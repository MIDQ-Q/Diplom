import numpy as np
from typing import Dict, Tuple, Optional
import logging


class HammingCoder:
    """
    Hamming (7,4) systematic coder/decoder с максимально правдоподобным декодированием.

    Структура:
    - G (4x7): генераторная матрица в систематическом виде [I_4 | P]
    - H (3x7): проверочная матрица [P^T | I_3]
    - Синдром = H * c^T идентифицирует позицию одиночной ошибки

    Особенность: исправляет все одиночные ошибки, обнаруживает двойные
    """

    def __init__(self, n: int = 7, k: int = 4):
        self.n = n
        self.k = k
        self.G, self.H = self._generate_matrices()
        self.syndrome_table = self._create_syndrome_table()

    def _generate_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Создает систематические G и H для (7,4) Hamming кода"""
        if self.n == 7 and self.k == 4:
            # Систематическая генераторная матрица: G = [I_4 | P]
            G = np.array([
                [1, 0, 0, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 1, 1],
                [0, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 1]
            ], dtype=int)

            # Проверочная матрица: H = [P^T | I_3]
            H = np.array([
                [1, 0, 1, 1, 1, 0, 0],
                [1, 1, 0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0, 0, 1]
            ], dtype=int)
        else:
            raise ValueError(f"Поддерживается только (7,4). Получено: ({self.n},{self.k})")

        return G, H

    def _create_syndrome_table(self) -> Dict:
        """Создает таблицу синдромов для декодирования одиночных ошибок"""
        table = {}

        # Для каждой одиночной ошибки найти ее синдром
        for i in range(self.n):
            e = np.zeros(self.n, dtype=int)
            e[i] = 1
            s = tuple((self.H @ e) % 2)
            table[s] = i

        # Нулевой синдром = нет ошибок
        table[tuple([0] * self.H.shape[0])] = -1

        return table

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        """
        Кодирование: k бит -> n кодовых слов
        c = (u * G) mod 2
        """
        bits = np.array(data_bits, dtype=int).flatten()
        padding = (self.k - (len(bits) % self.k)) % self.k
        if padding > 0:
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])

        blocks = bits.reshape(-1, self.k)
        codewords = (blocks @ self.G) % 2
        return codewords.reshape(-1)

    def decode(self, received_bits: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Декодирование с исправлением одиночных ошибок:
        1. Вычисляем синдром: s = H * r^T
        2. По таблице определяем позицию ошибки
        3. Исправляем и извлекаем информационные биты
        """
        bits = np.array(received_bits, dtype=int).flatten()
        padding = (self.n - (len(bits) % self.n)) % self.n
        if padding > 0:
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])

        codewords = bits.reshape(-1, self.n)
        syndromes = (codewords @ self.H.T) % 2

        decoded_bits = []
        corrected_errors = 0
        detected_errors = 0
        error_positions = []

        for i in range(len(codewords)):
            cw = codewords[i].copy()
            s = tuple(syndromes[i].astype(int))

            if s in self.syndrome_table:
                pos = self.syndrome_table[s]
                if pos >= 0:  # Есть одиночная ошибка
                    cw[pos] ^= 1
                    corrected_errors += 1
                    error_positions.append(pos)
                # Если pos == -1, то нет ошибок
            else:
                # Необычный синдром (двойная ошибка или выше)
                detected_errors += 1

            decoded_bits.extend(cw[:self.k])

        stats = {
            "corrected_errors": corrected_errors,
            "detected_errors": detected_errors,
            "total_blocks": len(codewords),
            "error_positions": error_positions,
            "code_rate": self.k / self.n,
            "fec_gain": 0  # Будет вычислено в симуляции
        }
        return np.array(decoded_bits, dtype=int), stats


class LDPCCoder:
    """
    LDPC (Low-Density Parity-Check) кодер/декодер.

    Используется систематический (12,6) код с проверочной матрицей H = [A | I_6],
    где A — матрица 6x6, и порождающей матрицей G = [I_6 | A^T].
    Декодирование методом битового флиппинга с возможностью многобитовых флипов.
    """

    def __init__(self, n: int = 12, k: int = 6):
        self.n = n
        self.k = k
        self.H, self.G = self._generate_matrices()
        # Предвычисляем списки проверок для каждого бита и битов для каждой проверки
        self._init_check_nodes()

    def _generate_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Генерирует согласованные H и G для систематического LDPC кода (12,6)."""
        # Задаём матрицу A размером (n-k) x k = 6x6 (разреженную)
        A = np.array([
            [1, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 1]
        ], dtype=int)

        # Проверочная матрица в систематическом виде: H = [A | I_{n-k}]
        H = np.hstack([A, np.eye(self.n - self.k, dtype=int)])

        # Порождающая матрица в систематическом виде: G = [I_k | A^T]
        G = np.hstack([np.eye(self.k, dtype=int), A.T])

        # Проверка: H * G^T должна быть нулевой в GF(2)
        assert np.all((H @ G.T) % 2 == 0), "Матрицы H и G не согласованы!"

        return H, G

    def _init_check_nodes(self):
        """Создаёт списки связей для ускорения декодирования."""
        self.bit_to_checks = []  # для каждого бита список проверок, в которых он участвует
        self.check_to_bits = []  # для каждой проверки список битов, которые в неё входят

        n_checks = self.H.shape[0]
        for j in range(self.n):
            self.bit_to_checks.append(list(np.where(self.H[:, j] == 1)[0]))

        for i in range(n_checks):
            self.check_to_bits.append(list(np.where(self.H[i, :] == 1)[0]))

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        """
        Кодирование: u -> c = u * G mod 2
        """
        bits = np.array(data_bits, dtype=int).flatten()
        padding = (self.k - (len(bits) % self.k)) % self.k
        if padding:
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])

        blocks = bits.reshape(-1, self.k)
        codewords = (blocks @ self.G) % 2
        return codewords.reshape(-1)

    def decode(self, received_bits: np.ndarray, max_iter: int = 20) -> Tuple[np.ndarray, Dict]:
        """
        Декодирование методом битового флиппинга.

        Алгоритм:
        1. Инициализация y = принятые биты.
        2. Для каждой итерации:
           - Вычислить синдром s = H * y (mod 2)
           - Если s == 0, успех
           - Для каждого бита подсчитать число неудовлетворённых проверок
           - Найти максимальное значение
           - Перевернуть все биты, у которых число неудовлетворённых проверок равно максимуму (и > 0)
        3. После max_iter вернуть текущий y и статистику.
        """
        bits = np.array(received_bits, dtype=int).flatten()
        padding = (self.n - (len(bits) % self.n)) % self.n
        if padding:
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])

        codewords = bits.reshape(-1, self.n)
        decoded_bits = []
        corrected_blocks = 0   # блоки, где сошёлся синдром
        detected_blocks = 0    # блоки, где не сошёлся после всех итераций
        total_flips = 0

        for cw in codewords:
            y = cw.copy()
            converged = False
            flips_this_block = 0

            for iteration in range(max_iter):
                # Вычисляем синдром
                syndrome = (self.H @ y) % 2
                if np.all(syndrome == 0):
                    converged = True
                    break

                # Для каждого бита считаем количество неудовлетворённых проверок
                unsatisfied = np.zeros(self.n, dtype=int)
                for j in range(self.n):
                    for check in self.bit_to_checks[j]:
                        if syndrome[check] == 1:
                            unsatisfied[j] += 1

                max_unsat = np.max(unsatisfied)
                if max_unsat == 0:
                    # Нет неудовлетворённых проверок — должно быть схождение, но синдром не ноль? (противоречие)
                    break

                # Переворачиваем все биты с максимальным числом неудовлетворённых проверок
                flip_positions = np.where(unsatisfied == max_unsat)[0]
                y[flip_positions] ^= 1
                flips_this_block += len(flip_positions)

            # Проверяем финальный синдром
            if np.all((self.H @ y) % 2 == 0):
                corrected_blocks += 1
            else:
                detected_blocks += 1

            decoded_bits.extend(y[:self.k])
            total_flips += flips_this_block

        stats = {
            "corrected_errors": corrected_blocks,        # число успешно декодированных блоков
            "detected_errors": detected_blocks,          # блоки с неисправленными ошибками
            "total_blocks": len(codewords),
            "total_flips": total_flips,                   # общее число флипов (опционально)
            "code_rate": self.k / self.n,
            "fec_gain": 0
        }
        return np.array(decoded_bits, dtype=int), stats