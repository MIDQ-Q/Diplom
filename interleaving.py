"""
interleaving.py — Блочное перемежение (interleaving) битового потока.
Python 3.12+

Назначение
──────────
Борьба с пакетными ошибками (burst errors): канал часто портит биты
подряд (глубокие замирания Rayleigh, импульсные помехи). Коды FEC
(Hamming, LDPC, Turbo) рассчитаны на случайные одиночные ошибки.
Интерливер разносит соседние биты во времени — пакетная ошибка
после деинтерливинга превращается в россыпь одиночных, которые FEC
уже может исправить.

Алгоритм блочного интерливера
───────────────────────────────
  Запись в матрицу (depth × width) по строкам:
    [b0  b1  b2  ...  b_{w-1}  ]
    [b_w b_{w+1}  ... b_{2w-1} ]
    ...

  Чтение по столбцам:
    [b0, b_w, b_{2w}, ...  b_1, b_{w+1}, ...]

  При деинтерливинге — обратная операция.

Параметры
─────────
  depth : int  — число строк матрицы (глубина перемежения).
                 Типичные значения: 4, 8, 16, 32.
                 Чем больше — тем лучше рассыпаются пакеты,
                 но больше задержка (depth × width бит).
  width : int  — число столбцов. По умолчанию = len(bits) // depth.
                 Фиксируется при создании объекта или выбирается
                 автоматически по длине блока.

Интерфейс
─────────
  BlockInterleaver(depth)
    .interleave(bits)    → ndarray  (перемежённые биты)
    .deinterleave(bits)  → ndarray  (восстановленный порядок)
    .depth               → int
    .overhead_bits(n)    → int      (паддинг для блока длиной n)
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class BlockInterleaver:
    """
    Блочный интерливер / деинтерливер.

    Запись в матрицу по строкам, чтение по столбцам.
    Если длина битового потока не кратна depth — добавляется нулевой паддинг,
    который снимается при деинтерливинге (срезается до исходной длины).

    Parameters
    ----------
    depth : int
        Глубина перемежения (число строк матрицы). Рекомендуется
        выбирать кратным длине кодового слова FEC.
    """

    def __init__(self, depth: int = 8) -> None:
        if depth < 2:
            raise ValueError(f"depth должна быть ≥ 2, получено {depth}")
        self.depth = depth

    # ── Перемежение ───────────────────────────────────────────────────────────

    def interleave(self, bits: np.ndarray) -> np.ndarray:
        """
        Перемежает битовый поток.

        Алгоритм:
          1. Дополнить до кратного depth нулями (паддинг).
          2. Записать в матрицу (depth × width) по строкам.
          3. Прочитать по столбцам → выходной поток.

        Parameters
        ----------
        bits : ndarray (uint8)
            Входной битовый поток.

        Returns
        -------
        ndarray (uint8)
            Перемежённый поток (той же длины, что вход + паддинг).
        """
        t0 = time.perf_counter()
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        original_len = len(bits)

        # Паддинг до кратного depth
        pad = (-original_len) % self.depth
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

        width = len(bits) // self.depth
        matrix = bits.reshape(self.depth, width)
        interleaved = matrix.T.ravel()   # читаем по столбцам

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        logger.debug(
            "BlockInterleaver.interleave: %d bits → %d bits "
            "(depth=%d, width=%d, pad=%d, %.2f ms)",
            original_len, len(interleaved), self.depth, width, pad, elapsed_ms,
        )
        return interleaved

    # ── Деинтерливинг ─────────────────────────────────────────────────────────

    def deinterleave(self, bits: np.ndarray, original_len: int | None = None) -> np.ndarray:
        """
        Восстанавливает исходный порядок бит.

        Алгоритм:
          1. Записать перемежённый поток в матрицу (width × depth) по строкам.
          2. Прочитать по столбцам → восстановленный поток.
          3. Срезать до original_len (убрать паддинг).

        Parameters
        ----------
        bits        : ndarray (uint8) — перемежённый поток.
        original_len: int | None      — исходная длина (без паддинга).
                      Если None — паддинг не срезается.

        Returns
        -------
        ndarray (uint8)
        """
        t0 = time.perf_counter()
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        n = len(bits)

        # Дополнить до кратного depth если нужно
        pad = (-n) % self.depth
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

        width = len(bits) // self.depth
        # Обратная операция: записать по столбцам (width × depth), читать по строкам
        matrix = bits.reshape(width, self.depth)
        deinterleaved = matrix.T.ravel()

        if original_len is not None:
            deinterleaved = deinterleaved[:original_len]

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        logger.debug(
            "BlockInterleaver.deinterleave: %d bits → %d bits "
            "(depth=%d, width=%d, %.2f ms)",
            n, len(deinterleaved), self.depth, width, elapsed_ms,
        )
        return deinterleaved

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def overhead_bits(self, n: int) -> int:
        """Число бит паддинга для блока длиной n."""
        return (-n) % self.depth

    def matrix_shape(self, n: int) -> tuple[int, int]:
        """Форма матрицы интерливера для блока длиной n (с учётом паддинга)."""
        total = n + self.overhead_bits(n)
        return self.depth, total // self.depth


# ── Фабрика ──────────────────────────────────────────────────────────────────

def get_interleaver(config: dict) -> "BlockInterleaver | None":
    """
    Создаёт интерливер по конфигурации.

    config["interleaving"] пример:
        {
            "enabled": true,
            "depth": 16
        }

    Возвращает BlockInterleaver или None если отключён.
    """
    cfg = config.get("interleaving", {})
    if not cfg.get("enabled", False):
        return None
    depth = int(cfg.get("depth", 8))
    return BlockInterleaver(depth=depth)
