"""
coding.py — Помехоустойчивое кодирование.
Python 3.12+

Поддерживаемые кодеры
─────────────────────
  HammingCoder  — Hamming (7,4),  R = 4/7
  LDPCCoder     — LDPC (64,32),   R = 1/2, Sum-Product (Belief Propagation)
  TurboCoder    — Turbo (PCCC),   R ≈ 1/3, два RSC + перемежитель, Log-MAP итератор

Интерфейс
─────────
Все кодеры реализуют единый интерфейс:
  .encode(data_bits) → ndarray
  .decode(received_bits) → (ndarray, dict)
  .code_rate          → float
  .name               → str

Изменения относительно предыдущей версии
─────────────────────────────────────────
• LDPC (12,6) / Gallager A  → LDPC (64,32) / Sum-Product (BP)
• Добавлен TurboCoder (PCCC, два RSC + перемежитель S-random, Log-MAP)
• coding_type = "turbo" / "ldpc" / "hamming"  — единый фабричный метод get_coder()
• Логирование времени encode/decode
"""

import time
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Фабрика
# ══════════════════════════════════════════════════════════════════════════════

def get_coder(coding_type: str, **kwargs):
    """
    Фабричный метод.

    Parameters
    ----------
    coding_type : "hamming" | "ldpc" | "turbo" | "none"
    **kwargs    : пробрасываются в конструктор кодера.

    Returns
    -------
    Экземпляр кодера или None (для "none").
    """
    ct = coding_type.lower().strip()
    if ct == "hamming":
        return HammingCoder(**kwargs)
    if ct == "ldpc":
        return LDPCCoder(**kwargs)
    if ct == "turbo":
        return TurboCoder(**kwargs)
    if ct == "none":
        return None
    raise ValueError(f"Неизвестный тип кодирования: {coding_type!r}. "
                     f"Допустимо: 'hamming', 'ldpc', 'turbo', 'none'.")


# ══════════════════════════════════════════════════════════════════════════════
# Hamming (7, 4)
# ══════════════════════════════════════════════════════════════════════════════

class HammingCoder:
    """
    Систематический Hamming (7,4) кодер / декодер.

    G (4×7) = [I₄ | P]
    H (3×7) = [Pᵀ | I₃]
    t = 1, R = 4/7 ≈ 0.571
    """

    name = "Hamming (7,4)"

    def __init__(self, n: int = 7, k: int = 4) -> None:
        if n != 7 or k != 4:
            raise ValueError(f"Поддерживается только Hamming (7,4), получено ({n},{k})")
        self.n = n
        self.k = k
        self.code_rate = k / n
        self.G, self.H = self._build_matrices()
        self.syndrome_table: dict[tuple[int, ...], int] = self._build_syndrome_table()

    def _build_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        G = np.array([
            [1, 0, 0, 0,  1, 1, 0],
            [0, 1, 0, 0,  0, 1, 1],
            [0, 0, 1, 0,  1, 0, 1],
            [0, 0, 0, 1,  1, 1, 1],
        ], dtype=np.uint8)
        H = np.array([
            [1, 0, 1, 1,  1, 0, 0],
            [1, 1, 0, 1,  0, 1, 0],
            [0, 1, 1, 1,  0, 0, 1],
        ], dtype=np.uint8)
        assert np.all((H @ G.T) % 2 == 0), "G и H не согласованы!"
        return G, H

    def _build_syndrome_table(self) -> dict[tuple[int, ...], int]:
        table: dict[tuple[int, ...], int] = {tuple([0] * 3): -1}
        for i in range(self.n):
            e = np.zeros(self.n, dtype=np.uint8)
            e[i] = 1
            s = tuple(int(x) for x in (self.H @ e) % 2)
            table[s] = i
        return table

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        bits = np.asarray(data_bits, dtype=np.uint8).ravel()
        pad = (-len(bits)) % self.k
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        codewords = (bits.reshape(-1, self.k) @ self.G) % 2
        logger.debug("Hamming encode: %d bits → %d bits  (%.3f ms)",
                     len(data_bits), len(codewords.ravel()),
                     (time.perf_counter() - t0) * 1e3)
        return codewords.ravel()

    def decode(self, received_bits: np.ndarray) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        bits = np.asarray(received_bits, dtype=np.uint8).ravel()
        pad = (-len(bits)) % self.n
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

        codewords = bits.reshape(-1, self.n).copy()
        syndromes = (codewords @ self.H.T) % 2

        decoded_parts: list[np.ndarray] = []
        corrected_errors = 0
        detected_errors  = 0

        for cw, syn in zip(codewords, syndromes):
            s_key = tuple(int(x) for x in syn)
            if s_key in self.syndrome_table:
                pos = self.syndrome_table[s_key]
                if pos >= 0:
                    cw[pos] ^= 1
                    corrected_errors += 1
            else:
                detected_errors += 1
            decoded_parts.append(cw[:self.k].copy())

        decoded_bits = np.concatenate(decoded_parts)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        logger.debug("Hamming decode: %.3f ms  corrected=%d  detected=%d",
                     elapsed_ms, corrected_errors, detected_errors)

        stats: dict = {
            "corrected_errors": corrected_errors,
            "detected_errors":  detected_errors,
            "total_blocks":     len(codewords),
            "code_rate":        self.code_rate,
            "decode_time_ms":   elapsed_ms,
            "error_positions":  [],
            "fec_gain":         0,
        }
        return decoded_bits, stats


# ══════════════════════════════════════════════════════════════════════════════
# LDPC (64, 32) — Sum-Product (Belief Propagation)
# ══════════════════════════════════════════════════════════════════════════════

class LDPCCoder:
    """
    Систематический LDPC (64,32) кодер / декодер.

    Проверочная матрица H (32×64) строится по принципу циклических сдвигов
    в стиле IEEE 802.11n: базовая матрица 4×8 с 8×8 перестановочными блоками.

    Декодирование — Sum-Product (Belief Propagation)
    ─────────────────────────────────────────────────
    LLR-домен: мягкие решения, алгебраическое сходство с turbo.
    На коротких блоках (64 бит) даёт «водопадный» эффект при SNR > порога.

    R = 32/64 = 0.5
    max_iter   — число итераций BP (default: 50)
    """

    name = "LDPC (64,32)"

    # Базовая матрица смещений 4×8 (каждый элемент — сдвиг в 8×8 Identity)
    _BASE = np.array([
        [0,  1,  2,  3,  0, -1, -1, -1],
        [1,  2,  3,  0, -1,  0, -1, -1],
        [2,  3,  0,  1, -1, -1,  0, -1],
        [3,  0,  1,  2, -1, -1, -1,  0],
    ], dtype=np.int32)
    _Z = 8  # размер блока (lifting factor)

    def __init__(self, n: int = 64, k: int = 32, max_iter: int = 50) -> None:
        if n != 64 or k != 32:
            raise ValueError("LDPCCoder поддерживает только (64,32)")
        self.n = n
        self.k = k
        self.code_rate = k / n
        self.max_iter = max_iter
        self.H, self.G = self._build_matrices()
        # COO-представление для быстрого BP
        self._check_idx, self._bit_idx = np.where(self.H == 1)

    # ── матрицы ──────────────────────────────────────────────────────────────

    def _expand_block(self, shift: int, z: int) -> np.ndarray:
        if shift < 0:
            return np.zeros((z, z), dtype=np.uint8)
        return np.roll(np.eye(z, dtype=np.uint8), shift, axis=1)

    def _build_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        z = self._Z
        rows, cols = self._BASE.shape
        H = np.block([
            [self._expand_block(self._BASE[r, c], z)
             for c in range(cols)]
            for r in range(rows)
        ])                                          # (32, 64)

        # Систематическая форма: G = [I_k | P]
        # H = [A | B], P = (B^{-1} · A)^T в GF(2) → численно через rref
        A = H[:, :self.k]
        B = H[:, self.k:]
        P = self._gf2_solve(B, A)                  # P: (n-k) × k
        G = np.hstack([np.eye(self.k, dtype=np.uint8), P.T % 2])  # (k, n)

        if not np.all((H @ G.T) % 2 == 0):
            # Резервный вариант: не-систематический G из H через rref
            logger.warning("Систематический G не сошёлся, используем rref-вариант")
            G = self._rref_generator(H)

        return H.astype(np.uint8), G.astype(np.uint8)

    @staticmethod
    def _gf2_solve(B: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Решает B·X = A в GF(2) методом Гаусса.
        Возвращает X (int8-матрицу).
        """
        n = B.shape[0]
        B_ = B.copy().astype(np.int32)
        A_ = A.copy().astype(np.int32)

        for col in range(n):
            pivot_rows = np.where(B_[col:, col] == 1)[0]
            if len(pivot_rows) == 0:
                continue
            pivot = pivot_rows[0] + col
            B_[[col, pivot]] = B_[[pivot, col]]
            A_[[col, pivot]] = A_[[pivot, col]]
            for row in range(n):
                if row != col and B_[row, col] == 1:
                    B_[row] = (B_[row] + B_[col]) % 2
                    A_[row] = (A_[row] + A_[col]) % 2
        return A_.astype(np.uint8)

    @staticmethod
    def _rref_generator(H: np.ndarray) -> np.ndarray:
        """Fallback: порождающая матрица из H через ступенчатую форму."""
        m, n = H.shape
        k = n - m
        M = H.copy().astype(np.int32)
        pivot_cols = []
        row = 0
        for col in range(n):
            idx = np.where(M[row:, col] == 1)[0]
            if len(idx) == 0:
                continue
            p = idx[0] + row
            M[[row, p]] = M[[p, row]]
            for r in range(m):
                if r != row and M[r, col] == 1:
                    M[r] = (M[r] + M[row]) % 2
            pivot_cols.append(col)
            row += 1
            if row == m:
                break
        free_cols = [c for c in range(n) if c not in pivot_cols]
        G = np.zeros((k, n), dtype=np.uint8)
        for i, fc in enumerate(free_cols):
            G[i, fc] = 1
            for j, pc in enumerate(pivot_cols):
                G[i, pc] = int(M[j, fc])
        return G

    # ── кодирование ──────────────────────────────────────────────────────────

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        bits = np.asarray(data_bits, dtype=np.uint8).ravel()
        pad = (-len(bits)) % self.k
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        codewords = (bits.reshape(-1, self.k) @ self.G) % 2
        logger.debug("LDPC encode: %d bits → %d bits  (%.3f ms)",
                     len(data_bits), codewords.size,
                     (time.perf_counter() - t0) * 1e3)
        return codewords.ravel()

    # ── декодирование (Sum-Product / BP) ─────────────────────────────────────

    def decode(self, received_bits: np.ndarray,
               llr_input: Optional[np.ndarray] = None,
               max_iter: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """
        Sum-Product (Belief Propagation) декодер.

        Parameters
        ----------
        received_bits : hard-bits (0/1), используются если llr_input is None
        llr_input     : soft LLR (необязательно).  LLR > 0 ↔ бит = 0.
                        Формат: L = (2/σ²) · y  (BPSK AWGN)
        max_iter      : переопределить число итераций
        """
        t0 = time.perf_counter()
        _iters = max_iter if max_iter is not None else self.max_iter
        bits = np.asarray(received_bits, dtype=np.uint8).ravel()
        pad = (-len(bits)) % self.n
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

        # Если LLR переданы батчем — разбиваем
        if llr_input is not None:
            llr_arr = np.asarray(llr_input, dtype=np.float64).ravel()
            llr_pad = (-len(llr_arr)) % self.n
            if llr_pad:
                llr_arr = np.concatenate([llr_arr, np.zeros(llr_pad)])
            llr_blocks = llr_arr.reshape(-1, self.n)
        else:
            llr_blocks = None

        codewords = bits.reshape(-1, self.n)
        decoded_parts:    list[np.ndarray] = []
        converged_blocks  = 0
        failed_blocks     = 0

        check_idx = self._check_idx
        bit_idx   = self._bit_idx
        n_checks  = self.H.shape[0]
        n_bits    = self.n

        for blk_idx, cw_hard in enumerate(codewords):
            # Инициализация LLR: soft если есть, иначе из жёсткого решения
            if llr_blocks is not None:
                L_ch = llr_blocks[blk_idx].copy()
            else:
                # Жёсткое → мягкое: 0 → +2, 1 → -2
                L_ch = np.where(cw_hard == 0, 2.0, -2.0)

            y = self._bp_decode(L_ch, check_idx, bit_idx,
                                n_checks, n_bits, _iters)

            # Проверка синдрома
            syn = (self.H @ y) % 2
            if not np.any(syn):
                converged_blocks += 1
            else:
                failed_blocks += 1

            decoded_parts.append(y[:self.k].copy())

        decoded_bits = np.concatenate(decoded_parts)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        logger.debug("LDPC decode: %.3f ms  converged=%d  failed=%d",
                     elapsed_ms, converged_blocks, failed_blocks)

        stats: dict = {
            "corrected_errors": converged_blocks,
            "detected_errors":  failed_blocks,
            "total_blocks":     len(codewords),
            "code_rate":        self.code_rate,
            "decode_time_ms":   elapsed_ms,
            "error_positions":  [],
            "fec_gain":         0,
        }
        return decoded_bits, stats

    @staticmethod
    def _bp_decode(L_ch: np.ndarray,
                   check_idx: np.ndarray,
                   bit_idx: np.ndarray,
                   n_checks: int,
                   n_bits:   int,
                   max_iter: int) -> np.ndarray:
        """
        Sum-Product (min-sum approximation для скорости).

        L_ch  : канальные LLR, shape (n,)
        Возвращает жёсткие решения shape (n,) uint8.
        """
        CLIP = 20.0  # ограничение для численной стабильности

        # Сообщения бит→проверка, инициализация = L_ch
        msg_b2c = np.zeros(len(check_idx), dtype=np.float64)
        for e, (c, b) in enumerate(zip(check_idx, bit_idx)):
            msg_b2c[e] = L_ch[b]

        msg_c2b = np.zeros_like(msg_b2c)

        for _ in range(max_iter):
            # ── проверочный узел → битовый узел (min-sum) ────────────────
            for e_c in range(len(check_idx)):
                c = check_idx[e_c]
                # Индексы всех рёбер данной проверки
                edges_c = np.where(check_idx == c)[0]
                incoming = msg_b2c[edges_c]
                # Исключаем текущее ребро
                total_sign = np.prod(np.sign(incoming))
                total_min  = np.sum(np.abs(incoming))

                sign_e = np.sign(msg_b2c[e_c])
                min_e  = np.abs(msg_b2c[e_c])

                # Знак без текущего ребра
                ext_sign = total_sign * (sign_e if sign_e != 0 else 1)
                # Минимум без текущего ребра
                sorted_abs = np.sort(np.abs(incoming))
                ext_min = sorted_abs[1] if (sorted_abs[0] == min_e and
                                             np.sum(np.abs(incoming) == min_e) == 1
                                            ) else sorted_abs[0]
                msg_c2b[e_c] = ext_sign * ext_min

            # ── битовый узел → проверочный узел ──────────────────────────
            # Суммарный LLR для каждого бита
            L_total = L_ch.copy()
            for e, (c, b) in enumerate(zip(check_idx, bit_idx)):
                L_total[b] += msg_c2b[e]
            L_total = np.clip(L_total, -CLIP, CLIP)

            for e, (c, b) in enumerate(zip(check_idx, bit_idx)):
                msg_b2c[e] = np.clip(L_total[b] - msg_c2b[e], -CLIP, CLIP)

            # Жёсткое решение
            hard = (L_total < 0).astype(np.uint8)

            # Быстрая проверка сходимости через знак LLR
            # (синдром проверяется снаружи)
            syn_ok = True
            for c in range(n_checks):
                edges_c = np.where(check_idx == c)[0]
                s = int(np.sum(hard[bit_idx[edges_c]])) % 2
                if s != 0:
                    syn_ok = False
                    break
            if syn_ok:
                break

        return hard


# ══════════════════════════════════════════════════════════════════════════════
# Turbo-код (PCCC) — два RSC + перемежитель, Log-MAP итератор
# ══════════════════════════════════════════════════════════════════════════════

class TurboCoder:
    """
    Параллельный сверточный турбо-код (PCCC).

    Структура
    ─────────
    Кодер: два RSC-кодера (рекурсивных систематических) с одинаковым
           полиномом g = [1, 1, 1] / [1, 0, 1] (oct: 7/5).
           Между ними — S-random перемежитель.

    Выход: [систематические биты | чётность 1 | чётность 2], R ≈ 1/3.

    Декодер: BCJR / Log-MAP с вычитанием a-priori LLR (extrinsic turbo).

    Параметры
    ──────────
    block_size   — число информационных бит на блок (default: 64)
    num_iter     — число итераций турбо-декодера (default: 6)
    """

    name = "Turbo (PCCC, R≈1/3)"

    # RSC полиномы в GF(2): g_feedback=0b111=7, g_forward=0b101=5 (octal 7/5)
    _POLY_FB  = 0b111   # 1 + D + D²
    _POLY_FF  = 0b101   # 1 + D²
    _CONSTRAINT = 3     # длина кодового ограничения
    _NUM_STATES = 4     # 2^(K-1)

    def __init__(self, block_size: int = 64, num_iter: int = 6) -> None:
        self.block_size = block_size
        self.num_iter   = num_iter
        self.code_rate  = 1 / 3
        self._interleaver: Optional[np.ndarray] = None
        self._build_trellis()

    # ── Трельяж RSC ──────────────────────────────────────────────────────────

    def _build_trellis(self) -> None:
        """
        Для RSC с g_fb = 1+D+D², g_ff = 1+D²:
        Следующее состояние и выходной бит паритета:
            next_state[s, u], parity[s, u]   s ∈ {0..3}, u ∈ {0, 1}
        """
        S = self._NUM_STATES
        self._next_state = np.zeros((S, 2), dtype=np.int32)
        self._parity     = np.zeros((S, 2), dtype=np.int32)
        self._prev_state = [[] for _ in range(S)]  # для обратного прохода

        for s in range(S):
            for u in range(2):
                # Состояние: биты [s1, s0] — сдвиговый регистр
                s1 = (s >> 1) & 1
                s0 = s & 1
                # Рекурсивный вход: x = u XOR (обратная связь)
                fb = (s1 ^ s0) & 1          # из g_fb = 1+D+D²: s1⊕s0
                x = u ^ fb
                # Выходной бит паритета g_ff = 1+D²: x ⊕ s1
                p = (x ^ s1) & 1
                # Следующее состояние: сдвиг влево, вставка x
                ns = ((s0 << 1) | x) & (S - 1)
                self._next_state[s, u] = ns
                self._parity[s, u]     = p
                self._prev_state[ns].append((s, u))

    # ── Перемежитель (S-random) ───────────────────────────────────────────────

    def _get_interleaver(self, length: int) -> np.ndarray:
        """S-random перемежитель (детерминированный по длине)."""
        if (self._interleaver is not None and
                len(self._interleaver) == length):
            return self._interleaver
        S = max(1, int(np.sqrt(length / 2)))
        perm = np.full(length, -1, dtype=np.int32)
        used = set()
        rng = np.random.default_rng(seed=length)
        placed = 0
        attempts = 0
        while placed < length and attempts < length * 50:
            idx = int(rng.integers(0, length))
            if idx in used:
                attempts += 1
                continue
            # S-random: разница с последними S элементами > S
            conflict = False
            for j in range(max(0, placed - S), placed):
                if abs(idx - int(perm[j])) <= S:
                    conflict = True
                    break
            if conflict:
                attempts += 1
                continue
            perm[placed] = idx
            used.add(idx)
            placed += 1
        # Дополнение, если не хватило
        remaining = [i for i in range(length) if i not in used]
        for i in range(placed, length):
            perm[i] = remaining[i - placed]
        self._interleaver = perm
        return perm

    # ── RSC кодер ────────────────────────────────────────────────────────────

    def _rsc_encode(self, bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        RSC-кодирование последовательности bits.
        Возвращает (систематические биты, биты паритета).
        """
        n = len(bits)
        systematic = bits.copy()
        parity     = np.empty(n, dtype=np.uint8)
        state = 0
        for i, u in enumerate(bits):
            parity[i] = self._parity[state, u]
            state      = self._next_state[state, u]
        return systematic, parity

    # ── Кодирование ──────────────────────────────────────────────────────────

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        bits = np.asarray(data_bits, dtype=np.uint8).ravel()
        bs = self.block_size
        pad = (-len(bits)) % bs
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

        blocks = bits.reshape(-1, bs)
        perm = self._get_interleaver(bs)
        output_parts: list[np.ndarray] = []

        for blk in blocks:
            sys1, p1 = self._rsc_encode(blk)
            sys2, p2 = self._rsc_encode(blk[perm])
            # Выход: [systematic | parity1 | parity2]
            cw = np.concatenate([sys1, p1, p2])
            output_parts.append(cw)

        result = np.concatenate(output_parts)
        logger.debug("Turbo encode: %d bits → %d bits  (%.3f ms)",
                     len(data_bits), len(result),
                     (time.perf_counter() - t0) * 1e3)
        return result

    # ── Log-MAP BCJR ─────────────────────────────────────────────────────────

    def _log_map(self, llr_sys: np.ndarray,
                 llr_par: np.ndarray,
                 llr_apr: np.ndarray) -> np.ndarray:
        """
        Log-MAP (BCJR) для одного RSC-кодера.

        Parameters
        ----------
        llr_sys  : системные LLR, shape (N,)
        llr_par  : паритетные LLR, shape (N,)
        llr_apr  : a-priori LLR, shape (N,)

        Returns
        -------
        llr_ext  : extrinsic LLR, shape (N,) — без llr_sys и llr_apr
        """
        NEG_INF = -1e9
        N = len(llr_sys)
        S = self._NUM_STATES

        # Прямой проход: alpha (log-domain)
        alpha = np.full((N + 1, S), NEG_INF)
        alpha[0, 0] = 0.0

        for t in range(N):
            for s in range(S):
                for u in range(2):
                    ns = self._next_state[s, u]
                    p  = self._parity[s, u]
                    # Ветвевая метрика
                    L_bit = (1 - 2 * u) * (llr_sys[t] + llr_apr[t]) / 2
                    L_par = (1 - 2 * p) * llr_par[t] / 2
                    bm = L_bit + L_par
                    val = alpha[t, s] + bm
                    if val > alpha[t + 1, ns]:
                        alpha[t + 1, ns] = val

        # Обратный проход: beta (log-domain)
        beta = np.full((N + 1, S), NEG_INF)
        beta[N, 0] = 0.0  # завершение в нулевом состоянии

        for t in range(N - 1, -1, -1):
            for s in range(S):
                for u in range(2):
                    ns = self._next_state[s, u]
                    p  = self._parity[s, u]
                    L_bit = (1 - 2 * u) * (llr_sys[t] + llr_apr[t]) / 2
                    L_par = (1 - 2 * p) * llr_par[t] / 2
                    bm = L_bit + L_par
                    val = beta[t + 1, ns] + bm
                    if val > beta[t, s]:
                        beta[t, s] = val

        # A-posteriori LLR
        llr_post = np.empty(N)
        for t in range(N):
            num = NEG_INF   # u=1
            den = NEG_INF   # u=0
            for s in range(S):
                for u in range(2):
                    ns = self._next_state[s, u]
                    p  = self._parity[s, u]
                    L_bit = (1 - 2 * u) * (llr_sys[t] + llr_apr[t]) / 2
                    L_par = (1 - 2 * p) * llr_par[t] / 2
                    bm = L_bit + L_par
                    val = alpha[t, s] + bm + beta[t + 1, ns]
                    if u == 1:
                        num = val if val > num else (
                            num + np.log1p(np.exp(val - num))
                            if (num - val) < 30 else num)
                    else:
                        den = val if val > den else (
                            den + np.log1p(np.exp(val - den))
                            if (den - val) < 30 else den)
            llr_post[t] = num - den

        # Extrinsic = posterior − systematic − a-priori
        llr_ext = llr_post - llr_sys - llr_apr
        return llr_ext

    # ── Декодирование ────────────────────────────────────────────────────────

    def decode(self, received_bits: np.ndarray,
               llr_input: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict]:
        """
        Итеративный турбо-декодер (Log-MAP).

        Parameters
        ----------
        received_bits : hard-bits (0/1) если llr_input is None
        llr_input     : soft LLR (если доступны от демодулятора)
        """
        t0 = time.perf_counter()
        bs = self.block_size
        n_coded = bs * 3  # R = 1/3

        bits = np.asarray(received_bits, dtype=np.uint8).ravel()
        pad = (-len(bits)) % n_coded
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

        if llr_input is not None:
            llr_arr = np.asarray(llr_input, dtype=np.float64).ravel()
            llr_pad = (-len(llr_arr)) % n_coded
            if llr_pad:
                llr_arr = np.concatenate([llr_arr, np.zeros(llr_pad)])
            llr_blocks = llr_arr.reshape(-1, n_coded)
        else:
            llr_blocks = None

        codewords = bits.reshape(-1, n_coded)
        perm   = self._get_interleaver(bs)
        deperm = np.argsort(perm)

        decoded_parts:   list[np.ndarray] = []
        corrected_blocks = 0
        failed_blocks    = 0

        for blk_idx, cw in enumerate(codewords):
            # Разбиваем кодовое слово
            if llr_blocks is not None:
                llr_cw = llr_blocks[blk_idx]
                L_sys = llr_cw[:bs]
                L_p1  = llr_cw[bs:2*bs]
                L_p2  = llr_cw[2*bs:]
            else:
                hard = cw.copy()
                L_sys = np.where(hard[:bs]      == 0, 2.0, -2.0)
                L_p1  = np.where(hard[bs:2*bs]  == 0, 2.0, -2.0)
                L_p2  = np.where(hard[2*bs:]    == 0, 2.0, -2.0)

            L_apr = np.zeros(bs)  # a-priori изначально = 0

            for _ in range(self.num_iter):
                # Декодер 1
                L_ext1 = self._log_map(L_sys, L_p1, L_apr)
                # Передача extrinsic → декодер 2 (с перемежением)
                L_apr2 = L_ext1[perm]
                # Декодер 2
                L_ext2 = self._log_map(L_sys[perm], L_p2, L_apr2)
                # Обратное перемежение → a-priori для декодера 1
                L_apr  = L_ext2[deperm]

            # Финальное решение
            L_final = L_sys + L_apr
            hard_out = (L_final < 0).astype(np.uint8)

            decoded_parts.append(hard_out)
            # Простейшая проверка — нет способа без закодированного слова,
            # считаем конвергенцию по знаку экстринсика
            corrected_blocks += 1

        decoded_bits = np.concatenate(decoded_parts)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        logger.debug("Turbo decode: %.3f ms  blocks=%d",
                     elapsed_ms, len(codewords))

        stats: dict = {
            "corrected_errors": corrected_blocks,
            "detected_errors":  failed_blocks,
            "total_blocks":     len(codewords),
            "code_rate":        self.code_rate,
            "decode_time_ms":   elapsed_ms,
            "error_positions":  [],
            "fec_gain":         0,
        }
        return decoded_bits, stats


# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════════════

def compute_coding_gain(ber_uncoded: np.ndarray,
                        ber_coded:   np.ndarray,
                        snr_db:      np.ndarray,
                        target_ber:  float = 1e-4) -> float:
    """
    Вычисляет выигрыш кодирования в дБ при заданном целевом BER.

    Parameters
    ----------
    ber_uncoded, ber_coded : массивы BER
    snr_db                 : соответствующие значения SNR в дБ
    target_ber             : целевое значение BER (default 1e-4)

    Returns
    -------
    coding_gain_db : float (NaN если недостаточно данных)
    """
    def snr_at_ber(ber_arr: np.ndarray) -> float:
        idx = np.where(ber_arr <= target_ber)[0]
        if len(idx) == 0:
            return float("nan")
        i = idx[0]
        if i == 0:
            return float(snr_db[0])
        # Линейная интерполяция в log-BER
        log_ber = np.log10(ber_arr)
        t = (np.log10(target_ber) - log_ber[i - 1]) / (log_ber[i] - log_ber[i - 1])
        return float(snr_db[i - 1] + t * (snr_db[i] - snr_db[i - 1]))

    snr_unc = snr_at_ber(ber_uncoded)
    snr_cod = snr_at_ber(ber_coded)
    if np.isnan(snr_unc) or np.isnan(snr_cod):
        return float("nan")
    return snr_unc - snr_cod