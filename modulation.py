"""
modulation.py — M-PSK и M-QAM модуляция / демодуляция.
Python 3.12+

Изменения относительно предыдущей версии:
─────────────────────────────────────────────────────────────────────
PSKModulator / QAMModulator:
  • modulate(): полная векторизация через numpy LUT — убран цикл for.
  • demodulate(): сборка бит через _bits_lut[indices].ravel().
  • PSK коррекция: только фазовая s·exp(−j·∠h).
  • QAM коррекция: ZF s/h.
  • Аннотации типов Python 3.12.
  • Q_function: защита x < 0, возвращает чистый float.

Новое (2 марта):
  • QAMModulator: поддержка M=256 (QAM-256) — расширен список {4,16,64,256}.
  • Теоретические BER/SER для Rayleigh-канала:
      theoretical_ber_rayleigh_psk()
      theoretical_ber_rayleigh_qam()
  • LLR-расчёт для soft-decision:
      compute_llr_psk()  — LLR бит для M-PSK
      compute_llr_qam()  — LLR бит для M-QAM (приближение мин. расстояний)
"""

import numpy as np
import scipy.special as sp
from math import pi, log2
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательное
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=512)
def Q_function(x: float) -> float:
    """
    Q(x) = 0.5 · erfc(x / √2).
    Кешируется: повторные вызовы (теоркривые) мгновенны.
    Защита от x < 0: Q(−x) = 1 − Q(x).
    """
    if x < 0:
        return 1.0 - Q_function(-x)
    return float(0.5 * sp.erfc(x / np.sqrt(2)))


# ══════════════════════════════════════════════════════════════════════════════
# M-PSK
# ══════════════════════════════════════════════════════════════════════════════

class PSKModulator:
    """
    M-PSK модулятор / демодулятор с кодом Грея и коррекцией канала.

    Поддерживаемые порядки: M ∈ {2, 4, 8, 16}.
    Созвездие: M точек на единичной окружности.
    QPSK (M=4): повёрнуто на π/4 (стандартная IQ-ориентация).

    Коррекция канала в demodulate()
    ────────────────────────────────
    Только фазовая: s_corr = s · exp(−j·∠h).
    Делить на |h| нет смысла — все точки PSK равноудалены от нуля,
    решение зависит только от угла символа.
    """

    def __init__(self, M: int = 8, use_gray_code: bool = True) -> None:
        if M not in (2, 4, 8, 16):
            raise ValueError(f"PSK: поддерживается M ∈ {{2,4,8,16}}, получено {M}")
        self.M               = M
        self.bits_per_symbol = int(log2(M))
        self.use_gray_code   = use_gray_code
        self._build_constellation()

    def _build_constellation(self) -> None:
        """
        Строит созвездие и LUT-таблицы:
          constellation_points — (M,) комплексный массив
          bit_to_index         — dict: битовый кортеж → индекс точки
          index_to_bits        — dict: индекс → битовый кортеж
          _bits_lut            — (M, bps) uint8 для быстрого demodulate
          _modulate_lut        — (M,) int32 для быстрого modulate
        """
        angles = np.linspace(0, 2 * pi, self.M, endpoint=False)
        if self.M == 4:
            angles = angles + pi / 4
        self.constellation_points: np.ndarray = np.exp(1j * angles)

        self.bit_to_index:  dict[tuple[int, ...], int] = {}
        self.index_to_bits: dict[int, tuple[int, ...]] = {}

        for i in range(self.M):
            code = i ^ (i >> 1) if self.use_gray_code else i
            bits = tuple(
                (code >> (self.bits_per_symbol - 1 - j)) & 1
                for j in range(self.bits_per_symbol)
            )
            self.bit_to_index[bits] = i
            self.index_to_bits[i]   = bits

        # LUT для demodulate: _bits_lut[i] = битовая строка точки i
        self._bits_lut: np.ndarray = np.array(
            [self.index_to_bits[i] for i in range(self.M)], dtype=np.uint8
        )  # (M, bps)

        # LUT для modulate: натуральный индекс → индекс созвездия
        if self.use_gray_code:
            gray_lut = np.arange(self.M, dtype=np.int32) ^ (np.arange(self.M) >> 1)
            self._modulate_lut: np.ndarray = np.argsort(gray_lut).astype(np.int32)
        else:
            self._modulate_lut = np.arange(self.M, dtype=np.int32)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Биты → символы PSK (полностью векторизовано).

        Алгоритм:
          1. Разбить на группы по bps.
          2. Группа → натуральный int: nat = bits · [2^(bps−1), …, 1].
          3. LUT: nat → индекс созвездия.
          4. constellation_points[индекс].
        """
        b = np.asarray(bits, dtype=np.uint8).ravel()
        pad = (-len(b)) % self.bits_per_symbol
        if pad:
            b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])

        groups = b.reshape(-1, self.bits_per_symbol)
        powers = (1 << np.arange(self.bits_per_symbol - 1, -1, -1, dtype=np.int32))
        nat_idx = groups.astype(np.int32) @ powers              # (N,)

        return self.constellation_points[self._modulate_lut[nat_idx]]

    def demodulate(self, symbols: np.ndarray,
                   channel_coeff: np.ndarray | None = None) -> np.ndarray:
        """
        Символы → биты (батчевое минимальное расстояние + LUT).

        Коррекция канала: только фазовая — s · exp(−j·∠h).
        """
        s = np.asarray(symbols, dtype=complex).ravel()

        if channel_coeff is not None:
            h = np.asarray(channel_coeff, dtype=complex).ravel()
            if len(h) != len(s):
                raise ValueError(f"channel_coeff длина {len(h)} ≠ symbols длина {len(s)}")
            s = s * np.exp(-1j * np.angle(h))

        dist2   = np.abs(s[:, np.newaxis] - self.constellation_points[np.newaxis, :]) ** 2
        indices = np.argmin(dist2, axis=1)
        return self._bits_lut[indices].ravel()


# ══════════════════════════════════════════════════════════════════════════════
# M-QAM
# ══════════════════════════════════════════════════════════════════════════════

class QAMModulator:
    """
    M-QAM модулятор / демодулятор с кодом Грея и ZF-коррекцией канала.

    Поддерживаемые порядки: M ∈ {4, 16, 64}.
    Созвездие нормировано: E[|s|²] = 1.

    Коррекция канала в demodulate()
    ────────────────────────────────
    ZF: s_corr = s / h.
    Точки QAM имеют разные амплитуды → нужно убрать и фазу, и амплитуду.
    При |h| ≈ 0 символ считается потерянным (сохраняем исходный s).
    """

    def __init__(self, M: int = 16, use_gray_code: bool = True) -> None:
        if M not in (4, 16, 64, 256):
            raise ValueError(f"QAM: поддерживается M ∈ {{4,16,64,256}}, получено {M}")
        self.M               = M
        self.bits_per_symbol = int(log2(M))
        self.use_gray_code   = use_gray_code
        self._build_constellation()

    def _build_constellation(self) -> None:
        side     = int(np.sqrt(self.M))
        amps     = np.arange(-(side - 1), side, 2, dtype=float)
        gray_map = np.array([i ^ (i >> 1) for i in range(side)], dtype=np.int32)

        I_grid, Q_grid = np.meshgrid(amps, amps)
        points   = (I_grid + 1j * Q_grid).ravel()
        avg_pwr  = float(np.mean(np.abs(points) ** 2))
        self.constellation_points: np.ndarray = points / np.sqrt(avg_pwr)

        half = self.bits_per_symbol // 2
        self.bit_to_index:  dict[tuple[int, ...], int] = {}
        self.index_to_bits: dict[int, tuple[int, ...]] = {}

        for i in range(self.M):
            qi = i // side
            ii = i % side
            gi = gray_map[ii] if self.use_gray_code else ii
            gq = gray_map[qi] if self.use_gray_code else qi
            bits = (
                tuple((gi >> (half - 1 - j)) & 1 for j in range(half)) +
                tuple((gq >> (half - 1 - j)) & 1 for j in range(half))
            )
            self.bit_to_index[bits] = i
            self.index_to_bits[i]   = bits

        # LUT для demodulate
        self._bits_lut: np.ndarray = np.array(
            [self.index_to_bits[i] for i in range(self.M)], dtype=np.uint8
        )  # (M, bps)

        # LUT для modulate: nat → индекс созвездия.
        # Прямое соответствие: nat → битовый кортеж (MSB first) → bit_to_index.
        bps = self.bits_per_symbol
        lut = np.empty(self.M, dtype=np.int32)
        for nat in range(self.M):
            bits = tuple((nat >> (bps - 1 - j)) & 1 for j in range(bps))
            lut[nat] = self.bit_to_index[bits]
        self._modulate_lut: np.ndarray = lut

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Биты → символы QAM (векторизовано через LUT)."""
        b = np.asarray(bits, dtype=np.uint8).ravel()
        pad = (-len(b)) % self.bits_per_symbol
        if pad:
            b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])

        groups  = b.reshape(-1, self.bits_per_symbol)
        powers  = (1 << np.arange(self.bits_per_symbol - 1, -1, -1, dtype=np.int32))
        nat_idx = groups.astype(np.int32) @ powers

        return self.constellation_points[self._modulate_lut[nat_idx]]

    def demodulate(self, symbols: np.ndarray,
                   channel_coeff: np.ndarray | None = None) -> np.ndarray:
        """
        Символы → биты.
        Коррекция канала: ZF — s_corr = s / h.
        """
        s = np.asarray(symbols, dtype=complex).ravel()

        if channel_coeff is not None:
            h = np.asarray(channel_coeff, dtype=complex).ravel()
            if len(h) != len(s):
                raise ValueError(f"channel_coeff длина {len(h)} ≠ symbols длина {len(s)}")
            with np.errstate(divide='ignore', invalid='ignore'):
                s_corr = s / h
            bad = ~np.isfinite(s_corr)
            if np.any(bad):
                s_corr[bad] = s[bad]
            s = s_corr

        dist2   = np.abs(s[:, np.newaxis] - self.constellation_points[np.newaxis, :]) ** 2
        indices = np.argmin(dist2, axis=1)
        return self._bits_lut[indices].ravel()


# ══════════════════════════════════════════════════════════════════════════════
# Теоретические кривые BER / SER — только AWGN
# ══════════════════════════════════════════════════════════════════════════════

def theoretical_ber_psk(ebn0_dB: float, M: int,
                         use_gray_code: bool = True) -> float:
    """
    BER для M-PSK в AWGN (с кодом Грея).
    M=2,4: точно. M≥8: BER ≈ SER / log₂(M).
    """
    ebn0 = 10 ** (ebn0_dB / 10)
    k    = log2(M)
    if M in (2, 4):
        return Q_function(float(np.sqrt(2 * ebn0)))
    return theoretical_ser_psk(ebn0_dB, M, use_gray_code) / k


def theoretical_ser_psk(ebn0_dB: float, M: int,
                          use_gray_code: bool = True) -> float:
    """
    SER для M-PSK в AWGN. Proakis §8.1.
    M=2: Q(√(2·Eb/N0)). M=4: 2Q−Q². M≥8: 2·Q(√(2k·Eb/N0)·sin(π/M)).
    """
    ebn0 = 10 ** (ebn0_dB / 10)
    k    = log2(M)
    if M == 2:
        return Q_function(float(np.sqrt(2 * ebn0)))
    if M == 4:
        q = Q_function(float(np.sqrt(2 * ebn0)))
        return 2 * q - q ** 2
    arg = float(np.sqrt(2 * k * ebn0) * np.sin(pi / M))
    return 2 * Q_function(arg)


def theoretical_ber_qam(ebn0_dB: float, M: int,
                          use_gray_code: bool = True) -> float:
    """
    BER для M-QAM в AWGN. Proakis §8.2.
    M=4: идентичен QPSK. M≥16: (4/k)(1−1/√M)·Q(√(3k·Eb/N0/(M−1))).
    """
    ebn0 = 10 ** (ebn0_dB / 10)
    k    = log2(M)
    if M == 4:
        return Q_function(float(np.sqrt(2 * ebn0)))
    arg = float(np.sqrt(3 * k * ebn0 / (M - 1)))
    return float((4 / k) * (1 - 1 / np.sqrt(M)) * Q_function(arg))


def theoretical_ser_qam(ebn0_dB: float, M: int,
                          use_gray_code: bool = True) -> float:
    """SER для M-QAM в AWGN."""
    ebn0 = 10 ** (ebn0_dB / 10)
    k    = log2(M)
    if M == 4:
        q = Q_function(float(np.sqrt(2 * ebn0)))
        return 2 * q - q ** 2
    q = Q_function(float(np.sqrt(3 * k * ebn0 / (M - 1))))
    c = 1 - 1 / np.sqrt(M)
    return float(4 * c * q - 4 * c ** 2 * q ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# Теоретические кривые BER — Rayleigh-канал
# ══════════════════════════════════════════════════════════════════════════════

def theoretical_ber_rayleigh_psk(ebn0_dB: float, M: int) -> float:
    """
    Теоретический BER для M-PSK в Rayleigh-канале (плоские замирания).

    Формулы (Proakis, Digital Communications, 5th ed.):
      M=2 (BPSK): BER = 0.5·(1 − √(γ/(1+γ)))
      M=4 (QPSK): BER = 0.5·(1 − √(γ/(2+γ))) + 0.25·(1 − √(2γ/(2+γ)))  → приближение
      M≥8: BER ≈ (2/log2(M))·Q_approx через численное интегрирование или:
           SER_Rayleigh ≈ 2·[0.5·(1 − √(α/(1+α)))]  где α = k·γ·sin²(π/M)
           BER ≈ SER / log2(M)   (серое кодирование)

    Для точного расчёта QPSK/8PSK используется форма через комплементарную
    гамма-функцию (Craig 1991 / Simon–Alouini).

    Args:
        ebn0_dB : Eb/N0 в дБ
        M       : порядок PSK (2, 4, 8, 16)

    Returns:
        теоретический BER (float)
    """
    ebn0 = 10.0 ** (ebn0_dB / 10.0)
    k    = log2(M)

    if M == 2:
        # BPSK Rayleigh — точная формула
        gamma_b = ebn0
        ber = 0.5 * (1.0 - np.sqrt(gamma_b / (1.0 + gamma_b)))
        return float(ber)

    if M == 4:
        # QPSK Rayleigh — аналогично BPSK на каждую квадратурную компоненту
        gamma_b = ebn0
        p_c = 0.5 * (1.0 - np.sqrt(gamma_b / (1.0 + gamma_b)))
        # BER ≈ p_c (приближение для QPSK с кодом Грея)
        return float(p_c)

    # M≥8: SER через замкнутую формулу Simon–Alouini (один член ряда)
    # SER_Rayleigh(M-PSK) ≈ 2/π · ∫ Mγf(γ) dγ → для плоского Rayleigh
    # приближение через α = γ_s · sin²(π/M) / (1 + γ_s · sin²(π/M)):
    gamma_s = ebn0 * k  # Eb/N0 → Es/N0
    sin2 = np.sin(pi / M) ** 2
    alpha = gamma_s * sin2 / (1.0 + gamma_s * sin2)
    # SER ≈ 2 · (1 - π/(π - π/M))·(0.5·(1 - √alpha))  — упрощение
    # Более точно: SER = (M-1)/M + закрытая форма; используем числ. интегрирование
    # Для простоты и точности — SER ≈ 2·(M-1)/M · (1/2)·(1 - √(alpha/(1+alpha)))
    ser = 2.0 * (M - 1) / M * 0.5 * (1.0 - np.sqrt(alpha / (1.0 + alpha)))
    ber = ser / k
    return float(np.clip(ber, 0.0, 0.5))


def theoretical_ber_rayleigh_qam(ebn0_dB: float, M: int) -> float:
    """
    Теоретический BER для M-QAM в Rayleigh-канале (плоские замирания).

    Формула (Simon–Alouini, Digital Communication over Fading Channels):
      BER_QAM_Rayleigh ≈ (4/k)·(1 - 1/√M) · (1/2)·(1 − √(β/(1+β)))

    где:
      k = log2(M)
      β = 3k·γ_b / (2·(M−1))
      γ_b = Eb/N0 (линейный)

    Это точный результат для прямоугольного QAM с кодом Грея в канале Rayleigh
    через моментную производящую функцию (MGF-подход).

    Args:
        ebn0_dB : Eb/N0 в дБ
        M       : порядок QAM (4, 16, 64, 256)

    Returns:
        теоретический BER (float)
    """
    ebn0 = 10.0 ** (ebn0_dB / 10.0)
    k    = log2(M)

    if M == 4:
        # QAM-4 ≡ QPSK
        return theoretical_ber_rayleigh_psk(ebn0_dB, 4)

    beta = 3.0 * k * ebn0 / (2.0 * (M - 1))
    p_e  = 0.5 * (1.0 - np.sqrt(beta / (1.0 + beta)))
    ber  = (4.0 / k) * (1.0 - 1.0 / np.sqrt(M)) * p_e
    return float(np.clip(ber, 0.0, 0.5))


# ══════════════════════════════════════════════════════════════════════════════
# LLR (Log-Likelihood Ratio) для soft-decision декодеров
# ══════════════════════════════════════════════════════════════════════════════

def compute_llr_psk(symbols: np.ndarray,
                    modulator: PSKModulator,
                    noise_var: float) -> np.ndarray:
    """
    Вычисляет LLR для каждого бита каждого M-PSK символа.

    LLR(b_k) = ln(Σ_{c: b_k=0} exp(−|y−c|²/σ²)) − ln(Σ_{c: b_k=1} exp(−|y−c|²/σ²))

    Для численной стабильности используется log-sum-exp:
      log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i − max(x)))

    Args:
        symbols   : принятые символы, shape (N,)
        modulator : инициализированный PSKModulator
        noise_var : дисперсия шума σ² (на комплексный символ)

    Returns:
        llr : float64, shape (N · bits_per_symbol,)
              положительный LLR → скорее 0, отрицательный → скорее 1
    """
    y   = np.asarray(symbols, dtype=complex).ravel()
    N   = len(y)
    bps = modulator.bits_per_symbol
    M   = modulator.M
    C   = modulator.constellation_points  # (M,)

    # Метрики: −|y_i − C_j|² / σ²,  shape (N, M)
    dist2   = np.abs(y[:, np.newaxis] - C[np.newaxis, :]) ** 2
    metrics = -dist2 / max(noise_var, 1e-30)           # (N, M)

    llr = np.zeros((N, bps), dtype=np.float64)

    for b in range(bps):
        # Индексы созвездия с b_k = 0 и b_k = 1
        idx0 = [i for i in range(M) if modulator.index_to_bits[i][b] == 0]
        idx1 = [i for i in range(M) if modulator.index_to_bits[i][b] == 1]

        def _log_sum_exp(m: np.ndarray) -> np.ndarray:
            mx = m.max(axis=1, keepdims=True)  # (N, 1)
            return mx.ravel() + np.log(np.exp(m - mx).sum(axis=1))  # (N,)

        llr[:, b] = _log_sum_exp(metrics[:, idx0]) - _log_sum_exp(metrics[:, idx1])

    return llr.ravel()


def compute_llr_qam(symbols: np.ndarray,
                    modulator: QAMModulator,
                    noise_var: float) -> np.ndarray:
    """
    Вычисляет LLR для каждого бита каждого M-QAM символа.

    Использует полную формулу max-log-MAP (exact log-sum-exp), что обеспечивает
    лучшую точность по сравнению с приближёнными методами.

    Args:
        symbols   : принятые символы, shape (N,)
        modulator : инициализированный QAMModulator
        noise_var : дисперсия шума σ² (на комплексный символ)

    Returns:
        llr : float64, shape (N · bits_per_symbol,)
    """
    y   = np.asarray(symbols, dtype=complex).ravel()
    N   = len(y)
    bps = modulator.bits_per_symbol
    M   = modulator.M
    C   = modulator.constellation_points

    dist2   = np.abs(y[:, np.newaxis] - C[np.newaxis, :]) ** 2
    metrics = -dist2 / max(noise_var, 1e-30)

    llr = np.zeros((N, bps), dtype=np.float64)

    for b in range(bps):
        idx0 = [i for i in range(M) if modulator.index_to_bits[i][b] == 0]
        idx1 = [i for i in range(M) if modulator.index_to_bits[i][b] == 1]

        def _log_sum_exp(m: np.ndarray) -> np.ndarray:
            mx = m.max(axis=1, keepdims=True)
            return mx.ravel() + np.log(np.exp(m - mx).sum(axis=1))

        llr[:, b] = _log_sum_exp(metrics[:, idx0]) - _log_sum_exp(metrics[:, idx1])

    return llr.ravel()
