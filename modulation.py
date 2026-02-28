import numpy as np
import scipy.special as sp
from math import pi, log2, sqrt
from typing import Tuple, Optional
from functools import lru_cache


class PSKModulator:
    """
    M-PSK modulator/demodulator with optional Gray coding and channel equalization.
    """

    def __init__(self, M: int = 8, use_gray_code: bool = True):
        assert M in (2, 4, 8, 16), "Поддерживаются только M=2,4,8,16"
        self.M = M
        self.bits_per_symbol = int(log2(M))
        self.use_gray_code = use_gray_code
        self._create_constellation()

    def _create_constellation(self):
        angles = np.linspace(0, 2 * pi, self.M, endpoint=False)
        if self.M == 4:
            angles = angles + pi / 4
        self.constellation_points = np.exp(1j * angles)

        self.bit_to_index = {}
        self.index_to_bits = {}
        for i in range(self.M):
            if self.use_gray_code:
                gray = i ^ (i >> 1)
                code_val = gray
            else:
                code_val = i
            bits = tuple(((code_val >> (self.bits_per_symbol - 1 - j)) & 1)
                         for j in range(self.bits_per_symbol))
            self.bit_to_index[bits] = i
            self.index_to_bits[i] = bits

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        b = np.array(bits, dtype=int).flatten()
        pad = (self.bits_per_symbol - (len(b) % self.bits_per_symbol)) % self.bits_per_symbol
        if pad:
            b = np.concatenate([b, np.zeros(pad, dtype=int)])
        groups = b.reshape(-1, self.bits_per_symbol)
        symbols = np.empty(len(groups), dtype=complex)
        for i, g in enumerate(groups):
            idx = self.bit_to_index.get(tuple(g.tolist()), 0)
            symbols[i] = self.constellation_points[idx]
        return symbols

    def demodulate(self, symbols: np.ndarray,
                   channel_coeff: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Демодуляция с возможностью коррекции по известному коэффициенту канала.

        Args:
            symbols: принятые комплексные символы
            channel_coeff: комплексные коэффициенты канала для каждого символа (должен быть той же длины)
                           Если None, коррекция не выполняется.
        Returns:
            биты
        """
        s = np.array(symbols, dtype=complex).flatten()
        if channel_coeff is not None:
            coeff = np.array(channel_coeff, dtype=complex).flatten()
            if len(coeff) != len(s):
                raise ValueError("Длина channel_coeff должна совпадать с длиной symbols")
            # Коррекция: поворачиваем фазу обратно (для PSK достаточно убрать фазу)
            # Для сохранения амплитуды не делим, а умножаем на exp(-j*arg(h))
            phase = np.angle(coeff)
            s_corrected = s * np.exp(-1j * phase)
        else:
            s_corrected = s

        pts = self.constellation_points[np.newaxis, :]
        sym_exp = s_corrected[:, np.newaxis]
        dist2 = np.abs(sym_exp - pts) ** 2
        indices = np.argmin(dist2, axis=1)

        bits = []
        for idx in indices:
            bits.extend(self.index_to_bits[int(idx)])
        return np.array(bits, dtype=int)


class QAMModulator:
    """
    M-QAM modulator/demodulator with Gray coding and channel equalization.
    """

    def __init__(self, M: int = 16, use_gray_code: bool = True):
        assert M in (4, 16, 64), "Поддерживаются только M=4,16,64"
        self.M = M
        self.bits_per_symbol = int(log2(M))
        self.use_gray_code = use_gray_code
        self._create_constellation()

    def _create_constellation(self):
        side = int(np.sqrt(self.M))
        amplitudes = np.arange(-(side - 1), side, 2)
        gray_map = np.array([i ^ (i >> 1) for i in range(side)])

        I, Q = np.meshgrid(amplitudes, amplitudes)
        points = (I + 1j * Q).flatten()
        avg_power = np.mean(np.abs(points) ** 2)
        self.constellation_points = points / np.sqrt(avg_power)

        self.bit_to_index = {}
        self.index_to_bits = {}
        for i in range(self.M):
            idx_Q = i // side
            idx_I = i % side
            if self.use_gray_code:
                gray_I = gray_map[idx_I]
                gray_Q = gray_map[idx_Q]
            else:
                gray_I = idx_I
                gray_Q = idx_Q
            bits = []
            for j in range(self.bits_per_symbol // 2 - 1, -1, -1):
                bits.append((gray_I >> j) & 1)
            for j in range(self.bits_per_symbol // 2 - 1, -1, -1):
                bits.append((gray_Q >> j) & 1)
            bits_tuple = tuple(bits)
            self.bit_to_index[bits_tuple] = i
            self.index_to_bits[i] = bits_tuple

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        b = np.array(bits, dtype=int).flatten()
        pad = (self.bits_per_symbol - (len(b) % self.bits_per_symbol)) % self.bits_per_symbol
        if pad:
            b = np.concatenate([b, np.zeros(pad, dtype=int)])
        groups = b.reshape(-1, self.bits_per_symbol)
        symbols = np.empty(len(groups), dtype=complex)
        for i, g in enumerate(groups):
            idx = self.bit_to_index.get(tuple(g.tolist()), 0)
            symbols[i] = self.constellation_points[idx]
        return symbols

    def demodulate(self, symbols: np.ndarray,
                   channel_coeff: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Демодуляция с коррекцией канала (деление на комплексный коэффициент).
        """
        s = np.array(symbols, dtype=complex).flatten()
        if channel_coeff is not None:
            coeff = np.array(channel_coeff, dtype=complex).flatten()
            if len(coeff) != len(s):
                raise ValueError("Длина channel_coeff должна совпадать с длиной symbols")
            # Избегаем деления на ноль
            with np.errstate(divide='ignore', invalid='ignore'):
                s_corrected = s / coeff
            # Заменяем inf/NaN на исходные символы (если coeff == 0, то информация потеряна)
            s_corrected = np.where(np.isfinite(s_corrected), s_corrected, s)
        else:
            s_corrected = s

        pts = self.constellation_points[np.newaxis, :]
        sym_exp = s_corrected[:, np.newaxis]
        dist2 = np.abs(sym_exp - pts) ** 2
        indices = np.argmin(dist2, axis=1)

        bits = []
        for idx in indices:
            bits.extend(self.index_to_bits[int(idx)])
        return np.array(bits, dtype=int)


@lru_cache(maxsize=256)
def Q_function(x: float) -> float:
    return 0.5 * sp.erfc(x / np.sqrt(2))


def theoretical_ber_psk(ebn0_dB: float, M: int, use_gray_code: bool = True) -> float:
    ebn0 = 10 ** (ebn0_dB / 10)
    k = log2(M)
    if M == 2:
        return Q_function(np.sqrt(2 * ebn0))
    elif M == 4:
        return Q_function(np.sqrt(2 * ebn0))
    else:
        ser = theoretical_ser_psk(ebn0_dB, M, use_gray_code)
        return ser / k


def theoretical_ser_psk(ebn0_dB: float, M: int, use_gray_code: bool = True) -> float:
    ebn0 = 10 ** (ebn0_dB / 10)
    k = log2(M)
    if M == 2:
        return Q_function(np.sqrt(2 * ebn0))
    elif M == 4:
        q = Q_function(np.sqrt(2 * ebn0))
        return 2 * q - q ** 2
    else:
        arg = np.sqrt(2 * k * ebn0) * np.sin(pi / M)
        return 2 * Q_function(arg)


def theoretical_ber_qam(ebn0_dB: float, M: int, use_gray_code: bool = True) -> float:
    ebn0 = 10 ** (ebn0_dB / 10)
    k = log2(M)
    if M == 4:
        return Q_function(np.sqrt(2 * ebn0))
    else:
        return (4 / k) * (1 - 1 / np.sqrt(M)) * Q_function(
            np.sqrt(3 * k * ebn0 / (M - 1))
        )


def theoretical_ser_qam(ebn0_dB: float, M: int, use_gray_code: bool = True) -> float:
    ebn0 = 10 ** (ebn0_dB / 10)
    k = log2(M)
    if M == 4:
        q = Q_function(np.sqrt(2 * ebn0))
        return 2 * q - q ** 2
    else:
        arg = np.sqrt(3 * k * ebn0 / (M - 1))
        q = Q_function(arg)
        return 4 * (1 - 1 / np.sqrt(M)) * q - 4 * (1 - 1 / np.sqrt(M)) ** 2 * q ** 2