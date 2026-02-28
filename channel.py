"""
Модели каналов связи с поддержкой когерентного приёма.
"""

import numpy as np
from typing import Dict, Optional
import numpy as np
import logging
from typing import Dict, Optional, Tuple


logger = logging.getLogger(__name__)


class ChannelModel:
    """Абстрактный базовый класс для всех моделей каналов"""

    def __init__(self, config: Dict):
        self.config = config
        self.name = "Channel"

    def apply(self, tx_symbols: np.ndarray, snr_linear: float,
              fs: float = 1.0) -> np.ndarray:
        """Применяет эффекты канала и возвращает искажённые символы"""
        raise NotImplementedError

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Применяет эффекты канала и возвращает (искажённые символы, комплексные коэффициенты).
        По умолчанию коэффициент = 1 для всех символов (аддитивные каналы).
        """
        rx = self.apply(tx_symbols, snr_linear, fs)
        coeff = np.ones(len(tx_symbols), dtype=complex)
        return rx, coeff


class AWGNChannel(ChannelModel):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "AWGN"

    def apply(self, tx_symbols: np.ndarray, snr_linear: float, fs: float = 1.0) -> np.ndarray:
        if not self.config.get("enabled", True):
            return tx_symbols.copy()
        sigma = 1.0 / np.sqrt(2 * snr_linear)
        noise = sigma * (np.random.randn(len(tx_symbols)) + 1j * np.random.randn(len(tx_symbols)))
        return tx_symbols + noise

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        rx = self.apply(tx_symbols, snr_linear, fs)
        coeff = np.ones(len(tx_symbols), dtype=complex)
        return rx, coeff


class RayleighFadingChannel(ChannelModel):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Rayleigh Fading"
        self.num_rays = config.get("num_rays", 4)
        self.normalized_doppler = config.get("normalized_doppler", 0.01)

        # Углы прихода
        self.angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)

        # Комплексные коэффициенты лучей (стационарные)
        power_per_ray = 1.0 / self.num_rays
        std_per_ray = np.sqrt(power_per_ray / 2)
        self.ray_coeff = (std_per_ray * np.random.randn(self.num_rays) +
                          1j * std_per_ray * np.random.randn(self.num_rays))

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False):
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        rx_symbols = np.zeros_like(tx_symbols)
        h = np.zeros(len(tx_symbols), dtype=complex)

        for n in range(len(tx_symbols)):
            coeff_n = 0.0j
            for k in range(self.num_rays):
                doppler_phase = 2 * np.pi * self.normalized_doppler * n * np.cos(self.angles[k])
                coeff_n += self.ray_coeff[k] * np.exp(1j * doppler_phase)
            h[n] = coeff_n
            rx_symbols[n] = coeff_n * tx_symbols[n]

        # Добавляем AWGN
        sigma = 1.0 / np.sqrt(2 * snr_linear)
        noise = sigma * (np.random.randn(len(tx_symbols)) + 1j * np.random.randn(len(tx_symbols)))
        rx_symbols += noise

        return rx_symbols, h

    def apply(self, tx_symbols: np.ndarray, snr_linear: float, fs: float = 1.0) -> np.ndarray:
        rx, _ = self.apply_with_coeff(tx_symbols, snr_linear, fs)
        return rx


class RicianFadingChannel(ChannelModel):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Rician Fading"
        self.num_rays = config.get("num_rays", 4)
        self.normalized_doppler = config.get("normalized_doppler", 0.01)
        self.rician_factor_k = config.get("rician_factor_k", 3.0)

        self.p_los = self.rician_factor_k / (self.rician_factor_k + 1)
        self.p_nlos = 1 / (self.rician_factor_k + 1)

        self.angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)

        # LOS компонента (фиксированная амплитуда)
        self.los_amp = np.sqrt(self.p_los)

        # NLOS компоненты
        power_per_ray = self.p_nlos / self.num_rays
        std_per_ray = np.sqrt(power_per_ray / 2)
        self.nlos_coeff = (std_per_ray * np.random.randn(self.num_rays) +
                           1j * std_per_ray * np.random.randn(self.num_rays))

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False):
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        rx_symbols = np.zeros_like(tx_symbols)
        h = np.zeros(len(tx_symbols), dtype=complex)

        for n in range(len(tx_symbols)):
            # LOS с доплером
            los = self.los_amp * np.exp(1j * 2 * np.pi * self.normalized_doppler * n)
            nlos = 0.0j
            for k in range(self.num_rays):
                doppler_phase = 2 * np.pi * self.normalized_doppler * n * np.cos(self.angles[k])
                nlos += self.nlos_coeff[k] * np.exp(1j * doppler_phase)
            h_n = los + nlos
            h[n] = h_n
            rx_symbols[n] = h_n * tx_symbols[n]

        # AWGN
        sigma = 1.0 / np.sqrt(2 * snr_linear)
        noise = sigma * (np.random.randn(len(tx_symbols)) + 1j * np.random.randn(len(tx_symbols)))
        rx_symbols += noise

        return rx_symbols, h

    def apply(self, tx_symbols: np.ndarray, snr_linear: float, fs: float = 1.0) -> np.ndarray:
        rx, _ = self.apply_with_coeff(tx_symbols, snr_linear, fs)
        return rx


class PhaseNoiseChannel(ChannelModel):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Phase Noise"
        self.phase_noise_variance = config.get("phase_noise_variance", 0.001)

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False):
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        phase_increments = np.sqrt(self.phase_noise_variance) * np.random.randn(len(tx_symbols))
        phase_noise = np.cumsum(phase_increments)
        phase_rotation = np.exp(1j * phase_noise)

        rx_symbols = tx_symbols * phase_rotation
        # Коэффициент канала – поворот фазы
        h = phase_rotation

        sigma = 1.0 / np.sqrt(2 * snr_linear)
        noise = sigma * (np.random.randn(len(tx_symbols)) + 1j * np.random.randn(len(tx_symbols)))
        rx_symbols += noise

        return rx_symbols, h

    def apply(self, tx_symbols: np.ndarray, snr_linear: float, fs: float = 1.0) -> np.ndarray:
        rx, _ = self.apply_with_coeff(tx_symbols, snr_linear, fs)
        return rx


class FrequencyOffsetChannel(ChannelModel):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Frequency Offset"
        self.normalized_freq_offset = config.get("normalized_freq_offset", 0.0)

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False):
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        phase = 2 * np.pi * self.normalized_freq_offset * np.arange(len(tx_symbols))
        offset = np.exp(1j * phase)

        rx_symbols = tx_symbols * offset
        h = offset

        sigma = 1.0 / np.sqrt(2 * snr_linear)
        noise = sigma * (np.random.randn(len(tx_symbols)) + 1j * np.random.randn(len(tx_symbols)))
        rx_symbols += noise

        return rx_symbols, h

    def apply(self, tx_symbols: np.ndarray, snr_linear: float, fs: float = 1.0) -> np.ndarray:
        rx, _ = self.apply_with_coeff(tx_symbols, snr_linear, fs)
        return rx


class TimingOffsetChannel(ChannelModel):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Timing Offset"
        self.timing_offset_range = config.get("timing_offset_range", 0.0)

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False) or self.timing_offset_range == 0:
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        timing_offsets = np.random.uniform(-self.timing_offset_range,
                                           self.timing_offset_range,
                                           len(tx_symbols))
        phase_shifts = 2 * np.pi * timing_offsets
        effect = np.exp(1j * phase_shifts)

        rx_symbols = tx_symbols * effect
        h = effect

        sigma = 1.0 / np.sqrt(2 * snr_linear)
        noise = sigma * (np.random.randn(len(tx_symbols)) + 1j * np.random.randn(len(tx_symbols)))
        rx_symbols += noise

        return rx_symbols, h

    def apply(self, tx_symbols: np.ndarray, snr_linear: float, fs: float = 1.0) -> np.ndarray:
        rx, _ = self.apply_with_coeff(tx_symbols, snr_linear, fs)
        return rx


class ImpulseNoiseChannel(ChannelModel):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Impulse Noise"
        self.impulse_probability = config.get("impulse_probability", 0.001)
        self.impulse_amplitude_sigma = config.get("impulse_amplitude_sigma", 10.0)
        self.impulse_width_from = config.get("impulse_width_from", 1)
        self.impulse_width_to = config.get("impulse_width_to", 5)

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False):
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        rx_symbols = tx_symbols.copy()
        sigma_awgn = 1.0 / np.sqrt(2 * snr_linear)

        # AWGN
        noise = sigma_awgn * (np.random.randn(len(tx_symbols)) + 1j * np.random.randn(len(tx_symbols)))
        rx_symbols += noise

        # Импульсы
        n = 0
        while n < len(tx_symbols):
            if np.random.rand() < self.impulse_probability:
                pulse_width = np.random.randint(self.impulse_width_from, self.impulse_width_to + 1)
                pulse_amplitude = self.impulse_amplitude_sigma * sigma_awgn * np.exp(
                    1j * 2 * np.pi * np.random.rand()
                ) * np.abs(np.random.randn())
                end = min(n + pulse_width, len(tx_symbols))
                rx_symbols[n:end] += pulse_amplitude
                n = end
            else:
                n += 1

        # Коэффициент канала для импульсного шума не определён (аддитивный),
        # поэтому возвращаем 1.
        h = np.ones(len(tx_symbols), dtype=complex)
        return rx_symbols, h

    def apply(self, tx_symbols: np.ndarray, snr_linear: float, fs: float = 1.0) -> np.ndarray:
        rx, _ = self.apply_with_coeff(tx_symbols, snr_linear, fs)
        return rx


class CompositeChannelModel:
    """
    Композитный канал: применяет несколько эффектов последовательно.
    Сохраняет совокупный комплексный коэффициент канала.
    """

    def __init__(self, channel_config: Dict):
        self.channel_config = channel_config
        self.channels = []

        if channel_config['rayleigh'].get('enabled', False):
            self.channels.append(RayleighFadingChannel(channel_config['rayleigh']))
        elif channel_config['rician'].get('enabled', False):
            self.channels.append(RicianFadingChannel(channel_config['rician']))
        else:
            self.channels.append(AWGNChannel(channel_config['awgn']))

        if channel_config['frequency_offset'].get('enabled', False):
            self.channels.append(FrequencyOffsetChannel(channel_config['frequency_offset']))
        if channel_config['timing_offset'].get('enabled', False):
            self.channels.append(TimingOffsetChannel(channel_config['timing_offset']))
        if channel_config['phase_noise'].get('enabled', False):
            self.channels.append(PhaseNoiseChannel(channel_config['phase_noise']))
        if channel_config['impulse_noise'].get('enabled', False):
            self.channels.append(ImpulseNoiseChannel(channel_config['impulse_noise']))

    def apply(self, tx_symbols: np.ndarray, snr_linear: float) -> np.ndarray:
        """Только искажённые символы (без коэффициентов)"""
        rx = tx_symbols.copy()
        for ch in self.channels:
            rx = ch.apply(rx, snr_linear)
        return rx

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Применяет все каналы и возвращает (rx_symbols, channel_coeff).
        channel_coeff – совокупный комплексный коэффициент для каждого символа.
        """
        rx = tx_symbols.copy()
        # Начальный коэффициент = 1 для всех символов
        h_total = np.ones(len(tx_symbols), dtype=complex)

        for ch in self.channels:
            # Каждый канал может изменить как символы, так и коэффициент
            rx, h_partial = ch.apply_with_coeff(rx, snr_linear)
            h_total *= h_partial   # последовательное умножение коэффициентов

        return rx, h_total

    def get_channel_names(self) -> list:
        return [ch.name for ch in self.channels]