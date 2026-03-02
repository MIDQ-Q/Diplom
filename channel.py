"""
Модели каналов связи для беспроводных систем.

Поддерживаемые модели:
  - AWGNChannel              : аддитивный белый гауссовский шум
  - RayleighFadingChannel    : частотно-плоские замирания (модель Jakes)
  - MultipathChannel         : многолучёвое распространение с ISI (FIR-канал + Jakes на каждом отводе)
  - PhaseNoiseChannel        : фазовый шум (случайное блуждание, модель Wiener)
  - FrequencyOffsetChannel   : сдвиг несущей частоты (CFO) с дрейфом
  - TimingOffsetChannel      : тайминговая ошибка
  - ImpulseNoiseChannel      : импульсные помехи (модель Бернулли–Гаусса)
  - ShadowingChannel         : медленные замирания (log-normal shadowing)
  - CompositeChannelModel    : последовательное объединение каналов с единым AWGN

Примечание (2 марта):
  - RicianFadingChannel удалён — фокус модели: AWGN + Rayleigh.
  - RayleighFadingChannel: параметр n_rays теперь по умолчанию 16 (было 4).
  - CompositeChannelModel: поддерживает настраиваемое число лучей Rayleigh через n_rays.

Ключевые особенности:
  - Нормировка мощности канала: E[|h|²] = 1 везде, где это применимо
  - Единый источник AWGN в CompositeChannelModel (нет накопления шума)
  - Модель Jakes с случайными начальными фазами (реалистичный Доплер)
  - MultipathChannel реализован как FIR-фильтр с независимыми Jakes-замираниями на каждом отводе
  - Все каналы возвращают (rx_symbols, channel_coeff) через apply_with_coeff()
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _awgn_noise(n: int, snr_linear: float, signal_power: float = 1.0) -> np.ndarray:
    """
    Генерирует комплексный AWGN с учётом реальной мощности сигнала.

    sigma² = signal_power / (2 * snr_linear)   [на каждую компоненту I и Q]

    Args:
        n            : количество отсчётов
        snr_linear   : линейное SNR (не дБ)
        signal_power : средняя мощность символов (E[|s|²])

    Returns:
        комплексный шум длиной n
    """
    sigma = np.sqrt(signal_power / (2.0 * snr_linear))
    return sigma * (np.random.randn(n) + 1j * np.random.randn(n))


def _jakes_channel(n_symbols: int,
                   n_rays: int,
                   normalized_doppler: float) -> np.ndarray:
    """
    Генерирует реализацию замираний по модели Jakes (сумма синусоид).

    Модель:
        h[n] = (1/√N) · Σ_{k=0}^{N-1}  exp(j·(2π·fd·cos(α_k)·n + φ_k))

    где:
        α_k  — углы прихода: 2π·k/N  (равномерно по кругу)
        φ_k  — случайные начальные фазы ~ Uniform[0, 2π)
        fd   — нормированная частота Доплера (fd = f_D / f_sym)

    Нормировка гарантирует E[|h|²] = 1.

    Args:
        n_symbols          : длина реализации (в символах)
        n_rays             : число лучей N (рекомендуется 8–64)
        normalized_doppler : f_D / f_sym (типично 0.001 – 0.05)

    Returns:
        h : ndarray, dtype=complex, shape=(n_symbols,)
    """
    angles = 2.0 * np.pi * np.arange(n_rays) / n_rays          # α_k
    init_phases = 2.0 * np.pi * np.random.rand(n_rays)          # φ_k ~ U[0,2π)
    n_idx = np.arange(n_symbols)                                 # [0, 1, ..., N-1]

    # Матрица фаз: shape (n_symbols, n_rays)
    doppler_phases = (2.0 * np.pi * normalized_doppler
                      * np.outer(n_idx, np.cos(angles)))         # 2π·fd·cos(α_k)·n
    total_phases = doppler_phases + init_phases[np.newaxis, :]   # + φ_k

    h = np.sum(np.exp(1j * total_phases), axis=1) / np.sqrt(n_rays)
    return h


# ---------------------------------------------------------------------------
# Базовый класс
# ---------------------------------------------------------------------------

class ChannelModel:
    """Абстрактный базовый класс для всех моделей каналов."""

    def __init__(self, config: Dict):
        self.config = config
        self.name = "Channel"

    def apply(self, tx_symbols: np.ndarray, snr_linear: float,
              add_noise: bool = True) -> np.ndarray:
        """Применяет канал, возвращает только искажённые символы."""
        rx, _ = self.apply_with_coeff(tx_symbols, snr_linear, add_noise)
        return rx

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применяет канал.

        Args:
            tx_symbols : переданные символы
            snr_linear : линейное Eb/N0
            add_noise  : добавлять ли AWGN (False используется в CompositeChannelModel,
                         чтобы AWGN добавлялся единожды)

        Returns:
            (rx_symbols, channel_coeff)
            channel_coeff — покомпонентный комплексный коэффициент для каждого символа.
            Для плоских каналов shape=(n,), для многолучёвых — усреднённый эквивалент.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# AWGN
# ---------------------------------------------------------------------------

class AWGNChannel(ChannelModel):
    """
    Аддитивный белый гауссовский шум.

    sigma² = P_signal / (2 · SNR_linear)
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "AWGN"

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", True):
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        signal_power = float(np.mean(np.abs(tx_symbols) ** 2))
        if signal_power == 0.0:
            signal_power = 1.0

        rx = tx_symbols.copy()
        if add_noise:
            rx = rx + _awgn_noise(len(tx_symbols), snr_linear, signal_power)

        coeff = np.ones(len(tx_symbols), dtype=complex)
        return rx, coeff


# ---------------------------------------------------------------------------
# Rayleigh Fading (Jakes)
# ---------------------------------------------------------------------------

class RayleighFadingChannel(ChannelModel):
    """
    Частотно-плоский канал Рэлея с доплеровским спектром по модели Jakes.

    Параметры config:
        enabled            : bool
        n_rays             : int, число лучей (по умолчанию 16)
        normalized_doppler : float, f_D/f_sym (по умолчанию 0.01)

    Канал нормирован: E[|h|²] = 1.
    Новая реализация h генерируется при каждом вызове apply_with_coeff().
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Rayleigh Fading"
        self.n_rays = config.get("n_rays", 16)
        self.normalized_doppler = config.get("normalized_doppler", 0.01)

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False):
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        n = len(tx_symbols)
        h = _jakes_channel(n, self.n_rays, self.normalized_doppler)

        rx = h * tx_symbols

        if add_noise:
            # Мощность сигнала после умножения на канал: E[|h·s|²] = E[|h|²]·E[|s|²] = E[|s|²]
            # (т.к. E[|h|²]=1), поэтому sigma считаем по исходной мощности символов
            signal_power = float(np.mean(np.abs(tx_symbols) ** 2))
            if signal_power == 0.0:
                signal_power = 1.0
            rx = rx + _awgn_noise(n, snr_linear, signal_power)

        return rx, h


# ---------------------------------------------------------------------------
# Multipath Channel (FIR + Jakes на каждом отводе)
# ---------------------------------------------------------------------------

class MultipathChannel(ChannelModel):
    """
    Многолучёвой канал с межсимвольной интерференцией (ISI).

    Модель:
        y[n] = Σ_{l=0}^{L-1}  h_l[n] · x[n - l]  +  w[n]

    где h_l[n] — комплексный коэффициент l-го отвода в момент n,
    моделируемый по Jakes с независимой реализацией для каждого отвода.

    Мощность отводов задаётся профилем PDP (Power Delay Profile).
    По умолчанию используется экспоненциальный PDP:
        p_l = exp(-l / decay)  ,  нормированный так, что Σ p_l = 1.

    Параметры config:
        enabled            : bool
        n_taps             : int, число отводов L (по умолчанию 6)
        normalized_doppler : float, f_D/f_sym (по умолчанию 0.01)
        n_rays             : int, лучей на отвод (по умолчанию 16)
        pdp_decay          : float, параметр экспоненциального затухания
                             (по умолчанию 1.0; больше → медленнее спадает)
        pdp_powers         : list[float] | None — явный PDP в линейном масштабе
                             (если задан, перекрывает pdp_decay и n_taps)

    Возвращаемый channel_coeff:
        Для совместимости с демодулятором возвращается h_0[n] — коэффициент
        нулевого (прямого) отвода. Полная коррекция ISI требует эквалайзера.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Multipath"
        self.n_taps = config.get("n_taps", 6)
        self.normalized_doppler = config.get("normalized_doppler", 0.01)
        self.n_rays = config.get("n_rays", 16)
        self.pdp_decay = config.get("pdp_decay", 1.0)

        # Профиль PDP
        pdp_explicit = config.get("pdp_powers", None)
        if pdp_explicit is not None:
            self.pdp = np.array(pdp_explicit, dtype=float)
            self.n_taps = len(self.pdp)
        else:
            taps = np.arange(self.n_taps, dtype=float)
            self.pdp = np.exp(-taps / self.pdp_decay)

        # Нормируем PDP: Σ p_l = 1  →  E[|h_total|²] = 1
        self.pdp = self.pdp / np.sum(self.pdp)
        logger.debug(f"MultipathChannel PDP (нормированный): {self.pdp}")

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False):
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        n = len(tx_symbols)
        rx = np.zeros(n, dtype=complex)

        # Генерируем коэффициенты каждого отвода и накапливаем свёртку
        h_taps = []
        for l in range(self.n_taps):
            tap_amp = np.sqrt(self.pdp[l])
            h_l = tap_amp * _jakes_channel(n, self.n_rays, self.normalized_doppler)
            h_taps.append(h_l)

            # Сдвиг сигнала на l символов (ISI)
            if l == 0:
                rx += h_l * tx_symbols
            else:
                rx[l:] += h_l[l:] * tx_symbols[:n - l]

        if add_noise:
            signal_power = float(np.mean(np.abs(tx_symbols) ** 2))
            if signal_power == 0.0:
                signal_power = 1.0
            rx = rx + _awgn_noise(n, snr_linear, signal_power)

        # Возвращаем h нулевого отвода как «эффективный» коэффициент
        return rx, h_taps[0]


# ---------------------------------------------------------------------------
# Phase Noise (модель Wiener / случайное блуждание)
# ---------------------------------------------------------------------------

class PhaseNoiseChannel(ChannelModel):
    """
    Фазовый шум на основе модели случайного блуждания (Wiener process).

    Физика: нестабильность гетеродина/генератора приводит к накапливающемуся
    случайному изменению фазы несущей.

    Модель:
        θ[n] = θ[n-1] + Δθ[n],   Δθ[n] ~ N(0, σ²_θ)
        y[n] = x[n] · exp(j·θ[n]) + w[n]

    Параметр σ²_θ (дисперсия приращений фазы) связан с linewidth генератора:
        σ²_θ = 2π · Δf / f_sym

    Параметры config:
        enabled             : bool
        phase_noise_std_deg : float, СКО приращения фазы в градусах (по умолчанию 1.0°)
                              Типичные значения: 0.5°–3° для реальных систем.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Phase Noise"
        std_deg = config.get("phase_noise_std_deg", 1.0)
        self.phase_noise_std = np.deg2rad(std_deg)

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False):
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        n = len(tx_symbols)

        # Случайное блуждание фазы
        delta_theta = self.phase_noise_std * np.random.randn(n)
        theta = np.cumsum(delta_theta)
        h = np.exp(1j * theta)

        rx = tx_symbols * h

        if add_noise:
            signal_power = float(np.mean(np.abs(tx_symbols) ** 2))
            if signal_power == 0.0:
                signal_power = 1.0
            rx = rx + _awgn_noise(n, snr_linear, signal_power)

        return rx, h


# ---------------------------------------------------------------------------
# Frequency Offset (CFO) с дрейфом
# ---------------------------------------------------------------------------

class FrequencyOffsetChannel(ChannelModel):
    """
    Сдвиг несущей частоты (Carrier Frequency Offset, CFO) с медленным дрейфом.

    Физика: расхождение тактовых генераторов передатчика и приёмника.
    Дрейф моделирует температурную нестабильность (TCXO/VCXO).

    Модель:
        φ_cfo[n] = 2π · Σ_{k=0}^{n} f_offset[k]
        y[n] = x[n] · exp(j·φ_cfo[n]) + w[n]

    f_offset[n] — нормированный CFO с медленным случайным блужданием:
        f_offset[n] = f_offset[n-1] + drift[n],  drift[n] ~ N(0, σ²_drift)

    Параметры config:
        enabled                  : bool
        normalized_freq_offset   : float, начальный нормированный CFO (f/f_sym),
                                   типично 1e-5 – 1e-3
        cfo_drift_std            : float, СКО шага дрейфа (по умолчанию 0),
                                   0 → постоянный CFO без дрейфа
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Frequency Offset"
        self.f0 = config.get("normalized_freq_offset", 0.0)
        self.drift_std = config.get("cfo_drift_std", 0.0)

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False):
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        n = len(tx_symbols)

        # Дрейф CFO
        if self.drift_std > 0:
            drift = self.drift_std * np.random.randn(n)
            freq = self.f0 + np.cumsum(drift)
        else:
            freq = np.full(n, self.f0)

        phase = 2.0 * np.pi * np.cumsum(freq)
        h = np.exp(1j * phase)
        rx = tx_symbols * h

        if add_noise:
            signal_power = float(np.mean(np.abs(tx_symbols) ** 2))
            if signal_power == 0.0:
                signal_power = 1.0
            rx = rx + _awgn_noise(n, snr_linear, signal_power)

        return rx, h


# ---------------------------------------------------------------------------
# Timing Offset
# ---------------------------------------------------------------------------

class TimingOffsetChannel(ChannelModel):
    """
    Тайминговая ошибка (дробный сдвиг момента выборки).

    Физика: несинхронность тактового генератора приёмника приводит к тому,
    что символы выбираются не в оптимальный момент, что вносит фазовый сдвиг.

    Упрощённая модель (без интерполяции):
        y[n] = x[n] · exp(j·2π·τ[n]) + w[n]
        τ[n] ~ Uniform(-τ_max, +τ_max)  [нормировано в долях символьного интервала]

    Параметры config:
        enabled              : bool
        timing_offset_range  : float, максимальная тайминговая ошибка (τ_max, доли символа)
                               Типично 0.0 – 0.5
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Timing Offset"
        self.tau_max = config.get("timing_offset_range", 0.0)

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False) or self.tau_max == 0.0:
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        n = len(tx_symbols)
        tau = np.random.uniform(-self.tau_max, self.tau_max, n)
        h = np.exp(1j * 2.0 * np.pi * tau)
        rx = tx_symbols * h

        if add_noise:
            signal_power = float(np.mean(np.abs(tx_symbols) ** 2))
            if signal_power == 0.0:
                signal_power = 1.0
            rx = rx + _awgn_noise(n, snr_linear, signal_power)

        return rx, h


# ---------------------------------------------------------------------------
# Impulse Noise (Бернулли–Гаусс)
# ---------------------------------------------------------------------------

class ImpulseNoiseChannel(ChannelModel):
    """
    Импульсные помехи по модели Бернулли–Гаусса.

    Физика: электромагнитные помехи от промышленного оборудования,
    грозовые разряды, переходные процессы в сети питания.

    Модель:
        b[n] ~ Bernoulli(p_impulse)          — индикатор импульса
        g[n] ~ CN(0, σ²_imp)                 — амплитуда импульса
        w[n] ~ CN(0, σ²_awgn)               — тепловой шум
        y[n] = x[n] + w[n] + b[n]·g[n]

    Импульсы могут иметь ширину > 1 символа (пачечные помехи).

    Параметры config:
        enabled                 : bool
        impulse_probability     : float, вероятность импульса на символ (по умолчанию 0.001)
        impulse_snr_dB          : float, превышение мощности импульса над AWGN, дБ (по умолчанию 20)
        impulse_width_min       : int, минимальная ширина импульса в символах (по умолчанию 1)
        impulse_width_max       : int, максимальная ширина импульса (по умолчанию 5)
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Impulse Noise"
        self.prob = config.get("impulse_probability", 0.001)
        self.impulse_snr_dB = config.get("impulse_snr_dB", 20.0)
        self.width_min = config.get("impulse_width_min", 1)
        self.width_max = config.get("impulse_width_max", 5)

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False):
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        n = len(tx_symbols)
        signal_power = float(np.mean(np.abs(tx_symbols) ** 2))
        if signal_power == 0.0:
            signal_power = 1.0

        rx = tx_symbols.copy()

        # Тепловой AWGN
        if add_noise:
            rx = rx + _awgn_noise(n, snr_linear, signal_power)

        # Импульсная мощность = AWGN мощность × 10^(impulse_snr_dB/10)
        sigma_awgn = np.sqrt(signal_power / (2.0 * snr_linear))
        impulse_amp = sigma_awgn * np.sqrt(10.0 ** (self.impulse_snr_dB / 10.0))

        idx = 0
        while idx < n:
            if np.random.rand() < self.prob:
                width = np.random.randint(self.width_min, self.width_max + 1)
                end = min(idx + width, n)
                # Случайная комплексная амплитуда
                amplitude = impulse_amp * np.exp(1j * 2.0 * np.pi * np.random.rand())
                rx[idx:end] += amplitude * (np.random.randn(end - idx)
                                            + 1j * np.random.randn(end - idx))
                idx = end
            else:
                idx += 1

        h = np.ones(n, dtype=complex)
        return rx, h


# ---------------------------------------------------------------------------
# Shadowing (Log-Normal)
# ---------------------------------------------------------------------------

class ShadowingChannel(ChannelModel):
    """
    Медленные замирания (теневые эффекты, shadowing) по log-normal модели.

    Физика: крупные препятствия (здания, рельеф местности) вызывают медленно
    изменяющееся затухание сигнала, которое в дБ описывается нормальным законом.

    Модель:
        S_dB ~ N(0, σ²_shadow)
        S    = 10^(S_dB / 20)       — линейный масштабный коэффициент
        y[n] = S · x[n] + w[n]

    В реальности S — константа или очень медленно меняется (время корреляции
    >> длины пакета). Здесь мы применяем одно значение S на весь вектор символов.

    Параметры config:
        enabled          : bool
        shadow_std_dB    : float, СКО затухания в дБ (по умолчанию 8.0 дБ)
                           Типичные значения: 6–10 дБ для городской среды
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "Shadowing"
        self.shadow_std_dB = config.get("shadow_std_dB", 8.0)

    def apply_with_coeff(self, tx_symbols: np.ndarray, snr_linear: float,
                         add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config.get("enabled", False):
            return tx_symbols.copy(), np.ones(len(tx_symbols), dtype=complex)

        n = len(tx_symbols)

        # Одно значение затухания на весь блок (медленные замирания)
        shadow_dB = self.shadow_std_dB * np.random.randn()
        shadow_linear = 10.0 ** (shadow_dB / 20.0)

        h = np.full(n, shadow_linear, dtype=complex)
        rx = tx_symbols * h

        if add_noise:
            # Нормируем sigma по исходной мощности символов (до тени)
            signal_power = float(np.mean(np.abs(tx_symbols) ** 2))
            if signal_power == 0.0:
                signal_power = 1.0
            rx = rx + _awgn_noise(n, snr_linear, signal_power)

        return rx, h


# ---------------------------------------------------------------------------
# Composite Channel Model
# ---------------------------------------------------------------------------

class CompositeChannelModel:
    """
    Последовательное объединение нескольких каналов.

    Ключевое отличие от наивной реализации:
    AWGN добавляется ОДИН РАЗ — после всех детерминированных искажений.
    Это физически корректно: тепловой шум возникает в приёмнике, а не
    в каждом «слое» канала.

    Порядок применения:
        1. Shadowing          (если включён)
        2. Rayleigh Fading    (если включён)
        3. Multipath          (если включён; взаимоисключает Rayleigh)
        4. Frequency Offset   (если включён)
        5. Phase Noise        (если включён)
        6. Timing Offset      (если включён)
        7. Единый AWGN
        8. Impulse Noise      (если включён; добавляет собственный AWGN + импульсы)

    Примечание: Rayleigh и Multipath взаимоисключают друг друга.
    Если оба включены, приоритет у Multipath.

    Ожидаемый формат channel_config:
    {
        "awgn":             {"enabled": True},
        "rayleigh":         {"enabled": False, "n_rays": 16, "normalized_doppler": 0.01},
        "multipath":        {"enabled": False, "n_taps": 6, "normalized_doppler": 0.01,
                             "n_rays": 16, "pdp_decay": 1.0},
        "frequency_offset": {"enabled": False, "normalized_freq_offset": 1e-4,
                             "cfo_drift_std": 0.0},
        "phase_noise":      {"enabled": False, "phase_noise_std_deg": 1.0},
        "timing_offset":    {"enabled": False, "timing_offset_range": 0.1},
        "impulse_noise":    {"enabled": False, "impulse_probability": 0.001,
                             "impulse_snr_dB": 20.0,
                             "impulse_width_min": 1, "impulse_width_max": 5},
        "shadowing":        {"enabled": False, "shadow_std_dB": 8.0}
    }
    """

    def __init__(self, channel_config: Dict):
        self.channel_config = channel_config
        self._build_pipeline()

    def _build_pipeline(self):
        """Строит список каналов в корректном порядке."""
        cfg = self.channel_config
        self.pre_noise_channels: List[ChannelModel] = []

        # 1. Shadowing
        if cfg.get("shadowing", {}).get("enabled", False):
            self.pre_noise_channels.append(ShadowingChannel(cfg["shadowing"]))

        # 2. Fading (Multipath имеет приоритет над Rayleigh)
        if cfg.get("multipath", {}).get("enabled", False):
            self.pre_noise_channels.append(MultipathChannel(cfg["multipath"]))
        elif cfg.get("rayleigh", {}).get("enabled", False):
            self.pre_noise_channels.append(RayleighFadingChannel(cfg["rayleigh"]))

        # 3. Frequency Offset
        if cfg.get("frequency_offset", {}).get("enabled", False):
            self.pre_noise_channels.append(FrequencyOffsetChannel(cfg["frequency_offset"]))

        # 4. Phase Noise
        if cfg.get("phase_noise", {}).get("enabled", False):
            self.pre_noise_channels.append(PhaseNoiseChannel(cfg["phase_noise"]))

        # 5. Timing Offset
        if cfg.get("timing_offset", {}).get("enabled", False):
            self.pre_noise_channels.append(TimingOffsetChannel(cfg["timing_offset"]))

        # Impulse Noise обрабатывается отдельно (имеет встроенный AWGN)
        self.impulse_channel: Optional[ImpulseNoiseChannel] = None
        if cfg.get("impulse_noise", {}).get("enabled", False):
            self.impulse_channel = ImpulseNoiseChannel(cfg["impulse_noise"])

        # Базовый AWGN (всегда присутствует как резерв)
        self.awgn_channel = AWGNChannel(cfg.get("awgn", {"enabled": True}))

        logger.info(f"CompositeChannel pipeline: "
                    f"{[ch.name for ch in self.pre_noise_channels]}"
                    f"{' + ImpulseNoise' if self.impulse_channel else ''}"
                    f" + AWGN")

    def apply_with_coeff(self, tx_symbols: np.ndarray,
                         snr_linear: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применяет полный канал.

        Returns:
            (rx_symbols, h_total)
            h_total — поэлементное произведение всех коэффициентов pre-noise каналов.
        """
        rx = tx_symbols.copy()
        n = len(tx_symbols)
        h_total = np.ones(n, dtype=complex)

        # --- Шаг 1: детерминированные искажения (без шума) ---
        for ch in self.pre_noise_channels:
            rx, h_partial = ch.apply_with_coeff(rx, snr_linear, add_noise=False)
            h_total *= h_partial

        # --- Шаг 2: единый AWGN ---
        # sigma считаем по исходной мощности символов (до всех искажений)
        signal_power = float(np.mean(np.abs(tx_symbols) ** 2))
        if signal_power == 0.0:
            signal_power = 1.0
        rx = rx + _awgn_noise(n, snr_linear, signal_power)

        # --- Шаг 3: импульсный шум (уже содержит свой AWGN) ---
        if self.impulse_channel is not None:
            # Передаём rx как «символы», add_noise=False чтобы не дублировать AWGN
            # Impulse канал добавит только импульсы поверх уже зашумлённого сигнала
            rx, _ = self.impulse_channel.apply_with_coeff(rx, snr_linear, add_noise=False)

        return rx, h_total

    def apply(self, tx_symbols: np.ndarray, snr_linear: float) -> np.ndarray:
        """Возвращает только искажённые символы."""
        rx, _ = self.apply_with_coeff(tx_symbols, snr_linear)
        return rx

    def get_channel_names(self) -> List[str]:
        """Список активных каналов."""
        names = [ch.name for ch in self.pre_noise_channels]
        names.append("AWGN")
        if self.impulse_channel:
            names.append(self.impulse_channel.name)
        return names


# ---------------------------------------------------------------------------
# Пример использования и быстрая проверка
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    print("=" * 60)
    print("Проверка нормировки мощности каналов")
    print("=" * 60)

    n_sym = 10_000
    snr_dB = 10.0
    snr_lin = 10 ** (snr_dB / 10)
    tx = (np.random.randn(n_sym) + 1j * np.random.randn(n_sym)) / np.sqrt(2)

    def check_power(name, h):
        p = np.mean(np.abs(h) ** 2)
        print(f"  {name:25s}: E[|h|²] = {p:.4f}  {'✓' if abs(p - 1.0) < 0.1 else '!'}")

    # AWGN
    ch = AWGNChannel({"enabled": True})
    rx, h = ch.apply_with_coeff(tx, snr_lin, add_noise=False)
    check_power("AWGN (h)", h)

    # Rayleigh Jakes
    ch = RayleighFadingChannel({"enabled": True, "n_rays": 32, "normalized_doppler": 0.01})
    rx, h = ch.apply_with_coeff(tx, snr_lin, add_noise=False)
    check_power("Rayleigh Jakes (h)", h)

    # Multipath
    ch = MultipathChannel({"enabled": True, "n_taps": 6, "normalized_doppler": 0.01,
                           "n_rays": 16, "pdp_decay": 1.5})
    rx, h = ch.apply_with_coeff(tx, snr_lin, add_noise=False)
    check_power("Multipath (h[0])", h)

    # Phase Noise
    ch = PhaseNoiseChannel({"enabled": True, "phase_noise_std_deg": 1.0})
    rx, h = ch.apply_with_coeff(tx, snr_lin, add_noise=False)
    check_power("Phase Noise (h)", h)

    # Shadowing
    ch = ShadowingChannel({"enabled": True, "shadow_std_dB": 8.0})
    rx, h = ch.apply_with_coeff(tx, snr_lin, add_noise=False)
    check_power("Shadowing (h)", h)

    print()
    print("=" * 60)
    print("Composite Channel")
    print("=" * 60)

    composite_cfg = {
        "awgn":             {"enabled": True},
        "rayleigh":         {"enabled": True, "n_rays": 16, "normalized_doppler": 0.01},
        "multipath":        {"enabled": False},
        "frequency_offset": {"enabled": True, "normalized_freq_offset": 1e-4,
                             "cfo_drift_std": 1e-7},
        "phase_noise":      {"enabled": True, "phase_noise_std_deg": 0.5},
        "timing_offset":    {"enabled": False},
        "impulse_noise":    {"enabled": True, "impulse_probability": 0.005,
                             "impulse_snr_dB": 15.0,
                             "impulse_width_min": 1, "impulse_width_max": 3},
        "shadowing":        {"enabled": True, "shadow_std_dB": 6.0},
    }

    composite = CompositeChannelModel(composite_cfg)
    print("Активные каналы:", composite.get_channel_names())
    rx, h = composite.apply_with_coeff(tx, snr_lin)
    print(f"E[|rx|²] = {np.mean(np.abs(rx)**2):.4f}  (ожидается ~ {np.mean(np.abs(tx)**2):.4f} + шум)")
    print("Готово.")