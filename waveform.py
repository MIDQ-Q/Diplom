"""
waveform.py — Дискретно-временная модель сигнала (samples-per-symbol) для симулятора.
Python 3.12+

Зачем нужно:
  - В реальной системе обычно есть >1 отсчёта на символ, импульсное формирование
    (pulse shaping) и согласованная фильтрация (matched filter).
  - Это даёт полосу, “глазок”, чувствительность к таймингу, а также позволяет
    позже добавить ISI/частотно-селективный канал и синхронизацию.

Текущий MVP:
  - RRC pulse shaping на TX + matched filter на RX
  - Downsample до symbol-rate (без timing recovery, фиксированная выборка)
  - AWGN и CFO предполагается добавлять на уровне отсчётов

Примечание по нормировке:
  - RRC taps нормируются по энергии: sum(|h|^2) = 1.
  - Для надёжности предусмотрена калибровка усиления так, чтобы для импульсной
    последовательности на выходе matched filter в момент символа получалось ~1.
"""

from __future__ import annotations

import numpy as np


def rrc_taps(beta: float, span_symbols: int, sps: int) -> np.ndarray:
    """
    Root Raised Cosine (RRC) FIR taps.

    Args:
        beta         : roll-off (0..1)
        span_symbols : длина фильтра в символах (типично 6..12)
        sps          : отсчётов на символ (>=2)

    Returns:
        taps : float64 ndarray, shape (span_symbols*sps + 1,)
               нормирован по энергии: sum(taps^2) = 1
    """
    if sps < 2:
        raise ValueError(f"sps должен быть >=2, получено {sps}")
    if span_symbols < 2:
        raise ValueError(f"span_symbols должен быть >=2, получено {span_symbols}")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta должен быть в [0..1], получено {beta}")

    n_taps = span_symbols * sps + 1
    t = (np.arange(n_taps) - (n_taps - 1) / 2.0) / sps  # в символах
    h = np.zeros_like(t, dtype=np.float64)

    # Формула RRC (стандартная, с обработкой особых точек t=0 и t=±1/(4β))
    # Нормировка будет сделана отдельно по энергии.
    eps = 1e-12
    for i, ti in enumerate(t):
        if abs(ti) < eps:
            # t = 0
            h[i] = 1.0 - beta + (4.0 * beta / np.pi)
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-10:
            # t = ±1/(4β)
            # Предельное значение
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = (
                np.sin(np.pi * ti * (1.0 - beta))
                + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            )
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den

    # Нормировка по энергии
    e = float(np.sum(h * h))
    if e <= 0:
        raise RuntimeError("RRC taps имеют нулевую энергию (проверь параметры)")
    h = h / np.sqrt(e)
    return h


def upsample_zeros(symbols: np.ndarray, sps: int) -> np.ndarray:
    """Вставка (sps-1) нулей между символами."""
    x = np.asarray(symbols, dtype=complex).ravel()
    y = np.zeros(len(x) * sps, dtype=complex)
    y[::sps] = x
    return y


def fir_filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Линейная FIR-фильтрация через np.convolve (для MVP достаточно)."""
    x = np.asarray(x)
    h = np.asarray(h)
    return np.convolve(x, h, mode="full")


def apply_cfo(samples: np.ndarray, cfo_norm: float) -> np.ndarray:
    """
    Применяет CFO (carrier frequency offset) к комплексным отсчётам.

    Args:
        samples  : комплексные отсчёты
        cfo_norm : нормированная частота, циклов/отсчёт.
                   Например 0.001 означает 0.001*2π рад/отсчёт.
                   Знак определяет направление вращения.
    """
    if cfo_norm == 0.0:
        return np.asarray(samples, dtype=complex).ravel()
    x = np.asarray(samples, dtype=complex).ravel()
    n = np.arange(len(x), dtype=np.float64)
    rot = np.exp(1j * 2.0 * np.pi * cfo_norm * n)
    return x * rot


def apply_phase_noise_wiener(
    samples: np.ndarray,
    *,
    step_std_rad: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Phase noise на отсчётах как Wiener process (random walk по фазе).

    Модель:
      theta[n] = theta[n-1] + dtheta[n],  dtheta[n] ~ N(0, step_std_rad^2)
      y[n] = x[n] * exp(j*theta[n])
    """
    x = np.asarray(samples, dtype=complex).ravel()
    if x.size == 0:
        return x.copy()
    step_std_rad = float(step_std_rad)
    if step_std_rad <= 0:
        return x.copy()

    rng = np.random.default_rng(seed=seed) if seed is not None else np.random.default_rng()
    dtheta = rng.standard_normal(x.size) * step_std_rad
    theta = np.cumsum(dtheta, dtype=np.float64)
    return x * np.exp(1j * theta)


def agc_block(
    samples: np.ndarray,
    *,
    target_rms: float = 1.0,
    max_gain: float = 100.0,
    min_gain: float = 0.01,
) -> tuple[np.ndarray, dict]:
    """
    Простой блоковый AGC: подстраивает усиление так, чтобы RMS стал target_rms.

    Это приближение реального AGC (без атаки/релиза), но даёт реалистичное поведение
    уровня сигнала перед АЦП.
    """
    x = np.asarray(samples, dtype=complex).ravel()
    if x.size == 0:
        return x.copy(), {"gain": 1.0, "rms_in": 0.0, "rms_out": 0.0}
    target_rms = float(target_rms)
    if target_rms <= 0:
        return x.copy(), {"gain": 1.0, "rms_in": float(np.sqrt(np.mean(np.abs(x) ** 2))), "rms_out": float(np.sqrt(np.mean(np.abs(x) ** 2)))}

    rms_in = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    if rms_in < 1e-30:
        return x.copy(), {"gain": 1.0, "rms_in": rms_in, "rms_out": rms_in}
    gain = target_rms / rms_in
    gain = float(np.clip(gain, float(min_gain), float(max_gain)))
    y = x * gain
    rms_out = float(np.sqrt(np.mean(np.abs(y) ** 2)))
    return y, {"gain": gain, "rms_in": rms_in, "rms_out": rms_out}


def clip_magnitude(samples: np.ndarray, *, clip_level: float) -> tuple[np.ndarray, dict]:
    """
    Клиппинг по модулю: |y| <= clip_level (с сохранением фазы).
    """
    x = np.asarray(samples, dtype=complex).ravel()
    A = float(clip_level)
    if x.size == 0 or A <= 0:
        return x.copy(), {"clip_level": A, "clipped_fraction": 0.0}
    mag = np.abs(x)
    mask = mag > A
    y = x.copy()
    if np.any(mask):
        y[mask] = x[mask] * (A / (mag[mask] + 1e-30))
    frac = float(np.mean(mask.astype(float)))
    return y, {"clip_level": A, "clipped_fraction": frac}


def adc_quantize_iq(
    samples: np.ndarray,
    *,
    n_bits: int = 10,
    full_scale: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Равномерное квантование I/Q (mid-tread) с насыщением.

    Range: [-full_scale, +full_scale].
    Levels: 2^n_bits.
    """
    x = np.asarray(samples, dtype=complex).ravel()
    n_bits = int(n_bits)
    if x.size == 0:
        return x.copy(), {"n_bits": n_bits, "full_scale": float(full_scale), "step": 0.0}
    if n_bits < 1:
        return x.copy(), {"n_bits": n_bits, "full_scale": float(full_scale), "step": 0.0}
    fs = float(full_scale)
    if fs <= 0:
        return x.copy(), {"n_bits": n_bits, "full_scale": fs, "step": 0.0}

    levels = 2 ** n_bits
    step = 2.0 * fs / levels

    def _q(v: np.ndarray) -> np.ndarray:
        v = np.clip(v, -fs, fs - step)  # чтобы верхняя граница не давала переполнения уровней
        return step * np.round(v / step)

    I = _q(x.real.astype(np.float64))
    Q = _q(x.imag.astype(np.float64))
    y = I + 1j * Q
    return y.astype(complex), {"n_bits": n_bits, "full_scale": fs, "step": float(step)}


def apply_dc_offset(samples: np.ndarray, *, dc_i: float = 0.0, dc_q: float = 0.0) -> tuple[np.ndarray, dict]:
    """Добавляет DC offset на I и Q."""
    x = np.asarray(samples, dtype=complex).ravel()
    di = float(dc_i)
    dq = float(dc_q)
    y = x + (di + 1j * dq)
    return y, {"dc_i": di, "dc_q": dq}


def apply_iq_imbalance(
    samples: np.ndarray,
    *,
    amp_imbalance_db: float = 0.0,
    phase_imbalance_deg: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """
    Простейшая модель IQ-imbalance: амплитудный дисбаланс I/Q + фазовый перекос.

    Реализация (реально-матричная):
      I' = gI * I
      Q' = gQ * (Q*cos(phi) + I*sin(phi))
    где gI/gQ задаются через amp_imbalance_db, phi — phase_imbalance.
    """
    x = np.asarray(samples, dtype=complex).ravel()
    a_db = float(amp_imbalance_db)
    phi = float(phase_imbalance_deg) * np.pi / 180.0

    # gI/gQ: если a_db>0, то I сильнее Q
    gI = 10.0 ** (a_db / 20.0)
    gQ = 1.0

    I = x.real.astype(np.float64)
    Q = x.imag.astype(np.float64)
    I2 = gI * I
    Q2 = gQ * (Q * np.cos(phi) + I * np.sin(phi))
    y = I2 + 1j * Q2
    return y.astype(complex), {"amp_imbalance_db": a_db, "phase_imbalance_deg": float(phase_imbalance_deg)}


def tdl_taps_from_symbol_delays(
    sps: int,
    delays_symbols: list[float] | tuple[float, ...] | np.ndarray,
    powers_dB: list[float] | tuple[float, ...] | np.ndarray,
    *,
    normalize: bool = True,
    fading: str = "rayleigh",
    rng: np.random.Generator | None = None,
    frac_taps: int = 21,
) -> np.ndarray:
    """
    Строит TDL (tapped-delay-line) дискретный FIR-канал на уровне ОТСЧЁТОВ.

    MVP-ограничения:
      - фейдинг статический на весь блок (один комплексный коэффициент на путь)
      - fractional delays реализованы через оконный sinc FIR (frac_taps)

    Args:
        sps          : samples per symbol
        delays_symbols : список задержек (в символах, можно дробные), напр. [0, 1, 3] или [0, 0.5, 2.25]
        powers_dB    : мощности путей (в дБ), напр. [0, -3, -6]
        normalize    : нормировать так, чтобы sum(|h|^2)=1
        fading       : "rayleigh" (по умолчанию) или "deterministic"
        rng          : генератор случайных чисел (для повторяемости)

    Returns:
        h : complex ndarray, shape (max_delay*sps + 1,)
    """
    if sps < 2:
        raise ValueError(f"sps должен быть >=2, получено {sps}")

    d_sym = np.asarray(delays_symbols, dtype=float).ravel()
    p = np.asarray(powers_dB, dtype=float).ravel()
    if d_sym.size == 0:
        return np.array([1.0 + 0.0j], dtype=complex)
    if d_sym.size != p.size:
        raise ValueError("delays_symbols и powers_dB должны иметь одинаковую длину")
    if np.any(d_sym < 0):
        raise ValueError("delays_symbols должны быть >= 0")

    if rng is None:
        rng = np.random.default_rng()

    amps = 10.0 ** (p / 20.0)  # dB power -> linear amplitude scale
    fading = (fading or "rayleigh").lower().strip()
    if fading in ("rayleigh", "randn", "random"):
        a = (rng.standard_normal(d_sym.size) + 1j * rng.standard_normal(d_sym.size)) / np.sqrt(2.0)
    else:
        a = np.ones(d_sym.size, dtype=complex)

    # Helper: fractional delay impulse (windowed sinc), centered
    frac_taps = int(frac_taps)
    if frac_taps < 5:
        frac_taps = 5
    if frac_taps % 2 == 0:
        frac_taps += 1
    m = (frac_taps - 1) // 2
    n = np.arange(-m, m + 1, dtype=np.float64)

    def _frac_impulse(delay_samples: float) -> tuple[np.ndarray, int]:
        """
        Returns:
          (g, k0) where g is frac_taps-length FIR, and k0 is integer sample index
          where g[0] should be added (i.e., g is centered around k0+ m).
        """
        k_int = int(np.floor(delay_samples))
        mu = float(delay_samples - k_int)  # [0..1)
        # sinc(n - mu)
        g = np.sinc(n - mu)
        # Hann window
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(frac_taps) / (frac_taps - 1)))
        g = g * w
        # Normalize DC gain ~1 for impulse reconstruction
        s = float(np.sum(g))
        if abs(s) > 1e-12:
            g = g / s
        k0 = k_int - m
        return g.astype(np.float64), k0

    d_samp = d_sym * float(sps)
    max_delay_samp = float(np.max(d_samp))
    max_int = int(np.floor(max_delay_samp))
    h = np.zeros(max_int + frac_taps + 1, dtype=complex)
    for delay_s, amp, coeff in zip(d_samp.tolist(), amps.tolist(), a.tolist()):
        g, k0 = _frac_impulse(delay_s)
        k0i = int(k0)
        k1 = k0i + frac_taps
        if k1 <= 0:
            continue
        if k0i < 0:
            g = g[-k0i:]
            k0i = 0
        k1 = k0i + g.size
        if k1 > h.size:
            h = np.pad(h, (0, k1 - h.size))
        h[k0i:k1] += (coeff * float(amp)) * g

    if normalize:
        e = float(np.sum(np.abs(h) ** 2))
        if e > 1e-30:
            h = h / np.sqrt(e)
    return h


def apply_tdl(samples: np.ndarray, h_tdl: np.ndarray) -> np.ndarray:
    """Применяет TDL канал к отсчётам (FIR свёртка)."""
    x = np.asarray(samples, dtype=complex).ravel()
    h = np.asarray(h_tdl, dtype=complex).ravel()
    if h.size <= 1:
        return x.copy()
    return np.convolve(x, h, mode="full")


def time_warp_samples(
    samples: np.ndarray,
    *,
    offset_samples: float = 0.0,
    drift_rate: float = 0.0,
    jitter_std_samples: float = 0.0,
    jitter_mode: str = "white",
    jitter_pll_bw_norm: float = 0.01,
    jitter_wiener_step_std_samples: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Имитация ошибок дискретизации (тайминга) на уровне отсчётов через time-warp.

    Выходной массив той же длины, что и вход.

    Модель:
      y_out[n] = x_in[t(n)]
      t(n) = offset_samples + n*(1 + drift_rate) + jitter[n]

    где drift_rate — относительная ошибка частоты дискретизации (например 50 ppm -> 50e-6).

    jitter_mode:
      - "white": jitter[n] независим по n (как сейчас)
      - "pll": коррелированный jitter как у PLL/clock recovery (AR(1) low-pass),
               стационарная СКО примерно jitter_std_samples.
      - "wiener": джиттер как random walk (Wiener), т.е. интеграл белого шума
                 (похоже на фазовый шум тактового, PSD ~ 1/f^2).
                 В этом режиме:
                   - если jitter_wiener_step_std_samples задан, это СКО шага (per-sample)
                   - иначе используется jitter_std_samples как СКО шага

    jitter_pll_bw_norm: нормированная “полоса” процесса на отсчёт (0..0.5),
    чем меньше — тем сильнее корреляция (медленнее меняется jitter).

    Это приближение реального “sampling clock offset + jitter”.
    """
    x = np.asarray(samples, dtype=complex).ravel()
    N = int(x.size)
    if N == 0:
        return x.copy()

    offset_samples = float(offset_samples)
    drift_rate = float(drift_rate)
    jitter_std_samples = float(jitter_std_samples)
    if jitter_std_samples < 0:
        jitter_std_samples = 0.0

    rng = np.random.default_rng(seed=seed) if seed is not None else np.random.default_rng()
    jitter = np.zeros(N, dtype=np.float64)
    if jitter_std_samples > 0:
        mode = (jitter_mode or "white").lower().strip()
        if mode in ("pll", "correlated", "ar1", "lowpass"):
            bw = float(jitter_pll_bw_norm)
            if bw < 1e-6:
                bw = 1e-6
            if bw > 0.5:
                bw = 0.5
            # AR(1): j[n] = rho*j[n-1] + sigma_e*w[n]
            # rho ~ exp(-2*pi*bw)  (bw in cycles/sample, crude but stable)
            rho = float(np.exp(-2.0 * np.pi * bw))
            sigma = float(jitter_std_samples)
            sigma_e = sigma * np.sqrt(max(1.0 - rho * rho, 1e-12))
            w = rng.standard_normal(N)
            j = 0.0
            for i in range(N):
                j = rho * j + sigma_e * float(w[i])
                jitter[i] = j
        elif mode in ("wiener", "rw", "random_walk", "brownian"):
            step_std = float(jitter_wiener_step_std_samples) if jitter_wiener_step_std_samples is not None else float(jitter_std_samples)
            if step_std < 0:
                step_std = 0.0
            w = rng.standard_normal(N) * step_std
            jitter = np.cumsum(w).astype(np.float64)
            # убрать линейный уклон, чтобы не дублировать drift_rate
            if N >= 2:
                jitter = jitter - np.linspace(jitter[0], jitter[-1], N, dtype=np.float64)
        else:
            jitter = rng.standard_normal(N) * jitter_std_samples

    t = offset_samples + (1.0 + drift_rate) * np.arange(N, dtype=np.float64) + jitter
    y = np.empty(N, dtype=complex)
    for n in range(N):
        y[n] = _lagrange4_interp(x, float(t[n]))
    return y


def _calibrate_gain(h: np.ndarray, sps: int) -> float:
    """
    Калибрует масштаб фильтра так, чтобы после TX(RRC)+RX(RRC) пик Nyquist
    в момент символа был ~1.
    """
    # Импульс: один символ = 1, остальные 0
    sym = np.zeros(1, dtype=complex)
    sym[0] = 1.0 + 0.0j
    tx_u = upsample_zeros(sym, sps)
    tx = fir_filter(tx_u, h)
    rx = fir_filter(tx, h)  # matched (для real taps time-reversal не нужен в амплитуде)
    gd = (len(h) - 1) // 2
    idx = 2 * gd  # суммарная групповая задержка
    if idx >= len(rx):
        return 1.0
    peak = abs(rx[idx])
    if peak < 1e-12:
        return 1.0
    return 1.0 / peak


def estimate_eye_opening_for_offset(
    rx_mf: np.ndarray,
    *,
    sps: int,
    h_len: int,
    offset_samples: float,
    n_traces: int = 200,
) -> dict:
    """
    Быстрая оценка "eye opening" для фиксированного fractional offset (без timing recovery).

    Берём окно длиной 2*sps вокруг каждого символа, sample time сдвигаем на offset_samples.
    Метрика: opening_norm = (std_mid - std_center) / (std_mid + eps) для I и Q.
    """
    y = np.asarray(rx_mf, dtype=complex).ravel()
    gd = (int(h_len) - 1) // 2
    start = float(2 * gd) + float(offset_samples)
    win = 2 * int(sps)
    center_idx = int(sps)
    mid_idx = int(sps // 2)

    n_traces = int(n_traces)
    n_traces = max(20, min(2000, n_traces))
    traces_i = []
    traces_q = []
    max_k = max(0, int((len(y) - (2 * gd) - win - 2) // max(int(sps), 1)))
    for k in range(min(n_traces, max_k)):
        t0 = start + k * float(sps) - float(sps)
        ts = t0 + np.arange(win, dtype=np.float64)
        seg = np.array([_lagrange4_interp(y, float(tt)) for tt in ts], dtype=complex)
        traces_i.append(seg.real.astype(float))
        traces_q.append(seg.imag.astype(float))

    def _metric(traces: list[np.ndarray]) -> dict | None:
        if not traces:
            return None
        arr = np.stack(traces, axis=0)  # (T, win)
        c = arr[:, center_idx]
        m = arr[:, mid_idx]
        std_c = float(np.std(c))
        std_m = float(np.std(m))
        opening = float(std_m - std_c)
        opening_norm = float(opening / (std_m + 1e-12))
        return {"std_center": std_c, "std_mid": std_m, "opening": opening, "opening_norm": opening_norm}

    return {
        "offset_samples": float(offset_samples),
        "I": _metric(traces_i),
        "Q": _metric(traces_q),
        "center_idx": center_idx,
        "mid_idx": mid_idx,
        "window_len": win,
        "n_traces_used": len(traces_i),
    }


def auto_init_offset_by_eye_opening(
    rx_mf: np.ndarray,
    *,
    sps: int,
    h_len: int,
    n_grid: int = 33,
    n_traces: int = 200,
    combine: str = "IQ",
) -> tuple[float, dict]:
    """
    Подбирает init_offset_samples (0..sps) по максимальному eye opening.

    combine:
      - "I": оптимизация только по I
      - "Q": только по Q
      - "IQ": сумма opening_norm по I и Q (по умолчанию)
    """
    n_grid = int(n_grid)
    if n_grid < 9:
        n_grid = 9
    combine = (combine or "IQ").upper().strip()
    grid = np.linspace(0.0, float(sps), n_grid, dtype=np.float64)

    best_off = 0.0
    best_score = -1e9
    best_stats = None
    scores = []
    for off in grid:
        st = estimate_eye_opening_for_offset(
            rx_mf, sps=sps, h_len=h_len, offset_samples=float(off), n_traces=n_traces
        )
        i = st.get("I") or {}
        q = st.get("Q") or {}
        si = float(i.get("opening_norm", -1e9)) if i else -1e9
        sq = float(q.get("opening_norm", -1e9)) if q else -1e9
        if combine == "I":
            score = si
        elif combine == "Q":
            score = sq
        else:
            score = si + sq
        scores.append((float(off), float(score)))
        if score > best_score:
            best_score = score
            best_off = float(off)
            best_stats = st

    diag = {
        "best_offset_samples": best_off,
        "best_score": float(best_score),
        "grid_scores": scores,
        "best_stats": best_stats,
        "combine": combine,
        "n_grid": n_grid,
        "n_traces": int(n_traces),
    }
    return best_off, diag


def pulse_shaping_tx(symbols: np.ndarray, h_rrc: np.ndarray, sps: int, calibrate: bool = True) -> np.ndarray:
    """
    TX: symbols → upsample → RRC filter → samples.
    """
    g = np.asarray(h_rrc, dtype=np.float64)
    scale = _calibrate_gain(g, sps) if calibrate else 1.0
    x_u = upsample_zeros(symbols, sps)
    return fir_filter(x_u, g * scale)


def matched_filter_rx(samples: np.ndarray, h_rrc: np.ndarray) -> np.ndarray:
    """RX matched filter (RRC)."""
    g = np.asarray(h_rrc, dtype=np.float64)
    return fir_filter(samples, g)


def downsample_to_symbols(rx_mf: np.ndarray, n_symbols: int, sps: int, h_len: int) -> np.ndarray:
    """
    Выборка после matched filter без recovery: берем фиксированный фазовый отсчёт.

    Args:
        rx_mf     : выход matched filter (full convolution)
        n_symbols : сколько символов ожидаем на выходе
        sps       : отсчётов на символ
        h_len     : длина RRC taps
    """
    gd = (h_len - 1) // 2
    start = 2 * gd  # TX gd + RX gd
    idx = start + np.arange(n_symbols) * sps
    idx = idx[idx < len(rx_mf)]
    out = rx_mf[idx]
    # Если не хватило по длине (края), дополняем нулями для стабильности пайплайна
    if len(out) < n_symbols:
        out = np.concatenate([out, np.zeros(n_symbols - len(out), dtype=complex)])
    return out


def _lagrange4_interp(x: np.ndarray, t: float) -> complex:
    """
    4-точечная (3-го порядка) Lagrange интерполяция комплексного сигнала.

    Args:
        x : входной массив (complex), 1D
        t : дробный индекс (может быть нецелым)

    Returns:
        интерполированное значение x(t)
    """
    xi = int(np.floor(t))
    mu = float(t - xi)
    # нужны точки x[xi-1], x[xi], x[xi+1], x[xi+2]
    # аккуратно обрабатываем границы: вне диапазона -> 0
    def _get(k: int) -> complex:
        if 0 <= k < x.size:
            return complex(x[k])
        return 0.0 + 0.0j

    xm1 = _get(xi - 1)
    x0 = _get(xi)
    x1 = _get(xi + 1)
    x2 = _get(xi + 2)

    # коэффициенты Lagrange для узлов [-1,0,1,2] при параметре mu in [0,1)
    c_m1 = (-mu * (mu - 1.0) * (mu - 2.0)) / 6.0
    c_0  = ((mu + 1.0) * (mu - 1.0) * (mu - 2.0)) / 2.0
    c_1  = (-(mu + 1.0) * mu * (mu - 2.0)) / 2.0
    c_2  = ((mu + 1.0) * mu * (mu - 1.0)) / 6.0
    return c_m1 * xm1 + c_0 * x0 + c_1 * x1 + c_2 * x2


def interp_lagrange4(samples: np.ndarray, t: float) -> complex:
    """Публичная обёртка для 4-точечной Lagrange интерполяции."""
    return _lagrange4_interp(np.asarray(samples, dtype=complex).ravel(), float(t))


def gardner_timing_recovery(
    rx_mf: np.ndarray,
    *,
    n_symbols: int,
    sps: int,
    h_len: int,
    alpha: float = 0.01,
    beta: float = 0.0001,
    init_offset_samples: float = 0.0,
    omega_rel_limit: float = 0.2,
) -> tuple[np.ndarray, dict]:
    """
    Gardner timing recovery + fractional resampler (через интерполяцию).

    Работает на выходе matched filter (rx_mf), где есть sps отсчётов/символ.
    Возвращает symbol-rate выборки без hard timing.

    TED Gardner:
      e_k = Re{ (x_k - x_{k+1}) * conj(x_{k+1/2}) }

    PI-петля по шагу выборки:
      omega_{k+1} = omega_k + beta * e_k
      t_{k+1}     = t_k + omega_{k+1} + alpha * e_k

    Args:
        rx_mf              : matched filter output (full convolution)
        n_symbols          : сколько символов выбрать
        sps                : samples per symbol (>=2, лучше чётное)
        h_len              : длина RRC taps (для групповой задержки)
        alpha, beta        : коэффициенты петли
        init_offset_samples: начальный сдвиг (в отсчётах) относительно "идеального" start=2*gd
        omega_rel_limit    : ограничение omega вокруг sps: sps*(1±omega_rel_limit)

    Returns:
        (rx_symbols, stats)
    """
    y = np.asarray(rx_mf, dtype=complex).ravel()
    if n_symbols <= 0:
        return np.array([], dtype=complex), {"timing_enabled": True}
    if sps < 2:
        raise ValueError("sps должен быть >=2")

    gd = (int(h_len) - 1) // 2
    t = float(2 * gd) + float(init_offset_samples)

    omega = float(sps)
    omega_min = float(sps) * (1.0 - float(abs(omega_rel_limit)))
    omega_max = float(sps) * (1.0 + float(abs(omega_rel_limit)))
    half = float(sps) / 2.0

    out = np.zeros(int(n_symbols), dtype=complex)
    errs = np.zeros(int(n_symbols), dtype=np.float64)
    t_hist = np.zeros(int(n_symbols), dtype=np.float64)
    omega_hist = np.zeros(int(n_symbols), dtype=np.float64)

    # Берём x_k, x_{k+1/2}, x_{k+1} на каждом шаге
    for k in range(int(n_symbols)):
        xk = _lagrange4_interp(y, t)
        xmid = _lagrange4_interp(y, t + half)
        xk1 = _lagrange4_interp(y, t + omega)

        e = float(np.real((xk - xk1) * np.conj(xmid)))
        errs[k] = e

        omega = omega + float(beta) * e
        if omega < omega_min:
            omega = omega_min
        elif omega > omega_max:
            omega = omega_max
        t = t + omega + float(alpha) * e

        out[k] = xk
        t_hist[k] = t
        omega_hist[k] = omega

    stats = {
        "timing_enabled": True,
        "alpha": float(alpha),
        "beta": float(beta),
        "init_offset_samples": float(init_offset_samples),
        "omega_final": float(omega),
        "timing_error_mean": float(np.mean(errs)) if errs.size else 0.0,
        "timing_error_std": float(np.std(errs)) if errs.size else 0.0,
        "t_hist": t_hist,              # sample times of symbol decisions (after update)
        "omega_hist": omega_hist,      # samples/symbol history
    }
    return out, stats


def estimate_symbol_spaced_channel(
    *,
    h_rrc: np.ndarray,
    sps: int,
    h_tdl: np.ndarray | None = None,
    n_symbols: int = 64,
) -> np.ndarray:
    """
    Оценивает эквивалентный symbol-rate ISI-канал для цепочки:
      symbols -> RRC(TX) -> [TDL on samples] -> RRC(RX) -> downsample

    Делается “численно” по импульсу, чтобы точно соответствовать всем нормировкам
    и реализации фильтров (включая calibrate в pulse_shaping_tx).

    Args:
        h_rrc    : RRC taps
        sps      : samples per symbol
        h_tdl    : taps TDL на отсчётах (или None/len=1 для отсутствия)
        n_symbols: длина оценки в символах (сколько ISI taps хотим)

    Returns:
        h_sym : complex ndarray, shape (n_symbols,)
    """
    n_symbols = int(n_symbols)
    if n_symbols < 8:
        n_symbols = 8

    x = np.zeros(n_symbols, dtype=complex)
    x[0] = 1.0 + 0.0j
    tx = pulse_shaping_tx(x, h_rrc=h_rrc, sps=sps, calibrate=True)
    if h_tdl is not None and np.asarray(h_tdl).size > 1:
        tx = apply_tdl(tx, h_tdl)
    rx = matched_filter_rx(tx, h_rrc=h_rrc)
    h_sym = downsample_to_symbols(rx, n_symbols=n_symbols, sps=sps, h_len=len(h_rrc))
    return np.asarray(h_sym, dtype=complex).ravel()


def estimate_channel_ls(
    tx_symbols: np.ndarray,
    rx_symbols: np.ndarray,
    *,
    channel_len: int,
    reg: float = 1e-3,
) -> np.ndarray:
    """
    Оценка symbol-rate ISI-канала по известной преамбуле (LS/ридж).

    Модель:
      rx ≈ tx (*) h
    где (*) — линейная свёртка. Оцениваем h длины channel_len.

    Args:
        tx_symbols  : известная последовательность (переданная), shape (N,)
        rx_symbols  : принятая последовательность, shape (N,) (можно взять первые N отсчётов)
        channel_len : длина оцениваемого канала (Lh)
        reg         : регуляризация (ridge), помогает при плохом обусловливании

    Returns:
        h_hat : shape (channel_len,)
    """
    tx = np.asarray(tx_symbols, dtype=complex).ravel()
    rx = np.asarray(rx_symbols, dtype=complex).ravel()
    Lh = int(channel_len)
    if Lh < 1:
        return np.array([1.0 + 0.0j], dtype=complex)
    N = int(min(tx.size, rx.size))
    if N < max(8, Lh + 2):
        # слишком мало данных — fallback
        h = np.zeros(Lh, dtype=complex)
        h[0] = 1.0 + 0.0j
        return h

    tx = tx[:N]
    rx = rx[:N]

    # Build convolution matrix X: (N, Lh), X[n, k] = tx[n-k] if n-k>=0 else 0
    X = np.zeros((N, Lh), dtype=complex)
    for k in range(Lh):
        X[k:, k] = tx[:N - k]

    A = X.conj().T @ X
    b = X.conj().T @ rx
    A = A + float(max(reg, 0.0)) * np.eye(Lh, dtype=complex)
    h_hat = np.linalg.solve(A, b)
    return np.asarray(h_hat, dtype=complex).ravel()


def design_linear_equalizer(
    h: np.ndarray,
    *,
    eq_len: int = 9,
    delay: int | None = None,
    kind: str = "mmse",
    noise_var: float = 0.0,
    reg: float = 1e-9,
) -> tuple[np.ndarray, int]:
    """
    Синтез FIR-эквалайзера (symbol-rate) для канала h (ISI taps).

    Решает задачу:
      w = argmin ||H w - d||^2 + λ||w||^2
    где H — свёрточная матрица канала h, d — единичный импульс с задержкой 'delay'.

    Args:
        h        : ISI taps, shape (Lh,)
        eq_len   : длина эквалайзера (Lw)
        delay    : куда “сфокусировать” импульс (0..Lh+Lw-2); None -> Lh-1
        kind     : "zf" или "mmse"
        noise_var: комплексная дисперсия шума на символ (для MMSE)
        reg      : маленькая регуляризация для устойчивости

    Returns:
        (w, delay_used)
    """
    h = np.asarray(h, dtype=complex).ravel()
    if h.size == 0:
        return np.array([1.0 + 0.0j], dtype=complex), 0
    eq_len = int(eq_len)
    if eq_len < 1:
        eq_len = 1

    Lh = int(h.size)
    Lw = eq_len
    Ly = Lh + Lw - 1
    if delay is None:
        delay = Lh - 1
    delay = int(delay)
    delay = max(0, min(Ly - 1, delay))

    # Build convolution matrix H: (Ly, Lw)
    H = np.zeros((Ly, Lw), dtype=complex)
    for col in range(Lw):
        H[col:col + Lh, col] = h

    d = np.zeros(Ly, dtype=complex)
    d[delay] = 1.0 + 0.0j

    kind = (kind or "mmse").lower().strip()
    lam = 0.0
    if kind in ("mmse", "lmmse"):
        lam = float(max(noise_var, 0.0))
    else:
        lam = 0.0

    A = H.conj().T @ H
    if lam > 0.0 or reg > 0.0:
        A = A + (lam + float(reg)) * np.eye(Lw, dtype=complex)
    b = H.conj().T @ d
    w = np.linalg.solve(A, b)
    return np.asarray(w, dtype=complex).ravel(), delay


def equalize_symbols(rx_symbols: np.ndarray, w: np.ndarray, delay: int) -> np.ndarray:
    """
    Применяет FIR-эквалайзер и возвращает поток той же длины, что и rx_symbols,
    вырезая задержку delay.
    """
    y = np.asarray(rx_symbols, dtype=complex).ravel()
    w = np.asarray(w, dtype=complex).ravel()
    if w.size <= 1:
        return y.copy()
    z = np.convolve(y, w, mode="full")
    start = int(max(0, delay))
    out = z[start:start + y.size]
    if out.size < y.size:
        out = np.concatenate([out, np.zeros(y.size - out.size, dtype=complex)])
    return out


def _nearest_constellation_indices(symbols: np.ndarray, constellation_points: np.ndarray) -> np.ndarray:
    """
    Возвращает индексы ближайших точек созвездия для каждого символа.
    """
    y = np.asarray(symbols, dtype=complex).ravel()
    c = np.asarray(constellation_points, dtype=complex).ravel()
    # dist2: (N, M)
    dist2 = np.abs(y[:, None] - c[None, :]) ** 2
    return np.argmin(dist2, axis=1).astype(np.int32)


def carrier_recovery_dd_pll(
    rx_symbols: np.ndarray,
    constellation_points: np.ndarray,
    alpha: float = 0.02,
    beta: float = 0.0004,
) -> tuple[np.ndarray, dict]:
    """
    Decision-Directed PLL (Costas-подобная петля) для компенсации CFO/фазы.

    Работает на symbol-rate (после matched filter + downsample).

    Модель:
      z_k = y_k * exp(-j*phi_k)
      d_k = nearest_constellation(z_k)
      e_k = angle(z_k * conj(d_k))              (фазовая ошибка)
      f_{k+1} = f_k + beta * e_k                (оценка частоты, рад/символ)
      phi_{k+1} = phi_k + f_{k+1} + alpha * e_k (фаза)

    Args:
        rx_symbols            : входные символы (complex)
        constellation_points  : точки созвездия (complex), shape (M,)
        alpha                 : пропорциональный коэффициент петли
        beta                  : интегральный коэффициент петли

    Returns:
        (corrected_symbols, stats)
        stats включает оценку CFO (rad/symbol) и простую диагностическую инфо.
    """
    y = np.asarray(rx_symbols, dtype=complex).ravel()
    c = np.asarray(constellation_points, dtype=complex).ravel()
    if y.size == 0:
        return y.copy(), {"cfo_rad_per_sym": 0.0, "alpha": alpha, "beta": beta}

    alpha = float(alpha)
    beta = float(beta)
    if alpha < 0 or beta < 0:
        raise ValueError("alpha и beta должны быть >= 0")

    phi = 0.0
    freq = 0.0
    out = np.empty_like(y)

    # Чтобы петля запускалась стабильнее, можно не обновляться на первых K символах,
    # но для MVP оставляем сразу обновления.
    for k in range(y.size):
        z = y[k] * np.exp(-1j * phi)
        # decision-directed: ближайшая точка созвездия
        idx = int(_nearest_constellation_indices(np.array([z]), c)[0])
        d = c[idx]
        # фазовая ошибка: угол между z и решением d
        e = np.angle(z * np.conj(d))

        freq += beta * e
        phi += freq + alpha * e
        out[k] = z

    return out, {
        "cfo_rad_per_sym": float(freq),
        "alpha": alpha,
        "beta": beta,
    }


def estimate_cfo_mth_power(rx_symbols: np.ndarray, m: int) -> tuple[float, float]:
    """
    Слепая оценка CFO/фазы для M-PSK методом m-й степени.

    Идея:
      y_k = s_k * exp(j*(phi0 + omega*k)) + n_k
      y_k^m ≈ exp(j*m*(phi0 + omega*k))  (данные PSK исчезают)
      Тогда по приращению фазы можно оценить omega.

    Args:
        rx_symbols : принятые символы (complex)
        m          : порядок PSK (2/4/8/16)

    Returns:
        (omega_rad_per_sym, phi0_rad)
    """
    y = np.asarray(rx_symbols, dtype=complex).ravel()
    if y.size < 2:
        return 0.0, 0.0
    m = int(m)
    if m < 2:
        return 0.0, 0.0

    y_m = y ** m

    # Оценка приращения фазы (рад/символ * m) через среднее произведение соседей
    prod = y_m[1:] * np.conj(y_m[:-1])
    s = np.mean(prod)
    if abs(s) < 1e-20:
        return 0.0, 0.0

    omega_m = float(np.angle(s))
    omega = omega_m / m

    # Оценка начальной фазы: сначала убираем оценённый наклон из y^m,
    # затем берём фазу среднего остатка (стабильнее, чем angle(mean(y^m))).
    k = np.arange(y_m.size, dtype=np.float64)
    y_m_derot = y_m * np.exp(-1j * omega_m * k)
    s0 = np.mean(y_m_derot)
    if abs(s0) < 1e-20:
        phi0 = 0.0
    else:
        phi0_m = float(np.angle(s0))
        phi0 = phi0_m / m

    return omega, phi0


def derotate_symbols(rx_symbols: np.ndarray, omega_rad_per_sym: float, phi0_rad: float = 0.0) -> np.ndarray:
    """
    Компенсирует линейную фазу: exp(-j*(phi0 + omega*k)).
    """
    y = np.asarray(rx_symbols, dtype=complex).ravel()
    if y.size == 0:
        return y.copy()
    k = np.arange(y.size, dtype=np.float64)
    rot = np.exp(-1j * (phi0_rad + omega_rad_per_sym * k))
    return y * rot


def estimate_cfo_from_repeated_preamble(rx_symbols: np.ndarray, half_len: int) -> float:
    """
    CFO оценка по повторяющейся преамбуле из двух одинаковых половин.

    Пусть r[k] = s[k] * exp(j*(phi0 + omega*k)) + n[k]
    Для повторения: s[k+N] = s[k], N = half_len
    Тогда:
      r[k+N] * conj(r[k]) ≈ |s[k]|^2 * exp(j*omega*N)
    И оценка:
      omega ≈ angle(sum_{k=0..N-1} r[k+N] * conj(r[k])) / N

    Args:
        rx_symbols : принятые символы, ожидается длина >= 2*half_len
        half_len  : длина половины (N)

    Returns:
        omega_rad_per_sym (float)
    """
    r = np.asarray(rx_symbols, dtype=complex).ravel()
    N = int(half_len)
    if N <= 0 or r.size < 2 * N:
        return 0.0
    a = r[:N]
    b = r[N:2 * N]
    s = np.vdot(a, b)  # sum(conj(a)*b)
    if abs(s) < 1e-20:
        return 0.0
    return float(np.angle(s) / N)


def estimate_cpe_from_known_preamble(rx_symbols: np.ndarray, tx_symbols: np.ndarray) -> float:
    """
    Оценка CPE (constant phase error) по известной преамбуле.

    Модель:
      r[k] ≈ s[k] * exp(j*phi) + n[k]
      phi_hat = angle(sum_k r[k] * conj(s[k]))

    Args:
        rx_symbols : принятые символы преамбулы (complex)
        tx_symbols : переданные символы преамбулы (complex)

    Returns:
        phi_rad (float) — оценка постоянного фазового сдвига (рад)
    """
    r = np.asarray(rx_symbols, dtype=complex).ravel()
    s = np.asarray(tx_symbols, dtype=complex).ravel()
    n = min(r.size, s.size)
    if n <= 0:
        return 0.0
    z = np.vdot(s[:n], r[:n])  # sum(conj(s)*r)
    if abs(z) < 1e-20:
        return 0.0
    return float(np.angle(z))


def estimate_cfo_cpe_from_known_preamble(
    rx_symbols: np.ndarray,
    tx_symbols: np.ndarray,
) -> tuple[float, float]:
    """
    Data-aided оценка CFO (omega, рад/символ) и CPE (phi0, рад) по известной преамбуле.

    Модель:
      r[k] ≈ s[k] * exp(j*(phi0 + omega*k)) + n[k]
      z[k] = r[k] * conj(s[k]) ≈ exp(j*(phi0 + omega*k)) + noise
      unwrap(angle(z[k])) ≈ phi0 + omega*k

    Оценка:
      линейная регрессия фазы (weighted least squares по |z|).

    Returns:
      (omega_rad_per_sym, phi0_rad)
    """
    r = np.asarray(rx_symbols, dtype=complex).ravel()
    s = np.asarray(tx_symbols, dtype=complex).ravel()
    n = min(r.size, s.size)
    if n < 4:
        return 0.0, 0.0

    z = r[:n] * np.conj(s[:n])
    ph = np.unwrap(np.angle(z))
    k = np.arange(n, dtype=np.float64)

    w = np.abs(z).astype(np.float64)
    w = np.maximum(w, 1e-6)
    w = w / np.mean(w)

    # Weighted least squares for y = a + b*k
    S0 = float(np.sum(w))
    S1 = float(np.sum(w * k))
    S2 = float(np.sum(w * k * k))
    T0 = float(np.sum(w * ph))
    T1 = float(np.sum(w * k * ph))

    det = S0 * S2 - S1 * S1
    if abs(det) < 1e-20:
        return 0.0, float(np.angle(np.mean(z)))

    phi0 = (T0 * S2 - T1 * S1) / det
    omega = (T1 * S0 - T0 * S1) / det
    return float(omega), float(phi0)


def estimate_cfo_cpe_ml_from_known_preamble(
    rx_symbols: np.ndarray,
    tx_symbols: np.ndarray,
    omega_coarse: float = 0.0,
    search_bw: float = 0.004,
    n_grid: int = 401,
) -> tuple[float, float]:
    """
    ML-подобная оценка CFO+CPE по известной преамбуле без unwrap.

    Рассматриваем:
      z[k] = r[k] * conj(s[k]) ≈ exp(j*(phi0 + omega*k)) + noise
    Тогда критерий для omega:
      J(omega) = |sum_k z[k] * exp(-j*omega*k)|
    Максимум достигается около истинного omega.
    После выбора omega:
      phi0 = angle(sum_k z[k] * exp(-j*omega*k))

    Args:
        omega_coarse : начальная грубая оценка (рад/символ)
        search_bw    : полуширина поиска вокруг omega_coarse (рад/символ)
        n_grid       : число точек сетки (>=11)
    """
    r = np.asarray(rx_symbols, dtype=complex).ravel()
    s = np.asarray(tx_symbols, dtype=complex).ravel()
    n = min(r.size, s.size)
    if n < 8:
        return 0.0, 0.0
    n_grid = int(n_grid)
    if n_grid < 11:
        n_grid = 11

    z = r[:n] * np.conj(s[:n])
    k = np.arange(n, dtype=np.float64)

    w0 = float(omega_coarse)
    bw = float(abs(search_bw))
    if bw <= 0:
        bw = 0.001

    grid = np.linspace(w0 - bw, w0 + bw, n_grid, dtype=np.float64)
    # compute J for each omega (vectorized)
    exps = np.exp(-1j * (k[None, :] * grid[:, None]))  # (G, n)
    sgrid = exps @ z  # (G,)
    J = np.abs(sgrid)
    imax = int(np.argmax(J))

    # Квадратичная интерполяция вокруг пика (субсеточная поправка omega).
    omega_hat = float(grid[imax])
    if 0 < imax < (n_grid - 1):
        y1 = float(J[imax - 1])
        y2 = float(J[imax])
        y3 = float(J[imax + 1])
        denom = (y1 - 2.0 * y2 + y3)
        if abs(denom) > 1e-20:
            delta = 0.5 * (y1 - y3) / denom  # в шагах сетки
            step = float(grid[1] - grid[0])
            omega_hat = float(grid[imax] + delta * step)

    # Пересчитать сумму для omega_hat, чтобы согласовать phi
    s_hat = np.sum(z * np.exp(-1j * omega_hat * k))
    phi_hat = float(np.angle(s_hat)) if abs(s_hat) > 1e-20 else 0.0
    return omega_hat, phi_hat


def carrier_recovery_costas_mpsk(
    rx_symbols: np.ndarray,
    m: int,
    alpha: float = 0.02,
    beta: float = 0.0004,
) -> tuple[np.ndarray, dict]:
    """
    Costas-подобная петля для M-PSK (blind M-th power detector).

    Работает так:
      z_k = y_k * exp(-j*phi_k)
      e_k = angle(z_k^M) / M
    И дальше PI-регулятор как в PLL.

    Плюсы: не требует решений ближайшей точки (устойчивее при плохом SNR).
    Минусы: работает именно для PSK (M-степень “убирает данные”).
    """
    y = np.asarray(rx_symbols, dtype=complex).ravel()
    if y.size == 0:
        return y.copy(), {"cfo_rad_per_sym": 0.0, "alpha": alpha, "beta": beta, "loop": "costas_mpsk"}

    m = int(m)
    if m < 2:
        return y.copy(), {"cfo_rad_per_sym": 0.0, "alpha": alpha, "beta": beta, "loop": "costas_mpsk"}

    alpha = float(alpha)
    beta = float(beta)
    if alpha < 0 or beta < 0:
        raise ValueError("alpha и beta должны быть >= 0")

    phi = 0.0
    freq = 0.0
    out = np.empty_like(y)

    for k in range(y.size):
        z = y[k] * np.exp(-1j * phi)
        # Ошибка по M-й степени (blind)
        e = float(np.angle(z ** m) / m)
        freq += beta * e
        phi += freq + alpha * e
        out[k] = z

    return out, {
        "cfo_rad_per_sym": float(freq),
        "alpha": alpha,
        "beta": beta,
        "loop": "costas_mpsk",
        "m": m,
    }

