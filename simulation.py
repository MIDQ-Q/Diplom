"""
simulation.py — Ядро симуляции цифровой связи. Python 3.12+

Пайплайн TX: данные → шифр → FEC → модуляция
Пайплайн RX: демодуляция → FEC → дешифр → сравнение

Кодирование: Hamming (7,4), Turbo (PCCC ≈ 1/3), Reed-Solomon (RSCodec)
Шифрование:  AES-128-CBC, AES-128-CTR
Модуляция:   M-PSK, M-QAM
"""

from pathlib import Path
import logging
import time
import numpy as np
from datetime import datetime
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_utils import normalize_config
from encryption import get_cipher, compute_encryption_stats, aes_available
from modulation import (
    PSKModulator, QAMModulator,
    theoretical_ber_psk, theoretical_ser_psk,
    theoretical_ber_qam, theoretical_ser_qam,
    theoretical_ber_rayleigh_psk, theoretical_ber_rayleigh_qam,
)
from coding import (
    HammingCoder,
    TurboCoder,
    ReedSolomonCoder,          # новый класс
    compute_coding_gain,
)
from channel import CompositeChannelModel
from interleaving import get_interleaver
from waveform_phy import WaveformPhy

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Конвертация текст ↔ биты
# ══════════════════════════════════════════════════════════════════════════════

def text_to_bits(text: str, encoding: str = "utf-8") -> np.ndarray:
    """Текст → одномерный массив бит (uint8), через np.unpackbits."""
    raw = np.frombuffer(text.encode(encoding), dtype=np.uint8)
    return np.unpackbits(raw)


def bits_to_text(bits: np.ndarray, encoding: str = "utf-8") -> str:
    """Биты → строка через np.packbits."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    pad = (-len(b)) % 8
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    raw = np.packbits(b)
    return raw.tobytes().decode(encoding, errors="replace")


def compare_texts(original: str, received: str) -> dict:
    """
    Посимвольное сравнение строк.
    Возвращает correct_percentage и edit_distance_sample (первые 1000 символов).
    """
    compared = min(len(original), len(received))
    correct  = sum(1 for i in range(compared) if original[i] == received[i])
    correct_pct = (correct / compared * 100) if compared > 0 else 0.0

    sample_len = min(compared, 1000)
    a, b = original[:sample_len], received[:sample_len]
    prev = list(range(len(b) + 1))
    for ch_a in a:
        curr = [prev[0] + 1]
        for j, ch_b in enumerate(b):
            curr.append(min(prev[j] + (ch_a != ch_b), curr[-1] + 1, prev[j + 1] + 1))
        prev = curr
    edit_dist_sample = prev[-1]

    return {
        "total_original_chars": len(original),
        "total_received_chars": len(received),
        "compared_chars":       compared,
        "correct_chars":        correct,
        "correct_percentage":   correct_pct,
        "edit_distance_sample": edit_dist_sample,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Фабричные функции
# ══════════════════════════════════════════════════════════════════════════════

def create_modulator(config: dict) -> PSKModulator | QAMModulator:
    mod_type = config["modulation"]["type"]
    order    = config["modulation"]["order"]
    use_gray = config["modulation"]["use_gray_code"]
    match mod_type:
        case "PSK": return PSKModulator(M=order, use_gray_code=use_gray)
        case "QAM": return QAMModulator(M=order, use_gray_code=use_gray)
        case _:     raise ValueError(f"Неизвестный тип модуляции: {mod_type}")


def create_coder(
    config: dict,
) -> tuple[HammingCoder | TurboCoder | ReedSolomonCoder | None, float]:
    """
    Возвращает (coder | None, code_rate).

    Поддерживаемые типы: "hamming", "turbo", "reed-solomon", (отсутствует → None).
    """
    if not config["coding"]["enabled"]:
        return None, 1.0

    coding_type = config["coding"]["type"]

    match coding_type:
        case "hamming":
            n, k = config["coding"].get("n", 7), config["coding"].get("k", 4)
            return HammingCoder(n=n, k=k), k / n
        case "turbo":
            coder = TurboCoder(
                num_iter=config["coding"].get("turbo_iterations", 6),
                block_size=config["coding"].get("turbo_block_size", 128),
            )
            return coder, coder.code_rate
        case "reed-solomon" | "rs":
            nsym   = config["coding"].get("rs_nsym", 10)
            nsize  = config["coding"].get("rs_nsize", 255)
            fcr    = config["coding"].get("rs_fcr", 0)
            prim   = config["coding"].get("rs_prim", 0x11d)
            coder = ReedSolomonCoder(nsym=nsym, nsize=nsize, fcr=fcr, prim=prim)
            return coder, coder.code_rate
        case _:
            raise ValueError(f"Неизвестный тип кодирования: {coding_type!r}")


def theoretical_ber(config: dict, ebn0_dB: float) -> float:
    mod_type = config["modulation"]["type"]
    order    = config["modulation"]["order"]
    use_gray = config["modulation"]["use_gray_code"]
    if mod_type == "PSK":
        return theoretical_ber_psk(ebn0_dB, order, use_gray)
    if mod_type == "QAM":
        return theoretical_ber_qam(ebn0_dB, order, use_gray)
    return 0.0


def theoretical_ser(config: dict, ebn0_dB: float) -> float:
    mod_type = config["modulation"]["type"]
    order    = config["modulation"]["order"]
    use_gray = config["modulation"]["use_gray_code"]
    if mod_type == "PSK":
        return theoretical_ser_psk(ebn0_dB, order, use_gray)
    if mod_type == "QAM":
        return theoretical_ser_qam(ebn0_dB, order, use_gray)
    return 0.0


def theoretical_ber_rayleigh(config: dict, ebn0_dB: float) -> float:
    """Теоретический BER для Rayleigh-канала (MGF-подход)."""
    mod_type = config["modulation"]["type"]
    order    = config["modulation"]["order"]
    if mod_type == "PSK":
        return theoretical_ber_rayleigh_psk(ebn0_dB, order)
    if mod_type == "QAM":
        return theoretical_ber_rayleigh_qam(ebn0_dB, order)
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════════════

def _compute_ser(mod: PSKModulator | QAMModulator,
                 tx_bits: np.ndarray,
                 rx_bits: np.ndarray,
                 ber: float) -> float:
    """
    SER: BPSK/QAM-4 аналитически через BER; остальные — явный подсчёт.
    """
    if mod.M == 2 or (mod.M == 4 and isinstance(mod, QAMModulator)):
        return 1.0 - (1.0 - ber) ** mod.bits_per_symbol if ber < 1.0 else 1.0
    bps = mod.bits_per_symbol
    total_sym = min(len(tx_bits), len(rx_bits)) // bps
    if total_sym == 0:
        return 0.0
    tx_g = tx_bits[:total_sym * bps].reshape(total_sym, bps)
    rx_g = rx_bits[:total_sym * bps].reshape(total_sym, bps)
    return int(np.any(tx_g != rx_g, axis=1).sum()) / total_sym


def _compute_per(tx_bits: np.ndarray,
                 rx_bits: np.ndarray,
                 packet_size: int) -> tuple[float, int, int]:
    """
    PER = доля пакетов с хотя бы одной битовой ошибкой.

    Returns:
        (per, num_packet_errors, total_packets)
    """
    n = min(len(tx_bits), len(rx_bits))
    if packet_size <= 0 or n == 0:
        return 0.0, 0, 0
    total_packets = n // packet_size
    if total_packets == 0:
        return 0.0, 0, 0
    tx_p = tx_bits[:total_packets * packet_size].reshape(total_packets, packet_size)
    rx_p = rx_bits[:total_packets * packet_size].reshape(total_packets, packet_size)
    errs = int(np.any(tx_p != rx_p, axis=1).sum())
    return errs / total_packets, errs, total_packets


def _adaptive_num_bits(base_bits: int,
                       prev_ber: float | None,
                       max_bits: int = 10_000_000) -> int:
    """
    Адаптивное масштабирование числа бит.

    prev_ber < 1e-5  → ×100
    prev_ber < 1e-4  → ×10
    иначе            → base_bits

    Результат ограничен max_bits.
    """
    if prev_ber is None:
        return base_bits
    if prev_ber < 1e-5:
        return min(base_bits * 100, max_bits)
    if prev_ber < 1e-4:
        return min(base_bits * 10, max_bits)
    return base_bits


def _empty_coding_stats() -> dict:
    return {
        "corrected_errors": 0,
        "detected_errors":  0,
        "total_blocks":     0,
        "error_positions":  [],
        "encode_time_ms":   0.0,
        "decode_time_ms":   0.0,
        "fec_gain":         0,
    }


def _log_config(config: dict, mode: str,
                data_bits_len: int | None = None,
                text_len: int | None = None) -> None:
    """Логирует полную конфигурацию симуляции один раз в начале прогона."""
    lines = [
        "=" * 80,
        "НОВАЯ СИМУЛЯЦИЯ",
        f"Режим: {mode}",
        (f"Модуляция: {config['modulation']['type']}-{config['modulation']['order']}, "
         f"Код Грея: {config['modulation']['use_gray_code']}"),
    ]

    wf = config.get("waveform", {})
    if wf.get("enabled", False):
        sps = wf.get("sps", "?")
        beta = wf.get("rrc_beta", "?")
        span = wf.get("rrc_span_symbols", "?")
        cfo = wf.get("cfo", {})
        cfo_label = (
            f", CFO_norm={cfo.get('cfo_norm', 0.0)}"
            if cfo.get("enabled", False) else ""
        )
        lines.append(f"Waveform: sps={sps}, RRC(beta={beta}, span={span}){cfo_label}")
    else:
        lines.append("Waveform: symbol-rate (1 отсчёт/символ)")
    if config["coding"]["enabled"]:
        ct = config["coding"]["type"]
        if ct == "turbo":
            lines.append("Кодирование: Turbo (PCCC), скорость ≈ 1/3")
        elif ct == "reed-solomon" or ct == "rs":
            nsym = config["coding"].get("rs_nsym", 10)
            nsize = config["coding"].get("rs_nsize", 255)
            lines.append(f"Кодирование: Reed-Solomon (nsym={nsym}, nsize={nsize}), скорость ≈ {nsize-nsym}/{nsize}")
        else:
            n, k = config["coding"].get("n", "?"), config["coding"].get("k", "?")
            lines.append(f"Кодирование: {ct} ({n},{k}), скорость={k}/{n}")
    else:
        lines.append("Кодирование: отключено")

    ch = config["channel"]
    active_ch: list[str] = []
    channel_log_map = [
        ("rayleigh",         lambda c: f"Rayleigh(n_rays={c.get('n_rays', 16)}, доплер={c.get('normalized_doppler', 0.01)})"),
        ("phase_noise",      lambda c: f"PhaseNoise(σ={c.get('phase_noise_std_deg', '?')}°)"),
        ("impulse_noise",    lambda c: (
            f"ImpulseNoise(mode={c.get('mode', 'bernoulli')}, "
            f"snr={c.get('impulse_snr_dB', 20.0)}дБ, "
            f"p={c.get('impulse_probability', 0.001)})"
        )),
    ]
    for key, label_fn in channel_log_map:
        sub = ch.get(key, {})
        if sub.get("enabled", False):
            active_ch.append(label_fn(sub))

    lines.append(f"Каналы: {', '.join(active_ch) or 'AWGN'}")

    per_cfg = config.get("per_settings", {})
    if per_cfg.get("enabled", False):
        lines.append(f"PER: пакет {per_cfg.get('packet_size', 1024)} бит")

    if data_bits_len is not None:
        lines.append(f"Длина данных: {data_bits_len} бит")
    if text_len is not None:
        lines.append(f"Длина текста: {text_len} символов")

    rng  = config["ebn0_dB_range"]
    step = round(rng[1] - rng[0], 4) if len(rng) > 1 else "N/A"
    lines.append(f"Диапазон Eb/N0: {min(rng)}..{max(rng)} дБ, шаг {step}")
    lines.append("=" * 80)
    for line in lines:
        logger.info(line)


# ══════════════════════════════════════════════════════════════════════════════
# Основные функции симуляции
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(
    config:    dict,
    data_bits: np.ndarray,
    ebn0_dB:   float,
    prev_ber:  float | None = None,
    log_prefix: str = "",
) -> dict:
    """
    Единый внутренний пайплайн симуляции одной SNR-точки.

    Принимает уже подготовленные информационные биты (до кодирования).
    Адаптивное масштабирование num_bits и логирование конфига —
    ответственность вызывающей функции.

    Args:
        config     : конфигурация симуляции
        data_bits  : информационные биты (uint8 ndarray)
        ebn0_dB    : Eb/N0 в дБ
        prev_ber   : BER предыдущей точки (используется только для adaptive_scale)
        log_prefix : префикс строки лога ("" для random, "TEXT " для text)

    Returns:
        dict со всеми метриками, включая 'decoded_bits' (np.ndarray).
        Поле 'adaptive_scale' равно 1 если prev_ber не передан.
        Поля 'text', 'original_text', 'text_comparison' НЕ включены —
        их добавляет simulate_text_transmission().
    """
    mod              = create_modulator(config)
    coder, code_rate = create_coder(config)
    channel_cfg      = config.get("channel", {})
    channel          = CompositeChannelModel(channel_cfg)

    # ── Опциональные метрики (важно: инициализировать для symbol-rate режима) ──
    evm_before = None
    evm_after = None
    const_before = None
    const_after = None
    eye_before = None
    eye_after = None
    eye_metrics = None
    evm_over_time = None
    timing_stats: dict = {"timing_enabled": False}
    cfo_stats: dict = {"enabled": False}
    adc_stats: dict = {"enabled": False}

    # ── Шифрование (до кодирования) ──────────────────────────────────────────
    enc_cfg  = config.get("encryption", {})
    enc_enabled = enc_cfg.get("enabled", False)
    cipher = None
    if enc_enabled:
        cipher = get_cipher(
            cipher_type=enc_cfg.get("type", "none"),
            mode=enc_cfg.get("aes_mode", "CBC"),
            key_hex=enc_cfg.get("key_hex") or None,
        )

    t_enc_cipher = time.perf_counter()
    if cipher is not None:
        encrypted_bits = cipher.encrypt(data_bits)
    else:
        encrypted_bits = data_bits
    encrypt_cipher_ms = (time.perf_counter() - t_enc_cipher) * 1e3

    # ── Кодирование ──────────────────────────────────────────────────────────
    t_enc = time.perf_counter()
    tx_bits = coder.encode(encrypted_bits) if coder is not None else encrypted_bits.copy()
    encode_time_ms = (time.perf_counter() - t_enc) * 1e3

    # ── Перемежение (TX) ─────────────────────────────────────────────────────
    interleaver = get_interleaver(config)
    tx_bits_len_before_il = len(tx_bits)  # длина до интерливинга (без padding)
    if interleaver is not None:
        tx_bits = interleaver.interleave(tx_bits)

    # ── Модуляция + канал ────────────────────────────────────────────────────
    tx_symbols = mod.modulate(tx_bits)
    bps        = mod.bits_per_symbol
    ebn0_lin   = 10.0 ** (ebn0_dB / 10.0)
    snr_lin    = ebn0_lin * code_rate * bps   # Es/N0 (линейный) для символов

    wf_cfg = config.get("waveform", {})
    wf_enabled = bool(wf_cfg.get("enabled", False))

    if wf_enabled:
        wf_out = WaveformPhy(wf_cfg).run(mod=mod, tx_symbols=tx_symbols, snr_lin=snr_lin)
        rx_bits = wf_out["rx_bits"]
        channel_names = wf_out["channel_names"]
        timing_stats = wf_out.get("timing_stats", {"timing_enabled": False})
        cfo_stats = wf_out.get("cfo_stats", {"enabled": False})
        adc_stats = wf_out.get("adc_stats", {"enabled": False})
        evm_before = wf_out.get("evm_before")
        evm_after = wf_out.get("evm_after")
        const_before = wf_out.get("const_before")
        const_after = wf_out.get("const_after")
        eye_before = wf_out.get("eye_before")
        eye_after = wf_out.get("eye_after")
        eye_metrics = wf_out.get("eye_metrics")
        evm_over_time = wf_out.get("evm_over_time")

        """
        Legacy inline waveform-branch code is preserved below for reference,
        but is no longer executed (moved to waveform_phy.py).

        # Discrete-time waveform: sps>1 + RRC(TX/RX) + AWGN on samples (+CFO optional)
        sps = int(wf_cfg.get("sps", 8))
        beta = float(wf_cfg.get("rrc_beta", 0.35))
        span = int(wf_cfg.get("rrc_span_symbols", 10))
        cfo_enabled = bool(wf_cfg.get("cfo", {}).get("enabled", False))
        cfo_norm = float(wf_cfg.get("cfo", {}).get("cfo_norm", 0.0))  # cycles/sample
        cfo_recovery_enabled = bool(wf_cfg.get("cfo", {}).get("recovery_enabled", True))
        pll_alpha = float(wf_cfg.get("cfo", {}).get("pll_alpha", 0.02))
        pll_beta = float(wf_cfg.get("cfo", {}).get("pll_beta", 0.0004))
        preamble_enabled = bool(wf_cfg.get("cfo", {}).get("preamble_enabled", True))
        pre_half_len = int(wf_cfg.get("cfo", {}).get("preamble_half_len_symbols", 64))
        loop_type = str(wf_cfg.get("cfo", {}).get("loop_type", "dd_pll")).lower().strip()
        ml_bw = float(wf_cfg.get("cfo", {}).get("ml_search_bw", 0.004))
        ml_grid = int(wf_cfg.get("cfo", {}).get("ml_grid_points", 401))

        # Если петля отключена, повышаем точность за счёт более длинной преамбулы.
        # Это реалистично: без сопровождения нужен более длинный пилот/преамбула.
        pll_off = (pll_alpha <= 0.0 and pll_beta <= 0.0)
        if pll_off and preamble_enabled and pre_half_len < 256:
            pre_half_len = 256

        h_rrc = rrc_taps(beta=beta, span_symbols=span, sps=sps)

        # ── Training sequence (не завязана на CFO) ──────────────────────────
        tr_cfg = wf_cfg.get("training", {}) if isinstance(wf_cfg.get("training", {}), dict) else {}
        tr_enabled = bool(tr_cfg.get("enabled", False))
        tr_len = int(tr_cfg.get("length_symbols", 128))
        tr_seed = tr_cfg.get("seed", 12345)
        tx_training = None
        if tr_enabled and tr_len > 0:
            rng_tr = np.random.default_rng(seed=tr_seed)
            tx_training = mod.constellation_points[
                rng_tr.integers(0, len(mod.constellation_points), size=tr_len)
            ]

        # ── CFO preamble (2 одинаковые половины) ─────────────────────────────
        pre_len_total = 0
        tx_preamble = None
        if cfo_enabled and cfo_recovery_enabled and preamble_enabled and pre_half_len > 0:
            rng = np.random.default_rng(seed=pre_half_len)
            pre_half = mod.constellation_points[
                rng.integers(0, len(mod.constellation_points), size=pre_half_len)
            ]
            preamble = np.concatenate([pre_half, pre_half])
            pre_len_total = len(preamble)
            tx_preamble = preamble
            tx_symbols_wf = np.concatenate([preamble, tx_symbols])
        else:
            tx_symbols_wf = tx_symbols

        # ── Pilots in data (symbol-rate) ─────────────────────────────────────
        pilots_cfg = wf_cfg.get("pilots", {}) if isinstance(wf_cfg.get("pilots", {}), dict) else {}
        pilots_enabled = bool(pilots_cfg.get("enabled", False))
        pilot_period = int(pilots_cfg.get("period_symbols", 256))
        pilot_len = int(pilots_cfg.get("length_symbols", 16))
        pilot_seed = pilots_cfg.get("seed", 4242)
        tx_pilot = None
        if pilots_enabled and pilot_period > 0 and pilot_len > 0:
            rng_p = np.random.default_rng(seed=pilot_seed)
            tx_pilot = mod.constellation_points[
                rng_p.integers(0, len(mod.constellation_points), size=pilot_len)
            ]
            # вставляем пилот перед каждым блоком данных длиной pilot_period
            data = tx_symbols_wf
            chunks = []
            for start in range(0, len(data), pilot_period):
                chunks.append(tx_pilot)
                chunks.append(data[start:start + pilot_period])
            tx_symbols_wf = np.concatenate(chunks)

        # Добавляем training в самое начало (перед CFO-переамбулой/пилотами)
        if tx_training is not None:
            tx_symbols_wf = np.concatenate([tx_training, tx_symbols_wf])

        # Сохраняем “истину” TX для метрик/графиков (training/CFO/pilots включены)
        tx_symbols_wf_full = tx_symbols_wf.copy()

        tx_samples = pulse_shaping_tx(tx_symbols_wf, h_rrc=h_rrc, sps=sps, calibrate=True)
        if cfo_enabled and abs(cfo_norm) > 0.0:
            tx_samples = apply_cfo(tx_samples, cfo_norm=cfo_norm)

        # TDL multipath on samples (FIR)
        tdl_cfg = wf_cfg.get("tdl", {}) if isinstance(wf_cfg.get("tdl", {}), dict) else {}
        tdl_enabled = bool(tdl_cfg.get("enabled", False))
        h_tdl = np.array([1.0 + 0.0j], dtype=complex)
        if tdl_enabled:
            delays_sym = tdl_cfg.get("delays_symbols", [0, 1, 3])
            powers_dB = tdl_cfg.get("powers_dB", [0.0, -3.0, -6.0])
            fading = str(tdl_cfg.get("fading", "rayleigh"))
            frac_taps = int(tdl_cfg.get("frac_taps", 21))
            seed = tdl_cfg.get("seed", None)
            rng_tdl = np.random.default_rng(seed=seed) if seed is not None else np.random.default_rng()
            h_tdl = tdl_taps_from_symbol_delays(
                sps=sps,
                delays_symbols=list(delays_sym),
                powers_dB=list(powers_dB),
                normalize=True,
                fading=fading,
                rng=rng_tdl,
                frac_taps=frac_taps,
            )
            tx_samples = apply_tdl(tx_samples, h_tdl)

        # Phase noise (Wiener) on samples (RX LO-like impairment)
        pn_cfg = wf_cfg.get("phase_noise", {}) if isinstance(wf_cfg.get("phase_noise", {}), dict) else {}
        pn_enabled = bool(pn_cfg.get("enabled", False))

        # AWGN на отсчётах.
        # В waveform-режиме мы калибруем RRC так, что после matched filter
        # выборка в момент символа имеет Es ≈ 1. Тогда:
        #   N0 = Es / (Es/N0) = 1 / snr_lin
        #   sigma^2 (на I и Q) = N0/2 = 1/(2*snr_lin)
        sigma = np.sqrt(1.0 / (2.0 * max(snr_lin, 1e-30)))
        noise = sigma * (np.random.randn(len(tx_samples)) + 1j * np.random.randn(len(tx_samples)))
        rx_samples = tx_samples + noise
        if pn_enabled:
            # step_std can be given in degrees or radians
            if "step_std_deg" in pn_cfg:
                step_std_rad = float(pn_cfg.get("step_std_deg", 0.0)) * np.pi / 180.0
            else:
                step_std_rad = float(pn_cfg.get("step_std_rad", 0.0))
            rx_samples = apply_phase_noise_wiener(
                rx_samples,
                step_std_rad=step_std_rad,
                seed=(int(pn_cfg.get("seed")) if pn_cfg.get("seed", None) is not None else None),
            )

        # Timing impairments on samples (ADC clock offset/drift/jitter)
        ti_cfg = wf_cfg.get("timing_impairments", {}) if isinstance(wf_cfg.get("timing_impairments", {}), dict) else {}
        ti_enabled = bool(ti_cfg.get("enabled", False))
        if ti_enabled:
            ppm = float(ti_cfg.get("drift_ppm", 0.0))
            drift_rate = ppm * 1e-6
            rx_samples = time_warp_samples(
                rx_samples,
                offset_samples=float(ti_cfg.get("offset_samples", 0.0)),
                drift_rate=drift_rate,
                jitter_std_samples=float(ti_cfg.get("jitter_std_samples", 0.0)),
                jitter_mode=str(ti_cfg.get("jitter_mode", "white")),
                jitter_pll_bw_norm=float(ti_cfg.get("jitter_pll_bw_norm", 0.01)),
                seed=(int(ti_cfg.get("seed")) if ti_cfg.get("seed", None) is not None else None),
            )

        # AGC + clipping + ADC quantization (receiver front-end)
        adc_cfg = wf_cfg.get("adc", {}) if isinstance(wf_cfg.get("adc", {}), dict) else {}
        adc_enabled = bool(adc_cfg.get("enabled", False))
        adc_stats: dict = {"enabled": adc_enabled}
        if adc_enabled:
            # DC offset
            if bool(adc_cfg.get("dc_enabled", False)):
                rx_samples, st = apply_dc_offset(
                    rx_samples,
                    dc_i=float(adc_cfg.get("dc_i", 0.0)),
                    dc_q=float(adc_cfg.get("dc_q", 0.0)),
                )
                adc_stats["dc_offset"] = st
            # IQ imbalance
            if bool(adc_cfg.get("iq_imbalance_enabled", False)):
                rx_samples, st = apply_iq_imbalance(
                    rx_samples,
                    amp_imbalance_db=float(adc_cfg.get("iq_amp_imbalance_db", 0.0)),
                    phase_imbalance_deg=float(adc_cfg.get("iq_phase_imbalance_deg", 0.0)),
                )
                adc_stats["iq_imbalance"] = st
            # AGC
            if bool(adc_cfg.get("agc_enabled", True)):
                rx_samples, st = agc_block(
                    rx_samples,
                    target_rms=float(adc_cfg.get("agc_target_rms", 1.0)),
                    max_gain=float(adc_cfg.get("agc_max_gain", 100.0)),
                    min_gain=float(adc_cfg.get("agc_min_gain", 0.01)),
                )
                adc_stats["agc"] = st
            # Clipping
            if bool(adc_cfg.get("clipping_enabled", True)):
                rx_samples, st = clip_magnitude(
                    rx_samples,
                    clip_level=float(adc_cfg.get("clip_level", 1.2)),
                )
                adc_stats["clipping"] = st
            # ADC quantization
            if bool(adc_cfg.get("quantization_enabled", True)):
                rx_samples, st = adc_quantize_iq(
                    rx_samples,
                    n_bits=int(adc_cfg.get("n_bits", 10)),
                    full_scale=float(adc_cfg.get("full_scale", 1.0)),
                )
                adc_stats["quantization"] = st
        else:
            adc_stats = {"enabled": False}

        rx_mf = matched_filter_rx(rx_samples, h_rrc=h_rrc)
        # Timing recovery (optional). Default = old fixed downsample.
        timing_cfg = wf_cfg.get("timing", {}) if isinstance(wf_cfg.get("timing", {}), dict) else {}
        timing_enabled = bool(timing_cfg.get("enabled", False))
        if timing_enabled:
            tr_method = str(timing_cfg.get("method", "gardner")).lower().strip()
            if tr_method != "gardner":
                tr_method = "gardner"
            init_off = float(timing_cfg.get("init_offset_samples", 0.0))
            auto_init = bool(timing_cfg.get("auto_init", False))
            auto_diag = None
            if auto_init:
                # Подбор по максимальному eye opening на первых N символах.
                n_tr = int(timing_cfg.get("auto_traces", 200))
                n_grid = int(timing_cfg.get("auto_grid", 33))
                # Используем кусок rx_mf для скорости
                max_len = int(timing_cfg.get("auto_rx_len", min(len(rx_mf), 20000)))
                rx_mf_slice = rx_mf[:max_len]
                init_off, auto_diag = auto_init_offset_by_eye_opening(
                    rx_mf_slice,
                    sps=sps,
                    h_len=len(h_rrc),
                    n_grid=n_grid,
                    n_traces=n_tr,
                    combine=str(timing_cfg.get("auto_combine", "IQ")),
                )
            rx_symbols_wf, timing_stats = gardner_timing_recovery(
                rx_mf,
                n_symbols=len(tx_symbols_wf),
                sps=sps,
                h_len=len(h_rrc),
                alpha=float(timing_cfg.get("alpha", 0.01)),
                beta=float(timing_cfg.get("beta", 0.0001)),
                init_offset_samples=float(init_off),
            )
            timing_stats["auto_init"] = bool(auto_init)
            if auto_init:
                timing_stats["auto_init_diag"] = auto_diag
                timing_stats["init_offset_used"] = float(init_off)
        else:
            rx_symbols_wf = downsample_to_symbols(
                rx_mf, n_symbols=len(tx_symbols_wf), sps=sps, h_len=len(h_rrc)
            )
            timing_stats = {"timing_enabled": False}

        channel_coeff = None

        # CFO recovery: now supports estimation from training/pilots/preamble (data-aided) and blind fallback for PSK
        cfo_est_cfg = wf_cfg.get("cfo", {}) if isinstance(wf_cfg.get("cfo", {}), dict) else {}
        cfo_est_source = str(cfo_est_cfg.get("estimation_source", "preamble")).lower().strip()
        if cfo_enabled and cfo_recovery_enabled and abs(cfo_norm) > 0.0:
            # CFO estimate via repeated preamble if available; fallback to blind for PSK
            omega = 0.0
            phi0 = 0.0
            tr_used = int(tr_len) if (tr_enabled and tx_training is not None) else 0
            carrier_track = None

            # 1) Prefer training/pilots if requested
            if cfo_est_source in ("training", "training_ls") and tx_training is not None and rx_symbols_wf.size >= tr_used and tr_used >= 8:
                omega0, phi00 = estimate_cfo_cpe_from_known_preamble(
                    rx_symbols_wf[:tr_used],
                    tx_training,
                )
                bw = max(float(ml_bw), 0.02 + 0.5 * abs(float(omega0)))
                omega, phi0 = estimate_cfo_cpe_ml_from_known_preamble(
                    rx_symbols_wf[:tr_used],
                    tx_training,
                    omega_coarse=float(omega0),
                    search_bw=bw,
                    n_grid=ml_grid,
                )
            elif cfo_est_source in ("pilots", "pilot") and pilots_enabled and tx_pilot is not None:
                # use the first pilot block after training/preamble insertion
                start_idx = tr_used
                if pre_len_total:
                    start_idx += pre_len_total
                rx_p = rx_symbols_wf[start_idx:start_idx + pilot_len]
                if rx_p.size >= min(8, pilot_len):
                    omega0, phi00 = estimate_cfo_cpe_from_known_preamble(
                        rx_p,
                        tx_pilot[:rx_p.size],
                    )
                    bw = max(float(ml_bw), 0.02 + 0.5 * abs(float(omega0)))
                    omega, phi0 = estimate_cfo_cpe_ml_from_known_preamble(
                        rx_p,
                        tx_pilot[:rx_p.size],
                        omega_coarse=float(omega0),
                        search_bw=bw,
                        n_grid=ml_grid,
                    )
            elif cfo_est_source in ("pilots_track", "pilots_track_lin", "pilot_track", "track") and pilots_enabled and tx_pilot is not None and pilot_len > 0 and pilot_period > 0:
                # Pilot-based tracking:
                #  - estimate omega_k (CFO slope) per pilot
                #  - estimate cpe_k (constant phase error) per pilot after removing slope
                #  - apply correction either piecewise-constant or linear-interpolated between pilots
                start0 = tr_used  # pilots are inserted after training
                pilot_omegas: list[float] = []
                pilot_cpes: list[float] = []
                pilot_starts: list[int] = []

                rx_corr = rx_symbols_wf.copy()
                idx = start0
                seg = 0
                while idx + pilot_len <= rx_symbols_wf.size:
                    rx_p = rx_symbols_wf[idx:idx + pilot_len]
                    if rx_p.size < max(8, min(pilot_len, 16)):
                        break
                    omega_k, phi0_k = estimate_cfo_cpe_from_known_preamble(rx_p, tx_pilot[:rx_p.size])
                    # Optional ML refinement around omega_k
                    bw = max(float(ml_bw), 0.02 + 0.5 * abs(float(omega_k)))
                    omega_k, phi0_k = estimate_cfo_cpe_ml_from_known_preamble(
                        rx_p, tx_pilot[:rx_p.size],
                        omega_coarse=float(omega_k),
                        search_bw=bw,
                        n_grid=ml_grid,
                    )
                    # CPE after removing slope from z[k]=r[k]*conj(s[k])
                    z = rx_p * np.conj(tx_pilot[:rx_p.size])
                    k_idx = np.arange(z.size, dtype=np.float64)
                    z_derot = z * np.exp(-1j * float(omega_k) * k_idx)
                    cpe_k = float(np.angle(np.mean(z_derot))) if z_derot.size else 0.0

                    pilot_omegas.append(float(omega_k))
                    pilot_cpes.append(float(cpe_k))
                    pilot_starts.append(int(idx))

                    # advance by [pilot][data]
                    idx_data = idx + pilot_len
                    data_len = min(int(pilot_period), int(rx_symbols_wf.size - idx_data))
                    idx = idx_data + max(0, data_len)
                    seg += 1

                # Apply tracking correction
                mode = "linear" if cfo_est_source in ("pilots_track_lin",) else "piecewise"
                if pilot_starts:
                    rx_corr = rx_symbols_wf.copy()
                    for k in range(len(pilot_starts)):
                        s0 = int(pilot_starts[k])
                        s1 = int(pilot_starts[k + 1]) if (k + 1) < len(pilot_starts) else None
                        w0 = float(pilot_omegas[k])
                        c0 = float(pilot_cpes[k])
                        if s1 is None:
                            w1 = w0
                            c1 = c0
                            L = int(rx_corr.size - s0)
                        else:
                            w1 = float(pilot_omegas[k + 1])
                            c1 = float(pilot_cpes[k + 1])
                            L = int(s1 - s0)
                        if L <= 0:
                            continue

                        # build omega/cpe over [s0 .. s0+L)
                        n = np.arange(L, dtype=np.float64)
                        if mode == "linear" and L > 1:
                            tau = n / float(L - 1)
                            omega_n = w0 + (w1 - w0) * tau
                            cpe_n = c0 + (c1 - c0) * tau
                        else:
                            omega_n = np.full(L, w0, dtype=np.float64)
                            cpe_n = np.full(L, c0, dtype=np.float64)

                        phase_inc = np.cumsum(omega_n, dtype=np.float64)
                        phase = cpe_n + (phase_inc - omega_n[0])
                        rx_corr[s0:s0 + L] = rx_corr[s0:s0 + L] * np.exp(-1j * phase)

                    rx_symbols_wf = rx_corr

                # For reporting: take first pilot as global estimate (rough)
                if pilot_omegas:
                    omega = float(pilot_omegas[0])
                    phi0 = float(pilot_cpes[0])
                carrier_track = {
                    "pilot_starts": pilot_starts,
                    "omega_est_rad_per_sym": pilot_omegas,
                    "cpe_est_rad": pilot_cpes,
                    "pilot_len": int(pilot_len),
                    "pilot_period": int(pilot_period),
                    "mode": mode,
                }
            # 2) Legacy preamble-based (if available)
            elif tx_preamble is not None and rx_symbols_wf.size >= (tr_used + pre_len_total):
                # Data-aided: грубо CFO по повторению половин, затем уточняем ML-поиском.
                omega0 = 0.0
                if pre_half_len > 0:
                    omega0 = estimate_cfo_from_repeated_preamble(
                        rx_symbols_wf[tr_used:tr_used + pre_len_total],
                        half_len=pre_half_len
                    )
                omega, phi0 = estimate_cfo_cpe_ml_from_known_preamble(
                    rx_symbols_wf[tr_used:tr_used + pre_len_total],
                    tx_preamble,
                    omega_coarse=omega0,
                    search_bw=ml_bw,
                    n_grid=ml_grid,
                )
            elif pre_len_total >= 2 * pre_half_len and rx_symbols_wf.size >= (tr_used + 2 * pre_half_len):
                # Blind (повторение): только CFO
                omega = estimate_cfo_from_repeated_preamble(rx_symbols_wf[tr_used:], half_len=pre_half_len)
                phi0 = 0.0
            elif isinstance(mod, PSKModulator):
                # Blind fallback for PSK
                omega, phi0 = estimate_cfo_mth_power(rx_symbols_wf, m=mod.M)

            # Компенсируем CFO + CPE
            if cfo_est_source not in ("pilots_track", "pilot_track", "track"):
                rx_symbols_wf = derotate_symbols(rx_symbols_wf, omega_rad_per_sym=omega, phi0_rad=phi0)
            cfo_stats = {
                "omega_est_rad_per_sym": float(omega),
                "phi0_est_rad": float(phi0),
                "estimation_source": cfo_est_source,
            }
            # true CFO if known from config (cfo_norm cycles/sample)
            omega_true = 2.0 * np.pi * float(cfo_norm) * float(sps)
            cfo_stats["omega_true_rad_per_sym"] = float(omega_true)
            cfo_stats["omega_err_rad_per_sym"] = float(omega - omega_true)
            if carrier_track is not None:
                cfo_stats["carrier_track"] = carrier_track

            # 3) Fine tracking loop — опционально (можно выключить alpha=beta=0)
            if (pll_alpha > 0.0 or pll_beta > 0.0) and loop_type in ("dd", "dd_pll", "decision", "decision_directed"):
                rx_symbols_wf, _ = carrier_recovery_dd_pll(
                    rx_symbols_wf,
                    constellation_points=mod.constellation_points,
                    alpha=pll_alpha,
                    beta=pll_beta,
                )
            elif (pll_alpha > 0.0 or pll_beta > 0.0) and loop_type in ("costas", "costas_mpsk"):
                if isinstance(mod, PSKModulator):
                    rx_symbols_wf, _ = carrier_recovery_costas_mpsk(
                        rx_symbols_wf,
                        m=mod.M,
                        alpha=pll_alpha,
                        beta=pll_beta,
                    )
                else:
                    # Для QAM Costas_mpsk не применим; fallback на DD-PLL
                    rx_symbols_wf, _ = carrier_recovery_dd_pll(
                        rx_symbols_wf,
                        constellation_points=mod.constellation_points,
                        alpha=pll_alpha,
                        beta=pll_beta,
                    )
        else:
            cfo_stats = {"omega_est_rad_per_sym": 0.0, "phi0_est_rad": 0.0, "estimation_source": cfo_est_source}

        # ── Equalizer (по training / пилотам / идеальный) ────────────────────
        eq_cfg = wf_cfg.get("equalizer", {}) if isinstance(wf_cfg.get("equalizer", {}), dict) else {}
        eq_enabled = bool(eq_cfg.get("enabled", False))
        eq_kind = str(eq_cfg.get("kind", "mmse")).lower().strip()
        if eq_enabled:
            eq_len = int(eq_cfg.get("eq_len", 9))
            eq_delay = eq_cfg.get("delay", None)
            eq_delay = int(eq_delay) if eq_delay is not None else None
            est_method = str(eq_cfg.get("estimation", "ideal")).lower().strip()
            chan_taps = int(eq_cfg.get("channel_taps", 32))
            # noise variance on complex symbol after MF sampling: N0 = 1/snr_lin
            noise_var = float(1.0 / max(snr_lin, 1e-30))
            ls_reg = float(eq_cfg.get("ls_reg", 1e-3))

            # Индексы разметки кадра в rx_symbols_wf:
            # [training][cfo_preamble+data_with_optional_pilots]
            tr_used = int(tr_len) if (tr_enabled and tx_training is not None) else 0

            def _design_from_h(h_est: np.ndarray) -> tuple[np.ndarray, int]:
                w, d = design_linear_equalizer(
                    h_est,
                    eq_len=eq_len,
                    delay=eq_delay,
                    kind=eq_kind,
                    noise_var=noise_var,
                )
                return w, d

            if est_method in ("training_ls", "training"):
                if tr_used > 0:
                    rx_tr = rx_symbols_wf[:tr_used]
                    h_sym = estimate_channel_ls(
                        tx_training,
                        rx_tr,
                        channel_len=chan_taps,
                        reg=ls_reg,
                    )
                    w_eq, d_used = _design_from_h(h_sym)
                else:
                    h_sym = estimate_symbol_spaced_channel(
                        h_rrc=h_rrc, sps=sps, h_tdl=h_tdl if tdl_enabled else None, n_symbols=chan_taps
                    )
                    w_eq, d_used = _design_from_h(h_sym)
            elif est_method in ("pilot_ls", "pilots"):
                # коэффициенты будут обновляться кусочно на данных по пилотам
                w_eq, d_used = _design_from_h(
                    estimate_symbol_spaced_channel(
                        h_rrc=h_rrc, sps=sps, h_tdl=h_tdl if tdl_enabled else None, n_symbols=chan_taps
                    )
                )
            elif est_method in ("preamble", "preamble_ls", "ls"):
                # legacy: оценка по CFO-переамбуле (если включена)
                if pre_len_total and tx_preamble is not None:
                    rx_pre = rx_symbols_wf[tr_used:tr_used + pre_len_total]
                    h_sym = estimate_channel_ls(tx_preamble, rx_pre, channel_len=chan_taps, reg=ls_reg)
                    w_eq, d_used = _design_from_h(h_sym)
                else:
                    h_sym = estimate_symbol_spaced_channel(
                        h_rrc=h_rrc, sps=sps, h_tdl=h_tdl if tdl_enabled else None, n_symbols=chan_taps
                    )
                    w_eq, d_used = _design_from_h(h_sym)
            else:
                # "ideal"
                h_sym = estimate_symbol_spaced_channel(
                    h_rrc=h_rrc,
                    sps=sps,
                    h_tdl=h_tdl if tdl_enabled else None,
                    n_symbols=chan_taps,
                )
                w_eq, d_used = _design_from_h(h_sym)

        # ── Снять training + CFO-преамбулу, затем убрать пилоты ───────────────
        tr_used = int(tr_len) if (tr_enabled and tx_training is not None) else 0
        start_data = tr_used + (pre_len_total if pre_len_total else 0)
        rx_stream = rx_symbols_wf[start_data:]
        tx_stream = tx_symbols_wf_full[start_data:]

        # Эквализация: либо одним блоком, либо с обновлением по пилотам
        # Также собираем потоки “до EQ / после EQ” для EVM/созвездия.
        rx_before_eq_stream = rx_stream
        if eq_enabled and est_method in ("pilot_ls", "pilots") and pilots_enabled and tx_pilot is not None:
            # структура: [pilot][data_chunk][pilot][data_chunk]...
            out_chunks = []
            tx_chunks = []
            idx = 0
            # для непрерывности FIR между чанками данных используем overlap на входе
            # overlap_len = (Lw-1) + delay
            in_hist = np.array([], dtype=complex)
            while idx < rx_stream.size:
                # pilot
                rx_p = rx_stream[idx:idx + pilot_len]
                tx_p = tx_stream[idx:idx + pilot_len]
                if rx_p.size < pilot_len:
                    break
                h_hat = estimate_channel_ls(tx_pilot, rx_p, channel_len=chan_taps, reg=ls_reg)
                w_eq, d_used = design_linear_equalizer(
                    h_hat, eq_len=eq_len, delay=eq_delay, kind=eq_kind, noise_var=noise_var
                )
                idx += pilot_len
                # data until next pilot (pilot_period) or remainder
                rx_d = rx_stream[idx:idx + pilot_period]
                tx_d = tx_stream[idx:idx + pilot_period]

                overlap_len = max(0, (len(w_eq) - 1) + int(d_used))
                if overlap_len > 0:
                    # держим немного истории входа, чтобы фильтр не “обнулялся” между блоками
                    if in_hist.size > overlap_len:
                        in_hist = in_hist[-overlap_len:]
                    x_blk = np.concatenate([in_hist, rx_d])
                    y_blk = equalize_symbols(x_blk, w_eq, delay=d_used)
                    rx_d_eq = y_blk[in_hist.size:]
                    # обновить историю входа (на границе блоков)
                    in_hist = x_blk[-overlap_len:]
                else:
                    rx_d_eq = equalize_symbols(rx_d, w_eq, delay=d_used)

                out_chunks.append(rx_d_eq)
                tx_chunks.append(tx_d)
                idx += rx_d.size
            rx_symbols = np.concatenate(out_chunks) if out_chunks else np.array([], dtype=complex)
            tx_symbols_ref = np.concatenate(tx_chunks) if tx_chunks else np.array([], dtype=complex)
        else:
            rx_symbols = equalize_symbols(rx_stream, w_eq, delay=d_used) if eq_enabled else rx_stream
            tx_symbols_ref = tx_stream.copy()

        # Удаляем пилоты из потока (если пилоты включены, но эквализация не делала удаление)
        if pilots_enabled and tx_pilot is not None:
            chunks = []
            tx_chunks = []
            i = 0
            while i < rx_symbols.size:
                # skip pilot
                i += pilot_len
                if i >= rx_symbols.size:
                    break
                chunks.append(rx_symbols[i:i + pilot_period])
                tx_chunks.append(tx_symbols_ref[i:i + pilot_period])
                i += pilot_period
            rx_symbols = np.concatenate(chunks) if chunks else np.array([], dtype=complex)
            tx_symbols_ref = np.concatenate(tx_chunks) if tx_chunks else np.array([], dtype=complex)

        # ── EVM + созвездие до/после EQ ──────────────────────────────────────
        evm_before = None
        evm_after = None
        const_before = None
        const_after = None
        eye_before = None
        eye_after = None
        eye_metrics = None
        evm_over_time = None
        if tx_symbols_ref.size and rx_symbols.size:
            plots_cfg = wf_cfg.get("plots", {}) if isinstance(wf_cfg.get("plots", {}), dict) else {}
            n = int(min(tx_symbols_ref.size, rx_symbols.size))
            txr = tx_symbols_ref[:n]
            rxa = rx_symbols[:n]
            # “до EQ”: если EQ выключен, то совпадает с rxa; иначе берём rx_stream без EQ, но уже без training/preamble/pilots
            # Для простоты и корректности по длине используем rx_before_eq_stream, затем вырезаем как и tx_symbols_ref.
            if eq_enabled:
                rx0 = rx_before_eq_stream.copy()
                # удалить пилоты из rx0 по той же схеме
                if pilots_enabled and tx_pilot is not None:
                    rx0_chunks = []
                    j = 0
                    while j < rx0.size:
                        j += pilot_len
                        if j >= rx0.size:
                            break
                        rx0_chunks.append(rx0[j:j + pilot_period])
                        j += pilot_period
                    rx0 = np.concatenate(rx0_chunks) if rx0_chunks else np.array([], dtype=complex)
                rx0 = rx0[:n] if rx0.size >= n else np.pad(rx0, (0, n - rx0.size))
            else:
                rx0 = rxa

            # RMS EVM, нормировано на мощность TX
            p = float(np.mean(np.abs(txr) ** 2)) if txr.size else 1.0
            p = p if p > 1e-30 else 1.0
            evm_before = float(np.sqrt(np.mean(np.abs(rx0 - txr) ** 2) / p))
            evm_after = float(np.sqrt(np.mean(np.abs(rxa - txr) ** 2) / p))

            # EVM over time (chunked)
            chunk = int(plots_cfg.get("evm_chunk_symbols", 256))
            chunk = max(32, min(8192, chunk))
            evm_list = []
            for i0 in range(0, n, chunk):
                i1 = min(n, i0 + chunk)
                if i1 - i0 < 8:
                    break
                txc = txr[i0:i1]
                rxc = rxa[i0:i1]
                pc = float(np.mean(np.abs(txc) ** 2))
                pc = pc if pc > 1e-30 else 1.0
                evm_c = float(np.sqrt(np.mean(np.abs(rxc - txc) ** 2) / pc))
                evm_list.append({"start": int(i0), "len": int(i1 - i0), "evm_rms": evm_c})
            evm_over_time = evm_list if evm_list else None
            if bool(plots_cfg.get("constellation", True)) and (eq_enabled or bool(plots_cfg.get("constellation_always", False))):
                n_pts = int(plots_cfg.get("constellation_points", 2000))
                n_pts = max(100, min(20000, n_pts))
                sel = slice(0, min(n_pts, n))
                const_before = np.column_stack([rx0[sel].real, rx0[sel].imag]).tolist()
                const_after = np.column_stack([rxa[sel].real, rxa[sel].imag]).tolist()

            # Eye diagram (до/после timing recovery)
            if bool(plots_cfg.get("eye_diagram", False)):
                n_tr = int(plots_cfg.get("eye_traces", 200))
                n_tr = max(20, min(2000, n_tr))
                # До: просто фиксированная сетка по sps вокруг идеального start (2*gd)
                gd = (len(h_rrc) - 1) // 2
                start = 2 * gd
                win = 2 * int(sps)
                traces_i = []
                traces_q = []
                for k in range(min(n_tr, max(0, (len(rx_mf) - start) // int(sps) - 2))):
                    i0 = start + k * int(sps) - int(sps)
                    if i0 < 0:
                        continue
                    seg = rx_mf[i0:i0 + win]
                    if seg.size != win:
                        break
                    traces_i.append(seg.real.astype(float).tolist())
                    traces_q.append(seg.imag.astype(float).tolist())
                eye_before = {"I": traces_i, "Q": traces_q} if traces_i else None

                # После: используем t_hist из Gardner, если timing включен
                if timing_stats.get("timing_enabled", False) and isinstance(timing_stats.get("t_hist"), np.ndarray):
                    t_hist = timing_stats["t_hist"]
                    traces2_i = []
                    traces2_q = []
                    for k in range(min(n_tr, int(t_hist.size))):
                        tk = float(t_hist[k])
                        # окно 2 символа вокруг tk: [-sps .. +sps)
                        ts = tk - float(sps) + np.arange(win, dtype=np.float64)
                        seg = np.array([interp_lagrange4(rx_mf, float(tt)) for tt in ts], dtype=complex)
                        traces2_i.append(seg.real.astype(float).tolist())
                        traces2_q.append(seg.imag.astype(float).tolist())
                    eye_after = {"I": traces2_i, "Q": traces2_q} if traces2_i else None

                # Eye opening metric (heuristic): сравнить std в центре глаза и в середине символа
                # Чем больше (std_mid - std_center), тем "открытее" глаз.
                center_idx = int(sps)  # центр окна (2*sps) для основного решения
                mid_idx = int(sps // 2)
                def _eye_stats(eye_dict, comp_key: str):
                    if not eye_dict or comp_key not in eye_dict:
                        return None
                    tr = eye_dict[comp_key]
                    if not tr:
                        return None
                    arr = np.asarray(tr, dtype=float)
                    if arr.ndim != 2 or arr.shape[1] <= center_idx:
                        return None
                    c = arr[:, center_idx]
                    m = arr[:, mid_idx] if arr.shape[1] > mid_idx else c
                    std_c = float(np.std(c))
                    std_m = float(np.std(m))
                    opening = float(std_m - std_c)
                    opening_norm = float(opening / (std_m + 1e-12))
                    return {"std_center": std_c, "std_mid": std_m, "opening": opening, "opening_norm": opening_norm}

                eye_metrics = {
                    "I": {
                        "before": _eye_stats(eye_before, "I"),
                        "after":  _eye_stats(eye_after, "I"),
                    },
                    "Q": {
                        "before": _eye_stats(eye_before, "Q"),
                        "after":  _eye_stats(eye_after, "Q"),
                    },
                    "center_idx": center_idx,
                    "mid_idx": mid_idx,
                    "window_len": win,
                }

        rx_bits = mod.demodulate(rx_symbols, channel_coeff)
        channel_names = ["AWGN", "RRC(sps)"]
        if timing_stats.get("timing_enabled", False):
            channel_names.append("Timing(Gardner)")
        if tdl_enabled:
            channel_names.append("TDL")
        if eq_enabled:
            channel_names.append(f"EQ({eq_kind.upper()})")
            if est_method in ("training_ls", "training"):
                channel_names.append("CH_EST(training_ls)")
            elif est_method in ("pilot_ls", "pilots"):
                channel_names.append("CH_EST(pilot_ls)")
            elif est_method in ("preamble", "preamble_ls", "ls"):
                channel_names.append("CH_EST(preamble_ls)")
        channel_names += (["CFO"] if (cfo_enabled and abs(cfo_norm) > 0.0) else [])
        if cfo_enabled and cfo_recovery_enabled and abs(cfo_norm) > 0.0:
            channel_names.append("CFO_recovery")
        if pre_len_total:
            channel_names.append("CFO_preamble")
        if tr_used:
            channel_names.append("TRAINING")
        if pilots_enabled and tx_pilot is not None:
            channel_names.append("PILOTS")

        """

    else:
        # Старый symbol-rate режим: комплексный символ = 1 отсчёт
        rx_symbols, channel_coeff = channel.apply_with_coeff(tx_symbols, snr_lin)
        rx_bits = mod.demodulate(rx_symbols, channel_coeff)
        channel_names = channel.get_channel_names()

    # ── Деинтерливинг (RX) ──────────────────────────────────────────────────
    if interleaver is not None:
        # Обрезаем до длины TX-потока (модулятор мог добавить padding по bps)
        rx_bits = rx_bits[:len(tx_bits)]
        rx_bits = interleaver.deinterleave(rx_bits, original_len=tx_bits_len_before_il)

    # ── Декодирование ────────────────────────────────────────────────────────
    t_dec = time.perf_counter()
    if coder is not None:
        decoded_bits, coding_stats = coder.decode(rx_bits)
    else:
        decoded_bits = rx_bits
        coding_stats = _empty_coding_stats()
    decode_time_ms = (time.perf_counter() - t_dec) * 1e3
    coding_stats["encode_time_ms"] = encode_time_ms
    coding_stats["decode_time_ms"] = decode_time_ms

    # ── Дешифровка (после декодирования) ─────────────────────────────────────
    # decoded_bits содержит зашифрованный поток (с ошибками канала после FEC).
    # Обрезаем до длины encrypted_bits перед дешифровкой.
    decoded_bits_raw = decoded_bits[:len(encrypted_bits)]

    t_dec_cipher = time.perf_counter()
    if cipher is not None:
        decrypted_bits = cipher.decrypt(decoded_bits_raw)
        # Выравниваем по длине оригинальных данных
        decrypted_bits = decrypted_bits[:len(data_bits)]
        if len(decrypted_bits) < len(data_bits):
            decrypted_bits = np.concatenate([
                decrypted_bits,
                np.zeros(len(data_bits) - len(decrypted_bits), dtype=np.uint8)
            ])
    else:
        decrypted_bits = decoded_bits_raw[:len(data_bits)]
    decrypt_cipher_ms = (time.perf_counter() - t_dec_cipher) * 1e3

    # ── BER: два уровня ───────────────────────────────────────────────────────
    # 1. BER после декодера (до дешифровки) — показывает влияние канала + FEC
    #    Сравниваем decoded_bits_raw с encrypted_bits (что было подано в кодер)
    min_enc = min(len(encrypted_bits), len(decoded_bits_raw))
    bit_errors_pre = int(np.sum(
        encrypted_bits[:min_enc] != decoded_bits_raw[:min_enc]
    ))
    ber_pre_decrypt = bit_errors_pre / min_enc if min_enc > 0 else 0.0

    # 2. BER после дешифровки — итоговый, сравниваем с оригинальными данными
    min_len    = min(len(data_bits), len(decrypted_bits))
    bit_errors = int(np.sum(data_bits[:min_len] != decrypted_bits[:min_len]))
    ber = bit_errors / min_len if min_len > 0 else 0.0

    ser = _compute_ser(mod, tx_bits, rx_bits, ber_pre_decrypt)

    # ── PER ──────────────────────────────────────────────────────────────────
    per_cfg     = config.get("per_settings", {})
    packet_size = per_cfg.get("packet_size", 1024) if per_cfg.get("enabled", False) else 0
    per, per_err, per_total = _compute_per(
        data_bits[:min_len], decrypted_bits[:min_len], packet_size
    )

    # ── Теоретические кривые ─────────────────────────────────────────────────
    theo_ber_val     = theoretical_ber(config, ebn0_dB)
    theo_ser_val     = theoretical_ser(config, ebn0_dB)
    rayleigh_ber_val = theoretical_ber_rayleigh(config, ebn0_dB)

    # ── Ранняя остановка ─────────────────────────────────────────────────────
    early_stop_ber = config.get("early_stop_ber", 1e-7)
    early_stop = bool(ber < early_stop_ber and early_stop_ber > 0)

    esn0_dB       = 10.0 * np.log10(snr_lin) if snr_lin > 0 else float("-inf")
    # channel_names определили выше (waveform или symbol-rate)

    # Для отображения в логе adaptive_scale считаем по base_bits если доступен
    base_bits = config.get("random_settings", {}).get("num_bits", min_len)
    adaptive_scale = (
        _adaptive_num_bits(
            base_bits, prev_ber,
            max_bits=config.get("random_settings", {}).get("max_adaptive_bits", 10_000_000),
        ) // base_bits
        if base_bits > 0 else 1
    )

    logger.info(
        f"{log_prefix}Eb/N0={ebn0_dB:.2f} dB | Es/N0={esn0_dB:.2f} dB | "
        f"Mod={mod.__class__.__name__}-{mod.M} | Gray={mod.use_gray_code} | "
        f"CodeRate={code_rate:.3f} | Bits={min_len} | BitErrors={bit_errors} | "
        f"BER={ber:.3e} | BER_pre={ber_pre_decrypt:.3e} | SER={ser:.3e} | PER={per:.3e} | "
        f"Cipher={cipher.name if cipher else 'none'} | "
        f"EncodeMs={encode_time_ms:.1f} | DecodeMs={decode_time_ms:.1f} | "
        f"EncryptMs={encrypt_cipher_ms:.1f} | DecryptMs={decrypt_cipher_ms:.1f} | "
        f"Corrected={coding_stats.get('corrected_errors', 0)} | "
        f"Detected={coding_stats.get('detected_errors', 0)} | "
        f"Blocks={coding_stats.get('total_blocks', 0)} | "
        f"Channels={channel_names}"
    )

    # Статистика ошибок с учётом шифрования
    enc_stats = compute_encryption_stats(
        data_bits, decrypted_bits, decoded_bits_raw, cipher
    )
    # Коррекция error_propagation_factor: используем ber_pre_decrypt как знаменатель
    if ber_pre_decrypt > 1e-12:
        enc_stats["error_propagation_factor"] = ber / ber_pre_decrypt
    else:
        enc_stats["error_propagation_factor"] = 1.0

    return {
        "snr":                      ebn0_dB,
        "ber":                      float(ber),
        "ber_pre_decrypt":          float(ber_pre_decrypt),
        "bit_errors":               int(bit_errors),
        "bit_errors_pre_decrypt":   int(bit_errors_pre),
        "ser":                      float(ser),
        "per":                      float(per),
        "per_packet_errors":        per_err,
        "per_total_packets":        per_total,
        "theoretical_ber":          float(theo_ber_val),
        "theoretical_ser":          float(theo_ser_val),
        "rayleigh_theoretical_ber": float(rayleigh_ber_val),
        "Es":                       1.0,
        "Eb":                       1.0 / (bps * code_rate),
        "spectral_efficiency":      float(bps * code_rate),
        "corrected_errors":         coding_stats.get("corrected_errors", 0),
        "detected_errors":          coding_stats.get("detected_errors", 0),
        "total_blocks":             coding_stats.get("total_blocks", 0),
        "encode_time_ms":           encode_time_ms,
        "decode_time_ms":           decode_time_ms,
        "active_channels":          channel_names,
        "num_bits_used":            min_len,
        "early_stop":               early_stop,
        "adaptive_scale":           adaptive_scale,
        # Шифрование
        "encryption_enabled":       enc_enabled,
        "cipher_name":              cipher.name if cipher else "none",
        "encrypt_time_ms":          encrypt_cipher_ms,
        "decrypt_time_ms":          decrypt_cipher_ms,
        "ber_post_decrypt":         enc_stats["ber_post_decrypt"],
        "aes_block_errors":         enc_stats["aes_block_errors"],
        "error_propagation_factor": enc_stats["error_propagation_factor"],
        # decoded_bits для text-режима (дешифрованные)
        "decoded_bits":             decrypted_bits,
        # Waveform metrics (optional)
        "evm_rms_before_eq":        float(evm_before) if evm_before is not None else None,
        "evm_rms_after_eq":         float(evm_after) if evm_after is not None else None,
        "constellation_before_eq":  const_before,
        "constellation_after_eq":   const_after,
        "timing_stats":             timing_stats,
        "eye_before":               eye_before,
        "eye_after":                eye_after,
        "eye_metrics":              eye_metrics,
        "cfo_stats":                cfo_stats,
        "evm_over_time":            evm_over_time,
        "adc_stats":                adc_stats,
    }


def simulate_transmission(config: dict,
                           ebn0_dB: float,
                           data_bits: np.ndarray | None = None,
                           log_config_once: bool = True,
                           prev_ber: float | None = None) -> dict:
    """
    Симуляция передачи случайных бит при фиксированном Eb/N0.

    Args:
        config          : конфигурация симуляции
        ebn0_dB         : Eb/N0 в дБ
        data_bits       : входные биты; None → случайные
        log_config_once : логировать конфиг только один раз
        prev_ber        : BER предыдущей точки для адаптивного масштабирования

    Returns:
        dict: snr, ber, ser, per, theoretical_ber, theoretical_ser,
              rayleigh_theoretical_ber, coding_gain_dB, early_stop, и т.д.
    """
    config = normalize_config(config)
    base_bits = config["random_settings"]["num_bits"]
    num_bits  = _adaptive_num_bits(
        base_bits, prev_ber,
        max_bits=config["random_settings"].get("max_adaptive_bits", 10_000_000),
    )

    if data_bits is None:
        data_bits = np.random.randint(0, 2, num_bits, dtype=np.uint8)

    if log_config_once:
        _log_config(config, "random", data_bits_len=len(data_bits))

    return _run_pipeline(config, data_bits, ebn0_dB, prev_ber=prev_ber)


def simulate_text_transmission(config: dict,
                                text: str,
                                ebn0_dB: float,
                                log_config_once: bool = True) -> dict:
    """
    Симуляция передачи текста при фиксированном Eb/N0.

    Использует apply_with_coeff() для когерентной демодуляции.
    Вычисляет PER если включено в config["per_settings"].
    Возвращает rayleigh_theoretical_ber.
    """
    config = normalize_config(config)
    if log_config_once:
        _log_config(config, "text", text_len=len(text))

    original_bits = text_to_bits(text, config["text_settings"]["text_encoding"])
    max_bits = config["text_settings"]["max_text_length"] * 8
    if len(original_bits) > max_bits:
        original_bits = original_bits[:max_bits]

    result = _run_pipeline(config, original_bits, ebn0_dB, log_prefix="TEXT ")

    # ── Text-специфичные поля ────────────────────────────────────────────────
    encoding = config["text_settings"]["text_encoding"]

    decoded_text = bits_to_text(result["decoded_bits"], encoding)

    text_comparison = compare_texts(text, decoded_text)

    # Логируем CER отдельно (не дублируем весь лог из _run_pipeline)
    logger.info(
        f"TEXT CER={text_comparison['correct_percentage']:.2f}% | "
        f"Eb/N0={ebn0_dB:.2f} dB"
    )

    result["text"]            = decoded_text
    result["original_text"]   = text
    result["text_comparison"] = text_comparison
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Сохранение результатов и графики
# ══════════════════════════════════════════════════════════════════════════════

def save_results_to_text(config: dict,
                          results: list[dict],
                          mode: str,
                          execution_time: float | None = None,
                          output_dir: str = ".") -> str:
    """
    Сохраняет результаты в текстовый файл.
    Включает: Rayleigh-теорию, PER, время encode/decode, coding_gain.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mod_type  = config["modulation"]["type"]
    order     = config["modulation"]["order"]
    mode_abbr = "Text" if mode == "text" else "Random"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = str(Path(output_dir) / f"results_{mod_type}{order}_{mode_abbr}_{timestamp}.txt")

    ber_arr  = np.array([r["ber"] for r in results])
    theo_arr = np.array([r.get("theoretical_ber", 0) for r in results])
    snr_arr  = np.array([r["snr"] for r in results])
    has_per  = any(r.get("per", 0) > 0 for r in results)
    coding_gain = (
        compute_coding_gain(theo_arr, ber_arr, snr_arr)
        if len(ber_arr) > 1 else float("nan")
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 95 + "\n")
        f.write("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ ЦИФРОВОЙ СВЯЗИ\n")
        f.write("=" * 95 + "\n\n")

        f.write(f"Режим:                 {mode}\n")
        f.write(f"Модуляция:             {mod_type}-{order}\n")
        coding_label = (
            config["coding"]["type"].upper()
            if config["coding"]["enabled"] else "ОТСУТСТВУЕТ"
        )
        f.write(f"Кодирование:           {coding_label}\n")
        if config["coding"]["enabled"]:
            ct = config["coding"]["type"]
            if ct == "turbo":
                f.write("  Тип кода:            Turbo (PCCC), скорость ≈ 1/3\n")
            elif ct == "reed-solomon" or ct == "rs":
                nsym = config["coding"].get("rs_nsym", 10)
                nsize = config["coding"].get("rs_nsize", 255)
                f.write(f"  Тип кода:            Reed-Solomon (nsym={nsym}, nsize={nsize})\n")
                f.write(f"  Скорость кода:       {(nsize-nsym)}/{nsize} ≈ {(nsize-nsym)/nsize:.3f}\n")
            else:
                n, k = config["coding"].get("n", "?"), config["coding"].get("k", "?")
                f.write(f"  Тип кода:            ({n},{k})\n")
                f.write(f"  Скорость кода:       {k}/{n}\n")
        f.write(f"Код Грея:              {'ДА' if config['modulation']['use_gray_code'] else 'НЕТ'}\n")
        rng = config["ebn0_dB_range"]
        f.write(f"Eb/N0 диапазон (дБ):   {min(rng):.1f} - {max(rng):.1f}\n")
        if not np.isnan(coding_gain):
            f.write(f"Выигрыш кодирования:   {coding_gain:+.2f} дБ (при BER=1e-4)\n")

        f.write("\nМодели канала:\n")
        active_channels: set[str] = set()
        for r in results:
            active_channels.update(r.get("active_channels", []))
        for ch in sorted(active_channels) or ["AWGN"]:
            f.write(f"  • {ch}\n")

        per_cfg = config.get("per_settings", {})
        if per_cfg.get("enabled", False):
            f.write(f"\nPER (размер пакета):   {per_cfg.get('packet_size', 1024)} бит\n")

        if execution_time:
            f.write(f"\nВремя выполнения:      {execution_time:.2f} сек\n")
        enc_ms = np.mean([r.get("encode_time_ms", 0) for r in results])
        dec_ms = np.mean([r.get("decode_time_ms", 0) for r in results])
        if enc_ms > 0 or dec_ms > 0:
            f.write(f"Среднее время encode:  {enc_ms:.2f} мс\n")
            f.write(f"Среднее время decode:  {dec_ms:.2f} мс\n")

        f.write("\n" + "=" * 95 + "\n\n")

        header = (
            f"{'SNR(дБ)':<10} {'BER':<12} {'SER':<12} "
            f"{'Теор.BER':<13} {'Теор.SER':<13} {'Rayleigh':<13}"
        )
        if has_per:
            header += f" {'PER':<12}"
        if mode == "text":
            header += f" {'CER(%)':<10}"
        f.write(header + "\n")
        f.write("-" * 95 + "\n")

        for r in results:
            cer = (
                r.get("cer", 0) if "cer" in r
                else r.get("text_comparison", {}).get("correct_percentage", 0)
                if mode == "text" else 0
            )
            row = (
                f"{r['snr']:<10.1f} {r['ber']:<12.2e} {r['ser']:<12.2e} "
                f"{r.get('theoretical_ber', 0):<13.2e} {r.get('theoretical_ser', 0):<13.2e} "
                f"{r.get('rayleigh_theoretical_ber', 0):<13.2e}"
            )
            if has_per:
                row += f" {r.get('per', 0):<12.2e}"
            if mode == "text":
                row += f" {cer:<10.2f}"
            f.write(row + "\n")

        f.write("\n" + "=" * 95 + "\n")

    logger.info(f"Результаты сохранены в файл: {filename}")
    return filename


def plot_and_save_results(config: dict,
                           results: list[dict],
                           mode: str,
                           show_theoretical: bool = True,
                           show_rayleigh_theo: bool = False,
                           output_dir: str = ".") -> tuple[str | None, Any]:
    """
    Строит и сохраняет графики.

    Subplots:
      1. BER/SER (всегда) — с аннотацией coding_gain
      2. PER (если есть данные)
      3. CER (только text-режим)
      4. EVM (если доступно evm_rms_* в results)
    """
    if not results:
        return None, None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mod_type  = config["modulation"]["type"]
    order     = config["modulation"]["order"]
    mode_abbr = "Text" if mode == "text" else "Random"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = str(Path(output_dir) / f"Plot_{mod_type}{order}_{mode_abbr}_{timestamp}.png")

    snr = np.array([r["snr"]         for r in results])
    ber = np.array([r["ber"]         for r in results])
    ser = np.array([r["ser"]         for r in results])
    per = np.array([r.get("per", 0)  for r in results])
    has_per = bool(np.any(per > 0))

    theo_ber_arr = np.array([r.get("theoretical_ber",  0) for r in results])
    theo_ser_arr = np.array([r.get("theoretical_ser",  0) for r in results])
    ray_ber_arr  = np.array([r.get("rayleigh_theoretical_ber", 0) for r in results])

    coding_gain = (
        compute_coding_gain(theo_ber_arr, ber, snr)
        if len(ber) > 1 else float("nan")
    )

    CYAN   = "#00d9ff"
    PINK   = "#ff006e"
    GREEN  = "#39ff14"
    YELLOW = "#ffbe0b"
    ORANGE = "#ff7700"
    BG     = "#1a1a2e"

    plt.style.use("dark_background")

    def _style_ax(ax: plt.Axes, title: str, ylabel: str) -> None:
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_xlabel("Eb/N0 (дБ)", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
        ax.legend(fontsize=9, framealpha=0.9)

    evm_before = np.array([r.get("evm_rms_before_eq") for r in results], dtype=object)
    evm_after  = np.array([r.get("evm_rms_after_eq")  for r in results], dtype=object)
    has_evm = any(v is not None for v in evm_before) or any(v is not None for v in evm_after)

    # Определяем число subplot-ов
    n_plots = 1 + int(has_per) + int(mode == "text") + int(has_evm)
    if n_plots == 1:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        axes = [ax1]
    elif n_plots == 2:
        fig, (ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))
        axes = [ax1, _ax2]
    else:
        fig, axarr = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        axes = list(axarr)

    ax1    = axes[0]
    ax_idx = 1

    # ── BER / SER ────────────────────────────────────────────────────────────
    ax1.semilogy(snr, np.clip(ber, 1e-9, 1), "o-",  color=CYAN, lw=2.5, ms=6,
                 label="BER (эксп.)",
                 markerfacecolor=CYAN, markeredgewidth=1.5, markeredgecolor="white")
    ax1.semilogy(snr, np.clip(ser, 1e-9, 1), "s--", color=PINK, lw=2.5, ms=6,
                 label="SER (эксп.)",
                 markerfacecolor=PINK, markeredgewidth=1.5, markeredgecolor="white")

    if show_theoretical:
        ax1.semilogy(snr, np.clip(theo_ber_arr, 1e-9, 1), ":",  color=GREEN,
                     lw=2, label="Теор. BER (AWGN)", alpha=0.85)
        ax1.semilogy(snr, np.clip(theo_ser_arr, 1e-9, 1), "-.", color=YELLOW,
                     lw=2, label="Теор. SER (AWGN)", alpha=0.85)

    if show_rayleigh_theo and np.any(ray_ber_arr > 0):
        ax1.semilogy(snr, np.clip(ray_ber_arr, 1e-9, 1), "--", color=ORANGE,
                     lw=2, label="Теор. BER (Rayleigh)", alpha=0.85)

    ax1.set_ylim(1e-6, 1)

    if not np.isnan(coding_gain):
        ax1.text(0.98, 0.97, f"Выигрыш кодирования: {coding_gain:+.2f} дБ",
                 transform=ax1.transAxes, ha="right", va="top", fontsize=9,
                 color=GREEN,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                           edgecolor=GREEN, alpha=0.8))

    title1 = "BER и SER" if mode == "text" else f"{mod_type}-{order} ({mode_abbr})"
    _style_ax(ax1, title1, "Вероятность ошибки")

    # ── PER ──────────────────────────────────────────────────────────────────
    if has_per:
        ax_per = axes[ax_idx]
        ax_idx += 1
        valid_per = np.where(per > 0, per, np.nan)
        ax_per.semilogy(snr, valid_per, "D-", color=ORANGE, lw=2.5, ms=7,
                        label="PER (эксп.)",
                        markerfacecolor=ORANGE, markeredgewidth=1.5,
                        markeredgecolor="white")
        ax_per.set_ylim(1e-5, 1)
        _style_ax(ax_per, "Packet Error Rate", "PER")

    # ── CER (только text) ────────────────────────────────────────────────────
    if mode == "text":
        ax_cer = axes[ax_idx]
        cer = np.array([
            r.get("cer", 0) if "cer" in r
            else r.get("text_comparison", {}).get("correct_percentage", 0)
            for r in results
        ])
        ax_cer.plot(snr, cer, "^-", color=CYAN, lw=2.5, ms=8,
                    markerfacecolor=CYAN, markeredgewidth=1.5,
                    markeredgecolor="white", label="CER (%)")
        for y_val, col, lbl in [(90, YELLOW, "90%"), (95, GREEN, "95%"), (99, CYAN, "99%")]:
            ax_cer.axhline(y=y_val, color=col, ls="--", alpha=0.5, lw=1.5, label=lbl)
        ax_cer.set_ylim(-2, 105)
        _style_ax(ax_cer, "Качество восстановления текста", "Правильных символов (%)")

    # ── EVM ──────────────────────────────────────────────────────────────────
    if has_evm:
        ax_evm = axes[ax_idx]
        # Переведём в float и проставим nan где None
        def _to_float_nan(arr_obj):
            out = np.full(len(arr_obj), np.nan, dtype=float)
            for i, v in enumerate(arr_obj):
                if v is None:
                    continue
                try:
                    out[i] = float(v)
                except Exception:
                    out[i] = np.nan
            return out
        evm_b = _to_float_nan(evm_before)
        evm_a = _to_float_nan(evm_after)
        if np.any(np.isfinite(evm_b)):
            ax_evm.plot(snr, evm_b, "o-", color=YELLOW, lw=2.5, ms=6,
                        label="EVM до EQ", markerfacecolor=YELLOW, markeredgewidth=1.5, markeredgecolor="white")
        if np.any(np.isfinite(evm_a)):
            ax_evm.plot(snr, evm_a, "s--", color=GREEN, lw=2.5, ms=6,
                        label="EVM после EQ", markerfacecolor=GREEN, markeredgewidth=1.5, markeredgecolor="white")
        ax_evm.set_ylim(0, max(0.5, float(np.nanmax(np.concatenate([evm_b, evm_a])) * 1.1) if np.any(np.isfinite(evm_a)) or np.any(np.isfinite(evm_b)) else 0.5))
        _style_ax(ax_evm, "EVM (RMS)", "EVM (норм.)")

    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(plot_filename, dpi=150, facecolor=BG)
    logger.info(f"График сохранён: {plot_filename}")

    # ── Созвездие до/после EQ (отдельный PNG) ────────────────────────────────
    try:
        last = results[-1] if results else {}
        cb = last.get("constellation_before_eq")
        ca = last.get("constellation_after_eq")
        if cb and ca and isinstance(cb, list) and isinstance(ca, list):
            cb_arr = np.asarray(cb, dtype=float)
            ca_arr = np.asarray(ca, dtype=float)
            if cb_arr.ndim == 2 and cb_arr.shape[1] == 2 and ca_arr.ndim == 2 and ca_arr.shape[1] == 2:
                const_filename = str(Path(output_dir) / f"ConstellationEQ_{mod_type}{order}_{mode_abbr}_{timestamp}.png")
                fig2, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(10, 5))
                fig2.patch.set_facecolor(BG)
                for ax, data, title, color in [
                    (ax_b, cb_arr, "До EQ", CYAN),
                    (ax_a, ca_arr, "После EQ", GREEN),
                ]:
                    ax.set_facecolor(BG)
                    ax.scatter(data[:, 0], data[:, 1], s=6, c=color, alpha=0.6, edgecolors="none")
                    ax.grid(True, alpha=0.25, linestyle="--")
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlabel("I", fontsize=11, fontweight="bold")
                    ax.set_ylabel("Q", fontsize=11, fontweight="bold")
                    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
                fig2.suptitle("Созвездие (последняя точка SNR)", fontsize=13, fontweight="bold", color=TEXT_LIGHT if 'TEXT_LIGHT' in globals() else "white")
                fig2.tight_layout()
                fig2.savefig(const_filename, dpi=150, facecolor=BG)
                plt.close(fig2)
                logger.info(f"Созвездие до/после EQ сохранено: {const_filename}")
    except Exception as e:
        logger.warning(f"Не удалось сохранить созвездие до/после EQ: {e}")

    # ── Eye diagram до/после timing recovery (отдельный PNG) ─────────────────
    try:
        last = results[-1] if results else {}
        eb = last.get("eye_before")
        ea = last.get("eye_after")
        if eb and isinstance(eb, dict) and "I" in eb and "Q" in eb:
            eye_filename = str(Path(output_dir) / f"Eye_{mod_type}{order}_{mode_abbr}_{timestamp}.png")
            fig3, axarr = plt.subplots(2, 2, figsize=(12, 8))
            fig3.patch.set_facecolor(BG)

            def _plot_eye(ax, traces, title, color):
                ax.set_facecolor(BG)
                for tr in traces:
                    arr = np.asarray(tr, dtype=float)
                    ax.plot(arr, color=color, alpha=0.08, lw=1.0)
                ax.grid(True, alpha=0.25, linestyle="--")
                ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
                ax.set_xlabel("Отсчёты", fontsize=11, fontweight="bold")
                ax.set_ylabel("Амплитуда", fontsize=11, fontweight="bold")

            ax_I_b = axarr[0, 0]
            ax_I_a = axarr[0, 1]
            ax_Q_b = axarr[1, 0]
            ax_Q_a = axarr[1, 1]

            _plot_eye(ax_I_b, eb.get("I", []), "Eye I: до timing recovery", CYAN)
            _plot_eye(ax_Q_b, eb.get("Q", []), "Eye Q: до timing recovery", PINK)

            if ea and isinstance(ea, dict):
                _plot_eye(ax_I_a, ea.get("I", []), "Eye I: после timing recovery", GREEN)
                _plot_eye(ax_Q_a, ea.get("Q", []), "Eye Q: после timing recovery", YELLOW)
            else:
                ax_I_a.axis("off")
                ax_Q_a.axis("off")

            fig3.tight_layout()
            fig3.savefig(eye_filename, dpi=150, facecolor=BG)
            plt.close(fig3)
            logger.info(f"Eye diagram сохранён: {eye_filename}")
    except Exception as e:
        logger.warning(f"Не удалось сохранить eye diagram: {e}")

    # ── Carrier tracking plot (omega/phi vs time + EVM over time) ────────────
    try:
        last = results[-1] if results else {}
        cs = last.get("cfo_stats") or {}
        tr = cs.get("carrier_track") if isinstance(cs, dict) else None
        evm_ot = last.get("evm_over_time")
        if tr and isinstance(tr, dict) and tr.get("pilot_starts") and (evm_ot is not None):
            tr_filename = str(Path(output_dir) / f"CarrierTrack_{mod_type}{order}_{mode_abbr}_{timestamp}.png")
            fig4, axarr = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
            fig4.patch.set_facecolor(BG)

            pstarts = np.asarray(tr.get("pilot_starts", []), dtype=float)
            omegas = np.asarray(tr.get("omega_est_rad_per_sym", []), dtype=float)
            cpes = np.asarray(tr.get("cpe_est_rad", []), dtype=float)

            ax_w = axarr[0]
            ax_p = axarr[1]
            ax_e = axarr[2]
            for ax in axarr:
                ax.set_facecolor(BG)
                ax.grid(True, alpha=0.25, linestyle="--")

            if pstarts.size and omegas.size:
                ax_w.plot(np.arange(len(omegas)), omegas, "o-", color=CYAN, lw=2.2, ms=5, label="omega est (rad/sym)")
                if "omega_true_rad_per_sym" in cs:
                    ax_w.axhline(float(cs["omega_true_rad_per_sym"]), color=YELLOW, ls="--", lw=1.6, alpha=0.7, label="omega true")
                ax_w.set_ylabel("ω (rad/sym)", fontweight="bold")
                ax_w.legend(fontsize=9, framealpha=0.9)

            if pstarts.size and cpes.size:
                ax_p.plot(np.arange(len(cpes)), np.unwrap(cpes), "o-", color=GREEN, lw=2.2, ms=5, label="CPE est (rad)")
                ax_p.set_ylabel("CPE φ (rad)", fontweight="bold")
                ax_p.legend(fontsize=9, framealpha=0.9)

            if isinstance(evm_ot, list) and evm_ot:
                xs = [d.get("start", 0) for d in evm_ot]
                ys = [d.get("evm_rms", np.nan) for d in evm_ot]
                ax_e.plot(xs, ys, "s-", color=PINK, lw=2.2, ms=5, label="EVM RMS (chunk)")
                ax_e.set_xlabel("Symbol index", fontweight="bold")
                ax_e.set_ylabel("EVM", fontweight="bold")
                ax_e.legend(fontsize=9, framealpha=0.9)

            fig4.tight_layout()
            fig4.savefig(tr_filename, dpi=150, facecolor=BG)
            plt.close(fig4)
            logger.info(f"Carrier tracking plot сохранён: {tr_filename}")
    except Exception as e:
        logger.warning(f"Не удалось сохранить carrier tracking plot: {e}")
    return plot_filename, fig


# ══════════════════════════════════════════════════════════════════════════════
# Сравнение нескольких прогонов на одном графике
# ══════════════════════════════════════════════════════════════════════════════

# Палитра для сравнения: каждый прогон получает свой цвет
_COMPARISON_PALETTE = [
    "#00d9ff",  # cyan
    "#ff006e",  # pink
    "#39ff14",  # green
    "#ffbe0b",  # yellow
    "#ff7700",  # orange
]

# Маркеры для различения прогонов при ч/б печати
_COMPARISON_MARKERS = ["o", "s", "^", "D", "v"]


def plot_comparison(
    results_list:     list,
    labels:           list,
    show_theoretical: bool = True,
    output_dir:       str  = ".",
) -> tuple:
    """
    Строит BER-кривые нескольких прогонов на одном графике.

    Args:
        results_list     : список прогонов; каждый прогон — список точек (dict),
                           как возвращает simulate_transmission().
        labels           : метки для легенды, len == len(results_list).
        show_theoretical : рисовать теоретический BER (AWGN) для первого прогона.
        output_dir       : папка для сохранения PNG.

    Returns:
        (plot_filename, fig) — путь к файлу и объект Figure.
        Возвращает (None, None) если results_list пуст.

    Примечания:
        - Каждая кривая строится по своим SNR-точкам независимо.
          Разные SNR-сетки у прогонов корректно отображаются без интерполяции.
        - Тот же dark theme что и plot_and_save_results().
    """
    if not results_list or not any(results_list):
        return None, None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = str(Path(output_dir) / f"Compare_{timestamp}.png")

    BG = "#1a1a2e"
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG)

    for run_idx, (points, label) in enumerate(zip(results_list, labels)):
        if not points:
            continue

        color  = _COMPARISON_PALETTE[run_idx % len(_COMPARISON_PALETTE)]
        marker = _COMPARISON_MARKERS[run_idx % len(_COMPARISON_MARKERS)]

        snr = np.array([r["snr"] for r in points])
        ber = np.array([r["ber"] for r in points])

        ax.semilogy(
            snr, np.clip(ber, 1e-9, 1),
            linestyle="-", marker=marker, color=color,
            lw=2.5, ms=6, label=label,
            markerfacecolor=color, markeredgewidth=1.5, markeredgecolor="white",
        )

        # Теоретический BER (AWGN) — только для первого прогона, пунктир того же цвета
        if show_theoretical and run_idx == 0:
            theo = np.array([r.get("theoretical_ber", 0) for r in points])
            if np.any(theo > 0):
                ax.semilogy(
                    snr, np.clip(theo, 1e-9, 1),
                    linestyle=":", color=color, lw=1.5, alpha=0.7,
                    label=f"{label} (теория AWGN)",
                )

    ax.set_ylim(1e-6, 1)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlabel("Eb/N0 (дБ)", fontsize=11, fontweight="bold")
    ax.set_ylabel("BER", fontsize=11, fontweight="bold")
    ax.set_title("Сравнение прогонов — BER", fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(plot_filename, dpi=150, facecolor=BG)
    logger.info(f"График сравнения сохранён: {plot_filename}")
    return plot_filename, fig