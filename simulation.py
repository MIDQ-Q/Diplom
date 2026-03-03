"""
simulation.py — Ядро симуляции цифровой связи.
Python 3.12+

Изменения (2 марта):
─────────────────────────────────────────────────────────────────────
create_coder():
  • Поддержка TurboCoder (coding_type = "turbo").
  • Корректные (n, k) по умолчанию: Hamming(7,4), LDPC(64,32).

_log_config():
  • Убран Rician из описания каналов.
  • Shadowing и Multipath добавлены.
  • Логирование PER packet_size.

_run_pipeline() [новое]:
  • Единый внутренний пайплайн: кодирование → модуляция → канал → декодирование → метрики.
  • Возвращает 'decoded_bits' (np.ndarray) для использования в text-режиме.
  • simulate_transmission / simulate_text_transmission — тонкие обёртки над ним.

simulate_transmission() / simulate_text_transmission():
  • Адаптивное число бит:
      prev_ber < 1e-5  → ×100   (до max_adaptive_bits = 10 млн)
      prev_ber < 1e-4  → ×10
      иначе            → base_bits
  • PER (Packet Error Rate): packet_size через config["per_settings"].
  • Время encode/decode — из coder и замера time.perf_counter().
  • Ранняя остановка: флаг early_stop=True если ber < early_stop_ber.
  • Поле rayleigh_theoretical_ber в каждой точке.

plot_and_save_results():
  • show_rayleigh_theo — флаг для кривых Rayleigh (оранжевый пунктир).
  • График PER как отдельный subplot.
  • Аннотация coding_gain на BER-графике.

save_results_to_text():
  • Колонки: Rayleigh-теория, PER, время encode/decode, coding_gain.
"""

import os
import logging
import time
import numpy as np
from datetime import datetime
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from encryption import get_cipher, compute_encryption_stats, aes_available
from modulation import (
    PSKModulator, QAMModulator,
    theoretical_ber_psk, theoretical_ser_psk,
    theoretical_ber_qam, theoretical_ser_qam,
    theoretical_ber_rayleigh_psk, theoretical_ber_rayleigh_qam,
)
from coding import HammingCoder, LDPCCoder, TurboCoder, compute_coding_gain
from results_manager import ResultsManager
from channel import CompositeChannelModel
from interleaving import get_interleaver
from text_recovery import recover_text, RecoveryResult

results_manager = ResultsManager()
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
    if mod_type == "PSK":
        return PSKModulator(M=order, use_gray_code=use_gray)
    if mod_type == "QAM":
        return QAMModulator(M=order, use_gray_code=use_gray)
    raise ValueError(f"Неизвестный тип модуляции: {mod_type}")


def create_coder(
    config: dict,
) -> tuple[HammingCoder | LDPCCoder | TurboCoder | None, float]:
    """
    Возвращает (coder | None, code_rate).

    Поддерживаемые типы: "hamming", "ldpc", "turbo", (отсутствует → None).
    TurboCoder имеет фиксированную скорость кода ≈ 1/3.
    """
    if not config["coding"]["enabled"]:
        return None, 1.0

    coding_type = config["coding"]["type"]

    if coding_type == "hamming":
        n, k = config["coding"].get("n", 7), config["coding"].get("k", 4)
        return HammingCoder(n=n, k=k), k / n

    if coding_type == "ldpc":
        n, k = config["coding"].get("n", 64), config["coding"].get("k", 32)
        return LDPCCoder(n=n, k=k), k / n

    if coding_type == "turbo":
        num_iter   = config["coding"].get("turbo_iterations", 6)
        block_size = config["coding"].get("turbo_block_size", 128)
        coder = TurboCoder(num_iter=num_iter, block_size=block_size)
        return coder, coder.code_rate

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
    if config["coding"]["enabled"]:
        ct = config["coding"]["type"]
        if ct == "turbo":
            lines.append("Кодирование: Turbo (PCCC), скорость ≈ 1/3")
        else:
            n, k = config["coding"].get("n", "?"), config["coding"].get("k", "?")
            lines.append(f"Кодирование: {ct} ({n},{k}), скорость={k}/{n}")
    else:
        lines.append("Кодирование: отключено")

    ch = config["channel"]
    active_ch: list[str] = []
    channel_log_map = [
        ("rayleigh",         lambda c: f"Rayleigh(n_rays={c.get('n_rays', 16)}, доплер={c.get('normalized_doppler', 0.01)})"),
        ("multipath",        lambda c: f"Multipath(taps={c.get('n_taps', 6)}, доплер={c.get('normalized_doppler', 0.01)})"),
        ("shadowing",        lambda c: f"Shadowing(std={c.get('shadow_std_dB', 8.0)}дБ)"),
        ("phase_noise",      lambda c: f"PhaseNoise(σ²={c.get('phase_noise_variance', '?')})"),
        ("frequency_offset", lambda c: f"FreqOffset(δf={c.get('normalized_freq_offset', '?')})"),
        ("timing_offset",    lambda c: f"TimingOffset(range={c.get('timing_offset_range', '?')})"),
        ("impulse_noise",    lambda c: (
            f"ImpulseNoise(p={c.get('impulse_probability', '?')}, "
            f"A={c.get('impulse_amplitude_sigma', '?')}σ, "
            f"ширина={c.get('impulse_width_from', 1)}-{c.get('impulse_width_to', 5)})"
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
    channel          = CompositeChannelModel(config.get("channel", {}))

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

    # ── Модуляция + канал ────────────────────────────────────────────────────
    tx_symbols = mod.modulate(tx_bits)
    bps        = mod.bits_per_symbol
    ebn0_lin   = 10.0 ** (ebn0_dB / 10.0)
    snr_lin    = ebn0_lin * code_rate * bps
    rx_symbols, channel_coeff = channel.apply_with_coeff(tx_symbols, snr_lin)
    rx_bits = mod.demodulate(rx_symbols, channel_coeff)

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
    channel_names = channel.get_channel_names()

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
    if log_config_once:
        _log_config(config, "text", text_len=len(text))

    original_bits = text_to_bits(text, config["text_settings"]["text_encoding"])
    max_bits = config["text_settings"]["max_text_length"] * 8
    if len(original_bits) > max_bits:
        original_bits = original_bits[:max_bits]

    result = _run_pipeline(config, original_bits, ebn0_dB, log_prefix="TEXT ")

    # ── Text-специфичные поля ────────────────────────────────────────────────
    encoding = config["text_settings"]["text_encoding"]

    # Восстановление текста: TextRecovery если включён, иначе стандартный путь
    tr_cfg = config.get("text_recovery", {})
    if tr_cfg.get("enabled", False):
        rec = recover_text(
            result["decoded_bits"],
            original_len=len(text),
            encoding=encoding,
            window_bytes=tr_cfg.get("window_bytes", 3),
        )
        decoded_text = rec.text
        result["recovery_stats"] = {
            "chars_ok":      rec.chars_ok,
            "chars_fixed":   rec.chars_fixed,
            "chars_lost":    rec.chars_lost,
            "total_chars":   rec.total_chars,
            "recovery_rate": rec.recovery_rate,
            "repair_ms":     rec.repair_time_ms,
        }
    else:
        decoded_text = bits_to_text(result["decoded_bits"], encoding)
        result["recovery_stats"] = None

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
    os.makedirs(output_dir, exist_ok=True)
    mod_type  = config["modulation"]["type"]
    order     = config["modulation"]["order"]
    mode_abbr = "Text" if mode == "text" else "Random"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(
        output_dir, f"results_{mod_type}{order}_{mode_abbr}_{timestamp}.txt"
    )

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
    """
    if not results:
        return None, None

    os.makedirs(output_dir, exist_ok=True)
    mod_type  = config["modulation"]["type"]
    order     = config["modulation"]["order"]
    mode_abbr = "Text" if mode == "text" else "Random"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(
        output_dir, f"Plot_{mod_type}{order}_{mode_abbr}_{timestamp}.png"
    )

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

    # Определяем число subplot-ов
    n_plots = 1 + int(has_per) + int(mode == "text")
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

    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(plot_filename, dpi=150, facecolor=BG)
    logger.info(f"График сохранён: {plot_filename}")
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

    os.makedirs(output_dir, exist_ok=True)
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"Compare_{timestamp}.png")

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
