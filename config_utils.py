"""
config_utils.py — профили (presets) и нормализация конфигурации.

Цели:
  - единая точка, где задаются дефолты и «сбалансированный реализм»
  - обратная совместимость: старые config.json без новых секций продолжают работать
  - минимальная инвазивность: не переопределяем явно заданные пользователем значения
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


ProfileName = str  # "fast" | "balanced" | "realistic"


def _deep_setdefault(dst: dict, src: Mapping[str, Any]) -> None:
    """
    Рекурсивно делает setdefault по дереву словарей.
    Если ключ уже есть в dst — не трогаем его (включая вложенные dict, кроме рекурсии).
    """
    for k, v in src.items():
        if isinstance(v, Mapping):
            if k not in dst or not isinstance(dst.get(k), dict):
                dst[k] = {}
            _deep_setdefault(dst[k], v)
        else:
            dst.setdefault(k, v)


def apply_profile(cfg: Mapping[str, Any], profile: ProfileName | None = None) -> dict:
    """
    Возвращает новый dict с применённым профилем (через setdefault).
    Профиль берётся из аргумента или из cfg['profile'].
    """
    out = deepcopy(dict(cfg))
    prof = (profile or out.get("profile") or "fast")
    prof = str(prof).lower().strip()
    out["profile"] = prof

    base_defaults: dict[str, Any] = {
        # Секции, которые часто отсутствуют в старых конфигурациях
        "channel": {
            "awgn": {"enabled": True},
            "rayleigh": {"enabled": False, "n_rays": 32, "normalized_doppler": 0.05},
            "phase_noise": {"enabled": False, "phase_noise_std_deg": 1.0},
            "impulse_noise": {"enabled": False, "mode": "bernoulli", "impulse_probability": 0.001, "impulse_snr_dB": 20.0},
        },
        "per_settings": {"enabled": False, "packet_size": 1024},
        "interleaving": {"enabled": False, "depth": 8},
        "encryption": {"enabled": False, "type": "aes", "aes_mode": "CBC", "key_hex": None},
        "random_settings": {"max_adaptive_bits": 1_000_000, "num_simulations": 1, "num_bits": 50_000},
        "text_settings": {"text_file": "input.txt", "text_encoding": "utf-8", "max_text_length": 50_000, "num_repetitions": 1},
        "early_stop_ber": 1e-7,
        # waveform оставляем пустым по умолчанию — включение/дефолты задаёт профиль
        "waveform": {},
    }
    _deep_setdefault(out, base_defaults)

    # Профили
    if prof == "fast":
        _deep_setdefault(out, {"waveform": {"enabled": False}})
        return out

    if prof == "balanced":
        balanced_defaults: dict[str, Any] = {
            "waveform": {
                "enabled": True,
                "sps": 8,
                "rrc_beta": 0.35,
                "rrc_span_symbols": 10,
                "cfo": {
                    "enabled": True,
                    # Умеренный CFO: хватает, чтобы было что компенсировать, но не «ломает» систему
                    "cfo_norm": 0.0,
                    "recovery_enabled": True,
                    "pll_alpha": 0.02,
                    "pll_beta": 0.0004,
                    "preamble_enabled": True,
                    "preamble_half_len_symbols": 64,
                    "loop_type": "dd_pll",
                    "estimation_source": "preamble",
                },
                "timing": {
                    "enabled": True,
                    "method": "gardner",
                    "alpha": 0.01,
                    "beta": 0.0001,
                    "init_offset_samples": 0.0,
                    "auto_init": True,
                },
                "plots": {
                    "constellation": True,
                    "constellation_points": 2000,
                    "eye_diagram": False,
                    "eye_traces": 200,
                    "evm_chunk_symbols": 256,
                },
                # В balanced не включаем тяжёлые блоки по умолчанию
                "tdl": {"enabled": False},
                "equalizer": {"enabled": False},
                "training": {"enabled": False},
                "pilots": {"enabled": False},
                "adc": {"enabled": False},
                "phase_noise": {"enabled": False},
                "timing_impairments": {"enabled": False},
            },
            # Каналы: AWGN по умолчанию, остальное выключено
            "channel": {
                "rayleigh": {"enabled": False},
                "phase_noise": {"enabled": False},
                "impulse_noise": {"enabled": False},
            },
        }
        _deep_setdefault(out, balanced_defaults)
        return out

    if prof == "realistic":
        realistic_defaults: dict[str, Any] = {
            "waveform": {
                "enabled": True,
                "sps": 8,
                "rrc_beta": 0.35,
                "rrc_span_symbols": 10,
                "cfo": {"enabled": True, "cfo_norm": 5e-4, "recovery_enabled": True, "preamble_enabled": True},
                "timing": {"enabled": True, "method": "gardner", "auto_init": True},
                "tdl": {"enabled": True},
                "equalizer": {"enabled": True, "kind": "mmse", "estimation": "pilot_ls"},
                "pilots": {"enabled": True},
                "adc": {"enabled": True},
                "phase_noise": {"enabled": True},
            },
            "channel": {
                "rayleigh": {"enabled": True, "n_rays": 128, "normalized_doppler": 0.05},
            },
            "interleaving": {"enabled": True, "depth": 16},
            "per_settings": {"enabled": True, "packet_size": 1024},
        }
        _deep_setdefault(out, realistic_defaults)
        return out

    # Неизвестный профиль — ведём себя как fast, но сохраняем строку профиля для UI
    return out


def normalize_config(cfg: Mapping[str, Any]) -> dict:
    """
    Нормализует конфиг: гарантирует наличие ключей и приводит некоторые значения к безопасным диапазонам.
    Не пытается быть «строгой» схемой — только safe defaults и clamp.
    """
    out = deepcopy(dict(cfg))

    # Секции (если конфиг пришёл извне без apply_profile)
    out = apply_profile(out, out.get("profile"))

    # Clamp некоторых чисел, чтобы симуляция не падала на мусорных значениях
    wf = out.get("waveform", {})
    if isinstance(wf, dict):
        if wf.get("enabled", False):
            sps = int(wf.get("sps", 8) or 8)
            wf["sps"] = 8 if sps < 2 else min(64, sps)

            beta = float(wf.get("rrc_beta", 0.35) or 0.35)
            wf["rrc_beta"] = 0.35 if beta < 0.0 or beta > 1.0 else beta

            span = int(wf.get("rrc_span_symbols", 10) or 10)
            wf["rrc_span_symbols"] = 10 if span < 2 else min(50, span)

            cfo = wf.get("cfo", {})
            if isinstance(cfo, dict) and cfo.get("enabled", False):
                cfo_norm = float(cfo.get("cfo_norm", 0.0) or 0.0)
                # циклы/отсчёт: ограничим разумным диапазоном, чтобы не улететь в aliasing
                cfo["cfo_norm"] = max(-0.01, min(0.01, cfo_norm))

    out["waveform"] = wf
    return out

