"""
waveform_phy.py — дискретно-временная (waveform) PHY-ветка симуляции.

Вынесено из simulation.py для упрощения архитектуры и тестирования.
Интерфейс: run(mod, tx_symbols, snr_lin) -> dict с rx_bits и опциональными метриками.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modulation import PSKModulator, QAMModulator
from waveform import (
    rrc_taps,
    pulse_shaping_tx,
    matched_filter_rx,
    downsample_to_symbols,
    gardner_timing_recovery,
    tdl_taps_from_symbol_delays,
    apply_tdl,
    estimate_symbol_spaced_channel,
    estimate_channel_ls,
    design_linear_equalizer,
    equalize_symbols,
    apply_cfo,
    apply_phase_noise_wiener,
    agc_block,
    clip_magnitude,
    adc_quantize_iq,
    apply_dc_offset,
    apply_iq_imbalance,
    time_warp_samples,
    interp_lagrange4,
    auto_init_offset_by_eye_opening,
    carrier_recovery_dd_pll,
    estimate_cfo_mth_power,
    derotate_symbols,
    estimate_cfo_from_repeated_preamble,
    estimate_cpe_from_known_preamble,
    carrier_recovery_costas_mpsk,
    estimate_cfo_cpe_from_known_preamble,
    estimate_cfo_cpe_ml_from_known_preamble,
)


Modulator = PSKModulator | QAMModulator


@dataclass
class WaveformPhy:
    wf_cfg: dict

    def run(self, mod: Modulator, tx_symbols: np.ndarray, snr_lin: float) -> dict[str, Any]:
        wf_cfg = self.wf_cfg
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

        pll_off = (pll_alpha <= 0.0 and pll_beta <= 0.0)
        if pll_off and preamble_enabled and pre_half_len < 256:
            pre_half_len = 256

        h_rrc = rrc_taps(beta=beta, span_symbols=span, sps=sps)

        # ── Training sequence ────────────────────────────────────────────────
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

        # Frame assembly with data-mask (so metrics/BER are computed on payload only)
        tx_symbols_wf = np.asarray(tx_symbols, dtype=complex).ravel()
        is_data = np.ones(tx_symbols_wf.size, dtype=bool)

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
            tx_symbols_wf = np.concatenate([preamble, tx_symbols_wf])
            is_data = np.concatenate([np.zeros(pre_len_total, dtype=bool), is_data])

        # ── Pilots insertion (symbol-rate) ───────────────────────────────────
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
            data = tx_symbols_wf
            data_mask = is_data
            chunks = []
            masks = []
            for start in range(0, len(data), pilot_period):
                chunks.append(tx_pilot)
                masks.append(np.zeros(pilot_len, dtype=bool))
                seg = data[start:start + pilot_period]
                seg_m = data_mask[start:start + pilot_period]
                chunks.append(seg)
                masks.append(seg_m)
            tx_symbols_wf = np.concatenate(chunks)
            is_data = np.concatenate(masks) if masks else np.ones(tx_symbols_wf.size, dtype=bool)

        # training в самое начало
        if tx_training is not None:
            tx_symbols_wf = np.concatenate([tx_training, tx_symbols_wf])
            is_data = np.concatenate([np.zeros(len(tx_training), dtype=bool), is_data])

        tx_symbols_wf_full = tx_symbols_wf.copy()

        # ── TX shaping + impairments ─────────────────────────────────────────
        tx_samples = pulse_shaping_tx(tx_symbols_wf, h_rrc=h_rrc, sps=sps, calibrate=True)
        if cfo_enabled and abs(cfo_norm) > 0.0:
            tx_samples = apply_cfo(tx_samples, cfo_norm=cfo_norm)

        # TDL multipath
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

        # AWGN on samples (calibrated so that symbol sample Es≈1)
        sigma = np.sqrt(1.0 / (2.0 * max(float(snr_lin), 1e-30)))
        noise = sigma * (np.random.randn(len(tx_samples)) + 1j * np.random.randn(len(tx_samples)))
        rx_samples = tx_samples + noise

        # Phase noise
        pn_cfg = wf_cfg.get("phase_noise", {}) if isinstance(wf_cfg.get("phase_noise", {}), dict) else {}
        pn_enabled = bool(pn_cfg.get("enabled", False))
        if pn_enabled:
            if "step_std_deg" in pn_cfg:
                step_std_rad = float(pn_cfg.get("step_std_deg", 0.0)) * np.pi / 180.0
            else:
                step_std_rad = float(pn_cfg.get("step_std_rad", 0.0))
            rx_samples = apply_phase_noise_wiener(
                rx_samples,
                step_std_rad=step_std_rad,
                seed=(int(pn_cfg.get("seed")) if pn_cfg.get("seed", None) is not None else None),
            )

        # Timing impairments
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

        # ADC / AGC / clipping / quant
        adc_cfg = wf_cfg.get("adc", {}) if isinstance(wf_cfg.get("adc", {}), dict) else {}
        adc_enabled = bool(adc_cfg.get("enabled", False))
        adc_stats: dict = {"enabled": adc_enabled}
        if adc_enabled:
            if bool(adc_cfg.get("dc_enabled", False)):
                rx_samples, st = apply_dc_offset(
                    rx_samples,
                    dc_i=float(adc_cfg.get("dc_i", 0.0)),
                    dc_q=float(adc_cfg.get("dc_q", 0.0)),
                )
                adc_stats["dc_offset"] = st
            if bool(adc_cfg.get("iq_imbalance_enabled", False)):
                rx_samples, st = apply_iq_imbalance(
                    rx_samples,
                    amp_imbalance_db=float(adc_cfg.get("iq_amp_imbalance_db", 0.0)),
                    phase_imbalance_deg=float(adc_cfg.get("iq_phase_imbalance_deg", 0.0)),
                )
                adc_stats["iq_imbalance"] = st
            if bool(adc_cfg.get("agc_enabled", True)):
                rx_samples, st = agc_block(
                    rx_samples,
                    target_rms=float(adc_cfg.get("agc_target_rms", 1.0)),
                    max_gain=float(adc_cfg.get("agc_max_gain", 100.0)),
                    min_gain=float(adc_cfg.get("agc_min_gain", 0.01)),
                )
                adc_stats["agc"] = st
            if bool(adc_cfg.get("clipping_enabled", True)):
                rx_samples, st = clip_magnitude(
                    rx_samples,
                    clip_level=float(adc_cfg.get("clip_level", 1.2)),
                )
                adc_stats["clipping"] = st
            if bool(adc_cfg.get("quantization_enabled", True)):
                rx_samples, st = adc_quantize_iq(
                    rx_samples,
                    n_bits=int(adc_cfg.get("n_bits", 10)),
                    full_scale=float(adc_cfg.get("full_scale", 1.0)),
                )
                adc_stats["quantization"] = st
        else:
            adc_stats = {"enabled": False}

        # ── RX matched filter + timing recovery ──────────────────────────────
        rx_mf = matched_filter_rx(rx_samples, h_rrc=h_rrc)
        timing_cfg = wf_cfg.get("timing", {}) if isinstance(wf_cfg.get("timing", {}), dict) else {}
        timing_enabled = bool(timing_cfg.get("enabled", False))
        if timing_enabled:
            init_off = float(timing_cfg.get("init_offset_samples", 0.0))
            auto_init = bool(timing_cfg.get("auto_init", False))
            auto_diag = None
            if auto_init:
                n_tr = int(timing_cfg.get("auto_traces", 200))
                n_grid = int(timing_cfg.get("auto_grid", 33))
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

        # ── CFO recovery ─────────────────────────────────────────────────────
        cfo_stats: dict
        cfo_est_cfg = wf_cfg.get("cfo", {}) if isinstance(wf_cfg.get("cfo", {}), dict) else {}
        cfo_est_source = str(cfo_est_cfg.get("estimation_source", "preamble")).lower().strip()
        if cfo_enabled and cfo_recovery_enabled and abs(cfo_norm) > 0.0:
            omega = 0.0
            phi0 = 0.0
            tr_used = int(tr_len) if (tr_enabled and tx_training is not None) else 0
            carrier_track = None

            if cfo_est_source in ("training", "training_ls") and tx_training is not None and rx_symbols_wf.size >= tr_used and tr_used >= 8:
                omega0, phi00 = estimate_cfo_cpe_from_known_preamble(rx_symbols_wf[:tr_used], tx_training)
                bw = max(float(ml_bw), 0.02 + 0.5 * abs(float(omega0)))
                omega, phi0 = estimate_cfo_cpe_ml_from_known_preamble(
                    rx_symbols_wf[:tr_used],
                    tx_training,
                    omega_coarse=float(omega0),
                    search_bw=bw,
                    n_grid=ml_grid,
                )
            elif cfo_est_source in ("pilots", "pilot") and pilots_enabled and tx_pilot is not None:
                start_idx = tr_used + (pre_len_total if pre_len_total else 0)
                rx_p = rx_symbols_wf[start_idx:start_idx + pilot_len]
                if rx_p.size >= min(8, pilot_len):
                    omega0, phi00 = estimate_cfo_cpe_from_known_preamble(rx_p, tx_pilot[:rx_p.size])
                    bw = max(float(ml_bw), 0.02 + 0.5 * abs(float(omega0)))
                    omega, phi0 = estimate_cfo_cpe_ml_from_known_preamble(
                        rx_p,
                        tx_pilot[:rx_p.size],
                        omega_coarse=float(omega0),
                        search_bw=bw,
                        n_grid=ml_grid,
                    )
            elif tx_preamble is not None and rx_symbols_wf.size >= (tr_used + pre_len_total):
                omega0 = 0.0
                if pre_half_len > 0:
                    omega0 = estimate_cfo_from_repeated_preamble(
                        rx_symbols_wf[tr_used:tr_used + pre_len_total],
                        half_len=pre_half_len,
                    )
                omega, phi0 = estimate_cfo_cpe_ml_from_known_preamble(
                    rx_symbols_wf[tr_used:tr_used + pre_len_total],
                    tx_preamble,
                    omega_coarse=omega0,
                    search_bw=ml_bw,
                    n_grid=ml_grid,
                )
            elif isinstance(mod, PSKModulator):
                omega, phi0 = estimate_cfo_mth_power(rx_symbols_wf, m=mod.M)

            rx_symbols_wf = derotate_symbols(rx_symbols_wf, omega_rad_per_sym=omega, phi0_rad=phi0)
            cfo_stats = {"omega_est_rad_per_sym": float(omega), "phi0_est_rad": float(phi0), "estimation_source": cfo_est_source}

            omega_true = 2.0 * np.pi * float(cfo_norm) * float(sps)
            cfo_stats["omega_true_rad_per_sym"] = float(omega_true)
            cfo_stats["omega_err_rad_per_sym"] = float(float(omega) - omega_true)
            if carrier_track is not None:
                cfo_stats["carrier_track"] = carrier_track

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
                        rx_symbols_wf, m=mod.M, alpha=pll_alpha, beta=pll_beta
                    )
                else:
                    rx_symbols_wf, _ = carrier_recovery_dd_pll(
                        rx_symbols_wf,
                        constellation_points=mod.constellation_points,
                        alpha=pll_alpha,
                        beta=pll_beta,
                    )
        else:
            cfo_stats = {"omega_est_rad_per_sym": 0.0, "phi0_est_rad": 0.0, "estimation_source": cfo_est_source}

        # ── Equalizer (optional) ─────────────────────────────────────────────
        eq_cfg = wf_cfg.get("equalizer", {}) if isinstance(wf_cfg.get("equalizer", {}), dict) else {}
        eq_enabled = bool(eq_cfg.get("enabled", False))
        eq_kind = str(eq_cfg.get("kind", "mmse")).lower().strip()

        evm_before = None
        evm_after = None
        const_before = None
        const_after = None
        eye_before = None
        eye_after = None
        eye_metrics = None
        evm_over_time = None

        rx0 = rx_symbols_wf.copy()
        rxa = rx_symbols_wf.copy()
        if eq_enabled:
            eq_len = int(eq_cfg.get("eq_len", 9))
            eq_delay = eq_cfg.get("delay", None)
            eq_delay = int(eq_delay) if eq_delay is not None else None
            est_method = str(eq_cfg.get("estimation", "ideal")).lower().strip()
            chan_taps = int(eq_cfg.get("channel_taps", 32))
            noise_var = float(1.0 / max(float(snr_lin), 1e-30))
            ls_reg = float(eq_cfg.get("ls_reg", 1e-3))

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

            if est_method in ("training_ls", "training") and tr_used > 0:
                rx_tr = rxa[:tr_used]
                h_sym = estimate_channel_ls(tx_training, rx_tr, channel_len=chan_taps, reg=ls_reg)
                w_eq, d_used = _design_from_h(h_sym)
            else:
                h_sym = estimate_symbol_spaced_channel(
                    h_rrc=h_rrc, sps=sps, h_tdl=h_tdl if tdl_enabled else None, n_symbols=chan_taps
                )
                w_eq, d_used = _design_from_h(h_sym)

            rxa = equalize_symbols(rxa, w_eq, delay=d_used)

        # Plot/metrics sampling (optional)
        plots_cfg = wf_cfg.get("plots", {}) if isinstance(wf_cfg.get("plots", {}), dict) else {}
        if bool(plots_cfg.get("constellation", True)):
            n_pts = int(plots_cfg.get("constellation_points", 2000))
            n_pts = max(100, min(20000, n_pts))
            n = min(int(n_pts), int(rx0.size))
            if n > 0:
                const_before = np.column_stack([rx0[:n].real, rx0[:n].imag]).tolist()
                const_after = np.column_stack([rxa[:n].real, rxa[:n].imag]).tolist()

        # EVM (payload only)
        n_sym = min(int(tx_symbols_wf_full.size), int(rx0.size), int(is_data.size))
        txr = tx_symbols_wf_full[:n_sym][is_data[:n_sym]]
        rx0m = rx0[:n_sym][is_data[:n_sym]]
        rxam = rxa[:n_sym][is_data[:n_sym]]
        if txr.size > 0:
            p = float(np.mean(np.abs(txr) ** 2))
            p = p if p > 1e-30 else 1.0
            evm_before = float(np.sqrt(np.mean(np.abs(rx0m - txr) ** 2) / p))
            evm_after = float(np.sqrt(np.mean(np.abs(rxam - txr) ** 2) / p))

        # Demodulate payload only (drop training/preamble/pilots)
        rxa_data = rxa[:n_sym][is_data[:n_sym]]
        rx_bits = mod.demodulate(rxa_data, channel_coeff=None)
        channel_names = ["AWGN", "RRC(sps)"]
        if timing_stats.get("timing_enabled", False):
            channel_names.append("Timing(Gardner)")
        if tdl_enabled:
            channel_names.append("TDL")
        if eq_enabled:
            channel_names.append(f"EQ({eq_kind.upper()})")
        channel_names += (["CFO"] if (cfo_enabled and abs(cfo_norm) > 0.0) else [])
        if cfo_enabled and cfo_recovery_enabled and abs(cfo_norm) > 0.0:
            channel_names.append("CFO_recovery")
        if pre_len_total:
            channel_names.append("CFO_preamble")
        if tr_enabled and tx_training is not None:
            channel_names.append("TRAINING")
        if pilots_enabled and tx_pilot is not None:
            channel_names.append("PILOTS")

        return {
            "rx_bits": rx_bits,
            "channel_names": channel_names,
            "timing_stats": timing_stats,
            "cfo_stats": cfo_stats,
            "adc_stats": adc_stats,
            "evm_before": evm_before,
            "evm_after": evm_after,
            "const_before": const_before,
            "const_after": const_after,
            "eye_before": eye_before,
            "eye_after": eye_after,
            "eye_metrics": eye_metrics,
            "evm_over_time": evm_over_time,
        }

