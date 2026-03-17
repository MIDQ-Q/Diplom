import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

import numpy as np

import simulation


@dataclass(frozen=True)
class Row:
    ebn0_db: float
    snr_db: float
    ber_sim: float
    ber_theory: float
    ser_sim: float
    ser_theory: float
    num_bits: int
    bit_errors: int


def _load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def _normalize_cfg(base: dict) -> dict:
    cfg = deepcopy(base)

    # simulation.py expects these keys in _log_config()
    cfg.setdefault("channel", {})
    cfg.setdefault("per_settings", {"enabled": False})
    cfg.setdefault("waveform", {"enabled": False})
    cfg.setdefault("interleaving", {"enabled": False})
    cfg.setdefault("encryption", {"enabled": False})

    # Make results deterministic
    cfg.setdefault("random_settings", {})
    cfg["random_settings"].setdefault("max_adaptive_bits", cfg["random_settings"].get("num_bits", 50_000))
    return cfg


def _set_awgn_only(cfg: dict) -> None:
    cfg["channel"] = {
        "awgn": {"enabled": True},
        "rayleigh": {"enabled": False},
        "phase_noise": {"enabled": False},
        "impulse_noise": {"enabled": False},
    }


def _set_rayleigh_awgn(cfg: dict, *, n_rays: int = 32, normalized_doppler: float = 0.05) -> None:
    cfg["channel"] = {
        "awgn": {"enabled": True},
        "rayleigh": {"enabled": True, "n_rays": int(n_rays), "normalized_doppler": float(normalized_doppler)},
        "phase_noise": {"enabled": False},
        "impulse_noise": {"enabled": False},
    }


def validate_awgn_uncoded_psk(
    *,
    M: int,
    use_gray: bool,
    ebn0_points_db: Iterable[float],
    num_bits: int,
    seed: int = 12345,
) -> list[Row]:
    base = _normalize_cfg(_load_config())
    base["simulation_mode"] = "random"

    base["coding"]["enabled"] = False
    base["modulation"]["type"] = "PSK"
    base["modulation"]["order"] = int(M)
    base["modulation"]["use_gray_code"] = bool(use_gray)
    base["random_settings"]["num_bits"] = int(num_bits)
    # Allow simulation.py to scale up when BER gets small (but cap for runtime)
    base["random_settings"]["max_adaptive_bits"] = int(max(num_bits, 5_000_000))
    _set_awgn_only(base)

    np.random.seed(seed)
    rows: list[Row] = []
    prev = None
    for i, eb in enumerate(ebn0_points_db):
        r = simulation.simulate_transmission(
            base,
            ebn0_dB=float(eb),
            log_config_once=(i == 0),
            prev_ber=prev,
        )
        prev = r["ber"]
        rows.append(
            Row(
                ebn0_db=float(eb),
                snr_db=float(r["snr"]),
                ber_sim=float(r["ber"]),
                ber_theory=float(r.get("theoretical_ber", 0.0)),
                ser_sim=float(r["ser"]),
                ser_theory=float(r.get("theoretical_ser", 0.0)),
                num_bits=int(r.get("num_bits_used", num_bits)),
                bit_errors=int(r.get("bit_errors", int(round(float(r["ber"]) * float(r.get("num_bits_used", num_bits)))))),
            )
        )
    return rows


def validate_rayleigh_uncoded_psk(
    *,
    M: int,
    ebn0_points_db: Iterable[float],
    num_bits: int,
    n_rays: int = 32,
    normalized_doppler: float = 0.05,
    seed: int = 23456,
) -> list[tuple[Row, float]]:
    """
    Returns list of (Row, ber_theory_rayleigh) where ber_theory_rayleigh is taken
    from simulation's own theoretical function (MGF-based) for consistency.
    """
    base = _normalize_cfg(_load_config())
    base["simulation_mode"] = "random"

    base["coding"]["enabled"] = False
    base["modulation"]["type"] = "PSK"
    base["modulation"]["order"] = int(M)
    base["modulation"]["use_gray_code"] = True
    base["random_settings"]["num_bits"] = int(num_bits)
    base["random_settings"]["max_adaptive_bits"] = int(max(num_bits, 8_000_000))
    _set_rayleigh_awgn(base, n_rays=n_rays, normalized_doppler=normalized_doppler)

    np.random.seed(seed)
    out: list[tuple[Row, float]] = []
    prev = None
    for i, eb in enumerate(ebn0_points_db):
        r = simulation.simulate_transmission(
            base,
            ebn0_dB=float(eb),
            log_config_once=(i == 0),
            prev_ber=prev,
        )
        prev = r["ber"]
        row = Row(
            ebn0_db=float(eb),
            snr_db=float(r["snr"]),
            ber_sim=float(r["ber"]),
            ber_theory=float(r.get("theoretical_ber", 0.0)),
            ser_sim=float(r["ser"]),
            ser_theory=float(r.get("theoretical_ser", 0.0)),
            num_bits=int(r.get("num_bits_used", num_bits)),
            bit_errors=int(r.get("bit_errors", int(round(float(r["ber"]) * float(r.get("num_bits_used", num_bits)))))),
        )
        out.append((row, float(r.get("rayleigh_theoretical_ber", 0.0))))
    return out


def _poisson_approx_ci95(k: int) -> tuple[float, float]:
    """
    95% CI for Poisson mean λ given observed count k.
    Uses chi-square quantiles via scipy if available; otherwise a conservative normal approx.
    Returns CI for λ (not probability).
    """
    try:
        import scipy.stats as st
        alpha = 0.05
        lo = 0.5 * st.chi2.ppf(alpha / 2, 2 * k) if k > 0 else 0.0
        hi = 0.5 * st.chi2.ppf(1 - alpha / 2, 2 * (k + 1))
        return float(lo), float(hi)
    except Exception:
        # Normal approx: λ ≈ k ± 1.96*sqrt(k+1) (keep non-negative)
        import math
        lo = max(0.0, k - 1.96 * math.sqrt(k + 1.0))
        hi = k + 1.96 * math.sqrt(k + 1.0)
        return float(lo), float(hi)


def _print_rows(title: str, rows: list[Row]) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print("Eb/N0(dB)  SNR(dB)    BER_sim      BER_theory    rel_err(BER)   k_err  E[k|theory]   95%CI(BER_sim)        Nbits")
    rel_errs = []
    for row in rows:
        rel = None
        if row.ber_theory > 0:
            rel = abs(row.ber_sim - row.ber_theory) / row.ber_theory
            rel_errs.append(rel)
        rel_s = f"{rel:>11.3e}" if rel is not None else f"{'n/a':>11}"

        exp_k = row.ber_theory * row.num_bits
        lam_lo, lam_hi = _poisson_approx_ci95(row.bit_errors)
        ber_lo = lam_lo / row.num_bits if row.num_bits > 0 else 0.0
        ber_hi = lam_hi / row.num_bits if row.num_bits > 0 else 0.0

        print(
            f"{row.ebn0_db:>8.2f}  {row.snr_db:>7.2f}  {row.ber_sim:>11.3e}  {row.ber_theory:>11.3e}  "
            f"{rel_s}  {row.bit_errors:>5d}  {exp_k:>10.2f}   [{ber_lo:.3e}, {ber_hi:.3e}]  {row.num_bits:>6d}"
        )
    if rel_errs:
        print(f"\nMean relative BER error: {float(np.mean(rel_errs)):.3e}")
        print(f"Max  relative BER error: {float(np.max(rel_errs)):.3e}")


def main() -> None:
    ebn0 = [0, 2, 4, 6, 8, 10]

    rows_bpsk = validate_awgn_uncoded_psk(M=2, use_gray=True, ebn0_points_db=ebn0, num_bits=200_000)
    _print_rows("AWGN, uncoded BPSK (PSK-2) — simulation vs theoretical BER/SER", rows_bpsk)

    rows_qpsk = validate_awgn_uncoded_psk(M=4, use_gray=True, ebn0_points_db=ebn0, num_bits=200_000)
    _print_rows("AWGN, uncoded QPSK (PSK-4, Gray) — simulation vs theoretical BER/SER", rows_qpsk)

    rows_8psk = validate_awgn_uncoded_psk(M=8, use_gray=True, ebn0_points_db=ebn0, num_bits=300_000)
    _print_rows("AWGN, uncoded 8-PSK (PSK-8, Gray) — simulation vs theoretical BER/SER", rows_8psk)

    ray = validate_rayleigh_uncoded_psk(M=2, ebn0_points_db=ebn0, num_bits=600_000)
    print("\n" + "=" * 110)
    print("Rayleigh (Jakes) + AWGN, uncoded BPSK — simulation vs Rayleigh theoretical BER")
    print("=" * 110)
    print("Eb/N0(dB)  BER_sim      BER_rayleigh_theory  rel_err    k_err  E[k|theory]    95%CI(BER_sim)")
    rels = []
    for row, ber_ray in ray:
        rel = abs(row.ber_sim - ber_ray) / ber_ray if ber_ray > 0 else float("nan")
        rels.append(rel)
        exp_k = ber_ray * row.num_bits
        lam_lo, lam_hi = _poisson_approx_ci95(row.bit_errors)
        print(
            f"{row.ebn0_db:>8.2f}  {row.ber_sim:>11.3e}  {ber_ray:>18.3e}  {rel:>7.3e}  "
            f"{row.bit_errors:>5d}  {exp_k:>10.2f}   [{lam_lo/row.num_bits:.3e}, {lam_hi/row.num_bits:.3e}]"
        )
    print(f"\nMean relative BER error (Rayleigh): {float(np.mean(rels)):.3e}")


if __name__ == "__main__":
    main()

