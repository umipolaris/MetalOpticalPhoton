#!/usr/bin/env python3
"""U1/U2 angle-scan figure — 2-panel layout with SF1 data (new geometry, vertical interface).

U1v5 (Air -> Glass, n1=1.0, n2=1.5):
   Vertical interface at x=0; source on radius-50 circle (Air side).
   T = SF1 going_in @ Glass/XMinusSurface (refracted)
   R = SF1 going_in @ AirRefl/XPlusSurface (reflected, caught by thin Air slab)

U2v4 (BC408 -> Air, n1=1.58, n2=1.0):
   Vertical interface at x=0; source INSIDE BC408 on radius-50 circle.
   T = SF1 going_in @ AirDet/XMinusSurface (refracted)
   R inferred = 1 - T (no mirror wrap; TIR photons stay trapped in BC408 → effectively R)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT  = Path(__file__).resolve().parent.parent
BUILT = ROOT / "results" / "built"
OUT   = ROOT / "results"

ANGLES   = [5, 15, 25, 30, 35, 38, 40, 45, 50, 60, 70, 85]
N_SOURCE = 100_000
BIN_AREA_V2 = 0.2 * 0.2 * 100.0  # 4 mm² (HLX=100, 100 bins)
BIN_AREA_V3 = 0.4 * 0.4 * 100.0  # 16 mm² (HLX=200, 100 bins)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.unicode_minus": False,
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "savefig.dpi": 300,
})


def load_sum(path: Path):
    if not path.exists():
        return None
    s = 0.0
    with path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 4:
                try:
                    s += float(parts[3])
                except ValueError:
                    pass
    return s


def load_sf1_count(path: Path):
    """SF1 single-bin scorer: last numeric line is the count."""
    if not path.exists():
        return None
    last = None
    with path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            try:
                last = float(line.strip().split(",")[-1])
            except (ValueError, IndexError):
                pass
    return last


def fluence_to_R(fluence_sum, theta_i_deg, bin_area):
    if fluence_sum is None:
        return None
    return fluence_sum * bin_area * np.cos(np.deg2rad(theta_i_deg)) / N_SOURCE


def fresnel(theta_i_deg, n1, n2):
    th_i = np.deg2rad(theta_i_deg)
    sin_t = (n1 / n2) * np.sin(th_i)
    valid = sin_t <= 1.0
    th_t = np.where(valid, np.arcsin(np.clip(sin_t, -1, 1)), np.nan)
    cos_i = np.cos(th_i)
    cos_t = np.cos(th_t)
    R_s = np.where(valid,
                   ((n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t))**2,
                   1.0)
    R_p = np.where(valid,
                   ((n1*cos_t - n2*cos_i) / (n1*cos_t + n2*cos_i))**2,
                   1.0)
    T_s = 1.0 - R_s
    T_p = 1.0 - R_p
    return R_s, T_s, R_p, T_p, np.rad2deg(th_t)


def collect_u1(prefix="U1v5"):
    """U1v5 (s-pol Y=1) or U1v5_ppol (p-pol X=1): vertical interface at x=0.
    T = SF1 going_in @ Glass/XMinusSurface (refracted into Glass)
    R = SF1 going_in @ AirRefl/XPlusSurface (reflected back into thin Air slab on -x side)
    """
    rows = []
    for th in ANGLES:
        T_gpu_cnt = load_sf1_count(BUILT / f"{prefix}_gpu_th{th}_T_sf1.csv")
        T_cpu_cnt = load_sf1_count(BUILT / f"{prefix}_cpu_th{th}_T_sf1.csv")
        R_gpu_cnt = load_sf1_count(BUILT / f"{prefix}_gpu_th{th}_R_sf1.csv")
        R_cpu_cnt = load_sf1_count(BUILT / f"{prefix}_cpu_th{th}_R_sf1.csv")
        rows.append({
            "th": th,
            "T_gpu": (T_gpu_cnt / N_SOURCE) if T_gpu_cnt is not None else None,
            "T_cpu": (T_cpu_cnt / N_SOURCE) if T_cpu_cnt is not None else None,
            "R_gpu": (R_gpu_cnt / N_SOURCE) if R_gpu_cnt is not None else None,
            "R_cpu": (R_cpu_cnt / N_SOURCE) if R_cpu_cnt is not None else None,
        })
    return rows


def collect_u2():
    """U2v5: vertical interface at x=0, source INSIDE BC408 + 5-face same-n absorber wrap.
    T = SF1 going_in @ AirDet/XMinusSurface (single-bounce only — returner photons absorbed)
    R = inferred 1 - T
    """
    rows = []
    for th in ANGLES:
        T_gpu_cnt = load_sf1_count(BUILT / f"U2v5_gpu_th{th}_T_sf1.csv")
        T_cpu_cnt = load_sf1_count(BUILT / f"U2v5_cpu_th{th}_T_sf1.csv")
        T_gpu = (T_gpu_cnt / N_SOURCE) if T_gpu_cnt is not None else None
        T_cpu = (T_cpu_cnt / N_SOURCE) if T_cpu_cnt is not None else None
        rows.append({
            "th": th,
            "T_gpu": T_gpu,
            "T_cpu": T_cpu,
            "R_gpu": (1.0 - T_gpu) if T_gpu is not None else None,
            "R_cpu": (1.0 - T_cpu) if T_cpu is not None else None,
        })
    return rows


def plot():
    u1_s = collect_u1("U1v5")            # s-pol (Y=1)
    u1_p = collect_u1("U1v5_ppol")       # p-pol (X=1)
    u1_u = collect_u1("U1v5_unpol")      # unpol (no BeamPolarization, TOPAS default)
    u2 = collect_u2()

    fig, (ax_U1, ax_U2) = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.91, bottom=0.12, wspace=0.22)

    # --- U1 (Air -> Glass, n=1.5)
    th_grid = np.linspace(0, 89, 360)
    R_s, T_s, R_p, T_p, _ = fresnel(th_grid, 1.0, 1.5)

    th_arr = np.array([r["th"]    for r in u1_s])
    # s-pol arrays
    Ts_gpu = np.array([r["T_gpu"] for r in u1_s])
    Ts_cpu = np.array([r["T_cpu"] for r in u1_s])
    Rs_gpu = np.array([r["R_gpu"] for r in u1_s])
    Rs_cpu = np.array([r["R_cpu"] for r in u1_s])
    # p-pol arrays
    Tp_gpu = np.array([r["T_gpu"] for r in u1_p])
    Tp_cpu = np.array([r["T_cpu"] for r in u1_p])
    Rp_gpu = np.array([r["R_gpu"] for r in u1_p])
    Rp_cpu = np.array([r["R_cpu"] for r in u1_p])

    # unpol arrays (TOPAS default — random pol)
    Tu_gpu = np.array([r["T_gpu"] for r in u1_u])
    Tu_cpu = np.array([r["T_cpu"] for r in u1_u])
    Ru_gpu = np.array([r["R_gpu"] for r in u1_u])
    Ru_cpu = np.array([r["R_cpu"] for r in u1_u])
    T_avg = 0.5 * (T_s + T_p)
    R_avg = 0.5 * (R_s + R_p)

    # Theory curves
    ax_U1.plot(th_grid, T_s, color="black",   lw=1.4, label=r"Theory $T_s$")
    ax_U1.plot(th_grid, R_s, color="black",   lw=1.0, ls=":", label=r"Theory $R_s$")
    ax_U1.plot(th_grid, T_p, color="#7a3f9a", lw=1.4, label=r"Theory $T_p$")
    ax_U1.plot(th_grid, R_p, color="#7a3f9a", lw=1.0, ls=":", label=r"Theory $R_p$")
    ax_U1.plot(th_grid, T_avg, color="#c75d00", lw=1.2, ls="--", label=r"Theory $T_{\rm avg}$")
    ax_U1.plot(th_grid, R_avg, color="#c75d00", lw=0.9, ls=(0,(1,1)), label=r"Theory $R_{\rm avg}$")

    # s-pol MC (blue/orange)
    ax_U1.plot(th_arr, Ts_gpu, "o", color="#2c7fb8", ms=6, label=r"GPU $T_s$")
    ax_U1.plot(th_arr, Ts_cpu, "s", color="#d95f0e", ms=5, mfc="none", mew=1.2, label=r"CPU $T_s$")
    ax_U1.plot(th_arr, Rs_gpu, "^", color="#2c7fb8", ms=5, mfc="white", mew=1.0, label=r"GPU $R_s$")
    ax_U1.plot(th_arr, Rs_cpu, "v", color="#d95f0e", ms=5, mfc="white", mew=1.0, label=r"CPU $R_s$")

    # p-pol MC (purple/green)
    ax_U1.plot(th_arr, Tp_gpu, "o", color="#5a3a8a", ms=6, label=r"GPU $T_p$")
    ax_U1.plot(th_arr, Tp_cpu, "s", color="#1b7837", ms=5, mfc="none", mew=1.2, label=r"CPU $T_p$")
    ax_U1.plot(th_arr, Rp_gpu, "^", color="#5a3a8a", ms=5, mfc="white", mew=1.0, label=r"GPU $R_p$")
    ax_U1.plot(th_arr, Rp_cpu, "v", color="#1b7837", ms=5, mfc="white", mew=1.0, label=r"CPU $R_p$")

    # unpol MC (TOPAS default random pol — affected by bug)
    ax_U1.plot(th_arr, Tu_gpu, "D", color="#c75d00", ms=5, label=r"GPU $T_{\rm unpol}$")
    ax_U1.plot(th_arr, Tu_cpu, "x", color="#c75d00", ms=6, mew=1.4, label=r"CPU $T_{\rm unpol}$")
    ax_U1.plot(th_arr, Ru_gpu, "D", color="#c75d00", ms=5, mfc="white", mew=1.0, label=r"GPU $R_{\rm unpol}$")
    ax_U1.plot(th_arr, Ru_cpu, "+", color="#c75d00", ms=7, mew=1.4, label=r"CPU $R_{\rm unpol}$")

    # T+R conservation marker (s-pol)
    ax_U1.plot(th_arr, Ts_gpu + Rs_gpu, "+", color="#888", ms=7, mew=1.2, label=r"GPU $T+R$")
    ax_U1.axhline(1.0, color="#888", lw=0.5, ls="--", alpha=0.5)

    ax_U1.set_xlim(0, 89)
    ax_U1.set_ylim(0, 1.05)
    ax_U1.set_xlabel(r"Incidence angle $\theta_i$ (deg)")
    ax_U1.set_ylabel("Apparent reflectance / transmittance")
    ax_U1.legend(loc="center left", bbox_to_anchor=(0.015, 0.45),
                 fontsize=11, ncol=2,
                 framealpha=0.9, handlelength=1.3, columnspacing=0.85,
                 labelspacing=0.25, handletextpad=0.4)
    ax_U1.grid(True, alpha=0.3)
    ax_U1.text(-0.105, 1.02, "(a)", transform=ax_U1.transAxes,
               fontsize=17, fontweight="bold", va="bottom", ha="left")

    # --- U2
    th_grid2 = np.linspace(0, 89, 360)
    R_s2, T_s2, _, _, _ = fresnel(th_grid2, 1.58, 1.0)
    theta_c = np.rad2deg(np.arcsin(1.0 / 1.58))

    th_u2  = np.array([r["th"]    for r in u2])
    T_gpu2 = np.array([r["T_gpu"] for r in u2])
    T_cpu2 = np.array([r["T_cpu"] for r in u2])
    R_gpu2 = np.array([r["R_gpu"] for r in u2])
    R_cpu2 = np.array([r["R_cpu"] for r in u2])
    TpR_gpu2 = T_gpu2 + R_gpu2

    ax_U2.plot(th_grid2, T_s2, color="black", lw=1.4, label=r"Theory $T_s = 1-R_s$")
    ax_U2.plot(th_grid2, R_s2, color="gray",  lw=1.2, ls=":", label=r"Theory $R_s$")
    ax_U2.plot(th_u2, T_gpu2, "o", color="#2c7fb8", ms=7, label="GPU $T$ (SF1 surface)")
    ax_U2.plot(th_u2, T_cpu2, "s", color="#d95f0e", ms=6, mfc="none", mew=1.4,
               label="CPU $T$ (SF1 surface)")
    ax_U2.plot(th_u2, R_gpu2, "^", color="#2c7fb8", ms=6, mfc="white", mew=1.2,
               label="GPU $R$ (SF1 surface)")
    ax_U2.plot(th_u2, R_cpu2, "v", color="#d95f0e", ms=6, mfc="white", mew=1.2,
               label="CPU $R$ (SF1 surface)")
    ax_U2.plot(th_u2, TpR_gpu2, "+", color="#1b7837", ms=8, mew=1.6,
               label="GPU $T+R$ (conservation)")
    ax_U2.axhline(1.0, color="#1b7837", lw=0.6, ls="--", alpha=0.5)
    ax_U2.axvline(theta_c, color="#7a3f9a", lw=1.3, ls="--",
                  label=fr"$\theta_c = {theta_c:.2f}°$")
    ax_U2.axvspan(theta_c, 89, color="#dcd0e8", alpha=0.4,
                  label="TIR region")
    ax_U2.set_xlim(0, 89)
    ax_U2.set_ylim(0, 1.15)
    ax_U2.set_xlabel(r"Incidence angle $\theta_i$ (deg)")
    ax_U2.set_ylabel("Apparent reflectance / transmittance")
    ax_U2.legend(loc="center right", bbox_to_anchor=(0.985, 0.5),
                 fontsize=11, ncol=1,
                 framealpha=0.9, handlelength=1.4, columnspacing=1.0,
                 labelspacing=0.3, handletextpad=0.4)
    ax_U2.grid(True, alpha=0.3)
    ax_U2.text(-0.105, 1.02, "(b)", transform=ax_U2.transAxes,
               fontsize=17, fontweight="bold", va="bottom", ha="left")

    # --- B1 validation numbers (12-angle max|Δ|, GPU vs analytic Fresnel / vs CPU Geant4) ---
    Rs_t, Ts_t, Rp_t, Tp_t, _ = fresnel(th_arr, 1.0, 1.5)   # U1: Air → Glass(n=1.5)
    Rs2_t, Ts2_t, _, _, _     = fresnel(th_u2, 1.58, 1.0)   # U2: BC-408(n=1.58) → Air
    def _mx(a, b): return 100.0 * float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    s_fres   = max(_mx(Rs_gpu, Rs_t),  _mx(Ts_gpu, Ts_t))
    p_fres   = max(_mx(Rp_gpu, Rp_t),  _mx(Tp_gpu, Tp_t))
    tir_fres = max(_mx(T_gpu2, Ts2_t), _mx(R_gpu2, Rs2_t))
    s_cpu    = max(_mx(Rs_gpu, Rs_cpu), _mx(Ts_gpu, Ts_cpu))
    p_cpu    = max(_mx(Rp_gpu, Rp_cpu), _mx(Tp_gpu, Tp_cpu))
    tir_cpu  = max(_mx(T_gpu2, T_cpu2), _mx(R_gpu2, R_cpu2))
    above = th_u2 > theta_c
    leak = 100.0 * float(np.max(np.abs(T_gpu2[above]))) if np.any(above) else 0.0
    print("=== B1 validation numbers (12-angle max|Δ|) ===")
    print(f"  U1 R/T vs analytic Fresnel : s-pol {s_fres:.3f} %   p-pol {p_fres:.3f} %                 [paper ≤ 0.29 %]")
    print(f"  U2 TIR vs analytic Fresnel : {tir_fres:.3f} %   (θc={theta_c:.1f}°, GPU leakage above critical angle {leak:.3f} %)   [paper ≤ 0.08 %]")
    print(f"  GPU vs CPU Geant4          : s-pol {s_cpu:.3f} %   p-pol {p_cpu:.3f} %   TIR {tir_cpu:.3f} %    [paper ≤ 0.36 %]")

    out_pdf = OUT / "fig_02_angle_scan.pdf"
    out_png = OUT / "fig_02_angle_scan.png"
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    print(f"wrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    plot()
