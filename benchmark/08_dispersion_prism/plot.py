#!/usr/bin/env python3
"""08 dispersion — combined figure: rainbow dispersion (CPU/GPU) + profile comparison.
Top: CPU/GPU rainbow with profile-line markers (X=-225, Z=175).
Bottom: z-profile (X=-225) + x-profile (Z=175), CPU vs GPU."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.unicode_minus": False,
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "savefig.dpi": 300,
})

# ── Inline constants/functions (former plot_phsp/plot_profile; merged into a single script) ──
from pathlib import Path
from matplotlib.patches import Polygon
import types

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# rainbow/xze grid + prism shapes (former plot_phsp)
_R = 40.0
_PRISM_CONFIGS = [(0, 0, 96.0), (-75, 67, 47.8), (-174, 56, 359.6)]  # (px, pz, phi_start) for display
_PRISMS = []
for _px, _pz, _phi in _PRISM_CONFIGS:
    _v = []
    for _k in range(3):
        _a = _phi + _k * 120
        _v.append((_R * np.cos(np.deg2rad(_a)) + _px, -_R * np.sin(np.deg2rad(_a)) + _pz))
    _PRISMS.append(_v)
_BEAM_X, _BEAM_Z = 10.0, -75.0


def draw_geometry(ax):
    LABEL_OFFSETS = [(70, 30), (60, 30), (70, 30)]
    for i, prism_verts in enumerate(_PRISMS, 1):
        poly = Polygon([(zv, xv) for (xv, zv) in prism_verts], closed=True,
                       facecolor="none", edgecolor="cyan", lw=1.8, alpha=0.95, zorder=5)
        ax.add_patch(poly)
        cz = sum(zv for xv, zv in prism_verts) / 3
        cx = sum(xv for xv, zv in prism_verts) / 3
        dz, dx = LABEL_OFFSETS[i - 1]
        ax.annotate(f"P{i}", xy=(cz, cx), xytext=(cz + dz, cx + dx), ha="center", va="center",
                    fontsize=11, color="cyan", zorder=6, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="cyan", lw=0.6, alpha=0.7))
    ax.plot(_BEAM_Z, _BEAM_X, "w*", ms=14, zorder=7, markeredgecolor="black", markeredgewidth=0.8)
    ax.set_xlabel("z (mm)", fontsize=11); ax.set_ylabel("x (mm)", fontsize=11)
    ax.set_xlim(-100, 200); ax.set_ylim(-300, 100)
    ax.set_aspect("equal"); ax.set_facecolor("black")
    ax.grid(True, alpha=0.15, color="gray")


RB = types.SimpleNamespace(RESULTS=RESULTS, N_X=200, N_Z=1000, HL_X=200.0,
                           SP_HL_Z=1000.0, SP_X0=-100.0, SP_Z0=0.0,
                           E_LOW=1.65, E_HIGH=3.10, draw_geometry=draw_geometry)

# profile grid (former plot_profile) — ScorePlane 0.125mm binning
_PF_N_X, _PF_N_Z = 3200, 2800
_PF_X_LO, _PF_X_HI = -300.0, 100.0
_PF_Z_LO, _PF_Z_HI = -125.0, 225.0


def load_grid(path):
    """TOPAS Fluence CSV (ix,iy,iz,value) → (N_X, N_Z) grid."""
    print(f"loading {path}")
    arr = np.loadtxt(path, comments="#", delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    g = np.zeros((_PF_N_X, _PF_N_Z))
    ix = arr[:, 0].astype(np.int32); iz = arr[:, 2].astype(np.int32); val = arr[:, 3]
    mask = (ix >= 0) & (ix < _PF_N_X) & (iz >= 0) & (iz < _PF_N_Z)
    g[ix[mask], iz[mask]] = val[mask]
    return g


PF = types.SimpleNamespace(RESULTS=RESULTS, N_X=_PF_N_X, N_Z=_PF_N_Z,
                           X_LO=_PF_X_LO, Z_LO=_PF_Z_LO,
                           DX=(_PF_X_HI - _PF_X_LO) / _PF_N_X,
                           DZ=(_PF_Z_HI - _PF_Z_LO) / _PF_N_Z, load_grid=load_grid)

X_PROFILE = -225.0
Z_PROFILE = 175.0

# 2026-05-20: physically accurate wavelength → color conversion.
# CIE 1931 color matching functions (Wyman et al. 2013 analytic multi-lobe Gaussian)
# → XYZ → linear sRGB (Rec.709 D65) → gamma. Reflects the actual spectral sensitivity of the human eye.
# The previous approximation mapped cyan(490nm) too brightly → relative dark gap in the green band.
# CIE accurately reflects eye luminance (max at 555nm, falling off at both ends) → natural rainbow.
def _gauss(x, mu, s1, s2):
    s = s1 if x < mu else s2
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def wl_to_rgb_cie(nm):
    X = (1.056 * _gauss(nm, 599.8, 37.9, 31.0)
         + 0.362 * _gauss(nm, 442.0, 16.0, 26.7)
         - 0.065 * _gauss(nm, 501.1, 20.4, 26.2))
    Y = (0.821 * _gauss(nm, 568.8, 46.9, 40.5)
         + 0.286 * _gauss(nm, 530.9, 16.3, 31.1))
    Z = (1.217 * _gauss(nm, 437.0, 11.8, 36.0)
         + 0.681 * _gauss(nm, 459.0, 26.0, 13.8))
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    r, g, b = max(r, 0.0), max(g, 0.0), max(b, 0.0)
    def gam(c):
        return 1.055 * c ** (1.0 / 2.4) - 0.055 if c > 0.0031308 else 12.92 * c
    return (gam(r), gam(g), gam(b))


def _blend_finalize(accum_rgb, accum_I, max_I):
    """intensity-weighted blend: intensity-weighted average of the multiple wavelength colors overlapping in one pixel.
    color  = Σ(rgb_λ·I_λ) / ΣI_λ   (mixes color only)
    bright = max_λ(I_λ)             (per-pixel max intensity → preserves brightness, same as max projection)
    In overlapping regions colors mix smoothly (bands removed), and brightness equals that of single-wavelength regions."""
    mask = accum_I > 1e-9
    color = np.zeros_like(accum_rgb)
    for ci in range(3):
        c = color[..., ci]
        c[mask] = accum_rgb[..., ci][mask] / accum_I[mask]
    return np.clip(color * max_I[..., None], 0, 1)


def xze_composite_blend(csv_name, n_bins=50):
    """X-Z-Energy CSV (same format for CPU/GPU) → intensity-weighted blend composite.
    voxel fluence + wavelength binning. Both CPU and GPU use the same voxel approach (not line raster)."""
    p = RB.RESULTS / csv_name
    d = np.loadtxt(p, comments="#", delimiter=",")
    print(f"  XZE CSV {csv_name}: {d.shape}")
    accum_rgb = np.zeros((RB.N_X, RB.N_Z, 3))
    accum_I = np.zeros((RB.N_X, RB.N_Z))
    max_I = np.zeros((RB.N_X, RB.N_Z))
    for i in range(n_bins):
        E_mid = RB.E_LOW + (i + 0.5) * (RB.E_HIGH - RB.E_LOW) / n_bins
        rgb = wl_to_rgb_cie(1239.84 / E_mid)
        g = d[:, 1 + i].reshape(RB.N_Z, RB.N_X).T  # Z-outer X-inner
        gm = g.max()
        if gm <= 0:
            continue
        I = np.power(g / gm, 0.5)
        I = np.where(I > 0.02, I, 0)
        for ci in range(3):
            accum_rgb[..., ci] += rgb[ci] * I
        accum_I += I
        max_I = np.maximum(max_I, I)
    return _blend_finalize(accum_rgb, accum_I, max_I)


def draw_geometry_with_lines(ax):
    """Rainbow panel: prisms + beam + profile-line markers."""
    RB.draw_geometry(ax)
    # Profile lines (axes: x-axis = z, y-axis = x)
    ax.axhline(X_PROFILE, color="white", ls="--", lw=1.3, alpha=0.85, zorder=8)
    ax.axvline(Z_PROFILE, color="yellow", ls="--", lw=1.3, alpha=0.85, zorder=8)
    ax.annotate(f"X={X_PROFILE:.0f} mm", xy=(170, X_PROFILE-15), color="white",
                fontsize=10, va="bottom", ha="right", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="white", alpha=0.6))
    ax.annotate(f"Z={Z_PROFILE:.0f} mm", xy=(Z_PROFILE+5, 90), color="yellow",
                fontsize=10, va="top", ha="left", zorder=9, rotation=90,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="yellow", alpha=0.6))


def main():
    print("[rainbow] GPU composite (XZE voxel fluence + EBins)")
    gpu_img = xze_composite_blend("disp_xze_GPU.csv", n_bins=50)
    print("[profile] load grids")
    cpu_grid = PF.load_grid(PF.RESULTS / "disp_prof_CPU.csv")
    gpu_grid = PF.load_grid(PF.RESULTS / "disp_prof_GPU_bk.csv")  # GPU 5M = CPU 5M (same count, no scaling needed)

    ix_prof = int(round((X_PROFILE - PF.X_LO) / PF.DX))
    iz_prof = int(round((Z_PROFILE - PF.Z_LO) / PF.DZ))
    z_axis = PF.Z_LO + (np.arange(PF.N_Z) + 0.5) * PF.DZ
    x_axis = PF.X_LO + (np.arange(PF.N_X) + 0.5) * PF.DX
    cpu_zprof, gpu_zprof = cpu_grid[ix_prof, :], gpu_grid[ix_prof, :]
    cpu_xprof, gpu_xprof = cpu_grid[:, iz_prof], gpu_grid[:, iz_prof]
    gc_z = gpu_zprof.sum() / cpu_zprof.sum() if cpu_zprof.sum() > 0 else 0
    gc_x = gpu_xprof.sum() / cpu_xprof.sum() if cpu_xprof.sum() > 0 else 0

    fig = plt.figure(figsize=(13.5, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.62, 1.0],
                          hspace=0.34, wspace=0.20,
                          left=0.05, right=0.97, top=0.94, bottom=0.12)
    ax_gpu = fig.add_subplot(gs[:, 0])   # GPU rainbow map (left, full height)
    ax_z = fig.add_subplot(gs[0, 1])     # z-profile (top right)
    ax_x = fig.add_subplot(gs[1, 1])     # x-profile (bottom right)

    # --- GPU rainbow panel (a); draw_geometry sets equal aspect + limits ---
    extent = (RB.SP_Z0 - RB.SP_HL_Z, RB.SP_Z0 + RB.SP_HL_Z,
              -RB.HL_X + RB.SP_X0, RB.HL_X + RB.SP_X0)
    ax_gpu.imshow(gpu_img, origin="lower", extent=extent, aspect="auto",
                  interpolation="bilinear", zorder=1)
    draw_geometry_with_lines(ax_gpu)
    ax_gpu.text(-0.02, 1.01, "(a)", transform=ax_gpu.transAxes, fontsize=17,
                fontweight="bold", va="bottom", ha="left")

    # --- Profile panels ---
    ax_z.plot(z_axis, cpu_zprof, color="tab:blue", lw=1.0, label="CPU")
    ax_z.plot(z_axis, gpu_zprof, color="tab:red", lw=1.0, alpha=0.7,
              label="GPU")
    ax_z.set_xlabel("z (mm)"); ax_z.set_ylabel("Fluence (mm$^{-2}$)")
    ax_z.text(-0.13, 1.02, "(b)", transform=ax_z.transAxes, fontsize=17,
              fontweight="bold", va="bottom", ha="left")
    ax_z.legend(); ax_z.grid(True, alpha=0.3); ax_z.set_xlim(-100, 225)

    ax_x.plot(x_axis, cpu_xprof, color="tab:blue", lw=1.0, label="CPU")
    ax_x.plot(x_axis, gpu_xprof, color="tab:red", lw=1.0, alpha=0.7,
              label="GPU")
    ax_x.set_xlabel("x (mm)"); ax_x.set_ylabel("Fluence (mm$^{-2}$)")
    ax_x.text(-0.13, 1.02, "(c)", transform=ax_x.transAxes, fontsize=17,
              fontweight="bold", va="bottom", ha="left")
    ax_x.legend(); ax_x.grid(True, alpha=0.3); ax_x.set_xlim(-300, 100)

    print(f"  [removed-from-title] z-profile X={X_PROFILE:.0f} mm  G/C = {gc_z:.4f}")
    print(f"  [removed-from-title] x-profile Z={Z_PROFILE:.0f} mm  G/C = {gc_x:.4f}")
    out = RB.RESULTS / "fig_08_combined_cpu_vs_gpu.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    out_pdf = RB.RESULTS / "fig_08_combined_cpu_vs_gpu.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
