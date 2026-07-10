#!/usr/bin/env python3
"""06 Mie (weak fog mfp=30mm) — paper figure.
  (a) beam + Mie-scattered fluence  x-z cross-section (GPU, top wide)
  (b) Z profile (phi-integrated)  CPU vs GPU   (barrel)
  (c) phi profile (z-integrated)  CPU vs GPU   (barrel)
Data: results/spread_m30_barrel_{CPU,GPU}.csv (barrel φ-z), spread_m30_xz_GPU.csv (x-z).
Regenerate: run spread_m30_{gpu,cpu,xz_gpu}.txt in configs/ then this script.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.unicode_minus": False,
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "savefig.dpi": 300,
})
plt.rcParams["mathtext.default"] = "regular"

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
N_PHI, N_Z = 180, 200
HL = 10.0; R_SHELL = 40.5; DZ = (2*HL*10.0)/N_Z


def load_barrel(p):
    g = np.zeros((N_PHI, N_Z))
    for r in np.loadtxt(p, comments="#", delimiter=","):
        if int(r[0]) == 0:
            g[int(r[1]), int(r[2])] = r[3]
    return g


def main():
    cpu = load_barrel(RES/"spread_m30_barrel_CPU.csv")
    gpu = load_barrel(RES/"spread_m30_barrel_GPU.csv")
    ratio = gpu.sum()/cpu.sum()
    z = (np.arange(N_Z)+0.5)*DZ - HL*10.0
    phi = (np.arange(N_PHI)+0.5)*(360.0/N_PHI)

    fig = plt.figure(figsize=(13.8, 4.7))
    g0 = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.55], wspace=0.20,
                          left=0.05, right=0.95, top=0.90, bottom=0.15)

    # (a) x-z beam + scatter
    ax = fig.add_subplot(g0[0, 0])
    NXx, NZx = 120, 300
    gxz = np.zeros((NXx, NZx))
    for r in np.loadtxt(RES/"spread_m30_xz_GPU.csv", comments="#", delimiter=","):
        gxz[int(r[0]), int(r[2])] = r[3]
    xx = -45 + (np.arange(NXx)+0.5)*90/NXx
    halo = gxz[np.abs(xx) > 4.0, :]
    vmx = np.percentile(halo[halo > 0], 99.5); vmn = vmx*1e-3
    im = ax.imshow(gxz, origin="lower", extent=[-110, 110, -45, 45], aspect="auto",
                   cmap="inferno", norm=LogNorm(vmin=vmn, vmax=vmx))
    # fog (Mie) region — cyan dashed box (z=±100, x=±40)
    ax.add_patch(Rectangle((-100, -40), 200, 80, fill=False, ec="cyan", ls="--", lw=1.3, alpha=0.85))
    # barrel scorer = (b)(c) profile measurement location (x=±40, R=40.5mm)
    ax.plot([-100, 100], [40, 40], color="#39ff14", lw=2.4)
    ax.plot([-100, 100], [-40, -40], color="#39ff14", lw=2.4)
    ax.annotate("beam (ø4mm)", xy=(-100, 0), xytext=(-98, 33), color="white", fontsize=11,
                ha="left", va="bottom", arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8))
    ax.text(35, 30, "Mie scattering region  (mfp=30 mm)", color="cyan", fontsize=11, ha="center", va="center")
    ax.text(86, 44, "barrel scorer",
            color="#39ff14", fontsize=11, ha="right", va="top")
    # vertical marker at the z = -50 mm cross-section shown in (b)
    ax.axvline(-50, color="white", ls="--", lw=1.2, alpha=0.95)
    ax.text(-47, 44, "z=−50 mm", color="white", fontsize=11, ha="left", va="top")
    ax.text(-0.105, 1.02, "(a)", transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="left")
    ax.set_xlabel("z (mm)"); ax.set_ylabel("x (mm)"); ax.set_xlim(-110, 110); ax.set_ylim(-45, 45)
    fig.colorbar(im, ax=ax, label="Fluence (mm$^{-2}$)", fraction=0.046, pad=0.02)

    # (b) z = -50 mm transverse (x-y) cross-section: GPU - CPU relative difference
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    NXY = 90
    def load_xy(p):
        a = np.zeros((NXY, NXY))
        for r in np.loadtxt(p, comments="#", delimiter=","):
            a[int(r[0]), int(r[1])] = r[3]      # ix, iy, iz=0, value
        return a
    gxy = load_xy(RES/"spread_m30_xy50_GPU.csv")
    cxy = load_xy(RES/"spread_m30_xy50_CPU.csv")
    rel = np.where(cxy > 0, 100.0*(gxy - cxy)/np.where(cxy > 0, cxy, 1.0), np.nan)
    ax = fig.add_subplot(g0[0, 1])
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("0.92")
    im = ax.imshow(rel.T, origin="lower", extent=[-45, 45, -45, 45], aspect="equal",
                   cmap=cmap, vmin=-3, vmax=3)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.text(-0.16, 1.02, "(b)", transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="left")
    # colourbar on the RIGHT, fixed ±3 % (matches B2)
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.10)
    fig.colorbar(im, cax=cax, label=r"$\Delta$  (%)", ticks=[-3, -1.5, 0, 1.5, 3])

    out = ROOT/"results"/"fig_06_mie.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  G/C={ratio:.4f}")


if __name__ == "__main__":
    main()
