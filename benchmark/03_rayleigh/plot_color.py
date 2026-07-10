#!/usr/bin/env python3
"""B2 Rayleigh colour-separation demo — "blue sky / red sunset" by scattering (not refraction).

A white continuous beam (1.65–3.10 eV, flat) travels +z through a non-refracting (n=1) box
whose Rayleigh attenuation length ∝ λ⁴.  Short-λ (blue) light scatters out strongly into a
side halo; long-λ (red) light penetrates, so the forward beam reddens with depth — the same
principle that makes the daytime sky blue and the setting sun red, shown here without any
refraction (n=1, the beam never bends).  Energy-resolved x–z fluence is rendered to true
colour (spectral → sRGB) for GPU and CPU.

Data: results/rayleigh_color_{GPU,CPU}.csv  (rayleigh_color_run.sh → configs/rayleigh_color_{gpu,cpu}.txt).
  Each row = one (iX,iZ) voxel (iX = row%NX, iZ = row//NX); columns: underflow, NE energy
  bins, overflow, no-track.  → results/fig_03_rayleigh_color.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.unicode_minus": False,
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "savefig.dpi": 300,
})

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# scorer binning (configs/rayleigh_color_gpu.txt): X 120 ×1mm, Z 180 ×1mm, E 30 bins 1.65–3.10 eV
NX, NZ, NE = 120, 180, 30
HLX, HLZ = 60.0, 90.0
EMIN, EMAX = 1.65, 3.10
SC_TRANSZ = 70.0                       # scorer centre in world z (medium: world z 0–150 mm)
ECEN = EMIN + (np.arange(NE) + 0.5) * (EMAX - EMIN) / NE   # bin centres (eV)
LAM = 1239.84 / ECEN                                       # bin centres (nm)
X_CEN = (np.arange(NX) + 0.5) - HLX                        # mm, beam at x≈0
WORLD_Z = ((np.arange(NZ) + 0.5) - HLZ) + SC_TRANSZ        # world z (mm), medium 0–150


# ---- spectral → sRGB (Bruton 380–780 nm; fades to dark red/violet at the ends, no
#      spurious hue return — the CIE-fit went near-zero past ~700 nm and a per-pixel
#      saturation step amplified the residual green channel into false yellow-green) ----
def wl_to_rgb_bruton(nm):
    nm = float(nm)
    if nm < 380 or nm > 780:
        return np.zeros(3)
    if nm < 440:   r, g, b = -(nm - 440) / 60.0, 0.0, 1.0
    elif nm < 490: r, g, b = 0.0, (nm - 440) / 50.0, 1.0
    elif nm < 510: r, g, b = 0.0, 1.0, -(nm - 510) / 20.0
    elif nm < 580: r, g, b = (nm - 510) / 70.0, 1.0, 0.0
    elif nm < 645: r, g, b = 1.0, -(nm - 645) / 65.0, 0.0
    else:          r, g, b = 1.0, 0.0, 0.0
    if nm < 420:   f = 0.3 + 0.7 * (nm - 380) / 40.0          # fade in at the violet end
    elif nm <= 700: f = 1.0
    else:          f = 0.3 + 0.7 * (780 - nm) / 80.0          # fade out at the red end
    gm = 0.8
    return np.array([(max(r, 0) * f) ** gm, (max(g, 0) * f) ** gm, (max(b, 0) * f) ** gm])


# Wavelength → colour lookup, shared by the map and its colourbar so they are the SAME mapping.
# Each voxel is coloured by its mean wavelength (not a broadband composite), so a displayed
# colour corresponds to exactly one wavelength on the bar.
CB_LAM = np.linspace(LAM.min(), LAM.max(), 256)               # nm
CB_RGB = np.clip(np.array([wl_to_rgb_bruton(l) for l in CB_LAM]), 0, 1)   # (256,3)


def load_cube(path: Path) -> np.ndarray:
    cube = np.zeros((NX, NZ, NE))
    n = 0
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split(",")
            iX, iZ = n % NX, n // NX
            if iX < NX and iZ < NZ:
                cube[iX, iZ, :] = [float(v) for v in p[1:1 + NE]]
            n += 1
    return cube


def composite_rgb(cube: np.ndarray) -> np.ndarray:
    """Spectral fluence cube (NX,NZ,NE) → sRGB image (NZ,NX,3) composited over WHITE.

    hue    = colour of the per-voxel MEAN wavelength (Σ_λ λ·I_λ / Σ_λ I_λ), looked up in CB_RGB
             — the same wavelength→colour map as the colourbar, so the two correspond exactly
    alpha  = log-scaled total fluence per voxel (faint halo visible next to ~10⁴× core)
    out    = white·(1−alpha) + hue·alpha  →  empty space is white, beam shows its colour."""
    I = cube.clip(min=0.0)
    tot = I.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_lam = np.where(tot > 0, np.tensordot(I, LAM, axes=([2], [0])) / tot, CB_LAM[0])
    hue = np.stack([np.interp(mean_lam, CB_LAM, CB_RGB[:, c]) for c in range(3)], axis=-1)
    pos = tot[tot > 0]
    vmax = pos.max(); vmin = vmax * 1e-3
    alpha = np.zeros_like(tot)
    m = tot > 0
    alpha[m] = np.clip((np.log10(tot[m]) - np.log10(vmin)) /
                       (np.log10(vmax) - np.log10(vmin)), 0, 1) ** 0.7    # lift faint halo
    img = 1.0 - alpha[..., None] * (1.0 - np.clip(hue, 0, 1))   # composite over white
    return np.clip(img.transpose(1, 0, 2), 0, 1)            # (NZ,NX,3)


def mean_lambda_vs_depth(cube, fwd_mask):
    """Forward-beam mean wavelength (nm) vs world z (reddening curve)."""
    z, lam = [], []
    for iz in range(NZ):
        sp = cube[fwd_mask, iz, :].sum(axis=0)
        if sp.sum() > 0:
            z.append(WORLD_Z[iz]); lam.append((LAM * sp).sum() / sp.sum())
    return np.array(z), np.array(lam)


def main():
    # rayleigh_color_GPU.csv is produced by the BEAM-kernel path (rayleigh_color_run.sh GPU
    # branch): kernel-side generation, ~560× faster than the TOPAS-Beam genstep path and
    # physics-identical (kernel/CPU sub-1%, kernel/TOPAS-beam 1.0002).
    G = load_cube(RESULTS / "rayleigh_color_GPU.csv")
    C = load_cube(RESULTS / "rayleigh_color_CPU.csv")

    fwd = np.abs(X_CEN) < 5
    side = np.abs(X_CEN) > 30
    for nm, c in (("GPU", G), ("CPU", C)):
        sf, ss = c[fwd].sum(axis=(0, 1)), c[side].sum(axis=(0, 1))
        ef = (ECEN * sf).sum() / sf.sum(); es = (ECEN * ss).sum() / ss.sum()
        print(f"{nm}: forward <E>={ef:.3f} eV ({1239.84/ef:.0f} nm)  "
              f"side <E>={es:.3f} eV ({1239.84/es:.0f} nm)  Δ={es-ef:+.3f} eV")

    import matplotlib.patheffects as pe
    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(1, 2, wspace=0.40)
    extent = [-HLX, HLX, WORLD_Z[0], WORLD_Z[-1]]

    # colourbar uses the SAME wavelength→colour map (CB_RGB) the voxels are coloured with
    cb_strip = CB_RGB

    def show_map(ax, cube, label, title):
        ax.imshow(composite_rgb(cube), origin="lower", extent=extent, aspect="auto")
        for zb in (0, 150):
            ax.axhline(zb, color="0.4", lw=0.6, ls=":", alpha=0.7)
        # beam-direction arrow at the entrance, along the axis (+z)
        ax.annotate("", xy=(0, 22), xytext=(0, WORLD_Z[0] + 2),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.4,
                                    mutation_scale=22))
        ax.text(6, WORLD_Z[0] + 9, "beam (+z)", color="black", fontsize=9,
                fontweight="bold", va="center", ha="left",
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
        ax.text(0.99, 0.975, "Rayleigh medium (n=1)", color="0.2", fontsize=10,
                ha="right", va="top", transform=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
        ax.text(-0.10, 1.02, label, transform=ax.transAxes, fontsize=17,
                fontweight="bold", va="bottom", ha="left")
        # wavelength colourbar on the right, matched to the panel height
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.10)
        cax.imshow(cb_strip[:, None, :], origin="lower", aspect="auto",
                   extent=[0, 1, LAM.min(), LAM.max()])
        cax.set_xticks([])
        cax.yaxis.set_label_position("right"); cax.yaxis.tick_right()
        ticks = [400, 450, 500, 550, 600, 650, 700, 750]
        cax.set_yticks([t for t in ticks if LAM.min() <= t <= LAM.max()])
        cax.set_ylabel(r"wavelength $\lambda$ (nm)", fontsize=9)
        cax.tick_params(labelsize=8)

    def show_diff(ax, gpu, cpu, label, title):
        # Relative difference 100·(GPU−CPU)/CPU [%], fixed ±2 % diverging scale.
        # Every voxel has signal (scattered light fills the whole cube), so nothing is masked;
        # the wider speckle away from the axis is the larger Poisson noise of the dimmer halo.
        gt = gpu.sum(axis=2); ct = cpu.sum(axis=2)
        rel = np.where(ct > 0, 100.0 * (gt - ct) / np.where(ct > 0, ct, 1.0), np.nan)
        cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("0.92")
        im = ax.imshow(rel.T, origin="lower", extent=extent, aspect="auto",
                       cmap=cmap, vmin=-3.0, vmax=3.0)
        for zb in (0, 150):
            ax.axhline(zb, color="0.4", lw=0.6, ls=":", alpha=0.7)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
        ax.text(-0.10, 1.02, label, transform=ax.transAxes, fontsize=17,
                fontweight="bold", va="bottom", ha="left")
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.10)
        fig.colorbar(im, cax=cax, label=r"$\Delta$  (%)",
                     ticks=[-3, -1.5, 0, 1.5, 3])
        cax.tick_params(labelsize=8)

    # (a) GPU true-colour map with beam arrow, (b) GPU vs CPU relative difference (±2 %)
    show_map(fig.add_subplot(gs[0, 0]), G, "(a)", "GPU (Metal, BEAM kernel)")
    show_diff(fig.add_subplot(gs[0, 1]), G, C, "(b)", "GPU vs CPU (relative difference)")

    out = RESULTS / "fig_03_rayleigh_color.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out)
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
