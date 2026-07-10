#!/usr/bin/env python3
"""07 Cherenkov benchmark — 2x2 panel (geometry sketch + e-/proton/C12 z-fluence).

So both CPU and GPU are clearly visible:
  - CPU: thick semi-transparent ribbon (lw=4, alpha=0.35)
  - GPU: crisp thin line on top (lw=1.4, alpha=1.0)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, FancyArrowPatch
from matplotlib.colors import LogNorm
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.unicode_minus": False,
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.linewidth": 0.8, "savefig.dpi": 300,
})

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# air-gap x-z scorer: x ±110mm (XBins 60), z ±165mm (ZBins 130)
NX, NZ = 60, 130
XC = (np.arange(NX) + 0.5) / NX * 220 - 110   # x bin center (mm)
ZC = (np.arange(NZ) + 0.5) / NZ * 330 - 165   # z bin center (mm)
ONAXIS = np.abs(XC) < 5                         # on-axis region (cone center)

GROUPS = {
    "electron": [
        ("e1",  "e- 1 MeV",  "#08306b"),
        ("e5",  "e- 5 MeV",  "#2171b5"),
        ("e10", "e- 10 MeV", "#41ab5d"),
        ("e50", "e- 50 MeV", "#cc4c02"),
    ],
    "proton": [
        ("p100", "p 100 MeV", "#08306b"),
        ("p150", "p 150 MeV", "#2171b5"),
        ("p200", "p 200 MeV", "#41ab5d"),
        ("p250", "p 250 MeV", "#cc4c02"),
    ],
    "C12 ion": [
        ("c12_200", "C12 200 MeV/u", "#08306b"),
        ("c12_300", "C12 300 MeV/u", "#41ab5d"),
        ("c12_430", "C12 430 MeV/u", "#cc4c02"),
    ],
}


def load_xz(tag: str, engine: str):
    """{tag}_xz_{engine}.csv (60×130) -> 2D grid. None if absent."""
    p = RESULTS / f"{tag}_xz_{engine}.csv"
    if not p.exists():
        return None
    g = np.zeros((NX, NZ))
    for r in np.loadtxt(p, comments="#", delimiter=","):
        g[int(r[0]), int(r[2])] = r[3]
    return g


def loadz_onaxis(tag: str, engine: str):
    """on-axis (|x|<5mm) local z-profile from the x-z map (130 bins, not integrated). None if absent."""
    g = load_xz(tag, engine)
    return None if g is None else g[ONAXIS].mean(axis=0)


def load_N(tag: str) -> int | None:
    cfg = ROOT / "configs" / f"cherenkov_{tag}_xz_cpu.txt"
    import re
    try:
        for ln in cfg.read_text(errors="ignore").splitlines():
            m = re.search(r"NumberOfHistoriesInRun\s*=\s*(\d+)", ln)
            if m:
                return int(m.group(1))
    except FileNotFoundError:
        pass
    return None


SEEDS = [11, 22, 33, 44, 55]
MULTISEED = RESULTS / "multiseed"

# Legend Δ values from benchmark/RESULTS_4232_11332.md (canonical 4.2.3/11.3.2 results).
# Protons: fresh 1e6 × 10 seeds (2026-07-03); electrons / 12C: marathon runs, 5 seeds.
# The local multiseed/ files hold only the 5 marathon seeds, so the 10-seed proton
# statistics cannot be recomputed here; values are quoted from the results table.
# (mean Δ %, SD %)
CANONICAL_DELTA = {
    "e1":      (+0.14, 1.61),
    "e5":      (-0.06, 0.29),
    "e10":     (+0.01, 0.38),
    "e50":     (-0.09, 0.33),
    "p100":    (+0.27, 1.63),
    "p150":    (+0.27, 1.29),
    "p200":    (+0.01, 0.56),
    "p250":    (-0.12, 0.36),
    "c12_200": (+0.15, 1.39),
    "c12_300": (-0.19, 0.07),
    "c12_430": (-0.15, 0.11),
}


def loadgrid_seed(tag: str, engine: str, seed: int):
    """full 2D grid (NX×NZ) of a single-seed multiseed file. None if absent."""
    p = MULTISEED / f"{tag}_{engine}_s{seed}.csv"
    if not p.exists():
        return None
    g = np.zeros((NX, NZ))
    for r in np.loadtxt(p, comments="#", delimiter=","):
        g[int(r[0]), int(r[2])] = r[3]
    return g


def loadgrid_pairs(tag: str):
    """list of (gpu_grid, cpu_grid) pairs for seeds present in both GPU and CPU."""
    pairs = []
    for s in SEEDS:
        g = loadgrid_seed(tag, "GPU", s)
        c = loadgrid_seed(tag, "CPU", s)
        if g is not None and c is not None:
            pairs.append((g, c))
    return pairs


def fmt_N(N) -> str:
    """particle count as 10^n mathtext (e.g. 100000 -> 10^{5})."""
    if not N:
        return "?"
    exp = int(round(np.log10(N)))
    if 10 ** exp == N:
        return f"10^{{{exp}}}"
    return f"{N / 10 ** exp:.0f}\\times 10^{{{exp}}}"


def draw_2d_fluence(ax, fig):
    """(a) proton 200MeV injected 5cm outside the water -> Cherenkov photon fluence x-z cross-section (GPU).
    Includes setup boundary (water box), beam arrow, and colorbar."""
    NX, NZ = 60, 130
    g = np.zeros((NX, NZ))
    for r in np.loadtxt(RESULTS / "c12_300_xz_GPU.csv", comments="#", delimiter=","):
        g[int(r[0]), int(r[2])] = r[3]
    vmax = np.percentile(g[g > 0], 99.0)   # linear scale — emphasize the bright core (cone)
    # scorer extent: x ±110mm (XBins 60), z ±165mm (ZBins 130)
    im = ax.imshow(g, origin="lower", extent=[-165, 165, -110, 110], aspect="auto",
                   cmap="inferno", vmin=0, vmax=vmax, interpolation="bilinear")
    # water box boundary (z=-100..100, x=-100..100)
    ax.add_patch(Rectangle((-100, -100), 200, 200, fill=False, ec="cyan", ls="--", lw=1.6, alpha=0.85))
    ax.text(0, 60, "Water (n = 1.333)", color="cyan", fontsize=11, ha="center", va="bottom")
    # World material (Air) label — dark region outside the water box
    ax.text(132, 0, "Air", color="white", fontsize=11, ha="center", va="center")
    # beam (entry 5 cm before water = z=-150 -> +z)
    ax.annotate("", xy=(-100, 0), xytext=(-150, 0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.2))
    ax.text(-162, 10, "$^{12}$C 300 MeV/u", color="white", fontsize=11, va="bottom")
    ax.axvline(-100, color="cyan", ls=":", lw=0.7, alpha=0.7)  # water entry face
    ax.set_xlabel("Depth $z$ (mm)", fontsize=10); ax.set_ylabel("$x$ (mm)", fontsize=10)
    ax.set_ylim(-110, 112)
    ax.text(-0.02, 1.02, "(a)", transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="bottom", ha="left")
    fig.colorbar(im, ax=ax, label="Fluence (mm$^{-2}$)", fraction=0.046, pad=0.02)


def plot_group(ax, group_name, cases, z, panel_label, group_tag):
    for tag, label, color in cases:
        pairs = loadgrid_pairs(tag)
        if not pairs:
            continue
        # for plotting: 5-seed mean of the on-axis (|x|<5mm) depth profile
        gpu_mean = np.array([g[ONAXIS].mean(axis=0) for g, c in pairs]).mean(axis=0)
        cpu_mean = np.array([c[ONAXIS].mean(axis=0) for g, c in pairs]).mean(axis=0)
        N = load_N(tag)
        # legend: signed % deviation of the voxel-sum, mean ± SD (same metric as supplementary Table S4).
        # Canonical values are quoted from benchmark/RESULTS_4232_11332.md (protons: fresh 1e6 × 10 seeds);
        # if a case is missing there, fall back to computing from the local multiseed files.
        if tag in CANONICAL_DELTA:
            dev, dsd = CANONICAL_DELTA[tag]
        else:
            ratios = np.array([g.sum() / c.sum() for g, c in pairs if c.sum() > 0])
            # 2026-06-12: unified to sample SD (ddof=1) — same definition as the "mean ± SD (n=5)" notation in the text
            dev, dsd = (ratios.mean() - 1) * 100, ratios.std(ddof=1) * 100

        # CPU = thick semi-transparent ribbon (5-seed mean, on-axis)
        ax.plot(z, cpu_mean, "-", color=color, lw=4.5, alpha=0.30,
                solid_capstyle="round")
        # GPU = dotted line on top (5-seed mean, on-axis). The legend match is the signed % deviation 100·(GPU/CPU-1) ± SD.
        ax.plot(z, gpu_mean, ":", color=color, lw=1.8, alpha=1.0,
                label=fr"{label}  $N\!=\!{fmt_N(N)}$  $\Delta\!=\!${dev:+.2f}$\,\pm\,${dsd:.2f}%")

    # single legend with the CPU/GPU labels
    from matplotlib.lines import Line2D
    handles, _ = ax.get_legend_handles_labels()
    extra = [
        Line2D([0], [0], color="gray", lw=4.5, alpha=0.30, label="CPU (5-seed mean)"),
        Line2D([0], [0], color="gray", lw=1.8, ls=":", alpha=1.0, label="GPU (5-seed mean)"),
    ]
    # legend near the center of the plot. (b)(c) shifted slightly up to avoid overlapping the green line.
    yc = 0.61 if panel_label == "(c)" else (0.64 if panel_label == "(b)" else 0.5)  # (c) legend 3% lower
    xc = 0.50 if panel_label == "(d)" else 0.48  # (d) shifted right so the rising C12 curves stay visible
    ax.legend(handles=extra + handles, loc="center", bbox_to_anchor=(xc, yc),
              fontsize=10, framealpha=0.92, borderpad=0.5, labelspacing=0.35)

    ax.set_xlim(z[0], z[-1])
    ax.set_ylim(bottom=0)  # linear scale
    ax.set_xlabel("Depth $z$ (mm)", fontsize=10)
    ax.set_ylabel("Track-length fluence (mm$^{-2}$)", fontsize=10)
    ax.text(-0.105, 1.02, panel_label, transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="bottom", ha="left")
    # particle-group identifier tag (replaces the removed title) — small non-boxed text
    ax.text(0.985, 0.97, group_tag, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="right")
    ax.grid(True, alpha=0.3)
    ax.axvspan(-165, -100, color="gray", alpha=0.08)  # region outside the water (air)
    ax.axvline(-100, color="cyan", lw=0.8, ls=":", alpha=0.7)  # water entry face


def main():
    z = ZC

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    draw_2d_fluence(axes[0, 0], fig)
    plot_group(axes[0, 1], "electron", GROUPS["electron"], z, "(b)", "electrons")
    plot_group(axes[1, 0], "proton",   GROUPS["proton"],   z, "(c)", "protons")
    plot_group(axes[1, 1], "C12 ion",  GROUPS["C12 ion"],  z, "(d)", "$^{12}$C ions")

    out = RESULTS / "fig_07_cherenkov_geometry_and_zfluence_cpu_gpu.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
