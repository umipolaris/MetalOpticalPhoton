#!/usr/bin/env python3
"""U6 torus + lens production scenario — CPU vs GPU benchmark figure.

  (a) GPU 10B focal-plane image (XY thin focal-plane) with the bright ring traced
      (fitted ellipse) and the line-out row marked; line-out (GPU vs CPU) below.
  (b) Azimuthal profile around the ring (fluence summed in angular wedges about the
      ring centroid), GPU vs CPU, with a Δ sub-panel — the high-signal comparison.
  (c) System design diagram (../common/fig6_design.png)

Data: results/torus_diffuse_10B_{CPU_fresh,beamgpu_GPU}.csv (816×624×1 thin focal-plane bins)
  → regenerate with run_gpu.sh + run_cpu.sh.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
RUNS = ROOT  # benchmark/05_torus
OUT = ROOT / "results"

# Sensor binning (from torus_diffuse_10B_beamgpu.txt)
NX, NY, NZ = 816, 624, 1   # 2026-05-28: thin focal-plane sensor (ZBins=1, HLZ 0.1mm @ imgShiftToZ 10.4mm)
HLX, HLY, HLZ = 3.672, 2.808, 0.1  # mm half-widths (ImagingSensor)
DX, DY = 2 * HLX / NX, 2 * HLY / NY
X_CENTERS = (np.arange(NX) + 0.5) * DX - HLX
Y_CENTERS = (np.arange(NY) + 0.5) * DY - HLY

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.unicode_minus": False,
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10,
    "axes.linewidth": 0.8, "savefig.dpi": 300,
})

GPU_C, CPU_C = "#2c7fb8", "#d95f0e"


def load_3d(path):
    """Load fluence CSV into (NX, NY, NZ) array."""
    arr = np.zeros((NX, NY, NZ), dtype=np.float32)
    with open(path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            p = line.strip().split(",")
            if len(p) < 4:
                continue
            try:
                ix, iy, iz, v = int(p[0]), int(p[1]), int(p[2]), float(p[3])
            except ValueError:
                continue
            if 0 <= ix < NX and 0 <= iy < NY and 0 <= iz < NZ:
                arr[ix, iy, iz] = v
    return arr


# ZBins=1: the thin sensor itself is the focal plane (single Z layer).
FOCAL_Z_BIN = 0

def to_xy(arr3d):
    return arr3d[:, :, FOCAL_Z_BIN]


print("loading CSVs (~14MB each, thin focal-plane)...")
cpu = load_3d(RUNS / "results" / "torus_diffuse_10B_CPU_fresh.csv")
gpu = load_3d(RUNS / "results" / "torus_diffuse_10B_beamgpu_GPU.csv")
# σ=1.5 px Gaussian smoothing applied to BOTH engines: the per-pixel Poisson noise is
# statistical (not a GPU/CPU difference), so the line-out, azimuthal profile and image
# are all formed from the same smoothed fields. Blur conserves the total (voxel sum).
SMOOTH = 1.5
cpu_xy = gaussian_filter(to_xy(cpu), sigma=SMOOTH)
gpu_xy = gaussian_filter(to_xy(gpu), sigma=SMOOTH)
print(f"  GPU/CPU sum (smoothed) = {gpu_xy.sum()/cpu_xy.sum():.5f}")

# Line-out: row through the CPU peak
peak_iy = int(np.unravel_index(cpu_xy.argmax(), cpu_xy.shape)[1])
cpu_x = cpu_xy[:, peak_iy]
gpu_x = gpu_xy[:, peak_iy]

# ---- ring geometry: fluence-weighted centroid (azimuthal ref) + ellipse on the bright peak ----
X, Y = np.meshgrid(X_CENTERS, Y_CENTERS, indexing="ij")   # (NX, NY) data coords
w = cpu_xy.copy()
x0 = (X * w).sum() / w.sum()
y0 = (Y * w).sum() / w.sum()
# ellipse overlay traces the bright ring at its peak radius (H/V cuts through the centroid)
ix0 = int(np.argmin(np.abs(X_CENTERS - x0))); iy0 = int(np.argmin(np.abs(Y_CENTERS - y0)))
row, col = cpu_xy[:, iy0], cpu_xy[ix0, :]
xrp = X_CENTERS[ix0 + int(np.argmax(row[ix0:]))]; xrn = X_CENTERS[int(np.argmax(row[:ix0]))]
yrp = Y_CENTERS[iy0 + int(np.argmax(col[iy0:]))]; yrn = Y_CENTERS[int(np.argmax(col[:iy0]))]
ex_c, ea = 0.5 * (xrp + xrn), 0.5 * (xrp - xrn)
ey_c, eb = 0.5 * (yrp + yrn), 0.5 * (yrp - yrn)
print(f"  ring ellipse centre=({ex_c:+.2f},{ey_c:+.2f}) semi-axes a={ea:.2f} b={eb:.2f} mm")

# ---- azimuthal profile: fluence summed in angular wedges about the centroid ----
NTH = 120
# θ in the DISPLAYED (vertically-flipped) frame: θ=0° → +x (right in (a)), θ=90° → +y (up in (a))
TH = np.degrees(np.arctan2(-(Y - y0), X - x0))
thb = np.linspace(-180, 180, NTH + 1)
thc = 0.5 * (thb[:-1] + thb[1:])
ti = np.digitize(TH.ravel(), thb)
g_az = np.array([gpu_xy.ravel()[ti == k].sum() for k in range(1, NTH + 1)])
c_az = np.array([cpu_xy.ravel()[ti == k].sum() for k in range(1, NTH + 1)])
az_dev = np.where(c_az > 0, (g_az - c_az) / c_az * 100, np.nan)
az_rms = np.sqrt(np.nanmean(az_dev ** 2))
print(f"  azimuthal profile: Δ RMS={az_rms:.2f}%  integral Δ={100*(g_az.sum()/c_az.sum()-1):+.3f}%")

# ============================================================
# Figure: (a) image + line-out | (b) azimuthal profile + Δ | (c) design
# ============================================================
fig = plt.figure(figsize=(16, 4.7))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.18], height_ratios=[2.0, 1.0],
                      hspace=0.06, wspace=0.40, left=0.05, right=0.985, top=0.92, bottom=0.14)

ext = [-HLX, HLX, -HLY, HLY]
prof_y_disp = -Y_CENTERS[peak_iy]   # line-out row, displayed y after the vertical flip

# (a) GPU image — vertically flipped to remove the lens inversion; y cropped to [-1.5, 2.5].
ax_a = fig.add_subplot(gs[0, 0])
im_a = ax_a.imshow(np.flipud((gpu_xy / gpu_xy.max()).T), origin="lower", extent=ext, aspect="auto",
                   cmap="inferno", vmin=0, vmax=1)
# ring overlay: dashed ellipse on the bright ring (y flips sign on the displayed image)
ax_a.add_patch(Ellipse((ex_c, -ey_c), 2 * ea, 2 * eb, angle=0,
                       fill=False, ec="#39ff14", lw=1.4, ls="--", alpha=0.95))
ax_a.set_ylim(-1.5, 2.5)
ax_a.set_xlim(-HLX, HLX)
ax_a.set_ylabel("y (mm)")
ax_a.axhline(prof_y_disp, color="yellow", ls="--", lw=1.4)
ax_a.text(-0.02, 1.04, "(a)", transform=ax_a.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="left")
plt.setp(ax_a.get_xticklabels(), visible=False)
cax = ax_a.inset_axes([1.022, 0.0, 0.03, 1.0])
plt.colorbar(im_a, cax=cax, label="Fluence (normalized)")

# (a, below) line-out GPU vs CPU at the marked row
ax_p = fig.add_subplot(gs[1, 0], sharex=ax_a)
ax_p.semilogy(X_CENTERS, gpu_x, "-", color=GPU_C, lw=1.4, label="GPU")
ax_p.semilogy(X_CENTERS, cpu_x, "--", color=CPU_C, lw=1.2, label="CPU")
ax_p.set_xlabel("x (mm)"); ax_p.set_ylabel("Fluence (mm$^{-2}$)")
ax_p.legend(fontsize=10, loc="lower center", ncol=2)
ax_p.grid(True, which="both", alpha=0.3)

# (b) azimuthal profile around the ring + Δ sub-panel
gsb = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 1], height_ratios=[2.6, 1.0], hspace=0.05)
ax_az = fig.add_subplot(gsb[0])
az_norm = c_az.max()
ax_az.plot(thc, c_az / az_norm, "-", color=CPU_C, lw=1.6, label="CPU")
ax_az.plot(thc, g_az / az_norm, "--", color=GPU_C, lw=1.3, label="GPU")
ax_az.set_ylabel("Fluence (normalized)")
ax_az.legend(fontsize=10, loc="upper center", ncol=2)
ax_az.grid(alpha=0.3)
ax_az.set_ylim(0.4, 1.1);
ax_az.text(-0.02, 1.04, "(b)", transform=ax_az.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="left")
plt.setp(ax_az.get_xticklabels(), visible=False)
ax_azd = fig.add_subplot(gsb[1], sharex=ax_az)
ax_azd.plot(thc, az_dev, "-", color="purple", lw=1.1)
ax_azd.axhline(0, color="k", lw=0.6); ax_azd.axhspan(-1, 1, color="green", alpha=0.12)
ax_azd.set_ylim(-3, 3); ax_azd.set_xlim(-180, 180)
ax_azd.set_xticks([-180, -90, 0, 90, 180])
ax_azd.set_xlabel("Azimuth angle (deg)"); ax_azd.set_ylabel(r"$\Delta$ (%)", labelpad=2)
ax_azd.grid(alpha=0.3)

# (c) design diagram
ax_d = fig.add_subplot(gs[:, 2])
design_path = RUNS.parent / "common" / "fig6_design.png"
if design_path.exists():
    import matplotlib.image as mpimg
    ax_d.imshow(mpimg.imread(design_path), aspect="auto")
else:
    ax_d.text(0.5, 0.5, f"design image not found:\n{design_path}",
              ha="center", va="center", transform=ax_d.transAxes, fontsize=9)
ax_d.text(-0.02, 1.04, "(c)", transform=ax_d.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="left")
ax_d.set_xticks([]); ax_d.set_yticks([])
for spine in ax_d.spines.values():
    spine.set_visible(False)

# halve only the (b)-(c) gap and give the freed width to (c)'s horizontal extent
fig.canvas.draw()
b_x1 = ax_az.get_position().x1
pd = ax_d.get_position()
new_x0 = b_x1 + (pd.x0 - b_x1) / 2.0
ax_d.set_position([new_x0, pd.y0, pd.x1 - new_x0, pd.height])

out_pdf = OUT / "fig_05_torus.pdf"
out_png = OUT / "fig_05_torus.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"wrote {out_png}")
