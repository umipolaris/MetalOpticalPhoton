#!/usr/bin/env python3
"""U5 short geometry — publication Birks figure (2x2).

Panels:
  (a) Setup geometry (XZ slice, true scale, large area)
  (b) Optical fluence dL/dz (sum over X) — CPU vs GPU + ratio subplot
  (c) Lateral fluence profile at z=-20 mm (dedicated x-slab scorer) + ratio subplot
  (d) Birks light-yield (L/E) — theory vs MC + explanation
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import exp1  # E_1(x), exponential integral

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results"
OUT = ROOT / "results"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.unicode_minus": False,
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 11,
    "axes.linewidth": 0.8, "savefig.dpi": 300,
})


def load_3d(path):
    """Load fluence CSV into (NX, NY, NZ) array."""
    import re
    bin_re = re.compile(r"#\s*([XYZ])\s+in\s+(\d+)\s+bins?")
    nx = ny = nz = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                m = bin_re.search(line)
                if m:
                    val = int(m.group(2))
                    if m.group(1) == "X":   nx = val
                    elif m.group(1) == "Y": ny = val
                    elif m.group(1) == "Z": nz = val
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    rows.append((int(parts[0]), int(parts[1]),
                                 int(parts[2]), float(parts[3])))
                except ValueError:
                    pass
    if None in (nx, ny, nz):
        raise RuntimeError(f"could not parse bin counts from {path}")
    arr = np.zeros((nx, ny, nz))
    for ix, iy, iz, v in rows:
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            arr[ix, iy, iz] = v
    return arr


def load_1d(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    rows.append(float(parts[3]))
                except ValueError:
                    pass
    return np.array(rows)


# --- Geometry constants ---
sci_HLZ = 75.0    # mm  (cube half-length along z)
sci_HLX = 25.0    # mm  (cube half-length along x = y)
N_EVENTS = 2500

# --- 2026-05-02: single-scorer rule enforced — GPU also uses 3 separate processes, each with only 1 scorer.
#     CSV file names are the same as before (cpu_z, cpu_x, gpu_z, gpu_x). Load directly as 1D.
cpu_dLdz     = load_1d(RUNS / "U5short_CPU_fluence_z.csv")
gpu_dLdz     = load_1d(RUNS / "U5short_GPU_fluence_z.csv")
cpu_dLdz_off = load_1d(RUNS / "U5short_CPU_fluence_z_birksoff.csv")
gpu_dLdz_off = load_1d(RUNS / "U5short_GPU_fluence_z_birksoff.csv")
NZ = len(cpu_dLdz)
bin_z = 2 * sci_HLZ / NZ
z_centers = np.linspace(-sci_HLZ + bin_z/2, sci_HLZ - bin_z/2, NZ)

# (c) lateral x-profile from the dedicated thin x-slab scorer, now placed at
# z = -20 mm (plateau, upstream of the Bragg peak) for both GPU and CPU.
Z_PROF = -20.0
cpu_lat  = load_1d(RUNS / "U5short_CPU_fluence_x.csv")
gpu_lat  = load_1d(RUNS / "U5short_GPU_fluence_x.csv")
NX = len(cpu_lat)
bin_x = 2 * sci_HLX / NX
x_centers = np.linspace(-sci_HLX + bin_x/2, sci_HLX - bin_x/2, NX)

def _csv_total(path):
    s = 0.0
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"): continue
        try: s += float(line.split(",")[-1])
        except: pass
    return s
cpu_full_on  = _csv_total(RUNS / "U5short_CPU_fluence.csv")
cpu_full_off = _csv_total(RUNS / "U5short_CPU_fluence_birksoff.csv")
gpu_full_on  = _csv_total(RUNS / "U5short_GPU_fluence.csv")
gpu_full_off = _csv_total(RUNS / "U5short_GPU_fluence_birksoff.csv")

# --- Edep on Sci (1×1×200) for theoretical Birks comparison curves ---
cpu_ed_1d = load_1d(RUNS / "U5short_CPU_edep.csv")
gpu_ed_1d = load_1d(RUNS / "U5short_GPU_edep.csv")
NZ_edep = len(cpu_ed_1d)
bin_z_edep = 2 * sci_HLZ / NZ_edep

# Birks
kB = 0.126
dEdz = cpu_ed_1d / (N_EVENTS * bin_z_edep)   # MeV/mm averaged over each edep Z bin
dLdz_birks = dEdz / (1 + kB * dEdz)

# --- NIST PSTAR for proton in PVT (polyvinyl-toluene, ρ=1.032 g/cm³, BC-408 base) ---
# Stopping power table (MeV/mm); log-log interpolation
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
# Proton stopping power in PVT. Below 0.5 MeV the table is extended with NIST-PSTAR
# proton ratios (anchored at the 0.5 MeV value) so the slowing-down curve captures the
# rise toward the Bragg-peak dE/dx (~0.08 MeV) instead of clamping flat; this matters for
# the low-energy (2-5 MeV) Birks integral, whose path spends ~13% below 0.5 MeV.
T_NIST = np.array([0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
S_NIST = np.array([114.0, 125.0, 122.0, 95.0, 77.0, 65.5, 57.0, 28.9, 16.7, 8.98, 4.85, 2.54, 1.30, 0.78, 0.51])
S_log = interp1d(np.log(T_NIST), np.log(S_NIST), kind='cubic', bounds_error=False,
                 fill_value=(np.log(S_NIST[0]), np.log(S_NIST[-1])))
def dEdx_T(T):
    """Stopping power [MeV/mm] for proton kinetic energy T [MeV] in PVT."""
    T_safe = np.maximum(T, 0.05)
    return np.where(T > 0.04, np.exp(S_log(np.log(T_safe))), 0.0)

# Solve dT/dz = -dE/dx(T) for proton entering at z=-50 with T=100 MeV
def dTdz(z, T):
    return -dEdx_T(np.array([max(T[0], 0.0)]))[0] if hasattr(T, '__len__') else -dEdx_T(np.array([T]))[0]
sol = solve_ivp(lambda z, T: -dEdx_T(T), [-sci_HLZ, sci_HLZ], [100.0],
                method='RK45', t_eval=np.linspace(-sci_HLZ, sci_HLZ, 50001),
                rtol=1e-8, atol=1e-9)
z_fine = sol.t
T_fine = np.maximum(sol.y[0], 0.0)
dEdx_fine = dEdx_T(T_fine)
# Beyond range: T=0, no energy deposit
dEdx_fine = np.where(T_fine > 0.05, dEdx_fine, 0.0)
R_proton = z_fine[np.where(T_fine < 0.5)[0][0]] if (T_fine < 0.5).any() else sci_HLZ
print(f"[fig_u5_short_birks] Proton stops at z = {R_proton:.2f} mm (range = {R_proton+sci_HLZ:.2f} mm)")

# Range straggling (Bohr): σ_R ≈ R/35 for proton
sigma_R = (R_proton + sci_HLZ) / 35.0
print(f"[fig_u5_short_birks] Range straggling σ_R = {sigma_R:.2f} mm")


# ---- 2x2 layout ----
fig = plt.figure(figsize=(15, 10.5))
gs_main = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.30,
                           left=0.06, right=0.97, top=0.96, bottom=0.08)

# (a) Setup geometry — top-left, expanded y-axis (x mm) for vertical breathing room
ax_geo = fig.add_subplot(gs_main[0, 0])

# World boundary (clipped view)
ax_geo.add_patch(mpatches.Rectangle((-160, -100), 320, 200,
                                     facecolor="white", edgecolor="gray",
                                     linewidth=0.7, linestyle=(0, (4, 3))))
# Sci BC408
ax_geo.add_patch(mpatches.Rectangle((-sci_HLZ, -25), 2*sci_HLZ, 50,
                                     facecolor="#fff7c2", edgecolor="black",
                                     linewidth=1.6,
                                     label=f"Scintillator BC408"))
# ScoreBox_Z (narrow axis tube, panel b)
ax_geo.add_patch(mpatches.Rectangle((-sci_HLZ, -1), 2*sci_HLZ, 2,
                                     facecolor="#c2185b", alpha=0.35,
                                     edgecolor="#c2185b", linewidth=1.4,
                                     label="Score Box (panel b)"))
# x-profile location (panel c): vertical slab at z = -20 mm (cube slice)
ax_geo.add_patch(mpatches.Rectangle((Z_PROF-1.5, -sci_HLX), 3, 2*sci_HLX,
                                     facecolor="#1f77b4", alpha=0.20,
                                     edgecolor="#1f77b4", linewidth=1.4,
                                     linestyle=(0, (4, 2)),
                                     label="X-axis profile (panel c)"))

# Proton beam — entering from outside Sci
src = (-130, 0)
ax_geo.annotate("", xy=(-sci_HLZ, 0), xytext=src,
                arrowprops=dict(arrowstyle="-|>", color="#1f77b4",
                                lw=2.6, mutation_scale=18))
ax_geo.plot([src[0]], [src[1]], "o", color="#1f77b4", markersize=8,
            markeredgecolor="white", markeredgewidth=0.8)
ax_geo.text(src[0], 12, "100 MeV\nproton beam", fontsize=10,
            color="#1f77b4", ha="center", va="bottom", fontweight="bold")

# Bragg peak indicator at the proton stopping point — vertical line (not a band)
z_bragg = R_proton
ax_geo.plot([z_bragg, z_bragg], [-sci_HLX, sci_HLX], color="#d33", lw=2.2,
            solid_capstyle="butt", label=f"Bragg peak (z ≈ {z_bragg:+.0f} mm)")


ax_geo.set_xlim(-160, 160)
ax_geo.set_ylim(-100, 100)
ax_geo.set_xlabel("z (mm)")
ax_geo.set_ylabel("x (mm)")
ax_geo.text(-0.105, 1.02, "(a)", transform=ax_geo.transAxes, fontsize=17,
            fontweight="bold", va="bottom", ha="left")
ax_geo.set_aspect("equal")
ax_geo.legend(fontsize=11, loc="upper left", framealpha=0.95)
ax_geo.grid(True, alpha=0.25)


# (b) Optical fluence dL/dz with ratio subplot
gs_b = gs_main[0, 1].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.06)
ax_b = fig.add_subplot(gs_b[0])
ax_b_r = fig.add_subplot(gs_b[1], sharex=ax_b)

ax_b.plot(z_centers, gpu_dLdz, "-", color="#2c7fb8", lw=1.8, label="GPU TOPAS")
ax_b.plot(z_centers, cpu_dLdz, "--", color="#d95f0e", lw=1.4, label="CPU TOPAS")
ax_b.set_ylabel("Optical fluence (mm$^{-2}$)")
ax_b.text(-0.105, 1.02, "(b)", transform=ax_b.transAxes, fontsize=17,
          fontweight="bold", va="bottom", ha="left")
ax_b.legend(fontsize=11, loc="upper left")
ax_b.grid(True, alpha=0.3)
plt.setp(ax_b.get_xticklabels(), visible=False)

valid_b = cpu_dLdz > 0
dev_per_b = np.where(valid_b, (gpu_dLdz - cpu_dLdz) / np.where(valid_b, cpu_dLdz, 1) * 100, np.nan)
ax_b_r.plot(z_centers, dev_per_b, "-", color="darkgreen", lw=1.4)
ax_b_r.axhline(0.0, color="red", ls="--", lw=1.0, alpha=0.7)
ax_b_r.set_xlabel("z (mm)")
ax_b_r.set_ylabel(r"$\Delta$ (%)")
ax_b_r.set_ylim(-2, 2)
ax_b_r.grid(True, alpha=0.3)


# (c) Lateral profile at z=-20 mm with ratio subplot
gs_c = gs_main[1, 0].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.06)
ax_c = fig.add_subplot(gs_c[0])
ax_c_r = fig.add_subplot(gs_c[1], sharex=ax_c)

ax_c.plot(x_centers, gpu_lat, "-", color="#2c7fb8", lw=1.8, label="GPU TOPAS")
ax_c.plot(x_centers, cpu_lat, "--", color="#d95f0e", lw=1.4, label="CPU TOPAS")
ax_c.set_ylabel("Optical fluence (mm$^{-2}$)")
ax_c.text(-0.105, 1.02, "(c)", transform=ax_c.transAxes, fontsize=17,
          fontweight="bold", va="bottom", ha="left")
ax_c.legend(fontsize=11, loc="upper right")
ax_c.grid(True, alpha=0.3)
plt.setp(ax_c.get_xticklabels(), visible=False)

valid_c = cpu_lat > 0
dev_per_c = np.where(valid_c, (gpu_lat - cpu_lat) / np.where(valid_c, cpu_lat, 1) * 100, np.nan)
ax_c_r.plot(x_centers, dev_per_c, "-", color="darkgreen", lw=1.4)
ax_c_r.axhline(0.0, color="red", ls="--", lw=1.0, alpha=0.7)
ax_c_r.set_xlabel("x (mm)")
ax_c_r.set_ylabel(r"$\Delta$ (%)")
ax_c_r.set_ylim(-2, 2)
ax_c_r.grid(True, alpha=0.3)


# (d) Birks light-yield <Q>(T) — MC scan vs theory, with MC/theory difference sub-panel
gs_d = gs_main[1, 1].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.06)
ax_d   = fig.add_subplot(gs_d[0])
ax_d_r = fig.add_subplot(gs_d[1], sharex=ax_d)

valid_e_cpu = cpu_ed_1d > 1e-2
valid_e_gpu = gpu_ed_1d > 1e-2
dEdz_gpu = gpu_ed_1d / (N_EVENTS * bin_z)

entry_cpu = np.where(valid_e_cpu)[0][:5]
entry_gpu = np.where(valid_e_gpu)[0][:5]

# --- 4 phenomenological quenching models from literature (Schmidt-Sommerfeld et al. 2021, arXiv:2007.08366) ---
# Q(ε) = quenching factor where ε = dE/dx [MeV/mm]; light yield dL/dE = S · Q(ε)
# Plotted as L/E_normalized = Q(ε(z)) / Q(ε(entry))

# Birks model only (kB = 0.126 mm/MeV, Geant4 input value for BC-408).
kB_birks = 0.126      # Birks kB

# FRET (Förster) + Track Structure model — Yoshida et al. PLOS ONE 2018, doi:10.1371/journal.pone.0202011
# Simplified Specific Energy (SE) form: l_N/N = 1 - 3·R_F / (4·r_se)
# where r_se is donor mean spacing in track, r_se(ε) ∝ ε^(-1/3)
R_F = 4.0   # nm (Förster radius, paper uses 4 nm for NE-102A — closest analog to BC-408)
# Mean inter-donor spacing in track: r_se [nm] ≈ ((π·r_track² × E_donor) / ε)^(1/3)
# Using r_track = 1 nm (electron track core), E_donor = 8.76 eV (PVT excitation)
# For ε in MeV/mm = eV/nm: r_se(ε) = (π × 8.76 / ε)^(1/3) nm
def Q_fret(eps):
    """FRET quenching Q(ε) using Yoshida SE method (track structure + Förster)."""
    eps_eV_per_nm = np.maximum(eps, 1e-6)  # MeV/mm = eV/nm
    r_se = (np.pi * 1.0**2 * 8.76 / eps_eV_per_nm) ** (1.0 / 3.0)  # nm, 3D mean spacing
    Q = np.maximum(0.0, 1.0 - 3.0 * R_F / (4.0 * r_se))
    return Q

def Q_birks(eps):  return 1.0 / (1.0 + kB_birks * eps)

# Compute each model's normalized L/E
def normalize_at_entry(Q_func, eps):
    Q_arr = Q_func(eps)
    Q_entry = np.nanmean(Q_arr[entry_cpu])
    return Q_arr / Q_entry

# --- Compute L/E per Z bin via FINE-GRID INTEGRATION (proper Birks application) ---
# For each model, integrate Q(ε_local(z)) × ε_local(z) over the fine grid,
# then bin to the measurement Z bins. This matches Geant4's step-by-step integration.
def integrate_to_bins(Q_func, z_fine, dEdx_fine, z_centers, bin_z, sigma_strag=0.0):
    """Integrate Q × dE/dx and dE/dx along proton path, bin to coarse grid.
    Optional Gaussian convolution with σ = sigma_strag for range straggling."""
    Q_loc = Q_func(dEdx_fine)
    dz_fine = z_fine[1] - z_fine[0]
    L_fine = Q_loc * dEdx_fine
    E_fine = dEdx_fine
    # Apply Gaussian convolution (range straggling)
    if sigma_strag > 0:
        from scipy.ndimage import gaussian_filter1d
        sigma_pts = sigma_strag / dz_fine
        L_fine = gaussian_filter1d(L_fine, sigma=sigma_pts, mode='constant', cval=0)
        E_fine = gaussian_filter1d(E_fine, sigma=sigma_pts, mode='constant', cval=0)
    L_fine = L_fine * dz_fine
    E_fine = E_fine * dz_fine
    NZ_coarse = len(z_centers)
    L_coarse = np.zeros(NZ_coarse)
    E_coarse = np.zeros(NZ_coarse)
    bin_edges = np.linspace(z_centers[0] - bin_z/2, z_centers[-1] + bin_z/2, NZ_coarse + 1)
    bin_idx = np.searchsorted(bin_edges, z_fine, side='right') - 1
    valid = (bin_idx >= 0) & (bin_idx < NZ_coarse)
    np.add.at(L_coarse, bin_idx[valid], L_fine[valid])
    np.add.at(E_coarse, bin_idx[valid], E_fine[valid])
    return np.where(E_coarse > 1e-6, L_coarse / E_coarse, np.nan)

# --- Compute theoretical EMISSION PROFILE dL/dz = Q(ε) × ε per bin (with straggling) ---
# This is what would be measured WITHOUT photon transport. Direct comparison to MC dL/dz
# shows the shape difference due to transport (cube has λ_abs=200mm vs cube=100mm so significant).
def emission_profile(Q_func, z_fine, dEdx_fine, z_centers, bin_z, sigma_strag=0.0):
    """Birks-quenched emission rate Q(ε)×ε binned to coarse Z grid (with straggling)."""
    L_fine = Q_func(dEdx_fine) * dEdx_fine
    dz_fine = z_fine[1] - z_fine[0]
    if sigma_strag > 0:
        from scipy.ndimage import gaussian_filter1d
        L_fine = gaussian_filter1d(L_fine, sigma=sigma_strag / dz_fine, mode='constant', cval=0)
    NZ_coarse = len(z_centers)
    L_coarse = np.zeros(NZ_coarse)
    bin_edges = np.linspace(z_centers[0] - bin_z/2, z_centers[-1] + bin_z/2, NZ_coarse + 1)
    bin_idx = np.searchsorted(bin_edges, z_fine, side='right') - 1
    valid = (bin_idx >= 0) & (bin_idx < NZ_coarse)
    np.add.at(L_coarse, bin_idx[valid], L_fine[valid] * dz_fine)
    return L_coarse

birks_emit = emission_profile(Q_birks,  z_fine, dEdx_fine, z_centers, bin_z, sigma_R)
# Pure dE/dx (no Birks) for reference — proton dose profile
pure_dose = emission_profile(lambda eps: np.ones_like(eps), z_fine, dEdx_fine, z_centers, bin_z, sigma_R)

# Q(T) — universal scintillation efficiency vs proton kinetic energy (Birks plot).
# Shows light reduction directly: Q=0 means total quenching, Q=1 means no quenching.
T_axis = np.logspace(np.log10(0.5), np.log10(150), 400)  # 0.5 to 150 MeV
S_axis = dEdx_T(T_axis)

# --- Plot integrated <Q>(T_in) — what MC scan measures ---
# <Q>(T_in) = ∫Q×dE / ∫dE  over proton path within cube
# For T_in such that proton stops in cube: integrate from T=0 to T_in
# For T_in such that proton exits: integrate from T_out (residual) to T_in
CUBE_LENGTH = 2 * sci_HLZ  # mm (matches Sci HLZ * 2)

from scipy.special import erfc as _erfc

def proton_range_T(T_in):
    """Mean CSDA range of proton with kinetic energy T_in [MeV] in BC-408 [mm]."""
    sol = solve_ivp(lambda z, T: [-dEdx_T(T[0])], [0, 1e4], [T_in],
                    method='RK45', rtol=1e-8, atol=1e-9,
                    events=lambda z, T: T[0] - 0.05)
    return sol.t_events[0][0] if len(sol.t_events[0]) else 1e4

def proton_T_out(T_in, cube_length=CUBE_LENGTH):
    """Residual energy of proton after traversing cube (0 if stops)."""
    sol = solve_ivp(lambda z, T: [-dEdx_T(T[0])], [0, cube_length], [T_in],
                    method='RK45', rtol=1e-8, atol=1e-9)
    return max(sol.y[0][-1], 0.0)

def avg_Q(Q_func, T_array, cube_length=CUBE_LENGTH):
    """<Q>(T_in) = light-yield reduction averaged over the in-cube proton path,
    smeared by Bohr range straggling so the stop→transit transition is
    continuous instead of producing the visible kink at T_in≈115 MeV.
    p_transit = 0.5·erfc((L-R)/(σ_R·√2)) blends the two regimes."""
    out = np.zeros_like(T_array)
    for i, Tin in enumerate(T_array):
        # avgQ_stop: proton fully stops in cube (uses entire 0..Tin path)
        Tg_s = np.linspace(0.05, Tin, 4000)
        num_s = np.trapz(Q_func(dEdx_T(Tg_s)), Tg_s)
        avgQ_stop = num_s / Tin if Tin > 0 else 0
        # avgQ_transit: proton exits cube with residual T_out (uses T_out..Tin)
        T_out = proton_T_out(Tin, cube_length)
        if T_out > 0.05 and Tin > T_out:
            Tg_t = np.linspace(T_out, Tin, 4000)
            avgQ_transit = np.trapz(Q_func(dEdx_T(Tg_t)), Tg_t) / (Tin - T_out)
        else:
            avgQ_transit = avgQ_stop
        # Blend by Bohr range-straggling probability of cube transit
        R = proton_range_T(Tin)
        sigR = max(R / 35.0, 1e-6)
        p_tr = 0.5 * _erfc((cube_length - R) / (sigR * np.sqrt(2)))
        out[i] = (1 - p_tr) * avgQ_stop + p_tr * avgQ_transit
    return out

avgQ_birks = avg_Q(Q_birks, T_axis)

ax_d.semilogx(T_axis, avgQ_birks, "-",  color="#7a3f9a", lw=2.4,
              label=fr"Birks model ($kB$={kB_birks})")

# --- Multi-energy MC scan: integrated <Q>(T_in) measurements ---
def load_csv_sum(p):
    import os
    if not os.path.exists(p): return None
    s = 0.0
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            try: s += float(line.split(",")[-1])
            except: pass
    return s

# Q(T) = L(Birks on)/L(Birks off) per energy, averaged over 5 independent seeds (mean ± SD).
scan_T = np.array([2, 5, 10, 20, 30, 50, 75, 100])
SCAN_SEEDS = [1, 2, 3, 4, 5]
def _scan_Q(pfx, T):
    qs = []
    for s in SCAN_SEEDS:
        lon = load_csv_sum(RUNS / f"U5seed_{pfx}T{T}_on_s{s}_fluence.csv")
        loff = load_csv_sum(RUNS / f"U5seed_{pfx}T{T}_off_s{s}_fluence.csv")
        if lon and loff and loff > 0:
            qs.append(lon / loff)
    qs = np.array(qs)
    return (np.nan, np.nan) if qs.size == 0 else (qs.mean(), qs.std())
cpu_m, cpu_sd, gpu_m, gpu_sd = [], [], [], []
for T in scan_T:
    m, sd = _scan_Q("", T);     cpu_m.append(m); cpu_sd.append(sd)
    m, sd = _scan_Q("GPU_", T); gpu_m.append(m); gpu_sd.append(sd)
scan_Q_cpu = np.array(cpu_m); scan_Q_gpu = np.array(gpu_m)
ax_d.errorbar(scan_T, scan_Q_cpu, yerr=np.array(cpu_sd), fmt="x", color="#d33", ms=10,
              mew=2.2, capsize=3, elinewidth=1.2, ecolor="#d33",
              label="MC CPU TOPAS", zorder=20)
ax_d.errorbar(scan_T, scan_Q_gpu, yerr=np.array(gpu_sd), fmt="s", color="#2c7fb8", ms=8,
              mfc="none", mew=1.6, capsize=3, elinewidth=1.2, ecolor="#2c7fb8",
              label="MC GPU TOPAS", zorder=21)

# Compute total light yield reduction L_birks / L_no_quench per proton (integrated over its full range)
def light_reduction(Q_func, T_initial=100.0):
    Tg = np.logspace(np.log10(0.05), np.log10(T_initial), 5000)
    return np.trapz(Q_func(dEdx_T(Tg)), Tg) / T_initial

red_birks = light_reduction(Q_birks)


# --- MC Q_eff(T) extracted from birks-on / birks-off ratio (cancels transport) ---
# At each bin z, Q_MC(z) = (dL/dz)_with_birks / (dL/dz)_no_birks
# Map z → T via analytical proton stopping curve, then plot Q_MC(T) overlay
# Total integrated reduction (cube-full ScoreBox single-bin scorer):
total_red_cpu = cpu_full_on / cpu_full_off if cpu_full_off > 0 else np.nan
total_red_gpu = gpu_full_on / gpu_full_off if gpu_full_off > 0 else np.nan

# Per-bin Q_MC(z) — scatter, transport-contaminated but reveals trend
ratio_cpu = np.where(cpu_dLdz_off > 1e-3, cpu_dLdz / np.maximum(cpu_dLdz_off, 1e-9), np.nan)
ratio_gpu = np.where(gpu_dLdz_off > 1e-3, gpu_dLdz / np.maximum(gpu_dLdz_off, 1e-9), np.nan)

# Map z (bin centers) to T using analytical solution
from scipy.interpolate import interp1d as interp1d_full
z_to_T = interp1d_full(z_fine, T_fine, bounds_error=False, fill_value=np.nan)
T_at_zbin = z_to_T(z_centers)

# Only plot points where T is well-defined and Q is sensible
mask_mc = (T_at_zbin > 0.5) & (T_at_zbin < 100) & np.isfinite(ratio_cpu) & np.isfinite(ratio_gpu)
# Per-bin MC ratio omitted — transport-contaminated; multi-energy scan is canonical

ax_d.set_ylabel(r"$\langle Q\rangle = L_{birks}/L_{no\,birks}$ (per proton)")
ax_d.text(-0.105, 1.02, "(d)", transform=ax_d.transAxes, fontsize=17,
          fontweight="bold", va="bottom", ha="left")
ax_d.set_xlim(0.5, 200)
ax_d.set_ylim(0, 1.10)
ax_d.legend(fontsize=11, loc="upper left", framealpha=0.92, ncol=1)
ax_d.grid(True, alpha=0.3, which="both")
plt.setp(ax_d.get_xticklabels(), visible=False)

# (d) lower panel — MC data points relative to the analytical theory curve at each
# scan energy: ratio Q_MC / Q_theory (both engines), within ±5 %.
Q_theory_scan = avg_Q(Q_birks, scan_T.astype(float))
dev_d_cpu = (scan_Q_cpu / Q_theory_scan - 1) * 100
dev_d_gpu = (scan_Q_gpu / Q_theory_scan - 1) * 100
ax_d_r.semilogx(scan_T, dev_d_cpu, "x", color="#d33", ms=9, mew=2.0,
                zorder=20, label="CPU vs theory")
ax_d_r.semilogx(scan_T, dev_d_gpu, "s", color="#2c7fb8", ms=7, mfc="none", mew=1.5,
                zorder=21, label="GPU vs theory")
ax_d_r.axhline(0.0, color="red", ls="--", lw=1.0, alpha=0.7)
ax_d_r.set_xlabel(r"Initial proton kinetic energy (MeV)")
ax_d_r.set_ylabel("Difference (%)")
ax_d_r.set_xlim(0.5, 200)
ax_d_r.set_ylim(-5, 5)
ax_d_r.grid(True, alpha=0.3, which="both")

# Suptitle
# (suptitle + bottom caption box removed 2026-06-01: house style — caption goes
#  in the manuscript body, not on the figure)

out_pdf = OUT / "fig_04_birks.pdf"
out_png = OUT / "fig_04_birks.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"wrote {out_pdf}\nwrote {out_png}")
