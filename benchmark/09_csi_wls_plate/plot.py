#!/usr/bin/env python3
"""CsI + WLS plate — (a) 2D wavelength + Bragg, (b) GPU vs CPU z-profile."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, Normalize
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.unicode_minus": False,
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "savefig.dpi": 300,
})

def _gauss(x, mu, s1, s2):
    s = s1 if x < mu else s2
    return np.exp(-0.5*((x-mu)/s)**2)
def wl_to_rgb_cie(nm):
    X = 1.056*_gauss(nm,599.8,37.9,31.0)+0.362*_gauss(nm,442.0,16.0,26.7)-0.065*_gauss(nm,501.1,20.4,26.2)
    Y = 0.821*_gauss(nm,568.8,46.9,40.5)+0.286*_gauss(nm,530.9,16.3,31.1)
    Z = 1.217*_gauss(nm,437.0,11.8,36.0)+0.681*_gauss(nm,459.0,26.0,13.8)
    r = 3.2406*X-1.5372*Y-0.4986*Z; g = -0.9689*X+1.8758*Y+0.0415*Z; b = 0.0557*X-0.2040*Y+1.0570*Z
    r, g, b = max(r,0), max(g,0), max(b,0)
    gam = lambda c: 1.055*c**(1/2.4)-0.055 if c > 0.0031308 else 12.92*c
    return np.array([gam(r), gam(g), gam(b)])

NX, NZ = 80, 145
# 2026-06-04: multi-E fluence CSV layout = col0:underflow, col1..NB:energy bins,
#   then overflow [, no-track]. There are NO iX,iY,iZ index columns — voxel position
#   is ROW ORDER (iZ-outer, iX-inner -> reshape(NZ,NX)), identical for GPU and CPU.
#   Energy bins are cols 1..NB (NB = i:Sc/PS/EBins). The previous [:,3:] wrongly applied the
#   index-format pattern, dropping underflow+e0+e1 and including overflow/no-track -> a 2-slot
#   energy-bin offset (sum and z-profile were unaffected since those bins are ~0; only the
#   wavelength color map was misaligned). Alignment corrected.
NB = 50
_graw = np.loadtxt("results/csi_wls_GPU.csv", comments="#", delimiter=",")
opt = _graw[:, 1:1+NB].reshape(NZ, NX, NB)
Zo = -70 + (np.arange(NZ)+0.5)*145/NZ
Xo = -40 + (np.arange(NX)+0.5)*80/NX
E = 1.5 + (np.arange(NB)+0.5)*2.0/NB
nm = 1239.84/E
flux = opt.sum(2).T
rgb_lut = np.array([wl_to_rgb_cie(nm[j]) for j in range(NB)])
rgb_avg = np.clip(np.tensordot(opt, rgb_lut, axes=([2],[0])).transpose(1,0,2) / (flux[...,None]+1e-30), 0, 1)
I = np.zeros((NX, NZ))
for col in [Zo < 20.0, (Zo >= 20.0) & (Zo < 25.0), Zo >= 25.0]:
    m = flux[:, col].max()
    if m > 0: I[:, col] = np.power(flux[:, col]/m, 0.5)
img = np.clip(1.0 - I[...,None]*(1.0 - rgb_avg), 0, 1)

# CPU z-profile (for comparison) — same layout/energy cols (1..NB) as GPU; PreStep CSV just lacks
# the trailing no-track column, which cols 1..NB correctly excludes.
cpu = np.loadtxt("results/csi_wls_CPU.csv", comments="#", delimiter=",")
cpu_opt = cpu[:, 1:1+NB].reshape(NZ, NX, NB)
cpu_flux = cpu_opt.sum(2).T

# dose (CsI box 20mm)
DNX, DNZ = 50, 80
dd = np.loadtxt("results/csi_wls_dose.csv", comments="#", delimiter=",")
# dose CSV is INDEX format (iX,iY,iZ,val) — placed by index columns (row-order independent, immune).
dose = np.zeros((DNZ, DNX))
for r in dd:
    dose[int(r[2]), int(r[0])] = r[3]
Zd = -20 + (np.arange(DNZ)+0.5)*40/DNZ
Xd = -10 + (np.arange(DNX)+0.5)*20/DNX

fig = plt.figure(figsize=(18,5.0), facecolor="white")
gs = gridspec.GridSpec(2,2,width_ratios=[1,1],height_ratios=[2.5,1],wspace=0.3,hspace=0.08)

# (a) 2D visible light (wavelength)
axa = fig.add_subplot(gs[0,0])
axa.imshow(img, extent=(-70,75,-40,40), origin="lower", aspect="auto", interpolation="bilinear")
axa.add_patch(Rectangle((-20,-10),40,20, fill=False, ec="blue", ls="-", lw=1.6))
axa.add_patch(Rectangle((20,-10),5,20, fill=False, ec="black", ls="-", lw=2.4))
axa.annotate("", xy=(-20,0), xytext=(-50,0), arrowprops=dict(arrowstyle="-|>",color="red",lw=2.5))
axa.text(-37, 1.5, "proton 100 MeV", color="red", fontsize=11, ha="center", va="bottom")
axa.annotate("CsI(Tl) (550nm)", xy=(0,10), xytext=(-22,18), color="blue", fontsize=11,
             ha="center", va="center", arrowprops=dict(arrowstyle="-", color="blue", lw=0.8))
axa.annotate("WLS plate (600nm)", xy=(22.5,10), xytext=(40,18), color="black", fontsize=11,
             ha="center", va="center", arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
axa.set_xlim(-70,75); axa.set_ylim(-20,20); axa.set_ylabel("X (mm)")
plt.setp(axa.get_xticklabels(), visible=False)
axa.text(-0.105, 1.02, "(a)", transform=axa.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="left")
axa.set_anchor('S')

cmap_w = ListedColormap([np.clip(wl_to_rgb_cie(w),0,1) for w in np.linspace(400,700,256)])
sm = mpl.cm.ScalarMappable(cmap=cmap_w, norm=Normalize(400, 700))
cax = axa.inset_axes([1.015, 0.0, 0.02, 1.0])
cb = fig.colorbar(sm, cax=cax); cb.set_label("Wavelength (nm)")
cb.set_ticks([400,450,500,550,600,650,700])

# (below a) proton dose (Bragg peak) — Z-aligned with a (sharex)
axbr = fig.add_subplot(gs[1,0], sharex=axa)
imbr = axbr.imshow(dose.T, extent=(-20,20,-10,10), origin="lower", aspect="auto",
                   cmap="inferno", interpolation="bilinear")
axbr.add_patch(Rectangle((-20,-10),40,20, fill=False, ec="white", ls="-", lw=1.0))
bz = Zd[np.argmax(dose.sum(1))]
axbr.axvline(bz, ls="--", color="yellow", lw=1, alpha=0.9)
axbr.text(bz-2, 6, f"Bragg-peak\nZ≈{bz:.0f} mm", color="yellow", fontsize=10, va="center", ha="right")
axbr.set_xlim(-70,75); axbr.set_ylim(-10,10)
axbr.set_xlabel("Z (mm)"); axbr.set_ylabel("X (mm)")
caxbr = axbr.inset_axes([1.015, 0.0, 0.02, 1.0])
cbbr = fig.colorbar(imbr, cax=caxbr); cbbr.set_label("Dose (Gy)")

# (b) GPU vs CPU z-profile (top 4) + (GPU−CPU)/CPU % (bottom 1)
gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:,1], height_ratios=[4,1], hspace=0.05)
axb = fig.add_subplot(gs_b[0])
axbd = fig.add_subplot(gs_b[1], sharex=axb)
gprof = flux.sum(0)
cprof = cpu_flux.sum(0)
gc = gprof.sum()/cprof.sum() if cprof.sum() else 0
axb.semilogy(Zo, np.maximum(gprof, 1e-30), color="navy", lw=1.6, label="GPU")
axb.semilogy(Zo, np.maximum(cprof, 1e-30), color="crimson", lw=1.3, ls="--", label="CPU")
axb.axvspan(-20, 20, color="blue", alpha=0.06)
axb.axvspan(20, 25, color="gray", alpha=0.12)
ymin = max(min(gprof[gprof>0].min(), cprof[cprof>0].min())*0.5, max(gprof.max(),cprof.max())*1e-5)
axb.set_xlim(-70,75); axb.set_ylim(ymin, max(gprof.max(),cprof.max())*2)
axb.set_ylabel("Photon fluence (mm⁻²)")
plt.setp(axb.get_xticklabels(), visible=False)
axb.legend(loc="upper right", fontsize=10)
axb.grid(True, which="both", ls=":", alpha=0.4)
axb.text(-0.105, 1.02, "(b)", transform=axb.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="left")
axb.text(0, ymin*2, "CsI(Tl)", color="blue", fontsize=11, va="bottom", ha="center")
axb.text(22.5, ymin*2, "WLS", color="gray", fontsize=11, va="bottom", ha="center")

# (below b) GPU-CPU difference % — all bins with photons (cprof>0)
diff = np.where(cprof > 0, (gprof - cprof)/cprof*100, np.nan)
axbd.plot(Zo, diff, color="purple", lw=1.2)
axbd.axhline(0, color="k", lw=0.6)
axbd.axhspan(-1, 1, color="green", alpha=0.12)
axbd.axvspan(-20, 20, color="blue", alpha=0.06)
axbd.axvspan(20, 25, color="gray", alpha=0.12)
axbd.set_xlim(-70,75); axbd.set_ylim(-0.5,0.5)
axbd.set_xlabel("Z (mm)"); axbd.set_ylabel(r"$\Delta$ (%)", labelpad=2)
axbd.grid(True, ls=":", alpha=0.4)

# b height = total height of a (image+Bragg), split 4:1 + position alignment
fig.canvas.draw()
pa = axa.get_position(); pbr = axbr.get_position(); pb = axb.get_position()
total_h = pa.y1 - pbr.y0; gap = 0.025
h_bot = total_h * 1.0/5.0; h_top = total_h * 4.0/5.0 - gap
axb.set_position([pb.x0, pbr.y0 + h_bot + gap, pb.width, h_top])
axbd.set_position([pb.x0, pbr.y0, pb.width, h_bot])

fig.savefig("results/csi_wls_dose.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("results/csi_wls_dose.pdf", bbox_inches="tight", facecolor="white")
print("wrote csi_wls_dose.png / .pdf  G/C=%.4f" % gc)
