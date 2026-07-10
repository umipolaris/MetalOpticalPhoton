#!/usr/bin/env python3
"""
Benchmark 10 analysis — curved-surface analytic vs mesh.

Reads results/<solid>_<method>_R<r>_<G|C>_s<seed>.csv (+ _trans) and reports,
for each (solid, radius), the GPU-computed chord for the analytic and mesh
intersection paths against the CPU Geant4 reference and the true geometric
chord (= 2*R). The chord is L = -Labs * ln(F_abs / F_transparent), Labs=10 mm.
"""
import glob, math, os, re, statistics as st

RESDIR = os.path.join(os.path.dirname(__file__), "results")
LABS = 10.0  # mm

def last_val(path):
    with open(path) as f:
        line = [l for l in f.read().strip().splitlines() if l and l[0].isdigit()][-1]
    return float(line.split(",")[-1])

def trans(tag):
    p = os.path.join(RESDIR, tag + ".csv")
    return last_val(p) if os.path.exists(p) else None

def seeds(prefix):
    out = []
    for p in sorted(glob.glob(os.path.join(RESDIR, prefix + "_s*.csv"))):
        try:
            out.append(last_val(p))
        except Exception:
            pass
    return out

def chord(Fabs, F0):
    return -LABS * math.log(Fabs / F0) if (Fabs > 0 and F0 and F0 > 0) else float("nan")

solids = ["sphere", "cylinder", "torus"]
radii = sorted({int(re.search(r"_R(\d+)_", os.path.basename(p)).group(1))
                for p in glob.glob(os.path.join(RESDIR, "*_R*_*.csv"))})

print(f"{'solid':<9}{'R':>3} {'true':>5} | {'CPU chord':>10} | "
      f"{'GPU-analytic':>22} | {'GPU-mesh':>22}")
print(f"{'':<9}{'':>3}{'(mm)':>6} | {'mean(mm)':>10} | "
      f"{'mean±SD(mm)':>13} {'Δvs CPU':>8} | {'mean±SD(mm)':>13} {'Δvs CPU':>8}")
print("-" * 96)
for solid in solids:
    for R in radii:
        rt = f"R{R}"
        true = 2.0 * R * 10  # mm
        c_trans = trans(f"{solid}_{rt}_C_trans")
        ga_trans = trans(f"{solid}_analytic_{rt}_G_trans")
        gm_trans = trans(f"{solid}_mesh_{rt}_G_trans")
        c = [chord(v, c_trans) for v in seeds(f"{solid}_{rt}_C")]
        ga = [chord(v, ga_trans) for v in seeds(f"{solid}_analytic_{rt}_G")]
        gm = [chord(v, gm_trans) for v in seeds(f"{solid}_mesh_{rt}_G")]
        if not c:
            continue
        cm = st.mean(c)
        def fmt(x):
            return (f"{st.mean(x):.2f}±{st.pstdev(x):.2f}", st.mean(x) - cm) if x else ("--", float("nan"))
        ga_s, ga_d = fmt(ga); gm_s, gm_d = fmt(gm)
        print(f"{solid:<9}{R:>3} {true:>5.0f} | {cm:>9.2f}  | "
              f"{ga_s:>13} {ga_d:>+7.2f}  | {gm_s:>13} {gm_d:>+7.2f}")
print("-" * 96)
print("Δ = GPU chord − CPU chord (mm). true = 2·R. Labs = 10 mm.")
