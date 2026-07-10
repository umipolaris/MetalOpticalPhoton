#!/bin/bash
# ============================================================
# 04_birks — Birks light yield + Q(T) scintillation
# Standalone: cd benchmark/04_birks && ./run_profile.sh
# ============================================================
set -e
cd "$(dirname "$0")"

. ../common/topas_env.sh
[ -x "$TOPAS" ] || { echo "ERROR: TOPAS binary not found at $TOPAS"; exit 1; }
export G4TRACE_DIR=OFF   # g4trace disk logging off (if not disabled, CPU is tens to hundreds of times slower; bit-identical)

mkdir -p results
cd configs

echo "=============================================="
echo " 04_birks: Birks scintillation (U5_short)"
echo "=============================================="

# Birks on / off, z scorer (depth dist) + main fluence.
# CPU main (U5_short_cpu) is multi-scorer (outputs fluence+z+x+edep in one run) → cpu + cpu_birksoff
# These 2 produce all of z/x/fluence/edep on·off (cpu_z/cpu_x/cpu_z_birksoff regenerate the same CSVs, so they are redundant and removed).
# GPU main only has fluence+edep, so z/x require separate configs (5 kept).
for cfg in U5_short_gpu_z U5_short_gpu_z_birksoff U5_short_gpu U5_short_gpu_birksoff U5_short_gpu_x \
           U5_short_cpu U5_short_cpu_birksoff; do
    [ -f $cfg.txt ] || { echo "  [skip] $cfg.txt not found"; continue; }
    echo "[$(date +%T)] running $cfg ..."
    "$TOPAS" $cfg.txt > ../results/$cfg.log 2>&1
    exec_t=$(grep "Execution:" ../results/$cfg.log | grep -oE "Real=[0-9.]+s" | head -1 | sed 's/Real=//;s/s//')
    printf "    Execution=%8ss\n" "$exec_t"
done

cd ..

echo
echo "=== U5 Birks quenching Q (whole-scintillator scorer) GPU vs CPU ==="
python3 - <<'PY'
import os
def s(f):
    if not os.path.exists(f): return None
    t = 0
    for l in open(f):
        if l.startswith('#') or not l.strip(): continue
        t += float(l.split(',')[-1])
    return t

cf_on  = s('results/U5short_CPU_fluence.csv')
cf_off = s('results/U5short_CPU_fluence_birksoff.csv')
gf_on  = s('results/U5short_GPU_fluence.csv')
gf_off = s('results/U5short_GPU_fluence_birksoff.csv')

# NOTE: use the WHOLE-scintillator scorer (U5short_*_fluence) for Q, not the thin
#   1x1 mm central z-tube (U5short_*_fluence_z): Birks shifts the light's spatial
#   distribution, so the tube captures the on/off ratio with a geometric bias. The
#   physical Birks quenching Q is the whole-volume on/off ratio.
if all((cf_on, cf_off, gf_on, gf_off)):
    Qc = cf_on / cf_off; Qg = gf_on / gf_off
    print(f"  CPU Birks Q (whole scintillator) = {Qc:.4f}")
    print(f"  GPU Birks Q (whole scintillator) = {Qg:.4f}")
    print(f"  GPU/CPU Q ratio = {Qg/Qc:.5f}  (Delta = {100*(Qg/Qc-1):+.3f} %)")
else:
    print(f"  CPU on={cf_on}, off={cf_off}, GPU on={gf_on}, off={gf_off} (some missing)")
PY

echo
echo "Results: results/*.log, configs/*.csv (TOPAS CSV)"

# === Figure generation ===
if command -v python3 >/dev/null 2>&1 && [ -f plot.py ]; then
    echo
    echo "=== Generating figure (plot.py) ==="
    python3 plot.py || echo "[warn] plot.py failed"
fi
