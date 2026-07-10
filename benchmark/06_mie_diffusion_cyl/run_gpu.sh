#!/bin/bash
# ============================================================
# 06_mie_diffusion_cyl — paper figure (weak fog mfp=30) GPU re-run
#   multi-scorer fix scores barrel φ-z + x-z box + z=−50 x-y simultaneously in **one run** (3 scorers):
#     spread_m30_both_gpu.txt (BEAM kernel 100M)
#       → results/spread_m30_barrel_GPU.csv  (φ-z, barrel total fluence G/C)
#       → results/spread_m30_xz_GPU.csv      (x-z beam trajectory, figure (a))
#       → results/spread_m30_xy50_GPU.csv    (z=−50 x-y, figure (b) GPU−CPU difference)
#   When done, if the CPU cache (spread_m30_barrel_CPU.csv) exists, plot.py runs automatically.
#
# Usage:  ./run_gpu.sh
#   The GPU uses the BEAM kernel (single trigger event), so it is independent of CPU threads.
# ============================================================
set -e
cd "$(dirname "$0")"

. ../common/topas_env.sh
[ -x "$TOPAS" ] || { echo "ERROR: topas-gpu not found at $TOPAS"; exit 1; }

mkdir -p results
cd configs

echo "[$(date +%T)] GPU barrel+x-z simultaneous (spread_m30_both_gpu.txt, 100M, multi-scorer) ..."
G4TRACE_DIR=OFF "$TOPAS" spread_m30_both_gpu.txt > ../results/spread_m30_both_gpu.log 2>&1
grep "Execution:" ../results/spread_m30_both_gpu.log | grep -oE "Real=[0-9.]+s" | head -1

cd ..
echo "→ results/spread_m30_barrel_GPU.csv, spread_m30_xz_GPU.csv"
if [ -f results/spread_m30_barrel_CPU.csv ]; then
    echo "[$(date +%T)] CPU cache found → regenerating figure with plot.py"
    G4TRACE_DIR=OFF python3 plot.py
else
    echo "CPU cache (spread_m30_barrel_CPU.csv) missing → run './run_cpu.sh' first, then 'python3 plot.py'"
fi
