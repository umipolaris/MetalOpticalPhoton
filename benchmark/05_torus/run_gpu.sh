#!/bin/bash
# ============================================================
# 05_torus — GPU 10B (BEAM kernel)
#   → results/torus_diffuse_10B_beamgpu_GPU.csv  (816×624×1, thin focal-plane)
#
# Usage:   ./run_gpu.sh
#   The BEAM kernel has the GPU generate 10B photons in a single trigger event,
#   so it is independent of CPU threads (NumberOfThreads).
#   N = 10B (config's u:BeamPhotons = 1.0e10), MaxStepsPerPhoton=1000.
# ============================================================
set -e
cd "$(dirname "$0")"

. ../common/topas_env.sh
[ -x "$TOPAS" ] || { echo "ERROR: topas-gpu not found at $TOPAS"; exit 1; }

mkdir -p results
ln -sf ../../common/simple_biconvex_lens_new.stl configs/simple_biconvex_lens_new.stl
cd configs

echo "[$(date +%T)] GPU 10B (BEAM kernel, MaxStepsPerPhoton=1000) ..."
G4TRACE_DIR=OFF "$TOPAS" torus_diffuse_10B_beamgpu.txt > ../results/torus_diffuse_10B_beamgpu.log 2>&1
grep "Execution:" ../results/torus_diffuse_10B_beamgpu.log | grep -oE "Real=[0-9.]+s" | head -1
grep "EndOfRun GPU buffer" ../results/torus_diffuse_10B_beamgpu.log | grep -oE "bins=[0-9]+ sum=[0-9.e+]+" | head -1
echo "→ results/torus_diffuse_10B_beamgpu_GPU.csv  (figure: after generating both GPU and CPU CSVs, run 'python3 plot.py')"
