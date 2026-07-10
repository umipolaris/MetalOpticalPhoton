#!/bin/bash
# ============================================================
# run_gpu.sh — 02 U1/U2 Fresnel reflection/refraction + TIR angle scan, GPU only
#   Builds + runs 3 tests (U1 s-pol / U1 p-pol / U2 TIR) over the specified angles.
#   ANGLES default = all 12 angles. Can be partially specified via argument or env.
#   e.g.: ./run_gpu.sh "30 45"     or     ANGLES="30 45" ./run_gpu.sh
#   thread: THREADS=8 ./run_gpu.sh   (default 1; when set, the runner injects i:Ts/NumberOfThreads)
#   override: TOPAS=/path/topas-gpu ./run_gpu.sh
#   CSV → results/built/U*_gpu_th*.csv. The figure is made after both (gpu+cpu) have run:
#         python3 scripts/fig_02_angle_scan.py
# ============================================================
cd "$(dirname "$0")"

. ../common/topas_env.sh
export MODE="gpu"
export ANGLES="${1:-${ANGLES:-5 15 25 30 35 38 40 45 50 60 70 85}}"
export THREADS="${THREADS:-1}"
export G4TRACE_DIR=OFF

[ -x "$TOPAS" ] || { echo "ERROR: TOPAS binary not found: $TOPAS"; exit 1; }
mkdir -p results/built
# The shared shell (common/unit_test_shell.txt → bc408.txt) uses the ../../common/ relative path,
# but TOPAS resolves includes relative to the run cwd (=results/built, 3-level) → ../../common
# points to 02_dir/common (nonexistent) and fails. The common→benchmark/common symlink corrects for the 2-level structure.
# -sfn: avoids the ln -sf trap where re-running burrows into the existing common symlink and
#       creates a ../common/common stray (no-dereference replaces the symlink itself).
ln -sfn ../common common

echo "=== 02 GPU angle scan (TOPAS-Beam, angles: $ANGLES, T=$THREADS) ==="
for s in build_run_u1v5.py build_run_u1v5_ppol.py build_run_u1v5_unpol.py build_run_u2v5.py; do
  echo "--- $s ---"
  python3 "scripts/$s" || echo "[warn] $s failed"
done
echo "CSV: results/built/U{1v5,1v5_ppol,1v5_unpol,2v5}_gpu_th*.csv"
echo "Standard figure (after both gpu+cpu have run): python3 scripts/fig_02_angle_scan.py"
