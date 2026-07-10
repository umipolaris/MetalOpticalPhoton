#!/bin/bash
# ============================================================
# run_gpu.sh — 07 Cherenkov in water, on-axis z-fluence validation, GPU only
#   Auto-runs 11 cases (C12 3 + e- 4 + proton 4).
#   CASES default = all 11 cases. A subset can be specified via argument or env.
#   e.g.: ./run_gpu.sh "p100 p250"     or     CASES="p100 p250" ./run_gpu.sh
#   thread: always fixed at 20 (unified GPU/CPU). T=1 serializes the charged-particle
#           transport (Geant4/CPU), so heavy cases (C12 ions etc.) slow to minutes — do not use.
#   override: TOPAS=/path/topas-gpu ./run_gpu.sh
#   CSV -> results/<case>_xz_GPU.csv. For the figure, after both (gpu+cpu) have run:
#         python3 plot.py
# ============================================================
cd "$(dirname "$0")"

. ../common/topas_env.sh
CASES="${1:-${CASES:-c12_200 c12_300 c12_430 e1 e5 e10 e50 p100 p150 p200 p250}}"
THREADS=20   # GPU/CPU always fixed at 20 (env override disabled)
export G4TRACE_DIR=OFF

[ -x "$TOPAS" ] || { echo "ERROR: TOPAS binary not found: $TOPAS"; exit 1; }
mkdir -p results
cd configs

echo "=== 07 GPU Cherenkov z-fluence  (cases: $CASES, T=$THREADS) ==="
printf "%-10s %16s\n" "case" "Execution(s)"
for c in $CASES; do
  cfg="cherenkov_${c}_xz_gpu.txt"
  [ -f "$cfg" ] || { printf "%-10s %16s\n" "$c" "[no config]"; continue; }
  tmp=".tmp_${c}_gpu.txt"
  sed "s|^i:Ts/NumberOfThreads = .*|i:Ts/NumberOfThreads = ${THREADS}|" "$cfg" > "$tmp"
  G4TRACE_DIR=OFF "$TOPAS" "$tmp" > "../results/run_${c}_gpu.log" 2>&1
  t=$(grep "Execution:" "../results/run_${c}_gpu.log" | grep -oE "Real=[0-9.]+s" | head -1 | sed 's/Real=//;s/s//')
  printf "%-10s %16s\n" "$c" "${t:-FAILED}"
  rm -f "$tmp"
done
echo "CSV: benchmark/07_cherenkov_water/results/<case>_xz_GPU.csv"
echo "figure (after running both gpu+cpu): python3 plot.py"
