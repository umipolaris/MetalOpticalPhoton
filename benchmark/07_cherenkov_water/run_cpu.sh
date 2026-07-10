#!/bin/bash
# ============================================================
# run_cpu.sh — 07 Cherenkov in water, on-axis z-fluence validation, CPU (Geant4) only
#   Auto-runs 11 cases (C12 3 + e- 4 + proton 4).
#   CASES default = all 11 cases. A subset can be specified via argument or env.
#   e.g.: ./run_cpu.sh "p100 p250"     or     CASES="p100 p250" ./run_cpu.sh
#   thread: always fixed at 20 (unified GPU/CPU).
#   override: TOPAS=/path/topas-gpu ./run_cpu.sh   (patched topas-gpu required)
#   CSV -> results/<case>_xz_CPU.csv. For the figure, after both (gpu+cpu) have run:
#         python3 plot.py
# ============================================================
cd "$(dirname "$0")"

. ../common/topas_env.sh
CASES="${1:-${CASES:-c12_200 c12_300 c12_430 e1 e5 e10 e50 p100 p150 p200 p250}}"
THREADS=20   # GPU/CPU always fixed at 20 (env override disabled)
export G4TRACE_DIR=OFF   # g4trace disk logging off (if left on, CPU is tens to hundreds of times slower; result is bit-identical)

[ -x "$TOPAS" ] || { echo "ERROR: TOPAS binary not found: $TOPAS"; exit 1; }
mkdir -p results
cd configs

echo "=== 07 CPU Cherenkov z-fluence  (cases: $CASES, T=$THREADS) ==="
printf "%-10s %16s\n" "case" "Execution(s)"
for c in $CASES; do
  cfg="cherenkov_${c}_xz_cpu.txt"
  [ -f "$cfg" ] || { printf "%-10s %16s\n" "$c" "[no config]"; continue; }
  tmp=".tmp_${c}_cpu.txt"
  sed "s|^i:Ts/NumberOfThreads = .*|i:Ts/NumberOfThreads = ${THREADS}|" "$cfg" > "$tmp"
  G4TRACE_DIR=OFF "$TOPAS" "$tmp" > "../results/run_${c}_cpu.log" 2>&1
  t=$(grep "Execution:" "../results/run_${c}_cpu.log" | grep -oE "Real=[0-9.]+s" | head -1 | sed 's/Real=//;s/s//')
  printf "%-10s %16s\n" "$c" "${t:-FAILED}"
  rm -f "$tmp"
done
echo "CSV: benchmark/07_cherenkov_water/results/<case>_xz_CPU.csv"
echo "figure (after running both gpu+cpu): python3 plot.py"
