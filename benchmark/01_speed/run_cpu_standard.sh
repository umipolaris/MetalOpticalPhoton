#!/bin/bash
# ============================================================
# run_cpu_standard.sh — sim_light CPU (Geant4), standard 1×1×50 (50 bin)
#   Runs 1K/10K/100K automatically, measures Execution: Real only (excludes init/finalize).
#   CPU default T=30. Uses the patched topas-gpu binary (unpatched topas forbidden).
#   SIZES default 1K. Specify via argument or env: ./run_cpu_standard.sh "10K 100K"  or  SIZES="10K 100K" ./run_cpu_standard.sh
#   override example: TOPAS=/path/topas-gpu THREADS=20 ./run_cpu_standard.sh "1K 100K"
# ============================================================
cd "$(dirname "$0")"

. ../common/topas_env.sh
THREADS="${THREADS:-30}"
SIZES="${1:-${SIZES:-1K}}"

[ -x "$TOPAS" ] || { echo "ERROR: TOPAS binary not found: $TOPAS"; exit 1; }
mkdir -p results
ln -sf ../../common/simple_biconvex_lens_new.stl configs/simple_biconvex_lens_new.stl

echo "=== CPU standard 1×1×50  (T=$THREADS, Execution Real) ==="
printf "%-6s %16s %16s\n" "N" "Execution(s)" "sum(/mm2)"
cd configs
for tag in $SIZES; do
  src="sim_light_cpu_${tag}.txt"
  [ -f "$src" ] || { echo "  [skip] $src not found"; continue; }
  tmp=".tmp_cpu_std_${tag}.txt"
  sed "s|^i:Ts/NumberOfThreads = .*|i:Ts/NumberOfThreads = ${THREADS}|" "$src" > "$tmp"
  G4TRACE_DIR=OFF "$TOPAS" "$tmp" > "../results/cpu_std_${tag}.log" 2>&1
  t=$(grep "Execution:" "../results/cpu_std_${tag}.log" | grep -oE "Real=[0-9.]+s" | head -1 | sed 's/Real=//;s/s//')
  s=$(awk -F',' '!/^#/{x+=$4} END{printf "%.4e", x}' ../results/SensorDepth_CPU.csv 2>/dev/null)
  printf "%-6s %16s %16s\n" "$tag" "${t:-FAILED}" "${s:-?}"
  rm -f "$tmp"
done
echo "logs: benchmark/01_speed/results/cpu_std_*.log"
