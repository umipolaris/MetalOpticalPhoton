#!/bin/bash
# ============================================================
# run_cpu_hires.sh — sim_light CPU (Geant4), high-resolution 816×624×50 (25.4M voxel)
#   Runs 1K/10K/100K automatically, measures Execution: Real only (excludes init/finalize).
#   CPU default T=30. Uses the patched topas-gpu binary (unpatched topas forbidden).
#   On CPU, runtime is G4 transport-bound, so binning has ~noise time impact (effectively identical to standard).
#   SIZES default 1K. Specify via argument or env: ./run_cpu_hires.sh "10K 100K"  or  SIZES="10K 100K" ./run_cpu_hires.sh
#   override example: TOPAS=/path/topas-gpu THREADS=20 ./run_cpu_hires.sh "100K"
# ============================================================
cd "$(dirname "$0")"

. ../common/topas_env.sh
THREADS="${THREADS:-30}"
SIZES="${1:-${SIZES:-1K}}"

[ -x "$TOPAS" ] || { echo "ERROR: TOPAS binary not found: $TOPAS"; exit 1; }
mkdir -p results
ln -sf ../../common/simple_biconvex_lens_new.stl configs/simple_biconvex_lens_new.stl

echo "=== CPU high-resolution 816×624×50  (T=$THREADS, Execution Real) ==="
printf "%-6s %16s %16s\n" "N" "Execution(s)" "sum(/mm2)"
cd configs
for tag in $SIZES; do
  src="sim_light_cpu_${tag}.txt"
  [ -f "$src" ] || { echo "  [skip] $src not found"; continue; }
  tmp=".tmp_cpu_hi_${tag}.txt"
  sed -e "s|^i:Ts/NumberOfThreads = .*|i:Ts/NumberOfThreads = ${THREADS}|" \
      -e 's|^i:Sc/SensorPhoton/XBins = .*|i:Sc/SensorPhoton/XBins = 816|' \
      -e 's|^i:Sc/SensorPhoton/YBins = .*|i:Sc/SensorPhoton/YBins = 624|' \
      "$src" > "$tmp"
  G4TRACE_DIR=OFF "$TOPAS" "$tmp" > "../results/cpu_hi_${tag}.log" 2>&1
  t=$(grep "Execution:" "../results/cpu_hi_${tag}.log" | grep -oE "Real=[0-9.]+s" | head -1 | sed 's/Real=//;s/s//')
  s=$(awk -F',' '!/^#/{x+=$4} END{printf "%.4e", x}' ../results/SensorDepth_CPU.csv 2>/dev/null)
  printf "%-6s %16s %16s\n" "$tag" "${t:-FAILED}" "${s:-?}"
  rm -f "$tmp" ../results/SensorDepth_CPU.csv   # clean up high-resolution CSV (~700MB)
done
echo "logs: benchmark/01_speed/results/cpu_hi_*.log"
