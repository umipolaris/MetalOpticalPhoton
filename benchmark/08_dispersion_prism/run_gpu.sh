#!/bin/bash
# ============================================================
# run_gpu.sh — 08 dispersion prism (BK7/SF1 n(λ) wavelength dispersion), GPU(BEAM kernel) only
#   Automatically runs 2 tests (profile, visible_xze).
#   TESTS default = "profile xze". Partially specify via first argument or env.
#   e.g.: ./run_gpu.sh "xze"     or     TESTS="xze" ./run_gpu.sh
#   photons (BEAM kernel photon count): PHOTONS=1.0e10 ./run_gpu.sh "xze"   (default 5M; injected via u:BeamPhotons, any N)
#   thread: GPU is BEAM kernel (kernel-side photon generation) so thread is irrelevant — trigger fixed at 1.
#           Can be overridden with THREADS=N but it has no effect on GPU workload.
#   override: TOPAS=/path/topas-gpu ./run_gpu.sh
#   CSV → results/disp_{prof_GPU_bk,xze_GPU}.csv. After both (gpu+cpu) have run, make the figure with:
#         python3 plot.py
# ============================================================
cd "$(dirname "$0")"

. ../common/topas_env.sh
TESTS="${1:-${TESTS:-profile xze}}"
THREADS="${THREADS:-1}"
PHOTONS="${PHOTONS:-5000000}"   # BEAM kernel photon count (injected via u:BeamPhotons; 5M·1e9·1e10·1e11 etc.)
export G4TRACE_DIR=OFF

[ -x "$TOPAS" ] || { echo "ERROR: TOPAS binary not found: $TOPAS"; exit 1; }
mkdir -p results
cd configs

echo "=== 08 GPU dispersion prism (BEAM kernel)  (tests: $TESTS, photons=$PHOTONS, T=$THREADS) ==="
printf "%-10s %16s\n" "test" "Execution(s)"
for t in $TESTS; do
  case "$t" in
    profile) cfg="dispersion_profile_gpu_bk.txt" ;;
    xze)     cfg="dispersion_visible_xze_gpu_bk.txt" ;;
    *) printf "%-10s %16s\n" "$t" "[unknown test]"; continue ;;
  esac
  [ -f "$cfg" ] || { printf "%-10s %16s\n" "$t" "[no config]"; continue; }
  tmp=".tmp_${t}_gpu.txt"
  sed -e "s|^i:Ts/NumberOfThreads = .*|i:Ts/NumberOfThreads = ${THREADS}|" \
      -e "s|^i:Ph/Default/GPUOptical/BeamPhotons = .*|u:Ph/Default/GPUOptical/BeamPhotons = ${PHOTONS}|" \
      "$cfg" > "$tmp"
  G4TRACE_DIR=OFF "$TOPAS" "$tmp" > "../results/run_${t}_gpu.log" 2>&1
  exec_t=$(grep "Execution:" "../results/run_${t}_gpu.log" | grep -oE "Real=[0-9.]+s" | head -1 | sed 's/Real=//;s/s//')
  printf "%-10s %16s\n" "$t" "${exec_t:-FAILED}"
  rm -f "$tmp"
done
echo "CSV: benchmark/08_dispersion_prism/results/disp_{prof_GPU_bk,xze_GPU}.csv"
echo "figure (after running both gpu+cpu): python3 plot.py"
