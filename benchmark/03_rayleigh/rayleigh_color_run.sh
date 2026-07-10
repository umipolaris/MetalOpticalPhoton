#!/bin/bash
# ============================================================
# rayleigh_color_run.sh — 03 B2 Rayleigh "blue sky / red sunset" colour demo.
#   White continuous beam (1.65-3.10 eV) through a Rayleigh box (RAYLEIGH length ~ lambda^4);
#   blue scatters into a side halo, red transmits -> energy-resolved x-z colour map.
#   Runs the committed configs  configs/rayleigh_color_{gpu,cpu}.txt  ->
#     results/rayleigh_color_{GPU,CPU}.csv   (then: python3 plot_color.py).
#
#   ENGINES (1st arg or env): "gpu cpu" (default), or just "gpu" / "cpu".
#     GPU = BEAM kernel (kernel-side photon gen, ~560x faster than TOPAS-Beam genstep,
#           physics-identical). CPU = Geant4 g4optical.
#   HIST=N         photon count (default 1000000000 = paper B2; HIST=5000000 for a quick demo). GPU -> u:BeamPhotons, CPU -> NumberOfHistoriesInRun.
#   THREADS_CPU=N  CPU threads (default 20; GPU is single-trigger, thread-independent).
#   TOPAS=/path/topas-gpu   patched binary (required; unpatched over-counts fluence ~4x).
#   e.g.:  ./rayleigh_color_run.sh gpu            # GPU only
#          HIST=5000000 ./rayleigh_color_run.sh   # quick demo (5M)
# ============================================================
cd "$(dirname "$0")"

. ../common/topas_env.sh
ENGINES="${1:-${ENGINES:-gpu cpu}}"
HIST="${HIST:-1000000000}"
THREADS_CPU="${THREADS_CPU:-20}"
export G4TRACE_DIR=OFF   # g4trace disk logging off (results bit-identical; on = much slower CPU)

[ -x "$TOPAS" ] || { echo "ERROR: TOPAS binary not found: $TOPAS"; exit 1; }
mkdir -p results
cd configs

echo "=== 03 B2 Rayleigh colour demo  (engines: $ENGINES, HIST=$HIST) ==="
for eng in $ENGINES; do
  case "$eng" in
    gpu)
      tmp=".tmp_gpu.txt"
      sed -e "s|^u:Ph/Default/GPUOptical/BeamPhotons=.*|u:Ph/Default/GPUOptical/BeamPhotons=${HIST}|" \
          rayleigh_color_gpu.txt > "$tmp"
      ;;
    cpu)
      tmp=".tmp_cpu.txt"
      sed -e "s|^i:Ts/NumberOfThreads=.*|i:Ts/NumberOfThreads=${THREADS_CPU}|" \
          -e "s|^i:So/Beam/NumberOfHistoriesInRun=.*|i:So/Beam/NumberOfHistoriesInRun=${HIST}|" \
          rayleigh_color_cpu.txt > "$tmp"
      ;;
    *) echo "  [unknown engine: $eng]"; continue ;;
  esac
  G4TRACE_DIR=OFF "$TOPAS" "$tmp" > "../results/run_${eng}.log" 2>&1 \
    && exec_t=$(grep "Execution:" "../results/run_${eng}.log" | grep -oE "Real=[0-9.]+s" | head -1) \
    || exec_t="FAILED"
  printf "  %-3s %s\n" "$eng" "${exec_t:-done}"
  rm -f "$tmp"
done
echo "done -> results/rayleigh_color_{GPU,CPU}.csv ; then: python3 plot_color.py"
