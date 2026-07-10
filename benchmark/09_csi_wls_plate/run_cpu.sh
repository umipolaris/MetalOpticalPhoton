#!/bin/bash
# ============================================================
# run_cpu.sh — 09 CsI(Tl) + WLS plate, CPU(Geant4 g4optical)
#   runs csi_wls_cpu.txt → results/csi_wls_CPU.csv + csi_wls_dose_cpu.csv
#   PHOTONS = proton count (NumberOfHistoriesInRun, default 10000). THREADS = default 20.
#   ⚠ CPU directly tracks scintillation photons and is very slow (10000 proton ≈ ~56 min).
#   e.g.: PHOTONS=1000 ./run_cpu.sh    THREADS=30 ./run_cpu.sh
#   override: TOPAS=/path/topas-gpu ./run_cpu.sh   (patched topas-gpu required)
#   plot (after running both gpu+cpu): python3 plot.py
# ============================================================
cd "$(dirname "$0")"

. ../common/topas_env.sh
PHOTONS="${PHOTONS:-10000}"   # proton count (integer)
THREADS="${THREADS:-20}"
export G4TRACE_DIR=OFF   # g4trace disk logging off (if left on, CPU is tens to hundreds of times slower; results bit-identical)

[ -x "$TOPAS" ] || { echo "ERROR: TOPAS binary not found: $TOPAS"; exit 1; }
mkdir -p results
cd configs

echo "=== 09 CPU CsI(Tl)+WLS plate  (protons=$PHOTONS, T=$THREADS) ==="
tmp=".tmp_cpu.txt"
sed -e "s|^i:Ts/NumberOfThreads = .*|i:Ts/NumberOfThreads = ${THREADS}|" \
    -e "s|^i:So/Beam/NumberOfHistoriesInRun = .*|i:So/Beam/NumberOfHistoriesInRun = ${PHOTONS}|" \
    csi_wls_cpu.txt > "$tmp"
G4TRACE_DIR=OFF "$TOPAS" "$tmp" > "../results/run_cpu.log" 2>&1
t=$(grep "Execution:" "../results/run_cpu.log" | grep -oE "Real=[0-9.]+s" | head -1 | sed 's/Real=//;s/s//')
echo "  Execution Real = ${t:-FAILED} s"
rm -f "$tmp"
echo "CSV: results/csi_wls_CPU.csv + csi_wls_dose_cpu.csv"
echo "plot (after running both gpu+cpu): python3 plot.py"
