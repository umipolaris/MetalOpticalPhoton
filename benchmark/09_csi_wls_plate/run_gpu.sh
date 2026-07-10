#!/bin/bash
# ============================================================
# run_gpu.sh — 09 CsI(Tl) + WLS plate, GPU (proton → scint/WLS genstep path)
#   runs csi_wls_gpu.txt → results/csi_wls_GPU.csv + csi_wls_dose.csv
#   PHOTONS = proton count (NumberOfHistoriesInRun, default 10000; genstep scintillation, not the beam kernel).
#   THREADS = default 1.
#   e.g.: PHOTONS=100000 ./run_gpu.sh    THREADS=8 ./run_gpu.sh
#   override: TOPAS=/path/topas-gpu ./run_gpu.sh
#   plot (after running both gpu+cpu): python3 plot.py
# ============================================================
cd "$(dirname "$0")"

. ../common/topas_env.sh
PHOTONS="${PHOTONS:-10000}"   # proton count (genstep → GPU scint/WLS photon generation)
THREADS="${THREADS:-1}"
export G4TRACE_DIR=OFF

[ -x "$TOPAS" ] || { echo "ERROR: TOPAS binary not found: $TOPAS"; exit 1; }
mkdir -p results
cd configs

echo "=== 09 GPU CsI(Tl)+WLS plate  (protons=$PHOTONS, T=$THREADS) ==="
tmp=".tmp_gpu.txt"
sed -e "s|^i:Ts/NumberOfThreads = .*|i:Ts/NumberOfThreads = ${THREADS}|" \
    -e "s|^i:So/Beam/NumberOfHistoriesInRun = .*|i:So/Beam/NumberOfHistoriesInRun = ${PHOTONS}|" \
    csi_wls_gpu.txt > "$tmp"
G4TRACE_DIR=OFF "$TOPAS" "$tmp" > "../results/run_gpu.log" 2>&1
t=$(grep "Execution:" "../results/run_gpu.log" | grep -oE "Real=[0-9.]+s" | head -1 | sed 's/Real=//;s/s//')
echo "  Execution Real = ${t:-FAILED} s"
rm -f "$tmp"
echo "CSV: results/csi_wls_GPU.csv + csi_wls_dose.csv"
echo "plot (after running both gpu+cpu): python3 plot.py"
