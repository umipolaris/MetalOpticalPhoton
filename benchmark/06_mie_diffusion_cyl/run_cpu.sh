#!/bin/bash
# ============================================================
# 06_mie_diffusion_cyl — paper figure (weak fog mfp=30) CPU reference
#   barrel φ-z : spread_m30_cpu.txt (g4optical 100M) → results/spread_m30_barrel_CPU.csv
#
# Usage:  ./run_cpu.sh                 # default 20 threads
#         THREADS=30 ./run_cpu.sh      # adjust thread count
#   (CPU profile reference for panels (b)(c) of the figure. GPU is run_gpu.sh.)
# ============================================================
set -e
cd "$(dirname "$0")"

. ../common/topas_env.sh
[ -x "$TOPAS" ] || { echo "ERROR: topas-gpu not found at $TOPAS"; exit 1; }
THREADS=${THREADS:-20}

mkdir -p results
cd configs

TMP=_run_cpu_tmp.txt
sed "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $THREADS|" spread_m30_cpu.txt > $TMP
echo "[$(date +%T)] CPU barrel φ-z (spread_m30_cpu.txt, 100M, threads=$THREADS) ..."
G4TRACE_DIR=OFF "$TOPAS" $TMP > ../results/spread_m30_cpu.log 2>&1
rm -f $TMP
grep "Execution:" ../results/spread_m30_cpu.log | grep -oE "Real=[0-9.]+s" | head -1
echo "→ results/spread_m30_barrel_CPU.csv  (figure: after generating both GPU and CPU, 'python3 plot.py')"
