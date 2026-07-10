#!/bin/bash
# ============================================================
# 05_torus — CPU 10B (TOPAS Beam, Geant4 reference)
#   TOPAS rejects NumberOfHistoriesInRun > 10^9 per source (hard cap; above that there is also an int32 limit).
#   So CPU 10B = **seed 1..10 × 1e9 = 10B** are run separately and the fluence CSVs are summed (statistically equivalent).
#   → results/torus_diffuse_10B_CPU_fresh.csv  (element-wise sum of 10 per-seed CSVs)
#
# Usage:   ./run_cpu.sh                      # default 30 threads, seed 1-10 (1e9 each) = 10B
#          THREADS=20 ./run_cpu.sh           # adjust thread count
#          SEEDS="1 2 3" ./run_cpu.sh        # seed subset (= 3e9)
#   Ts/MaxStepNumber=1000 (same as GPU) — kills torus_diffuse air-gap TIR trapped photons early, result unchanged.
# ============================================================
set -e
cd "$(dirname "$0")"

. ../common/topas_env.sh
[ -x "$TOPAS" ] || { echo "ERROR: topas-gpu not found at $TOPAS"; exit 1; }
THREADS=${THREADS:-30}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}

mkdir -p results
ln -sf ../../common/simple_biconvex_lens_new.stl configs/simple_biconvex_lens_new.stl
cd configs

CSVS=()
for s in $SEEDS; do
    TMP=_run_cpu_tmp.txt
    sed -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $THREADS|" \
        -e "s|^i:Ts/Seed.*|i:Ts/Seed = $s|" \
        -e "s|torus_diffuse_10B_CPU_fresh\"|torus_diffuse_10B_CPU_fresh_s$s\"|" \
        torus_diffuse_10B_cpu_fresh.txt > $TMP
    echo "[$(date +%T)] CPU 1e9  seed=$s  (threads=$THREADS) ..."
    G4TRACE_DIR=OFF "$TOPAS" $TMP > ../results/torus_diffuse_10B_cpu_s$s.log 2>&1
    rm -f $TMP
    echo "    $(grep 'Execution:' ../results/torus_diffuse_10B_cpu_s$s.log | grep -oE 'Real=[0-9.]+s' | head -1)"
    CSVS+=("../results/torus_diffuse_10B_CPU_fresh_s$s.csv")
done

echo "[$(date +%T)] summing ${#CSVS[@]} seed CSVs → torus_diffuse_10B_CPU_fresh.csv"
python3 - "${CSVS[@]}" <<'PY'
import sys
files = sys.argv[1:]
header, acc, order = [], {}, []
for fi, f in enumerate(files):
    with open(f) as fh:
        for line in fh:
            if line.startswith('#'):
                if fi == 0: header.append(line.rstrip('\n'))
                continue
            p = line.rstrip('\n').split(',')
            if len(p) < 4: continue
            try:
                ix, iy, iz, v = int(p[0]), int(p[1]), int(p[2]), float(p[3])
            except ValueError:
                continue
            key = (ix, iy, iz)
            if key not in acc:
                acc[key] = 0.0; order.append(key)
            acc[key] += v
with open('../results/torus_diffuse_10B_CPU_fresh.csv', 'w') as out:
    for h in header: out.write(h + '\n')
    for k in order:
        out.write(f"{k[0]}, {k[1]}, {k[2]}, {acc[k]:.6e}\n")
print(f"  summed {len(files)} seeds, {len(order)} bins, total = {sum(acc.values()):.4e}")
PY
echo "→ results/torus_diffuse_10B_CPU_fresh.csv  (figure: after generating both GPU and CPU CSVs, run 'python3 plot.py')"
