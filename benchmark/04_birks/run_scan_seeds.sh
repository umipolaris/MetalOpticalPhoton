#!/bin/bash
# ============================================================
# run_scan_seeds.sh — 04 Birks Q(T) energy scan, multi-seed (Fig 5d error bars).
#   Per proton energy T and seed s, Birks on/off integrated fluence →
#   results/U5seed_{GPU_,}T{T}_{on,off}_s{s}_fluence.csv  (plot.py panel d reads these).
#   ENG=gpu|cpu  SEEDS="1 2 3 4 5"  ENERGIES="2 5 10 20 30 50 75 100"  PHOTONS=2500
#   GPU ~6 min, CPU ~1.8 h (5 seeds x 8 energies x on/off).
# ============================================================
cd "$(dirname "$0")"
. ../common/topas_env.sh
ENG="${ENG:-gpu}"
ENERGIES="${ENERGIES:-2 5 10 20 30 50 75 100}"
SEEDS="${SEEDS:-1 2 3 4 5}"
NHIST="${PHOTONS:-2500}"
export G4TRACE_DIR=OFF
[ -x "$TOPAS" ] || { echo "ERROR: TOPAS not found: $TOPAS"; exit 1; }
mkdir -p results; cd configs
inc="U5_short_${ENG}.txt"; pfx=""; [ "$ENG" = "gpu" ] && pfx="GPU_"
echo "=== 04 ${ENG} Birks Q(T) multi-seed (T:$ENERGIES; seeds:$SEEDS; NHist=$NHIST) ==="
for T in $ENERGIES; do for s in $SEEDS; do for mode in on off; do
  kB=0.126; [ "$mode" = "off" ] && kB=0.0
  tmp=".tmp_seed_${ENG}_${T}_${s}_${mode}.txt"
  cat > "$tmp" <<EOF
includeFile = $inc
i:Ts/Seed = $s
d:So/Beam/BeamEnergy = $T MeV
u:Ma/Buapfcfm/BirksConstant = $kB
i:So/Beam/NumberOfHistoriesInRun = $NHIST
i:Sc/U5short_Fluence/XBins = 1
i:Sc/U5short_Fluence/YBins = 1
i:Sc/U5short_Fluence/ZBins = 1
s:Sc/U5short_Fluence/OutputFile = "../results/U5seed_${pfx}T${T}_${mode}_s${s}_fluence"
i:Sc/U5short_Edep/XBins = 1
i:Sc/U5short_Edep/YBins = 1
i:Sc/U5short_Edep/ZBins = 1
s:Sc/U5short_Edep/OutputFile = "../results/.U5seed_ed_${ENG}_${T}_${mode}_s${s}"
EOF
  if [ "$ENG" = "cpu" ]; then cat >> "$tmp" <<EOF
i:Sc/U5short_Fluence_Z/XBins = 1
i:Sc/U5short_Fluence_Z/YBins = 1
i:Sc/U5short_Fluence_Z/ZBins = 1
s:Sc/U5short_Fluence_Z/OutputFile = "../results/.U5seed_z_${T}_${mode}_s${s}"
i:Sc/U5short_Fluence_X/XBins = 1
i:Sc/U5short_Fluence_X/YBins = 1
i:Sc/U5short_Fluence_X/ZBins = 1
s:Sc/U5short_Fluence_X/OutputFile = "../results/.U5seed_x_${T}_${mode}_s${s}"
EOF
  fi
  "$TOPAS" "$tmp" > /dev/null 2>&1 && st=ok || st=FAIL
  printf "  %s T=%-3s s%s %-3s %s\n" "$ENG" "$T" "$s" "$mode" "$st"
  rm -f "$tmp"
done; done; done
echo "→ results/U5seed_${pfx}T*_{on,off}_s*_fluence.csv"
