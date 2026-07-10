#!/bin/bash
# ============================================================
# Benchmark 10 — Curved-surface geometry validation: analytic vs mesh
# ============================================================
# Purpose
#   Verify that the GPU curved-surface intersection (sphere, cylinder, torus)
#   reproduces the CPU Geant4 solid, for BOTH the in-kernel ANALYTIC path
#   (quadratic / torus sphere-tracing+bisection) and the triangulated MESH path (BVH).
#
# Method (why this setup)
#   A pencil opticalphoton beam is fired through the geometric centre of an
#   absorbing solid (absorption length = 10 mm), so the surviving fluence at a
#   downstream detector is F = F0 * exp(-L/Labs), where L is the chord the beam
#   travels inside the solid. Inverting gives a DIRECT, sub-mm measurement of
#   the GPU-computed chord:  L_GPU = -Labs * ln(F_abs / F_transparent).
#   The transparent run (same solid, non-absorbing) supplies F0 per engine,
#   cancelling detector/normalisation factors. The chord is the physically
#   meaningful metric; the raw transmission ratio carries Poisson noise that
#   grows as transmission falls, so we report both but trust the chord.
#
#   Beam orientation is chosen so the beam crosses the CURVED surface along a
#   known chord = 2*R (a diameter):
#     - sphere   : beam +Z through centre  -> chord 2R
#     - cylinder : beam +X (perp. to Z axis) through centre -> chord 2R
#                  (axis must stay Z for the analytic path; we rotate the beam)
#     - torus    : beam +Z through the tube centred at radius RTor=5cm -> chord 2R
#
#   ANALYTIC vs MESH is selected purely at the script level via
#     b:Ge/<Component>/GPUForceMesh="True"   (-> triangulated BVH path)
#   default (absent) = analytic in-kernel intersection.
#
# Output: results/<solid>_<method>_R<r>_<G|C>_s<seed>.csv  (+ transparent _trans)
# Analyse with: python3 analyze.py
# ============================================================
set -e
cd "$(dirname "$0")"
. ../common/topas_env.sh
HIST=${HIST:-2000000}
SEEDS=(11 22 33 44 55)
RADII=(1.0 2.0 3.0)
RESDIR=results
mkdir -p "$RESDIR"

emit_geom(){ # $1 solid $2 R(cm) $3 mat $4 forcemesh(0/1)
  local solid=$1 R=$2 mat=$3 fm=$4
  local fmline=""
  [ "$fm" = "1" ] && fmline='b:Ge/Obj/GPUForceMesh="True"'
  case $solid in
    sphere)
      cat <<EOF
s:Ge/Obj/Parent="World"
s:Ge/Obj/Type="TsSphere"
d:Ge/Obj/RMax=$R cm
s:Ge/Obj/Material="$mat"
$fmline
d:Ge/Det/TransZ=12 cm
d:Ge/Det/HLX=2 cm
d:Ge/Det/HLY=2 cm
d:Ge/Det/HLZ=0.5 cm
d:Ge/OptSrcGroup/TransZ=-20 cm
EOF
      ;;
    cylinder)
      cat <<EOF
s:Ge/Obj/Parent="World"
s:Ge/Obj/Type="TsCylinder"
d:Ge/Obj/RMax=$R cm
d:Ge/Obj/HL=8 cm
s:Ge/Obj/Material="$mat"
$fmline
d:Ge/Det/TransX=12 cm
d:Ge/Det/HLX=0.5 cm
d:Ge/Det/HLY=2 cm
d:Ge/Det/HLZ=2 cm
d:Ge/OptSrcGroup/TransX=-20 cm
d:Ge/OptSrcGroup/RotY=-90 deg
EOF
      ;;
    torus)
      cat <<EOF
s:Ge/Obj/Parent="World"
s:Ge/Obj/Type="G4Torus"
d:Ge/Obj/RMin=0.0 cm
d:Ge/Obj/RMax=$R cm
d:Ge/Obj/RTor=5.0 cm
d:Ge/Obj/SPhi=0 deg
d:Ge/Obj/DPhi=360 deg
s:Ge/Obj/Material="$mat"
$fmline
d:Ge/Det/TransX=5 cm
d:Ge/Det/TransZ=12 cm
d:Ge/Det/HLX=2 cm
d:Ge/Det/HLY=2 cm
d:Ge/Det/HLZ=0.5 cm
d:Ge/OptSrcGroup/TransX=5 cm
d:Ge/OptSrcGroup/TransZ=-20 cm
EOF
      ;;
  esac
}

run_one(){ # $1 solid $2 R $3 mat(Air/AbsMat) $4 fm(0/1) $5 engine(G/C) $6 outtag $7 seed(opt,default 42)
  local solid=$1 R=$2 mat=$3 fm=$4 eng=$5 tag=$6 seed=${7:-42}
  local cfg="$RESDIR/.cfg_${tag}.txt"
  cat > "$cfg" <<EOF
s:Ph/ListName="Optical"
s:Ph/Optical/Type="Geant4_Modular"
i:Ph/Default/GPUOptical/GPUCores=80
i:Ph/Default/GPUOptical/LogLevel=0
dv:Ma/Air/RIndex/Energies=2 1.5 5.0 eV
uv:Ma/Air/RIndex/Values=2 1.0 1.0
dv:Ma/Air/AbsLength/Energies=2 1.5 5.0 eV
dv:Ma/Air/AbsLength/Values=2 1e6 1e6 cm
sv:Ma/AbsMat/Components=1 "Hydrogen"
uv:Ma/AbsMat/Fractions=1 1.0
d:Ma/AbsMat/Density=0.001 g/cm3
b:Ma/AbsMat/EnableOpticalProperties="True"
dv:Ma/AbsMat/RIndex/Energies=2 1.5 5.0 eV
uv:Ma/AbsMat/RIndex/Values=2 1.0 1.0
dv:Ma/AbsMat/AbsLength/Energies=2 1.5 5.0 eV
dv:Ma/AbsMat/AbsLength/Values=2 1.0 1.0 cm
s:Ge/World/Material="Air"
s:Ge/World/Type="TsBox"
d:Ge/World/HLX=30 cm
d:Ge/World/HLY=30 cm
d:Ge/World/HLZ=30 cm
s:Ge/Det/Parent="World"
s:Ge/Det/Type="TsBox"
s:Ge/Det/Material="Air"
s:So/OptSrc/Type="Beam"
s:So/OptSrc/Component="OptSrcGroup"
s:So/OptSrc/BeamParticle="opticalphoton"
s:So/OptSrc/BeamPositionDistribution="Flat"
s:So/OptSrc/BeamPositionCutoffShape="Rectangle"
d:So/OptSrc/BeamPositionCutoffX=0.5 mm
d:So/OptSrc/BeamPositionCutoffY=0.5 mm
d:So/OptSrc/BeamEnergy=2.5 eV
u:So/OptSrc/BeamEnergySpread=0.0
s:So/OptSrc/BeamAngularDistribution="None"
i:So/OptSrc/NumberOfHistoriesInRun=$HIST
s:Ge/OptSrcGroup/Parent="World"
s:Ge/OptSrcGroup/Type="Group"
i:Sc/Det/XBins=1
i:Sc/Det/YBins=1
i:Sc/Det/ZBins=1
s:Sc/Det/Component="Det"
sv:Sc/Det/OnlyIncludeParticlesNamed=1 "opticalphoton"
s:Sc/Det/OutputType="csv"
s:Sc/Det/IfOutputFileAlreadyExists="Overwrite"
i:Ts/Seed=$seed
b:Gr/Enable="False"
b:Ts/UseQt="False"
b:Ts/QuitIfManyHistoriesSeemAnomalous="False"
$(emit_geom $solid $R $mat $fm)
EOF
  if [ "$eng" = "G" ]; then
    printf '%s\n' 'sv:Ph/Optical/Modules=3 "g4em-standard_opt4" "g4optical" "gpuoptical"' >> "$cfg"
    printf '%s\n' 'i:Ts/NumberOfThreads=1'                                                >> "$cfg"
    printf '%s\n' 's:Sc/Det/Quantity="GPUOpticalPhotonFluence"'                           >> "$cfg"
  else
    printf '%s\n' 'sv:Ph/Optical/Modules=2 "g4em-standard_opt4" "g4optical"'              >> "$cfg"
    printf '%s\n' 'i:Ts/NumberOfThreads=20'                                              >> "$cfg"
    printf '%s\n' 's:Sc/Det/Quantity="Fluence"'                                          >> "$cfg"
  fi
  printf '%s\n' "s:Sc/Det/OutputFile=\"$RESDIR/$tag\"" >> "$cfg"
  G4TRACE_DIR=OFF "$TOPAS" "$cfg" >/dev/null 2>&1 || true
  rm -f "$cfg"
}

for solid in sphere cylinder torus; do
  for R in "${RADII[@]}"; do
    rtag="R${R%%.*}"
    # transparent baselines (one per engine; method-independent on CPU,
    #   measured per method on GPU for completeness)
    run_one $solid $R Air 0 G "${solid}_analytic_${rtag}_G_trans"
    run_one $solid $R Air 1 G "${solid}_mesh_${rtag}_G_trans"
    run_one $solid $R Air 0 C "${solid}_${rtag}_C_trans"
    for s in "${SEEDS[@]}"; do
      run_one $solid $R AbsMat 0 G "${solid}_analytic_${rtag}_G_s${s}" $s
      run_one $solid $R AbsMat 1 G "${solid}_mesh_${rtag}_G_s${s}"     $s
      run_one $solid $R AbsMat 0 C "${solid}_${rtag}_C_s${s}"          $s
    done
    # NOTE: run_one signature is (solid R mat fm eng tag [seed]); transparent
    #       calls above omit seed (default 42).
    printf '%s\n' "done: $solid $rtag"
  done
done
printf '%s\n' "ALL DONE — run: python3 analyze.py"
