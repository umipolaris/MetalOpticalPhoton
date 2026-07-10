#!/usr/bin/env bash
# ============================================================
# GPU optical engine — installation smoke test (benchmarks 01–10, GPU + CPU)
#
# Runs every benchmark on drastically reduced photon/event counts and checks:
#   [run]  the TOPAS run exits 0
#   [log]  no crash/error markers (segfault, FATAL, dyld, …)
#   [gpu]  GPU rows: engine engaged ("[TsGPU]" in log);
#          CPU rows: engine NOT engaged (module wiring check)
#   [out]  expected output CSV(s) exist with a nonzero sum
#
# This validates the INSTALLATION (build, patches, Metal pipeline, scorers) —
# NOT physics statistics. For real numbers run each benchmark's own scripts.
#
#   ./run_smoke_test.sh                     # all 10, GPU+CPU (~3–5 min)
#   DEVICES=gpu ./run_smoke_test.sh         # GPU only (~1 min)
#   ONLY="03 07" ./run_smoke_test.sh        # subset of benchmarks
#   SCALE=10 ./run_smoke_test.sh            # 10x the (tiny) default event counts
#
# Report: smoke_logs/smoke_report.html  (+ console summary)
# ============================================================
set -u
cd "$(dirname "$0")"
. common/topas_env.sh   # resolves $TOPAS (env > PATH > default)

# ---------- knobs ----------
ONLY="${ONLY:-01 02 03 04 05 06 07 08 09 10}"
DEVICES="${DEVICES:-gpu cpu}"
SCALE="${SCALE:-1}"                 # global multiplier on every event count below
CPU_THREADS="${CPU_THREADS:-20}"

# per-benchmark base event counts (× SCALE)
B01=100      # protons          (sim_light)
B02=100000   # photons ×1 angle (Fresnel U1v5)
B03=100000   # photons          (Rayleigh)
B04=100      # protons          (Birks)
B05=1000000  # photons          (torus BEAM)
B06=1000000  # photons          (Mie)
B07=200      # electrons        (Cherenkov)
B08=100000   # photons          (prism)
B09=100      # protons          (CsI+WLS)
B10=20000    # photons ×6 runs  (analytic vs mesh)
n() { awk -v b="$1" -v s="$SCALE" 'BEGIN{printf "%d", b*s}'; }

SMOKE="$PWD/smoke_logs"; mkdir -p "$SMOKE"
ROWS="$SMOKE/.rows.tsv";     : > "$ROWS"
PREROWS="$SMOKE/.pre.tsv";   : > "$PREROWS"
FAILDET="$SMOKE/.fail.html"; : > "$FAILDET"
REPORT="$SMOKE/smoke_report.html"
NPASS=0; NFAIL=0
T_ALL0=$(date +%s)

# ---------- helpers ----------
esc() { sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g'; }
csv_sum() { awk -F',' '!/^#/{for(i=1;i<=NF;i++)s+=$i+0}END{printf "%.10g", s+0}' "$1" 2>/dev/null; }

pre_check() { # label ok|fail detail
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$PREROWS"
  [ "$2" = ok ] && printf '  ✅ %-26s %s\n' "$1" "$3" || printf '  ❌ %-26s %s\n' "$1" "$3"
}

# finish_case NUM NAME DEV EXERCISES EVENTS LOG RC MARKER(require|auto|skip|absent) OUT...
finish_case() {
  local num="$1" name="$2" dev="$3" ex="$4" ev="$5" log="$6" rc="$7" mk="$8"; shift 8
  local c_run=ok c_log=ok c_gpu=ok c_out=ok outdesc="" f s dt
  dt=$(( $(date +%s) - T_CASE0 ))
  [ "$rc" -eq 0 ] || c_run=fail
  grep -qE 'Segmentation fault|FATAL|Abort trap|dyld\[|Library not loaded|command not found' "$log" && c_log=fail
  case "$mk" in
    require) grep -q '\[TsGPU\]' "$log" || c_gpu=fail ;;
    absent)  grep -q '\[TsGPU\]' "$log" && c_gpu=fail || c_gpu=ok ;;
    auto)    grep -q '\[TsGPU\]' "$log" && c_gpu=ok || c_gpu=na ;;
    skip)    c_gpu=na ;;
  esac
  for f in "$@"; do
    if [ -s "$f" ]; then
      s=$(csv_sum "$f")
      if awk -v s="$s" 'BEGIN{exit !(s>0)}'; then outdesc+="$(basename "$f") (sum=$s); "
      else c_out=fail; outdesc+="$(basename "$f") (sum=0 ⚠); "; fi
    else c_out=fail; outdesc+="$(basename "$f") (MISSING ⚠); "; fi
  done
  local status=PASS
  { [ $c_run = fail ] || [ $c_log = fail ] || [ $c_gpu = fail ] || [ $c_out = fail ]; } && status=FAIL
  [ $status = PASS ] && NPASS=$((NPASS+1)) || NFAIL=$((NFAIL+1))
  printf '%s\t%s\t%s\t%s\t%s\t%ss\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$num" "$name" "$dev" "$ex" "$ev" "$dt" "$c_run" "$c_log" "$c_gpu" "$c_out" "$status" "$outdesc" >> "$ROWS"
  printf '  %s %s [%s]  (%ss)  run:%s log:%s gpu:%s out:%s\n' \
    "$([ $status = PASS ] && echo '✅' || echo '❌')" "$num $name" "$dev" "$dt" "$c_run" "$c_log" "$c_gpu" "$c_out"
  if [ $status = FAIL ]; then
    { printf '<details open><summary><b>%s %s [%s]</b> — log tail (%s)</summary><pre>' "$num" "$name" "$dev" "$(basename "$log")"
      tail -40 "$log" | esc; printf '</pre></details>\n'; } >> "$FAILDET"
  fi
}

run_cfg() { ( cd "$1" && G4TRACE_DIR=OFF "$TOPAS" "$2" ) > "$3" 2>&1; }

# sed-inject helper: make_cfg SRC DST SEDEXPR...
make_cfg() { local src="$1" dst="$2"; shift 2; sed "$@" "$src" > "$dst"; }

# ---------- pre-flight ----------
echo "── Pre-flight ──  (DEVICES='$DEVICES'  SCALE=$SCALE)"
[ "$(uname -m)" = arm64 ] && pre_check "Apple Silicon (arm64)" ok "$(sysctl -n machdep.cpu.brand_string 2>/dev/null)" \
                          || pre_check "Apple Silicon (arm64)" fail "uname -m = $(uname -m)"
if [ -x "$TOPAS" ] || command -v "$TOPAS" >/dev/null 2>&1; then
  pre_check "topas-gpu binary" ok "$TOPAS"
else
  pre_check "topas-gpu binary" fail "not found: $TOPAS"; echo "ABORT: no binary"; exit 2
fi
TBIN="$(command -v "$TOPAS" || echo "$TOPAS")"
LIBDIR="$(cd "$(dirname "$TBIN")/.." 2>/dev/null && pwd)/lib"
[ -f "$LIBDIR/libMetalOpticalPhoton.dylib" ] && pre_check "engine dylib" ok "$LIBDIR/libMetalOpticalPhoton.dylib" \
                                             || pre_check "engine dylib" fail "missing in $LIBDIR"
[ -f "$LIBDIR/default.metallib" ] && pre_check "Metal shader library" ok "$LIBDIR/default.metallib" \
                                  || pre_check "Metal shader library" fail "missing in $LIBDIR"

# ---------- benchmarks ----------
for NB in $ONLY; do
for DEV in $DEVICES; do
T_CASE0=$(date +%s)
LOG="$SMOKE/${NB}_${DEV}.log"
case "$NB-$DEV" in

01-gpu|01-cpu)
  d=01_speed/configs
  ln -sfn ../../common/simple_biconvex_lens_new.stl "$d/simple_biconvex_lens_new.stl"
  if [ $DEV = gpu ]; then src=sim_light_gpu_1K.txt; th=1; mk=require; out=01_speed/results/SensorDepth_GPU.csv
  else src=sim_light_cpu_1K.txt; th=$CPU_THREADS; mk=absent; out=01_speed/results/SensorDepth_CPU.csv; fi
  make_cfg "$d/$src" "$d/_smoke.txt" \
    -e "s|^i:So/Example1/NumberOfHistoriesInRun.*|i:So/Example1/NumberOfHistoriesInRun   = $(n $B01)|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $th|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 01_speed "$DEV" "proton→scintillation genstep; STL-lens BVH; DDA scorer" "$(n $B01) p" "$LOG" $rc $mk "$out"
  ;;

02-gpu|02-cpu)
  d=02_reflection_refraction_tir
  sed "s|NumberOfHistoriesInRun = 100000|NumberOfHistoriesInRun = $(n $B02)|" \
    "$d/scripts/build_run_u1v5.py" > "$d/scripts/_smoke_u1v5.py"
  ( cd "$d" && rm -f results/built/U1v5_${DEV}_th30_*_sf1.csv \
    && ANGLES=30 MODE=$DEV THREADS=$([ $DEV = gpu ] && echo 1 || echo $CPU_THREADS) TOPAS="$TOPAS" \
       python3 scripts/_smoke_u1v5.py ) > "$LOG" 2>&1; rc=$?
  rm -f "$d/scripts/_smoke_u1v5.py"
  finish_case "$NB" 02_reflection_refraction_tir "$DEV" "TOPAS-Beam photon hand-off; Fresnel R/T at 30°" "$(n $B02) γ ×1 angle" \
    "$LOG" $rc $([ $DEV = gpu ] && echo auto || echo skip) \
    "$d/results/built/U1v5_${DEV}_th30_R_sf1.csv" "$d/results/built/U1v5_${DEV}_th30_T_sf1.csv"
  ;;

03-gpu)
  d=03_rayleigh/configs
  make_cfg "$d/rayleigh_color_gpu.txt" "$d/_smoke.txt" \
    -e "s|^u:Ph/Default/GPUOptical/BeamPhotons=.*|u:Ph/Default/GPUOptical/BeamPhotons=$(n $B03)|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 03_rayleigh gpu "BEAM kernel; Rayleigh λ⁴; energy-resolved x–z scorer" "$(n $B03) γ" \
    "$LOG" $rc require 03_rayleigh/results/rayleigh_color_GPU.csv
  ;;
03-cpu)
  d=03_rayleigh/configs
  make_cfg "$d/rayleigh_color_cpu.txt" "$d/_smoke.txt" \
    -e "s|^i:So/Beam/NumberOfHistoriesInRun=.*|i:So/Beam/NumberOfHistoriesInRun=$(n $B03)|" \
    -e "s|^i:Ts/NumberOfThreads=.*|i:Ts/NumberOfThreads=$CPU_THREADS|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 03_rayleigh cpu "Rayleigh λ⁴ on g4optical; energy-resolved x–z scorer" "$(n $B03) γ" \
    "$LOG" $rc absent 03_rayleigh/results/rayleigh_color_CPU.csv
  ;;

04-gpu)
  d=04_birks/configs
  make_cfg "$d/U5_short_gpu.txt" "$d/_smoke.txt" \
    -e "s|^i:So/Beam/NumberOfHistoriesInRun.*|i:So/Beam/NumberOfHistoriesInRun   = $(n $B04)|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = 1|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 04_birks gpu "Birks-quenched scintillation genstep; analytic box; edep scorer" "$(n $B04) p" \
    "$LOG" $rc require 04_birks/results/U5short_GPU_fluence.csv 04_birks/results/U5short_GPU_edep.csv
  ;;
04-cpu)
  d=04_birks/configs
  make_cfg "$d/U5_short_cpu.txt" "$d/_smoke.txt" \
    -e "s|^i:So/Beam/NumberOfHistoriesInRun.*|i:So/Beam/NumberOfHistoriesInRun   = $(n $B04)|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $CPU_THREADS|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 04_birks cpu "CPU optical multi-scorer; parallel-world patch path (U5 cube)" "$(n $B04) p" \
    "$LOG" $rc absent \
    04_birks/results/U5short_CPU_fluence.csv 04_birks/results/U5short_CPU_fluence_z.csv \
    04_birks/results/U5short_CPU_fluence_x.csv 04_birks/results/U5short_CPU_edep.csv
  ;;

05-gpu)
  d=05_torus/configs
  ln -sfn ../../common/simple_biconvex_lens_new.stl "$d/simple_biconvex_lens_new.stl"
  make_cfg "$d/torus_diffuse_10B_beamgpu.txt" "$d/_smoke.txt" \
    -e "s|^u:Ph/Default/GPUOptical/BeamPhotons = .*|u:Ph/Default/GPUOptical/BeamPhotons = $(n $B05)|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 05_torus gpu "BEAM kernel; torus SDF analytic; STL lens; camera scorer" "$(n $B05) γ" \
    "$LOG" $rc require 05_torus/results/torus_diffuse_10B_beamgpu_GPU.csv
  ;;
05-cpu)
  d=05_torus/configs
  ln -sfn ../../common/simple_biconvex_lens_new.stl "$d/simple_biconvex_lens_new.stl"
  make_cfg "$d/torus_diffuse_10B_cpu_fresh.txt" "$d/_smoke.txt" \
    -e "s|^i:So/Light/NumberOfHistoriesInRun.*|i:So/Light/NumberOfHistoriesInRun   = $(n $B05)|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $CPU_THREADS|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 05_torus cpu "torus + diffuse mirror + STL lens on g4optical" "$(n $B05) γ" \
    "$LOG" $rc absent 05_torus/results/torus_diffuse_10B_CPU_fresh.csv
  ;;

06-gpu)
  d=06_mie_diffusion_cyl/configs
  make_cfg "$d/spread_m30_both_gpu.txt" "$d/_smoke.txt" \
    -e "s|^i:Ph/Default/GPUOptical/BeamPhotons = .*|i:Ph/Default/GPUOptical/BeamPhotons = $(n $B06)|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 06_mie_diffusion_cyl gpu "Mie (HG); analytic cylinder; 3-scorer deferred-trigger" "$(n $B06) γ" \
    "$LOG" $rc require \
    06_mie_diffusion_cyl/results/spread_m30_barrel_GPU.csv \
    06_mie_diffusion_cyl/results/spread_m30_xz_GPU.csv \
    06_mie_diffusion_cyl/results/spread_m30_xy50_GPU.csv
  ;;
06-cpu)
  d=06_mie_diffusion_cyl/configs
  make_cfg "$d/spread_m30_cpu.txt" "$d/_smoke.txt" \
    -e "s|^i:So/OptSrc/NumberOfHistoriesInRun.*|i:So/OptSrc/NumberOfHistoriesInRun = $(n $B06)|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $CPU_THREADS|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 06_mie_diffusion_cyl cpu "Mie (G4OpMieHG) on g4optical" "$(n $B06) γ" \
    "$LOG" $rc absent \
    06_mie_diffusion_cyl/results/spread_m30_barrel_CPU.csv 06_mie_diffusion_cyl/results/spread_m30_xy50_CPU.csv
  ;;

07-gpu|07-cpu)
  d=07_cherenkov_water/configs
  if [ $DEV = gpu ]; then src=cherenkov_e5_xz_gpu.txt; th=1; mk=require; out=07_cherenkov_water/results/e5_xz_GPU.csv
  else src=cherenkov_e5_xz_cpu.txt; th=$CPU_THREADS; mk=absent; out=07_cherenkov_water/results/e5_xz_CPU.csv; fi
  make_cfg "$d/$src" "$d/_smoke.txt" \
    -e "s|^i:So/Beam/NumberOfHistoriesInRun.*|i:So/Beam/NumberOfHistoriesInRun = $(n $B07)|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $th|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 07_cherenkov_water "$DEV" "e⁻ Cherenkov genstep; water absorption spline; TIR air-gap" "$(n $B07) e⁻" \
    "$LOG" $rc $mk "$out"
  ;;

08-gpu)
  d=08_dispersion_prism/configs
  make_cfg "$d/dispersion_visible_xze_gpu_bk.txt" "$d/_smoke.txt" \
    -e "s|^i:Ph/Default/GPUOptical/BeamPhotons = .*|i:Ph/Default/GPUOptical/BeamPhotons = $(n $B08)|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 08_dispersion_prism gpu "continuous-spectrum BEAM; polyhedra prisms n(λ); E-binned scorer" "$(n $B08) γ" \
    "$LOG" $rc require 08_dispersion_prism/results/disp_xze_GPU.csv
  ;;
08-cpu)
  d=08_dispersion_prism/configs
  make_cfg "$d/dispersion_profile_cpu.txt" "$d/_smoke.txt" \
    -e "s|^i:So/Beam/NumberOfHistoriesInRun.*|i:So/Beam/NumberOfHistoriesInRun = $(n $B08)|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $CPU_THREADS|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 08_dispersion_prism cpu "prism dispersion profile on g4optical" "$(n $B08) γ" \
    "$LOG" $rc absent 08_dispersion_prism/results/disp_prof_CPU.csv
  ;;

09-gpu|09-cpu)
  d=09_csi_wls_plate/configs
  if [ $DEV = gpu ]; then src=csi_wls_gpu.txt; th=1; mk=require
       outs="09_csi_wls_plate/results/csi_wls_GPU.csv"
  else src=csi_wls_cpu.txt; th=$CPU_THREADS; mk=absent
       outs="09_csi_wls_plate/results/csi_wls_CPU.csv"; fi
  make_cfg "$d/$src" "$d/_smoke.txt" \
    -e "s|^i:So/Beam/NumberOfHistoriesInRun.*|i:So/Beam/NumberOfHistoriesInRun = $(n $B09)|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $th|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 09_csi_wls_plate "$DEV" "proton→CsI(Tl) scintillation; WLS plate; reflector wrap" "$(n $B09) p" \
    "$LOG" $rc $mk $outs
  ;;

10-gpu)
  d=10_analytic_vs_mesh
  sed -e 's|^SEEDS=.*|SEEDS=(11)|' -e 's|^RADII=.*|RADII=(1.0)|' \
      -e '/_C_/d' -e '/_trans"/d' "$d/run.sh" > "$d/_smoke_run.sh"
  chmod +x "$d/_smoke_run.sh"
  ( cd "$d" && HIST=$(n $B10) ./_smoke_run.sh ) > "$LOG" 2>&1; rc=$?
  rm -f "$d/_smoke_run.sh"
  finish_case "$NB" 10_analytic_vs_mesh gpu "analytic sphere/cylinder/torus vs GPUForceMesh BVH (6 runs)" "$(n $B10) γ ×6" \
    "$LOG" $rc skip \
    $d/results/sphere_analytic_R1_G_s11.csv   $d/results/sphere_mesh_R1_G_s11.csv \
    $d/results/cylinder_analytic_R1_G_s11.csv $d/results/cylinder_mesh_R1_G_s11.csv \
    $d/results/torus_analytic_R1_G_s11.csv    $d/results/torus_mesh_R1_G_s11.csv
  ;;
10-cpu)
  d=10_analytic_vs_mesh
  sed -e 's|^SEEDS=.*|SEEDS=(11)|' -e 's|^RADII=.*|RADII=(1.0)|' \
      -e '/ G "/d' -e '/_trans"/d' "$d/run.sh" > "$d/_smoke_run.sh"
  chmod +x "$d/_smoke_run.sh"
  ( cd "$d" && HIST=$(n $B10) ./_smoke_run.sh ) > "$LOG" 2>&1; rc=$?
  rm -f "$d/_smoke_run.sh"
  finish_case "$NB" 10_analytic_vs_mesh cpu "same solids on g4optical (3 runs)" "$(n $B10) γ ×3" \
    "$LOG" $rc skip \
    $d/results/sphere_R1_C_s11.csv $d/results/cylinder_R1_C_s11.csv $d/results/torus_R1_C_s11.csv
  ;;
esac
done
done

# ---------- HTML report ----------
T_ALL=$(( $(date +%s) - T_ALL0 ))
OVERALL=PASS; [ $NFAIL -gt 0 ] && OVERALL=FAIL
{
cat <<HTML
<!doctype html><html><head><meta charset="utf-8">
<title>GPU optical engine — installation smoke test</title>
<style>
 :root{color-scheme:light dark}
 body{font:14px/1.5 -apple-system,'Segoe UI',sans-serif;max-width:1150px;margin:2em auto;padding:0 1em}
 h1{font-size:1.4em} table{border-collapse:collapse;width:100%;margin:1em 0}
 th,td{border:1px solid #8884;padding:.4em .6em;text-align:left;vertical-align:top}
 th{background:#8881} tr.fail{background:#e5393522} td.st.P{color:#2e7d32;font-weight:700}
 td.st.F{color:#c62828;font-weight:700}
 .badge{display:inline-block;padding:.2em .8em;border-radius:1em;color:#fff;font-weight:700}
 .P{background:#2e7d32}.F{background:#c62828}
 .dev{font-size:.85em;padding:.05em .5em;border-radius:.8em;color:#fff}
 .dgpu{background:#1565c0}.dcpu{background:#6d4c41}
 .meta{color:#888;font-size:.92em} code{background:#8882;padding:0 .3em;border-radius:3px}
 pre{background:#8881;padding:.8em;overflow-x:auto;font-size:.85em}
 .ok::before{content:"✅"}.bad::before{content:"❌"}.na::before{content:"—"}
</style></head><body>
<h1>GPU optical engine — installation smoke test
 <span class="badge $([ $OVERALL = PASS ] && echo P || echo F)">$OVERALL</span></h1>
<p class="meta">$(date '+%Y-%m-%d %H:%M %Z') · $(sw_vers -productName) $(sw_vers -productVersion) ·
$(sysctl -n machdep.cpu.brand_string 2>/dev/null) · binary: <code>$(esc <<<"$TBIN")</code> ·
DEVICES=$DEVICES · SCALE=$SCALE · total $((T_ALL/60))m $((T_ALL%60))s</p>
<p><b>$NPASS/$((NPASS+NFAIL)) checks passed.</b>
Event counts are drastically reduced (global multiplier <code>SCALE=$SCALE</code>) —
this validates the <i>installation</i> (build, patches, Metal pipeline, scorers), not physics statistics.</p>
<h2>Pre-flight</h2>
<table><tr><th>check</th><th></th><th>detail</th></tr>
HTML
while IFS=$'\t' read -r lbl st det; do
  printf '<tr class="%s"><td>%s</td><td class="%s"></td><td><code>%s</code></td></tr>\n' \
    "$([ "$st" = ok ] && echo pass || echo fail)" "$(esc <<<"$lbl")" \
    "$([ "$st" = ok ] && echo ok || echo bad)" "$(esc <<<"$det")"
done < "$PREROWS"
cat <<HTML
</table>
<h2>Benchmarks</h2>
<table><tr><th>#</th><th>benchmark</th><th>dev</th><th>exercises</th><th>events</th><th>time</th>
<th>run</th><th>log</th><th>GPU</th><th>outputs</th><th>status</th></tr>
HTML
while IFS=$'\t' read -r num name dev ex ev dt r l g o st outd; do
  cls=pass; [ "$st" = FAIL ] && cls=fail
  m(){ case $1 in ok) echo '<td class="ok"></td>';; na) echo '<td class="na"></td>';; *) echo '<td class="bad"></td>';; esac; }
  printf '<tr class="%s"><td>%s</td><td><b>%s</b></td><td><span class="dev d%s">%s</span></td><td>%s</td><td>%s</td><td>%s</td>%s%s%s<td class="meta">%s</td><td class="st %s">%s</td></tr>\n' \
    "$cls" "$num" "$(esc <<<"$name")" "$dev" "$dev" "$(esc <<<"$ex")" "$(esc <<<"$ev")" "$dt" \
    "$(m $r)" "$(m $l)" "$(m $g)" "$(esc <<<"$outd")" "$([ "$st" = PASS ] && echo P || echo F)" "$st"
done < "$ROWS"
echo '</table>'
if [ -s "$FAILDET" ]; then echo '<h2>Failure details</h2>'; cat "$FAILDET"; fi
cat <<HTML
<p class="meta">Checks — <b>run</b>: exit code 0 · <b>log</b>: no segfault/FATAL/dyld markers ·
<b>GPU</b>: GPU rows must show the <code>[TsGPU]</code> engine marker, CPU rows must NOT
(“—” = driver script hides engine stdout) · <b>outputs</b>: expected CSVs exist with sum &gt; 0.
Logs: <code>smoke_logs/</code> · Knobs: <code>DEVICES</code>, <code>ONLY</code>, <code>SCALE</code>, <code>CPU_THREADS</code></p>
</body></html>
HTML
} > "$REPORT"

echo ""
echo "── Result: $NPASS/$((NPASS+NFAIL)) passed ($OVERALL) in $((T_ALL/60))m $((T_ALL%60))s ──"
echo "── HTML report: $REPORT ──"
[ $OVERALL = PASS ]
