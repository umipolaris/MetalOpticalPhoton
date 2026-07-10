#!/usr/bin/env bash
# ============================================================
# GPU optical engine — installation smoke test (benchmarks 01–10, GPU + CPU + figures)
#
# Runs every benchmark at a reduced fraction of its ORIGINAL event count —
# GPU at 1/GPU_DIV (default 1/10), CPU at 1/CPU_DIV (default 1/200, since the
# CPU is O(20×) slower) — and checks, per run:
#   [run]    the TOPAS run exits 0
#   [log]    no crash/error markers (segfault, FATAL, dyld, …)
#   [engine] GPU rows: engine engaged ("[TsGPU]" in log);
#            CPU rows: engine NOT engaged (module wiring check)
#   [out]    expected output CSV(s) exist with a nonzero sum
# and, when both devices ran, executes each benchmark's plot script and checks
# that the figure PNG is actually produced (committed reference figures are
# backed up and restored; the smoke-generated PNGs are kept in smoke_logs/figs/
# and embedded in the HTML report).
#
# This validates the INSTALLATION (build, patches, Metal pipeline, scorers,
# python plotting) — NOT physics statistics.
#
#   ./run_smoke_test.sh                     # all 10, GPU+CPU+figures
#   DEVICES=gpu ./run_smoke_test.sh         # GPU only, no figures
#   ONLY="03 07" ./run_smoke_test.sh        # subset of benchmarks
#   GPU_DIV=100 CPU_DIV=2000 ./run_smoke_test.sh   # even smaller fractions
#   PLOTS=0 ./run_smoke_test.sh             # skip the figure stage
#   OPEN=0 ./run_smoke_test.sh              # don't auto-open the report in the browser
#
# Report: smoke_logs/smoke_report.html — opened in the browser at start and
# LIVE-UPDATED as each stage finishes (auto-refresh every 2 s while running).
# ============================================================
set -u
cd "$(dirname "$0")"
. common/topas_env.sh   # resolves $TOPAS (env > PATH > default)

# ---------- knobs ----------
ONLY="${ONLY:-01 02 03 04 05 06 07 08 09 10}"
DEVICES="${DEVICES:-gpu cpu}"
GPU_DIV="${GPU_DIV:-10}"     # GPU event count = original / GPU_DIV
CPU_DIV="${CPU_DIV:-200}"    # CPU event count = original / CPU_DIV
CPU_THREADS="${CPU_THREADS:-20}"
PLOTS="${PLOTS:-1}"          # 1 = run plot scripts + check PNGs (needs both devices)
OPEN="${OPEN:-1}"            # 1 = auto-open the live HTML report at start (macOS 'open')

# original (full-benchmark) event counts
O01=1000         # protons   (sim_light 1K)
O02=100000       # photons   (Fresnel U1v5, per angle)
O03=1000000000   # photons   (Rayleigh 10^9)
O04=2500         # protons   (Birks)
O05=10000000000  # photons   (torus 10B)
O06=1000000000   # photons   (Mie 10^9)
O07=10000        # electrons (Cherenkov e5 case)
O08=5000000      # photons   (prism 5M)
O09=10000        # protons   (CsI+WLS)
O10=2000000      # photons   (analytic-vs-mesh, per run)
ng() { awk -v b="$1" -v d="$GPU_DIV" 'BEGIN{v=int(b/d); if(v<1)v=1; printf "%d", v}'; }
nc() { awk -v b="$1" -v d="$CPU_DIV" 'BEGIN{v=int(b/d); if(v<1)v=1; printf "%d", v}'; }
nn() { if [ "$DEV" = gpu ]; then ng "$1"; else nc "$1"; fi; }

SMOKE="$PWD/smoke_logs"; mkdir -p "$SMOKE" "$SMOKE/figs"
STATE="$SMOKE/state"; rm -rf "$STATE"; mkdir -p "$STATE"
PREROWS="$SMOKE/.pre.tsv";   : > "$PREROWS"
FAILDET="$SMOKE/.fail.html"; : > "$FAILDET"
REPORT="$SMOKE/smoke_report.html"
NPASS=0; NFAIL=0; CUR_ROW=0; RUNNING=1
T_ALL0=$(date +%s)
BOTH_DEV=0; case " $DEVICES " in *" gpu "*) case " $DEVICES " in *" cpu "*) BOTH_DEV=1;; esac;; esac

# ---------- helpers ----------
esc() { sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g'; }
csv_sum() { awk -F',' '!/^#/{for(i=1;i<=NF;i++)s+=$i+0}END{printf "%.10g", s+0}' "$1" 2>/dev/null; }

pre_check() { # label ok|fail detail
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$PREROWS"
  [ "$2" = ok ] && printf '  ✓ %-26s %s\n' "$1" "$3" || printf '  ✗ %-26s %s\n' "$1" "$3"
}

engine_text() { # mode result -> human text
  case "$1-$2" in
    require-ok) echo "GPU engaged" ;;  require-fail) echo "GPU NOT engaged ⚠" ;;
    absent-ok)  echo "GPU off ✓" ;;    absent-fail)  echo "GPU engaged on CPU run ⚠" ;;
    *) echo "—" ;;
  esac
}

# finish_case NUM NAME DEV EXERCISES EVENTS LOG RC MARKER(require|auto|skip|absent) OUT...
finish_case() {
  local num="$1" name="$2" dev="$3" ex="$4" ev="$5" log="$6" rc="$7" mk="$8"; shift 8
  local c_run=ok c_log=ok c_eng=ok c_out=ok outdesc="" f s dt engtxt
  dt=$(( $(date +%s) - T_CASE0 ))
  [ "$rc" -eq 0 ] || c_run=fail
  grep -qE 'Segmentation fault|FATAL|Abort trap|dyld\[|Library not loaded|command not found|Traceback' "$log" && c_log=fail
  case "$mk" in
    require) grep -q '\[TsGPU\]' "$log" || c_eng=fail ;;
    absent)  grep -q '\[TsGPU\]' "$log" && c_eng=fail || c_eng=ok ;;
    auto)    grep -q '\[TsGPU\]' "$log" && { mk=require; c_eng=ok; } || { mk=na; c_eng=na; } ;;
    skip)    mk=na; c_eng=na ;;
  esac
  engtxt="$(engine_text "$mk" "$c_eng")"
  for f in "$@"; do
    if [ -s "$f" ]; then
      s=$(csv_sum "$f")
      if awk -v s="$s" 'BEGIN{exit !(s>0)}'; then outdesc+="$(basename "$f") (sum=$s); "
      else c_out=fail; outdesc+="$(basename "$f") (sum=0 ⚠); "; fi
    else c_out=fail; outdesc+="$(basename "$f") (MISSING ⚠); "; fi
  done
  local status=PASS
  { [ $c_run = fail ] || [ $c_log = fail ] || [ $c_eng = fail ] || [ $c_out = fail ]; } && status=FAIL
  [ $status = PASS ] && NPASS=$((NPASS+1)) || NFAIL=$((NFAIL+1))
  printf '%s\t%s\t%s\t%s\t%s\t%ss\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$num" "$name" "$dev" "$ex" "$ev" "$dt" "$c_run" "$c_log" "$engtxt" "$c_out" "$status" "$outdesc" \
    > "$STATE/$(printf '%02d' $CUR_ROW).tsv"
  printf '  %s %s [%s]  (%ss)  run:%s log:%s engine:%s out:%s\n' \
    "$([ $status = PASS ] && echo '✓' || echo '✗ FAIL')" "$num $name" "$dev" "$dt" "$c_run" "$c_log" "$engtxt" "$c_out"
  if [ $status = FAIL ]; then
    { printf '<details open><summary><b>%s %s [%s]</b> — log tail (%s)</summary><pre>' "$num" "$name" "$dev" "$(basename "$log")"
      tail -40 "$log" | esc; printf '</pre></details>\n'; } >> "$FAILDET"
  fi
  render_report
}

# mark the next planned row as running and refresh the live page
begin_row() { # NUM NAME STAGE
  CUR_ROW=$((CUR_ROW+1)); T_CASE0=$(date +%s)
  printf '%s\t%s\t%s\t—\t—\t…\t-\t-\t—\t-\tRUNNING\t\n' "$1" "$2" "$3" \
    > "$STATE/$(printf '%02d' $CUR_ROW).tsv"
  render_report
}

run_cfg() { ( cd "$1" && G4TRACE_DIR=OFF "$TOPAS" "$2" ) > "$3" 2>&1; }
make_cfg() { local src="$1" dst="$2"; shift 2; sed "$@" "$src" > "$dst"; }
# ensure an override exists even when the source config inherits the parameter
# via includeFile (sed no-op there); in TOPAS the including file's definition wins.
force_param() { grep -qE "$2" "$1" || printf '%s\n' "$3" >> "$1"; }

# fig_stage NUM NAME DIR PLOTCMD PNG...
# Backs up committed reference PNGs, runs the plot script, verifies the PNGs
# were (re)generated, keeps copies in smoke_logs/figs/, restores the originals.
fig_stage() {
  local num="$1" name="$2" dir="$3" cmd="$4"; shift 4
  [ "$PLOTS" = 1 ] || return 0
  if [ $BOTH_DEV -ne 1 ]; then return 0; fi   # figures need both devices' CSVs
  begin_row "$num" "$name" fig
  local log="$SMOKE/${num}_fig.log" stamp="$SMOKE/.stamp" f bak rc
  local c_run=ok c_log=ok c_out=ok outdesc="" dt
  for f in "$@"; do
    [ -f "$dir/$f" ] && cp -p "$dir/$f" "$SMOKE/figs/.bak_${num}_$(basename "$f")"
  done
  touch "$stamp"; sleep 1
  ( cd "$dir" && MPLBACKEND=Agg python3 $cmd ) > "$log" 2>&1; rc=$?
  [ $rc -eq 0 ] || c_run=fail
  grep -qE 'Traceback|Error' "$log" && c_log=fail
  for f in "$@"; do
    if [ -f "$dir/$f" ] && [ "$dir/$f" -nt "$stamp" ] && [ "$(stat -f%z "$dir/$f")" -gt 1000 ]; then
      cp -p "$dir/$f" "$SMOKE/figs/${num}_$(basename "$f")"
      outdesc+="$(basename "$f") ($(( $(stat -f%z "$SMOKE/figs/${num}_$(basename "$f")") /1024 )) kB); "
    else
      c_out=fail; outdesc+="$(basename "$f") (NOT regenerated ⚠); "
    fi
    bak="$SMOKE/figs/.bak_${num}_$(basename "$f")"
    [ -f "$bak" ] && cp -p "$bak" "$dir/$f" && rm -f "$bak"
  done
  dt=$(( $(date +%s) - T_CASE0 ))
  local status=PASS
  { [ $c_run = fail ] || [ $c_log = fail ] || [ $c_out = fail ]; } && status=FAIL
  [ $status = PASS ] && NPASS=$((NPASS+1)) || NFAIL=$((NFAIL+1))
  printf '%s\t%s\t%s\t%s\t%s\t%ss\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$num" "$name" fig "figure generation ($cmd)" "—" "$dt" "$c_run" "$c_log" "—" "$c_out" "$status" "$outdesc" \
    > "$STATE/$(printf '%02d' $CUR_ROW).tsv"
  printf '  %s %s [fig]  (%ss)  run:%s log:%s png:%s\n' \
    "$([ $status = PASS ] && echo '✓' || echo '✗ FAIL')" "$num $name" "$dt" "$c_run" "$c_log" "$c_out"
  if [ $status = FAIL ]; then
    { printf '<details open><summary><b>%s %s [fig]</b> — log tail</summary><pre>' "$num" "$name"
      tail -40 "$log" | esc; printf '</pre></details>\n'; } >> "$FAILDET"
  fi
  render_report
}

bench_name() { case $1 in
  01) echo 01_speed;; 02) echo 02_reflection_refraction_tir;; 03) echo 03_rayleigh;;
  04) echo 04_birks;; 05) echo 05_torus;; 06) echo 06_mie_diffusion_cyl;;
  07) echo 07_cherenkov_water;; 08) echo 08_dispersion_prism;;
  09) echo 09_csi_wls_plate;; 10) echo 10_analytic_vs_mesh;; esac; }

# (re)write the HTML report atomically — called after every state change, so an
# auto-refreshing browser tab shows live progress.
render_report() {
  local tmp="$REPORT.tmp" ELAP badge btxt refresh done_n
  ELAP=$(( $(date +%s) - T_ALL0 )); done_n=$((NPASS+NFAIL))
  if [ "$RUNNING" = 1 ]; then
    badge=R; btxt="RUNNING ${done_n}/${PLAN_TOTAL}"; refresh='<meta http-equiv="refresh" content="2">'
  elif [ $NFAIL -gt 0 ]; then badge=F; btxt=FAIL; refresh=""
  else badge=P; btxt=PASS; refresh=""; fi
  {
cat <<HTML
<!doctype html><html><head><meta charset="utf-8">$refresh
<title>GPU optical engine — installation smoke test</title>
<style>
 :root{color-scheme:light dark}
 body{font:14px/1.5 -apple-system,'Segoe UI',sans-serif;max-width:1150px;margin:2em auto;padding:0 1em}
 h1{font-size:1.4em} table{border-collapse:collapse;width:100%;margin:1em 0}
 th,td{border:1px solid #8884;padding:.4em .6em;text-align:left;vertical-align:top}
 th{background:#8881} tr.fail{background:#e5393522} tr.running{background:#f9a82522}
 td.st{text-align:center} td.st.P{color:#888} td.st.F{color:#c62828;font-weight:700}
 td.st.RU{color:#e65100;font-weight:700} td.st.PE{color:#8886}
 .badge{display:inline-block;padding:.2em .8em;border-radius:1em;color:#fff;font-weight:700}
 .P{background:#2e7d32}.F{background:#c62828}.R{background:#f9a825;color:#000}
 .dev{font-size:.85em;padding:.05em .5em;border-radius:.8em;color:#fff}
 .dgpu{background:#1565c0}.dcpu{background:#6d4c41}.dfig{background:#7b1fa2}
 .meta{color:#888;font-size:.92em} code{background:#8882;padding:0 .3em;border-radius:3px}
 pre{background:#8881;padding:.8em;overflow-x:auto;font-size:.85em}
 .ok::before{content:"✓"}.bad::before{content:"✗";color:#c62828;font-weight:700}.na::before{content:"—"}
 .gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:1em}
 .gallery figure{margin:0;border:1px solid #8884;border-radius:6px;padding:.5em}
 .gallery img{max-width:100%;height:auto} .gallery figcaption{font-size:.85em;color:#888}
</style></head><body>
<h1>GPU optical engine — installation smoke test
 <span class="badge $badge">$btxt</span></h1>
<p class="meta">$(date '+%Y-%m-%d %H:%M %Z') · $(sw_vers -productName) $(sw_vers -productVersion) ·
$(sysctl -n machdep.cpu.brand_string 2>/dev/null) · binary: <code>$(esc <<<"$TBIN")</code> ·
DEVICES=$DEVICES · GPU 1/$GPU_DIV · CPU 1/$CPU_DIV of full benchmark counts · PLOTS=$PLOTS ·
elapsed $((ELAP/60))m $((ELAP%60))s</p>
<p><b>$done_n/$PLAN_TOTAL stages finished — pass $NPASS · fail $NFAIL.</b>
Event counts are reduced fractions of each benchmark's full scale (GPU ÷$GPU_DIV, CPU ÷$CPU_DIV) —
this validates the <i>installation</i> (build, patches, Metal pipeline, scorers, plotting),
not physics statistics. Figures below are therefore noisier than the reference ones.</p>
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
<table><tr><th>#</th><th>benchmark</th><th>stage</th><th>exercises</th><th>events</th><th>time</th>
<th>run</th><th>log</th><th>engine</th><th>outputs</th><th>status</th></tr>
HTML
local sf num name dev ex ev dt r l engtxt o st outd cls stcls sttxt
for sf in "$STATE"/*.tsv; do
  [ -f "$sf" ] || continue
  IFS=$'\t' read -r num name dev ex ev dt r l engtxt o st outd < "$sf"
  case "$st" in
    FAIL)    cls=fail;    stcls=F;  sttxt=FAIL ;;
    PASS)    cls=pass;    stcls=P;  sttxt='✓' ;;
    RUNNING) cls=running; stcls=RU; sttxt='▶ running' ;;
    *)       cls=pending; stcls=PE; sttxt='·' ;;
  esac
  m(){ case $1 in ok) echo '<td class="ok"></td>';; na) echo '<td class="na"></td>';; -) echo '<td></td>';; *) echo '<td class="bad"></td>';; esac; }
  printf '<tr class="%s"><td>%s</td><td><b>%s</b></td><td><span class="dev d%s">%s</span></td><td>%s</td><td>%s</td><td>%s</td>%s%s<td>%s</td><td class="meta">%s</td><td class="st %s">%s</td></tr>\n' \
    "$cls" "$num" "$(esc <<<"$name")" "$dev" "$dev" "$(esc <<<"$ex")" "$(esc <<<"$ev")" "$dt" \
    "$(m $r)" "$(m $l)" "$(esc <<<"$engtxt")" "$(esc <<<"$outd")" "$stcls" "$sttxt"
done
echo '</table>'
if [ -s "$FAILDET" ]; then echo '<h2>Failure details</h2>'; cat "$FAILDET"; fi
if ls "$SMOKE"/figs/*.png >/dev/null 2>&1; then
  echo '<h2>Generated figures (reduced statistics — noise expected)</h2><div class="gallery">'
  local f
  for f in "$SMOKE"/figs/*.png; do
    printf '<figure><img src="figs/%s" alt=""><figcaption>%s</figcaption></figure>\n' \
      "$(basename "$f")" "$(basename "$f")"
  done
  echo '</div>'
fi
cat <<HTML
<p class="meta">Checks — <b>run</b>: exit code 0 · <b>log</b>: no segfault/FATAL/dyld/Traceback markers ·
<b>engine</b>: GPU rows must engage the GPU engine (<code>[TsGPU]</code>), CPU rows must not;
"—" = the stage's driver hides engine stdout · <b>outputs</b>: expected CSVs exist with sum &gt; 0;
figure rows: PNG regenerated (&gt;1 kB). Committed reference figures are restored after the check —
the smoke-generated PNGs live in <code>smoke_logs/figs/</code>.
Logs: <code>smoke_logs/</code> · Knobs: <code>DEVICES</code>, <code>ONLY</code>, <code>GPU_DIV</code>, <code>CPU_DIV</code>, <code>PLOTS</code>, <code>OPEN</code>, <code>CPU_THREADS</code></p>
</body></html>
HTML
  } > "$tmp"
  mv "$tmp" "$REPORT"
}

# ---------- pre-flight ----------
echo "── Pre-flight ──  (DEVICES='$DEVICES'  GPU_DIV=$GPU_DIV  CPU_DIV=$CPU_DIV  PLOTS=$PLOTS)"
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
if [ "$PLOTS" = 1 ]; then
  python3 -c 'import numpy, matplotlib' >/dev/null 2>&1 \
    && pre_check "python3 + numpy/matplotlib" ok "$(python3 -V 2>&1)" \
    || { pre_check "python3 + numpy/matplotlib" fail "figure stage disabled"; PLOTS=0; }
fi

# ---------- plan all stages as pending + open the live report ----------
PLAN_TOTAL=0
for NB in $ONLY; do
  for DEV in $DEVICES; do
    PLAN_TOTAL=$((PLAN_TOTAL+1))
    printf '%s\t%s\t%s\t—\t—\t·\t-\t-\t—\t-\tPENDING\t\n' "$NB" "$(bench_name $NB)" "$DEV" \
      > "$STATE/$(printf '%02d' $PLAN_TOTAL).tsv"
  done
  if [ "$PLOTS" = 1 ] && [ $BOTH_DEV = 1 ]; then case $NB in 02|03|04|05|06|07|08|09)
    PLAN_TOTAL=$((PLAN_TOTAL+1))
    printf '%s\t%s\tfig\t—\t—\t·\t-\t-\t—\t-\tPENDING\t\n' "$NB" "$(bench_name $NB)" \
      > "$STATE/$(printf '%02d' $PLAN_TOTAL).tsv" ;; esac
  fi
done
render_report
[ "$OPEN" = 1 ] && command -v open >/dev/null 2>&1 && open "$REPORT"
trap 'RUNNING=0; render_report' INT TERM
CUR_ROW=0

# ---------- benchmarks ----------
for NB in $ONLY; do
for DEV in $DEVICES; do
begin_row "$NB" "$(bench_name $NB)" "$DEV"
LOG="$SMOKE/${NB}_${DEV}.log"
case "$NB-$DEV" in

01-gpu|01-cpu)
  d=01_speed/configs; NN=$(nn $O01)
  ln -sfn ../../common/simple_biconvex_lens_new.stl "$d/simple_biconvex_lens_new.stl"
  if [ $DEV = gpu ]; then src=sim_light_gpu_1K.txt; th=1; mk=require; out=01_speed/results/SensorDepth_GPU.csv
  else src=sim_light_cpu_1K.txt; th=$CPU_THREADS; mk=absent; out=01_speed/results/SensorDepth_CPU.csv; fi
  make_cfg "$d/$src" "$d/_smoke.txt" \
    -e "s|^i:So/Example1/NumberOfHistoriesInRun.*|i:So/Example1/NumberOfHistoriesInRun   = $NN|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $th|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 01_speed "$DEV" "proton→scintillation genstep; STL-lens BVH; DDA scorer" "$NN p" "$LOG" $rc $mk "$out"
  ;;

02-gpu|02-cpu)
  d=02_reflection_refraction_tir; NN=$(nn $O02)
  sed "s|NumberOfHistoriesInRun = 100000|NumberOfHistoriesInRun = $NN|" \
    "$d/scripts/build_run_u1v5.py" > "$d/scripts/_smoke_u1v5.py"
  ( cd "$d" && rm -f results/built/U1v5_${DEV}_th30_*_sf1.csv \
    && ANGLES=30 MODE=$DEV THREADS=$([ $DEV = gpu ] && echo 1 || echo $CPU_THREADS) TOPAS="$TOPAS" \
       python3 scripts/_smoke_u1v5.py ) > "$LOG" 2>&1; rc=$?
  rm -f "$d/scripts/_smoke_u1v5.py"
  finish_case "$NB" 02_reflection_refraction_tir "$DEV" "TOPAS-Beam photon hand-off; Fresnel R/T at 30°" "$NN γ ×1 angle" \
    "$LOG" $rc $([ $DEV = gpu ] && echo auto || echo skip) \
    "$d/results/built/U1v5_${DEV}_th30_R_sf1.csv" "$d/results/built/U1v5_${DEV}_th30_T_sf1.csv"
  ;;

03-gpu)
  d=03_rayleigh/configs; NN=$(ng $O03)
  make_cfg "$d/rayleigh_color_gpu.txt" "$d/_smoke.txt" \
    -e "s|^u:Ph/Default/GPUOptical/BeamPhotons=.*|u:Ph/Default/GPUOptical/BeamPhotons=$NN|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 03_rayleigh gpu "BEAM kernel; Rayleigh λ⁴; energy-resolved x–z scorer" "$NN γ" \
    "$LOG" $rc require 03_rayleigh/results/rayleigh_color_GPU.csv
  ;;
03-cpu)
  d=03_rayleigh/configs; NN=$(nc $O03)
  make_cfg "$d/rayleigh_color_cpu.txt" "$d/_smoke.txt" \
    -e "s|^i:So/Beam/NumberOfHistoriesInRun=.*|i:So/Beam/NumberOfHistoriesInRun=$NN|" \
    -e "s|^i:Ts/NumberOfThreads=.*|i:Ts/NumberOfThreads=$CPU_THREADS|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 03_rayleigh cpu "Rayleigh λ⁴ on g4optical; energy-resolved x–z scorer" "$NN γ" \
    "$LOG" $rc absent 03_rayleigh/results/rayleigh_color_CPU.csv
  ;;

04-gpu)
  d=04_birks/configs; NN=$(ng $O04)
  rc=0
  for cfg in U5_short_gpu U5_short_gpu_birksoff U5_short_gpu_z U5_short_gpu_z_birksoff U5_short_gpu_x; do
    make_cfg "$d/$cfg.txt" "$d/_smoke.txt" \
      -e "s|^i:So/Beam/NumberOfHistoriesInRun.*|i:So/Beam/NumberOfHistoriesInRun   = $NN|" \
      -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = 1|"
    # *_birksoff variants inherit these via includeFile — append overrides
    force_param "$d/_smoke.txt" '^i:So/Beam/NumberOfHistoriesInRun' "i:So/Beam/NumberOfHistoriesInRun = $NN"
    force_param "$d/_smoke.txt" '^i:Ts/NumberOfThreads' "i:Ts/NumberOfThreads = 1"
    run_cfg "$d" _smoke.txt "$LOG.$cfg"; r=$?; [ $r -ne 0 ] && rc=$r
    rm -f "$d/_smoke.txt"
  done
  cat "$LOG".U5_short_gpu* > "$LOG"; rm -f "$LOG".U5_short_gpu*
  finish_case "$NB" 04_birks gpu "Birks scintillation genstep; analytic box; 5 configs (on/off/z/x)" "$NN p ×5" \
    "$LOG" $rc require \
    04_birks/results/U5short_GPU_fluence.csv 04_birks/results/U5short_GPU_edep.csv \
    04_birks/results/U5short_GPU_fluence_z.csv 04_birks/results/U5short_GPU_fluence_x.csv \
    04_birks/results/U5short_GPU_fluence_birksoff.csv 04_birks/results/U5short_GPU_fluence_z_birksoff.csv
  ;;
04-cpu)
  d=04_birks/configs; NN=$(nc $O04)
  rc=0
  for cfg in U5_short_cpu U5_short_cpu_birksoff; do
    make_cfg "$d/$cfg.txt" "$d/_smoke.txt" \
      -e "s|^i:So/Beam/NumberOfHistoriesInRun.*|i:So/Beam/NumberOfHistoriesInRun   = $NN|" \
      -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $CPU_THREADS|"
    # U5_short_cpu_birksoff inherits these via includeFile — append overrides
    force_param "$d/_smoke.txt" '^i:So/Beam/NumberOfHistoriesInRun' "i:So/Beam/NumberOfHistoriesInRun = $NN"
    force_param "$d/_smoke.txt" '^i:Ts/NumberOfThreads' "i:Ts/NumberOfThreads = $CPU_THREADS"
    run_cfg "$d" _smoke.txt "$LOG.$cfg"; r=$?; [ $r -ne 0 ] && rc=$r
    rm -f "$d/_smoke.txt"
  done
  cat "$LOG".U5_short_cpu* > "$LOG"; rm -f "$LOG".U5_short_cpu*
  finish_case "$NB" 04_birks cpu "CPU optical multi-scorer ×2 (Birks on/off); parallel-world patch path" "$NN p ×2" \
    "$LOG" $rc absent \
    04_birks/results/U5short_CPU_fluence.csv 04_birks/results/U5short_CPU_fluence_z.csv \
    04_birks/results/U5short_CPU_fluence_x.csv 04_birks/results/U5short_CPU_edep.csv \
    04_birks/results/U5short_CPU_fluence_birksoff.csv 04_birks/results/U5short_CPU_fluence_z_birksoff.csv
  ;;

05-gpu)
  d=05_torus/configs; NN=$(ng $O05)
  ln -sfn ../../common/simple_biconvex_lens_new.stl "$d/simple_biconvex_lens_new.stl"
  make_cfg "$d/torus_diffuse_10B_beamgpu.txt" "$d/_smoke.txt" \
    -e "s|^u:Ph/Default/GPUOptical/BeamPhotons = .*|u:Ph/Default/GPUOptical/BeamPhotons = $NN|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 05_torus gpu "BEAM kernel; torus SDF analytic; STL lens; camera scorer" "$NN γ" \
    "$LOG" $rc require 05_torus/results/torus_diffuse_10B_beamgpu_GPU.csv
  ;;
05-cpu)
  d=05_torus/configs; NN=$(nc $O05)
  ln -sfn ../../common/simple_biconvex_lens_new.stl "$d/simple_biconvex_lens_new.stl"
  make_cfg "$d/torus_diffuse_10B_cpu_fresh.txt" "$d/_smoke.txt" \
    -e "s|^i:So/Light/NumberOfHistoriesInRun.*|i:So/Light/NumberOfHistoriesInRun   = $NN|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $CPU_THREADS|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 05_torus cpu "torus + diffuse mirror + STL lens on g4optical" "$NN γ" \
    "$LOG" $rc absent 05_torus/results/torus_diffuse_10B_CPU_fresh.csv
  ;;

06-gpu)
  d=06_mie_diffusion_cyl/configs; NN=$(ng $O06)
  make_cfg "$d/spread_m30_both_gpu.txt" "$d/_smoke.txt" \
    -e "s|^i:Ph/Default/GPUOptical/BeamPhotons = .*|i:Ph/Default/GPUOptical/BeamPhotons = $NN|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 06_mie_diffusion_cyl gpu "Mie (HG); analytic cylinder; 3-scorer deferred-trigger" "$NN γ" \
    "$LOG" $rc require \
    06_mie_diffusion_cyl/results/spread_m30_barrel_GPU.csv \
    06_mie_diffusion_cyl/results/spread_m30_xz_GPU.csv \
    06_mie_diffusion_cyl/results/spread_m30_xy50_GPU.csv
  ;;
06-cpu)
  d=06_mie_diffusion_cyl/configs; NN=$(nc $O06)
  make_cfg "$d/spread_m30_cpu.txt" "$d/_smoke.txt" \
    -e "s|^i:So/OptSrc/NumberOfHistoriesInRun.*|i:So/OptSrc/NumberOfHistoriesInRun = $NN|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $CPU_THREADS|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 06_mie_diffusion_cyl cpu "Mie (G4OpMieHG) on g4optical" "$NN γ" \
    "$LOG" $rc absent \
    06_mie_diffusion_cyl/results/spread_m30_barrel_CPU.csv 06_mie_diffusion_cyl/results/spread_m30_xy50_CPU.csv
  ;;

07-gpu)
  d=07_cherenkov_water/configs; NN=$(ng $O07)
  rc=0
  # e5 (electron) + c12_300 (carbon ion — panel (a) of the figure needs its GPU map)
  for cfg in cherenkov_e5_xz_gpu cherenkov_c12_300_xz_gpu; do
    make_cfg "$d/$cfg.txt" "$d/_smoke.txt" \
      -e "s|^i:So/Beam/NumberOfHistoriesInRun.*|i:So/Beam/NumberOfHistoriesInRun = $NN|" \
      -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $CPU_THREADS|"
    run_cfg "$d" _smoke.txt "$LOG.$cfg"; r=$?; [ $r -ne 0 ] && rc=$r
    rm -f "$d/_smoke.txt"
  done
  cat "$LOG".cherenkov_* > "$LOG"; rm -f "$LOG".cherenkov_*
  finish_case "$NB" 07_cherenkov_water gpu "Cherenkov genstep (e⁻ + ¹²C); water absorption spline; TIR air-gap" "$NN e⁻ + $NN C12" \
    "$LOG" $rc require 07_cherenkov_water/results/e5_xz_GPU.csv 07_cherenkov_water/results/c12_300_xz_GPU.csv
  ;;
07-cpu)
  d=07_cherenkov_water/configs; NN=$(nc $O07)
  make_cfg "$d/cherenkov_e5_xz_cpu.txt" "$d/_smoke.txt" \
    -e "s|^i:So/Beam/NumberOfHistoriesInRun.*|i:So/Beam/NumberOfHistoriesInRun = $NN|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $CPU_THREADS|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 07_cherenkov_water cpu "e⁻ Cherenkov on g4optical; water absorption spline" "$NN e⁻" \
    "$LOG" $rc absent 07_cherenkov_water/results/e5_xz_CPU.csv
  ;;

08-gpu)
  d=08_dispersion_prism/configs; NN=$(ng $O08)
  rc=0
  for cfg in dispersion_visible_xze_gpu_bk dispersion_profile_gpu_bk; do
    make_cfg "$d/$cfg.txt" "$d/_smoke.txt" \
      -e "s|^i:Ph/Default/GPUOptical/BeamPhotons = .*|i:Ph/Default/GPUOptical/BeamPhotons = $NN|"
    run_cfg "$d" _smoke.txt "$LOG.$cfg"; r=$?; [ $r -ne 0 ] && rc=$r
    rm -f "$d/_smoke.txt"
  done
  cat "$LOG".dispersion_* > "$LOG"; rm -f "$LOG".dispersion_*
  finish_case "$NB" 08_dispersion_prism gpu "continuous-spectrum BEAM; polyhedra prisms n(λ); xze + profile" "$NN γ ×2" \
    "$LOG" $rc require \
    08_dispersion_prism/results/disp_xze_GPU.csv 08_dispersion_prism/results/disp_prof_GPU_bk.csv
  ;;
08-cpu)
  d=08_dispersion_prism/configs; NN=$(nc $O08)
  make_cfg "$d/dispersion_profile_cpu.txt" "$d/_smoke.txt" \
    -e "s|^i:So/Beam/NumberOfHistoriesInRun.*|i:So/Beam/NumberOfHistoriesInRun = $NN|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $CPU_THREADS|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 08_dispersion_prism cpu "prism dispersion profile on g4optical" "$NN γ" \
    "$LOG" $rc absent 08_dispersion_prism/results/disp_prof_CPU.csv
  ;;

09-gpu|09-cpu)
  d=09_csi_wls_plate/configs; NN=$(nn $O09)
  if [ $DEV = gpu ]; then src=csi_wls_gpu.txt; th=1; mk=require
       outs="09_csi_wls_plate/results/csi_wls_GPU.csv"
  else src=csi_wls_cpu.txt; th=$CPU_THREADS; mk=absent
       outs="09_csi_wls_plate/results/csi_wls_CPU.csv"; fi
  make_cfg "$d/$src" "$d/_smoke.txt" \
    -e "s|^i:So/Beam/NumberOfHistoriesInRun.*|i:So/Beam/NumberOfHistoriesInRun = $NN|" \
    -e "s|^i:Ts/NumberOfThreads.*|i:Ts/NumberOfThreads = $th|"
  run_cfg "$d" _smoke.txt "$LOG"; rc=$?; rm -f "$d/_smoke.txt"
  finish_case "$NB" 09_csi_wls_plate "$DEV" "proton→CsI(Tl) scintillation; WLS plate; reflector wrap" "$NN p" \
    "$LOG" $rc $mk $outs
  ;;

10-gpu)
  d=10_analytic_vs_mesh; NN=$(ng $O10)
  sed -e 's|^SEEDS=.*|SEEDS=(11)|' -e 's|^RADII=.*|RADII=(1.0)|' \
      -e '/_C_/d' -e '/_trans"/d' "$d/run.sh" > "$d/_smoke_run.sh"
  chmod +x "$d/_smoke_run.sh"
  ( cd "$d" && HIST=$NN ./_smoke_run.sh ) > "$LOG" 2>&1; rc=$?
  rm -f "$d/_smoke_run.sh"
  finish_case "$NB" 10_analytic_vs_mesh gpu "analytic sphere/cylinder/torus vs GPUForceMesh BVH (6 runs)" "$NN γ ×6" \
    "$LOG" $rc skip \
    $d/results/sphere_analytic_R1_G_s11.csv   $d/results/sphere_mesh_R1_G_s11.csv \
    $d/results/cylinder_analytic_R1_G_s11.csv $d/results/cylinder_mesh_R1_G_s11.csv \
    $d/results/torus_analytic_R1_G_s11.csv    $d/results/torus_mesh_R1_G_s11.csv
  ;;
10-cpu)
  d=10_analytic_vs_mesh; NN=$(nc $O10)
  sed -e 's|^SEEDS=.*|SEEDS=(11)|' -e 's|^RADII=.*|RADII=(1.0)|' \
      -e '/ G "/d' -e '/_trans"/d' "$d/run.sh" > "$d/_smoke_run.sh"
  chmod +x "$d/_smoke_run.sh"
  ( cd "$d" && HIST=$NN ./_smoke_run.sh ) > "$LOG" 2>&1; rc=$?
  rm -f "$d/_smoke_run.sh"
  finish_case "$NB" 10_analytic_vs_mesh cpu "same solids on g4optical (3 runs)" "$NN γ ×3" \
    "$LOG" $rc skip \
    $d/results/sphere_R1_C_s11.csv $d/results/cylinder_R1_C_s11.csv $d/results/torus_R1_C_s11.csv
  ;;
esac
done   # DEV

# ---- figure stage (needs both devices' CSVs) ----
case $NB in
  02) fig_stage 02 02_reflection_refraction_tir 02_reflection_refraction_tir "scripts/fig_02_angle_scan.py" results/fig_02_angle_scan.png ;;
  03) fig_stage 03 03_rayleigh 03_rayleigh "plot_color.py" results/fig_03_rayleigh_color.png ;;
  04) fig_stage 04 04_birks 04_birks "plot.py" results/fig_04_birks.png ;;
  05) fig_stage 05 05_torus 05_torus "plot.py" results/fig_05_torus.png ;;
  06) fig_stage 06 06_mie_diffusion_cyl 06_mie_diffusion_cyl "plot.py" results/fig_06_mie.png ;;
  07) fig_stage 07 07_cherenkov_water 07_cherenkov_water "plot.py" results/fig_07_cherenkov_geometry_and_zfluence_cpu_gpu.png ;;
  08) fig_stage 08 08_dispersion_prism 08_dispersion_prism "plot.py" results/fig_08_combined_cpu_vs_gpu.png ;;
  09) fig_stage 09 09_csi_wls_plate 09_csi_wls_plate "plot.py" results/csi_wls_dose.png ;;
esac
done   # NB

# ---------- finalize ----------
RUNNING=0
T_ALL=$(( $(date +%s) - T_ALL0 ))
OVERALL=PASS; [ $NFAIL -gt 0 ] && OVERALL=FAIL
render_report

echo ""
echo "── Result: $NPASS/$((NPASS+NFAIL)) passed ($OVERALL) in $((T_ALL/60))m $((T_ALL%60))s ──"
echo "── HTML report: $REPORT ──"
[ $OVERALL = PASS ]
