#!/bin/bash
# ============================================================
# MetalOpticalPhoton + TOPAS GPU 빌드/설치 스크립트  (macOS Apple Silicon)
#
# 사용법:
#   ./build_topas_gpu.sh            # 대화형 — 빌드 모드 선택 프롬프트
#   ./build_topas_gpu.sh full       # 전체   — vanilla OpenTOPAS 패치 + TOPAS 빌드 + GPU 엔진 + 설치
#   ./build_topas_gpu.sh wrapper    # 엔진만 — dylib/metallib 재빌드 + 래퍼 갱신 (TOPAS 재빌드 X)
#   MODE=wrapper ./build_topas_gpu.sh
#
# 경로는 자동 탐색. Geant4/OpenTOPAS 설치가 여러 개면 대화형에서 방향키(↑/↓)로 선택. override 가능:
#   Geant4_DIR / TOPAS_SRC / TOPAS_INSTALL / TOPAS_BUILD / GDCM_DIR / TOPAS_G4_DATA_DIR
# Qt6 사용 (TOPAS 4.2.3+ 는 Qt6 필수). 강제: TOPAS_QT=6
# 색 끄기: NO_COLOR=1
# ============================================================
set -e
export PATH="/opt/homebrew/bin:$PATH"

# ---- 터미널 스타일 (비-TTY 또는 NO_COLOR 시 자동 비활성) ----
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    B=$'\033[1m'; D=$'\033[2m'; R=$'\033[0m'
    CY=$'\033[36m'; GR=$'\033[32m'; YL=$'\033[33m'; RD=$'\033[31m'; BL=$'\033[34m'
else
    B=; D=; R=; CY=; GR=; YL=; RD=; BL=
fi
ok()   { printf "   ${GR}✓${R} %s\n" "$*"; }
warn() { printf "   ${YL}⚠${R} %s\n" "$*"; }
note() { printf "     ${D}%s${R}\n" "$*"; }
die()  { printf "   ${RD}✗ %s${R}\n" "$*"; exit 1; }
step() { printf "\n  ${B}${BL}▸ %s${R}\n" "$*"; }
# 로그파일($1)에서 첫 컴파일 에러들을 골라 보여준다 (tail 만으론 에러줄이 스크롤돼 안 보이므로).
show_log_errors() {
    local _log="$1" _hits
    [ -f "$_log" ] || return 0
    _hits="$(grep -nE 'error:|fatal error|undefined symbol|ld: |Error [0-9]+' "$_log" 2>/dev/null | head -8)"
    if [ -n "$_hits" ]; then
        printf "   ${RD}↳ 로그에서 추출한 에러:${R}\n"
        printf "%s\n" "$_hits" | sed "s/^/       ${D}/; s/\$/${R}/"
    fi
}
# 점(.)으로 구분된 버전 비교: $1 >= $2 면 0(true). MAJOR.MINOR.PATCH 만 본다.
# (macOS /bin/bash 3.2 호환 — 인덱스 배열 O, 연관배열/-A 사용 안 함)
_ver_ge() {
    local IFS=. i a b; local -a A B
    read -ra A <<< "$1"; read -ra B <<< "$2"
    for i in 0 1 2; do
        a="${A[i]:-0}"; b="${B[i]:-0}"
        a="${a//[!0-9]/}"; b="${b//[!0-9]/}"; a="${a:-0}"; b="${b:-0}"
        [ "$a" -gt "$b" ] && return 0
        [ "$a" -lt "$b" ] && return 1
    done
    return 0   # 완전히 같음 → >= 성립
}
banner() {
    printf "\n${B}${CY}  ╔══════════════════════════════════════════════════════════╗${R}\n"
    printf   "${B}${CY}  ║${R}   ${B}MetalOpticalPhoton  ·  TOPAS GPU Optical Engine${R}        ${B}${CY}║${R}\n"
    printf   "${B}${CY}  ╚══════════════════════════════════════════════════════════╝${R}\n"
    printf   "   ${D}Apple Metal · macOS Apple Silicon · 빌드/설치${R}\n"
}

# ---- 경로 유효성 검사 + (대화형) 보충 입력 ----
_path_ok() {  # $1=값  $2=종류(dir|topassrc|topasinstall)
    case "$2" in
        dir)          [ -n "$1" ] && [ -d "$1" ] ;;
        topassrc)     [ -n "$1" ] && [ -f "$1/CMakeLists.txt" ] ;;
        topasinstall) [ -n "$1" ] && [ -x "$1/bin/topas" ] ;;
        *)            [ -n "$1" ] ;;
    esac
}
# 자동탐색이 실패했으면: 대화형이면 사용자에게 묻고, 비대화형이면 에러.
# 결과를 $1 으로 지정한 변수에 다시 써넣는다.
ensure_path() {  # $1=변수명  $2=라벨  $3=종류  $4=예시
    local _n="$1" _label="$2" _kind="$3" _hint="$4" _v
    _v="${!_n}"
    while ! _path_ok "$_v" "$_kind"; do
        if [ -t 0 ]; then
            [ -n "$_v" ] && printf "   ${YL}⚠${R} %s 경로가 유효하지 않음: '%s'\n" "$_label" "$_v"
            printf "   ${YL}?${R} %s 자동탐색 실패 — 경로를 입력하세요 ${D}(예: %s)${R}\n     > " "$_label" "$_hint"
            read -r _v
            _v="${_v/#\~/$HOME}"
        else
            die "$_label 못 찾음 (자동탐색 실패). 환경변수로 지정하세요. 예: $_hint"
        fi
    done
    printf -v "$_n" '%s' "$_v"
}

# ---- 경로 (스크립트 기준 상대경로 — 항상 정확) ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOP_DIR="$SCRIPT_DIR"
MOP_BUILD_DIR="$MOP_DIR/build"
MOP_INCLUDE_DIR="$MOP_DIR/include"
MOP_SRC_DIR="$MOP_DIR/src"
MOP_SHADER_DIR="$MOP_DIR/shaders"
MOP_EXT_DIR="$MOP_DIR/topas_extension"

# ============================================================
# 외부 의존성 자동 탐색 함수
#   env(Geant4_DIR/TOPAS_SRC/TOPAS_INSTALL/TOPAS_BUILD/GDCM_DIR/TOPAS_G4_DATA_DIR)가
#   설정돼 있으면 ${VAR:-$(detect)} 로 그 값을 우선 사용(탐색 함수 호출 안 됨).
#   없을 때만 흔한 위치를 자동 탐색 → 새 Mac 도 git clone 후 그대로 실행 가능.
# ============================================================
detect_geant4_dir() {
    # 1) geant4-config (PATH 에 있으면 가장 정확)
    if command -v geant4-config >/dev/null 2>&1; then
        local p; p="$(geant4-config --prefix 2>/dev/null)"
        [ -n "$p" ] && [ -d "$p" ] && { echo "$p"; return 0; }
    fi
    # 2) 기존 설치의 topas-gpu 래퍼에 박힌 Geant4 경로 (이전 빌드값 — 비표준 위치도 회수)
    local _inst _wrap _g4 _cache _cdir
    _inst="$(detect_topas_install)"; _wrap="$_inst/bin/topas-gpu"
    if [ -n "$_inst" ] && [ -f "$_wrap" ]; then
        _g4="$(grep 'DYLD_LIBRARY_PATH=' "$_wrap" 2>/dev/null | grep -v 'SCRIPT_DIR' | sed -nE 's#.*="([^"]*)/lib:.*#\1#p' | head -1)"
        [ -n "$_g4" ] && [ -d "$_g4" ] && { echo "$_g4"; return 0; }
    fi
    # 3) 기존 TOPAS build 의 CMakeCache 의 Geant4_DIR → prefix 로 거슬러 올라감
    for _cache in /Applications/TOPAS/OpenTOPAS-build*/CMakeCache.txt; do
        [ -f "$_cache" ] || continue
        _cdir="$(grep -E '^Geant4_DIR(:[A-Za-z]+)?=' "$_cache" 2>/dev/null | head -1 | cut -d= -f2)"
        while [ -n "$_cdir" ] && [ "$_cdir" != "/" ] && [ "$_cdir" != "." ]; do
            { [ -d "$_cdir/include/Geant4" ] || [ -x "$_cdir/bin/geant4-config" ]; } && { echo "$_cdir"; return 0; }
            _cdir="$(dirname "$_cdir")"
        done
    done
    # 4) 흔한 설치 위치 (Geant4Config.cmake 존재로 검증)
    local c
    for c in \
        /Applications/GEANT4/geant4-install \
        /Applications/GEANT4/*install* \
        /opt/homebrew/opt/geant4 /opt/homebrew/Cellar/geant4/* \
        /usr/local/opt/geant4 /usr/local/geant4* \
        "$HOME"/geant4* "$HOME"/GEANT4/*install*; do
        [ -d "$c" ] || continue
        if find "$c" -maxdepth 4 -name Geant4Config.cmake 2>/dev/null | grep -q .; then
            echo "$c"; return 0
        fi
    done
    return 0
}

# TOPAS_SRC 의 버전(MAJOR.MINOR.PATCH) — CMakeLists 우선, git describe 폴백 (Qt 자동선택용)
detect_topas_version() {
    local v="" vM vm vp tv
    if [ -f "$TOPAS_SRC/CMakeLists.txt" ]; then
        vM=$(grep -E 'set ?\(TOPAS_VERSION_MAJOR' "$TOPAS_SRC/CMakeLists.txt" 2>/dev/null | grep -oE '[0-9]+' | head -1)
        vm=$(grep -E 'set ?\(TOPAS_VERSION_MINOR' "$TOPAS_SRC/CMakeLists.txt" 2>/dev/null | grep -oE '[0-9]+' | head -1)
        vp=$(grep -E 'set ?\(TOPAS_VERSION_PATCH' "$TOPAS_SRC/CMakeLists.txt" 2>/dev/null | grep -oE '[0-9]+' | head -1)
        [ -n "$vM" ] && v="${vM}.${vm:-0}.${vp:-0}"
    fi
    if [ -z "$v" ] && git -C "$TOPAS_SRC" rev-parse >/dev/null 2>&1; then
        tv="$(git -C "$TOPAS_SRC" describe --tags 2>/dev/null)"; tv="${tv#v}"; v="${tv%%-*}"
    fi
    echo "$v"
}

detect_topas_src() {
    local c
    for c in \
        /Applications/TOPAS/OpenTOPAS /Applications/TOPAS/topas \
        /Applications/OpenTOPAS "$HOME"/OpenTOPAS "$HOME"/topas \
        /opt/OpenTOPAS /opt/topas /Applications/*/OpenTOPAS; do
        if [ -f "$c/CMakeLists.txt" ] && grep -qi topas "$c/CMakeLists.txt" 2>/dev/null; then
            echo "$c"; return 0
        fi
    done
    return 0
}

# 기존 TOPAS 설치(install) 디렉토리 탐색 — bin/topas 가 있어야 (wrapper 모드용)
detect_topas_install() {
    local c
    for c in \
        /Applications/TOPAS/OpenTOPAS-install-gpu \
        /Applications/TOPAS/OpenTOPAS-install \
        /Applications/TOPAS/*install* \
        "$HOME"/OpenTOPAS-install* /opt/OpenTOPAS-install*; do
        [ -x "$c/bin/topas" ] && { echo "$c"; return 0; }
    done
    return 0
}

detect_gdcm_dir() {
    local c bp
    # 0) 기존 topas 바이너리가 실제 링크한 GDCM → cmake config dir(lib/gdcm-*) 회수 (가장 정확)
    local _inst _dep _libdir
    _inst="$(detect_topas_install)"
    if [ -n "$_inst" ] && [ -x "$_inst/bin/topas" ] && command -v otool >/dev/null 2>&1; then
        _dep="$(otool -L "$_inst/bin/topas" 2>/dev/null | grep -oE '/[^ ]*/libgdcm[A-Za-z]+\.[0-9.]+\.dylib' | head -1)"
        if [ -n "$_dep" ]; then
            _libdir="${_dep%/*}"
            for c in "$_libdir"/gdcm-*; do [ -d "$c" ] && { echo "$c"; return 0; }; done
        fi
    fi
    if command -v brew >/dev/null 2>&1; then
        bp="$(brew --prefix gdcm 2>/dev/null)"
        for c in "$bp"/lib/gdcm-*; do
            [ -d "$c" ] && { echo "$c"; return 0; }
        done
    fi
    for c in \
        /opt/homebrew/opt/gdcm/lib/gdcm-* \
        /usr/local/opt/gdcm/lib/gdcm-* \
        /Applications/GDCM/gdcm-install/lib/gdcm-* \
        /Applications/GDCM/gdcm-install; do
        [ -d "$c" ] && { echo "$c"; return 0; }
    done
    for c in /Applications/TOPAS/OpenTOPAS-build*/CMakeCache.txt; do
        [ -f "$c" ] || continue
        local cached; cached="$(grep '^GDCM_DIR:PATH=' "$c" 2>/dev/null | cut -d= -f2)"
        [ -d "$cached" ] && { echo "$cached"; return 0; }
    done
    return 0
}

detect_g4_data_dir() {
    # 0) 기존 topas-gpu 래퍼에 박힌 TOPAS_G4_DATA_DIR (이전 빌드값)
    local _inst _wrap _g4d
    _inst="$(detect_topas_install)"; _wrap="$_inst/bin/topas-gpu"
    if [ -n "$_inst" ] && [ -f "$_wrap" ]; then
        _g4d="$(sed -nE 's/.*TOPAS_G4_DATA_DIR="([^"]*)".*/\1/p' "$_wrap" 2>/dev/null | head -1)"
        [ -n "$_g4d" ] && [ -d "$_g4d" ] && { echo "$_g4d"; return 0; }
    fi
    local c
    for c in \
        "$GEANT4_DATA_DIR" \
        /Applications/GEANT4/G4DATA \
        "$GEANT4_DIR"/share/Geant4*/data "$GEANT4_DIR"/share/geant4*/data \
        "$GEANT4_DIR"/data "$GEANT4_DIR"/../G4DATA; do
        [ -d "$c" ] && ls "$c" 2>/dev/null | grep -qi '^G4' && { echo "$c"; return 0; }
    done
    if command -v geant4-config >/dev/null 2>&1; then
        local first; first="$(geant4-config --datasets 2>/dev/null | head -1 | awk '{print $NF}')"
        [ -n "$first" ] && [ -d "$first" ] && { dirname "$first"; return 0; }
    fi
    return 0
}

# ---- 후보 스캔 + 방향키 선택 (Geant4 / OpenTOPAS 소스가 여러 개일 때) ----
# 각 scan_* 는 "경로<TAB>버전" 줄을 후보별로 출력.
scan_geant4_candidates() {
    local c p ver
    {
        command -v geant4-config >/dev/null 2>&1 && geant4-config --prefix 2>/dev/null
        for c in \
            /Applications/GEANT4/*install* \
            /opt/homebrew/opt/geant4 /opt/homebrew/Cellar/geant4/* \
            /usr/local/opt/geant4 /usr/local/geant4* \
            "$HOME"/geant4*install* "$HOME"/GEANT4/*install*; do
            { [ -x "$c/bin/geant4-config" ] || [ -f "$c/lib/cmake/Geant4/Geant4Config.cmake" ]; } && echo "$c"
        done
    } | sort -u | while read -r p; do
        [ -n "$p" ] || continue
        if [ -x "$p/bin/geant4-config" ]; then ver="$("$p/bin/geant4-config" --version 2>/dev/null)"
        else ver="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' "$p/lib/cmake/Geant4/Geant4ConfigVersion.cmake" 2>/dev/null | head -1)"; fi
        printf '%s\t%s\n' "$p" "${ver:-?}"
    done
}
scan_topas_candidates() {
    local c ver
    for c in \
        /Applications/TOPAS/OpenTOPAS /Applications/TOPAS/OpenTOPAS-* \
        /Applications/TOPAS/topas /Applications/OpenTOPAS \
        "$HOME"/OpenTOPAS* "$HOME"/topas /opt/OpenTOPAS /opt/topas; do
        # 소스 트리만 (install/build 디렉터리 제외): 핵심 소스파일 존재로 판별
        { [ -f "$c/CMakeLists.txt" ] && [ -f "$c/geometry/TsBox.cc" ]; } || continue
        ver=""
        [ -d "$c/.git" ] && ver="$(git -C "$c" describe --tags 2>/dev/null)"
        [ -n "$ver" ] || ver="$(grep -E 'set ?\(TOPAS_VERSION' "$c/CMakeLists.txt" 2>/dev/null | grep -oE '[0-9]+' | head -3 | paste -sd. -)"
        printf '%s\t%s\n' "$c" "${ver:-?}"
    done | sort -u
}
# 방향키 메뉴: pick_menu "제목" "라벨0" "라벨1" ...  → 선택 인덱스를 PICK_INDEX 에.
pick_menu() {
    local title="$1"; shift
    local n=$# sel=0 key esc i first=1
    local -a opts; opts=("$@")
    PICK_INDEX=0
    { [ "$n" -le 1 ] || [ ! -t 0 ] || [ ! -t 1 ]; } && return 0
    printf "  ${B}%s${R}  ${D}(↑/↓ 또는 숫자, Enter 확정)${R}\n" "$title"
    printf '\033[?25l' 2>/dev/null || true
    while true; do
        if [ "$first" -eq 1 ]; then first=0; else printf "\033[%dA" "$n"; fi
        for ((i=0; i<n; i++)); do
            if [ "$i" -eq "$sel" ]; then printf "\r    ${GR}▶ %s${R}\033[K\n" "${opts[$i]}"
            else printf "\r      ${D}%s${R}\033[K\n" "${opts[$i]}"; fi
        done
        IFS= read -rsn1 key 2>/dev/null || true
        if [ "$key" = $'\033' ]; then
            read -rsn2 -t 1 esc 2>/dev/null || true
            case "$esc" in '[A') sel=$(( (sel - 1 + n) % n ));; '[B') sel=$(( (sel + 1) % n ));; esac
        elif [ -z "$key" ]; then break
        elif printf '%s' "$key" | grep -q '^[1-9]$' && [ "$key" -le "$n" ]; then sel=$((key - 1)); break
        fi
    done
    printf '\033[?25h' 2>/dev/null || true
    PICK_INDEX=$sel
}
# env 값 있으면 그대로. 없고 대화형+후보 2개 이상이면 방향키 선택. 그 외엔 빈값(호출부가 기존 detect 사용).
choose_from_scan() {
    local envval="$1" scan="$2" title="$3"
    CHOSEN_PATH=""
    [ -n "$envval" ] && { CHOSEN_PATH="$envval"; return 0; }
    { [ -t 0 ] && [ -t 1 ]; } || return 0
    [ -n "$scan" ] || return 0
    local -a paths labels; local p v cnt=0
    while IFS="$(printf '\t')" read -r p v; do
        [ -n "$p" ] || continue
        v="${v#v}"; paths[$cnt]="$p"; labels[$cnt]="$(printf 'v%-18s %s' "$v" "$p")"; cnt=$((cnt + 1))
    done <<< "$scan"
    [ "$cnt" -ge 2 ] || return 0
    pick_menu "$title" "${labels[@]}"
    CHOSEN_PATH="${paths[$PICK_INDEX]}"
}

banner

# ============================================================
# 빌드 모드 선택: full / wrapper
#   인자 → MODE env → (대화형 프롬프트) → 비대화형 기본 full
# ============================================================
MODE="${1:-${MODE:-}}"
if [ -z "$MODE" ]; then
    if [ -t 0 ]; then
        printf "\n  ${B}설치 모드를 선택하세요${R}\n\n"
        printf "    ${B}${GR}f${R}  ${B}전체 빌드${R}    ${D}vanilla OpenTOPAS 패치 + TOPAS 빌드 + GPU 엔진 + 설치${R}\n"
        printf "          ${D}처음 설치 / TOPAS·확장 C++ 변경 시${R}\n\n"
        printf "    ${B}${CY}w${R}  ${B}GPU 엔진만${R}   ${D}dylib·metallib 재빌드 + 래퍼 갱신 (TOPAS 재빌드 안 함)${R}\n"
        printf "          ${D}엔진/셰이더만 수정했을 때 — 훨씬 빠름${R}\n\n"
        printf "  ${B}선택${R} ${D}[f/w]${R} (기본 f): "
        read -r _ans
        case "$_ans" in w|W|wrapper|engine) MODE=wrapper ;; *) MODE=full ;; esac
    else
        MODE=full   # 비대화형(파이프/CI) + 미지정 → 안전하게 전체 빌드
    fi
fi
case "$MODE" in
    full|f|FULL)              MODE=full ;;
    wrapper|w|WRAPPER|engine) MODE=wrapper ;;
    *) die "MODE 는 full | wrapper 만 가능 (받은 값: '$MODE')" ;;
esac
if [ "$MODE" = full ]; then
    printf "\n  ${B}▶ 모드:${R} ${GR}전체 빌드${R} ${D}(TOPAS 패치+빌드 + GPU 엔진 + 설치)${R}\n"
else
    printf "\n  ${B}▶ 모드:${R} ${CY}GPU 엔진만${R} ${D}(dylib/metallib + 래퍼만, TOPAS 재빌드 없음)${R}\n"
fi

# ---- 경로 결정: env 우선, 없으면 자동탐색 (모드별 필요한 것만) ----
# Geant4: 후보 2개 이상이면 방향키 선택 (env Geant4_DIR 있으면 그대로, 비대화형/1개면 기존 자동탐색)
choose_from_scan "${Geant4_DIR:-}" "$(scan_geant4_candidates)" "Geant4 설치를 선택하세요"
GEANT4_DIR="${CHOSEN_PATH:-$(detect_geant4_dir)}"
G4_DATA_DIR="${TOPAS_G4_DATA_DIR:-$(detect_g4_data_dir)}"
# 선택한 Geant4 에 번들 데이터가 있으면 버전 일치 위해 우선 (env override 없을 때만)
if [ -z "${TOPAS_G4_DATA_DIR:-}" ]; then
    for _gd in "$GEANT4_DIR"/share/Geant4*/data "$GEANT4_DIR"/share/geant4*/data; do
        [ -d "$_gd" ] && ls "$_gd" 2>/dev/null | grep -qi '^G4' && { G4_DATA_DIR="$_gd"; break; }
    done
fi
# Geant4 버전 floor: 11.3.2 이상만 지원
GEANT4_REQUIRED="11.3.2"
_g4ver=""
if [ -x "$GEANT4_DIR/bin/geant4-config" ]; then
    _g4ver="$("$GEANT4_DIR/bin/geant4-config" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
elif command -v geant4-config >/dev/null 2>&1; then
    _g4ver="$(geant4-config --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
fi
[ -z "$_g4ver" ] && _g4ver="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' "$GEANT4_DIR"/lib/cmake/Geant4/Geant4ConfigVersion.cmake 2>/dev/null | head -1)"
if [ -z "$_g4ver" ]; then
    warn "Geant4 버전 확인 불가 — >= ${GEANT4_REQUIRED} 필수. 수동 확인하세요."
elif _ver_ge "$_g4ver" "$GEANT4_REQUIRED"; then
    ok "Geant4 $_g4ver (>= ${GEANT4_REQUIRED})"
else
    die "Geant4 $_g4ver 미지원 — >= ${GEANT4_REQUIRED} 필수."
fi
if [ "$MODE" = full ]; then
    # OpenTOPAS 소스: 후보 2개 이상이면 방향키 선택 (env TOPAS_SRC 있으면 그대로)
    choose_from_scan "${TOPAS_SRC:-}" "$(scan_topas_candidates)" "OpenTOPAS 소스를 선택하세요"
    TOPAS_SRC="${CHOSEN_PATH:-$(detect_topas_src)}"
    TOPAS_VER="$(detect_topas_version)"
    GDCM_DIR="${GDCM_DIR:-$(detect_gdcm_dir)}"
    # TOPAS_BUILD/INSTALL 파생은 TOPAS_SRC 확정(Step1, 프롬프트 가능) 후로 미룸
else
    TOPAS_INSTALL="${TOPAS_INSTALL:-$(detect_topas_install)}"
fi
NPROC=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
JOBS="${JOBS:-$NPROC}"   # 컴파일 병렬 job 수 (make -j). 기본 = 전체 코어. 예: JOBS=4 ./build_topas_gpu.sh

# 시작 단계 — 컴파일이 실패하면 그 단계부터 재시도(이전 산출물 재사용). step1 은 항상 실행.
#   인자2 또는 START env. 값: 2|3|4|5 (shaders|engine|topas|install)
START="${START:-${2:-2}}"
case "$START" in
    2|shaders)      START=2 ;;
    3|engine|dylib) START=3 ;;
    4|topas|build)  START=4 ;;
    5|install)      START=5 ;;
    *) die "START 는 2|3|4|5 (shaders|engine|topas|install). 받은 값: '$START'" ;;
esac

# 실패 시 "그 단계부터 재시도" 명령을 안내 (컴파일은 자주 실패하므로)
_CUR_STEP=1
_finish() {
    local code=$?
    { [ "$code" -eq 0 ] || [ "${_CUR_STEP:-1}" -lt 2 ]; } && return
    printf "\n   ${RD}✗ 단계 %s 에서 실패 (exit %s)${R}\n" "$_CUR_STEP" "$code"
    printf "   ${D}고친 뒤 그 단계부터 재시도:${R}  ${B}START=%s \"%s\" %s${R}\n" "$_CUR_STEP" "$0" "$MODE"
    [ "$_CUR_STEP" -ge 4 ] && printf "   ${D}cmake 재구성 생략하고 make 만:${R}  ${B}START=4 SKIP_CMAKE=1 \"%s\" %s${R}\n" "$0" "$MODE"
}
trap _finish EXIT

[ "$START" -ne 2 ] && printf "  ${B}▶ 시작 단계:${R} %s ${D}(이전 단계 산출물 재사용)${R}\n" "$START"
[ "$JOBS" != "$NPROC" ] && printf "  ${B}▶ 컴파일 job:${R} %s ${D}(전체 %s 코어 중)${R}\n" "$JOBS" "$NPROC"

# ============================================================
step "1 · 환경 점검"
# ============================================================
command -v xcrun >/dev/null 2>&1 || die "xcrun 없음 — Xcode CLT 설치: sudo xcode-select --install"
xcrun metal --version >/dev/null 2>&1 || die $'Metal 컴파일러 없음 — Xcode + Metal 툴체인 필요 (CLT 만으론 부족, Xcode 26+ 는 별도 다운로드).\n     sudo xcode-select -s /Applications/Xcode.app/Contents/Developer\n     sudo xcodebuild -license accept && xcodebuild -runFirstLaunch\n     xcodebuild -downloadComponent MetalToolchain   # ~700MB (재부팅 후 풀리면 같은 명령으로 재마운트)\n     자세히: INSTALL.md \xc2\xa71-1'
ok "Metal 컴파일러: $(xcrun metal --version 2>&1 | head -1)"

ensure_path GEANT4_DIR "Geant4 설치 prefix" dir "/Applications/GEANT4/geant4-install"
ok "Geant4: $GEANT4_DIR"

# G4 데이터(G4DATA)는 런타임용 — 못 찾아도 빌드는 됨. 대화형이면 보충 입력(Enter 로 건너뜀).
if [ -d "$G4_DATA_DIR" ]; then
    ok "G4 데이터: $G4_DATA_DIR"
elif [ -t 0 ]; then
    printf "   ${YL}?${R} G4 데이터(G4DATA) 자동탐색 실패 — 경로 입력 ${D}(없으면 Enter, 나중에 wrapper 수정 가능)${R}\n     > "
    read -r G4_DATA_DIR; G4_DATA_DIR="${G4_DATA_DIR/#\~/$HOME}"
    if [ -d "$G4_DATA_DIR" ]; then ok "G4 데이터: $G4_DATA_DIR"; else warn "G4DATA 미설정 — 실행 시 TOPAS_G4_DATA_DIR 필요"; fi
else
    warn "G4 데이터(G4DATA) 자동탐색 실패 — 실행 시 TOPAS_G4_DATA_DIR 확인 필요"
fi

if [ "$MODE" = wrapper ]; then
    # ---- wrapper 모드: 이미 빌드된 TOPAS 설치만 있으면 됨 ----
    ensure_path TOPAS_INSTALL "기존 TOPAS 설치(install)" topasinstall "/Applications/TOPAS/OpenTOPAS-install-gpu"
    ok "기존 TOPAS 설치: $TOPAS_INSTALL"
    note "엔진 dylib/metallib + 래퍼만 교체 (TOPAS 재빌드/패치 없음)"
else
    # ---- full 모드: TOPAS 빌드용 cmake / GDCM / 소스 + 패치 ----
    command -v cmake >/dev/null 2>&1 || die "cmake 없음 — brew install cmake"
    ok "CMake: $(cmake --version | head -1)"

    ensure_path GDCM_DIR "GDCM (lib/gdcm-*)" dir "/opt/homebrew/opt/gdcm/lib/gdcm-3.2"
    ok "GDCM: $GDCM_DIR"

    ensure_path TOPAS_SRC "OpenTOPAS 소스" topassrc "/Applications/TOPAS/OpenTOPAS"
    ok "TOPAS 소스: $TOPAS_SRC"
    # TOPAS_SRC 가 프롬프트로 바뀌었을 수 있으니 출력경로 파생 (env 로 직접 준 값은 유지)
    TOPAS_BUILD="${TOPAS_BUILD:-${TOPAS_SRC}-build-gpu}"
    TOPAS_INSTALL="${TOPAS_INSTALL:-${TOPAS_SRC}-install-gpu}"

    # ---- TOPAS 소스 패치 자동 적용 (vanilla OpenTOPAS → 정상 동작) ----
    # 패치 없이 빌드하면 optical fluence 4x over + parallel-world abort 발생.
    #   primary/TsVGenerator.cc : polarization (Opticks식 cross-product) + continuous-spectrum zero-bin fix
    #   geometry/Ts{Box,Cylinder,Sphere}.cc : PW envelope 1nm auto-shrink (+ TsSphere R-div fix)
    TOPAS_PATCH="$MOP_DIR/topas_patches/opentopas_local.patch"   # vanilla 단일 패치 (TOPAS 4.2.3 기준)
    TOPAS_REQUIRED="4.2.3"
    PATCH_MARKER_FILES="primary/TsVGenerator.cc geometry/TsBox.cc geometry/TsCylinder.cc geometry/TsSphere.cc"
    if [ -f "$TOPAS_PATCH" ]; then
        # 버전 탐지: CMakeLists(가장 신뢰, git 불필요) → git describe 폴백.
        _tvnum=""
        if [ -f "$TOPAS_SRC/CMakeLists.txt" ]; then
            _vM=$(grep -E 'set ?\(TOPAS_VERSION_MAJOR' "$TOPAS_SRC/CMakeLists.txt" 2>/dev/null | grep -oE '[0-9]+' | head -1)
            _vm=$(grep -E 'set ?\(TOPAS_VERSION_MINOR' "$TOPAS_SRC/CMakeLists.txt" 2>/dev/null | grep -oE '[0-9]+' | head -1)
            _vp=$(grep -E 'set ?\(TOPAS_VERSION_PATCH' "$TOPAS_SRC/CMakeLists.txt" 2>/dev/null | grep -oE '[0-9]+' | head -1)
            [ -n "$_vM" ] && _tvnum="${_vM}.${_vm:-0}.${_vp:-0}"
        fi
        if [ -z "$_tvnum" ] && git -C "$TOPAS_SRC" rev-parse >/dev/null 2>&1; then
            _tv="$(git -C "$TOPAS_SRC" describe --tags 2>/dev/null)"
            _tvnum="${_tv#v}"; _tvnum="${_tvnum%%-*}"   # v4.2.3-2-g… → 4.2.3
        fi
        if [ -z "$_tvnum" ]; then
            warn "OpenTOPAS 버전 확인 불가 — >= ${TOPAS_REQUIRED} 필수. 수동 확인하세요."
        elif _ver_ge "$_tvnum" "$TOPAS_REQUIRED"; then
            ok "OpenTOPAS $_tvnum (>= ${TOPAS_REQUIRED})"
        else
            die "OpenTOPAS $_tvnum 미지원 — >= ${TOPAS_REQUIRED} 필수 (패치는 4.2.3 기준)."
        fi
        # 이미 적용? geometry(TsBox) + polarization(TsVGenerator) 마커 둘 다 있어야.
        if grep -q "MetalOpticalPhoton" "$TOPAS_SRC/geometry/TsBox.cc" 2>/dev/null \
           && grep -q "MetalOpticalPhoton" "$TOPAS_SRC/primary/TsVGenerator.cc" 2>/dev/null; then
            ok "TOPAS 소스 패치 이미 적용됨 (skip)"
        elif patch -p1 -d "$TOPAS_SRC" --forward --dry-run < "$TOPAS_PATCH" >/dev/null 2>&1; then
            patch -p1 -d "$TOPAS_SRC" --forward < "$TOPAS_PATCH" >/dev/null
            _applied=0
            for _f in $PATCH_MARKER_FILES; do
                grep -q "MetalOpticalPhoton" "$TOPAS_SRC/$_f" 2>/dev/null && _applied=$((_applied+1))
            done
            if [ "$_applied" -eq 4 ]; then
                ok "TOPAS 소스 패치 적용 + 검증 (4/4 파일)"
            else
                warn "패치 부분 적용($_applied/4) — TOPAS 버전 확인 + .rej 파일 점검"
            fi
        else
            warn "TOPAS 소스 패치 자동 적용 실패 — context 불일치(미지원 버전?), 잘못 적용 대신 중단"
            note "수동 검토: patch -p1 -d \"$TOPAS_SRC\" < \"$TOPAS_PATCH\"  (실패 hunk 는 .rej)"
            note "패치 없이 빌드하면 optical fluence 4x over + parallel-world abort 버그"
        fi
    else
        warn "TOPAS 패치 파일 없음 ($TOPAS_PATCH) — 소스가 이미 패치됐다고 가정하고 진행"
    fi
fi

# ============================================================
# ============================================================
_CUR_STEP=2
mkdir -p "$MOP_BUILD_DIR"
if [ "$START" -le 2 ]; then
    step "2 · Metal 셰이더 컴파일"
    cd "$MOP_BUILD_DIR"
    SHADER_FILES=(
        "$MOP_SHADER_DIR/PhotonGeneration.metal"
        "$MOP_SHADER_DIR/OpticalPhotonKernel.metal"
        "$MOP_SHADER_DIR/DDAScoring.metal"
    )
    AIR_FILES=()
    for shader in "${SHADER_FILES[@]}"; do
        bn=$(basename "$shader" .metal)
        air_file="$MOP_BUILD_DIR/${bn}.air"
        note "compile $bn.metal → $bn.air"
        xcrun metal -c "$shader" -I "$MOP_INCLUDE_DIR" -I "$MOP_SHADER_DIR" -o "$air_file" -std=metal3.1
        AIR_FILES+=("$air_file")
    done
    xcrun metallib "${AIR_FILES[@]}" -o "$MOP_BUILD_DIR/default.metallib"
    ok "default.metallib 빌드 완료"
else
    step "2 · Metal 셰이더 — 건너뜀 (START=$START)"
    [ -f "$MOP_BUILD_DIR/default.metallib" ] || die "이전 metallib 없음 ($MOP_BUILD_DIR/default.metallib) — START 를 2 이하로 다시 실행"
    ok "기존 metallib 재사용"
fi

# ============================================================
# ============================================================
_CUR_STEP=3
if [ "$START" -le 3 ]; then
    step "3 · libMetalOpticalPhoton.dylib (GPU 엔진)"
    note "compile MetalOpticalEngine.mm"
    # -framework 는 링커 전용 플래그 — 컴파일(-c) 단계엔 불필요(링크 단계 아래에 있음).
    clang++ -std=c++17 -ObjC++ -fobjc-arc -O2 \
        -c "$MOP_SRC_DIR/MetalOpticalEngine.mm" \
        -I "$MOP_INCLUDE_DIR" -I "$MOP_EXT_DIR" \
        -o "$MOP_BUILD_DIR/MetalOpticalEngine.o"
    note "compile TopasParameterParser.cc"
    clang++ -std=c++17 -O2 \
        -c "$MOP_EXT_DIR/TopasParameterParser.cc" \
        -I "$MOP_EXT_DIR" -I "$MOP_INCLUDE_DIR" \
        -o "$MOP_BUILD_DIR/TopasParameterParser.o"
    note "link libMetalOpticalPhoton.dylib"
    clang++ -dynamiclib -O2 \
        "$MOP_BUILD_DIR/MetalOpticalEngine.o" \
        "$MOP_BUILD_DIR/TopasParameterParser.o" \
        -o "$MOP_BUILD_DIR/libMetalOpticalPhoton.dylib" \
        -framework Metal -framework MetalPerformanceShaders -framework Foundation \
        -install_name "@rpath/libMetalOpticalPhoton.dylib"
    ok "libMetalOpticalPhoton.dylib 빌드 완료"
else
    step "3 · GPU 엔진 — 건너뜀 (START=$START)"
    [ -f "$MOP_BUILD_DIR/libMetalOpticalPhoton.dylib" ] || die "이전 dylib 없음 ($MOP_BUILD_DIR/libMetalOpticalPhoton.dylib) — START 를 3 이하로 다시 실행"
    ok "기존 libMetalOpticalPhoton.dylib 재사용"
fi

# ============================================================
# ============================================================
_CUR_STEP=4
if [ "$MODE" = full ] && [ "$START" -le 4 ]; then
    step "4 · TOPAS + GPU Extension 빌드"
    mkdir -p "$TOPAS_BUILD"
    cd "$TOPAS_BUILD"
    export Geant4_DIR="$GEANT4_DIR"
    export GDCM_DIR="$GDCM_DIR"
    GEANT4_CMAKE_DIR=""
    for candidate in "$GEANT4_DIR/lib/cmake/Geant4" "$GEANT4_DIR/lib64/cmake/Geant4" "$GEANT4_DIR"; do
        if [ -f "$candidate/Geant4Config.cmake" ]; then GEANT4_CMAKE_DIR="$candidate"; break; fi
    done
    [ -z "$GEANT4_CMAKE_DIR" ] && GEANT4_CMAKE_DIR="$GEANT4_DIR"
    ok "Geant4 CMake: $GEANT4_CMAKE_DIR"
    # CXX_FLAGS 두 가지 — 둘 다 vanilla TOPAS 빌드엔 없고, 우리 셋업에 필수:
    #   (1) -include cmath : macOS SDK <math.h> 의 isinf 매크로가 libc++ <complex> 의
    #       std::isinf 와 충돌 → 강제 include 로 우회 (system header 는 안 건드림).
    #   (2) -iquote <include> <topas_extension> : TOPAS 의 CMakeHandleExtensions 는
    #       확장 .cc/.hh 를 build/extensions/ 로 "평탄화 복사" 한다. 그래서 wrapper 헤더
    #       (topas_extension/MOPTypes.hh, MetalOpticalEngine.hh) 안의 상대 include
    #       `#include "../include/MOPTypes.h"` 가 build/extensions/../include = build/include
    #       (존재 안 함) 를 가리켜 'file not found' 로 죽는다. -iquote 로 원본 트리의
    #       include/ · topas_extension/ 를 quoted-include 검색경로에 넣어 canonical 헤더로
    #       해결시킨다 (include/ 를 먼저 둬서 항상 정식 헤더가 선택되게 한다).
    MOP_INCLUDE_FLAGS="-iquote ${MOP_INCLUDE_DIR} -iquote ${MOP_EXT_DIR}"
    # Qt6 사용 (TOPAS 4.2.3+ 는 Qt6 필수 — 없으면 TsQt5 링크 실패). env TOPAS_QT 로 override 가능.
    QT_VER="${TOPAS_QT:-6}"
    QT_CMAKE_FLAGS=""
    if [ "$QT_VER" = 6 ]; then
        _qtp="$(brew --prefix qt 2>/dev/null || true)"
        [ -d "$_qtp" ] && export CMAKE_PREFIX_PATH="${_qtp}:${CMAKE_PREFIX_PATH:-}"
        QT_CMAKE_FLAGS="-DTOPAS_USE_QT=ON -DTOPAS_USE_QT6=ON"
        ok "Qt6 (${_qtp:-brew qt}) — TOPAS ${TOPAS_VER:-?} 용"
    else
        ok "Qt5 — TOPAS ${TOPAS_VER:-?} 용 (TOPAS_QT=5 강제)"
    fi
    if [ -n "${SKIP_CMAKE:-}" ] && [ -f "$TOPAS_BUILD/CMakeCache.txt" ]; then
        note "SKIP_CMAKE — cmake 재구성 생략 (기존 CMakeCache 사용)"
    else
        note "cmake 구성 중... (전체 로그: $TOPAS_BUILD/cmake.log)"
        cmake "$TOPAS_SRC" \
            -DCMAKE_INSTALL_PREFIX="$TOPAS_INSTALL" \
            -DCMAKE_BUILD_TYPE=Release \
            -DTOPAS_EXTENSIONS_DIR="$MOP_EXT_DIR" \
            -DGeant4_DIR="$GEANT4_CMAKE_DIR" \
            -DGDCM_DIR="$GDCM_DIR" \
            -DEXPAT_INCLUDE_DIR="$(xcrun --show-sdk-path)/usr/include" \
            -DZLIB_INCLUDE_DIR="$(xcrun --show-sdk-path)/usr/include" \
            -DEXPAT_LIBRARY="$(xcrun --show-sdk-path)/usr/lib/libexpat.tbd" \
            -DZLIB_LIBRARY_RELEASE="$(xcrun --show-sdk-path)/usr/lib/libz.tbd" \
            -DCMAKE_CXX_FLAGS="-include cmath ${MOP_INCLUDE_FLAGS}" \
            -DCMAKE_EXE_LINKER_FLAGS="-L${MOP_BUILD_DIR} -lMetalOpticalPhoton -Wl,-rpath,${MOP_BUILD_DIR} -framework Metal -framework Foundation -framework MetalPerformanceShaders" \
            -DCMAKE_OSX_ARCHITECTURES=arm64 \
            ${QT_CMAKE_FLAGS} \
            2>&1 | tee "$TOPAS_BUILD/cmake.log" | tail -30
        [ "${PIPESTATUS[0]}" -eq 0 ] || { warn "cmake 실패 — 전체 로그: $TOPAS_BUILD/cmake.log"; show_log_errors "$TOPAS_BUILD/cmake.log"; exit 1; }
    fi
    note "make -j${JOBS} ... (전체 로그: $TOPAS_BUILD/make.log)"
    make -j"$JOBS" 2>&1 | tee "$TOPAS_BUILD/make.log" | tail -15
    [ "${PIPESTATUS[0]}" -eq 0 ] || { warn "make 실패 — 전체 로그: $TOPAS_BUILD/make.log"; show_log_errors "$TOPAS_BUILD/make.log"; exit 1; }
    ok "TOPAS + GPU Extension 빌드 완료"
elif [ "$MODE" = full ]; then
    step "4 · TOPAS 빌드 — 건너뜀 (START=$START, 기존 $TOPAS_BUILD 재사용)"
else
    step "4 · TOPAS 빌드 — 건너뜀 (wrapper 모드)"
fi

# ============================================================
# ============================================================
_CUR_STEP=5
step "5 · 설치 + 실행 래퍼 생성"
if [ "$MODE" = full ]; then
    cd "$TOPAS_BUILD" 2>/dev/null || die "TOPAS build dir 없음: $TOPAS_BUILD — full 빌드를 START 낮춰 먼저 실행"
    make install 2>&1 | tee "$TOPAS_BUILD/install.log" | tail -3
    [ "${PIPESTATUS[0]}" -eq 0 ] || { warn "make install 실패 — 전체 로그: $TOPAS_BUILD/install.log"; show_log_errors "$TOPAS_BUILD/install.log"; exit 1; }
else
    note "wrapper 모드 — TOPAS make install 생략"
fi

# GPU 엔진(dylib + metallib) 을 설치 디렉토리에 복사 (양 모드 공통)
[ -f "$MOP_BUILD_DIR/libMetalOpticalPhoton.dylib" ] && [ -f "$MOP_BUILD_DIR/default.metallib" ] \
    || die "엔진 산출물 없음 ($MOP_BUILD_DIR) — START 를 낮춰 엔진(2·3)부터 빌드"
mkdir -p "$TOPAS_INSTALL/lib"
cp "$MOP_BUILD_DIR/libMetalOpticalPhoton.dylib" "$TOPAS_INSTALL/lib/"
cp "$MOP_BUILD_DIR/default.metallib" "$TOPAS_INSTALL/lib/"
ok "엔진 설치: $TOPAS_INSTALL/lib/{libMetalOpticalPhoton.dylib, default.metallib}"

# topas-gpu 래퍼 — 탐색된 G4_DATA_DIR / GEANT4_DIR 를 박아 넣음.
# (${...} 는 생성 시점 확장, \$... 는 실행 시점 확장으로 남김)
cat > "$TOPAS_INSTALL/bin/topas-gpu" << SCRIPT
#!/bin/bash
# TOPAS with Metal GPU Optical Photon Acceleration
SCRIPT_DIR="\$(cd "\$(dirname "\$0")/.." && pwd)"
export QT_QPA_PLATFORM_PLUGIN_PATH="\$SCRIPT_DIR/Frameworks"
export TOPAS_G4_DATA_DIR="${G4_DATA_DIR}"
export DYLD_LIBRARY_PATH="\$SCRIPT_DIR/lib:\${DYLD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="${GEANT4_DIR}/lib:\${DYLD_LIBRARY_PATH}"
export METAL_DEVICE_WRAPPER_TYPE=1  # Metal GPU 디버그 레이어 비활성화
export G4TRACE_DIR=OFF              # g4trace instrumentation runtime opt-out (CPU disk I/O 차단)
exec "\$SCRIPT_DIR/bin/topas" "\$@"
SCRIPT
chmod +x "$TOPAS_INSTALL/bin/topas-gpu"
ok "실행 래퍼: $TOPAS_INSTALL/bin/topas-gpu"

# ============================================================
printf "\n${B}${GR}  ╔══════════════════════════════════════════════════════════╗${R}\n"
printf   "${B}${GR}  ║${R}   ${B}✓ 빌드 완료${R}                                              ${B}${GR}║${R}\n"
printf   "${B}${GR}  ╚══════════════════════════════════════════════════════════╝${R}\n"
printf   "   ${D}모드:${R} %s\n" "$MODE"
printf   "   ${D}엔진:${R} $MOP_BUILD_DIR/{libMetalOpticalPhoton.dylib, default.metallib}\n"
printf   "   ${D}실행:${R} ${B}topas-gpu your_simulation.txt${R}\n"
printf   "   ${D}모듈 추가:${R} sv:Ph/Default/Modules = 3 \"g4em-standard_opt4\" \"g4optical\" \"gpuoptical\"\n\n"
