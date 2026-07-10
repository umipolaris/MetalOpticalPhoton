# topas_env.sh — shared resolver for the patched topas-gpu binary.
# Sourced by every benchmark runner:   . ../common/topas_env.sh   (runners cd to their own dir first).
#
# Resolution order (first match wins):
#   1. $TOPAS environment variable, if already set   — per-call override:  TOPAS=/path ./run_gpu.sh
#   2. `topas-gpu` found on $PATH                     — works out-of-the-box for a standard install
#   3. the default install path below                — EDIT THIS ONE LINE to change it globally
#
# NOTE: must be the *patched* topas-gpu (unpatched over-counts optical fluence ~4x).
: "${TOPAS:=$(command -v topas-gpu || echo /Applications/TOPAS/OpenTOPAS-install-gpu/bin/topas-gpu)}"
export TOPAS
[ -x "$TOPAS" ] || echo "WARN: topas-gpu not found at '$TOPAS' — set TOPAS=/path/to/topas-gpu, add it to PATH, or edit benchmark/common/topas_env.sh" >&2
