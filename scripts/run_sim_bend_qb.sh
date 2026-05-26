#!/usr/bin/env bash
# run_sim_bend_qb.sh — launch a PolyFEM bending simulation with a chosen material model.
#
# USAGE
#   ./scripts/run_sim_bend_qb.sh [JSON] [--iso | --fibered | --layered]
#
#   JSON          Path to the simulation JSON (default: kigumi-bending.json).
#                 Relative paths are resolved from the scripts/ directory.
#
# MATERIAL FLAGS  (default: --iso)
#   --iso         Linear-elastic isotropic  (HookeLinearElasticity, E/nu only).
#                 Fast; good for sanity-checks.
#   --fibered     Saint-Venant transversely isotropic.
#                 Stiff along X (grain direction); weaker in the YZ plane.
#                 Captures the dominant along-grain / across-grain contrast.
#   --layered     Saint-Venant orthotropic with three distinct principal stiffnesses.
#                 XY-plane layers are stiff; Z (through-thickness) is weak.
#                 Best match for laminated / ring-porous wood; use for Tsai-Wu analysis.
#
# EXAMPLES
#   ./scripts/run_sim_bend_qb.sh                          # iso, default JSON
#   ./scripts/run_sim_bend_qb.sh kigumi-bending.json --layered
#   ./scripts/run_sim_bend_qb.sh kigumi-bending.json --fibered
#
# OUTPUT
#   Results are written to a timestamped folder inside:
#     ~/Library/CloudStorage/.../kigumi-project/simulations/bending/
#   Folder name format: MMDD_HHMM_<material>_simulation
#
# DEPENDENCIES
#   - PolyFEM binary at ~/Research/polyfem/build/PolyFEM_bin
#   - jq  (used to patch the materials block before passing JSON to PolyFEM)
#   - Material snippets: scripts/materials-{iso,fibered,layered}.json
set -euo pipefail

# スクリプトの場所を基準に相対パスを解決する
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# 引数でJSONファイルを指定（指定がなければデフォルト）
JSON_ARG="${1:-kigumi-bending.json}"
if [[ "${JSON_ARG}" = /* ]]; then
    JSON_FILE="${JSON_ARG}"
else
    JSON_FILE="${SCRIPT_DIR}/${JSON_ARG}"
fi

if [[ ! -f "${JSON_FILE}" ]]; then
    echo "Error: JSON file not found: ${JSON_FILE}" >&2
    exit 1
fi

# Material model: --iso | --fibered | --layered (default: --iso)
# --iso     : HookeLinearElasticity, isotropic E/nu
# --fibered : SaintVenant, transversely isotropic, stiff along X (grain along beam axis)
# --layered : SaintVenant, orthotropic, stiff in XY (layers in XY plane, weak through-thickness Z)
MATERIAL="iso"
for arg in "${@:2}"; do
    case "${arg}" in
        --iso)      MATERIAL="iso" ;;
        --fibered)  MATERIAL="fibered" ;;
        --layered)  MATERIAL="layered" ;;
    esac
done

POLYFEM_BIN="${HOME}/Research/polyfem/build/PolyFEM_bin"
if [[ ! -x "${POLYFEM_BIN}" ]]; then
    echo "Error: PolyFEM binary not found or not executable: ${POLYFEM_BIN}" >&2
    exit 1
fi

# Patch materials block from the appropriate snippet file, write to a temp JSON
MATERIAL_FILE="${SCRIPT_DIR}/materials-${MATERIAL}.json"
if [[ ! -f "${MATERIAL_FILE}" ]]; then
    echo "Error: Material file not found: ${MATERIAL_FILE}" >&2
    exit 1
fi
PATCHED_JSON="$(mktemp /tmp/kigumi-bending-XXXXXX.json)"
trap 'rm -f "${PATCHED_JSON}"' EXIT
jq --slurpfile mat "${MATERIAL_FILE}" '.materials = $mat[0]' "${JSON_FILE}" > "${PATCHED_JSON}"

# 現在の日時を取得 (月日_時分 例: 0127_1705)
DATETIME="$(date +%m%d_%H%M)"
OUT_DIR="/Users/quentinbecker/Library/CloudStorage/GoogleDrive-quentinbecker@g.ecc.u-tokyo.ac.jp/My Drive/kigumi-project/simulations/bending/${DATETIME}_${MATERIAL}_simulation"
mkdir -p "${OUT_DIR}"

echo "Running simulation with: ${JSON_FILE} [material: ${MATERIAL}]"
"${POLYFEM_BIN}" --json "${PATCHED_JSON}" --output_dir "${OUT_DIR}"

echo "Simulation finished. Results are in ${OUT_DIR}"
