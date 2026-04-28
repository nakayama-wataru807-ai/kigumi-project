#!/usr/bin/env bash
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

# Material model: --iso or --ortho (default: --iso)
# NOTE: --ortho (HookeLinearElasticity) does not converge with IPC contact due to
#       near-singular stiffness (M piece has rigid body modes). Use --iso (NeoHookean)
#       for two-piece bending+contact simulations.
MATERIAL="iso"
for arg in "${@:2}"; do
    case "${arg}" in
        --iso)   MATERIAL="iso" ;;
        --ortho) MATERIAL="ortho" ;;
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
