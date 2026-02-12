#!/usr/bin/env bash
set -euo pipefail

# スクリプトの場所を基準に相対パスを解決する
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# 引数でJSONファイルを指定（指定がなければデフォルト）
JSON_ARG="${1:-kigumi-tension.json}"
if [[ "${JSON_ARG}" = /* ]]; then
    JSON_FILE="${JSON_ARG}"
else
    JSON_FILE="${SCRIPT_DIR}/${JSON_ARG}"
fi

if [[ ! -f "${JSON_FILE}" ]]; then
    echo "Error: JSON file not found: ${JSON_FILE}" >&2
    exit 1
fi

POLYFEM_BIN="${HOME}/polyfem/build/PolyFEM_bin"
if [[ ! -x "${POLYFEM_BIN}" ]]; then
    echo "Error: PolyFEM binary not found or not executable: ${POLYFEM_BIN}" >&2
    exit 1
fi

# 現在の日時を取得 (月日_時分 例: 0127_1705)
DATETIME="$(date +%m%d_%H%M)"
OUT_DIR="${PROJECT_ROOT}/output/${DATETIME}_simulation"
mkdir -p "${OUT_DIR}"

echo "Running simulation with: ${JSON_FILE}"
"${POLYFEM_BIN}" --json "${JSON_FILE}" --output_dir "${OUT_DIR}"

echo "Simulation finished. Results are in ${OUT_DIR}"
