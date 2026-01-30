#!/bin/bash

# 1. 引数でJSONファイルを指定（指定がなければデフォルトの *.json を使用）
JSON_FILE="${1:-kigumi-incline.json}"

# 2. 現在の日時を取得 (月日_時分 例: 0127_1705)
DATETIME=$(date +%m%d_%H%M)

# 3. 出力フォルダ名を定義 (output/0127_1705_simulation)
OUT_DIR="../output/${DATETIME}_simulation"

# 4. フォルダを作成
mkdir -p "$OUT_DIR"

# 5. PolyFEMを実行
# 実行するJSONファイル名を表示
echo "Running simulation with: $JSON_FILE"
~/polyfem/build/PolyFEM_bin --json "$JSON_FILE" --output_dir "$OUT_DIR"

echo "Simulation finished. Results are in $OUT_DIR"