#!/bin/bash

SYMBOLS=(
  "ADAUSDT" "AVAXUSDT" "DOTUSDT" "LINKUSDT" "LTCUSDT"
  "ATOMUSDT" "NEARUSDT" "APTUSDT" "ARBUSDT" "OPUSDT"
  "INJUSDT" "STXUSDT" "TIAUSDT" "SEIUSDT" "WLDUSDT"
  "FETUSDT" "RNDRUSDT" "AAVEUSDT" "UNIUSDT" "MKRUSDT"
)

BASE_URL="https://data.binance.vision/data/futures/um/monthly/klines"
INTERVAL="1d"
OUTPUT_DIR="./binance_data"
mkdir -p $OUTPUT_DIR

for SYM in "${SYMBOLS[@]}"; do
  for YEAR in 2025; do
    for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
      FILE="${SYM}-${INTERVAL}-${YEAR}-${MONTH}.zip"
      URL="${BASE_URL}/${SYM}/${INTERVAL}/${FILE}"
      OUT="${OUTPUT_DIR}/${FILE}"
      
      if curl -sf -o "$OUT" "$URL"; then
        echo "✅ $FILE"
        unzip -q -o "$OUT" -d "$OUTPUT_DIR"
        rm "$OUT"
      else
        echo "⏭ Skip $FILE (not found)"
      fi
      sleep 0.2
    done
  done
done

echo "Done!"
