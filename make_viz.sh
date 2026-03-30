#!/usr/bin/env bash
# Regenerate visualization videos for a range of training scenes.
# Each scene's WAV is paired with its own JSON (correct GT labels).
#
# Usage:
#   bash make_viz.sh                  # scenes 0..9
#   bash make_viz.sh 0 19             # scenes 0..19
#   bash make_viz.sh 5 5              # scene 5 only

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

START=${1:-0}
END=${2:-9}

CKPT="./checkpoints_ver6_50db/sled_best.pt"
CLASS_MAP="./data/meta/class_map.json"
AUDIO_DIR="./data/audio/train"

for i in $(seq "$START" "$END"); do
  NAME=$(printf "scene_%06d" "$i")
  WAV="${AUDIO_DIR}/${NAME}.wav"
  JSON="${AUDIO_DIR}/${NAME}.json"
  OUT="viz_${NAME}.mp4"

  if [[ ! -f "$WAV" || ! -f "$JSON" ]]; then
    echo "[SKIP] ${NAME}: WAV or JSON missing"
    continue
  fi

  echo "[VIZ]  ${NAME} ..."
  python -m sled.visualize \
    --audio     "$WAV"       \
    --ckpt      "$CKPT"      \
    --gt-json   "$JSON"      \
    --class-map "$CLASS_MAP" \
    --output    "$OUT"
done

echo "Done."
