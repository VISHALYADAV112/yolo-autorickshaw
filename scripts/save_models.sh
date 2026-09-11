#!/usr/bin/env bash
# Archive every trained model into models/ with descriptive, versioned names.
# Scans both the local runs/ dir and the ultralytics runs_dir (model_comparison_lab).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/venv/bin/python}"
[ -x "$PY" ] || PY="python3"
OUT="$ROOT/models"
mkdir -p "$OUT"

date_tag="$(date +%Y%m%d)"

# all plausible search roots for trained weights
roots=(
  "$ROOT/runs"
  "$ROOT/model_comparison_lab/runs/detect/runs"
  "$HOME/model_comparison_lab/runs/detect/runs"
)

index=0
find "${roots[@]}" -maxdepth 4 -type f -name best.pt 2>/dev/null | sort -u > "$OUT/.scan_tmp" || true
while IFS= read -r best; do
  [ -n "$best" ] || continue
  run_name="$(basename "$(dirname "$(dirname "$best")")")"
  index=$((index + 1))

  # determine number of classes
  nc="$("$PY" -c 'import sys
from ultralytics import YOLO
try:
    print(len(YOLO(sys.argv[1]).names))
except Exception:
    print("?")' "$best" 2>/dev/null || echo "?")"

  dest="$OUT/${date_tag}_${run_name}_nc${nc}.pt"
  cp "$best" "$dest"
  printf "  %2d) %-30s -> %s (%s classes, %s MB)\n" \
    "$index" "$run_name" "$(basename "$dest")" "$nc" "$(du -m "$dest" | cut -f1)"
done < "$OUT/.scan_tmp"
rm -f "$OUT/.scan_tmp"

if [ "$index" -eq 0 ]; then
  echo "No trained models found." >&2
  exit 0
fi

echo
echo "Archived $index model(s) to $OUT/"
ls -lh "$OUT" | tail -n +2