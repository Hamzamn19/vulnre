#!/usr/bin/env bash
# T2c — شريحة «الحداثة» (R9): كود حديث متعدد اللغات + ويب/واجهات + Python عالي الجودة
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF="$ROOT/.venv/bin/hf"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

# "repo|local-dir"
REPOS=(
  "nvidia/OpenCodeInstruct|opencodeinstruct"
  "EER6/nvidia-OpenCodeInstruct-refined|opencodeinstruct-refined"
  "bigcode/self-oss-instruct-sc2-exec-filter-50k|oss-instruct-50k"
  "AdhyanshVerma/html-css-js-cot|html-css-js-cot"
)

FAILED=()
for entry in "${REPOS[@]}"; do
  repo="${entry%%|*}"
  dir="${entry##*|}"
  echo "[$(date +%H:%M:%S)] START $repo"
  if "$HF" download "$repo" --repo-type dataset --local-dir "$RAW/$dir"; then
    echo "[$(date +%H:%M:%S)] OK    $repo"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $repo"
    FAILED+=("$repo")
  fi
done

echo "=== MODERN-DATA SUMMARY: ${#FAILED[@]} failed: ${FAILED[*]:-none}"
