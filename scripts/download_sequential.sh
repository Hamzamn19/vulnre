#!/usr/bin/env bash
# download_sequential.sh — تحميل ما تبقى من البيانات والموديلات تسلسلياً
# لتفادي 429 Rate Limit من HF (كانت 8 وحدات متوازية تُحظر)
set -u
ROOT="/home/hamzah/Documents/Default Project/vulnre"
LOG="$ROOT/logs/dl-sequential.log"
DONE="$ROOT/logs/dl-sequential.DONE"
mkdir -p "$ROOT/logs"
exec >>"$LOG" 2>&1
rm -f "$DONE"

entries=(
  "dataset|colin/PrimeVul|data/raw/primevul"
  "dataset|nvidia/OpenCodeInstruct|data/raw/opencodeinstruct"
  "dataset|EER6/nvidia-OpenCodeInstruct-refined|data/raw/opencodeinstruct-refined"
  "dataset|bigcode/self-oss-instruct-sc2-exec-filter-50k|data/raw/oss-instruct-50k"
  "dataset|AdhyanshVerma/html-css-js-cot|data/raw/html-css-js-cot"
  "model|deepseek-ai/deepseek-coder-6.7b-base|models/deepseek-coder-6.7b-base"
  "model|LLM4Binary/llm4decompile-6.7b-v1.5|models/llm4decompile-6.7b-v1.5"
  "model|LittleBitLLM/littlebit-qwen3-8b-0.55bpw|models/littlebit-qwen3-8b-0.55bpw"
)

for entry in "${entries[@]}"; do
  kind="${entry%%|*}"; rest="${entry#*|}"; repo="${rest%%|*}"; dirpart="${rest##*|}"
  echo ""
  echo "===== $(date) START $kind $repo -> $dirpart ====="
  bash "$ROOT/scripts/download_aria2.sh" "$kind" "$repo" "$ROOT/$dirpart"
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "===== $(date) OK   $kind $repo ====="
  else
    echo "===== $(date) FAIL $kind $repo (rc=$rc) — سيتم تخطيها والمتابعة ====="
  fi
  echo "===== $(date) استراحة 15 ثانية قبل العنصر التالي ====="
  sleep 15
done

echo ""
echo "===== $(date) كل العناصر انتهت من المعالجة ====="
touch "$DONE"
