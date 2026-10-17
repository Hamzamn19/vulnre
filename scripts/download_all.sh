#!/usr/bin/env bash
# استئناف شامل لكل التنزيلات المتبقية — مناعة ضد إعادة تشغيل الخادم (يُشغَّل بـ setsid)
# الأساسي + النماذج + الحداثة → سجل في logs/download_all.log وعلامة اكتمال logs/download_all.DONE
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF="$ROOT/.venv/bin/hf"
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/download_all.log"
DONE="$LOG_DIR/download_all.DONE"
mkdir -p "$LOG_DIR"
rm -f "$DONE"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "=== START download_all (resume) pid=$$ ==="

# "kind|repo|local-dir-relative-to-ROOT"
JOBS=(
  "dataset|bstee615/bigvul|data/raw/bigvul"
  "dataset|bstee615/diversevul|data/raw/diversevul"
  "dataset|starsofchance/CVEfixes_v1.0.8|data/raw/cvefixes"
  "dataset|colin/PrimeVul|data/raw/primevul"
  "dataset|nvidia/OpenCodeInstruct|data/raw/opencodeinstruct"
  "dataset|EER6/nvidia-OpenCodeInstruct-refined|data/raw/opencodeinstruct-refined"
  "dataset|bigcode/self-oss-instruct-sc2-exec-filter-50k|data/raw/oss-instruct-50k"
  "dataset|AdhyanshVerma/html-css-js-cot|data/raw/html-css-js-cot"
  "model|deepseek-ai/deepseek-coder-6.7b-base|models/deepseek-coder-6.7b-base"
  "model|LLM4Binary/llm4decompile-6.7b-v1.5|models/llm4decompile-6.7b-v1.5"
  "model|LittleBitLLM/littlebit-qwen3-8b-0.55bpw|models/littlebit-qwen3-8b-0.55bpw"
)

FAILED=()
for entry in "${JOBS[@]}"; do
  case "$entry" in
    model|*) log "SKIP [model] ${entry#*|} (يدوي عبر hf_aria2_repo.sh)"; continue ;;
  esac
  kind="${entry%%|*}"; rest="${entry#*|}"; repo="${rest%%|*}"; dir="$ROOT/${rest##*|}"
  mkdir -p "$dir"
  log "START [$kind] $repo"
  if [ "$kind" = "dataset" ]; then
    "$HF" download "$repo" --repo-type dataset --local-dir "$dir" >>"$LOG" 2>&1
  else
    "$HF" download "$repo" --local-dir "$dir" >>"$LOG" 2>&1
  fi
  if [ $? -eq 0 ]; then
    log "OK    [$kind] $repo"
  else
    log "FAIL  [$kind] $repo"
    FAILED+=("$repo")
  fi
done

log "=== SUMMARY: ${#FAILED[@]} failed: ${FAILED[*]:-none}"
date '+%F %T' > "$DONE"
log "=== ALL DONE ==="
