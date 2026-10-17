#!/usr/bin/env bash
# Evaluate a trained LittleBit checkpoint (perplexity + zero-shot).
#   CKPT=/workspace/outputs/Qwen2.5-0.5B-1.0bpw bash eval.sh
# Set FP_BASELINE=1 with CKPT=<hf model id> to score the original FP16 model.
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
CKPT="${CKPT:?set CKPT to a checkpoint dir or HF model id}"
ZEROSHOT="${ZEROSHOT:-boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa}"

export HF_HOME="$WORKDIR/hf_cache"
# shellcheck disable=SC1091
source "$WORKDIR/venv/bin/activate"
cd "$WORKDIR/LittleBit"
mkdir -p "$WORKDIR/logs"

EXTRA=()
if [ "${FP_BASELINE:-0}" = "1" ]; then EXTRA+=(--fp_eval True); fi

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --model_id "$CKPT" \
  --seqlen 2048 \
  --ppl_task wikitext2,c4 \
  --zeroshot_task "$ZEROSHOT" \
  "${EXTRA[@]}" 2>&1 | tee "$WORKDIR/logs/eval-$(basename "$CKPT")-$(date +%Y%m%d-%H%M).log"
