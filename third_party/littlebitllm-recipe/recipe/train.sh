#!/usr/bin/env bash
# LittleBit QAT training on RunPod. Override any setting via env vars, e.g.
#   MODEL_ID=Qwen/Qwen3-14B EFF_BIT=0.55 EPOCHS=5 bash train.sh
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="$WORKDIR/LittleBit"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
EFF_BIT="${EFF_BIT:-0.55}"                  # bits per weight: 1.0, 0.55, 0.3, 0.1 ...
EPOCHS="${EPOCHS:-1.0}"                     # paper uses 5.0
BATCH="${BATCH:-2}"                         # Qwen3's 151k vocab makes KD logits large
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-4e-05}"
USE_ITQ="${USE_ITQ:-True}"                  # True = LittleBit-2 init (Joint-ITQ)
TEACHER_OFFLOAD="${TEACHER_OFFLOAD:-False}" # False = teacher stays on GPU (faster, patched)
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MODEL_SLUG="$(basename "$MODEL_ID" | tr '[:upper:]' '[:lower:]')"
RUN_NAME="${RUN_NAME:-littlebit-${MODEL_SLUG}-${EFF_BIT}bpw}"
SAVE_DIR="${SAVE_DIR:-$WORKDIR/outputs}"

if [ -z "${REPORT:-}" ]; then
  if [ -n "${WANDB_API_KEY:-}" ]; then REPORT=wandb; else REPORT=tensorboard; fi
fi

export HF_HOME="$WORKDIR/hf_cache"
# shellcheck disable=SC1091
source "$WORKDIR/venv/bin/activate"
cd "$REPO_DIR"
mkdir -p "$WORKDIR/logs"
LOG="$WORKDIR/logs/train-$RUN_NAME-$(date +%Y%m%d-%H%M).log"

ARGS=(
  --model_id "$MODEL_ID"
  --dataset c4_wiki
  --data_root "$WORKDIR/data"
  --save_dir "$SAVE_DIR"
  --f_name "$RUN_NAME"
  --run_name "$RUN_NAME"
  --num_train_epochs "$EPOCHS"
  --per_device_train_batch_size "$BATCH"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --lr "$LR"
  --warmup_ratio 0.02
  --report "$REPORT"
  --quant_func SmoothSign
  --quant_mod LittleBitLinear
  --residual True
  --eff_bit "$EFF_BIT"
  --kv_factor 1.0
  --min_split_dim 8
  --l2l_loss_scale 10.0
  --use_itq "$USE_ITQ"
  --teacher_offload "$TEACHER_OFFLOAD"
)
# Multi-GPU: ZeRO-3 sharding. Default keeps optimizer/params on GPU (fast);
# set DS_CONFIG=configs/zero3.json for upstream CPU offload (slow, less VRAM).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS_CONFIG="${DS_CONFIG:-$SCRIPT_DIR/zero3_nooffload.json}"
if [ "$NUM_GPUS" -gt 1 ]; then
  ARGS+=(--ds_config_path "$DS_CONFIG")
fi

# Always go through the deepspeed launcher: it sets LOCAL_RANK, which main.py
# needs to place the student on a concrete GPU (plain `python -m main` passes
# device_map="auto" to model.to(), which fails).
echo ">> $RUN_NAME on $NUM_GPUS GPU(s), log: $LOG"
deepspeed --num_gpus="$NUM_GPUS" main.py "${ARGS[@]}" 2>&1 | tee "$LOG"

echo ">> Checkpoint: $SAVE_DIR/$RUN_NAME"
