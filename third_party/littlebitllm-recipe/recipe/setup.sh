#!/usr/bin/env bash
# One-time setup for LittleBit on a RunPod GPU pod.
# Run from the pod terminal:  bash /workspace/runpod/setup.sh
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="$WORKDIR/LittleBit"
VENV_DIR="$WORKDIR/venv"

# Keep every cache on the persistent volume so it survives pod restarts.
export HF_HOME="$WORKDIR/hf_cache"
export PIP_CACHE_DIR="$WORKDIR/pip_cache"

echo ">> GPU check"
nvidia-smi --query-gpu=name,memory.total --format=csv || { echo "No NVIDIA GPU found"; exit 1; }

echo ">> System packages (mpi4py needs OpenMPI headers)"
apt-get update -qq && apt-get install -y -qq libopenmpi-dev git >/dev/null

if [ ! -d "$REPO_DIR" ]; then
  echo ">> Cloning LittleBit"
  git clone https://github.com/SamsungLabs/LittleBit.git "$REPO_DIR"
  # Pin the commit the patches were written against.
  git -C "$REPO_DIR" checkout -q 42d658b
fi

echo ">> Applying local patches"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for p in "$SCRIPT_DIR"/patches/*.patch; do
  if git -C "$REPO_DIR" apply --reverse --check "$p" 2>/dev/null; then
    echo "   already applied: $(basename "$p")"
  else
    git -C "$REPO_DIR" apply "$p" && echo "   applied: $(basename "$p")"
  fi
done

echo ">> Python venv (inherits the pod's preinstalled CUDA torch)"
python3 -m venv --system-site-packages "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip

# requirements.txt pins torch==2.6.0, which would replace the pod's CUDA build.
# Install everything except torch, then the versions the README asks for.
grep -viE '^torch(==|$)' "$REPO_DIR/requirements.txt" > /tmp/req-notorch.txt
pip install -q -r /tmp/req-notorch.txt
pip install -q "transformers==4.51.*" deepspeed wandb

echo ">> Sanity check"
python - <<'PY'
import torch, transformers, deepspeed
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "deepspeed", deepspeed.__version__)
for i in range(torch.cuda.device_count()):
    print(" gpu", i, torch.cuda.get_device_name(i))
assert torch.cuda.is_available(), "CUDA not available"
PY

cat <<EOF

Setup done. Next:
  export HF_TOKEN=hf_xxx        # needed for gated models like Llama-2
  export WANDB_API_KEY=xxx      # optional; without it, REPORT=tensorboard is used
  bash $WORKDIR/runpod/train.sh
EOF
