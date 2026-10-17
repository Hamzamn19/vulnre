---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3-8B
base_model_relation: quantized
library_name: transformers
tags: [littlebit, littlebit-2, sub-1-bit, quantization, qat, qwen3]
---

![LittleBit.](assets/banner.png)

# LittleBit

**Big models. Little bits.** Sub-1-bit Qwen3 models built with LittleBit-2:
binarized latent factorization, recovered by distillation from the original model.

> **Status:** first sub-1-bit weights are out:
> [littlebit-qwen3-8b-0.55bpw](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw) (early checkpoint, 0.5 epoch).
> This repo holds the training recipe and measured throughput.
> Follow [@LittleBit_llm](https://x.com/LittleBit_llm) for updates.

> Independent community project. Not affiliated with or endorsed by Samsung Research.
> Built on the LittleBit method and code by Lee, Kim, You & Kim ([SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit)).

## Planned releases

| Model | Base | Target bpw | Status |
|---|---|---|---|
| littlebit-qwen3-4b | Qwen/Qwen3-4B | 0.55 | planned |
| [littlebit-qwen3-8b-0.55bpw](https://huggingface.co/LittleBitLLM/littlebit-qwen3-8b-0.55bpw) | Qwen/Qwen3-8B | 0.55 | released: early checkpoint, 0.5 epoch |
| littlebit-qwen3-14b | Qwen/Qwen3-14B | 0.55 | planned (5 epochs, after the 8B eval) |

Bits per weight apply to linear layers. Embeddings and `lm_head` stay BF16.

The first sub-1-bit run is Qwen3-8B with half an epoch of distillation (about 10,350 steps,
~8.5 h on 1× H200). That is 10% of the paper's 5-epoch recipe, so expect quality below the
paper's numbers. It is out as an early checkpoint: WikiText-2 PPL 24.48 (BF16: 9.73), ARC-Challenge at chance.

## Method

Each linear layer `W` is approximated as `sign(U) · diag(h·g·ℓ) · sign(V)ᵀ`:
low-rank latent factors binarized to ±1, plus three thin learned scale vectors.

1. **Latent factorization:** SVD splits each linear layer into rank-r factors sized to the bit budget.
2. **Joint-ITQ rotation (LittleBit-2):** aligns the factors with the binary hypercube before training. It folds into the factors, so it adds no inference cost.
3. **SmoothSign binarization:** a smooth surrogate gradient keeps the sign step trainable.
4. **Residual compensation:** a second binarized path learns what the first one missed.
5. **Distillation:** quantization-aware training on C4 + WikiText-2 (seq len 2048), with the BF16 model as teacher (logit KL + layer-to-layer MSE).

## Measured throughput

Measured on RunPod with the recipe in [`recipe/`](recipe/). One step = 4 sequences × 2048 tokens.
One epoch of the C4-shard-0 + WikiText-2 mix is about 20,750 steps with the Qwen3 tokenizer.

| Model | GPU | Sec / step | Peak VRAM | One-time init (SVD + Joint-ITQ) | Est. 1 epoch |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B @ 0.55 bpw | 1× H100 80GB | 1.03 | — | ~2.3 min | ~6 h |
| Qwen3-8B @ 0.55 bpw | 1× H200 141GB | 2.84 | ~107 GB | ~14 min | ~16.5 h |

Qwen3-8B does not fit on a single 80 GB GPU: about 3.7B latent parameters are trainable,
and their optimizer state alone exceeds the memory. Use a 141 GB GPU or ≥ 2 GPUs with ZeRO-3.

## Recipe

[`recipe/`](recipe/) runs the official LittleBit code on a RunPod GPU pod:

- `setup.sh`: clones SamsungLabs/LittleBit at a pinned commit, applies the patches, and installs dependencies (`transformers==4.51.*`, DeepSpeed).
- `train.sh`: runs QAT. Defaults: Qwen3-8B, 0.55 bpw, LittleBit-2 init, SmoothSign, residual. Override settings with env vars (`MODEL_ID`, `EFF_BIT`, `EPOCHS`, `NUM_GPUS`, …).
- `eval.sh`: measures WikiText-2/C4 perplexity and zero-shot accuracy (lm-eval).
- `zero3_nooffload.json`: multi-GPU ZeRO-3 config without CPU offload.
- `patches/teacher-on-gpu.patch`: keeps the teacher on GPU instead of ZeRO-3 CPU offload (`--teacher_offload False`).
- `patches/eval-import-fix.patch`: fixes a circular import between lm-eval, transformers, and DeepSpeed in `eval.py`.

```bash
bash recipe/setup.sh
MODEL_ID=Qwen/Qwen3-8B EFF_BIT=0.55 EPOCHS=0.5 NUM_GPUS=1 bash recipe/train.sh
CKPT=/workspace/outputs/littlebit-qwen3-8b-0.55bpw bash recipe/eval.sh
```

## License

CC BY-NC 4.0 (non-commercial), inherited from the LittleBit code. Released weights
are also subject to the base model's license (Qwen3: Apache 2.0).

## Citation

```bibtex
@inproceedings{lee2025littlebit,
  title     = {LittleBit: Ultra Low-Bit Quantization via Latent Factorization},
  author    = {Lee, Banseok and Kim, Dongkyu and You, Youngcheon and Kim, Youngmin},
  booktitle = {NeurIPS},
  year      = {2025}
}
@inproceedings{lee2026littlebit2,
  title     = {LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment},
  author    = {Lee, Banseok and Kim, Youngmin},
  booktitle = {ICML},
  year      = {2026}
}
```
