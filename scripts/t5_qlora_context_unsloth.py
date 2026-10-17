#!/usr/bin/env python3
"""T5 — QLoRA context test with Unsloth on deepseek-coder-6.7b-base.
Tests seq_len = 8K / 16K with LoRA r=16 using:
  - FastLanguageModel.from_pretrained (load_in_4bit=True)
  - use_gradient_checkpointing="unsloth"
  - paged_adamw_8bit optimizer
Measures TRUE peak VRAM right after loss.backward() (before any cleanup).
"""
import os
import csv
import gc
import torch
import unsloth  # noqa: F401  (must be imported before transformers for optimizations)
from datetime import datetime
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from bitsandbytes.optim import PagedAdamW8bit

MODEL_ID = "models/deepseek-coder-6.7b-base"
OUTPUT_DIR = "data/processed/smoke/t5_ctx_unsloth_test"
RESULTS_CSV = "data/processed/smoke/t5_ctx_unsloth_results.csv"

CONTEXTS = [16384, 32768]  # 16K first, then 32K if 16K passes
R_VALUE = 16

BASE_BLOCK = """#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void func_{idx}(char *input) {
    char buffer[256];
    int i;
    for (i = 0; i < 256; i++) {
        buffer[i] = input[i];
    }
    buffer[255] = '\\0';
    printf("Output: %s\\n", buffer);
}

int main_{idx}(int argc, char *argv[]) {
    char *data = getenv("INPUT");
    if (data == NULL) {
        data = "default";
    }
    func_{idx}(data);
    return 0;
}

"""


def build_long_text(target_chars):
    parts = []
    idx = 0
    total = 0
    while total < target_chars:
        parts.append(BASE_BLOCK.replace("{idx}", str(idx)))
        total += len(BASE_BLOCK.replace("{idx}", str(idx)))
        idx += 1
    return "".join(parts)


def run_context(seq_len):
    print(f"\n{'='*60}")
    print(f"Testing context length: {seq_len} tokens, LoRA r={R_VALUE} (Unsloth)")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_chars = seq_len * 6
    text = build_long_text(target_chars)

    # Tokenize full, then manually truncate to seq_len
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    raw_tokens = enc.input_ids.shape[1]
    print(f"  Raw token count: {raw_tokens}")

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = enc.input_ids.to("cuda")
    labels = input_ids.clone()
    print(f"  input_ids.shape after cut: {tuple(input_ids.shape)}")

    print("Loading model in 4-bit via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=seq_len,
        dtype=None,
        load_in_4bit=True,
        token=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=R_VALUE,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=seq_len,
    )

    optimizer = PagedAdamW8bit(model.parameters(), lr=2e-4)

    # Reset peak stats BEFORE forward/backward
    torch.cuda.reset_peak_memory_stats(0)

    status = "OK"
    peak_after_backward = 0
    input_ids_shape = None
    try:
        model.train()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
        loss.backward()
        peak_after_backward = torch.cuda.max_memory_allocated(0)
        input_ids_shape = tuple(input_ids.shape)
        print(f"  loss: {loss.item():.4f}")
        print(f"  PEAK VRAM after backward: {peak_after_backward / 1024**3:.3f} GB")
        optimizer.zero_grad()
    except Exception as e:
        status = f"FAIL: {type(e).__name__}: {str(e)[:140]}"
        print(f"  !! {status}")
        try:
            peak_after_backward = torch.cuda.max_memory_allocated(0)
        except Exception:
            peak_after_backward = 0

    # Cleanup
    del model, tokenizer, input_ids, labels, enc
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,
        "r": R_VALUE,
        "raw_tokens": raw_tokens,
        "input_ids_shape": input_ids_shape,
        "status": status,
        "peak_vram_gb": round(peak_after_backward / 1024**3, 3),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    print(f"Contexts to test: {CONTEXTS}")
    print(f"LoRA r={R_VALUE} (Unsloth + PagedAdamW8bit)")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

    results = []
    for ctx in CONTEXTS:
        r = run_context(ctx)
        results.append(r)
        print(f"\n>>> {ctx}: {r['status']} | input_ids={r['input_ids_shape']} | peak VRAM: {r['peak_vram_gb']} GB")

        # If this context failed, don't try larger ones
        if r["status"] != "OK":
            print(f"\nStopping at {ctx} due to failure.")
            break

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq_len", "r", "raw_tokens", "input_ids_shape", "status", "peak_vram_gb"])
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*60}")
    print("SUMMARY — UNSLOTH TRUE PEAK VRAM AFTER BACKWARD")
    print(f"{'='*60}")
    print(f"{'Context':<10} {'InputIDs':<16} {'Status':<12} {'PeakVRAM(GB)':<14}")
    for r in results:
        print(f"{r['seq_len']:<10} {str(r['input_ids_shape']):<16} {r['status']:<12} {r['peak_vram_gb']:<14}")
    print(f"\nResults saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
