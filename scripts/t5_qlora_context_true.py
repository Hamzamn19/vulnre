#!/usr/bin/env python3
"""T5 — QLoRA context-length TRUE stress test on deepseek-coder-6.7b-base.
Tests seq_len = 4K / 8K / 16K / 32K with LoRA r=32.
Measures TRUE peak VRAM right after loss.backward() (before any cleanup),
and prints actual input_ids.shape per context.
"""
import os
import csv
import gc
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "models/deepseek-coder-6.7b-base"
OUTPUT_DIR = "data/processed/smoke/t5_ctx_true_test"
RESULTS_CSV = "data/processed/smoke/t5_ctx_true_results.csv"

CONTEXTS = [8192]
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
    print(f"Testing context length: {seq_len} tokens, LoRA r={R_VALUE}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_chars = seq_len * 6
    text = build_long_text(target_chars)

    # Tokenize full, then manually truncate to seq_len
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    real_tokens = enc.input_ids.shape[1]
    print(f"  Raw token count: {real_tokens}")

    # Manual hard truncation to exactly seq_len tokens
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = enc.input_ids.to("cuda")
    labels = input_ids.clone()
    print(f"  input_ids.shape after cut: {tuple(input_ids.shape)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=R_VALUE,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # Reset peak stats BEFORE forward/backward
    torch.cuda.reset_peak_memory_stats(0)

    status = "OK"
    peak_after_backward = 0
    input_shape_recorded = None
    try:
        model.train()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
        loss.backward()
        peak_after_backward = torch.cuda.max_memory_allocated(0)
        input_shape_recorded = tuple(input_ids.shape)
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
        "raw_tokens": real_tokens,
        "input_ids_shape": str(input_shape_recorded),
        "status": status,
        "peak_vram_gb": round(peak_after_backward / 1024**3, 3),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    print(f"Contexts to test: {CONTEXTS}")
    print(f"LoRA r={R_VALUE}")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

    results = []
    for ctx in CONTEXTS:
        r = run_context(ctx)
        results.append(r)
        print(f"\n>>> {ctx}: {r['status']} | input_ids={r['input_ids_shape']} | peak VRAM: {r['peak_vram_gb']} GB")

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq_len", "r", "raw_tokens", "input_ids_shape", "status", "peak_vram_gb"])
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*60}")
    print("SUMMARY — TRUE PEAK VRAM AFTER BACKWARD")
    print(f"{'='*60}")
    print(f"{'Context':<10} {'InputIDs':<16} {'Status':<12} {'PeakVRAM(GB)':<14}")
    for r in results:
        print(f"{r['seq_len']:<10} {r['input_ids_shape']:<16} {r['status']:<12} {r['peak_vram_gb']:<14}")
    print(f"\nResults saved to: {RESULTS_CSV}")

    viable = [r for r in results if r["status"] == "OK"]
    if viable:
        print("\nPeak VRAM trend across contexts:")
        for r in viable:
            print(f"  {r['seq_len']}: {r['peak_vram_gb']} GB")


if __name__ == "__main__":
    main()
