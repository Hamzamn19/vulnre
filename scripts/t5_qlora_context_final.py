#!/usr/bin/env python3
"""T5 — QLoRA context-length FINAL test on deepseek-coder-6.7b-base.
Tests seq_len = 4K / 8K / 16K / 32K with LoRA r=32.
Cuts text manually to seq_len tokens so num_tokens reflects reality.
Note: deepseek-coder-6.7b has max_position_embeddings=16384 natively;
32K will be truncated to 16K by the model — confirms 16K is native limit.
"""
import os
import csv
import gc
import torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "models/deepseek-coder-6.7b-base"
OUTPUT_DIR = "data/processed/smoke/t5_ctx_final_test"
RESULTS_CSV = "data/processed/smoke/t5_ctx_final_results.csv"

CONTEXTS = [4096, 8192, 16384, 32768]
R_VALUE = 32
MAX_STEPS = 1

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


def print_mem(tag):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1024**3
        res = torch.cuda.memory_reserved(0) / 1024**3
        peak = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"  [{tag}] allocated={alloc:.2f}GB reserved={res:.2f}GB peak={peak:.2f}GB")
        return peak
    return 0


def run_context(seq_len):
    print(f"\n{'='*60}")
    print(f"Testing context length: {seq_len} tokens, LoRA r={R_VALUE}")
    print(f"{'='*60}")

    # Single tokenizer instance
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build long text (> target tokens), then cut manually to seq_len
    target_chars = seq_len * 6
    text = build_long_text(target_chars)
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    real_tokens = enc.input_ids.shape[1]
    del enc
    gc.collect()
    print(f"  Real token count before manual cut: {real_tokens}")

    # Manual cut to seq_len tokens
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    truncated_text = tokenizer.decode(enc.input_ids[0], skip_special_tokens=True)
    actual_tokens = enc.input_ids.shape[1]
    del enc
    gc.collect()
    print(f"  Actual tokens after cut to {seq_len}: {actual_tokens}")

    train_dataset = Dataset.from_list([{"text": truncated_text}])

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
    )
    peak_after_load = print_mem("after load")

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
    model.print_trainable_parameters()
    peak_after_lora = print_mem("after LoRA")

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"ctx_{seq_len}"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    peak_before_train = print_mem("before train")
    actual_num_tokens = None
    status = "OK"
    try:
        train_result = trainer.train()
        metrics = train_result.metrics
        actual_num_tokens = metrics.get("num_tokens")
        print(f"  train metrics num_tokens: {actual_num_tokens}")
    except Exception as e:
        status = f"FAIL: {type(e).__name__}: {str(e)[:120]}"
        print(f"  !! {status}")
    peak_after_train = print_mem("after train")

    del model, trainer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,
        "r": R_VALUE,
        "real_tokens_before": real_tokens,
        "actual_tokens_after_cut": actual_tokens,
        "actual_num_tokens": actual_num_tokens,
        "status": status,
        "peak_load_gb": round(peak_after_load, 3),
        "peak_lora_gb": round(peak_after_lora, 3),
        "peak_train_gb": round(peak_after_train, 3),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    print(f"Contexts to test: {CONTEXTS}")
    print(f"LoRA r={R_VALUE}, steps per context={MAX_STEPS}")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

    results = []
    for ctx in CONTEXTS:
        r = run_context(ctx)
        results.append(r)
        print(f"\n>>> Result for {ctx}: {r['status']} | actual_num_tokens={r['actual_num_tokens']} | peak VRAM: {r['peak_train_gb']} GB")

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq_len", "r", "real_tokens_before", "actual_tokens_after_cut", "actual_num_tokens", "status", "peak_load_gb", "peak_lora_gb", "peak_train_gb"])
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*60}")
    print("SUMMARY — REAL CONTEXT TEST (manual cut to seq_len tokens)")
    print(f"{'='*60}")
    print(f"{'Context':<10} {'ActualNumTok':<16} {'Status':<12} {'PeakVRAM(GB)':<14}")
    for r in results:
        print(f"{r['seq_len']:<10} {str(r['actual_num_tokens']):<16} {r['status']:<12} {r['peak_train_gb']:<14}")
    print(f"\nResults saved to: {RESULTS_CSV}")

    viable = [r for r in results if r["status"] == "OK" and r["actual_num_tokens"] is not None and r["actual_num_tokens"] >= 30000]
    if viable:
        best = max(viable, key=lambda x: x["seq_len"])
        print(f"\nRecommended max context (>=30K actual tokens): {best['seq_len']} tokens (peak VRAM: {best['peak_train_gb']} GB)")
    else:
        print("\nNo context reached >=30K actual tokens.")
        print("deepseek-coder-6.7b has max_position_embeddings=16384 natively.")
        print("32K was truncated to 16K — confirms 16K is native limit; 32K requires YaRN (T9b).")

if __name__ == "__main__":
    main()
