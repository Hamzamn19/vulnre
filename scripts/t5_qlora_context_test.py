#!/usr/bin/env python3
"""T5 — QLoRA context-length stress test on deepseek-coder-6.7b-base.
Tests seq_len = 4K / 8K / 16K / 32K with LoRA r=32.
Measures peak VRAM per context to decide final context (R8).
Memory-safe: runs contexts sequentially, clears CUDA cache between them.
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
OUTPUT_DIR = "data/processed/smoke/t5_ctx_test"
RESULTS_CSV = "data/processed/smoke/t5_ctx_results.csv"

CONTEXTS = [4096, 8192, 16384, 32768]
R_VALUE = 32
MAX_STEPS = 1

# Base text repeated to reach target length
BASE_TEXT = """# C code example with potential vulnerability
#include <stdio.h>
#include <string.h>

void vulnerable_function(char *input) {
    char buffer[256];
    strcpy(buffer, input);  // potential buffer overflow
    printf(buffer);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        vulnerable_function(argv[1]);
    }
    return 0;
}
""" * 20  # ~8KB per repetition


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

    # Build dataset: one long text padded/truncated to seq_len
    text = (BASE_TEXT * ((seq_len // len(BASE_TEXT)) + 2))[:seq_len]
    train_dataset = Dataset.from_list([{"text": text}])

    # 4-bit quant config
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    try:
        trainer.train()
        status = "OK"
    except Exception as e:
        status = f"FAIL: {type(e).__name__}: {str(e)[:120]}"
        print(f"  !! {status}")
    peak_after_train = print_mem("after train")

    # Cleanup for next context
    del model, trainer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,
        "r": R_VALUE,
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
        print(f"\n>>> Result for {ctx}: {r['status']} | peak train VRAM: {r['peak_train_gb']} GB")

    # Write CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq_len", "r", "status", "peak_load_gb", "peak_lora_gb", "peak_train_gb"])
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Context':<10} {'Status':<20} {'Peak VRAM (GB)':<15}")
    for r in results:
        print(f"{r['seq_len']:<10} {r['status']:<20} {r['peak_train_gb']:<15}")
    print(f"\nResults saved to: {RESULTS_CSV}")

    # Recommendation
    viable = [r for r in results if r["status"] == "OK"]
    if viable:
        best = max(viable, key=lambda x: x["seq_len"])
        print(f"\nRecommended max context: {best['seq_len']} tokens (peak VRAM: {best['peak_train_gb']} GB)")


if __name__ == "__main__":
    main()
