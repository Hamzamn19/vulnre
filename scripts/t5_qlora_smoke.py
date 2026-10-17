#!/usr/bin/env python3
"""T5 — QLoRA smoke test on deepseek-coder-6.7b-base.
Runs 2 training steps on a tiny dataset to verify training works.
Memory-safe: 4-bit NF4, batch_size=1, seq_len=512, LoRA r=8 (q_proj/v_proj only).
"""
import os
import gc
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "models/deepseek-coder-6.7b-base"
OUTPUT_DIR = "data/processed/smoke/t5_qlora_test"

# Tiny dataset — 3 code examples (vulnerable + safe)
TRAIN_DATA = [
    {"text": "# C vulnerability example\n#include <stdio.h>\nint main() { char buf[64]; gets(buf); return 0; }\n"},
    {"text": "# Python unsafe eval\ndef run(code):\n    eval(code)\n    return True\n"},
    {"text": "# Safe version\ndef run_safe(code):\n    allowed = {'print', 'len'}\n    return True\n"},
]


def print_memory(tag):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"[{tag}] GPU allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print_memory("start")

    # 4-bit quant config (NF4)
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
    print_memory("after model load")

    model = prepare_model_for_kbit_training(model)

    # Light LoRA: only q_proj/v_proj to save memory
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print_memory("after LoRA")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SFT config — tiny smoke test
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=2,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )

    train_dataset = Dataset.from_list(TRAIN_DATA)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    print_memory("before training")
    trainer.train()
    print_memory("after training")

    print("\n" + "=" * 60)
    print("✅ T5 QLoRA smoke test PASSED — training works!")
    print("=" * 60)

    # Cleanup
    del model, trainer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("after cleanup")


if __name__ == "__main__":
    main()
