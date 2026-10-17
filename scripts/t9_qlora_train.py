#!/usr/bin/env python3
"""T9 — QLoRA full-run training on the final mix (RTX 3060 12GB).

المصدر: data/processed/final_mix/train_final.parquet (يكتبه t9_final_mix.py).
القالب القانوني الوحيد (توكنaiser التعليمات الأصلي لـ DeepSeek-Coder):
  <｜begin▁of▁sentence｜>؟ + <|User|>\n{system\n\nuser}\n<|Assistant|>\n{assistant}<|EOT|>
  - أدوات: tool_calls تُصيّر JSON داخل دور المساعد ونتائجها دور |User| تالي.
  - telemetry: خسارة كاملة (لا assistant-only) — قاعدة base.

موازنة: --token-budget (بالتوكن التقديري chars/3.5) أو --max-records كحدّ أسطر.
ملاحظة 3060: NF4 + LoRA r16 (كل الإسقاطات) + packing 4096 + grad ckpt + paged AdamW
≈ 4.1GB ثوابت + 2-3GB تنشيط → ذروة ~7-8GB (تتحقق مع T5: 4.12GB عند 512).
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainerCallback)
from trl import SFTConfig, SFTTrainer

MODEL_ID = "models/deepseek-coder-6.7b-base"
DEFAULT_PARQUET = "data/processed/final_mix/train_final.parquet"

BOS = "<｜begin▁of▁sentence｜>"
USER_T = "<|User|>"
ASST_T = "<|Assistant|>"
EOT_T = "<|EOT|>"

STATS = {"encoded": 0, "skipped_toolshape": 0}


class LiveLossLogger(TrainerCallback):
    """يوقّع كل خطوة خرج تلقائي بملف JSONL منفصل (متطلب T9)."""

    def __init__(self, path):
        self.path = path
        self.t0 = time.time()

    def _write(self, payload):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=float) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and ("loss" in logs or "total_loss" in logs):
            self._write({"wall_h": (time.time() - self.t0) / 3600.0, **logs,
                         "step": state.global_step})

    def on_train_end(self, args, state, control, **kwargs):
        self._write({"event": "train_end", "step": state.global_step,
                     "wall_h": (time.time() - self.t0) / 3600.0})


def _assistant_text(m):
    """يزيم رسالة مساعد: محتوى + JSON لأدوات لو موجودة."""
    txt = m.get("content") or ""
    calls = m.get("tool_calls")
    if calls:
        calls_repr = [{"name": c.get("function", {}).get("name"),
                       "arguments": c.get("function", {}).get("arguments")}
                      for c in calls]
        txt = (txt + "\n" if txt.strip() else "") + \
            json.dumps(calls_repr, ensure_ascii=False)
    return txt


def template_record(messages):
    """سجل قانوني (messages JSON) → نص جاهز للتدريب. None لسجلات غير قابلة للتصيير."""
    out = []
    sys_buf = None
    for m in messages:
        role = m.get("role")
        if role == "system":
            sys_buf = (sys_buf + "\n\n" if sys_buf else "") + (m.get("content") or "")
        elif role in ("user", "tool"):
            body = m.get("content") or ""
            if role == "tool":
                body = "Tool result:\n" + body
            if sys_buf:
                body = sys_buf + "\n\n" + body
                sys_buf = None
            out.append(f"{USER_T}\n{body}\n{ASST_T}\n")
        elif role == "assistant":
            out.append(_assistant_text(m) + EOT_T + "\n")
    if sys_buf:  # system بلا دور مستخدم لاحق → سجل غير قابل للتصيير
        return None
    if not any(r.strip() for r in out):
        return None
    seq = "".join(out)
    return BOS + seq if seq else None


def load_mix(parquet, token_budget, max_records, seed):
    """يرجّع Dataset من Parquet مع 'text' مُصيَّر ومحذوف تذبذب الحجم (≤24K chars)."""
    pf = pq.ParquetFile(parquet)
    n = pf.metadata.num_rows
    take_n = min(n, max_records or n)
    print(f"[data] {parquet}: {n:,} rows → taking {take_n:,}", flush=True)

    # قراءة تدفّقية من parquet بياض الذاكرة، نمسك أول take_n من ترتيب الخلط المخزن
    ds = Dataset.from_parquet(parquet)
    ds = ds.take(take_n)

    def _render(batch):
        texts, skipped = [], 0
        for msg_json in batch["messages"]:
            try:
                msgs = json.loads(msg_json)
            except Exception:
                skipped += 1
                texts.append(None)
                continue
            t = template_record(msgs)
            if (t is None) or len(t) > 24000:
                skipped += 1
                texts.append(None)
            else:
                texts.append(t)
        STATS["encoded"] += len(texts) - skipped
        STATS["skipped_toolshape"] += skipped
        return {"text": texts}

    ds = ds.map(_render, batched=True, batch_size=256, num_proc=8,
                remove_columns=[c for c in ds.column_names if c != "text"])
    ds = ds.filter(lambda x: x["text"] is not None, num_proc=8)

    if token_budget:
        # مختصر: استقصاء الرصيد بوفر (chars/3.5 ≈ tokens)
        accum, chars = 0, 0
        keep_idx = []
        for i in range(0, len(ds), 2048):
            chunk_chars = [len(t) for t in ds[i: i + 2048]["text"]]
            for j, c in enumerate(chunk_chars):
                if accum >= token_budget * 3.5:
                    break
                accum += c
                chars += c
                keep_idx.append(i + j)
            if accum >= token_budget * 3.5:
                break
        ds = ds.select(keep_idx)
        print(f"[data] token-budget {token_budget/1e6:.1f}M → est chars {chars/1e9:.2f}GB "
              f"→ {len(ds):,} records", flush=True)
    ds = ds.shuffle(seed=seed, keep_in_memory=False)
    print(f"[data] final training dataset: {len(ds):,} records, "
          f"~{sum(len(t) for t in ds['text'][:2000])/2000*len(ds)/1e9:.2f}GB chars", flush=True)
    return ds


def main():
    ap = argparse.ArgumentParser(description="T9 QLoRA train")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--output", default="data/processed/t9_qlora_r1")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--max-records", type=int, default=0, help="0 = الكل")
    ap.add_argument("--token-budget", type=int, default=0,
                    help="سقف توكن تقديري (مثلاً 55000000 ≈ 55M)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="سقف صارم بخطوات (>0 يلغي epochs) — تدريب مكتمل منطقياً على نطاق المحدود")
    ap.add_argument("--warmup-frac", type=float, default=0.03, help="نسبة warmup من max_steps")
    ap.add_argument("--logging-steps", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-targets", choices=["all", "qv"], default="all",
                    help="all=7 إسقاطات (الخطي الحالي ~40M بارامتر)؛ qv=إعداد القياس الأصلي 6.72GB")
    ap.add_argument("--attn", default="sdpa",
                    help="sdpa | flash_attention_2 | kernels-community/flash-attn2")
    ap.add_argument("--torch-compile", action="store_true",
                    help="تفعيل torch.compile (رخيص التجربة — قد يفشل مع bnb)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--dry-run", action="store_true", help="التحقق من القالب والبيانات فقط")
    a = ap.parse_args()

    ds = load_mix(a.parquet, a.token_budget, a.max_records, a.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # فحص قالب: 3 عينات
    for i in range(3):
        t = ds[i]["text"]
        print(f"[probe {i}] chars={len(t):,} head={t[:160]!r}… tail={t[-90:]!r}", flush=True)

    if a.dry_run:
        print("[dry-run] تم التحقق — لا تدريب.", flush=True)
        return

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True,
        attn_implementation=a.attn)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    targets = (["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
               if a.lora_targets == "all" else ["q_proj", "v_proj"])  # qv = إعداد القياس الأصلي (6.72GB)
    print(f"[lora] r={a.lora_r}, alpha={a.lora_r}, targets={a.lora_targets}", flush=True)
    lora = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0.05,  # alpha=r (احتياط رخيص — ليس حل مشكلة lora_gate الجوهرية: غياب eval)
        target_modules=targets,
        bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    import torch
    print(f"[gpu] allocated {torch.cuda.memory_allocated(0)/1024**3:.2f}GB", flush=True)

    args = SFTConfig(
        output_dir=a.output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",          # cosine → صفر عند num_training_steps
        logging_steps=a.logging_steps,
        save_strategy="steps",
        save_steps=a.save_steps,
        save_total_limit=3,                  # آخر 3 فقط (ضبط القرص)
        bf16=True,
        weight_decay=0.01,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=a.max_length,
        packing=True,
        torch_compile=a.torch_compile,
        report_to="none",
        remove_unused_columns=False,
        seed=a.seed,
    )
    if a.max_steps:
        # سقف صارم بخطوات — يتجاوز epochs ويجعل LR cosine ينتهي صفاً عند السقف
        # (نبقي num_train_epochs رقمياً — لا يقبل None في transformers 5.17)
        args.num_train_epochs = max(float(a.epochs), 1.0)
        args.max_steps = a.max_steps
        args.warmup_steps = max(5, int(a.warmup_frac * a.max_steps))
        print(f"[sched] max_steps={a.max_steps} (يتجاوز epochs={args.num_train_epochs}), "
              f"warmup={args.warmup_steps}, cosine→0 عند السقف", flush=True)
    else:
        args.num_train_epochs = a.epochs
        args.warmup_steps = 40

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        callbacks=[LiveLossLogger(os.path.join(a.output, "loss_log.jsonl"))],
    )
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    trainer.save_model(os.path.join(a.output, "adapter_final"))
    tokenizer.save_pretrained(os.path.join(a.output, "adapter_final"))
    print(f"[saved] adapter → {a.output}/adapter_final", flush=True)

    # ملخص التدريب: منحنى الخسارة + أفضل نقطة بالخسارة (نسخة محفوظة) + توكن تقديري
    import shutil
    log_rows = []
    lp = os.path.join(a.output, "loss_log.jsonl")
    if os.path.exists(lp):
        with open(lp) as f:
            log_rows = [json.loads(l) for l in f if l.strip()]
    losses = [(r["step"], r["loss"]) for r in log_rows if "loss" in r]
    first_loss = losses[0][1] if losses else None
    last_loss = losses[-1][1] if losses else None
    best = min(losses, key=lambda x: x[1]) if losses else None
    best_dir = None
    if best:
        cand = os.path.join(a.output, f"checkpoint-{best[0]}")
        if os.path.isdir(cand):
            best_dir = os.path.join(a.output, "best")
            shutil.copytree(cand, best_dir, dirs_exist_ok=True)
    steps_done = trainer.state.global_step
    tok_per_step = a.max_length * a.grad_accum
    summary = {
        "steps": steps_done, "planned_max_steps": a.max_steps or None,
        "max_length": a.max_length, "grad_accum": a.grad_accum,
        "loss_first": first_loss, "loss_last": last_loss,
        "best": {"step": best[0], "loss": best[1]} if best else None,
        "batch_records_dir_best": best_dir,
        "est_tokens_trained": steps_done * tok_per_step,
        "wall_hours": dt / 3600.0,
        "loss_log": lp,
        "events": [r for r in log_rows if "event" in r],
    }
    with open(os.path.join(a.output, "train_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[done] train finished in %.2fh; steps=%d; loss %.4f→%.4f; est_tokens=%.1fM"
          % (dt / 3600, steps_done,
             first_loss if first_loss is not None else -1,
             last_loss if last_loss is not None else -1,
             steps_done * tok_per_step / 1e6), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
