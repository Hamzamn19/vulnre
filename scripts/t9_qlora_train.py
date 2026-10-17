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
import unsloth  # يجب قبل transformers/trl (patching إلزامي — خارطة unsloth)
from unsloth import FastLanguageModel  # مسار t5_context_unsloth الذي نجح (قديم/ثالثة أسرع)
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

MODEL_ID = "models/deepseek-coder-6.7b-base"
DEFAULT_PARQUET = "data/processed/final_mix/train_final.parquet"

BOS = "<｜begin▁of▁sentence｜>"
USER_T = "<|User|>"
ASST_T = "<|Assistant|>"
EOT_T = "<|EOT|>"

STATS = {"encoded": 0, "skipped_toolshape": 0}


class LiveLossLogger(TrainerCallback):
    """يوقّع كل خطوة خرج تلقائي بملف JSONL منفصل (متطلب T9).
    منهجية المستخدم: يطبع s/step لحظياً بعد أول خطوات (لا انتظار 15-30) —
    بايلوت قياس السرعة = --max-steps 8-10 + --logging-steps 1، والحساب من فرق 3→8."""

    def __init__(self, path, tok_per_step):
        self.path = path
        self.t0 = time.time()
        self.tok_per_step = tok_per_step
        self._prev = None

    def _write(self, payload):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=float) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        wall = (time.time() - self.t0) / 3600.0
        # صفوف eval غير مقصودة في توقيت التدريب — تسجَّل بخفة وتطبع
        if "eval_loss" in logs and "loss" not in logs:
            print(f"[eval@{state.global_step}] eval_loss={logs.get('eval_loss', 0):.4f}", flush=True)
            self._write({"kind": "eval", "wall_h": wall, "step": state.global_step,
                         **logs})
            return
        if "loss" not in logs and "total_loss" not in logs:
            return
        row = {"kind": "train", "wall_h": wall, **logs, "step": state.global_step}
        if self._prev is not None:
            dstep = row["step"] - self._prev["step"]
            dwall = (row["wall_h"] - self._prev["wall_h"]) * 3600.0
            if dstep > 0:
                sps = dwall / dstep
                print(f"[timing] steps {self._prev['step']}→{row['step']}: "
                      f"{sps:.1f}s/step, {self.tok_per_step/sps:.0f} tok/s "
                      f"({self.tok_per_step} tok/step)", flush=True)
        self._prev = row
        self._write(row)

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


def load_mix(parquet, token_budget, max_records, seed, eval_frac=0.0, max_eval_records=0):
    """يرجّع (ds_train, ds_eval|None) من Parquet مع 'text' مُصيَّر (≤24K chars).

    eval: اقتطاع stratified بنسبة eval_frac من كل source (قبل map)، مقصوص بmax_eval_records
    لتخفيف GPU كلفة كل eval pass (eval كل 100 خطوة — يجب أن تبقى بدقائق لا ساعات)."""
    pf = pq.ParquetFile(parquet)
    n = pf.metadata.num_rows
    take_n = min(n, max_records or n)
    print(f"[data] {parquet}: {n:,} rows → taking {take_n:,}", flush=True)
    ds = Dataset.from_parquet(parquet)
    ds = ds.take(take_n)

    import collections
    train_idx, eval_idx = list(range(len(ds))), []
    if eval_frac > 0:
        src = ds["source"]  # قراءة عمود صغيرة قبل إزالة stubs
        by_src = collections.defaultdict(list)
        for i, s in enumerate(src):
            by_src[s].append(i)
        rng = random.Random(seed + 777)
        for _s, idxs in by_src.items():
            eval_idx.extend(rng.sample(idxs, max(1, int(len(idxs) * eval_frac))))
        if max_eval_records and len(eval_idx) > max_eval_records:
            eval_idx = rng.sample(eval_idx, max_eval_records)
        eval_set = set(eval_idx)
        sel_counts = collections.defaultdict(int)
        for i in eval_idx:
            sel_counts[src[i]] += 1
        print(f"[eval] stratified x{eval_frac:.3f} → eval={len(eval_idx):,}, "
              f"train={len(train_idx) - len(eval_idx):,} per-source: {dict(sel_counts)}", flush=True)

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

    orig = ds  # المرجع الأصلي (بكل الأعمدة) — eval يُستخرج منه قبل التصيير
    train_part = orig.select(train_idx) if eval_idx else orig
    ds = train_part.map(_render, batched=True, batch_size=256, num_proc=8,
                        remove_columns=[c for c in train_part.column_names if c != "text"])
    ds = ds.filter(lambda x: x["text"] is not None, num_proc=8)

    eval_ds = None
    if eval_idx:
        raw_eval = orig.select(eval_idx)
        eval_ds = raw_eval.map(_render, batched=True, batch_size=256, num_proc=8,
                               remove_columns=[c for c in raw_eval.column_names if c != "text"])
        eval_ds = eval_ds.filter(lambda x: x["text"] is not None, num_proc=8)
        print(f"[eval] بعد التصيير: eval={len(eval_ds):,} سجل — متاح للpass", flush=True)

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
    return ds, eval_ds


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
    ap.add_argument("--save-steps", type=int, default=100,
                    help="مواءم مع eval_steps افتراضياً (شرط load_best_model_at_end)")
    ap.add_argument("--eval-frac", type=float, default=0.07,
                    help="حصة eval stratified بنسبة من كل source (0 يلغي)")
    ap.add_argument("--max-eval-records", type=int, default=2000,
                    help="سقف سجلات eval (اقتصاد pass — كل eval لا يمخ بجدول التدريب)")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--es-patience", type=int, default=3,
                    help="EarlyStopping: عدد eval بدون تحسن قبل التوقف")
    ap.add_argument("--dry-run", action="store_true", help="التحقق من القالب والبيانات فقط")
    a = ap.parse_args()

    ds, eval_ds = load_mix(a.parquet, a.token_budget, a.max_records, a.seed,
                           a.eval_frac, a.max_eval_records)

    # فحص قالب: 3 عينات (التباده لا يحتاج notice loader؛ unsloth يعيد tokenizer لاحقاً)
    for i in range(3):
        t = ds[i]["text"]
        print(f"[probe {i}] chars={len(t):,} head={t[:160]!r}… tail={t[-90:]!r}", flush=True)

    if a.dry_run:
        print("[dry-run] تم التحقق — لا تدريب.", flush=True)
        return

    print(f"[loader] unsloth FastLanguageModel (max_seq_length={a.max_length}); "
          f"attn flag ignored — unsloth يدير الانتباه داخلياً", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=a.max_length,
        dtype=None,          # unsloth يختار bf16 تلقائياً (مدعوم على 3060)
        load_in_4bit=True,
    )

    targets = (["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
               if a.lora_targets == "all" else ["q_proj", "v_proj"])  # qv = إعداد القياس الأصلي (6.72GB)
    print(f"[lora] r={a.lora_r}, alpha={a.lora_r}, targets={a.lora_targets}, "
          f"ckpt=unsloth", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=a.lora_r,
        lora_alpha=a.lora_r,  # alpha=r — توافق t5-unsloth الناجح (ليس حل مشكلة lora_gate: غياب eval)
        lora_dropout=0,       # صفر كما في المسار الناجح (unsloth توصيته)
        bias="none",
        target_modules=targets,
        use_gradient_checkpointing="unsloth",   # نسخة unsloth — أسرع وأخف من HF القياسية
        random_state=a.seed,
        max_seq_length=a.max_length,
    )
    model.print_trainable_parameters()

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
        # gradient_checkpointing: unsloth يديره عبر use_gradient_checkpointing="unsloth" —
        # تفعيل هذا هنا سيطبق مرتين (unsloth + HF) فيضر الأداء/الذاكرة.
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

    callbacks = []  # LiveLossLogger لاحقاً
    if eval_ds is not None and len(eval_ds) > 0:
        # بوابة lora_gate: فرق "التعلم الحقيقي" عن "الحفظ" — قبل أي تشغيل طويل
        args.eval_strategy = "steps"
        args.eval_steps = a.eval_steps
        args.metric_for_best_model = "eval_loss"
        args.greater_is_better = False
        args.load_best_model_at_end = True
        args.per_device_eval_batch_size = 1
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=a.es_patience))
        print(f"[eval] مفعّل كل {a.eval_steps} خطوات ({len(eval_ds):,} سجل): "
              f"best-keep + early-stop patience={a.es_patience}", flush=True)
    else:
        print("[eval] معطّل (eval_frac=0) — تحذير: لا حكم ضد الحفظ", flush=True)

    callbacks.insert(0, LiveLossLogger(os.path.join(a.output, "loss_log.jsonl"),
                                       a.max_length * a.grad_accum))

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks,  # LiveLossLogger دقة أدناه
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
