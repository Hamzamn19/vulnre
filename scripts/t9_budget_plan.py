#!/usr/bin/env python3
"""حساب سقف الخطوات للتشغيل الفعلي T9 من زمن الخطوة المقاس في البايلوت.

مبدأ لا اسمياً (موافقة المستخدم): max_steps = floor(ميزانية ساعات / زمن\u200alخطوة فعلي)
مع هامش أمان 10% + منح بين تدريب العملي والحفظ — السقف من زمن الخطوة **ثابت** في
الوسط الأخير من البايلوت (لتجنب برودة البداية العربية).
"""
import json
import sys
from pathlib import Path

BUDGET_H = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
LOG = Path("data/processed/t9_pilot/loss_log.jsonl")

rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
losses = [(r["step"], r["wall_h"], r["loss"]) for r in rows if "loss" in r]
if not losses:
    sys.exit("no loss rows — pilot not complete?")

last_h = losses[-1][1]
per_step_h = last_h / losses[-1][0]          # متوسط عام (شامل الإحماء)

# steady-state: زمن الخطوة من آخر ~60% من المسافة
if len(losses) >= 4:
    mid = losses[max(1, int(len(losses) * 0.4))]
    per_step_h = (losses[-1][1] - mid[1]) / (losses[-1][0] - mid[0])

max_steps = int(BUDGET_H * 0.9 / per_step_h)   # هامش 10% للنهاية/الحفظ
est_tokens = max_steps * 65536 / 1e6

print(json.dumps({
    "pilot_loss_first": losses[0][2],
    "pilot_loss_last": losses[-1][2],
    "pilot_steps": losses[-1][0],
    "pilot_wall_h": round(last_h, 3),
    "per_step_min_steady": round(per_step_h * 60, 3),
    "budget_hours": BUDGET_H,
    "recommended_max_steps": max_steps,
    "est_tokens_M": round(est_tokens, 1),
    "runtime_max_sec_suggested": int(BUDGET_H * 3600 * 1.05),
}, indent=2))
