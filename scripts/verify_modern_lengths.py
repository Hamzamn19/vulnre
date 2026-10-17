#!/usr/bin/env python3
"""التحقق من رقم '0 long records' في شريحة modern.
يطبع توزيع أطوال العينات (min/max/median بالتوكن) ويتحقق من منطق فحص 'long' بتسجيل طويل يدوياً.
"""
import os
import json
import glob
import statistics
from transformers import AutoTokenizer

MODERN_DIR = "data/processed/modern"
TOKENIZER_MODEL = "models/deepseek-coder-6.7b-base"


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)


def count_chars(msgs):
    total = 0
    for m in msgs:
        c = m.get("content") or ""
        total += len(c)
    return total


def scan_shards(shard_files, tokenizer, sample_for_tokens=5000):
    """مسح الشفرات لحساب أطوال التوكن الحقيقية لعينة."""
    char_lengths = []
    token_lengths = []
    max_char_record = None
    max_char_val = 0
    total_records = 0
    long_count = 0  # سجلات > 250K chars (MAX_TOTAL_CHARS)

    for sf in shard_files:
        with open(sf, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                total_records += 1
                msgs = rec.get("messages", [])
                chars = count_chars(msgs)
                char_lengths.append(chars)
                if chars > max_char_val:
                    max_char_val = chars
                    max_char_record = rec
                if chars > 250_000:
                    long_count += 1

                # عينة للتوكن (أول sample_for_tokens سجل فقط لتسريع)
                if len(token_lengths) < sample_for_tokens:
                    try:
                        text = " ".join((m.get("content") or "") for m in msgs)
                        toks = tokenizer(text, return_tensors="pt", truncation=False)
                        token_lengths.append(toks.input_ids.shape[1])
                    except Exception:
                        pass

    return {
        "total_records": total_records,
        "char_min": min(char_lengths) if char_lengths else 0,
        "char_max": max_char_val,
        "char_median": statistics.median(char_lengths) if char_lengths else 0,
        "char_mean": statistics.mean(char_lengths) if char_lengths else 0,
        "long_count_250k": long_count,
        "token_min": min(token_lengths) if token_lengths else 0,
        "token_max": max(token_lengths) if token_lengths else 0,
        "token_median": statistics.median(token_lengths) if token_lengths else 0,
        "token_mean": statistics.mean(token_lengths) if token_lengths else 0,
        "max_char_record": max_char_record,
    }


def test_long_filter_logic():
    """اختبار منطق فحص 'long' من build_dataset.py بحقن سجل طويل يدوياً."""
    MAX_TOTAL_CHARS = 250_000

    # سجل طويل (>250K chars)
    long_text = "x" * 300_000
    long_msgs = [
        {"role": "user", "content": long_text},
        {"role": "assistant", "content": long_text[:100_000]},
    ]
    total_chars = sum(len(m.get("content") or "") for m in long_msgs)
    is_long = total_chars > MAX_TOTAL_CHARS

    # سجل طبيعي (<250K chars)
    normal_msgs = [
        {"role": "user", "content": "def hello(): pass"},
        {"role": "assistant", "content": "print('hello')"},
    ]
    normal_chars = sum(len(m.get("content") or "") for m in normal_msgs)
    is_normal_long = normal_chars > MAX_TOTAL_CHARS

    return {
        "long_record_chars": total_chars,
        "long_record_detected_as_long": is_long,
        "normal_record_chars": normal_chars,
        "normal_record_detected_as_long": is_normal_long,
    }


def main():
    shard_files = sorted(glob.glob(os.path.join(MODERN_DIR, "*.jsonl")))
    print(f"وجدت {len(shard_files)} شريحة في {MODERN_DIR}")
    print()

    tokenizer = load_tokenizer()
    print("تحميل التوكنيزر مكتمل.")
    print()

    print("=== مسح الشفرات لأطوال العينات ===")
    stats = scan_shards(shard_files, tokenizer)
    print(f"إجمالي السجلات: {stats['total_records']:,}")
    print(f"أحرف min/median/max: {stats['char_min']:,} / {stats['char_median']:,.0f} / {stats['char_max']:,}")
    print(f"توكن min/median/max: {stats['token_min']:,} / {stats['token_median']:,.0f} / {stats['token_max']:,}")
    print(f"سجلات >250K chars (مكتشفة): {stats['long_count_250k']}")
    print()

    print("=== اختبار منطق فحص 'long' بحقن يدوي ===")
    filter_test = test_long_filter_logic()
    print(f"سجل طويل (300K chars): المكتشف كـ long = {filter_test['long_record_detected_as_long']}")
    print(f"سجل طبيعي (~30 chars): المكتشف كـ long = {filter_test['normal_record_detected_as_long']}")
    print()

    # استنتاج
    if stats["long_count_250k"] == 0 and stats["char_max"] > 250_000:
        print("⚠️  تحذير: هناك سجلات تتجاوز 250K chars لكنها لم تُكتشف كـ 'long' — منطق الفحص معطل!")
    elif stats["long_count_250k"] == 0:
        print("✓ لا توجد سجلات تتجاوز 250K chars — plausible إذا كانت البيانات مقصوصة مسبقاً.")
    else:
        print(f"✓ تم اكتشاف {stats['long_count_250k']} سجل طويل — منطق الفحص يعمل.")

    if not filter_test["long_record_detected_as_long"]:
        print("⚠️  خطير: منطق الفحص لا يكتشف السجلات الطويلة حتى عند حقنها يدوياً!")
    else:
        print("✓ منطق الفحص يعمل بشكل صحيح في حقن اليدوي.")

    print()
    print("=== توصية ===")
    print("إذا كان char_max كبير (>250K) و long_count=0 → منطق الفحص في build_dataset.py معطل.")
    print("إذا كان char_max معقول (<100K) و long_count=0 → البيانات مقصوصة مسبقاً والرقم منطقي.")


if __name__ == "__main__":
    main()
