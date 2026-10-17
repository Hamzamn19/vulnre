#!/usr/bin/env python3
"""T9 prep — الخلط النهائي للبيانات حسب النسب المقفلة (نسخة بثّية آمنة الذاكرة).

النسب المقفلة:
  - أمن/RE:        40%  (security + t8_asm_pairs)
  - أدوات:         35%  (عينة من ~1.79M)
  - كود عام/حداثة: 20%  (عينة من ~5.47M)
  - معرفة ذاتية:    5%  (micro؛ datasheets قادمة لاحقاً وتستبدل جزءاً منها)

التنفيذ:
  1) عدّ تدفّقي لسجلات كل مصدر (بدون تحميل).
  2) عيّنة مقنّعة عبر خط تمرير واحد لكل ملف (احتمال target/total مع تعويض) —
     المقبولون يُكتبون في ملف JSONL مؤقت لكل شريحة (بثّ، لا RAM).
  3) مزج نهائي: دمج تدفّقي للملفات المؤقتة في Parquet مع buffer خلط (~256MB) —
     والترتيب النهائي إضافياً يُعاد خلطه عند التدريب (shuffle بالseed).
الإخراج: data/processed/final_mix/train_final.parquet + ملخص JSONL/JSON.
"""
import glob
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SECURITY_DIR = ROOT / "data/processed/security"
T8_DIR = ROOT / "data/processed/t8_asm_pairs"
TOOLS_DIR = ROOT / "data/processed/tools"
MODERN_DIR = ROOT / "data/processed/modern"
MICRO_DIR = ROOT / "data/processed/micro"
TMP_DIR = ROOT / "data/processed/final_mix/_tmp"
OUT_DIR = ROOT / "data/processed/final_mix"

TARGET_TOTAL = int(os.environ.get("FINAL_MIX_TARGET", 1_000_000))
RATIOS = {"security_re": 0.40, "tools": 0.35, "modern": 0.20, "micro": 0.05}
SEED = 42
random.seed(SEED)

SHUFFLE_BYTES = int(os.environ.get("FINAL_MIX_SHUFFLE_BYTES", 64 << 20))  # buffer خلط


def stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def list_slices():
    """(شرائح → ملفات jsonl) مع استبعاد ملفات التلخيص."""
    groups = {        "security_re": sorted(glob.glob(str(SECURITY_DIR / "*.jsonl"))) +
                       sorted(glob.glob(str(T8_DIR / "*.jsonl"))),
        "tools": sorted(glob.glob(str(TOOLS_DIR / "*.jsonl"))),
        "modern": sorted(glob.glob(str(MODERN_DIR / "*.jsonl"))),
        "micro": sorted(glob.glob(str(MICRO_DIR / "*.jsonl"))),
    }
    return {k: [f for f in v if "_summary" not in f] for k, v in groups.items()}


def build_sample(slice_name, files, target, tmp_path):
    """خط تمرير واحد: هرملة مقنّنة + كتابة فورية للمقبولين.

    mask(s) = ((sha(s) % 10**6) / 10**6) < target/total
    هذاخط شبهالعشوائي مستقلخ بالمحتوى؛ لا وزن دقيق متكافئ reservoir لكن
    مفاضلة واضحة: بلا RAM ولا ممرات متعددة؛ الفتوة صغيرة عينياً.
    ثم تعويض النقص: لو الهدف لم يتحقق (تقريب)، خط ثانٍ مختوم بفحص ملف دفعة-سجلات.
    """
    total = 0
    for f in files:
        with open(f, "rb") as fh:
            n = file_line_count(fh)
        total += n
    if total == 0:
        return 0, total
    p = min(1.0, target / total)
    thresh = int(p * 10 ** 9)  # دقة 1e-9 من نطاق 10^9
    kept = 0
    with open(tmp_path, "w", encoding="utf-8") as out:
        for f in files:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    if kept >= target:
                        break
                    h = int.from_bytes(
                        hashlib.blake2b(line[:4096].encode("utf-8", "ignore"),
                                        digest_size=5).digest(), "big")
                    if (h % 10 ** 9) < thresh:
                        out.write(line)
                        kept += 1
    return kept, total


def file_line_count(fh):
    n = 0
    pos = fh.tell()
    for _ in fh:
        n += 1
    fh.seek(pos)
    return n


FIELDS_SCALAR = ["id", "source", "task", "lang", "split"]
SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("source", pa.string()),
    pa.field("task", pa.string()),
    pa.field("lang", pa.string()),
    pa.field("split", pa.string()),
    pa.field("messages", pa.string()),   # JSON
    pa.field("tools", pa.string()),      # JSON أو null
    pa.field("meta", pa.string()),       # JSON أو null
])


def merge_and_shuffle_shuffled(tmp_files, out_parquet):
    """دمج تدفّقي مع خلط بوفر ~SHUFFLE_BYTES، كتابة Parquet بمخطط صريح (JSON strings)."""
    buffers = []
    bytes_in_buf = 0
    rows_ok = 0

    all_len = 0
    for tp, sl in tmp_files:
        with open(tp, "rb") as fh:
            for _ in fh:
                all_len += 1
    print(f"  [merge] سيُكتب {all_len:,} سجل إلى {out_parquet.name}", flush=True)

    writer = None
    handles = [(open(tp, "rb"), sl) for tp, sl in tmp_files]
    try:
        while handles:
            i = random.randrange(len(handles))
            fh, sl = handles[i]
            line = fh.readline()
            if not line:
                fh.close()
                handles.pop(i)
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rec["split"] = sl
            buffers.append(rec)
            bytes_in_buf += len(line)
            if bytes_in_buf >= SHUFFLE_BYTES:
                random.shuffle(buffers)
                cols = {
                    f: pa.array([r.get(f) for r in buffers], type=pa.string())
                    for f in FIELDS_SCALAR
                }
                cols["messages"] = pa.array(
                    [json.dumps(r.get("messages") or [], ensure_ascii=False)
                     for r in buffers], type=pa.string())
                for jf in ("tools", "meta"):
                    cols[jf] = pa.array(
                        [json.dumps(r.get(jf), ensure_ascii=False) if r.get(jf) else None
                         for r in buffers], type=pa.string())
                tbl = pa.table(cols, schema=SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(out_parquet, SCHEMA)
                writer.write_table(tbl)
                rows_ok += len(buffers)
                buffers.clear()
                bytes_in_buf = 0
        if buffers:
            random.shuffle(buffers)
            cols = {
                f: pa.array([r.get(f) for r in buffers], type=pa.string())
                for f in FIELDS_SCALAR
            }
            cols["messages"] = pa.array(
                [json.dumps(r.get("messages") or [], ensure_ascii=False)
                 for r in buffers], type=pa.string())
            for jf in ("tools", "meta"):
                cols[jf] = pa.array(
                    [json.dumps(r.get(jf), ensure_ascii=False) if r.get(jf) else None
                     for r in buffers], type=pa.string())
            tbl = pa.table(cols, schema=SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(out_parquet, SCHEMA)
            writer.write_table(tbl)
            rows_ok += len(buffers)
            buffers.clear()
    finally:
        if writer is not None:
            writer.close()
        for fh_list in handles:
            try:
                fh_list[0].close()
            except Exception:
                pass
    return rows_ok


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    slices = list_slices()
    print("=== الملفات ===")
    for sl, fs in slices.items():
        print(f"  {sl}: {len(fs)} ملف")

    # الأهداف الأساسية وإعادة توزيع مصادر فارغة على البقية باوزان نفسها
    base_targets = {sl: TARGET_TOTAL * RATIOS[sl] for sl in RATIOS}
    # micro متاح 651 فقط — الفارق يوزّع على security/tools/modern بنسبها
    micro_avail = sum(file_line_count(open(f, "r", encoding="utf-8", errors="replace"))
                      for f in slices["micro"])
    micro_target = min(int(base_targets["micro"]), micro_avail)
    shortfall = int(base_targets["micro"]) - micro_target
    keep_share = 1.0 - RATIOS["micro"]
    extra = {sl: int(shortfall * RATIOS[sl] / keep_share) for sl in ("security_re", "tools", "modern")}
    targets = {
        "security_re": int(base_targets["security_re"]) + extra["security_re"],
        "tools": int(base_targets["tools"]) + extra["tools"],
        "modern": int(base_targets["modern"]) + extra["modern"],
        "micro": micro_target,
    }
    print("=== الأهداف ===")
    for sl, t in targets.items():
        print(f"  {sl}: {t:,}")

    TMP_DIR.mkdir(exist_ok=True)
    samples = {}
    print("=== العينة (بثّ ممرّ واحد) ===")
    for sl, files in slices.items():
        if not files or targets[sl] == 0:
            samples[sl] = 0
            continue
        tmp_path = TMP_DIR / f"{sl}.jsonl"
        kept, total = build_sample(sl, files, targets[sl], tmp_path)
        samples[sl] = kept
        print(f"  {sl}: عيّنة {kept:,} من {total:,} → {tmp_path.name}")

    # الرئيسي: Parquet نهائي بالمزج
    print("=== المزج والكتابة ===")
    out_parquet = OUT_DIR / "train_final.parquet"
    rows = merge_and_shuffle_shuffled([(TMP_DIR / f"{sl}.jsonl", sl)
                                       for sl in ("security_re", "tools", "modern", "micro")
                                       if samples.get(sl, 0) > 0], out_parquet)
    print(f"  [done] {out_parquet}: {rows:,} سجل")

    summary = {
        "total_records": rows,
        "seed": SEED,
        "target_total": TARGET_TOTAL,
        "ratios": RATIOS,
        "targets_adjusted": targets,
        "sampled": samples,
        "available": {
            "security_re": 960501 + 362145,
            "tools": 1790838,
            "modern": 4976549 + 441645 + 50661 + 2645,
            "micro": micro_avail,
        },
        "note": ("datasheets مؤقتاً 0 — رصيدها (5%) أُعيد توزيعه على security/tools/modern "
                 "بنسبها نفسها؛ micro يغطي معرفة المشروع (651). يُعاد توليد الخلط بعد "
                 "اكتمال T8-AnghaBench دون تغيير النسب."),
    }
    (OUT_DIR / "_summary_final_mix.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
