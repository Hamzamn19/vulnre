#!/usr/bin/env python3
"""T7 — فحص مخططات المصادر الخام (آمن الذاكرة): أول سجل من كل مصدر مكتمل.

ملاحظة: النسخة السابقة حمّلت ملفات ضخمة كاملة (json.load لـ 256MB + pandas.read_parquet
لملفات 1–2GB) → ضغط ذاكرة متكرر. هذه النسخة:
- parquet: pyarrow iter_batches لسجل واحد + تخطي قراءة البيانات إذا كانت أول row group > 300MB
- json كبير: رأس نصي فقط
- jsonl: قراءة أول سطر بحد أقصى 4MB
"""
import json
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def enc(x, n=200):
    s = str(x).replace("\n", "\\n")
    return s[:n] + ("…" if len(s) > n else "")


def _files(dirp):
    if not dirp.exists():
        return []
    return [p for p in sorted(dirp.rglob("*")) if p.is_file() and ".cache" not in p.parts]


def first_file(dirp, exts):
    for ext in exts:
        hits = [p for p in _files(dirp) if p.suffix == ext]
        if hits:
            return hits[0]
    return None


def peek_json(path, n=220, max_load=90_000_000):
    if not path.exists():
        print(f"  MISSING {path}")
        return
    size = path.stat().st_size
    print(f"  file: {path.name} ({size/1e6:.1f}MB) [json]")
    if size > max_load:
        with open(path, errors="replace") as fh:
            print("  [big json → raw head]", enc(fh.read(1200), 1200))
        return
    try:
        data = json.loads(path.read_text(errors="replace"))
        if isinstance(data, list):
            print(f"  list[{len(data)}]")
            r0 = data[0] if data else None
            if isinstance(r0, dict):
                print(f"  keys: {list(r0.keys())}")
                for k in list(r0)[:6]:
                    print(f"    {k}: {enc(r0[k], n)}")
        elif isinstance(data, dict):
            print(f"  dict keys: {list(data.keys())[:15]}")
            k0 = next(iter(data), None)
            if k0 is not None:
                print(f"  first value: {enc(data[k0], n)}")
    except Exception as e:
        print(f"  ERR json: {e}")


def peek_jsonl(path, n=220):
    if not path.exists():
        print(f"  MISSING {path}")
        return
    print(f"  file: {path.name} ({path.stat().st_size/1e9:.2f}GB) [jsonl]")
    try:
        with open(path, errors="replace") as f:
            line = f.readline(4_000_000)
        if not line:
            print("  EMPTY")
            return
        try:
            rec = json.loads(line)
        except Exception:
            print("  (raw head)", enc(line, 600))
            return
        if isinstance(rec, dict):
            print(f"  keys: {list(rec.keys())}")
            for k in list(rec)[:7]:
                print(f"    {k}: {enc(rec[k], n)}")
        else:
            print(f"  type {type(rec)}: {enc(rec, n)}")
    except Exception as e:
        print(f"  ERR jsonl: {e}")


def peek_parquet(path, n=200, max_rg=300_000_000):
    if not path.exists():
        print(f"  MISSING {path}")
        return
    print(f"  file: {path.name} ({path.stat().st_size/1e9:.3f}GB) [parquet]")
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        print(f"  cols({len(cols)}): {cols[:16]}")
        nrg = pf.metadata.num_row_groups
        rg0 = pf.metadata.row_group(0).total_byte_size if nrg else 0
        if rg0 > max_rg:
            print(f"  (first row group {rg0/1e6:.0f}MB > cap → schema only)")
            return
        batch = next(pf.iter_batches(batch_size=1))
        for c in cols[:7]:
            v = batch.column(batch.schema.get_field_index(c))[0].as_py()
            print(f"    {c}: {enc(v, n)}")
    except Exception as e:
        print(f"  ERR parquet: {e}")


def peek_any(dirp, label=None):
    label = label or str(dirp)
    print(f"\n## {label}")
    files = _files(dirp)
    if not files:
        print("  (no files / missing)")
        return
    for p in files[:10]:
        print(f"  {p.relative_to(RAW)} ({p.stat().st_size/1e6:.1f}MB)")
    f = first_file(dirp, [".parquet", ".jsonl", ".json", ".csv"])
    if not f:
        return
    print(f"  >> peek first: {f.name}")
    if f.suffix == ".parquet":
        peek_parquet(f)
    elif f.suffix == ".jsonl":
        peek_jsonl(f)
    elif f.suffix == ".json" and f.stat().st_size < 400e6:
        peek_json(f)
    else:
        with open(f, errors="replace") as fh:
            print("  [head]", enc(fh.read(500), 500))


if __name__ == "__main__":
    print("=" * 30, "VULNRE SCHEMA PEEK", "=" * 30)

    print("\n## ToolACE (json)")
    peek_json(RAW / "toolace/data.json")

    print("\n## ToolMind graphsyn (jsonl)")
    peek_jsonl(RAW / "toolmind/graph_syn_datasets/graphsyn.jsonl")

    print("\n## ToolMind open query sample")
    peek_jsonl(RAW / "toolmind/open_datasets/xlam-function-calling-60k-query.jsonl")

    print("\n## Nemotron Terminal adapters")
    peek_parquet(RAW / "nemotron-terminal/dataset_adapters/code.parquet")
    peek_parquet(RAW / "nemotron-terminal/dataset_adapters/swe.parquet")

    print("\n## Nemotron Terminal synthetic (first parquet)")
    st = RAW / "nemotron-terminal/synthetic_tasks"
    f = first_file(st, [".parquet"])
    if f:
        print(f"  hint: {f.relative_to(RAW)}")
        peek_parquet(f)
    else:
        peek_any(st)

    print("\n## Nemotron agentic (jsonl)")
    peek_jsonl(RAW / "nemotron-agentic/data/tool_calling.jsonl")
    peek_jsonl(RAW / "nemotron-agentic/data/interactive_agent.jsonl")
    peek_jsonl(RAW / "nemotron-agentic/data/search.jsonl")

    print("\n## SWE-rebench OpenHands trajectories")
    peek_parquet(RAW / "swe-rebench-openhands/trajectories.parquet")

    peek_any(RAW / "agent-flan/data", "Agent-FLAN (data dir)")
    peek_any(RAW / "glaive-fc", "Glaive FC v2 (dir)")
    peek_any(RAW / "xlam", "xlam")
    peek_any(RAW / "bigvul", "BigVul")
    peek_any(RAW / "diversevul", "DiverseVul")
    peek_any(RAW / "mind2web", "Mind2Web")
    peek_any(RAW / "fenrir", "Fenrir")

    print("\nDONE")
