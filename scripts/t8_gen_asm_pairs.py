#!/usr/bin/env python3
"""T8 — توليد أزواج Assembly (gcc -O0/-O2 -S) للتدريب على RE.
المصادر (بترتيب الأولوية):
1. REx86 (atul10/ على HF) — يحتوي على assembly_code و decompiled_code_original مباشرة، لا حاجة compile.
2. AnghaBench (من GitHub brenocfg/AnghaBench) — C source compilable، compile إلى assembly via gcc -O0/-O2.

إخراج بالسجل القانوني: task ∈ {c_to_asm, asm_to_c}.
مقيد بـ ≤8K توكن لكل سجل (R8).
"""
import os
import re as _re
import json
import subprocess
import tempfile
import hashlib
import argparse
import multiprocessing
from pathlib import Path
from transformers import AutoTokenizer
import pyarrow.parquet as pq

RAW = Path("data/raw")
PROC = Path("data/processed")
MAX_TOTAL_CHARS = 250_000
MAX_USER_CHARS = 20_000
MAX_ASSISTANT_CHARS = 24_000
TOKENIZER_MODEL = "models/deepseek-coder-6.7b-base"
WORKERS = int(os.environ.get("T8_WORKERS",
                             max(1, min(12, (os.cpu_count() or 4) - 1))))

DET_SYS = ("You are an expert in binary analysis and reverse engineering (x86/x64 assembly). "
           "Answer precisely. When asked for JSON, output only valid JSON.")


def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)


ANGHA_PLACEHOLDER = "/*<<< orphan*/"   # بديل الأنواع المعقّدة في AnghaBench


def clean_c(c_code):
    """تنظيف بقايا استخراج AnghaBench: إزالة علامة /*<<< orphan*/ (يتبقى 'int')."""
    if ANGHA_PLACEHOLDER in c_code:
        c_code = c_code.replace(ANGHA_PLACEHOLDER, "")
    return c_code


def c_to_asm(c_code, opt_level):
    """Compile C function to assembly using gcc -O{level} -S.

    ملاحظات gcc 16:
      - -std=gnu89: ملفات AnghaBench تعرف `typedef int bool` (ممنوع في c23)
      - -w: قمع تحذيرات (تعارضات printf و builtin) — نحن نريد -S فقط بلا لينك
      - prelude بلا includes: بعض الملفات تعرّف printf/malloc بنفسها فتصطدم بالترويسات
    """
    c_code = clean_c(c_code)
    with tempfile.TemporaryDirectory() as td:
        c_path = os.path.join(td, "input.c")
        asm_path = os.path.join(td, "output.s")
        wrapper = f"""typedef unsigned long size_t;
typedef long intptr_t;
typedef unsigned long uintptr_t;
typedef int bool;
#define NULL ((void*)0)
{c_code}
"""
        with open(c_path, "w") as f:
            f.write(wrapper)
        try:
            result = subprocess.run(
                ["gcc", "-std=gnu89", "-w", f"-O{opt_level}", "-S", c_path, "-o", asm_path],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return None, f"gcc error: {result.stderr[:300]}"
            with open(asm_path, "r") as f:
                asm = f.read()
            return asm, None
        except subprocess.TimeoutExpired:
            return None, "gcc timeout"
        except Exception as e:
            return None, str(e)


def content_key(msgs):
    h = hashlib.blake2b(digest_size=8)
    for m in msgs:
        h.update(((m.get("content") or "")[:4096]).encode("utf-8", "ignore"))
        h.update(b"\x1e")
    return int.from_bytes(h.digest(), "big")


class ShardWriter:
    def __init__(self, out_dir: Path, name: str):
        self.out_dir = out_dir
        self.name = name
        self.shard = 0
        self.chars = 0
        self.fh = None
        self._open()

    def _open(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        p = self.out_dir / f"{self.name}-shard-{self.shard:05d}.jsonl"
        self.fh = open(p, "w", encoding="utf-8")
        self.chars = 0

    def write(self, rec):
        line = json.dumps(rec, ensure_ascii=False)
        self.fh.write(line + "\n")
        self.chars += len(line) + 1
        if self.chars >= 192_000_000:
            self.fh.close()
            self.shard += 1
            self._open()

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


class Ctx:
    def __init__(self, out_dir: Path, limit: int = 0):
        self.out_dir = Path(out_dir)
        self.limit = limit
        self.writers = {}
        self.stats = {}
        self.dedupe = set()

    def st(self, source):
        return self.stats.setdefault(source, {"kept": 0, "dups": 0, "long": 0, "empty": 0, "gcc_fail": 0})

    def writer(self, source):
        w = self.writers.get(source)
        if w is None:
            w = ShardWriter(self.out_dir, source)
            self.writers[source] = w
        return w

    def reached(self, source):
        return bool(self.limit) and self.st(source)["kept"] >= self.limit

    def emit(self, source, task, msgs, lang="asm", tools=None, meta=None):
        st = self.st(source)
        msgs = [m for m in msgs if (m.get("content") or m.get("tool_calls"))]
        if not msgs:
            st["empty"] += 1
            return False
        total_chars = sum(len(m.get("content") or "") for m in msgs)
        if total_chars > MAX_TOTAL_CHARS:
            st["long"] += 1
            return False
        k = content_key(msgs)
        if k in self.dedupe:
            st["dups"] += 1
            return False
        self.dedupe.add(k)
        rec = {"id": f"{source}-{st['kept']:07d}", "source": source, "task": task,
               "lang": lang, "messages": msgs}
        if tools:
            rec["tools"] = tools
        if meta:
            rec["meta"] = {k2: v for k2, v in meta.items() if v not in (None, "", [])}
        self.writer(source).write(rec)
        st["kept"] += 1
        if st["kept"] % 50_000 == 0:
            print(f"  … {source}: {st['kept']:,} kept", flush=True)
        return True

    def close(self):
        for w in self.writers.values():
            w.close()

    def summary(self, slice_name):
        s = {"slice": slice_name, "sources": self.stats}
        p = self.out_dir / f"_summary_{slice_name}.json"
        p.write_text(json.dumps(s, ensure_ascii=False, indent=2))
        print(json.dumps(s, ensure_ascii=False, indent=2), flush=True)


def msg(role, content, cap=MAX_ASSISTANT_CHARS, **extra):
    m = {"role": role, "content": (content or "")[:cap]}
    for k, v in extra.items():
        if v:
            m[k] = v
    return m


def conv_rex86(ctx, tokenizer, limit):
    """REx86 من atul10/ على HF — يحتوي على assembly_code و decompiled_code_original."""
    src = "rex86"
    count = 0
    for opt_level in ["O1", "O2"]:
        base = RAW / f"rex86_{opt_level.lower()}_x86_{opt_level.lower()}" / "data"
        if not base.exists():
            print(f"  [rex86] skipping {opt_level}: {base} not found", flush=True)
            continue
        for path in sorted(base.glob("*.parquet")):
            if ctx.reached(src):
                return
            for r in pq.read_table(path, columns=["assembly_code", "decompiled_code_original", "original_function_name", "opt_level", "arch"]).to_pylist():
                if ctx.reached(src):
                    return
                count += 1
                if limit and count > limit:
                    return
                asm = r.get("assembly_code") or ""
                c_code = r.get("decompiled_code_original") or ""
                opt = r.get("opt_level") or "unknown"
                func_name = r.get("original_function_name") or ""
                if not asm or not c_code:
                    continue
                meta = {"opt": opt, "function": func_name, "source": "rex86"}

                # C -> asm
                msgs = [
                    msg("system", DET_SYS, 2500),
                    msg("user", f"Generate the x86-64 assembly for the following C function:\n\n```c\n{c_code}\n```", MAX_USER_CHARS),
                    msg("assistant", f"```asm\n{asm}\n```", MAX_ASSISTANT_CHARS),
                ]
                ctx.emit(src, "c_to_asm", msgs, lang="asm", meta={**meta, "direction": "c_to_asm"})

                # asm -> c
                msgs = [
                    msg("system", DET_SYS, 2500),
                    msg("user", f"Decompile the following x86-64 assembly to C:\n\n```asm\n{asm}\n```", MAX_USER_CHARS),
                    msg("assistant", f"```c\n{c_code}\n```", MAX_ASSISTANT_CHARS),
                ]
                ctx.emit(src, "asm_to_c", msgs, lang="asm", meta={**meta, "direction": "asm_to_c"})


def _ab_worker(args):
    """عامل multiprocessing: ينظّف ويرجم ملفاً واحداً بمستويين ويُرجع النتيجة."""
    (c_path, rel) = args
    try:
        with open(c_path, "r") as f:
            c_code = f.read()
    except Exception:
        return None
    if not c_code.strip() or len(c_code) > 131072:
        return None
    c_clean = clean_c(c_code)
    asm_o0, _ = c_to_asm(c_clean, 0)
    asm_o2, _ = c_to_asm(c_clean, 2)
    if not asm_o0 and not asm_o2:
        return ("fail", rel)
    return ("ok", rel, c_clean, asm_o0, asm_o2)


def conv_anghabench(ctx, tokenizer, limit):
    """AnghaBench (brenocfg/AnghaBench) — مليون ملف C مستقل، ترجمة متوازية إلى asm."""
    src = "anghabench"
    base = RAW / "anghabench"
    src_ctx = ctx.st(src)
    print(f"  [anghabench] compiling with {WORKERS} workers, limit={limit or 'all'}",
          flush=True)
    files = (os.path.join(dp, fn)
             for dp, dns, fns in os.walk(base) for fn in fns if fn.endswith(".c"))
    n_seen = n_ok = 0
    pool = multiprocessing.Pool(processes=WORKERS)
    try:
        got = 0
        for res in pool.imap(_ab_worker, ((p, os.path.relpath(p, base)) for p in files),
                             chunksize=64):
            n_seen += 1
            if res is not None and res[0] == "fail":
                src_ctx["gcc_fail"] += 1
                continue
            if res is None:
                continue
            n_ok += 1
            _, rel, c_clean, asm_o0, asm_o2 = res
            meta = {"source": "anghabench", "file": rel}
            if asm_o0:
                msgs = [
                    msg("system", DET_SYS, 2500),
                    msg("user", f"Generate the x86-64 assembly for the following C function (gcc -O0):\n\n```c\n{c_clean}\n```", MAX_USER_CHARS),
                    msg("assistant", f"```asm\n{asm_o0}\n```", MAX_ASSISTANT_CHARS),
                ]
                if ctx.emit(src, "c_to_asm", msgs, lang="asm",
                            meta={**meta, "opt": "O0", "direction": "c_to_asm"}):
                    got += 1
            if asm_o2:
                msgs = [
                    msg("system", DET_SYS, 2500),
                    msg("user", f"Generate the x86-64 assembly for the following C function (gcc -O2):\n\n```c\n{c_clean}\n```", MAX_USER_CHARS),
                    msg("assistant", f"```asm\n{asm_o2}\n```", MAX_ASSISTANT_CHARS),
                ]
                if ctx.emit(src, "c_to_asm", msgs, lang="asm",
                            meta={**meta, "opt": "O2", "direction": "c_to_asm"}):
                    got += 1
            if limit and got >= limit:
                break
            # نبّه كل 20 ألف ملف
            if n_seen % 20000 == 0:
                print(f"  [anghabench] seen={n_seen:,} ok={n_ok:,} kept={src_ctx['kept']:,}",
                      flush=True)
    finally:
        pool.terminate()
        pool.join()
    print(f"  [anghabench] done: seen={n_seen:,} compiled_ok={n_ok:,} kept={src_ctx['kept']:,}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description="T8 - Generate C<->Assembly pairs from REx86 + AnghaBench")
    ap.add_argument("--limit", type=int, default=0, help="max records per source (0 = all)")
    ap.add_argument("--out", default=str(PROC / "t8_asm_pairs"), help="output directory")
    ap.add_argument("--source", choices=["rex86", "anghabench", "all"], default="all",
                    help="which source(s) to process")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer()
    ctx = Ctx(out_dir, limit=a.limit)

    if a.source in ("rex86", "all"):
        print("== T8: REx86 asm pairs ==", flush=True)
        conv_rex86(ctx, tokenizer, limit=a.limit)
    if a.source in ("anghabench", "all"):
        print("== T8: AnghaBench asm pairs ==", flush=True)
        conv_anghabench(ctx, tokenizer, limit=a.limit)

    ctx.close()
    ctx.summary("t8_asm_pairs")


if __name__ == "__main__":
    main()
