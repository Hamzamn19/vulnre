#!/usr/bin/env python3
"""T7 — بايبلاين توحيد البيانات للتدريب: مصادر خام → سجل قانوني → شرائح JSONL.

السجل القانوني (Canonical Record):
  {
    "id":       "<source>-0000123",
    "source":   "bigvul",
    "task":     "vuln_detect" | "vuln_fix" | "tool_calling" | "agent_terminal" | "cyber_qa" | ...
    "lang":     "c" | "python" | "shell" | "text" | ...
    "messages": [{"role": "...", "content": "...", "tool_calls": ..., "name": ...}],
    "tools":    [OpenAI tool schemas] (اختياري),
    "meta":     {...} (اختياري، حقول صغيرة)
  }

الاستخدام:
  python build_dataset.py security [--limit N] [--out DIR]
  python build_dataset.py tools    [--limit N] [--out DIR]
  python build_dataset.py all      [--limit N] [--out DIR]
  python build_dataset.py stats    [--out DIR]

ملاحظات تصميم:
- بثّي بالكامل (jsonl سطر-سطر، parquet بـ iter_batches) → ذاكرة محدودة.
- إزالة التكرار عبر بصمة blake2b لكل مصدر/سجل (مجموعة مشتركة للجلسة).
- قصّ آمن للرسائل الطويلة + تخطي السجلات الأطول من السقف.
- JSONL للمراجعة؛ تحويل Parquet خطوة لاحقة (to_parquet) قبل التدريب.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

MAX_TOTAL_CHARS = 250_000
MAX_ASSISTANT_CHARS = 24_000
MAX_USER_CHARS = 16_000
MAX_TOOL_CHARS = 6_000
SHARD_CHARS = 192_000_000

DET_SYS = ("You are an expert vulnerability analyst for C/C++ source code "
           "(x86/x64, ARM, embedded/IoT/firmware). Answer precisely. "
           "When asked for a JSON verdict, output only valid JSON.")


# ---------------------------------------------------------------- helpers

def clean(s, cap):
    if not isinstance(s, str):
        try:
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
    if "\x00" in s:
        s = s.replace("\x00", "")
    if len(s) > cap:
        s = s[:cap] + "\n…[truncated]"
    return s


def msg(role, content, cap=MAX_ASSISTANT_CHARS, **extra):
    m = {"role": role, "content": clean(content, cap)}
    for k, v in extra.items():
        if v:
            m[k] = v
    return m


def norm_msg(m):
    role = m.get("role") or "assistant"
    cap = {"user": MAX_USER_CHARS, "tool": MAX_TOOL_CHARS, "system": 8000}.get(role, MAX_ASSISTANT_CHARS)
    out = {"role": role, "content": clean(m.get("content") or "", cap)}
    if m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    if m.get("name"):
        out["name"] = m["name"]
    return out


def total_chars(msgs):
    n = 0
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
    return n


def content_key(msgs):
    h = hashlib.blake2b(digest_size=8)
    for m in msgs:
        h.update(((m.get("content") or "")[:4096]).encode("utf-8", "ignore"))
        h.update(b"\x1e")
    return int.from_bytes(h.digest(), "big")


def cap_tools(tools, max_n=25, max_json=12_000):
    if not tools:
        return None
    t = list(tools[:max_n])
    while t and len(json.dumps(t, ensure_ascii=False)) > max_json:
        t = t[:-1]
    return t or None


# ---------------------------------------------------------------- iterators

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def iter_json_array(path, chunk=8 << 20, max_buf=64 << 20):
    """بثّ عناصر مصفوفة JSON كبيرة (مثل glaive 271MB) بذاكرة محدودة."""
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        buf = ""
        while "[" not in buf:
            more = f.read(chunk)
            if not more:
                return
            buf += more
        buf = buf[buf.index("[") + 1:]
        while True:
            buf = buf.lstrip()
            while buf.startswith(","):
                buf = buf[1:].lstrip()
            if buf.startswith("]"):
                return
            if not buf:
                more = f.read(chunk)
                if not more:
                    return
                buf += more
                continue
            try:
                obj, idx = dec.raw_decode(buf)
            except ValueError:
                more = f.read(chunk)
                if not more:
                    return
                buf += more
                if len(buf) > max_buf:
                    raise RuntimeError(f"JSON object larger than {max_buf} bytes in {path.name}")
                continue
            yield obj
            buf = buf[idx:]


def iter_parquet(path, columns=None, batch_size=64):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        names = batch.schema.names
        cols = {c: batch.column(i).to_pylist() for i, c in enumerate(names)}
        for i in range(batch.num_rows):
            yield {c: cols[c][i] for c in names}


# ---------------------------------------------------------------- writer / ctx

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
        if self.chars >= SHARD_CHARS:
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
        return self.stats.setdefault(source, {"kept": 0, "dups": 0, "long": 0, "empty": 0})

    def writer(self, source):
        w = self.writers.get(source)
        if w is None:
            w = ShardWriter(self.out_dir, source)
            self.writers[source] = w
        return w

    def reached(self, source):
        return bool(self.limit) and self.st(source)["kept"] >= self.limit

    def emit(self, source, task, msgs, lang="text", tools=None, meta=None):
        st = self.st(source)
        if self.reached(source):
            return False
        msgs = [m for m in msgs if (m.get("content") or m.get("tool_calls"))]
        if not msgs:
            st["empty"] += 1
            return False
        if total_chars(msgs) > MAX_TOTAL_CHARS:
            st["long"] += 1
            return False
        k = content_key(msgs)
        if k in self.dedupe:
            st["dups"] += 1
            return False
        self.dedupe.add(k)
        rec = {"id": f"{source}-{st['kept']:07d}", "source": source, "task": task,
               "lang": lang, "messages": msgs}
        t = cap_tools(tools)
        if t:
            rec["tools"] = t
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
        s = {"slice": slice_name, "time": time.strftime("%F %T"), "sources": self.stats}
        p = self.out_dir / f"_summary_{slice_name}.json"
        p.write_text(json.dumps(s, ensure_ascii=False, indent=2))
        print(json.dumps(s, ensure_ascii=False, indent=2), flush=True)


# ---------------------------------------------------------------- converters

def conv_fenrir(ctx):
    src = "fenrir"
    p = RAW / "fenrir" / "220426-CyberSec-Dataset_escaped.jsonl"
    for r in iter_jsonl(p):
        if ctx.reached(src):
            break
        msgs = [msg("system", r.get("system", ""), 2500),
                msg("user", r.get("user", ""), MAX_USER_CHARS),
                msg("assistant", r.get("assistant", ""))]
        ctx.emit(src, "cyber_qa", msgs, lang="text")


def _detect_pair(code, lang, vulnerable, cwe_list, reason):
    user = ("Analyze the following %s function for security vulnerabilities. "
            "Respond with JSON only: {\"vulnerable\": true|false, \"cwe\": [\"CWE-XXX\"], \"reason\": \"<short>\"}."
            "\n\n```%s\n%s\n```" % (lang.upper(), lang, clean(code, MAX_USER_CHARS)))
    a = {"vulnerable": bool(vulnerable), "cwe": list(cwe_list or []), "reason": reason}
    return [msg("system", DET_SYS, 2500), msg("user", user, MAX_USER_CHARS + 4000),
            msg("assistant", json.dumps(a, ensure_ascii=False), 2000)]


def conv_bigvul(ctx):
    src = "bigvul"
    for path in sorted((RAW / "bigvul" / "data").glob("*.parquet")):
        for r in iter_parquet(path, columns=["func_before", "func_after", "CWE ID", "project", "lang", "commit_id"]):
            if ctx.reached(src):
                break
            before = r.get("func_before") or ""
            after = r.get("func_after") or ""
            cwe = r.get("CWE ID") or ""
            lng = (r.get("lang") or "c").lower()
            meta = {"cwe": cwe, "project": r.get("project"), "commit": (r.get("commit_id") or "")[:12]}
            ctx.emit(src, "vuln_detect",
                     _detect_pair(before, lng, True, [cwe] if cwe else [],
                                  "Vulnerable code; weakness consistent with %s." % cwe),
                     lang=lng, meta=meta)
            user2 = ("Rewrite the following vulnerable %s function so it is safe. "
                     "Return only the fixed code in a code block.\n\n```%s\n%s\n```"
                     % (lng.upper(), lng, clean(before, MAX_USER_CHARS)))
            a2 = "```%s\n%s\n```" % (lng, clean(after, MAX_ASSISTANT_CHARS))
            ctx.emit(src, "vuln_fix",
                     [msg("system", DET_SYS, 2500), msg("user", user2, MAX_USER_CHARS + 4000),
                      msg("assistant", a2)],
                     lang=lng, meta=meta)


def conv_diversevul(ctx):
    src = "diversevul"
    for path in sorted((RAW / "diversevul" / "data").glob("*.parquet")):
        for r in iter_parquet(path, columns=["func", "target", "cwe", "project", "message"]):
            if ctx.reached(src):
                break
            func = r.get("func") or ""
            target = int(r.get("target") or 0)
            cwe = r.get("cwe") or []
            if target:
                reason = "Vulnerable code; weakness consistent with %s." % ", ".join(cwe) if cwe else "Vulnerable code."
            else:
                reason = "No vulnerability detected in this function."
            meta = {"cwe": ",".join(cwe), "project": r.get("project")}
            ctx.emit(src, "vuln_detect", _detect_pair(func, "c", target, cwe, reason),
                     lang="c", meta=meta)


def conv_nemotron_agentic(ctx):
    base = RAW / "nemotron-agentic" / "data"
    for fname, src, task in [
        ("tool_calling.jsonl", "nemotron_tool_calling", "tool_calling"),
        ("interactive_agent.jsonl", "nemotron_interactive", "agentic_interactive"),
        ("search.jsonl", "nemotron_search", "web_search_agent"),
    ]:
        p = base / fname
        for r in iter_jsonl(p):
            if ctx.reached(src):
                break
            msgs = [norm_msg(m) for m in (r.get("messages") or []) if isinstance(m, dict)]
            md = r.get("metadata") or {}
            meta = {"model": md.get("sdg_model") or r.get("model"), "file": fname}
            if r.get("domain"):
                meta["domain"] = r["domain"]
            ctx.emit(src, task, msgs, lang="text", tools=r.get("tools"), meta=meta)


def conv_nemotron_terminal(ctx):
    base = RAW / "nemotron-terminal"
    for fname, src, step in [
        ("code.parquet", "nemotron_terminal_code", 1),
        ("swe.parquet", "nemotron_terminal_swe", 1),
        ("math.parquet", "nemotron_terminal_math", 8),  # عيّنة 1/8 فقط
    ]:
        p = base / "dataset_adapters" / fname
        for i, r in enumerate(iter_parquet(p, columns=["conversations", "agent", "task", "model"])):
            if ctx.reached(src):
                break
            if step > 1 and (i % step):
                continue
            msgs = [norm_msg(m) for m in (r.get("conversations") or []) if isinstance(m, dict)]
            meta = {"agent": r.get("agent"), "task": r.get("task"), "model": r.get("model")}
            ctx.emit(src, "agent_terminal", msgs, lang="shell", meta=meta)
    src = "nemotron_terminal_skill"
    for p in sorted((base / "synthetic_tasks").rglob("data_filtered.parquet")):
        for r in iter_parquet(p, columns=["conversations", "agent", "task"]):
            if ctx.reached(src):
                break
            msgs = [norm_msg(m) for m in (r.get("conversations") or []) if isinstance(m, dict)]
            meta = {"agent": r.get("agent"), "task": r.get("task"),
                    "skill": str(p.parent.relative_to(base / "synthetic_tasks"))}
            ctx.emit(src, "agent_terminal", msgs, lang="shell", meta=meta)


def traj_to_msgs(traj):
    out = []
    if isinstance(traj, list):
        for item in traj:
            if not isinstance(item, dict):
                continue
            if item.get("role"):
                out.append(norm_msg(item))
            else:
                txt = item.get("content") or item.get("message") or item.get("action") or item.get("observation")
                role = "assistant" if item.get("action") else "tool"
                if txt:
                    out.append(msg(role, txt, MAX_TOOL_CHARS))
    elif isinstance(traj, str):
        out.append(msg("assistant", traj, MAX_ASSISTANT_CHARS))
    return out


def conv_swe_rebench(ctx):
    src = "swe_rebench"
    p = RAW / "swe-rebench-openhands" / "trajectories.parquet"
    cols = ["trajectory", "tools", "resolved", "repo", "instance_id", "exit_status"]
    for r in iter_parquet(p, columns=cols):
        if ctx.reached(src):
            break
        if not r.get("resolved"):
            continue
        msgs = traj_to_msgs(r.get("trajectory"))
        meta = {"repo": r.get("repo"), "instance": r.get("instance_id"), "status": r.get("exit_status")}
        ctx.emit(src, "swe_agent", msgs, lang="python", tools=r.get("tools"), meta=meta)


def conv_toolace(ctx):
    src = "toolace"
    data = json.loads((RAW / "toolace" / "data.json").read_text(encoding="utf-8", errors="replace"))
    for r in data:
        if ctx.reached(src):
            break
        msgs = [msg("system", r.get("system", ""), 8000)]
        for t in r.get("conversations") or []:
            role = "user" if (t.get("from") == "user") else "assistant"
            msgs.append(msg(role, t.get("value", "")))
        ctx.emit(src, "tool_calling", msgs, lang="text")


def conv_toolmind(ctx):
    src = "toolmind"
    files = [RAW / "toolmind" / "graph_syn_datasets" / "graphsyn.jsonl"]
    files += sorted((RAW / "toolmind" / "open_datasets").glob("*.jsonl"))
    for p in files:
        for r in iter_jsonl(p):
            if ctx.reached(src):
                break
            msgs = [norm_msg(m) for m in (r.get("conversations") or []) if isinstance(m, dict)]
            ctx.emit(src, "tool_calling", msgs, lang="text", tools=r.get("tools"), meta={"file": p.name})


def conv_agent_flan(ctx):
    src = "agent_flan"
    for p in sorted((RAW / "agent-flan" / "data").glob("*.jsonl")):
        for r in iter_jsonl(p):
            if ctx.reached(src):
                break
            convs = r.get("conversation") or r.get("conversations") or []
            msgs = [norm_msg(m) for m in convs if isinstance(m, dict)]
            ctx.emit(src, "agent_react", msgs, lang="text", meta={"file": p.name, "id": r.get("id")})


def conv_xlam(ctx):
    src = "xlam"
    p = RAW / "xlam" / "data" / "train-00000-of-00001.parquet"
    for r in iter_parquet(p, columns=["messages"]):
        if ctx.reached(src):
            break
        msgs = [norm_msg(m) for m in (r.get("messages") or []) if isinstance(m, dict)]
        ctx.emit(src, "tool_calling", msgs, lang="text")


GLAIVE_RE = re.compile(r"(?:^|\n)(USER|ASSISTANT|FUNCTION RESPONSE):\s?")


def parse_glaive_chat(chat):
    parts = GLAIVE_RE.split(chat or "")
    out = []
    for role, content in zip(parts[1::2], parts[2::2]):
        role = {"USER": "user", "ASSISTANT": "assistant", "FUNCTION RESPONSE": "tool"}.get(role, "assistant")
        content = content.replace("<|endoftext|>", "").strip()
        if content:
            out.append(msg(role, content))
    return out


def conv_glaive(ctx):
    src = "glaive"
    p = RAW / "glaive-fc" / "glaive-function-calling-v2.json"
    for r in iter_json_array(p):
        if ctx.reached(src):
            break
        msgs = [msg("system", r.get("system", ""), 8000)] + parse_glaive_chat(r.get("chat", ""))
        ctx.emit(src, "tool_calling", msgs, lang="text")


def conv_opencodeinstruct(ctx):
    src = "opencodeinstruct"
    base = RAW / "opencodeinstruct" / "data"
    for path in sorted(base.glob("*.parquet")):
        if ctx.reached(src):
            break
        for r in iter_parquet(path, columns=["id", "input", "output", "domain", "generation_algorithm"]):
            if ctx.reached(src):
                break
            code_input = r.get("input") or ""
            code_output = r.get("output") or ""
            if not code_input and not code_output:
                ctx.st(src)["empty"] += 1
                continue
            msgs = [
                msg("user", code_input, MAX_USER_CHARS),
                msg("assistant", code_output, MAX_ASSISTANT_CHARS),
            ]
            meta = {
                "domain": r.get("domain") or "",
                "generation_algorithm": r.get("generation_algorithm") or "",
            }
            ctx.emit(src, "code_gen", msgs, lang="text", meta=meta)


def conv_openrefined(ctx):
    src = "openrefined"
    base = RAW / "opencodeinstruct-refined" / "data"
    for path in sorted(base.glob("train-*.parquet")):
        if ctx.reached(src):
            break
        for r in iter_parquet(path, columns=["input", "output", "domain",
                                             "generation_algorithm", "average_test_score"]):
            if ctx.reached(src):
                break
            # عامل الجودة: نستبعد ما لم تكتمل اختباراته (نسخة مصفّاة أصلاً — R9)
            score = float(r.get("average_test_score") or 0)
            if score < 0.9:
                ctx.st(src)["empty"] += 1
                continue
            msgs = [msg("user", r.get("input") or "", MAX_USER_CHARS),
                    msg("assistant", r.get("output") or "", MAX_ASSISTANT_CHARS)]
            meta = {"domain": r.get("domain") or "", "avg_score": score,
                    "slow_gen": r.get("generation_algorithm") or ""}
            ctx.emit(src, "code_gen", msgs, lang="text", meta=meta)


def conv_oss_instruct(ctx):
    src = "oss_instruct"
    p = RAW / "oss-instruct-50k" / "data" / "train-00000-of-00001.parquet"
    for r in iter_parquet(p, columns=["instruction", "response", "concepts"]):
        if ctx.reached(src):
            break
        if not (r.get("instruction") or "").strip():
            ctx.st(src)["empty"] += 1
            continue
        msgs = [msg("user", r.get("instruction", ""), MAX_USER_CHARS),
                msg("assistant", r.get("response", ""), MAX_ASSISTANT_CHARS)]
        concepts = r.get("concepts") or []
        ctx.emit(src, "code_gen", msgs, lang="python",
                 meta={"concepts": concepts[:8] if isinstance(concepts, list) else concepts})


def conv_html_cot(ctx):
    src = "html_cot"
    p = RAW / "html-css-js-cot" / "data_0.parquet"
    for r in iter_parquet(p, columns=["question_text", "solution_full_response", "stack"]):
        if ctx.reached(src):
            break
        q = (r.get("question_text") or "").strip()
        a = (r.get("solution_full_response") or "").strip()
        if not q or not a:
            ctx.st(src)["empty"] += 1
            continue
        # R10 (الفخ السطحي): استبدال وسم <thinking> الزائف بصياغة طبيعية للتفكير
        if a.startswith("<thinking>"):
            body = a[len("<thinking>"):]
            if "</thinking>" in body:
                think, rest = body.split("</thinking>", 1)
                a = "Reasoning:\n" + think.strip() + "\n\n" + rest.strip()
            else:
                a = "Reasoning:\n" + body.strip()
        msgs = [msg("user", q, MAX_USER_CHARS),
                msg("assistant", a, MAX_ASSISTANT_CHARS)]
        ctx.emit(src, "web_ui_gen", msgs, lang="html", meta={"stack": r.get("stack")})


WEB_SYS = ("You are a web navigation agent that operates a text browser "
           "through page HTML and accessibility structure. Act step by step.")


def conv_mind2web(ctx):
    src = "mind2web"
    for p in sorted((RAW / "mind2web" / "data" / "train").glob("train_*.json")):
        if ctx.reached(src):
            break
        for ep in iter_json_array(p, max_buf=256 << 20):
            if ctx.reached(src):
                break
            acts = ep.get("actions") or []
            reprs = [a for a in (ep.get("action_reprs") or []) if isinstance(a, str)]
            website = ep.get("website") or ""
            domain = "%s > %s" % (ep.get("domain") or "", ep.get("subdomain") or "")
            task_txt = (ep.get("confirmed_task") or "").strip()
            anno = ep.get("annotation_id") or ""
            if not task_txt:
                ctx.st(src)["empty"] += 1
                continue
            # (1) تسجيل الخطة الكاملة بلا HTML (ركيصة)
            plan_user = "Website: %s (%s)\nTask: %s\n\nPlan the click sequence to complete the task." % (
                website, domain, task_txt)
            plan_assist = "\n".join(reprs) if reprs else ""
            if plan_assist:
                ctx.emit(src, "agent_browser_plan",
                         [msg("system", WEB_SYS, 1500), msg("user", plan_user), msg("assistant", plan_assist)],
                         lang="html", meta={"website": website, "domain": domain, "annotation": anno})
            # (2) سجل لكل خطوة: حالة الصفحة → الفعل
            n = len(acts)
            for i, a in enumerate(acts):
                if ctx.reached(src):
                    break
                if not isinstance(a, dict):
                    continue
                html = (a.get("cleaned_html") or "")[:12_000]
                if not html.strip():
                    ctx.st(src)["empty"] += 1
                    continue
                act = reprs[i] if i < len(reprs) else ""
                step_user = ("Website: %s (%s)\nTask: %s\nStep %d of %d.\n\nCurrent page (truncated HTML):\n"
                             "```html\n%s\n```" % (website, domain, task_txt, i + 1, n, html))
                ctx.emit(src, "agent_browser_step",
                         [msg("system", WEB_SYS, 1500), msg("user", step_user, MAX_USER_CHARS),
                          msg("assistant", act, 500)],
                         lang="html", meta={"website": website, "step": i + 1, "total": n,
                                            "annotation": anno})


def _pw_msg(m):
    """messages قد تكون نصوص JSON-مشفّرة مصغّرة (train_v3) أو قواميس."""
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except Exception:
            return None
    if not isinstance(m, dict):
        return None
    return norm_msg(m)


def conv_playwright_mcp(ctx):
    src = "playwright_mcp"
    files = [RAW / "playwright-mcp-toolcalling" / "data" / "train_v3.jsonl",
             RAW / "playwright-mcp-toolcalling" / "data" / "train_v4.jsonl"]
    for p in files:
        for r in iter_jsonl(p):
            if ctx.reached(src):
                break
            msgs = [x for x in (_pw_msg(m) for m in (r.get("messages") or [])) if x]
            meta = {"model": r.get("model"), "num_tools": r.get("num_tools")}
            ctx.emit(src, "tool_calling_browse", msgs, lang="text", meta=meta)
    for path in [RAW / "playwright-mcp-toolcalling" / "data" / "train.parquet",
                 RAW / "playwright-mcp-toolcalling" / "data" / "train_v3.parquet"]:
        for r in iter_parquet(path, columns=["messages", "model", "num_tools"]):
            if ctx.reached(src):
                break
            raw = r.get("messages") or []
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = []
            msgs = [x for x in (_pw_msg(m) for m in raw) if x]
            meta = {"model": r.get("model"), "num_tools": r.get("num_tools")}
            ctx.emit(src, "tool_calling_browse", msgs, lang="text", meta=meta)


def conv_primevul(ctx):
    src = "primevul"
    for p in [RAW / "primevul" / "primevul_train.jsonl",
              RAW / "primevul" / "primevul_valid.jsonl"]:
        if ctx.reached(src):
            break
        for r in iter_jsonl(p):
            if ctx.reached(src):
                break
            func = r.get("func") or ""
            target = int(r.get("target") or 0)
            cwe = r.get("cwe") or []
            cwe = [cwe] if isinstance(cwe, str) else list(cwe or [])
            cve = r.get("cve") or ""
            reason = ("Vulnerable code; weakness consistent with %s (%s)." % (", ".join(cwe), cve)
                      if target else "No vulnerability detected in this function.")
            meta = {"cwe": ",".join(cwe), "cve": cve, "project": r.get("project"),
                    "desc": (r.get("cve_desc") or "")[:400]}
            ctx.emit(src, "vuln_detect", _detect_pair(func, "c", target, cwe, reason),
                     lang="c", meta=meta)


# ---------------------------------------------------------------- run / stats

def run_slice(name, out_dir, limit):
    ctx = Ctx(out_dir, limit)
    try:
        if name in ("security", "security-new", "all"):
            print("== security slice:", flush=True)
            if name in ("security", "all"):
                conv_fenrir(ctx)
                conv_bigvul(ctx)
                conv_diversevul(ctx)
            if name in ("security-new", "all"):
                conv_primevul(ctx)
        if name in ("tools", "all"):
            print("== tools slice:", flush=True)
            conv_nemotron_agentic(ctx)
            conv_nemotron_terminal(ctx)
            conv_swe_rebench(ctx)
            conv_toolace(ctx)
            conv_toolmind(ctx)
            conv_agent_flan(ctx)
            conv_xlam(ctx)
            conv_glaive(ctx)
        if name in ("tools-new", "all"):
            print("== tools-new slice:", flush=True)
            conv_mind2web(ctx)
            conv_playwright_mcp(ctx)
        if name in ("modern", "all"):
            print("== modern slice:", flush=True)
            conv_opencodeinstruct(ctx)
        if name in ("modern-new", "all"):
            print("== modern-new slice:", flush=True)
            conv_openrefined(ctx)
            conv_oss_instruct(ctx)
            conv_html_cot(ctx)
    finally:
        ctx.close()
    ctx.summary(name)


def stats(out_dir):
    total_n = 0
    total_b = 0
    by_src = {}
    for p in sorted(out_dir.glob("*shard-*.jsonl")):
        src = p.name.split("-shard-")[0]
        n = 0
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                n += 1
        b = p.stat().st_size
        by_src[src] = by_src.get(src, (0, 0))
        by_src[src] = (by_src[src][0] + n, by_src[src][1] + b)
        total_n += n
        total_b += b
    for src, (n, b) in sorted(by_src.items()):
        print(f"{src:34s} {n:>12,d} records  {b/1e9:>7.2f} GB")
    print(f"{'TOTAL':34s} {total_n:>12,d} records  {total_b/1e9:>7.2f} GB")


def main():
    ap = argparse.ArgumentParser(description="T7 unified data pipeline")
    ap.add_argument("slice", choices=["security", "tools", "modern", "all", "stats",
                                      "security-new", "tools-new", "modern-new"])
    ap.add_argument("--limit", type=int, default=0, help="حد سجلات لكل مصدر (0 = بلا حد)")
    # افتراضي: نفس مجلد الشريحة (data/processed/security إلخ) — لا يكتب في جذر processed
    ap.add_argument("--out", default=None, help="مجلد الإخراج (افتراضياً data/processed/<slice-primary>)")
    a = ap.parse_args()
    default_out = PROC / {"security-new": "security", "tools-new": "tools",
                          "modern-new": "modern"}.get(a.slice, a.slice if a.slice != "stats" else ".")
    out = Path(a.out) if a.out else default_out
    if a.slice == "stats":
        stats(out)
        return
    run_slice(a.slice, out, a.limit)


if __name__ == "__main__":
    main()
