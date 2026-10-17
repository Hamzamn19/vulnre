#!/usr/bin/env python3
"""تنزيل ملفات تسميات PrimeVul الناقصة (file_contents فقط نزّل سابقاً)."""
from huggingface_hub import hf_hub_download

for f in ["primevul_train.jsonl", "primevul_valid.jsonl", "primevul_test.jsonl"]:
    try:
        p = hf_hub_download("colin/PrimeVul", f, repo_type="dataset",
                            local_dir="data/raw/primevul")
        print("OK", p, flush=True)
    except Exception as e:
        print("ERR", f, type(e).__name__, str(e)[:200], flush=True)
