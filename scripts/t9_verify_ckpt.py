#!/usr/bin/env python3
"""تحقق T9: تحميل checkpoint محفوظ + توليد — يجب أن ينجح قبل اعتبار البايلوت مُقيّماً."""
import sys
import gc
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "models/deepseek-coder-6.7b-base"
PROBE = ("Analyze this C snippet for vulnerabilities and answer with JSON:\n\n"
         "```c\nvoid f(char *s){ char buf[32]; strcpy(buf, s); }\n```")


def main(ckpt):
    auto = torch.cuda.is_available()
    print(f"[verify] {ckpt}; cuda={auto}", flush=True)
    kw = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
          "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True},
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=kw[0], torch_dtype=torch.bfloat16 if auto else None,
        device_map="auto", local_files_only=True)
    model = PeftModel.from_pretrained(model, ckpt, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    prompt = f"<|User|>\n{PROBE}\n<|Assistant|>\n"
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print("[verify] generation OK →", text[:200].replace("\n", " "), flush=True)
    del model
    gc.collect()
    if auto:
        torch.cuda.empty_cache()
    print("[verify] PASSED — checkpoint يعمل للتحميل والتوليد", flush=True)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    if not ckpt:
        p = Path("data/processed/t9_pilot")
        cands = sorted(p.glob("checkpoint-*")) + sorted(p.glob("best")) + \
            sorted(p.glob("adapter_final"))
        ckpt = str(cands[-1]) if cands else None
        if not ckpt:
            print("no checkpoint found", flush=True)
            sys.exit(2)
    main(ckpt)
