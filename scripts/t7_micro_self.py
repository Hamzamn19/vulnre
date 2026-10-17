#!/usr/bin/env python3
"""T7 — المصغّر الذاتي (self-micro): 500–2000 زوج سؤال/جواب من معرفة المشروع + نطاقه.

النطاقات (R9 + R10):
  project_pipeline — وثائق مشروع vulnre (السجل القانوني، السكربتات، القرارات R1..R10)
  qlora            — تدريب QLoRA على RTX 3060 (NF4، LoRA ranks، checkpointing)
  project_littlebit— ضغط LittleBit/EVREN (0.55bpw)
  re_knowledge     — دلالات تعليمات x86/RE
  re_toolchain     — أعلام gcc وأدوات التحليل الثنائي
  vuln_knowledge   — CWE/CVE وأنماط الثغرات
  firmware_knowledge — أساسيات Cortex-M4/ESP32/STM32F407
  tool_format      — استدعاء الأدوات بأسلوب الوكلاء

كل حقيقة تُطلق 3 صياغات. الإخراج: data/processed/micro/micro_self-shard-*.jsonl + ملخص.
"""
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "micro"
SEED = 42
TARGET = 2000
SHARD_CHARS = 64_000_000

random.seed(SEED)


def variants(q):
    """ثلاث صياغات لكل حقيقة."""
    return [q,
            "Give a concise expert answer. " + q,
            "In one short paragraph, as an expert: " + q]


RECS = []


def add(task, question, answer, lang="text", system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    for q in variants(question):
        mm = list(msgs)
        mm.append({"role": "user", "content": q})
        mm.append({"role": "assistant", "content": answer})
        RECS.append((task, lang, mm))


# ================================================================== 1) المشروع
PIPE_SYS = ("You are the internal knowledge base of the vulnre project: a local security LLM "
            "for low-level code, reverse engineering, IoT/firmware, and agentic tool use.")

add("project_pipeline", "What is the vulnre project's final goal?",
    "Train a specialized local model strong in low-level code, reverse engineering (x86/x64 plus ARM), "
    "vulnerability detection (IoT/firmware), and agentic tool control (terminal, browser, tool creation) — "
    "via local QLoRA on an RTX 3060, then compress with LittleBit-2 on EVREN (first target ~0.55 bpw) "
    "and publish on Hugging Face.", system=PIPE_SYS)
add("project_pipeline", "What is the Canonical Record schema used by the vulnre data pipeline?",
    '{"id": "<source>-0000123", "source": "<raw repo slug>", '
    '"task": "vuln_detect|vuln_fix|tool_calling|agent_terminal|c_to_asm|asm_to_c|cyber_qa|code_gen|agent_browser_step|...", '
    '"lang": "c|python|shell|html|text", "messages": [{"role", "content", "tool_calls?"}], '
    'optional "tools": [OpenAI tool schemas], optional "meta": {small fields}}. '
    "Each slice writes layered JSONL shards (~192MB each) under data/processed/<slice>/.", system=PIPE_SYS)
add("project_pipeline", "Which base models were selected for vulnre and why?",
    "Both deepseek-ai/deepseek-coder-6.7b-base and LLM4Binary/llm4decompile-6.7b-v1.5 (user choice R2). "
    "Both are Llama-family with ~32K vocab — the only line compatible with the LittleBit small-vocab "
    "compression target; llm4decompile is itself built on deepseek-coder-6.7b-base.", system=PIPE_SYS)
add("project_pipeline", "Which datasets make up the vulnre security slice?",
    "fenrir (cyber QA) ~99,870 + bigvul (vuln_detect and vuln_fix from func_before/func_after pairs) "
    "~330,865 + diversevul (vuln_detect) ~330,110 = 760,845 records (1.7GB output). "
    "PrimeVul is added by security-new as a labeled extra with target/cwe/cve_desc fields.", system=PIPE_SYS)
add("project_pipeline", "How does scripts/build_dataset.py keep memory bounded on a 16GB machine?",
    "Fully streaming: line-by-line JSONL, parquet via iter_batches (~64 rows), JSON arrays via incremental "
    "raw_decode with a bounded buffer; per-record dedupe with blake2b of truncated message contents; caps "
    "per record (250K total chars, 16K user / 24K assistant) — oversized records are counted as 'long' and "
    "skipped instead of overflowing memory.", system=PIPE_SYS)
add("project_pipeline", "What is the T8 asm-pair data source and format?",
    "REx86 already ships assembly plus decompiled original code (362,145 records kept; 770,477 duplicates "
    "removed). AnghaBench C sources (1M single-file programs) are compiled locally with gcc -O0/-O2 -S to "
    "produce (C, asm) pairs; records use tasks c_to_asm and asm_to_c with a system prompt, user code or asm, "
    "and the counterpart as the assistant message.", system=PIPE_SYS)
add("project_pipeline", "Why is the T8/AnghaBench clone run under systemd --user?",
    "The OpenCode server restarted during long git clones and killed them (the first AnghaBench clone died "
    "after receiving objects, leaving an empty directory). Clone and checkout now run in a vulnre-* systemd "
    "unit with a log file, surviving server restarts; the plan reuses this pattern for every long job.", system=PIPE_SYS)
add("project_pipeline", "What ratios does the final mix (t9_final_mix.py) lock?",
    "security_re 40% (security slice + t8_asm_pairs), tools 35% (sampled from ~1.75M), modern/general code "
    "20% (sampled from ~4.97M), and the last 5% for micro_self knowledge knowledge (1,000,000 records "
    "target total). Datasheet QA is deferred to later work, so the 5% initially carries micro_self.", system=PIPE_SYS)
add("project_pipeline", "How are long records handled before training (R8)?",
    "The base supports 16K tokens (4K native times linear rope scaling 4) and the KV cache is heavy "
    "(~0.5MB/token FP16, no GQA), so 16K costs about 8GB VRAM. Strategy: keep records at or below 8K tokens, "
    "use compressed KV (int8/int4) at inference, and reach very large firmware files via RAG retrieval; a "
    "32K extension is a separate optional task (T9b).", system=PIPE_SYS)
add("project_pipeline", "Which finding from R10's competitor scan validates this project, and which gaps remain?",
    "Cisco Antares (July 2026; 350M/1B agentic vulnerability locating) proves small local security models "
    "are worth building. Remaining protected gaps: low-level RE (asm/binaries/decompilation), firmware/IoT "
    "specifics, broadened tools (browser/MCP/terminal), extreme near-1GB compression; competitors like "
    "Foundation-Sec 8B need 16GB+ FP16 and a huge vocab — unsuited to our targets.", system=PIPE_SYS)
add("project_pipeline", "Which evaluation benchmarks are adopted for T10?",
    "FAITH suite (CTIBench/CyberMetric/SecEval/SecBench/MMLU-Security), IoTVulBench (first human-verified "
    "firmware benchmark, Aug 2026), CWE-Trace 834-sample Linux set (74 CWEs) as a clean temporal eval, "
    "REFORGE/HOBRE for RE quality, plus a direct comparison against Antares-350M/1B on localization.", system=PIPE_SYS)
add("project_pipeline", "How do the three 'recency' mechanisms of R9 keep answers current?",
    "(1) The modern data slice includes recent JS/TS/HTML/CSS and general code in the mix; (2) live tools "
    "and RAG documentation lookup at inference — the base model is from 2023 so it must call tools for 2025+ "
    "facts; (3) cheap periodic LoRA refresh (~100MB adapters) reusing the frozen base.", system=PIPE_SYS)
add("project_pipeline", "Why is DeepSeek-Coder-V2-Lite rejected despite being newer?",
    "It is a MoE, not Llama-family, and incompatible with the LittleBit compression target that requires "
    "Llama-style small-vocab architecture; ruling R2 keeps the 6.7B base plus llm4decompile pair.", system=PIPE_SYS)
add("project_pipeline", "What went wrong with the first inspect_schemas.py and how was it fixed?",
    "It loaded whole artifacts (json.load on a 256MB file, pandas read_parquet on 1-2GB files), peaked at "
    "~10.5GB RSS, and the kernel OOM-killed the server 3 times. Fix: a streaming schema probe (pyarrow "
    "iter_batches, skipping row groups above 300MB, 4MB line caps, text head for big files), flock-protected "
    "launches, and all long jobs moved into systemd --user units.", system=PIPE_SYS)
add("project_pipeline", "Where is the LittleBit community recipe and how is training invoked later (T12)?",
    "Kernel code: third_party/LittleBit (Samsung official). Recipe: third_party/littlebitllm-recipe/recipe/"
    "train.sh with MODEL_ID, EFF_BIT, EPOCHS, NUM_GPUS. The Qwen3-8B at 0.55 recipe needs ~107GB aggregate "
    "VRAM, so it runs on EVREN (H200), not the local 3060; community weights littlebit-qwen3-8b-0.55bpw are "
    "kept locally only for inference experiments.", system=PIPE_SYS)

# ================================================================== 2) QLoRA
QL_SYS = ("You advise on QLoRA fine-tuning of 6-7B base models on a single RTX 3060 12GB "
          "(torch cu130, bitsandbytes 0.50.2, peft and trl).")

add("qlora", "What does 4-bit NF4 quantization change in QLoRA?",
    "Weights are quantized to 4-bit NormalFloat groups (normally group size 64) with double-quantized FP32 "
    "scale constants, while LoRA adapters stay FP16/BF16 and train normally. A 6.7B model drops from ~13GB "
    "FP16 to ~3.5-4GB of weights, leaving activation headroom on a 12GB card.", system=QL_SYS)
add("qlora", "Why do paged optimizers matter on 12GB?",
    "paged_adamw_32bit pages optimizer states between GPU memory and host RAM under churn spikes, turning "
    "transient OOM into slower survivable steps. The trade-off is paging latency, and on a 16GB-RAM host you "
    "must watch system memory so paging does not exhaust the host instead.", system=QL_SYS)
add("qlora", "What LoRA rank/alpha is sane for a specialized 6.7B SFT on one 3060?",
    "Start at r=16, alpha=32, dropout 0.05, targeting q,k,v,o and the MLP projections (gate/up/down) for "
    "knowledge addition. r=8 is enough for style and format learning; r=64+ on 12GB eats optimizer memory "
    "for marginal return. Trainable parameters end up around 0.5-1.5% of the total.", system=QL_SYS)
add("qlora", "Which memory knobs rescue a training step that OOMs at sequence length 4096?",
    "In order: enable gradient checkpointing (recompute instead of store, ~30% slower); drop per-device "
    "batch to 1 and raise gradient accumulation; shorten max sequence length or pack at 2K; switch to an "
    "8-bit or paged optimizer. On a 3060 with a 6.7B NF4 model, batch 1 with accumulation 8 at 4K plus "
    "checkpointing is the reliable floor.", system=QL_SYS)
add("qlora", "How do you verify the training pipeline before a long run?",
    "A T5-style smoke test: load the model 4-bit NF4, attach LoRA, run 2 optimizer steps on a tiny subset, "
    "assert the loss decreases, and record torch.cuda.max_memory_allocated. Our run peaked at 4.12GB on "
    "deepseek-coder-6.7b-base, confirming headroom for full sequences.", system=QL_SYS)
add("qlora", "What chat-template caution applies to a BASE model like deepseek-coder-6.7b-base?",
    "A base model has no chat template, so never rely on tokenizer.apply_chat_template without supplying "
    "one. Use the model's documented wrapper (user/assistant separator tokens as in the repo's examples) or "
    "define a strict custom template in the SFT dataset; test tokenization on a few samples once, and never "
    "mix templates inside one run.", system=QL_SYS)
add("qlora", "When is bf16 preferred over fp16 on these GPUs?",
    "The RTX 3060 is Ampere, so bf16 works and avoids fp16 loss-scaling pain — use bf16 autocast with FP16/ "
    "BF16 LoRA weights and FP32 optimizer masters (TRL default). If a bitsandbytes op complains, fall back "
    "to fp16 with a GradScaler.", system=QL_SYS)
add("qlora", "How do you estimate trainable tokens per hour for a 6.7B NF4 on a 3060?",
    "Roughly 1.2-2.5K tokens per second forward+backward at 4K sequence length with checkpointing enabled, "
    "so about 5-9M tokens per hour. Budget runs by token count, not record count, and recalibrate after the "
    "first checkpoint.", system=QL_SYS)
add("qlora", "What is gradient checkpointing and when do you enable it?",
    "It drops stored activations between layers and recomputes them during backward, trading about 20-30% "
    "throughput for large peak-memory savings — effectively mandatory on 12GB for a 6.7B model with long "
    "sequences. In transformers 5.x use use_reentrant=False for the modern implementation.", system=QL_SYS)
add("qlora", "Why warmup plus cosine schedule instead of constant LR?",
    "For LoRA SFT on a 6.7B, learning rates of 1e-4 to 2e-4 with 30-100 warmup steps prevent early loss "
    "spikes from noisy batches; cosine decay to near zero recovers the last bits of quality. For very short "
    "runs (under 1000 steps) constant-with-warmup is better because cosine never gets time to matter.", system=QL_SYS)
add("qlora", "What does peft's modules_to_save do beyond LoRA targets?",
    "It trains complete copies of small modules (lm_head, classifier heads, norm layers) alongside the "
    "frozen base — needed when a head must adapt (vocabulary or classifier) or a head can't be expressed "
    "as a rank delta. Unlike LoRA, these are stored as full tensors inside the adapter.", system=QL_SYS)
add("qlora", "What is the correct merge_and_unload workflow after QLoRA SFT, and its danger?",
    "Load the base in FP16, load the adapter, call model.merge_and_unload(), then save. Danger: merging "
    "into the 4-bit base bakes dequantization error into a corrupted artifact — dequantize the base to FP16 "
    "first, then merge, and verify with probe prompts and token counts before publishing.", system=QL_SYS)
add("qlora", "How should eval samples be held out without leaking training content?",
    "Hold out by source group — whole project, repo, or dataset — not by random per-record split, because "
    "random splits leak near-duplicates across train and eval. This matches the temporal-eval discipline of "
    "T10: train on data published through 2024 and evaluate on 2025+ material.", system=QL_SYS)
add("qlora", "What catastrophic forgetting looks like and how the mix limits it",
    "The model excels at the trained task while degrading on general code questions it previously handled. "
    "Limits: keep LoRA rank modest (16 rather than 128), and include 15-20% general modern code in the "
    "mixture so base competence is reinforced; the plan's T7 ratio does exactly this.", system=QL_SYS)
add("qlora", "What does validation loss vs training loss tell you for LoRA SFT?",
    "With strongly deduplicated data LoRA often reaches its best eval loss inside the first epoch; watching "
    "eval loss every N steps reveals overfitting (eval rises while train falls). Stop and checkpoint on the "
    "eval plateau instead of finishing a fixed epoch count blindly.", system=QL_SYS)

# ================================================================== 3) LittleBit
LB_SYS = ("You summarize the LittleBit-2 extreme compression recipe (Samsung paper + community recipe) "
          "for post-training bit-level compression on EVREN.")

add("project_littlebit", "What is LittleBit's target bit-width idea and the first milestone here?",
    "Extreme post-training weight compression below 1 bit per parameter using error-feedback bit-level "
    "distillation. First milestone: Qwen3-8B around 0.55 bpw (a ~1GB-class model). The community train.sh "
    "takes MODEL_ID, EFF_BIT, EPOCHS, NUM_GPUS.", system=LB_SYS)
add("project_littlebit", "Why can't the local 3060 run the LittleBit 8B recipe directly?",
    "The recipe needs roughly 107GB aggregate VRAM (quantized model plus full-size reference plus "
    "distillation buffers). EVREN's H200 covers that; the 3060 only serves inference experiments with "
    "released community weights like littlebit-qwen3-8b-0.55bpw.", system=LB_SYS)
add("project_littlebit", "How is 0.55 bpw compared with 0.8-1.0 bpw in this project?",
    "Ruling R7 made the size cap flexible: produce both compression points on EVREN, evaluate both, and "
    "prefer quality per GB over raw size — a 13B model compressed to 2-3GB is an open option fitting the "
    "same storage budget as a heavily crushed 8B.", system=LB_SYS)
add("project_littlebit", "What is BitNet/ternary and how does it relate to LittleBit here?",
    "Ternary 1.58-bit training became mainstream in 2026. LittleBit remains the chosen local compression "
    "route; a possible later experiment is progressive ternary conversion of our final merged model, with "
    "bitnet.cpp providing fast ternary inference.", system=LB_SYS)
add("project_littlebit", "Why does vocab size matter for compression feasibility?",
    "A small vocabulary (~32K) shrinks the embedding table's share of post-compression weights and keeps "
    "the total dominated by transformer blocks. That is why R2 restricted base models to Llama-vocab "
    "instead of 100-128K vocab models like Antares or Foundation-Sec.", system=LB_SYS)
add("project_littlebit", "How does error-feedback compression actually work?",
    "Bit-level distillation: weights are compressed level by level while a student is trained to "
    "compensate the residual error left by previously quantized levels. Each pass lowers the remaining "
    "error budget, so the final network keeps accuracy at half a bit per parameter. It requires the full "
    "teacher in DRAM plus staged training, hence the H200-class demand.", system=LB_SYS)
add("project_littlebit", "What does EFF_BIT 0.55 mean concretely?",
    "Effective bits per weight: 0.55 bpw averages ~0.55 bits per parameter, so an 8B model's weights take "
    "about 0.66GB before overhead. Lower values shrink further but demand better error-feedback training; "
    "the plan asks for both 0.55 and 0.8-1.0 and lets evaluation pick.", system=LB_SYS)
add("project_littlebit", "Where do the community 0.55bpw Qwen3 weights come from, and what are they for?",
    "They were compressed by community recipe runs on rented GPUs and uploaded to Hugging Face. Locally they "
    "enable inference experiments (API and structure checks, small evals) but not more compression training "
    "on a 3060. Our own model gets compressed in the T12 EVREN phase.", system=LB_SYS)
add("project_littlebit", "What memory and time do 0.55bpw runs need?",
    "The Qwen3-8B at 0.55 recipe takes ~107GB VRAM aggregated over stages and multiple H200-hours per epoch "
    "(community reports days). On EVREN the free API window runs to Nov 1, 2026, and training bills about "
    "10 CR per hour for an H200; budget both compression points as the biggest T12 expense.", system=LB_SYS)
add("project_littlebit", "What does the T12 EVREN plan cover in order?",
    "Phase 1: compress Qwen3-8B at 0.55 bpw through the community recipe to validate tooling against known "
    "numbers. Phase 2: compress our own merged QLoRA vulnre model (deepseek-6.7b bases) at 0.55 and ~0.9 "
    "bpw, evaluate, then produce the hydra-style set of stage-specific compressed copies (T13).", system=LB_SYS)

# ================================================================== 4) x86
X86_SYS = "You are an x86/x64 reverse-engineering tutor. Answer precisely, Intel-syntax where useful."

X86_FACTS = [
    ("What does the x86 instruction `mov eax, [ebx+8]` do?",
     "Loads a 32-bit value from memory at effective address EBX+8 into EAX. In Intel syntax with size "
     "keyword: mov eax, DWORD PTR [ebx+8]. The bracketed form means dereference with displacement 8."),
    ("What is `lea` and why do compilers use it for arithmetic like a*8+b?",
     "lea eax, [ebx*8+ecx] computes the address expression into a register WITHOUT touching flags or "
     "memory. Compilers exploit the [base+index*scale+disp] address math as a fast 3-operand arithmetic "
     "unit: multiply-by-constant plus add in one instruction with no flag clobber."),
    ("What does `test eax, eax` do before a conditional jump?",
     "ANDs EAX with itself without storing the result, setting ZF=1 iff EAX==0 and SF from the sign bit, "
     "clearing CF/OF. It is the standard zero-check idiom: jz jumps when zero, jnz when nonzero."),
    ("How is `cmp` different from `sub`?",
     "Both compute the subtraction, but cmp discards the result and only updates flags (ZF, SF, OF, CF); "
     "the following conditional jump reads those flags, so cmp-then-jcc implements a comparison."),
    ("What do `push` and `pop` do to ESP?",
     "push decrements the stack pointer by 4 (in 32-bit mode) and stores the operand at the new ESP; pop "
     "loads from [ESP] into the destination then increments ESP by 4. The stack grows downward."),
    ("How do `call` and `ret` work?",
     "call pushes the address of the next instruction and jumps to the callee; ret pops that return "
     "address into EIP. ret imm16 additionally nimm16 bytes off the stack — the callee-cleans-args forms "
     "such as stdcall use it."),
    ("What does the `leave` instruction do?",
     "leave is shorthand for mov esp, ebp followed by pop ebp — the standard function epilogue matching "
     "the push ebp / mov ebp, esp prologue. It restores the stack pointer from the frame pointer."),
    ("What is the difference between `sar` and `shr`?",
     "sar (arithmetic shift right) copies the sign bit into the vacated bits, used for signed division by "
     "a power of two — compilers sometimes add a rounding adjustment (add a bias before shifting for "
     "negative values). shr fills with zeros and is the logical, unsigned version."),
    ("How does the compiler express modulo by a constant in x86 (without div)?",
     "Granlund-Montgomery magic multiplication: multiply by a magic constant (for %10: 0xCCCCCCCD as a "
     "32-bit value), take the high half of the product, then shift right (by 5 for mod-10 style sequences). "
     "Seeing mov edx, 0xCCCCCCCD followed by mul then a shr reveals a %10 or /10 in source."),
    ("What compiler patterns reveal a switch statement in disassembly?",
     "Dense cases compile to a jump table: a bounds check, then an indirect jump like jmp DWORD PTR "
     "[ecx*4+table]; sparse cases become a check chain (cmp eax, N; je label). Either pattern in a "
     "disassembly is the running fingerprint for Switch statements."),
    ("What do `movsx` and `movzx` do?",
     "They load with extension: movsx fills high bits with the sign bit (signed char/short), movzx fills "
     "with zeros (unsigned). The choice in compiled code tells you the C-level signedness of casts."),
    ("How is a stack canary check recognized in compiled code?",
     "The prologue loads a cookie from fs:[0x28] into a local; the epilogue compares the local against it "
     "and jumps to __stack_chk_fail when mismatched. In assembly: mov eax, DWORD PTR fs:0x28 near the "
     "top, a compare near the bottom, and a call to the fail routine."),
    ("What does the LOCK prefix do?",
     "It makes the following instruction an atomic read-modify-write (lock xchg, lock cmpxchg, lock inc "
     "DWORD PTR [...]) via cache-line locking. In reverse engineering, lock-prefixed counters reveal "
     "refcounts, atomics, and spinlocks."),
    ("What are multi-byte NOPs and why do they appear in binaries?",
     "Padding instructions (nop, xchg eax,eax as the 2-byte form, and longer nop variants) are used to "
     "align code to 16-byte boundaries for loop performance or jump-target alignment. They carry no logic "
     "and should be skipped in flow analysis."),
    ("What does position-independent x86-64 code look like in disassembly?",
     "Data is referenced RIP-relative (lea rax, [rip+0x1234]) instead of by absolute addresses, and calls "
     "to shared library functions go through PLT stubs that jump through the GOT. That indirection is what "
     "makes ASLR possible for code and data."),
    ("What does `mov eax, [ebp+8]` mean in a standard 32-bit stdcall callee?",
     "With the classic prologue push ebp; mov ebp, esp, the saved EBP sits at [ebp+0], the return address "
     "at [ebp+4], and the first stack-passed argument at [ebp+8] (second at [ebp+0xC]). This is the "
     "32-bit cdecl/stack-args convention."),
    ("What does the REX prefix expand in x86-64?",
     "The REX bits add R8-R15 (via B and R bits), allow new byte registers (SIL, BPL, DIL; R8B-R15B), and "
     "select 64-bit operand size (W bit). Without REX.W=1 operations default to 32-bit, which still "
     "zero-extends the upper half in x86-64."),
    ("What do `cdq` and `cqo` before `idiv` establish?",
     "They sign-extend EAX into EDX:EAX (cdq) or RAX into RDX:RAX (cqo) to form the 64-bit (or 128-bit) "
     "dividend that idiv consumes. Their presence flags signed division; unsigned division uses div with "
     "EDX zeroed via xor edx, edx."),
    ("What does `rdtsc` measure and how do obfuscators abuse it?",
     "It reads the 64-bit processor timestamp counter into EDX:EAX (cycles since reset). Anti-debugging "
     "tricks pair two rdtsc calls around sensitive code and treat unusually large deltas (breakpoint or "
     "single-step slowdown) as evidence of a debugger."),
    ("How do Linux syscalls differ between int 0x80 and syscall?",
     "On 32-bit x86, int 0x80 enters the kernel with the syscall number in EAX and arguments in EBX, ECX, "
     "EDX and so on. On x86-64, syscall uses RAX for the number, RDI, RSI, RDX, R10, R8, R9 for arguments, "
     "and returns in RAX — faster and without the legacy segment machinery."),
    ("How do you recognize XOR string obfuscation loops in x86?",
     "A byte loop like movzx ecx, BYTE PTR [esi]; xor ecx, key; mov [esi], cl; inc esi; cmp; jne back — "
     "the immediate in the xor leaks the key. Decoders (FLOSS-class tools) probe per-byte against this "
     "pattern to recover strings statically."),
    ("What is the System V x86-64 shadow space (red zone) and what uses it?",
     "The ABI reserves 32 bytes below the return address on every call — actually Linux's version sits "
     "128 bytes BELOW RSP for leaf functions (the red zone). Windows instead allocates 32 bytes above RSP "
     "for callee use (shadow space). Distinguishing the conventions matters when reading prologues: sub "
     "rsp, 0x28 on Windows-style builds vs sub rsp, N on Linux."),
    ("What does `rep stos` set up?",
     "A memset-style fill: EAX = value, EDI = destination, ECX = count, with the direction flag clear "
     "(cld). Compilers emit rep stosd/stosb for zero-initializing buffers, so the pattern of eax=0 with "
     "rep marks array or structure clears."),
    ("What does `sete al` followed by `movzx eax, al` represent?",
     "A boolean-returning comparison: setcc materializes the flag predicate into AL and the movzx widens "
     "it to int, avoiding partial-register stalls. It is the compiled form of an == or != comparison "
     "returning 0/1."),
    ("What does `cmpxchg` implement on x86-64?",
     "Atomic compare-and-exchange: if the operand equals the value in the accumulator, the new value is "
     "stored and ZF set; otherwise the operand is loaded into the accumulator with ZF cleared. Compiler "
     "output typically seeds the accumulator first (mov rax, [mem]) then executes lock cmpxchg — the "
     "signature of spinlocks and refcounts."),
    ("How does a format-string vulnerability appear in a disassembly?",
     "A printf-family call receives its format argument from a runtime buffer rather than a .rodata "
     "constant — the register used for the first argument (rdi on x86-64) comes from a prior load instead "
     "of a lea rip-relative to a literal. That missing constant reference signals uncontrolled format."),
    ("How do 1-op, 2-op, and 3-op multiply differ in x86?",
     "mul r/m is unsigned with the full 64-bit product split into EDX:EAX; imul with one operand uses "
     "EDX:EAX for signed; imul 2-op and 3-op forms truncate the result to the destination size and only "
     "set CF/OF when the true product doesn't fit. Reading which form appears tells you the C expression's "
     "signedness and where the high half was needed."),
    ("Why are indirect call and jmp chokepoints for RE analysis?",
     "Targets like call eax, jmp DWORD PTR [ebx*4+table], or call QWORD PTR [rax+8] route control through "
     "a register or memory: function-pointer tables, vtables, switch dispatches, and obfuscated dispatchers "
     "all hide the destination there. Building call graphs requires resolving these manually or by dataflow."),
    ("What do rol and ror do and where do they appear?",
     "Circular shifts: bits that exit one side wrap around to the other (through CF when a count form is "
     "used). They appear in hash mixing rounds, some crypto primitives, and integrity checks — distinct "
     "from shl/shr which just drop the shifted-out bits."),
    ("What does `and ax, 0xff` (or similar masking) signal?",
     "An unsigned narrowing: the C expression truncated a value to 8 bits (unsigned char cast or x & 0ff). "
     "Compilers emit the mask after arithmetic to implement well-defined narrowing before use as an index "
     "or store."),
]
for q, r in X86_FACTS:
    add("re_knowledge", q, r, system=X86_SYS)

X86_EXTRA = [
    ("What does `endbr64` mean in modern x86-64 binaries?",
     "An Intel CET indirect-branch landing pad: it marks legal targets of indirect calls/jumps when "
     "hardware CET is enabled. Its presence in a stripped ELF signals a CET-enabled build, and binary "
     "patchers must preserve it before any new indirect destination."),
    ("How do `push rbp; mov rbp, rsp; sub rsp, 0x28` on Windows-style x86-64 map to stack usage?",
     "Save and anchor the caller's frame, then reserve 0x28 bytes: 0x20 of shadow space for the next call "
     "plus 8 bytes for alignment or locals. Linux builds skip shadow space and size locals directly."),
    ("How does GCC inline a tiny function at -O2, and what traces remain in the assembly?",
     "The callee's operations are absorbed into the caller body; no call/ret survives. Traces appear in "
     "debug maps (inlined ranges) or as repeated constant loads at multiple sites; without symbols, "
     "cloning patterns and repeated idioms suggest inlining."),
    ("What is the lea pattern for multiplying by a constant that isn't a power of two?",
     "The compiler decomposes the constant into shifts and adds: x*10 becomes lea eax,[rax*4]; lea "
     "eax,[rax*2+eax] style minute chains, or an imul for awkward constants. Factoring the constant back "
     "out of the instruction chain recovers the multiplier."),
    ("How do x86 instruction prefixes stack, and why must RE tools respect order?",
     "Legacy prefixes (0xF0 lock, 0xF2/0xF3 rep/repnz) precede REX (0x40-0x4F), and operand size depends "
     "on REX bits. Disassemblers must parse prefixes in order because the effective opcode and addressing "
     "change with the REX combination."),
    ("What does a PLT stub do in stripped x86-64 binaries?",
     "First calls jump to the dynamic resolver, which resolves the symbol and patches the GOT so later "
     "calls go direct; in disassembly the call shows as a call to plt-N rather than a symbol. Analysts "
     "map PLT indices back through relocations to recover called-function names."),
    ("How does an unbounded array index show up in assembly (CWE-129 view)?",
     "The index path lacks a bounds comparison right before the access: no cmp/jae guarding the register "
     "used as [base+reg*scale]. The load then reaches outside the buffer when input exceeds the nominal "
     "range — look for the missing guard, not for fancy arithmetic."),
    ("How is string comparison implemented by the compiler and what does its pattern reveal?",
     "repz cmpsb/cmpsq with a length register (or SSE-compares like pcmpistri on modern builds), followed "
     "by test/branch on the result — test al, al; jz equal. Recognizing the comparison chain helps you "
     "locate authentication branches in stripped binaries."),
    ("What does `and rsp, -16` accomplish?",
     "Right-aligns RSP to a 16-byte boundary, demanded by the ABI before calls and required for movaps/SSE "
     "operations on stack locals. Misaligned RSP plus movaps is a classic crash when hand-assembling stubs."),
    ("What is ROP and how does a reverse engineer spot a ROP chain attempt?",
     "Return-Oriented Programming chains short gadget sequences ending in ret to compose behavior without "
     "injecting code. Analysis spots a flood of 'ret' addresses written over the stack, often anchored at "
     "an overflowing strcpy/memcpy, with payloads built from binary gadgets rather than crafted instructions."),
]
for q, r in X86_EXTRA:
    add("re_knowledge", q, r, system=X86_SYS)

# ================================================================== 5) gcc/binutils
GCC_SYS = ("You explain compiler and binutils flags (GCC 13+, objdump, nm, readelf) and disassembly "
           "workflow on Linux x86-64 and ARM, concise and practical.")

GCC_FACTS = [
    ("What actually differs between gcc -O0, -O1, -O2, -O3 and -Os?",
     "-O0 emits unoptimized code that maps lines and variables cleanly (the debug view). -O1 applies basic "
     "local optimizations. -O2 is the standard release level (inlining, scheduling, common subexpression). "
     "-O3 adds aggressive vectorization and unrolling, sometimes larger or slower. -Os optimizes for size, "
     "skipping expansion-heavy transforms."),
    ("When would you compile with -fno-stack-protector for RE work?",
     "To reproduce un-patched historical binaries or bare-metal targets: it removes canary loads and "
     "epilogue checks, making emitted assembly cleaner for hand-diffing against a reference binary. Never "
     "for shipping code."),
    ("What does -m32 require on a modern x86-64 distro?",
     "GCC multilib support (gcc-multilib, 32-bit glibc and headers). It produces 32-bit binaries that use "
     "stack-based argument passing, used to mirror legacy or 32-bit-only targets found in RE."),
    ("How does `gcc -S` output differ from `gcc -c` output?",
     "-S stops after code generation and emits .s assembly text with directives — perfect for reading the "
     "compiler's back-end view. -c produces a .o object file with sections and relocations, still "
     "incomplete until linking."),
    ("What does -fPIC change in generated code?",
     "It enables position-independent code: data through RIP-relative or GOT references and external calls "
     "through the PLT. Required for shared libraries and what makes code and data segments relocatable "
     "under ASLR."),
    ("Does -g affect runtime behavior?",
     "No — -g only embeds DWARF debug metadata (files, lines, variables, types) used by debuggers and "
     "decompilers. Runtime behavior is identical; strip removes the DWARF without changing semantics."),
    ("What does -fsanitize=address detect and what is the cost?",
     "AddressSanitizer catches heap and stack overflows, use-after-free, double-free, and leaks at runtime "
     "using redzones and shadow memory. Expect around 2x slowdown and 2-3x memory overhead — much faster "
     "than valgrind and catches more classes of bugs."),
    ("What is the basic objdump workflow for a stripped ELF?",
     "objdump -d -M intel for Intel-syntax disassembly (add -j .text to scope it), readelf -h/-s/-a for "
     "headers, sections, symbols; nm -D for dynamic symbols; strings -a -tx for offset-anchored strings. "
     "Together they build the initial picture of a stripped binary."),
    ("What is LTO (-flto) and what should an RE analyst expect?",
     "Link-time optimization defers most optimization to the final link across translation units: heavy "
     "cross-module inlining, dead code removal, merged functions, and blurred per-file attribution. It "
     "makes recovering structure from symbols much harder."),
    ("What does -static change for analysis?",
     "It links libc into the ELF: no dynamic loader, no PLT/GOT, direct calls, and library-rich strings — "
     "plus syscalls machinery visible inline. Size grows and diffs hide inside glibc internals."),
    ("How are -I and -D flags arranged in a firmware build?",
     "-I paths add header search directories per-board or per-HAL; -D asserts feature macros (BOARD_X, "
     "CPU_OVERRIDE). Reproducing a firmware exactly means capturing both plus the -O level, "
     "march/mcpu/mfloat-abi, linker script, and map file."),
    ("What is -nostdlib for and when do firmware builds need it?",
     "It skips the standard C runtime and library link, so the build owns the entry point (_start), stack "
     "setup, and any freestanding helpers. Kernels and firmware with a custom linker script use it; the "
     "reachable code starts from the vector table instead of crt0."),
    ("Why does the plan compile both -O0 and -O2 for asm-training pairs?",
     "-O0 maps C structure to assembly cleanly for learning basic correspondence; -O2 produces realistic "
     "inlining, register allocation, and jump tables — the shapes the model actually sees in shipped "
     "binaries. Training pairs teach c_to_asm and asm_to_c at both levels."),
    ("How do you extract one function's disassembly with objdump?",
     "Reading symbol offsets from nm -S or readelf -s, then objdump -d -M intel --start-address=0xX "
     "--stop-address=0xY. Or, with a debug build: gdb -batch -ex 'disassemble /m func' for source-"
     "interleaved assembly at -O0."),
    ("Why use `strings -t x`?",
     "The -t x option prints each string's file offset in hex, anchoring matches for hex-editor or patch "
     "work and letting you check which section (readelf -S) contains them — useful to confirm whether a "
     "suspicious string lives in .rodata or an overlay region."),
    ("What does nm -C add over nm?",
     "-C demangles C++ symbols from the Itanium ABI naming, turning _ZN3Foo3barEi into Foo::bar(int). On "
     "shared objects combine with -D to demangle dynamic symbols — the practical map from mangled to "
     "human names."),
    ("What does -ffunction-sections -fdata-sections accomplish with --gc-sections?",
     "Each function or data object lives in its own section so the linker can garbage-collect unreferenced "
     "ones, shrinking firmware or tiny binaries. The reverse effect for RE: function order in the binary "
     "no longer matches source order, relying on the map file to recover layout."),
    ("How do you disassemble without symbols in gdb?",
     "Disassemble by raw address ranges (disassemble 0xstart,0xend), use /r to show raw bytes, or rely on "
     "rizin/radare2 analysis plus a decompiler view for structure. Treat decompiler output as a hypothesis "
     "and verify against recognizable assembly idioms and data references."),
    ("What does `strip` remove, and what does it leave?",
     "strip removes debug sections (.debug_*) and the symbol table (.symtab) but keeps what's needed to "
     "run: program headers, .text, .data, .bss, and the dynamic symbol table (.dynsym) for shared "
     "libraries. So stripped binaries still export their dynamic API for readelf/nm -D."),
    ("Why doesn't --gc-sections usually remove as much as a project hopes?",
     "Runtime-registered handlers (interrupt vectors, callbacks), GOT-driven references, init/fini "
     "sections, and weak symbols survive the collector because the linker can't see runtime registration. "
     "Audit with -Wl,--print-gc-sections and post-link size reports."),
]
for q, r in GCC_FACTS:
    add("re_toolchain", q, r, system=GCC_SYS)

GCC_EXTRA = [
    ("What is -Werror=format-security and what does it block?",
     "A hardened-build flag that turns dynamic format strings (printf(some_variable)) into compile errors, "
     "closing the CWE-134 path at build time. Distros enable it by default; for security-critical C branches "
     "reproduce it explicitly with -Werror=format-security."),
    ("What does -D_FORTIFY_SOURCE (2 or 3) change in linked code?",
     "With optimization on it swaps calls like sprintf, strcpy, memcpy to known-bounded __*_chk variants "
     "that abort on overflow. The presence of __strcpy_chk symbols in a stripped binary signals a hardened "
     "build; the semantics of the callsite are otherwise the same for RE."),
    ("How do objcopy --only-keep-debug and strip differ for shipping debug info?",
     "--only-keep-debug creates a sidecar file containing the DWARF while strip removes the debug sections "
     "from the shipped binary; the two match by build-id so debuggers can load the sidecar. Kernel and "
     "daemon teams ship such sidecars for symbolication."),
    ("What is the march/mcpu/mfloat-abi triplet for ARM cross-compilation?",
     "march or mcpu names the CPU variant, the second token names the FPU version (fpv4-sp-d16 etc.), and "
     "mfloat-abi picks soft, softfp, or hard calling for float arguments. Mixing hard and softfp binaries "
     "breaks calls with float parameters; reproduction must match all three exactly."),
    ("What does --specs=nano.specs do in embedded toolchains?",
     "Links newlib-nano, a minimized C library (no-float printf, small malloc) for size-constrained "
     "firmware. The linked runtime shrinks radically — RE of such images sees _write and minimal internals "
     "rather than full glibc-style paths."),
    ("Why isn't -Wunsafe-loop-optimization a CVE fixer, and what is the practical loop-bounds review?",
     "It warns when loop transforms might violate overflow assumptions — a translator concern, not a loop-"
     "bound check. Real review looks at loop guard types (int vs size_t), upper bounds with <=, and "
     "increment wraparound, backed by fuzzing at extremes."),
    ("What is the `readelf -S` view for locating a suspicious string's section?",
     "Each row lists name, type, address and offset; compare a string's file offset (strings -t x) against "
     "section ranges to see whether it sits in .rodata, an overlay region, or a writable section. Writable "
     "placement for what looks like a constant is itself a RED flag in patch analysis."),
    ("What is DWARF and which sections does it live in?",
     "DWARF is the debug format GCC emits (v5 default in GCC 13), in sections .debug_info, .debug_line, "
     ".debug_loc, and peers. Strip removes them but leaves code identical — decompilers lose variable and "
     "type data and fall back on assembly idioms and strings."),
    ("What warning flags give best coverage for C memory bugs at build time?",
     "-Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wformat=2 -Wformat-security -Werror=bounds "
     "catch most accidental memory arithmetic. None substitute for sanitizers + fuzzing; the goal is build-"
     "time funneling of memory discipline before runtime testing."),
    ("How does --coverage output help an RE workflow for a fuzzing harness?",
     "It instruments the binary with gcov counters that can dump after runs, revealing which code regions "
     "a corpus touched; fuzzing teams use this to drive corpus evolution. In reverse engineering of a "
     "research artifact, coverage instrumentation lets you correlate behaviors with source lines."),
]
for q, r in GCC_EXTRA:
    add("re_toolchain", q, r, system=GCC_SYS)

# ================================================================== 6) vuln
SEC_SYS = ("You are a vulnerability-analysis expert (CWE/CVE taxonomies, C/C++ memory and integer "
           "safety, web and firmware weaknesses).")

SEC_FACTS = [
    ("What is CWE-79 (XSS) at its root and what is CWE-78 for terminal agents?",
     "CWE-79 Cross-site Scripting: attacker-controlled data reaches an interpreter of active content "
     "(HTML/JS) without output-context encoding. CWE-78 OS Command Injection: attacker input concatenated "
     "into a shell command string — remediate with argv arrays and subprocess without shell=True."),
    ("What distinguishes CWE-120 (buffer overflow) from CWE-129 and CWE-190?",
     "120 is improper bounds on a write (copy loops, memcpy with attacker-controlled length). 129 is "
     "incorrect validation of an array index. 190 is integer overflow or wraparound that often leads to "
     "120: length computed with wraparound, under-allocated, then overflowed. The 190 -> allocation -> 120 "
     "chain is the most common CVE fingerprint."),
    ("What is CWE-416 (use-after-free) and its distinctive C fingerprint?",
     "Access via a pointer after its owner freed it: free(ptr) followed by dereference or a reference "
     "retained in a structure. Exploitation relies on heap grooming; defensive style nulls the pointer at "
     "free and rechecks ownership on reuse."),
    ("What is a double free (CWE-415) and why does the allocator matter?",
     "Calling free twice on occupied memory lets allocator freelists (glibc tcache) duplicate pointers, "
     "enabling write-what-where primitives. The code smell is two frees on ownership paths, frequently in "
     "error branches that free before returning."),
    ("What is CWE-401 (memory leak) and how does it differ from UAF?",
     "A leak means the resource is allocated but unreachable on some path (forgot to free on early return) "
     "— an availability problem. UAF means the resource was freed but an alias remains — a corruption and "
     "exploitation problem, not just growth."),
    ("What is CWE-476 (NULL dereference) and what source patterns favor it?",
     "Dereferencing a pointer that can be NULL on unchecked-return, failed-allocation, or missing-config "
     "paths: p->field right after p = malloc(...) or get_config(...). Especially CVE-worthy when network "
     "input can trigger the NULL (malformed packet -> crash)."),
    ("What is CWE-22 (path traversal) and how is it defended?",
     "Attacker-controlled path components (../ or absolute) escape an intended root. Defenses resolve with "
     "realpath and verify the result stays under a fixed prefix; extra defense in file services rejects "
     "symlink jumps and normalizes before any use."),
    ("What is CWE-89 (SQL injection) and what is the fingerprint?",
     "String-concatenated SQL executed against the database: input flows into query text without "
     "parameterization. Fix with prepared statements plus parameter binding, and grant the DB user least "
     "privilege."),
    ("What does CWE-77 include and how does it relate to CWE-78?",
     "77 is injection into any sub-language/interpreter (shell, templates, another mini-language), while 78 "
     "is the OS-command case specifically. Policy: avoid constructing commands or templates from raw "
     "strings; pass structured arguments or escape per the target's rules."),
    ("Why is unsafe deserialization (CWE-502) a threat in Python tooling?",
     "pickle and torch.load can instantiate attacker-chosen objects, executing code during load. Use JSON "
     "with schema validation, or torch.load(..., weights_only=True) for model files, and treat any "
     "deserialized payload as untrusted input (validate shape and type before use)."),
    ("What pattern signals missing authentication (CWE-306) in daemons?",
     "Mutating or privileged operations exposed by default with no auth guard: bind 0.0.0.0 plus admin "
     "endpoints, default passwords, or telnet enabled at boot. In IoT firmware this is the single most "
     "common CVE family."),
    ("What is CWE-190's role in firmware vulnerabilities specifically?",
     "Length and offset arithmetic in narrow types (8/16-bit) or unchecked sums wrap around, producing "
     "tiny allocations followed by large copies. Firmware code often does raw arithmetic on attacker-supplied "
     "length fields — the classic IoT binary protocol bug."),
    ("How do stack (CWE-121) and heap (CWE-122) overflows differ in review?",
     "121 writes past fixed-size locals (char buf[64] with attacker length) and may trash the return "
     "address; 122 writes past malloc'd buffers and corrupts adjacent allocations or heap metadata. Both "
     "start the same way — an unbounded length — but their reach and exploitation differ."),
    ("What is TOCTOU (CWE-367) for file operations?",
     "Time-of-check to time-of-use: the state can change between checking a path and acting on it "
     "(symlink swap). Counter it by opening with O_NOFOLLOW/O_EXCL and fstat'ing the resulting descriptor "
     "rather than trusting the path twice."),
    ("What is CWE-362 (race condition) and its C signature?",
     "Two execution contexts (threads, ISRs, processes) race on shared state without synchronization: "
     "check-then-act sequences, double-checked lazy init, shared buffers without atomicity. In firmware, "
     "ISR handlers reading shared flags need critical sections."),
    ("What is CWE-798 and why do hardcoded credentials dominate IoT CVE counts?",
     "Hardcoded or default passwords shipped in firmware (admin:admin over telnet, exported keys) are "
     "scanner-discoverable at massive scale, so every affected firmware yields broad CVE reports and bot "
     "activity. Mitigations are per-device provisioning and disabling legacy management transports."),
    ("What does reusing a nonce in a CTR-mode cipher cause (CWE-323)?",
     "Two keystreams with the same key and nonce cancel: their XOR leaks the XOR of the two plaintexts, "
     "often enough to recover both with known structure. Firmware protocols that skip nonces for speed end "
     "up with replayable, decryptable streams."),
    ("Why is %n dangerous in format strings (CWE-134)?",
     "%n writes the count of bytes printed so far into a pointer argument, turning attacker-controlled "
     "format strings into an arbitrary-write primitive (padded %08x leaks stack, then %n writes). The "
     "defensive rule: never let user input supply the format argument."),
    ("How does integer promotion turn CWE-190 into CWE-122 in one line?",
     "A comparison between types of different rank (int vs uint32_t) can promote -1 into a huge unsigned "
     "value, or an allocation like (size_t)len + 1 can wrap on a huge len — under-allocating and then "
     "overflowing. Width-aware checks (copy into uint64_t, compare, then cast) break the chain."),
    ("What is sign confusion (CWE-195) in recv loops?",
     "recv returns ssize_t (can be negative); code storing it in int and comparing against unsigned bounds "
     "mis-assigns negative or truncated values into accepted ranges. Concrete fix: validate bounds with "
     "the same signedness used by the API return."),
    ("How do CWE-20 and CWE-1284 differ in dataset labels?",
     "Both describe input validation failures; 1284 names mismatched quantity or type (too long, negative, "
     "wrong type) while 20 is the broader improper-validation umbrella (including injection families). When "
     "labeling, prefer the most specific weakness the patch actually introduces."),
    ("What is a patch gap and why do fork-based ecosystems stay vulnerable longer?",
     "The interval between a vendor's fix commit and downstream forks or release channels carrying it: "
     "embedded ecosystems (OpenWRT forks, router SDKs) rebuild slowly, so scanners keep finding vulnerable "
     "deployments years later. Mining the upstream commit and checking fork update commits is how researchers "
     "find them."),
    ("What makes a good vuln_detect training label (PrimeVul-style)?",
     "Pair-based records: a vulnerable function (before) with its fixed version (after) linked to the same "
     "commit, plus project and CWE metadata. Pairing suppresses pattern-only memorization and supports "
     "both detection learning and repair learning."),
    ("Why is surface-pattern memorization a training trap for vuln detection?",
     "Models latch on incidental scaffolds (function names, includes like 'from flask import') instead of "
     "the causal flaw. Countermeasures: pair-based labels, multi-project coverage per weakness class, and "
     "temporal (later-year) evaluation sets the model never saw."),
    ("What does CWE-6 enable among common fake-dataset-shortcut risks in generated labels?",
     "It represents 'unsafe external input handling' — the most over-generated label when datasets are "
     "auto-mined by keyword. Labels must come from real patch diffs with commit provenance (bigvul/primevul "
     "style), not keyword tagging."),
    ("What is CWE-611 (XXE) in XML-using services and agents?",
     "External entity resolution left on lets DOCTYPE declarations read local files or perform SSRF. "
     "Disable DTDs and external entities via parser options (libxml2 XML_PARSE_NOENT off) or parse with a "
     "restricted parser — for any agent that parses untrusted XML/RSS."),
    ("What is the practical workflow to convert a CVE into training records?",
     "Fetch the CVE and patch commit; extract the changed function before and after; classify CWE from the "
     "missing check; filter noise (build-only changes, style); store as schema-compatible paired records "
     "with CWE/CVE/project metadata. That is exactly the bigvul/primevul pipeline our data pipeline mirrors."),
    ("What is CWE-404 (improper resource shutdown) versus a plain leak?",
     "404 means shutdown happens wrongly or not at all across error paths (missing close, missed unlock, "
     "or shutdown in wrong order) — a state/correctness issue. A plain leak (401) is unreleased allocated "
     "memory. Firmware often shows 404 on interrupt disable paths."),
    ("What is a 'weak sandbox' confusions in tool agents relevant to security review?",
     "Agents that execute mixed-trust code with shared credentials or host filesystem access turn every "
     "tool call into an exfiltration or corruption risk. The safe design separates the sandbox's network, "
     "filesystem, and credential scopes per task and treats every result as untrusted data."),
    ("What is the 'semantic trap' named in literature and how does our pipeline avoid it?",
     "Literature (2026) shows fine-tuned security models can memorize shallow patterns while failing "
     "outside distribution. Our design avoids it through pair-based labels, multi-project diversity per "
     "CWE, and evaluation on later-year datasets the model never saw (CWE-Trace/benchmarks after 2025), "
     "as recorded in plan R10."),
    ("How should the assistant reason about a suspicious function when only the code is given?",
     "Trace data flow from user-controlled entry to sensitive sinks, check bounds on every size computation, "
     "notice ownership lifetime across all error paths, and name the concrete fault (missing bound, missing "
     "NULL check, wrong free) rather than pattern-matching on style features."),
    ("What makes a data-derived 'vulnerable code' sample different from merely 'style-odd' code?",
     "Vulnerable means a concretely reachable fault exists with attacker-influenceable input; style-odd "
     "code can be safe but unusual. Datasets label vulnerability only when a patch changed behavior for a "
     "CVE/security reason — that provenance is what makes the label honest."),
]
for q, r in SEC_FACTS:
    add("vuln_knowledge", q, r, system=SEC_SYS)

SEC_EXTRA = [
    ("How should the reason text in a dataset JSON be written to avoid pseudo-pattern learning?",
     "Reference the actual trigger in generic English (missing bound check, wrong free order) rather than "
     "project names or dataset-specific phrasing, so the model learns investigative semantics instead of "
     "memorizing scaffolds."),
    ("How are multi-CWE labels handled in training data?",
     "Keep the primary CWE as the reason's lead and include the rest in meta.cwe; outputs may report "
     "multiple CWEs during evaluation but must rank-order them, teaching the model to lead with the "
     "dominant weakness first."),
    ("Why keep general code in a specialization mix (15-20% modern slice)?",
     "Specialized SFT on one narrow corpus erodes general code competence; mixing general data maintains "
     "the base model's fluency (DAPT-style: general plus domain). Records in the modern slice serve exactly "
     "that counterweight in the locked final mix."),
    ("When is air-gapped analysis the right setting and when is an online sandbox needed?",
     "Air-gapped: malware RE, firmware patch audits, malicious document analysis — nothing leaves the "
     "machine. Online disposable VM: unknown payloads or network artifacts that need live behavior "
     "observation. The agent's policy statement should declare which mode it operates in and never mix "
     "the two."),
    ("What is N-day versus 0-day for an assistant security workflow?",
     "N-day: published vulnerability with known patch and artifacts — the agent can diff, locate, and "
     "verify. 0-day: unknown to vendors — the agent's role narrows to defensive triage (locate "
     "suspicious reachability, bound blast radius, prepare disclosure) and never weaponization guidance."),
    ("How does version-fingerprint trust affect security tooling outputs?",
     "Banner versions are forgeable and often stale. Prefer active verification (running commands, reading "
     "actual files) over banner-trusting claims, and label unverified version statements clearly in "
     "reports."),
    ("Why does the model need to distinguish local CVE (package on disk) from public CVE research?",
     "Local analysis works offline: query installed package versions against known vulnerability ranges in "
     "NVD metadata. Public research adds patch commits and advisories from network sources. The tool plan "
     "chooses accordingly, and the agent should never fabricate version data it didn't read."),
    ("What is patch-gap mining in fork-heavy ecosystems?",
     "Fork maintainers lag upstream fixes; scanning fork repos for their latest update commits and "
     "comparing against upstream CSV ranges locates still-vulnerable deployments. This pattern appears "
     "repeatedly in embedded ecosystems and informs both dataset labeling and tooling workflow."),
    ("What is a good question/answer derivation from a CVE description for training data?",
     "Derive the attacker's controlled entry (input channel), the mechanism (integer/bounds/state), and "
     "the resulting CWE, in 2-3 sentences of structured reasoning. Short, verifiable reasoning sequences "
     "train the model to classify rather than recite."),
    ("What is the exact purpose of holding the security slice at 40% of the final mix?",
     "It matches the specialization's centrality (the model is a security/local-RE model) while leaving "
     "room for tools (35%) and modern code (20%) to preserve breadth; the 5% micro slice supplies "
     "project-internal knowledge. The ratios were locked after the plan's data-volume survey."),
]
for q, r in SEC_EXTRA:
    add("vuln_knowledge", q, r, system=SEC_SYS)

# ================================================================== 7) firmware
FW_SYS = ("You are an embedded/firmware mentor (ARM Cortex-M, ESP32, STM32F4 register families, "
          "freeRTOS-free patterns).")

FW_FACTS = [
    ("Why is volatile mandatory for memory-mapped registers in C?",
     "Register state can change outside the C data flow (interrupts, DMA, external writes), and register "
     "accesses are side effects. volatile forbids the optimizer from caching or reordering those accesses: "
     "without it, a spin like while (!(REG & FLAG)) may be optimized into an infinite loop reading a stale "
     "value."),
    ("What is memory-mapped I/O versus port-mapped I/O?",
     "MMIO exposes peripheral registers as ordinary memory addresses on the bus (STM32 peripherals at "
     "0x40000000, accessed via normal loads and stores). Port-mapped I/O (x86 legacy) uses a separate 16-"
     "bit address space accessed with IN/OUT — irrelevant for Cortex-M and ESP32, which are pure MMIO."),
    ("What does the Cortex-M vector table contain and what happens on reset?",
     "Index 0 holds the initial Main Stack Pointer value and index 1 the Reset handler address, followed "
     "by NMI, HardFault, and further exception and IRQ entries. Reset loads MSP from vector 0 and PC from "
     "vector 1; Cortex-M3 and later can relocate the table via VTOR."),
    ("What are MSP and PSP on Cortex-M and why does an RTOS swap them?",
     "Two banked stack pointers: MSP (reset and exception mode) and PSP (thread mode). A real-time OS runs "
     "tasks on PSP to isolate them from exception-time startup, and exceptions always use MSP stacks, "
     "making context save/restore bookkeeping reliable."),
    ("What is BASEPRI versus PRIMASK for masking?",
     "PRIMASK is a single-bit switch disabling all maskable interrupts; BASEPRI masks interrupts at or "
     "below a chosen priority (numeric value; lower number = higher priority). RTOS critical sections use "
     "BASEPRI to prefer coarse blocking of just the scheduler-critical range."),
    ("What does a Cortex-M exception entry push automatically (no FPU)?",
     "Eight words: R0-R3, R12, LR, PC, and xPSR. The handler can then be a plain C function; a special "
     "EXC_RETURN value in LR drives unwind, and hardware handles stacking/unstacking. With FPCA active in "
     "a Cortex-M4F, 17 more words carry S0-S15 plus FPSCR via lazy stacking."),
    ("Which STM32 register enables the GPIOA clock and where does the register family live?",
     "RCC->AHB1ENR bit 0 (GPIOAEN); the RCC block sits at 0x40023800 on STM32F4. The pattern holds for "
     "every peripheral: enable the clock before touching registers, else reads appear as zero — the "
     "number-one bring-up bug."),
    ("Why does the ESP32's internal RAM banking matter for DMA buffers?",
     "The classic ESP32 splits its DRAM into DMA-capable and non-capable banks, with part pre-reserved for "
     "Wi-Fi. DMA-capable allocation must use heap_caps_malloc(..., MALLOC_CAP_DMA): a wrongly-placed "
     "buffer yields silent corruption rather than an error."),
    ("What is the watchdog's purpose and the safe feeding pattern?",
     "A hardware watchdog resets the MCU if not fed within its window — the only reliable guard against a "
     "wedged main loop. Feed only from the top of the certified main loop after successful polling; never "
     "feed from an interrupt or a timer that runs independently of application health."),
    ("How does a secure bootloader verify a firmware image (concept)?",
     "The bootloader reads the image and checks a cryptographic signature (ECDSA/RSA) against a public key "
     "held in eFuse/OTP storage before jumping; OTA updates land in a staged slot and commit only after a "
     "successful test boot. A magic header alone is NOT integrity — the signature check is the actual "
     "security."),
    ("What gotchas surround DMA transfers in firmware?",
     "Cache coherency (Cortex-M7 and similar need explicit cache invalidates), alignment of buffers, DMA-"
     "capable RAM placement (ESP32), completion callbacks racing the last byte, and interrupt priorities "
     "set so DMA completion doesn't starve higher-value work."),
    ("Why must ISRs stay short, and what is the correct division of labor with tasks?",
     "ISRs should only record the event (queue it, set a flag, wake a task via direct-to-task notification) "
     "and return quickly; longer work belongs in task context. Busy-waiting or calling non-reentrant APIs "
     "(printf, malloc) inside an ISR creates priority inversion and data corruption."),
    ("What is a safe critical section on single-core Cortex-M inside an RTOS?",
     "FreeRTOS provides taskENTER_CRITICAL()/taskEXIT_CRITICAL(), which raise BASEPRI to a level that "
     "blocks the racing interrupt. Keep the section tiny, never block or sleep inside it, and on dual-core "
     "(ESP32) pair it with a spinlock because another core can race independently."),
    ("What's the rule for TX ring buffers driving UART output?",
     "TX drains asynchronously via DMA or interrupt; ring capacity must exceed producer burst size or "
     "data silently overwrites. Design: wait-on-space via semaphore, count dropped frames for telemetry, "
     "and drop whole frames rather than partial bytes."),
    ("When to choose SPI versus I2C versus UART?",
     "SPI: full-duplex, fast, 4 wires, per-device chip-select — flash and audio. I2C: 2 wires, multi-device "
     "addressing with pull-ups, slower — sensors and configuration. UART: asynchronous point-to-point at a "
     "baud rate — consoles and GPS. Pin muxing tables in the device's reference manual decide the mapping."),
    ("How should you shape firmware strings for attacker data (the snprintf style)?",
     "Always snprintf(buf, sizeof(buf), formatted_with_const_literal, args) — format string literal, "
     "never attacker-controlled. Building messages with runtime-controlled format strings converts any "
     "delivery into CWE-134."),
    ("Why are C bitfields risky for register maps?",
     "Bitfield layout order and packing are implementation-defined; cross-compiler and cross-version "
     "emits differ. Safer: named 32-bit volatile integers per register plus explicit mask macros, or "
     "CMSIS-style accessor inlines compiled per architecture."),
    ("What do a linker script's MEMORY and SECTIONS encode for a Cortex-M build?",
     "MEMORY declares regions (FLASH at 0x08000000, RAM at 0x20000000, optional CCM RAM at 0x10000000); "
     "SECTIONS place .isr_vector first, .text in FLASH, .data at RAM with LOADADDR, .bss zeroed via "
     "_sbss.._sebss symbols, keeping the bootloader's boot-ram underlying section anchored."),
    ("What is bit-banding (Cortex-M3/M4) and when should you NOT reach for it?",
     "A hardware alias map that turns a single bit in certain regions into a full word at 0x22000000 (SRAM "
     "range) or 0x42000000 (peripheral range), letting a word-store atomically set/clear that bit. Only "
     "those regions support it — many peripheral registers (RCC, TIM, UART control) fall outside; use "
     "direct BSRR/W1TS-style registers where provided."),
    ("How does interrupt latency scale on a 168MHz Cortex-M4 (order of magnitude)?",
     "Entry is about 12 cycles; exit similar, plus 17 more with contents if FPU stacking is used. So "
     "byte-interrupts at 115200 baud (roughly 11.5 kB/s) are trivial, while calling printf from a high-"
     "frequency ISR at 168MHz consumes a sizeable fraction of each interval — keep ISRs lean."),
    ("How should fault handlers be instrumented for bring-up?",
     "Replace default handlers (HardFault/MemManage/BusFault/UsageFault) with ones that log SCB->CFSR and "
     "HFSR, dump the stacked PC/LR to UART, and reset via NVI SystemReset or watchdog. They should not "
     "silently loop; an LED heartbeat plus WDT reset makes the fault visible instead of hanging."),
    ("Why is Thumb-2 the only ISA for Cortex-M and how does it show in binaries?",
     "Cortex-M executes only the Thumb instruction set (mixed 16/32-bit Thumb-2 encodings). In binary "
     "images, the LSB of every function pointer is the Thumb bit; disassemblers decode automatically and "
     "ARM-A-style wide instruction handling does not appear."),
    ("What is CCM SRAM on STM32F407 and when should you avoid it?",
     "64KB of core-clocked RAM at 0x10000000, the fastest MCU-internal access. It is NOT reachable by "
     "other bus masters (no DMA, no Ethernet); use it for hot functions and ISR stacks, and never place "
     "DMA buffers there."),
    ("What is the F407 flash layout (RM0090)?",
     "1MB main flash split into 12 sectors: four 16KB, one 64KB, seven 128KB. Programming granularity is "
     "16 bits (a halfword) with 30K minimum erase/write cycles. Erase of a whole sector is the unit of "
     "erase — the bootloader must schedule rewrites accordingly."),
    ("What configuration drives a 168MHz Cortex-M4F PLL (STM32F407 case)?",
     "From an 8MHz HSE: M=8, N=336, P=2 gives 168MHz system clock. Flash wait states must rise to 5 at "
     "168MHz; forgetting the latency ramp causes hard faults on instruction fetch after the clock switch."),
    ("How do per-piece flash-encryption paths change RE strategy (ESP32 classic)?",
     "With AES-XTS eFuse-controlled flash encryption, a raw flash dump isn't plaintext — decryption "
     "happens during boot into RAM. Analysts typically pivot elsewhere (JTAG access if not fused off, or "
     "recovered config data), and must state that barrier before planning attack paths."),
    ("What does the ESP32 partition table hold and how does OTA A/B selection work?",
     "A table (usually at 0x8000) listing nvs, phy_init, otadata, ota_0/ota_1 app slots, plus custom "
     "schemes. OTA writes the new image into the inactive slot, updates otadata to boot it, and rolls "
     "back on failed boots — the reverse requires a valid previous slot's otadata marker."),
    ("What is NVS in ESP32 and its crypto caveat?",
     "Non-Volatile Storage: a wear-leveled key-value store for configuration and device secrets. Standard "
     "NVS is NOT encrypted unless NVS encryption is enabled alongside flash encryption — a raw partition "
     "read exposes plaintext config, so secrets belong in encrypted NVS when that risk matters."),
    ("What are eFuses in ESP32 and which flags matter for device reviews?",
     "One-time-programmable configuration bits: SECURE_BOOT, FLASH_CRYPT_CNT, DISABLE_JTAG, "
     "DIS_DOWNLOAD_MODE. Once burned, simple reflashing is blocked (soft path), so RE strategies must "
     "either target the already-configured state or replace the chip."),
    ("What is SysTick's role in a Cortex-M bare metal application?",
     "A core-internal 24-bit down-counter used for timebase: configure LOAD=(HCLK/HZ)-1 and a handler "
     "for a periodic tick, used by HAL for delay/timestamp and by an RTOS for preemption. Disabling it "
     "during deep sleep (WFI) is the classic power-saving pattern."),
    ("What does exhausting the main stack produce (and what's the detection)?",
     "Silent corruption (stack encroaching on .bss or .data) or, with an MPU guard, a MemManage fault. "
     "RTOS code detects near-exhaustion via high-water-mark bookkeeping (freelist of 'stack remaining') "
     "so leaks and overflows surface in telemetry before corruption does."),
    ("How does the Cortex-M FPU's lazy stacking help latency?",
     "FPU state stacking (S0-S15 and FPSCR, 17 words) is deferred when LSPEN is set: the exception frame "
     "reserves the space but copies only if the handler actually touches FPU registers, saving latency "
     "on hot ISRs that avoid floating-point math."),
    ("What is the BT (BootRAM trampoline) idea in a bootloader boot RAM setup?",
     "Bootloader's flash is swapped to public BootRAM at 0x0 with a minimal trampoline that moves the "
     "real bootloader image out of flash into RAM, so the same flash region can later be rewritten under "
     "safe boot semantics without change of vector table."),
]
for q, r in FW_FACTS:
    add("firmware_knowledge", q, r, system=FW_SYS)

FW_EXTRA = [
    ("What flags and guards make an ISR-safe single-core Cortex-M critical section?",
     "BASEPRI masking is the workhorse: suspend just the racing interrupt level (taskENTER_CRITICAL/"
     "taskEXIT_CRITICAL in FreeRTOS), keep the section tiny, and never block inside. Global PRIMASK-"
     "disable starves the scheduler tick and high-priority events — coarse, avoid in task code."),
    ("What does W1TS/W1TC (ESP32) or BSRR (STM32) avoid in GPIO updates?",
     "One-shot set and reset registers let concurrent code paths (tasks and ISRs) update disjoint bits "
     "without a read-modify-write race — the atomic-OR and atomic-ANDNOT of one word, unlike ODR-style "
     "read-then-write that can lose updates under interrupts."),
    ("What is the reason for the 5 flash wait states at 168MHz on STM32F4?",
     "Flash reads pipeline at a fixed clock; above ~150MHz on F4 the RAW flash cannot sustain zero-wait "
     "access, so FLASH->ACR latency of 5 cycles is required. Missing it corrupts execution after clock "
     "reconfiguration — the bring-up bug that manifests as HardFault in random code paths."),
    ("What is the bootloader/BootRAM distinction in Cortex-M0 (e.g., Raspberry Pi bootloader patterns)?",
     "Cortex-M0 lacks a VTOR, so its vector table is fixed at address 0. To run code from another region, "
     "either alias memory mapping (boot from flash or RAM via pins) or a small trampoline copies the real "
     "image to fixed-0 RAM first. M3/M4 relax this with VTOR writes."),
    ("How does the 'bounded stack check' pattern for deferred tasks work in FreeRTOS?",
     "Each task's stack top holds a known pattern; uxTaskGetStackHighWaterMark counts the minimum "
     "remaining ever seen, letting telemetry report near-overflow per task. Stack overflows have NO "
     "natural error — the watermark is the crash detection substitute."),
    ("What is the value of bit-banding speed for legacy code and when does the compiler help anyway?",
     "Hardware bit-banding guarantees an atomic single-bit operation; in many cases the compiler already "
     "emits a read-modify-write that is acceptable when the bit is task-local. Use banding primarily when "
     "ISRs interoperate with task-owned bits in certain peripheral regions."),
    ("What does flash sector layout mean for firmware update size?",
     "Because erase granularity is the sector (16KB/64KB/128KB on F407), updating a small app may erase a "
     "whole sector and rewrite it — real update byte count and wear equal the covered sectors, not the "
     "code delta. Design wear-leveling around sector grouping rather than line-level."),
    ("What does 'lazy stacking' defer exactly in a Cortex-M4F exception?",
     "The hardware reserves 17 words of frame for FPU state but skips the actual copies unless the handler "
     "provably touches FPU registers; consuming those words (via lazy stacking error path) may surrender "
     "the benefit on stacks not yet drained. Keep hot ISRs FP-free so the deferral completes unused."),
    ("What is the recommended handling of a device whose flash dump appears plaintext yet contains odd gaps?",
     "Gaps and unreadable regions mean encrypted regions (or vendor-hidden) — do not assume the entire "
     "image is plaintext. Analysts should read the partition map, identify which regions are encrypted "
     "(e.g., on ESP32, encrypted NVS/slot areas) and pursue hardware-level access paths further."),
    ("What is the practical reading for 'WDT window' style configuration?",
     "Setting a WDT with an early-warning interrupt (WWDG on STM32, ESP32's task WDT with priority) lets "
     "the software record diagnostics before reset; timing windows must allow the longest expected healthy "
     "operation. Configure the interrupt to capture state (which task/loop was wedged) before reset."),
    ("How should a 'no pins free' situation be handled on a pin-constrained design?",
     "First inventory alternate functions via the device's mux tables (STM32 AF table, ESP32 IO matrix) — "
     "most pins multiplex several peripherals. Serial alternatives (SWD, boot ROM UART) preserve debug "
     "channels when the design is constrained; always route them out to a test header."),
    ("What is the difference between SWD and JTAG in ARM Cortex-M debugging terms?",
     "SWD uses 2 wires (SWDIO, SWCLK) with the intrinsic trace port optional; JTAG uses 4-5 wires per the "
     "standard. Cortex-M devices support SWD universally; JTAG offers daisy-chaining for multi-chip "
     "designs. Both rely on the debug unit responding — fused-off debug is a hard stop for either."),
    ("Why is `while (1)` in a default handler a poor choice during bring-up?",
     "It hides the fault: execution stops silently until watchdog resets (or hangs forever if WDT is off). "
     "Instead, call fault handlers that log SCB->CFSR/HFSR and the stacked frame, then trigger a reset "
     "from telemetry — turning every fault into recoverable diagnostics."),
    ("What does 'constexpr static volatile' misinterpretation mean for register constants?",
     "Some developers attempt const on register defines; but register access requires volatile, and const "
     "confuses the optimizer differently than volatile-discard semantics. Keep register values in plain "
     "volatile typed pointers/defines; reserve const for true compile-time constants."),
    ("What is the mental rule to keep firmware buffer arithmetic CWE-190-free?",
     "Promote every attacker-influenced length to the widest unsigned type (uint64_t or size_t) BEFORE "
     "comparisons and arithmetic, run all bounds checks in that width, and only then cast down. Most "
     "classic wraparound bugs arise from keeping arithmetic in narrow types end-to-end."),
    ("What register change makes sense when moving shared firmware buffers to a new partition region "
     "(DMA and MPU)?",
     "Move buffers into DMA-capable RAM and mark their region via the MPU as no-cache or explicitly "
     "invalidated; device regions without cache-coherency need explicit maintenance after each transfer. "
     "Then re-audit all pointers still referring to the old region (compile-time checks and linker map "
     "diff catch leftovers)."),
]
for q, r in FW_EXTRA:
    add("firmware_knowledge", q, r, system=FW_SYS)

# ================================================================== 8) tools
TOOLS_SYS = ("You explain modern agent tool-calling (OpenAI-style JSON function calling, terminal "
             "agents, browser agents, MCP) for a code model trained on agentic data.")

add("tool_format", "What does a well-formed OpenAI-style function-call record contain?",
    "The assistant message carries tool_calls: [{id, type:'function', function:{name, arguments: '... a "
    "JSON STRING, not an object ...'}}], followed by one or more role:'tool' result messages; tools are "
    "declared once per request as [{type:'function', function:{name, description, parameters:<JSON "
    "Schema>}}]. Good training data keeps arguments-as-string and binds every call to its tool result.", system=TOOLS_SYS)
add("tool_format", "Why must tool results arrive with role='tool' rather than plain user text?",
    "The rendering layer binds results to their call via call id and special separators, marking the data "
    "as tool-produced rather than as the user speaking — a different signal that prevents inventing "
    "results. Flattening results into chat text loses that attribution and trains the model to fabricate.", system=TOOLS_SYS)
add("tool_format", "What should a terminal agent do when a shell command fails?",
    "Read stderr, form one concrete hypothesis, and retry with a changed action (missing dependency -> "
    "install; wrong path -> locate file; permission -> report and request elevation). Never claim success "
    "without verification output — the failure transcript itself is what teaches the model discipline.", system=TOOLS_SYS)
add("tool_format", "How does a browser agent act from a truncated HTML snapshot?",
    "The snapshot preserves page state with element references; the agent maps the user task to the next "
    "single interaction (click/fill/hover on a referenced element) and emits it in the dataset's repr "
    "format ([link] X -> CLICK). It reasons from the truncated DOM state and requests the next snapshot "
    "instead of inventing page content.", system=TOOLS_SYS)
add("tool_format", "What is MCP (Model Context Protocol) and how does it differ from function calling?",
    "MCP standardizes how hosts talk to servers that expose tools, resources, and prompts over a "
    "transport (stdio/HTTP), so many heterogeneous servers plug into one session. Function calling is the "
    "per-model JSON call format; MCP is the plumbing standard carrying those JSON-shaped calls from any "
    "compliant server.", system=TOOLS_SYS)
add("tool_format", "When should an agent define a NEW tool instead of hand-running a workflow?",
    "When the same multi-step operation recurs: wrap it once as a parameterized tool with validation and a "
    "short description, then reuse. A recurring compile-and-convert step, for example, is tool material; "
    "a genuinely one-off inspection stays an ad-hoc shell step.", system=TOOLS_SYS)
add("tool_format", "How does the agent choose between a search tool and a fetch tool?",
    "Search for discovery across unknown sources; fetch when the target URL or document is already known "
    "(doc page, PDF, code file). Standard chain: search, select the best link, fetch, read, answer with "
    "citations — and skip fetching pages whose search snippets already answer the question.", system=TOOLS_SYS)
add("tool_format", "What makes a tool schema GOOD?",
    "Exact name, one clear purpose, typed JSON-Schema parameters with descriptions and required fields, "
    "enums where valid, and a description stating when to use it and what it returns. Call quality scales "
    "with schema quality: vague descriptions produce hallucinated arguments.", system=TOOLS_SYS)
add("tool_format", "What is the verification rule for agent claims?",
    "Anything checkable must be checked: run the command, read the file, fetch the URL, and quote the "
    "observed output. Uncertainty is declared when verification is impossible. Data that records failures "
    "and empty results faithfully is what trains this behavior.", system=TOOLS_SYS)
add("tool_format", "What is RAG in this toolset and when does it beat long context?",
    "Retrieval-Augmented Generation fetches the few relevant chunks of a large artifact (datasheet PDF, "
    "huge source file) by keyword or embedding and feeds only those to the model. On 12GB hardware it "
    "beats giant contexts on cost and precision; the model must cite what it retrieved and re-query when "
    "evidence is partial.", system=TOOLS_SYS)
add("tool_format", "Can an agent emit multiple tool calls in one assistant message?",
    "Yes — batch independent calls (read two files, query two IDs); the runtime executes and returns "
    "results in order. But calls that need each other's OUTPUT must be sequenced across turns, otherwise "
    "the second call's arguments get fabricated before the first result exists.", system=TOOLS_SYS)
add("tool_format", "What is the most common tool-using-model failure, and how is data supposed to fix it?",
    "Hallucinated results: narrating an outcome that never occurred, especially after an unexpected "
    "failure. Training data fixes it by recording failures and empty results faithfully, and pairing them "
    "with the policy 'say so, then re-query differently' — so the model learns verification instead of "
    "narration.", system=TOOLS_SYS)
add("tool_format", "Why design tools idempotently?",
    "Idempotent calls tolerate retries without compounding side effects (mkdir -p, SET instead of "
    "INCREMENT). Network flaps and re-emitted calls happen constantly in agent loops; idempotent "
    "interfaces turn those into no-ops instead of corruption.", system=TOOLS_SYS)
add("tool_format", "How should file-edit tool calls stay reviewable?",
    "Prefer exact-match replacement with a unique anchor (oldString) or a whole-file rewrite when the "
    "change is wholesale; avoid fragile pattern-based edits. Reviewable diffs are the audit trail — "
    "non-deterministic rewrites make verification and rollback impossible.", system=TOOLS_SYS)
add("tool_format", "What is the difference between an agent's plan and an explicit todo state?",
    "A plan is reasoning about sequence (usually in thinking text); a todo list is observable, checkable "
    "state the runtime can inspect. Better agentic data externalizes progress as tool-mediated todo "
    "updates (add step, complete step) so users and harnesses can track it — narration-only plans are "
    "weaker.", system=TOOLS_SYS)
add("tool_format", "What is an agent sandbox and its hard walls?",
    "A restricted execution environment with scoped network, filesystem, and credential access. Hard "
    "walls: never touch resources outside granted scopes (except an explicit temp dir), never echo "
    "secrets into logs or tool output, and never open network shell access without an explicit, user-"
    "approved need.", system=TOOLS_SYS)

# ================================================================== count & write

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    random.shuffle(RECS)
    if len(RECS) > TARGET:
        RECS[:] = RECS[:TARGET]
    seen = set()
    deduped = []
    for task, lang, msgs in RECS:
        key = (tuple(m["role"] + m["content"][:64] for m in msgs))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((task, lang, msgs))
    RECS[:] = deduped
    shard, chars, fh, total = 0, 0, None, 0
    per_task = {}

    def writer():
        nonlocal fh, shard, chars
        fh = open(OUT / f"micro_self-shard-{shard:05d}.jsonl", "w", encoding="utf-8")
        chars = 0

    writer()
    for i, (task, lang, msgs) in enumerate(RECS):
        per_task[task] = per_task.get(task, 0) + 1
        rec = {"id": f"micro_self-{i:07d}", "source": "micro_self", "task": task,
               "lang": lang, "messages": msgs}
        line = json.dumps(rec, ensure_ascii=False)
        fh.write(line + "\n")
        chars += len(line) + 1
        total += 1
        if chars >= SHARD_CHARS:
            fh.close()
            shard += 1
            writer()
    fh.close()
    (OUT / "_summary_micro.json").write_text(json.dumps(
        {"slice": "micro", "time": time.strftime("%F %T"),
         "kept": total, "by_task": per_task}, ensure_ascii=False, indent=2))
    print(json.dumps({"slice": "micro", "kept": total, "by_task": per_task},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
