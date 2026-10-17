# خطة مشروع النموذج المتخصص في الأمن منخفض المستوى — `vulnre`

> مبنية على 3 ملفات خطط (في `~/Downloads`):
> 1. تدريب-نموذج-محسّن-على-RTX-3060.md (المسار A: QLoRA محلي / المسار B: LittleBit على EVREN)
> 2. تدريب-نموذج-ذكاء-اصطناعي-للكشف-عن-الثغرات.md (ثغرات IoT/Embedded + Datasheets + Assembly)
> 3. استشارة-صحة-شخصية-على-موديل-مختار.md (LittleBit / EVREN / ضغط الموديلات الكبيرة)
>
> الجهاز: RTX 3060 12GB · Ryzen 5 5600 · 16GB RAM · ~216GB حرة · Fedora

## الهدف النهائي
نموذج متخصص قوي في: الكود منخفض المستوى + الهندسة العكسية (x86/x64 + ARM) + كشف الثغرات (IoT/فيرموير) + استدعاء الأدوات والتحكم بالكمبيوتر بأسلوب الوكلاء — يُدرَّب محلياً بـ QLoRA، ثم يُضغط لاحقاً بـ LittleBit-2 على EVREN (هدف مرحلة أولى: ~0.55 bpw)، ويُنشر بحثياً/دفاعياً على Hugging Face.

## قرارات مثبتة (Rulings)
- **R1 — البيئة**: Python 3.12 عبر `uv` في `vulnre/.venv`. تم التحقق: `torch==2.14.1+cu130`, `cuda available: True` على RTX 3060، `bitsandbytes 0.50.2`.
- **R2 — القاعدة (اختيار المستخدم)**: تنزيل **الاثنين معاً**: `deepseek-ai/deepseek-coder-6.7b-base` + `LLM4Binary/llm4decompile-6.7b-v1.5` (كلاهما عائلة llama، vocab ~32K → الوحيدة المتوافقة مع هدف LittleBit الصغير؛ الثانية مبنية أصلاً على الأولى).
- **R3 — تصحيح مستودعات وردت خطأً في الملفات**: `benjaminbeilharz/better_bigvul` ❌ غير موجود، `vul-hub/cvefixes-c` ❌ غير موجود. البدائل المتحققة:
  | الغرض | المستودع الفعلي |
  |---|---|
  | BigVul | `bstee615/bigvul` |
  | DiverseVul | `bstee615/diversevul` |
  | CVEfixes (splits) | `starsofchance/CVEfixes_v1.0.8` |
  | PrimeVul | `colin/PrimeVul` |
  | Fenrir v2.1 (طبقة عامة) | `AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1` |
  | Tool-use (xlam 60k) | `eitanturok/Salesforce-xlam-function-calling-60k` |
- **R4 — LittleBit**: الوصفة الرسمية للمجتمع محفوظة في `third_party/littlebitllm-recipe/recipe/` (setup/train/eval). التدريب يحتاج ~107GB VRAM لـ Qwen3-8B @0.55 ← **على EVREN لاحقاً**، وليس على 3060. أوزان المجتمع التجريبية تُنزَّل للتجربة المحلية (inference).
- **R5 — إدارة العمل**: المشروع غير مخزّن في git؛ التتبع عبر `PLAN.md` + `progress.md`.
- **R6 — طبقة بيانات الأدوات (إضافة المستخدم 2026-10-16)**: «قوي في التحكم بالكمبيوتر واستخدام الأدوات مثلك: متصفح + موجه أوامر + إنشاء أدوات». المستودعات المتحقق منها (الأحجام الفعلية من HF API):

  | الطبقة | المستودع | الحجم |
  |---|---|---|
  | طرفية/أوامر (مهام مهارية: تشغيل، إدارة نظام، أمن، سكربتات) | `nvidia/Nemotron-Terminal-Corpus` | ~8.2GB |
  | مسارات أدوات حديثة (tool_calling + وكيل تفاعلي + بحث) | `nvidia/Nemotron-SFT-Agentic-v2` | ~21.8GB |
  | تحكم فعلي بالكمبيوتر (مسارات OpenHands كاملة) | `nebius/SWE-rebench-openhands-trajectories` | ~2.1GB |
  | Tool-use موسّع (160k + 200k) | `Nanbeige/ToolMind` | ~4.5GB |
  | Tool-use مدمج (26k API) | `Team-ACE/ToolACE` | 37MB |
  | ReAct + ToolBench + عينات سلبية | `internlm/Agent-FLAN` | ~220MB |
  | Function calling كلاسيكي | `glaiveai/glaive-function-calling-v2` | ~1GB |
  | متصفح (HTML/إجراءات) | `osunlp/Mind2Web` | ~6.5GB |
  | متصفح (أدوات Playwright بصيغة tool-calling — MIT) | `jdaddyalbs/playwright-mcp-toolcalling` | صغير |
  | تقييم MCP لاحقاً | `ScaleAI/MCP-Atlas` + `marin-community/mcp-atlas-easy` | صغير |

  الإجمالي ≈ 44GB. **حدود النطاق**: التحكم الرسومي (GUI/لقطات شاشة) يحتاج نموذجاً بصرياً — خارج نطاق قاعدة نصية؛ المتصفح يُغطى عبر HTML/A11y + استدعاء الأدوات. سكربت التنزيل: `scripts/download_tool_data.sh`.

- **R7 — سقف الحجم مرن (تحديث المستخدم 2026-10-16)**: >1GB مقبول إذا زاد الخبرة → **الأولوية للجودة لا للحجم**. الخطة: إنتاج نقطتي ضغط (0.55bpw و ~0.8–1.0bpw) ومقارنتهما بالأرقام في التقييم؛ خيار مفتوح: ضغط قاعدة أكبر (13B) على EVREN ضمن ~2–3GB.
- **R8 — السياق (تحقق 2026-10-16)**: القاعدة `deepseek-coder-6.7b` تدعم **16K توكن** حالياً (`max_position_embeddings: 16384` = 4K أصلي × rope linear ×4، و`tie_word_embeddings: false` أي embedding وlm_head منفصلان 264MB لكل منهما FP16). خزن KV ثقيل لغياب GQA: ~0.5MB/توكن FP16 → سياق 16K = ~8GB VRAM. المسار: T9b (تمديد اختياري إلى 32K) + خزن KV مضغوط (int8/int4) للتشغيل؛ الملفات الضخمة (فيرموير) عبر الاسترجاع RAG لا السياق العملاق.
- **R9 — شريحة «الحداثة» + هندسة تدريب النماذج (طلب المستخدم 2026-10-16)**: تغطية أحدث التقنيات (ويب/واجهات) + معرفة بطُرق تدريب/نسخ النماذج:

  | الطبقة | المستودع | الحجم |
  |---|---|---|
  | كود حديث متعدد اللغات (5M عينة، 2025) | `nvidia/OpenCodeInstruct` | ~7GB |
  | نسخة مصفّاة (اختبارات وحدة + حكم LLM) | `EER6/nvidia-OpenCodeInstruct-refined` | ~470MB |
  | Python عالي الجودة (مُنفَّذ/متحقق) | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | صغير |
  | واجهات ويب CoT (2026، رخصة GPL-3 — تنبيه) | `AdhyanshVerma/html-css-js-cot` | صغير |

  **آليات الحداثة الثلاث**: (1) البيانات الحديثة تُفلتر لغاتاً في T7 (JS/TS/HTML/CSS + عام)، (2) استدعاء أدوات/توثيق حيّ (RAG) — القاعدة نفسها من 2023 فلا تحفظ تفاصيل 2025+ داخلياً، (3) تحديث دوري رخيص عبر LoRA (~100MB) عند الحاجة. **معرفة تدريب النماذج**: مصغّر ذاتي (500–2000 QA) من وثائق مشروعنا (QLoRA/3060، LittleBit، EVREN) + تمارين الأكواد/الأدوات. ملاحظة: `DeepSeek-Coder-V2-Lite` (MoE) أحدث لكن ليس عائلة Llama → لا يصلح لهدف الضغط؛ نبقى على القاعدة (R2).

## هيكل المشروع
```
vulnre/
├── .venv/                          # بيئة Python 3.12 (torch cu130)
├── PLAN.md / progress.md / .gitignore
├── data/
│   ├── raw/       # bigvul, diversevul, cvefixes, primevul, fenrir, xlam
│   │              # + أدوات (R6): toolace, toolmind, nemotron-terminal, nemotron-agentic,
│   │              #   swe-rebench-openhands, agent-flan, glaive-fc, mind2web, mcp-atlas(+easy), playwright-mcp
│   ├── datasheets/# ESP32 TRM, STM32 RM0090, ARM Cortex-M4 DDI0439 ✓
│   └── processed/ # JSONL/Parquet جاهز للتدريب
├── models/        # deepseek-coder-6.7b-base, llm4decompile-6.7b-v1.5, littlebit-qwen3-8b-0.55bpw
├── scripts/       # سكربتات المعالجة والتدريب (download_tool_data.sh ✓)
└── third_party/
    ├── LittleBit/                # كود سامسونج الرسمي (استُنسخ ✓)
    └── littlebitllm-recipe/      # وصفة المجتمع الكاملة (setup/train/eval) ✓
```

## المهام
- [x] **T0** البيئة + CUDA + bitsandbytes — تم التحقق فعلياً ✓
- [x] **T1** بيانات أساسية: bigvul + diversevul + cvefixes + primevul (تحميل خلفي — جارٍ)
- [x] **T2** بيانات إضافية: fenrir ✓ + xlam ✓ (اكتمل)
- [x] **T2b** بيانات الأدوات (R6): 11 مستودعاً ≈ 44GB — اكتمل ✓ (11/11 بدون فشل)
- [ ] **T2c** شريحة الحداثة (R9): OpenCodeInstruct + مصفّاة + self-oss + CoT ويب (~8GB) — تنزيل خلفي جارٍ
- [x] **T3** استنساخ LittleBit + وصفة المجتمع ✓
- [ ] **T4** تحميل النماذج: deepseek-coder-6.7b-base + llm4decompile-6.7b-v1.5 + littlebit-qwen3-8b-0.55bpw (جارٍ)
- [ ] **T5** اختبار دخان: تحميل 4-bit + توليد ✓ (Qwen3.5-4B، 3.12GB VRAM) — متبقٍ: خطوة QLoRA فعلية
- [x] **T6** Datasheets ✓: ESP32 TRM (784ص) + STM32 RM0090 (1741ص، عبر Wayback) + ARM Cortex-M4 DDI0439C (121ص، عبر Keil)
- [🚧] **T7** بايبلاين البيانات: **v1 منجزة** (`scripts/build_dataset.py` — بثّي، سجل قانوني موحّد، dedupe، شرائح JSONL) — فحص مخططات 16 مصدراً ✓ + اختبار دخان 4800 سجل ✓ + تحويل كامل جارٍ (security+tools كوحدات systemd). المتبقي: الشريحة الحديثة (R9) + Mind2Web + المصغّر الذاتي + الخلط النهائي + Parquet
## R10 — مسح المنافسين والأدبيات (2026-10-16، آخر 6 أشهر)
- **Cisco Antares (يوليو 2026)**: عائلة نماذج أمنية صغيرة 350M/1B (مفتوحة الأوزان) + 3B (محتجزة) على IBM Granite 4.0 — «توطين ثغرات» agentic عبر طرفية (يبحث في المستودع بأوامر shell ويحدد الملفات المصابة)، تدريب SFT+GRPO، وبِنشمارك VLoc Bench (500 مهمة/290 مستودع/147 CWE). Antares-1B: File F1 ‏0.209 ويتفوق على GLM-5.2 (753B) ونماذج أكبر 250×. → **يصدّق أطروحتنا جوهرياً** (نماذج أمنية صغيرة محلية). فجواتنا المتبقية أمامه: (1) لا أحد يخدم RE المنخفض المستوى (asm/ثنائيات/تفكيك)، (2) لا أحد يخدم firmware/IoT، (3) أدوات موسّعة (متصفح/MCP/طرفية) لا طرفية فقط، (4) ضغط متطرف إلى ~1GB لا يجرؤون عليه (vocab 100K عندهم).
- **Foundation-Sec (Cisco)**: سلسلة 8B على Llama-3.1: Base → Instruct (2025) → Reasoning (يناير 2026، أول نموذج تفكير أمني مفتوح) → 1.1-8B قيد الإنتاج لدى Deloitte/SoftBank (يونيو 2026). قوي لكن 16GB FP16 + vocab 128K → لا يصلح لأهدافنا الضاغطة (يؤكد R2/R7).
- **تحذيرات أدبية مهمة (2026)**: «Calibration Without Comprehension» (arXiv 2606.20502) و«Semantic Trap» (2601.22655): النماذج المضبوطة قد تحفظ أنماطاً سطحية وتفشل خارج التوزيع. الحل: **تقييم زمني نظيف** (تدريب ≤2024، تقييم 2025+) + فحص فهم بدل الأنماط الشكلية. قالب CWE-Trace (834 عيّنة Linux، 74 CWE) جاهز للاستخدام كتقييم.
- **بنشماركات للتبني في T10**: FAITH (CTIBench/CyberMetric/SecEval/SecBench/MMLU-Security — أداة Cisco الرسمية)، IoTVulBench (أغسطس 2026 — أول بنشمارك firmware IoT بشري التحقق)، CWE-Trace (زمني)، REFORGE/HOBRE لفهم جودة RE. مقارنة مباشرة مع Antares-350M/1B (جاري تنزيلهما الآن).
- **أدوات/بيانات مرشحة جديدة**: patchwork-cve (16,597 CVE مع root-cause+taint، سبتمبر 2026)، OpenAnt (Apache-2.0 — قالب orchestration استكشاف ثغرات)، NeverD (محرك تفكيك صديق للـ LLM)، اتجاه «decompilation مع حلقات تحسين متعددة الجولات» (arXiv 2606.16162) لتشكيل T8b.
- **الضغط**: BitNet/ternary صار تياراً رئيسياً في 2026 (TerMeZO سبتمبر 2026، BitNet v2 بتفعيلات 4-bit، VibeVoice QAT تدريجي). LittleBit يبقى الخيار الأساسي محلياً؛ تجربة «تحويل ternary تدريجي» لموديلنا النهائي ممكنة على EVREN لاحقاً؛ bitnet.cpp للاستدلال السريع.

- [ ] **T8** توليد أزواج Assembly (gcc -O0/-O2 -S من BigVul/DiverseVul + إضافة ARM لاحقاً)
- [ ] **T9** QLoRA تدريب حقيقي (DeepSeek-Coder أو LLM4Decompile كقاعدة)
- [ ] **T9b** (اختياري — R8) تمديد السياق إلى 32K: YaRN/linear + تدريب قصير على تسلسلات طويلة
- [ ] **T10** تقييم: defect detection + LLM4Decompile eval + عينات RE يدوية + MCP-Atlas (أدوات)
- [ ] **T11** نشر على HF (إطار بحثي/دفاعي موثّق في البطاقة)
- [ ] **T12** (المسار B) LittleBit-2 على EVREN: Qwen3-8B @0.55 أولاً (5 epochs) ثم نماذج أكبر
- [ ] **T13** harness توزيع العمل على عدة نسخ مضغوطة (رؤية المستخدم الأصلية: نسخة لكل قسم/مرحلة)

## مراجع سريعة
- أوامر LittleBit المجتمعية: `third_party/littlebitllm-recipe/recipe/train.sh` (متغيرات: `MODEL_ID`, `EFF_BIT`, `EPOCHS`, `NUM_GPUS`)
- كود سامسونج: `third_party/LittleBit/`
- نقاط تفتيش LLM4Decompile جاهزة للفك: `LLM4Binary/llm4decompile-6.7b-v1.5`
- EVREN: مجاني حتى 1 نوفمبر 2026 للـ API؛ التدريب بالرصيد (H200 ≈ 10 CR/ساعة)
- بيانات الأدوات (R6): جدول R6 أعلاه + `scripts/download_tool_data.sh`
- تقييم الأدوات: MCP-Atlas (500 مهمة، 36 سيرفر MCP/220 أداة) + MCP-Atlas-Easy (استدعاء واحد — مناسب للقواعد base)
- المتصفح: Mind2Web (HTML traces) + playwright-mcp-toolcalling (أدوات Playwright بصيغة tool-calling جاهزة)
