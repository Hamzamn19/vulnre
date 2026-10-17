# Ledger — vulnre

(سجل التنفيذ والقرارات. يُحدَّث بعد كل مهمة.)

## 2026-10-16
- **بدء الجلسة**: فحص 3 ملفات خطط + مشروع `Default Project` السابق (bench/lora_gate/CLEF) + العتاد (RTX 3060 12GB, 16GB RAM, 216GB حرة).
- **R1**: إنشاء `vulnre/.venv` (Python 3.12.14 عبر uv) + torch 2.14.1+cu130 + transformers 5.20 + peft/trl/bitsandbytes → `ENV_READY`. التحقق: `cuda available: True`, `gpu: NVIDIA GeForce RTX 3060`, `bnb: 0.50.2`.
- **R3**: تحقق فعلي من مستودعات HF — رُفضت أسماء غير موجودة وردت في الخطط (`benjaminbeilharz/better_bigvul`, `vul-hub/cvefixes-c`) واستُبدلت بأسماء متحققة (انظر PLAN.md).
- **T2 اكتمل**: fenrir (434MB) ✓ + xlam parquet ✓ في `data/raw/`.
- **T3 اكتمل**: `git clone SamsungLabs/LittleBit` ✓ + وصفة المجتمع `littlebitllm-recipe` ✓.
- **سؤال القاعدة للمستخدم** → اختار: DeepSeek-Coder + LLM4Decompile معاً.
- **T1 جارٍ**: تحميل bigvul/diversevul/cvefixes/primevul (خلفية).
- **T4 جارٍ**: تحميل القاعدتين + أوزان littlebit-qwen3-8b-0.55bpw (خلفية).
- **T6 جارٍ**: تحميل Datasheets (خلفية).
- **T5 جارٍ**: اختبار دخان 4-bit على Qwen3.5-4B (من كاش HF) (خلفية).

## 2026-10-16 (تكملة)
- **T6 اكتمل ✓**: `ESP32_TRM_en.pdf` (784ص) + `STM32F407_RM0090.pdf` (1741ص) + `ARM_CortexM4_TRM_DDI0439C.pdf` (121ص) في `data/datasheets/`. مصادر بديلة: st.com غير متاح (HTTP 000) → نسخة Wayback موثقة (2026-10-08)؛ developer.arm.com يرفض (18) → مرآة Keil الرسمية.
- **T5 جزئي ✓**: Qwen3.5-4B تحميل 4-bit NF4 على GPU (3.12GB VRAM) + توليد ناجح. متبقٍ: خطوة تدريب QLoRA فعلية (إثبات أن التدريب يعمل).
- **طلب المستخدم الجديد**: طبقة بيانات أدوات قوية — «يفهم كيف يستخدم الأدوات مثلك: متصفح، موجه أوامر، إنشاء أدوات» بأسلوب الوكلاء الحديث.
- **بحث HF موسّع ✓ → R6** أُضيف إلى PLAN.md — 11 مستودعاً متحققاً منها بالحجم الفعلي (≈44GB):
  - `nvidia/Nemotron-Terminal-Corpus` (~8.2GB): مهام طرفية مهارية (تشغيل/إدارة/أمن/سكربتات) — موجه الأوامر.
  - `nvidia/Nemotron-SFT-Agentic-v2` (~21.8GB): tool_calling 15GB + مسارات وكيل تفاعلية 6.3GB +بحث 0.6GB.
  - `nebius/SWE-rebench-openhands-trajectories` (~2.1GB): مسارات OpenHands كاملة (تفكير+أفعال+ملاحظات) — تحكم حقيقي بالكمبيوتر.
  - `Nanbeige/ToolMind` (~4.5GB): 160k مؤلفة بأدوات + 200k معززة (مع ToolACE/xlam/glaive للاستعلامات).
  - `Team-ACE/ToolACE` (37MB) + `internlm/Agent-FLAN` (~220MB، ReAct + ToolBench + سلبيات) + `glaiveai/glaive-function-calling-v2` (~1GB).
  - `osunlp/Mind2Web` (~6.5GB، HTML للتصفح) + `jdaddyalbs/playwright-mcp-toolcalling` (أدوات متصفح Playwright — MIT).
  - `ScaleAI/MCP-Atlas` + `marin-community/mcp-atlas-easy` (تقييم MCP لاحقاً؛ صغيرة).
- **T2b بدأ**: تنزيل مجموعة الأدوات أعلاه خلفياً (`scripts/download_tool_data.sh`).
- **حدود النطاق**: التحكم الرسومي (GUI/لقطات شاشة) يحتاج نموذجاً بصرياً — خارج نطاق قاعدة نصية؛ المتصفح يُغطى عبر HTML/A11y + استدعاء أدوات.
- **تحديثات المستخدم (حجم/سياق/لغات)**:
  - سقف الحجم مرن: >1GB مقبول إن زاد الخبرة → **R7** (نقطتا ضغط 0.55 و 0.8–1.0 + خيار 13B على EVREN).
  - سقف السياق: القاعدة 16K (linear×4 من 4K أصلي)؛ KV ~0.5MB/توكن FP16 (16K = ~8GB VRAM) → **R8** + T9b (تمديد 32K اختياري) + KV مضغوط + RAG للملفات الضخمة.
  - لغات كتابة الكود: C/Python محور التحسين؛ حماية بقية اللغات (القاعدة أصلاً تدرّبت على 87 لغة) بخلط 10–20% كود عام في T7.
- **R9/T2c — شريحة الحداثة**: المستخدم يريد تغطية أحدث التقنيات (ويب/واجهات) + هندسة تدريب النماذج → اعتماد شريحة كود حديث (~8GB: `nvidia/OpenCodeInstruct` 5M + نسخة مصفّاة + Python متحقق + CoT ويب 2026) وبدأ تنزيلها خلفياً. آليات الحداثة: بيانات حديثة + أدوات/توثيق حيّ (RAG) + تحديث LoRA دوري رخيص. معرفة تدريب النماذج: مصغّر ذاتي (500–2000 QA) من وثائق المشروع في T7.

## 2026-10-16 (نافذة استقلالية — المستخدم خارج المنزل ~7 ساعات)
- المستخدم منح صلاحية تنفيذ ما يلزم عند اكتمال التنزيلات (مع كلمة sudo عند الحاجة القصوى).
- خطة التنفيذ الذاتي: (1) التحقق من كل تنزيل عند الإشعار (أحجام/عدد ملفات/اكتمال المخططات)، (2) T5: اختبار خطوة QLoRA فعلية على `deepseek-coder-6.7b-base` بعد اكتمال نماذجه، (3) T7: فحص مخططات المصادر المكتملة وبناء/تشغيل سكربت التوحيد (JSONL→Parquet)، (4) T8: توليد أزواج asm↔C (x86، gcc موجود حسب الفحص) على عينة ثم دفعة كاملة، (5) تحديث هذا السجل بعد كل خطوة.
- **لن يبدأ التدريب الكامل T9** (الخلطة النهائية) دون موافقة المستخدم.
- مُثبِّط الإسبات مُفعّل (systemd-inhibit، 8 ساعات) كي لا تتوقف التنزيلات.
- **إنذار (إعادة تشغيل الخادم)**: أُلغيت الوظائف الخلفية الأربع أثناء النافذة → تمت الاستئناف فوراً (hf download يستكمل تلقائياً): cvefixes/primevul (أساسي)، deepseek والمتبقي (نماذج)، opencodeinstruct والمتبقي (حداثة)، وأعيد تفعيل مانع الإسبات، وأعيد تشغيل فحص المخططات.
- **إعادة تشغيل ثانية للخادم**: أُلغيت أيضاً محاولة الاستئناف عبر الوظائف الخلفية → تحوّلنا إلى **تشغيل منفصل** (setsid/nohup) لا يتأثر بإعادة تشغيل الخادم: سكربت موحّد `scripts/download_all.sh` (أساسي+نماذج+حداثة؛ سجل: `logs/download_all.log`؛ علامة اكتمال: `logs/download_all.DONE`) + مانع إسبات منفصل + مراقب (إشعار عند الاكتمال).
- **إعادة تشغيل ثالثة + السبب الجذري المُرَجَّح**: تكرر 3/3 مرات بالتزامن مع تشغيل `inspect_schemas.py` القديم الذي كان يحمّل ملفات ضخمة كاملة للذاكرة (json.load لـ 256MB + pandas.read_parquet لملفات 1–2GB) → ضغط ذاكرة أسقط الخادم. الإصلاحات: (1) سكربت فحص آمن الذاكرة (pyarrow iter_batches لسجل واحد، تخطي قراءة row group > 300MB، رأس نصي للملفات الكبيرة، حد 4MB لسطور jsonl)، (2) أداة إطلاق `scripts/launch_downloads.sh` بقفل flock (منع التكرار — سبق أن فشل الفحص الذاتي pgrep مرتين)، (3) التنزيلات ستُطلق مفصولة عن الخادم وتُوثّق في `logs/download_all.log`.
- **OOM مؤكَّد بالدليل (kernel log)**: `python` PID 24731 قُتل بـ ~10.5GB RSS — السكربت القديم هو الرصاصة. التعزيز النهائي: كل التشغيل الطويل الآن عبر **وحدات systemd مستخدم مؤقتة** (`systemd-run --user`): `vulnre-downloads` (التنزيلات) + `vulnre-inhibit` (مانع الإسبات) — تعيش خارج نطاق تطبيق OpenCode فلا تقتلها إعادات تشغيله؛ الفحص الذكي أُعيد بنسخة آمنة (سقف 8GB) لإكمال T7.

## T7 — بايبلاين توحيد البيانات (v1) — 2026-10-16
- `scripts/inspect_schemas.py` (آمن الذاكرة): فحص 16 مصدراً مكتملاً — المخططات الرئيسية:
  - أدوات: ToolACE 11.3k {system,conversations[from,value]}؛ ToolMind {conversations,tools}+7 ملفات query؛ Nemotron-Terminal parquet {conversations,agent,task,model}؛ Nemotron-Agentic {messages,tools,metadata}؛ SWE-rebench {trajectory,tools,resolved,repo}؛ Agent-FLAN (7 ملفات {conversation})؛ xlam {messages}؛ Glaive {system,chat نصي}؛ Mind2Web {website,confirmed_task,action_reprs}.
  - أمني: BigVul {func_before,func_after,"CWE ID",lang,project,commit_id,…}؛ DiverseVul {func,target,cwe,project,…}؛ Fenrir {system,user,assistant}.
- `scripts/build_dataset.py` v1: بثّي بالكامل، سجل قانوني `{id,source,task,lang,messages[,tools,meta]}`، إزالة تكرار blake2b، سقف 250K حرف/سجل (تقطيع النوافذ في T9)، شرائح JSONL كل ~192MB.
  - مهام مولَّدة: `vuln_detect`/`vuln_fix` (BigVul قبل→بعد)، `vuln_detect` (DiverseVul)، `cyber_qa` (Fenrir)، `tool_calling` (Nemotron/ToolACE/ToolMind/xlam/Glaive)، `agent_terminal` (Nemotron-Terminal + عيّنة 1/8 للرياضيات + skill_based)، `swe_agent` (SWE-rebench المنجحة فقط)، `agent_react` (Agent-FLAN).
- اختبار الدخان: 16 مصدراً × 300 سجل ✓؛ صفر إسقاطات طويلة بعد رفع السقف؛ كشف التكرار فعّال (145 تكراراً في Agent-FLAN).
- **التشغيل الكامل الآن**: `vulnre-build-sec` + `vulnre-build-tools` (وحدات systemd) → `data/processed/{security,tools}/`، سجلات `logs/build_*.log`.
- ✅ T7 rimanente — OpenCodeInstruct مكتمل: 4,976,549 سجل في `data/processed/modern/` (11GB، 57 shard). skeema: task=code_gen، lang=text، meta={domain, generation_algorithm}.
- ✅ التحقق من "0 long records": عدد فعال — max 16,303 chars (~2K توكن)، لا توجد سجلات >250K chars. منطق الفحص في build_dataset.py يعمل (تحقق بحقن يدوي 300K chars).
- ✅ T8 (أزواج Assembly) — REx86 مكتمل:
  - ✅ REx86 O1 و O2: تم التحميل والمعالجة — 362,145 سجل في data/processed/t8_asm_pairs/ (770,477 مكرر 제거، 0 فشل gcc، 0 طويل، 0 فارغ).
  - 🔄 AnghaBench: جاري التحميل من GitHub (مليون ملف C، يستغرق وقتاً) — يمكن إضافته لاحقاً كمصدر إضافي.
  - السكريبت: scripts/t8_gen_asm_pairs.py (يدعم --source rex86/anghabench/all، --limit للاختبار).
  - البوابة لسا سارية: لا T9 بدون موافقة المستخدم الصrique.
- البوابة الصارمة: T9 معطل إلا بموافقة المستخدم الصريحة.
- البوابة الصارمة: T9 معطل إلا بموافقة المستخدم الصريحة.
- **مكتمل الآن**: الشريحة الأمنية انتهت (fenrir 99,870 + bigvul 330,865 + diversevul 330,110 = **760,845 سجلاً**، 1.7GB خرج)؛ شريحة الأدوات مستمرة (تجاوزت 250K سجل في nemotron_interactive)؛ CVEfixes أكبر من المتوقع (59.1GB إجمالاً: test 17.9 + validation 12.5 + training 16 + zip 12.7 → سنحذف الـzip بعد اكتماله لتوفير المساحة).
- **R10 (مسح المنافسين/الأدبيات آخر 6 أشهر)** — التفاصيل في PLAN.md: Cisco Antares (350M/1B، يوليو 2026) يصدّق أطروحة «النموذج الأمني الصغير المحلي»؛ فجوتنا المحمية: RE منخفض المستوى + firmware/IoT + أدوات موسّعة + ضغط ~1GB. قرارات: تنزيل Antares-350M/1B للمقارنة (جارٍ)، إضافة FAITH + IoTVulBench + تقييم زمني نظيف لـ T10، CLI agent harness كجزء من التسليم، ومتابعة data مرشحة (patchwork-cve).
- **2026-10-16 (مساءً)**: ✅ حساب HF (HAMZAHshAL) قبل تراخيص Antares + توكن `hf_` رسمي مخزّن عبر `hf auth login` (موثّق: whoami + resolve للنموذجين يعملان). 🚫 طريقة الكوكي المؤقتة أُلغيت وحُذف ملف الكوكي (.secrets/). ✅ اكتملت جميع التنزيلات: antares-350m (680MB)، antares-1b (3.5GB)، deepseek-coder-6.7b-base (31GB)، llm4decompile-6.7b-v1.5 (14GB)، littlebit-qwen3-8b (2.8GB)، OpenCodeInstruct (6.4GB)، PrimeVul (630MB)، refined+oss-instruct+html-css-js-cot. إجمالي المحمل ~62GB البيانات + ~52GB النماذج.
- **حالة T7**: security مكتمل (760,845 سجل، 1.7GB)، tools مكتمل (~1.75M سجل، 13GB).
- **✅ T5 اكتمل**: اختبار دخان QLoRA على `deepseek-coder-6.7b-base` نجح — 2 خطوة تدريب، 4.12GB GPU ذروة استهلاك، لا OOM. السكربت: `scripts/t5_qlora_smoke.py`. النموذج جاهز للتدريب الكامل T9.
- **بعد T5**: T7 rimanente (تحويل شريحة الحداثة OpenCodeInstruct → Parquet)، T8 (أزواج Assembly gcc -O0/-O2 -S)، T9 (تدريب QLoRA كامل — ينتظر موافقة المستخدم).
- **ملف التوكن**: `.secrets/hf_token` (37 بايت، chmod 600). التوكن الرسمي في `~/.cache/huggingface/token`. يُنصح بتدوير التوكن بعد الانتهاء من جميع التنزيلات.

## 2026-10-17 (تجهيز كامل للخلط النهائي)
- **إكمال المصادر الجديدة** (شرائح إضافية لا تُلمس القديمة — لمنع كتابة فوق الشظايا الموجودة):
  - `tools-new`: mind2web **8,784** (خطة كاملة بلا HTML + سجل لكل خطوة) + playwright_mcp **3,869** (v3+v4+parquet؛ إصلاح رسائل مصفوفات نصوص JSON-مشفرة) → tools الإجمالي **1,790,838 سجل / 40.6GB**.
  - `modern-new`: openrefined **441,645** (فلتر جودة average_test_score≥0.9) + oss_instruct **50,661** + html_cot **2,645** (وسوم `<thinking>` الزائفة → `Reasoning:` طبيعي حسب R10) → modern الإجمالي **5,471,500**.
  - `security-new`: **PrimeVul 199,656** (تم تنزيل ملفات التسميات المفقودة: train/valid/test jsonl ~575MB عبر `fetch_primevul_labels.py` — سابقاً نزّل file_contents فقط) → security الإجمالي **960,501**.
  - إصلاح `build_dataset.py`: `--out` الافتراضي أصبح مجلد الشريحة تلقائياً (كان يكتب في جذر processed)؛ السجلات الخاطئة نُقلت `mv` إلى مجلداتها ودُمجت التلخيصات.
- **المصغّر الذاتي ✅**: `scripts/t7_micro_self.py` → **651 سجلاً** نظيفاً (8 مهام: project_pipeline/qlora/littlebit/re_knowledge/re_toolchain/vuln_knowledge/firmware_knowledge/tool_format؛ 3 صياغات لكل حقيقة؛ فحص خربشات CJK/Cyrillic = 0) في `data/processed/micro/`.
- **AnghaBench ✅**: الرابط الصحيح `brenocfg/AnghaBench` (المستودع `anghabench/anghabench` غير موجود — سبب فشل أول مستنسخ)؛ اكتمل الاستنساخ 4.8GB بوحدة systemd → **1,044,021 ملف C** مستقل.
- **إصلاحات T8 الحاسمة لـ gcc16**: (1) فشل 200/200 في أول دخان؛ السبب: gcc 16 يعتمد `-std=c23` فتصبح `typedef int bool` خطأ → **`-std=gnu89 -w`**، (2) الملفات تحمل علامات `/*<<< orphan*/` (بدائل أنواع من استخراج AnghaBench) → إزالة حرفية قبل الترجمة والتسجيل، (3) prelude بلا `#include` (typedefs فقط) لأن الملفات تعرّف printf بنفسها، (4) `conv_anghabench` أعيد كتابتها: `os.walk` بثّي + **multiprocessing** (T8_WORKERS قابل للضبط، افتراضي 11) + `emit` يعيد True.
  - دخان: 3,000/3,000 سجل، 6 dups، gcc_fail=64 فقط (~96% نجاح ترجمة).
  - 🔄 **التشغيل الكامل**: `vulnre-t8-anghabench` (هدف 400K سجل ≈ 200K ملف)؛ أُطلق 4 عمال ثم رفعه لـ **12 عامل** بموافقة المستخدم الصريحة على استهلاك كامل المعالج (سجل: `logs/t8_anghabench.log`).
  - ✅ **اكتمل (2026-10-17 01:47)**: **400,000 سجل** (14,340 مكرر مُزال، gcc_fail=1,577 من ~212K ملف ≈ 99.3% نجاح، 0 طويل/فارغ). `t8_asm_pairs` الآن **762,145 سجلاً / 2.8GB** (REx86 362,145 + AnghaBench 400,000).
- **t9_final_mix.py أعيد كتابته (بثّي + مخطط Arrow صريح)**: sampling بقناع hash بلا RAM (ملف tmp لكل شريحة) → دمج+خلط ببوفر 64MB → Parquet بكاتب واحد، الحقول الكائنية (messages/tools/meta) مخزّنة JSON strings (فشل سابق: `cannot mix struct and non-struct`).
  - دخان ناجح: `train_final.parquet` 5,953 سجلًا بمخطط نظيف؛ إعادة توزيع رصيد datasheets (0 على 5%) على security/tools/modern بنسبها نفسها؛ micro يغطي معرفة المشروع.
  - **النسب النهائية على هدف 1M**: security_re 400K (security+t8) / tools 350K / modern 200K / micro 651 + فائض الـ5% موزع بنسب 40/35/20 على الثلاث الأخرى.
- **الخطوة التالية**: عند اكتمال AnghaBench → تشغيل الخلط النهائي الكامل (FINAL_MIX_TARGET=1000000، وحدة systemd، ~20-40 دقيقة) ثم استشارة المستخدم لإطلاق تدريب T9 (بوابة صارمة).
- **✅ الخلط النهائي الكامل اكتمل (01:52)**: `train_final.parquet` = **999,908 سجلاً / ~12.6GB** — security_re **420,687** / tools **368,181** / modern **210,389** / micro **651** (dups/long/empty inspected: 92 سجلاً نقصاً فقط عبر الأقنعة). ~3.4B توكن تقديرياً → التدريب الكامل 100% غير عملي على 3060 (~2-3 أسابيع) → **قرار موازِن في T9**: تمريرات محدودة بالتوكن (token budget) بنسبة 40/35/20/5 محفوظة بالخلط المسبق. `t9_qlora_train.py` جاهز ومدقّق قالبياً (dry-run على الدخام: 1,858/2,000 سجل قابلياً).
- **موازنة T9 المقترحة (انتظر موافقة المستخدم)**: R1 = ~10-14 ساعة / ~55M توكن (زيجة r16، packing 4096، paged AdamW، bf16... ذروة VRAM متوقعة ~7-8GB) ثم رحلة تقييم سريعة T10-lite قبل الإطالة إلى مزيد بـ LoRA refresh.

## 2026-10-17 — البوابة الجديدة: T9 pilot محدود (موافقة المستخدم بنص الإلزام)
- موافقته: تشغيل **pilot ~50 خطوة على البيانات الحقيقية** (8K seq / LoRA r16) ثم تشغيل فعلي ~6 ساعات بحد صارم (max_steps محسوب من زمن الخطوة المقاس، warmup 3% + cosine→0، checkpoint كل 250 بخلق آخر 3، سجل خسارة منفصل، systemd RuntimeMaxSec دفاع ثانٍ). بعد التشغيل: **لا تدريب إضافي ولا T10 تلقائي** — checkpoint مُقيَّم ينتظر مراجعته؛ أي تدريب إضافي يتطلب موافقة جديدة.
- **محاولات البايلوت الفاشلة (أخطاء إعداد API فقط — ليست فشل تدريبي):**
  1. `TypeError: save_safetensors` — وسيط أُزيل في transformers 5.17 → حذفه (safetensors هو الافتراضي).
  2. `TypeError: NoneType > int` في `_validate_args` — تمرير `num_train_epochs=None` مع max_steps يكسر مقارنة التحقق → تبقيها رقمية (1.0) و max_steps يتجاوزها.
- **البايلوت (إعادة إطلاق ثالثة) تعمل الآن**: `vulnre-t9-pilot` (RuntimeMaxSec=3600s) — max_steps=50, save كل 25 (لاختبار حفظ checkpoint حرفياً), grad_accum=8 → نظرياً 65,536 توكن/خطوة؛ عينة بيانات أول 120K سجل من الخليط الحقيقي (999,908).

- ❌ **ثالثة محاولة pilot فشلت OOM (القرار: وقف وسجل — لا استمرار حسب نص الموافقة)**:
  - الوضع:_Message `torch.OutOfMemoryError` عند أول خطوة (0/50) رacebook على alloc 342-358MB بينما المتكلفة الكلية 10.78GB لبروسيس التدريب (أصل بطاقة 11.68GiB، منشغلة من تطبيقات سطح المكتب ~456MB إضافى).
  - الخلاصة الحاسمة عند 死: `1.83GiB reserved-but-unallocated` → التشظية (fragmentation) هي المسببة المؤكدة، رسالة نفسها توصي `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - لم تُتسجل أي خطوة خسارة ولا أي checkpoint — لم يبدأ عدد تحديث فعلي.
  - المسببات المرشحة بترتيب تحريكها: (1) 8192 توكن/سلسل = ذروة تنشيط+logits نوعاً ما مرتفع (8192×32022 fp32 ≈ 1.05GB في سقف)، (2) ضغط تطبيقات سطح المكتب على الVRAM، (3) LoRA r16&utm؛ كل الإسقاطات (39.97M بارامتر تدريب) — قابل ونهاط خفфорq16، (4) grad_accum=8 على نفس الخطوة.
  - الأموية العيّن+8192+accum8+full-targets على 3060: **impossible حافة**؛ محتاج قرار الصيروري تحت المراجعة قبل إكاكل أي محاولة (expandable_segments أو seq 6144/4096 أو LoRA أضيق). لا تدريب حتى موافقة جديدة.

## 2026-10-17 — دورة ما بعد OOM: شبكة أمان + بينيلوت A (تعليمات المستخدم)
- ✅ **git init** في vulnre/ (commit ‏972c258 = حالة كاملة قبل البايلوت؛ data/models/logs/third_party خارج التتبع). سبب: 3 انهيارات سابقة بلا شبكة أمان كان خطأ.
- ✅ تصحيح مسجّل (ملاحظة المستخدم): ربط alpha=2r بفشل lora_gate مبالغة — alpha=2r إعداد شائع ناجح؛ سبب lora_gate الأرجح **غياب eval/early-stopping**. ننزل alpha=r كاحتياط رخيص فقط، لا كحل جوهري.
- ✅ دمج آمنان في البايلوت: `lora_alpha=r` + `weight_decay=0.01`.
- ✅ **قرار البايلوت: A فقط لا D** — متغير واحد (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` عبر بيئة الوحدة) لتحديد السبب الحقيقي (تشظيّة الذاكرة على 8K عبر 50 خطوة، لا حظ السياق).
- 🟢 **البايلوت A يعمل الآن** (`vulnre-t9-pilot-a`، RuntimeMaxSec=3600): 50 خطوة، نفس 8K@r16@accum8، حفظ كل 25، سجل منفصل `logs/t9_pilot_a.log` + `loss_log.jsonl`، معاداة to 8K مع تعديلي alpha/wd (كلاهما آمن).
- ⏳ **مؤجّل (بوابة T9 الكامل — قبل أي موافقة تدريب كامل):**
  1. eval split + harness شامل (غاية: منع نمط lora_gate: train ممتاز/test عشوائي) — **الأخطر**.
  2. assistant-only loss (بدل تدريب على توكنات المستخدم).
  3. خفض وزن micro_self إلى 1-2% (من 651 سجل حالياً — وزنه في الخليط مرتفع نسبياً).
  4. فحص/استبعاد بيانات GPL-3 (html-css-js-cot) من الخليط.
  5. قرار ميزانية: 55M توكن = 1.6% فقط من خليط 1M سجل مع 4 مهام متنوعة → إما تضييق النطاق (مهمة/مهمتين) أو زيادة الميزانية الزمنية — قرار المستخدم.

- ❌ **البايلوت A (expandable_segments:True) — فشل بنفس OOM لكن بجواب علمي قاطع**:
  - التشظية عولجت فعلاً: reserved-unallocated نزل من 1.83GB → **300MB** (الإعداد أثبت نجاحه).
  - ومع ذلك الموت صار: PyTorch مخصص **10.39GB** حقيقي + طلب 342MB آخران → **الذرة الأساسية لا تتسع**: 8K seq + LoRA r16 على كل الإسقاطات (bf16 activations+logits@8K) أثقل من 11.68GB بذاتها، بلا تشظية.
  - **استنتاج المتغير الواحد**: الحاجز ليس التعريفا بل حظ الذاكرة نفسه → الحل الوحيد المتبقي: تقليص seq (B: 6144/4096) أو تقليص أهداف LoRA (C: q,v) — العقد المتغير القادم يعتمد موافقة المختار.
  - حمل الرامات أثناء التشغيل (ملاحظة المستخدم): عادي المتوقع — تحميل النموذج يمر عبر الصفحة الرئيسية (31GB shards → تحويل NF4 ثم نسخ 3.6GB للGPU) + خريطة التوكنايزة (8 لغسال) + paged optimizer (صفحات حالات آدم بالرام CPU) + page cache للparquet (9.3GB available قابلة للتجديد — ليست ضغطاً). التدريب نفسه يجري فوق الGPU (10.83GB proceso مباشرةً).

## 2026-10-17 — تصحيح منهجي من المستخدم: البايلوت A كان غيّر متغيرين لا واحد
- ملاحظته الصائبة: القياس الأصلي (6.72GB @8K) تم بـ **q,v فقط** (~11.5M بارامتر تدريب) بينما فيلميتنا واحدة على logvaat/البايلوت السابقين: **كل الإسقاطات السبعة** (~40M) → المقارنة ضمنية بأكثر من متغير (ALLOC_CONF + نطاق LoRA) → قراراتها الفوقية (B/C) انحرفت.
- الحساب: gradients+cases آدم+ constituيت تدريبية حفز LoRA retention عملياً تفصل فرق 6.72→10.39GB المعِقد أضيق من «السياق نفسه».
- ✅ تنفيذ: `--lora-targets {all,qv}` في t9_qlora_train.py — **qv = إعداد القياس الأصلي تماماً** (q_proj,v_proj فقط) — commit مستقل، كلا المسارين محفوظان.
- 🟢 **بايلوت A-qv يشتغل الآن** (`vulnre-t9-pilot-aqv`): 50 خطوة @8K + **q,v فقط** + expandable_segments — *متغير فرعي واحد* مقidine إلى القياس الأصلي: إن مرّ؛ العائق كان منطاب LoRA ر15 الحالي (الحل: نثبت كذاك أو hankingڢٍ نزول seq بند الوضع qv)، إن فشل — السياق 8K حتى بالصپي قيسّ و لا يتسع؛ الحل: seq 6144/4096.

- ❌ **البايلوت A-qv (q,v فقط = إعداد القياس الأصلي + expandable_segments) — فشل بنفس OOM**:
  - نفس النقطة حرفياً (`bnb.matmul_4bit`، طلب 342MB، متاح 208MB)؛ PyTorch مخصص **10.33GB**.
  - **الخلاصة المقارنة العادلة (المتغير الوحيد = النطاق):** all-7 (10.39GB) مقابل q,v (10.33GB) → **فرق 60MB فقط** → نطاق LoRA ليس العنصر المسبب مطلقاً؛ العائلة الحاملة هي مسار SFT الكامل عند 8K (logits 8192×32022 + CE + backward + paged page-in).
- 🔍 **تشريح سكربت القياس الأصلي (t5_qlora_context_final.py) — لمَ مبالغته:**
  1. **لا backward() ولا optimizer ولا zero_grad في الملف بأكمله** (grep حرفي = صفر) → قاس **forward فقط** + empty_cache بين السياقات.
  2. r=32 مو r16 (وأيضاً forward-only يجعل الحالات/unloaded optimizer غير موجودة أصلاً).
  3. لا مسار SFTTrainer: لا CE نسبة لمخزن logits الكامل (8192×32022 ≈ 0.5GB bf16 + نسخ fp32 ≈ 1-2GB + تدرجات logits للخلف) ولا packing.
  → «6.72GB» كانت ذروة **تخطي** لا ذروة تدريب؛ كل بايلوت لاحق = أول اختبار حقيقي.
- 📉 **العقل المتبقي الوحيد (بحساب تقديري):** خطوة الاحتداد + فقي logits/CE تتناسب خطياً مع seq؛ المتوقع: allocated ~7.5-8GB عند 4096 (هامش ~3GB)، ~9.2-9.6GB عند 6144 (ضيق).

## 2026-10-17 — موافقة B1 (المستخدم، بنص شرطين)
- ✅ موافقته: **B1 = seq 4096 فقط** — بشرط سابق للتشغيل: السكربت يحتوي backward + optimizer.step فعلياً (SFTTrainer كامل، ليس forward-only كخطأ القياس الأصلي) — **تحقق grep: SFTTrainer + trainer.train() ✓** وأبلاغُهُُ.
- ✅ بنيته كـ**متغير واحد** على A-qv (سلسلة العدل): نفس qv + expandable_segments + 50 خطوة + save كل 25 + accum 8 — **الفرق الوحيد: 8192→4096**.
- 🟢  `vulnre-t9-pilot-b1` يشتغل (RuntimeMaxSec=3600). تقدير ذروة allocated ~7.5-8.5GB (هامش ~3GB).
- ⛔ **شرط التوقف المطلق (نص موافقته): لو فشل 4096 → وقف تام، لا أي تجربة أخرى، والرجوع له لمراجعة الأساس** (r أقل، أو batch/accum، أو تأجيل). هذا أصبح قاعدة البوابة.

## 2026-10-17 — البايلوت B1: نجاح كامل (الشرط الثلاثة) — والتقييم الفعلي
- ✅ **لا OOM @4096**: 9.16-9.18GB من 11.68 مستعمل، ≥28 خطوة تدريباً حقيقياً،杀死ُ المرحَل في 60 دقيقة بالسقف الذاتي (RuntimeMaxSec، كما صمّم) — لمسنا صفر خطوات موتية.
- ✅ **خسارة منفصلة**: نقطة 15 = **1.604** | grad_norm ‏0.149 | دقة توكن 68.4% | entropy 1.40 | 487K توكن مقنّن — مقيّد معقول ونظيف؛ السلسلة الثانية (30) مات قبلها بقسمتها إلى 60 دقيقة بعيدَ الإلحاق من 2 دقيقة/خطوة.
- ✅ **checkpoint-25**: حفظ كامل (83MB: adapter+optimizer+scheduler+trainer_state) **ومُحمّل بعناية** — `t9_verify_ckpt.py` **PASSED**: تحميل جديده قيادات NF4 + تلييد قارئ «buffer overflow» في جهاز، معرف حساب strcpy > 32.
- ⚠️ تحاية سجل التقييم فقط **نقطة دانة واحدة** (logging من 15 خطوة) — اهتم المنحنى الكامل يحصل في الفعلي.
- ⚠️ تحذير TRL المستمر (packing + sdpa = cross-contamination محتمل) — بنيه سجل للبوابة المؤجلة (يحل باهث إله flash-attn-2 أو packing=False).
- 📉 **التصغير الزمني المقاس (القياس الحل):** ثابت 121-120.8 s/خطوة → **~272 توكن/ث** → نافذة 6 ساعات ≈ 163 خطوة ≈ **5.3M توكن** (وليس 55M؛ الفرق ~10x). 55M يحتاج ~55 ساعة.
- 🧭 **قائمة القرار التالية (كل ما بعد بوابة موافقة جديدة):** قبول 5.3M+‌ضيّق المهمتين / زيةń النافذة / تثبيت flash-attn-2 ثم إعادة الاندماج pilot (سرعة + نقاء ماسکرد) / تأجيل كامل.

## 2026-10-17 — تجريب flash-attn-2 + torch.compile (موافقة المستخدم بنص قياس)
- تعليماته: (1) ثبّت flash-attn-2 مع محاولة توافق torch2.14/cu130، (2) جرّب torch.compile إن كان سهلاً، (3) **أعد نفس البايلوت 28-30 خطوة وقارن tok/s فقط**، (4) كل ما четر < ‎1.3x → توقف عن مطاردة السرعة وانتقل لقرار تضييق النطاق (مهمتين).
- التنفيذ: `pip install kernels` أُضيف (kernels 0.17.2 عبر uv — pip غير موجود داخل .venv). الحصول على flash-attn الرسمي مستحيل التوافق (prebuilts لا وجود لtorch2.14/cu130؛ nvcc غير متوفر لبناء المصدر) → الحيلة الحديثة: `attn_implementation="kernels-community/flash-attn2"` (كما أبلغت رسالة TRL نفسها بالمتكفيل-support).
- سكربت: `--attn` + `--torch-compile` أضيفا (commit مستقل). baseline سصق تمثيل: **122 s/خطوة (2/دق) = 268.6 tok/s ideiasي** (من دليلة الجدञ bar: الخطوة 12→28 في 1952s/16).
- 🟢 بايلوت F1 يعمل الآن `vulnre-t9-fa2`: **متغير واحد = attn فقط** — نفس كل بتّ: qv + 4096 + 30 خطوة + accum8 + segs، (RuntimeMaxSec=7200 احتياطاً أوسع لأن القتال 3600 كان يقدّ في القياس).
- مصير النتيجة تقياسها من فقد wall بين صفوف 15و30 في loss_log (إبعاد الوقت الاستمراري) = steady s/step بللمكوين القيسي.

## 2026-10-17 — موافقة جديدة: قلب المحرك إلى unsloth (أقوى تغيير متاح، وتعليمة المستخدم)
- تعليماته: أوقف ملاحقة flash-attn-2/torch.compile → حوّل t9_qlora_train.py إلى **unsloth.FastLanguageModel + use_gradient_checkpointing="unsloth"** (نفس مسار t5_context_unsloth الناجح) → إعادة نفس البايلوت (qv+4096، 30 خطوة) وقارن VRAM وtok/s. توقعه: **~2x** سرعة + هامش ذاكرة يفتح احتمال 8K.
- ✅ التنفيذ (commit ‏«24+/26−»):
  - `import unsloth` **أول الاستيرادات** (قبل transformers/trl — شرط الpatching).
  - FastLanguageModel.from_pretrained(max_seq_length=max_length, dtype=None, load_in_4bit=True) + get_peft_model(r16, alpha=r, dropout=0, use_gradient_checkpointing="unsloth", qv/all حسب --lora-targets).
  - **إزالة** gradient_checkpointing من SFTConfig (تطبيق مزدوج يضر)، إزالة AutoTokenizer/peft/typeof HF loader القديم (unsloth يعيد tokenizer). unsloth 2026.10.6 موجود بالبيئة منذ البداية — هذا أغفلنا أنه مثبت وسليم لبيئتنا.
  - dry-run نجاح (probes نفس القالب).
- ⏹ F1 (kernels flash-attn2) **وُقف بتعليمه** (كان will dead أصلاً — kernels لم توافق ABI torch2.14/cu130 على الأرجح).
- 🟢 بايلوت **T9-uns1 يعمل** (`vulnre-t9-uns1`, 7200s): متغير واحد = محرّك التنفيذ (HF/peft → unsloth). كل البتات مطابقة B1: qv+4096+accum8+120K سجل+segs+paged32.
- 🖇 وسيلлетا القياس الموحدة: s/step من فارق wall بين صفوف 15و30؛ VRAM من nvidia-smi أثناء الخطوات. **Baseline: 122 s/خطوة = 268.6 tok/s مع 9.18GB.**
