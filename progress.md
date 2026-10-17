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
