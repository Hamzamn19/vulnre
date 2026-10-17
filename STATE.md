# STATE — دليل استئناف مشروع vulnre (يُحدَّث مع كل دورة)

_آخر تحديث: 2026-10-17 (بعد بداية بايلوت uns8k). الاقتران: progress.md (سجل يومي تفصيلي) + PLAN.md (الخطة الأصلية R/T)._
_هذا الملف = «الكل وصلنا له وما قمنا به وما نحاول القيام به» لبدء محادثة جديدة والاستئناف من النقطة الحالية._

---

## 0) المشروع في سطر
نموذج متخصص في **الأمن منخفض المستوى / RE / Firmware** عبر **fine-tune بالحزمة** على بطاقة **RTX 3060 12GB**: أساس `deepseek-coder-6.7b-base` بـQLoRA (NF4) عبر **unsloth**، على خليط بيانات نهائية `data/processed/final_mix/train_final.parquet` (999,908 سجل / ~12.6GB).

## 1) البنية والمخطط
- مسار: `vulnre/` (الجذر، git repo منذ 2026-10-17) — commits كاملة من ‏`972c258` (قبل البايلوت A) إلى ‏`cb9b196` (eval stratified).
- بيئة: `.venv/` (uv بدون pip داخلها — استدعِ تثبيت بحزم بـ`uv pip install --python .venv/bin/python`). mtfers 5.17 / trl 1.15 / peft 0.21.2 / torch ‏2.14.1+cu130 / bnb 0.50.2 / **unsloth 2026.10.6** (كان مثبتاً من قبل).
- systemd-run لكل شغل طويل: `vulnre-t9-*` — `--collect --property=RuntimeMaxSec=… --property=TimeoutStopSec=600` + `PX‑envs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- ملفات رئيسية:
  - `scripts/t9_qlora_train.py` — سكربت التدريب الحيّ (unsloth + eval + سجل حي)
  - `scripts/t9_verify_ckpt.py` — فحص تحميل checkpoint + توليد (PASSED مسبقاً)
  - `scripts/run_logged.sh` — تشغيل موسوع logs
  - `scripts/t9_final_mix.py / t7_micro_self.py / t8_gen_asm_pairs.py` — خط أنابيب البيانات (مكتمل)
  - `logs/*.log` — كل بايلوت سجل عَلَن
  - `data/processed/final_mix/train_final.parquet` — الخليط (لا تدخل في git؛ 195G data مُستثناة من التتبع)

## 2) ما قمنا به (بترتيب الزمن الحيّ)

### 💉 المرحلة 1 — تجهيز البيانات (مكتمل 100% حتى train_final)
- **t7**: سلاسل الأمان/أدوات/الحديث (~960K/1.79M/5.47M سجلاً مؤهلة) مع تدفقات بياض ذاكرة، وسلاسل عيّنات للأدوات الجديدة (mind2web/playwright/oss_instruct).
- **t8 + AnghaBench**: 400,000 سجل asm-pairs جديد (المجموع 762,145). إصلاحات حاسمة: `--std=gnu89 -w`، إزالة `/*<<<o rphan>>>*/`، prelude typedefs فقط، بثّي 11+ general worker.
- **micro_self**: 651 سجل أدق من الحقائق الموثوقة بدورات جرعات صياغة.
- **t9_final_mix**: خليط نهائي **999,908** سجلاً بنِسَب 40/35/20/5 (security_re 420,687 / tools 368,181 / modern 210,389 / micro 651) — packing 999,899 قابل في دقائق.

### 🧔 المرحلة 2 — تشريح الذاكرة (الأرقام الحاكمة الثمانية)
| المرحلة | النتيجة | الدرس |
|---|---|---|
| القياس الأصلي t5_context (6.72GB@8K) | — | كان **forward-only** (لا backward/optimizer) — كل القياسات الحقيقية بالبايلوطة |
| A (8K, كل الإسقاطات r16) | OOM عندما: 342MB خالياً مقابل 10.39GB مخصصاً | التشظية |1.83GB محجوز-غير مستخدم؛ بينما الحاجز أعمق |
| A+expandable_segments | **التشظية نزلت 1.83→300MB لكن الموت بقى** | الحاجز ذروة حقيقية لا تعريفاً |
| A-qv (q,v فقط = إعداد القياس الأصلي) | 10.33GB (فرق 60MB فقط عن all-7) | **نطاق LoRA ليس العنقي** — الحائمة في مسار 8K+SFT |
| B1 (4096, q+v, expandable_segments) | **نجاح**: 9.18GB@100%، حتى الخطوة 28/50 ثم قتل بسقف 3600s الذاتي | 4096 يعمل على 11.68GB؛ الخسارة ‏1.604 سجّلة grad_norm ‏0.149 نظيف؛ checkpoint-25 حُفظ ومُحمّل وولَを見 مختار (verify PASS) |
| uns1 (unsloth @4096) | **61 s/خطوة = 537 tok/s = 2.00x بالحرب + ‏5.95GB (توفير 3.2GB)** | unsloth أقوى تغيير متاح |
| uns8k (قيد الشغل الآن) | جارٍ — يجب إثبات الذاكرة والسرعة عند 8K + عمل eval | يعمل حالياً `vulnre-t9-uns8k` |

### 🌐 المرحلة 3 — بناء التجريب الحي (commit الأكثر حلة)
`scripts/t9_qlora_train.py` الآن:
- **unsloth**: `FastLanguageModel(from_pretrained(load_in_4bit=True, max_seq_length)) + get_peft_model(use_gradient_checkpointing="unsloth", dropout=0, alpha=r)` + إزالة قتل HF-ckpt من SFTConfig (مرتين يضر).
- **eval lora_gate بوابة**: `load_mix` يعيد `(train, eval)`، **stratified على source قبل تصيير السعر** (سقف 2000 سجل — لتحمّل eval pass بدقائق لا ساعات؛ الحسبة 7% لتحطيم كلف قياس البايلوت pass)، SFTConfig: `eval_strategy=steps / eval_steps=100 / metric_for_best_model=eval_loss / greater_is_better=False / load_best_model_at_end=True / per_device_eval_batch_size=1 / EarlyStoppingCallback(patience=3)`. **شرط alignment الثابت: save_steps=eval_steps (default 100)**.
- **LiveLossLogger**: `[timing] steps X→Y: Zs/step, N tok/s` (وقتاي مباشر بشكل لا يتجاوز 8-10 خطوات) + JSONL حي (train/eval/ref فرفعة) + `[eval@S] eval_loss`.
- вес_decay=0.01، lora r16/alpha=r/dropout=0، packing + expandable_segments env.
- CLI هتمة: `--max-length` (8192 المفوّرة), `--max-steps`, `--token-budget`, `--lora-targets {all,qv}`, `--attn` (غير فعّال في مس unsloth), `--torch-compile` (اختبار رخيص لم يُستكَمل لمس unsloth), `--eval-frac 0.07`, `--max-eval-records 2000`, `--eval-steps 100`, `--es-patience 3`, `--logging-steps`, `--save-steps`, `--dry-run`.

## 3) البوابات وتاريخ صلاحيات المستخدم (لا تُكسر أبداً)
1. **T9 pilot B1** (موافق عليه bcharge لكن بنص صرحة): «mo الموافقات القادمة كل مص المصعد ولمس المستخدم عليه صراحة».
2. **عقد الفشل**: «لو فشل، وقف وسجّل السبب بـ progress.md فقط — ما تكمل» — متشدد على أي pilot (تطبيق مرتين) والإنهاء داكتَب في اللحظة.
3. **قاعدة السرعة (توقّع خطر وأُعمم الموافقة)**: مطاردة flash-attn-2/compile بشر «لو ما تحسنَ فوق 1.3x توقف وانتقل للقرار التالي» — فالسرعة أنجز unsloth ‏2.00x.
4. **unsloth (متحفظ مع 2.00x)**: العامل الذياء الفعلي — ويُسمح به revisit للاتجاه المطلوب حياً.
5. **البوابة الحالية (الحاسمة/معلق)**: **ما يبدأ التشغيل الطويل حتى**:
   - تظهر [eval@5] و [eval@10] وeval_loss לגי على uns8k (إثبات تفريّق تعلّم/حفظ) — **بقية مهمة uns8k**؛
   - تقرير نتيجي (108K VRAM وطوّه)…
   - → مطلوب **موافقة صريحة من المستخدم** لبدء التشغيل الطويل رأساً في كل دورة أحدثة.
6. **منهجية القياس الجديدة (مُثبتة كحكم مُلحّق)**: بايلوت قياس السرعة = `--max-steps 8-10 + --logging-steps 1` (happening بعد 5-8 دقائق لا 45) — الحساب من فرق 3→8.

## 4) الميزانية المعتمدة (تحمل المستخدم نصياً)
- قبل في قصور الحفظ: **200-300M توكن** (الموضع كان 55M) بنِسَب الخلط الطلبية للمزوج (40/35/20/5 — const حصة القوت في الخليط حالياً).
- بحسابية الصيغة المستخدمة: تأقص `max_steps = floor(budget / tok_per_step)` حيث `tok_per_step = max_length × grad_accum` بعد تأكيد uns8k: (تثبث فور تصفّد متي)`. oído: 250M ÷ 65,536 = **3,815 خطوة** عند 8K؛ 250M ÷ 32,768 = 7,630 خطوة عند 4096.
- الزمن التقديري (بشان 537 tok/s): 250M / 537 ≈ **129 ساعة ≈ 5.4 أيام**-مليزة.

## 5) ما نحاول القيام به الآن (خيط حيّ عند هذه النقطة)
🟢 ‏**vulnre-t9-uns8k** قيد العمل الآن (آخر معلومة عند كتابة الإطار: النموذج محمل، VRAM ‏6.86GB قبل بدء الخطوات):
- **يجياجها فاصلًا**: [eval@5] و [eval@10] تُطبع بيق sandbox يانسبة (`[eval] مفعّل كل 5 خطوات (1,995 سجل): best-keep + early-stop patience=3` طبع بالسجل ✓)؛ [timing] ال8K؛ VRAM الذوق؛ وتتواجد steps 10 كاملة.
- **ماذا بعد النجاح**: التقرير المستخدم → الموافقة → تشغيل طويل 200-300M ضمن B1-style constraints:
  - تبقّىباً: `--lora-targets qv --max-length 8192 --grad-accum 8 --max-steps <مفحوص> --eval-steps 100 --save-steps 100 --es-patience 3 --warmup-frac 0.03`
  - RuntimeMaxSec على الميزانية+20%؛ checkpoints كل 100 بخلق آخر 3.
- ❒ فترة تشغيل uns1 (old task) حصة المجال أُخرجها بعد الإرجع — لا با ي旦ت بعده.

## 6) المؤجلات (بوابة T9 الكامل، أرجعها بعد النجاح)
1. **assistant-only loss** (حالي: خسارة كاملة بنسج base) — قرار مستخدم.
2. **خفض micro_self إلى 1-2%** (حالياً 651 سجل بوزن مرضيّ).
3. **فحص/استبعاد GPL-3** (html-css-js-cot من الخليط).
4. **تضييق النطاق مفوض**: مهمتين مقابل كامل (الأولية صقر بنصه «تضييق النطاق أو زيادة الميزانية» — ميزانية رُفعت 200-300M؛ النطاق القرار يبقى معك).
5. **تحذير TRL المتواخي**: packing + أي ب(BytesImpl) sdpa — `cross-contamination` محتمل بين العينات؛ unsloth يدار اختيار attention داخلياً — أتجرب عيّنة نفس التوجيه القبل.
6. — بالإشراق: `--torch-compile` كان اختبار رخيص غير مترقد واقعاً (أوقف بتعتمد unsloth).

## 7) كيف تستأنف محادثة جديدة (وَصِلْ من هنا)
1. `cat vulnre/STATE.md` + last ~80 خطوط من `vulnre/progress.md`.
2. تحقق خيوط الشغل: `systemctl --user list-units 'vulnre-t9-*'` + `nvidia-smi`.
3. إذا uns8k لم يُنسى بعد طلباً من المستخدم (光的 الابن evalrowth) — تتوقع من logs:
   `grep -E "^\[timing|^\[eval" logs/t9_uns8k.log`
4. لأي pilot جديد: نمط الإطلاق الموثق:
   ```
   systemd-run --user --unit=NAME --collect --property=RuntimeMaxSec=X --property=TimeoutStopSec=600 \
     -E "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
     ./scripts/run_logged.sh logs/NS.log .venv/bin/python scripts/t9_qlora_train.py \
     --max-steps 10 --logging-steps 1 --eval-steps 5 --save-steps 5 \
     --max-records 200000 --lora-targets qv --max-length 4096|8192 --output data/processed/NAME
   ```
5. Muzzle harness: أقل من موافقة لا تدريب طويل — وكل شيءukes كما بعت في المرحلة 3/5.

## 8) طريق إث بالسر الفعلي الخيابان (الخريطة المطلوبة بعد muست 8K)
- حساب quick: `max_steps = budget / (max_length × grad_accum)`; وz تحققي أوقات؛ المشوّم بالتواريع: save/الروتانات وكل 100 خطوة… compose والقي (3 أيام ~5.4).
- unit contract: RuntimeMaxSec = الزمن المتوقع +20% (ليلة كاملة ~12h ⇒ 52,000s هي مر عليه).
- الأمان: stop (SIGTERM) يقطع الحفظ التشغيلي؛ 즐ق save كل 100 (نطاق ~100 دقيقة نقطة إتحالات مفقدة بالأسواء).
