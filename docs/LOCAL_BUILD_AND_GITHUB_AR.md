# ترتيب بناء الورقة ورفع المستودع

## البنية المعتمدة

```text
PACBayes_Sparse_TSK_V4_GitHub_Ready/
├── src/                         كود بايثون الأساسي
├── scripts/                     تشغيل التجارب والتحقق
├── configs/                     الإعدادات المجمدة
├── tests/                       اختبارات V3
├── results/                     النتائج المسموح برفعها
├── artifacts/                   الأقفال والبصمات وسجلات التدقيق
├── paper/
│   ├── main.tex                 ملف الورقة الرئيسي
│   ├── refs.bib                 المراجع
│   ├── sections/                أقسام الورقة بالترتيب
│   ├── tables/                  جداول LaTeX المولدة
│   ├── figures/                 الصور المولدة
│   ├── source_data/             ملخصات CSV التي تولد الجداول والصور
│   └── PACBayes_TSK_Manuscript_V4.pdf
└── tools/                       أوامر Windows الجاهزة
```

## الترتيب الصحيح

1. `tools\00_verify_prerequisites.cmd`
2. `tools\01_setup_environment.cmd`
3. `tools\02_generate_manuscript_assets.cmd`
4. `tools\03_validate_code_and_results.cmd`
5. `tools\04_compile_manuscript.cmd`
6. `tools\05_publish_github.cmd ...`

يمكن تنفيذ المراحل 1-5 دفعة واحدة عبر:

```cmd
tools\BUILD_ALL_LOCAL.cmd
```

هذا الأمر لا يرفع شيئاً إلى GitHub.

## إنشاء مستودع GitHub

أنشئ مستودعاً فارغاً بلا README أو License أو `.gitignore`، ثم نفذ:

```cmd
tools\05_publish_github.cmd https://github.com/OWNER/REPOSITORY.git "Your Name" "you@example.com"
```

## ملفات لا تُرفع

لا تضع داخل المستودع:

- `lab.zip`
- `results.rar`
- البيئات مثل `.venv`
- ملفات LaTeX المؤقتة مثل `*.aux` و`*.log`
- بيانات خام خارج الترخيص أو غير مطلوبة لإعادة الإنتاج

## إعادة توليد أصول الورقة فقط

```cmd
.venv\Scripts\python.exe tools\generate_manuscript_assets_v4.py --repo-root . --paper-root paper
```

سينشئ أو يحدث تلقائياً:

```text
paper\tables\structural_overall.tex
paper\tables\structural_pairwise.tex
paper\tables\process_results.tex
paper\tables\energy_decisions.tex
paper\tables\pjm_family.tex
paper\figures\structural_certificate_by_family.png
paper\figures\pjm_test_improvements.png
```
