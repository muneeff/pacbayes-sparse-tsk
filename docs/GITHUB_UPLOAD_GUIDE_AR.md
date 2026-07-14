# دليل نشر مستودع V3.4 على GitHub

## ما يجب رفعه

يرفع المستودع الكود والبروتوكولات والاختبارات والنتائج النهائية الصغيرة اللازمة للتحكيم، بما في ذلك:

- `configs/v3/pjm_confirmatory_case_v3.yaml`
- `protocol/PJM_CONFIRMATORY_CASE_V3_PROTOCOL.md`
- `artifacts/pjm_confirmatory_v3_preoutcome_lock.json`
- `artifacts/pjm_confirmatory_v3_aborted_attempt_01.json`
- `artifacts/pjm_confirmatory_v3_run_manifest.json`
- `results/confirmatory/pjm_case_v3_4/`
- تقارير Development وTetouan وPJM.

لا ترفع ملفات PJM الخام؛ يستعيدها `scripts/download_pjm_energy_v3.py` ويتحقق من بصماتها. يمكن رفع البيانات والنتائج الكبيرة إلى Zenodo ثم إضافة DOI إلى `README.md` و`CITATION.cff`.

## التحقق قبل الرفع

من PowerShell داخل المشروع:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
python scripts\validate_v3.py
```

يجب أن تنجح الاختبارات كلها. لا تعِد تشغيل PJM؛ ملف `COMPLETED.json` جزء من سجل التجربة التأكيدية.

## تهيئة Git

```powershell
git init
git add .
git status
git commit -m "Release PAC-Bayesian Sparse TSK V3.4 reproducibility artifact"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

## إصدار ثابت للتحكيم

```powershell
git tag -a v3.4-confirmatory -m "Frozen PJM confirmatory artifact"
git push origin v3.4-confirmatory
```

بعد ربط GitHub بـZenodo، أنشئ Release من الوسم نفسه، ثم أضف DOI الناتج إلى `CITATION.cff` والمخطوط دون تغيير النتائج أو البروتوكول.

## ما لا يجوز فعله

- حذف `COMPLETED.json` ثم إعادة تشغيل PJM بوصفه تأكيدياً.
- تغيير العتبات أو السلاسل أو الفترة بعد الاطلاع على النتائج.
- حذف سجل محاولة الإيقاف الأولى.
- تقديم حالة PJM كدليل على تفوق Sparse TSK؛ النماذج المختارة كانت Ridge أو TSK بقاعدة واحدة.
