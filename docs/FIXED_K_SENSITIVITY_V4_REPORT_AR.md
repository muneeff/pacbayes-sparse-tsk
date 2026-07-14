# تقرير حساسية Fixed-K متعددة القيم — V4.1

## الهدف

اختبار ما إذا كان تفوق Radius-controlled TSK على خط الأساس ذي 12 قاعدة ناتجاً عن آلية نصف القطر نفسها، أم ببساطة عن تقليل عدد القواعد والمعاملات.

## التصميم

- العمليات: AR(2)، SETAR، NARMA-10، Mackey–Glass، GARCH، Structural Break.
- خمس بذور لكل عملية: 30 سلسلة كاملة.
- قيم Fixed-K: 2، 3، 4، 6، 8، 12.
- استُبعد K=1 من المقارنة الضبابية الأساسية لأنه يتحول إلى نموذج affine.
- بقيت بقية الشبكات والبروتوكول كما هي.
- كل قيمة K دفعت كلفة rule-count نفسها ضمن prior الهندسي؛ لم يكن اختيار K مجانياً.
- التحليل تطويري استكشافي، ولم يُعدّل بوابة PJM أو قراراتها أو نتائجها.
- عدد المرشحين المؤهلين: 16,450.
- فشل البناء: صفر.
- الاختبارات الآلية: 34/34 ناجحة.

## المتوسطات عند الاختيار بأقل Validation RMSE

| K | Dimension | Gaussian KL | Certificate | Test RMSE |
|---:|---:|---:|---:|---:|
| 2 | 23.7 | 10.180 | 0.1326 | 0.8850 |
| 3 | 34.3 | 13.469 | 0.1465 | 0.8846 |
| 4 | 41.9 | 17.294 | 0.1568 | 0.8869 |
| 6 | 60.0 | 21.011 | 0.1701 | 0.8950 |
| 8 | 82.9 | 26.413 | 0.1875 | 0.8951 |
| 12 | 135.6 | 39.547 | 0.2221 | 0.9149 |
| Radius TSK | 39.7 | 16.630 | 0.1589 | 0.8814 |

## النتيجة الصريحة

- الشهادة تزداد عموماً مع K.
- Radius TSK ليس أفضل من جميع Fixed-K.
- Fixed K=2 وK=3 حققا شهادات أضيق من Radius TSK في المتوسط.
- K=4 كان قريباً جداً من Radius TSK.
- Radius TSK أصبح أفضل بوضوح من القيم الكبيرة K=8 وK=12.
- عند اختيار أفضل Fixed-K لكل سلسلة بواسطة validation RMSE، كان فرق الشهادة Radius minus best Fixed-K يساوي -0.0068 فقط، وفترة bootstrap 95% هي [-0.0187, 0.0049]؛ أي إن الفرق غير حاسم.
- فرق Test RMSE كان -0.0054 وفترة bootstrap [-0.0118, -0.0002]، لكن Wilcoxon بعد تصحيح التعدد لم يكن دالاً.
- عند الاختيار بأقل شهادة، اختارت جميع السلاسل K=2 داخل عائلة Fixed-K.

## أثر النتيجة على ادعاء الورقة

الادعاء القديم كان واسعاً أكثر من اللازم. النتيجة المدعومة الآن هي:

> إزالة كتل قواعد كاملة تخفض البعد وKL والشهادة مقارنة ببنية كثيفة كبيرة، لكن آلية radius لا تتفوق بصورة عامة على شبكة Fixed-K صغيرة ومشحونة بعدالة.

هذا يرفع مصداقية الورقة، لكنه يقلل قوة الادعاء حول أفضلية radius نفسها.

## الملفات الناتجة

- `results/development/fixed_k_sensitivity_v4/fixed_k_sensitivity_selected.csv`
- `results/development/fixed_k_sensitivity_v4/fixed_k_sensitivity_summary.csv`
- `results/development/fixed_k_sensitivity_v4/fixed_k_vs_radius_paired_statistics.csv`
- `results/development/fixed_k_sensitivity_v4/best_fixed_k_selected.csv`
- `results/development/fixed_k_sensitivity_v4/radius_vs_best_fixed_k_statistics.csv`
- `paper/tables/fixed_k_sensitivity.tex`
- `paper/figures/fixed_k_certificate_sensitivity.png`
- `paper/figures/fixed_k_pareto_sensitivity.png`
