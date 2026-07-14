# تقرير تحليل التناثر البنيوي V3.2

## نطاق المرحلة

هذه مرحلة **Development استكشافية** وليست Confirmatory. قارنت أربع عائلات تحت prior هرمي واحد:

1. Ridge.
2. Fixed-K Dense TSK بعدد ثابت قدره 12 قاعدة وكل القواعد فعالة.
3. Radius-Dense TSK بعدد قواعد يتحدد بنصف القطر وكل القواعد فعالة.
4. Radius-Top3 TSK بالبنية نفسها مع الإبقاء على أكبر ثلاث درجات تفعيل لكل عينة.

الغرض هو فصل **التناثر البنيوي** (تقليل K والبعد) عن **تناثر التفعيل** (Top-3 فقط).

## سلامة التنفيذ

- 6 عمليات × 5 بذور = **30 سلسلة** مكتملة.
- عدد المرشحين الصالحين: **12,700**.
- مرشحو Fixed-K: **750**، وجميعهم استخدموا K=12 فعلياً.
- أخطاء بناء النماذج: **صفر**.
- الاختيار والشهادات استبعدت بيانات الاختبار.
- الاختبارات الآلية بعد الإضافة: **22/22 ناجحة**.
- Confirmatory V3 لم يُشغّل.

## النتيجة الأساسية

**التناثر البنيوي مدعوم بوضوح، أما Top-3 activation sparsity فلا يقدم فائدة منهجية للشهادة في هذا التصميم.**

عند الاختيار بأقل Validation RMSE، مقارنة Radius-Dense مع Fixed-K Dense عبر 30 سلسلة:

- خفّض البعد العشوائي بمتوسط **95.9** معامل.
- خفّض Gaussian KL بمتوسط **22.917**.
- خفّض الشهادة بمتوسط مطلق **0.0631**.
- كانت شهادة Radius-Dense أقل في **28 من 30** سلسلة، وأعلى في سلسلتين.
- كان Test RMSE أقل في **24 من 30** سلسلة.

عند الاختيار بأقل شهادة:

- اختارت عائلتا Radius-Dense وRadius-Top3 في المتوسط قاعدة واحدة فقط، ولذلك تطابقت نتائجهما تماماً.
- كانت شهادة Radius-Dense أقل من Fixed-K في **30 من 30** سلسلة.
- متوسط الشهادة: Ridge=0.1017، Radius-Dense=0.1114، Fixed-K=0.1679.

## ملخص شامل عبر 30 سلسلة

| العائلة | أسلوب الاختيار | متوسط K | متوسط البعد | متوسط Gaussian KL | متوسط الشهادة | متوسط Test RMSE |
|---|---|---:|---:|---:|---:|---:|
| Ridge | validation_rmse | 1.00 | 13.40 | 5.159 | 0.1145 | 0.8866 |
| Fixed-K Dense TSK | validation_rmse | 12.00 | 135.60 | 39.547 | 0.2221 | 0.9149 |
| Radius-Dense TSK | validation_rmse | 3.80 | 39.70 | 16.630 | 0.1589 | 0.8814 |
| Radius-Top3 TSK | validation_rmse | 3.57 | 40.23 | 16.831 | 0.1589 | 0.8845 |
| Ridge | certificate | 1.00 | 4.23 | 0.592 | 0.1017 | 0.9116 |
| Fixed-K Dense TSK | certificate | 12.00 | 75.20 | 16.665 | 0.1679 | 0.9188 |
| Radius-Dense TSK | certificate | 1.00 | 4.93 | 1.088 | 0.1114 | 0.9021 |
| Radius-Top3 TSK | certificate | 1.00 | 4.93 | 1.088 | 0.1114 | 0.9021 |

## النتائج حسب العملية عند اختيار Validation RMSE

| العملية | Radius-Dense RMSE | Fixed-K RMSE | فرق الشهادة Radius−Fixed | فرق البعد Radius−Fixed |
|---|---:|---:|---:|---:|
| ar2 | 0.9126 | 0.9342 | -0.0564 | -44.0 |
| garch | 1.0926 | 1.1544 | -0.0355 | -153.6 |
| mackey_glass | 0.0539 | 0.0521 | -0.0616 | -130.2 |
| narma10 | 0.7656 | 0.8558 | -0.1266 | -166.8 |
| setar | 0.9798 | 0.9936 | -0.0421 | -40.0 |
| structural_break | 1.4841 | 1.4992 | -0.0567 | -40.8 |

القيم السالبة في العمودين الأخيرين تعني أن Radius-Dense أقل تعقيداً وأضيق شهادة. الاستثناء التنبئي الوحيد على مستوى متوسط العملية كان Mackey–Glass، حيث حقق Fixed-K RMSE أقل بفارق صغير، لكنه بقي أسوأ بوضوح في KL والشهادة.

## تحليل تناثر التفعيل Top-3

- تمت مطابقة **5,600 زوج مرشح** بين all-active وTop-3 عند نفس العملية والبذرة والـlag والـradius وridge alpha وعدد القواعد.
- في 4,383 زوجاً لم يتغير Validation RMSE إطلاقاً، لأن عدد القواعد الفعلي كان غالباً ≤3 أو لأن الأوزان المستبعدة كانت ضئيلة.
- متوسط فرق الشهادة Top3−All كان قرابة الصفر: **−0.00005** فقط.
- تناثر التفعيل لا يغير البعد العشوائي K(p+1)، ولذلك لا يخفض بند Gaussian dimension تلقائياً.
- النتيجة: لا يجوز تقديم Top-3 بوصفه سبباً لتحسن PAC-Bayes certificate. فائدته الأساسية حسابية في التنبؤ عندما K>3.

## الحكم العلمي

1. **الادعاء المدعوم:** نصف القطر الذي يخفض عدد القواعد يمكنه خفض البعد وKL والشهادة مقارنة ببنية ثابتة من 12 قاعدة.
2. **الادعاء غير المدعوم:** الاقتصار على أكبر ثلاث قواعد فعالة يؤدي بحد ذاته إلى شهادة أضيق.
3. **قيد مهم:** Ridge ما زال يحقق أضيق شهادة إجمالاً؛ لذلك مساهمة الورقة يجب أن تركز على شهادة TSK القابلة للحساب ومقايضة الدقة–التعقيد، لا على تفوق عالمي على النماذج الخطية.
4. Structural Break يبقى حالة فشل تشغيلي بسبب القص المرتفع، ولا يجوز استخدامه كحالة نجاح.

## القرار للمرحلة التالية

يُعتمد Radius-Dense TSK بوصفه النموذج الأساسي في تحليل الشهادة البنيوية، ويُحتفظ بـTop-3 كتحسين حسابي منفصل. المرحلة التالية هي بناء حالة طاقة حقيقية ذات قرار مسبق، clipping منخفض، ومقارنة Ridge وFixed-K وRadius-Dense.

## ملفات النتائج

- `results/development/structural_ablation_v3/development_v3_selected_all.csv`
- `results/development/structural_ablation_v3/development_v3_candidates_all.csv`
- `results/development/structural_ablation_v3/development_v3_summary.csv`
- `results/development/structural_ablation_v3/structural_sparsity_selected_pairs.csv`
- `results/development/structural_ablation_v3/activation_sparsity_selected_pairs.csv`
- `results/development/structural_ablation_v3/activation_sparsity_candidate_matched.csv`
- `results/development/structural_ablation_v3/structural_ablation_comparison_summary.csv`
- `results/development/structural_ablation_v3/structural_ablation_family_overall.csv`
- `results/development/structural_ablation_v3/structural_ablation_integrity.json`
- `results/development/structural_ablation_v3/development_v3_audit.json`
