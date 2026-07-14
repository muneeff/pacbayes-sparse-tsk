# Auditable PAC-Bayesian Forecasting — V4.3

This repository contains the reproducibility artifact for the revised manuscript:

> **An Auditable Complexity-Aware PAC-Bayesian Framework for Linear and Takagi--Sugeno Time-Series Forecasting under Temporal Dependence**

The artifact preserves the corrected synthetic development study, Fixed-\(K\) sensitivity, statistical inference, the negative Tetouan development case, the internally locked PJM holdout case, and the V4.3 secondary post-hoc and interpretability analyses.

## Scientific status

- The general time-uniform PAC-Bayes inequality is treated as an established result; the contribution is its chronological and computable specialization.
- Radius-controlled structural sparsity reduces dimension, KL, and certificate relative to a fixed 12-rule TSK reference, but it is not significantly better in certificate value than the best small Fixed-\(K\) model.
- Top-three activation sparsity does not materially tighten the certificate because it does not reduce the randomized parameter dimension.
- Ridge retains the tightest average certificate.
- The Tetouan development gate failed and is retained as negative evidence.
- The locked PJM gate improved over its predeclared seven-day seasonal-naive fallback, but the selected models were three ridge predictors and one one-rule TSK predictor.
- Secondary post-hoc SARIMA and ETS benchmarks, added only after the PJM holdout was opened, outperform the originally deployed predictors on average. They do not alter the frozen gate or its decisions.
- The PAC-Bayes certificate directly covers the clipped Gibbs predictor and, by convexity, the clipped Bayesian model average. The deterministic posterior-center predictor used for validation, test RMSE, and deployment is not directly certified after clipping; the certificate is used as a screening signal.

## Main V4.3 additions

1. Secondary post-hoc SARIMA and ETS benchmarks on the four PJM regions.
2. Explicit distinction between the certified Gibbs/BMA objects and the operational point predictor.
3. Actual multi-rule examples reconstructed from a five-rule SETAR model and a four-rule Tetouan model.
4. A complete augmented hierarchical prior over model structure, prior variant, scale, and parameters.
5. Full gate and infrastructure-recovery details moved to the supplementary material.
6. Discussion and conclusion rewritten around the auditable integration claim rather than TSK superiority.

## Quick local build on Windows

Activate an existing Python environment, install the project without changing dependencies, then run:

```cmd
python -m pip install -e . --no-deps
python -m pytest tests\v3
tools\04_compile_manuscript.cmd
```

Final files:

```text
paper\PACBayes_TSK_Manuscript_V4_3_Integrated.pdf
paper\PACBayes_TSK_Supplementary_V4_3.pdf
```

## Reproduce the V4.3 additions

The supplied CSV and LaTeX outputs are already included. Regeneration is optional.

### Secondary post-hoc PJM benchmarks

```cmd
tools\08_run_posthoc_energy_benchmarks_v4_3.cmd
```

This fits SARIMA and ETS candidates using prior/bound/validation data and evaluates the previously opened PJM test horizons. The analysis is explicitly post-hoc and never changes the frozen deployment gate.

### Actual fuzzy-rule examples

```cmd
tools\09_extract_fuzzy_rules_v4_3.cmd
```

This reconstructs the selected posterior-center TSK models and writes rule antecedents, firing summaries, and local affine consequents in original units.

### Fixed-K sensitivity and statistical inference

```cmd
tools\06_run_fixed_k_sensitivity_v4.cmd
tools\07_run_statistical_analysis_v4_2.cmd
```

## Complete reproducibility build

```cmd
tools\BUILD_ALL_LOCAL.cmd
```

The complete build regenerates manuscript assets, V4.2 statistical summaries, V4.3 post-hoc benchmarks and rule examples, validates the artifact, and compiles both PDFs. It does **not** rerun or modify the locked PJM confirmatory decisions.

## Important files

- `paper/main.tex`
- `paper/supplementary.tex`
- `paper/sections/03_model_and_chronology.tex`
- `paper/sections/04_certificate.tex`
- `paper/sections/05_experimental_protocol.tex`
- `paper/sections/06_results.tex`
- `paper/sections/07_discussion.tex`
- `paper/sections/08_conclusion.tex`
- `results/posthoc/v4_3/energy_posthoc_summary_all.csv`
- `results/interpretability/v4_3/fuzzy_rule_examples_summary.csv`
- `docs/INTEGRATED_REVISION_V4_3_REPORT_AR.md`
- `artifacts/pjm_confirmatory_v3_preoutcome_lock.json`
- `artifacts/pjm_confirmatory_v3_aborted_attempt_01.json`

## Integrity restriction

Do not delete the PJM completion marker or rerun the PJM case while describing a later run as the original locked evaluation. The V4.3 SARIMA/ETS analysis is post-hoc by construction and must remain labeled as such.
