# PAC-Bayesian Sparse TSK — V3.4

This repository is the clean rebuild of the experimental artifact. V2 is preserved as a historical run. The revised evidence consists of corrected synthetic development, structural-sparsity ablations, a negative Tetouan development decision case, and a frozen independent PJM confirmatory case.

## Current status

- Corrected V3 synthetic development completed.
- Fixed-K versus radius-controlled structural-sparsity ablation completed.
- Tetouan development energy case completed; its original deployment gate failed operationally.
- Redesigned robust gate frozen using Tetouan development evidence only.
- Independent PJM confirmatory case completed once under a pre-outcome SHA-256 lock.
- All four PJM deployment decisions improved independent test RMSE and asymmetric cost relative to the predeclared seven-day seasonal-naive fallback.
- Three selected models were Ridge and the fourth was a one-rule TSK; therefore the PJM case validates the decision gate, not nonlinear Sparse TSK superiority.
- Theoretical and manuscript reconstruction is the next phase.

## Evidence summary

### Synthetic and structural ablation

- Ridge gives the tightest certificates overall.
- Radius-controlled structural sparsity reduces rule count, randomized dimension, KL, and certificate relative to a fixed-K TSK reference when it reduces the realized rule count.
- Top-3 activation sparsity is computational only; it does not reduce the randomized parameter dimension or systematically tighten the certificate.

### Tetouan development case

The first frozen deployment gate admitted two TSK decisions that later underperformed the seasonal fallback. This result is retained as a negative development case and motivated the stricter gate.

### PJM confirmatory case

The stricter gate required certificate <= 0.10, clipping <= 2%, at least 5% validation improvement in both RMSE and asymmetric cost, and stability across four chronological validation blocks. On AEP, COMED, DAYTON, and PJME, mean independent test improvements were 53.55% in RMSE and 53.62% in cost relative to the frozen fallback. See `docs/PJM_CONFIRMATORY_CASE_V3_REPORT_AR.md`.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e .[dev]
pytest
python scripts/validate_v3.py
```

## Retrieve and prepare the PJM data

The underlying Hourly Energy Consumption collection is identified as CC0 by its primary Kaggle page. The transport mirror is used only to retrieve exact files whose SHA-256 hashes are frozen.

```bash
python scripts/download_pjm_energy_v3.py
python scripts/prepare_pjm_energy_v3.py \
  --raw-dir data/raw/pjm \
  --processed data/processed/pjm_daily.csv \
  --audit artifacts/pjm_data_preparation_audit.json
```

## Confirmatory-run integrity

The completed PJM run is protected by `results/confirmatory/pjm_case_v3_4/COMPLETED.json`. The original pre-outcome lock is `artifacts/pjm_confirmatory_v3_preoutcome_lock.json`. Do not delete the completion marker or rerun the case while describing it as confirmatory.

A first infrastructure attempt stopped at a command timeout before test opening. It is documented in `artifacts/pjm_confirmatory_v3_aborted_attempt_01.json`. The exact unchanged retry completed successfully; no retuning occurred.

## Key documentation

- `protocol/EXPERIMENT_PROTOCOL_V3.md`
- `protocol/PJM_CONFIRMATORY_CASE_V3_PROTOCOL.md`
- `docs/PACBAYES_VALIDITY_CHECKLIST.md`
- `docs/DEVELOPMENT_V3_REPORT_AR.md`
- `docs/STRUCTURAL_SPARSITY_ABLATION_V3_REPORT_AR.md`
- `docs/ENERGY_CASE_STUDY_V3_REPORT_AR.md`
- `docs/PJM_CONFIRMATORY_CASE_V3_REPORT_AR.md`
- `docs/PJM_DATASET_MANIFEST.json`
- `docs/REPRODUCIBILITY.md`

## Exploratory multi-K sensitivity

The development-only Fixed-K sensitivity over `K={2,3,4,6,8,12}` can be reproduced on Windows with an active Python environment:

```cmd
tools\06_run_fixed_k_sensitivity_v4.cmd
```

This analysis does not modify the locked PJM confirmatory protocol or outcomes. See `docs/FIXED_K_SENSITIVITY_V4_REPORT_AR.md`.
