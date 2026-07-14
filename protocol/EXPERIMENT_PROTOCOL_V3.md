# Experimental Protocol V3

## Status
Development is open. Confirmatory V3 has **not** been authorized or run.

## Non-negotiable rules
1. All model indices are predeclared as `M=(p,r,gamma,K)`.
2. Scaling, antecedent geometry, prior means, and prior covariances use only `Dprior`.
3. Posterior centers use `Dbound`; validation may select the posterior/model because the bound is simultaneous over the frozen model family and the full structural code length is charged.
4. Certification uses `Dbound + Dvalidation`; `Dtest` is excluded.
5. Real benchmark series use their official forecast horizon for validation and test. Synthetic series use 20/45/15/20.
6. Confirmatory code, configs, data IDs, seeds, and environment are hashed before authorization. One authorized run only.

## V3 synthetic definitions
- AR(2): `Y_t=0.6Y_{t-1}-0.2Y_{t-2}+eps_t`, `eps~N(0,0.2^2)`.
- SETAR: `0.65Y_{t-1}+eps_t` when `Y_{t-1}<=0`; otherwise `-0.45Y_{t-1}+0.25Y_{t-2}+eps_t`, `eps~N(0,0.2^2)`.
- NARMA-10: standard recursion in `synthetic_v3.py`.
- Mackey–Glass: Euler recursion with delay 17 and additive `N(0,0.01^2)` noise.
- GARCH: zero-mean GARCH(1,1) with standardized Student-t(5) innovations.
- Structural break: AR(1) parameter/mean/noise change after 60% of the post-burn-in sample.

## Selection
Primary ordering is validation RMSE. Exact/numerical ties use validation MAE, consequent dimension, rule count, radius, ridge alpha, then lag.

## Required analyses
Sparse TSK vs dense TSK vs Ridge; radius-rule-dimension-KL-certificate chain; localized vs zero prior; RMSE vs certificate-aware selection; clipping analysis; energy decision case study.

## Structural-sparsity development ablation (V3.2)

Before confirmatory freezing, an exploratory ablation compares four families
under one finite hierarchical prior:

1. `ridge`;
2. `fixed_k_dense_tsk`: exactly 12 rules, all active;
3. `dense_tsk`: radius-controlled realized rule count, all active;
4. `sparse_tsk`: radius-controlled realized rule count, top-3 activation.

The fixed-K antecedent is constructed only from `Dprior` using deterministic
origin-seeded farthest-first traversal. Its realized covering radius is logged
as a diagnostic but is not selected as a hyperparameter. This comparison
separates structural sparsity (fixed K versus radius-controlled K) from
activation sparsity (all-active versus top-3 with the same radius grid).
