# EXPERIMENT PROTOCOL V6

## Prospective certificate-aware structural learning

**Protocol status:** `prospective_pre_download_frozen`  
**Experiment:** `neso_gb_demand_2025_certificate_subset_v6`

This protocol governs the first prospective evaluation of certificate-aware
complete rule-subset learning in the repository. The NESO 2025 source file must
not be downloaded until this protocol, its configuration, implementation, and
tests have been committed and hashed.

## 1. Scientific question

Can a first-order TSK forecaster select a complete subset of a rule dictionary
by directly minimizing a time-uniform PAC-Bayesian certificate, while charging
both subset size and subset identity, and then deploy the exact clipped
Gaussian Bayesian model average without using test observations during
selection?

The outcome is admissible even if the prelocked decision gate abstains.

## 2. Source and immutable snapshot policy

The source is the National Energy System Operator (NESO) resource
`Historic Demand Data 2025`, resource identifier
`b2bde559-3455-4021-b179-dfe60c0337b0`.

Exactly one download is authorized after protocol freeze. The raw bytes,
download time in UTC, HTTP metadata, and SHA-256 must be recorded. NESO may
make retrospective corrections; the authorized experiment must not refresh
the source after the snapshot has been recorded.

Required attribution in resulting documentation:

> Supported by National Energy SO Open Data

## 3. Target and preprocessing

The target is `ND`, National Demand in MW. Rows are keyed by
`SETTLEMENT_DATE` and `SETTLEMENT_PERIOD`.

Pre-outcome validation rules:

1. Parse settlement dates day-first.
2. Reject duplicate `(date, period)` keys.
3. Reject missing or non-finite `ND`.
4. Require calendar coverage from 2025-01-01 through 2025-12-31.
5. Require each day to contain 46, 48, or 50 settlement periods.
6. Aggregate `ND` to one daily arithmetic mean.
7. Require exactly 365 daily rows.
8. Use no exogenous variables.
9. Construct centering and scaling from `D_prior` only.

Any violation aborts before outcome review and does not authorize parameter
changes.

## 4. Chronology

The 365 daily observations are split chronologically and without shuffling:

| Role | Count |
|---|---:|
| `D_prior` | 73 |
| `D_bound` | 164 |
| `D_validation` | 55 |
| `D_test` | 73 |

Lagged rows inherit the role of their target date. Test observations remain
inaccessible until all model, posterior-scale, subset, and gate decisions have
been frozen.

## 5. Frozen TSK dictionary

For each lag in `{1, 7, 14}`, construct a deterministic fixed-`K` antecedent
from lagged `D_prior` only:

- `K = 8`;
- all eight rules may fire;
- two deterministic refinement iterations;
- first-order consequents;
- ridge grid `{0.001, 0.01, 0.1, 1.0}`.

The rule dictionary is not reconstructed after subset selection.

## 6. Prior over structural identities

For a frozen dictionary with `K=8`, every non-empty subset `S` is supported.
The conditional prior is

\[
\pi(S\mid K)
=
\frac{\exp(-\eta |S|)}
{Z_K {K\choose |S|}},
\qquad
Z_K=\sum_{j=1}^{K}\exp(-\eta j),
\qquad
\eta=\log 2.
\]

Thus the subset charge is

\[
-\log \pi(S\mid K)
=
\eta |S|+\log Z_K+\log {K\choose |S|}.
\]

Lag and ridge-alpha receive independent uniform masses. The total structural
charge is the negative log mass of lag, ridge-alpha, subset size, and exact
subset identity. No structure selected after viewing certification data is
free.

## 7. Posterior construction and certificate objective

For each candidate `(lag, ridge_alpha, S)`:

1. Reduce the frozen dictionary to the complete rule subset `S`.
2. Renormalize firing strengths over retained rules.
3. Estimate the localized Gaussian prior mean and base scale from
   `D_prior`.
4. Estimate the posterior mean from `D_bound`.
5. Search only the predeclared prior-scale, posterior-ratio, and temperature
   grids.
6. Evaluate the exact posterior expectation of normalized squared loss after
   clipping prediction and target on the chronological certification sequence
   `D_bound + D_validation`.
7. Add Gaussian KL, base structural charge, exact subset-identity charge, and
   prior-scale grid charge.
8. compute the time-uniform certificate.

The selected structure is the candidate with the smallest **untruncated**
certificate. The frozen tie-break order is total KL, exact empirical Gibbs
risk, subset size, lexicographic subset identity, ridge alpha, then lag.

All `255` non-empty subsets are enumerated for each lag and ridge-alpha pair:
`255 × 3 × 4 = 3060` structural candidates.

## 8. Operational prediction object

The operational forecast is the exact clipped Gaussian Bayesian model average
of the selected posterior. The posterior is not refitted on validation data
after selection. This keeps the deployed prediction object aligned with the
certified posterior.

## 9. Prelocked abstention gate

The gate is inherited unchanged from the frozen PJM robust gate:

- untruncated certificate at most `0.10`;
- certification target clipping rate at most `0.02`;
- validation RMSE improvement of at least `5%` over seasonal-naive-7;
- validation asymmetric-cost improvement of at least `5%`;
- four chronological validation blocks;
- nonnegative improvement in at least three blocks;
- worst block degradation no greater than `5%`.

Underforecast cost has weight `2`, overforecast cost weight `1`.

If every condition passes, deploy the selected exact clipped BMA on test
without refitting. Otherwise abstain and use seasonal-naive-7. Abstention is a valid prospective outcome and must not trigger retuning or rerunning.

## 10. Authorized execution and audit

Only one outcome-bearing run is authorized. Publication must be atomic. The
audit must contain:

- protocol/configuration hashes;
- source URL, resource ID, UTC download timestamp, HTTP metadata, raw SHA-256;
- preprocessing counts and date coverage;
- split counts;
- all 3060 candidate identities and certificate components;
- selected lag, alpha, subset, scales, temperature, and tie-break fields;
- proof that selection and certification excluded test;
- frozen validation gate decision;
- one-time test metrics for selected deployment or fallback;
- hashes of all result and audit files.

Unexpected data or compute failure before outcome review produces an abort
record. It does not authorize silent repair, parameter changes, or an
outcome-bearing rerun.
