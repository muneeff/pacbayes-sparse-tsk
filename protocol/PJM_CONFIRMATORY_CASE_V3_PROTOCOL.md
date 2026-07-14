# Independent PJM Confirmatory Energy Case — Protocol V3.4

## Status

This protocol is confirmatory. It was designed after the Tetouan development case and before inspecting PJM test outcomes. Any retuning or rerun after outcome inspection is forbidden.

## Independent dataset

Four predeclared PJM regional hourly load series are used: AEP, COMED, DAYTON, and PJME. The source dataset is the CC0 Kaggle *Hourly Energy Consumption* collection. A public GitHub mirror is used only as a transport layer. Raw and processed SHA-256 hashes are recorded.

Hourly values are sorted, duplicate timestamps are collapsed by their mean, and the common interval 2012-01-01 through 2018-08-02 is aggregated to daily means. Each retained day must have at least 23 hourly observations. No exogenous variables are used.

## Chronological roles

The 2,406 daily observations are split without shuffling:

- Prior: 20%
- Bound: 45%
- Validation: 15%
- Test: 20%

Scaling, antecedent construction, prior means, and prior variances use the prior segment only. Posterior centers use the bound segment. PAC-Bayes certification uses bound plus validation. Test is excluded from fitting, certification, model selection, and deployment decisions.

## Model family

The frozen candidate family contains:

- Ridge
- Fixed-K dense TSK with K=8
- Radius-controlled dense TSK

Lags are 7, 14, and 28 days. Radius values are 0.75, 1.00, 1.25, 1.50, 2.00, and 2.50. Ridge penalties are 1e-4, 1e-3, 1e-2, 1e-1, and 1.

The hierarchical prior charges family, lag, radius, ridge penalty, and realized rule count. The familywise confidence allocation is 0.05/4.

## Redesigned decision gate

A model is eligible only when all conditions hold:

1. Untruncated certificate <= 0.10.
2. Certification target clipping <= 2%.
3. Validation RMSE improves over the 7-day seasonal-naive fallback by at least 5%.
4. Validation asymmetric operational cost improves by at least 5%.
5. RMSE improvement is nonnegative in at least 3 of 4 chronological validation blocks.
6. Cost improvement is nonnegative in at least 3 of 4 blocks.
7. Neither the worst RMSE block nor the worst cost block degrades by more than 5%.

Underforecast errors have weight 2 and overforecast errors weight 1. Among eligible candidates, the lowest validation operational cost wins; ties use RMSE, certificate, dimension, rule count, family, radius, ridge penalty, and lag. If no candidate qualifies, the system deploys 7-day seasonal naive.

## Test-opening rule

All four regional decisions must be serialized and hashed before any test forecast or test metric is computed. The run publishes atomically. A completed run cannot be overwritten.

## Interpretation

Success requires positive test improvement in the chosen operational cost and RMSE relative to fallback. A non-vacuous PAC-Bayes certificate alone is not interpreted as superiority to seasonal naive. Abstention is a valid outcome.
