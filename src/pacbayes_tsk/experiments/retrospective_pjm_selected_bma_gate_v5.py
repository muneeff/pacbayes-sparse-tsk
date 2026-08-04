from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pacbayes_tsk.data.energy_pjm import load_pjm_daily
from pacbayes_tsk.data.splits_v3 import ratio_split
from pacbayes_tsk.evaluation.bma_gate_v5 import (
    BMAGateThresholds,
    evaluate_bma_validation_gate,
)
from pacbayes_tsk.evaluation.clipped_gaussian_bma import (
    clipped_gaussian_bma,
)
from pacbayes_tsk.evaluation.exact_gibbs_certificate_v5 import (
    search_exact_clipped_gibbs_certificate,
)
from pacbayes_tsk.experiments.energy_case_study_v3 import (
    _base_prior_std,
    _build_designs,
    _gaussian_kl_ratio,
    _lagged,
    _ridge_solution,
    _rmse_mae,
    _role_masks,
    _weighted_cost,
)
from pacbayes_tsk.experiments.pjm_confirmatory_case_v3 import (
    PJMConfirmatorySettings,
)
from pacbayes_tsk.experiments.retrospective_bma_v5 import (
    POSTHOC_STATUS,
    RetrospectiveBMAError,
    _single_row,
)
from pacbayes_tsk.pac_bayes.certificates_v3 import (
    martingale_certificate,
    pointwise_gibbs_upper,
)


@dataclass(frozen=True)
class PJMSelectedBMAGateRetrospectiveResult:
    summary: pd.DataFrame
    predictions: dict[str, pd.DataFrame]


def _relative_improvement(
    baseline: float,
    candidate: float,
) -> float:
    baseline_value = float(baseline)
    candidate_value = float(candidate)
    return (
        baseline_value - candidate_value
    ) / max(abs(baseline_value), 1e-12)


def _gate_thresholds(
    settings: PJMConfirmatorySettings,
) -> BMAGateThresholds:
    return BMAGateThresholds(
        certificate_threshold=settings.certificate_threshold,
        clipping_rate_maximum=settings.clipping_rate_maximum,
        minimum_rmse_improvement=settings.minimum_rmse_improvement,
        minimum_asymmetric_score_improvement=(
            settings.minimum_cost_improvement
        ),
        validation_blocks=settings.validation_blocks,
        minimum_nonnegative_blocks=(
            settings.minimum_nonnegative_blocks
        ),
        maximum_worst_block_degradation=(
            settings.maximum_worst_block_degradation
        ),
        underforecast_weight=settings.underforecast_weight,
        overforecast_weight=settings.overforecast_weight,
    )


def _reconstruct_candidate(
    *,
    original: np.ndarray,
    labels: np.ndarray,
    settings: PJMConfirmatorySettings,
    candidate: pd.Series,
) -> dict[str, object]:
    prior_raw = original[labels == "prior"]
    scaling_mean = float(np.mean(prior_raw))
    scaling_std = float(np.std(prior_raw, ddof=0))
    if not np.isfinite(scaling_std) or scaling_std <= 1e-12:
        raise RetrospectiveBMAError(
            "Invalid prior-only scaling standard deviation."
        )

    scaled = (original - scaling_mean) / scaling_std
    lag = int(candidate["lag"])
    features, target_scaled, target_labels, target_indices = _lagged(
        scaled,
        labels,
        lag,
    )
    masks = _role_masks(target_labels)
    radius = (
        None
        if pd.isna(candidate["radius"])
        else float(candidate["radius"])
    )
    designs, _, _, capped = _build_designs(
        family=str(candidate["family"]),
        features=features,
        masks=masks,
        settings=settings,
        radius=radius,
    )
    if capped:
        raise RetrospectiveBMAError(
            "A frozen selected candidate unexpectedly reached the rule cap."
        )

    ridge_alpha = float(candidate["ridge_alpha"])
    prior_mean, prior_inverse_diag = _ridge_solution(
        designs["prior"],
        target_scaled[masks["prior"]],
        ridge_alpha,
    )
    posterior_mean, _ = _ridge_solution(
        designs["bound"],
        target_scaled[masks["bound"]],
        ridge_alpha,
    )
    base_prior_std = _base_prior_std(
        designs["prior"],
        target_scaled[masks["prior"]],
        prior_mean,
        prior_inverse_diag,
    )
    certification_design = np.vstack(
        [designs["bound"], designs["validation"]]
    )
    certification_target = np.concatenate(
        [
            target_scaled[masks["bound"]],
            target_scaled[masks["validation"]],
        ]
    )
    return {
        "scaling_mean": scaling_mean,
        "scaling_std": scaling_std,
        "target_scaled": target_scaled,
        "target_indices": target_indices,
        "masks": masks,
        "designs": designs,
        "prior_mean": prior_mean,
        "posterior_mean": posterior_mean,
        "base_prior_std": base_prior_std,
        "certification_design": certification_design,
        "certification_target": certification_target,
    }


def _verify_frozen_certificate(
    *,
    candidate: pd.Series,
    reconstruction: dict[str, object],
    settings: PJMConfirmatorySettings,
) -> float:
    prior_scale = float(candidate["prior_scale"])
    posterior_ratio = float(candidate["posterior_ratio"])
    posterior_std = (
        np.asarray(reconstruction["base_prior_std"])
        * prior_scale
        * posterior_ratio
    )
    empirical = float(
        np.mean(
            pointwise_gibbs_upper(
                np.asarray(
                    reconstruction["certification_design"]
                ),
                np.asarray(
                    reconstruction["certification_target"]
                ),
                np.asarray(reconstruction["posterior_mean"]),
                posterior_std,
                clip_bound=float(candidate["clip_bound_scaled"]),
            )
        )
    )
    gaussian_kl = _gaussian_kl_ratio(
        np.asarray(reconstruction["posterior_mean"]),
        np.asarray(reconstruction["prior_mean"]),
        np.asarray(reconstruction["base_prior_std"]),
        prior_scale,
        posterior_ratio,
    )
    total_kl = (
        float(candidate["structure_kl"])
        + float(np.log(len(settings.prior_scales)))
        + gaussian_kl
    )
    certificate = martingale_certificate(
        empirical_risk=empirical,
        total_kl=total_kl,
        n=len(
            np.asarray(reconstruction["certification_target"])
        ),
        delta=settings.familywise_delta,
        temperatures=settings.temperatures,
    )

    expected = {
        "empirical_gibbs_risk": empirical,
        "gaussian_kl": gaussian_kl,
        "total_kl": total_kl,
        "certificate_untruncated": float(certificate.untruncated),
    }
    return max(
        abs(float(candidate[column]) - value)
        for column, value in expected.items()
    )


def _role_arrays(
    *,
    original: np.ndarray,
    role: str,
    reconstruction: dict[str, object],
    posterior_std: np.ndarray,
    clip_bound: float,
    seasonal_lag: int,
) -> dict[str, np.ndarray]:
    masks = reconstruction["masks"]
    role_mask = masks[role]
    indices = np.asarray(
        reconstruction["target_indices"]
    )[role_mask]
    if np.any(indices < seasonal_lag):
        raise RetrospectiveBMAError(
            f"{role} indices are incompatible with the seasonal lag."
        )

    bma_scaled = clipped_gaussian_bma(
        np.asarray(reconstruction["designs"][role]),
        np.asarray(reconstruction["posterior_mean"]),
        posterior_std,
        clip_bound=clip_bound,
    )
    scaling_mean = float(reconstruction["scaling_mean"])
    scaling_std = float(reconstruction["scaling_std"])
    target_raw = (
        np.asarray(reconstruction["target_scaled"])[role_mask]
        * scaling_std
        + scaling_mean
    )
    bma_raw = bma_scaled * scaling_std + scaling_mean
    seasonal_raw = original[indices - seasonal_lag]
    return {
        "indices": indices,
        "target": target_raw,
        "bma": bma_raw,
        "seasonal": seasonal_raw,
    }


def _test_metrics(
    *,
    arrays: dict[str, np.ndarray],
    settings: PJMConfirmatorySettings,
) -> dict[str, float]:
    target = arrays["target"]
    bma = arrays["bma"]
    seasonal = arrays["seasonal"]

    baseline_rmse, baseline_mae = _rmse_mae(target, seasonal)
    bma_rmse, bma_mae = _rmse_mae(target, bma)
    baseline_score, _ = _weighted_cost(
        target,
        seasonal,
        under_weight=settings.underforecast_weight,
        over_weight=settings.overforecast_weight,
    )
    bma_score, _ = _weighted_cost(
        target,
        bma,
        under_weight=settings.underforecast_weight,
        over_weight=settings.overforecast_weight,
    )
    return {
        "test_baseline_rmse": baseline_rmse,
        "test_bma_rmse": bma_rmse,
        "test_rmse_improvement_vs_fallback": _relative_improvement(
            baseline_rmse,
            bma_rmse,
        ),
        "test_baseline_mae": baseline_mae,
        "test_bma_mae": bma_mae,
        "test_baseline_asymmetric_score": baseline_score,
        "test_bma_asymmetric_score": bma_score,
        "test_score_improvement_vs_fallback": _relative_improvement(
            baseline_score,
            bma_score,
        ),
    }


def evaluate_pjm_selected_bma_gate_retrospective(
    *,
    case_data_path: str | Path,
    config_path: str | Path,
    frozen_results_dir: str | Path,
    reproduction_tolerance: float = 1e-8,
) -> PJMSelectedBMAGateRetrospectiveResult:
    """Re-evaluate frozen PJM selections with exact BMA and the frozen gate.

    Candidate identities come only from serialized pre-test decisions. The
    exact clipped-Gibbs posterior scale and BMA validation metrics are
    retrospective additions; they do not modify or relabel the original
    confirmatory protocol.
    """

    if (
        not np.isfinite(reproduction_tolerance)
        or reproduction_tolerance <= 0.0
    ):
        raise RetrospectiveBMAError(
            "reproduction_tolerance must be finite and positive."
        )

    frozen_results = Path(frozen_results_dir)
    completion = frozen_results / "COMPLETED.json"
    if not completion.is_file():
        raise RetrospectiveBMAError(
            f"Missing frozen completion marker: {completion}"
        )

    settings = PJMConfirmatorySettings.from_yaml(config_path)
    thresholds = _gate_thresholds(settings)
    thresholds.validate()
    data = load_pjm_daily(case_data_path)
    split = ratio_split(len(data), settings.split_fractions)
    decisions = pd.read_csv(
        frozen_results / "pjm_pretest_decisions.csv"
    )

    rows: list[dict[str, object]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}

    for region in settings.regions:
        decision = _single_row(
            decisions,
            decisions["region"].eq(region),
            label=f"frozen pretest decision for {region}",
        )
        candidate_id = decision["selected_candidate_id"]
        if pd.isna(candidate_id):
            rows.append(
                {
                    "status": POSTHOC_STATUS,
                    "region": region,
                    "frozen_deployment": str(
                        decision["deployment"]
                    ),
                    "frozen_selected_candidate_id": None,
                    "bma_gate_applicable": False,
                    "gate_passed": False,
                    "failed_conditions": (
                        "frozen_fallback_has_no_selected_posterior",
                    ),
                    "selection_uses_test": False,
                    "interpretation_uses_test_posthoc": True,
                    "confirmatory_decision_modified": False,
                }
            )
            continue

        candidate_id = str(candidate_id)
        candidates = pd.read_csv(
            frozen_results / f"{region}_candidates.csv"
        )
        candidate = _single_row(
            candidates,
            candidates["candidate_id"].eq(candidate_id),
            label=f"candidate {candidate_id}",
        )
        original = data[region].to_numpy(dtype=float)
        reconstruction = _reconstruct_candidate(
            original=original,
            labels=split.labels,
            settings=settings,
            candidate=candidate,
        )

        frozen_certificate_error = _verify_frozen_certificate(
            candidate=candidate,
            reconstruction=reconstruction,
            settings=settings,
        )
        if frozen_certificate_error > reproduction_tolerance:
            raise RetrospectiveBMAError(
                f"Frozen certificate was not reproduced for "
                f"{region}/{candidate_id}: "
                f"{frozen_certificate_error:.17g}."
            )

        exact = search_exact_clipped_gibbs_certificate(
            design_cert=np.asarray(
                reconstruction["certification_design"]
            ),
            target_cert=np.asarray(
                reconstruction["certification_target"]
            ),
            posterior_mean=np.asarray(
                reconstruction["posterior_mean"]
            ),
            prior_mean=np.asarray(
                reconstruction["prior_mean"]
            ),
            base_prior_std=np.asarray(
                reconstruction["base_prior_std"]
            ),
            clip_bound=float(candidate["clip_bound_scaled"]),
            structure_kl=float(candidate["structure_kl"]),
            prior_scale_multipliers=settings.prior_scales,
            posterior_scale_ratios=settings.posterior_ratios,
            temperatures=settings.temperatures,
            delta=settings.familywise_delta,
        )
        posterior_std = (
            np.asarray(reconstruction["base_prior_std"])
            * exact.prior_scale_multiplier
            * exact.posterior_scale_ratio
        )

        validation = _role_arrays(
            original=original,
            role="validation",
            reconstruction=reconstruction,
            posterior_std=posterior_std,
            clip_bound=float(candidate["clip_bound_scaled"]),
            seasonal_lag=settings.seasonal_lag,
        )
        gate = evaluate_bma_validation_gate(
            target=validation["target"],
            bma_prediction=validation["bma"],
            seasonal_prediction=validation["seasonal"],
            certificate_untruncated=(
                exact.certificate_untruncated
            ),
            certification_target_clipping_rate=float(
                candidate["certification_target_clipping_rate"]
            ),
            thresholds=thresholds,
        )

        test = _role_arrays(
            original=original,
            role="test",
            reconstruction=reconstruction,
            posterior_std=posterior_std,
            clip_bound=float(candidate["clip_bound_scaled"]),
            seasonal_lag=settings.seasonal_lag,
        )
        stored = pd.read_csv(
            frozen_results
            / "predictions"
            / f"{region}_test_predictions.csv"
        )
        if len(stored) != len(test["target"]):
            raise RetrospectiveBMAError(
                f"Stored prediction length mismatch for {region}."
            )
        target_error = float(
            np.max(
                np.abs(
                    stored["actual"].to_numpy(dtype=float)
                    - test["target"]
                )
            )
        )
        seasonal_error = float(
            np.max(
                np.abs(
                    stored["seasonal_naive_7d"].to_numpy(
                        dtype=float
                    )
                    - test["seasonal"]
                )
            )
        )
        if (
            target_error > reproduction_tolerance
            or seasonal_error > reproduction_tolerance
        ):
            raise RetrospectiveBMAError(
                f"Frozen test alignment was not reproduced for {region}."
            )

        test_metrics = _test_metrics(
            arrays=test,
            settings=settings,
        )
        rows.append(
            {
                "status": POSTHOC_STATUS,
                "region": region,
                "frozen_deployment": str(decision["deployment"]),
                "frozen_selected_candidate_id": candidate_id,
                "family": str(candidate["family"]),
                "bma_gate_applicable": True,
                "exact_prior_scale": (
                    exact.prior_scale_multiplier
                ),
                "exact_posterior_ratio": (
                    exact.posterior_scale_ratio
                ),
                "exact_certificate_untruncated": (
                    exact.certificate_untruncated
                ),
                "max_old_certificate_reproduction_error": (
                    frozen_certificate_error
                ),
                "max_target_reproduction_error": target_error,
                "max_seasonal_reproduction_error": seasonal_error,
                **gate.as_dict(),
                **test_metrics,
                "selection_uses_test": False,
                "interpretation_uses_test_posthoc": True,
                "confirmatory_decision_modified": False,
            }
        )

        prediction_frames[region] = pd.DataFrame(
            {
                "timestamp": stored["timestamp"],
                "actual": test["target"],
                "seasonal_naive_7d": test["seasonal"],
                "exact_clipped_bma_forecast": test["bma"],
                "analysis_status": POSTHOC_STATUS,
            }
        )

    return PJMSelectedBMAGateRetrospectiveResult(
        summary=pd.DataFrame(rows),
        predictions=prediction_frames,
    )
