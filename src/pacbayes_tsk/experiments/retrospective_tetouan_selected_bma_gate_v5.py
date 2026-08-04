from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pacbayes_tsk.data.energy_tetouan import load_two_hour_tetouan
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
    EnergySettings,
    _base_prior_std,
    _build_designs,
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
from pacbayes_tsk.experiments.retrospective_tetouan_bma_v5 import (
    evaluate_tetouan_bma_retrospective,
)


@dataclass(frozen=True)
class TetouanSelectedBMAGateRetrospectiveResult:
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
    settings: EnergySettings,
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


def _role_arrays(
    *,
    original: np.ndarray,
    role: str,
    reconstruction: dict[str, object],
    posterior_std: np.ndarray,
    clip_bound: float,
    seasonal_lag: int,
) -> dict[str, np.ndarray]:
    role_mask = reconstruction["masks"][role]
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
    target = (
        np.asarray(reconstruction["target_scaled"])[role_mask]
        * scaling_std
        + scaling_mean
    )
    bma = bma_scaled * scaling_std + scaling_mean
    seasonal = original[indices - seasonal_lag]
    return {
        "indices": indices,
        "target": target,
        "bma": bma,
        "seasonal": seasonal,
    }


def _test_metrics(
    *,
    arrays: dict[str, np.ndarray],
    settings: EnergySettings,
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
        "test_exact_bma_rmse": bma_rmse,
        "test_exact_bma_rmse_improvement_vs_fallback": (
            _relative_improvement(baseline_rmse, bma_rmse)
        ),
        "test_baseline_mae": baseline_mae,
        "test_exact_bma_mae": bma_mae,
        "test_baseline_asymmetric_score": baseline_score,
        "test_exact_bma_asymmetric_score": bma_score,
        "test_exact_bma_score_improvement_vs_fallback": (
            _relative_improvement(baseline_score, bma_score)
        ),
    }


def evaluate_tetouan_selected_bma_gate_retrospective(
    *,
    case_data_path: str | Path,
    energy_config_path: str | Path,
    gate_config_path: str | Path,
    frozen_results_dir: str | Path,
    reproduction_tolerance: float = 1e-8,
) -> TetouanSelectedBMAGateRetrospectiveResult:
    """Audit frozen Tetouan selections with exact BMA and the frozen gate.

    The candidate identities remain the completed V3 development choices.
    Posterior-scale selection, validation-gate evaluation, and test
    interpretation are retrospective V5 analyses and never modify the
    frozen development outputs.
    """

    if (
        not np.isfinite(reproduction_tolerance)
        or reproduction_tolerance <= 0.0
    ):
        raise RetrospectiveBMAError(
            "reproduction_tolerance must be finite and positive."
        )

    results = Path(frozen_results_dir)
    completion = results / "COMPLETED.json"
    if not completion.is_file():
        raise RetrospectiveBMAError(
            f"Missing frozen completion marker: {completion}"
        )

    energy = EnergySettings.from_yaml(energy_config_path)
    gate_settings = PJMConfirmatorySettings.from_yaml(
        gate_config_path
    )
    thresholds = _gate_thresholds(gate_settings)
    thresholds.validate()

    data = load_two_hour_tetouan(case_data_path)
    split = ratio_split(len(data), energy.split_fractions)
    decisions = pd.read_csv(
        results / "energy_deployment_decisions.csv"
    )
    frozen_scale_result = evaluate_tetouan_bma_retrospective(
        case_data_path=case_data_path,
        config_path=energy_config_path,
        frozen_results_dir=results,
        reproduction_tolerance=reproduction_tolerance,
    )

    rows: list[dict[str, object]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}

    for zone in energy.zones:
        decision = _single_row(
            decisions,
            decisions["zone"].eq(zone),
            label=f"frozen decision for {zone}",
        )
        candidate_id = decision["selected_candidate_id"]

        if pd.isna(candidate_id):
            rows.append(
                {
                    "status": POSTHOC_STATUS,
                    "zone": zone,
                    "frozen_deployment": str(
                        decision["deployment"]
                    ),
                    "frozen_selected_candidate_id": None,
                    "bma_gate_applicable": False,
                    "gate_passed": False,
                    "failed_conditions": (
                        "frozen_fallback_has_no_selected_posterior",
                    ),
                    "abstention_action": "seasonal_fallback",
                    "selection_uses_test": False,
                    "interpretation_uses_test_posthoc": True,
                    "development_decision_modified": False,
                }
            )
            continue

        candidate_id = str(candidate_id)
        candidates = pd.read_csv(
            results / f"{zone}_candidates.csv"
        )
        candidate = _single_row(
            candidates,
            candidates["candidate_id"].eq(candidate_id),
            label=f"candidate {candidate_id}",
        )
        original = data[zone].to_numpy(dtype=float)
        reconstruction = _reconstruct_candidate(
            original=original,
            labels=split.labels,
            settings=energy,
            candidate=candidate,
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
            prior_scale_multipliers=energy.prior_scales,
            posterior_scale_ratios=energy.posterior_ratios,
            temperatures=energy.temperatures,
            delta=energy.familywise_delta,
        )
        exact_posterior_std = (
            np.asarray(reconstruction["base_prior_std"])
            * exact.prior_scale_multiplier
            * exact.posterior_scale_ratio
        )
        frozen_posterior_std = (
            np.asarray(reconstruction["base_prior_std"])
            * float(candidate["prior_scale"])
            * float(candidate["posterior_ratio"])
        )

        validation = _role_arrays(
            original=original,
            role="validation",
            reconstruction=reconstruction,
            posterior_std=exact_posterior_std,
            clip_bound=float(candidate["clip_bound_scaled"]),
            seasonal_lag=energy.seasonal_lag,
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

        test_exact = _role_arrays(
            original=original,
            role="test",
            reconstruction=reconstruction,
            posterior_std=exact_posterior_std,
            clip_bound=float(candidate["clip_bound_scaled"]),
            seasonal_lag=energy.seasonal_lag,
        )
        test_frozen_scale = _role_arrays(
            original=original,
            role="test",
            reconstruction=reconstruction,
            posterior_std=frozen_posterior_std,
            clip_bound=float(candidate["clip_bound_scaled"]),
            seasonal_lag=energy.seasonal_lag,
        )
        metrics = _test_metrics(
            arrays=test_exact,
            settings=energy,
        )

        tracked_frozen_prediction = frozen_scale_result.predictions[
            zone
        ]["exact_clipped_bma_forecast"].to_numpy(dtype=float)
        frozen_scale_error = float(
            np.max(
                np.abs(
                    tracked_frozen_prediction
                    - test_frozen_scale["bma"]
                )
            )
        )
        if frozen_scale_error > reproduction_tolerance:
            raise RetrospectiveBMAError(
                f"Tracked frozen-scale BMA was not reproduced for "
                f"{zone}: {frozen_scale_error:.17g}."
            )

        exact_minus_frozen = (
            test_exact["bma"] - test_frozen_scale["bma"]
        )
        rmse_avoided = bool(
            not gate.gate_passed
            and metrics[
                "test_exact_bma_rmse_improvement_vs_fallback"
            ]
            < 0.0
        )
        score_avoided = bool(
            not gate.gate_passed
            and metrics[
                "test_exact_bma_score_improvement_vs_fallback"
            ]
            < 0.0
        )

        rows.append(
            {
                "status": POSTHOC_STATUS,
                "zone": zone,
                "frozen_deployment": str(decision["deployment"]),
                "frozen_selected_candidate_id": candidate_id,
                "family": str(candidate["family"]),
                "bma_gate_applicable": True,
                "frozen_prior_scale": float(
                    candidate["prior_scale"]
                ),
                "frozen_posterior_ratio": float(
                    candidate["posterior_ratio"]
                ),
                "exact_prior_scale": (
                    exact.prior_scale_multiplier
                ),
                "exact_posterior_ratio": (
                    exact.posterior_scale_ratio
                ),
                "posterior_scale_changed": bool(
                    exact.prior_scale_multiplier
                    != float(candidate["prior_scale"])
                    or exact.posterior_scale_ratio
                    != float(candidate["posterior_ratio"])
                ),
                "exact_certificate_untruncated": (
                    exact.certificate_untruncated
                ),
                **gate.as_dict(),
                **metrics,
                "mean_abs_exact_bma_minus_frozen_scale_bma": float(
                    np.mean(np.abs(exact_minus_frozen))
                ),
                "max_abs_exact_bma_minus_frozen_scale_bma": float(
                    np.max(np.abs(exact_minus_frozen))
                ),
                "max_frozen_scale_bma_reproduction_error": (
                    frozen_scale_error
                ),
                "abstention_action": (
                    "deploy_candidate"
                    if gate.gate_passed
                    else "seasonal_fallback"
                ),
                "abstention_avoided_test_rmse_deterioration": (
                    rmse_avoided
                ),
                "abstention_avoided_test_score_deterioration": (
                    score_avoided
                ),
                "selection_uses_test": False,
                "interpretation_uses_test_posthoc": True,
                "development_decision_modified": False,
            }
        )
        prediction_frames[zone] = pd.DataFrame(
            {
                "index": test_exact["indices"],
                "actual": test_exact["target"],
                "seasonal_fallback": test_exact["seasonal"],
                "frozen_scale_bma": test_frozen_scale["bma"],
                "exact_scale_bma": test_exact["bma"],
                "exact_minus_frozen_scale_bma": exact_minus_frozen,
                "analysis_status": POSTHOC_STATUS,
            }
        )

    return TetouanSelectedBMAGateRetrospectiveResult(
        summary=pd.DataFrame(rows),
        predictions=prediction_frames,
    )
