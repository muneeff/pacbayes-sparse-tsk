from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pacbayes_tsk.data.energy_pjm import load_pjm_daily
from pacbayes_tsk.data.splits_v3 import ratio_split
from pacbayes_tsk.evaluation.clipped_gaussian_bma import clipped_gaussian_bma
from pacbayes_tsk.experiments.energy_case_study_v3 import (
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


POSTHOC_STATUS = "retrospective_posthoc_only"


class RetrospectiveBMAError(RuntimeError):
    """Raised when frozen PJM artifacts cannot be reproduced safely."""


@dataclass(frozen=True)
class PJMBMARetrospectiveResult:
    summary: pd.DataFrame
    predictions: dict[str, pd.DataFrame]


def _single_row(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    label: str,
) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise RetrospectiveBMAError(
            f"Expected exactly one {label} row, found {len(selected)}."
        )
    return selected.iloc[0]


def _relative_change(new_value: float, old_value: float) -> float:
    if old_value == 0.0:
        raise RetrospectiveBMAError("Cannot compute relative change from zero.")
    return float((new_value - old_value) / old_value)


def _improvement(model_value: float, baseline_value: float) -> float:
    if baseline_value == 0.0:
        raise RetrospectiveBMAError("Cannot compute improvement from zero baseline.")
    return float((baseline_value - model_value) / baseline_value)


def evaluate_pjm_bma_retrospective(
    *,
    case_data_path: str | Path,
    config_path: str | Path,
    frozen_results_dir: str | Path,
    reproduction_tolerance: float = 1e-8,
) -> PJMBMARetrospectiveResult:
    """Reconstruct frozen PJM point forecasts and compare exact clipped BMA.

    This function is strictly retrospective. It reads the pre-test decisions and
    frozen outputs from the completed V3.4 PJM run. It does not reselect models,
    change the original gate, or write into the confirmatory results directory.
    """

    if not np.isfinite(reproduction_tolerance) or reproduction_tolerance <= 0.0:
        raise RetrospectiveBMAError(
            "reproduction_tolerance must be finite and positive."
        )

    settings = PJMConfirmatorySettings.from_yaml(config_path)
    case_data = load_pjm_daily(case_data_path)
    split = ratio_split(len(case_data), settings.split_fractions)
    frozen_results = Path(frozen_results_dir)

    completion = frozen_results / "COMPLETED.json"
    if not completion.is_file():
        raise RetrospectiveBMAError(
            f"Missing frozen completion marker: {completion}"
        )

    decisions = pd.read_csv(frozen_results / "pjm_pretest_decisions.csv")
    rows: list[dict[str, object]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}

    for region in settings.regions:
        decision = _single_row(
            decisions,
            decisions["region"].eq(region),
            label=f"pre-test decision for {region}",
        )
        candidate_id = str(decision["selected_candidate_id"])
        if not candidate_id or candidate_id.lower() == "nan":
            raise RetrospectiveBMAError(
                f"Region {region} has no selected candidate to reconstruct."
            )

        candidates = pd.read_csv(
            frozen_results / f"{region}_candidates.csv"
        )
        candidate = _single_row(
            candidates,
            candidates["candidate_id"].eq(candidate_id),
            label=f"candidate {candidate_id}",
        )

        original = case_data[region].to_numpy(dtype=float)
        prior = original[split.labels == "prior"]
        scaling_mean = float(prior.mean())
        scaling_std = float(prior.std(ddof=0))
        if not np.isfinite(scaling_std) or scaling_std <= 0.0:
            raise RetrospectiveBMAError(
                f"Invalid prior scaling standard deviation for {region}."
            )

        values = (original - scaling_mean) / scaling_std
        lag = int(candidate["lag"])
        features, target_scaled, labels, _ = _lagged(
            values,
            split.labels,
            lag,
        )
        masks = _role_masks(labels)
        radius = (
            None
            if pd.isna(candidate["radius"])
            else float(candidate["radius"])
        )
        designs, _, _, rule_cap_reached = _build_designs(
            family=str(candidate["family"]),
            features=features,
            masks=masks,
            settings=settings,
            radius=radius,
        )
        if rule_cap_reached:
            raise RetrospectiveBMAError(
                f"Frozen selected candidate unexpectedly reaches rule cap for {region}."
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
        base_std = _base_prior_std(
            designs["prior"],
            target_scaled[masks["prior"]],
            prior_mean,
            prior_inverse_diag,
        )
        posterior_std = (
            base_std
            * float(candidate["prior_scale"])
            * float(candidate["posterior_ratio"])
        )
        clip_bound = float(candidate["clip_bound_scaled"])

        point_scaled = np.clip(
            designs["test"] @ posterior_mean,
            -clip_bound,
            clip_bound,
        )
        bma_scaled = clipped_gaussian_bma(
            designs["test"],
            posterior_mean,
            posterior_std,
            clip_bound=clip_bound,
        )
        point_forecast = point_scaled * scaling_std + scaling_mean
        bma_forecast = bma_scaled * scaling_std + scaling_mean
        target = target_scaled[masks["test"]] * scaling_std + scaling_mean

        stored_path = (
            frozen_results
            / "predictions"
            / f"{region}_test_predictions.csv"
        )
        stored = pd.read_csv(stored_path)
        if len(stored) != len(target):
            raise RetrospectiveBMAError(
                f"Stored prediction length mismatch for {region}: "
                f"{len(stored)} != {len(target)}."
            )

        point_reproduction_error = float(
            np.max(
                np.abs(
                    point_forecast
                    - stored["deployed_forecast"].to_numpy(dtype=float)
                )
            )
        )
        target_reproduction_error = float(
            np.max(
                np.abs(
                    target
                    - stored["actual"].to_numpy(dtype=float)
                )
            )
        )
        if point_reproduction_error > reproduction_tolerance:
            raise RetrospectiveBMAError(
                f"Frozen point forecasts were not reproduced for {region}: "
                f"{point_reproduction_error:.17g}."
            )
        if target_reproduction_error > reproduction_tolerance:
            raise RetrospectiveBMAError(
                f"Frozen targets were not reproduced for {region}: "
                f"{target_reproduction_error:.17g}."
            )

        seasonal = stored["seasonal_naive_7d"].to_numpy(dtype=float)
        baseline_rmse, baseline_mae = _rmse_mae(target, seasonal)
        point_rmse, point_mae = _rmse_mae(target, point_forecast)
        bma_rmse, bma_mae = _rmse_mae(target, bma_forecast)
        baseline_score, _ = _weighted_cost(
            target,
            seasonal,
            under_weight=settings.underforecast_weight,
            over_weight=settings.overforecast_weight,
        )
        point_score, _ = _weighted_cost(
            target,
            point_forecast,
            under_weight=settings.underforecast_weight,
            over_weight=settings.overforecast_weight,
        )
        bma_score, _ = _weighted_cost(
            target,
            bma_forecast,
            under_weight=settings.underforecast_weight,
            over_weight=settings.overforecast_weight,
        )

        rows.append(
            {
                "status": POSTHOC_STATUS,
                "region": region,
                "candidate_id": candidate_id,
                "family": str(candidate["family"]),
                "max_point_reproduction_error": point_reproduction_error,
                "max_target_reproduction_error": target_reproduction_error,
                "baseline_rmse": baseline_rmse,
                "point_rmse": point_rmse,
                "bma_rmse": bma_rmse,
                "point_rmse_improvement_vs_fallback": _improvement(
                    point_rmse,
                    baseline_rmse,
                ),
                "bma_rmse_improvement_vs_fallback": _improvement(
                    bma_rmse,
                    baseline_rmse,
                ),
                "bma_rmse_change_vs_point": _relative_change(
                    bma_rmse,
                    point_rmse,
                ),
                "baseline_mae": baseline_mae,
                "point_mae": point_mae,
                "bma_mae": bma_mae,
                "baseline_asymmetric_score": baseline_score,
                "point_asymmetric_score": point_score,
                "bma_asymmetric_score": bma_score,
                "point_score_improvement_vs_fallback": _improvement(
                    point_score,
                    baseline_score,
                ),
                "bma_score_improvement_vs_fallback": _improvement(
                    bma_score,
                    baseline_score,
                ),
                "bma_score_change_vs_point": _relative_change(
                    bma_score,
                    point_score,
                ),
                "mean_abs_bma_minus_point": float(
                    np.mean(np.abs(bma_forecast - point_forecast))
                ),
                "confirmatory_decision_modified": False,
            }
        )

        output_predictions = stored.copy()
        output_predictions["reconstructed_point_forecast"] = point_forecast
        output_predictions["exact_clipped_bma_forecast"] = bma_forecast
        output_predictions["bma_minus_point"] = (
            bma_forecast - point_forecast
        )
        output_predictions["analysis_status"] = POSTHOC_STATUS
        prediction_frames[region] = output_predictions

    summary = pd.DataFrame(rows)
    return PJMBMARetrospectiveResult(
        summary=summary,
        predictions=prediction_frames,
    )
