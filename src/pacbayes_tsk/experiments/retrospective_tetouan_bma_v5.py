from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pacbayes_tsk.data.energy_tetouan import load_two_hour_tetouan
from pacbayes_tsk.data.splits_v3 import ratio_split
from pacbayes_tsk.evaluation.clipped_gaussian_bma import clipped_gaussian_bma
from pacbayes_tsk.experiments.energy_case_study_v3 import (
    EnergySettings,
    _base_prior_std,
    _build_designs,
    _lagged,
    _ridge_solution,
    _rmse_mae,
    _role_masks,
    _seasonal_predictions,
    _weighted_cost,
)
from pacbayes_tsk.experiments.retrospective_bma_v5 import (
    POSTHOC_STATUS,
    RetrospectiveBMAError,
    _relative_change,
    _single_row,
)


@dataclass(frozen=True)
class TetouanBMARetrospectiveResult:
    summary: pd.DataFrame
    predictions: dict[str, pd.DataFrame]


def evaluate_tetouan_bma_retrospective(
    *,
    case_data_path: str | Path,
    config_path: str | Path,
    frozen_results_dir: str | Path,
    reproduction_tolerance: float = 1e-8,
) -> TetouanBMARetrospectiveResult:
    """Compare exact clipped BMA with frozen Tetouan deployment forecasts.

    This is strictly retrospective. The completed V3 development decisions
    remain unchanged. A zone deployed with the seasonal fallback has no
    selected posterior, so its BMA output is explicitly not applicable.
    """

    if (
        not np.isfinite(reproduction_tolerance)
        or reproduction_tolerance <= 0.0
    ):
        raise RetrospectiveBMAError(
            "reproduction_tolerance must be finite and positive."
        )

    settings = EnergySettings.from_yaml(config_path)
    case_data = load_two_hour_tetouan(case_data_path)
    split = ratio_split(len(case_data), settings.split_fractions)
    frozen_results = Path(frozen_results_dir)

    completion = frozen_results / "COMPLETED.json"
    if not completion.is_file():
        raise RetrospectiveBMAError(
            f"Missing frozen completion marker: {completion}"
        )

    decisions = pd.read_csv(
        frozen_results / "energy_deployment_decisions.csv"
    )
    rows: list[dict[str, object]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}

    for zone in settings.zones:
        decision = _single_row(
            decisions,
            decisions["zone"].eq(zone),
            label=f"deployment decision for {zone}",
        )
        deployment = str(decision["deployment"])
        stored = pd.read_csv(
            frozen_results
            / "predictions"
            / f"{zone}_test_predictions.csv"
        )

        original = case_data[zone].to_numpy(dtype=float)
        _, seasonal_target, seasonal_forecast = _seasonal_predictions(
            original,
            split.labels,
            "test",
            settings.seasonal_lag,
        )
        if len(stored) != len(seasonal_target):
            raise RetrospectiveBMAError(
                f"Stored prediction length mismatch for {zone}: "
                f"{len(stored)} != {len(seasonal_target)}."
            )

        target_error = float(
            np.max(
                np.abs(
                    seasonal_target
                    - stored["actual"].to_numpy(dtype=float)
                )
            )
        )
        seasonal_error = float(
            np.max(
                np.abs(
                    seasonal_forecast
                    - stored["seasonal_naive_24h"].to_numpy(dtype=float)
                )
            )
        )
        if target_error > reproduction_tolerance:
            raise RetrospectiveBMAError(
                f"Frozen targets were not reproduced for {zone}: "
                f"{target_error:.17g}."
            )
        if seasonal_error > reproduction_tolerance:
            raise RetrospectiveBMAError(
                f"Frozen seasonal forecasts were not reproduced for {zone}: "
                f"{seasonal_error:.17g}."
            )

        baseline_rmse, baseline_mae = _rmse_mae(
            seasonal_target,
            seasonal_forecast,
        )
        baseline_score, _ = _weighted_cost(
            seasonal_target,
            seasonal_forecast,
            under_weight=settings.underforecast_weight,
            over_weight=settings.overforecast_weight,
        )

        if deployment == "seasonal_naive_24h":
            point_forecast = seasonal_forecast
            point_error = float(
                np.max(
                    np.abs(
                        point_forecast
                        - stored["deployed_forecast"].to_numpy(dtype=float)
                    )
                )
            )
            if point_error > reproduction_tolerance:
                raise RetrospectiveBMAError(
                    f"Frozen fallback deployment was not reproduced for "
                    f"{zone}: {point_error:.17g}."
                )

            rows.append(
                {
                    "status": POSTHOC_STATUS,
                    "zone": zone,
                    "deployment": deployment,
                    "candidate_id": None,
                    "family": None,
                    "bma_applicable": False,
                    "reason": "frozen_fallback_has_no_selected_posterior",
                    "max_point_reproduction_error": point_error,
                    "max_target_reproduction_error": target_error,
                    "max_seasonal_reproduction_error": seasonal_error,
                    "baseline_rmse": baseline_rmse,
                    "point_rmse": baseline_rmse,
                    "bma_rmse": np.nan,
                    "bma_rmse_change_vs_point": np.nan,
                    "baseline_mae": baseline_mae,
                    "point_mae": baseline_mae,
                    "bma_mae": np.nan,
                    "baseline_asymmetric_score": baseline_score,
                    "point_asymmetric_score": baseline_score,
                    "bma_asymmetric_score": np.nan,
                    "bma_score_change_vs_point": np.nan,
                    "mean_abs_bma_minus_point": np.nan,
                    "development_decision_modified": False,
                }
            )

            output_predictions = stored.copy()
            output_predictions[
                "reconstructed_point_forecast"
            ] = point_forecast
            output_predictions["exact_clipped_bma_forecast"] = np.nan
            output_predictions["bma_minus_point"] = np.nan
            output_predictions["bma_applicable"] = False
            output_predictions["analysis_status"] = POSTHOC_STATUS
            prediction_frames[zone] = output_predictions
            continue

        candidate_id = str(decision["selected_candidate_id"])
        if not candidate_id or candidate_id.lower() == "nan":
            raise RetrospectiveBMAError(
                f"Zone {zone} has no selected candidate to reconstruct."
            )

        candidates = pd.read_csv(
            frozen_results / f"{zone}_candidates.csv"
        )
        candidate = _single_row(
            candidates,
            candidates["candidate_id"].eq(candidate_id),
            label=f"candidate {candidate_id}",
        )

        prior = original[split.labels == "prior"]
        scaling_mean = float(prior.mean())
        scaling_std = float(prior.std(ddof=0))
        if not np.isfinite(scaling_std) or scaling_std <= 0.0:
            raise RetrospectiveBMAError(
                f"Invalid prior scaling standard deviation for {zone}."
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
                f"Frozen selected candidate unexpectedly reaches rule cap "
                f"for {zone}."
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

        point_error = float(
            np.max(
                np.abs(
                    point_forecast
                    - stored["deployed_forecast"].to_numpy(dtype=float)
                )
            )
        )
        model_target_error = float(
            np.max(
                np.abs(
                    target
                    - stored["actual"].to_numpy(dtype=float)
                )
            )
        )
        if point_error > reproduction_tolerance:
            raise RetrospectiveBMAError(
                f"Frozen point forecasts were not reproduced for {zone}: "
                f"{point_error:.17g}."
            )
        if model_target_error > reproduction_tolerance:
            raise RetrospectiveBMAError(
                f"Model-aligned targets were not reproduced for {zone}: "
                f"{model_target_error:.17g}."
            )

        point_rmse, point_mae = _rmse_mae(target, point_forecast)
        bma_rmse, bma_mae = _rmse_mae(target, bma_forecast)
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
                "zone": zone,
                "deployment": deployment,
                "candidate_id": candidate_id,
                "family": str(candidate["family"]),
                "bma_applicable": True,
                "reason": "selected_posterior_reconstructed",
                "max_point_reproduction_error": point_error,
                "max_target_reproduction_error": max(
                    target_error,
                    model_target_error,
                ),
                "max_seasonal_reproduction_error": seasonal_error,
                "baseline_rmse": baseline_rmse,
                "point_rmse": point_rmse,
                "bma_rmse": bma_rmse,
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
                "bma_score_change_vs_point": _relative_change(
                    bma_score,
                    point_score,
                ),
                "mean_abs_bma_minus_point": float(
                    np.mean(np.abs(bma_forecast - point_forecast))
                ),
                "development_decision_modified": False,
            }
        )

        output_predictions = stored.copy()
        output_predictions[
            "reconstructed_point_forecast"
        ] = point_forecast
        output_predictions[
            "exact_clipped_bma_forecast"
        ] = bma_forecast
        output_predictions["bma_minus_point"] = (
            bma_forecast - point_forecast
        )
        output_predictions["bma_applicable"] = True
        output_predictions["analysis_status"] = POSTHOC_STATUS
        prediction_frames[zone] = output_predictions

    return TetouanBMARetrospectiveResult(
        summary=pd.DataFrame(rows),
        predictions=prediction_frames,
    )
