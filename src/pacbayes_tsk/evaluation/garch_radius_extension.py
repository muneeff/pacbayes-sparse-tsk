from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class RadiusExtensionError(ValueError):
    """Raised when GARCH radius-extension outputs are incomplete or invalid."""


@dataclass(frozen=True)
class RadiusAcceptanceThresholds:
    max_boundary_rate: float = 0.10
    max_failure_rate: float = 0.0
    max_capped_rate: float = 0.0
    max_mean_rmse_penalty: float = 0.02

    def validate(self) -> None:
        for name, value in {
            "max_boundary_rate": self.max_boundary_rate,
            "max_failure_rate": self.max_failure_rate,
            "max_capped_rate": self.max_capped_rate,
            "max_mean_rmse_penalty": self.max_mean_rmse_penalty,
        }.items():
            if not np.isfinite(value) or value < 0.0:
                raise RadiusExtensionError(
                    f"{name} must be finite and non-negative."
                )


def select_analysis_rows(
    test_results: pd.DataFrame,
    *,
    dataset: str = "garch",
    selection_mode: str = "uncapped_only",
    max_rules: int = 12,
) -> pd.DataFrame:
    required = {
        "source",
        "dataset",
        "series_id",
        "selection_mode",
        "max_rules",
        "status",
        "selected_radius",
        "radius_cap_reached",
        "rmse",
        "mase",
        "rule_count",
        "total_parameter_count",
    }
    missing = required.difference(test_results.columns)
    if missing:
        raise RadiusExtensionError(
            f"Test results are missing columns: {sorted(missing)}."
        )

    selected = test_results.loc[
        test_results["dataset"].astype(str).eq(str(dataset))
        & test_results["selection_mode"].astype(str).eq(str(selection_mode))
        & test_results["max_rules"].astype(int).eq(int(max_rules))
    ].copy()

    if selected.empty:
        raise RadiusExtensionError(
            "No rows match the requested dataset, selection mode, and max_rules."
        )

    duplicate_mask = selected.duplicated(
        subset=["source", "dataset", "series_id"],
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            selected.loc[
                duplicate_mask,
                ["source", "dataset", "series_id"],
            ]
            .head(10)
            .to_dict("records")
        )
        raise RadiusExtensionError(
            f"Selected test rows contain duplicate series: {examples}."
        )

    return selected.sort_values("series_id").reset_index(drop=True)


def build_radius_distribution(
    selected_rows: pd.DataFrame,
    *,
    maximum_radius: float,
) -> pd.DataFrame:
    if not np.isfinite(maximum_radius):
        raise RadiusExtensionError("maximum_radius must be finite.")

    successful = selected_rows.loc[
        selected_rows["status"].astype(str).eq("PASS")
    ].copy()
    if successful.empty:
        return pd.DataFrame(
            columns=[
                "selected_radius",
                "series",
                "rate",
                "at_upper_boundary",
            ]
        )

    counts = (
        successful.groupby("selected_radius", as_index=False)
        .agg(series=("series_id", "nunique"))
        .sort_values("selected_radius")
    )
    total = int(successful["series_id"].nunique())
    counts["rate"] = counts["series"] / total
    counts["at_upper_boundary"] = np.isclose(
        counts["selected_radius"].astype(float),
        float(maximum_radius),
        rtol=0.0,
        atol=1e-12,
    )
    return counts.reset_index(drop=True)


def compare_with_previous_grid(
    selected_rows: pd.DataFrame,
    previous_test: pd.DataFrame,
    *,
    dataset: str = "garch",
    previous_selection_mode: str = "uncapped_only",
    previous_max_rules: int = 12,
) -> pd.DataFrame:
    current = selected_rows.loc[
        selected_rows["status"].astype(str).eq("PASS")
    ].copy()

    required_previous = {
        "source",
        "dataset",
        "series_id",
        "selection_mode",
        "max_rules",
        "status",
        "selected_radius",
        "rmse",
        "mase",
        "rule_count",
        "total_parameter_count",
    }
    missing = required_previous.difference(previous_test.columns)
    if missing:
        raise RadiusExtensionError(
            f"Previous sensitivity table is missing columns: {sorted(missing)}."
        )

    previous = previous_test.loc[
        previous_test["dataset"].astype(str).eq(str(dataset))
        & previous_test["selection_mode"].astype(str).eq(
            str(previous_selection_mode)
        )
        & previous_test["max_rules"].astype(int).eq(int(previous_max_rules))
        & previous_test["status"].astype(str).eq("PASS")
    ].copy()

    previous = previous[
        [
            "source",
            "dataset",
            "series_id",
            "selected_radius",
            "rmse",
            "mase",
            "rule_count",
            "total_parameter_count",
        ]
    ].rename(
        columns={
            "selected_radius": "previous_selected_radius",
            "rmse": "previous_rmse",
            "mase": "previous_mase",
            "rule_count": "previous_rule_count",
            "total_parameter_count": "previous_total_parameter_count",
        }
    )

    merged = current.merge(
        previous,
        on=["source", "dataset", "series_id"],
        how="left",
        validate="one_to_one",
    )
    merged["rmse_ratio_to_previous"] = (
        merged["rmse"] / merged["previous_rmse"]
    )
    merged["mase_ratio_to_previous"] = (
        merged["mase"] / merged["previous_mase"]
    )
    merged["radius_change"] = (
        merged["selected_radius"] - merged["previous_selected_radius"]
    )
    merged["rule_change"] = (
        merged["rule_count"] - merged["previous_rule_count"]
    )
    merged["parameter_change"] = (
        merged["total_parameter_count"]
        - merged["previous_total_parameter_count"]
    )
    return merged.sort_values("series_id").reset_index(drop=True)


def build_boundary_summary(
    selected_rows: pd.DataFrame,
    *,
    maximum_radius: float,
    expected_series: int | None = None,
    comparison: pd.DataFrame | None = None,
) -> pd.DataFrame:
    total = int(len(selected_rows))
    pass_mask = selected_rows["status"].astype(str).eq("PASS")
    successful = selected_rows.loc[pass_mask].copy()
    pass_count = int(len(successful))
    failure_count = total - pass_count

    if expected_series is not None and total != int(expected_series):
        raise RadiusExtensionError(
            f"Expected {expected_series} selected rows; found {total}."
        )

    boundary_mask = np.isclose(
        successful["selected_radius"].astype(float),
        float(maximum_radius),
        rtol=0.0,
        atol=1e-12,
    )
    capped_mask = (
        successful["radius_cap_reached"]
        .fillna(False)
        .astype(bool)
    )

    mean_rmse_ratio = np.nan
    median_rmse_ratio = np.nan
    matched_previous = 0
    if comparison is not None and not comparison.empty:
        ratios = (
            comparison["rmse_ratio_to_previous"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        matched_previous = int(len(ratios))
        if not ratios.empty:
            mean_rmse_ratio = float(ratios.mean())
            median_rmse_ratio = float(ratios.median())

    record = {
        "dataset": (
            str(successful["dataset"].iloc[0])
            if pass_count
            else "garch"
        ),
        "selection_mode": (
            str(selected_rows["selection_mode"].iloc[0])
            if total
            else "uncapped_only"
        ),
        "max_rules": (
            int(selected_rows["max_rules"].iloc[0])
            if total
            else 12
        ),
        "maximum_radius": float(maximum_radius),
        "series": total,
        "pass_count": pass_count,
        "failure_count": failure_count,
        "failure_rate": (
            float(failure_count / total)
            if total
            else np.nan
        ),
        "upper_boundary_count": int(boundary_mask.sum()),
        "upper_boundary_rate": (
            float(boundary_mask.mean())
            if pass_count
            else np.nan
        ),
        "selected_cap_count": int(capped_mask.sum()),
        "selected_cap_rate": (
            float(capped_mask.mean())
            if pass_count
            else np.nan
        ),
        "mean_selected_radius": (
            float(successful["selected_radius"].mean())
            if pass_count
            else np.nan
        ),
        "median_selected_radius": (
            float(successful["selected_radius"].median())
            if pass_count
            else np.nan
        ),
        "mean_rmse": (
            float(successful["rmse"].mean())
            if pass_count
            else np.nan
        ),
        "median_rmse": (
            float(successful["rmse"].median())
            if pass_count
            else np.nan
        ),
        "mean_mase": (
            float(successful["mase"].mean())
            if pass_count
            else np.nan
        ),
        "median_mase": (
            float(successful["mase"].median())
            if pass_count
            else np.nan
        ),
        "mean_rule_count": (
            float(successful["rule_count"].mean())
            if pass_count
            else np.nan
        ),
        "median_rule_count": (
            float(successful["rule_count"].median())
            if pass_count
            else np.nan
        ),
        "mean_total_parameters": (
            float(successful["total_parameter_count"].mean())
            if pass_count
            else np.nan
        ),
        "matched_previous_series": matched_previous,
        "mean_rmse_ratio_to_previous": mean_rmse_ratio,
        "median_rmse_ratio_to_previous": median_rmse_ratio,
    }
    return pd.DataFrame([record])


def build_radius_recommendation(
    boundary_summary: pd.DataFrame,
    *,
    thresholds: RadiusAcceptanceThresholds,
) -> pd.DataFrame:
    thresholds.validate()
    if len(boundary_summary) != 1:
        raise RadiusExtensionError(
            "boundary_summary must contain exactly one row."
        )

    row = boundary_summary.iloc[0]
    mean_ratio = float(row["mean_rmse_ratio_to_previous"])
    rmse_penalty = (
        mean_ratio - 1.0
        if np.isfinite(mean_ratio)
        else np.nan
    )

    boundary_ok = bool(
        np.isfinite(row["upper_boundary_rate"])
        and float(row["upper_boundary_rate"])
        <= thresholds.max_boundary_rate
    )
    failures_ok = bool(
        np.isfinite(row["failure_rate"])
        and float(row["failure_rate"])
        <= thresholds.max_failure_rate
    )
    caps_ok = bool(
        np.isfinite(row["selected_cap_rate"])
        and float(row["selected_cap_rate"])
        <= thresholds.max_capped_rate
    )
    performance_ok = bool(
        np.isfinite(rmse_penalty)
        and rmse_penalty <= thresholds.max_mean_rmse_penalty
    )

    grid_sufficient = bool(
        boundary_ok
        and failures_ok
        and caps_ok
        and performance_ok
    )

    if grid_sufficient:
        decision = "ACCEPT_RADIUS_GRID"
        next_maximum_radius = float(row["maximum_radius"])
    else:
        decision = "EXTEND_RADIUS_GRID"
        next_maximum_radius = float(row["maximum_radius"]) + 1.0

    return pd.DataFrame(
        [
            {
                "dataset": str(row["dataset"]),
                "maximum_radius": float(row["maximum_radius"]),
                "upper_boundary_rate": float(row["upper_boundary_rate"]),
                "failure_rate": float(row["failure_rate"]),
                "selected_cap_rate": float(row["selected_cap_rate"]),
                "mean_rmse_penalty_vs_previous": rmse_penalty,
                "boundary_threshold": thresholds.max_boundary_rate,
                "failure_threshold": thresholds.max_failure_rate,
                "cap_threshold": thresholds.max_capped_rate,
                "rmse_penalty_threshold": (
                    thresholds.max_mean_rmse_penalty
                ),
                "boundary_ok": boundary_ok,
                "failures_ok": failures_ok,
                "caps_ok": caps_ok,
                "performance_ok": performance_ok,
                "grid_sufficient": grid_sufficient,
                "decision": decision,
                "suggested_next_maximum_radius": next_maximum_radius,
            }
        ]
    )
