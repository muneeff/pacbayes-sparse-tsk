from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SELECTION_MODES = (
    "unconstrained",
    "uncapped_only",
)


class SensitivitySelectionError(RuntimeError):
    """Raised when a sensitivity candidate cannot be selected safely."""


def select_candidate(
    validation: pd.DataFrame,
    *,
    allow_capped: bool,
) -> pd.Series:
    """
    Select a validation candidate using deterministic tie breaking.

    Capped candidates are retained in the audit table. They can be excluded
    from the scientifically admissible selection with allow_capped=False.
    """
    required = {
        "status",
        "rmse",
        "mae",
        "total_parameter_count",
        "rule_count",
        "radius",
        "ridge_alpha",
        "lag_order",
        "radius_cap_reached",
    }
    missing = required.difference(validation.columns)
    if missing:
        raise SensitivitySelectionError(
            f"Validation table is missing columns: {sorted(missing)}."
        )

    successful = validation.loc[
        validation["status"].astype(str).eq("PASS")
    ].copy()

    if not allow_capped:
        successful = successful.loc[
            ~successful["radius_cap_reached"].astype(bool)
        ].copy()

    if successful.empty:
        mode = "unconstrained" if allow_capped else "uncapped-only"
        raise SensitivitySelectionError(
            f"No successful {mode} validation candidate is available."
        )

    finite = np.isfinite(
        successful[
            [
                "rmse",
                "mae",
                "total_parameter_count",
                "rule_count",
                "radius",
                "ridge_alpha",
                "lag_order",
            ]
        ].to_numpy(dtype=float)
    ).all(axis=1)
    successful = successful.loc[finite].copy()
    if successful.empty:
        raise SensitivitySelectionError(
            "No candidate has finite selection fields."
        )

    # Prefer the smallest radius after equal performance and complexity.
    # Larger radii can collapse to the same one-rule model, so preferring the
    # largest value creates a false upper-boundary signal.
    ordered = successful.sort_values(
        [
            "rmse",
            "mae",
            "total_parameter_count",
            "rule_count",
            "radius",
            "ridge_alpha",
            "lag_order",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
        kind="stable",
    )
    return ordered.iloc[0]


def completed_series_keys(
    test_results: pd.DataFrame,
    *,
    max_rules_values: Iterable[int],
) -> set[tuple[str, str, str]]:
    """
    A series is complete only when every max-rules value has both selection
    modes recorded. PASS is not required, because an explicit no-eligible-
    candidate row is a legitimate sensitivity outcome.
    """
    if test_results.empty:
        return set()

    required_pairs = {
        (int(max_rules), mode)
        for max_rules in max_rules_values
        for mode in SELECTION_MODES
    }

    completed: set[tuple[str, str, str]] = set()
    grouped = test_results.groupby(
        ["source", "dataset", "series_id"],
        dropna=False,
        sort=False,
    )
    for key, group in grouped:
        observed_pairs = {
            (int(row.max_rules), str(row.selection_mode))
            for row in group.itertuples(index=False)
        }
        if required_pairs.issubset(observed_pairs):
            completed.add(tuple(str(value) for value in key))
    return completed


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def build_setting_summary(
    test_results: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "source",
        "dataset",
        "series_id",
        "max_rules",
        "selection_mode",
        "status",
        "radius_cap_reached",
        "rmse",
        "mase",
        "rule_count",
        "total_parameter_count",
    }
    missing = required.difference(test_results.columns)
    if missing:
        raise ValueError(
            f"Test table is missing columns: {sorted(missing)}."
        )

    records: list[dict[str, float | int | str]] = []
    grouped = test_results.groupby(
        [
            "source",
            "dataset",
            "max_rules",
            "selection_mode",
        ],
        dropna=False,
        sort=True,
    )

    for (
        source,
        dataset,
        max_rules,
        selection_mode,
    ), group in grouped:
        successful = group.loc[
            group["status"].astype(str).eq("PASS")
        ].copy()
        total = int(len(group))
        pass_count = int(len(successful))
        failure_count = total - pass_count
        capped_count = int(
            successful["radius_cap_reached"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        records.append(
            {
                "source": str(source),
                "dataset": str(dataset),
                "max_rules": int(max_rules),
                "selection_mode": str(selection_mode),
                "series": total,
                "pass_count": pass_count,
                "failure_count": failure_count,
                "failure_rate": _safe_rate(failure_count, total),
                "selected_cap_count": capped_count,
                "selected_cap_rate": _safe_rate(capped_count, pass_count),
                "mean_rmse": float(successful["rmse"].mean())
                if pass_count
                else np.nan,
                "median_rmse": float(successful["rmse"].median())
                if pass_count
                else np.nan,
                "mean_mase": float(successful["mase"].mean())
                if pass_count
                else np.nan,
                "median_mase": float(successful["mase"].median())
                if pass_count
                else np.nan,
                "mean_rule_count": float(successful["rule_count"].mean())
                if pass_count
                else np.nan,
                "median_rule_count": float(successful["rule_count"].median())
                if pass_count
                else np.nan,
                "mean_total_parameters": float(
                    successful["total_parameter_count"].mean()
                )
                if pass_count
                else np.nan,
                "median_total_parameters": float(
                    successful["total_parameter_count"].median()
                )
                if pass_count
                else np.nan,
            }
        )

    return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class AcceptanceThresholds:
    max_dataset_selected_cap_rate: float = 0.10
    max_dataset_uncapped_failure_rate: float = 0.0
    max_dataset_uncapped_rmse_penalty: float = 0.02

    def validate(self) -> None:
        for name, value in {
            "max_dataset_selected_cap_rate": self.max_dataset_selected_cap_rate,
            "max_dataset_uncapped_failure_rate": self.max_dataset_uncapped_failure_rate,
            "max_dataset_uncapped_rmse_penalty": self.max_dataset_uncapped_rmse_penalty,
        }.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")


def build_recommendation_table(
    setting_summary: pd.DataFrame,
    *,
    thresholds: AcceptanceThresholds,
) -> pd.DataFrame:
    """
    Compare natural validation selection against uncapped-only selection.

    Acceptance requires, for every dataset:
      1. Natural selected cap rate <= configured threshold.
      2. Uncapped-only selection failure rate <= threshold.
      3. Mean RMSE penalty of uncapped-only versus natural selection <= threshold.

    The smallest acceptable max_rules value is the recommended setting.
    """
    thresholds.validate()

    required = {
        "source",
        "dataset",
        "max_rules",
        "selection_mode",
        "failure_rate",
        "selected_cap_rate",
        "mean_rmse",
        "mean_rule_count",
        "mean_total_parameters",
    }
    missing = required.difference(setting_summary.columns)
    if missing:
        raise ValueError(
            f"Setting summary is missing columns: {sorted(missing)}."
        )

    unconstrained = setting_summary.loc[
        setting_summary["selection_mode"].eq("unconstrained")
    ].copy()
    uncapped = setting_summary.loc[
        setting_summary["selection_mode"].eq("uncapped_only")
    ].copy()

    merge_keys = [
        "source",
        "dataset",
        "max_rules",
    ]
    merged = unconstrained.merge(
        uncapped,
        on=merge_keys,
        suffixes=("_natural", "_uncapped"),
        how="outer",
        validate="one_to_one",
    )

    merged["uncapped_rmse_penalty"] = (
        merged["mean_rmse_uncapped"]
        / merged["mean_rmse_natural"]
        - 1.0
    )

    records: list[dict[str, float | int | bool]] = []
    for max_rules, group in merged.groupby(
        "max_rules",
        sort=True,
    ):
        natural_cap = group["selected_cap_rate_natural"]
        uncapped_failure = group["failure_rate_uncapped"]
        rmse_penalty = group["uncapped_rmse_penalty"]

        max_cap = float(natural_cap.max())
        max_failure = float(uncapped_failure.max())
        max_penalty = float(rmse_penalty.max())

        acceptable = bool(
            np.isfinite(max_cap)
            and np.isfinite(max_failure)
            and np.isfinite(max_penalty)
            and max_cap
            <= thresholds.max_dataset_selected_cap_rate
            and max_failure
            <= thresholds.max_dataset_uncapped_failure_rate
            and max_penalty
            <= thresholds.max_dataset_uncapped_rmse_penalty
        )

        records.append(
            {
                "max_rules": int(max_rules),
                "datasets": int(group["dataset"].nunique()),
                "max_dataset_natural_selected_cap_rate": max_cap,
                "max_dataset_uncapped_failure_rate": max_failure,
                "max_dataset_uncapped_rmse_penalty": max_penalty,
                "mean_uncapped_rule_count": float(
                    group["mean_rule_count_uncapped"].mean()
                ),
                "mean_uncapped_total_parameters": float(
                    group["mean_total_parameters_uncapped"].mean()
                ),
                "acceptable": acceptable,
            }
        )

    output = pd.DataFrame.from_records(records).sort_values("max_rules")
    acceptable_values = output.loc[
        output["acceptable"], "max_rules"
    ]
    recommended = (
        int(acceptable_values.min())
        if not acceptable_values.empty
        else None
    )
    output["recommended"] = (
        output["max_rules"].eq(recommended)
        if recommended is not None
        else False
    )
    return output.reset_index(drop=True)
