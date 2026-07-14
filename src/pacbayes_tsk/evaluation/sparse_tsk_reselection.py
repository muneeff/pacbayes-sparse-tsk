from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class ReselectionError(ValueError):
    """Raised when the consolidated Sparse TSK reselection is invalid."""


CANDIDATE_KEY = [
    "source",
    "dataset",
    "series_id",
    "lag_order",
    "radius",
    "ridge_alpha",
]


@dataclass(frozen=True)
class CandidateSourceSpec:
    name: str
    dataset: str
    expected_rows: int


def coerce_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(
        set(normalized).difference(mapping)
    )
    if unknown:
        raise ReselectionError(
            f"Invalid Boolean values: {unknown}."
        )
    return normalized.map(mapping).astype(bool)


def validate_candidate_columns(frame: pd.DataFrame, *, label: str) -> None:
    required = {
        *CANDIDATE_KEY,
        "status",
        "radius_cap_reached",
        "rmse",
        "mae",
        "rule_count",
        "total_parameter_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ReselectionError(
            f"{label} is missing columns: {sorted(missing)}."
        )


def _prepare_candidate_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    validation_source: str,
) -> pd.DataFrame:
    validate_candidate_columns(frame, label=label)
    output = frame.copy()
    output["source"] = output["source"].astype(str)
    output["dataset"] = output["dataset"].astype(str)
    output["series_id"] = output["series_id"].astype(str)
    output["lag_order"] = output["lag_order"].astype(int)
    output["radius"] = output["radius"].astype(float)
    output["ridge_alpha"] = output["ridge_alpha"].astype(float)
    output["radius_cap_reached"] = coerce_boolean(
        output["radius_cap_reached"]
    )
    output["validation_source"] = validation_source
    return output


def build_consolidated_candidate_pool(
    baseline_validation: pd.DataFrame,
    garch_extension_validation: pd.DataFrame,
    structural_extension_validation: pd.DataFrame,
    *,
    garch_expected_rows: int = 6500,
    structural_expected_rows: int = 4000,
) -> pd.DataFrame:
    baseline = _prepare_candidate_frame(
        baseline_validation,
        label="baseline validation",
        validation_source="baseline_grid",
    )
    garch = _prepare_candidate_frame(
        garch_extension_validation,
        label="GARCH extension validation",
        validation_source="garch_radius_to_5",
    )
    structural = _prepare_candidate_frame(
        structural_extension_validation,
        label="structural-break extension validation",
        validation_source="structural_radius_to_2_5",
    )

    garch = garch.loc[
        garch["source"].eq("synthetic")
        & garch["dataset"].eq("garch")
    ].copy()
    if "max_rules" in garch.columns:
        garch = garch.loc[
            garch["max_rules"].astype(int).eq(12)
        ].copy()

    structural = structural.loc[
        structural["source"].eq("synthetic")
        & structural["dataset"].eq("structural_break")
    ].copy()
    if "max_rules" in structural.columns:
        structural = structural.loc[
            structural["max_rules"].astype(int).eq(12)
        ].copy()

    if len(garch) != int(garch_expected_rows):
        raise ReselectionError(
            "Unexpected GARCH extension row count: "
            f"expected {garch_expected_rows}, found {len(garch)}."
        )
    if len(structural) != int(structural_expected_rows):
        raise ReselectionError(
            "Unexpected structural-break extension row count: "
            f"expected {structural_expected_rows}, found {len(structural)}."
        )

    retained_baseline = baseline.loc[
        ~(
            baseline["source"].eq("synthetic")
            & baseline["dataset"].isin(
                ["garch", "structural_break"]
            )
        )
    ].copy()

    pool = pd.concat(
        [retained_baseline, garch, structural],
        ignore_index=True,
        sort=False,
    )

    duplicate_mask = pool.duplicated(
        subset=CANDIDATE_KEY,
        keep=False,
    )
    if duplicate_mask.any():
        examples = (
            pool.loc[
                duplicate_mask,
                CANDIDATE_KEY + ["validation_source"],
            ]
            .head(20)
            .to_dict("records")
        )
        raise ReselectionError(
            f"Consolidated candidate pool has duplicate candidates: {examples}."
        )

    return pool.sort_values(CANDIDATE_KEY).reset_index(drop=True)


def select_candidate_smallest_radius(
    validation: pd.DataFrame,
    *,
    allow_capped: bool = False,
) -> pd.Series:
    validate_candidate_columns(
        validation,
        label="series validation candidates",
    )
    successful = validation.loc[
        validation["status"].astype(str).eq("PASS")
    ].copy()
    successful["radius_cap_reached"] = coerce_boolean(
        successful["radius_cap_reached"]
    )

    if not allow_capped:
        successful = successful.loc[
            ~successful["radius_cap_reached"]
        ].copy()

    if successful.empty:
        mode = "unconstrained" if allow_capped else "uncapped_only"
        raise ReselectionError(
            f"No successful candidate is available for selection mode {mode}."
        )

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


def build_eligibility_audit(
    candidate_pool: pd.DataFrame,
) -> pd.DataFrame:
    validate_candidate_columns(
        candidate_pool,
        label="consolidated candidate pool",
    )
    frame = candidate_pool.copy()
    frame["pass"] = frame["status"].astype(str).eq("PASS")
    frame["capped"] = coerce_boolean(
        frame["radius_cap_reached"]
    )
    frame["eligible_uncapped"] = frame["pass"] & ~frame["capped"]

    audit = (
        frame.groupby(
            ["source", "dataset", "series_id"],
            as_index=False,
        )
        .agg(
            validation_candidates=("series_id", "size"),
            pass_candidates=("pass", "sum"),
            capped_pass_candidates=(
                "capped",
                lambda values: int(
                    values[
                        frame.loc[values.index, "pass"]
                    ].sum()
                ),
            ),
            uncapped_pass_candidates=("eligible_uncapped", "sum"),
            maximum_candidate_radius=("radius", "max"),
            validation_source_count=("validation_source", "nunique"),
        )
    )
    audit["has_uncapped_candidate"] = (
        audit["uncapped_pass_candidates"] > 0
    )
    return audit.sort_values(
        ["source", "dataset", "series_id"]
    ).reset_index(drop=True)


def compare_old_and_new(
    new_results: pd.DataFrame,
    old_results: pd.DataFrame,
) -> pd.DataFrame:
    required_new = {
        "source",
        "dataset",
        "series_id",
        "status",
        "selected_lag",
        "selected_radius",
        "selected_ridge_alpha",
        "selected_validation_rmse",
        "selected_validation_mae",
        "rule_count",
        "total_parameter_count",
        "rmse",
        "mae",
        "mase",
    }
    required_old = required_new.difference({"selected_validation_mae"})
    missing_new = required_new.difference(new_results.columns)
    missing_old = required_old.difference(old_results.columns)
    if missing_new:
        raise ReselectionError(
            f"New test results are missing columns: {sorted(missing_new)}."
        )
    if missing_old:
        raise ReselectionError(
            f"Old test results are missing columns: {sorted(missing_old)}."
        )

    current = new_results.loc[
        new_results["status"].astype(str).eq("PASS")
    ].copy()
    old = old_results.loc[
        old_results["status"].astype(str).eq("PASS")
    ].copy()

    old_columns = [
        "source",
        "dataset",
        "series_id",
        "selected_lag",
        "selected_radius",
        "selected_ridge_alpha",
        "selected_validation_rmse",
        "rule_count",
        "total_parameter_count",
        "rmse",
        "mae",
        "mase",
    ]
    if "selected_validation_mae" in old.columns:
        old_columns.append("selected_validation_mae")

    old = old[old_columns].rename(
        columns={
            "selected_lag": "old_selected_lag",
            "selected_radius": "old_selected_radius",
            "selected_ridge_alpha": "old_selected_ridge_alpha",
            "selected_validation_rmse": "old_validation_rmse",
            "selected_validation_mae": "old_validation_mae",
            "rule_count": "old_rule_count",
            "total_parameter_count": "old_total_parameter_count",
            "rmse": "old_test_rmse",
            "mae": "old_test_mae",
            "mase": "old_test_mase",
        }
    )

    comparison = current.merge(
        old,
        on=["source", "dataset", "series_id"],
        how="left",
        validate="one_to_one",
    )
    comparison["lag_change"] = (
        comparison["selected_lag"] - comparison["old_selected_lag"]
    )
    comparison["radius_change"] = (
        comparison["selected_radius"] - comparison["old_selected_radius"]
    )
    comparison["alpha_change"] = (
        comparison["selected_ridge_alpha"]
        - comparison["old_selected_ridge_alpha"]
    )
    comparison["validation_rmse_change"] = (
        comparison["selected_validation_rmse"]
        - comparison["old_validation_rmse"]
    )
    if "old_validation_mae" in comparison.columns:
        comparison["validation_mae_change"] = (
            comparison["selected_validation_mae"]
            - comparison["old_validation_mae"]
        )
    else:
        comparison["validation_mae_change"] = np.nan

    comparison["rule_change"] = (
        comparison["rule_count"] - comparison["old_rule_count"]
    )
    comparison["parameter_change"] = (
        comparison["total_parameter_count"]
        - comparison["old_total_parameter_count"]
    )
    comparison["test_rmse_ratio_to_old"] = (
        comparison["rmse"] / comparison["old_test_rmse"]
    )
    comparison["test_mase_ratio_to_old"] = (
        comparison["mase"] / comparison["old_test_mase"]
    )
    comparison["selection_changed"] = (
        comparison["selected_lag"].ne(comparison["old_selected_lag"])
        | comparison["selected_radius"].ne(comparison["old_selected_radius"])
        | comparison["selected_ridge_alpha"].ne(
            comparison["old_selected_ridge_alpha"]
        )
    )

    tolerance = 1e-12
    comparison["exact_validation_plateau"] = (
        comparison["validation_rmse_change"].abs().le(tolerance)
        & (
            comparison["validation_mae_change"].isna()
            | comparison["validation_mae_change"].abs().le(tolerance)
        )
        & comparison["rule_change"].eq(0)
        & comparison["parameter_change"].eq(0)
    )
    comparison["radius_reduced_on_plateau"] = (
        comparison["selection_changed"]
        & comparison["radius_change"].lt(0.0)
        & comparison["exact_validation_plateau"]
    )
    return comparison.sort_values(
        ["source", "dataset", "series_id"]
    ).reset_index(drop=True)


def summarize_selection_changes(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()

    return (
        comparison.groupby(
            ["source", "dataset"],
            as_index=False,
        )
        .agg(
            series=("series_id", "nunique"),
            changed_selections=("selection_changed", "sum"),
            reduced_radius=("radius_change", lambda x: int((x < 0).sum())),
            increased_radius=("radius_change", lambda x: int((x > 0).sum())),
            plateau_radius_reductions=(
                "radius_reduced_on_plateau",
                "sum",
            ),
            mean_old_radius=("old_selected_radius", "mean"),
            mean_new_radius=("selected_radius", "mean"),
            mean_old_rules=("old_rule_count", "mean"),
            mean_new_rules=("rule_count", "mean"),
            mean_test_rmse_ratio=("test_rmse_ratio_to_old", "mean"),
            median_test_rmse_ratio=("test_rmse_ratio_to_old", "median"),
        )
        .sort_values(["source", "dataset"])
        .reset_index(drop=True)
    )
