from __future__ import annotations

import numpy as np
import pandas as pd


class TourismFineRadiusError(ValueError):
    """Raised when the Tourism fine-radius experiment is inconsistent."""


CANDIDATE_KEY = [
    "source",
    "dataset",
    "series_id",
    "lag_order",
    "radius",
    "ridge_alpha",
]

SERIES_KEY = [
    "source",
    "dataset",
    "series_id",
]


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
        raise TourismFineRadiusError(
            f"Invalid Boolean values: {unknown}."
        )
    return normalized.map(mapping).astype(bool)


def validate_candidate_columns(
    frame: pd.DataFrame,
    *,
    label: str,
) -> None:
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
        raise TourismFineRadiusError(
            f"{label} is missing columns: {sorted(missing)}."
        )


def prepare_candidates(
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


def consolidate_tourism_candidates(
    baseline_validation: pd.DataFrame,
    added_validation: pd.DataFrame,
    *,
    baseline_radii: list[float],
    added_radii: list[float],
    expected_series: int,
    lags_per_series: int,
    alphas_per_setting: int,
) -> pd.DataFrame:
    baseline = prepare_candidates(
        baseline_validation,
        label="baseline validation",
        validation_source="baseline_grid_reused",
    )
    added = prepare_candidates(
        added_validation,
        label="fine-radius validation",
        validation_source="fine_radius_new",
    )

    baseline = baseline.loc[
        baseline["source"].eq("real")
        & baseline["dataset"].eq("tourism_monthly")
        & baseline["radius"].isin(
            [float(value) for value in baseline_radii]
        )
    ].copy()

    added = added.loc[
        added["source"].eq("real")
        & added["dataset"].eq("tourism_monthly")
        & added["radius"].isin(
            [float(value) for value in added_radii]
        )
    ].copy()

    expected_baseline = (
        int(expected_series)
        * int(lags_per_series)
        * len(baseline_radii)
        * int(alphas_per_setting)
    )
    expected_added = (
        int(expected_series)
        * int(lags_per_series)
        * len(added_radii)
        * int(alphas_per_setting)
    )

    if len(baseline) != expected_baseline:
        raise TourismFineRadiusError(
            "Unexpected reused Tourism validation row count: "
            f"expected {expected_baseline}, found {len(baseline)}."
        )
    if len(added) != expected_added:
        raise TourismFineRadiusError(
            "Unexpected new fine-radius row count: "
            f"expected {expected_added}, found {len(added)}."
        )

    overlap = set(
        float(value) for value in baseline_radii
    ).intersection(
        float(value) for value in added_radii
    )
    if overlap:
        raise TourismFineRadiusError(
            f"Baseline and added radius grids overlap: {sorted(overlap)}."
        )

    pool = pd.concat(
        [baseline, added],
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
        raise TourismFineRadiusError(
            f"Duplicate consolidated candidates: {examples}."
        )

    expected_total = expected_baseline + expected_added
    if len(pool) != expected_total:
        raise TourismFineRadiusError(
            f"Expected {expected_total} consolidated candidates; "
            f"found {len(pool)}."
        )

    counts = (
        pool.groupby(SERIES_KEY)
        .size()
    )
    expected_per_series = (
        int(lags_per_series)
        * (
            len(baseline_radii)
            + len(added_radii)
        )
        * int(alphas_per_setting)
    )
    if len(counts) != int(expected_series):
        raise TourismFineRadiusError(
            f"Expected {expected_series} Tourism series; found {len(counts)}."
        )
    if not counts.eq(expected_per_series).all():
        bad = counts.loc[
            ~counts.eq(expected_per_series)
        ].head(10)
        raise TourismFineRadiusError(
            "Incorrect candidate count for some Tourism series:\n"
            + bad.to_string()
        )

    return pool.sort_values(CANDIDATE_KEY).reset_index(drop=True)


def select_uncapped_smallest_radius(
    validation: pd.DataFrame,
) -> pd.Series:
    validate_candidate_columns(
        validation,
        label="Tourism series validation",
    )
    successful = validation.loc[
        validation["status"].astype(str).eq("PASS")
    ].copy()
    successful["radius_cap_reached"] = coerce_boolean(
        successful["radius_cap_reached"]
    )
    successful = successful.loc[
        ~successful["radius_cap_reached"]
    ].copy()

    if successful.empty:
        raise TourismFineRadiusError(
            "No successful uncapped Tourism candidate is available."
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


def compare_tourism_results(
    fine_results: pd.DataFrame,
    reference_results: pd.DataFrame,
    *,
    reference_label: str,
) -> pd.DataFrame:
    required = {
        *SERIES_KEY,
        "status",
        "selected_lag",
        "selected_radius",
        "selected_ridge_alpha",
        "selected_validation_rmse",
        "rule_count",
        "total_parameter_count",
        "rmse",
        "mae",
        "mase",
    }
    missing_fine = required.difference(fine_results.columns)
    missing_reference = required.difference(reference_results.columns)
    if missing_fine:
        raise TourismFineRadiusError(
            f"Fine Tourism results are missing: {sorted(missing_fine)}."
        )
    if missing_reference:
        raise TourismFineRadiusError(
            f"Reference Tourism results are missing: {sorted(missing_reference)}."
        )

    fine = fine_results.loc[
        fine_results["source"].astype(str).eq("real")
        & fine_results["dataset"].astype(str).eq("tourism_monthly")
        & fine_results["status"].astype(str).eq("PASS")
    ].copy()

    reference = reference_results.loc[
        reference_results["source"].astype(str).eq("real")
        & reference_results["dataset"].astype(str).eq("tourism_monthly")
        & reference_results["status"].astype(str).eq("PASS")
    ].copy()

    reference_columns = [
        *SERIES_KEY,
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
    if "selected_validation_mae" in reference.columns:
        reference_columns.append("selected_validation_mae")

    rename = {
        column: f"{reference_label}_{column}"
        for column in reference_columns
        if column not in SERIES_KEY
    }
    reference = reference[
        reference_columns
    ].rename(columns=rename)

    comparison = fine.merge(
        reference,
        on=SERIES_KEY,
        how="left",
        validate="one_to_one",
    )

    prefix = f"{reference_label}_"
    comparison["selection_changed"] = (
        comparison["selected_lag"].ne(
            comparison[f"{prefix}selected_lag"]
        )
        | comparison["selected_radius"].ne(
            comparison[f"{prefix}selected_radius"]
        )
        | comparison["selected_ridge_alpha"].ne(
            comparison[f"{prefix}selected_ridge_alpha"]
        )
    )
    comparison["radius_change"] = (
        comparison["selected_radius"]
        - comparison[f"{prefix}selected_radius"]
    )
    comparison["rule_change"] = (
        comparison["rule_count"]
        - comparison[f"{prefix}rule_count"]
    )
    comparison["validation_rmse_change"] = (
        comparison["selected_validation_rmse"]
        - comparison[f"{prefix}selected_validation_rmse"]
    )
    comparison["test_rmse_change"] = (
        comparison["rmse"]
        - comparison[f"{prefix}rmse"]
    )
    comparison["test_rmse_ratio"] = (
        comparison["rmse"]
        / comparison[f"{prefix}rmse"]
    )
    comparison["test_mase_ratio"] = (
        comparison["mase"]
        / comparison[f"{prefix}mase"]
    )
    return comparison.sort_values("series_id").reset_index(drop=True)


def build_radius_distribution(
    test_results: pd.DataFrame,
    *,
    maximum_radius: float,
) -> pd.DataFrame:
    successful = test_results.loc[
        test_results["status"].astype(str).eq("PASS")
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

    distribution = (
        successful.groupby("selected_radius", as_index=False)
        .agg(series=("series_id", "nunique"))
        .sort_values("selected_radius")
    )
    total = int(successful["series_id"].nunique())
    distribution["rate"] = distribution["series"] / total
    distribution["at_upper_boundary"] = np.isclose(
        distribution["selected_radius"].astype(float),
        float(maximum_radius),
        rtol=0.0,
        atol=1e-12,
    )
    return distribution.reset_index(drop=True)


def build_fine_grid_summary(
    test_results: pd.DataFrame,
    comparison_current: pd.DataFrame,
    *,
    maximum_radius: float,
) -> pd.DataFrame:
    total = int(len(test_results))
    pass_mask = test_results["status"].astype(str).eq("PASS")
    successful = test_results.loc[pass_mask].copy()
    failures = total - len(successful)

    capped = coerce_boolean(
        successful["radius_cap_reached"]
    ) if not successful.empty else pd.Series(dtype=bool)
    boundary = np.isclose(
        successful["selected_radius"].astype(float),
        float(maximum_radius),
        rtol=0.0,
        atol=1e-12,
    ) if not successful.empty else np.array([], dtype=bool)

    ratios = (
        comparison_current["test_rmse_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    record = {
        "source": "real",
        "dataset": "tourism_monthly",
        "study_status": "exploratory_test_informed",
        "series": total,
        "pass_count": int(len(successful)),
        "failure_count": int(failures),
        "failure_rate": float(failures / total) if total else np.nan,
        "selected_cap_count": int(capped.sum()) if len(capped) else 0,
        "selected_cap_rate": float(capped.mean()) if len(capped) else np.nan,
        "upper_boundary_count": int(boundary.sum()),
        "upper_boundary_rate": float(boundary.mean()) if len(boundary) else np.nan,
        "maximum_radius": float(maximum_radius),
        "mean_selected_radius": float(successful["selected_radius"].mean()),
        "median_selected_radius": float(successful["selected_radius"].median()),
        "mean_rule_count": float(successful["rule_count"].mean()),
        "median_rule_count": float(successful["rule_count"].median()),
        "mean_total_parameters": float(
            successful["total_parameter_count"].mean()
        ),
        "mean_rmse": float(successful["rmse"].mean()),
        "median_rmse": float(successful["rmse"].median()),
        "mean_mase": float(successful["mase"].mean()),
        "median_mase": float(successful["mase"].median()),
        "changed_vs_current": int(
            comparison_current["selection_changed"].sum()
        ),
        "mean_rmse_ratio_vs_current": float(ratios.mean()),
        "median_rmse_ratio_vs_current": float(ratios.median()),
        "maximum_rmse_ratio_vs_current": float(ratios.max()),
    }
    return pd.DataFrame([record])


def merge_tourism_into_all210(
    current_all210: pd.DataFrame,
    fine_tourism: pd.DataFrame,
    *,
    expected_total: int = 210,
    expected_tourism: int = 30,
) -> pd.DataFrame:
    current = current_all210.copy()
    fine = fine_tourism.copy()

    current_keys = current[SERIES_KEY].astype(str)
    if current_keys.duplicated().any():
        raise TourismFineRadiusError(
            "Current 210-series table contains duplicate series keys."
        )
    fine_keys = fine[SERIES_KEY].astype(str)
    if fine_keys.duplicated().any():
        raise TourismFineRadiusError(
            "Fine Tourism table contains duplicate series keys."
        )

    current_tourism = current.loc[
        current["source"].astype(str).eq("real")
        & current["dataset"].astype(str).eq("tourism_monthly")
    ].copy()
    fine_tourism_rows = fine.loc[
        fine["source"].astype(str).eq("real")
        & fine["dataset"].astype(str).eq("tourism_monthly")
    ].copy()

    if len(current) != int(expected_total):
        raise TourismFineRadiusError(
            f"Expected current table with {expected_total} rows; found {len(current)}."
        )
    if len(current_tourism) != int(expected_tourism):
        raise TourismFineRadiusError(
            "Current all-series table has an unexpected Tourism count: "
            f"{len(current_tourism)}."
        )
    if len(fine_tourism_rows) != int(expected_tourism):
        raise TourismFineRadiusError(
            "Fine Tourism table has an unexpected row count: "
            f"{len(fine_tourism_rows)}."
        )

    old_key_set = {
        tuple(row)
        for row in current_tourism[SERIES_KEY].astype(str).to_numpy()
    }
    new_key_set = {
        tuple(row)
        for row in fine_tourism_rows[SERIES_KEY].astype(str).to_numpy()
    }
    if old_key_set != new_key_set:
        raise TourismFineRadiusError(
            "Fine Tourism series keys do not exactly match the current 210-series table."
        )

    retained = current.loc[
        ~(
            current["source"].astype(str).eq("real")
            & current["dataset"].astype(str).eq("tourism_monthly")
        )
    ].copy()
    merged = pd.concat(
        [retained, fine_tourism_rows],
        ignore_index=True,
        sort=False,
    )
    merged = merged.sort_values(SERIES_KEY).reset_index(drop=True)

    if len(merged) != int(expected_total):
        raise TourismFineRadiusError(
            f"Merged table should contain {expected_total} rows; found {len(merged)}."
        )
    if merged[SERIES_KEY].astype(str).duplicated().any():
        raise TourismFineRadiusError(
            "Merged table contains duplicate series keys."
        )
    return merged
