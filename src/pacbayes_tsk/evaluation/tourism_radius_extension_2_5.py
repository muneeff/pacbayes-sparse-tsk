from __future__ import annotations

import numpy as np
import pandas as pd


class TourismRadiusExtensionError(ValueError):
    """Raised when the Tourism radius-extension experiment is inconsistent."""


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
        raise TourismRadiusExtensionError(
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
        raise TourismRadiusExtensionError(
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


def consolidate_extension_candidates(
    previous_validation: pd.DataFrame,
    extension_validation: pd.DataFrame,
    *,
    previous_radii: list[float],
    extension_radii: list[float],
    expected_series: int,
    lags_per_series: int,
    alphas_per_setting: int,
) -> pd.DataFrame:
    previous = prepare_candidates(
        previous_validation,
        label="previous Tourism validation",
        validation_source="tourism_grid_to_2_0_reused",
    )
    extension = prepare_candidates(
        extension_validation,
        label="Tourism radius extension",
        validation_source="tourism_radius_2_1_to_2_5_new",
    )

    previous = previous.loc[
        previous["source"].eq("real")
        & previous["dataset"].eq("tourism_monthly")
        & previous["radius"].isin(
            [float(value) for value in previous_radii]
        )
    ].copy()
    extension = extension.loc[
        extension["source"].eq("real")
        & extension["dataset"].eq("tourism_monthly")
        & extension["radius"].isin(
            [float(value) for value in extension_radii]
        )
    ].copy()

    previous_set = set(float(value) for value in previous_radii)
    extension_set = set(float(value) for value in extension_radii)
    overlap = previous_set.intersection(extension_set)
    if overlap:
        raise TourismRadiusExtensionError(
            f"Previous and extension radius grids overlap: {sorted(overlap)}."
        )

    expected_previous = (
        int(expected_series)
        * int(lags_per_series)
        * len(previous_radii)
        * int(alphas_per_setting)
    )
    expected_extension = (
        int(expected_series)
        * int(lags_per_series)
        * len(extension_radii)
        * int(alphas_per_setting)
    )
    if len(previous) != expected_previous:
        raise TourismRadiusExtensionError(
            "Unexpected previous Tourism validation row count: "
            f"expected {expected_previous}, found {len(previous)}."
        )
    if len(extension) != expected_extension:
        raise TourismRadiusExtensionError(
            "Unexpected extension validation row count: "
            f"expected {expected_extension}, found {len(extension)}."
        )

    pool = pd.concat(
        [previous, extension],
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
        raise TourismRadiusExtensionError(
            f"Duplicate consolidated candidates: {examples}."
        )

    expected_total = expected_previous + expected_extension
    if len(pool) != expected_total:
        raise TourismRadiusExtensionError(
            f"Expected {expected_total} consolidated candidates; "
            f"found {len(pool)}."
        )

    expected_per_series = (
        int(lags_per_series)
        * (
            len(previous_radii)
            + len(extension_radii)
        )
        * int(alphas_per_setting)
    )
    counts = pool.groupby(SERIES_KEY).size()
    if len(counts) != int(expected_series):
        raise TourismRadiusExtensionError(
            f"Expected {expected_series} Tourism series; found {len(counts)}."
        )
    if not counts.eq(expected_per_series).all():
        bad = counts.loc[
            ~counts.eq(expected_per_series)
        ].head(10)
        raise TourismRadiusExtensionError(
            "Incorrect candidate count for some Tourism series:\n"
            + bad.to_string()
        )

    return pool.sort_values(CANDIDATE_KEY).reset_index(drop=True)


def select_uncapped_smallest_radius(
    validation: pd.DataFrame,
) -> pd.Series:
    validate_candidate_columns(
        validation,
        label="Tourism extension series validation",
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
        raise TourismRadiusExtensionError(
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


def build_upper_boundary_audit(
    validation: pd.DataFrame,
    test_results: pd.DataFrame,
    *,
    maximum_radius: float,
    tolerance: float = 1.0e-10,
) -> pd.DataFrame:
    validate_candidate_columns(
        validation,
        label="consolidated Tourism extension validation",
    )
    required_test = {
        *SERIES_KEY,
        "status",
        "selected_lag",
        "selected_radius",
        "selected_ridge_alpha",
        "selected_validation_rmse",
        "selected_validation_mae",
        "rule_count",
        "total_parameter_count",
    }
    missing = required_test.difference(test_results.columns)
    if missing:
        raise TourismRadiusExtensionError(
            f"Tourism extension test results are missing: {sorted(missing)}."
        )

    selected_boundary = test_results.loc[
        test_results["status"].astype(str).eq("PASS")
        & np.isclose(
            test_results["selected_radius"].astype(float),
            float(maximum_radius),
            rtol=0.0,
            atol=1.0e-12,
        )
    ].copy()

    if selected_boundary.empty:
        return pd.DataFrame(
            columns=[
                *SERIES_KEY,
                "selected_lag",
                "selected_ridge_alpha",
                "selected_radius",
                "previous_radius",
                "selected_validation_rmse",
                "previous_validation_rmse",
                "selected_validation_mae",
                "previous_validation_mae",
                "selected_rule_count",
                "previous_rule_count",
                "selected_parameter_count",
                "previous_parameter_count",
                "rmse_improvement",
                "mae_improvement",
                "classification",
                "effective_upper_boundary",
            ]
        )

    prepared = prepare_candidates(
        validation,
        label="consolidated Tourism extension validation",
        validation_source="combined",
    )
    records: list[dict] = []

    for row in selected_boundary.itertuples(index=False):
        same = prepared.loc[
            prepared["source"].eq(str(row.source))
            & prepared["dataset"].eq(str(row.dataset))
            & prepared["series_id"].eq(str(row.series_id))
            & prepared["status"].astype(str).eq("PASS")
            & ~prepared["radius_cap_reached"]
            & prepared["lag_order"].eq(int(row.selected_lag))
            & np.isclose(
                prepared["ridge_alpha"].astype(float),
                float(row.selected_ridge_alpha),
                rtol=0.0,
                atol=1.0e-12,
            )
            & (prepared["radius"].astype(float) < float(maximum_radius))
        ].copy()

        if same.empty:
            records.append(
                {
                    "source": row.source,
                    "dataset": row.dataset,
                    "series_id": row.series_id,
                    "selected_lag": row.selected_lag,
                    "selected_ridge_alpha": row.selected_ridge_alpha,
                    "selected_radius": row.selected_radius,
                    "previous_radius": np.nan,
                    "selected_validation_rmse": row.selected_validation_rmse,
                    "previous_validation_rmse": np.nan,
                    "selected_validation_mae": row.selected_validation_mae,
                    "previous_validation_mae": np.nan,
                    "selected_rule_count": row.rule_count,
                    "previous_rule_count": np.nan,
                    "selected_parameter_count": row.total_parameter_count,
                    "previous_parameter_count": np.nan,
                    "rmse_improvement": np.nan,
                    "mae_improvement": np.nan,
                    "classification": "unresolved_missing_lower_comparator",
                    "effective_upper_boundary": True,
                }
            )
            continue

        previous_radius = float(same["radius"].max())
        previous = same.loc[
            np.isclose(
                same["radius"].astype(float),
                previous_radius,
                rtol=0.0,
                atol=1.0e-12,
            )
        ].sort_values(
            [
                "rmse",
                "mae",
                "total_parameter_count",
                "rule_count",
            ],
            kind="stable",
        ).iloc[0]

        rmse_improvement = (
            float(previous["rmse"])
            - float(row.selected_validation_rmse)
        )
        mae_improvement = (
            float(previous["mae"])
            - float(row.selected_validation_mae)
        )
        rmse_tie = abs(rmse_improvement) <= tolerance
        mae_tie = abs(mae_improvement) <= tolerance
        lower_complexity = (
            int(row.total_parameter_count)
            < int(previous["total_parameter_count"])
            or int(row.rule_count)
            < int(previous["rule_count"])
        )
        same_complexity = (
            int(row.total_parameter_count)
            == int(previous["total_parameter_count"])
            and int(row.rule_count)
            == int(previous["rule_count"])
        )

        if rmse_improvement > tolerance:
            classification = "true_rmse_improvement"
            effective = True
        elif rmse_tie and mae_improvement > tolerance:
            classification = "mae_improvement_on_rmse_tie"
            effective = True
        elif rmse_tie and mae_tie and lower_complexity:
            classification = "complexity_reduction_on_performance_tie"
            effective = False
        elif rmse_tie and mae_tie and same_complexity:
            classification = "exact_plateau"
            effective = False
        else:
            classification = "other_boundary_selection"
            effective = True

        records.append(
            {
                "source": row.source,
                "dataset": row.dataset,
                "series_id": row.series_id,
                "selected_lag": row.selected_lag,
                "selected_ridge_alpha": row.selected_ridge_alpha,
                "selected_radius": row.selected_radius,
                "previous_radius": previous_radius,
                "selected_validation_rmse": row.selected_validation_rmse,
                "previous_validation_rmse": float(previous["rmse"]),
                "selected_validation_mae": row.selected_validation_mae,
                "previous_validation_mae": float(previous["mae"]),
                "selected_rule_count": row.rule_count,
                "previous_rule_count": int(previous["rule_count"]),
                "selected_parameter_count": row.total_parameter_count,
                "previous_parameter_count": int(
                    previous["total_parameter_count"]
                ),
                "rmse_improvement": rmse_improvement,
                "mae_improvement": mae_improvement,
                "classification": classification,
                "effective_upper_boundary": bool(effective),
            }
        )

    return pd.DataFrame(records).sort_values(
        "series_id"
    ).reset_index(drop=True)


def build_grid_decision(
    test_results: pd.DataFrame,
    boundary_audit: pd.DataFrame,
    *,
    maximum_radius: float,
    boundary_threshold: float,
) -> pd.DataFrame:
    total = int(len(test_results))
    pass_mask = test_results["status"].astype(str).eq("PASS")
    pass_rows = test_results.loc[pass_mask].copy()
    failure_count = total - len(pass_rows)

    capped = (
        coerce_boolean(pass_rows["radius_cap_reached"])
        if not pass_rows.empty
        else pd.Series(dtype=bool)
    )
    selected_cap_count = int(capped.sum()) if len(capped) else 0

    raw_boundary_count = int(
        np.isclose(
            pass_rows["selected_radius"].astype(float),
            float(maximum_radius),
            rtol=0.0,
            atol=1.0e-12,
        ).sum()
    )
    effective_boundary_count = int(
        boundary_audit.get(
            "effective_upper_boundary",
            pd.Series(dtype=bool),
        ).fillna(False).astype(bool).sum()
    )

    nonfinite_count = 0
    for metric in ["mae", "rmse", "mase", "smape"]:
        if metric in pass_rows.columns:
            nonfinite_count += int(
                (~np.isfinite(pass_rows[metric])).sum()
            )

    raw_boundary_rate = (
        raw_boundary_count / total
        if total
        else np.nan
    )
    effective_boundary_rate = (
        effective_boundary_count / total
        if total
        else np.nan
    )

    failures_ok = failure_count == 0
    caps_ok = selected_cap_count == 0
    finite_ok = nonfinite_count == 0
    boundary_ok = (
        effective_boundary_rate <= float(boundary_threshold)
        if total
        else False
    )
    accepted = (
        failures_ok
        and caps_ok
        and finite_ok
        and boundary_ok
    )

    return pd.DataFrame(
        [
            {
                "source": "real",
                "dataset": "tourism_monthly",
                "study_status": "exploratory_test_informed",
                "maximum_radius": float(maximum_radius),
                "series": total,
                "pass_count": int(len(pass_rows)),
                "failure_count": int(failure_count),
                "selected_cap_count": selected_cap_count,
                "nonfinite_metric_count": nonfinite_count,
                "raw_upper_boundary_count": raw_boundary_count,
                "raw_upper_boundary_rate": raw_boundary_rate,
                "effective_upper_boundary_count": effective_boundary_count,
                "effective_upper_boundary_rate": effective_boundary_rate,
                "boundary_threshold": float(boundary_threshold),
                "failures_ok": failures_ok,
                "caps_ok": caps_ok,
                "finite_ok": finite_ok,
                "boundary_ok": boundary_ok,
                "grid_sufficient": accepted,
                "decision": (
                    "ACCEPT_RADIUS_GRID"
                    if accepted
                    else "EXTEND_RADIUS_GRID"
                ),
                "suggested_next_maximum_radius": (
                    float(maximum_radius)
                    if accepted
                    else float(maximum_radius) + 0.5
                ),
            }
        ]
    )
