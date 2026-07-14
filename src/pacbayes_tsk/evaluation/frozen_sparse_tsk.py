from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


class FrozenSparseTSKError(ValueError):
    """Raised when the development Sparse TSK result cannot be frozen."""


SERIES_KEY = [
    "source",
    "dataset",
    "series_id",
]

LOCK_COLUMNS = [
    "source",
    "dataset",
    "series_id",
    "model",
    "selection_mode",
    "tie_break_policy",
    "validation_source",
    "study_status",
    "selected_lag",
    "selected_radius",
    "selected_ridge_alpha",
    "rule_count",
    "effective_rule_count",
    "radius_cap_reached",
    "antecedent_parameter_count",
    "consequent_parameter_count",
    "total_parameter_count",
    "model_path",
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
        raise FrozenSparseTSKError(
            f"Invalid Boolean values: {unknown}."
        )
    return normalized.map(mapping).astype(bool)


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def stable_row_id(row: pd.Series) -> str:
    payload = "|".join(
        [
            str(row["source"]),
            str(row["dataset"]),
            str(row["series_id"]),
            str(int(row["selected_lag"])),
            format(float(row["selected_radius"]), ".12g"),
            format(
                float(row["selected_ridge_alpha"]),
                ".12g",
            ),
            str(int(row["rule_count"])),
            str(int(row["total_parameter_count"])),
        ]
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def validate_final_results(
    frame: pd.DataFrame,
    *,
    expected_series: int = 210,
) -> pd.DataFrame:
    required = {
        *SERIES_KEY,
        *LOCK_COLUMNS,
        "status",
        "mae",
        "rmse",
        "mase",
        "smape",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise FrozenSparseTSKError(
            "Final Sparse TSK table is missing columns: "
            f"{sorted(missing)}."
        )

    output = frame.copy()
    if len(output) != int(expected_series):
        raise FrozenSparseTSKError(
            f"Expected {expected_series} rows; found {len(output)}."
        )

    duplicate = output.duplicated(
        subset=SERIES_KEY,
        keep=False,
    )
    if duplicate.any():
        examples = (
            output.loc[duplicate, SERIES_KEY]
            .head(10)
            .to_dict("records")
        )
        raise FrozenSparseTSKError(
            f"Duplicate series keys: {examples}."
        )

    failures = output.loc[
        ~output["status"].astype(str).eq("PASS")
    ]
    if not failures.empty:
        raise FrozenSparseTSKError(
            f"Final table contains {len(failures)} failures."
        )

    output["radius_cap_reached"] = coerce_boolean(
        output["radius_cap_reached"]
    )
    capped = output.loc[
        output["radius_cap_reached"]
    ]
    if not capped.empty:
        raise FrozenSparseTSKError(
            f"Final table contains {len(capped)} capped selections."
        )

    for metric in ["mae", "rmse", "mase", "smape"]:
        values = pd.to_numeric(
            output[metric],
            errors="coerce",
        ).to_numpy(float)
        if not np.all(np.isfinite(values)):
            raise FrozenSparseTSKError(
                f"Final table contains non-finite {metric}."
            )

    if not output["selection_mode"].astype(str).eq(
        "uncapped_only"
    ).all():
        raise FrozenSparseTSKError(
            "All frozen models must use selection_mode=uncapped_only."
        )
    if not output["tie_break_policy"].astype(str).eq(
        "smallest_radius"
    ).all():
        raise FrozenSparseTSKError(
            "All frozen models must use tie_break_policy=smallest_radius."
        )
    if not output["model"].astype(str).eq(
        "sparse_tsk"
    ).all():
        raise FrozenSparseTSKError(
            "The frozen table must contain only sparse_tsk models."
        )

    return output.sort_values(
        SERIES_KEY
    ).reset_index(drop=True)


def build_selection_lock(
    validated: pd.DataFrame,
) -> pd.DataFrame:
    missing = set(LOCK_COLUMNS).difference(
        validated.columns
    )
    if missing:
        raise FrozenSparseTSKError(
            f"Cannot build lock; missing columns: {sorted(missing)}."
        )

    lock = validated[LOCK_COLUMNS].copy()
    lock["radius_cap_reached"] = coerce_boolean(
        lock["radius_cap_reached"]
    )
    lock["frozen_model_id"] = lock.apply(
        stable_row_id,
        axis=1,
    )
    lock["frozen"] = True
    lock["certificate_status"] = (
        "exploratory_development_only"
    )
    return lock.sort_values(
        SERIES_KEY
    ).reset_index(drop=True)


def validate_lock_has_no_test_metrics(
    lock: pd.DataFrame,
) -> None:
    forbidden = {
        "mae",
        "rmse",
        "mase",
        "smape",
        "y_true",
        "y_pred",
        "test_mean_active_rules",
        "test_rule_usage_entropy",
    }
    overlap = forbidden.intersection(lock.columns)
    if overlap:
        raise FrozenSparseTSKError(
            "Frozen selection lock contains forbidden test metrics: "
            f"{sorted(overlap)}."
        )
