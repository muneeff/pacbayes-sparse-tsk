from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


REQUIRED_SERIES_COLUMNS = ("series_id", "time", "value", "regime", "seed")
SPLIT_ORDER = ("prior", "bound", "validation", "test")


class DataValidationError(ValueError):
    """Raised when a time series fails a required data-quality check."""


@dataclass(frozen=True)
class SplitFractions:
    prior: float
    bound: float
    validation: float
    test: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SplitFractions":
        fractions = cls(
            prior=float(values["prior"]),
            bound=float(values["bound"]),
            validation=float(values["validation"]),
            test=float(values["test"]),
        )
        fractions.validate()
        return fractions

    def validate(self) -> None:
        values = np.asarray(
            [self.prior, self.bound, self.validation, self.test],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Split fractions must be finite.")
        if np.any(values <= 0.0):
            raise ValueError("Every split fraction must be strictly positive.")
        if not np.isclose(values.sum(), 1.0, atol=1e-12):
            raise ValueError(
                f"Split fractions must sum to 1.0; received {values.sum():.12f}."
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "prior": self.prior,
            "bound": self.bound,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class ScalerStats:
    method: str
    fit_split: str
    mean: float
    scale: float
    used_fallback_scale: bool


def compute_split_sizes(
    n_observations: int,
    fractions: SplitFractions,
) -> dict[str, int]:
    """Return exact split sizes using a deterministic largest-remainder rule."""
    fractions.validate()

    if n_observations < len(SPLIT_ORDER):
        raise ValueError(
            f"At least {len(SPLIT_ORDER)} observations are required; "
            f"received {n_observations}."
        )

    fraction_values = np.asarray(
        [
            fractions.prior,
            fractions.bound,
            fractions.validation,
            fractions.test,
        ],
        dtype=float,
    )
    raw_sizes = fraction_values * int(n_observations)
    sizes = np.floor(raw_sizes).astype(int)

    remainder = int(n_observations - sizes.sum())
    fractional_parts = raw_sizes - sizes

    # Stable ordering makes equal fractional parts deterministic.
    order = np.argsort(-fractional_parts, kind="stable")
    for index in order[:remainder]:
        sizes[index] += 1

    if np.any(sizes <= 0):
        raise ValueError(
            "The requested fractions produce an empty temporal split. "
            "Use a longer series or larger fractions."
        )

    result = dict(zip(SPLIT_ORDER, sizes.tolist()))
    if sum(result.values()) != n_observations:
        raise RuntimeError("Internal error: split sizes do not sum to series length.")
    return result


def build_split_labels(
    n_observations: int,
    fractions: SplitFractions,
) -> tuple[np.ndarray, dict[str, int]]:
    """Create ordered prior→bound→validation→test labels."""
    sizes = compute_split_sizes(n_observations, fractions)
    labels = np.concatenate(
        [
            np.repeat(split_name, sizes[split_name])
            for split_name in SPLIT_ORDER
        ]
    )
    if len(labels) != n_observations:
        raise RuntimeError("Internal error: label count does not match series length.")
    return labels.astype(object), sizes


def inspect_series_frame(
    frame: pd.DataFrame,
    *,
    expected_length: int | None = None,
) -> dict[str, Any]:
    """Inspect one generated series without silently repairing it."""
    missing_columns = [
        column for column in REQUIRED_SERIES_COLUMNS if column not in frame.columns
    ]

    report: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "missing_columns": "|".join(missing_columns),
        "missing_value_count": 0,
        "nonfinite_value_count": 0,
        "duplicate_time_count": 0,
        "time_strictly_increasing": False,
        "single_series_id": False,
        "single_seed": False,
        "expected_length": expected_length,
        "expected_length_match": (
            expected_length is None or len(frame) == int(expected_length)
        ),
        "status": "FAIL",
        "issues": "",
    }

    issues: list[str] = []

    if missing_columns:
        issues.append(f"missing_columns={','.join(missing_columns)}")
        report["issues"] = "; ".join(issues)
        return report

    required = frame.loc[:, REQUIRED_SERIES_COLUMNS]
    report["missing_value_count"] = int(required.isna().sum().sum())

    numeric_values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(float)
    report["nonfinite_value_count"] = int((~np.isfinite(numeric_values)).sum())

    time_values = pd.to_numeric(frame["time"], errors="coerce").to_numpy(float)
    finite_time = np.isfinite(time_values)
    report["duplicate_time_count"] = int(frame["time"].duplicated().sum())
    report["time_strictly_increasing"] = bool(
        finite_time.all()
        and len(time_values) > 0
        and (len(time_values) == 1 or np.all(np.diff(time_values) > 0))
    )

    report["single_series_id"] = bool(frame["series_id"].nunique(dropna=False) == 1)
    report["single_seed"] = bool(frame["seed"].nunique(dropna=False) == 1)

    if report["missing_value_count"] > 0:
        issues.append(f"missing_values={report['missing_value_count']}")
    if report["nonfinite_value_count"] > 0:
        issues.append(f"nonfinite_values={report['nonfinite_value_count']}")
    if report["duplicate_time_count"] > 0:
        issues.append(f"duplicate_times={report['duplicate_time_count']}")
    if not report["time_strictly_increasing"]:
        issues.append("time_not_strictly_increasing")
    if not report["single_series_id"]:
        issues.append("multiple_series_ids")
    if not report["single_seed"]:
        issues.append("multiple_seeds")
    if not report["expected_length_match"]:
        issues.append(
            f"unexpected_length={len(frame)};expected={int(expected_length)}"
        )

    report["issues"] = "; ".join(issues)
    report["status"] = "PASS" if not issues else "FAIL"
    return report


def validate_series_frame(
    frame: pd.DataFrame,
    *,
    expected_length: int | None = None,
) -> dict[str, Any]:
    """Return a quality report or raise a clear validation exception."""
    report = inspect_series_frame(frame, expected_length=expected_length)
    if report["status"] != "PASS":
        raise DataValidationError(report["issues"])
    return report


def fit_prior_zscore(
    values: np.ndarray,
    split_labels: np.ndarray,
    *,
    fit_split: str = "prior",
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, ScalerStats]:
    """Fit z-score parameters on one past split and apply them to all times."""
    values = np.asarray(values, dtype=float)
    split_labels = np.asarray(split_labels, dtype=object)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if split_labels.shape != values.shape:
        raise ValueError("split_labels must have the same shape as values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contain NaN or infinite entries.")

    fit_mask = split_labels == fit_split
    if not np.any(fit_mask):
        raise ValueError(f"No observations found for fit_split={fit_split!r}.")

    fit_values = values[fit_mask]
    mean = float(np.mean(fit_values))
    raw_scale = float(np.std(fit_values, ddof=0))
    used_fallback = bool((not np.isfinite(raw_scale)) or raw_scale <= epsilon)
    scale = 1.0 if used_fallback else raw_scale

    transformed = (values - mean) / scale
    stats = ScalerStats(
        method="zscore",
        fit_split=fit_split,
        mean=mean,
        scale=scale,
        used_fallback_scale=used_fallback,
    )
    return transformed, stats


def preprocess_series_frame(
    frame: pd.DataFrame,
    *,
    fractions: SplitFractions,
    expected_length: int | None = None,
    fit_split: str = "prior",
    epsilon: float = 1e-12,
    strict: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Validate, temporally split, and scale one series without future leakage."""
    quality = inspect_series_frame(frame, expected_length=expected_length)
    if strict and quality["status"] != "PASS":
        raise DataValidationError(quality["issues"])

    missing_columns = [
        column for column in REQUIRED_SERIES_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise DataValidationError(
            f"Cannot preprocess because columns are missing: {missing_columns}"
        )

    ordered = frame.sort_values("time", kind="stable").reset_index(drop=True).copy()
    labels, sizes = build_split_labels(len(ordered), fractions)

    raw_values = pd.to_numeric(ordered["value"], errors="raise").to_numpy(float)
    scaled_values, scaler = fit_prior_zscore(
        raw_values,
        labels,
        fit_split=fit_split,
        epsilon=epsilon,
    )

    processed = pd.DataFrame(
        {
            "series_id": ordered["series_id"].astype(str),
            "time": ordered["time"],
            "value_raw": raw_values,
            "value_scaled": scaled_values,
            "split": labels,
            "regime": ordered["regime"],
            "seed": ordered["seed"],
        }
    )

    boundaries: dict[str, Any] = {}
    cursor = 0
    for split_name in SPLIT_ORDER:
        split_size = sizes[split_name]
        start_index = cursor
        end_index = cursor + split_size - 1
        boundaries[f"{split_name}_start_index"] = int(start_index)
        boundaries[f"{split_name}_end_index"] = int(end_index)
        boundaries[f"{split_name}_start_time"] = processed.loc[start_index, "time"]
        boundaries[f"{split_name}_end_time"] = processed.loc[end_index, "time"]
        boundaries[f"{split_name}_n"] = int(split_size)
        cursor = end_index + 1

    metadata: dict[str, Any] = {
        **boundaries,
        "scaler_method": scaler.method,
        "scaler_fit_split": scaler.fit_split,
        "scaler_mean": scaler.mean,
        "scaler_scale": scaler.scale,
        "scaler_used_fallback_scale": scaler.used_fallback_scale,
    }
    return processed, metadata, quality


def sha256_file(path: str | Path) -> str:
    """Return a reproducibility checksum for a file."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_mapping(
    values: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested configuration dictionaries for tabular export."""
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, prefix=full_key))
        elif isinstance(value, (list, tuple)):
            flattened[full_key] = json.dumps(value, ensure_ascii=False)
        else:
            flattened[full_key] = value
    return flattened
