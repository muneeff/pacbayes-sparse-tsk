from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pacbayes_tsk.data.causal_imputation import (
    ImputationReport,
    causal_seasonal_impute,
)
from pacbayes_tsk.data.tsf_reader import TSFMetadata, TSFRecord


VALID_SPLITS = ("prior", "bound", "validation", "test")


class RealDataError(ValueError):
    """Raised when a real series cannot satisfy the experimental protocol."""


@dataclass(frozen=True)
class SplitResult:
    labels: np.ndarray
    prior_n: int
    bound_n: int
    validation_n: int
    test_n: int


def make_fixed_horizon_splits(
    n_observations: int,
    *,
    horizon: int,
    prior_fraction: float,
    min_prior: int,
    min_bound: int,
) -> SplitResult:
    """
    Reserve the final horizon for test and the preceding horizon for validation.

    The earlier development prefix is divided chronologically into prior and bound.
    """
    if horizon <= 0:
        raise RealDataError("horizon must be positive.")
    if not 0.0 < prior_fraction < 1.0:
        raise RealDataError("prior_fraction must be between 0 and 1.")

    development_n = n_observations - 2 * horizon
    if development_n <= 0:
        raise RealDataError(
            "The series is too short for separate validation and test horizons."
        )

    prior_n = int(np.floor(prior_fraction * development_n))
    prior_n = max(prior_n, int(min_prior))
    bound_n = development_n - prior_n

    if prior_n < min_prior:
        raise RealDataError(
            f"Prior split is too short: {prior_n} < {min_prior}."
        )
    if bound_n < min_bound:
        raise RealDataError(
            f"Bound split is too short: {bound_n} < {min_bound}."
        )

    labels = np.concatenate(
        [
            np.repeat("prior", prior_n),
            np.repeat("bound", bound_n),
            np.repeat("validation", horizon),
            np.repeat("test", horizon),
        ]
    ).astype(object)

    if len(labels) != n_observations:
        raise RuntimeError("Split labels do not match the series length.")

    return SplitResult(
        labels=labels,
        prior_n=prior_n,
        bound_n=bound_n,
        validation_n=horizon,
        test_n=horizon,
    )


def zscore_from_prior(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, float, float, bool]:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)

    prior = values[labels == "prior"]
    if prior.size == 0:
        raise RealDataError("Prior split is empty.")
    if not np.isfinite(values).all():
        raise RealDataError("Values contain NaN or infinite entries.")

    mean = float(prior.mean())
    raw_scale = float(prior.std(ddof=0))
    fallback = (not np.isfinite(raw_scale)) or raw_scale <= epsilon
    scale = 1.0 if fallback else raw_scale
    return (values - mean) / scale, mean, scale, bool(fallback)


def infer_cif_origin(attributes: dict[str, Any]) -> str:
    """
    Infer CIF origin only from explicit metadata text.

    No ordering-based assumption is used. Unknown remains 'unverified'.
    """
    text = " ".join(
        f"{key}={value}" for key, value in attributes.items()
    ).lower()

    artificial_tokens = (
        "artificial",
        "synthetic",
        "generated",
        "simulated",
    )
    real_tokens = (
        "real",
        "bank",
        "banking",
    )

    if any(token in text for token in artificial_tokens):
        return "artificial"
    if any(token in text for token in real_tokens):
        return "real"
    return "unverified"


def read_verified_id_file(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    file_path = Path(path)
    if not file_path.exists():
        return set()

    ids: set[str] = set()
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        ids.add(line)
    return ids


def apply_cif_verified_ids(
    series_id: str,
    inferred_origin: str,
    verified_real_ids: set[str],
) -> str:
    if not verified_real_ids:
        return inferred_origin
    return "real" if series_id in verified_real_ids else "artificial"


def serialize_attributes(attributes: dict[str, Any]) -> str:
    serializable = {
        str(key): (
            value.isoformat()
            if hasattr(value, "isoformat")
            else value
        )
        for key, value in attributes.items()
    }
    return json.dumps(
        serializable,
        ensure_ascii=False,
        sort_keys=True,
    )


def prepare_real_series(
    *,
    dataset: str,
    series_id: str,
    record: TSFRecord,
    metadata: TSFMetadata,
    horizon: int,
    seasonal_period: int,
    prior_fraction: float,
    min_prior: int,
    min_bound: int,
    impute_missing: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    original = np.asarray(record.values, dtype=np.float64)
    original_length = len(original)

    if impute_missing or not np.isfinite(original).all():
        imputed, sources, report = causal_seasonal_impute(
            original,
            seasonal_period=seasonal_period,
            drop_leading=True,
        )
    else:
        imputed = original.copy()
        sources = np.repeat("observed", len(imputed)).astype("U24")
        report = ImputationReport(
            original_missing=0,
            seasonal_fills=0,
            last_observation_fills=0,
            leading_values_dropped=0,
            remaining_missing=0,
        )

    split = make_fixed_horizon_splits(
        len(imputed),
        horizon=horizon,
        prior_fraction=prior_fraction,
        min_prior=min_prior,
        min_bound=min_bound,
    )

    scaled, scaler_mean, scaler_scale, scaler_fallback = zscore_from_prior(
        imputed,
        split.labels,
    )

    original_aligned = original[report.leading_values_dropped :]
    if len(original_aligned) != len(imputed):
        raise RuntimeError("Imputation alignment failure.")

    time_index = np.arange(
        report.leading_values_dropped,
        report.leading_values_dropped + len(imputed),
        dtype=np.int64,
    )

    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "series_id": series_id,
            "time": time_index,
            "value_raw": original_aligned,
            "value_imputed": imputed,
            "value_scaled": scaled,
            "split": split.labels,
            "was_imputed": sources != "observed",
            "imputation_source": sources,
        }
    )

    metadata_row = {
        "dataset": dataset,
        "series_id": series_id,
        "original_length": original_length,
        "processed_length": len(imputed),
        "frequency": metadata.frequency,
        "forecast_horizon": horizon,
        "seasonal_period": seasonal_period,
        "prior_n": split.prior_n,
        "bound_n": split.bound_n,
        "validation_n": split.validation_n,
        "test_n": split.test_n,
        "original_missing": report.original_missing,
        "seasonal_fills": report.seasonal_fills,
        "last_observation_fills": report.last_observation_fills,
        "leading_values_dropped": report.leading_values_dropped,
        "remaining_missing": report.remaining_missing,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "scaler_fallback": scaler_fallback,
        "attributes_json": serialize_attributes(record.attributes),
    }
    return frame, metadata_row


def deterministic_length_stratified_sample(
    frame: pd.DataFrame,
    *,
    target_n: int,
    seed: int,
    length_column: str = "processed_length",
    id_column: str = "series_id",
    bins: int = 3,
) -> pd.DataFrame:
    """Select a deterministic sample distributed across length strata."""
    if target_n <= 0:
        return frame.iloc[0:0].copy()
    if len(frame) <= target_n:
        return frame.sort_values(id_column).copy()

    ordered = frame.sort_values([length_column, id_column]).reset_index(drop=True)
    ranks = ordered[length_column].rank(method="first")
    ordered["_length_stratum"] = pd.qcut(
        ranks,
        q=min(bins, len(ordered)),
        labels=False,
        duplicates="drop",
    )

    strata = sorted(ordered["_length_stratum"].unique().tolist())
    base = target_n // len(strata)
    remainder = target_n % len(strata)
    rng = np.random.default_rng(seed)

    chosen_indices: list[int] = []
    for position, stratum in enumerate(strata):
        group = ordered.index[
            ordered["_length_stratum"] == stratum
        ].to_numpy()
        take = base + (1 if position < remainder else 0)
        take = min(take, len(group))
        chosen = rng.choice(group, size=take, replace=False)
        chosen_indices.extend(int(value) for value in chosen)

    if len(chosen_indices) < target_n:
        remaining = ordered.index[
            ~ordered.index.isin(chosen_indices)
        ].to_numpy()
        additional = rng.choice(
            remaining,
            size=target_n - len(chosen_indices),
            replace=False,
        )
        chosen_indices.extend(int(value) for value in additional)

    return (
        ordered.loc[sorted(chosen_indices)]
        .drop(columns="_length_stratum")
        .sort_values(id_column)
        .reset_index(drop=True)
    )


def stable_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
