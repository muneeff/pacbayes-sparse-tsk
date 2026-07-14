from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class CausalImputationError(ValueError):
    """Raised when a series cannot be imputed without future information."""


@dataclass(frozen=True)
class ImputationReport:
    original_missing: int
    seasonal_fills: int
    last_observation_fills: int
    leading_values_dropped: int
    remaining_missing: int


def trim_leading_missing(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop only the missing prefix; never inspect values after a nonmissing start."""
    array = np.asarray(values, dtype=np.float64)
    finite_positions = np.flatnonzero(np.isfinite(array))
    if finite_positions.size == 0:
        raise CausalImputationError("The series contains no finite observations.")
    first = int(finite_positions[0])
    return array[first:].copy(), first


def causal_seasonal_impute(
    values: np.ndarray,
    *,
    seasonal_period: int,
    drop_leading: bool = True,
) -> tuple[np.ndarray, np.ndarray, ImputationReport]:
    """
    Causally impute missing values.

    At time t:
      1. use the already available imputed/observed value at t-seasonal_period;
      2. otherwise use the latest past finite observation;
      3. never use t+1 or any later value.
    """
    if seasonal_period <= 0:
        raise CausalImputationError("seasonal_period must be positive.")

    original = np.asarray(values, dtype=np.float64)
    original_missing = int((~np.isfinite(original)).sum())

    if drop_leading:
        working, dropped = trim_leading_missing(original)
    else:
        working = original.copy()
        dropped = 0

    sources = np.full(len(working), "observed", dtype="U24")
    seasonal_fills = 0
    locf_fills = 0
    last_finite: float | None = None

    for index in range(len(working)):
        value = working[index]
        if np.isfinite(value):
            last_finite = float(value)
            continue

        seasonal_index = index - seasonal_period
        if seasonal_index >= 0 and np.isfinite(working[seasonal_index]):
            working[index] = working[seasonal_index]
            sources[index] = "seasonal_past"
            seasonal_fills += 1
            last_finite = float(working[index])
            continue

        if last_finite is not None:
            working[index] = last_finite
            sources[index] = "last_observation"
            locf_fills += 1
            continue

        raise CausalImputationError(
            "A leading missing value remains but no past observation exists."
        )

    remaining = int((~np.isfinite(working)).sum())
    if remaining:
        raise CausalImputationError(
            f"Causal imputation left {remaining} missing observations."
        )

    report = ImputationReport(
        original_missing=original_missing,
        seasonal_fills=seasonal_fills,
        last_observation_fills=locf_fills,
        leading_values_dropped=dropped,
        remaining_missing=remaining,
    )
    return working, sources, report
