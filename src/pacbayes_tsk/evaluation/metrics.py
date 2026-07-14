from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MetricError(ValueError):
    """Raised when forecasting metrics cannot be computed safely."""


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    mase: float
    smape: float
    n_observations: int
    mase_scale: float
    mase_scale_pairs: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mase": self.mase,
            "smape": self.smape,
            "n_observations": self.n_observations,
            "mase_scale": self.mase_scale,
            "mase_scale_pairs": self.mase_scale_pairs,
        }


def _validate_pair(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if true.shape != pred.shape:
        raise MetricError(
            f"y_true shape {true.shape} does not match y_pred shape {pred.shape}."
        )
    if true.size == 0:
        raise MetricError("At least one observation is required.")
    if not np.all(np.isfinite(true)):
        raise MetricError("y_true contains NaN or infinite values.")
    if not np.all(np.isfinite(pred)):
        raise MetricError("y_pred contains NaN or infinite values.")
    return true, pred


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _validate_pair(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _validate_pair(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(true - pred))))


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> float:
    """
    Symmetric mean absolute percentage error in the [0, 200] convention.

    Terms with |y| + |y_hat| <= epsilon contribute zero.
    """
    true, pred = _validate_pair(y_true, y_pred)
    denominator = np.abs(true) + np.abs(pred)
    numerator = 2.0 * np.abs(true - pred)

    terms = np.zeros_like(numerator)
    valid = denominator > float(epsilon)
    terms[valid] = numerator[valid] / denominator[valid]
    return float(100.0 * np.mean(terms))


def _mase_denominator_details(
    insample: np.ndarray,
    *,
    seasonal_period: int,
    epsilon: float,
    observed_mask: np.ndarray | None,
) -> tuple[float, int]:
    values = np.asarray(insample, dtype=np.float64).reshape(-1)

    if seasonal_period <= 0:
        raise MetricError("seasonal_period must be strictly positive.")
    if values.size <= seasonal_period:
        raise MetricError(
            "insample must contain more observations than seasonal_period."
        )
    if not np.all(np.isfinite(values)):
        raise MetricError("insample contains NaN or infinite values.")

    if observed_mask is None:
        observed = np.ones(values.size, dtype=bool)
    else:
        observed = np.asarray(observed_mask)
        if observed.ndim != 1:
            observed = observed.reshape(-1)
        if observed.size != values.size:
            raise MetricError(
                "observed_mask must have the same length as insample."
            )
        if observed.dtype != np.bool_:
            raise MetricError("observed_mask must have Boolean dtype.")

    valid_pairs = (
        observed[seasonal_period:]
        & observed[:-seasonal_period]
    )
    differences = np.abs(
        values[seasonal_period:] - values[:-seasonal_period]
    )
    eligible = differences[valid_pairs]
    pair_count = int(eligible.size)

    if pair_count == 0:
        return float("nan"), 0

    scale = float(np.mean(eligible))
    if not np.isfinite(scale) or scale <= float(epsilon):
        return float("nan"), pair_count
    return scale, pair_count


def mase_denominator(
    insample: np.ndarray,
    *,
    seasonal_period: int = 1,
    epsilon: float = 1e-12,
    observed_mask: np.ndarray | None = None,
) -> float:
    """
    Return the mean absolute seasonal difference used by MASE.

    The input must contain only data available before validation/test. When
    ``observed_mask`` is provided, a seasonal difference contributes only when
    both endpoints were originally observed. This prevents causally imputed
    NN5 values from changing the MASE scaling denominator.
    """
    scale, _ = _mase_denominator_details(
        insample,
        seasonal_period=seasonal_period,
        epsilon=epsilon,
        observed_mask=observed_mask,
    )
    return scale


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    scale: float,
) -> float:
    true, pred = _validate_pair(y_true, y_pred)
    if not np.isfinite(scale) or scale <= 0.0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)) / scale)


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    insample: np.ndarray,
    seasonal_period: int = 1,
    epsilon: float = 1e-12,
    insample_observed_mask: np.ndarray | None = None,
) -> ForecastMetrics:
    true, pred = _validate_pair(y_true, y_pred)
    scale, pair_count = _mase_denominator_details(
        insample,
        seasonal_period=seasonal_period,
        epsilon=epsilon,
        observed_mask=insample_observed_mask,
    )
    return ForecastMetrics(
        mae=mae(true, pred),
        rmse=rmse(true, pred),
        mase=mase(true, pred, scale=scale),
        smape=smape(true, pred, epsilon=epsilon),
        n_observations=int(true.size),
        mase_scale=scale,
        mase_scale_pairs=pair_count,
    )
