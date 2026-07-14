from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge


class BaselineModelError(ValueError):
    """Raised when a baseline model receives invalid data or settings."""


@dataclass
class FittedBaseline:
    model_name: str
    estimator: Any | None
    alpha: float | None
    warning_messages: tuple[str, ...]

    def predict(self, X: np.ndarray) -> np.ndarray:
        matrix = validate_feature_matrix(X)
        if self.model_name == "naive":
            return naive_prediction(matrix)
        if self.estimator is None:
            raise BaselineModelError(
                f"Estimator is missing for model {self.model_name!r}."
            )
        prediction = np.asarray(
            self.estimator.predict(matrix),
            dtype=np.float64,
        ).reshape(-1)
        if not np.all(np.isfinite(prediction)):
            raise BaselineModelError(
                f"{self.model_name} produced non-finite predictions."
            )
        return prediction


def validate_feature_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise BaselineModelError("X must be a two-dimensional matrix.")
    if matrix.shape[0] == 0:
        raise BaselineModelError("X contains no rows.")
    if matrix.shape[1] == 0:
        raise BaselineModelError("X contains no features.")
    if not np.all(np.isfinite(matrix)):
        raise BaselineModelError("X contains NaN or infinite values.")
    return matrix


def validate_target(y: np.ndarray, n_rows: int) -> np.ndarray:
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if target.size != n_rows:
        raise BaselineModelError(
            f"y contains {target.size} rows but X contains {n_rows}."
        )
    if target.size == 0:
        raise BaselineModelError("y contains no observations.")
    if not np.all(np.isfinite(target)):
        raise BaselineModelError("y contains NaN or infinite values.")
    return target


def naive_prediction(X: np.ndarray) -> np.ndarray:
    """Persistence forecast: y_hat_t = y_(t-1)."""
    matrix = validate_feature_matrix(X)
    return matrix[:, 0].astype(np.float64, copy=True)


def seasonal_naive_prediction(
    values: np.ndarray,
    target_indices: np.ndarray,
    *,
    seasonal_period: int,
) -> np.ndarray:
    """
    Predict each target from the same season in the observed past.

    `values` must be a finite causally available series. Every target index
    must be at least `seasonal_period`.
    """
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    indices = np.asarray(target_indices, dtype=np.int64).reshape(-1)

    if seasonal_period <= 0:
        raise BaselineModelError(
            "seasonal_period must be strictly positive."
        )
    if not np.all(np.isfinite(series)):
        raise BaselineModelError(
            "values contains NaN or infinite entries."
        )
    if indices.size == 0:
        raise BaselineModelError(
            "target_indices contains no observations."
        )
    if np.any(indices < seasonal_period):
        raise BaselineModelError(
            "Every target index must have seasonal history."
        )
    if np.any(indices >= len(series)):
        raise BaselineModelError(
            "target_indices contains an out-of-range index."
        )

    return series[indices - seasonal_period]


def fit_baseline(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float | None = None,
    lasso_max_iter: int = 50000,
    lasso_tolerance: float = 1e-6,
) -> FittedBaseline:
    matrix = validate_feature_matrix(X)
    target = validate_target(y, matrix.shape[0])
    normalized = model_name.strip().lower()

    if normalized == "naive":
        return FittedBaseline(
            model_name="naive",
            estimator=None,
            alpha=None,
            warning_messages=(),
        )

    if normalized == "ols":
        estimator = LinearRegression(fit_intercept=True)
    elif normalized == "ridge":
        if alpha is None or alpha < 0.0:
            raise BaselineModelError(
                "Ridge requires alpha >= 0."
            )
        estimator = Ridge(
            alpha=float(alpha),
            fit_intercept=True,
            solver="auto",
        )
    elif normalized == "lasso":
        if alpha is None or alpha <= 0.0:
            raise BaselineModelError(
                "Lasso requires alpha > 0."
            )
        estimator = Lasso(
            alpha=float(alpha),
            fit_intercept=True,
            max_iter=int(lasso_max_iter),
            tol=float(lasso_tolerance),
            selection="cyclic",
        )
    else:
        raise BaselineModelError(
            f"Unknown baseline model: {model_name!r}."
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.simplefilter("always", RuntimeWarning)
        estimator.fit(matrix, target)

    messages = tuple(
        f"{warning.category.__name__}: {warning.message}"
        for warning in caught
    )

    return FittedBaseline(
        model_name=normalized,
        estimator=estimator,
        alpha=(float(alpha) if alpha is not None else None),
        warning_messages=messages,
    )


def inverse_scale(
    values_scaled: np.ndarray,
    *,
    mean: float,
    scale: float,
) -> np.ndarray:
    values = np.asarray(values_scaled, dtype=np.float64)
    if not np.isfinite(mean):
        raise BaselineModelError("Scaler mean must be finite.")
    if not np.isfinite(scale) or scale <= 0.0:
        raise BaselineModelError(
            "Scaler scale must be finite and positive."
        )
    result = values * float(scale) + float(mean)
    if not np.all(np.isfinite(result)):
        raise BaselineModelError(
            "Inverse scaling produced non-finite values."
        )
    return result
