from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class StatisticalBaselineError(ValueError):
    """Raised when an ARIMA/SARIMA baseline cannot be fitted safely."""


@dataclass(frozen=True)
class StatisticalModelSpec:
    model_name: str
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    trend: str

    def __post_init__(self) -> None:
        normalized = self.model_name.strip().lower()
        if normalized not in {"arima", "sarima"}:
            raise StatisticalBaselineError(
                f"Unknown statistical model: {self.model_name!r}."
            )

        if len(self.order) != 3:
            raise StatisticalBaselineError(
                "order must contain exactly (p, d, q)."
            )
        if len(self.seasonal_order) != 4:
            raise StatisticalBaselineError(
                "seasonal_order must contain exactly (P, D, Q, m)."
            )

        if any(
            isinstance(value, bool) or int(value) != value or value < 0
            for value in (*self.order, *self.seasonal_order)
        ):
            raise StatisticalBaselineError(
                "ARIMA and seasonal orders must be non-negative integers."
            )

        p, d, q = (int(value) for value in self.order)
        P, D, Q, m = (
            int(value) for value in self.seasonal_order
        )

        if normalized == "arima" and (P, D, Q, m) != (0, 0, 0, 0):
            raise StatisticalBaselineError(
                "ARIMA must use seasonal_order=(0, 0, 0, 0)."
            )
        if normalized == "sarima":
            if m <= 1:
                raise StatisticalBaselineError(
                    "SARIMA requires a seasonal period greater than one."
                )
            if P == D == Q == 0:
                raise StatisticalBaselineError(
                    "SARIMA requires at least one seasonal term."
                )

        if self.trend not in {"n", "c", "t", "ct"}:
            raise StatisticalBaselineError(
                "trend must be one of: n, c, t, ct."
            )

        object.__setattr__(self, "model_name", normalized)
        object.__setattr__(self, "order", (p, d, q))
        object.__setattr__(
            self,
            "seasonal_order",
            (P, D, Q, m),
        )

    @property
    def candidate_id(self) -> str:
        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        return (
            f"{self.model_name}"
            f"__p{p}_d{d}_q{q}"
            f"__P{P}_D{D}_Q{Q}_m{m}"
            f"__trend_{self.trend}"
        )

    @property
    def nominal_complexity(self) -> int:
        p, d, q = self.order
        P, D, Q, _ = self.seasonal_order
        trend_parameters = {
            "n": 0,
            "c": 1,
            "t": 1,
            "ct": 2,
        }[self.trend]
        return int(
            p + q + P + Q + trend_parameters
        )


@dataclass
class FittedStatisticalModel:
    spec: StatisticalModelSpec
    results: Any
    warning_messages: tuple[str, ...]
    converged: bool
    iterations: int | None
    aic: float
    bic: float
    log_likelihood: float
    n_parameters: int
    n_train: int

    def one_step_predictions(
        self,
        update_values: np.ndarray,
        *,
        prediction_offset: int = 0,
    ) -> np.ndarray:
        """
        Return one-step-ahead predictions while conditioning on observed past.

        `update_values` are appended without parameter refitting. Prediction at
        time t uses observations only through t-1. `prediction_offset` can
        discard an initial state-update segment, e.g. validation before test.
        """
        updates = validate_series(
            update_values,
            name="update_values",
        )
        if not isinstance(prediction_offset, int):
            raise StatisticalBaselineError(
                "prediction_offset must be an integer."
            )
        if prediction_offset < 0 or prediction_offset >= len(updates):
            raise StatisticalBaselineError(
                "prediction_offset must satisfy "
                "0 <= prediction_offset < len(update_values)."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extended = self.results.append(
                updates,
                refit=False,
            )
            start = self.n_train
            end = self.n_train + len(updates) - 1
            prediction = extended.get_prediction(
                start=start,
                end=end,
                dynamic=False,
            ).predicted_mean

        values = np.asarray(
            prediction,
            dtype=np.float64,
        ).reshape(-1)
        if values.size != len(updates):
            raise StatisticalBaselineError(
                "Statsmodels returned an unexpected prediction length."
            )
        values = values[prediction_offset:]
        if not np.all(np.isfinite(values)):
            raise StatisticalBaselineError(
                "The statistical model produced non-finite predictions."
            )
        return values


def validate_series(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if series.size == 0:
        raise StatisticalBaselineError(
            f"{name} contains no observations."
        )
    if not np.all(np.isfinite(series)):
        raise StatisticalBaselineError(
            f"{name} contains NaN or infinite values."
        )
    return series


def trend_for_orders(
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> str:
    """
    Use an intercept only when the model has no ordinary/seasonal differencing.
    """
    _, d, _ = order
    _, D, _, _ = seasonal_order
    return "c" if int(d) + int(D) == 0 else "n"


def make_arima_spec(
    order: tuple[int, int, int] | list[int],
) -> StatisticalModelSpec:
    normalized_order = tuple(int(value) for value in order)
    seasonal_order = (0, 0, 0, 0)
    return StatisticalModelSpec(
        model_name="arima",
        order=normalized_order,
        seasonal_order=seasonal_order,
        trend=trend_for_orders(
            normalized_order,
            seasonal_order,
        ),
    )


def make_sarima_spec(
    order: tuple[int, int, int] | list[int],
    seasonal_terms: tuple[int, int, int] | list[int],
    *,
    seasonal_period: int,
) -> StatisticalModelSpec:
    normalized_order = tuple(int(value) for value in order)
    P, D, Q = (
        int(value) for value in seasonal_terms
    )
    seasonal_order = (
        P,
        D,
        Q,
        int(seasonal_period),
    )
    return StatisticalModelSpec(
        model_name="sarima",
        order=normalized_order,
        seasonal_order=seasonal_order,
        trend=trend_for_orders(
            normalized_order,
            seasonal_order,
        ),
    )


def minimum_training_length(
    spec: StatisticalModelSpec,
) -> int:
    """
    Conservative structural minimum before calling statsmodels.
    """
    p, d, q = spec.order
    P, D, Q, m = spec.seasonal_order
    seasonal_memory = (
        (P + D + Q) * m
        if m > 0
        else 0
    )
    ordinary_memory = p + d + q
    return max(
        20,
        ordinary_memory + seasonal_memory + 8,
    )


def fit_statistical_model(
    y_train: np.ndarray,
    spec: StatisticalModelSpec,
    *,
    max_iter: int = 200,
    method: str = "lbfgs",
    tolerance: float = 1e-6,
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
) -> FittedStatisticalModel:
    training = validate_series(
        y_train,
        name="y_train",
    )

    required = minimum_training_length(spec)
    if len(training) < required:
        raise StatisticalBaselineError(
            f"Training length {len(training)} is below the structural "
            f"minimum {required} for {spec.candidate_id}."
        )
    if max_iter <= 0:
        raise StatisticalBaselineError(
            "max_iter must be strictly positive."
        )
    if tolerance <= 0.0:
        raise StatisticalBaselineError(
            "tolerance must be strictly positive."
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SARIMAX(
            training,
            order=spec.order,
            seasonal_order=spec.seasonal_order,
            trend=spec.trend,
            enforce_stationarity=bool(
                enforce_stationarity
            ),
            enforce_invertibility=bool(
                enforce_invertibility
            ),
            simple_differencing=False,
        )
        results = model.fit(
            disp=False,
            maxiter=int(max_iter),
            method=str(method),
            pgtol=float(tolerance),
        )

    warning_messages = tuple(
        f"{warning.category.__name__}: {warning.message}"
        for warning in caught
    )

    mle_retvals = getattr(
        results,
        "mle_retvals",
        {},
    ) or {}
    converged = bool(
        mle_retvals.get("converged", True)
    )
    raw_iterations = mle_retvals.get("iterations")
    iterations = (
        int(raw_iterations)
        if raw_iterations is not None
        else None
    )

    metadata_values = {
        "aic": float(results.aic),
        "bic": float(results.bic),
        "log_likelihood": float(results.llf),
    }
    if not all(
        np.isfinite(value)
        for value in metadata_values.values()
    ):
        raise StatisticalBaselineError(
            "The fitted model has non-finite likelihood criteria."
        )

    params = np.asarray(
        results.params,
        dtype=np.float64,
    ).reshape(-1)
    if not np.all(np.isfinite(params)):
        raise StatisticalBaselineError(
            "The fitted model has non-finite parameters."
        )

    return FittedStatisticalModel(
        spec=spec,
        results=results,
        warning_messages=warning_messages,
        converged=converged,
        iterations=iterations,
        aic=metadata_values["aic"],
        bic=metadata_values["bic"],
        log_likelihood=metadata_values[
            "log_likelihood"
        ],
        n_parameters=int(params.size),
        n_train=int(len(training)),
    )


def inverse_scale(
    values_scaled: np.ndarray,
    *,
    mean: float,
    scale: float,
) -> np.ndarray:
    values = np.asarray(
        values_scaled,
        dtype=np.float64,
    )
    if not np.isfinite(mean):
        raise StatisticalBaselineError(
            "Scaler mean must be finite."
        )
    if not np.isfinite(scale) or scale <= 0.0:
        raise StatisticalBaselineError(
            "Scaler scale must be finite and positive."
        )
    output = (
        values * float(scale)
        + float(mean)
    )
    if not np.all(np.isfinite(output)):
        raise StatisticalBaselineError(
            "Inverse scaling produced non-finite values."
        )
    return output
