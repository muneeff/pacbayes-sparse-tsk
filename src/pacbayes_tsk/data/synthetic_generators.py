"""Reproducible generators for synthetic univariate time series.

The generators in this module are intentionally self-contained.  They use
``numpy.random.default_rng`` and return both observations and regime labels so
that later experiments can evaluate forecasting accuracy and behaviour under
nonlinearity, heteroskedasticity, and structural change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SyntheticSeries:
    """Container returned by every generator.

    Attributes
    ----------
    values:
        One-dimensional array containing the generated observations after the
        burn-in period.
    regimes:
        Integer regime labels aligned with ``values``.  Stationary generators
        use zero throughout; SETAR and structural-break generators use labels
        0 and 1.
    metadata:
        Generator-specific information useful for manifests and diagnostics.
    """

    values: FloatArray
    regimes: IntArray
    metadata: dict[str, Any]

    def validate(self, expected_length: int) -> None:
        """Validate shape and numerical integrity."""
        if self.values.ndim != 1 or self.regimes.ndim != 1:
            raise ValueError("Generated values and regimes must be one-dimensional.")
        if len(self.values) != expected_length:
            raise ValueError(
                f"Expected {expected_length} observations, got {len(self.values)}."
            )
        if len(self.regimes) != expected_length:
            raise ValueError("Regime labels must have the same length as values.")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("Generated series contains NaN or infinite values.")


def _validate_lengths(length: int, burn_in: int) -> None:
    if length < 20:
        raise ValueError("length must be at least 20.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")


def _finalize(
    values: NDArray[np.floating[Any]],
    regimes: NDArray[np.integer[Any]],
    *,
    length: int,
    burn_in: int,
    metadata: Mapping[str, Any],
) -> SyntheticSeries:
    start = burn_in
    stop = burn_in + length
    result = SyntheticSeries(
        values=np.asarray(values[start:stop], dtype=np.float64),
        regimes=np.asarray(regimes[start:stop], dtype=np.int64),
        metadata=dict(metadata),
    )
    result.validate(length)
    return result


def generate_ar(
    *,
    length: int,
    burn_in: int,
    seed: int,
    phi: Sequence[float] = (0.6, -0.2),
    intercept: float = 0.0,
    noise_std: float = 0.5,
) -> SyntheticSeries:
    """Generate an autoregressive AR(p) series."""
    _validate_lengths(length, burn_in)
    phi_array = np.asarray(phi, dtype=np.float64)
    if phi_array.ndim != 1 or len(phi_array) == 0:
        raise ValueError("phi must contain at least one autoregressive coefficient.")
    if noise_std <= 0:
        raise ValueError("noise_std must be positive.")

    rng = np.random.default_rng(seed)
    order = len(phi_array)
    total = length + burn_in + order
    values = np.zeros(total, dtype=np.float64)
    values[:order] = rng.normal(0.0, noise_std, size=order)

    for t in range(order, total):
        lags = values[t - order : t][::-1]
        values[t] = intercept + float(phi_array @ lags) + rng.normal(0.0, noise_std)

    regimes = np.zeros(total, dtype=np.int64)
    return _finalize(
        values[order:],
        regimes[order:],
        length=length,
        burn_in=burn_in,
        metadata={
            "generator": "ar",
            "order": order,
            "phi": phi_array.tolist(),
            "intercept": intercept,
            "noise_std": noise_std,
        },
    )


def generate_setar(
    *,
    length: int,
    burn_in: int,
    seed: int,
    threshold: float = 0.0,
    delay: int = 1,
    low_intercept: float = -0.15,
    low_phi: Sequence[float] = (0.25, -0.10),
    high_intercept: float = 0.20,
    high_phi: Sequence[float] = (0.65, -0.20),
    noise_std: float = 0.35,
) -> SyntheticSeries:
    """Generate a two-regime self-exciting threshold AR process."""
    _validate_lengths(length, burn_in)
    low = np.asarray(low_phi, dtype=np.float64)
    high = np.asarray(high_phi, dtype=np.float64)
    if low.ndim != 1 or high.ndim != 1 or len(low) == 0 or len(low) != len(high):
        raise ValueError("low_phi and high_phi must have the same non-zero length.")
    if delay < 1:
        raise ValueError("delay must be at least one.")
    if noise_std <= 0:
        raise ValueError("noise_std must be positive.")

    rng = np.random.default_rng(seed)
    order = len(low)
    warmup = max(order, delay)
    total = length + burn_in + warmup
    values = np.zeros(total, dtype=np.float64)
    regimes = np.zeros(total, dtype=np.int64)
    values[:warmup] = rng.normal(0.0, noise_std, size=warmup)

    for t in range(warmup, total):
        state = values[t - delay]
        if state <= threshold:
            coeffs = low
            intercept = low_intercept
            regimes[t] = 0
        else:
            coeffs = high
            intercept = high_intercept
            regimes[t] = 1
        lags = values[t - order : t][::-1]
        values[t] = intercept + float(coeffs @ lags) + rng.normal(0.0, noise_std)

    return _finalize(
        values[warmup:],
        regimes[warmup:],
        length=length,
        burn_in=burn_in,
        metadata={
            "generator": "setar",
            "threshold": threshold,
            "delay": delay,
            "low_intercept": low_intercept,
            "low_phi": low.tolist(),
            "high_intercept": high_intercept,
            "high_phi": high.tolist(),
            "noise_std": noise_std,
        },
    )


def generate_narma10(
    *,
    length: int,
    burn_in: int,
    seed: int,
    input_low: float = 0.0,
    input_high: float = 0.5,
    observation_noise_std: float = 0.0,
) -> SyntheticSeries:
    """Generate the standard nonlinear NARMA-10 benchmark process."""
    _validate_lengths(length, burn_in)
    if input_high <= input_low:
        raise ValueError("input_high must be greater than input_low.")
    if observation_noise_std < 0:
        raise ValueError("observation_noise_std cannot be negative.")

    rng = np.random.default_rng(seed)
    order = 10
    total = length + burn_in + order + 1
    inputs = rng.uniform(input_low, input_high, size=total)
    values = np.zeros(total, dtype=np.float64)

    for t in range(order, total - 1):
        recent_sum = float(np.sum(values[t - order + 1 : t + 1]))
        next_value = (
            0.3 * values[t]
            + 0.05 * values[t] * recent_sum
            + 1.5 * inputs[t - 9] * inputs[t]
            + 0.1
        )
        if observation_noise_std > 0:
            next_value += rng.normal(0.0, observation_noise_std)
        values[t + 1] = next_value

    regimes = np.zeros(total, dtype=np.int64)
    return _finalize(
        values[order + 1 :],
        regimes[order + 1 :],
        length=length,
        burn_in=burn_in,
        metadata={
            "generator": "narma10",
            "order": order,
            "input_low": input_low,
            "input_high": input_high,
            "observation_noise_std": observation_noise_std,
        },
    )


def generate_mackey_glass(
    *,
    length: int,
    burn_in: int,
    seed: int,
    beta: float = 0.2,
    gamma: float = 0.1,
    exponent: int = 10,
    tau: int = 17,
    dt: float = 1.0,
    initial_value: float = 1.2,
    initial_jitter: float = 0.02,
) -> SyntheticSeries:
    """Generate a discretised Mackey--Glass delay differential process."""
    _validate_lengths(length, burn_in)
    if tau < 1:
        raise ValueError("tau must be at least one.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if beta <= 0 or gamma <= 0 or exponent < 1:
        raise ValueError("beta, gamma, and exponent must be positive.")

    rng = np.random.default_rng(seed)
    total = length + burn_in + tau + 1
    values = np.full(total, initial_value, dtype=np.float64)
    values[: tau + 1] += rng.normal(0.0, initial_jitter, size=tau + 1)

    for t in range(tau, total - 1):
        delayed = values[t - tau]
        derivative = beta * delayed / (1.0 + delayed**exponent) - gamma * values[t]
        values[t + 1] = values[t] + dt * derivative

    regimes = np.zeros(total, dtype=np.int64)
    return _finalize(
        values[tau + 1 :],
        regimes[tau + 1 :],
        length=length,
        burn_in=burn_in,
        metadata={
            "generator": "mackey_glass",
            "beta": beta,
            "gamma": gamma,
            "exponent": exponent,
            "tau": tau,
            "dt": dt,
            "initial_value": initial_value,
            "initial_jitter": initial_jitter,
        },
    )


def generate_garch(
    *,
    length: int,
    burn_in: int,
    seed: int,
    omega: float = 0.05,
    alpha: float = 0.10,
    beta: float = 0.85,
    degrees_of_freedom: float = 6.0,
    mean: float = 0.0,
) -> SyntheticSeries:
    """Generate a GARCH(1,1) process with standardized Student-t innovations."""
    _validate_lengths(length, burn_in)
    if omega <= 0 or alpha < 0 or beta < 0:
        raise ValueError("omega must be positive; alpha and beta must be non-negative.")
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be smaller than one for covariance stationarity.")
    if degrees_of_freedom <= 2:
        raise ValueError("degrees_of_freedom must exceed two.")

    rng = np.random.default_rng(seed)
    total = length + burn_in
    values = np.zeros(total, dtype=np.float64)
    variances = np.empty(total, dtype=np.float64)
    variances[0] = omega / (1.0 - alpha - beta)

    # A Student-t random variable has variance df/(df-2).  This scaling gives
    # unit-variance innovations so the GARCH recursion has its usual meaning.
    innovation_scale = np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
    innovations = rng.standard_t(degrees_of_freedom, size=total) * innovation_scale
    values[0] = mean + np.sqrt(variances[0]) * innovations[0]

    for t in range(1, total):
        previous_residual = values[t - 1] - mean
        variances[t] = (
            omega + alpha * previous_residual**2 + beta * variances[t - 1]
        )
        values[t] = mean + np.sqrt(variances[t]) * innovations[t]

    regimes = np.zeros(total, dtype=np.int64)
    return _finalize(
        values,
        regimes,
        length=length,
        burn_in=burn_in,
        metadata={
            "generator": "garch",
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "degrees_of_freedom": degrees_of_freedom,
            "mean": mean,
            "unconditional_variance": omega / (1.0 - alpha - beta),
        },
    )


def generate_structural_break(
    *,
    length: int,
    burn_in: int,
    seed: int,
    break_fraction: float = 0.5,
    pre_mean: float = 0.0,
    pre_phi: float = 0.75,
    pre_noise_std: float = 0.30,
    post_mean: float = 1.5,
    post_phi: float = 0.30,
    post_noise_std: float = 0.55,
) -> SyntheticSeries:
    """Generate an AR(1) process with a persistent parameter break."""
    _validate_lengths(length, burn_in)
    if not 0.05 <= break_fraction <= 0.95:
        raise ValueError("break_fraction must be between 0.05 and 0.95.")
    if abs(pre_phi) >= 1 or abs(post_phi) >= 1:
        raise ValueError("pre_phi and post_phi must lie strictly between -1 and 1.")
    if pre_noise_std <= 0 or post_noise_std <= 0:
        raise ValueError("Noise standard deviations must be positive.")

    rng = np.random.default_rng(seed)
    total = length + burn_in
    break_index = burn_in + int(round(length * break_fraction))
    break_index = min(max(burn_in + 1, break_index), total - 1)

    values = np.zeros(total, dtype=np.float64)
    regimes = np.zeros(total, dtype=np.int64)
    values[0] = rng.normal(pre_mean, pre_noise_std)

    for t in range(1, total):
        if t < break_index:
            process_mean = pre_mean
            phi = pre_phi
            noise_std = pre_noise_std
            regimes[t] = 0
        else:
            process_mean = post_mean
            phi = post_phi
            noise_std = post_noise_std
            regimes[t] = 1
        values[t] = (
            process_mean
            + phi * (values[t - 1] - process_mean)
            + rng.normal(0.0, noise_std)
        )

    break_index_after_burn_in = int(round(length * break_fraction))
    return _finalize(
        values,
        regimes,
        length=length,
        burn_in=burn_in,
        metadata={
            "generator": "structural_break",
            "break_fraction": break_fraction,
            "break_index_after_burn_in": break_index_after_burn_in,
            "pre_mean": pre_mean,
            "pre_phi": pre_phi,
            "pre_noise_std": pre_noise_std,
            "post_mean": post_mean,
            "post_phi": post_phi,
            "post_noise_std": post_noise_std,
        },
    )


Generator = Callable[..., SyntheticSeries]

_GENERATORS: dict[str, Generator] = {
    "ar": generate_ar,
    "setar": generate_setar,
    "narma": generate_narma10,
    "narma10": generate_narma10,
    "mackey_glass": generate_mackey_glass,
    "garch": generate_garch,
    "structural_break": generate_structural_break,
}


def generate_synthetic_series(
    generator: str,
    *,
    length: int,
    burn_in: int,
    seed: int,
    parameters: Mapping[str, Any] | None = None,
) -> SyntheticSeries:
    """Dispatch to a named generator.

    Parameters in ``parameters`` are passed directly to the selected generator.
    """
    normalized_name = generator.strip().lower().replace("-", "_")
    try:
        function = _GENERATORS[normalized_name]
    except KeyError as exc:
        available = ", ".join(sorted(_GENERATORS))
        raise ValueError(
            f"Unknown generator '{generator}'. Available generators: {available}."
        ) from exc

    kwargs = dict(parameters or {})
    return function(length=length, burn_in=burn_in, seed=seed, **kwargs)
