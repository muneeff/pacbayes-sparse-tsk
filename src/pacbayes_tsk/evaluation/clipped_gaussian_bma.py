from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr


class ClippedGaussianBMAError(ValueError):
    """Raised when an exact clipped-Gaussian BMA calculation is invalid."""


_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _finite_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ClippedGaussianBMAError(f"{name} is empty.")
    if not np.all(np.isfinite(array)):
        raise ClippedGaussianBMAError(f"{name} contains non-finite values.")
    return array


def _standard_normal_pdf(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    magnitude = np.abs(array)
    safe_magnitude = np.minimum(magnitude, 40.0)
    density = np.exp(-0.5 * safe_magnitude * safe_magnitude) / _SQRT_2PI
    return np.where(magnitude <= 40.0, density, 0.0)


def _z_times_pdf(values: np.ndarray, density: np.ndarray) -> np.ndarray:
    product = np.zeros_like(values, dtype=np.float64)
    np.multiply(
        values,
        density,
        out=product,
        where=np.isfinite(values),
    )
    return product


def clipped_gaussian_mean(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Compute E[clip(Z, lower, upper)] exactly for Gaussian Z."""
    mu = _finite_array(mean, name="mean")
    sigma = _finite_array(std, name="std")
    try:
        mu, sigma = np.broadcast_arrays(mu, sigma)
    except ValueError as error:
        raise ClippedGaussianBMAError(
            "mean and std are not broadcast-compatible."
        ) from error

    lo = float(lower)
    hi = float(upper)
    if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
        raise ClippedGaussianBMAError(
            "lower and upper must be finite with lower < upper."
        )
    if np.any(sigma < 0.0):
        raise ClippedGaussianBMAError("std must be nonnegative.")

    result = np.clip(mu, lo, hi).astype(np.float64, copy=True)
    positive = sigma > 0.0
    if not np.any(positive):
        return result

    mu_pos = mu[positive]
    sigma_pos = sigma[positive]
    alpha = (lo - mu_pos) / sigma_pos
    beta = (hi - mu_pos) / sigma_pos
    cdf_alpha = ndtr(alpha)
    cdf_beta = ndtr(beta)
    pdf_alpha = _standard_normal_pdf(alpha)
    pdf_beta = _standard_normal_pdf(beta)

    result[positive] = (
        lo * cdf_alpha
        + mu_pos * (cdf_beta - cdf_alpha)
        + sigma_pos * (pdf_alpha - pdf_beta)
        + hi * ndtr(-beta)
    )
    return np.clip(result, lo, hi)


def clipped_gaussian_second_moment(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Compute E[clip(Z, lower, upper)^2] exactly for Gaussian Z."""
    mu = _finite_array(mean, name="mean")
    sigma = _finite_array(std, name="std")
    try:
        mu, sigma = np.broadcast_arrays(mu, sigma)
    except ValueError as error:
        raise ClippedGaussianBMAError(
            "mean and std are not broadcast-compatible."
        ) from error

    lo = float(lower)
    hi = float(upper)
    if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
        raise ClippedGaussianBMAError(
            "lower and upper must be finite with lower < upper."
        )
    if np.any(sigma < 0.0):
        raise ClippedGaussianBMAError("std must be nonnegative.")

    result = np.square(np.clip(mu, lo, hi)).astype(np.float64, copy=True)
    positive = sigma > 0.0
    if not np.any(positive):
        return result

    mu_pos = mu[positive]
    sigma_pos = sigma[positive]
    alpha = (lo - mu_pos) / sigma_pos
    beta = (hi - mu_pos) / sigma_pos
    cdf_alpha = ndtr(alpha)
    cdf_beta = ndtr(beta)
    pdf_alpha = _standard_normal_pdf(alpha)
    pdf_beta = _standard_normal_pdf(beta)
    probability = cdf_beta - cdf_alpha
    first = pdf_alpha - pdf_beta
    second = (
        probability
        + _z_times_pdf(alpha, pdf_alpha)
        - _z_times_pdf(beta, pdf_beta)
    )

    interior = (
        np.square(mu_pos) * probability
        + 2.0 * mu_pos * sigma_pos * first
        + np.square(sigma_pos) * second
    )
    result[positive] = (
        lo * lo * cdf_alpha
        + interior
        + hi * hi * ndtr(-beta)
    )
    maximum_second_moment = max(lo * lo, hi * hi)
    return np.clip(result, 0.0, maximum_second_moment)


def clipped_gaussian_bma(
    design: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    *,
    clip_bound: float,
) -> np.ndarray:
    """Return the exact clipped Bayesian model-average prediction."""
    matrix = _finite_array(design, name="design")
    mean = _finite_array(
        posterior_mean,
        name="posterior_mean",
    ).reshape(-1)
    std = _finite_array(
        posterior_std,
        name="posterior_std",
    ).reshape(-1)

    if matrix.ndim != 2:
        raise ClippedGaussianBMAError("design must be two-dimensional.")
    if matrix.shape[1] != len(mean) or len(std) != len(mean):
        raise ClippedGaussianBMAError(
            "design, posterior_mean, and posterior_std dimensions do not match."
        )
    if np.any(std < 0.0):
        raise ClippedGaussianBMAError(
            "posterior_std must be nonnegative."
        )

    bound = float(clip_bound)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ClippedGaussianBMAError(
            "clip_bound must be finite and positive."
        )

    predictive_mean = matrix @ mean
    predictive_variance = np.square(matrix) @ np.square(std)
    predictive_std = np.sqrt(np.maximum(predictive_variance, 0.0))
    return clipped_gaussian_mean(
        predictive_mean,
        predictive_std,
        lower=-bound,
        upper=bound,
    )


def expected_normalized_clipped_squared_loss(
    design: np.ndarray,
    target: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    *,
    clip_bound: float,
) -> np.ndarray:
    """Return exact E[(clip(Y)-clip(target))^2]/(4 B^2) pointwise."""
    matrix = _finite_array(design, name="design")
    y = _finite_array(target, name="target").reshape(-1)
    mean = _finite_array(
        posterior_mean,
        name="posterior_mean",
    ).reshape(-1)
    std = _finite_array(
        posterior_std,
        name="posterior_std",
    ).reshape(-1)

    if matrix.ndim != 2 or matrix.shape != (len(y), len(mean)):
        raise ClippedGaussianBMAError(
            "design dimensions do not match target and posterior_mean."
        )
    if len(std) != len(mean) or np.any(std < 0.0):
        raise ClippedGaussianBMAError("posterior_std is invalid.")

    bound = float(clip_bound)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ClippedGaussianBMAError(
            "clip_bound must be finite and positive."
        )

    predictive_mean = matrix @ mean
    predictive_variance = np.square(matrix) @ np.square(std)
    predictive_std = np.sqrt(np.maximum(predictive_variance, 0.0))
    clipped_mean = clipped_gaussian_mean(
        predictive_mean,
        predictive_std,
        lower=-bound,
        upper=bound,
    )
    clipped_second = clipped_gaussian_second_moment(
        predictive_mean,
        predictive_std,
        lower=-bound,
        upper=bound,
    )
    clipped_target = np.clip(y, -bound, bound)
    raw_loss = (
        clipped_second
        - 2.0 * clipped_target * clipped_mean
        + np.square(clipped_target)
    )
    return np.clip(
        raw_loss / (4.0 * bound * bound),
        0.0,
        1.0,
    )
