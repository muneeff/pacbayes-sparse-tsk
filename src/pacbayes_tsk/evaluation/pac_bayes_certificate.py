from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


class PACBayesCertificateError(ValueError):
    """Raised when a PAC-Bayes certificate cannot be computed safely."""


@dataclass(frozen=True)
class CertificateChoice:
    prior_scale_multiplier: float
    posterior_scale_ratio: float
    temperature: float
    empirical_gibbs_risk_upper: float
    gaussian_kl: float
    structure_kl: float
    prior_component_penalty: float
    total_kl_upper: float
    certificate: float


def _as_finite_vector(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    vector = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)
    if vector.size == 0:
        raise PACBayesCertificateError(
            f"{name} is empty."
        )
    if not np.all(np.isfinite(vector)):
        raise PACBayesCertificateError(
            f"{name} contains non-finite values."
        )
    return vector


def stable_logsumexp(values: np.ndarray) -> float:
    vector = _as_finite_vector(
        values,
        name="log weights",
    )
    maximum = float(np.max(vector))
    return maximum + float(
        np.log(
            np.sum(
                np.exp(vector - maximum)
            )
        )
    )


def gaussian_kl_diag(
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
) -> float:
    mu_q = _as_finite_vector(
        posterior_mean,
        name="posterior_mean",
    )
    sigma_q = _as_finite_vector(
        posterior_std,
        name="posterior_std",
    )
    mu_p = _as_finite_vector(
        prior_mean,
        name="prior_mean",
    )
    sigma_p = _as_finite_vector(
        prior_std,
        name="prior_std",
    )
    if not (
        len(mu_q)
        == len(sigma_q)
        == len(mu_p)
        == len(sigma_p)
    ):
        raise PACBayesCertificateError(
            "Gaussian vectors must have equal dimensions."
        )
    if np.any(sigma_q <= 0.0) or np.any(sigma_p <= 0.0):
        raise PACBayesCertificateError(
            "Gaussian standard deviations must be positive."
        )

    ratio = np.square(sigma_q / sigma_p)
    displacement = np.square(
        (mu_q - mu_p) / sigma_p
    )
    kl = 0.5 * np.sum(
        -np.log(ratio)
        - 1.0
        + ratio
        + displacement
    )
    if not np.isfinite(kl) or kl < -1e-10:
        raise PACBayesCertificateError(
            f"Invalid Gaussian KL value: {kl}."
        )
    return float(max(0.0, kl))


def prior_diagonal_std(
    design: np.ndarray,
    target: np.ndarray,
    prior_mean: np.ndarray,
    *,
    ridge_alpha: float,
    variance_floor: float,
    std_floor: float,
    std_ceiling: float,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(
        design,
        dtype=np.float64,
    )
    y = _as_finite_vector(
        target,
        name="prior target",
    )
    mu = _as_finite_vector(
        prior_mean,
        name="prior mean",
    )
    if matrix.ndim != 2:
        raise PACBayesCertificateError(
            "Prior design must be a matrix."
        )
    if matrix.shape != (len(y), len(mu)):
        raise PACBayesCertificateError(
            "Prior design dimensions do not match target and mean."
        )
    if ridge_alpha < 0.0:
        raise PACBayesCertificateError(
            "ridge_alpha cannot be negative."
        )

    residual = y - matrix @ mu
    noise_variance = max(
        float(np.mean(np.square(residual))),
        float(variance_floor),
    )
    gram = (
        matrix.T @ matrix
        + (
            float(ridge_alpha)
            + 1.0e-10
        )
        * np.eye(matrix.shape[1])
    )
    inverse = np.linalg.pinv(
        gram,
        hermitian=True,
    )
    diagonal = np.maximum(
        np.diag(inverse),
        0.0,
    )
    std = np.sqrt(
        noise_variance * diagonal
    )
    std = np.clip(
        std,
        float(std_floor),
        float(std_ceiling),
    )
    if not np.all(np.isfinite(std)):
        raise PACBayesCertificateError(
            "Prior standard deviations are non-finite."
        )
    return std, noise_variance


def prior_only_clip_bound(
    prior_target: np.ndarray,
    *,
    minimum_bound: float,
) -> float:
    target = _as_finite_vector(
        prior_target,
        name="prior target",
    )
    if minimum_bound <= 0.0:
        raise PACBayesCertificateError(
            "minimum_bound must be positive."
        )
    return float(
        max(
            float(minimum_bound),
            float(np.max(np.abs(target))),
        )
    )


def bounded_gibbs_empirical_upper(
    design: np.ndarray,
    target: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    *,
    clip_bound: float,
) -> tuple[float, int, float]:
    matrix = np.asarray(
        design,
        dtype=np.float64,
    )
    y = _as_finite_vector(
        target,
        name="certification target",
    )
    mu = _as_finite_vector(
        posterior_mean,
        name="posterior mean",
    )
    sigma = _as_finite_vector(
        posterior_std,
        name="posterior std",
    )
    if matrix.ndim != 2:
        raise PACBayesCertificateError(
            "Certification design must be a matrix."
        )
    if matrix.shape != (len(y), len(mu)):
        raise PACBayesCertificateError(
            "Certification design dimensions do not match."
        )
    if len(sigma) != len(mu) or np.any(sigma <= 0.0):
        raise PACBayesCertificateError(
            "Posterior standard deviations are invalid."
        )
    if clip_bound <= 0.0:
        raise PACBayesCertificateError(
            "clip_bound must be positive."
        )

    clipped_target = np.clip(
        y,
        -float(clip_bound),
        float(clip_bound),
    )
    target_clip_count = int(
        np.sum(
            np.abs(y) > float(clip_bound)
        )
    )
    mean_prediction = matrix @ mu
    prediction_variance = (
        np.square(matrix)
        @ np.square(sigma)
    )

    second_moment_upper = (
        np.square(
            clipped_target
            - mean_prediction
        )
        + prediction_variance
    )
    pointwise_upper = np.minimum(
        1.0,
        second_moment_upper
        / (
            4.0
            * float(clip_bound) ** 2
        ),
    )
    empirical_upper = float(
        np.mean(pointwise_upper)
    )
    posterior_mean_loss = float(
        np.mean(
            np.square(
                clipped_target
                - np.clip(
                    mean_prediction,
                    -float(clip_bound),
                    float(clip_bound),
                )
            )
            / (
                4.0
                * float(clip_bound) ** 2
            )
        )
    )
    return (
        empirical_upper,
        target_clip_count,
        posterior_mean_loss,
    )


def structure_prior_penalty(
    candidates: pd.DataFrame,
    selected: pd.Series,
    *,
    eta_rule: float,
    eta_dimension: float,
    eta_lag: float,
) -> tuple[float, int]:
    required = {
        "lag_order",
        "radius",
        "ridge_alpha",
        "rule_count",
    }
    missing = required.difference(
        candidates.columns
    )
    if missing:
        raise PACBayesCertificateError(
            "Candidate structure table is missing: "
            f"{sorted(missing)}."
        )

    frame = candidates.copy()
    if "consequent_parameter_count" not in frame.columns:
        frame["consequent_parameter_count"] = (
            frame["rule_count"].astype(int)
            * (
                frame["lag_order"].astype(int)
                + 1
            )
        )

    frame = frame[
        [
            "lag_order",
            "radius",
            "ridge_alpha",
            "rule_count",
            "consequent_parameter_count",
        ]
    ].drop_duplicates()

    if frame.empty:
        raise PACBayesCertificateError(
            "Candidate structure table is empty."
        )

    scores = -(
        float(eta_rule)
        * frame["rule_count"].astype(float)
        + float(eta_dimension)
        * frame[
            "consequent_parameter_count"
        ].astype(float)
        + float(eta_lag)
        * frame["lag_order"].astype(float)
    )
    log_normalizer = stable_logsumexp(
        scores.to_numpy(float)
    )

    mask = (
        frame["lag_order"].astype(int).eq(
            int(selected["selected_lag"])
        )
        & np.isclose(
            frame["radius"].astype(float),
            float(selected["selected_radius"]),
            rtol=0.0,
            atol=1.0e-12,
        )
        & np.isclose(
            frame["ridge_alpha"].astype(float),
            float(
                selected[
                    "selected_ridge_alpha"
                ]
            ),
            rtol=0.0,
            atol=1.0e-12,
        )
        & frame["rule_count"].astype(int).eq(
            int(selected["rule_count"])
        )
    )
    matched = frame.loc[mask]
    if len(matched) != 1:
        raise PACBayesCertificateError(
            "Frozen structure does not match exactly one candidate; "
            f"matched {len(matched)} rows."
        )

    selected_score = -(
        float(eta_rule)
        * int(selected["rule_count"])
        + float(eta_dimension)
        * int(
            matched.iloc[0][
                "consequent_parameter_count"
            ]
        )
        + float(eta_lag)
        * int(selected["selected_lag"])
    )
    penalty = float(
        log_normalizer
        - selected_score
    )
    return penalty, int(len(frame))


def anytime_bounded_certificate(
    *,
    empirical_risk_upper: float,
    total_kl_upper: float,
    temperature: float,
    sample_size: int,
    delta: float,
    temperature_mass: float,
) -> float:
    if not 0.0 <= empirical_risk_upper <= 1.0:
        raise PACBayesCertificateError(
            "empirical_risk_upper must be in [0, 1]."
        )
    if total_kl_upper < 0.0:
        raise PACBayesCertificateError(
            "total_kl_upper cannot be negative."
        )
    if temperature <= 0.0:
        raise PACBayesCertificateError(
            "temperature must be positive."
        )
    if sample_size <= 0:
        raise PACBayesCertificateError(
            "sample_size must be positive."
        )
    if not 0.0 < delta < 1.0:
        raise PACBayesCertificateError(
            "delta must be in (0, 1)."
        )
    if not 0.0 < temperature_mass <= 1.0:
        raise PACBayesCertificateError(
            "temperature_mass must be in (0, 1]."
        )

    complexity = (
        float(total_kl_upper)
        + math_log(
            1.0
            / (
                float(delta)
                * float(temperature_mass)
            )
        )
    ) / (
        float(temperature)
        * int(sample_size)
    )
    value = (
        float(empirical_risk_upper)
        + float(temperature) / 8.0
        + complexity
    )
    return float(min(1.0, value))


def math_log(value: float) -> float:
    if value <= 0.0:
        raise PACBayesCertificateError(
            "Logarithm argument must be positive."
        )
    return float(np.log(value))


def search_certificate(
    *,
    design_cert: np.ndarray,
    target_cert: np.ndarray,
    posterior_mean: np.ndarray,
    prior_mean: np.ndarray,
    base_prior_std: np.ndarray,
    clip_bound: float,
    structure_kl: float,
    prior_scale_multipliers: Iterable[float],
    posterior_scale_ratios: Iterable[float],
    temperatures: Iterable[float],
    delta: float,
) -> tuple[CertificateChoice, int, float]:
    prior_scales = [
        float(value)
        for value in prior_scale_multipliers
    ]
    posterior_ratios = [
        float(value)
        for value in posterior_scale_ratios
    ]
    lambda_grid = [
        float(value)
        for value in temperatures
    ]
    if not prior_scales:
        raise PACBayesCertificateError(
            "At least one prior scale is required."
        )
    if not posterior_ratios:
        raise PACBayesCertificateError(
            "At least one posterior scale ratio is required."
        )
    if not lambda_grid:
        raise PACBayesCertificateError(
            "At least one temperature is required."
        )
    if any(value <= 0.0 for value in prior_scales):
        raise PACBayesCertificateError(
            "Prior scale multipliers must be positive."
        )
    if any(value <= 0.0 for value in posterior_ratios):
        raise PACBayesCertificateError(
            "Posterior scale ratios must be positive."
        )
    if any(value <= 0.0 for value in lambda_grid):
        raise PACBayesCertificateError(
            "Temperatures must be positive."
        )

    prior_component_penalty = math_log(
        len(prior_scales)
    )
    temperature_mass = 1.0 / len(lambda_grid)
    best: CertificateChoice | None = None
    best_clip_count = 0
    best_mean_loss = np.nan

    for prior_multiplier in prior_scales:
        prior_std = (
            np.asarray(
                base_prior_std,
                dtype=np.float64,
            )
            * prior_multiplier
        )
        for posterior_ratio in posterior_ratios:
            posterior_std = (
                prior_std
                * posterior_ratio
            )
            gaussian_kl = gaussian_kl_diag(
                posterior_mean,
                posterior_std,
                prior_mean,
                prior_std,
            )
            (
                empirical_upper,
                target_clip_count,
                posterior_mean_loss,
            ) = bounded_gibbs_empirical_upper(
                design_cert,
                target_cert,
                posterior_mean,
                posterior_std,
                clip_bound=clip_bound,
            )
            total_kl = (
                gaussian_kl
                + float(structure_kl)
                + prior_component_penalty
            )

            for temperature in lambda_grid:
                certificate = (
                    anytime_bounded_certificate(
                        empirical_risk_upper=(
                            empirical_upper
                        ),
                        total_kl_upper=total_kl,
                        temperature=temperature,
                        sample_size=len(target_cert),
                        delta=delta,
                        temperature_mass=(
                            temperature_mass
                        ),
                    )
                )
                choice = CertificateChoice(
                    prior_scale_multiplier=(
                        prior_multiplier
                    ),
                    posterior_scale_ratio=(
                        posterior_ratio
                    ),
                    temperature=temperature,
                    empirical_gibbs_risk_upper=(
                        empirical_upper
                    ),
                    gaussian_kl=gaussian_kl,
                    structure_kl=float(
                        structure_kl
                    ),
                    prior_component_penalty=(
                        prior_component_penalty
                    ),
                    total_kl_upper=total_kl,
                    certificate=certificate,
                )
                if (
                    best is None
                    or (
                        choice.certificate,
                        choice.total_kl_upper,
                        choice.empirical_gibbs_risk_upper,
                        choice.posterior_scale_ratio,
                        choice.prior_scale_multiplier,
                        choice.temperature,
                    )
                    < (
                        best.certificate,
                        best.total_kl_upper,
                        best.empirical_gibbs_risk_upper,
                        best.posterior_scale_ratio,
                        best.prior_scale_multiplier,
                        best.temperature,
                    )
                ):
                    best = choice
                    best_clip_count = (
                        target_clip_count
                    )
                    best_mean_loss = (
                        posterior_mean_loss
                    )

    if best is None:
        raise PACBayesCertificateError(
            "Certificate search produced no result."
        )
    return best, best_clip_count, float(best_mean_loss)
