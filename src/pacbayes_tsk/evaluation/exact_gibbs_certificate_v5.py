from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pacbayes_tsk.evaluation.clipped_gaussian_bma import (
    expected_normalized_clipped_squared_loss,
)
from pacbayes_tsk.evaluation.pac_bayes_certificate import gaussian_kl_diag
from pacbayes_tsk.pac_bayes.certificates_v3 import (
    martingale_certificate,
    pointwise_gibbs_upper,
)


class ExactGibbsCertificateError(ValueError):
    """Raised when exact clipped-Gibbs certificate selection is invalid."""


@dataclass(frozen=True)
class ExactGibbsCertificateChoice:
    prior_scale_multiplier: float
    posterior_scale_ratio: float
    temperature: float
    exact_empirical_gibbs_risk: float
    conservative_empirical_gibbs_risk_upper: float
    gaussian_kl: float
    structure_kl: float
    prior_component_penalty: float
    total_kl: float
    certificate_untruncated: float
    certificate: float


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ExactGibbsCertificateError(f"{name} is empty.")
    if not np.all(np.isfinite(vector)):
        raise ExactGibbsCertificateError(
            f"{name} contains non-finite values."
        )
    return vector


def _finite_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ExactGibbsCertificateError(
            f"{name} must be a non-empty matrix."
        )
    if not np.all(np.isfinite(matrix)):
        raise ExactGibbsCertificateError(
            f"{name} contains non-finite values."
        )
    return matrix


def _positive_grid(
    values: Iterable[float],
    *,
    name: str,
) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid:
        raise ExactGibbsCertificateError(
            f"{name} must contain at least one value."
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in grid):
        raise ExactGibbsCertificateError(
            f"{name} values must be finite and positive."
        )
    return grid


def search_exact_clipped_gibbs_certificate(
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
) -> ExactGibbsCertificateChoice:
    """Select posterior scale using exact clipped empirical Gibbs risk.

    The posterior mean and model structure are fixed inputs. The search only
    ranges over the predeclared prior-scale, posterior-ratio, and temperature
    grids. The empirical risk is the exact posterior expectation of the
    normalized squared loss after clipping both prediction and target.
    """

    design = _finite_matrix(design_cert, name="design_cert")
    target = _finite_vector(target_cert, name="target_cert")
    q_mean = _finite_vector(posterior_mean, name="posterior_mean")
    p_mean = _finite_vector(prior_mean, name="prior_mean")
    base_std = _finite_vector(base_prior_std, name="base_prior_std")

    if design.shape != (len(target), len(q_mean)):
        raise ExactGibbsCertificateError(
            "design_cert dimensions do not match target and posterior mean."
        )
    if len(p_mean) != len(q_mean) or len(base_std) != len(q_mean):
        raise ExactGibbsCertificateError(
            "Prior and posterior vectors must have equal dimensions."
        )
    if np.any(base_std <= 0.0):
        raise ExactGibbsCertificateError(
            "base_prior_std must be strictly positive."
        )

    bound = float(clip_bound)
    structural_penalty = float(structure_kl)
    confidence = float(delta)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ExactGibbsCertificateError(
            "clip_bound must be finite and positive."
        )
    if not np.isfinite(structural_penalty) or structural_penalty < 0.0:
        raise ExactGibbsCertificateError(
            "structure_kl must be finite and nonnegative."
        )
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ExactGibbsCertificateError(
            "delta must lie strictly between zero and one."
        )

    prior_scales = _positive_grid(
        prior_scale_multipliers,
        name="prior_scale_multipliers",
    )
    posterior_ratios = _positive_grid(
        posterior_scale_ratios,
        name="posterior_scale_ratios",
    )
    temperature_grid = _positive_grid(
        temperatures,
        name="temperatures",
    )
    prior_component_penalty = float(np.log(len(prior_scales)))

    best: ExactGibbsCertificateChoice | None = None
    for prior_scale in prior_scales:
        prior_std = base_std * prior_scale
        for posterior_ratio in posterior_ratios:
            posterior_std = prior_std * posterior_ratio
            exact_pointwise = expected_normalized_clipped_squared_loss(
                design,
                target,
                q_mean,
                posterior_std,
                clip_bound=bound,
            )
            exact_empirical = float(np.mean(exact_pointwise))
            conservative_empirical = float(
                np.mean(
                    pointwise_gibbs_upper(
                        design,
                        target,
                        q_mean,
                        posterior_std,
                        clip_bound=bound,
                    )
                )
            )
            if exact_empirical > conservative_empirical + 1e-12:
                raise ExactGibbsCertificateError(
                    "Exact clipped Gibbs risk exceeds its conservative upper."
                )

            gaussian_kl = gaussian_kl_diag(
                q_mean,
                posterior_std,
                p_mean,
                prior_std,
            )
            total_kl = (
                gaussian_kl
                + structural_penalty
                + prior_component_penalty
            )
            result = martingale_certificate(
                empirical_risk=exact_empirical,
                total_kl=total_kl,
                n=len(target),
                delta=confidence,
                temperatures=temperature_grid,
            )
            choice = ExactGibbsCertificateChoice(
                prior_scale_multiplier=prior_scale,
                posterior_scale_ratio=posterior_ratio,
                temperature=float(result.temperature),
                exact_empirical_gibbs_risk=exact_empirical,
                conservative_empirical_gibbs_risk_upper=(
                    conservative_empirical
                ),
                gaussian_kl=gaussian_kl,
                structure_kl=structural_penalty,
                prior_component_penalty=prior_component_penalty,
                total_kl=total_kl,
                certificate_untruncated=float(result.untruncated),
                certificate=float(result.certificate),
            )
            if best is None or (
                choice.certificate_untruncated,
                choice.total_kl,
                choice.exact_empirical_gibbs_risk,
                choice.posterior_scale_ratio,
                choice.prior_scale_multiplier,
                choice.temperature,
            ) < (
                best.certificate_untruncated,
                best.total_kl,
                best.exact_empirical_gibbs_risk,
                best.posterior_scale_ratio,
                best.prior_scale_multiplier,
                best.temperature,
            ):
                best = choice

    if best is None:
        raise ExactGibbsCertificateError(
            "Exact certificate search produced no result."
        )
    return best
