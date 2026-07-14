from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from pacbayes_tsk.evaluation.pac_bayes_certificate import (
    PACBayesCertificateError,
    gaussian_kl_diag,
)


class TemporalPACBayesError(ValueError):
    """Raised when a temporal PAC-Bayes calculation is not valid."""


@dataclass(frozen=True)
class BetaCertificateChoice:
    block_length: int
    effective_sample_size: int
    used_observations: int
    assumed_beta_upper: float
    mixing_residual: float
    allocated_delta: float
    effective_delta: float
    prior_scale_multiplier: float
    posterior_scale_ratio: float
    temperature: float
    empirical_block_gibbs_risk_upper: float
    gaussian_kl: float
    structure_kl: float
    prior_component_penalty: float
    total_kl_upper: float
    certificate: float


def _finite_vector(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise TemporalPACBayesError(f"{name} is empty.")
    if not np.all(np.isfinite(vector)):
        raise TemporalPACBayesError(
            f"{name} contains non-finite values."
        )
    return vector


def pointwise_gibbs_loss_upper(
    design: np.ndarray,
    target: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    *,
    clip_bound: float,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Return an upper bound on expected clipped squared loss per observation.

    The target is explicitly clipped to the prior-fixed interval. Projection
    of the random forecast cannot increase squared distance to a target inside
    that interval, so the unprojected Gaussian second moment is a valid upper
    bound for the projected-forecast loss.
    """
    matrix = np.asarray(design, dtype=np.float64)
    y = _finite_vector(target, name="target")
    mean = _finite_vector(
        posterior_mean,
        name="posterior_mean",
    )
    std = _finite_vector(
        posterior_std,
        name="posterior_std",
    )

    if matrix.ndim != 2:
        raise TemporalPACBayesError(
            "Design must be a two-dimensional matrix."
        )
    if matrix.shape != (len(y), len(mean)):
        raise TemporalPACBayesError(
            "Design dimensions do not match target and posterior mean."
        )
    if len(std) != len(mean) or np.any(std <= 0.0):
        raise TemporalPACBayesError(
            "Posterior standard deviations are invalid."
        )
    if clip_bound <= 0.0:
        raise TemporalPACBayesError(
            "clip_bound must be positive."
        )

    bound = float(clip_bound)
    clipped_target = np.clip(y, -bound, bound)
    target_clip_count = int(np.sum(np.abs(y) > bound))

    prediction_mean = matrix @ mean
    prediction_variance = np.square(matrix) @ np.square(std)
    second_moment = (
        np.square(clipped_target - prediction_mean)
        + prediction_variance
    )
    pointwise_upper = np.minimum(
        1.0,
        second_moment / (4.0 * bound * bound),
    )
    clipped_mean_prediction = np.clip(
        prediction_mean,
        -bound,
        bound,
    )
    posterior_mean_loss = np.square(
        clipped_target - clipped_mean_prediction
    ) / (4.0 * bound * bound)

    if not np.all(np.isfinite(pointwise_upper)):
        raise TemporalPACBayesError(
            "Pointwise Gibbs loss upper bounds are non-finite."
        )
    return (
        pointwise_upper.astype(np.float64),
        target_clip_count,
        posterior_mean_loss.astype(np.float64),
    )


def odd_block_empirical_risk(
    pointwise_loss_upper: np.ndarray,
    *,
    block_length: int,
) -> tuple[float, int, int]:
    """Average losses in retained odd blocks from n=2*mu*a observations."""
    loss = _finite_vector(
        pointwise_loss_upper,
        name="pointwise_loss_upper",
    )
    a = int(block_length)
    if a <= 0:
        raise TemporalPACBayesError(
            "block_length must be positive."
        )

    mu = len(loss) // (2 * a)
    if mu < 2:
        raise TemporalPACBayesError(
            "At least two retained blocks are required."
        )
    used = 2 * mu * a
    retained_means = []
    for block_index in range(mu):
        start = 2 * block_index * a
        stop = start + a
        retained_means.append(
            float(np.mean(loss[start:stop]))
        )
    return float(np.mean(retained_means)), int(mu), int(used)


def geometric_beta_upper(
    block_length: int,
    *,
    tau: float,
    multiplier: float = 1.0,
) -> float:
    """Assumption-only geometric envelope beta(a) <= c exp(-a/tau)."""
    a = int(block_length)
    if a <= 0:
        raise TemporalPACBayesError(
            "block_length must be positive."
        )
    if tau <= 0.0:
        raise TemporalPACBayesError(
            "tau must be positive."
        )
    if multiplier <= 0.0:
        raise TemporalPACBayesError(
            "multiplier must be positive."
        )
    return float(
        min(
            1.0,
            float(multiplier)
            * np.exp(-a / float(tau)),
        )
    )


def beta_effective_delta(
    *,
    allocated_delta: float,
    effective_sample_size: int,
    beta_upper: float,
) -> tuple[float, float]:
    if not 0.0 < allocated_delta < 1.0:
        raise TemporalPACBayesError(
            "allocated_delta must be in (0, 1)."
        )
    mu = int(effective_sample_size)
    if mu < 2:
        raise TemporalPACBayesError(
            "effective_sample_size must be at least two."
        )
    if not 0.0 <= beta_upper <= 1.0:
        raise TemporalPACBayesError(
            "beta_upper must be in [0, 1]."
        )
    residual = 2.0 * (mu - 1) * float(beta_upper)
    return float(allocated_delta - residual), float(residual)


def beta_mixing_certificate_value(
    *,
    empirical_block_risk_upper: float,
    total_kl_upper: float,
    temperature: float,
    effective_sample_size: int,
    effective_delta: float,
    temperature_mass: float,
) -> float:
    if not 0.0 <= empirical_block_risk_upper <= 1.0:
        raise TemporalPACBayesError(
            "empirical_block_risk_upper must be in [0, 1]."
        )
    if total_kl_upper < 0.0:
        raise TemporalPACBayesError(
            "total_kl_upper cannot be negative."
        )
    if temperature <= 0.0:
        raise TemporalPACBayesError(
            "temperature must be positive."
        )
    if effective_sample_size < 2:
        raise TemporalPACBayesError(
            "effective_sample_size must be at least two."
        )
    if effective_delta <= 0.0:
        raise TemporalPACBayesError(
            "effective_delta must be positive."
        )
    if not 0.0 < temperature_mass <= 1.0:
        raise TemporalPACBayesError(
            "temperature_mass must be in (0, 1]."
        )

    complexity = (
        float(total_kl_upper)
        + np.log(
            1.0
            / (
                float(effective_delta)
                * float(temperature_mass)
            )
        )
    ) / (
        float(temperature)
        * int(effective_sample_size)
    )
    value = (
        float(empirical_block_risk_upper)
        + float(temperature) / 8.0
        + float(complexity)
    )
    return float(min(1.0, value))


def fixed_structure_code_penalty(
    *,
    selected_rule_count: int,
    lag_count: int,
    radius_count: int,
    alpha_count: int,
    max_rules: int,
    eta_rule: float,
) -> float:
    """Data-independent code over frozen grids and K in {1,...,max_rules}."""
    k = int(selected_rule_count)
    if not 1 <= k <= int(max_rules):
        raise TemporalPACBayesError(
            "selected_rule_count is outside the frozen rule support."
        )
    for name, value in {
        "lag_count": lag_count,
        "radius_count": radius_count,
        "alpha_count": alpha_count,
        "max_rules": max_rules,
    }.items():
        if int(value) <= 0:
            raise TemporalPACBayesError(
                f"{name} must be positive."
            )
    if eta_rule < 0.0:
        raise TemporalPACBayesError(
            "eta_rule cannot be negative."
        )

    rule_values = np.arange(1, int(max_rules) + 1)
    log_rule_normalizer = float(
        np.log(
            np.sum(
                np.exp(
                    -float(eta_rule)
                    * rule_values
                )
            )
        )
    )
    negative_log_rule_mass = (
        float(eta_rule) * k
        + log_rule_normalizer
    )
    grid_penalty = float(
        np.log(
            int(lag_count)
            * int(radius_count)
            * int(alpha_count)
        )
    )
    return grid_penalty + negative_log_rule_mass


def beta_verified_allowed(
    *,
    stationarity_status: str,
    frozen_feature_map_status: str,
    envelope_verification_status: str,
    source_reference: str,
) -> bool:
    return bool(
        stationarity_status == "verified_stationary"
        and frozen_feature_map_status
        == "verified_independent_or_valid_gap"
        and envelope_verification_status
        == "verified_external_upper_bound"
        and bool(str(source_reference).strip())
    )


def search_beta_profile(
    *,
    design_cert: np.ndarray,
    target_cert: np.ndarray,
    posterior_mean: np.ndarray,
    clip_bound: float,
    structure_kl: float,
    prior_scale_multipliers: Iterable[float],
    posterior_scale_ratios: Iterable[float],
    temperatures: Iterable[float],
    block_lengths: Iterable[int],
    delta_series_route: float,
    beta_tau: float,
    beta_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, BetaCertificateChoice | None, int, float]:
    """Search a fixed beta-envelope profile over a predeclared block grid."""
    matrix = np.asarray(design_cert, dtype=np.float64)
    target = _finite_vector(target_cert, name="target_cert")
    posterior = _finite_vector(
        posterior_mean,
        name="posterior_mean",
    )
    if matrix.shape != (len(target), len(posterior)):
        raise TemporalPACBayesError(
            "Certification design dimensions do not match."
        )

    prior_scales = [float(x) for x in prior_scale_multipliers]
    posterior_ratios = [
        float(x) for x in posterior_scale_ratios
    ]
    lambda_grid = [float(x) for x in temperatures]
    blocks = [int(x) for x in block_lengths]

    if not prior_scales or not posterior_ratios or not lambda_grid:
        raise TemporalPACBayesError(
            "Prior, posterior, and temperature grids must be non-empty."
        )
    if not blocks:
        raise TemporalPACBayesError(
            "Block grid must be non-empty."
        )

    prior_component_penalty = float(
        np.log(len(prior_scales))
    )
    block_mass = 1.0 / len(blocks)
    temperature_mass = 1.0 / len(lambda_grid)
    allocated_delta = (
        float(delta_series_route)
        * block_mass
    )

    rows: list[dict] = []
    best: BetaCertificateChoice | None = None
    best_clip_count = 0
    best_mean_loss = np.nan

    for block_length in blocks:
        try:
            _, mu, used = odd_block_empirical_risk(
                np.zeros(len(target)),
                block_length=block_length,
            )
        except TemporalPACBayesError:
            rows.append(
                {
                    "block_length": block_length,
                    "status": "INSUFFICIENT_BLOCKS",
                    "effective_sample_size": 0,
                    "used_observations": 0,
                    "assumed_beta_upper": np.nan,
                    "mixing_residual": np.nan,
                    "allocated_delta": allocated_delta,
                    "effective_delta": np.nan,
                    "certificate": np.nan,
                }
            )
            continue

        beta_upper = geometric_beta_upper(
            block_length,
            tau=beta_tau,
            multiplier=beta_multiplier,
        )
        effective_delta, residual = beta_effective_delta(
            allocated_delta=allocated_delta,
            effective_sample_size=mu,
            beta_upper=beta_upper,
        )
        if effective_delta <= 0.0:
            rows.append(
                {
                    "block_length": block_length,
                    "status": "INFEASIBLE_CONFIDENCE",
                    "effective_sample_size": mu,
                    "used_observations": used,
                    "assumed_beta_upper": beta_upper,
                    "mixing_residual": residual,
                    "allocated_delta": allocated_delta,
                    "effective_delta": effective_delta,
                    "certificate": np.nan,
                }
            )
            continue

        block_best: BetaCertificateChoice | None = None
        block_clip_count = 0
        block_mean_loss = np.nan

        for prior_multiplier in prior_scales:
            prior_std = np.full(
                len(posterior),
                prior_multiplier,
                dtype=np.float64,
            )
            prior_mean = np.zeros(
                len(posterior),
                dtype=np.float64,
            )
            for posterior_ratio in posterior_ratios:
                posterior_std = (
                    prior_std
                    * posterior_ratio
                )
                gaussian_kl = gaussian_kl_diag(
                    posterior,
                    posterior_std,
                    prior_mean,
                    prior_std,
                )
                pointwise, clip_count, mean_loss = (
                    pointwise_gibbs_loss_upper(
                        matrix,
                        target,
                        posterior,
                        posterior_std,
                        clip_bound=clip_bound,
                    )
                )
                block_risk, _, _ = odd_block_empirical_risk(
                    pointwise,
                    block_length=block_length,
                )
                total_kl = (
                    float(gaussian_kl)
                    + float(structure_kl)
                    + prior_component_penalty
                )

                for temperature in lambda_grid:
                    certificate = beta_mixing_certificate_value(
                        empirical_block_risk_upper=block_risk,
                        total_kl_upper=total_kl,
                        temperature=temperature,
                        effective_sample_size=mu,
                        effective_delta=effective_delta,
                        temperature_mass=temperature_mass,
                    )
                    choice = BetaCertificateChoice(
                        block_length=block_length,
                        effective_sample_size=mu,
                        used_observations=used,
                        assumed_beta_upper=beta_upper,
                        mixing_residual=residual,
                        allocated_delta=allocated_delta,
                        effective_delta=effective_delta,
                        prior_scale_multiplier=prior_multiplier,
                        posterior_scale_ratio=posterior_ratio,
                        temperature=temperature,
                        empirical_block_gibbs_risk_upper=block_risk,
                        gaussian_kl=gaussian_kl,
                        structure_kl=float(structure_kl),
                        prior_component_penalty=prior_component_penalty,
                        total_kl_upper=total_kl,
                        certificate=certificate,
                    )
                    if (
                        block_best is None
                        or (
                            choice.certificate,
                            choice.total_kl_upper,
                            choice.empirical_block_gibbs_risk_upper,
                            choice.block_length,
                            choice.temperature,
                        )
                        < (
                            block_best.certificate,
                            block_best.total_kl_upper,
                            block_best.empirical_block_gibbs_risk_upper,
                            block_best.block_length,
                            block_best.temperature,
                        )
                    ):
                        block_best = choice
                        block_clip_count = clip_count
                        block_mean_loss = float(
                            np.mean(mean_loss[:used])
                        )

        if block_best is None:
            rows.append(
                {
                    "block_length": block_length,
                    "status": "NO_VALID_CHOICE",
                    "effective_sample_size": mu,
                    "used_observations": used,
                    "assumed_beta_upper": beta_upper,
                    "mixing_residual": residual,
                    "allocated_delta": allocated_delta,
                    "effective_delta": effective_delta,
                    "certificate": np.nan,
                }
            )
            continue

        rows.append(
            {
                "block_length": block_best.block_length,
                "status": "PASS",
                "effective_sample_size": block_best.effective_sample_size,
                "used_observations": block_best.used_observations,
                "assumed_beta_upper": block_best.assumed_beta_upper,
                "mixing_residual": block_best.mixing_residual,
                "allocated_delta": block_best.allocated_delta,
                "effective_delta": block_best.effective_delta,
                "prior_scale_multiplier": (
                    block_best.prior_scale_multiplier
                ),
                "posterior_scale_ratio": (
                    block_best.posterior_scale_ratio
                ),
                "temperature": block_best.temperature,
                "empirical_block_gibbs_risk_upper": (
                    block_best.empirical_block_gibbs_risk_upper
                ),
                "gaussian_kl": block_best.gaussian_kl,
                "structure_kl": block_best.structure_kl,
                "prior_component_penalty": (
                    block_best.prior_component_penalty
                ),
                "total_kl_upper": block_best.total_kl_upper,
                "certificate": block_best.certificate,
                "certificate_vacuous": bool(
                    block_best.certificate
                    >= 1.0 - 1.0e-12
                ),
                "target_clip_count": block_clip_count,
                "posterior_mean_block_loss": block_mean_loss,
            }
        )

        if (
            best is None
            or (
                block_best.certificate,
                block_best.total_kl_upper,
                block_best.empirical_block_gibbs_risk_upper,
                block_best.block_length,
            )
            < (
                best.certificate,
                best.total_kl_upper,
                best.empirical_block_gibbs_risk_upper,
                best.block_length,
            )
        ):
            best = block_best
            best_clip_count = block_clip_count
            best_mean_loss = block_mean_loss

    return (
        pd.DataFrame(rows),
        best,
        best_clip_count,
        float(best_mean_loss),
    )
