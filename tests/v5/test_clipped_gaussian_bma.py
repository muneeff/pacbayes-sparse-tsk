from __future__ import annotations

import numpy as np
import pytest

from pacbayes_tsk.evaluation.clipped_gaussian_bma import (
    ClippedGaussianBMAError,
    clipped_gaussian_bma,
    clipped_gaussian_mean,
    clipped_gaussian_second_moment,
    expected_normalized_clipped_squared_loss,
)


def test_zero_variance_reduces_to_deterministic_clipping():
    mean = np.array([-3.0, -0.5, 0.5, 3.0])
    result = clipped_gaussian_mean(
        mean,
        np.zeros_like(mean),
        lower=-1.0,
        upper=1.0,
    )
    np.testing.assert_allclose(
        result,
        [-1.0, -0.5, 0.5, 1.0],
    )


def test_symmetric_zero_mean_has_zero_clipped_expectation():
    std = np.array([0.1, 1.0, 10.0])
    result = clipped_gaussian_mean(
        np.zeros_like(std),
        std,
        lower=-2.0,
        upper=2.0,
    )
    np.testing.assert_allclose(result, 0.0, atol=1e-14)


def test_second_moment_matches_deterministic_case():
    mean = np.array([-3.0, -0.5, 0.5, 3.0])
    result = clipped_gaussian_second_moment(
        mean,
        np.zeros_like(mean),
        lower=-1.0,
        upper=1.0,
    )
    np.testing.assert_allclose(
        result,
        [1.0, 0.25, 0.25, 1.0],
    )


def test_exact_mean_matches_monte_carlo():
    rng = np.random.default_rng(20260804)
    mean = np.array([-1.8, -0.2, 0.9, 2.2])
    std = np.array([0.7, 1.4, 0.4, 1.1])
    exact = clipped_gaussian_mean(
        mean,
        std,
        lower=-1.5,
        upper=1.5,
    )
    samples = rng.normal(
        mean,
        std,
        size=(400_000, len(mean)),
    )
    monte_carlo = np.mean(
        np.clip(samples, -1.5, 1.5),
        axis=0,
    )
    np.testing.assert_allclose(
        exact,
        monte_carlo,
        atol=3e-3,
        rtol=0.0,
    )


def test_exact_second_moment_matches_monte_carlo():
    rng = np.random.default_rng(20260804)
    mean = np.array([-1.2, 0.3, 1.8])
    std = np.array([0.8, 1.1, 0.6])
    exact = clipped_gaussian_second_moment(
        mean,
        std,
        lower=-1.5,
        upper=1.5,
    )
    samples = rng.normal(
        mean,
        std,
        size=(400_000, len(mean)),
    )
    monte_carlo = np.mean(
        np.square(np.clip(samples, -1.5, 1.5)),
        axis=0,
    )
    np.testing.assert_allclose(
        exact,
        monte_carlo,
        atol=4e-3,
        rtol=0.0,
    )


def test_design_level_bma_matches_scalar_distribution():
    design = np.array([
        [1.0, 2.0],
        [1.0, -1.0],
    ])
    posterior_mean = np.array([0.2, 0.5])
    posterior_std = np.array([0.3, 0.4])
    result = clipped_gaussian_bma(
        design,
        posterior_mean,
        posterior_std,
        clip_bound=1.0,
    )
    predictive_mean = design @ posterior_mean
    predictive_std = np.sqrt(
        np.square(design) @ np.square(posterior_std)
    )
    expected = clipped_gaussian_mean(
        predictive_mean,
        predictive_std,
        lower=-1.0,
        upper=1.0,
    )
    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-14,
    )


def test_exact_expected_loss_is_bounded_and_matches_deterministic_limit():
    design = np.array([
        [1.0],
        [1.0],
        [1.0],
    ])
    target = np.array([-2.0, 0.25, 2.0])
    posterior_mean = np.array([0.25])
    posterior_std = np.array([0.0])
    result = expected_normalized_clipped_squared_loss(
        design,
        target,
        posterior_mean,
        posterior_std,
        clip_bound=1.0,
    )
    expected = (
        np.square(
            np.clip(target, -1.0, 1.0) - 0.25
        )
        / 4.0
    )
    np.testing.assert_allclose(result, expected)
    assert np.all((result >= 0.0) & (result <= 1.0))



def test_extreme_small_std_remains_finite():
    mean = np.array([-1e6, 0.0, 1e6])
    std = np.array([1e-300, 1e-300, 1e-300])
    clipped_mean = clipped_gaussian_mean(
        mean,
        std,
        lower=-1.0,
        upper=1.0,
    )
    clipped_second = clipped_gaussian_second_moment(
        mean,
        std,
        lower=-1.0,
        upper=1.0,
    )
    np.testing.assert_allclose(clipped_mean, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(clipped_second, [1.0, 0.0, 1.0])
    assert np.all(np.isfinite(clipped_mean))
    assert np.all(np.isfinite(clipped_second))


def test_exact_expected_loss_matches_monte_carlo():
    rng = np.random.default_rng(20260804)
    design = np.array([
        [1.0, 0.5],
        [1.0, -1.5],
        [1.0, 2.0],
    ])
    target = np.array([-1.8, 0.2, 1.7])
    posterior_mean = np.array([0.15, 0.45])
    posterior_std = np.array([0.35, 0.55])
    bound = 1.25

    exact = expected_normalized_clipped_squared_loss(
        design,
        target,
        posterior_mean,
        posterior_std,
        clip_bound=bound,
    )
    parameter_samples = rng.normal(
        posterior_mean,
        posterior_std,
        size=(400_000, len(posterior_mean)),
    )
    forecast_samples = parameter_samples @ design.T
    clipped_forecasts = np.clip(forecast_samples, -bound, bound)
    clipped_target = np.clip(target, -bound, bound)
    monte_carlo = np.mean(
        np.square(clipped_forecasts - clipped_target)
        / (4.0 * bound * bound),
        axis=0,
    )
    np.testing.assert_allclose(
        exact,
        monte_carlo,
        atol=4e-3,
        rtol=0.0,
    )


def test_bma_loss_is_bounded_by_exact_gibbs_loss():
    design = np.array([
        [1.0, 0.25],
        [1.0, -1.0],
        [1.0, 1.75],
    ])
    target = np.array([-0.8, 0.4, 1.3])
    posterior_mean = np.array([0.1, 0.6])
    posterior_std = np.array([0.45, 0.7])
    bound = 1.0

    bma = clipped_gaussian_bma(
        design,
        posterior_mean,
        posterior_std,
        clip_bound=bound,
    )
    clipped_target = np.clip(target, -bound, bound)
    bma_loss = (
        np.square(bma - clipped_target)
        / (4.0 * bound * bound)
    )
    gibbs_loss = expected_normalized_clipped_squared_loss(
        design,
        target,
        posterior_mean,
        posterior_std,
        clip_bound=bound,
    )
    assert np.all(bma_loss <= gibbs_loss + 1e-13)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ClippedGaussianBMAError):
        clipped_gaussian_mean(
            np.array([0.0]),
            np.array([-1.0]),
            lower=-1.0,
            upper=1.0,
        )
    with pytest.raises(ClippedGaussianBMAError):
        clipped_gaussian_bma(
            np.ones((2, 3)),
            np.ones(2),
            np.ones(2),
            clip_bound=1.0,
        )
