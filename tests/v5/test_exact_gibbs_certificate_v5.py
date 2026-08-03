from __future__ import annotations

import numpy as np
import pytest

from pacbayes_tsk.evaluation.clipped_gaussian_bma import (
    expected_normalized_clipped_squared_loss,
)
from pacbayes_tsk.evaluation.exact_gibbs_certificate_v5 import (
    ExactGibbsCertificateError,
    search_exact_clipped_gibbs_certificate,
)
from pacbayes_tsk.evaluation.pac_bayes_certificate import gaussian_kl_diag
from pacbayes_tsk.pac_bayes.certificates_v3 import (
    martingale_certificate,
    pointwise_gibbs_upper,
)


DESIGN = np.array(
    [
        [1.0, -1.5],
        [1.0, -0.5],
        [1.0, 0.5],
        [1.0, 1.5],
        [1.0, 2.0],
    ]
)
TARGET = np.array([-1.4, -0.2, 0.7, 1.3, 1.8])
POSTERIOR_MEAN = np.array([0.1, 0.75])
PRIOR_MEAN = np.array([0.0, 0.4])
BASE_STD = np.array([0.35, 0.55])
PRIOR_SCALES = (0.5, 1.0, 2.0)
POSTERIOR_RATIOS = (0.1, 0.4, 1.0)
TEMPERATURES = (0.125, 0.25, 0.5, 1.0)


def test_exact_pointwise_risk_is_below_conservative_upper() -> None:
    posterior_std = BASE_STD * 1.5 * 0.4
    exact = expected_normalized_clipped_squared_loss(
        DESIGN,
        TARGET,
        POSTERIOR_MEAN,
        posterior_std,
        clip_bound=1.25,
    )
    conservative = pointwise_gibbs_upper(
        DESIGN,
        TARGET,
        POSTERIOR_MEAN,
        posterior_std,
        clip_bound=1.25,
    )

    assert np.all(exact <= conservative + 1e-13)
    assert np.any(exact < conservative - 1e-6)


def test_search_matches_independent_grid_enumeration() -> None:
    structure_kl = 0.7
    delta = 0.05
    bound = 1.25

    selected = search_exact_clipped_gibbs_certificate(
        design_cert=DESIGN,
        target_cert=TARGET,
        posterior_mean=POSTERIOR_MEAN,
        prior_mean=PRIOR_MEAN,
        base_prior_std=BASE_STD,
        clip_bound=bound,
        structure_kl=structure_kl,
        prior_scale_multipliers=PRIOR_SCALES,
        posterior_scale_ratios=POSTERIOR_RATIOS,
        temperatures=TEMPERATURES,
        delta=delta,
    )

    rows = []
    prior_penalty = float(np.log(len(PRIOR_SCALES)))
    for prior_scale in PRIOR_SCALES:
        prior_std = BASE_STD * prior_scale
        for posterior_ratio in POSTERIOR_RATIOS:
            posterior_std = prior_std * posterior_ratio
            exact_empirical = float(
                np.mean(
                    expected_normalized_clipped_squared_loss(
                        DESIGN,
                        TARGET,
                        POSTERIOR_MEAN,
                        posterior_std,
                        clip_bound=bound,
                    )
                )
            )
            gaussian_kl = gaussian_kl_diag(
                POSTERIOR_MEAN,
                posterior_std,
                PRIOR_MEAN,
                prior_std,
            )
            total_kl = gaussian_kl + structure_kl + prior_penalty
            result = martingale_certificate(
                empirical_risk=exact_empirical,
                total_kl=total_kl,
                n=len(TARGET),
                delta=delta,
                temperatures=TEMPERATURES,
            )
            rows.append(
                (
                    float(result.untruncated),
                    total_kl,
                    exact_empirical,
                    posterior_ratio,
                    prior_scale,
                    float(result.temperature),
                )
            )

    expected = min(rows)
    actual = (
        selected.certificate_untruncated,
        selected.total_kl,
        selected.exact_empirical_gibbs_risk,
        selected.posterior_scale_ratio,
        selected.prior_scale_multiplier,
        selected.temperature,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0.0)


def test_search_reports_exact_and_conservative_risks() -> None:
    selected = search_exact_clipped_gibbs_certificate(
        design_cert=DESIGN,
        target_cert=TARGET,
        posterior_mean=POSTERIOR_MEAN,
        prior_mean=PRIOR_MEAN,
        base_prior_std=BASE_STD,
        clip_bound=1.25,
        structure_kl=0.7,
        prior_scale_multipliers=PRIOR_SCALES,
        posterior_scale_ratios=POSTERIOR_RATIOS,
        temperatures=TEMPERATURES,
        delta=0.05,
    )

    assert 0.0 <= selected.exact_empirical_gibbs_risk <= 1.0
    assert (
        selected.exact_empirical_gibbs_risk
        <= selected.conservative_empirical_gibbs_risk_upper + 1e-13
    )
    assert selected.total_kl >= selected.structure_kl
    assert selected.certificate == min(
        1.0,
        selected.certificate_untruncated,
    )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("base_prior_std", np.array([0.0, 0.5])),
        ("prior_scale_multipliers", ()),
        ("posterior_scale_ratios", (0.0,)),
        ("temperatures", (-1.0,)),
    ],
)
def test_invalid_search_inputs_are_rejected(
    argument: str,
    value: object,
) -> None:
    kwargs = {
        "design_cert": DESIGN,
        "target_cert": TARGET,
        "posterior_mean": POSTERIOR_MEAN,
        "prior_mean": PRIOR_MEAN,
        "base_prior_std": BASE_STD,
        "clip_bound": 1.25,
        "structure_kl": 0.7,
        "prior_scale_multipliers": PRIOR_SCALES,
        "posterior_scale_ratios": POSTERIOR_RATIOS,
        "temperatures": TEMPERATURES,
        "delta": 0.05,
    }
    kwargs[argument] = value

    with pytest.raises(ExactGibbsCertificateError):
        search_exact_clipped_gibbs_certificate(**kwargs)
