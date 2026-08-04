from __future__ import annotations

import math

import numpy as np
import pytest

from pacbayes_tsk.models.sparse_tsk import SparseTSKAntecedent
from pacbayes_tsk.optimization.rule_subset_v6 import (
    RuleSubsetPriorV6,
    RuleSubsetV6Error,
    canonical_rule_subset,
    subset_antecedent,
)


def _antecedent() -> SparseTSKAntecedent:
    antecedent = SparseTSKAntecedent(
        feature_mean=np.array([0.5, -0.25]),
        feature_scale=np.array([1.5, 0.75]),
        centers=np.array(
            [
                [-1.0, -0.5],
                [0.0, 0.25],
                [0.75, 1.0],
                [1.5, -1.25],
            ]
        ),
        spreads=np.array(
            [
                [0.5, 0.75],
                [0.8, 0.6],
                [0.7, 0.9],
                [1.0, 0.55],
            ]
        ),
        prior_support=np.array([11, 7, 5, 3]),
        radius=1.25,
        max_active_rules=3,
        radius_cap_reached=False,
    )
    antecedent.validate()
    return antecedent


def test_subset_prior_is_normalized_over_every_declared_identity() -> None:
    prior = RuleSubsetPriorV6(
        total_rule_count=5,
        eta_rule=math.log(2.0),
    )
    mass = sum(
        math.exp(-prior.negative_log_mass(subset))
        for subset in prior.iter_support()
    )
    assert np.isclose(mass, 1.0, atol=1e-14, rtol=0.0)


def test_subset_prior_charges_identity_and_size_explicitly() -> None:
    prior = RuleSubsetPriorV6(
        total_rule_count=6,
        eta_rule=0.4,
    )
    same_size_a = prior.negative_log_mass((0, 1))
    same_size_b = prior.negative_log_mass((2, 5))
    size_three = prior.negative_log_mass((0, 1, 2))

    assert same_size_a == pytest.approx(same_size_b)
    expected_difference = (
        0.4
        + math.log(math.comb(6, 3))
        - math.log(math.comb(6, 2))
    )
    assert (
        size_three - same_size_a
        == pytest.approx(expected_difference)
    )


def test_truncated_subset_prior_remains_normalized() -> None:
    prior = RuleSubsetPriorV6(
        total_rule_count=6,
        eta_rule=0.7,
        minimum_size=2,
        maximum_size=4,
    )
    support = tuple(prior.iter_support())
    assert all(2 <= len(subset) <= 4 for subset in support)
    assert sum(
        math.exp(-prior.negative_log_mass(subset))
        for subset in support
    ) == pytest.approx(1.0)


def test_subset_antecedent_preserves_selected_blocks_and_renormalizes() -> None:
    original = _antecedent()
    reduced = subset_antecedent(original, (3, 1))

    assert reduced.rule_count == 2
    assert reduced.max_active_rules == 2
    np.testing.assert_array_equal(
        reduced.centers,
        original.centers[[1, 3]],
    )
    np.testing.assert_array_equal(
        reduced.spreads,
        original.spreads[[1, 3]],
    )
    np.testing.assert_array_equal(
        reduced.prior_support,
        original.prior_support[[1, 3]],
    )

    X = np.array(
        [
            [0.0, -0.25],
            [1.5, 0.5],
            [-1.0, -0.75],
        ]
    )
    firing = reduced.firing_strengths(X)
    design = reduced.design_matrix(X)
    assert firing.shape == (3, 2)
    assert design.shape == (3, 2 * (2 + 1))
    np.testing.assert_allclose(
        firing.sum(axis=1),
        1.0,
        atol=1e-14,
    )


def test_subset_construction_does_not_mutate_dictionary() -> None:
    original = _antecedent()
    centers_before = original.centers.copy()
    support_before = original.prior_support.copy()

    reduced = subset_antecedent(original, (0, 2))
    reduced.centers[0, 0] = 999.0
    reduced.prior_support[0] = 999

    np.testing.assert_array_equal(
        original.centers,
        centers_before,
    )
    np.testing.assert_array_equal(
        original.prior_support,
        support_before,
    )


@pytest.mark.parametrize(
    "indices",
    [
        (),
        (0, 0),
        (-1,),
        (4,),
        (0.5,),
        (True,),
    ],
)
def test_invalid_rule_subsets_are_rejected(
    indices: tuple[object, ...],
) -> None:
    with pytest.raises(RuleSubsetV6Error):
        canonical_rule_subset(
            indices,
            total_rule_count=4,
        )


def test_subset_outside_truncated_prior_support_is_rejected() -> None:
    prior = RuleSubsetPriorV6(
        total_rule_count=4,
        minimum_size=2,
        maximum_size=3,
    )
    with pytest.raises(RuleSubsetV6Error):
        prior.negative_log_mass((0,))
    with pytest.raises(RuleSubsetV6Error):
        prior.negative_log_mass((0, 1, 2, 3))
