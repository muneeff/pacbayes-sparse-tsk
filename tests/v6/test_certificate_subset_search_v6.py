from __future__ import annotations

import math

import numpy as np
import pytest

from pacbayes_tsk.models.sparse_tsk import SparseTSKAntecedent
from pacbayes_tsk.optimization.certificate_subset_search_v6 import (
    CertificateSubsetSearchV6Error,
    search_certificate_aware_rule_subsets_v6,
)
from pacbayes_tsk.optimization.rule_subset_v6 import (
    RuleSubsetPriorV6,
)


def _dictionary(rule_count: int = 4) -> SparseTSKAntecedent:
    centers = np.linspace(-1.5, 1.5, rule_count).reshape(-1, 1)
    antecedent = SparseTSKAntecedent(
        feature_mean=np.array([0.0]),
        feature_scale=np.array([1.0]),
        centers=centers,
        spreads=np.full((rule_count, 1), 0.8),
        prior_support=np.arange(rule_count, 0, -1),
        radius=1.0,
        max_active_rules=rule_count,
        radius_cap_reached=False,
    )
    antecedent.validate()
    return antecedent


def _data() -> dict[str, np.ndarray]:
    x = np.linspace(-2.0, 2.0, 42).reshape(-1, 1)
    y = 0.35 + 0.8 * x[:, 0] + 0.15 * np.sin(2.0 * x[:, 0])
    return {
        "X_prior": x[:14],
        "y_prior": y[:14],
        "X_bound": x[14:30],
        "y_bound": y[14:30],
        "X_validation": x[30:],
        "y_validation": y[30:],
    }


def _search(
    *,
    strategy: str,
    rule_count: int = 4,
    maximum_size: int | None = None,
):
    return search_certificate_aware_rule_subsets_v6(
        dictionary_antecedent=_dictionary(rule_count),
        **_data(),
        ridge_alpha=0.1,
        clip_bound=2.5,
        base_structure_kl=0.7,
        subset_prior=RuleSubsetPriorV6(
            total_rule_count=rule_count,
            eta_rule=math.log(2.0),
            maximum_size=maximum_size,
        ),
        prior_scale_multipliers=(0.5, 1.0),
        posterior_scale_ratios=(0.2, 0.6),
        temperatures=(0.25, 0.5, 1.0),
        delta=0.05,
        strategy=strategy,
    )


def test_exhaustive_search_evaluates_full_nonempty_support() -> None:
    result = _search(strategy="exhaustive", rule_count=4)
    assert result.evaluated_subset_count == 2**4 - 1
    assert 1 <= result.selected.subset_size <= 4
    assert result.selected is min(
        result.evaluated_candidates,
        key=lambda row: row.selection_key,
    )


def test_backward_search_is_deterministic_and_smaller_than_exhaustive() -> None:
    first = _search(strategy="backward", rule_count=4)
    second = _search(strategy="backward", rule_count=4)

    assert first.selected.subset == second.selected.subset
    assert first.selected.selection_key == second.selected.selection_key
    assert first.evaluated_subset_count < 2**4 - 1
    assert any(
        row.subset_size == 4
        for row in first.evaluated_candidates
    )
    assert any(
        row.subset_size == 1
        for row in first.evaluated_candidates
    )


def test_structure_kl_adds_base_and_subset_identity_charge() -> None:
    result = _search(strategy="backward", rule_count=4)
    for row in result.evaluated_candidates:
        assert row.total_structure_kl == pytest.approx(
            row.base_structure_kl + row.subset_identity_kl
        )
        assert row.certificate.structure_kl == pytest.approx(
            row.total_structure_kl
        )


def test_selected_operational_forecast_is_exact_clipped_bma() -> None:
    result = _search(strategy="backward", rule_count=4)
    prediction = result.predict_bma(_data()["X_validation"])
    assert prediction.shape == _data()["y_validation"].shape
    assert np.all(np.isfinite(prediction))
    assert np.all(np.abs(prediction) <= 2.5 + 1e-14)


def test_truncated_maximum_size_is_respected() -> None:
    result = _search(
        strategy="exhaustive",
        rule_count=4,
        maximum_size=2,
    )
    assert result.evaluated_subset_count == (
        math.comb(4, 1) + math.comb(4, 2)
    )
    assert all(
        row.subset_size <= 2
        for row in result.evaluated_candidates
    )


def test_exhaustive_search_respects_evaluation_guard() -> None:
    with pytest.raises(
        CertificateSubsetSearchV6Error,
        match="max_evaluations",
    ):
        search_certificate_aware_rule_subsets_v6(
            dictionary_antecedent=_dictionary(4),
            **_data(),
            ridge_alpha=0.1,
            clip_bound=2.5,
            base_structure_kl=0.7,
            subset_prior=RuleSubsetPriorV6(4),
            prior_scale_multipliers=(1.0,),
            posterior_scale_ratios=(0.5,),
            temperatures=(0.5,),
            delta=0.05,
            strategy="exhaustive",
            max_evaluations=3,
        )


def test_dictionary_and_prior_rule_counts_must_match() -> None:
    with pytest.raises(CertificateSubsetSearchV6Error):
        search_certificate_aware_rule_subsets_v6(
            dictionary_antecedent=_dictionary(4),
            **_data(),
            ridge_alpha=0.1,
            clip_bound=2.5,
            base_structure_kl=0.7,
            subset_prior=RuleSubsetPriorV6(3),
            prior_scale_multipliers=(1.0,),
            posterior_scale_ratios=(0.5,),
            temperatures=(0.5,),
            delta=0.05,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ridge_alpha", -0.1),
        ("clip_bound", 0.0),
        ("base_structure_kl", -1.0),
        ("delta", 1.0),
    ],
)
def test_invalid_scalar_inputs_are_rejected(
    name: str,
    value: float,
) -> None:
    kwargs = dict(
        dictionary_antecedent=_dictionary(4),
        **_data(),
        ridge_alpha=0.1,
        clip_bound=2.5,
        base_structure_kl=0.7,
        subset_prior=RuleSubsetPriorV6(4),
        prior_scale_multipliers=(1.0,),
        posterior_scale_ratios=(0.5,),
        temperatures=(0.5,),
        delta=0.05,
    )
    kwargs[name] = value
    with pytest.raises(CertificateSubsetSearchV6Error):
        search_certificate_aware_rule_subsets_v6(**kwargs)
