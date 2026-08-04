from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pacbayes_tsk.evaluation.bma_gate_v5 import (
    BMAGateThresholds,
)
import pacbayes_tsk.experiments.neso_prospective_v6 as neso_v6
from pacbayes_tsk.experiments.neso_prospective_v6 import (
    NESOV6Settings,
    ProspectiveNESOV6Error,
    _verify_execution_snapshot,
    evaluate_test_phase_v6,
    fixed_count_labels,
    run_pretest_selection_v6,
)


def _tiny_settings(
    *,
    certificate_threshold: float = 1e-12,
) -> NESOV6Settings:
    settings = NESOV6Settings(
        split_counts={
            "prior": 12,
            "bound": 14,
            "validation": 8,
            "test": 10,
        },
        lags=(1,),
        ridge_alphas=(0.1,),
        rule_count=3,
        max_active_rules=3,
        refinement_iterations=1,
        eta_rule=math.log(2.0),
        minimum_subset_size=1,
        maximum_subset_size=3,
        prior_scales=(1.0,),
        posterior_ratios=(0.4,),
        temperatures=(0.5,),
        delta_total=0.05,
        clip_multiplier=1.25,
        clip_minimum_scaled=1.0,
        seasonal_lag=7,
        gate_thresholds=BMAGateThresholds(
            certificate_threshold=certificate_threshold,
            clipping_rate_maximum=0.99,
            minimum_rmse_improvement=0.0,
            minimum_asymmetric_score_improvement=0.0,
            validation_blocks=4,
            minimum_nonnegative_blocks=1,
            maximum_worst_block_degradation=0.99,
            underforecast_weight=2.0,
            overforecast_weight=1.0,
        ),
    )
    settings.validate()
    return settings


def _series(settings: NESOV6Settings) -> tuple[np.ndarray, pd.DatetimeIndex]:
    index = np.arange(settings.expected_rows, dtype=np.float64)
    values = (
        25_000.0
        + 20.0 * index
        + 500.0 * np.sin(2.0 * np.pi * index / 7.0)
    )
    dates = pd.date_range(
        "2025-01-01",
        periods=settings.expected_rows,
        freq="D",
    )
    return values, dates


def test_fixed_count_labels_are_chronological_and_exact() -> None:
    settings = _tiny_settings()
    labels = fixed_count_labels(settings.split_counts)
    assert len(labels) == settings.expected_rows
    assert list(pd.Series(labels).value_counts(sort=False)) == [
        12,
        14,
        8,
        10,
    ]
    coded = np.array(
        [
            ("prior", "bound", "validation", "test").index(role)
            for role in labels
        ]
    )
    assert np.all(np.diff(coded) >= 0)


def test_pretest_search_enumerates_full_support_without_test() -> None:
    settings = _tiny_settings()
    values, dates = _series(settings)
    pretest = run_pretest_selection_v6(
        values_prefix_original=values[: settings.pretest_rows],
        timestamps_prefix=dates[: settings.pretest_rows],
        settings=settings,
    )

    assert settings.subset_count == 2**3 - 1
    assert len(pretest.candidates) == 7
    assert pretest.candidates["selection_uses_test"].eq(False).all()
    assert pretest.candidates["certificate_uses_test"].eq(False).all()
    assert int(pretest.candidates["globally_selected"].sum()) == 1
    assert pretest.decision[
        "decision_made_before_test_opening"
    ] is True
    assert pretest.decision["candidate_count"] == 7


def test_pretest_rejects_any_input_beyond_frozen_prefix() -> None:
    settings = _tiny_settings()
    values, dates = _series(settings)
    with pytest.raises(
        ProspectiveNESOV6Error,
        match="prefix count",
    ):
        run_pretest_selection_v6(
            values_prefix_original=values[
                : settings.pretest_rows + 1
            ],
            timestamps_prefix=dates[
                : settings.pretest_rows + 1
            ],
            settings=settings,
        )


def test_test_phase_uses_frozen_posterior_and_fallback_after_abstention() -> None:
    settings = _tiny_settings(certificate_threshold=1e-12)
    values, dates = _series(settings)
    pretest = run_pretest_selection_v6(
        values_prefix_original=values[: settings.pretest_rows],
        timestamps_prefix=dates[: settings.pretest_rows],
        settings=settings,
    )
    assert pretest.context.gate_passed is False
    assert pretest.context.deployment == "seasonal_naive_7d"

    posterior_before = (
        pretest.context.selected_candidate.posterior_mean.copy()
    )
    metrics, predictions = evaluate_test_phase_v6(
        values_full_original=values,
        timestamps_full=dates,
        settings=settings,
        context=pretest.context,
    )
    np.testing.assert_array_equal(
        posterior_before,
        pretest.context.selected_candidate.posterior_mean,
    )
    np.testing.assert_allclose(
        predictions["deployed_forecast"],
        predictions["seasonal_naive_7d"],
    )
    assert metrics["posterior_refit_after_selection"] is False
    assert metrics["test_observation_count"] == 10
    assert metrics["deployment"] == "seasonal_naive_7d"


def test_base_structure_charge_is_uniform_lag_alpha_mass() -> None:
    settings = _tiny_settings()
    assert settings.base_structure_kl == pytest.approx(0.0)

    expanded = NESOV6Settings(
        **{
            **settings.__dict__,
            "lags": (1, 2),
            "ridge_alphas": (0.1, 1.0, 10.0),
        }
    )
    expanded.validate()
    assert expanded.base_structure_kl == pytest.approx(
        math.log(2) + math.log(3)
    )


def test_execution_snapshot_detects_hash_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    tracked = root / "tracked.txt"
    tracked.write_text("frozen", encoding="utf-8")
    digest = hashlib.sha256(tracked.read_bytes()).hexdigest()
    snapshot = artifacts / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "status": "FROZEN_NOT_RUN",
                "tracked_files": [
                    {
                        "path": "tracked.txt",
                        "sha256": digest,
                        "size_bytes": tracked.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _verify_execution_snapshot(snapshot)
    tracked.write_text("changed", encoding="utf-8")
    with pytest.raises(
        ProspectiveNESOV6Error,
        match="snapshot mismatch",
    ):
        _verify_execution_snapshot(snapshot)

def test_full_processed_file_is_touched_only_after_pretest_decision() -> None:
    pretest_verifier = inspect.getsource(neso_v6._verify_data_chain)
    runner = inspect.getsource(neso_v6.run_prospective_neso_v6)

    assert "sha256_file(processed_path)" not in pretest_verifier
    decision_position = runner.index(
        "_atomic_json(decision_path, pretest.decision)"
    )
    deferred_hash_position = runner.index(
        "_verify_processed_data_hash_after_decision"
    )
    full_load_position = runner.index(
        "full = load_neso_daily(processed_path)"
    )
    assert decision_position < deferred_hash_position < full_load_position

def test_amended_run_signature_requires_parent_snapshot() -> None:
    signature = inspect.signature(neso_v6.run_prospective_neso_v6)
    assert "execution_snapshot_path" in signature.parameters
    assert "parent_execution_snapshot_path" in signature.parameters


def test_parent_snapshot_hash_lookup_is_exact() -> None:
    snapshot = {
        "tracked_files": [
            {
                "path": "configs/v6/neso_demand_2025_v6.yaml",
                "sha256": "abc",
            }
        ]
    }
    assert neso_v6._snapshot_tracked_hash(
        snapshot,
        "configs/v6/neso_demand_2025_v6.yaml",
    ) == "abc"
    with pytest.raises(
        neso_v6.ProspectiveNESOV6Error,
        match="exactly one",
    ):
        neso_v6._snapshot_tracked_hash(
            snapshot,
            "protocol/EXPERIMENT_PROTOCOL_V6.md",
        )

