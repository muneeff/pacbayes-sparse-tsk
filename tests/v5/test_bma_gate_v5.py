from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pacbayes_tsk.evaluation.bma_gate_v5 import (
    BMAGateError,
    BMAGateThresholds,
    asymmetric_forecasting_error_score,
    bma_block_stability,
    evaluate_bma_validation_gate,
    select_bma_candidate,
)


STRICT = BMAGateThresholds(
    certificate_threshold=0.10,
    clipping_rate_maximum=0.02,
    minimum_rmse_improvement=0.05,
    minimum_asymmetric_score_improvement=0.05,
    validation_blocks=4,
    minimum_nonnegative_blocks=3,
    maximum_worst_block_degradation=0.05,
    underforecast_weight=2.0,
    overforecast_weight=1.0,
)


def test_asymmetric_score_penalizes_underforecast_more() -> None:
    target = np.array([10.0, 10.0])
    under = asymmetric_forecasting_error_score(
        target,
        np.array([8.0, 8.0]),
        underforecast_weight=2.0,
        overforecast_weight=1.0,
    )
    over = asymmetric_forecasting_error_score(
        target,
        np.array([12.0, 12.0]),
        underforecast_weight=2.0,
        overforecast_weight=1.0,
    )
    assert under == 2.0 * over


def test_uniform_bma_improvement_passes_all_seven_conditions() -> None:
    target = np.arange(1.0, 17.0)
    seasonal = target + 2.0
    bma = target + 1.0

    result = evaluate_bma_validation_gate(
        target=target,
        bma_prediction=bma,
        seasonal_prediction=seasonal,
        certificate_untruncated=0.08,
        certification_target_clipping_rate=0.01,
        thresholds=STRICT,
    )

    assert result.gate_passed
    assert result.failed_conditions == ()
    assert result.validation_bma_rmse_nonnegative_blocks == 4
    assert result.validation_bma_score_nonnegative_blocks == 4
    assert result.validation_bma_rmse_improvement_vs_fallback == 0.5
    assert result.validation_bma_score_improvement_vs_fallback == 0.5


def test_gate_reports_each_failed_condition_without_hiding_it() -> None:
    target = np.arange(1.0, 17.0)
    seasonal = target + 1.0
    bma = target + 2.0

    result = evaluate_bma_validation_gate(
        target=target,
        bma_prediction=bma,
        seasonal_prediction=seasonal,
        certificate_untruncated=0.11,
        certification_target_clipping_rate=0.03,
        thresholds=STRICT,
    )

    assert not result.gate_passed
    assert "certificate" in result.failed_conditions
    assert "clipping" in result.failed_conditions
    assert "rmse_margin" in result.failed_conditions
    assert "asymmetric_score_margin" in result.failed_conditions
    assert "block_rmse_stability" in result.failed_conditions
    assert "block_score_stability" in result.failed_conditions
    assert "worst_block_degradation" in result.failed_conditions


def test_block_stability_detects_one_bad_chronological_block() -> None:
    target = np.arange(1.0, 17.0)
    seasonal = target + 2.0
    bma = target + 1.0
    bma[-4:] = target[-4:] + 4.0

    result = bma_block_stability(
        target,
        bma,
        seasonal,
        blocks=4,
        underforecast_weight=2.0,
        overforecast_weight=1.0,
    )

    assert result["validation_bma_rmse_nonnegative_blocks"] == 3
    assert result["validation_bma_score_nonnegative_blocks"] == 3
    assert (
        result["validation_bma_worst_block_rmse_improvement"]
        < 0.0
    )
    assert (
        result["validation_bma_worst_block_score_improvement"]
        < 0.0
    )


def test_selection_prioritizes_bma_asymmetric_score_before_rmse() -> None:
    base = {
        "validation_bma_asymmetric_score": 10.0,
        "validation_bma_rmse": 8.0,
        "exact_certificate_untruncated": 0.07,
        "consequent_dimension": 20,
        "rule_count": 1,
        "family": "ridge",
        "radius": np.nan,
        "ridge_alpha": 0.1,
        "lag": 7,
    }
    lower_score = dict(base)
    lower_score["validation_bma_asymmetric_score"] = 9.0
    lower_score["validation_bma_rmse"] = 9.0
    pool = pd.DataFrame([base, lower_score])

    selected = select_bma_candidate(pool)

    assert selected["validation_bma_asymmetric_score"] == 9.0
    assert selected["validation_bma_rmse"] == 9.0


def test_selection_uses_exact_certificate_as_third_tie_break() -> None:
    first = {
        "validation_bma_asymmetric_score": 9.0,
        "validation_bma_rmse": 8.0,
        "exact_certificate_untruncated": 0.08,
        "consequent_dimension": 20,
        "rule_count": 1,
        "family": "ridge",
        "radius": np.nan,
        "ridge_alpha": 0.1,
        "lag": 7,
    }
    second = dict(first)
    second["exact_certificate_untruncated"] = 0.06
    pool = pd.DataFrame([first, second])

    selected = select_bma_candidate(pool)

    assert selected["exact_certificate_untruncated"] == 0.06


def test_invalid_thresholds_and_misaligned_vectors_are_rejected() -> None:
    invalid = BMAGateThresholds(
        certificate_threshold=1.0,
        clipping_rate_maximum=0.02,
        minimum_rmse_improvement=0.05,
        minimum_asymmetric_score_improvement=0.05,
        validation_blocks=4,
        minimum_nonnegative_blocks=3,
        maximum_worst_block_degradation=0.05,
        underforecast_weight=2.0,
        overforecast_weight=1.0,
    )
    with pytest.raises(BMAGateError):
        invalid.validate()

    with pytest.raises(BMAGateError):
        evaluate_bma_validation_gate(
            target=np.ones(8),
            bma_prediction=np.ones(7),
            seasonal_prediction=np.ones(8),
            certificate_untruncated=0.08,
            certification_target_clipping_rate=0.01,
            thresholds=STRICT,
        )
