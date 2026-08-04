from pathlib import Path

import numpy as np

from pacbayes_tsk.experiments.retrospective_bma_v5 import (
    POSTHOC_STATUS,
)
from pacbayes_tsk.experiments.retrospective_pjm_selected_bma_gate_v5 import (
    evaluate_pjm_selected_bma_gate_retrospective,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_pjm_selections_pass_exact_bma_gate_retrospectively() -> None:
    result = evaluate_pjm_selected_bma_gate_retrospective(
        case_data_path=ROOT / "data/processed/pjm_daily.csv",
        config_path=(
            ROOT / "configs/v3/pjm_confirmatory_case_v3.yaml"
        ),
        frozen_results_dir=(
            ROOT / "results/confirmatory/pjm_case_v3_4"
        ),
    )
    summary = result.summary

    assert summary["region"].tolist() == [
        "aep",
        "comed",
        "dayton",
        "pjme",
    ]
    assert summary["status"].eq(POSTHOC_STATUS).all()
    assert summary["bma_gate_applicable"].all()
    assert summary["gate_passed"].all()
    assert summary["failed_conditions"].map(len).eq(0).all()
    assert summary["exact_certificate_untruncated"].lt(0.10).all()
    assert summary[
        "validation_bma_rmse_nonnegative_blocks"
    ].eq(4).all()
    assert summary[
        "validation_bma_score_nonnegative_blocks"
    ].eq(4).all()
    assert summary[
        "validation_bma_worst_block_rmse_improvement"
    ].gt(0.0).all()
    assert summary[
        "validation_bma_worst_block_score_improvement"
    ].gt(0.0).all()

    assert summary["test_rmse_improvement_vs_fallback"].gt(0.45).all()
    assert summary["test_score_improvement_vs_fallback"].gt(0.45).all()
    assert summary[
        "max_old_certificate_reproduction_error"
    ].max() <= 1e-8
    assert summary["max_target_reproduction_error"].max() <= 1e-8
    assert summary["max_seasonal_reproduction_error"].max() <= 1e-8

    assert not summary["selection_uses_test"].any()
    assert summary["interpretation_uses_test_posthoc"].all()
    assert not summary["confirmatory_decision_modified"].any()

    assert set(result.predictions) == {
        "aep",
        "comed",
        "dayton",
        "pjme",
    }
    for frame in result.predictions.values():
        assert np.isfinite(
            frame["exact_clipped_bma_forecast"]
        ).all()
        assert frame["analysis_status"].eq(POSTHOC_STATUS).all()
