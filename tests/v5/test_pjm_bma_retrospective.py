from pathlib import Path

import numpy as np

from pacbayes_tsk.experiments.retrospective_bma_v5 import (
    POSTHOC_STATUS,
    evaluate_pjm_bma_retrospective,
)


ROOT = Path(__file__).resolve().parents[2]


def test_pjm_bma_retrospective_reproduces_frozen_forecasts() -> None:
    result = evaluate_pjm_bma_retrospective(
        case_data_path=ROOT / "data/processed/pjm_daily.csv",
        config_path=ROOT / "configs/v3/pjm_confirmatory_case_v3.yaml",
        frozen_results_dir=ROOT / "results/confirmatory/pjm_case_v3_4",
    )
    summary = result.summary

    assert summary["region"].tolist() == [
        "aep",
        "comed",
        "dayton",
        "pjme",
    ]
    assert summary["status"].eq(POSTHOC_STATUS).all()
    assert not summary["confirmatory_decision_modified"].any()
    assert summary["max_point_reproduction_error"].max() <= 1e-8
    assert summary["max_target_reproduction_error"].max() <= 1e-8
    assert np.isfinite(summary.select_dtypes(include=[float, int])).all().all()
    assert np.abs(summary["bma_rmse_change_vs_point"]).max() < 0.01
    assert np.abs(summary["bma_score_change_vs_point"]).max() < 0.01

    assert set(result.predictions) == {
        "aep",
        "comed",
        "dayton",
        "pjme",
    }
    for frame in result.predictions.values():
        assert "exact_clipped_bma_forecast" in frame
        assert "reconstructed_point_forecast" in frame
        assert frame["analysis_status"].eq(POSTHOC_STATUS).all()
