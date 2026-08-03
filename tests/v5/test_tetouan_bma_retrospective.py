from pathlib import Path

import numpy as np

from pacbayes_tsk.experiments.retrospective_bma_v5 import POSTHOC_STATUS
from pacbayes_tsk.experiments.retrospective_tetouan_bma_v5 import (
    evaluate_tetouan_bma_retrospective,
)


ROOT = Path(__file__).resolve().parents[2]


def test_tetouan_bma_retrospective_handles_selected_and_fallback_zones() -> None:
    result = evaluate_tetouan_bma_retrospective(
        case_data_path=ROOT / "data/processed/tetouan_two_hour.csv",
        config_path=ROOT / "configs/v3/energy_case_study.yaml",
        frozen_results_dir=ROOT / "results/development/energy_case_v3",
    )
    summary = result.summary

    assert summary["zone"].tolist() == [
        "zone_1",
        "zone_2",
        "zone_3",
    ]
    assert summary["status"].eq(POSTHOC_STATUS).all()
    assert summary["bma_applicable"].tolist() == [True, True, False]
    assert not summary["development_decision_modified"].any()
    assert summary["max_point_reproduction_error"].max() <= 1e-8
    assert summary["max_target_reproduction_error"].max() <= 1e-8
    assert summary["max_seasonal_reproduction_error"].max() <= 1e-8

    selected = summary.loc[summary["bma_applicable"]]
    assert np.isfinite(
        selected[
            [
                "point_rmse",
                "bma_rmse",
                "point_asymmetric_score",
                "bma_asymmetric_score",
            ]
        ]
    ).all().all()

    fallback = summary.loc[summary["zone"].eq("zone_3")].iloc[0]
    assert fallback["reason"] == (
        "frozen_fallback_has_no_selected_posterior"
    )
    assert np.isnan(fallback["bma_rmse"])
    assert np.isnan(fallback["bma_asymmetric_score"])

    zone_1 = summary.loc[summary["zone"].eq("zone_1")].iloc[0]
    zone_2 = summary.loc[summary["zone"].eq("zone_2")].iloc[0]
    assert zone_1["bma_rmse_change_vs_point"] < 0.0
    assert zone_2["bma_rmse_change_vs_point"] > 0.20

    assert set(result.predictions) == {
        "zone_1",
        "zone_2",
        "zone_3",
    }
    assert result.predictions["zone_1"][
        "exact_clipped_bma_forecast"
    ].notna().all()
    assert result.predictions["zone_2"][
        "exact_clipped_bma_forecast"
    ].notna().all()
    assert result.predictions["zone_3"][
        "exact_clipped_bma_forecast"
    ].isna().all()
