from pathlib import Path

import numpy as np

from pacbayes_tsk.experiments.retrospective_bma_v5 import (
    POSTHOC_STATUS,
)
from pacbayes_tsk.experiments.retrospective_tetouan_selected_bma_gate_v5 import (
    evaluate_tetouan_selected_bma_gate_retrospective,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_tetouan_selections_are_rejected_by_exact_bma_gate() -> None:
    result = evaluate_tetouan_selected_bma_gate_retrospective(
        case_data_path=(
            ROOT / "data/processed/tetouan_two_hour.csv"
        ),
        energy_config_path=(
            ROOT / "configs/v3/energy_case_study.yaml"
        ),
        gate_config_path=(
            ROOT / "configs/v3/pjm_confirmatory_case_v3.yaml"
        ),
        frozen_results_dir=(
            ROOT / "results/development/energy_case_v3"
        ),
    )
    summary = result.summary

    assert summary["zone"].tolist() == [
        "zone_1",
        "zone_2",
        "zone_3",
    ]
    assert summary["status"].eq(POSTHOC_STATUS).all()
    assert summary["bma_gate_applicable"].tolist() == [
        True,
        True,
        False,
    ]

    selected = summary.loc[summary["bma_gate_applicable"]]
    assert not selected["gate_passed"].any()
    assert selected["posterior_scale_changed"].tolist() == [
        True,
        False,
    ]
    assert selected["exact_certificate_untruncated"].gt(0.10).all()
    assert selected[
        "test_exact_bma_rmse_improvement_vs_fallback"
    ].lt(0.0).all()
    assert selected[
        "test_exact_bma_score_improvement_vs_fallback"
    ].lt(0.0).all()
    assert selected[
        "abstention_avoided_test_rmse_deterioration"
    ].all()
    assert selected[
        "abstention_avoided_test_score_deterioration"
    ].all()
    assert selected[
        "max_frozen_scale_bma_reproduction_error"
    ].max() <= 1e-8

    zone_1 = selected.loc[selected["zone"].eq("zone_1")].iloc[0]
    zone_2 = selected.loc[selected["zone"].eq("zone_2")].iloc[0]
    assert set(zone_1["failed_conditions"]) == {
        "certificate",
        "worst_block_degradation",
    }
    assert tuple(zone_2["failed_conditions"]) == ("certificate",)
    assert (
        zone_1[
            "mean_abs_exact_bma_minus_frozen_scale_bma"
        ]
        > 50.0
    )
    assert (
        zone_2[
            "max_abs_exact_bma_minus_frozen_scale_bma"
        ]
        == 0.0
    )

    fallback = summary.loc[summary["zone"].eq("zone_3")].iloc[0]
    assert not fallback["bma_gate_applicable"]
    assert fallback["abstention_action"] == "seasonal_fallback"

    assert set(result.predictions) == {"zone_1", "zone_2"}
    for frame in result.predictions.values():
        assert np.isfinite(frame["exact_scale_bma"]).all()
        assert frame["analysis_status"].eq(POSTHOC_STATUS).all()

    assert not summary["selection_uses_test"].any()
    assert summary["interpretation_uses_test_posthoc"].all()
    assert not summary["development_decision_modified"].any()
