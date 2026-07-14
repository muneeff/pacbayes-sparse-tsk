from pathlib import Path
import inspect
import numpy as np
import pandas as pd

from pacbayes_tsk.data.energy_pjm import (
    EXPECTED_DAILY_ROWS,
    REGION_COLUMNS,
    load_pjm_daily,
)
from pacbayes_tsk.experiments.pjm_confirmatory_case_v3 import (
    PJMConfirmatorySettings,
    _block_stability,
    _deployment_selection_key,
    run_pjm_confirmatory_case,
)

ROOT = Path(__file__).resolve().parents[2]


def test_pjm_confirmatory_config_is_frozen_and_strict():
    settings = PJMConfirmatorySettings.from_yaml(
        ROOT / "configs/v3/pjm_confirmatory_case_v3.yaml"
    )
    assert settings.regions == REGION_COLUMNS
    assert settings.familywise_delta == settings.delta_total / 4
    assert settings.certificate_threshold == 0.10
    assert settings.clipping_rate_maximum == 0.02
    assert settings.minimum_rmse_improvement == 0.05
    assert settings.minimum_cost_improvement == 0.05
    assert settings.validation_blocks == 4
    assert settings.minimum_nonnegative_blocks == 3
    assert settings.maximum_worst_block_degradation == 0.05


def test_processed_pjm_data_are_complete_daily_and_positive():
    frame = load_pjm_daily(ROOT / "data/processed/pjm_daily.csv")
    assert len(frame) == EXPECTED_DAILY_ROWS
    assert tuple(frame.columns[1:]) == REGION_COLUMNS
    assert frame["timestamp"].is_monotonic_increasing
    assert not frame.isna().any().any()
    assert (frame.loc[:, REGION_COLUMNS] > 0).all().all()


def test_block_stability_detects_uniform_improvement():
    target = np.arange(1.0, 17.0)
    baseline = target + 2.0
    candidate = target + 1.0
    result = _block_stability(
        target,
        candidate,
        baseline,
        blocks=4,
        under_weight=2.0,
        over_weight=1.0,
    )
    assert result["validation_rmse_nonnegative_blocks"] == 4
    assert result["validation_cost_nonnegative_blocks"] == 4
    assert result["validation_worst_block_rmse_improvement"] > 0
    assert result["validation_worst_block_cost_improvement"] > 0


def test_deployment_key_prefers_lower_operational_cost_first():
    base = pd.Series(
        {
            "validation_weighted_cost": 10.0,
            "validation_rmse_clipped": 8.0,
            "certificate_untruncated": 0.07,
            "consequent_dimension": 20,
            "rule_count": 1,
            "family": "ridge",
            "radius": np.nan,
            "ridge_alpha": 0.1,
            "lag": 7,
        }
    )
    alternative = base.copy()
    alternative["validation_weighted_cost"] = 9.0
    alternative["validation_rmse_clipped"] = 9.0
    assert _deployment_selection_key(alternative) < _deployment_selection_key(base)


def test_confirmatory_runner_serializes_all_decisions_before_test_phase():
    source = inspect.getsource(run_pjm_confirmatory_case)
    phase1 = source.index("Phase 1: candidate construction and all pre-test decisions")
    pretest_write = source.index("pjm_pretest_decisions.csv")
    phase2 = source.index("Phase 2: open test only after all regional decisions are serialized")
    test_call = source.index('"test", settings.seasonal_lag', phase2)
    assert phase1 < pretest_write < phase2 < test_call
