from pathlib import Path
import numpy as np
import pandas as pd

from pacbayes_tsk.data.energy_tetouan import (
    CLEAN_ZONE_COLUMNS,
    EXPECTED_TWO_HOUR_ROWS,
    load_two_hour_tetouan,
)
from pacbayes_tsk.experiments.energy_case_study_v3 import (
    EnergySettings,
    _selection_key,
    _weighted_cost,
)

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_energy_config_is_valid():
    settings = EnergySettings.from_yaml(ROOT / "configs/v3/energy_case_study.yaml")
    assert settings.familywise_series_count == 3
    assert settings.familywise_delta == settings.delta_total / 3
    assert settings.clip_multiplier >= 1
    assert settings.families == ("ridge", "fixed_k_dense_tsk", "dense_tsk")


def test_processed_tetouan_data_are_complete_and_chronological():
    frame = load_two_hour_tetouan(ROOT / "data/processed/tetouan_two_hour.csv")
    assert len(frame) == EXPECTED_TWO_HOUR_ROWS
    assert tuple(frame.columns[1:]) == CLEAN_ZONE_COLUMNS
    assert frame["timestamp"].is_monotonic_increasing
    assert not frame.isna().any().any()


def test_weighted_cost_penalizes_underforecast_more():
    target = np.array([10.0, 10.0])
    under_cost, _ = _weighted_cost(
        target, np.array([8.0, 8.0]), under_weight=2.0, over_weight=1.0
    )
    over_cost, _ = _weighted_cost(
        target, np.array([12.0, 12.0]), under_weight=2.0, over_weight=1.0
    )
    assert under_cost == 2 * over_cost


def test_selection_key_prefers_lower_validation_rmse_before_certificate():
    first = pd.Series(
        {
            "validation_rmse": 2.0,
            "certificate_untruncated": 0.10,
            "consequent_dimension": 10,
            "rule_count": 1,
            "family": "ridge",
            "radius": np.nan,
            "ridge_alpha": 0.1,
            "lag": 3,
        }
    )
    second = first.copy()
    second["validation_rmse"] = 1.9
    second["certificate_untruncated"] = 0.19
    assert _selection_key(second) < _selection_key(first)


def test_ridge_path_matches_single_alpha_solver():
    from pacbayes_tsk.experiments.energy_case_study_v3 import _ridge_path, _ridge_solution

    rng = np.random.default_rng(7)
    design = rng.normal(size=(80, 12))
    target = rng.normal(size=80)
    alphas = (0.001, 0.1, 1.0)
    path = _ridge_path(
        design, target, alphas, include_inverse_diagonal=True
    )
    for alpha in alphas:
        expected_mean, expected_diag = _ridge_solution(design, target, alpha)
        actual_mean, actual_diag = path[alpha]
        assert np.allclose(actual_mean, expected_mean, rtol=1e-8, atol=1e-8)
        assert actual_diag is not None
        assert np.allclose(actual_diag, expected_diag, rtol=1e-8, atol=1e-8)
