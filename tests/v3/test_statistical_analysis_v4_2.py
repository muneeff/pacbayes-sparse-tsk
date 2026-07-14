from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STAT_ROOT = ROOT / "results/statistics/v4_2"


def test_synthetic_statistical_outputs_are_complete() -> None:
    frame = pd.read_csv(STAT_ROOT / "synthetic_primary_paired_statistics.csv")
    assert len(frame) == 16
    assert set(frame["comparison_id"]) == {
        "radius_vs_fixed_12",
        "radius_vs_best_fixed",
        "top3_vs_all_active",
        "radius_vs_ridge",
    }
    assert (frame["pair_count"] == 30).all()
    assert np.isfinite(frame["bootstrap_ci_low"]).all()
    assert np.isfinite(frame["bootstrap_ci_high"]).all()
    assert ((frame["wilcoxon_p_holm"] >= 0) & (frame["wilcoxon_p_holm"] <= 1)).all()


def test_primary_radius_vs_fixed_12_intervals_support_structural_effect() -> None:
    frame = pd.read_csv(STAT_ROOT / "synthetic_primary_paired_statistics.csv")
    primary = frame[frame["comparison_id"] == "radius_vs_fixed_12"].set_index("metric")
    for metric in (
        "consequent_dimension",
        "localized_gaussian_kl",
        "localized_certificate_familywise",
        "test_rmse_scaled",
    ):
        assert primary.loc[metric, "bootstrap_ci_high"] < 0
        assert primary.loc[metric, "wilcoxon_p_holm"] < 0.05


def test_pjm_primary_block_intervals_are_positive_and_match_decisions() -> None:
    bootstrap = pd.read_csv(STAT_ROOT / "pjm_moving_block_bootstrap_ci.csv")
    primary = bootstrap[bootstrap["analysis_role"] == "primary"].sort_values("region")
    assert len(primary) == 4
    assert (primary["block_length"] == 14).all()
    assert (primary["rmse_ci_low"] > 0).all()
    assert (primary["cost_ci_low"] > 0).all()

    decisions = pd.read_csv(
        ROOT / "results/confirmatory/pjm_case_v3_4/pjm_deployment_decisions.csv"
    ).copy()
    decisions["region"] = decisions["region"].str.upper()
    merged = primary.merge(decisions, on="region", how="inner")
    assert len(merged) == 4
    assert np.allclose(
        merged["observed_rmse_improvement"],
        merged["test_rmse_improvement_vs_fallback"],
        atol=1e-10,
    )
    assert np.allclose(
        merged["observed_cost_improvement"],
        merged["test_cost_improvement_vs_fallback"],
        atol=1e-10,
    )


def test_statistical_analysis_does_not_modify_frozen_pjm_decisions() -> None:
    manifest = ROOT / "artifacts/statistical_analysis_v4_2_manifest.json"
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert '"selection_or_gate_changed": false' in text
    assert '"post_outcome_disclosure": true' in text
