from pathlib import Path
import pandas as pd

from pacbayes_tsk.experiments.development_v3 import DevelopmentSettings, run_series


def tiny_settings():
    return DevelopmentSettings(
        processes=("ar2",),
        seeds=(3000,),
        families=("ridge", "dense_tsk", "sparse_tsk"),
        lags=(3,),
        radii=(2.0,),
        ridge_alphas=(0.1,),
        max_rules=6,
        fixed_k_values=(6,),
        sparse_active_rules=3,
        temperatures=(0.25, 0.5, 1.0),
        prior_scales=(0.5, 1.0),
        posterior_ratios=(0.1, 0.3),
        prior_variants=("localized", "zero_mean"),
        delta_total=0.05,
        familywise_series_count=1,
        length=160,
        burn_in=50,
        split_fractions={"prior": 0.20, "bound": 0.45, "validation": 0.15, "test": 0.20},
        process_parameters={"ar2": {"phi1": 0.6, "phi2": -0.2, "noise_std": 0.2}},
    )


def test_candidate_stage_excludes_test_and_selected_opens_after_selection(tmp_path: Path):
    settings = tiny_settings()
    result = run_series(process="ar2", seed=3000, settings=settings, output_dir=tmp_path)
    candidates = pd.read_csv(result["candidate_path"])
    selected = pd.read_csv(result["selected_path"])
    assert "test_rmse_scaled" not in candidates.columns
    assert (candidates["selection_uses_test"] == False).all()  # noqa: E712
    assert (candidates["certificate_uses_test"] == False).all()  # noqa: E712
    assert "test_rmse_scaled" in selected.columns
    assert selected["test_opened_after_selection"].all()
    assert set(selected["selection_strategy"]) == {"validation_rmse", "certificate"}
    assert len(selected) == 6


def test_prior_variant_mixture_is_charged(tmp_path: Path):
    settings = tiny_settings()
    result = run_series(process="ar2", seed=3000, settings=settings, output_dir=tmp_path)
    candidates = pd.read_csv(result["candidate_path"])
    assert (candidates["localized_prior_variant_penalty"] > 0).all()
    assert (candidates["zero_mean_prior_variant_penalty"] > 0).all()
