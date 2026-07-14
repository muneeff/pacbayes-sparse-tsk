from pathlib import Path

import numpy as np
import pandas as pd

from pacbayes_tsk.experiments.development_v3 import (
    DevelopmentSettings,
    run_series,
)
from pacbayes_tsk.models.sparse_tsk import fit_fixed_k_antecedent
from pacbayes_tsk.pac_bayes.priors_v3 import (
    HierarchicalModelPrior,
    ModelIndex,
)


def test_fixed_k_antecedent_preserves_requested_rule_count():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(160, 4))
    antecedent = fit_fixed_k_antecedent(
        X,
        rule_count=12,
        max_active_rules=12,
    )
    assert antecedent.rule_count == 12
    assert antecedent.radius_cap_reached is False
    assert antecedent.max_active_rules == 12
    firing = antecedent.firing_strengths(X[:8])
    assert firing.shape == (8, 12)
    assert np.allclose(firing.sum(axis=1), 1.0)


def test_structural_prior_accepts_fixed_k_and_charges_rule_count():
    prior = HierarchicalModelPrior(
        families=(
            "ridge",
            "fixed_k_dense_tsk",
            "dense_tsk",
            "sparse_tsk",
        ),
        lags=(3, 5),
        radii=(1.0, 2.0),
        ridge_alphas=(0.1, 1.0),
        max_rules=12,
    )
    fixed = prior.negative_log_mass(
        ModelIndex(
            family="fixed_k_dense_tsk",
            lag=3,
            ridge_alpha=0.1,
            radius=None,
            rule_count=12,
        )
    )
    radius_sparse = prior.negative_log_mass(
        ModelIndex(
            family="dense_tsk",
            lag=3,
            ridge_alpha=0.1,
            radius=1.0,
            rule_count=4,
        )
    )
    assert fixed > 0
    assert radius_sparse > 0
    assert fixed > radius_sparse


def _tiny_structural_settings() -> DevelopmentSettings:
    return DevelopmentSettings(
        processes=("ar2",),
        seeds=(3000,),
        families=(
            "ridge",
            "fixed_k_dense_tsk",
            "dense_tsk",
            "sparse_tsk",
        ),
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
        length=180,
        burn_in=50,
        split_fractions={
            "prior": 0.20,
            "bound": 0.45,
            "validation": 0.15,
            "test": 0.20,
        },
        process_parameters={
            "ar2": {"phi1": 0.6, "phi2": -0.2, "noise_std": 0.2}
        },
    )


def test_run_series_includes_fixed_k_family(tmp_path: Path):
    result = run_series(
        process="ar2",
        seed=3000,
        settings=_tiny_structural_settings(),
        output_dir=tmp_path,
    )
    candidates = pd.read_csv(result["candidate_path"])
    selected = pd.read_csv(result["selected_path"])
    fixed = candidates[candidates["family"] == "fixed_k_dense_tsk"]
    assert not fixed.empty
    assert (fixed["rule_count"] == 6).all()
    assert fixed["radius"].isna().all()
    assert set(selected["family"]) == {
        "ridge",
        "fixed_k_dense_tsk",
        "dense_tsk",
        "sparse_tsk",
    }
    assert len(selected) == 8


def test_structural_prior_is_normalized_over_declared_support():
    prior = HierarchicalModelPrior(
        families=(
            "ridge",
            "fixed_k_dense_tsk",
            "dense_tsk",
            "sparse_tsk",
        ),
        lags=(3, 5),
        radii=(1.0, 2.0),
        ridge_alphas=(0.1, 1.0),
        max_rules=3,
    )
    mass = 0.0
    for family in prior.families:
        for lag in prior.lags:
            for alpha in prior.ridge_alphas:
                if family == "ridge":
                    mass += np.exp(
                        -prior.negative_log_mass(
                            ModelIndex(family, lag, alpha, None, 1)
                        )
                    )
                elif family == "fixed_k_dense_tsk":
                    for k in range(1, prior.max_rules + 1):
                        mass += np.exp(
                            -prior.negative_log_mass(
                                ModelIndex(family, lag, alpha, None, k)
                            )
                        )
                else:
                    for radius in prior.radii:
                        for k in range(1, prior.max_rules + 1):
                            mass += np.exp(
                                -prior.negative_log_mass(
                                    ModelIndex(family, lag, alpha, radius, k)
                                )
                            )
    assert np.isclose(mass, 1.0)


def test_fixed_k_sensitivity_evaluates_predeclared_k_grid(tmp_path: Path):
    base = _tiny_structural_settings()
    settings = DevelopmentSettings(
        **{
            **base.__dict__,
            "fixed_k_values": (2, 4, 6),
        }
    )
    result = run_series(
        process="ar2",
        seed=3000,
        settings=settings,
        output_dir=tmp_path,
    )
    candidates = pd.read_csv(result["candidate_path"])
    fixed = candidates[candidates["family"] == "fixed_k_dense_tsk"]
    assert set(fixed["rule_count"].astype(int)) == {2, 4, 6}
    assert (fixed["requested_fixed_k"].astype(int) == fixed["rule_count"].astype(int)).all()


def test_fixed_k_refit_uses_selected_rule_count(tmp_path: Path):
    base = _tiny_structural_settings()
    settings = DevelopmentSettings(
        **{
            **base.__dict__,
            "fixed_k_values": (2, 6),
        }
    )
    result = run_series(
        process="ar2",
        seed=3000,
        settings=settings,
        output_dir=tmp_path,
    )
    selected = pd.read_csv(result["selected_path"])
    fixed = selected[selected["family"] == "fixed_k_dense_tsk"]
    assert not fixed.empty
    assert set(fixed["rule_count"].astype(int)).issubset({2, 6})
    assert np.isfinite(fixed["test_rmse_scaled"]).all()
