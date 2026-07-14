from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_posthoc_benchmark_is_explicitly_secondary_and_does_not_modify_gate():
    cfg = yaml.safe_load(
        (ROOT / "configs/v4/posthoc_energy_benchmarks_v4_3.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert cfg["status"] == "secondary_post_hoc_only"
    assert list(cfg["datasets"]) == ["pjm"]
    summary = pd.read_csv(
        ROOT / "results/posthoc/v4_3/energy_posthoc_summary_all.csv"
    )
    assert len(summary) == 8
    assert set(summary["family"]) == {"SARIMA", "ETS"}
    assert not summary["test_used_for_selection"].astype(bool).any()
    audit = json.loads(
        (ROOT / "results/posthoc/v4_3/posthoc_energy_benchmarks_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["confirmatory_gate_modified"] is False
    assert audit["confirmatory_run_reexecuted"] is False
    assert audit["test_used_for_hyperparameter_selection"] is False


def test_rule_examples_are_multi_rule_and_posthoc_only():
    summary = pd.read_csv(
        ROOT / "results/interpretability/v4_3/fuzzy_rule_examples_summary.csv"
    )
    assert len(summary[summary["example"] == "SETAR seed 3001"]) == 5
    assert len(summary[summary["example"] == "Tetouan Zone 1"]) == 4
    assert (summary["prior_support"] > 0).all()
    assert (summary["mean_firing"] > 0).all()
    audit = json.loads(
        (ROOT / "results/interpretability/v4_3/fuzzy_rule_examples_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "post_hoc_interpretability_only"
    assert audit["selection_modified"] is False
    assert audit["confirmatory_gate_modified"] is False


def test_manuscript_distinguishes_certified_and_operational_predictors():
    model_text = (ROOT / "paper/sections/03_model_and_chronology.tex").read_text(
        encoding="utf-8"
    )
    assert "Gibbs predictor" in model_text
    assert "clipped Bayesian model average" in model_text
    assert "deterministic posterior-center point forecast" in model_text
    assert "need not equal" in model_text


def test_joint_prior_is_explicit_on_augmented_space():
    text = (ROOT / "paper/sections/04_certificate.tex").read_text(encoding="utf-8")
    assert "h=(M,v,s,a)" in text
    assert "P(dM,dv,ds,da)" in text
    assert "Q=\\delta_{\\widehat M}" in text
    assert "\\mathrm{KL}(Q\\|P)" in text
    assert "dummy hypothesis" in text


def test_operational_incident_details_are_in_supplement():
    protocol = (ROOT / "paper/sections/05_experimental_protocol.tex").read_text(
        encoding="utf-8"
    )
    supplement = (ROOT / "paper/supplementary.tex").read_text(encoding="utf-8")
    assert "reported only in the supplementary audit" in protocol
    assert "first execution stopped" in supplement
    assert "seven conditions" in supplement
