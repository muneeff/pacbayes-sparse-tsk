from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v6/neso_demand_2025_v6.yaml"
PROTOCOL = ROOT / "protocol/EXPERIMENT_PROTOCOL_V6.md"


def _load() -> dict:
    with CONFIG.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return payload


def test_v6_protocol_is_pre_download_and_single_run() -> None:
    cfg = _load()
    assert cfg["status"] == "prospective_pre_outcome_amended_001"
    assert cfg["amendment"]["id"] == "V6-001"
    assert cfg["amendment"]["scope"] == "input_date_parser_only"
    assert cfg["amendment"]["outcome_reviewed_before_amendment"] is False
    assert cfg["amendment"]["redownload_forbidden"] is True
    assert cfg["execution"]["single_authorized_run_only"] is True
    assert cfg["execution"]["download_before_protocol_freeze"] == "forbidden"
    assert cfg["execution"]["rerun_after_outcome"] == "forbidden"
    assert cfg["execution"]["retuning_after_outcome"] == "forbidden"


def test_v6_source_identity_and_snapshot_policy_are_frozen() -> None:
    cfg = _load()
    source = cfg["source"]
    assert source["publisher"] == "National Energy System Operator"
    assert source["resource_id"] == (
        "b2bde559-3455-4021-b179-dfe60c0337b0"
    )
    assert source["snapshot_policy"] == (
        "one_download_after_protocol_freeze"
    )
    assert source["required_attribution"] == (
        "Supported by National Energy SO Open Data"
    )


def test_v6_daily_preprocessing_and_split_are_exact() -> None:
    cfg = _load()
    assert cfg["preprocessing"]["expected_daily_rows"] == 365
    assert cfg["preprocessing"]["allowed_periods_per_day"] == [46, 48, 50]
    counts = cfg["split"]["counts"]
    assert counts == {
        "prior": 73,
        "bound": 164,
        "validation": 55,
        "test": 73,
    }
    assert sum(counts.values()) == 365


def test_v6_structural_support_and_candidate_count_are_exact() -> None:
    cfg = _load()
    model = cfg["model"]
    search = cfg["structural_search"]
    assert model["dictionary"]["rule_count"] == 8
    assert cfg["subset_prior"]["minimum_size"] == 1
    assert cfg["subset_prior"]["maximum_size"] == 8
    assert search["strategy"] == "exhaustive"
    assert search["expected_subset_count_per_lag_alpha"] == 255
    expected = (
        (2 ** model["dictionary"]["rule_count"] - 1)
        * len(model["lags"])
        * len(model["ridge_alphas"])
    )
    assert expected == 3060
    assert search["expected_total_structural_candidates"] == expected


def test_v6_certificate_and_operational_object_are_aligned() -> None:
    cfg = _load()
    certificate = cfg["certificate"]
    assert certificate["certified_roles"] == ["bound", "validation"]
    assert certificate["test_excluded"] is True
    assert certificate["empirical_risk"] == (
        "exact_expected_normalized_squared_loss_after_clipping"
    )
    assert certificate["operational_forecast"] == (
        "exact_clipped_gaussian_bayesian_model_average"
    )
    assert cfg["model"]["consequent_fit"]["refit_after_selection"] == (
        "forbidden"
    )


def test_v6_gate_is_prelocked_and_allows_abstention() -> None:
    cfg = _load()
    gate = cfg["decision"]["robust_gate"]
    assert gate["certificate_threshold_untruncated"] == 0.10
    assert gate["certification_target_clipping_rate_maximum"] == 0.02
    assert gate["minimum_validation_rmse_improvement"] == 0.05
    assert gate["minimum_validation_cost_improvement"] == 0.05
    assert gate["validation_blocks"] == 4
    assert gate["minimum_nonnegative_blocks"] == 3
    assert gate["maximum_worst_block_degradation"] == 0.05
    assert cfg["decision"]["outcome"]["fail"].startswith("abstain")


def test_v6_protocol_document_exists_and_names_core_claims() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    required = [
        "prospective_pre_download_frozen",
        "subset identity",
        "untruncated",
        "exact clipped Gaussian Bayesian model average",
        "Abstention is a valid prospective outcome",
        "3060",
    ]
    for phrase in required:
        assert phrase in text
