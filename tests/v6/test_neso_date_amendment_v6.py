from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v6/neso_demand_2025_v6.yaml"
AMENDMENT = ROOT / "protocol/AMENDMENT_V6_001_NESO_ISO_DATE.md"

RAW_SHA256 = (
    "8b8be946bbb084bc28fbf95a71fad0fceb5543b98e02755235130a42beaee4e0"
)
PARENT_SNAPSHOT_SHA256 = (
    "c0bb169c88073854fd3d936190238d110a2d5bce5931e2bebd6e7db42d338d34"
)


def test_amended_config_uses_strict_iso_date_contract() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["status"] == "prospective_pre_outcome_amended_001"
    assert cfg["preprocessing"]["date_format"] == "%Y-%m-%d"
    assert cfg["preprocessing"]["parse_dates_day_first"] is False
    assert cfg["amendment"]["id"] == "V6-001"
    assert cfg["amendment"]["scope"] == "input_date_parser_only"
    assert cfg["amendment"]["downloaded_raw_sha256"] == RAW_SHA256
    assert (
        cfg["amendment"]["parent_execution_snapshot_sha256"]
        == PARENT_SNAPSHOT_SHA256
    )
    assert cfg["amendment"]["outcome_reviewed_before_amendment"] is False
    assert cfg["amendment"]["redownload_forbidden"] is True


def test_amendment_document_preserves_auditable_chain() -> None:
    text = AMENDMENT.read_text(encoding="utf-8")
    assert "V6-001" in text
    assert RAW_SHA256 in text
    assert PARENT_SNAPSHOT_SHA256 in text
    assert "No `ND` values" in text
    assert "must not be refreshed" in text
    assert "pre-outcome parser amendment" in text
