from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/freeze_amendment_v6_001.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "freeze_amendment_v6_001",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_amendment_snapshot_is_distinct_and_chained() -> None:
    module = _module()
    assert module.OUTPUT.name == (
        "prospective_execution_snapshot_v6_amendment_001.json"
    )
    assert module.PARENT_SNAPSHOT.name == (
        "prospective_execution_snapshot_v6.json"
    )
    assert module.OUTPUT != module.PARENT_SNAPSHOT


def test_required_amendment_inputs_exist() -> None:
    module = _module()
    missing = sorted(
        path
        for path in module.REQUIRED_FILES
        if not (module.ROOT / path).is_file()
    )
    assert missing == []


def test_amendment_snapshot_never_hashes_itself() -> None:
    module = _module()
    selected = module.select_snapshot_paths(
        [
            "artifacts/prospective_execution_snapshot_v6.json",
            "artifacts/prospective_execution_snapshot_v6_amendment_001.json",
            "artifacts/neso_v6_download_audit.json",
        ]
    )
    assert selected == [
        "artifacts/neso_v6_download_audit.json",
        "artifacts/prospective_execution_snapshot_v6.json",
    ]


def test_local_generators_and_manuscript_builds_are_excluded() -> None:
    module = _module()
    selected = module.select_snapshot_paths(
        [
            "V6_LOCAL_GENERATOR.py",
            "paper/Manuscript.pdf",
            "scripts/prepare_neso_demand_2025_v6.py",
            "protocol/AMENDMENT_V6_001_NESO_ISO_DATE.md",
        ]
    )
    assert selected == [
        "protocol/AMENDMENT_V6_001_NESO_ISO_DATE.md",
        "scripts/prepare_neso_demand_2025_v6.py",
    ]
