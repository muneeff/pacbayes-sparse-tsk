from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/freeze_execution_v6.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "freeze_execution_v6",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_execution_snapshot_is_distinct_from_protocol_snapshot() -> None:
    module = _module()
    assert module.OUTPUT.name == (
        "prospective_execution_snapshot_v6.json"
    )
    assert module.PROTOCOL_SNAPSHOT.name == (
        "prospective_protocol_snapshot_v6.json"
    )
    assert module.OUTPUT != module.PROTOCOL_SNAPSHOT


def test_required_execution_inputs_exist() -> None:
    module = _module()
    missing = sorted(
        path
        for path in module.REQUIRED_FILES
        if not (module.ROOT / path).is_file()
    )
    assert missing == []


def test_path_selection_is_deterministic_and_domain_limited() -> None:
    module = _module()
    candidates = [
        "paper/Manuscript.pdf",
        "V6_LOCAL_GENERATOR.py",
        "src/pacbayes_tsk/example.py",
        "scripts/run_neso_prospective_v6.py",
        "artifacts/prospective_protocol_snapshot_v6.json",
        "tests/v6/test_example.py",
        "src\\pacbayes_tsk\\windows_path.py",
    ]
    selected = module.select_snapshot_paths(candidates)
    assert selected == sorted(
        [
            "artifacts/prospective_protocol_snapshot_v6.json",
            "scripts/run_neso_prospective_v6.py",
            "src/pacbayes_tsk/example.py",
            "src/pacbayes_tsk/windows_path.py",
            "tests/v6/test_example.py",
        ]
    )
    assert "paper/Manuscript.pdf" not in selected
    assert "V6_LOCAL_GENERATOR.py" not in selected


def test_final_snapshot_is_never_self_hashed() -> None:
    module = _module()
    selected = module.select_snapshot_paths(
        [
            "artifacts/prospective_execution_snapshot_v6.json",
            "artifacts/prospective_protocol_snapshot_v6.json",
        ]
    )
    assert selected == [
        "artifacts/prospective_protocol_snapshot_v6.json"
    ]
