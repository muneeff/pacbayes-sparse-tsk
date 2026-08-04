#!/usr/bin/env python
"""Freeze the prospective V6 protocol before downloading NESO data."""

from pathlib import Path

from pacbayes_tsk.experiments.freeze_v3 import freeze


ROOT = Path(".")
TRACKED: list[str] = []

for base in [
    "protocol",
    "configs/v6",
    "src/pacbayes_tsk",
    "scripts",
    "tests/v6",
    "docs",
    ".github",
]:
    root = ROOT / base
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ):
            TRACKED.append(path.as_posix())

TRACKED += [
    "pyproject.toml",
    "requirements-lock.txt",
    "README.md",
    "CITATION.cff",
    ".gitignore",
    "EXECUTION_STATUS_AR.md",
]

if __name__ == "__main__":
    freeze(
        "protocol/EXPERIMENT_PROTOCOL_V6.md",
        sorted(set(TRACKED)),
        "artifacts/prospective_protocol_snapshot_v6.json",
    )
