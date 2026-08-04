#!/usr/bin/env python
"""Create the final pre-download execution snapshot for V6.

This snapshot is intentionally distinct from the earlier protocol-only snapshot.
It hashes only Git-tracked implementation inputs and refuses to run from a
modified index or working tree.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/prospective_execution_snapshot_v6.json"
PROTOCOL = ROOT / "protocol/EXPERIMENT_PROTOCOL_V6.md"
PROTOCOL_SNAPSHOT = (
    ROOT / "artifacts/prospective_protocol_snapshot_v6.json"
)

TRACKED_PREFIXES = (
    ".github/",
    "configs/v6/",
    "docs/",
    "protocol/",
    "scripts/",
    "src/pacbayes_tsk/",
    "tests/v6/",
)
TRACKED_FIXED_FILES = frozenset(
    {
        ".gitignore",
        "CITATION.cff",
        "EXECUTION_STATUS_AR.md",
        "README.md",
        "pyproject.toml",
        "requirements-lock.txt",
        "artifacts/prospective_protocol_snapshot_v6.json",
    }
)
REQUIRED_FILES = frozenset(
    {
        "artifacts/prospective_protocol_snapshot_v6.json",
        "configs/v6/neso_demand_2025_v6.yaml",
        "protocol/EXPERIMENT_PROTOCOL_V6.md",
        "scripts/download_neso_demand_2025_v6.py",
        "scripts/freeze_execution_v6.py",
        "scripts/prepare_neso_demand_2025_v6.py",
        "scripts/run_neso_prospective_v6.py",
        "src/pacbayes_tsk/data/energy_neso_v6.py",
        "src/pacbayes_tsk/experiments/neso_prospective_v6.py",
        "src/pacbayes_tsk/optimization/certificate_subset_search_v6.py",
        "src/pacbayes_tsk/optimization/rule_subset_v6.py",
        "tests/v6/test_execution_freeze_v6.py",
        "tests/v6/test_neso_data_v6.py",
        "tests/v6/test_neso_prospective_v6.py",
        "tests/v6/test_protocol_v6.py",
    }
)


class ExecutionFreezeV6Error(RuntimeError):
    """Raised when a final V6 execution snapshot cannot be frozen safely."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=check,
        capture_output=True,
    )


def select_snapshot_paths(paths: Iterable[str]) -> list[str]:
    selected = {
        path.replace("\\", "/")
        for path in paths
        if (
            path.replace("\\", "/") in TRACKED_FIXED_FILES
            or any(
                path.replace("\\", "/").startswith(prefix)
                for prefix in TRACKED_PREFIXES
            )
        )
    }
    return sorted(selected)


def git_tracked_snapshot_paths() -> list[str]:
    raw = _git("ls-files", "-z").stdout
    tracked = [
        item.decode("utf-8")
        for item in raw.split(b"\0")
        if item
    ]
    selected = select_snapshot_paths(tracked)
    missing = sorted(REQUIRED_FILES.difference(selected))
    if missing:
        raise ExecutionFreezeV6Error(
            "Required V6 execution files are not Git-tracked: "
            f"{missing}"
        )
    return selected


def assert_clean_tracked_state() -> None:
    unstaged = _git("diff", "--quiet", check=False)
    staged = _git("diff", "--cached", "--quiet", check=False)
    if unstaged.returncode not in {0, 1}:
        raise ExecutionFreezeV6Error(
            "git diff failed while checking the working tree."
        )
    if staged.returncode not in {0, 1}:
        raise ExecutionFreezeV6Error(
            "git diff --cached failed while checking the index."
        )
    if unstaged.returncode == 1 or staged.returncode == 1:
        raise ExecutionFreezeV6Error(
            "Tracked files must be clean before freezing execution."
        )

    raw_untracked = _git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).stdout
    untracked = [
        item.decode("utf-8")
        for item in raw_untracked.split(b"\0")
        if item
    ]
    relevant_untracked = select_snapshot_paths(untracked)
    if relevant_untracked:
        raise ExecutionFreezeV6Error(
            "Untracked files exist inside the frozen execution domains: "
            f"{relevant_untracked}"
        )


def _require_initial_protocol_snapshot() -> dict:
    if not PROTOCOL_SNAPSHOT.is_file():
        raise FileNotFoundError(PROTOCOL_SNAPSHOT)
    payload = json.loads(
        PROTOCOL_SNAPSHOT.read_text(encoding="utf-8")
    )
    if payload.get("status") != "FROZEN_NOT_RUN":
        raise ExecutionFreezeV6Error(
            "The initial protocol snapshot is not FROZEN_NOT_RUN."
        )
    if not payload.get("tracked_files"):
        raise ExecutionFreezeV6Error(
            "The initial protocol snapshot contains no tracked files."
        )
    return payload


def create_execution_snapshot() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite final execution snapshot: {OUTPUT}"
        )
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)

    assert_clean_tracked_state()
    protocol_snapshot = _require_initial_protocol_snapshot()
    tracked_paths = git_tracked_snapshot_paths()
    rows = []
    for relative in tracked_paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    head = _git("rev-parse", "HEAD").stdout.decode("ascii").strip()
    branch = (
        _git("rev-parse", "--abbrev-ref", "HEAD")
        .stdout.decode("utf-8")
        .strip()
    )
    payload = {
        "schema_version": "6.0",
        "snapshot_kind": "FINAL_EXECUTION_PRE_DOWNLOAD",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_NOT_RUN",
        "single_authorized_run_only": True,
        "git": {
            "branch": branch,
            "head": head,
            "tracked_state_clean": True,
        },
        "protocol": {
            "path": PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(PROTOCOL),
        },
        "initial_protocol_snapshot": {
            "path": PROTOCOL_SNAPSHOT.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(PROTOCOL_SNAPSHOT),
            "created_utc": protocol_snapshot.get("created_utc"),
        },
        "tracked_files": rows,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, OUTPUT)
    return payload


def main() -> None:
    payload = create_execution_snapshot()
    print(f"FILE: {OUTPUT.relative_to(ROOT)}")
    print(f"STATUS: {payload['status']}")
    print(f"KIND: {payload['snapshot_kind']}")
    print(f"GIT_HEAD: {payload['git']['head']}")
    print(f"TRACKED_COUNT: {len(payload['tracked_files'])}")
    print(
        "PROTOCOL_SNAPSHOT_SHA256: "
        f"{payload['initial_protocol_snapshot']['sha256']}"
    )


if __name__ == "__main__":
    main()
