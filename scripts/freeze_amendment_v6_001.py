#!/usr/bin/env python
"""Freeze the chained pre-outcome V6-001 execution amendment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "artifacts/prospective_execution_snapshot_v6_amendment_001.json"
)
PARENT_SNAPSHOT = (
    ROOT / "artifacts/prospective_execution_snapshot_v6.json"
)
DOWNLOAD_AUDIT = ROOT / "artifacts/neso_v6_download_audit.json"
RAW_DATA = ROOT / "data/raw/neso_demanddata_2025.csv"
CONFIG = ROOT / "configs/v6/neso_demand_2025_v6.yaml"
AMENDMENT_DOCUMENT = (
    ROOT / "protocol/AMENDMENT_V6_001_NESO_ISO_DATE.md"
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
        "artifacts/prospective_execution_snapshot_v6.json",
        "artifacts/neso_v6_download_audit.json",
    }
)
REQUIRED_FILES = frozenset(
    {
        "artifacts/neso_v6_download_audit.json",
        "artifacts/prospective_execution_snapshot_v6.json",
        "artifacts/prospective_protocol_snapshot_v6.json",
        "configs/v6/neso_demand_2025_v6.yaml",
        "protocol/AMENDMENT_V6_001_NESO_ISO_DATE.md",
        "protocol/EXPERIMENT_PROTOCOL_V6.md",
        "scripts/download_neso_demand_2025_v6.py",
        "scripts/freeze_amendment_v6_001.py",
        "scripts/freeze_execution_v6.py",
        "scripts/prepare_neso_demand_2025_v6.py",
        "scripts/run_neso_prospective_v6.py",
        "src/pacbayes_tsk/data/energy_neso_v6.py",
        "src/pacbayes_tsk/experiments/neso_prospective_v6.py",
        "tests/v6/test_amendment_freeze_v6.py",
        "tests/v6/test_execution_freeze_v6.py",
        "tests/v6/test_neso_data_v6.py",
        "tests/v6/test_neso_date_amendment_v6.py",
        "tests/v6/test_neso_prospective_v6.py",
        "tests/v6/test_protocol_v6.py",
    }
)


class AmendmentFreezeV6Error(RuntimeError):
    """Raised when the V6-001 chained snapshot cannot be frozen safely."""


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
    selected.discard(
        "artifacts/prospective_execution_snapshot_v6_amendment_001.json"
    )
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
        raise AmendmentFreezeV6Error(
            "Required V6-001 files are not Git-tracked: "
            f"{missing}"
        )
    return selected


def assert_clean_tracked_state() -> None:
    unstaged = _git("diff", "--quiet", check=False)
    staged = _git("diff", "--cached", "--quiet", check=False)
    if unstaged.returncode not in {0, 1}:
        raise AmendmentFreezeV6Error(
            "git diff failed while checking the working tree."
        )
    if staged.returncode not in {0, 1}:
        raise AmendmentFreezeV6Error(
            "git diff --cached failed while checking the index."
        )
    if unstaged.returncode == 1 or staged.returncode == 1:
        raise AmendmentFreezeV6Error(
            "Tracked files must be clean before freezing V6-001."
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
    relevant = select_snapshot_paths(untracked)
    if relevant:
        raise AmendmentFreezeV6Error(
            "Untracked files exist inside frozen V6-001 domains: "
            f"{relevant}"
        )


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AmendmentFreezeV6Error(
            f"Expected a JSON mapping in {path}."
        )
    return payload


def _load_config() -> dict:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AmendmentFreezeV6Error(
            "The amended V6 configuration is not a mapping."
        )
    return payload


def _verify_chain_inputs() -> tuple[dict, dict, dict, str]:
    for path in (
        PARENT_SNAPSHOT,
        DOWNLOAD_AUDIT,
        RAW_DATA,
        CONFIG,
        AMENDMENT_DOCUMENT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = _load_config()
    amendment = config.get("amendment", {})
    if amendment.get("id") != "V6-001":
        raise AmendmentFreezeV6Error(
            "The configuration is not the V6-001 amendment."
        )
    if amendment.get("outcome_reviewed_before_amendment") is not False:
        raise AmendmentFreezeV6Error(
            "V6-001 must remain explicitly pre-outcome."
        )
    if amendment.get("redownload_forbidden") is not True:
        raise AmendmentFreezeV6Error(
            "V6-001 must explicitly forbid redownload."
        )

    parent = _load_json(PARENT_SNAPSHOT)
    parent_sha = sha256_file(PARENT_SNAPSHOT)
    if parent.get("snapshot_kind") != "FINAL_EXECUTION_PRE_DOWNLOAD":
        raise AmendmentFreezeV6Error(
            "Unexpected parent execution-snapshot kind."
        )
    if parent.get("status") != "FROZEN_NOT_RUN":
        raise AmendmentFreezeV6Error(
            "The parent execution snapshot is not FROZEN_NOT_RUN."
        )
    if amendment.get("parent_execution_snapshot_sha256") != parent_sha:
        raise AmendmentFreezeV6Error(
            "The amended configuration does not bind the parent snapshot."
        )

    download = _load_json(DOWNLOAD_AUDIT)
    raw_sha = sha256_file(RAW_DATA)
    if download.get("status") != "DOWNLOADED_NOT_PREPARED":
        raise AmendmentFreezeV6Error(
            "The download audit is not DOWNLOADED_NOT_PREPARED."
        )
    if download.get("execution_snapshot_sha256") != parent_sha:
        raise AmendmentFreezeV6Error(
            "The download is not bound to the parent snapshot."
        )
    if download.get("raw_sha256") != raw_sha:
        raise AmendmentFreezeV6Error(
            "The raw-file hash differs from its download audit."
        )
    if amendment.get("downloaded_raw_sha256") != raw_sha:
        raise AmendmentFreezeV6Error(
            "The amended configuration does not bind the raw bytes."
        )
    return config, parent, download, raw_sha


def create_amendment_snapshot() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite V6-001 snapshot: {OUTPUT}"
        )
    assert_clean_tracked_state()
    config, parent, download, raw_sha = _verify_chain_inputs()
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
        "snapshot_kind": "PRE_OUTCOME_AMENDMENT_V6_001",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_NOT_RUN",
        "single_authorized_run_only": True,
        "git": {
            "branch": branch,
            "head": head,
            "tracked_state_clean": True,
        },
        "amendment": {
            "id": "V6-001",
            "scope": "input_date_parser_only",
            "document_path": (
                AMENDMENT_DOCUMENT.relative_to(ROOT).as_posix()
            ),
            "document_sha256": sha256_file(AMENDMENT_DOCUMENT),
            "outcome_reviewed_before_amendment": False,
            "redownload_forbidden": True,
        },
        "parent_execution_snapshot": {
            "path": PARENT_SNAPSHOT.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(PARENT_SNAPSHOT),
            "git_head": parent.get("git", {}).get("head"),
        },
        "download": {
            "audit_path": DOWNLOAD_AUDIT.relative_to(ROOT).as_posix(),
            "audit_sha256": sha256_file(DOWNLOAD_AUDIT),
            "raw_path": RAW_DATA.relative_to(ROOT).as_posix(),
            "raw_sha256": raw_sha,
            "raw_size_bytes": RAW_DATA.stat().st_size,
            "resource_id": download.get("resource_id"),
            "downloaded_utc": download.get("downloaded_utc"),
        },
        "config": {
            "path": CONFIG.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(CONFIG),
            "status": config.get("status"),
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
    payload = create_amendment_snapshot()
    print(f"FILE: {OUTPUT.relative_to(ROOT)}")
    print(f"STATUS: {payload['status']}")
    print(f"KIND: {payload['snapshot_kind']}")
    print(f"GIT_HEAD: {payload['git']['head']}")
    print(f"TRACKED_COUNT: {len(payload['tracked_files'])}")
    print(
        "PARENT_SNAPSHOT_SHA256: "
        f"{payload['parent_execution_snapshot']['sha256']}"
    )
    print(f"RAW_SHA256: {payload['download']['raw_sha256']}")


if __name__ == "__main__":
    main()
