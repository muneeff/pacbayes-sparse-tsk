#!/usr/bin/env python
"""Prepare the one-shot NESO 2025 daily dataset under V6-001."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pacbayes_tsk.data.energy_neso_v6 import (
    prepare_neso_daily_dataset,
    sha256_file,
)
from pacbayes_tsk.experiments.neso_prospective_v6 import (
    ProspectiveNESOV6Error,
    _verify_execution_snapshot,
)


def _load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProspectiveNESOV6Error(
            f"Expected a JSON mapping in {path}."
        )
    return payload


def _verify_amendment_chain_before_preparation(
    *,
    raw_path: str | Path,
    config_path: str | Path,
    amendment_snapshot_path: str | Path,
    parent_snapshot_path: str | Path,
    download_audit_path: str | Path,
) -> None:
    current = _verify_execution_snapshot(amendment_snapshot_path)
    if current.get("snapshot_kind") != "PRE_OUTCOME_AMENDMENT_V6_001":
        raise ProspectiveNESOV6Error(
            "Preparation requires the V6-001 amendment snapshot."
        )

    config = yaml.safe_load(
        Path(config_path).read_text(encoding="utf-8")
    )
    amendment = config.get("amendment", {})
    parent_sha = sha256_file(parent_snapshot_path)
    raw_sha = sha256_file(raw_path)
    download = _load_json(download_audit_path)

    checks = {
        "amendment_id": (
            amendment.get("id"),
            "V6-001",
        ),
        "config_parent_snapshot": (
            amendment.get("parent_execution_snapshot_sha256"),
            parent_sha,
        ),
        "snapshot_parent": (
            current.get("parent_execution_snapshot", {}).get(
                "sha256"
            ),
            parent_sha,
        ),
        "download_parent": (
            download.get("execution_snapshot_sha256"),
            parent_sha,
        ),
        "config_raw": (
            amendment.get("downloaded_raw_sha256"),
            raw_sha,
        ),
        "snapshot_raw": (
            current.get("download", {}).get("raw_sha256"),
            raw_sha,
        ),
        "download_raw": (
            download.get("raw_sha256"),
            raw_sha,
        ),
    }
    failed = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in checks.items()
        if observed != expected
    }
    if failed:
        raise ProspectiveNESOV6Error(
            f"V6-001 preparation-chain mismatch: {failed}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        default="data/raw/neso_demanddata_2025.csv",
    )
    parser.add_argument(
        "--processed",
        default="data/processed/neso_demand_2025_daily.csv",
    )
    parser.add_argument(
        "--audit",
        default="artifacts/neso_v6_preparation_audit.json",
    )
    parser.add_argument(
        "--config",
        default="configs/v6/neso_demand_2025_v6.yaml",
    )
    parser.add_argument(
        "--amendment-snapshot",
        default=(
            "artifacts/"
            "prospective_execution_snapshot_v6_amendment_001.json"
        ),
    )
    parser.add_argument(
        "--parent-snapshot",
        default="artifacts/prospective_execution_snapshot_v6.json",
    )
    parser.add_argument(
        "--download-audit",
        default="artifacts/neso_v6_download_audit.json",
    )
    args = parser.parse_args()

    _verify_amendment_chain_before_preparation(
        raw_path=args.raw,
        config_path=args.config,
        amendment_snapshot_path=args.amendment_snapshot,
        parent_snapshot_path=args.parent_snapshot,
        download_audit_path=args.download_audit,
    )
    audit = prepare_neso_daily_dataset(
        args.raw,
        args.processed,
        args.audit,
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
