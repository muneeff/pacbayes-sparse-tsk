#!/usr/bin/env python
"""Run the single authorized prospective NESO V6 experiment."""

from __future__ import annotations

import argparse
import json

from pacbayes_tsk.experiments.neso_prospective_v6 import (
    run_prospective_neso_v6,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed",
        default="data/processed/neso_demand_2025_daily.csv",
    )
    parser.add_argument(
        "--config",
        default="configs/v6/neso_demand_2025_v6.yaml",
    )
    parser.add_argument(
        "--protocol",
        default="protocol/EXPERIMENT_PROTOCOL_V6.md",
    )
    parser.add_argument(
        "--execution-snapshot",
        default="artifacts/prospective_execution_snapshot_v6.json",
    )
    parser.add_argument(
        "--download-audit",
        default="artifacts/neso_v6_download_audit.json",
    )
    parser.add_argument(
        "--preparation-audit",
        default="artifacts/neso_v6_preparation_audit.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/neso_prospective_v6",
    )
    args = parser.parse_args()

    audit = run_prospective_neso_v6(
        processed_path=args.processed,
        config_path=args.config,
        protocol_path=args.protocol,
        execution_snapshot_path=args.execution_snapshot,
        download_audit_path=args.download_audit,
        preparation_audit_path=args.preparation_audit,
        output_dir=args.output_dir,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
