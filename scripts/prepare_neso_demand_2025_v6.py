#!/usr/bin/env python
"""Prepare the one-shot NESO 2025 daily dataset for V6."""

from __future__ import annotations

import argparse
import json

from pacbayes_tsk.data.energy_neso_v6 import (
    prepare_neso_daily_dataset,
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
    args = parser.parse_args()

    audit = prepare_neso_daily_dataset(
        args.raw,
        args.processed,
        args.audit,
    )
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
