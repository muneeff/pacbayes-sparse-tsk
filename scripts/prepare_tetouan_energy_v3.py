from __future__ import annotations
import argparse, json
from pathlib import Path
from pacbayes_tsk.data.energy_tetouan import prepare_two_hour_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    audit = prepare_two_hour_dataset(args.raw, args.processed)
    Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit).write_text(json.dumps(audit.__dict__, indent=2), encoding="utf-8")
    print(json.dumps(audit.__dict__, indent=2))


if __name__ == "__main__":
    main()
