from __future__ import annotations
import argparse, json
from pacbayes_tsk.data.energy_pjm import prepare_pjm_daily_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    audit = prepare_pjm_daily_dataset(args.raw_dir, args.processed, args.audit)
    print(json.dumps(audit.to_dict(), indent=2))


if __name__ == "__main__":
    main()
