from __future__ import annotations
import argparse, datetime as dt, json
from pathlib import Path
from pacbayes_tsk.data.energy_tetouan import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--case-data", required=True)
    parser.add_argument("--experiment-code", required=True)
    parser.add_argument("--model-code", required=True)
    parser.add_argument("--prior-code", required=True)
    parser.add_argument("--certificate-code", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "config": args.config,
        "case_data": args.case_data,
        "experiment_code": args.experiment_code,
        "model_code": args.model_code,
        "prior_code": args.prior_code,
        "certificate_code": args.certificate_code,
    }
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Energy-case lock already exists; refusing to overwrite it.")
    payload = {
        "schema_version": "3.2",
        "status": "pre_outcome_lock",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "single_authorized_run_only": True,
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "paths": {name: str(Path(path)) for name, path in paths.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
