from __future__ import annotations
import argparse, datetime as dt, json
from pathlib import Path
from pacbayes_tsk.data.energy_pjm import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--case-data", required=True)
    parser.add_argument("--data-audit", required=True)
    parser.add_argument("--protocol-doc", required=True)
    parser.add_argument("--experiment-code", required=True)
    parser.add_argument("--data-code", required=True)
    parser.add_argument("--model-code", required=True)
    parser.add_argument("--shared-energy-code", required=True)
    parser.add_argument("--prior-code", required=True)
    parser.add_argument("--certificate-code", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "config": args.config,
        "case_data": args.case_data,
        "data_audit": args.data_audit,
        "protocol_doc": args.protocol_doc,
        "experiment_code": args.experiment_code,
        "data_code": args.data_code,
        "model_code": args.model_code,
        "shared_energy_code": args.shared_energy_code,
        "prior_code": args.prior_code,
        "certificate_code": args.certificate_code,
    }
    output = Path(args.output)
    if output.exists():
        raise SystemExit("PJM confirmatory lock already exists; refusing overwrite.")
    payload = {
        "schema_version": "3.4",
        "status": "pre_outcome_lock",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "single_authorized_run_only": True,
        "retuning_after_outcome": "forbidden",
        "rerun_after_outcome": "forbidden",
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "paths": {name: str(Path(path)) for name, path in paths.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
