from __future__ import annotations
import argparse, json
from pathlib import Path
from pacbayes_tsk.experiments.pjm_confirmatory_case_v3 import run_pjm_confirmatory_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    verification = {
        "config": Path(args.config),
        "case_data": Path(args.case_data),
        "data_audit": root / "artifacts/pjm_data_preparation_audit.json",
        "protocol_doc": root / "protocol/PJM_CONFIRMATORY_CASE_V3_PROTOCOL.md",
        "experiment_code": root / "src/pacbayes_tsk/experiments/pjm_confirmatory_case_v3.py",
        "data_code": root / "src/pacbayes_tsk/data/energy_pjm.py",
        "model_code": root / "src/pacbayes_tsk/models/sparse_tsk.py",
        "shared_energy_code": root / "src/pacbayes_tsk/experiments/energy_case_study_v3.py",
        "prior_code": root / "src/pacbayes_tsk/pac_bayes/priors_v3.py",
        "certificate_code": root / "src/pacbayes_tsk/pac_bayes/certificates_v3.py",
    }
    result = run_pjm_confirmatory_case(
        case_data_path=args.case_data,
        config_path=args.config,
        lock_path=args.lock,
        output_dir=args.output,
        verification_paths=verification,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
