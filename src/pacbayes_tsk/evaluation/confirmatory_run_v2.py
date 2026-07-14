from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


class ConfirmatoryRunV2Error(RuntimeError):
    """Raised when the one-time confirmatory protocol is violated."""


SERIES_KEY = ["source", "dataset", "series_id"]
REAL_KEY = ["dataset", "series_id"]


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfirmatoryRunV2Error(
            f"JSON root must be an object: {path}"
        )
    return value


def resolve_path(
    project_root: Path,
    value: str | Path,
) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def verify_hashed_artifacts(
    project_root: Path,
    artifacts: Iterable[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for artifact in artifacts:
        project_path = resolve_path(
            project_root,
            artifact["project_path"],
        )
        if not project_path.exists():
            errors.append(
                f"Missing frozen artifact: {project_path}"
            )
            continue
        actual = sha256_file(project_path)
        if actual != str(artifact["sha256"]):
            errors.append(
                f"Modified frozen artifact: {project_path}"
            )
    return errors


def verify_temporal_freeze(
    project_root: Path,
    freeze_directory: str | Path,
) -> dict[str, Any]:
    freeze_dir = resolve_path(
        project_root,
        freeze_directory,
    )
    manifest_path = freeze_dir / "freeze_manifest.json"
    marker_path = freeze_dir / "FROZEN.lock"
    if not manifest_path.exists() or not marker_path.exists():
        raise ConfirmatoryRunV2Error(
            "temporal_pac_bayes_v2 freeze is incomplete."
        )

    manifest = load_json(manifest_path)
    errors = verify_hashed_artifacts(
        project_root,
        manifest.get("artifacts", []),
    )
    if errors:
        raise ConfirmatoryRunV2Error(
            "Temporal freeze verification failed:\n"
            + "\n".join(errors[:30])
        )
    if manifest.get("freeze_name") != "temporal_pac_bayes_v2":
        raise ConfirmatoryRunV2Error(
            "Unexpected temporal freeze name."
        )
    return manifest


def verify_protocol(
    project_root: Path,
    protocol_directory: str | Path,
    *,
    allowed_states: set[str],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
]:
    protocol_dir = resolve_path(
        project_root,
        protocol_directory,
    )
    lock_path = (
        protocol_dir
        / "confirmatory_protocol.lock.json"
    )
    state_path = protocol_dir / "execution_state.json"
    manifest_path = (
        protocol_dir
        / "confirmatory_series_manifest.csv"
    )
    for path in (lock_path, state_path, manifest_path):
        if not path.exists():
            raise ConfirmatoryRunV2Error(
                f"Confirmatory protocol artifact missing: {path}"
            )

    lock = load_json(lock_path)
    state = load_json(state_path)
    current_state = str(state.get("state", ""))
    if current_state not in allowed_states:
        raise ConfirmatoryRunV2Error(
            f"Protocol state {current_state!r} is not allowed; "
            f"expected one of {sorted(allowed_states)}."
        )

    if sha256_file(manifest_path) != str(
        lock["series_manifest_sha256"]
    ):
        raise ConfirmatoryRunV2Error(
            "Confirmatory series manifest hash mismatch."
        )

    frame = pd.read_csv(manifest_path)
    required = {"source", "dataset", "series_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ConfirmatoryRunV2Error(
            "Confirmatory manifest is missing columns: "
            f"{sorted(missing)}."
        )
    if len(frame) != 210:
        raise ConfirmatoryRunV2Error(
            f"Expected 210 confirmatory series; found {len(frame)}."
        )
    if frame.duplicated(SERIES_KEY).any():
        raise ConfirmatoryRunV2Error(
            "Confirmatory manifest contains duplicate composite keys."
        )

    synthetic = frame.loc[
        frame["source"].astype(str).eq("synthetic")
    ].copy()
    real = frame.loc[
        frame["source"].astype(str).eq("real")
    ].copy()
    if len(synthetic) != 120 or len(real) != 90:
        raise ConfirmatoryRunV2Error(
            "Confirmatory source counts must be 120 synthetic and 90 real."
        )

    expected_regimes = {
        "ar",
        "setar",
        "narma",
        "mackey_glass",
        "garch",
        "structural_break",
    }
    observed_regimes = set(
        synthetic["dataset"].astype(str)
    )
    if observed_regimes != expected_regimes:
        raise ConfirmatoryRunV2Error(
            "Synthetic regime set does not match the locked protocol."
        )
    regime_counts = (
        synthetic.groupby("dataset")["series_id"].nunique()
    )
    if not (regime_counts == 20).all():
        raise ConfirmatoryRunV2Error(
            "Every synthetic regime must contain 20 series."
        )
    if "seed" not in synthetic.columns:
        raise ConfirmatoryRunV2Error(
            "Synthetic confirmatory manifest has no seed column."
        )
    for dataset, group in synthetic.groupby("dataset"):
        seeds = sorted(
            pd.to_numeric(
                group["seed"],
                errors="raise",
            ).astype(int).tolist()
        )
        if seeds != list(range(2000, 2020)):
            raise ConfirmatoryRunV2Error(
                f"{dataset}: synthetic seeds are not 2000 through 2019."
            )

    expected_real = {
        "m4_hourly",
        "nn5_daily",
        "tourism_monthly",
    }
    observed_real = set(real["dataset"].astype(str))
    if observed_real != expected_real:
        raise ConfirmatoryRunV2Error(
            "Real dataset set does not match the locked protocol."
        )
    real_counts = real.groupby("dataset")["series_id"].nunique()
    if not (real_counts == 30).all():
        raise ConfirmatoryRunV2Error(
            "Every real dataset must contain 30 unseen series."
        )

    return lock, state, frame


def verify_no_real_overlap(
    confirmatory_manifest: pd.DataFrame,
    development_manifest: pd.DataFrame,
) -> None:
    real = confirmatory_manifest.loc[
        confirmatory_manifest["source"].astype(str).eq("real"),
        REAL_KEY,
    ].copy()
    development = development_manifest[REAL_KEY].copy()
    for frame in (real, development):
        frame["dataset"] = frame["dataset"].astype(str)
        frame["series_id"] = frame["series_id"].astype(str)
    overlap = real.merge(
        development.drop_duplicates(),
        on=REAL_KEY,
        how="inner",
    )
    if not overlap.empty:
        raise ConfirmatoryRunV2Error(
            "Confirmatory real series overlap development series:\n"
            + overlap.head(20).to_string(index=False)
        )


def create_exclusive_run_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            str(path),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as error:
        raise ConfirmatoryRunV2Error(
            "The one-time confirmatory run lock already exists."
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            "One-time confirmatory execution lock.\n"
            f"Created UTC: {utc_now()}\n"
        )


def transition_state(
    state_path: Path,
    *,
    expected_state: str,
    new_state: str,
    updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = load_json(state_path)
    current = str(state.get("state", ""))
    if current != expected_state:
        raise ConfirmatoryRunV2Error(
            f"Cannot transition {current!r} to {new_state!r}; "
            f"expected {expected_state!r}."
        )
    state["state"] = new_state
    state["updated_utc"] = utc_now()
    if updates:
        state.update(updates)
    atomic_write_json(state_path, state)
    return state


def normalized_stage_plan(
    python_executable: str,
    plan: list[dict[str, Any]],
) -> list[Stage]:
    stages: list[Stage] = []
    names: set[str] = set()
    for item in plan:
        name = str(item["name"])
        if name in names:
            raise ConfirmatoryRunV2Error(
                f"Duplicate stage name: {name}"
            )
        names.add(name)
        raw = [str(value) for value in item["command"]]
        command = tuple(
            python_executable if value == "{python}" else value
            for value in raw
        )
        if not command:
            raise ConfirmatoryRunV2Error(
                f"Stage {name} has an empty command."
            )
        stages.append(Stage(name=name, command=command))
    return stages


def command_plan_hash(stages: Iterable[Stage]) -> str:
    payload = [
        {"name": stage.name, "command": list(stage.command)}
        for stage in stages
    ]
    return sha256_json(payload)


def run_stage(
    stage: Stage,
    *,
    project_root: Path,
    log_directory: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    log_directory.mkdir(parents=True, exist_ok=True)
    stdout_path = log_directory / f"{stage.name}.stdout.log"
    stderr_path = log_directory / f"{stage.name}.stderr.log"
    started = utc_now()
    result = subprocess.run(
        list(stage.command),
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
    )
    stdout_path.write_text(
        result.stdout,
        encoding="utf-8",
    )
    stderr_path.write_text(
        result.stderr,
        encoding="utf-8",
    )
    record = {
        "name": stage.name,
        "command": list(stage.command),
        "started_utc": started,
        "completed_utc": utc_now(),
        "return_code": int(result.returncode),
        "stdout_path": str(
            stdout_path.relative_to(project_root)
        ).replace("\\", "/"),
        "stderr_path": str(
            stderr_path.relative_to(project_root)
        ).replace("\\", "/"),
    }
    if result.returncode != 0:
        tail = "\n".join(
            result.stderr.splitlines()[-20:]
        )
        raise ConfirmatoryRunV2Error(
            f"Stage {stage.name!r} failed with code "
            f"{result.returncode}.\n{tail}"
        )
    return record


def assert_fresh_outputs(
    project_root: Path,
    paths: Iterable[str | Path],
) -> None:
    existing = [
        resolve_path(project_root, value)
        for value in paths
        if resolve_path(project_root, value).exists()
    ]
    if existing:
        raise ConfirmatoryRunV2Error(
            "Confirmatory outputs already exist; one-time execution "
            "requires a fresh output namespace:\n"
            + "\n".join(str(path) for path in existing[:30])
        )


def finite_metric_audit(
    frame: pd.DataFrame,
    *,
    metrics: Iterable[str],
    table_name: str,
) -> list[str]:
    errors: list[str] = []
    for metric in metrics:
        if metric not in frame.columns:
            errors.append(
                f"{table_name}: missing metric column {metric}."
            )
            continue
        values = pd.to_numeric(
            frame[metric],
            errors="coerce",
        ).to_numpy(float)
        if not np.all(np.isfinite(values)):
            errors.append(
                f"{table_name}: non-finite values in {metric}."
            )
    return errors


def validate_predictive_table(
    frame: pd.DataFrame,
    *,
    expected_rows: int,
    table_name: str,
) -> list[str]:
    errors: list[str] = []
    if len(frame) != expected_rows:
        errors.append(
            f"{table_name}: expected {expected_rows} rows; found {len(frame)}."
        )
    if "status" not in frame.columns:
        errors.append(f"{table_name}: missing status column.")
        return errors
    failures = int(
        (~frame["status"].astype(str).eq("PASS")).sum()
    )
    if failures:
        errors.append(
            f"{table_name}: contains {failures} failed rows."
        )
    errors.extend(
        finite_metric_audit(
            frame.loc[
                frame["status"].astype(str).eq("PASS")
            ],
            metrics=["mae", "rmse", "mase", "smape"],
            table_name=table_name,
        )
    )
    return errors


def average_ranks(
    predictive: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "source",
        "dataset",
        "series_id",
        "model",
        "rmse",
    }
    missing = required.difference(predictive.columns)
    if missing:
        raise ConfirmatoryRunV2Error(
            "Predictive table is missing ranking columns: "
            f"{sorted(missing)}."
        )
    frame = predictive.copy()
    frame["rank"] = frame.groupby(
        SERIES_KEY,
        sort=False,
    )["rmse"].rank(
        method="average",
        ascending=True,
    )
    return (
        frame.groupby(
            ["source", "dataset", "model"],
            as_index=False,
        )
        .agg(
            series=("series_id", "nunique"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            mean_mase=("mase", "mean"),
            average_rank=("rank", "mean"),
        )
        .sort_values(
            ["source", "dataset", "average_rank", "model"]
        )
        .reset_index(drop=True)
    )


def collect_output_hashes(
    project_root: Path,
    paths: Iterable[str | Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in paths:
        path = resolve_path(project_root, value)
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(
                    path.relative_to(project_root)
                ).replace("\\", "/"),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows
