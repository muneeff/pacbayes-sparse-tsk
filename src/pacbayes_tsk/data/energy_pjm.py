"""Deterministic preparation of the independent PJM confirmatory energy case.

The source consists of four predeclared hourly regional load series from the
CC0 Kaggle "Hourly Energy Consumption" snapshot. The execution bundle uses a
public GitHub mirror only as a transport layer. Raw file hashes are frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

REGION_SPECS = {
    "aep": ("AEP_hourly.csv", "AEP_MW"),
    "comed": ("COMED_hourly.csv", "COMED_MW"),
    "dayton": ("DAYTON_hourly.csv", "DAYTON_MW"),
    "pjme": ("PJME_hourly.csv", "PJME_MW"),
}
REGION_COLUMNS = tuple(REGION_SPECS)
COMMON_START = pd.Timestamp("2012-01-01 00:00:00")
COMMON_END_EXCLUSIVE = pd.Timestamp("2018-08-03 00:00:00")
EXPECTED_DAILY_ROWS = 2406


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PJMPreparationAudit:
    source_files: dict[str, str]
    source_sha256: dict[str, str]
    source_rows: dict[str, int]
    duplicate_timestamps_collapsed: dict[str, int]
    common_start: str
    common_end: str
    processed_rows: int
    minimum_hourly_observations_per_day: dict[str, int]
    processed_sha256: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _load_region(path: Path, value_column: str) -> tuple[pd.Series, dict[str, int]]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["Datetime", value_column]:
        raise ValueError(f"Unexpected schema for {path.name}: {frame.columns.tolist()}")
    timestamp = pd.to_datetime(frame["Datetime"], errors="raise")
    value = pd.to_numeric(frame[value_column], errors="raise")
    if value.isna().any() or not np.isfinite(value.to_numpy(float)).all():
        raise ValueError(f"Non-finite values in {path.name}.")
    if (value <= 0).any():
        raise ValueError(f"Non-positive load values in {path.name}.")
    loaded = pd.DataFrame({"timestamp": timestamp, "value": value})
    duplicates = int(loaded["timestamp"].duplicated(keep=False).sum())
    loaded = (
        loaded.groupby("timestamp", as_index=True, sort=True)["value"]
        .mean()
        .sort_index()
    )
    loaded = loaded.loc[(loaded.index >= COMMON_START) & (loaded.index < COMMON_END_EXCLUSIVE)]
    if loaded.empty:
        raise ValueError(f"No observations in the frozen common period for {path.name}.")
    return loaded, {"raw_rows": int(len(frame)), "duplicate_rows": duplicates}


def prepare_pjm_daily_dataset(
    raw_dir: str | Path,
    processed_path: str | Path,
    audit_path: str | Path | None = None,
) -> PJMPreparationAudit:
    raw_root = Path(raw_dir)
    daily_columns: dict[str, pd.Series] = {}
    source_files: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    source_rows: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    minimum_counts: dict[str, int] = {}

    for region, (filename, value_column) in REGION_SPECS.items():
        path = raw_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        series, meta = _load_region(path, value_column)
        daily_count = series.resample("D").count()
        daily_mean = series.resample("D").mean()
        if len(daily_mean) != EXPECTED_DAILY_ROWS:
            raise ValueError(
                f"{region} produced {len(daily_mean)} daily rows; expected {EXPECTED_DAILY_ROWS}."
            )
        if int(daily_count.min()) < 23:
            raise ValueError(f"{region} has a day with fewer than 23 hourly values.")
        if daily_mean.isna().any():
            raise ValueError(f"{region} daily aggregation contains missing values.")
        daily_columns[region] = daily_mean
        source_files[region] = filename
        source_hashes[region] = sha256_file(path)
        source_rows[region] = meta["raw_rows"]
        duplicate_counts[region] = meta["duplicate_rows"]
        minimum_counts[region] = int(daily_count.min())

    combined = pd.DataFrame(daily_columns)
    combined.index.name = "timestamp"
    combined = combined.reset_index()
    if tuple(combined.columns[1:]) != REGION_COLUMNS:
        raise RuntimeError("PJM region column order changed unexpectedly.")
    if not combined["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Processed PJM timestamps are not chronological.")
    if not (combined["timestamp"].diff().dropna() == pd.Timedelta(days=1)).all():
        raise RuntimeError("Processed PJM data are not daily and contiguous.")
    if combined.isna().any().any():
        raise RuntimeError("Processed PJM data contain missing values.")

    output = Path(processed_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, float_format="%.10f")
    audit = PJMPreparationAudit(
        source_files=source_files,
        source_sha256=source_hashes,
        source_rows=source_rows,
        duplicate_timestamps_collapsed=duplicate_counts,
        common_start=combined["timestamp"].iloc[0].isoformat(),
        common_end=combined["timestamp"].iloc[-1].isoformat(),
        processed_rows=int(len(combined)),
        minimum_hourly_observations_per_day=minimum_counts,
        processed_sha256=sha256_file(output),
    )
    if audit_path is not None:
        audit_output = Path(audit_path)
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit_output.write_text(json.dumps(audit.to_dict(), indent=2), encoding="utf-8")
    return audit


def load_pjm_daily(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    expected = ("timestamp", *REGION_COLUMNS)
    if tuple(frame.columns) != expected:
        raise ValueError(f"Unexpected PJM processed schema: {frame.columns.tolist()}")
    if len(frame) != EXPECTED_DAILY_ROWS:
        raise ValueError(f"Unexpected PJM daily row count: {len(frame)}")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("PJM timestamps are not chronological.")
    if not (frame["timestamp"].diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("PJM timestamps are not daily and contiguous.")
    values = frame.loc[:, REGION_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("PJM processed loads must be finite and positive.")
    return frame
