"""Loading and deterministic hourly aggregation for the Tetouan energy case study."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

RAW_ZONE_COLUMNS = (
    "Zone 1 Power Consumption",
    "Zone 2  Power Consumption",
    "Zone 3  Power Consumption",
)
CLEAN_ZONE_COLUMNS = ("zone_1", "zone_2", "zone_3")
EXPECTED_RAW_ROWS = 52_416
EXPECTED_HOURLY_ROWS = 8_736
EXPECTED_TWO_HOUR_ROWS = 4_368


@dataclass(frozen=True)
class TetouanDataAudit:
    raw_rows: int
    processed_rows: int
    start: str
    end: str
    raw_sha256: str
    processed_sha256: str | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_tetouan(path: str | Path) -> pd.DataFrame:
    """Load the UCI Tetouan CSV and enforce its frozen schema and chronology."""
    frame = pd.read_csv(path)
    expected = {"DateTime", *RAW_ZONE_COLUMNS}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Tetouan file is missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_RAW_ROWS:
        raise ValueError(
            f"Tetouan raw row count is {len(frame)}; expected {EXPECTED_RAW_ROWS}."
        )
    timestamps = pd.to_datetime(
        frame["DateTime"], format="%m/%d/%Y %H:%M", errors="raise"
    )
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Tetouan timestamps must be unique and chronological.")
    differences = timestamps.diff().dropna()
    if not (differences == pd.Timedelta(minutes=10)).all():
        raise ValueError("Tetouan observations are not uniformly ten minutes apart.")
    output = frame.loc[:, ["DateTime", *RAW_ZONE_COLUMNS]].copy()
    output["DateTime"] = timestamps
    for column in RAW_ZONE_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="raise")
        values = output[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"Invalid power-consumption values in {column}.")
    return output


def aggregate_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate six ten-minute readings into one chronological hourly mean."""
    frame = raw.copy().set_index("DateTime")
    counts = frame[RAW_ZONE_COLUMNS[0]].resample("1h").count()
    if not (counts == 6).all():
        raise ValueError("Each hourly bin must contain exactly six observations.")
    hourly = frame.loc[:, list(RAW_ZONE_COLUMNS)].resample("1h").mean()
    hourly.columns = list(CLEAN_ZONE_COLUMNS)
    hourly = hourly.reset_index().rename(columns={"DateTime": "timestamp"})
    if len(hourly) != EXPECTED_HOURLY_ROWS:
        raise ValueError(
            f"Tetouan hourly row count is {len(hourly)}; expected {EXPECTED_HOURLY_ROWS}."
        )
    if hourly.isna().any().any():
        raise ValueError("Hourly Tetouan aggregation contains missing values.")
    return hourly


def aggregate_two_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate twelve ten-minute readings into one two-hour mean."""
    frame = raw.copy().set_index("DateTime")
    counts = frame[RAW_ZONE_COLUMNS[0]].resample("2h").count()
    if not (counts == 12).all():
        raise ValueError("Each two-hour bin must contain exactly twelve observations.")
    aggregated = frame.loc[:, list(RAW_ZONE_COLUMNS)].resample("2h").mean()
    aggregated.columns = list(CLEAN_ZONE_COLUMNS)
    aggregated = aggregated.reset_index().rename(columns={"DateTime": "timestamp"})
    if len(aggregated) != EXPECTED_TWO_HOUR_ROWS:
        raise ValueError(
            f"Tetouan two-hour row count is {len(aggregated)}; expected {EXPECTED_TWO_HOUR_ROWS}."
        )
    if aggregated.isna().any().any():
        raise ValueError("Two-hour Tetouan aggregation contains missing values.")
    return aggregated


def prepare_hourly_dataset(
    raw_path: str | Path, processed_path: str | Path
) -> TetouanDataAudit:
    raw = load_raw_tetouan(raw_path)
    hourly = aggregate_hourly(raw)
    output = Path(processed_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output, index=False, float_format="%.10f")
    return TetouanDataAudit(
        raw_rows=len(raw),
        processed_rows=len(hourly),
        start=hourly["timestamp"].iloc[0].isoformat(),
        end=hourly["timestamp"].iloc[-1].isoformat(),
        raw_sha256=sha256_file(raw_path),
        processed_sha256=sha256_file(output),
    )


def load_hourly_tetouan(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", *CLEAN_ZONE_COLUMNS}
    if set(frame.columns) != required:
        raise ValueError(
            f"Unexpected hourly Tetouan schema: {frame.columns.tolist()}"
        )
    if len(frame) != EXPECTED_HOURLY_ROWS:
        raise ValueError("Unexpected hourly Tetouan row count.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Hourly Tetouan timestamps are not chronological.")
    for column in CLEAN_ZONE_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"Invalid hourly values in {column}.")
    return frame


def prepare_two_hour_dataset(
    raw_path: str | Path, processed_path: str | Path
) -> TetouanDataAudit:
    raw = load_raw_tetouan(raw_path)
    aggregated = aggregate_two_hourly(raw)
    output = Path(processed_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output, index=False, float_format="%.10f")
    return TetouanDataAudit(
        raw_rows=len(raw),
        processed_rows=len(aggregated),
        start=aggregated["timestamp"].iloc[0].isoformat(),
        end=aggregated["timestamp"].iloc[-1].isoformat(),
        raw_sha256=sha256_file(raw_path),
        processed_sha256=sha256_file(output),
    )


def load_two_hour_tetouan(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", *CLEAN_ZONE_COLUMNS}
    if set(frame.columns) != required:
        raise ValueError(
            f"Unexpected two-hour Tetouan schema: {frame.columns.tolist()}"
        )
    if len(frame) != EXPECTED_TWO_HOUR_ROWS:
        raise ValueError("Unexpected two-hour Tetouan row count.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Two-hour Tetouan timestamps are not chronological.")
    if not (frame["timestamp"].diff().dropna() == pd.Timedelta(hours=2)).all():
        raise ValueError("Tetouan case timestamps are not two hours apart.")
    for column in CLEAN_ZONE_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"Invalid two-hour values in {column}.")
    return frame
