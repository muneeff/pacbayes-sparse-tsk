"""Deterministic NESO 2025 preparation for the prospective V6 case."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_DATE_COLUMN = "SETTLEMENT_DATE"
RAW_PERIOD_COLUMN = "SETTLEMENT_PERIOD"
RAW_TARGET_COLUMN = "ND"
EXPECTED_START = pd.Timestamp("2025-01-01")
EXPECTED_END = pd.Timestamp("2025-12-31")
EXPECTED_DAILY_ROWS = 365
ALLOWED_PERIOD_COUNTS = frozenset({46, 48, 50})
PROCESSED_COLUMNS = ("timestamp", "nd_mw")


class NESODataV6Error(ValueError):
    """Raised when the frozen NESO V6 data contract is violated."""


@dataclass(frozen=True)
class NESOPreparationAuditV6:
    raw_file: str
    raw_sha256: str
    raw_rows: int
    raw_columns: tuple[str, ...]
    duplicate_keys: int
    settlement_period_count_distribution: dict[str, int]
    processed_rows: int
    start: str
    end: str
    processed_file: str
    processed_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [
        str(column).lstrip("\ufeff").strip()
        for column in output.columns
    ]
    if len(set(output.columns)) != len(output.columns):
        raise NESODataV6Error(
            "NESO column normalization produced duplicate names."
        )
    return output


def load_raw_neso_2025(path: str | Path) -> pd.DataFrame:
    """Load and validate the frozen half-hourly NESO 2025 source."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    frame = _clean_columns(pd.read_csv(input_path))
    required = {
        RAW_DATE_COLUMN,
        RAW_PERIOD_COLUMN,
        RAW_TARGET_COLUMN,
    }
    missing = required - set(frame.columns)
    if missing:
        raise NESODataV6Error(
            f"NESO source is missing columns: {sorted(missing)}"
        )
    if frame.empty:
        raise NESODataV6Error("NESO source is empty.")

    dates = pd.to_datetime(
        frame[RAW_DATE_COLUMN],
        dayfirst=True,
        errors="raise",
    )
    if dates.isna().any():
        raise NESODataV6Error(
            "NESO settlement dates contain missing values."
        )
    normalized_dates = dates.dt.normalize()
    if not dates.equals(normalized_dates):
        raise NESODataV6Error(
            "SETTLEMENT_DATE must not contain time-of-day values."
        )

    periods_numeric = pd.to_numeric(
        frame[RAW_PERIOD_COLUMN],
        errors="raise",
    )
    periods_array = periods_numeric.to_numpy(dtype=np.float64)
    if (
        not np.all(np.isfinite(periods_array))
        or not np.all(periods_array == np.floor(periods_array))
    ):
        raise NESODataV6Error(
            "SETTLEMENT_PERIOD must contain finite integers."
        )
    periods = periods_numeric.astype(np.int64)
    if (periods < 1).any():
        raise NESODataV6Error(
            "SETTLEMENT_PERIOD must be strictly positive."
        )

    target_numeric = pd.to_numeric(
        frame[RAW_TARGET_COLUMN],
        errors="raise",
    )
    target = target_numeric.to_numpy(dtype=np.float64)
    if (
        target_numeric.isna().any()
        or not np.all(np.isfinite(target))
    ):
        raise NESODataV6Error(
            "ND contains missing or non-finite values."
        )

    output = pd.DataFrame(
        {
            "settlement_date": normalized_dates,
            "settlement_period": periods,
            "nd_mw": target,
        }
    )
    duplicate_count = int(
        output.duplicated(
            ["settlement_date", "settlement_period"],
            keep=False,
        ).sum()
    )
    if duplicate_count:
        raise NESODataV6Error(
            "Duplicate (SETTLEMENT_DATE, SETTLEMENT_PERIOD) keys "
            f"were found: {duplicate_count} rows."
        )

    output.sort_values(
        ["settlement_date", "settlement_period"],
        kind="stable",
        inplace=True,
        ignore_index=True,
    )
    observed_dates = pd.DatetimeIndex(
        output["settlement_date"].drop_duplicates()
    )
    expected_dates = pd.date_range(
        EXPECTED_START,
        EXPECTED_END,
        freq="D",
    )
    if not observed_dates.equals(expected_dates):
        missing_dates = expected_dates.difference(observed_dates)
        extra_dates = observed_dates.difference(expected_dates)
        raise NESODataV6Error(
            "NESO calendar coverage differs from frozen 2025 coverage; "
            f"missing={list(map(str, missing_dates[:5]))}, "
            f"extra={list(map(str, extra_dates[:5]))}."
        )

    daily_counts = output.groupby(
        "settlement_date",
        sort=True,
    ).size()
    invalid = daily_counts[
        ~daily_counts.isin(ALLOWED_PERIOD_COUNTS)
    ]
    if not invalid.empty:
        preview = {
            timestamp.date().isoformat(): int(count)
            for timestamp, count in invalid.iloc[:10].items()
        }
        raise NESODataV6Error(
            "A settlement day has a count outside {46, 48, 50}: "
            f"{preview}"
        )
    return output


def aggregate_neso_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate validated half-hourly ND observations to daily means."""
    required = {
        "settlement_date",
        "settlement_period",
        "nd_mw",
    }
    if set(raw.columns) != required:
        raise NESODataV6Error(
            f"Unexpected validated raw schema: {raw.columns.tolist()}"
        )

    daily = (
        raw.groupby("settlement_date", sort=True)["nd_mw"]
        .mean()
        .rename("nd_mw")
        .reset_index()
        .rename(columns={"settlement_date": "timestamp"})
    )
    if len(daily) != EXPECTED_DAILY_ROWS:
        raise NESODataV6Error(
            f"Daily aggregation produced {len(daily)} rows; "
            f"expected {EXPECTED_DAILY_ROWS}."
        )
    expected_dates = pd.date_range(
        EXPECTED_START,
        EXPECTED_END,
        freq="D",
    )
    if not pd.DatetimeIndex(daily["timestamp"]).equals(
        expected_dates
    ):
        raise NESODataV6Error(
            "Daily NESO timestamps are not the complete 2025 calendar."
        )
    values = daily["nd_mw"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise NESODataV6Error(
            "Daily NESO aggregation contains non-finite values."
        )
    return daily.loc[:, list(PROCESSED_COLUMNS)]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def prepare_neso_daily_dataset(
    raw_path: str | Path,
    processed_path: str | Path,
    audit_path: str | Path,
) -> NESOPreparationAuditV6:
    """Prepare and atomically publish the frozen daily NESO dataset."""
    raw_input = Path(raw_path)
    output = Path(processed_path)
    audit_output = Path(audit_path)
    for target in (output, audit_output):
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite prospective V6 artifact: {target}"
            )

    raw = load_raw_neso_2025(raw_input)
    daily = aggregate_neso_daily(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    daily.to_csv(
        temporary,
        index=False,
        float_format="%.10f",
        date_format="%Y-%m-%d",
    )
    os.replace(temporary, output)

    counts = (
        raw.groupby("settlement_date").size().value_counts().sort_index()
    )
    audit = NESOPreparationAuditV6(
        raw_file=raw_input.name,
        raw_sha256=sha256_file(raw_input),
        raw_rows=int(len(raw)),
        raw_columns=tuple(
            _clean_columns(pd.read_csv(raw_input, nrows=0)).columns
        ),
        duplicate_keys=0,
        settlement_period_count_distribution={
            str(int(period_count)): int(day_count)
            for period_count, day_count in counts.items()
        },
        processed_rows=int(len(daily)),
        start=daily["timestamp"].iloc[0].date().isoformat(),
        end=daily["timestamp"].iloc[-1].date().isoformat(),
        processed_file=output.name,
        processed_sha256=sha256_file(output),
    )
    try:
        _atomic_json(audit_output, audit.to_dict())
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return audit


def load_neso_daily(path: str | Path) -> pd.DataFrame:
    """Load a prepared prospective V6 daily dataset."""
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    if tuple(frame.columns) != PROCESSED_COLUMNS:
        raise NESODataV6Error(
            f"Unexpected NESO daily schema: {frame.columns.tolist()}"
        )
    if len(frame) != EXPECTED_DAILY_ROWS:
        raise NESODataV6Error(
            f"Unexpected NESO daily row count: {len(frame)}"
        )
    expected_dates = pd.date_range(
        EXPECTED_START,
        EXPECTED_END,
        freq="D",
    )
    if not pd.DatetimeIndex(frame["timestamp"]).equals(
        expected_dates
    ):
        raise NESODataV6Error(
            "Processed NESO timestamps are not the complete 2025 calendar."
        )
    values = frame["nd_mw"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise NESODataV6Error(
            "Processed NESO demand contains non-finite values."
        )
    return frame
