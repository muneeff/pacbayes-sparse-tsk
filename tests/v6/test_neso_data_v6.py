from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pacbayes_tsk.data.energy_neso_v6 import (
    EXPECTED_DAILY_ROWS,
    NESODataV6Error,
    load_neso_daily,
    load_raw_neso_2025,
    prepare_neso_daily_dataset,
    sha256_file,
)


def _raw_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    for day_number, date in enumerate(dates):
        period_count = 48
        if date == pd.Timestamp("2025-03-30"):
            period_count = 46
        elif date == pd.Timestamp("2025-10-26"):
            period_count = 50
        for period in range(1, period_count + 1):
            rows.append(
                {
                    "SETTLEMENT_DATE": date.strftime("%Y-%m-%d"),
                    "SETTLEMENT_PERIOD": period,
                    "ND": (
                        25_000.0
                        + 10.0 * day_number
                        + 0.25 * period
                    ),
                    "TSD": 30_000.0,
                }
            )
    return pd.DataFrame(rows)


def _write_raw(path: Path, frame: pd.DataFrame | None = None) -> Path:
    payload = _raw_frame() if frame is None else frame
    payload.to_csv(path, index=False)
    return path


def test_raw_loader_accepts_frozen_calendar_and_clock_change_counts(
    tmp_path: Path,
) -> None:
    raw = load_raw_neso_2025(_write_raw(tmp_path / "raw.csv"))
    assert raw["settlement_date"].nunique() == EXPECTED_DAILY_ROWS
    counts = raw.groupby("settlement_date").size()
    assert set(counts.unique()) == {46, 48, 50}
    assert not raw.duplicated(
        ["settlement_date", "settlement_period"]
    ).any()


def test_prepare_publishes_daily_data_and_hash_audit(
    tmp_path: Path,
) -> None:
    raw_path = _write_raw(tmp_path / "raw.csv")
    processed = tmp_path / "daily.csv"
    audit_path = tmp_path / "audit.json"

    audit = prepare_neso_daily_dataset(
        raw_path,
        processed,
        audit_path,
    )
    loaded = load_neso_daily(processed)
    stored = json.loads(audit_path.read_text(encoding="utf-8"))

    assert len(loaded) == EXPECTED_DAILY_ROWS
    assert loaded["timestamp"].iloc[0] == pd.Timestamp("2025-01-01")
    assert loaded["timestamp"].iloc[-1] == pd.Timestamp("2025-12-31")
    assert audit.processed_sha256 == sha256_file(processed)
    assert stored["raw_sha256"] == sha256_file(raw_path)
    assert stored["settlement_period_count_distribution"] == {
        "46": 1,
        "48": 363,
        "50": 1,
    }


def test_daily_mean_uses_only_nd_values(tmp_path: Path) -> None:
    frame = _raw_frame()
    first_day = frame["SETTLEMENT_DATE"].eq("2025-01-01")
    expected = float(frame.loc[first_day, "ND"].mean())
    raw_path = _write_raw(tmp_path / "raw.csv", frame)
    processed = tmp_path / "daily.csv"
    prepare_neso_daily_dataset(
        raw_path,
        processed,
        tmp_path / "audit.json",
    )
    loaded = load_neso_daily(processed)
    assert loaded["nd_mw"].iloc[0] == pytest.approx(expected)


def test_duplicate_settlement_key_is_rejected(tmp_path: Path) -> None:
    frame = _raw_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = _write_raw(tmp_path / "duplicate.csv", frame)
    with pytest.raises(NESODataV6Error, match="Duplicate"):
        load_raw_neso_2025(path)


def test_invalid_daily_period_count_is_rejected(tmp_path: Path) -> None:
    frame = _raw_frame()
    mask = (
        frame["SETTLEMENT_DATE"].eq("2025-01-02")
        & frame["SETTLEMENT_PERIOD"].eq(48)
    )
    frame = frame.loc[~mask].copy()
    path = _write_raw(tmp_path / "count.csv", frame)
    with pytest.raises(NESODataV6Error, match="46, 48, 50"):
        load_raw_neso_2025(path)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_missing_or_nonfinite_nd_is_rejected(
    tmp_path: Path,
    bad_value: float,
) -> None:
    frame = _raw_frame()
    frame.loc[0, "ND"] = bad_value
    path = _write_raw(tmp_path / "bad_nd.csv", frame)
    with pytest.raises((NESODataV6Error, ValueError)):
        load_raw_neso_2025(path)


def test_missing_calendar_day_is_rejected(tmp_path: Path) -> None:
    frame = _raw_frame()
    frame = frame[
        ~frame["SETTLEMENT_DATE"].eq("2025-06-15")
    ].copy()
    path = _write_raw(tmp_path / "missing_day.csv", frame)
    with pytest.raises(NESODataV6Error, match="calendar coverage"):
        load_raw_neso_2025(path)


def test_prospective_artifacts_are_never_overwritten(
    tmp_path: Path,
) -> None:
    raw_path = _write_raw(tmp_path / "raw.csv")
    processed = tmp_path / "daily.csv"
    audit = tmp_path / "audit.json"
    processed.write_text("already exists", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing"):
        prepare_neso_daily_dataset(raw_path, processed, audit)

def test_non_iso_date_representation_is_rejected(tmp_path: Path) -> None:
    frame = _raw_frame()
    frame["SETTLEMENT_DATE"] = pd.to_datetime(
        frame["SETTLEMENT_DATE"],
        format="%Y-%m-%d",
        errors="raise",
    ).dt.strftime("%d/%m/%Y")
    path = _write_raw(tmp_path / "non_iso_dates.csv", frame)
    with pytest.raises(ValueError):
        load_raw_neso_2025(path)
