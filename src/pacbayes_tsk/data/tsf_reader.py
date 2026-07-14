from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class TSFFormatError(ValueError):
    """Raised when a TSF file is malformed or unsupported."""


@dataclass(frozen=True)
class TSFMetadata:
    frequency: str | None
    horizon: int | None
    contains_missing: bool | None
    equal_length: bool | None
    attributes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TSFRecord:
    attributes: dict[str, Any]
    values: np.ndarray


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise TSFFormatError(f"Invalid Boolean value: {value!r}")


def _parse_attribute_value(value: str, attribute_type: str) -> Any:
    kind = attribute_type.lower()
    if kind == "string":
        return value
    if kind == "numeric":
        number = float(value)
        return int(number) if number.is_integer() else number
    if kind == "date":
        try:
            return datetime.strptime(value, "%Y-%m-%d %H-%M-%S")
        except ValueError as exc:
            raise TSFFormatError(
                f"Invalid TSF date {value!r}; expected YYYY-MM-DD HH-MM-SS."
            ) from exc
    raise TSFFormatError(f"Unsupported attribute type: {attribute_type!r}")


def read_tsf(path: str | Path) -> tuple[list[TSFRecord], TSFMetadata]:
    """
    Parse a Monash TSF file.

    Missing observations represented by '?' are returned as np.nan.
    The implementation follows the public Monash TSF specification and
    supports string, numeric, and date attributes.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    attribute_names: list[str] = []
    attribute_types: list[str] = []
    records: list[TSFRecord] = []

    frequency: str | None = None
    horizon: int | None = None
    contains_missing: bool | None = None
    equal_length: bool | None = None

    found_data_tag = False

    with input_path.open("r", encoding="cp1252") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("@"):
                lower = line.lower()

                if lower == "@data":
                    if not attribute_names:
                        raise TSFFormatError(
                            "@data encountered before any @attribute declaration."
                        )
                    found_data_tag = True
                    continue

                parts = line.split()
                if lower.startswith("@attribute"):
                    if len(parts) != 3:
                        raise TSFFormatError(
                            f"Invalid @attribute declaration at line {line_number}."
                        )
                    attribute_names.append(parts[1])
                    attribute_types.append(parts[2])
                    continue

                if len(parts) != 2:
                    raise TSFFormatError(
                        f"Invalid metadata declaration at line {line_number}."
                    )

                if lower.startswith("@frequency"):
                    frequency = parts[1]
                elif lower.startswith("@horizon"):
                    horizon = int(parts[1])
                elif lower.startswith("@missing"):
                    contains_missing = _parse_bool(parts[1])
                elif lower.startswith("@equallength"):
                    equal_length = _parse_bool(parts[1])
                else:
                    # Unknown metadata are allowed by the TSF format.
                    continue
                continue

            if not found_data_tag:
                raise TSFFormatError(
                    f"Data row encountered before @data at line {line_number}."
                )

            fields = line.split(":", len(attribute_names))
            if len(fields) != len(attribute_names) + 1:
                raise TSFFormatError(
                    f"Expected {len(attribute_names)} attributes plus a series "
                    f"at line {line_number}; received {len(fields)} fields."
                )

            attributes = {
                name: _parse_attribute_value(value, kind)
                for name, kind, value in zip(
                    attribute_names,
                    attribute_types,
                    fields[:-1],
                )
            }

            value_tokens = fields[-1].split(",")
            if not value_tokens:
                raise TSFFormatError(
                    f"Empty time series at line {line_number}."
                )

            values = np.asarray(
                [
                    np.nan if token.strip() == "?" else float(token)
                    for token in value_tokens
                ],
                dtype=np.float64,
            )

            if np.isnan(values).all():
                raise TSFFormatError(
                    f"All observations are missing at line {line_number}."
                )

            records.append(TSFRecord(attributes=attributes, values=values))

    if not attribute_names:
        raise TSFFormatError("No @attribute declarations were found.")
    if not found_data_tag:
        raise TSFFormatError("Missing @data declaration.")
    if not records:
        raise TSFFormatError("No series records were found.")

    metadata = TSFMetadata(
        frequency=frequency,
        horizon=horizon,
        contains_missing=contains_missing,
        equal_length=equal_length,
        attributes=tuple(zip(attribute_names, attribute_types)),
    )
    return records, metadata


def infer_series_id(record: TSFRecord, index: int) -> str:
    """Return a stable identifier from common TSF attribute names."""
    candidates = (
        "series_name",
        "series_id",
        "item_id",
        "id",
        "name",
    )
    lowered = {str(key).lower(): key for key in record.attributes}
    for candidate in candidates:
        original_key = lowered.get(candidate)
        if original_key is not None:
            value = str(record.attributes[original_key]).strip()
            if value:
                return value
    return f"series_{index:04d}"


def infer_record_horizon(
    record: TSFRecord,
    metadata: TSFMetadata,
    fallback: int | None = None,
) -> int:
    """Prefer a per-series horizon, then dataset metadata, then a fallback."""
    lowered = {str(key).lower(): key for key in record.attributes}
    for candidate in ("horizon", "forecast_horizon", "prediction_length"):
        original_key = lowered.get(candidate)
        if original_key is not None:
            value = int(record.attributes[original_key])
            if value <= 0:
                raise TSFFormatError(
                    f"Non-positive per-series horizon: {value}."
                )
            return value

    if metadata.horizon is not None:
        if metadata.horizon <= 0:
            raise TSFFormatError(
                f"Non-positive dataset horizon: {metadata.horizon}."
            )
        return int(metadata.horizon)

    if fallback is not None and fallback > 0:
        return int(fallback)

    raise TSFFormatError("No valid forecast horizon is available.")
