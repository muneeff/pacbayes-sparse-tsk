from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


REQUIRED_PROCESSED_COLUMNS = (
    "series_id",
    "time",
    "value_raw",
    "value_scaled",
    "split",
    "regime",
    "seed",
)

VALID_SPLITS = ("prior", "bound", "validation", "test")
HistoryPolicy = Literal["past_context", "same_split_only"]


class SupervisedDataError(ValueError):
    """Raised when lagged forecasting data cannot be built safely."""


@dataclass(frozen=True)
class LaggedDataset:
    """One supervised forecasting dataset for one series and one lag order."""

    X: np.ndarray
    y_scaled: np.ndarray
    y_raw: np.ndarray
    target_time: np.ndarray
    target_index: np.ndarray
    split: np.ndarray
    series_id: str
    seed: int
    lag_order: int
    feature_names: tuple[str, ...]
    history_policy: str

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def split_counts(self) -> dict[str, int]:
        return {
            split_name: int(np.sum(self.split == split_name))
            for split_name in VALID_SPLITS
        }

    def validate(self) -> None:
        """Validate shape, ordering, finiteness, and no-target-leakage invariants."""
        n_samples = self.n_samples

        if self.X.ndim != 2:
            raise SupervisedDataError("X must be a two-dimensional matrix.")
        if self.n_features != self.lag_order:
            raise SupervisedDataError(
                "The number of X columns must equal lag_order."
            )

        aligned_arrays = {
            "y_scaled": self.y_scaled,
            "y_raw": self.y_raw,
            "target_time": self.target_time,
            "target_index": self.target_index,
            "split": self.split,
        }
        for name, array in aligned_arrays.items():
            if len(array) != n_samples:
                raise SupervisedDataError(
                    f"{name} has {len(array)} rows but X has {n_samples}."
                )

        if n_samples == 0:
            raise SupervisedDataError("The lagged dataset contains no samples.")

        if not np.all(np.isfinite(self.X)):
            raise SupervisedDataError("X contains NaN or infinite values.")
        if not np.all(np.isfinite(self.y_scaled)):
            raise SupervisedDataError("y_scaled contains NaN or infinite values.")
        if not np.all(np.isfinite(self.y_raw)):
            raise SupervisedDataError("y_raw contains NaN or infinite values.")
        if not np.all(np.isfinite(self.target_time.astype(float))):
            raise SupervisedDataError("target_time contains non-finite values.")

        if np.any(np.diff(self.target_index) <= 0):
            raise SupervisedDataError("target_index must be strictly increasing.")
        if np.any(np.diff(self.target_time.astype(float)) <= 0):
            raise SupervisedDataError("target_time must be strictly increasing.")

        observed_splits = set(np.unique(self.split).tolist())
        unknown_splits = observed_splits.difference(VALID_SPLITS)
        if unknown_splits:
            raise SupervisedDataError(
                f"Unknown split labels found: {sorted(unknown_splits)}"
            )

        expected_features = tuple(
            f"lag_{lag}" for lag in range(1, self.lag_order + 1)
        )
        if self.feature_names != expected_features:
            raise SupervisedDataError(
                "feature_names must be ordered as lag_1, lag_2, ..., lag_p."
            )

    def save_npz(self, path: str | Path, *, compressed: bool = True) -> Path:
        """Save all arrays and metadata in a portable NumPy archive."""
        self.validate()

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "X": self.X,
            "y_scaled": self.y_scaled,
            "y_raw": self.y_raw,
            "target_time": self.target_time,
            "target_index": self.target_index,
            "split": self.split.astype("U16"),
            "series_id": np.asarray(self.series_id),
            "seed": np.asarray(self.seed, dtype=np.int64),
            "lag_order": np.asarray(self.lag_order, dtype=np.int64),
            "feature_names": np.asarray(self.feature_names, dtype="U32"),
            "history_policy": np.asarray(self.history_policy),
        }

        save_function = np.savez_compressed if compressed else np.savez
        save_function(output_path, **payload)
        return output_path


def validate_processed_frame(frame: pd.DataFrame) -> None:
    """Validate the processed time-series schema and chronological order."""
    missing_columns = [
        column
        for column in REQUIRED_PROCESSED_COLUMNS
        if column not in frame.columns
    ]
    if missing_columns:
        raise SupervisedDataError(
            f"Processed series is missing columns: {missing_columns}"
        )

    if frame.empty:
        raise SupervisedDataError("Processed series is empty.")

    required = frame.loc[:, REQUIRED_PROCESSED_COLUMNS]
    if required.isna().any().any():
        raise SupervisedDataError("Processed series contains missing values.")

    time_values = pd.to_numeric(frame["time"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(time_values)):
        raise SupervisedDataError("time contains non-finite values.")
    if len(time_values) > 1 and np.any(np.diff(time_values) <= 0):
        raise SupervisedDataError("time must be strictly increasing.")

    for value_column in ("value_raw", "value_scaled"):
        values = pd.to_numeric(
            frame[value_column], errors="coerce"
        ).to_numpy(float)
        if not np.all(np.isfinite(values)):
            raise SupervisedDataError(
                f"{value_column} contains NaN or infinite values."
            )

    if frame["series_id"].nunique(dropna=False) != 1:
        raise SupervisedDataError(
            "Each processed file must contain exactly one series_id."
        )
    if frame["seed"].nunique(dropna=False) != 1:
        raise SupervisedDataError(
            "Each processed file must contain exactly one seed."
        )

    observed_splits = set(frame["split"].astype(str).unique().tolist())
    unknown_splits = observed_splits.difference(VALID_SPLITS)
    if unknown_splits:
        raise SupervisedDataError(
            f"Unknown split labels found: {sorted(unknown_splits)}"
        )

    split_order = {
        split_name: index for index, split_name in enumerate(VALID_SPLITS)
    }
    numeric_split_order = (
        frame["split"].astype(str).map(split_order).to_numpy(int)
    )
    if len(numeric_split_order) > 1 and np.any(
        np.diff(numeric_split_order) < 0
    ):
        raise SupervisedDataError(
            "Split labels must follow prior→bound→validation→test."
        )


def build_lagged_dataset(
    frame: pd.DataFrame,
    *,
    lag_order: int,
    history_policy: HistoryPolicy = "past_context",
) -> LaggedDataset:
    """
    Build one-step-ahead forecasting samples.

    For target index t:
        X_t = [y_(t-1), y_(t-2), ..., y_(t-p)]
        target = y_t

    Split membership is determined exclusively by the target row.
    """
    validate_processed_frame(frame)

    if not isinstance(lag_order, int) or isinstance(lag_order, bool):
        raise SupervisedDataError("lag_order must be an integer.")
    if lag_order <= 0:
        raise SupervisedDataError("lag_order must be strictly positive.")
    if lag_order >= len(frame):
        raise SupervisedDataError(
            f"lag_order={lag_order} must be smaller than series length={len(frame)}."
        )
    if history_policy not in ("past_context", "same_split_only"):
        raise SupervisedDataError(
            "history_policy must be 'past_context' or 'same_split_only'."
        )

    ordered = frame.reset_index(drop=True)
    scaled_values = ordered["value_scaled"].to_numpy(float)
    raw_values = ordered["value_raw"].to_numpy(float)
    times = ordered["time"].to_numpy()
    splits = ordered["split"].astype(str).to_numpy(object)

    target_index = np.arange(lag_order, len(ordered), dtype=np.int64)
    lag_offsets = np.arange(1, lag_order + 1, dtype=np.int64)
    lag_indices = target_index[:, None] - lag_offsets[None, :]

    # lag_1 is the immediate past; lag_p is the oldest observation.
    X = scaled_values[lag_indices]
    y_scaled = scaled_values[target_index]
    y_raw = raw_values[target_index]
    target_time = times[target_index]
    target_split = splits[target_index]

    if history_policy == "same_split_only":
        history_splits = splits[lag_indices]
        keep_mask = np.all(
            history_splits == target_split[:, None],
            axis=1,
        )
        X = X[keep_mask]
        y_scaled = y_scaled[keep_mask]
        y_raw = y_raw[keep_mask]
        target_time = target_time[keep_mask]
        target_index = target_index[keep_mask]
        target_split = target_split[keep_mask]

    dataset = LaggedDataset(
        X=np.asarray(X, dtype=np.float64),
        y_scaled=np.asarray(y_scaled, dtype=np.float64),
        y_raw=np.asarray(y_raw, dtype=np.float64),
        target_time=np.asarray(target_time),
        target_index=np.asarray(target_index, dtype=np.int64),
        split=np.asarray(target_split, dtype=object),
        series_id=str(ordered["series_id"].iloc[0]),
        seed=int(ordered["seed"].iloc[0]),
        lag_order=lag_order,
        feature_names=tuple(
            f"lag_{lag}" for lag in range(1, lag_order + 1)
        ),
        history_policy=history_policy,
    )
    dataset.validate()
    return dataset


def load_lagged_npz(path: str | Path) -> LaggedDataset:
    """Load and validate a lagged dataset archive."""
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as archive:
        dataset = LaggedDataset(
            X=archive["X"],
            y_scaled=archive["y_scaled"],
            y_raw=archive["y_raw"],
            target_time=archive["target_time"],
            target_index=archive["target_index"],
            split=archive["split"].astype(object),
            series_id=str(archive["series_id"].item()),
            seed=int(archive["seed"].item()),
            lag_order=int(archive["lag_order"].item()),
            feature_names=tuple(
                archive["feature_names"].astype(str).tolist()
            ),
            history_policy=str(archive["history_policy"].item()),
        )
    dataset.validate()
    return dataset


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum of an output archive."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
