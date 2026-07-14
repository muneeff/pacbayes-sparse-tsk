from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


VALID_SPLITS = ("prior", "bound", "validation", "test")
HistoryPolicy = Literal["past_context", "same_split_only"]

REQUIRED_REAL_COLUMNS = (
    "dataset",
    "series_id",
    "time",
    "value_raw",
    "value_imputed",
    "value_scaled",
    "split",
    "was_imputed",
    "imputation_source",
)


class RealSupervisedDataError(ValueError):
    """Raised when real lagged forecasting data cannot be built safely."""


@dataclass(frozen=True)
class RealLaggedDataset:
    """
    Lagged one-step-ahead data for one real series and one lag order.

    `y_raw` is the finite original-scale target after causal imputation.
    `y_original_raw` preserves the pre-imputation target and can contain NaN
    only when `target_observed_mask` is False.

    Downstream training/evaluation should use `split_view`, whose default
    excludes targets that were originally missing.
    """

    X: np.ndarray
    y_scaled: np.ndarray
    y_raw: np.ndarray
    y_original_raw: np.ndarray
    target_time: np.ndarray
    target_index: np.ndarray
    split: np.ndarray
    target_observed_mask: np.ndarray
    context_imputed_count: np.ndarray
    dataset: str
    series_id: str
    lag_order: int
    feature_names: tuple[str, ...]
    history_policy: str

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_observed_targets(self) -> int:
        return int(np.sum(self.target_observed_mask))

    @property
    def n_excluded_imputed_targets(self) -> int:
        return int(self.n_samples - self.n_observed_targets)

    @property
    def n_samples_with_imputed_context(self) -> int:
        return int(np.sum(self.context_imputed_count > 0))

    def all_split_counts(self) -> dict[str, int]:
        return {
            split_name: int(np.sum(self.split == split_name))
            for split_name in VALID_SPLITS
        }

    def observed_split_counts(self) -> dict[str, int]:
        return {
            split_name: int(
                np.sum(
                    (self.split == split_name)
                    & self.target_observed_mask
                )
            )
            for split_name in VALID_SPLITS
        }

    def split_view(
        self,
        split_name: str,
        *,
        observed_targets_only: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Return one temporal split.

        By default, originally missing targets are excluded from both training
        and evaluation. Imputed past values may remain in X as causal context.
        """
        if split_name not in VALID_SPLITS:
            raise RealSupervisedDataError(
                f"Unknown split {split_name!r}; expected one of {VALID_SPLITS}."
            )

        mask = self.split == split_name
        if observed_targets_only:
            mask = mask & self.target_observed_mask

        return {
            "X": self.X[mask],
            "y_scaled": self.y_scaled[mask],
            "y_raw": self.y_raw[mask],
            "y_original_raw": self.y_original_raw[mask],
            "target_time": self.target_time[mask],
            "target_index": self.target_index[mask],
            "target_observed_mask": self.target_observed_mask[mask],
            "context_imputed_count": self.context_imputed_count[mask],
        }

    def validate(self) -> None:
        if self.X.ndim != 2:
            raise RealSupervisedDataError("X must be two-dimensional.")
        if self.X.shape[1] != self.lag_order:
            raise RealSupervisedDataError(
                "The number of feature columns must equal lag_order."
            )
        if self.n_samples == 0:
            raise RealSupervisedDataError("No lagged samples were created.")

        aligned = {
            "y_scaled": self.y_scaled,
            "y_raw": self.y_raw,
            "y_original_raw": self.y_original_raw,
            "target_time": self.target_time,
            "target_index": self.target_index,
            "split": self.split,
            "target_observed_mask": self.target_observed_mask,
            "context_imputed_count": self.context_imputed_count,
        }
        for name, array in aligned.items():
            if len(array) != self.n_samples:
                raise RealSupervisedDataError(
                    f"{name} has {len(array)} rows but X has {self.n_samples}."
                )

        if not np.all(np.isfinite(self.X)):
            raise RealSupervisedDataError("X contains NaN or infinite values.")
        if not np.all(np.isfinite(self.y_scaled)):
            raise RealSupervisedDataError(
                "y_scaled contains NaN or infinite values."
            )
        if not np.all(np.isfinite(self.y_raw)):
            raise RealSupervisedDataError(
                "y_raw contains NaN or infinite values."
            )
        if not np.all(np.isfinite(self.target_time.astype(float))):
            raise RealSupervisedDataError(
                "target_time contains non-finite values."
            )

        if self.target_observed_mask.dtype != np.bool_:
            raise RealSupervisedDataError(
                "target_observed_mask must have Boolean dtype."
            )

        observed_original = self.y_original_raw[
            self.target_observed_mask
        ]
        if not np.all(np.isfinite(observed_original)):
            raise RealSupervisedDataError(
                "Observed targets must have finite y_original_raw values."
            )
        if not np.allclose(
            observed_original,
            self.y_raw[self.target_observed_mask],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RealSupervisedDataError(
                "Observed original targets must equal y_raw."
            )

        missing_original = self.y_original_raw[
            ~self.target_observed_mask
        ]
        if missing_original.size and np.any(np.isfinite(missing_original)):
            raise RealSupervisedDataError(
                "Targets marked unobserved must have NaN y_original_raw."
            )

        if np.any(self.context_imputed_count < 0):
            raise RealSupervisedDataError(
                "context_imputed_count cannot be negative."
            )
        if np.any(self.context_imputed_count > self.lag_order):
            raise RealSupervisedDataError(
                "context_imputed_count cannot exceed lag_order."
            )

        if np.any(np.diff(self.target_index) <= 0):
            raise RealSupervisedDataError(
                "target_index must be strictly increasing."
            )
        if np.any(np.diff(self.target_time.astype(float)) <= 0):
            raise RealSupervisedDataError(
                "target_time must be strictly increasing."
            )

        observed_splits = set(np.unique(self.split).tolist())
        unknown = observed_splits.difference(VALID_SPLITS)
        if unknown:
            raise RealSupervisedDataError(
                f"Unknown split labels: {sorted(unknown)}."
            )

        expected_names = tuple(
            f"lag_{lag}" for lag in range(1, self.lag_order + 1)
        )
        if self.feature_names != expected_names:
            raise RealSupervisedDataError(
                "feature_names must be lag_1, ..., lag_p."
            )

        if not self.dataset.strip():
            raise RealSupervisedDataError("dataset cannot be empty.")
        if not self.series_id.strip():
            raise RealSupervisedDataError("series_id cannot be empty.")

    def save_npz(
        self,
        path: str | Path,
        *,
        compressed: bool = True,
    ) -> Path:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "X": self.X,
            "y_scaled": self.y_scaled,
            "y_raw": self.y_raw,
            "y_original_raw": self.y_original_raw,
            "target_time": self.target_time,
            "target_index": self.target_index,
            "split": self.split.astype("U16"),
            "target_observed_mask": self.target_observed_mask,
            "eligible_sample_indices": np.flatnonzero(
                self.target_observed_mask
            ).astype(np.int64),
            "context_imputed_count": self.context_imputed_count,
            "context_contains_imputation": (
                self.context_imputed_count > 0
            ),
            "dataset": np.asarray(self.dataset),
            "series_id": np.asarray(self.series_id),
            "lag_order": np.asarray(self.lag_order, dtype=np.int64),
            "feature_names": np.asarray(
                self.feature_names,
                dtype="U32",
            ),
            "history_policy": np.asarray(self.history_policy),
        }

        save_function = np.savez_compressed if compressed else np.savez
        save_function(output, **payload)
        return output


def _coerce_boolean(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(set(normalized).difference(mapping))
    if unknown:
        raise RealSupervisedDataError(
            f"Invalid Boolean values in was_imputed: {unknown}."
        )
    return normalized.map(mapping).to_numpy(dtype=bool)


def validate_real_processed_frame(frame: pd.DataFrame) -> None:
    missing_columns = [
        column
        for column in REQUIRED_REAL_COLUMNS
        if column not in frame.columns
    ]
    if missing_columns:
        raise RealSupervisedDataError(
            f"Processed real series is missing columns: {missing_columns}."
        )

    if frame.empty:
        raise RealSupervisedDataError(
            "Processed real series is empty."
        )

    if frame["dataset"].nunique(dropna=False) != 1:
        raise RealSupervisedDataError(
            "Each file must contain exactly one dataset value."
        )
    if frame["series_id"].nunique(dropna=False) != 1:
        raise RealSupervisedDataError(
            "Each file must contain exactly one series_id."
        )

    time = pd.to_numeric(
        frame["time"],
        errors="coerce",
    ).to_numpy(float)
    if not np.all(np.isfinite(time)):
        raise RealSupervisedDataError(
            "time contains non-finite values."
        )
    if len(time) > 1 and np.any(np.diff(time) <= 0):
        raise RealSupervisedDataError(
            "time must be strictly increasing."
        )

    for column in ("value_imputed", "value_scaled"):
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(float)
        if not np.all(np.isfinite(values)):
            raise RealSupervisedDataError(
                f"{column} contains NaN or infinite values."
            )

    raw = pd.to_numeric(
        frame["value_raw"],
        errors="coerce",
    ).to_numpy(float)
    imputed_mask = _coerce_boolean(frame["was_imputed"])

    if np.any(~imputed_mask & ~np.isfinite(raw)):
        raise RealSupervisedDataError(
            "An observed target has a non-finite value_raw."
        )
    if np.any(imputed_mask & np.isfinite(raw)):
        raise RealSupervisedDataError(
            "Rows marked imputed must preserve missing value_raw as NaN."
        )

    observed_splits = set(
        frame["split"].astype(str).unique().tolist()
    )
    unknown = observed_splits.difference(VALID_SPLITS)
    if unknown:
        raise RealSupervisedDataError(
            f"Unknown split labels: {sorted(unknown)}."
        )

    split_order = {
        split_name: index
        for index, split_name in enumerate(VALID_SPLITS)
    }
    numeric_order = (
        frame["split"]
        .astype(str)
        .map(split_order)
        .to_numpy(int)
    )
    if len(numeric_order) > 1 and np.any(np.diff(numeric_order) < 0):
        raise RealSupervisedDataError(
            "Splits must follow prior→bound→validation→test."
        )


def build_real_lagged_dataset(
    frame: pd.DataFrame,
    *,
    lag_order: int,
    history_policy: HistoryPolicy = "past_context",
) -> RealLaggedDataset:
    """
    Build one-step-ahead real forecasting samples.

    X_t = [y_(t-1), y_(t-2), ..., y_(t-p)]
    target = y_t

    The target split determines sample membership. Every feature index is
    strictly earlier than the target index.
    """
    validate_real_processed_frame(frame)

    if not isinstance(lag_order, int) or isinstance(lag_order, bool):
        raise RealSupervisedDataError(
            "lag_order must be an integer."
        )
    if lag_order <= 0:
        raise RealSupervisedDataError(
            "lag_order must be strictly positive."
        )
    if lag_order >= len(frame):
        raise RealSupervisedDataError(
            f"lag_order={lag_order} must be smaller than "
            f"series length={len(frame)}."
        )
    if history_policy not in ("past_context", "same_split_only"):
        raise RealSupervisedDataError(
            "history_policy must be 'past_context' or "
            "'same_split_only'."
        )

    ordered = frame.reset_index(drop=True)
    values_scaled = ordered["value_scaled"].to_numpy(float)
    values_imputed = ordered["value_imputed"].to_numpy(float)
    values_original = pd.to_numeric(
        ordered["value_raw"],
        errors="coerce",
    ).to_numpy(float)
    imputed_mask = _coerce_boolean(ordered["was_imputed"])
    times = ordered["time"].to_numpy()
    splits = ordered["split"].astype(str).to_numpy(object)

    target_index = np.arange(
        lag_order,
        len(ordered),
        dtype=np.int64,
    )
    lag_offsets = np.arange(
        1,
        lag_order + 1,
        dtype=np.int64,
    )
    lag_indices = (
        target_index[:, None] - lag_offsets[None, :]
    )

    X = values_scaled[lag_indices]
    y_scaled = values_scaled[target_index]
    y_raw = values_imputed[target_index]
    y_original_raw = values_original[target_index]
    target_time = times[target_index]
    target_split = splits[target_index]
    target_observed_mask = ~imputed_mask[target_index]
    context_imputed_count = np.sum(
        imputed_mask[lag_indices],
        axis=1,
    ).astype(np.int16)

    if history_policy == "same_split_only":
        history_splits = splits[lag_indices]
        keep = np.all(
            history_splits == target_split[:, None],
            axis=1,
        )
        X = X[keep]
        y_scaled = y_scaled[keep]
        y_raw = y_raw[keep]
        y_original_raw = y_original_raw[keep]
        target_time = target_time[keep]
        target_index = target_index[keep]
        target_split = target_split[keep]
        target_observed_mask = target_observed_mask[keep]
        context_imputed_count = context_imputed_count[keep]

    dataset = RealLaggedDataset(
        X=np.asarray(X, dtype=np.float64),
        y_scaled=np.asarray(y_scaled, dtype=np.float64),
        y_raw=np.asarray(y_raw, dtype=np.float64),
        y_original_raw=np.asarray(
            y_original_raw,
            dtype=np.float64,
        ),
        target_time=np.asarray(target_time),
        target_index=np.asarray(
            target_index,
            dtype=np.int64,
        ),
        split=np.asarray(target_split, dtype=object),
        target_observed_mask=np.asarray(
            target_observed_mask,
            dtype=bool,
        ),
        context_imputed_count=np.asarray(
            context_imputed_count,
            dtype=np.int16,
        ),
        dataset=str(ordered["dataset"].iloc[0]),
        series_id=str(ordered["series_id"].iloc[0]),
        lag_order=lag_order,
        feature_names=tuple(
            f"lag_{lag}"
            for lag in range(1, lag_order + 1)
        ),
        history_policy=history_policy,
    )
    dataset.validate()
    return dataset


def load_real_lagged_npz(path: str | Path) -> RealLaggedDataset:
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as archive:
        dataset = RealLaggedDataset(
            X=archive["X"],
            y_scaled=archive["y_scaled"],
            y_raw=archive["y_raw"],
            y_original_raw=archive["y_original_raw"],
            target_time=archive["target_time"],
            target_index=archive["target_index"],
            split=archive["split"].astype(object),
            target_observed_mask=archive[
                "target_observed_mask"
            ].astype(bool),
            context_imputed_count=archive[
                "context_imputed_count"
            ].astype(np.int16),
            dataset=str(archive["dataset"].item()),
            series_id=str(archive["series_id"].item()),
            lag_order=int(archive["lag_order"].item()),
            feature_names=tuple(
                archive["feature_names"]
                .astype(str)
                .tolist()
            ),
            history_policy=str(
                archive["history_policy"].item()
            ),
        )
    dataset.validate()
    return dataset


def parse_compatible_lags(
    value: Any,
) -> list[int]:
    """
    Parse compatible_lags from a selected-series manifest.

    Supported examples:
      "3|5|10"
      "3,5,10"
      [3, 5, 10]
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        raw_values = list(value)
    else:
        if pd.isna(value):
            return []
        text = str(value).strip()
        if not text:
            return []
        normalized = text.replace(",", "|").replace(";", "|")
        raw_values = [
            token.strip()
            for token in normalized.split("|")
            if token.strip()
        ]

    parsed = sorted(set(int(token) for token in raw_values))
    if any(lag <= 0 for lag in parsed):
        raise RealSupervisedDataError(
            "compatible_lags must contain positive integers."
        )
    return parsed


def safe_filename(value: str) -> str:
    cleaned = "".join(
        character
        if character.isalnum() or character in "-_."
        else "_"
        for character in str(value)
    )
    cleaned = cleaned.strip("._")
    return cleaned or "series"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()
