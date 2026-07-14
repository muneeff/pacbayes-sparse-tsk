from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


class ConfirmatoryProtocolError(ValueError):
    """Raised when the confirmatory protocol cannot be locked safely."""


KEY = ["dataset", "series_id"]


def deterministic_score(
    *,
    namespace: str,
    dataset: str,
    series_id: str,
) -> str:
    payload = (
        f"{namespace}|{dataset}|{series_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_unseen_real_series(
    full_manifest: pd.DataFrame,
    development_manifest: pd.DataFrame,
    *,
    counts: dict[str, int],
    namespace: str,
) -> pd.DataFrame:
    required = {"dataset", "series_id"}
    missing_full = required.difference(
        full_manifest.columns
    )
    missing_development = required.difference(
        development_manifest.columns
    )
    if missing_full:
        raise ConfirmatoryProtocolError(
            f"Full manifest is missing: {sorted(missing_full)}."
        )
    if missing_development:
        raise ConfirmatoryProtocolError(
            "Development manifest is missing: "
            f"{sorted(missing_development)}."
        )

    full = full_manifest.copy()
    development = development_manifest.copy()
    full["dataset"] = full["dataset"].astype(str)
    full["series_id"] = full["series_id"].astype(str)
    development["dataset"] = (
        development["dataset"].astype(str)
    )
    development["series_id"] = (
        development["series_id"].astype(str)
    )

    if "status" in full.columns:
        accepted = {
            "PASS",
            "SELECTED",
            "ELIGIBLE",
            "OK",
            "PROCESSED",
        }
        status = full["status"].astype(str).str.upper()
        if status.isin(accepted).any():
            full = full.loc[
                status.isin(accepted)
            ].copy()

    development_keys = {
        (row.dataset, row.series_id)
        for row in development[KEY].itertuples(index=False)
    }

    selected_frames = []
    for dataset, count in counts.items():
        candidates = full.loc[
            full["dataset"].eq(str(dataset))
        ].drop_duplicates(KEY).copy()
        candidates = candidates.loc[
            ~candidates.apply(
                lambda row: (
                    str(row["dataset"]),
                    str(row["series_id"]),
                )
                in development_keys,
                axis=1,
            )
        ].copy()
        candidates["selection_hash"] = candidates.apply(
            lambda row: deterministic_score(
                namespace=namespace,
                dataset=str(row["dataset"]),
                series_id=str(row["series_id"]),
            ),
            axis=1,
        )
        candidates = candidates.sort_values(
            ["selection_hash", "series_id"]
        )
        if len(candidates) < int(count):
            raise ConfirmatoryProtocolError(
                f"{dataset}: requires {count} unseen series; "
                f"only {len(candidates)} are available."
            )
        chosen = candidates.head(int(count)).copy()
        chosen["source"] = "real"
        chosen["confirmatory_role"] = "unseen_real_series"
        selected_frames.append(chosen)

    selected = pd.concat(
        selected_frames,
        ignore_index=True,
        sort=False,
    )
    overlap = selected.merge(
        development[KEY].drop_duplicates(),
        on=KEY,
        how="inner",
    )
    if not overlap.empty:
        raise ConfirmatoryProtocolError(
            "Confirmatory and development real-series sets overlap."
        )
    return selected


def synthetic_confirmatory_manifest(
    *,
    regimes: list[str],
    first_seed: int,
    last_seed: int,
) -> pd.DataFrame:
    if first_seed > last_seed:
        raise ConfirmatoryProtocolError(
            "Synthetic seed range is empty."
        )
    rows = []
    for regime in regimes:
        for seed in range(
            int(first_seed),
            int(last_seed) + 1,
        ):
            rows.append(
                {
                    "source": "synthetic",
                    "dataset": str(regime),
                    "series_id": (
                        f"{regime}_confirmatory_seed_{seed}"
                    ),
                    "seed": seed,
                    "confirmatory_role": (
                        "new_synthetic_trajectory"
                    ),
                }
            )
    return pd.DataFrame(rows)
