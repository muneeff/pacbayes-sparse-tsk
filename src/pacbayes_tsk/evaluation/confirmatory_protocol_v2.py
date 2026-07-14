from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd


class ConfirmatoryProtocolV2Error(ValueError):
    """Raised when confirmatory protocol V2 cannot be locked safely."""


KEY = ["dataset", "series_id"]


def deterministic_score(
    *,
    namespace: str,
    dataset: str,
    series_id: str,
) -> str:
    payload = f"{namespace}|{dataset}|{series_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_key_frame(
    frame: pd.DataFrame,
    *,
    require_seed: bool = False,
) -> pd.DataFrame:
    required = {"dataset", "series_id"}
    if require_seed:
        required.add("seed")
    missing = required.difference(frame.columns)
    if missing:
        raise ConfirmatoryProtocolV2Error(
            f"Missing key columns: {sorted(missing)}."
        )

    normalized = frame.copy()
    normalized["dataset"] = normalized["dataset"].astype(str)
    normalized["series_id"] = normalized["series_id"].astype(str)
    if require_seed:
        normalized["seed"] = pd.to_numeric(
            normalized["seed"],
            errors="raise",
        ).astype("int64")
    return normalized


def key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    normalized = normalize_key_frame(frame)
    return {
        (row.dataset, row.series_id)
        for row in normalized[KEY].drop_duplicates().itertuples(index=False)
    }


def select_unseen_real_series_v2(
    full_manifest: pd.DataFrame,
    development_manifest: pd.DataFrame,
    failed_v1_manifest: pd.DataFrame,
    *,
    counts: dict[str, int],
    namespace: str,
) -> pd.DataFrame:
    full = normalize_key_frame(full_manifest)
    development = normalize_key_frame(development_manifest)
    failed_v1 = normalize_key_frame(failed_v1_manifest)

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
            full = full.loc[status.isin(accepted)].copy()

    excluded = key_set(development) | key_set(failed_v1)
    selected_frames: list[pd.DataFrame] = []

    for dataset, requested_count in counts.items():
        dataset = str(dataset)
        requested_count = int(requested_count)
        candidates = (
            full.loc[full["dataset"].eq(dataset)]
            .drop_duplicates(KEY)
            .copy()
        )
        candidate_keys = list(
            zip(
                candidates["dataset"].astype(str),
                candidates["series_id"].astype(str),
            )
        )
        candidates = candidates.loc[
            [item not in excluded for item in candidate_keys]
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
            ["selection_hash", "series_id"],
            kind="stable",
        )

        if len(candidates) < requested_count:
            raise ConfirmatoryProtocolV2Error(
                f"{dataset}: requires {requested_count} fresh real series; "
                f"only {len(candidates)} remain after excluding development "
                "and failed V1."
            )

        chosen = candidates.head(requested_count).copy()
        chosen["source"] = "real"
        chosen["confirmatory_role"] = "fresh_unseen_real_series_v2"
        selected_frames.append(chosen)

    selected = pd.concat(
        selected_frames,
        ignore_index=True,
        sort=False,
    )

    selected_keys = key_set(selected)
    if selected_keys & key_set(development):
        raise ConfirmatoryProtocolV2Error(
            "V2 real selection overlaps development data."
        )
    if selected_keys & key_set(failed_v1):
        raise ConfirmatoryProtocolV2Error(
            "V2 real selection overlaps failed V1 data."
        )
    if selected.duplicated(KEY).any():
        raise ConfirmatoryProtocolV2Error(
            "V2 real selection contains duplicate composite keys."
        )
    return selected


def synthetic_confirmatory_manifest_v2(
    *,
    regimes: Iterable[str],
    first_seed: int,
    last_seed: int,
    failed_v1_manifest: pd.DataFrame,
) -> pd.DataFrame:
    first_seed = int(first_seed)
    last_seed = int(last_seed)
    if first_seed > last_seed:
        raise ConfirmatoryProtocolV2Error(
            "Synthetic V2 seed range is empty."
        )

    failed = normalize_key_frame(
        failed_v1_manifest.loc[
            failed_v1_manifest["source"].astype(str).eq("synthetic")
        ],
        require_seed=True,
    )
    failed_seeds = set(failed["seed"].astype(int).tolist())

    rows = []
    for regime in regimes:
        regime = str(regime)
        for seed in range(first_seed, last_seed + 1):
            if seed in failed_seeds:
                raise ConfirmatoryProtocolV2Error(
                    f"V2 synthetic seed {seed} was already used by failed V1."
                )
            rows.append(
                {
                    "source": "synthetic",
                    "dataset": regime,
                    "series_id": f"{regime}_confirmatory_v2_seed_{seed}",
                    "seed": int(seed),
                    "confirmatory_role": "fresh_synthetic_trajectory_v2",
                }
            )

    result = pd.DataFrame(rows)
    result = normalize_key_frame(result, require_seed=True)
    if result.duplicated(["source", "dataset", "series_id"]).any():
        raise ConfirmatoryProtocolV2Error(
            "V2 synthetic manifest contains duplicate keys."
        )
    return result
