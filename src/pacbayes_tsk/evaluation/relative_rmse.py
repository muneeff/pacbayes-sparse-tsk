from __future__ import annotations

import numpy as np
import pandas as pd


SERIES_KEY = [
    "source",
    "dataset",
    "series_id",
]


def refresh_relative_rmse(
    sparse_results: pd.DataFrame,
    linear_results: pd.DataFrame | None,
) -> pd.DataFrame:
    """Rebuild naive RMSE columns without duplicate merge suffixes.

    The function is deliberately idempotent: it first removes any existing
    ``naive_rmse`` and ``relative_rmse_to_naive`` columns, then recreates them
    from the Naive baseline rows.
    """
    required_sparse = {
        *SERIES_KEY,
        "rmse",
    }
    missing_sparse = required_sparse.difference(
        sparse_results.columns
    )
    if missing_sparse:
        raise ValueError(
            "Sparse results are missing columns: "
            f"{sorted(missing_sparse)}."
        )

    output = sparse_results.drop(
        columns=[
            "naive_rmse",
            "relative_rmse_to_naive",
        ],
        errors="ignore",
    ).copy()

    if linear_results is None or linear_results.empty:
        output["naive_rmse"] = np.nan
        output["relative_rmse_to_naive"] = np.nan
        return output

    required_linear = {
        *SERIES_KEY,
        "status",
        "model",
        "rmse",
    }
    missing_linear = required_linear.difference(
        linear_results.columns
    )
    if missing_linear:
        raise ValueError(
            "Linear results are missing columns: "
            f"{sorted(missing_linear)}."
        )

    naive = linear_results.loc[
        linear_results["status"].astype(str).eq("PASS")
        & linear_results["model"].astype(str).eq("naive"),
        SERIES_KEY + ["rmse"],
    ].rename(
        columns={
            "rmse": "naive_rmse",
        }
    )

    duplicates = naive.duplicated(
        subset=SERIES_KEY,
        keep=False,
    )
    if duplicates.any():
        examples = (
            naive.loc[
                duplicates,
                SERIES_KEY,
            ]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Naive baseline has duplicate series keys: "
            f"{examples}."
        )

    output = output.merge(
        naive,
        on=SERIES_KEY,
        how="left",
        validate="one_to_one",
    )
    output["relative_rmse_to_naive"] = (
        output["rmse"]
        / output["naive_rmse"]
    )
    return output
