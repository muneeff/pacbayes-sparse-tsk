#!/usr/bin/env python
"""Summarize the exploratory multi-K fixed-rule sensitivity analysis.

This script never changes the frozen PJM protocol or outcomes. It operates only
on synthetic development candidate files, selects candidates within each
predeclared K, opens the synthetic test role after each selection, and reports
paired uncertainty against radius-controlled all-active TSK.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pacbayes_tsk.data.synthetic_v3 import generate
from pacbayes_tsk.data.splits_v3 import ratio_split
from pacbayes_tsk.experiments.development_v3 import (
    DevelopmentSettings,
    _refit_test_metrics,
    _selection_key_certificate,
    _selection_key_rmse,
)

METRICS = {
    "validation_rmse_scaled": "Validation RMSE",
    "test_rmse_scaled": "Test RMSE",
    "consequent_dimension": "Dimension",
    "localized_gaussian_kl": "Gaussian KL",
    "localized_certificate_familywise": "Certificate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/v3/protocol_v3.yaml")
    parser.add_argument("--synthetic", default="configs/v3/synthetic_v3.yaml")
    parser.add_argument("--development", default="configs/v3/fixed_k_sensitivity_v4.yaml")
    parser.add_argument("--input", default="results/development/fixed_k_sensitivity_v4")
    parser.add_argument("--output", default="results/development/fixed_k_sensitivity_v4")
    parser.add_argument("--paper-root", default="paper")
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    return parser.parse_args()


def _regenerate_standardized(process: str, seed: int, settings: DevelopmentSettings):
    generated = generate(
        process,
        length=settings.length,
        burn_in=settings.burn_in,
        seed=seed,
        parameters=settings.process_parameters[process],
    )
    split = ratio_split(settings.length, settings.split_fractions)
    prior = generated.values[split.labels == "prior"]
    mean = float(np.mean(prior))
    std = float(np.std(prior, ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        std = 1.0
    values = (generated.values - mean) / std
    return values, split.labels, std


def _select(pool: pd.DataFrame, strategy: str) -> pd.Series:
    if pool.empty:
        raise ValueError("Cannot select from an empty pool.")
    key = _selection_key_rmse if strategy == "validation_rmse" else _selection_key_certificate
    idx = min(pool.index, key=lambda i: key(pool.loc[i]))
    return pool.loc[idx]


def _bootstrap_ci(values: np.ndarray, *, n: int, rng: np.random.Generator):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return math.nan, math.nan
    draw = rng.integers(0, len(values), size=(n, len(values)))
    means = values[draw].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _wilcoxon_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.allclose(values, 0.0):
        return 1.0
    try:
        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def _holm_adjust(pvalues: pd.Series) -> pd.Series:
    p = pvalues.astype(float).to_numpy()
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return pd.Series(adjusted, index=pvalues.index)


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def main() -> int:
    args = parse_args()
    settings = DevelopmentSettings.from_files(
        protocol_path=args.protocol,
        synthetic_path=args.synthetic,
        development_path=args.development,
    )
    input_root = Path(args.input)
    output_root = Path(args.output)
    paper_root = Path(args.paper_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (paper_root / "tables").mkdir(parents=True, exist_ok=True)
    (paper_root / "figures").mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict] = []
    series_files = sorted((input_root / "series").glob("*_candidates.csv"))
    if not series_files:
        raise FileNotFoundError(f"No candidate files found in {input_root / 'series'}")

    for path in series_files:
        candidates = pd.read_csv(path)
        process = str(candidates.iloc[0]["process"])
        seed = int(candidates.iloc[0]["seed"])
        values, labels, original_scale = _regenerate_standardized(process, seed, settings)
        for strategy in ("validation_rmse", "certificate"):
            radius_pool = candidates[candidates["family"] == "dense_tsk"]
            radius_row = _select(radius_pool, strategy).copy()
            radius_payload = radius_row.to_dict()
            radius_payload.update(
                _refit_test_metrics(
                    values=values,
                    labels=labels,
                    row=radius_row,
                    settings=settings,
                    original_scale=original_scale,
                )
            )
            radius_payload.update(
                {
                    "comparison_family": "radius_tsk",
                    "fixed_k": np.nan,
                    "selection_strategy": strategy,
                }
            )
            selected_rows.append(radius_payload)

            fixed_pool_all = candidates[candidates["family"] == "fixed_k_dense_tsk"]
            for k in settings.fixed_k_values:
                pool = fixed_pool_all[fixed_pool_all["rule_count"].astype(int) == int(k)]
                row = _select(pool, strategy).copy()
                payload = row.to_dict()
                payload.update(
                    _refit_test_metrics(
                        values=values,
                        labels=labels,
                        row=row,
                        settings=settings,
                        original_scale=original_scale,
                    )
                )
                payload.update(
                    {
                        "comparison_family": "fixed_k_tsk",
                        "fixed_k": int(k),
                        "selection_strategy": strategy,
                    }
                )
                selected_rows.append(payload)

    selected = pd.DataFrame(selected_rows)
    selected_path = output_root / "fixed_k_sensitivity_selected.csv"
    selected.to_csv(selected_path, index=False)

    summary = (
        selected.groupby(["selection_strategy", "comparison_family", "fixed_k"], dropna=False)
        .agg(
            series_count=("seed", "count"),
            mean_validation_rmse=("validation_rmse_scaled", "mean"),
            sd_validation_rmse=("validation_rmse_scaled", "std"),
            mean_test_rmse=("test_rmse_scaled", "mean"),
            sd_test_rmse=("test_rmse_scaled", "std"),
            mean_rule_count=("rule_count", "mean"),
            mean_dimension=("consequent_dimension", "mean"),
            mean_gaussian_kl=("localized_gaussian_kl", "mean"),
            mean_certificate=("localized_certificate_familywise", "mean"),
            sd_certificate=("localized_certificate_familywise", "std"),
        )
        .reset_index()
    )
    summary_path = output_root / "fixed_k_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    rng = np.random.default_rng(args.bootstrap_seed)
    comparisons: list[dict] = []
    index_cols = ["process", "seed", "selection_strategy"]
    radius = selected[selected["comparison_family"] == "radius_tsk"].set_index(index_cols)
    for strategy in ("validation_rmse", "certificate"):
        radius_strategy = radius.xs(strategy, level="selection_strategy")
        for k in settings.fixed_k_values:
            fixed = selected[
                (selected["comparison_family"] == "fixed_k_tsk")
                & (selected["selection_strategy"] == strategy)
                & (selected["fixed_k"].eq(float(k)))
            ].set_index(["process", "seed"])
            common = radius_strategy.index.intersection(fixed.index)
            for metric, label in METRICS.items():
                difference = (
                    radius_strategy.loc[common, metric].astype(float).to_numpy()
                    - fixed.loc[common, metric].astype(float).to_numpy()
                )
                low, high = _bootstrap_ci(
                    difference,
                    n=args.bootstrap_resamples,
                    rng=rng,
                )
                comparisons.append(
                    {
                        "selection_strategy": strategy,
                        "fixed_k": int(k),
                        "metric": metric,
                        "metric_label": label,
                        "pair_count": int(len(difference)),
                        "mean_radius_minus_fixed": float(np.mean(difference)),
                        "median_radius_minus_fixed": float(np.median(difference)),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "radius_lower_count": int(np.sum(difference < 0)),
                        "wilcoxon_p": _wilcoxon_p(difference),
                    }
                )
    comparison = pd.DataFrame(comparisons)
    comparison["wilcoxon_p_holm"] = np.nan
    for (strategy, metric), idx in comparison.groupby(["selection_strategy", "metric"]).groups.items():
        comparison.loc[idx, "wilcoxon_p_holm"] = _holm_adjust(comparison.loc[idx, "wilcoxon_p"]).to_numpy()
    comparison_path = output_root / "fixed_k_vs_radius_paired_statistics.csv"
    comparison.to_csv(comparison_path, index=False)

    # Best Fixed-K chosen across the predeclared K grid, paired against radius.
    best_rows = []
    for strategy in ("validation_rmse", "certificate"):
        key_metric = "validation_rmse_scaled" if strategy == "validation_rmse" else "localized_certificate_familywise_untruncated"
        fixed_strategy = selected[
            (selected["comparison_family"] == "fixed_k_tsk")
            & (selected["selection_strategy"] == strategy)
        ]
        for (_, _), group in fixed_strategy.groupby(["process", "seed"]):
            order = [key_metric, "validation_rmse_scaled", "validation_mae_scaled", "consequent_dimension", "rule_count"]
            row = group.sort_values(order, kind="stable").iloc[0].to_dict()
            row["comparison_family"] = "best_fixed_k_tsk"
            best_rows.append(row)
    best = pd.DataFrame(best_rows)
    best_path = output_root / "best_fixed_k_selected.csv"
    best.to_csv(best_path, index=False)

    best_comparisons = []
    for strategy in ("validation_rmse", "certificate"):
        r = selected[(selected["comparison_family"] == "radius_tsk") & (selected["selection_strategy"] == strategy)].set_index(["process", "seed"])
        b = best[best["selection_strategy"] == strategy].set_index(["process", "seed"])
        common = r.index.intersection(b.index)
        for metric, label in METRICS.items():
            difference = r.loc[common, metric].astype(float).to_numpy() - b.loc[common, metric].astype(float).to_numpy()
            low, high = _bootstrap_ci(difference, n=args.bootstrap_resamples, rng=rng)
            best_comparisons.append({
                "selection_strategy": strategy,
                "metric": metric,
                "metric_label": label,
                "pair_count": int(len(difference)),
                "mean_radius_minus_best_fixed": float(np.mean(difference)),
                "median_radius_minus_best_fixed": float(np.median(difference)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "radius_lower_count": int(np.sum(difference < 0)),
                "wilcoxon_p": _wilcoxon_p(difference),
            })
    best_comparison = pd.DataFrame(best_comparisons)
    best_comparison["wilcoxon_p_holm"] = _holm_adjust(best_comparison["wilcoxon_p"])
    best_comparison_path = output_root / "radius_vs_best_fixed_k_statistics.csv"
    best_comparison.to_csv(best_comparison_path, index=False)

    # Paper table: validation-RMSE selection only.
    primary = summary[
        (summary["selection_strategy"] == "validation_rmse")
        & (summary["comparison_family"] == "fixed_k_tsk")
    ].sort_values("fixed_k")
    cert_diffs = comparison[
        (comparison["selection_strategy"] == "validation_rmse")
        & (comparison["metric"] == "localized_certificate_familywise")
    ].set_index("fixed_k")
    lines = [
        r"\begin{table}[t]",
        r"\caption{Fixed-$K$ sensitivity under validation-RMSE selection across 30 synthetic development series. The paired difference is Radius TSK minus Fixed-$K$ TSK; negative values favor radius control.}",
        r"\label{tab:fixed_k_sensitivity}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$K$ & Dimension & Gaussian KL & Certificate & Test RMSE & $\Delta$ Cert. \\",
        r"\midrule",
    ]
    for _, row in primary.iterrows():
        k = int(row["fixed_k"])
        delta = float(cert_diffs.loc[k, "mean_radius_minus_fixed"])
        lines.append(
            f"{k} & {row['mean_dimension']:.1f} & {row['mean_gaussian_kl']:.3f} & "
            f"{row['mean_certificate']:.4f} & {row['mean_test_rmse']:.4f} & {delta:+.4f} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (paper_root / "tables" / "fixed_k_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figures with bootstrap confidence intervals for per-K means.
    fixed_primary = selected[
        (selected["comparison_family"] == "fixed_k_tsk")
        & (selected["selection_strategy"] == "validation_rmse")
    ]
    radius_primary = selected[
        (selected["comparison_family"] == "radius_tsk")
        & (selected["selection_strategy"] == "validation_rmse")
    ]
    ks = list(settings.fixed_k_values)
    cert_means, cert_low, cert_high = [], [], []
    rmse_means = []
    for k in ks:
        vals = fixed_primary[fixed_primary["fixed_k"].astype(int) == int(k)]["localized_certificate_familywise"].to_numpy(float)
        cert_means.append(float(vals.mean()))
        lo, hi = _bootstrap_ci(vals, n=args.bootstrap_resamples, rng=rng)
        cert_low.append(lo)
        cert_high.append(hi)
        rmse_means.append(float(fixed_primary[fixed_primary["fixed_k"].astype(int) == int(k)]["test_rmse_scaled"].mean()))
    cert_means = np.asarray(cert_means)
    yerr = np.vstack([cert_means - np.asarray(cert_low), np.asarray(cert_high) - cert_means])
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.errorbar(ks, cert_means, yerr=yerr, marker="o", capsize=3, label="Fixed-K TSK")
    ax.axhline(float(radius_primary["localized_certificate_familywise"].mean()), linestyle="--", label="Radius TSK")
    ax.set_xlabel("Fixed rule count K")
    ax.set_ylabel("Mean familywise certificate")
    ax.set_xticks(ks)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paper_root / "figures" / "fixed_k_certificate_sensitivity.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(cert_means, rmse_means, marker="o", label="Fixed-K TSK")
    for k, x, y in zip(ks, cert_means, rmse_means):
        ax.annotate(f"K={k}", (x, y), xytext=(4, 4), textcoords="offset points")
    radius_x = float(radius_primary["localized_certificate_familywise"].mean())
    radius_y = float(radius_primary["test_rmse_scaled"].mean())
    ax.scatter([radius_x], [radius_y], marker="x", s=70, label="Radius TSK")
    ax.set_xlabel("Mean familywise certificate")
    ax.set_ylabel("Mean standardized test RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paper_root / "figures" / "fixed_k_pareto_sensitivity.png", dpi=220)
    plt.close(fig)

    audit = {
        "phase": "exploratory_development_extension",
        "confirmatory": False,
        "pjm_protocol_modified": False,
        "fixed_k_values": list(settings.fixed_k_values),
        "series_count": int(selected[["process", "seed"]].drop_duplicates().shape[0]),
        "selected_rows": int(len(selected)),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "outputs": [
            selected_path.name,
            summary_path.name,
            comparison_path.name,
            best_path.name,
            best_comparison_path.name,
        ],
    }
    (output_root / "fixed_k_sensitivity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
