#!/usr/bin/env python
"""Secondary statistical analysis and confidence intervals for manuscript V4.2.

This script does not refit models, change the PJM deployment gate, or reopen any
selection decisions. It computes:
1) series-level paired bootstrap intervals and Wilcoxon tests for the synthetic
   development comparisons; and
2) paired circular moving-block bootstrap intervals for the already frozen
   Tetouan and PJM test predictions.

The PJM and Tetouan uncertainty analyses are explicitly secondary post-outcome
analyses. They quantify uncertainty around fixed decisions but do not alter or
relabel the original confirmatory protocol.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


PRIMARY_METRICS = {
    "consequent_dimension": "Dimension",
    "localized_gaussian_kl": "Gaussian KL",
    "localized_certificate_familywise": "Certificate",
    "test_rmse_scaled": "Test RMSE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="results/statistics/v4_2")
    parser.add_argument("--paper-root", default="paper")
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--pjm-primary-block", type=int, default=14)
    parser.add_argument("--tetouan-primary-block", type=int, default=84)
    return parser.parse_args()


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def _percentile_bootstrap_mean_ci(
    difference: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float]:
    difference = np.asarray(difference, dtype=float)
    if difference.size == 0:
        return math.nan, math.nan
    if np.allclose(difference, difference[0]):
        return float(difference[0]), float(difference[0])
    draw = rng.integers(0, difference.size, size=(n_resamples, difference.size))
    means = difference[draw].mean(axis=1)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def _wilcoxon_p(difference: np.ndarray) -> float:
    difference = np.asarray(difference, dtype=float)
    if difference.size == 0 or np.allclose(difference, 0.0):
        return 1.0
    try:
        result = wilcoxon(
            difference,
            zero_method="pratt",
            alternative="two-sided",
            correction=False,
            method="auto",
        )
        return float(result.pvalue)
    except ValueError:
        return 1.0


def _rank_biserial(difference: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation.

    Negative values mean the left member of the reported left-minus-right
    comparison tends to be smaller.
    """
    difference = np.asarray(difference, dtype=float)
    nonzero = difference[~np.isclose(difference, 0.0)]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    denominator = positive + negative
    if denominator <= 0:
        return 0.0
    return (positive - negative) / denominator


def _paired_frame(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_cols = ["process", "seed"]
    left_indexed = left.set_index(index_cols).sort_index()
    right_indexed = right.set_index(index_cols).sort_index()
    common = left_indexed.index.intersection(right_indexed.index)
    if len(common) != 30:
        raise ValueError(f"Expected 30 paired synthetic series, found {len(common)}")
    return left_indexed.loc[common], right_indexed.loc[common]


def _synthetic_statistics(
    repo_root: Path,
    output_root: Path,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    fixed_root = repo_root / "results/development/fixed_k_sensitivity_v4"
    selected = pd.read_csv(fixed_root / "fixed_k_sensitivity_selected.csv")
    development = pd.read_csv(fixed_root / "development_v3_selected_all.csv")
    best_fixed = pd.read_csv(fixed_root / "best_fixed_k_selected.csv")

    required = ["process", "seed", "selection_strategy", *PRIMARY_METRICS.keys()]
    _ensure_columns(selected, required + ["comparison_family", "fixed_k"], "fixed_k_sensitivity_selected.csv")
    _ensure_columns(development, required + ["family"], "development_v3_selected_all.csv")
    _ensure_columns(best_fixed, required, "best_fixed_k_selected.csv")

    validation_selected = selected[selected["selection_strategy"] == "validation_rmse"].copy()
    development_validation = development[development["selection_strategy"] == "validation_rmse"].copy()
    best_validation = best_fixed[best_fixed["selection_strategy"] == "validation_rmse"].copy()

    radius_from_sensitivity = validation_selected[
        validation_selected["comparison_family"] == "radius_tsk"
    ].copy()
    fixed_12 = validation_selected[
        (validation_selected["comparison_family"] == "fixed_k_tsk")
        & np.isclose(validation_selected["fixed_k"].astype(float), 12.0)
    ].copy()

    comparisons = [
        (
            "Radius TSK - Fixed K=12",
            "radius_vs_fixed_12",
            radius_from_sensitivity,
            fixed_12,
        ),
        (
            "Radius TSK - Best fixed K",
            "radius_vs_best_fixed",
            radius_from_sensitivity,
            best_validation,
        ),
        (
            "Radius Top-3 - Radius all-active",
            "top3_vs_all_active",
            development_validation[development_validation["family"] == "sparse_tsk"].copy(),
            development_validation[development_validation["family"] == "dense_tsk"].copy(),
        ),
        (
            "Radius TSK - Ridge",
            "radius_vs_ridge",
            development_validation[development_validation["family"] == "dense_tsk"].copy(),
            development_validation[development_validation["family"] == "ridge"].copy(),
        ),
    ]

    rows: list[dict] = []
    for comparison_label, comparison_id, left, right in comparisons:
        left_paired, right_paired = _paired_frame(left, right)
        start = len(rows)
        raw_p: list[float] = []
        for metric, metric_label in PRIMARY_METRICS.items():
            difference = (
                left_paired[metric].astype(float).to_numpy()
                - right_paired[metric].astype(float).to_numpy()
            )
            low, high = _percentile_bootstrap_mean_ci(
                difference,
                n_resamples=n_resamples,
                rng=rng,
            )
            pvalue = _wilcoxon_p(difference)
            raw_p.append(pvalue)
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "comparison": comparison_label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "pair_count": int(difference.size),
                    "mean_left_minus_right": float(np.mean(difference)),
                    "median_left_minus_right": float(np.median(difference)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "left_lower_count": int(np.sum(difference < 0)),
                    "left_higher_count": int(np.sum(difference > 0)),
                    "ties": int(np.sum(np.isclose(difference, 0.0))),
                    "wilcoxon_p": pvalue,
                    "rank_biserial": _rank_biserial(difference),
                }
            )
        adjusted = _holm_adjust(np.asarray(raw_p, dtype=float))
        for offset, value in enumerate(adjusted):
            rows[start + offset]["wilcoxon_p_holm"] = float(value)

    result = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_root / "synthetic_primary_paired_statistics.csv", index=False)
    return result


def _weighted_cost_per_observation(
    actual: np.ndarray,
    prediction: np.ndarray,
    *,
    under_weight: float = 2.0,
    over_weight: float = 1.0,
) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    under = np.maximum(actual - prediction, 0.0)
    over = np.maximum(prediction - actual, 0.0)
    return under_weight * under + over_weight * over


def _circular_block_indices(
    n: int,
    block_length: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if block_length <= 0 or block_length > n:
        raise ValueError(f"Invalid block length {block_length} for n={n}")
    block_count = int(math.ceil(n / block_length))
    starts = rng.integers(0, n, size=(n_resamples, block_count))
    offsets = np.arange(block_length, dtype=int)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    return indices.reshape(n_resamples, -1)[:, :n]


def _block_bootstrap_improvements(
    actual: np.ndarray,
    baseline: np.ndarray,
    deployed: np.ndarray,
    *,
    block_length: int,
    n_resamples: int,
    rng: np.random.Generator,
    chunk_size: int = 1000,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    deployed = np.asarray(deployed, dtype=float)
    if not (len(actual) == len(baseline) == len(deployed)):
        raise ValueError("Prediction arrays must have the same length.")
    n = len(actual)

    baseline_sq = (actual - baseline) ** 2
    deployed_sq = (actual - deployed) ** 2
    baseline_cost = _weighted_cost_per_observation(actual, baseline)
    deployed_cost = _weighted_cost_per_observation(actual, deployed)

    observed_rmse = 1.0 - math.sqrt(float(np.mean(deployed_sq))) / math.sqrt(float(np.mean(baseline_sq)))
    observed_cost = 1.0 - float(np.mean(deployed_cost)) / float(np.mean(baseline_cost))

    rmse_values = np.empty(n_resamples, dtype=float)
    cost_values = np.empty(n_resamples, dtype=float)
    position = 0
    while position < n_resamples:
        current = min(chunk_size, n_resamples - position)
        indices = _circular_block_indices(n, block_length, current, rng)
        baseline_rmse = np.sqrt(np.mean(baseline_sq[indices], axis=1))
        deployed_rmse = np.sqrt(np.mean(deployed_sq[indices], axis=1))
        baseline_mean_cost = np.mean(baseline_cost[indices], axis=1)
        deployed_mean_cost = np.mean(deployed_cost[indices], axis=1)
        rmse_values[position : position + current] = 1.0 - deployed_rmse / baseline_rmse
        cost_values[position : position + current] = 1.0 - deployed_mean_cost / baseline_mean_cost
        position += current

    def summarize(values: np.ndarray, observed: float, prefix: str) -> dict[str, float]:
        return {
            f"observed_{prefix}_improvement": float(observed),
            f"{prefix}_ci_low": float(np.quantile(values, 0.025)),
            f"{prefix}_ci_high": float(np.quantile(values, 0.975)),
            f"{prefix}_bootstrap_probability_nonpositive": float((np.sum(values <= 0.0) + 1) / (len(values) + 1)),
        }

    result = {
        "sample_size": n,
        "block_length": int(block_length),
        "bootstrap_resamples": int(n_resamples),
    }
    result.update(summarize(rmse_values, observed_rmse, "rmse"))
    result.update(summarize(cost_values, observed_cost, "cost"))
    return result


def _energy_bootstrap_statistics(
    repo_root: Path,
    output_root: Path,
    *,
    n_resamples: int,
    rng: np.random.Generator,
    pjm_primary_block: int,
    tetouan_primary_block: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pjm_rows: list[dict] = []
    for region in ("aep", "comed", "dayton", "pjme"):
        path = repo_root / f"results/confirmatory/pjm_case_v3_4/predictions/{region}_test_predictions.csv"
        frame = pd.read_csv(path)
        _ensure_columns(frame, ["actual", "seasonal_naive_7d", "deployed_forecast"], path.name)
        for block in (7, pjm_primary_block, 28):
            statistics = _block_bootstrap_improvements(
                frame["actual"].to_numpy(),
                frame["seasonal_naive_7d"].to_numpy(),
                frame["deployed_forecast"].to_numpy(),
                block_length=block,
                n_resamples=n_resamples,
                rng=rng,
            )
            statistics.update(
                {
                    "region": region.upper(),
                    "analysis_role": "primary" if block == pjm_primary_block else "sensitivity",
                }
            )
            pjm_rows.append(statistics)
    pjm = pd.DataFrame(pjm_rows)
    pjm.to_csv(output_root / "pjm_moving_block_bootstrap_ci.csv", index=False)

    tetouan_rows: list[dict] = []
    for zone in ("zone_1", "zone_2", "zone_3"):
        path = repo_root / f"results/development/energy_case_v3/predictions/{zone}_test_predictions.csv"
        frame = pd.read_csv(path)
        _ensure_columns(frame, ["actual", "seasonal_naive_24h", "deployed_forecast"], path.name)
        for block in (12, 42, tetouan_primary_block):
            statistics = _block_bootstrap_improvements(
                frame["actual"].to_numpy(),
                frame["seasonal_naive_24h"].to_numpy(),
                frame["deployed_forecast"].to_numpy(),
                block_length=block,
                n_resamples=n_resamples,
                rng=rng,
            )
            statistics.update(
                {
                    "zone": zone.replace("_", " ").title(),
                    "analysis_role": "primary" if block == tetouan_primary_block else "sensitivity",
                }
            )
            tetouan_rows.append(statistics)
    tetouan = pd.DataFrame(tetouan_rows)
    tetouan.to_csv(output_root / "tetouan_moving_block_bootstrap_ci.csv", index=False)
    return pjm, tetouan


def _format_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def _write_latex_tables(
    paper_root: Path,
    synthetic: pd.DataFrame,
    pjm: pd.DataFrame,
) -> None:
    tables = paper_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    primary = synthetic[synthetic["comparison_id"] == "radius_vs_fixed_12"].copy()
    primary_order = [
        "consequent_dimension",
        "localized_gaussian_kl",
        "localized_certificate_familywise",
        "test_rmse_scaled",
    ]
    primary = primary.set_index("metric").loc[primary_order].reset_index()
    rows = []
    for _, row in primary.iterrows():
        mean = float(row["mean_left_minus_right"])
        low = float(row["bootstrap_ci_low"])
        high = float(row["bootstrap_ci_high"])
        decimals = 1 if row["metric"] == "consequent_dimension" else (3 if row["metric"] == "localized_gaussian_kl" else 4)
        ci = f"[{low:.{decimals}f}, {high:.{decimals}f}]"
        rows.append(
            f"{row['metric_label']} & {mean:.{decimals}f} & {ci} & "
            f"{int(row['left_lower_count'])}/30 & {_format_p(float(row['wilcoxon_p_holm']))} & "
            f"{float(row['rank_biserial']):.3f} \\\\"
        )
    text = r"""\begin{table*}[t]
\centering
\caption{Paired Radius-TSK minus fixed-$K=12$ differences across 30 synthetic development series under validation-RMSE selection. Confidence intervals are 95\% paired percentile-bootstrap intervals for the mean difference. $p_H$ is the two-sided Wilcoxon signed-rank $p$-value after Holm correction across the four reported endpoints; $r_{rb}$ is matched-pairs rank-biserial correlation. Negative values favor Radius TSK.}
\label{tab:structural_inference}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrlrr}
\toprule
Metric & Mean $\Delta$ & 95\% CI & Radius lower & $p_H$ & $r_{rb}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (tables / "structural_inference.tex").write_text(text, encoding="utf-8")

    pjm_primary = pjm[pjm["analysis_role"] == "primary"].copy().sort_values("region")
    pjm_rows = []
    for _, row in pjm_primary.iterrows():
        rmse_obs = 100.0 * float(row["observed_rmse_improvement"])
        rmse_low = 100.0 * float(row["rmse_ci_low"])
        rmse_high = 100.0 * float(row["rmse_ci_high"])
        cost_obs = 100.0 * float(row["observed_cost_improvement"])
        cost_low = 100.0 * float(row["cost_ci_low"])
        cost_high = 100.0 * float(row["cost_ci_high"])
        pjm_rows.append(
            f"{row['region']} & {rmse_obs:.2f}\\% [{rmse_low:.2f}, {rmse_high:.2f}] & "
            f"{cost_obs:.2f}\\% [{cost_low:.2f}, {cost_high:.2f}] \\\\"
        )
    pjm_text = r"""\begin{table}[t]
\centering
\caption{Secondary paired moving-block-bootstrap uncertainty intervals for the frozen PJM test improvements. Circular blocks of 14 days and 20,000 resamples are used; 7- and 28-day sensitivity results are provided in the artifact. The analysis was added after the frozen test outcomes and does not alter the decisions.}
\label{tab:pjm_bootstrap_ci}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lll}
\toprule
Region & RMSE improvement [95\% CI] & Cost improvement [95\% CI] \\
\midrule
""" + "\n".join(pjm_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (tables / "pjm_bootstrap_ci.tex").write_text(pjm_text, encoding="utf-8")


def _write_report(
    repo_root: Path,
    output_root: Path,
    synthetic: pd.DataFrame,
    pjm: pd.DataFrame,
    tetouan: pd.DataFrame,
    *,
    n_resamples: int,
) -> None:
    main = synthetic[synthetic["comparison_id"] == "radius_vs_fixed_12"].set_index("metric")
    best = synthetic[synthetic["comparison_id"] == "radius_vs_best_fixed"].set_index("metric")
    top3 = synthetic[synthetic["comparison_id"] == "top3_vs_all_active"].set_index("metric")
    pjm_primary = pjm[pjm["analysis_role"] == "primary"].sort_values("region")
    tet_primary = tetouan[tetouan["analysis_role"] == "primary"].sort_values("zone")

    def metric_line(frame: pd.DataFrame, metric: str) -> str:
        row = frame.loc[metric]
        return (
            f"mean={row['mean_left_minus_right']:.6f}, 95% CI="
            f"[{row['bootstrap_ci_low']:.6f}, {row['bootstrap_ci_high']:.6f}], "
            f"Holm p={row['wilcoxon_p_holm']:.6g}, r_rb={row['rank_biserial']:.3f}"
        )

    lines = [
        "# تقرير التحليل الإحصائي وفترات الثقة V4.2",
        "",
        "## المنهج",
        f"- استخدمت المقارنات الاصطناعية الوحدة الإحصائية الصحيحة: السلسلة الكاملة (30 زوجاً)، مع {n_resamples:,} إعادة سحب زوجية على فروق المتوسط.",
        "- استخدم اختبار Wilcoxon signed-rank ثنائي الطرف مع تصحيح Holm لأربعة مؤشرات أساسية داخل كل مقارنة.",
        "- أضيف معامل matched-pairs rank-biserial لقياس حجم الأثر واتجاهه.",
        "- استخدمت PJM إعادة سحب دائرية بكتل متحركة طولها الأساسي 14 يوماً، مع تحليل حساسية لكتل 7 و28 يوماً.",
        "- تحليل PJM الإحصائي ثانوي بعد فتح نتائج الاختبار؛ لا يغيّر القرارات المجمدة ولا يُقدَّم بوصفه تسجيلاً مسبقاً.",
        "",
        "## النتيجة البنيوية الأساسية: Radius مقابل Fixed K=12",
        f"- Dimension: {metric_line(main, 'consequent_dimension')}",
        f"- Gaussian KL: {metric_line(main, 'localized_gaussian_kl')}",
        f"- Certificate: {metric_line(main, 'localized_certificate_familywise')}",
        f"- Test RMSE: {metric_line(main, 'test_rmse_scaled')}",
        "",
        "## Radius مقابل أفضل Fixed-K مختار بعدالة",
        f"- Certificate: {metric_line(best, 'localized_certificate_familywise')}",
        f"- Test RMSE: {metric_line(best, 'test_rmse_scaled')}",
        "",
        "## Top-3 مقابل all-active Radius TSK",
        f"- Certificate: {metric_line(top3, 'localized_certificate_familywise')}",
        f"- Test RMSE: {metric_line(top3, 'test_rmse_scaled')}",
        "",
        "## فترات الثقة لحالة PJM",
    ]
    for _, row in pjm_primary.iterrows():
        lines.append(
            f"- {row['region']}: RMSE {100*row['observed_rmse_improvement']:.2f}% "
            f"[{100*row['rmse_ci_low']:.2f}, {100*row['rmse_ci_high']:.2f}]؛ "
            f"الكلفة {100*row['observed_cost_improvement']:.2f}% "
            f"[{100*row['cost_ci_low']:.2f}, {100*row['cost_ci_high']:.2f}]."
        )
    lines.extend(["", "## Tetouan (تحليل سلبي تطويري)"])
    for _, row in tet_primary.iterrows():
        lines.append(
            f"- {row['zone']}: RMSE {100*row['observed_rmse_improvement']:.2f}% "
            f"[{100*row['rmse_ci_low']:.2f}, {100*row['rmse_ci_high']:.2f}]؛ "
            f"الكلفة {100*row['observed_cost_improvement']:.2f}% "
            f"[{100*row['cost_ci_low']:.2f}, {100*row['cost_ci_high']:.2f}]."
        )
    lines.extend(
        [
            "",
            "## الحكم العلمي",
            "- الأثر البنيوي مقارنةً بـK=12 قوي ومدعوم بفترات ثقة واختبارات مقترنة.",
            "- أفضلية Radius على أفضل شبكة Fixed-K صغيرة ليست حاسمة للشهادة؛ وهذا يقيد الادعاء بدقة.",
            "- Top-3 لا يحقق تحسناً إحصائياً ذا معنى في الشهادة عندما يبقى البعد العشوائي ثابتاً.",
            "- فترات PJM تقيس عدم اليقين حول قرارات ثابتة، لكنها لا تحول التحليل إلى تجربة تأكيدية مسجلة مسبقاً.",
            "",
        ]
    )
    report_path = repo_root / "docs/STATISTICAL_ANALYSIS_V4_2_REPORT_AR.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_root = repo_root / args.output
    paper_root = repo_root / args.paper_root
    output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    synthetic = _synthetic_statistics(
        repo_root,
        output_root,
        n_resamples=args.bootstrap_resamples,
        rng=rng,
    )
    pjm, tetouan = _energy_bootstrap_statistics(
        repo_root,
        output_root,
        n_resamples=args.bootstrap_resamples,
        rng=rng,
        pjm_primary_block=args.pjm_primary_block,
        tetouan_primary_block=args.tetouan_primary_block,
    )
    _write_latex_tables(paper_root, synthetic, pjm)
    _write_report(
        repo_root,
        output_root,
        synthetic,
        pjm,
        tetouan,
        n_resamples=args.bootstrap_resamples,
    )

    manifest = {
        "analysis_version": "4.2",
        "bootstrap_resamples": args.bootstrap_resamples,
        "random_seed": args.seed,
        "synthetic_unit": "complete paired series",
        "synthetic_pair_count": 30,
        "synthetic_ci": "paired percentile bootstrap for mean difference",
        "synthetic_test": "two-sided Wilcoxon signed-rank with Pratt zeros and Holm correction across four endpoints per comparison",
        "effect_size": "matched-pairs rank-biserial correlation",
        "pjm_bootstrap": "paired circular moving-block bootstrap",
        "pjm_primary_block_days": args.pjm_primary_block,
        "pjm_sensitivity_blocks_days": [7, 28],
        "tetouan_primary_block_two_hour_steps": args.tetouan_primary_block,
        "post_outcome_disclosure": True,
        "selection_or_gate_changed": False,
        "outputs": [
            str(output_root / "synthetic_primary_paired_statistics.csv"),
            str(output_root / "pjm_moving_block_bootstrap_ci.csv"),
            str(output_root / "tetouan_moving_block_bootstrap_ci.csv"),
            str(paper_root / "tables/structural_inference.tex"),
            str(paper_root / "tables/pjm_bootstrap_ci.tex"),
            str(repo_root / "docs/STATISTICAL_ANALYSIS_V4_2_REPORT_AR.md"),
        ],
    }
    (repo_root / "artifacts/statistical_analysis_v4_2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Statistical analysis written to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
