"""Extract auditable multi-rule TSK examples from development models.

The script reconstructs the exact antecedents and posterior-center consequents
for one synthetic SETAR development series and the selected Tetouan zone-1
model. These are post-hoc interpretability summaries; they do not alter any
selection, certificate, gate, or confirmatory result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pacbayes_tsk.data.synthetic_v3 import generate
from pacbayes_tsk.data.splits_v3 import ratio_split
from pacbayes_tsk.experiments.development_v3 import DevelopmentSettings
from pacbayes_tsk.experiments.energy_case_study_v3 import EnergySettings
from pacbayes_tsk.models.sparse_tsk import fit_radius_antecedent

ROLES = ("prior", "bound", "validation", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lagged(values: np.ndarray, labels: np.ndarray, lag: int):
    targets = np.arange(lag, len(values))
    X = np.column_stack([values[targets - j] for j in range(1, lag + 1)])
    return X, values[targets], labels[targets]


def ridge_solution(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    gram = design.T @ design + float(alpha) * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ target)


def local_coefficients_original_scale(
    coefficients: np.ndarray,
    antecedent: Any,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert each local affine consequent to original target/input units."""
    K = antecedent.rule_count
    p = antecedent.n_features
    blocks = np.asarray(coefficients, float).reshape(K, p + 1)
    intercepts = np.empty(K, float)
    slopes = np.empty((K, p), float)
    input_offsets_original = target_mean + target_std * antecedent.feature_mean
    for k, block in enumerate(blocks):
        b0 = float(block[0])
        bj = np.asarray(block[1:], float)
        slopes[k] = bj / antecedent.feature_scale
        intercepts[k] = (
            target_mean
            + target_std * b0
            - float(np.dot(slopes[k], input_offsets_original))
        )
    return intercepts, slopes


def centers_original_scale(antecedent: Any, target_mean: float, target_std: float):
    centers_scaled_target = (
        antecedent.centers * antecedent.feature_scale[None, :]
        + antecedent.feature_mean[None, :]
    )
    centers_original = target_mean + target_std * centers_scaled_target
    spreads_original = (
        antecedent.spreads * antecedent.feature_scale[None, :] * target_std
    )
    return centers_original, spreads_original


def top_terms(slopes: np.ndarray, top_n: int = 3) -> str:
    order = np.argsort(-np.abs(slopes))[: min(top_n, len(slopes))]
    parts = []
    for j in order:
        parts.append(f"{slopes[j]:+.3f} y(t-{j+1})")
    return "; ".join(parts)


def summarize_rules(
    *,
    example: str,
    antecedent: Any,
    coefficients: np.ndarray,
    target_mean: float,
    target_std: float,
    certification_features: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    centers, spreads = centers_original_scale(antecedent, target_mean, target_std)
    intercepts, slopes = local_coefficients_original_scale(
        coefficients, antecedent, target_mean, target_std
    )
    firing = antecedent.firing_strengths(certification_features)
    mean_firing = firing.mean(axis=0)
    hard = np.argmax(firing, axis=1)
    hard_support = np.bincount(hard, minlength=antecedent.rule_count)

    summary_rows = []
    detail_rows = []
    for k in range(antecedent.rule_count):
        center = centers[k]
        spread = spreads[k]
        summary_rows.append(
            {
                "example": example,
                "rule": k + 1,
                "prior_support": int(antecedent.prior_support[k]),
                "certification_hard_support": int(hard_support[k]),
                "mean_firing": float(mean_firing[k]),
                "center_mean": float(center.mean()),
                "center_lag1": float(center[0]),
                "center_lag_last": float(center[-1]),
                "center_recent_minus_oldest": float(center[0] - center[-1]),
                "mean_spread": float(spread.mean()),
                "local_intercept_original": float(intercepts[k]),
                "top_consequent_terms": top_terms(slopes[k], top_n=3),
            }
        )
        for j in range(antecedent.n_features):
            detail_rows.append(
                {
                    "example": example,
                    "rule": k + 1,
                    "lag": j + 1,
                    "antecedent_center_original": float(center[j]),
                    "antecedent_spread_original": float(spread[j]),
                    "consequent_coefficient_original": float(slopes[k, j]),
                    "local_intercept_original": float(intercepts[k]),
                    "prior_support": int(antecedent.prior_support[k]),
                    "mean_firing": float(mean_firing[k]),
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values("mean_firing", ascending=False)
    detail = pd.DataFrame(detail_rows)
    return summary, detail


def synthetic_example(root: Path):
    settings = DevelopmentSettings.from_files(
        protocol_path=root / "configs/v3/protocol_v3.yaml",
        synthetic_path=root / "configs/v3/synthetic_v3.yaml",
        development_path=root / "configs/v3/fixed_k_sensitivity_v4.yaml",
    )
    selected_path = root / "results/development/fixed_k_sensitivity_v4/development_v3_selected_all.csv"
    selected = pd.read_csv(selected_path)
    row = selected[
        (selected["process"] == "setar")
        & (selected["seed"] == 3001)
        & (selected["family"] == "dense_tsk")
        & (selected["selection_strategy"] == "validation_rmse")
    ].iloc[0]
    generated = generate(
        "setar",
        length=settings.length,
        burn_in=settings.burn_in,
        seed=3001,
        parameters=settings.process_parameters["setar"],
    )
    split = ratio_split(settings.length, settings.split_fractions)
    target_mean = float(generated.values[split.labels == "prior"].mean())
    target_std = float(generated.values[split.labels == "prior"].std(ddof=0))
    values = (generated.values - target_mean) / target_std
    X, y, ylabels = lagged(values, split.labels, int(row["lag"]))
    masks = {role: ylabels == role for role in ROLES}
    ant = fit_radius_antecedent(
        X[masks["prior"]],
        radius=float(row["radius"]),
        max_rules=settings.max_rules,
        max_active_rules=settings.max_rules,
    )
    coef = ridge_solution(
        ant.design_matrix(X[masks["bound"]]),
        y[masks["bound"]],
        float(row["ridge_alpha"]),
    )
    cert_X = np.vstack([X[masks["bound"]], X[masks["validation"]]])
    summary, detail = summarize_rules(
        example="SETAR seed 3001",
        antecedent=ant,
        coefficients=coef,
        target_mean=target_mean,
        target_std=target_std,
        certification_features=cert_X,
    )
    metadata = {
        "example": "SETAR seed 3001",
        "candidate_id": str(row["candidate_id"]),
        "rule_count": int(ant.rule_count),
        "lag": int(row["lag"]),
        "radius": float(row["radius"]),
        "ridge_alpha": float(row["ridge_alpha"]),
        "selection": "development validation RMSE",
        "selected_csv_sha256": sha256(selected_path),
    }
    return summary, detail, metadata


def tetouan_example(root: Path):
    settings = EnergySettings.from_yaml(root / "configs/v3/energy_case_study.yaml")
    decision_path = root / "results/development/energy_case_v3/energy_deployment_decisions.csv"
    decisions = pd.read_csv(decision_path)
    decision = decisions[decisions["zone"] == "zone_1"].iloc[0]
    candidate_path = root / "results/development/energy_case_v3/energy_candidates_all.csv"
    candidates = pd.read_csv(candidate_path)
    row = candidates[
        (candidates["zone"] == "zone_1")
        & (candidates["candidate_id"] == decision["selected_candidate_id"])
    ].iloc[0]
    data = pd.read_csv(root / "data/processed/tetouan_two_hour.csv")
    values_original = data["zone_1"].to_numpy(float)
    split = ratio_split(len(data), settings.split_fractions)
    target_mean = float(values_original[split.labels == "prior"].mean())
    target_std = float(values_original[split.labels == "prior"].std(ddof=0))
    values = (values_original - target_mean) / target_std
    X, y, ylabels = lagged(values, split.labels, int(row["lag"]))
    masks = {role: ylabels == role for role in ROLES}
    ant = fit_radius_antecedent(
        X[masks["prior"]],
        radius=float(row["radius"]),
        max_rules=settings.max_rules,
        max_active_rules=settings.max_rules,
    )
    coef = ridge_solution(
        ant.design_matrix(X[masks["bound"]]),
        y[masks["bound"]],
        float(row["ridge_alpha"]),
    )
    cert_X = np.vstack([X[masks["bound"]], X[masks["validation"]]])
    summary, detail = summarize_rules(
        example="Tetouan Zone 1",
        antecedent=ant,
        coefficients=coef,
        target_mean=target_mean,
        target_std=target_std,
        certification_features=cert_X,
    )
    metadata = {
        "example": "Tetouan Zone 1",
        "candidate_id": str(row["candidate_id"]),
        "rule_count": int(ant.rule_count),
        "lag": int(row["lag"]),
        "radius": float(row["radius"]),
        "ridge_alpha": float(row["ridge_alpha"]),
        "selection": "development deployment gate; test failure retained",
        "decision_csv_sha256": sha256(decision_path),
        "candidate_csv_sha256": sha256(candidate_path),
    }
    return summary, detail, metadata


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def write_latex(setar: pd.DataFrame, tet: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Post-hoc examples of actual multi-rule posterior-center TSK models. Antecedent prototypes and consequent coefficients are reported in original data units. Mean firing is computed on bound plus validation observations. These deterministic rules are the point predictors used for validation/test evaluation; the PAC-Bayes guarantee directly covers the Gibbs predictor and, by convexity, the clipped Bayesian model average, not these point forecasts.}",
        r"\label{tab:fuzzy_rule_examples}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrp{5.4cm}}",
        r"\toprule",
        r"Example & Rule & Prior support & Mean firing & Center mean & Recent-oldest & Dominant local consequent terms \\",
        r"\midrule",
    ]
    for frame in (setar, tet):
        for _, row in frame.iterrows():
            lines.append(
                f"{latex_escape(str(row['example']))} & R{int(row['rule'])} & "
                f"{int(row['prior_support'])} & {100*row['mean_firing']:.1f}\\% & "
                f"{row['center_mean']:.2f} & {row['center_recent_minus_oldest']:.2f} & "
                f"{latex_escape(str(row['top_consequent_terms']))} \\\\"
            )
        if frame is setar:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="results/interpretability/v4_3")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)

    s_summary, s_detail, s_meta = synthetic_example(root)
    t_summary, t_detail, t_meta = tetouan_example(root)
    summary = pd.concat([s_summary, t_summary], ignore_index=True)
    detail = pd.concat([s_detail, t_detail], ignore_index=True)
    summary.to_csv(out / "fuzzy_rule_examples_summary.csv", index=False)
    detail.to_csv(out / "fuzzy_rule_examples_full.csv", index=False)
    write_latex(s_summary, t_summary, root / "paper/tables/fuzzy_rule_examples.tex")
    audit = {
        "schema_version": "4.3",
        "status": "post_hoc_interpretability_only",
        "selection_modified": False,
        "confirmatory_gate_modified": False,
        "examples": [s_meta, t_meta],
        "summary_sha256": sha256(out / "fuzzy_rule_examples_summary.csv"),
        "full_sha256": sha256(out / "fuzzy_rule_examples_full.csv"),
    }
    (out / "fuzzy_rule_examples_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
