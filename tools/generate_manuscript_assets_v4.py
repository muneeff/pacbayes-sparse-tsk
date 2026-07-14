from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROCESS_LABELS = {
    "ar2": "AR(2)",
    "setar": "SETAR",
    "narma10": "NARMA-10",
    "mackey_glass": "Mackey--Glass",
    "garch": "GARCH",
    "structural_break": "Structural break",
}
PROCESS_ORDER = ["ar2", "setar", "narma10", "mackey_glass", "garch", "structural_break"]
FAMILY_LABELS = {
    "ridge": "Ridge",
    "fixed_k_dense_tsk": "Fixed-$K$ TSK",
    "dense_tsk": "Radius TSK",
    "sparse_tsk": "Radius Top-3 TSK",
}
FAMILY_ORDER = ["ridge", "fixed_k_dense_tsk", "dense_tsk", "sparse_tsk"]


def pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}\\%"


def ensure_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def generate_structural_tables(source_dir: Path, tables_dir: Path) -> dict[str, float]:
    overall = pd.read_csv(source_dir / "structural_ablation_family_overall.csv")
    pairwise = pd.read_csv(source_dir / "structural_ablation_comparison_summary.csv")
    ensure_columns(
        overall,
        {"family", "selection_strategy", "mean_rule_count", "mean_dimension", "mean_gaussian_kl", "mean_certificate"},
        "structural_ablation_family_overall.csv",
    )
    ensure_columns(
        pairwise,
        {"comparison", "selection_strategy", "metric", "mean_delta_left_minus_right", "left_lower_count", "paired_series"},
        "structural_ablation_comparison_summary.csv",
    )

    sel = overall[overall["selection_strategy"] == "validation_rmse"].copy()
    rows = []
    for family in FAMILY_ORDER:
        row = sel[sel["family"] == family]
        if row.empty:
            raise ValueError(f"Missing validation_rmse row for family={family}")
        r = row.iloc[0]
        rows.append(
            f"{FAMILY_LABELS[family]} & {r['mean_rule_count']:.2f} & {r['mean_dimension']:.1f} & "
            f"{r['mean_gaussian_kl']:.2f} & {r['mean_certificate']:.4f} \\\\" 
        )

    structural_overall = r"""\begin{table}[t]
\centering
\caption{Development ablation over 30 synthetic series. Models are selected by validation RMSE; lower values are better.}
\label{tab:structural_overall}
\small
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrrr}
\toprule
Family & $\bar K$ & $\bar D$ & Gaussian KL & Certificate \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_text(tables_dir / "structural_overall.tex", structural_overall)

    psel = pairwise[
        (pairwise["comparison"] == "radius_dense_vs_fixed_k_dense")
        & (pairwise["selection_strategy"] == "validation_rmse")
    ].copy()
    metric_map = [
        ("consequent_dimension", "Dimension", 1),
        ("localized_gaussian_kl", "Gaussian KL", 3),
        ("localized_certificate_familywise", "Certificate", 4),
        ("test_rmse_scaled", "Test RMSE", 4),
    ]
    prow = []
    summary: dict[str, float] = {}
    for metric, label, decimals in metric_map:
        row = psel[psel["metric"] == metric]
        if row.empty:
            raise ValueError(f"Missing pairwise metric={metric}")
        r = row.iloc[0]
        value = float(r["mean_delta_left_minus_right"])
        summary[metric] = value
        prow.append(
            f"{label} & {value:.{decimals}f} & {int(r['left_lower_count'])}/{int(r['paired_series'])} \\\\" 
        )

    structural_pairwise = r"""\begin{table}[t]
\centering
\caption{Paired Radius-TSK minus Fixed-$K$-TSK differences over 30 development series. Negative values favor Radius TSK.}
\label{tab:structural_pairwise}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrr}
\toprule
Metric & Mean difference & Radius lower \\
\midrule
""" + "\n".join(prow) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write_text(tables_dir / "structural_pairwise.tex", structural_pairwise)
    return summary


def generate_process_table(source_dir: Path, tables_dir: Path) -> None:
    df = pd.read_csv(source_dir / "development_v3_summary.csv")
    ensure_columns(
        df,
        {
            "process",
            "family",
            "selection_strategy",
            "mean_test_rmse",
            "mean_certificate_familywise",
            "mean_certification_clipping",
            "mean_test_clipping",
        },
        "development_v3_summary.csv",
    )
    df = df[df["selection_strategy"] == "validation_rmse"].copy()
    rows = []
    for process in PROCESS_ORDER:
        ridge = df[(df["process"] == process) & (df["family"] == "ridge")]
        radius = df[(df["process"] == process) & (df["family"] == "dense_tsk")]
        if ridge.empty or radius.empty:
            raise ValueError(f"Missing process summary for {process}")
        rr = ridge.iloc[0]
        rt = radius.iloc[0]
        rows.append(
            f"{PROCESS_LABELS[process]} & {rr['mean_test_rmse']:.4f} & {rt['mean_test_rmse']:.4f} & "
            f"{rr['mean_certificate_familywise']:.4f} & {rt['mean_certificate_familywise']:.4f} & "
            f"{pct(rr['mean_certification_clipping'])} & {pct(rr['mean_test_clipping'])} \\\\" 
        )

    text = r"""\begin{table*}[t]
\centering
\caption{Synthetic development results selected by validation RMSE (five seeds per process). RMSE is on the prior-standardized scale.}
\label{tab:process_results}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{2}{c}{Test RMSE} & \multicolumn{2}{c}{Certificate} & \multicolumn{2}{c}{Target clipping} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Process & Ridge & Radius TSK & Ridge & Radius TSK & Cert. & Test \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_text(tables_dir / "process_results.tex", text)


def generate_energy_table(repo_root: Path, tables_dir: Path) -> None:
    tet = pd.read_csv(repo_root / "results/development/energy_case_v3/energy_deployment_decisions.csv")
    pjm = pd.read_csv(repo_root / "results/confirmatory/pjm_case_v3_4/pjm_deployment_decisions.csv")
    ensure_columns(
        tet,
        {"zone", "deployment", "selected_certificate", "test_rmse_improvement_vs_fallback", "test_cost_improvement_vs_fallback"},
        "energy_deployment_decisions.csv",
    )
    ensure_columns(
        pjm,
        {"region", "deployment", "selected_certificate", "test_rmse_improvement_vs_fallback", "test_cost_improvement_vs_fallback"},
        "pjm_deployment_decisions.csv",
    )

    rows = []
    for _, r in tet.iterrows():
        cert = "--" if pd.isna(r["selected_certificate"]) else f"{r['selected_certificate']:.4f}"
        deployment = str(r["deployment"]).replace("_", r"\_")
        rows.append(
            f"Tetouan dev. & {str(r['zone']).replace('_', ' ')} & {deployment} & {cert} & "
            f"{pct(r['test_rmse_improvement_vs_fallback'])} & {pct(r['test_cost_improvement_vs_fallback'])} \\\\" 
        )
    rows.append(r"\midrule")
    for _, r in pjm.iterrows():
        deployment = str(r["deployment"]).replace("_", r"\_")
        rows.append(
            f"PJM confirm. & {str(r['region']).upper()} & {deployment} & "
            f"{r['selected_certificate']:.4f} & {pct(r['test_rmse_improvement_vs_fallback'])} & "
            f"{pct(r['test_cost_improvement_vs_fallback'])} \\\\" 
        )

    text = r"""\begin{table*}[t]
\centering
\caption{Operational decision studies. Improvements are relative to the predeclared seasonal-naive fallback; negative values indicate degradation.}
\label{tab:energy_decisions}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllr rr}
\toprule
Phase & Series & Deployment & Certificate & Test RMSE imp. & Test cost imp. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_text(tables_dir / "energy_decisions.tex", text)


def generate_pjm_family_table(repo_root: Path, tables_dir: Path) -> None:
    df = pd.read_csv(repo_root / "results/confirmatory/pjm_case_v3_4/pjm_family_results.csv")
    ensure_columns(
        df,
        {"region", "family", "lag", "rule_count", "consequent_dimension", "certificate", "test_rmse_raw"},
        "pjm_family_results.csv",
    )
    family_order = {"ridge": 0, "fixed_k_dense_tsk": 1, "dense_tsk": 2}
    df = df[df["family"].isin(family_order)].copy()
    df["family_order"] = df["family"].map(family_order)
    df = df.sort_values(["region", "family_order"])
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{str(r['region']).upper()} & {FAMILY_LABELS[r['family']]} & {int(r['lag'])} & {int(r['rule_count'])} & "
            f"{int(r['consequent_dimension'])} & {r['certificate']:.4f} & {r['test_rmse_raw']:.2f} \\\\" 
        )
    text = r"""\begin{table*}[t]
\centering
\caption{Best validation-cost candidate within each family in the confirmatory PJM study.}
\label{tab:pjm_family}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrr}
\toprule
Region & Family & Lag & $K$ & $D$ & Certificate & Test RMSE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write_text(tables_dir / "pjm_family.tex", text)


def generate_figures(repo_root: Path, source_dir: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    overall = pd.read_csv(source_dir / "structural_ablation_family_overall.csv")
    sel = overall[overall["selection_strategy"] == "validation_rmse"].copy()
    labels, values = [], []
    for family in FAMILY_ORDER:
        row = sel[sel["family"] == family]
        if row.empty:
            continue
        labels.append(FAMILY_LABELS[family].replace("$", ""))
        values.append(float(row.iloc[0]["mean_certificate"]))
    plt.figure(figsize=(7.0, 4.2))
    plt.bar(labels, values)
    plt.ylabel("Mean certificate")
    plt.xlabel("Model family")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "structural_certificate_by_family.png", dpi=220, bbox_inches="tight")
    plt.close()

    pjm = pd.read_csv(repo_root / "results/confirmatory/pjm_case_v3_4/pjm_deployment_decisions.csv")
    regions = [str(x).upper() for x in pjm["region"]]
    rmse = 100.0 * pjm["test_rmse_improvement_vs_fallback"].astype(float)
    cost = 100.0 * pjm["test_cost_improvement_vs_fallback"].astype(float)
    x = list(range(len(regions)))
    width = 0.36
    plt.figure(figsize=(7.0, 4.2))
    plt.bar([i - width / 2 for i in x], rmse, width=width, label="RMSE")
    plt.bar([i + width / 2 for i in x], cost, width=width, label="Asymmetric cost")
    plt.axhline(0.0, linewidth=0.8)
    plt.ylabel("Improvement over fallback (%)")
    plt.xticks(x, regions)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "pjm_test_improvements.png", dpi=220, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate V4 manuscript tables and figures from frozen result summaries.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--paper-root", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    paper_root = (args.paper_root or (repo_root / "paper")).resolve()
    source_dir = paper_root / "source_data"
    tables_dir = paper_root / "tables"
    figures_dir = paper_root / "figures"

    for required in [
        source_dir / "development_v3_summary.csv",
        source_dir / "structural_ablation_family_overall.csv",
        source_dir / "structural_ablation_comparison_summary.csv",
        repo_root / "results/development/energy_case_v3/energy_deployment_decisions.csv",
        repo_root / "results/confirmatory/pjm_case_v3_4/pjm_deployment_decisions.csv",
        repo_root / "results/confirmatory/pjm_case_v3_4/pjm_family_results.csv",
    ]:
        if not required.exists():
            raise FileNotFoundError(required)

    pairwise_summary = generate_structural_tables(source_dir, tables_dir)
    generate_process_table(source_dir, tables_dir)
    generate_energy_table(repo_root, tables_dir)
    generate_pjm_family_table(repo_root, tables_dir)
    generate_figures(repo_root, source_dir, figures_dir)

    snapshot = {
        "generator": "tools/generate_manuscript_assets_v4.py",
        "source_files": [
            "paper/source_data/development_v3_summary.csv",
            "paper/source_data/structural_ablation_family_overall.csv",
            "paper/source_data/structural_ablation_comparison_summary.csv",
            "results/development/energy_case_v3/energy_deployment_decisions.csv",
            "results/confirmatory/pjm_case_v3_4/pjm_deployment_decisions.csv",
            "results/confirmatory/pjm_case_v3_4/pjm_family_results.csv",
        ],
        "generated_tables": sorted(p.name for p in tables_dir.glob("*.tex")),
        "generated_figures": sorted(p.name for p in figures_dir.glob("*.png")),
        "pairwise_summary": pairwise_summary,
    }
    write_text(paper_root / "artifacts/manuscript_asset_generation_v4.json", json.dumps(snapshot, indent=2))
    print(f"Generated manuscript assets in: {paper_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
