"""Secondary post-hoc SARIMA and ETS benchmarks for the energy studies.

This script is deliberately isolated from the frozen PJM confirmatory runner.
It never edits the gate, candidate files, decisions, certificates, or lock
artifacts. Hyperparameters are selected on the original validation segment;
selected models are re-estimated on all pre-test observations and evaluated by
rolling one-step-ahead forecasts that update the state with each observed target
without parameter refitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pacbayes_tsk.data.splits_v3 import ratio_split


@dataclass(frozen=True)
class Metrics:
    rmse: float
    mae: float
    weighted_cost: float


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metrics(actual: np.ndarray, forecast: np.ndarray, *, under: float, over: float) -> Metrics:
    actual = np.asarray(actual, float)
    forecast = np.asarray(forecast, float)
    if actual.shape != forecast.shape or actual.ndim != 1 or len(actual) == 0:
        raise ValueError("Invalid metric arrays.")
    error = actual - forecast
    rmse = float(np.sqrt(np.mean(error * error)))
    mae = float(np.mean(np.abs(error)))
    weighted = np.where(error >= 0.0, under * error, over * (-error))
    return Metrics(rmse=rmse, mae=mae, weighted_cost=float(np.mean(weighted)))


def _improvement(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / max(abs(baseline), 1e-12))


def _rolling_forecast(results: Any, observed: np.ndarray) -> np.ndarray:
    """Batch-compute rolling one-step predictions with fixed parameters.

    Appending all realized observations and requesting non-dynamic predictions is
    algebraically identical to forecasting one step and appending the target at
    every origin, but is much faster than a Python loop.
    """
    observed = np.asarray(observed, float)
    n_train = int(results.nobs)
    updated = results.append(observed, refit=False)
    preds = np.asarray(
        updated.get_prediction(
            start=n_train, end=n_train + len(observed) - 1, dynamic=False
        ).predicted_mean,
        float,
    ).reshape(-1)
    if len(preds) != len(observed) or not np.all(np.isfinite(preds)):
        raise RuntimeError("Non-finite rolling one-step forecast.")
    return preds


def _fit_sarima(history: np.ndarray, spec: dict[str, Any], period: int):
    order = tuple(int(x) for x in spec["order"])
    seasonal = tuple(int(x) for x in spec["seasonal_order"]) + (int(period),)
    model = SARIMAX(
        np.asarray(history, float),
        order=order,
        seasonal_order=seasonal,
        trend="c" if order[1] == 0 and seasonal[1] == 0 else "n",
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = model.fit(disp=False, maxiter=250)
    converged = bool(getattr(results, "mle_retvals", {}).get("converged", True))
    warning_text = " | ".join(str(w.message) for w in caught if issubclass(w.category, Warning))
    return results, converged, warning_text


def _fit_ets(history: np.ndarray, spec: dict[str, Any], period: int):
    trend = bool(spec["trend"])
    damped = bool(spec["damped_trend"])
    seasonal_period = int(period) if bool(spec["seasonal"]) else None
    model = ExponentialSmoothing(
        np.asarray(history, float),
        trend=trend,
        damped_trend=damped,
        seasonal=seasonal_period,
        initialization_method="estimated",
        concentrate_scale=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = model.fit(disp=False, maxiter=250)
    converged = bool(getattr(results, "mle_retvals", {}).get("converged", True))
    warning_text = " | ".join(str(w.message) for w in caught if issubclass(w.category, Warning))
    return results, converged, warning_text


def _candidate_rows(
    *,
    values: np.ndarray,
    labels: np.ndarray,
    period: int,
    sarima_specs: Iterable[dict[str, Any]],
    ets_specs: Iterable[dict[str, Any]],
    under: float,
    over: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    prevalidation = values[np.isin(labels, ["prior", "bound"])]
    validation = values[labels == "validation"]
    rows: list[dict[str, Any]] = []
    forecasts: dict[str, np.ndarray] = {}

    for family, specs, fitter in (
        ("SARIMA", sarima_specs, _fit_sarima),
        ("ETS", ets_specs, _fit_ets),
    ):
        for spec in specs:
            started = time.perf_counter()
            row: dict[str, Any] = {
                "family": family,
                "candidate": str(spec["name"]),
                "status": "ok",
                "error": "",
                "converged": False,
                "warnings": "",
            }
            try:
                fitted, converged, warning_text = fitter(prevalidation, spec, period)
                prediction = _rolling_forecast(fitted, validation)
                score = _metrics(validation, prediction, under=under, over=over)
                row.update(
                    converged=converged,
                    warnings=warning_text,
                    aic=float(fitted.aic),
                    bic=float(fitted.bic),
                    validation_rmse=score.rmse,
                    validation_mae=score.mae,
                    validation_weighted_cost=score.weighted_cost,
                )
                forecasts[f"{family}|{spec['name']}"] = prediction
            except Exception as exc:  # keep the audit complete
                row.update(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    aic=math.nan,
                    bic=math.nan,
                    validation_rmse=math.nan,
                    validation_mae=math.nan,
                    validation_weighted_cost=math.nan,
                )
            row["runtime_seconds"] = float(time.perf_counter() - started)
            row["spec_json"] = json.dumps(spec, sort_keys=True)
            rows.append(row)
    return pd.DataFrame(rows), forecasts


def _selection_key(row: pd.Series, primary: str) -> tuple[float, ...]:
    if primary == "validation_weighted_cost":
        return (
            float(row["validation_weighted_cost"]),
            float(row["validation_rmse"]),
            float(row["aic"]),
        )
    if primary == "validation_rmse":
        return (
            float(row["validation_rmse"]),
            float(row["validation_weighted_cost"]),
            float(row["aic"]),
        )
    raise ValueError(f"Unknown selection primary: {primary}")


def _select_family(candidates: pd.DataFrame, family: str, primary: str) -> pd.Series:
    pool = candidates[(candidates["family"] == family) & (candidates["status"] == "ok")].copy()
    if pool.empty:
        raise RuntimeError(f"No successful {family} candidate.")
    index = min(pool.index, key=lambda i: _selection_key(pool.loc[i], primary))
    return pool.loc[index]


def _fit_selected(history: np.ndarray, selected: pd.Series, period: int):
    spec = json.loads(str(selected["spec_json"]))
    if selected["family"] == "SARIMA":
        return _fit_sarima(history, spec, period)[0]
    if selected["family"] == "ETS":
        return _fit_ets(history, spec, period)[0]
    raise ValueError("Unknown selected family.")


def _existing_prediction_path(root: Path, dataset: str, series: str, directory: str) -> Path:
    filename = f"{series}_test_predictions.csv"
    path = root / directory / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path



def _run_one_series(
    root: Path,
    name: str,
    cfg: dict[str, Any],
    shared: dict[str, Any],
    out: Path,
    series: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = root / cfg["data_path"]
    data = pd.read_csv(data_path)
    if "timestamp" not in data:
        raise ValueError(f"{name}: timestamp column missing")
    timestamps = pd.to_datetime(data["timestamp"], errors="raise")
    fractions = {k: float(v) for k, v in shared["split_fractions"].items()}
    split = ratio_split(len(data), fractions)
    period = int(cfg["seasonal_period"])
    under = float(shared["underforecast_weight"])
    over = float(shared["overforecast_weight"])
    values = data[str(series)].to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}/{series}: non-finite values")

    candidates, _ = _candidate_rows(
        values=values,
        labels=split.labels,
        period=period,
        sarima_specs=shared["sarima_candidates"],
        ets_specs=shared["ets_candidates"],
        under=under,
        over=over,
    )
    candidates.insert(0, "dataset", name)
    candidates.insert(1, "series", series)
    selected_rows = {
        family: _select_family(candidates, family, str(cfg["selection_primary"]))
        for family in ("SARIMA", "ETS")
    }
    pretest = values[split.labels != "test"]
    test = values[split.labels == "test"]
    test_timestamps = timestamps[split.labels == "test"].reset_index(drop=True)
    existing_path = _existing_prediction_path(
        root, name, str(series), str(cfg["existing_predictions_dir"])
    )
    existing = pd.read_csv(existing_path)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], errors="raise")
    if len(existing) != len(test) or not np.allclose(existing["actual"], test):
        raise RuntimeError(f"{name}/{series}: existing test predictions do not align")
    if not np.array_equal(existing["timestamp"].to_numpy(), test_timestamps.to_numpy()):
        raise RuntimeError(f"{name}/{series}: timestamps do not align")

    baseline = existing[str(cfg["existing_fallback_column"])].to_numpy(float)
    deployed = existing[str(cfg["existing_deployed_column"])].to_numpy(float)
    baseline_metrics = _metrics(test, baseline, under=under, over=over)
    deployed_metrics = _metrics(test, deployed, under=under, over=over)
    output_predictions = existing.copy()
    summary_rows: list[dict[str, Any]] = []
    for family, selected in selected_rows.items():
        fitted = _fit_selected(pretest, selected, period)
        prediction = _rolling_forecast(fitted, test)
        output_predictions[f"{family.lower()}_posthoc_forecast"] = prediction
        score = _metrics(test, prediction, under=under, over=over)
        summary_rows.append(
            {
                "dataset": name,
                "series": series,
                "analysis_status": cfg["posthoc_label"],
                "family": family,
                "selected_candidate": selected["candidate"],
                "selection_primary": cfg["selection_primary"],
                "validation_rmse": float(selected["validation_rmse"]),
                "validation_mae": float(selected["validation_mae"]),
                "validation_weighted_cost": float(selected["validation_weighted_cost"]),
                "test_rmse": score.rmse,
                "test_mae": score.mae,
                "test_weighted_cost": score.weighted_cost,
                "test_rmse_improvement_vs_fallback": _improvement(baseline_metrics.rmse, score.rmse),
                "test_cost_improvement_vs_fallback": _improvement(baseline_metrics.weighted_cost, score.weighted_cost),
                "test_rmse_improvement_vs_deployed": _improvement(deployed_metrics.rmse, score.rmse),
                "test_cost_improvement_vs_deployed": _improvement(deployed_metrics.weighted_cost, score.weighted_cost),
                "fallback_rmse": baseline_metrics.rmse,
                "fallback_weighted_cost": baseline_metrics.weighted_cost,
                "deployed_rmse": deployed_metrics.rmse,
                "deployed_weighted_cost": deployed_metrics.weighted_cost,
                "test_count": len(test),
                "seasonal_period": period,
                "rolling_one_step": True,
                "parameters_refit_within_test": False,
                "test_used_for_selection": False,
            }
        )

    series_out = out / "series" / name / str(series)
    series_out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(series_out / "candidates.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(series_out / "summary.csv", index=False)
    output_predictions.to_csv(series_out / "predictions.csv", index=False)
    return candidates, summary


def _collect_outputs(root: Path, cfg: dict[str, Any], out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates_all: list[pd.DataFrame] = []
    summaries_all: list[pd.DataFrame] = []
    audit_datasets: dict[str, Any] = {}
    for dataset_name, dataset_cfg in cfg["datasets"].items():
        dataset_candidates: list[pd.DataFrame] = []
        dataset_summaries: list[pd.DataFrame] = []
        for series in dataset_cfg["series"]:
            series_out = out / "series" / dataset_name / str(series)
            c_path = series_out / "candidates.csv"
            s_path = series_out / "summary.csv"
            if not c_path.exists() or not s_path.exists():
                raise FileNotFoundError(f"Missing worker output for {dataset_name}/{series}")
            dataset_candidates.append(pd.read_csv(c_path))
            dataset_summaries.append(pd.read_csv(s_path))
        candidate_frame = pd.concat(dataset_candidates, ignore_index=True)
        summary_frame = pd.concat(dataset_summaries, ignore_index=True)
        candidate_frame.to_csv(out / f"{dataset_name}_posthoc_candidates.csv", index=False)
        summary_frame.to_csv(out / f"{dataset_name}_posthoc_summary.csv", index=False)
        candidates_all.append(candidate_frame)
        summaries_all.append(summary_frame)
        audit_datasets[dataset_name] = {
            "candidate_rows": int(len(candidate_frame)),
            "successful_candidates": int((candidate_frame["status"] == "ok").sum()),
            "failed_candidates": int((candidate_frame["status"] != "ok").sum()),
            "selected_rows": int(len(summary_frame)),
            "data_sha256": _sha256(root / dataset_cfg["data_path"]),
        }
    return (
        pd.concat(candidates_all, ignore_index=True),
        pd.concat(summaries_all, ignore_index=True),
        audit_datasets,
    )

def _latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        dataset = "PJM" if row["dataset"] == "pjm" else "Tetouan"
        series = str(row["series"]).replace("_", "\\_").upper() if dataset == "PJM" else str(row["series"]).replace("zone_", "Zone~")
        selected_spec = str(row["selected_candidate"]).replace("_", r"\_")
        rows.append(
            f"{dataset} & {series} & {row['family']} & {selected_spec} & "
            f"{row['test_rmse']:.2f} & {100*row['test_rmse_improvement_vs_fallback']:.2f}\\% & "
            f"{100*row['test_rmse_improvement_vs_deployed']:.2f}\\% \\\\"
        )
    content = r"""\begin{table*}[t]
\centering
\caption{Secondary post-hoc SARIMA and ETS benchmarks. Hyperparameters are selected without test outcomes, but the analysis was designed after the original energy test results were opened and does not alter the frozen PJM gate. Positive improvements indicate lower RMSE.}
\label{tab:posthoc_energy}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{ll ll rrr}
\toprule
Dataset & Series & Family & Selected specification & Test RMSE & vs. fallback & vs. deployed \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/v4/posthoc_energy_benchmarks_v4_3.yaml")
    parser.add_argument("--output", default="results/posthoc/v4_3")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--series")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = root / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("status") != "secondary_post_hoc_only":
        raise ValueError("The benchmark must remain explicitly post-hoc.")
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)

    if args.worker:
        if not args.dataset or not args.series:
            raise ValueError("Worker mode requires --dataset and --series.")
        dataset_cfg = cfg["datasets"][args.dataset]
        if args.series not in [str(x) for x in dataset_cfg["series"]]:
            raise ValueError("Series is outside the configured post-hoc benchmark.")
        _run_one_series(root, args.dataset, dataset_cfg, cfg["shared"], out, args.series)
        return

    if not args.aggregate_only:
        raise SystemExit(
            "Run one fresh worker process per series, then call --aggregate-only. "
            "On Windows use tools\\08_run_posthoc_energy_benchmarks_v4_3.cmd."
        )

    _, combined, audit_datasets = _collect_outputs(root, cfg, out)
    combined.to_csv(out / "energy_posthoc_summary_all.csv", index=False)
    _latex_table(combined, root / "paper/tables/posthoc_energy_benchmarks.tex")
    audit: dict[str, Any] = {
        "schema_version": "4.3",
        "status": "secondary_post_hoc_only",
        "confirmatory_gate_modified": False,
        "confirmatory_run_reexecuted": False,
        "test_used_for_hyperparameter_selection": False,
        "config_sha256": _sha256(config_path),
        "datasets": audit_datasets,
        "summary_sha256": _sha256(out / "energy_posthoc_summary_all.csv"),
    }
    (out / "posthoc_energy_benchmarks_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
