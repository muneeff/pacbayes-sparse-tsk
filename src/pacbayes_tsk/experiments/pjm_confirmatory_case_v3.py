"""Independent PJM confirmatory deployment case with a predeclared robust gate.

The gate was redesigned using only the earlier Tetouan development case. PJM
outcomes are not used until every regional pre-test decision has been written.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import yaml

from pacbayes_tsk.data.energy_pjm import REGION_COLUMNS, load_pjm_daily, sha256_file
from pacbayes_tsk.data.splits_v3 import ratio_split
from pacbayes_tsk.experiments.energy_case_study_v3 import (
    _build_designs,
    _candidate_rows_for_zone,
    _lagged,
    _role_masks,
    _ridge_solution,
    _rmse_mae,
    _seasonal_predictions,
    _weighted_cost,
)

ROLES = ("prior", "bound", "validation", "test")


@dataclass(frozen=True)
class PJMConfirmatorySettings:
    regions: tuple[str, ...]
    families: tuple[str, ...]
    lags: tuple[int, ...]
    radii: tuple[float, ...]
    ridge_alphas: tuple[float, ...]
    max_rules: int
    split_fractions: dict[str, float]
    temperatures: tuple[float, ...]
    prior_scales: tuple[float, ...]
    posterior_ratios: tuple[float, ...]
    delta_total: float
    familywise_series_count: int
    clip_multiplier: float
    certificate_threshold: float
    clipping_rate_maximum: float
    seasonal_lag: int
    underforecast_weight: float
    overforecast_weight: float
    minimum_rmse_improvement: float
    minimum_cost_improvement: float
    validation_blocks: int
    minimum_nonnegative_blocks: int
    maximum_worst_block_degradation: float

    @property
    def zones(self) -> tuple[str, ...]:
        """Compatibility alias for the shared candidate builder."""
        return self.regions

    @property
    def familywise_delta(self) -> float:
        return self.delta_total / self.familywise_series_count

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PJMConfirmatorySettings":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        fractions = cfg["split"]["fractions"]
        gate = cfg["decision"]["robust_gate"]
        settings = cls(
            regions=tuple(str(x) for x in cfg["preprocessing"]["target_series"]),
            families=tuple(str(x) for x in cfg["model"]["families"]),
            lags=tuple(int(x) for x in cfg["model"]["lags"]),
            radii=tuple(float(x) for x in cfg["model"]["radii"]),
            ridge_alphas=tuple(float(x) for x in cfg["model"]["ridge_alphas"]),
            max_rules=int(cfg["model"]["max_rules"]),
            split_fractions={role: float(fractions[role]) for role in ROLES},
            temperatures=tuple(float(x) for x in cfg["certificate"]["temperatures"]),
            prior_scales=tuple(float(x) for x in cfg["certificate"]["prior_scales"]),
            posterior_ratios=tuple(float(x) for x in cfg["certificate"]["posterior_ratios"]),
            delta_total=float(cfg["certificate"]["delta_total"]),
            familywise_series_count=int(cfg["certificate"]["familywise_series_count"]),
            clip_multiplier=float(cfg["certificate"]["clip_bound_prior_multiplier"]),
            certificate_threshold=float(gate["certificate_threshold"]),
            clipping_rate_maximum=float(gate["clipping_rate_maximum"]),
            seasonal_lag=int(cfg["selection"]["seasonal_fallback_lag"]),
            underforecast_weight=float(cfg["decision"]["underforecast_weight"]),
            overforecast_weight=float(cfg["decision"]["overforecast_weight"]),
            minimum_rmse_improvement=float(gate["minimum_validation_rmse_improvement"]),
            minimum_cost_improvement=float(gate["minimum_validation_cost_improvement"]),
            validation_blocks=int(gate["validation_blocks"]),
            minimum_nonnegative_blocks=int(gate["minimum_nonnegative_blocks"]),
            maximum_worst_block_degradation=float(gate["maximum_worst_block_degradation"]),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.regions != REGION_COLUMNS:
            raise ValueError("The PJM regions and their order must remain predeclared.")
        if self.familywise_series_count != len(self.regions):
            raise ValueError("Familywise count must equal the number of PJM regions.")
        if self.families != ("ridge", "fixed_k_dense_tsk", "dense_tsk"):
            raise ValueError("Unexpected PJM model families.")
        if abs(sum(self.split_fractions.values()) - 1.0) > 1e-12:
            raise ValueError("Split fractions must sum to one.")
        if self.validation_blocks < 2:
            raise ValueError("At least two validation blocks are required.")
        if not 1 <= self.minimum_nonnegative_blocks <= self.validation_blocks:
            raise ValueError("Invalid block-stability requirement.")
        if not 0 < self.certificate_threshold < 1:
            raise ValueError("Invalid certificate threshold.")
        if not 0 <= self.clipping_rate_maximum < 1:
            raise ValueError("Invalid clipping threshold.")
        if not 0 <= self.minimum_rmse_improvement < 1:
            raise ValueError("Invalid RMSE improvement threshold.")
        if not 0 <= self.minimum_cost_improvement < 1:
            raise ValueError("Invalid cost improvement threshold.")
        if not 0 <= self.maximum_worst_block_degradation < 1:
            raise ValueError("Invalid worst-block degradation allowance.")


def _predict_candidate_role(
    *,
    values_original: np.ndarray,
    labels: np.ndarray,
    row: pd.Series,
    settings: PJMConfirmatorySettings,
    role: str,
) -> dict[str, Any]:
    if role not in {"validation", "test"}:
        raise ValueError("Only validation or test prediction is supported.")
    prior_raw = values_original[labels == "prior"]
    mean = float(np.mean(prior_raw))
    std = float(np.std(prior_raw, ddof=0))
    if std <= 1e-12:
        raise ValueError("Invalid prior-only scale.")
    values = (values_original - mean) / std
    lag = int(row["lag"])
    features, target, target_labels, target_indices = _lagged(values, labels, lag)
    masks = _role_masks(target_labels)
    radius = None if pd.isna(row["radius"]) else float(row["radius"])
    designs, _, _, cap_reached = _build_designs(
        family=str(row["family"]),
        features=features,
        masks=masks,
        settings=settings,
        radius=radius,
    )
    if cap_reached:
        raise RuntimeError("A selected candidate unexpectedly reached the rule cap.")
    posterior_mean, _ = _ridge_solution(
        designs["bound"], target[masks["bound"]], float(row["ridge_alpha"])
    )
    prediction_scaled_raw = designs[role] @ posterior_mean
    clip_bound = float(row["clip_bound_scaled"])
    prediction_scaled_clipped = np.clip(prediction_scaled_raw, -clip_bound, clip_bound)
    target_scaled = target[masks[role]]
    prediction_raw = prediction_scaled_raw * std + mean
    prediction_clipped = prediction_scaled_clipped * std + mean
    target_original = target_scaled * std + mean
    indices = target_indices[masks[role]]
    rmse_raw, mae_raw = _rmse_mae(target_original, prediction_raw)
    rmse_clipped, mae_clipped = _rmse_mae(target_original, prediction_clipped)
    cost, normalized_cost = _weighted_cost(
        target_original,
        prediction_clipped,
        under_weight=settings.underforecast_weight,
        over_weight=settings.overforecast_weight,
    )
    return {
        "indices": indices,
        "target": target_original,
        "prediction_raw": prediction_raw,
        "prediction_clipped": prediction_clipped,
        f"{role}_rmse_raw": rmse_raw,
        f"{role}_mae_raw": mae_raw,
        f"{role}_rmse_clipped": rmse_clipped,
        f"{role}_mae_clipped": mae_clipped,
        f"{role}_weighted_cost": cost,
        f"{role}_normalized_weighted_cost": normalized_cost,
        f"{role}_target_clipping_rate": float(np.mean(np.abs(target_scaled) > clip_bound)),
        f"{role}_forecast_clipping_rate": float(np.mean(np.abs(prediction_scaled_raw) > clip_bound)),
    }


def _block_stability(
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    blocks: int,
    under_weight: float,
    over_weight: float,
) -> dict[str, Any]:
    indices = np.array_split(np.arange(len(target)), blocks)
    rmse_improvements: list[float] = []
    cost_improvements: list[float] = []
    for block in indices:
        if len(block) == 0:
            raise ValueError("Validation block is empty.")
        base_rmse, _ = _rmse_mae(target[block], baseline[block])
        cand_rmse, _ = _rmse_mae(target[block], candidate[block])
        base_cost, _ = _weighted_cost(
            target[block], baseline[block],
            under_weight=under_weight, over_weight=over_weight,
        )
        cand_cost, _ = _weighted_cost(
            target[block], candidate[block],
            under_weight=under_weight, over_weight=over_weight,
        )
        rmse_improvements.append((base_rmse - cand_rmse) / max(base_rmse, 1e-12))
        cost_improvements.append((base_cost - cand_cost) / max(base_cost, 1e-12))
    return {
        "validation_block_rmse_improvements": rmse_improvements,
        "validation_block_cost_improvements": cost_improvements,
        "validation_rmse_nonnegative_blocks": int(np.sum(np.asarray(rmse_improvements) >= 0.0)),
        "validation_cost_nonnegative_blocks": int(np.sum(np.asarray(cost_improvements) >= 0.0)),
        "validation_worst_block_rmse_improvement": float(np.min(rmse_improvements)),
        "validation_worst_block_cost_improvement": float(np.min(cost_improvements)),
    }


def _deployment_selection_key(row: pd.Series) -> tuple[Any, ...]:
    radius = math.inf if pd.isna(row["radius"]) else float(row["radius"])
    return (
        float(row["validation_weighted_cost"]),
        float(row["validation_rmse_clipped"]),
        float(row["certificate_untruncated"]),
        int(row["consequent_dimension"]),
        int(row["rule_count"]),
        str(row["family"]),
        radius,
        float(row["ridge_alpha"]),
        int(row["lag"]),
    )


def _select(pool: pd.DataFrame) -> pd.Series:
    if pool.empty:
        raise ValueError("Cannot select from an empty pool.")
    idx = min(pool.index, key=lambda i: _deployment_selection_key(pool.loc[i]))
    return pool.loc[idx]


def _verify_lock(lock_path: str | Path, paths: dict[str, str | Path]) -> dict[str, Any]:
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    expected = lock.get("sha256", {})
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if expected != actual:
        mismatch = {
            name: {"expected": expected.get(name), "actual": actual.get(name)}
            for name in sorted(set(expected) | set(actual))
            if expected.get(name) != actual.get(name)
        }
        raise RuntimeError(f"PJM confirmatory lock mismatch: {mismatch}")
    return lock


def run_pjm_confirmatory_case(
    *,
    case_data_path: str | Path,
    config_path: str | Path,
    lock_path: str | Path,
    output_dir: str | Path,
    verification_paths: dict[str, str | Path],
) -> dict[str, Any]:
    output = Path(output_dir)
    if (output / "COMPLETED.json").exists():
        raise RuntimeError("The authorized PJM confirmatory run already completed.")
    settings = PJMConfirmatorySettings.from_yaml(config_path)
    lock = _verify_lock(lock_path, verification_paths)
    case_data = load_pjm_daily(case_data_path)
    split = ratio_split(len(case_data), settings.split_fractions)
    started = time.perf_counter()

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="pjm_confirmatory_v3_", dir=str(output.parent)))
    try:
        all_candidates: list[pd.DataFrame] = []
        pretest_decisions: list[dict[str, Any]] = []
        region_context: dict[str, dict[str, Any]] = {}
        predictions_dir = temp_root / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: candidate construction and all pre-test decisions.
        for region in settings.regions:
            values = case_data[region].to_numpy(float)
            candidates, audit = _candidate_rows_for_zone(
                zone=region,
                values_original=values,
                labels=split.labels,
                settings=settings,
            )
            val_idx, val_target, val_seasonal = _seasonal_predictions(
                values, split.labels, "validation", settings.seasonal_lag
            )
            seasonal_rmse, seasonal_mae = _rmse_mae(val_target, val_seasonal)
            seasonal_cost, seasonal_norm_cost = _weighted_cost(
                val_target, val_seasonal,
                under_weight=settings.underforecast_weight,
                over_weight=settings.overforecast_weight,
            )

            augmented_rows: list[dict[str, Any]] = []
            for _, row in candidates.iterrows():
                forecast = _predict_candidate_role(
                    values_original=values,
                    labels=split.labels,
                    row=row,
                    settings=settings,
                    role="validation",
                )
                if not np.array_equal(forecast["indices"], val_idx):
                    raise RuntimeError("Candidate and seasonal validation indices are not aligned.")
                stability = _block_stability(
                    val_target,
                    forecast["prediction_clipped"],
                    val_seasonal,
                    blocks=settings.validation_blocks,
                    under_weight=settings.underforecast_weight,
                    over_weight=settings.overforecast_weight,
                )
                enriched = row.to_dict()
                enriched.update({
                    key: value for key, value in forecast.items()
                    if key not in {"indices", "target", "prediction_raw", "prediction_clipped"}
                })
                enriched.update(stability)
                enriched["seasonal_validation_rmse"] = seasonal_rmse
                enriched["seasonal_validation_mae"] = seasonal_mae
                enriched["seasonal_validation_weighted_cost"] = seasonal_cost
                enriched["seasonal_validation_normalized_weighted_cost"] = seasonal_norm_cost
                enriched["validation_rmse_improvement_vs_fallback"] = (
                    seasonal_rmse - float(forecast["validation_rmse_clipped"])
                ) / max(seasonal_rmse, 1e-12)
                enriched["validation_cost_improvement_vs_fallback"] = (
                    seasonal_cost - float(forecast["validation_weighted_cost"])
                ) / max(seasonal_cost, 1e-12)
                augmented_rows.append(enriched)

            augmented = pd.DataFrame(augmented_rows)
            eligible_mask = (
                (augmented["certificate_untruncated"] <= settings.certificate_threshold)
                & (augmented["certification_target_clipping_rate"] <= settings.clipping_rate_maximum)
                & (augmented["validation_rmse_improvement_vs_fallback"] >= settings.minimum_rmse_improvement)
                & (augmented["validation_cost_improvement_vs_fallback"] >= settings.minimum_cost_improvement)
                & (augmented["validation_rmse_nonnegative_blocks"] >= settings.minimum_nonnegative_blocks)
                & (augmented["validation_cost_nonnegative_blocks"] >= settings.minimum_nonnegative_blocks)
                & (
                    augmented["validation_worst_block_rmse_improvement"]
                    >= -settings.maximum_worst_block_degradation
                )
                & (
                    augmented["validation_worst_block_cost_improvement"]
                    >= -settings.maximum_worst_block_degradation
                )
            )
            eligible = augmented[eligible_mask]
            selected = None if eligible.empty else _select(eligible)
            deployment = "seasonal_naive_7d" if selected is None else str(selected["family"])
            reason = (
                "no_candidate_passed_robust_gate"
                if selected is None
                else "robust_certificate_validation_gate_passed"
            )
            decision = {
                "region": region,
                "decision_made_before_test_opening": True,
                "deployment": deployment,
                "reason": reason,
                "eligible_candidate_count": int(len(eligible)),
                "seasonal_validation_rmse": seasonal_rmse,
                "seasonal_validation_mae": seasonal_mae,
                "seasonal_validation_weighted_cost": seasonal_cost,
                "selected_candidate_id": None if selected is None else str(selected["candidate_id"]),
                "selected_family": None if selected is None else str(selected["family"]),
                "selected_certificate": None if selected is None else float(selected["certificate"]),
                "selected_validation_rmse": None if selected is None else float(selected["validation_rmse_clipped"]),
                "selected_validation_cost": None if selected is None else float(selected["validation_weighted_cost"]),
                "selected_validation_rmse_improvement": None if selected is None else float(selected["validation_rmse_improvement_vs_fallback"]),
                "selected_validation_cost_improvement": None if selected is None else float(selected["validation_cost_improvement_vs_fallback"]),
            }
            (temp_root / f"{region}_pretest_decision.json").write_text(
                json.dumps(decision, indent=2), encoding="utf-8"
            )
            augmented.to_csv(temp_root / f"{region}_candidates.csv", index=False)
            all_candidates.append(augmented)
            pretest_decisions.append(decision)
            region_context[region] = {
                "values": values,
                "selected": selected,
                "audit": audit,
            }

        pretest_frame = pd.DataFrame(pretest_decisions)
        pretest_frame.to_csv(temp_root / "pjm_pretest_decisions.csv", index=False)
        pretest_hash = sha256_file(temp_root / "pjm_pretest_decisions.csv")

        # Phase 2: open test only after all regional decisions are serialized.
        final_decisions: list[dict[str, Any]] = []
        family_results: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for region in settings.regions:
            values = region_context[region]["values"]
            selected = region_context[region]["selected"]
            candidates = all_candidates[list(settings.regions).index(region)]
            test_idx, test_target, test_seasonal = _seasonal_predictions(
                values, split.labels, "test", settings.seasonal_lag
            )
            baseline_rmse, baseline_mae = _rmse_mae(test_target, test_seasonal)
            baseline_cost, baseline_norm_cost = _weighted_cost(
                test_target, test_seasonal,
                under_weight=settings.underforecast_weight,
                over_weight=settings.overforecast_weight,
            )

            family_forecasts: dict[str, dict[str, Any]] = {}
            for family in settings.families:
                winner = _select(candidates[candidates["family"] == family])
                fc = _predict_candidate_role(
                    values_original=values,
                    labels=split.labels,
                    row=winner,
                    settings=settings,
                    role="test",
                )
                if not np.array_equal(fc["indices"], test_idx):
                    raise RuntimeError("Family test forecasts are not aligned.")
                family_forecasts[family] = fc
                family_results.append({
                    "region": region,
                    "family": family,
                    "candidate_id": str(winner["candidate_id"]),
                    "lag": int(winner["lag"]),
                    "radius": winner["radius"],
                    "ridge_alpha": float(winner["ridge_alpha"]),
                    "rule_count": int(winner["rule_count"]),
                    "consequent_dimension": int(winner["consequent_dimension"]),
                    "certificate": float(winner["certificate"]),
                    "certificate_untruncated": float(winner["certificate_untruncated"]),
                    "total_kl": float(winner["total_kl"]),
                    "validation_rmse": float(winner["validation_rmse_clipped"]),
                    "validation_weighted_cost": float(winner["validation_weighted_cost"]),
                    **{
                        key: value for key, value in fc.items()
                        if key not in {"indices", "target", "prediction_raw", "prediction_clipped"}
                    },
                })

            if selected is None:
                deployed_prediction = test_seasonal
                deployed_rmse, deployed_mae = baseline_rmse, baseline_mae
                deployed_cost, deployed_norm_cost = baseline_cost, baseline_norm_cost
            else:
                deployed = _predict_candidate_role(
                    values_original=values,
                    labels=split.labels,
                    row=selected,
                    settings=settings,
                    role="test",
                )
                deployed_prediction = deployed["prediction_clipped"]
                deployed_rmse = float(deployed["test_rmse_clipped"])
                deployed_mae = float(deployed["test_mae_clipped"])
                deployed_cost = float(deployed["test_weighted_cost"])
                deployed_norm_cost = float(deployed["test_normalized_weighted_cost"])

            base_decision = pretest_frame[pretest_frame["region"] == region].iloc[0].to_dict()
            final_decisions.append({
                **base_decision,
                "test_opened_after_all_decisions": True,
                "test_baseline_rmse": baseline_rmse,
                "test_baseline_mae": baseline_mae,
                "test_baseline_weighted_cost": baseline_cost,
                "test_baseline_normalized_weighted_cost": baseline_norm_cost,
                "test_deployed_rmse": deployed_rmse,
                "test_deployed_mae": deployed_mae,
                "test_deployed_weighted_cost": deployed_cost,
                "test_deployed_normalized_weighted_cost": deployed_norm_cost,
                "test_rmse_improvement_vs_fallback": (baseline_rmse - deployed_rmse) / max(baseline_rmse, 1e-12),
                "test_cost_improvement_vs_fallback": (baseline_cost - deployed_cost) / max(baseline_cost, 1e-12),
            })
            prediction_frame = pd.DataFrame({
                "timestamp": case_data["timestamp"].iloc[test_idx].to_numpy(),
                "actual": test_target,
                "seasonal_naive_7d": test_seasonal,
                "deployed_forecast": deployed_prediction,
            })
            for family, fc in family_forecasts.items():
                prediction_frame[f"{family}_forecast"] = fc["prediction_clipped"]
            prediction_frame.to_csv(predictions_dir / f"{region}_test_predictions.csv", index=False)
            audit = dict(region_context[region]["audit"])
            audit.update({
                "split_counts": split.counts,
                "familywise_delta": settings.familywise_delta,
                "candidate_selection_uses_test": False,
                "certificate_uses_test": False,
                "deployment_decision_uses_test": False,
                "all_region_decisions_serialized_before_test": True,
            })
            audits.append(audit)

        candidates_all = pd.concat(all_candidates, ignore_index=True)
        decisions = pd.DataFrame(final_decisions)
        family_frame = pd.DataFrame(family_results)
        candidates_all.to_csv(temp_root / "pjm_candidates_all.csv", index=False)
        decisions.to_csv(temp_root / "pjm_deployment_decisions.csv", index=False)
        family_frame.to_csv(temp_root / "pjm_family_results.csv", index=False)

        aggregate = {
            "phase": "confirmatory_real_energy_case",
            "confirmatory": True,
            "case_study": "pjm_four_region_daily_load",
            "regions": list(settings.regions),
            "completed_regions": len(settings.regions),
            "total_candidates": int(len(candidates_all)),
            "all_candidate_selection_excluded_test": bool(candidates_all["selection_uses_test"].eq(False).all()),
            "all_certificates_excluded_test": bool(candidates_all["certificate_uses_test"].eq(False).all()),
            "all_decisions_preceded_test": bool(decisions["decision_made_before_test_opening"].all()),
            "pretest_decisions_sha256": pretest_hash,
            "deployed_model_regions": int(np.sum(decisions["deployment"] != "seasonal_naive_7d")),
            "mean_test_rmse_improvement_vs_fallback": float(decisions["test_rmse_improvement_vs_fallback"].mean()),
            "mean_test_cost_improvement_vs_fallback": float(decisions["test_cost_improvement_vs_fallback"].mean()),
            "runtime_seconds": float(time.perf_counter() - started),
            "lock_sha256": sha256_file(lock_path),
            "lock_created_utc": lock.get("created_utc"),
            "data_sha256": sha256_file(case_data_path),
            "region_audits": audits,
        }
        (temp_root / "pjm_confirmatory_audit.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        completion = {
            "status": "completed",
            "single_authorized_run": True,
            "outcomes_published_atomically": True,
            "runtime_seconds": aggregate["runtime_seconds"],
            "audit_sha256": sha256_file(temp_root / "pjm_confirmatory_audit.json"),
            "decisions_sha256": sha256_file(temp_root / "pjm_deployment_decisions.csv"),
            "family_results_sha256": sha256_file(temp_root / "pjm_family_results.csv"),
        }
        (temp_root / "COMPLETED.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        temp_root.replace(output)
        return aggregate
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
