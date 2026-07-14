"""Predeclared real-energy case study for PAC-Bayesian deployment gating.

The test segment is never used for candidate construction, certificate
selection, or the deployment decision. Test forecasts are computed only after
all per-zone decisions have been serialized inside the atomic run directory.
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

from pacbayes_tsk.data.energy_tetouan import (
    CLEAN_ZONE_COLUMNS,
    load_two_hour_tetouan,
    sha256_file,
)
from pacbayes_tsk.data.splits_v3 import ratio_split
from pacbayes_tsk.models.sparse_tsk import (
    fit_fixed_k_antecedent,
    fit_radius_antecedent,
)
from pacbayes_tsk.pac_bayes.certificates_v3 import martingale_certificate
from pacbayes_tsk.pac_bayes.priors_v3 import HierarchicalModelPrior, ModelIndex

ROLES = ("prior", "bound", "validation", "test")


@dataclass(frozen=True)
class EnergySettings:
    zones: tuple[str, ...]
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
    require_no_worse_than_seasonal: bool
    underforecast_weight: float
    overforecast_weight: float

    @property
    def familywise_delta(self) -> float:
        return self.delta_total / self.familywise_series_count

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnergySettings":
        with Path(path).open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        fractions = cfg["split"]["fractions"]
        settings = cls(
            zones=tuple(str(x) for x in cfg["preprocessing"]["target_series"]),
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
            certificate_threshold=float(cfg["decision"]["certificate_threshold"]),
            clipping_rate_maximum=float(cfg["decision"]["clipping_rate_maximum"]),
            seasonal_lag=int(cfg["selection"]["seasonal_fallback_lag"]),
            require_no_worse_than_seasonal=bool(
                cfg["selection"]["require_no_worse_than_seasonal_validation"]
            ),
            underforecast_weight=float(cfg["decision"]["underforecast_weight"]),
            overforecast_weight=float(cfg["decision"]["overforecast_weight"]),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if set(self.zones) != set(CLEAN_ZONE_COLUMNS):
            raise ValueError("The frozen energy case must use all three Tetouan zones.")
        if self.familywise_series_count != len(self.zones):
            raise ValueError("Familywise count must equal the number of zones.")
        if self.families != ("ridge", "fixed_k_dense_tsk", "dense_tsk"):
            raise ValueError("Unexpected energy-case family support.")
        if any(x <= 0 for x in self.lags + self.radii + self.ridge_alphas):
            raise ValueError("Model grids must be positive.")
        if self.max_rules < 2 or self.seasonal_lag < 1:
            raise ValueError("Invalid rule cap or seasonal lag.")
        if not 0 < self.delta_total < 1:
            raise ValueError("Invalid confidence level.")
        if self.clip_multiplier < 1:
            raise ValueError("The prior-fixed clipping multiplier must be at least one.")
        if not 0 < self.certificate_threshold < 1:
            raise ValueError("Invalid deployment certificate threshold.")
        if not 0 <= self.clipping_rate_maximum < 1:
            raise ValueError("Invalid clipping threshold.")


@dataclass(frozen=True)
class CandidateSpec:
    family: str
    lag: int
    radius: float | None
    ridge_alpha: float
    rule_count: int

    @property
    def candidate_id(self) -> str:
        radius = "none" if self.radius is None else f"{self.radius:.8g}"
        return (
            f"{self.family}|p={self.lag}|r={radius}|"
            f"a={self.ridge_alpha:.8g}|k={self.rule_count}"
        )


def _lagged(values: np.ndarray, labels: np.ndarray, lag: int):
    targets = np.arange(lag, len(values))
    features = np.column_stack(
        [values[targets - step] for step in range(1, lag + 1)]
    )
    return features, values[targets], labels[targets], targets


def _role_masks(target_labels: np.ndarray) -> dict[str, np.ndarray]:
    return {role: target_labels == role for role in ROLES}


def _ridge_design(features: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(features), dtype=float), features])


def _ridge_solution(design: np.ndarray, target: np.ndarray, alpha: float):
    """Reference single-alpha solver retained for tests and test refits."""
    gram = design.T @ design
    regularized = gram + float(alpha) * np.eye(gram.shape[0])
    rhs = design.T @ target
    try:
        coefficients = np.linalg.solve(regularized, rhs)
        inverse = np.linalg.inv(regularized)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(regularized)
        coefficients = inverse @ rhs
    return np.asarray(coefficients, float), np.asarray(np.diag(inverse), float)


def _ridge_path(
    design: np.ndarray,
    target: np.ndarray,
    alphas: tuple[float, ...],
    *,
    include_inverse_diagonal: bool,
) -> dict[float, tuple[np.ndarray, np.ndarray | None]]:
    """Solve an entire ridge grid from one symmetric eigendecomposition.

    This is algebraically equivalent to solving each regularized normal system
    separately, but avoids repeated cubic-cost matrix inversions for the large
    fixed-K TSK designs.
    """
    gram = np.asarray(design.T @ design, dtype=float)
    rhs = np.asarray(design.T @ target, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_rhs = eigenvectors.T @ rhs
    squared_vectors = eigenvectors * eigenvectors
    output: dict[float, tuple[np.ndarray, np.ndarray | None]] = {}
    for alpha in alphas:
        inverse_spectrum = 1.0 / (eigenvalues + float(alpha))
        coefficients = eigenvectors @ (projected_rhs * inverse_spectrum)
        inverse_diagonal = (
            squared_vectors @ inverse_spectrum
            if include_inverse_diagonal
            else None
        )
        output[float(alpha)] = (
            np.asarray(coefficients, dtype=float),
            None
            if inverse_diagonal is None
            else np.asarray(inverse_diagonal, dtype=float),
        )
    return output


def _base_prior_std(design, target, prior_mean, inverse_diagonal):
    residual = target - design @ prior_mean
    residual_variance = max(1e-4, float(np.mean(residual * residual)))
    variance = residual_variance * np.maximum(inverse_diagonal, 0.0)
    return np.sqrt(np.clip(variance, 1e-6, 100.0))


def _gaussian_kl_ratio(q_mean, p_mean, base_std, prior_scale, posterior_ratio):
    dimension = len(q_mean)
    ratio2 = posterior_ratio * posterior_ratio
    location = np.sum(((q_mean - p_mean) / (base_std * prior_scale)) ** 2)
    value = 0.5 * (
        dimension * (math.log(1.0 / ratio2) - 1.0 + ratio2) + location
    )
    return float(max(0.0, value))


def _rmse_mae(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    error = np.asarray(target, float) - np.asarray(prediction, float)
    return float(np.sqrt(np.mean(error * error))), float(np.mean(np.abs(error)))


def _weighted_cost(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    under_weight: float,
    over_weight: float,
) -> tuple[float, float]:
    target = np.asarray(target, float)
    prediction = np.asarray(prediction, float)
    under = np.maximum(target - prediction, 0.0)
    over = np.maximum(prediction - target, 0.0)
    cost = float(np.mean(under_weight * under + over_weight * over))
    denominator = max(1e-12, float(np.mean(np.abs(target))))
    return cost, cost / denominator


def _best_certificate(
    *,
    design_certification: np.ndarray,
    target_certification: np.ndarray,
    q_mean: np.ndarray,
    p_mean: np.ndarray,
    base_std: np.ndarray,
    clip_bound: float,
    structure_kl: float,
    settings: EnergySettings,
) -> dict[str, float]:
    clipped_target = np.clip(target_certification, -clip_bound, clip_bound)
    squared_bias = (clipped_target - design_certification @ q_mean) ** 2
    base_variance = (design_certification * design_certification) @ (
        base_std * base_std
    )
    denominator = 4.0 * clip_bound * clip_bound
    scale_penalty = math.log(len(settings.prior_scales))
    best: dict[str, float] | None = None
    for prior_scale in settings.prior_scales:
        for posterior_ratio in settings.posterior_ratios:
            variance_multiplier = (prior_scale * posterior_ratio) ** 2
            pointwise = np.minimum(
                1.0,
                (squared_bias + variance_multiplier * base_variance) / denominator,
            )
            empirical = float(np.mean(pointwise))
            gaussian_kl = _gaussian_kl_ratio(
                q_mean, p_mean, base_std, prior_scale, posterior_ratio
            )
            total_kl = structure_kl + scale_penalty + gaussian_kl
            result = martingale_certificate(
                empirical_risk=empirical,
                total_kl=total_kl,
                n=len(target_certification),
                delta=settings.familywise_delta,
                temperatures=settings.temperatures,
            )
            row = {
                "prior_scale": float(prior_scale),
                "posterior_ratio": float(posterior_ratio),
                "empirical_gibbs_risk": empirical,
                "gaussian_kl": gaussian_kl,
                "structure_kl": structure_kl,
                "prior_scale_penalty": scale_penalty,
                "total_kl": total_kl,
                "temperature": result.temperature,
                "certificate": result.certificate,
                "certificate_untruncated": result.untruncated,
            }
            if best is None or row["certificate_untruncated"] < best["certificate_untruncated"]:
                best = row
    assert best is not None
    return best


def _build_designs(
    *,
    family: str,
    features: np.ndarray,
    masks: dict[str, np.ndarray],
    settings: EnergySettings,
    radius: float | None,
):
    if family == "ridge":
        designs = {role: _ridge_design(features[masks[role]]) for role in ROLES}
        diagnostics = {
            "effective_rule_count": 1,
            "minimum_hard_rule_support": int(np.sum(masks["prior"])),
            "maximum_hard_rule_support": int(np.sum(masks["prior"])),
            "mean_active_rules": 1.0,
            "maximum_active_rules": 1,
            "rule_usage_entropy": 0.0,
        }
        return designs, diagnostics, 1, False
    if family == "fixed_k_dense_tsk":
        antecedent = fit_fixed_k_antecedent(
            features[masks["prior"]],
            rule_count=settings.max_rules,
            max_active_rules=settings.max_rules,
        )
    elif family == "dense_tsk":
        if radius is None:
            raise ValueError("Radius-controlled TSK requires a radius.")
        antecedent = fit_radius_antecedent(
            features[masks["prior"]],
            radius=radius,
            max_rules=settings.max_rules,
            max_active_rules=settings.max_rules,
        )
    else:
        raise ValueError(f"Unsupported family: {family}")
    designs = {
        role: antecedent.design_matrix(features[masks[role]]) for role in ROLES
    }
    certification_features = np.vstack(
        [features[masks["bound"]], features[masks["validation"]]]
    )
    diagnostics = antecedent.diagnostics(certification_features)
    diagnostics["realized_covering_radius"] = float(antecedent.radius)
    return designs, diagnostics, antecedent.rule_count, antecedent.radius_cap_reached


def _candidate_rows_for_zone(
    *,
    zone: str,
    values_original: np.ndarray,
    labels: np.ndarray,
    settings: EnergySettings,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prior_raw = values_original[labels == "prior"]
    scaling_mean = float(np.mean(prior_raw))
    scaling_std = float(np.std(prior_raw, ddof=0))
    if not np.isfinite(scaling_std) or scaling_std <= 1e-12:
        raise ValueError("Invalid prior-only target scale.")
    values = (values_original - scaling_mean) / scaling_std
    prior = HierarchicalModelPrior(
        settings.families,
        settings.lags,
        settings.radii,
        settings.ridge_alphas,
        settings.max_rules,
    )
    rows: list[dict[str, Any]] = []
    skipped_capped = 0
    build_failures: list[dict[str, Any]] = []
    for lag in settings.lags:
        features, target, target_labels, _ = _lagged(values, labels, lag)
        masks = _role_masks(target_labels)
        targets = {role: target[masks[role]] for role in ROLES}
        for family in settings.families:
            radii: tuple[float | None, ...]
            if family == "dense_tsk":
                radii = settings.radii
            else:
                radii = (None,)
            for radius in radii:
                try:
                    designs, diagnostics, rule_count, cap_reached = _build_designs(
                        family=family,
                        features=features,
                        masks=masks,
                        settings=settings,
                        radius=radius,
                    )
                except Exception as error:
                    build_failures.append(
                        {
                            "family": family,
                            "lag": lag,
                            "radius": radius,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                if cap_reached:
                    skipped_capped += len(settings.ridge_alphas)
                    continue
                prior_path = _ridge_path(
                    designs["prior"],
                    targets["prior"],
                    settings.ridge_alphas,
                    include_inverse_diagonal=True,
                )
                posterior_path = _ridge_path(
                    designs["bound"],
                    targets["bound"],
                    settings.ridge_alphas,
                    include_inverse_diagonal=False,
                )
                for alpha in settings.ridge_alphas:
                    spec = CandidateSpec(family, lag, radius, alpha, rule_count)
                    prior_mean, prior_inverse_diag = prior_path[float(alpha)]
                    posterior_mean, _ = posterior_path[float(alpha)]
                    assert prior_inverse_diag is not None
                    validation_scaled = designs["validation"] @ posterior_mean
                    validation_original = validation_scaled * scaling_std + scaling_mean
                    validation_target_original = (
                        targets["validation"] * scaling_std + scaling_mean
                    )
                    validation_rmse, validation_mae = _rmse_mae(
                        validation_target_original, validation_original
                    )
                    base_std = _base_prior_std(
                        designs["prior"],
                        targets["prior"],
                        prior_mean,
                        prior_inverse_diag,
                    )
                    certification_design = np.vstack(
                        [designs["bound"], designs["validation"]]
                    )
                    certification_target = np.concatenate(
                        [targets["bound"], targets["validation"]]
                    )
                    clip_bound = max(
                        1.0,
                        settings.clip_multiplier
                        * float(np.max(np.abs(targets["prior"]))),
                    )
                    clipping_rate = float(
                        np.mean(np.abs(certification_target) > clip_bound)
                    )
                    model_index = ModelIndex(
                        family=family,
                        lag=lag,
                        ridge_alpha=alpha,
                        radius=radius,
                        rule_count=rule_count,
                    )
                    structure_kl = prior.negative_log_mass(model_index)
                    certificate = _best_certificate(
                        design_certification=certification_design,
                        target_certification=certification_target,
                        q_mean=posterior_mean,
                        p_mean=prior_mean,
                        base_std=base_std,
                        clip_bound=clip_bound,
                        structure_kl=structure_kl,
                        settings=settings,
                    )
                    row: dict[str, Any] = {
                        "zone": zone,
                        "candidate_id": spec.candidate_id,
                        "family": family,
                        "lag": lag,
                        "radius": radius,
                        "ridge_alpha": alpha,
                        "rule_count": rule_count,
                        "consequent_dimension": int(designs["bound"].shape[1]),
                        "validation_rmse": validation_rmse,
                        "validation_mae": validation_mae,
                        "clip_bound_scaled": clip_bound,
                        "clip_lower_original": scaling_mean - clip_bound * scaling_std,
                        "clip_upper_original": scaling_mean + clip_bound * scaling_std,
                        "certification_target_clipping_rate": clipping_rate,
                        "certificate_sample_size": int(len(certification_target)),
                        "selection_uses_test": False,
                        "certificate_uses_test": False,
                    }
                    row.update(certificate)
                    row.update(diagnostics)
                    rows.append(row)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError(f"No eligible candidates were built for {zone}.")
    audit = {
        "zone": zone,
        "scaling_constructed_from": "Dprior_only",
        "scaling_mean": scaling_mean,
        "scaling_std": scaling_std,
        "candidate_count": int(len(candidates)),
        "skipped_capped_candidates": int(skipped_capped),
        "build_failures": build_failures,
    }
    return candidates, audit


def _seasonal_predictions(
    values: np.ndarray, labels: np.ndarray, role: str, lag: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(labels == role)
    indices = indices[indices >= lag]
    return indices, values[indices], values[indices - lag]


def _selection_key(row: pd.Series) -> tuple[Any, ...]:
    radius = math.inf if pd.isna(row["radius"]) else float(row["radius"])
    return (
        float(row["validation_rmse"]),
        float(row["certificate_untruncated"]),
        int(row["consequent_dimension"]),
        int(row["rule_count"]),
        str(row["family"]),
        radius,
        float(row["ridge_alpha"]),
        int(row["lag"]),
    )


def _select_row(pool: pd.DataFrame) -> pd.Series:
    if pool.empty:
        raise ValueError("Cannot select from an empty candidate pool.")
    index = min(pool.index, key=lambda idx: _selection_key(pool.loc[idx]))
    return pool.loc[index]


def _forecast_candidate(
    *,
    values_original: np.ndarray,
    labels: np.ndarray,
    row: pd.Series,
    settings: EnergySettings,
) -> dict[str, Any]:
    prior_raw = values_original[labels == "prior"]
    mean = float(np.mean(prior_raw))
    std = float(np.std(prior_raw, ddof=0))
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
        raise RuntimeError("Selected radius candidate unexpectedly reached the cap.")
    posterior_mean, _ = _ridge_solution(
        designs["bound"], target[masks["bound"]], float(row["ridge_alpha"])
    )
    prediction_scaled_raw = designs["test"] @ posterior_mean
    clip_bound = float(row["clip_bound_scaled"])
    prediction_scaled_clipped = np.clip(
        prediction_scaled_raw, -clip_bound, clip_bound
    )
    test_target_scaled = target[masks["test"]]
    prediction_raw = prediction_scaled_raw * std + mean
    prediction_clipped = prediction_scaled_clipped * std + mean
    test_target = test_target_scaled * std + mean
    indices = target_indices[masks["test"]]
    rmse_raw, mae_raw = _rmse_mae(test_target, prediction_raw)
    rmse_clipped, mae_clipped = _rmse_mae(test_target, prediction_clipped)
    cost, normalized_cost = _weighted_cost(
        test_target,
        prediction_clipped,
        under_weight=settings.underforecast_weight,
        over_weight=settings.overforecast_weight,
    )
    return {
        "indices": indices,
        "target": test_target,
        "prediction_raw": prediction_raw,
        "prediction_clipped": prediction_clipped,
        "test_rmse_raw": rmse_raw,
        "test_mae_raw": mae_raw,
        "test_rmse_clipped": rmse_clipped,
        "test_mae_clipped": mae_clipped,
        "test_weighted_cost": cost,
        "test_normalized_weighted_cost": normalized_cost,
        "test_target_clipping_rate": float(
            np.mean(np.abs(test_target_scaled) > clip_bound)
        ),
        "test_forecast_clipping_rate": float(
            np.mean(np.abs(prediction_scaled_raw) > clip_bound)
        ),
    }


def _verify_lock(lock_path: str | Path, paths: dict[str, str | Path]) -> dict[str, Any]:
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    expected = lock.get("sha256", {})
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if expected != actual:
        mismatches = {
            name: {"expected": expected.get(name), "actual": actual.get(name)}
            for name in sorted(set(expected) | set(actual))
            if expected.get(name) != actual.get(name)
        }
        raise RuntimeError(f"Energy-case protocol lock mismatch: {mismatches}")
    return lock


def run_energy_case_study(
    *,
    case_data_path: str | Path,
    config_path: str | Path,
    lock_path: str | Path,
    output_dir: str | Path,
    verification_paths: dict[str, str | Path],
) -> dict[str, Any]:
    """Execute the single authorized energy case and publish atomically."""
    output = Path(output_dir)
    completion = output / "COMPLETED.json"
    if completion.exists():
        raise RuntimeError("The authorized energy case has already completed.")
    settings = EnergySettings.from_yaml(config_path)
    lock = _verify_lock(lock_path, verification_paths)
    case_data = load_two_hour_tetouan(case_data_path)
    split = ratio_split(len(case_data), settings.split_fractions)
    started = time.perf_counter()

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix="energy_case_v3_", dir=str(output.parent))
    )
    try:
        all_candidates: list[pd.DataFrame] = []
        all_family_results: list[dict[str, Any]] = []
        all_decisions: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        predictions_dir = temp_root / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        for zone in settings.zones:
            values = case_data[zone].to_numpy(dtype=float)
            candidates, zone_audit = _candidate_rows_for_zone(
                zone=zone,
                values_original=values,
                labels=split.labels,
                settings=settings,
            )
            candidates.to_csv(temp_root / f"{zone}_candidates.csv", index=False)
            all_candidates.append(candidates)

            validation_indices, validation_target, validation_seasonal = _seasonal_predictions(
                values, split.labels, "validation", settings.seasonal_lag
            )
            seasonal_validation_rmse, seasonal_validation_mae = _rmse_mae(
                validation_target, validation_seasonal
            )

            family_winners: list[pd.Series] = []
            for family in settings.families:
                winner = _select_row(candidates[candidates["family"] == family])
                family_winners.append(winner)

            eligible = candidates[
                (candidates["certificate_untruncated"] <= settings.certificate_threshold)
                & (
                    candidates["certification_target_clipping_rate"]
                    <= settings.clipping_rate_maximum
                )
            ]
            if eligible.empty:
                selected = None
                deployment = "seasonal_naive_24h"
                reason = "no_certificate_eligible_candidate"
            else:
                candidate = _select_row(eligible)
                if (
                    settings.require_no_worse_than_seasonal
                    and float(candidate["validation_rmse"]) > seasonal_validation_rmse
                ):
                    selected = None
                    deployment = "seasonal_naive_24h"
                    reason = "eligible_models_failed_seasonal_validation_gate"
                else:
                    selected = candidate
                    deployment = str(candidate["family"])
                    reason = "certificate_and_validation_gates_passed"

            pretest_decision = {
                "zone": zone,
                "decision_made_before_test_opening": True,
                "deployment": deployment,
                "reason": reason,
                "eligible_candidate_count": int(len(eligible)),
                "seasonal_validation_rmse": seasonal_validation_rmse,
                "seasonal_validation_mae": seasonal_validation_mae,
                "selected_candidate_id": None if selected is None else str(selected["candidate_id"]),
                "selected_family": None if selected is None else str(selected["family"]),
                "selected_certificate": None if selected is None else float(selected["certificate"]),
                "selected_validation_rmse": None if selected is None else float(selected["validation_rmse"]),
            }
            (temp_root / f"{zone}_pretest_decision.json").write_text(
                json.dumps(pretest_decision, indent=2), encoding="utf-8"
            )

            test_indices, test_target, test_seasonal = _seasonal_predictions(
                values, split.labels, "test", settings.seasonal_lag
            )
            baseline_rmse, baseline_mae = _rmse_mae(test_target, test_seasonal)
            baseline_cost, baseline_normalized_cost = _weighted_cost(
                test_target,
                test_seasonal,
                under_weight=settings.underforecast_weight,
                over_weight=settings.overforecast_weight,
            )

            family_forecasts: dict[str, dict[str, Any]] = {}
            for winner in family_winners:
                forecast = _forecast_candidate(
                    values_original=values,
                    labels=split.labels,
                    row=winner,
                    settings=settings,
                )
                family = str(winner["family"])
                family_forecasts[family] = forecast
                all_family_results.append(
                    {
                        "zone": zone,
                        "family": family,
                        "candidate_id": winner["candidate_id"],
                        "lag": int(winner["lag"]),
                        "radius": winner["radius"],
                        "ridge_alpha": float(winner["ridge_alpha"]),
                        "rule_count": int(winner["rule_count"]),
                        "consequent_dimension": int(winner["consequent_dimension"]),
                        "validation_rmse": float(winner["validation_rmse"]),
                        "certificate": float(winner["certificate"]),
                        "certificate_untruncated": float(winner["certificate_untruncated"]),
                        "empirical_gibbs_risk": float(winner["empirical_gibbs_risk"]),
                        "total_kl": float(winner["total_kl"]),
                        "gaussian_kl": float(winner["gaussian_kl"]),
                        "certification_target_clipping_rate": float(
                            winner["certification_target_clipping_rate"]
                        ),
                        "certificate_rms_allowance_original": float(
                            2
                            * winner["clip_bound_scaled"]
                            * zone_audit["scaling_std"]
                            * math.sqrt(min(1.0, float(winner["certificate"])))
                        ),
                        **{
                            key: value
                            for key, value in forecast.items()
                            if key not in {"indices", "target", "prediction_raw", "prediction_clipped"}
                        },
                    }
                )

            if selected is None:
                deployed_prediction = test_seasonal
                deployed_rmse, deployed_mae = baseline_rmse, baseline_mae
                deployed_cost, deployed_normalized_cost = (
                    baseline_cost,
                    baseline_normalized_cost,
                )
                selected_certificate = None
                selected_clipping = None
            else:
                selected_family = str(selected["family"])
                selected_forecast = _forecast_candidate(
                    values_original=values,
                    labels=split.labels,
                    row=selected,
                    settings=settings,
                )
                deployed_prediction = selected_forecast["prediction_clipped"]
                deployed_rmse = selected_forecast["test_rmse_clipped"]
                deployed_mae = selected_forecast["test_mae_clipped"]
                deployed_cost = selected_forecast["test_weighted_cost"]
                deployed_normalized_cost = selected_forecast[
                    "test_normalized_weighted_cost"
                ]
                selected_certificate = float(selected["certificate"])
                selected_clipping = float(
                    selected["certification_target_clipping_rate"]
                )

            decision_row = {
                **pretest_decision,
                "test_opened_after_decision": True,
                "test_baseline_rmse": baseline_rmse,
                "test_baseline_mae": baseline_mae,
                "test_baseline_weighted_cost": baseline_cost,
                "test_baseline_normalized_weighted_cost": baseline_normalized_cost,
                "test_deployed_rmse": deployed_rmse,
                "test_deployed_mae": deployed_mae,
                "test_deployed_weighted_cost": deployed_cost,
                "test_deployed_normalized_weighted_cost": deployed_normalized_cost,
                "test_rmse_improvement_vs_fallback": (
                    baseline_rmse - deployed_rmse
                ) / baseline_rmse,
                "test_cost_improvement_vs_fallback": (
                    baseline_cost - deployed_cost
                ) / baseline_cost,
                "selected_certificate": selected_certificate,
                "selected_certification_clipping_rate": selected_clipping,
            }
            all_decisions.append(decision_row)

            prediction_frame = pd.DataFrame(
                {
                    "timestamp": case_data["timestamp"].iloc[test_indices].to_numpy(),
                    "actual": test_target,
                    "seasonal_naive_24h": test_seasonal,
                    "deployed_forecast": deployed_prediction,
                }
            )
            for family, forecast in family_forecasts.items():
                if not np.array_equal(forecast["indices"], test_indices):
                    raise RuntimeError("Family forecasts are not test-aligned.")
                prediction_frame[f"{family}_forecast"] = forecast[
                    "prediction_clipped"
                ]
            prediction_frame.to_csv(
                predictions_dir / f"{zone}_test_predictions.csv", index=False
            )
            zone_audit.update(
                {
                    "split_counts": split.counts,
                    "familywise_delta": settings.familywise_delta,
                    "candidate_selection_uses_test": False,
                    "certificate_uses_test": False,
                    "deployment_decision_uses_test": False,
                    "test_opened_after_decision": True,
                    "seasonal_validation_points": int(len(validation_indices)),
                    "seasonal_test_points": int(len(test_indices)),
                }
            )
            audits.append(zone_audit)

        candidates_all = pd.concat(all_candidates, ignore_index=True)
        family_results = pd.DataFrame(all_family_results)
        decisions = pd.DataFrame(all_decisions)
        candidates_all.to_csv(temp_root / "energy_candidates_all.csv", index=False)
        family_results.to_csv(temp_root / "energy_family_results.csv", index=False)
        decisions.to_csv(temp_root / "energy_deployment_decisions.csv", index=False)

        aggregate = {
            "phase": "development_real_case_study",
            "confirmatory": False,
            "case_study": "tetouan_city_hourly_power_consumption",
            "zones": list(settings.zones),
            "completed_zones": len(audits),
            "total_candidates": int(len(candidates_all)),
            "all_candidate_selection_excluded_test": bool(
                candidates_all["selection_uses_test"].eq(False).all()
            ),
            "all_certificates_excluded_test": bool(
                candidates_all["certificate_uses_test"].eq(False).all()
            ),
            "all_decisions_preceded_test": bool(
                decisions["decision_made_before_test_opening"].all()
            ),
            "deployed_model_zones": int(
                np.sum(decisions["deployment"] != "seasonal_naive_24h")
            ),
            "mean_test_rmse_improvement_vs_fallback": float(
                decisions["test_rmse_improvement_vs_fallback"].mean()
            ),
            "mean_test_cost_improvement_vs_fallback": float(
                decisions["test_cost_improvement_vs_fallback"].mean()
            ),
            "runtime_seconds": float(time.perf_counter() - started),
            "lock_sha256": sha256_file(lock_path),
            "lock_created_utc": lock.get("created_utc"),
            "data_sha256": sha256_file(case_data_path),
            "zone_audits": audits,
        }
        (temp_root / "energy_case_audit.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        completion_payload = {
            "status": "completed",
            "single_authorized_run": True,
            "outcomes_published_atomically": True,
            "runtime_seconds": aggregate["runtime_seconds"],
            "audit_sha256": sha256_file(temp_root / "energy_case_audit.json"),
            "decisions_sha256": sha256_file(
                temp_root / "energy_deployment_decisions.csv"
            ),
            "family_results_sha256": sha256_file(
                temp_root / "energy_family_results.csv"
            ),
        }
        (temp_root / "COMPLETED.json").write_text(
            json.dumps(completion_payload, indent=2), encoding="utf-8"
        )
        if output.exists():
            shutil.rmtree(output)
        temp_root.replace(output)
        return aggregate
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
