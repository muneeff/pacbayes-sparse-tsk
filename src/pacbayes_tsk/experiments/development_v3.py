"""Full V3 synthetic-development experiment engine.

The engine deliberately separates candidate selection/certification from test
measurement. Candidate rows never contain test metrics. Test data are opened
only after a selection decision has been frozen for a series.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
import yaml

from pacbayes_tsk.data.synthetic_v3 import generate
from pacbayes_tsk.data.splits_v3 import ratio_split
from pacbayes_tsk.models.sparse_tsk import (
    fit_fixed_k_antecedent,
    fit_radius_antecedent,
)
from pacbayes_tsk.pac_bayes.priors_v3 import HierarchicalModelPrior, ModelIndex
from pacbayes_tsk.pac_bayes.certificates_v3 import (
    clip_bound_from_prior,
    martingale_certificate,
)

ROLES = ("prior", "bound", "validation", "test")


@dataclass(frozen=True)
class DevelopmentSettings:
    processes: tuple[str, ...]
    seeds: tuple[int, ...]
    families: tuple[str, ...]
    lags: tuple[int, ...]
    radii: tuple[float, ...]
    ridge_alphas: tuple[float, ...]
    max_rules: int
    fixed_k_values: tuple[int, ...]
    sparse_active_rules: int
    temperatures: tuple[float, ...]
    prior_scales: tuple[float, ...]
    posterior_ratios: tuple[float, ...]
    prior_variants: tuple[str, ...]
    delta_total: float
    familywise_series_count: int
    length: int
    burn_in: int
    split_fractions: dict[str, float]
    process_parameters: dict[str, dict[str, Any]]

    @property
    def familywise_delta(self) -> float:
        return self.delta_total / self.familywise_series_count

    @classmethod
    def from_files(
        cls,
        *,
        protocol_path: str | Path,
        synthetic_path: str | Path,
        development_path: str | Path,
    ) -> "DevelopmentSettings":
        protocol = _read_yaml(protocol_path)
        synthetic = _read_yaml(synthetic_path)
        development = _read_yaml(development_path)
        processes = tuple(str(x) for x in development["processes"])
        seeds = tuple(int(x) for x in development["seeds"])
        families = tuple(str(x) for x in development["model_families"])
        prior_variants = tuple(str(x) for x in development["prior_variants"])
        required_families = {"ridge", "dense_tsk", "sparse_tsk"}
        allowed_families = required_families | {"fixed_k_dense_tsk"}
        if not required_families.issubset(families):
            raise ValueError(
                "V3 development requires ridge, dense_tsk, and sparse_tsk."
            )
        if set(families) - allowed_families:
            raise ValueError("Unsupported V3 model family.")
        if set(prior_variants) != {"localized", "zero_mean"}:
            raise ValueError("Unsupported prior-variant set.")
        fractions = protocol["data"]["synthetic"]["split_fractions"]
        split_fractions = {role: float(fractions[role]) for role in ROLES}
        params = {
            str(name): dict(values or {})
            for name, values in synthetic["processes"].items()
        }
        missing = set(processes) - set(params)
        if missing:
            raise ValueError(f"Missing synthetic parameters for: {sorted(missing)}")
        settings = cls(
            processes=processes,
            seeds=seeds,
            families=families,
            lags=tuple(int(x) for x in protocol["model"]["lags"]),
            radii=tuple(float(x) for x in protocol["model"]["radii"]),
            ridge_alphas=tuple(float(x) for x in protocol["model"]["ridge_alphas"]),
            max_rules=int(protocol["model"]["max_rules"]),
            fixed_k_values=tuple(
                int(x)
                for x in development.get("ablation", {}).get(
                    "fixed_k_values",
                    [development.get("ablation", {}).get("fixed_k", protocol["model"]["max_rules"])],
                )
            ),
            sparse_active_rules=int(protocol["model"]["sparse_active_rules"]),
            temperatures=tuple(float(x) for x in protocol["certificate"]["temperatures"]),
            prior_scales=tuple(float(x) for x in protocol["certificate"]["prior_scales"]),
            posterior_ratios=tuple(float(x) for x in protocol["certificate"]["posterior_ratios"]),
            prior_variants=prior_variants,
            delta_total=float(development["confidence"]["delta_total"]),
            familywise_series_count=int(development["confidence"]["familywise_series_count"]),
            length=int(synthetic["global"]["length"]),
            burn_in=int(synthetic["global"]["burn_in"]),
            split_fractions=split_fractions,
            process_parameters=params,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if len(set(self.families)) != len(self.families):
            raise ValueError("Model families must be unique.")
        if len(self.processes) * len(self.seeds) != self.familywise_series_count:
            raise ValueError("familywise_series_count must equal processes × seeds.")
        if not 0 < self.delta_total < 1:
            raise ValueError("delta_total must lie in (0,1).")
        if any(x <= 0 for x in self.lags + self.radii + self.ridge_alphas):
            raise ValueError("Model grids must be strictly positive.")
        if any(x <= 0 for x in self.temperatures + self.prior_scales + self.posterior_ratios):
            raise ValueError("Certificate grids must be strictly positive.")
        if not self.fixed_k_values:
            raise ValueError("fixed_k_values cannot be empty.")
        if len(set(self.fixed_k_values)) != len(self.fixed_k_values):
            raise ValueError("fixed_k_values must be unique.")
        if any(k < 1 or k > self.max_rules for k in self.fixed_k_values):
            raise ValueError("fixed_k_values must lie in [1, max_rules].")
        if self.length < 100 or self.burn_in < 0:
            raise ValueError("Invalid synthetic length or burn-in.")


@dataclass(frozen=True)
class CandidateSpec:
    family: str
    lag: int
    radius: float | None
    ridge_alpha: float
    rule_count: int

    @property
    def id(self) -> str:
        radius = "none" if self.radius is None else f"{self.radius:.8g}"
        return (
            f"{self.family}|p={self.lag}|r={radius}|"
            f"a={self.ridge_alpha:.8g}|k={self.rule_count}"
        )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lagged(values: np.ndarray, labels: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.arange(lag, len(values))
    features = np.column_stack([values[targets - step] for step in range(1, lag + 1)])
    return features, values[targets], labels[targets]


def _role_masks(target_labels: np.ndarray) -> dict[str, np.ndarray]:
    return {role: target_labels == role for role in ROLES}


def _ridge_design(features: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(features), dtype=float), features])


def _ridge_solution(design: np.ndarray, target: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    gram = design.T @ design
    regularized = gram + float(alpha) * np.eye(gram.shape[0])
    rhs = design.T @ target
    try:
        coefficients = np.linalg.solve(regularized, rhs)
        inverse = np.linalg.inv(regularized)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(regularized)
        coefficients = inverse @ rhs
    return np.asarray(coefficients, dtype=float), np.asarray(np.diag(inverse), dtype=float)


def _rmse_mae(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    error = np.asarray(target, dtype=float) - np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean(error * error))), float(np.mean(np.abs(error)))


def _base_prior_std(
    design_prior: np.ndarray,
    target_prior: np.ndarray,
    prior_mean: np.ndarray,
    inverse_gram_diagonal: np.ndarray,
) -> np.ndarray:
    residual = target_prior - design_prior @ prior_mean
    residual_variance = max(1e-4, float(np.mean(residual * residual)))
    variance = residual_variance * np.maximum(inverse_gram_diagonal, 0.0)
    # The floor/ceiling applies to the base standard deviation. Mixture scales
    # multiply this base without a second clipping operation.
    return np.sqrt(np.clip(variance, 1e-6, 100.0))


def _gaussian_kl_ratio(
    *,
    q_mean: np.ndarray,
    p_mean: np.ndarray,
    base_std: np.ndarray,
    prior_scale: float,
    posterior_ratio: float,
) -> float:
    dimension = len(q_mean)
    ratio2 = posterior_ratio * posterior_ratio
    location = np.sum(
        ((q_mean - p_mean) / (base_std * prior_scale)) ** 2
    )
    value = 0.5 * (
        dimension * (math.log(1.0 / ratio2) - 1.0 + ratio2)
        + location
    )
    return float(max(0.0, value))


def _best_certificate(
    *,
    design_certification: np.ndarray,
    target_certification: np.ndarray,
    q_mean: np.ndarray,
    p_mean: np.ndarray,
    base_std: np.ndarray,
    clip_bound: float,
    structure_kl: float,
    settings: DevelopmentSettings,
    prior_variant_penalty: float,
) -> dict[str, float]:
    clipped_target = np.clip(target_certification, -clip_bound, clip_bound)
    squared_bias = (clipped_target - design_certification @ q_mean) ** 2
    base_variance = (design_certification * design_certification) @ (base_std * base_std)
    denominator = 4.0 * clip_bound * clip_bound
    prior_scale_penalty = math.log(len(settings.prior_scales))
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
                q_mean=q_mean,
                p_mean=p_mean,
                base_std=base_std,
                prior_scale=prior_scale,
                posterior_ratio=posterior_ratio,
            )
            total_kl = (
                structure_kl
                + prior_scale_penalty
                + prior_variant_penalty
                + gaussian_kl
            )
            familywise = martingale_certificate(
                empirical_risk=empirical,
                total_kl=total_kl,
                n=len(target_certification),
                delta=settings.familywise_delta,
                temperatures=settings.temperatures,
            )
            pointwise_cert = martingale_certificate(
                empirical_risk=empirical,
                total_kl=total_kl,
                n=len(target_certification),
                delta=settings.delta_total,
                temperatures=settings.temperatures,
            )
            row = {
                "prior_scale": float(prior_scale),
                "posterior_ratio": float(posterior_ratio),
                "empirical_gibbs_risk": empirical,
                "gaussian_kl": gaussian_kl,
                "structure_kl": structure_kl,
                "prior_scale_penalty": prior_scale_penalty,
                "prior_variant_penalty": prior_variant_penalty,
                "total_kl": total_kl,
                "temperature_familywise": familywise.temperature,
                "certificate_familywise": familywise.certificate,
                "certificate_familywise_untruncated": familywise.untruncated,
                "temperature_pointwise": pointwise_cert.temperature,
                "certificate_pointwise": pointwise_cert.certificate,
                "certificate_pointwise_untruncated": pointwise_cert.untruncated,
            }
            if best is None or row["certificate_familywise_untruncated"] < best["certificate_familywise_untruncated"]:
                best = row
    assert best is not None
    return best


def _candidate_row(
    *,
    process: str,
    seed: int,
    spec: CandidateSpec,
    designs: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    prior: HierarchicalModelPrior,
    settings: DevelopmentSettings,
    original_scale: float,
    diagnostics: dict[str, float | int],
) -> dict[str, Any]:
    prior_mean, prior_inverse_diagonal = _ridge_solution(
        designs["prior"], targets["prior"], spec.ridge_alpha
    )
    posterior_mean, _ = _ridge_solution(
        designs["bound"], targets["bound"], spec.ridge_alpha
    )
    validation_prediction = designs["validation"] @ posterior_mean
    validation_rmse, validation_mae = _rmse_mae(
        targets["validation"], validation_prediction
    )
    base_std = _base_prior_std(
        designs["prior"], targets["prior"], prior_mean, prior_inverse_diagonal
    )
    certification_design = np.vstack([designs["bound"], designs["validation"]])
    certification_target = np.concatenate([targets["bound"], targets["validation"]])
    clip_bound = clip_bound_from_prior(targets["prior"])
    clipping_rate = float(np.mean(np.abs(certification_target) > clip_bound))
    model_index = ModelIndex(
        family=spec.family,
        lag=spec.lag,
        ridge_alpha=spec.ridge_alpha,
        radius=spec.radius,
        rule_count=spec.rule_count,
    )
    structure_kl = prior.negative_log_mass(model_index)
    prior_variant_penalty = math.log(len(settings.prior_variants))
    localized = _best_certificate(
        design_certification=certification_design,
        target_certification=certification_target,
        q_mean=posterior_mean,
        p_mean=prior_mean,
        base_std=base_std,
        clip_bound=clip_bound,
        structure_kl=structure_kl,
        settings=settings,
        prior_variant_penalty=prior_variant_penalty,
    )
    zero = _best_certificate(
        design_certification=certification_design,
        target_certification=certification_target,
        q_mean=posterior_mean,
        p_mean=np.zeros_like(prior_mean),
        base_std=base_std,
        clip_bound=clip_bound,
        structure_kl=structure_kl,
        settings=settings,
        prior_variant_penalty=prior_variant_penalty,
    )
    row: dict[str, Any] = {
        "process": process,
        "seed": int(seed),
        "candidate_id": spec.id,
        "family": spec.family,
        "lag": spec.lag,
        "radius": spec.radius,
        "ridge_alpha": spec.ridge_alpha,
        "rule_count": spec.rule_count,
        "consequent_dimension": int(designs["bound"].shape[1]),
        "validation_rmse_scaled": validation_rmse,
        "validation_mae_scaled": validation_mae,
        "validation_rmse_original": validation_rmse * original_scale,
        "validation_mae_original": validation_mae * original_scale,
        "clip_bound_scaled": clip_bound,
        "certification_target_clipping_rate": clipping_rate,
        "certificate_sample_size": int(len(certification_target)),
        "selection_uses_test": False,
        "certificate_uses_test": False,
    }
    row.update({f"localized_{key}": value for key, value in localized.items()})
    row.update({f"zero_mean_{key}": value for key, value in zero.items()})
    row.update(diagnostics)
    return row


def _selection_key_rmse(row: pd.Series) -> tuple[Any, ...]:
    radius = math.inf if pd.isna(row["radius"]) else float(row["radius"])
    return (
        float(row["validation_rmse_scaled"]),
        float(row["validation_mae_scaled"]),
        int(row["consequent_dimension"]),
        int(row["rule_count"]),
        radius,
        float(row["ridge_alpha"]),
        int(row["lag"]),
    )


def _selection_key_certificate(row: pd.Series) -> tuple[Any, ...]:
    return (
        float(row["localized_certificate_familywise_untruncated"]),
        *_selection_key_rmse(row),
    )


def _refit_test_metrics(
    *,
    values: np.ndarray,
    labels: np.ndarray,
    row: pd.Series,
    settings: DevelopmentSettings,
    original_scale: float,
) -> dict[str, Any]:
    lag = int(row["lag"])
    features, target, target_labels = _lagged(values, labels, lag)
    masks = _role_masks(target_labels)
    family = str(row["family"])
    alpha = float(row["ridge_alpha"])
    if family == "ridge":
        designs = {role: _ridge_design(features[masks[role]]) for role in ROLES}
    elif family == "fixed_k_dense_tsk":
        requested_k = int(row["rule_count"])
        antecedent = fit_fixed_k_antecedent(
            features[masks["prior"]],
            rule_count=requested_k,
            max_active_rules=requested_k,
        )
        designs = {
            role: antecedent.design_matrix(features[masks[role]])
            for role in ROLES
        }
    else:
        radius = float(row["radius"])
        max_active = (
            settings.sparse_active_rules
            if family == "sparse_tsk"
            else settings.max_rules
        )
        antecedent = fit_radius_antecedent(
            features[masks["prior"]],
            radius=radius,
            max_rules=settings.max_rules,
            max_active_rules=max_active,
        )
        designs = {
            role: antecedent.design_matrix(features[masks[role]])
            for role in ROLES
        }
    posterior_mean, _ = _ridge_solution(designs["bound"], target[masks["bound"]], alpha)
    prediction = designs["test"] @ posterior_mean
    test_target = target[masks["test"]]
    rmse, mae = _rmse_mae(test_target, prediction)
    clip_bound = float(row["clip_bound_scaled"])
    return {
        "test_rmse_scaled": rmse,
        "test_mae_scaled": mae,
        "test_rmse_original": rmse * original_scale,
        "test_mae_original": mae * original_scale,
        "test_target_clipping_rate": float(np.mean(np.abs(test_target) > clip_bound)),
        "test_forecast_clipping_rate": float(np.mean(np.abs(prediction) > clip_bound)),
    }


def run_series(
    *,
    process: str,
    seed: int,
    settings: DevelopmentSettings,
    output_dir: str | Path,
    protocol_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    if process not in settings.processes:
        raise ValueError(f"Process {process!r} is outside the development protocol.")
    if seed not in settings.seeds:
        raise ValueError(f"Seed {seed} is outside the development protocol.")
    started = time.perf_counter()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    series_dir = output_root / "series"
    series_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{process}_seed{seed}"
    candidate_path = series_dir / f"{stem}_candidates.csv"
    selected_path = series_dir / f"{stem}_selected.csv"
    audit_path = series_dir / f"{stem}_audit.json"

    generated = generate(
        process,
        length=settings.length,
        burn_in=settings.burn_in,
        seed=seed,
        parameters=settings.process_parameters[process],
    )
    split = ratio_split(settings.length, settings.split_fractions)
    raw_prior = generated.values[split.labels == "prior"]
    scaling_mean = float(np.mean(raw_prior))
    scaling_std = float(np.std(raw_prior, ddof=0))
    if not np.isfinite(scaling_std) or scaling_std <= 1e-12:
        scaling_std = 1.0
    values = (generated.values - scaling_mean) / scaling_std

    prior = HierarchicalModelPrior(
        settings.families,
        settings.lags,
        settings.radii,
        settings.ridge_alphas,
        settings.max_rules,
    )
    candidate_rows: list[dict[str, Any]] = []
    skipped_capped = 0
    build_failures: list[dict[str, Any]] = []

    for lag in settings.lags:
        features, target, target_labels = _lagged(values, split.labels, lag)
        masks = _role_masks(target_labels)
        targets = {role: target[masks[role]] for role in ROLES}

        ridge_designs = {role: _ridge_design(features[masks[role]]) for role in ROLES}
        ridge_diagnostics = {
            "effective_rule_count": 1,
            "minimum_hard_rule_support": int(len(targets["prior"])),
            "maximum_hard_rule_support": int(len(targets["prior"])),
            "mean_active_rules": 1.0,
            "maximum_active_rules": 1,
            "rule_usage_entropy": 0.0,
        }
        for alpha in settings.ridge_alphas:
            spec = CandidateSpec("ridge", lag, None, alpha, 1)
            candidate_rows.append(
                _candidate_row(
                    process=process,
                    seed=seed,
                    spec=spec,
                    designs=ridge_designs,
                    targets=targets,
                    prior=prior,
                    settings=settings,
                    original_scale=scaling_std,
                    diagnostics=ridge_diagnostics,
                )
            )

        if "fixed_k_dense_tsk" in settings.families:
            for fixed_k in settings.fixed_k_values:
                try:
                    fixed_antecedent = fit_fixed_k_antecedent(
                        features[masks["prior"]],
                        rule_count=fixed_k,
                        max_active_rules=fixed_k,
                    )
                    fixed_designs = {
                        role: fixed_antecedent.design_matrix(features[masks[role]])
                        for role in ROLES
                    }
                    fixed_diagnostics = fixed_antecedent.diagnostics(
                        np.vstack(
                            [
                                features[masks["bound"]],
                                features[masks["validation"]],
                            ]
                        )
                    )
                    fixed_diagnostics["realized_covering_radius"] = float(
                        fixed_antecedent.radius
                    )
                    fixed_diagnostics["requested_fixed_k"] = int(fixed_k)
                    for alpha in settings.ridge_alphas:
                        spec = CandidateSpec(
                            "fixed_k_dense_tsk",
                            lag,
                            None,
                            alpha,
                            fixed_antecedent.rule_count,
                        )
                        candidate_rows.append(
                            _candidate_row(
                                process=process,
                                seed=seed,
                                spec=spec,
                                designs=fixed_designs,
                                targets=targets,
                                prior=prior,
                                settings=settings,
                                original_scale=scaling_std,
                                diagnostics=fixed_diagnostics,
                            )
                        )
                except Exception as error:
                    build_failures.append(
                        {
                            "family": "fixed_k_dense_tsk",
                            "lag": lag,
                            "radius": None,
                            "rule_count": int(fixed_k),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

        for family in ("dense_tsk", "sparse_tsk"):
            if family not in settings.families:
                continue
            active_rules = (
                settings.max_rules
                if family == "dense_tsk"
                else settings.sparse_active_rules
            )
            for radius in settings.radii:
                try:
                    antecedent = fit_radius_antecedent(
                        features[masks["prior"]],
                        radius=radius,
                        max_rules=settings.max_rules,
                        max_active_rules=active_rules,
                    )
                except Exception as error:  # logged and surfaced in audit
                    build_failures.append(
                        {
                            "family": family,
                            "lag": lag,
                            "radius": radius,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                if antecedent.radius_cap_reached:
                    skipped_capped += len(settings.ridge_alphas)
                    continue
                designs = {
                    role: antecedent.design_matrix(features[masks[role]])
                    for role in ROLES
                }
                diagnostics = antecedent.diagnostics(
                    np.vstack([features[masks["bound"]], features[masks["validation"]]])
                )
                for alpha in settings.ridge_alphas:
                    spec = CandidateSpec(
                        family,
                        lag,
                        radius,
                        alpha,
                        antecedent.rule_count,
                    )
                    candidate_rows.append(
                        _candidate_row(
                            process=process,
                            seed=seed,
                            spec=spec,
                            designs=designs,
                            targets=targets,
                            prior=prior,
                            settings=settings,
                            original_scale=scaling_std,
                            diagnostics=diagnostics,
                        )
                    )

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        raise RuntimeError(f"No eligible candidates for {process} seed {seed}.")
    candidates.sort_values(
        ["family", "lag", "radius", "ridge_alpha"],
        na_position="first",
        inplace=True,
        kind="stable",
    )
    candidates.to_csv(candidate_path, index=False)

    selected_rows: list[dict[str, Any]] = []
    for family in settings.families:
        pool = candidates[candidates["family"] == family]
        if pool.empty:
            raise RuntimeError(f"No eligible {family} candidate for {process} seed {seed}.")
        for strategy, key_function in (
            ("validation_rmse", _selection_key_rmse),
            ("certificate", _selection_key_certificate),
        ):
            selected_index = min(pool.index, key=lambda idx: key_function(pool.loc[idx]))
            row = pool.loc[selected_index].to_dict()
            row["selection_strategy"] = strategy
            row["test_opened_after_selection"] = True
            row.update(
                _refit_test_metrics(
                    values=values,
                    labels=split.labels,
                    row=pool.loc[selected_index],
                    settings=settings,
                    original_scale=scaling_std,
                )
            )
            selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(selected_path, index=False)

    protocol_hashes = {
        str(Path(path).name): _sha256(path)
        for path in protocol_paths
    }
    audit = {
        "phase": "development",
        "confirmatory": False,
        "process": process,
        "seed": seed,
        "generator_metadata": generated.metadata,
        "split_counts_raw": split.counts,
        "scaling_constructed_from": "Dprior_only",
        "scaling_mean": scaling_mean,
        "scaling_std": scaling_std,
        "candidate_count": int(len(candidates)),
        "eligible_by_family": {
            family: int(np.sum(candidates["family"] == family))
            for family in settings.families
        },
        "skipped_capped_candidates": int(skipped_capped),
        "build_failures": build_failures,
        "selection_uses_test": False,
        "certificate_uses_test": False,
        "test_opened_only_after_selection": True,
        "familywise_delta": settings.familywise_delta,
        "pointwise_delta": settings.delta_total,
        "protocol_hashes": protocol_hashes,
        "runtime_seconds": float(time.perf_counter() - started),
        "candidate_file": candidate_path.name,
        "selected_file": selected_path.name,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {
        "process": process,
        "seed": seed,
        "candidate_path": str(candidate_path),
        "selected_path": str(selected_path),
        "audit_path": str(audit_path),
        "candidate_count": len(candidates),
        "runtime_seconds": audit["runtime_seconds"],
    }


def aggregate_development(output_dir: str | Path) -> dict[str, str]:
    output_root = Path(output_dir)
    series_dir = output_root / "series"
    selected_files = sorted(series_dir.glob("*_selected.csv"))
    candidate_files = sorted(series_dir.glob("*_candidates.csv"))
    if not selected_files:
        raise FileNotFoundError("No per-series selected files were found.")
    selected = pd.concat([pd.read_csv(path) for path in selected_files], ignore_index=True)
    candidates = pd.concat([pd.read_csv(path) for path in candidate_files], ignore_index=True)
    selected_path = output_root / "development_v3_selected_all.csv"
    candidate_path = output_root / "development_v3_candidates_all.csv"
    selected.to_csv(selected_path, index=False)
    candidates.to_csv(candidate_path, index=False)

    summary = (
        selected.groupby(["process", "family", "selection_strategy"], dropna=False)
        .agg(
            series_count=("seed", "count"),
            mean_validation_rmse=("validation_rmse_scaled", "mean"),
            mean_test_rmse=("test_rmse_scaled", "mean"),
            mean_rule_count=("rule_count", "mean"),
            mean_dimension=("consequent_dimension", "mean"),
            mean_empirical_risk=("localized_empirical_gibbs_risk", "mean"),
            mean_gaussian_kl=("localized_gaussian_kl", "mean"),
            mean_total_kl=("localized_total_kl", "mean"),
            mean_certificate_familywise=("localized_certificate_familywise", "mean"),
            median_certificate_familywise=("localized_certificate_familywise", "median"),
            vacuous_familywise=("localized_certificate_familywise", lambda x: int(np.sum(np.asarray(x) >= 1.0))),
            mean_certificate_pointwise=("localized_certificate_pointwise", "mean"),
            mean_zero_prior_certificate=("zero_mean_certificate_familywise", "mean"),
            mean_certification_clipping=("certification_target_clipping_rate", "mean"),
            mean_test_clipping=("test_target_clipping_rate", "mean"),
        )
        .reset_index()
    )
    summary_path = output_root / "development_v3_summary.csv"
    summary.to_csv(summary_path, index=False)

    chain = (
        candidates[candidates["family"] != "ridge"]
        .groupby(["process", "family", "lag", "radius"], dropna=False)
        .agg(
            candidates=("candidate_id", "count"),
            mean_rule_count=("rule_count", "mean"),
            mean_dimension=("consequent_dimension", "mean"),
            mean_validation_rmse=("validation_rmse_scaled", "mean"),
            mean_gaussian_kl=("localized_gaussian_kl", "mean"),
            mean_total_kl=("localized_total_kl", "mean"),
            mean_certificate=("localized_certificate_familywise", "mean"),
        )
        .reset_index()
    )
    chain_path = output_root / "development_v3_radius_chain.csv"
    chain.to_csv(chain_path, index=False)

    audit_files = sorted(series_dir.glob("*_audit.json"))
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in audit_files]
    aggregate_audit = {
        "phase": "development",
        "confirmatory": False,
        "completed_series": len(audits),
        "total_candidates": int(len(candidates)),
        "all_selection_excluded_test": all(not item["selection_uses_test"] for item in audits),
        "all_certificates_excluded_test": all(not item["certificate_uses_test"] for item in audits),
        "total_runtime_seconds_sum": float(sum(item["runtime_seconds"] for item in audits)),
        "build_failure_count": int(sum(len(item["build_failures"]) for item in audits)),
        "skipped_capped_candidates": int(sum(item["skipped_capped_candidates"] for item in audits)),
        "selected_file_sha256": _sha256(selected_path),
        "candidates_file_sha256": _sha256(candidate_path),
        "summary_file_sha256": _sha256(summary_path),
        "radius_chain_file_sha256": _sha256(chain_path),
    }
    aggregate_audit_path = output_root / "development_v3_audit.json"
    aggregate_audit_path.write_text(json.dumps(aggregate_audit, indent=2), encoding="utf-8")
    return {
        "selected": str(selected_path),
        "candidates": str(candidate_path),
        "summary": str(summary_path),
        "radius_chain": str(chain_path),
        "audit": str(aggregate_audit_path),
    }
