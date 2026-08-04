"""Prospective one-shot NESO V6 structural-learning experiment.

The implementation enforces two outcome phases:

1. Pre-test phase:
   * read only the chronological prior/bound/validation prefix;
   * build fixed rule dictionaries from D_prior only;
   * enumerate every declared complete rule subset;
   * select the global candidate by the frozen untruncated-certificate key;
   * evaluate and serialize the prelocked deployment/abstention gate.

2. Test phase:
   * open the full processed series only after the pre-test decision exists;
   * use the already selected posterior without refitting;
   * publish the selected exact clipped BMA and deployed forecast atomically.

Any failure writes a permanent abort marker and does not overwrite an existing
prospective result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pacbayes_tsk.data.energy_neso_v6 import (
    EXPECTED_START,
    PROCESSED_COLUMNS,
    load_neso_daily,
    sha256_file,
)
from pacbayes_tsk.evaluation.bma_gate_v5 import (
    BMAGateThresholds,
    asymmetric_forecasting_error_score,
    evaluate_bma_validation_gate,
    relative_improvement,
    rmse,
)
from pacbayes_tsk.evaluation.clipped_gaussian_bma import (
    clipped_gaussian_bma,
)
from pacbayes_tsk.models.sparse_tsk import (
    SparseTSKAntecedent,
    fit_fixed_k_antecedent,
)
from pacbayes_tsk.optimization.certificate_subset_search_v6 import (
    RuleSubsetCandidateV6,
    search_certificate_aware_rule_subsets_v6,
)
from pacbayes_tsk.optimization.rule_subset_v6 import (
    RuleSubsetPriorV6,
    subset_antecedent,
)


ROLE_ORDER = ("prior", "bound", "validation", "test")
PRETEST_ROLES = ("prior", "bound", "validation")


class ProspectiveNESOV6Error(RuntimeError):
    """Raised when the frozen prospective execution contract is violated."""


@dataclass(frozen=True)
class NESOV6Settings:
    split_counts: dict[str, int]
    lags: tuple[int, ...]
    ridge_alphas: tuple[float, ...]
    rule_count: int
    max_active_rules: int
    refinement_iterations: int
    eta_rule: float
    minimum_subset_size: int
    maximum_subset_size: int
    prior_scales: tuple[float, ...]
    posterior_ratios: tuple[float, ...]
    temperatures: tuple[float, ...]
    delta_total: float
    clip_multiplier: float
    clip_minimum_scaled: float
    seasonal_lag: int
    gate_thresholds: BMAGateThresholds

    @property
    def expected_rows(self) -> int:
        return int(sum(self.split_counts.values()))

    @property
    def pretest_rows(self) -> int:
        return int(
            sum(self.split_counts[role] for role in PRETEST_ROLES)
        )

    @property
    def subset_count(self) -> int:
        return int(
            sum(
                math.comb(self.rule_count, size)
                for size in range(
                    self.minimum_subset_size,
                    self.maximum_subset_size + 1,
                )
            )
        )

    @property
    def expected_candidate_count(self) -> int:
        return int(
            self.subset_count
            * len(self.lags)
            * len(self.ridge_alphas)
        )

    @property
    def base_structure_kl(self) -> float:
        return float(
            math.log(len(self.lags))
            + math.log(len(self.ridge_alphas))
        )

    def validate(self) -> None:
        if tuple(self.split_counts) != ROLE_ORDER:
            raise ProspectiveNESOV6Error(
                f"split_counts must be ordered as {ROLE_ORDER}."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
            for value in self.split_counts.values()
        ):
            raise ProspectiveNESOV6Error(
                "Every split count must be a positive integer."
            )
        if not self.lags or len(set(self.lags)) != len(self.lags):
            raise ProspectiveNESOV6Error(
                "The lag grid must be non-empty and unique."
            )
        if not self.ridge_alphas or len(set(self.ridge_alphas)) != len(
            self.ridge_alphas
        ):
            raise ProspectiveNESOV6Error(
                "The ridge grid must be non-empty and unique."
            )
        if any(lag < 1 for lag in self.lags):
            raise ProspectiveNESOV6Error(
                "All lags must be positive."
            )
        if any(
            not np.isfinite(alpha) or alpha < 0.0
            for alpha in self.ridge_alphas
        ):
            raise ProspectiveNESOV6Error(
                "Ridge alphas must be finite and nonnegative."
            )
        if not 1 <= self.max_active_rules <= self.rule_count:
            raise ProspectiveNESOV6Error(
                "max_active_rules must lie in [1, rule_count]."
            )
        if self.refinement_iterations < 0:
            raise ProspectiveNESOV6Error(
                "refinement_iterations cannot be negative."
            )
        RuleSubsetPriorV6(
            total_rule_count=self.rule_count,
            eta_rule=self.eta_rule,
            minimum_size=self.minimum_subset_size,
            maximum_size=self.maximum_subset_size,
        ).validate()
        for name, grid in {
            "prior_scales": self.prior_scales,
            "posterior_ratios": self.posterior_ratios,
            "temperatures": self.temperatures,
        }.items():
            if (
                not grid
                or any(
                    not np.isfinite(value) or value <= 0.0
                    for value in grid
                )
            ):
                raise ProspectiveNESOV6Error(
                    f"{name} must be a non-empty positive grid."
                )
        if not 0.0 < self.delta_total < 1.0:
            raise ProspectiveNESOV6Error(
                "delta_total must lie strictly between zero and one."
            )
        if (
            not np.isfinite(self.clip_multiplier)
            or self.clip_multiplier <= 0.0
            or not np.isfinite(self.clip_minimum_scaled)
            or self.clip_minimum_scaled <= 0.0
        ):
            raise ProspectiveNESOV6Error(
                "Clip-bound settings must be finite and positive."
            )
        if self.seasonal_lag < 1:
            raise ProspectiveNESOV6Error(
                "seasonal_lag must be positive."
            )
        if max(self.lags) >= self.split_counts["prior"]:
            raise ProspectiveNESOV6Error(
                "Every lag must leave prior targets for dictionary fitting."
            )
        self.gate_thresholds.validate()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "NESOV6Settings":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ProspectiveNESOV6Error(
                "The V6 configuration must be a YAML mapping."
            )
        counts_raw = cfg["split"]["counts"]
        gate = cfg["decision"]["robust_gate"]
        cost = cfg["decision"]["asymmetric_cost"]
        dictionary = cfg["model"]["dictionary"]
        subset = cfg["subset_prior"]
        clip = cfg["certificate"]["clip_bound"]
        settings = cls(
            split_counts={
                role: int(counts_raw[role])
                for role in ROLE_ORDER
            },
            lags=tuple(int(value) for value in cfg["model"]["lags"]),
            ridge_alphas=tuple(
                float(value)
                for value in cfg["model"]["ridge_alphas"]
            ),
            rule_count=int(dictionary["rule_count"]),
            max_active_rules=int(dictionary["max_active_rules"]),
            refinement_iterations=int(
                dictionary["refinement_iterations"]
            ),
            eta_rule=float(subset["eta_rule"]),
            minimum_subset_size=int(subset["minimum_size"]),
            maximum_subset_size=int(subset["maximum_size"]),
            prior_scales=tuple(
                float(value)
                for value in cfg["certificate"]["prior_scales"]
            ),
            posterior_ratios=tuple(
                float(value)
                for value in cfg["certificate"][
                    "posterior_ratios"
                ]
            ),
            temperatures=tuple(
                float(value)
                for value in cfg["certificate"]["temperatures"]
            ),
            delta_total=float(cfg["certificate"]["delta_total"]),
            clip_multiplier=float(clip["multiplier"]),
            clip_minimum_scaled=float(clip["minimum_scaled"]),
            seasonal_lag=7,
            gate_thresholds=BMAGateThresholds(
                certificate_threshold=float(
                    gate["certificate_threshold_untruncated"]
                ),
                clipping_rate_maximum=float(
                    gate[
                        "certification_target_clipping_rate_maximum"
                    ]
                ),
                minimum_rmse_improvement=float(
                    gate["minimum_validation_rmse_improvement"]
                ),
                minimum_asymmetric_score_improvement=float(
                    gate["minimum_validation_cost_improvement"]
                ),
                validation_blocks=int(gate["validation_blocks"]),
                minimum_nonnegative_blocks=int(
                    gate["minimum_nonnegative_blocks"]
                ),
                maximum_worst_block_degradation=float(
                    gate["maximum_worst_block_degradation"]
                ),
                underforecast_weight=float(
                    cost["underforecast_weight"]
                ),
                overforecast_weight=float(
                    cost["overforecast_weight"]
                ),
            ),
        )
        settings.validate()
        expected = int(
            cfg["structural_search"][
                "expected_total_structural_candidates"
            ]
        )
        if settings.expected_candidate_count != expected:
            raise ProspectiveNESOV6Error(
                "The computed candidate count differs from the frozen "
                f"configuration: {settings.expected_candidate_count} != "
                f"{expected}."
            )
        return settings


@dataclass(frozen=True)
class PretestContextV6:
    scaling_mean: float
    scaling_std: float
    selected_lag: int
    selected_ridge_alpha: float
    clip_bound_scaled: float
    selected_candidate: RuleSubsetCandidateV6
    selected_antecedent: SparseTSKAntecedent
    gate_passed: bool
    deployment: str


@dataclass(frozen=True)
class PretestOutcomeV6:
    candidates: pd.DataFrame
    decision: dict[str, Any]
    validation_predictions: pd.DataFrame
    context: PretestContextV6


def fixed_count_labels(
    counts: dict[str, int],
) -> np.ndarray:
    if tuple(counts) != ROLE_ORDER:
        raise ProspectiveNESOV6Error(
            f"Counts must be ordered as {ROLE_ORDER}."
        )
    if any(
        isinstance(count, bool)
        or not isinstance(count, (int, np.integer))
        or count < 1
        for count in counts.values()
    ):
        raise ProspectiveNESOV6Error(
            "All role counts must be positive integers."
        )
    return np.concatenate(
        [
            np.repeat(role, int(counts[role]))
            for role in ROLE_ORDER
        ]
    ).astype(object)


def _pretest_labels(settings: NESOV6Settings) -> np.ndarray:
    counts = {
        role: settings.split_counts[role]
        for role in PRETEST_ROLES
    }
    return np.concatenate(
        [
            np.repeat(role, int(counts[role]))
            for role in PRETEST_ROLES
        ]
    ).astype(object)


def _lagged(
    values: np.ndarray,
    labels: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    role_labels = np.asarray(labels, dtype=object).reshape(-1)
    if len(series) != len(role_labels):
        raise ProspectiveNESOV6Error(
            "Values and temporal labels have different lengths."
        )
    if lag < 1 or lag >= len(series):
        raise ProspectiveNESOV6Error("Invalid lag.")
    if not np.all(np.isfinite(series)):
        raise ProspectiveNESOV6Error(
            "The time series contains non-finite values."
        )
    target_indices = np.arange(lag, len(series), dtype=np.int64)
    features = np.column_stack(
        [
            series[target_indices - step]
            for step in range(1, lag + 1)
        ]
    )
    return (
        features,
        series[target_indices],
        role_labels[target_indices],
        target_indices,
    )


def _role_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    return {
        role: np.asarray(labels, dtype=object) == role
        for role in PRETEST_ROLES
    }


def _candidate_id(
    lag: int,
    alpha: float,
    subset: tuple[int, ...],
) -> str:
    subset_text = "-".join(str(index + 1) for index in subset)
    return (
        f"lag={int(lag)}|alpha={float(alpha):.8g}|"
        f"rules={subset_text}"
    )


def _global_selection_key(
    *,
    lag: int,
    ridge_alpha: float,
    candidate: RuleSubsetCandidateV6,
) -> tuple[object, ...]:
    return (
        float(candidate.certificate.certificate_untruncated),
        float(candidate.certificate.total_kl),
        float(candidate.certificate.exact_empirical_gibbs_risk),
        int(candidate.subset_size),
        tuple(candidate.subset),
        float(ridge_alpha),
        int(lag),
    )


def _candidate_row(
    *,
    lag: int,
    ridge_alpha: float,
    dictionary_rule_count: int,
    candidate: RuleSubsetCandidateV6,
) -> dict[str, Any]:
    certificate = candidate.certificate
    return {
        "candidate_id": _candidate_id(
            lag,
            ridge_alpha,
            candidate.subset,
        ),
        "lag": int(lag),
        "ridge_alpha": float(ridge_alpha),
        "dictionary_rule_count": int(dictionary_rule_count),
        "subset_zero_based": ",".join(
            str(value) for value in candidate.subset
        ),
        "subset_one_based": ",".join(
            str(value + 1) for value in candidate.subset
        ),
        "subset_size": int(candidate.subset_size),
        "consequent_dimension": int(
            candidate.posterior_mean.size
        ),
        "subset_identity_kl": float(
            candidate.subset_identity_kl
        ),
        "base_structure_kl": float(
            candidate.base_structure_kl
        ),
        "total_structure_kl": float(
            candidate.total_structure_kl
        ),
        "prior_scale_multiplier": float(
            certificate.prior_scale_multiplier
        ),
        "posterior_scale_ratio": float(
            certificate.posterior_scale_ratio
        ),
        "temperature": float(certificate.temperature),
        "exact_empirical_gibbs_risk": float(
            certificate.exact_empirical_gibbs_risk
        ),
        "conservative_empirical_gibbs_risk_upper": float(
            certificate.conservative_empirical_gibbs_risk_upper
        ),
        "gaussian_kl": float(certificate.gaussian_kl),
        "prior_component_penalty": float(
            certificate.prior_component_penalty
        ),
        "total_kl": float(certificate.total_kl),
        "certificate_untruncated": float(
            certificate.certificate_untruncated
        ),
        "certificate": float(certificate.certificate),
        "validation_bma_rmse_scaled": float(
            candidate.validation_bma_rmse
        ),
        "validation_bma_mae_scaled": float(
            candidate.validation_bma_mae
        ),
        "selection_uses_test": False,
        "certificate_uses_test": False,
        "globally_selected": False,
    }


def _clip_bound_from_prior(
    prior_target_scaled: np.ndarray,
    settings: NESOV6Settings,
) -> float:
    values = np.asarray(
        prior_target_scaled,
        dtype=np.float64,
    ).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ProspectiveNESOV6Error(
            "Prior targets are empty or non-finite."
        )
    return float(
        max(
            settings.clip_minimum_scaled,
            settings.clip_multiplier
            * float(np.max(np.abs(values))),
        )
    )


def run_pretest_selection_v6(
    *,
    values_prefix_original: np.ndarray,
    timestamps_prefix: np.ndarray,
    settings: NESOV6Settings,
) -> PretestOutcomeV6:
    """Run the full structural search without accepting test observations."""
    settings.validate()
    original = np.asarray(
        values_prefix_original,
        dtype=np.float64,
    ).reshape(-1)
    timestamps = pd.DatetimeIndex(timestamps_prefix)
    if len(original) != settings.pretest_rows:
        raise ProspectiveNESOV6Error(
            "Pre-test input length differs from the frozen prefix count."
        )
    if len(timestamps) != len(original):
        raise ProspectiveNESOV6Error(
            "Pre-test timestamps and values have different lengths."
        )
    if not np.all(np.isfinite(original)):
        raise ProspectiveNESOV6Error(
            "Pre-test values contain non-finite observations."
        )
    expected_dates = pd.date_range(
        EXPECTED_START,
        periods=settings.pretest_rows,
        freq="D",
    )
    if not timestamps.equals(expected_dates):
        raise ProspectiveNESOV6Error(
            "Pre-test timestamps are not the frozen chronological prefix."
        )

    labels = _pretest_labels(settings)
    prior_raw = original[labels == "prior"]
    scaling_mean = float(np.mean(prior_raw))
    scaling_std = float(np.std(prior_raw, ddof=0))
    if (
        not np.isfinite(scaling_std)
        or scaling_std <= 1e-12
    ):
        raise ProspectiveNESOV6Error(
            "D_prior produced an invalid scaling standard deviation."
        )
    scaled = (original - scaling_mean) / scaling_std

    subset_prior = RuleSubsetPriorV6(
        total_rule_count=settings.rule_count,
        eta_rule=settings.eta_rule,
        minimum_size=settings.minimum_subset_size,
        maximum_size=settings.maximum_subset_size,
    )

    candidate_rows: list[dict[str, Any]] = []
    candidate_objects: list[
        tuple[
            tuple[object, ...],
            int,
            float,
            RuleSubsetCandidateV6,
            SparseTSKAntecedent,
        ]
    ] = []

    for lag in settings.lags:
        (
            features,
            target,
            target_labels,
            _,
        ) = _lagged(scaled, labels, lag)
        masks = _role_masks(target_labels)
        if not all(np.any(masks[role]) for role in PRETEST_ROLES):
            raise ProspectiveNESOV6Error(
                f"Lag {lag} leaves an empty pre-test role."
            )
        dictionary = fit_fixed_k_antecedent(
            features[masks["prior"]],
            rule_count=settings.rule_count,
            max_active_rules=settings.max_active_rules,
            refinement_iterations=settings.refinement_iterations,
        )
        if dictionary.rule_count != settings.rule_count:
            raise ProspectiveNESOV6Error(
                "The frozen dictionary did not preserve K."
            )

        clip_bound = _clip_bound_from_prior(
            target[masks["prior"]],
            settings,
        )
        for alpha in settings.ridge_alphas:
            search = search_certificate_aware_rule_subsets_v6(
                dictionary_antecedent=dictionary,
                X_prior=features[masks["prior"]],
                y_prior=target[masks["prior"]],
                X_bound=features[masks["bound"]],
                y_bound=target[masks["bound"]],
                X_validation=features[masks["validation"]],
                y_validation=target[masks["validation"]],
                ridge_alpha=alpha,
                clip_bound=clip_bound,
                base_structure_kl=settings.base_structure_kl,
                subset_prior=subset_prior,
                prior_scale_multipliers=settings.prior_scales,
                posterior_scale_ratios=settings.posterior_ratios,
                temperatures=settings.temperatures,
                delta=settings.delta_total,
                strategy="exhaustive",
                max_evaluations=settings.subset_count,
            )
            if search.evaluated_subset_count != settings.subset_count:
                raise ProspectiveNESOV6Error(
                    "An exhaustive search did not evaluate its full support."
                )
            for candidate in search.evaluated_candidates:
                candidate_rows.append(
                    _candidate_row(
                        lag=lag,
                        ridge_alpha=alpha,
                        dictionary_rule_count=dictionary.rule_count,
                        candidate=candidate,
                    )
                )
                candidate_objects.append(
                    (
                        _global_selection_key(
                            lag=lag,
                            ridge_alpha=alpha,
                            candidate=candidate,
                        ),
                        int(lag),
                        float(alpha),
                        candidate,
                        dictionary,
                    )
                )

    if len(candidate_rows) != settings.expected_candidate_count:
        raise ProspectiveNESOV6Error(
            "The observed structural candidate count differs from the "
            f"frozen count: {len(candidate_rows)} != "
            f"{settings.expected_candidate_count}."
        )

    (
        _,
        selected_lag,
        selected_alpha,
        selected_candidate,
        selected_dictionary,
    ) = min(candidate_objects, key=lambda item: item[0])
    selected_antecedent = subset_antecedent(
        selected_dictionary,
        selected_candidate.subset,
    )

    candidates = pd.DataFrame(candidate_rows)
    selected_id = _candidate_id(
        selected_lag,
        selected_alpha,
        selected_candidate.subset,
    )
    candidates.loc[
        candidates["candidate_id"].eq(selected_id),
        "globally_selected",
    ] = True
    if int(candidates["globally_selected"].sum()) != 1:
        raise ProspectiveNESOV6Error(
            "The global structural selection is not unique."
        )
    candidates.sort_values(
        [
            "lag",
            "ridge_alpha",
            "subset_size",
            "subset_zero_based",
        ],
        ascending=[True, True, False, True],
        kind="stable",
        inplace=True,
        ignore_index=True,
    )

    (
        selected_features,
        selected_target,
        selected_labels,
        selected_indices,
    ) = _lagged(scaled, labels, selected_lag)
    selected_masks = _role_masks(selected_labels)
    validation_design = selected_antecedent.design_matrix(
        selected_features[selected_masks["validation"]]
    )
    validation_bma_scaled = clipped_gaussian_bma(
        validation_design,
        selected_candidate.posterior_mean,
        selected_candidate.posterior_std,
        clip_bound=float(
            max(
                settings.clip_minimum_scaled,
                settings.clip_multiplier
                * np.max(
                    np.abs(
                        selected_target[
                            selected_masks["prior"]
                        ]
                    )
                ),
            )
        ),
    )
    clip_bound_selected = _clip_bound_from_prior(
        selected_target[selected_masks["prior"]],
        settings,
    )
    validation_indices = selected_indices[
        selected_masks["validation"]
    ]
    if np.any(validation_indices < settings.seasonal_lag):
        raise ProspectiveNESOV6Error(
            "Validation indices are incompatible with the fallback lag."
        )
    validation_target_original = original[validation_indices]
    seasonal_validation = original[
        validation_indices - settings.seasonal_lag
    ]
    validation_bma_original = (
        validation_bma_scaled * scaling_std + scaling_mean
    )
    certification_target = np.concatenate(
        [
            selected_target[selected_masks["bound"]],
            selected_target[selected_masks["validation"]],
        ]
    )
    clipping_rate = float(
        np.mean(
            np.abs(certification_target)
            > clip_bound_selected
        )
    )
    gate = evaluate_bma_validation_gate(
        target=validation_target_original,
        bma_prediction=validation_bma_original,
        seasonal_prediction=seasonal_validation,
        certificate_untruncated=(
            selected_candidate.certificate.certificate_untruncated
        ),
        certification_target_clipping_rate=clipping_rate,
        thresholds=settings.gate_thresholds,
    )
    deployment = (
        "selected_exact_clipped_gaussian_bma"
        if gate.gate_passed
        else "seasonal_naive_7d"
    )
    decision: dict[str, Any] = {
        "schema_version": "6.0",
        "status": "PRETEST_DECISION_FROZEN",
        "decision_made_before_test_opening": True,
        "selection_uses_test": False,
        "certificate_uses_test": False,
        "gate_uses_test": False,
        "candidate_count": int(len(candidates)),
        "selected_candidate_id": selected_id,
        "selected_lag": int(selected_lag),
        "selected_ridge_alpha": float(selected_alpha),
        "selected_subset_zero_based": list(
            selected_candidate.subset
        ),
        "selected_subset_one_based": [
            value + 1 for value in selected_candidate.subset
        ],
        "selected_subset_size": int(
            selected_candidate.subset_size
        ),
        "selected_consequent_dimension": int(
            selected_candidate.posterior_mean.size
        ),
        "selected_subset_identity_kl": float(
            selected_candidate.subset_identity_kl
        ),
        "selected_base_structure_kl": float(
            selected_candidate.base_structure_kl
        ),
        "selected_total_structure_kl": float(
            selected_candidate.total_structure_kl
        ),
        "selected_prior_scale_multiplier": float(
            selected_candidate.certificate.prior_scale_multiplier
        ),
        "selected_posterior_scale_ratio": float(
            selected_candidate.certificate.posterior_scale_ratio
        ),
        "selected_temperature": float(
            selected_candidate.certificate.temperature
        ),
        "selected_exact_empirical_gibbs_risk": float(
            selected_candidate.certificate.exact_empirical_gibbs_risk
        ),
        "selected_gaussian_kl": float(
            selected_candidate.certificate.gaussian_kl
        ),
        "selected_total_kl": float(
            selected_candidate.certificate.total_kl
        ),
        "selected_certificate_untruncated": float(
            selected_candidate.certificate.certificate_untruncated
        ),
        "selected_certificate": float(
            selected_candidate.certificate.certificate
        ),
        "clip_bound_scaled": float(clip_bound_selected),
        "certification_target_clipping_rate": clipping_rate,
        "deployment": deployment,
        "gate": gate.as_dict(),
    }
    validation_predictions = pd.DataFrame(
        {
            "timestamp": timestamps[validation_indices],
            "actual": validation_target_original,
            "seasonal_naive_7d": seasonal_validation,
            "selected_exact_clipped_bma": (
                validation_bma_original
            ),
        }
    )
    context = PretestContextV6(
        scaling_mean=scaling_mean,
        scaling_std=scaling_std,
        selected_lag=int(selected_lag),
        selected_ridge_alpha=float(selected_alpha),
        clip_bound_scaled=float(clip_bound_selected),
        selected_candidate=selected_candidate,
        selected_antecedent=selected_antecedent,
        gate_passed=bool(gate.gate_passed),
        deployment=deployment,
    )
    return PretestOutcomeV6(
        candidates=candidates,
        decision=decision,
        validation_predictions=validation_predictions,
        context=context,
    )


def _metric_bundle(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    settings: NESOV6Settings,
) -> dict[str, float]:
    return {
        "rmse": rmse(target, prediction),
        "mae": float(
            np.mean(
                np.abs(
                    np.asarray(target, dtype=np.float64)
                    - np.asarray(prediction, dtype=np.float64)
                )
            )
        ),
        "asymmetric_score": asymmetric_forecasting_error_score(
            target,
            prediction,
            underforecast_weight=(
                settings.gate_thresholds.underforecast_weight
            ),
            overforecast_weight=(
                settings.gate_thresholds.overforecast_weight
            ),
        ),
    }


def evaluate_test_phase_v6(
    *,
    values_full_original: np.ndarray,
    timestamps_full: np.ndarray,
    settings: NESOV6Settings,
    context: PretestContextV6,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Open and evaluate test using the frozen pre-test context."""
    settings.validate()
    original = np.asarray(
        values_full_original,
        dtype=np.float64,
    ).reshape(-1)
    timestamps = pd.DatetimeIndex(timestamps_full)
    if len(original) != settings.expected_rows:
        raise ProspectiveNESOV6Error(
            "Full-series length differs from the frozen split total."
        )
    if len(timestamps) != len(original):
        raise ProspectiveNESOV6Error(
            "Full timestamps and values have different lengths."
        )
    expected_dates = pd.date_range(
        EXPECTED_START,
        periods=settings.expected_rows,
        freq="D",
    )
    if not timestamps.equals(expected_dates):
        raise ProspectiveNESOV6Error(
            "Full timestamps are not the frozen chronological calendar."
        )
    labels = fixed_count_labels(settings.split_counts)
    scaled = (
        original - context.scaling_mean
    ) / context.scaling_std
    features, target_scaled, target_labels, target_indices = (
        _lagged(scaled, labels, context.selected_lag)
    )
    test_mask = target_labels == "test"
    if not np.any(test_mask):
        raise ProspectiveNESOV6Error(
            "The selected lag leaves no test observations."
        )
    test_indices = target_indices[test_mask]
    if np.any(test_indices < settings.seasonal_lag):
        raise ProspectiveNESOV6Error(
            "Test indices are incompatible with the fallback lag."
        )

    design_test = context.selected_antecedent.design_matrix(
        features[test_mask]
    )
    selected_bma_scaled = clipped_gaussian_bma(
        design_test,
        context.selected_candidate.posterior_mean,
        context.selected_candidate.posterior_std,
        clip_bound=context.clip_bound_scaled,
    )
    selected_bma = (
        selected_bma_scaled
        * context.scaling_std
        + context.scaling_mean
    )
    target = original[test_indices]
    seasonal = original[
        test_indices - settings.seasonal_lag
    ]
    deployed = (
        selected_bma
        if context.gate_passed
        else seasonal
    )

    baseline_metrics = _metric_bundle(
        target,
        seasonal,
        settings=settings,
    )
    selected_metrics = _metric_bundle(
        target,
        selected_bma,
        settings=settings,
    )
    deployed_metrics = _metric_bundle(
        target,
        deployed,
        settings=settings,
    )
    metrics: dict[str, Any] = {
        "schema_version": "6.0",
        "test_opened_after_pretest_decision": True,
        "posterior_refit_after_selection": False,
        "deployment": context.deployment,
        "test_observation_count": int(len(target)),
        "seasonal_naive_7d": baseline_metrics,
        "selected_exact_clipped_bma": selected_metrics,
        "deployed_forecast": deployed_metrics,
        "selected_bma_rmse_improvement_vs_fallback": (
            relative_improvement(
                baseline_metrics["rmse"],
                selected_metrics["rmse"],
            )
        ),
        "selected_bma_score_improvement_vs_fallback": (
            relative_improvement(
                baseline_metrics["asymmetric_score"],
                selected_metrics["asymmetric_score"],
            )
        ),
        "deployed_rmse_improvement_vs_fallback": (
            relative_improvement(
                baseline_metrics["rmse"],
                deployed_metrics["rmse"],
            )
        ),
        "deployed_score_improvement_vs_fallback": (
            relative_improvement(
                baseline_metrics["asymmetric_score"],
                deployed_metrics["asymmetric_score"],
            )
        ),
        "test_target_clipping_rate": float(
            np.mean(
                np.abs(target_scaled[test_mask])
                > context.clip_bound_scaled
            )
        ),
    }
    predictions = pd.DataFrame(
        {
            "timestamp": timestamps[test_indices],
            "actual": target,
            "seasonal_naive_7d": seasonal,
            "selected_exact_clipped_bma": selected_bma,
            "deployed_forecast": deployed,
        }
    )
    return metrics, predictions


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _verify_execution_snapshot(
    snapshot_path: str | Path,
) -> dict[str, Any]:
    snapshot_file = Path(snapshot_path).resolve()
    payload = json.loads(
        snapshot_file.read_text(encoding="utf-8")
    )
    if payload.get("status") != "FROZEN_NOT_RUN":
        raise ProspectiveNESOV6Error(
            "The execution snapshot is not in FROZEN_NOT_RUN state."
        )
    tracked = payload.get("tracked_files")
    if not isinstance(tracked, list) or not tracked:
        raise ProspectiveNESOV6Error(
            "The execution snapshot has no tracked files."
        )
    root = snapshot_file.parent.parent
    mismatches: list[dict[str, str | None]] = []
    for row in tracked:
        path = root / str(row["path"])
        expected = str(row["sha256"])
        actual = (
            sha256_file(path)
            if path.is_file()
            else None
        )
        if actual != expected:
            mismatches.append(
                {
                    "path": str(row["path"]),
                    "expected": expected,
                    "actual": actual,
                }
            )
    if mismatches:
        raise ProspectiveNESOV6Error(
            f"Execution snapshot mismatch: {mismatches[:10]}"
        )
    return payload


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProspectiveNESOV6Error(
            f"Expected a JSON mapping in {path}."
        )
    return payload


def _verify_data_chain(
    *,
    config_path: str | Path,
    protocol_path: str | Path,
    execution_snapshot_path: str | Path,
    download_audit_path: str | Path,
    preparation_audit_path: str | Path,
) -> dict[str, Any]:
    download = _load_json(download_audit_path)
    preparation = _load_json(preparation_audit_path)
    expected_resource = (
        "b2bde559-3455-4021-b179-dfe60c0337b0"
    )
    checks = {
        "resource_id": (
            download.get("resource_id"),
            expected_resource,
        ),
        "execution_snapshot_sha256": (
            download.get("execution_snapshot_sha256"),
            sha256_file(execution_snapshot_path),
        ),
        "config_sha256": (
            download.get("config_sha256"),
            sha256_file(config_path),
        ),
        "protocol_sha256": (
            download.get("protocol_sha256"),
            sha256_file(protocol_path),
        ),
        "raw_chain": (
            preparation.get("raw_sha256"),
            download.get("raw_sha256"),
        ),
    }
    failed = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in checks.items()
        if observed != expected
    }
    if failed:
        raise ProspectiveNESOV6Error(
            f"Prospective data-chain mismatch: {failed}"
        )
    return {
        "download": download,
        "preparation": preparation,
    }


def _verify_processed_data_hash_after_decision(
    *,
    processed_path: str | Path,
    preparation_audit_path: str | Path,
) -> str:
    # Verify the complete processed file only after the pre-test decision.
    preparation = _load_json(preparation_audit_path)
    expected = preparation.get("processed_sha256")
    actual = sha256_file(processed_path)
    if expected != actual:
        raise ProspectiveNESOV6Error(
            "Processed-data hash mismatch after the pre-test decision: "
            f"expected={expected}, actual={actual}"
        )
    return actual


def _read_pretest_prefix(
    path: str | Path,
    *,
    settings: NESOV6Settings,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        nrows=settings.pretest_rows,
        parse_dates=["timestamp"],
    )
    if tuple(frame.columns) != PROCESSED_COLUMNS:
        raise ProspectiveNESOV6Error(
            f"Unexpected processed schema: {frame.columns.tolist()}"
        )
    if len(frame) != settings.pretest_rows:
        raise ProspectiveNESOV6Error(
            "Processed data do not contain the frozen pre-test prefix."
        )
    expected_dates = pd.date_range(
        EXPECTED_START,
        periods=settings.pretest_rows,
        freq="D",
    )
    if not pd.DatetimeIndex(frame["timestamp"]).equals(
        expected_dates
    ):
        raise ProspectiveNESOV6Error(
            "The processed pre-test prefix has unexpected dates."
        )
    values = frame["nd_mw"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ProspectiveNESOV6Error(
            "The processed pre-test prefix contains non-finite demand."
        )
    return frame


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _write_abort_marker(
    *,
    output_dir: Path,
    error: Exception,
) -> Path:
    marker = output_dir.parent / (
        output_dir.name + "_ABORTED.json"
    )
    if marker.exists():
        return marker
    payload = {
        "schema_version": "6.0",
        "status": "ABORTED",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "outcome_reviewed": False,
        "rerun_authorized": False,
    }
    _atomic_json(marker, payload)
    return marker


def run_prospective_neso_v6(
    *,
    processed_path: str | Path,
    config_path: str | Path,
    protocol_path: str | Path,
    execution_snapshot_path: str | Path,
    download_audit_path: str | Path,
    preparation_audit_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Execute and atomically publish the single authorized V6 run."""
    output = Path(output_dir)
    abort_marker = output.parent / (
        output.name + "_ABORTED.json"
    )
    if output.exists():
        raise ProspectiveNESOV6Error(
            f"Prospective output already exists: {output}"
        )
    if abort_marker.exists():
        raise ProspectiveNESOV6Error(
            f"A permanent V6 abort marker already exists: {abort_marker}"
        )

    started = time.perf_counter()
    temp_root: Path | None = None
    try:
        snapshot = _verify_execution_snapshot(
            execution_snapshot_path
        )
        settings = NESOV6Settings.from_yaml(config_path)
        if settings.expected_rows != 365:
            raise ProspectiveNESOV6Error(
                "The frozen NESO V6 run must contain 365 daily rows."
            )
        data_chain = _verify_data_chain(
            config_path=config_path,
            protocol_path=protocol_path,
            execution_snapshot_path=execution_snapshot_path,
            download_audit_path=download_audit_path,
            preparation_audit_path=preparation_audit_path,
        )

        prefix = _read_pretest_prefix(
            processed_path,
            settings=settings,
        )
        pretest = run_pretest_selection_v6(
            values_prefix_original=prefix[
                "nd_mw"
            ].to_numpy(dtype=np.float64),
            timestamps_prefix=prefix["timestamp"].to_numpy(),
            settings=settings,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        temp_root = Path(
            tempfile.mkdtemp(
                prefix="neso_v6_prospective_",
                dir=str(output.parent),
            )
        )
        candidates_path = temp_root / "candidates_all.csv"
        decision_path = temp_root / "pretest_decision.json"
        validation_path = (
            temp_root / "validation_predictions.csv"
        )
        pretest.candidates.to_csv(
            candidates_path,
            index=False,
            float_format="%.17g",
        )
        _atomic_json(decision_path, pretest.decision)
        pretest.validation_predictions.to_csv(
            validation_path,
            index=False,
            float_format="%.17g",
            date_format="%Y-%m-%d",
        )
        pretest_decision_sha256 = sha256_file(decision_path)

        # The complete processed file is not hashed or loaded until the
        # pre-test decision above has been serialized and hashed.
        processed_data_sha256 = (
            _verify_processed_data_hash_after_decision(
                processed_path=processed_path,
                preparation_audit_path=preparation_audit_path,
            )
        )
        full = load_neso_daily(processed_path)
        test_metrics, test_predictions = evaluate_test_phase_v6(
            values_full_original=full["nd_mw"].to_numpy(
                dtype=np.float64
            ),
            timestamps_full=full["timestamp"].to_numpy(),
            settings=settings,
            context=pretest.context,
        )
        metrics_path = temp_root / "test_metrics.json"
        predictions_path = temp_root / "test_predictions.csv"
        _atomic_json(metrics_path, test_metrics)
        test_predictions.to_csv(
            predictions_path,
            index=False,
            float_format="%.17g",
            date_format="%Y-%m-%d",
        )

        audit: dict[str, Any] = {
            "schema_version": "6.0",
            "status": "COMPLETED",
            "experiment_id": (
                "neso_gb_demand_2025_certificate_subset_v6"
            ),
            "prospective": True,
            "single_authorized_run": True,
            "outcomes_published_atomically": True,
            "split_counts": settings.split_counts,
            "candidate_count": int(
                len(pretest.candidates)
            ),
            "expected_candidate_count": int(
                settings.expected_candidate_count
            ),
            "all_candidate_selection_excluded_test": bool(
                pretest.candidates[
                    "selection_uses_test"
                ].eq(False).all()
            ),
            "all_certificates_excluded_test": bool(
                pretest.candidates[
                    "certificate_uses_test"
                ].eq(False).all()
            ),
            "pretest_decision_preceded_test": True,
            "pretest_decision_sha256": (
                pretest_decision_sha256
            ),
            "posterior_refit_after_selection": False,
            "deployment": pretest.context.deployment,
            "gate_passed": bool(
                pretest.context.gate_passed
            ),
            "execution_snapshot_created_utc": snapshot.get(
                "created_utc"
            ),
            "execution_snapshot_sha256": sha256_file(
                execution_snapshot_path
            ),
            "config_sha256": sha256_file(config_path),
            "protocol_sha256": sha256_file(protocol_path),
            "processed_data_sha256": processed_data_sha256,
            "download_audit_sha256": sha256_file(
                download_audit_path
            ),
            "preparation_audit_sha256": sha256_file(
                preparation_audit_path
            ),
            "source_resource_id": data_chain[
                "download"
            ].get("resource_id"),
            "raw_data_sha256": data_chain[
                "download"
            ].get("raw_sha256"),
            "runtime_seconds": float(
                time.perf_counter() - started
            ),
            "required_attribution": (
                "Supported by National Energy SO Open Data"
            ),
        }
        audit_path = temp_root / "run_audit.json"
        _atomic_json(audit_path, audit)

        completion = {
            "schema_version": "6.0",
            "status": "COMPLETED",
            "single_authorized_run": True,
            "completed_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "deployment": pretest.context.deployment,
            "gate_passed": bool(
                pretest.context.gate_passed
            ),
            "runtime_seconds": audit["runtime_seconds"],
            "hashes": {
                "candidates_all.csv": sha256_file(
                    candidates_path
                ),
                "pretest_decision.json": sha256_file(
                    decision_path
                ),
                "validation_predictions.csv": sha256_file(
                    validation_path
                ),
                "test_metrics.json": sha256_file(
                    metrics_path
                ),
                "test_predictions.csv": sha256_file(
                    predictions_path
                ),
                "run_audit.json": sha256_file(audit_path),
            },
        }
        _atomic_json(
            temp_root / "COMPLETED.json",
            completion,
        )
        temp_root.replace(output)
        temp_root = None
        return audit
    except Exception as error:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)
        _write_abort_marker(
            output_dir=output,
            error=error,
        )
        raise
