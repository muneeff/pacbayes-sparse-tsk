from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


class BMAGateError(ValueError):
    """Raised when a V5 BMA validation gate cannot be evaluated safely."""


@dataclass(frozen=True)
class BMAGateThresholds:
    certificate_threshold: float
    clipping_rate_maximum: float
    minimum_rmse_improvement: float
    minimum_asymmetric_score_improvement: float
    validation_blocks: int
    minimum_nonnegative_blocks: int
    maximum_worst_block_degradation: float
    underforecast_weight: float
    overforecast_weight: float

    def validate(self) -> None:
        finite_values = (
            self.certificate_threshold,
            self.clipping_rate_maximum,
            self.minimum_rmse_improvement,
            self.minimum_asymmetric_score_improvement,
            self.maximum_worst_block_degradation,
            self.underforecast_weight,
            self.overforecast_weight,
        )
        if not all(math.isfinite(float(value)) for value in finite_values):
            raise BMAGateError("All gate thresholds must be finite.")
        if not 0.0 < self.certificate_threshold < 1.0:
            raise BMAGateError(
                "certificate_threshold must lie strictly between zero and one."
            )
        if not 0.0 <= self.clipping_rate_maximum < 1.0:
            raise BMAGateError(
                "clipping_rate_maximum must lie in [0, 1)."
            )
        if not 0.0 <= self.minimum_rmse_improvement < 1.0:
            raise BMAGateError(
                "minimum_rmse_improvement must lie in [0, 1)."
            )
        if not 0.0 <= self.minimum_asymmetric_score_improvement < 1.0:
            raise BMAGateError(
                "minimum_asymmetric_score_improvement must lie in [0, 1)."
            )
        if self.validation_blocks < 2:
            raise BMAGateError(
                "validation_blocks must be at least two."
            )
        if not (
            1
            <= self.minimum_nonnegative_blocks
            <= self.validation_blocks
        ):
            raise BMAGateError(
                "minimum_nonnegative_blocks is outside the valid range."
            )
        if not 0.0 <= self.maximum_worst_block_degradation < 1.0:
            raise BMAGateError(
                "maximum_worst_block_degradation must lie in [0, 1)."
            )
        if self.underforecast_weight <= 0.0:
            raise BMAGateError(
                "underforecast_weight must be positive."
            )
        if self.overforecast_weight <= 0.0:
            raise BMAGateError(
                "overforecast_weight must be positive."
            )


@dataclass(frozen=True)
class BMAValidationGateResult:
    validation_bma_rmse: float
    validation_bma_asymmetric_score: float
    seasonal_validation_rmse: float
    seasonal_validation_asymmetric_score: float
    validation_bma_rmse_improvement_vs_fallback: float
    validation_bma_score_improvement_vs_fallback: float
    validation_bma_rmse_nonnegative_blocks: int
    validation_bma_score_nonnegative_blocks: int
    validation_bma_worst_block_rmse_improvement: float
    validation_bma_worst_block_score_improvement: float
    certificate_passed: bool
    clipping_passed: bool
    rmse_margin_passed: bool
    asymmetric_score_margin_passed: bool
    block_rmse_stability_passed: bool
    block_score_stability_passed: bool
    worst_block_degradation_passed: bool
    gate_passed: bool
    failed_conditions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise BMAGateError(f"{name} is empty.")
    if not np.all(np.isfinite(vector)):
        raise BMAGateError(f"{name} contains non-finite values.")
    return vector


def _aligned_vectors(
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _finite_vector(target, name="target")
    prediction = _finite_vector(candidate, name="candidate")
    fallback = _finite_vector(baseline, name="baseline")
    if not (len(y) == len(prediction) == len(fallback)):
        raise BMAGateError(
            "target, candidate, and baseline must have equal lengths."
        )
    return y, prediction, fallback


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    y = _finite_vector(target, name="target")
    yhat = _finite_vector(prediction, name="prediction")
    if len(y) != len(yhat):
        raise BMAGateError(
            "target and prediction must have equal lengths."
        )
    return float(np.sqrt(np.mean(np.square(y - yhat))))


def asymmetric_forecasting_error_score(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    underforecast_weight: float,
    overforecast_weight: float,
) -> float:
    y = _finite_vector(target, name="target")
    yhat = _finite_vector(prediction, name="prediction")
    if len(y) != len(yhat):
        raise BMAGateError(
            "target and prediction must have equal lengths."
        )
    under = float(underforecast_weight)
    over = float(overforecast_weight)
    if (
        not math.isfinite(under)
        or not math.isfinite(over)
        or under <= 0.0
        or over <= 0.0
    ):
        raise BMAGateError(
            "Asymmetric-score weights must be finite and positive."
        )
    under_error = np.maximum(y - yhat, 0.0)
    over_error = np.maximum(yhat - y, 0.0)
    return float(np.mean(under * under_error + over * over_error))


def relative_improvement(
    baseline_value: float,
    candidate_value: float,
) -> float:
    baseline = float(baseline_value)
    candidate = float(candidate_value)
    if (
        not math.isfinite(baseline)
        or not math.isfinite(candidate)
        or baseline < 0.0
        or candidate < 0.0
    ):
        raise BMAGateError(
            "Metric values must be finite and nonnegative."
        )
    return (baseline - candidate) / max(baseline, 1e-12)


def bma_block_stability(
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    blocks: int,
    underforecast_weight: float,
    overforecast_weight: float,
) -> dict[str, Any]:
    y, prediction, fallback = _aligned_vectors(
        target,
        candidate,
        baseline,
    )
    block_count = int(blocks)
    if block_count < 2 or block_count > len(y):
        raise BMAGateError(
            "blocks must be between two and the number of observations."
        )

    rmse_improvements: list[float] = []
    score_improvements: list[float] = []
    for index in np.array_split(np.arange(len(y)), block_count):
        if len(index) == 0:
            raise BMAGateError("A validation block is empty.")
        baseline_rmse = rmse(y[index], fallback[index])
        candidate_rmse = rmse(y[index], prediction[index])
        baseline_score = asymmetric_forecasting_error_score(
            y[index],
            fallback[index],
            underforecast_weight=underforecast_weight,
            overforecast_weight=overforecast_weight,
        )
        candidate_score = asymmetric_forecasting_error_score(
            y[index],
            prediction[index],
            underforecast_weight=underforecast_weight,
            overforecast_weight=overforecast_weight,
        )
        rmse_improvements.append(
            relative_improvement(
                baseline_rmse,
                candidate_rmse,
            )
        )
        score_improvements.append(
            relative_improvement(
                baseline_score,
                candidate_score,
            )
        )

    return {
        "validation_bma_block_rmse_improvements": tuple(
            rmse_improvements
        ),
        "validation_bma_block_score_improvements": tuple(
            score_improvements
        ),
        "validation_bma_rmse_nonnegative_blocks": int(
            np.sum(np.asarray(rmse_improvements) >= 0.0)
        ),
        "validation_bma_score_nonnegative_blocks": int(
            np.sum(np.asarray(score_improvements) >= 0.0)
        ),
        "validation_bma_worst_block_rmse_improvement": float(
            np.min(rmse_improvements)
        ),
        "validation_bma_worst_block_score_improvement": float(
            np.min(score_improvements)
        ),
    }


def evaluate_bma_validation_gate(
    *,
    target: np.ndarray,
    bma_prediction: np.ndarray,
    seasonal_prediction: np.ndarray,
    certificate_untruncated: float,
    certification_target_clipping_rate: float,
    thresholds: BMAGateThresholds,
) -> BMAValidationGateResult:
    thresholds.validate()
    y, bma, seasonal = _aligned_vectors(
        target,
        bma_prediction,
        seasonal_prediction,
    )

    certificate = float(certificate_untruncated)
    clipping_rate = float(certification_target_clipping_rate)
    if not math.isfinite(certificate) or certificate < 0.0:
        raise BMAGateError(
            "certificate_untruncated must be finite and nonnegative."
        )
    if (
        not math.isfinite(clipping_rate)
        or not 0.0 <= clipping_rate <= 1.0
    ):
        raise BMAGateError(
            "certification_target_clipping_rate must lie in [0, 1]."
        )

    seasonal_rmse = rmse(y, seasonal)
    bma_rmse = rmse(y, bma)
    seasonal_score = asymmetric_forecasting_error_score(
        y,
        seasonal,
        underforecast_weight=thresholds.underforecast_weight,
        overforecast_weight=thresholds.overforecast_weight,
    )
    bma_score = asymmetric_forecasting_error_score(
        y,
        bma,
        underforecast_weight=thresholds.underforecast_weight,
        overforecast_weight=thresholds.overforecast_weight,
    )
    rmse_improvement = relative_improvement(
        seasonal_rmse,
        bma_rmse,
    )
    score_improvement = relative_improvement(
        seasonal_score,
        bma_score,
    )
    stability = bma_block_stability(
        y,
        bma,
        seasonal,
        blocks=thresholds.validation_blocks,
        underforecast_weight=thresholds.underforecast_weight,
        overforecast_weight=thresholds.overforecast_weight,
    )

    conditions = {
        "certificate": (
            certificate <= thresholds.certificate_threshold
        ),
        "clipping": (
            clipping_rate <= thresholds.clipping_rate_maximum
        ),
        "rmse_margin": (
            rmse_improvement
            >= thresholds.minimum_rmse_improvement
        ),
        "asymmetric_score_margin": (
            score_improvement
            >= thresholds.minimum_asymmetric_score_improvement
        ),
        "block_rmse_stability": (
            stability[
                "validation_bma_rmse_nonnegative_blocks"
            ]
            >= thresholds.minimum_nonnegative_blocks
        ),
        "block_score_stability": (
            stability[
                "validation_bma_score_nonnegative_blocks"
            ]
            >= thresholds.minimum_nonnegative_blocks
        ),
        "worst_block_degradation": (
            stability[
                "validation_bma_worst_block_rmse_improvement"
            ]
            >= -thresholds.maximum_worst_block_degradation
            and stability[
                "validation_bma_worst_block_score_improvement"
            ]
            >= -thresholds.maximum_worst_block_degradation
        ),
    }
    failed = tuple(
        name
        for name, passed in conditions.items()
        if not passed
    )

    return BMAValidationGateResult(
        validation_bma_rmse=bma_rmse,
        validation_bma_asymmetric_score=bma_score,
        seasonal_validation_rmse=seasonal_rmse,
        seasonal_validation_asymmetric_score=seasonal_score,
        validation_bma_rmse_improvement_vs_fallback=rmse_improvement,
        validation_bma_score_improvement_vs_fallback=score_improvement,
        validation_bma_rmse_nonnegative_blocks=stability[
            "validation_bma_rmse_nonnegative_blocks"
        ],
        validation_bma_score_nonnegative_blocks=stability[
            "validation_bma_score_nonnegative_blocks"
        ],
        validation_bma_worst_block_rmse_improvement=stability[
            "validation_bma_worst_block_rmse_improvement"
        ],
        validation_bma_worst_block_score_improvement=stability[
            "validation_bma_worst_block_score_improvement"
        ],
        certificate_passed=conditions["certificate"],
        clipping_passed=conditions["clipping"],
        rmse_margin_passed=conditions["rmse_margin"],
        asymmetric_score_margin_passed=conditions[
            "asymmetric_score_margin"
        ],
        block_rmse_stability_passed=conditions[
            "block_rmse_stability"
        ],
        block_score_stability_passed=conditions[
            "block_score_stability"
        ],
        worst_block_degradation_passed=conditions[
            "worst_block_degradation"
        ],
        gate_passed=not failed,
        failed_conditions=failed,
    )


_SELECTION_COLUMNS = {
    "validation_bma_asymmetric_score",
    "validation_bma_rmse",
    "exact_certificate_untruncated",
    "consequent_dimension",
    "rule_count",
    "family",
    "radius",
    "ridge_alpha",
    "lag",
}


def bma_deployment_selection_key(row: pd.Series) -> tuple[Any, ...]:
    missing = _SELECTION_COLUMNS.difference(row.index)
    if missing:
        raise BMAGateError(
            f"Candidate row is missing selection columns: {sorted(missing)}."
        )
    radius = (
        math.inf
        if pd.isna(row["radius"])
        else float(row["radius"])
    )
    return (
        float(row["validation_bma_asymmetric_score"]),
        float(row["validation_bma_rmse"]),
        float(row["exact_certificate_untruncated"]),
        int(row["consequent_dimension"]),
        int(row["rule_count"]),
        str(row["family"]),
        radius,
        float(row["ridge_alpha"]),
        int(row["lag"]),
    )


def select_bma_candidate(pool: pd.DataFrame) -> pd.Series:
    if pool.empty:
        raise BMAGateError(
            "Cannot select from an empty BMA candidate pool."
        )
    missing = _SELECTION_COLUMNS.difference(pool.columns)
    if missing:
        raise BMAGateError(
            f"Candidate pool is missing selection columns: {sorted(missing)}."
        )
    index = min(
        pool.index,
        key=lambda item: bma_deployment_selection_key(
            pool.loc[item]
        ),
    )
    return pool.loc[index]
