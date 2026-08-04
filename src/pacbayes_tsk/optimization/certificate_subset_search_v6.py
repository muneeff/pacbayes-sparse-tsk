"""Certificate-aware structural learning for first-order TSK models.

The frozen rule dictionary must be constructed from D_prior only.  Candidate
subsets are then evaluated without access to test data:

* prior location and scale are estimated from D_prior;
* posterior location is estimated from D_bound;
* exact clipped Gaussian Gibbs risk and the time-uniform certificate are
  evaluated on the predeclared certification sequence D_bound + D_validation;
* the structural KL charge is the caller-supplied base charge plus the
  normalized rule-subset identity charge;
* the selected candidate minimizes the untruncated certificate.

Two deterministic search strategies are supported:

``exhaustive``
    Evaluate every subset in the declared prior support.

``backward``
    Start from the full dictionary. At each level, evaluate every one-rule
    deletion of the current subset and continue through the child with the
    smallest certificate. The final selection is the smallest-certificate
    candidate among every subset evaluated along this path.

The backward strategy is a scalable direct certificate-minimization heuristic;
the exhaustive strategy provides an exact finite-support audit for small K.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np

from pacbayes_tsk.evaluation.clipped_gaussian_bma import (
    clipped_gaussian_bma,
)
from pacbayes_tsk.evaluation.exact_gibbs_certificate_v5 import (
    ExactGibbsCertificateChoice,
    search_exact_clipped_gibbs_certificate,
)
from pacbayes_tsk.models.sparse_tsk import SparseTSKAntecedent
from pacbayes_tsk.optimization.rule_subset_v6 import (
    RuleSubsetPriorV6,
    canonical_rule_subset,
    subset_antecedent,
)


class CertificateSubsetSearchV6Error(ValueError):
    """Raised when certificate-aware subset search inputs are invalid."""


@dataclass(frozen=True)
class RuleSubsetCandidateV6:
    subset: tuple[int, ...]
    subset_size: int
    subset_identity_kl: float
    base_structure_kl: float
    total_structure_kl: float
    prior_mean: np.ndarray
    posterior_mean: np.ndarray
    base_prior_std: np.ndarray
    posterior_std: np.ndarray
    certificate: ExactGibbsCertificateChoice
    validation_bma_rmse: float
    validation_bma_mae: float

    @property
    def selection_key(self) -> tuple[object, ...]:
        return (
            float(self.certificate.certificate_untruncated),
            float(self.certificate.total_kl),
            float(self.certificate.exact_empirical_gibbs_risk),
            int(self.subset_size),
            tuple(self.subset),
        )


@dataclass(frozen=True)
class CertificateSubsetSearchResultV6:
    strategy: str
    selected: RuleSubsetCandidateV6
    selected_antecedent: SparseTSKAntecedent
    evaluated_candidates: tuple[RuleSubsetCandidateV6, ...]

    @property
    def evaluated_subset_count(self) -> int:
        return len(self.evaluated_candidates)

    def predict_bma(self, X: np.ndarray) -> np.ndarray:
        design = self.selected_antecedent.design_matrix(X)
        return clipped_gaussian_bma(
            design,
            self.selected.posterior_mean,
            self.selected.posterior_std,
            clip_bound=self._clip_bound,
        )

    @property
    def _clip_bound(self) -> float:
        # The exact search does not currently expose the clip bound in its
        # choice dataclass. It is attached to the posterior array by the
        # factory below through a private immutable scalar attribute.
        value = getattr(self, "__clip_bound", None)
        if value is None:
            raise CertificateSubsetSearchV6Error(
                "The search result does not contain a clip bound."
            )
        return float(value)


def _finite_matrix(
    values: np.ndarray,
    *,
    name: str,
    expected_features: int,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise CertificateSubsetSearchV6Error(
            f"{name} must be a non-empty matrix."
        )
    if matrix.shape[1] != expected_features:
        raise CertificateSubsetSearchV6Error(
            f"{name} has an unexpected feature dimension."
        )
    if not np.all(np.isfinite(matrix)):
        raise CertificateSubsetSearchV6Error(
            f"{name} contains non-finite values."
        )
    return matrix


def _finite_target(
    values: np.ndarray,
    *,
    name: str,
    expected_rows: int,
) -> np.ndarray:
    target = np.asarray(values, dtype=np.float64).reshape(-1)
    if target.shape != (expected_rows,):
        raise CertificateSubsetSearchV6Error(
            f"{name} has an unexpected length."
        )
    if not np.all(np.isfinite(target)):
        raise CertificateSubsetSearchV6Error(
            f"{name} contains non-finite values."
        )
    return target


def _ridge_solution(
    design: np.ndarray,
    target: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    gram = design.T @ design
    regularized = (
        gram
        + float(ridge_alpha)
        * np.eye(gram.shape[0], dtype=np.float64)
    )
    rhs = design.T @ target
    try:
        inverse = np.linalg.inv(regularized)
        coefficients = inverse @ rhs
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(regularized)
        coefficients = inverse @ rhs
    return (
        np.asarray(coefficients, dtype=np.float64).reshape(-1),
        np.asarray(np.diag(inverse), dtype=np.float64).reshape(-1),
    )


def _base_prior_std(
    design_prior: np.ndarray,
    target_prior: np.ndarray,
    prior_mean: np.ndarray,
    inverse_gram_diagonal: np.ndarray,
) -> np.ndarray:
    residual = target_prior - design_prior @ prior_mean
    residual_variance = max(
        1e-4,
        float(np.mean(np.square(residual))),
    )
    variance = residual_variance * np.maximum(
        inverse_gram_diagonal,
        0.0,
    )
    return np.sqrt(np.clip(variance, 1e-6, 100.0))


def _validate_scalar_inputs(
    *,
    ridge_alpha: float,
    clip_bound: float,
    base_structure_kl: float,
    delta: float,
    max_evaluations: int | None,
) -> None:
    if not np.isfinite(ridge_alpha) or ridge_alpha < 0.0:
        raise CertificateSubsetSearchV6Error(
            "ridge_alpha must be finite and nonnegative."
        )
    if not np.isfinite(clip_bound) or clip_bound <= 0.0:
        raise CertificateSubsetSearchV6Error(
            "clip_bound must be finite and positive."
        )
    if (
        not np.isfinite(base_structure_kl)
        or base_structure_kl < 0.0
    ):
        raise CertificateSubsetSearchV6Error(
            "base_structure_kl must be finite and nonnegative."
        )
    if not np.isfinite(delta) or not 0.0 < delta < 1.0:
        raise CertificateSubsetSearchV6Error(
            "delta must lie strictly between zero and one."
        )
    if max_evaluations is not None and (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, (int, np.integer))
        or max_evaluations < 1
    ):
        raise CertificateSubsetSearchV6Error(
            "max_evaluations must be a positive integer or None."
        )


def search_certificate_aware_rule_subsets_v6(
    *,
    dictionary_antecedent: SparseTSKAntecedent,
    X_prior: np.ndarray,
    y_prior: np.ndarray,
    X_bound: np.ndarray,
    y_bound: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    ridge_alpha: float,
    clip_bound: float,
    base_structure_kl: float,
    subset_prior: RuleSubsetPriorV6,
    prior_scale_multipliers: Iterable[float],
    posterior_scale_ratios: Iterable[float],
    temperatures: Iterable[float],
    delta: float,
    strategy: Literal["exhaustive", "backward"] = "backward",
    max_evaluations: int | None = None,
) -> CertificateSubsetSearchResultV6:
    """Select a complete rule subset by direct certificate minimization."""
    dictionary_antecedent.validate()
    subset_prior.validate()
    if (
        int(subset_prior.total_rule_count)
        != dictionary_antecedent.rule_count
    ):
        raise CertificateSubsetSearchV6Error(
            "The subset prior and frozen dictionary have different K."
        )
    if strategy not in {"exhaustive", "backward"}:
        raise CertificateSubsetSearchV6Error(
            "strategy must be 'exhaustive' or 'backward'."
        )
    _validate_scalar_inputs(
        ridge_alpha=float(ridge_alpha),
        clip_bound=float(clip_bound),
        base_structure_kl=float(base_structure_kl),
        delta=float(delta),
        max_evaluations=max_evaluations,
    )

    n_features = dictionary_antecedent.n_features
    Xp = _finite_matrix(
        X_prior,
        name="X_prior",
        expected_features=n_features,
    )
    Xb = _finite_matrix(
        X_bound,
        name="X_bound",
        expected_features=n_features,
    )
    Xv = _finite_matrix(
        X_validation,
        name="X_validation",
        expected_features=n_features,
    )
    yp = _finite_target(
        y_prior,
        name="y_prior",
        expected_rows=len(Xp),
    )
    yb = _finite_target(
        y_bound,
        name="y_bound",
        expected_rows=len(Xb),
    )
    yv = _finite_target(
        y_validation,
        name="y_validation",
        expected_rows=len(Xv),
    )

    prior_scales = tuple(float(x) for x in prior_scale_multipliers)
    posterior_ratios = tuple(
        float(x) for x in posterior_scale_ratios
    )
    temperature_grid = tuple(float(x) for x in temperatures)

    cache: dict[tuple[int, ...], RuleSubsetCandidateV6] = {}

    def evaluate(indices: Iterable[int]) -> RuleSubsetCandidateV6:
        subset = canonical_rule_subset(
            indices,
            total_rule_count=dictionary_antecedent.rule_count,
        )
        if subset in cache:
            return cache[subset]
        if (
            max_evaluations is not None
            and len(cache) >= int(max_evaluations)
        ):
            raise CertificateSubsetSearchV6Error(
                "The predeclared max_evaluations limit was exceeded."
            )

        antecedent = subset_antecedent(
            dictionary_antecedent,
            subset,
        )
        design_prior = antecedent.design_matrix(Xp)
        design_bound = antecedent.design_matrix(Xb)
        design_validation = antecedent.design_matrix(Xv)

        prior_mean, inverse_diagonal = _ridge_solution(
            design_prior,
            yp,
            float(ridge_alpha),
        )
        posterior_mean, _ = _ridge_solution(
            design_bound,
            yb,
            float(ridge_alpha),
        )
        base_std = _base_prior_std(
            design_prior,
            yp,
            prior_mean,
            inverse_diagonal,
        )

        subset_identity_kl = subset_prior.negative_log_mass(
            subset
        )
        total_structure_kl = (
            float(base_structure_kl)
            + float(subset_identity_kl)
        )
        design_cert = np.vstack(
            [design_bound, design_validation]
        )
        target_cert = np.concatenate([yb, yv])

        exact = search_exact_clipped_gibbs_certificate(
            design_cert=design_cert,
            target_cert=target_cert,
            posterior_mean=posterior_mean,
            prior_mean=prior_mean,
            base_prior_std=base_std,
            clip_bound=float(clip_bound),
            structure_kl=total_structure_kl,
            prior_scale_multipliers=prior_scales,
            posterior_scale_ratios=posterior_ratios,
            temperatures=temperature_grid,
            delta=float(delta),
        )
        posterior_std = (
            base_std
            * exact.prior_scale_multiplier
            * exact.posterior_scale_ratio
        )
        validation_bma = clipped_gaussian_bma(
            design_validation,
            posterior_mean,
            posterior_std,
            clip_bound=float(clip_bound),
        )
        validation_error = yv - validation_bma
        candidate = RuleSubsetCandidateV6(
            subset=subset,
            subset_size=len(subset),
            subset_identity_kl=float(subset_identity_kl),
            base_structure_kl=float(base_structure_kl),
            total_structure_kl=float(total_structure_kl),
            prior_mean=prior_mean,
            posterior_mean=posterior_mean,
            base_prior_std=base_std,
            posterior_std=posterior_std,
            certificate=exact,
            validation_bma_rmse=float(
                np.sqrt(np.mean(np.square(validation_error)))
            ),
            validation_bma_mae=float(
                np.mean(np.abs(validation_error))
            ),
        )
        cache[subset] = candidate
        return candidate

    if strategy == "exhaustive":
        for subset in subset_prior.iter_support():
            evaluate(subset)
    else:
        current = tuple(
            range(dictionary_antecedent.rule_count)
        )
        if not (
            subset_prior.minimum_size
            <= len(current)
            <= subset_prior.resolved_maximum_size
        ):
            # If the prior truncates the maximum size, begin with the
            # lexicographically first supported maximum-size subset.
            current = tuple(
                range(subset_prior.resolved_maximum_size)
            )
        evaluate(current)
        while len(current) > int(subset_prior.minimum_size):
            children = [
                current[:position] + current[position + 1 :]
                for position in range(len(current))
            ]
            eligible = [
                child
                for child in children
                if len(child) >= int(subset_prior.minimum_size)
            ]
            evaluated_children = [
                evaluate(child) for child in eligible
            ]
            current = min(
                evaluated_children,
                key=lambda row: row.selection_key,
            ).subset

    candidates = tuple(
        sorted(
            cache.values(),
            key=lambda row: (
                -row.subset_size,
                row.subset,
            ),
        )
    )
    if not candidates:
        raise CertificateSubsetSearchV6Error(
            "The structural search evaluated no candidates."
        )
    selected = min(
        candidates,
        key=lambda row: row.selection_key,
    )
    selected_antecedent = subset_antecedent(
        dictionary_antecedent,
        selected.subset,
    )
    result = CertificateSubsetSearchResultV6(
        strategy=strategy,
        selected=selected,
        selected_antecedent=selected_antecedent,
        evaluated_candidates=candidates,
    )
    object.__setattr__(result, "__clip_bound", float(clip_bound))
    return result
