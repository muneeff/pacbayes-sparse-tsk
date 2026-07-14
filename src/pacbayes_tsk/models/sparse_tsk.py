from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge


class SparseTSKError(ValueError):
    """Raised when a sparse Takagi-Sugeno model cannot be built safely."""


def _validate_matrix(
    X: np.ndarray,
    *,
    name: str,
    expected_features: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise SparseTSKError(f"{name} must be a two-dimensional matrix.")
    if matrix.shape[0] == 0:
        raise SparseTSKError(f"{name} contains no rows.")
    if matrix.shape[1] == 0:
        raise SparseTSKError(f"{name} contains no features.")
    if expected_features is not None and matrix.shape[1] != expected_features:
        raise SparseTSKError(
            f"{name} has {matrix.shape[1]} features; "
            f"expected {expected_features}."
        )
    if not np.all(np.isfinite(matrix)):
        raise SparseTSKError(f"{name} contains NaN or infinite values.")
    return matrix


def _validate_target(
    y: np.ndarray,
    *,
    n_rows: int,
    name: str,
) -> np.ndarray:
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if target.size != n_rows:
        raise SparseTSKError(
            f"{name} contains {target.size} rows but X contains {n_rows}."
        )
    if target.size == 0:
        raise SparseTSKError(f"{name} contains no values.")
    if not np.all(np.isfinite(target)):
        raise SparseTSKError(f"{name} contains NaN or infinite values.")
    return target


def _standardizer(
    X: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_mean = np.mean(X, axis=0)
    feature_scale = np.std(X, axis=0, ddof=0)
    feature_scale = np.where(
        feature_scale > float(epsilon),
        feature_scale,
        1.0,
    )
    standardized = (X - feature_mean) / feature_scale
    return feature_mean, feature_scale, standardized


def _rms_squared_distances(
    X: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """
    Dimension-normalized squared Euclidean distances.

    Dividing by the feature dimension makes the radius grid comparable across
    lag orders.
    """
    difference = X[:, None, :] - centers[None, :, :]
    return np.mean(np.square(difference), axis=2)


def _hard_assign(
    X: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    distances = _rms_squared_distances(X, centers)
    return np.argmin(distances, axis=1).astype(np.int64)


@dataclass(frozen=True)
class SparseTSKAntecedent:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    centers: np.ndarray
    spreads: np.ndarray
    prior_support: np.ndarray
    radius: float
    max_active_rules: int
    radius_cap_reached: bool

    @property
    def rule_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.centers.shape[1])

    @property
    def antecedent_parameter_count(self) -> int:
        return int(2 * self.rule_count * self.n_features)

    def validate(self) -> None:
        feature_mean = np.asarray(self.feature_mean, dtype=np.float64)
        feature_scale = np.asarray(self.feature_scale, dtype=np.float64)
        centers = np.asarray(self.centers, dtype=np.float64)
        spreads = np.asarray(self.spreads, dtype=np.float64)
        support = np.asarray(self.prior_support, dtype=np.int64)

        if centers.ndim != 2 or centers.shape[0] == 0:
            raise SparseTSKError("centers must be a non-empty 2D matrix.")
        if spreads.shape != centers.shape:
            raise SparseTSKError("spreads must have the same shape as centers.")
        if feature_mean.shape != (centers.shape[1],):
            raise SparseTSKError("feature_mean has an invalid shape.")
        if feature_scale.shape != (centers.shape[1],):
            raise SparseTSKError("feature_scale has an invalid shape.")
        if support.shape != (centers.shape[0],):
            raise SparseTSKError("prior_support has an invalid shape.")

        for name, array in {
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "centers": centers,
            "spreads": spreads,
        }.items():
            if not np.all(np.isfinite(array)):
                raise SparseTSKError(f"{name} contains non-finite values.")

        if np.any(feature_scale <= 0.0):
            raise SparseTSKError("feature_scale must be strictly positive.")
        if np.any(spreads <= 0.0):
            raise SparseTSKError("spreads must be strictly positive.")
        if np.any(support <= 0):
            raise SparseTSKError(
                "Every retained rule must have positive prior support."
            )
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise SparseTSKError("radius must be finite and positive.")
        if self.max_active_rules <= 0:
            raise SparseTSKError(
                "max_active_rules must be strictly positive."
            )

    def standardize(self, X: np.ndarray) -> np.ndarray:
        self.validate()
        matrix = _validate_matrix(
            X,
            name="X",
            expected_features=self.n_features,
        )
        standardized = (
            matrix - self.feature_mean
        ) / self.feature_scale
        if not np.all(np.isfinite(standardized)):
            raise SparseTSKError(
                "Feature standardization produced non-finite values."
            )
        return standardized

    def firing_strengths(self, X: np.ndarray) -> np.ndarray:
        """
        Return sparse normalized Gaussian firing strengths.

        At most `max_active_rules` rules are retained for each sample.
        """
        standardized = self.standardize(X)
        scaled_difference = (
            standardized[:, None, :]
            - self.centers[None, :, :]
        ) / self.spreads[None, :, :]
        log_strength = -0.5 * np.sum(
            np.square(scaled_difference),
            axis=2,
        )

        active_count = min(
            int(self.max_active_rules),
            self.rule_count,
        )
        if active_count < self.rule_count:
            order = np.argsort(
                log_strength,
                axis=1,
                kind="stable",
            )
            active_indices = order[:, -active_count:]
            active_mask = np.zeros_like(
                log_strength,
                dtype=bool,
            )
            rows = np.arange(
                log_strength.shape[0]
            )[:, None]
            active_mask[rows, active_indices] = True
            sparse_log = np.where(
                active_mask,
                log_strength,
                -np.inf,
            )
        else:
            sparse_log = log_strength

        row_max = np.max(
            sparse_log,
            axis=1,
            keepdims=True,
        )
        exponentiated = np.exp(
            sparse_log - row_max
        )
        denominator = np.sum(
            exponentiated,
            axis=1,
            keepdims=True,
        )
        firing = exponentiated / denominator

        if not np.all(np.isfinite(firing)):
            raise SparseTSKError(
                "Firing-strength normalization produced non-finite values."
            )
        if not np.allclose(
            np.sum(firing, axis=1),
            1.0,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise SparseTSKError(
                "Normalized firing strengths do not sum to one."
            )
        return firing

    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Construct the first-order TSK design matrix.

        For each rule k, the block is:
            w_k(x) * [1, z_1, ..., z_p]
        where z is standardized using D_prior only.
        """
        standardized = self.standardize(X)
        firing = self.firing_strengths(X)
        augmented = np.concatenate(
            [
                np.ones(
                    (standardized.shape[0], 1),
                    dtype=np.float64,
                ),
                standardized,
            ],
            axis=1,
        )
        design = (
            firing[:, :, None]
            * augmented[:, None, :]
        ).reshape(
            standardized.shape[0],
            self.rule_count
            * (self.n_features + 1),
        )
        if not np.all(np.isfinite(design)):
            raise SparseTSKError(
                "The TSK design matrix contains non-finite values."
            )
        return design

    def diagnostics(
        self,
        X: np.ndarray,
        *,
        activation_threshold: float = 1e-8,
    ) -> dict[str, float | int]:
        if activation_threshold < 0.0:
            raise SparseTSKError(
                "activation_threshold cannot be negative."
            )
        firing = self.firing_strengths(X)
        hard_assignments = np.argmax(
            firing,
            axis=1,
        )
        hard_support = np.bincount(
            hard_assignments,
            minlength=self.rule_count,
        )
        mean_rule_weight = np.mean(
            firing,
            axis=0,
        )
        active_per_sample = np.sum(
            firing > activation_threshold,
            axis=1,
        )

        if self.rule_count > 1:
            positive = mean_rule_weight[
                mean_rule_weight > 0.0
            ]
            entropy = float(
                -np.sum(
                    positive * np.log(positive)
                )
                / np.log(self.rule_count)
            )
        else:
            entropy = 0.0

        return {
            "effective_rule_count": int(
                np.sum(hard_support > 0)
            ),
            "minimum_hard_rule_support": int(
                np.min(hard_support)
            ),
            "maximum_hard_rule_support": int(
                np.max(hard_support)
            ),
            "mean_active_rules": float(
                np.mean(active_per_sample)
            ),
            "maximum_active_rules": int(
                np.max(active_per_sample)
            ),
            "rule_usage_entropy": entropy,
        }


@dataclass(frozen=True)
class SparseTSKModel:
    antecedent: SparseTSKAntecedent
    coefficients: np.ndarray
    ridge_alpha: float

    @property
    def rule_count(self) -> int:
        return self.antecedent.rule_count

    @property
    def n_features(self) -> int:
        return self.antecedent.n_features

    @property
    def consequent_parameter_count(self) -> int:
        return int(
            self.rule_count
            * (self.n_features + 1)
        )

    @property
    def total_parameter_count(self) -> int:
        return int(
            self.antecedent.antecedent_parameter_count
            + self.consequent_parameter_count
        )

    def validate(self) -> None:
        self.antecedent.validate()
        coefficients = np.asarray(
            self.coefficients,
            dtype=np.float64,
        ).reshape(-1)
        if coefficients.shape != (
            self.consequent_parameter_count,
        ):
            raise SparseTSKError(
                "coefficients have an invalid shape."
            )
        if not np.all(np.isfinite(coefficients)):
            raise SparseTSKError(
                "coefficients contain non-finite values."
            )
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha < 0.0:
            raise SparseTSKError(
                "ridge_alpha must be finite and non-negative."
            )

    def predict_scaled(self, X: np.ndarray) -> np.ndarray:
        self.validate()
        design = self.antecedent.design_matrix(X)
        prediction = design @ self.coefficients
        prediction = np.asarray(
            prediction,
            dtype=np.float64,
        ).reshape(-1)
        if not np.all(np.isfinite(prediction)):
            raise SparseTSKError(
                "Sparse TSK produced non-finite predictions."
            )
        return prediction

    def save_npz(
        self,
        path: str | Path,
        *,
        compressed: bool = True,
    ) -> Path:
        self.validate()
        output = Path(path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        payload: dict[str, Any] = {
            "feature_mean": self.antecedent.feature_mean,
            "feature_scale": self.antecedent.feature_scale,
            "centers": self.antecedent.centers,
            "spreads": self.antecedent.spreads,
            "prior_support": self.antecedent.prior_support,
            "radius": np.asarray(
                self.antecedent.radius,
                dtype=np.float64,
            ),
            "max_active_rules": np.asarray(
                self.antecedent.max_active_rules,
                dtype=np.int64,
            ),
            "radius_cap_reached": np.asarray(
                self.antecedent.radius_cap_reached,
                dtype=np.bool_,
            ),
            "coefficients": self.coefficients,
            "ridge_alpha": np.asarray(
                self.ridge_alpha,
                dtype=np.float64,
            ),
        }
        saver = (
            np.savez_compressed
            if compressed
            else np.savez
        )
        saver(output, **payload)
        return output


def load_sparse_tsk_npz(
    path: str | Path,
) -> SparseTSKModel:
    input_path = Path(path)
    with np.load(
        input_path,
        allow_pickle=False,
    ) as archive:
        antecedent = SparseTSKAntecedent(
            feature_mean=archive["feature_mean"],
            feature_scale=archive["feature_scale"],
            centers=archive["centers"],
            spreads=archive["spreads"],
            prior_support=archive[
                "prior_support"
            ].astype(np.int64),
            radius=float(
                archive["radius"].item()
            ),
            max_active_rules=int(
                archive[
                    "max_active_rules"
                ].item()
            ),
            radius_cap_reached=bool(
                archive[
                    "radius_cap_reached"
                ].item()
            ),
        )
        model = SparseTSKModel(
            antecedent=antecedent,
            coefficients=archive[
                "coefficients"
            ].astype(np.float64),
            ridge_alpha=float(
                archive[
                    "ridge_alpha"
                ].item()
            ),
        )
    model.validate()
    return model


def fit_radius_antecedent(
    X_prior: np.ndarray,
    *,
    radius: float,
    max_rules: int,
    max_active_rules: int,
    spread_floor: float = 0.15,
    spread_ceiling: float = 3.0,
    spread_multiplier: float = 1.0,
    refinement_iterations: int = 2,
    epsilon: float = 1e-12,
) -> SparseTSKAntecedent:
    """
    Build a deterministic radius cover from D_prior only.

    Centers are selected by farthest-first traversal in standardized feature
    space. The distance is dimension-normalized RMS Euclidean distance.
    """
    X = _validate_matrix(
        X_prior,
        name="X_prior",
    )

    if not np.isfinite(radius) or radius <= 0.0:
        raise SparseTSKError(
            "radius must be finite and positive."
        )
    if max_rules <= 0:
        raise SparseTSKError(
            "max_rules must be strictly positive."
        )
    if max_active_rules <= 0:
        raise SparseTSKError(
            "max_active_rules must be strictly positive."
        )
    if (
        not np.isfinite(spread_floor)
        or spread_floor <= 0.0
    ):
        raise SparseTSKError(
            "spread_floor must be finite and positive."
        )
    if (
        not np.isfinite(spread_ceiling)
        or spread_ceiling < spread_floor
    ):
        raise SparseTSKError(
            "spread_ceiling must be finite and >= spread_floor."
        )
    if (
        not np.isfinite(spread_multiplier)
        or spread_multiplier <= 0.0
    ):
        raise SparseTSKError(
            "spread_multiplier must be finite and positive."
        )
    if refinement_iterations < 0:
        raise SparseTSKError(
            "refinement_iterations cannot be negative."
        )

    feature_mean, feature_scale, standardized = (
        _standardizer(
            X,
            epsilon=epsilon,
        )
    )

    maximum_rules = min(
        int(max_rules),
        len(standardized),
    )

    distance_to_origin = np.mean(
        np.square(standardized),
        axis=1,
    )
    first_index = int(
        np.argmin(distance_to_origin)
    )
    centers = [
        standardized[first_index].copy()
    ]

    nearest_squared = _rms_squared_distances(
        standardized,
        np.asarray(centers),
    )[:, 0]

    while len(centers) < maximum_rules:
        farthest_index = int(
            np.argmax(nearest_squared)
        )
        farthest_distance = float(
            np.sqrt(
                nearest_squared[
                    farthest_index
                ]
            )
        )
        if farthest_distance <= float(radius):
            break

        centers.append(
            standardized[
                farthest_index
            ].copy()
        )
        new_squared = _rms_squared_distances(
            standardized,
            np.asarray(
                centers[-1:],
                dtype=np.float64,
            ),
        )[:, 0]
        nearest_squared = np.minimum(
            nearest_squared,
            new_squared,
        )

    radius_cap_reached = bool(
        len(centers) == maximum_rules
        and float(
            np.sqrt(
                np.max(nearest_squared)
            )
        )
        > float(radius)
    )

    center_matrix = np.asarray(
        centers,
        dtype=np.float64,
    )

    for _ in range(
        int(refinement_iterations)
    ):
        assignment = _hard_assign(
            standardized,
            center_matrix,
        )
        refined = center_matrix.copy()
        for rule_index in range(
            len(center_matrix)
        ):
            members = standardized[
                assignment == rule_index
            ]
            if len(members) > 0:
                refined[rule_index] = (
                    np.mean(
                        members,
                        axis=0,
                    )
                )
        center_matrix = refined

    assignment = _hard_assign(
        standardized,
        center_matrix,
    )
    support = np.bincount(
        assignment,
        minlength=len(center_matrix),
    )

    nonempty = support > 0
    center_matrix = center_matrix[
        nonempty
    ]
    assignment = _hard_assign(
        standardized,
        center_matrix,
    )
    support = np.bincount(
        assignment,
        minlength=len(center_matrix),
    ).astype(np.int64)

    global_spread = np.std(
        standardized,
        axis=0,
        ddof=0,
    )
    global_spread = np.where(
        global_spread > epsilon,
        global_spread,
        1.0,
    )

    spreads = np.empty_like(
        center_matrix,
        dtype=np.float64,
    )
    for rule_index in range(
        len(center_matrix)
    ):
        members = standardized[
            assignment == rule_index
        ]
        if len(members) >= 2:
            rule_spread = np.std(
                members,
                axis=0,
                ddof=0,
            )
            rule_spread = np.where(
                rule_spread > epsilon,
                rule_spread,
                global_spread,
            )
        else:
            rule_spread = global_spread.copy()

        spreads[rule_index] = np.clip(
            rule_spread
            * float(spread_multiplier),
            float(spread_floor),
            float(spread_ceiling),
        )

    antecedent = SparseTSKAntecedent(
        feature_mean=np.asarray(
            feature_mean,
            dtype=np.float64,
        ),
        feature_scale=np.asarray(
            feature_scale,
            dtype=np.float64,
        ),
        centers=center_matrix,
        spreads=spreads,
        prior_support=support,
        radius=float(radius),
        max_active_rules=int(
            max_active_rules
        ),
        radius_cap_reached=(
            radius_cap_reached
        ),
    )
    antecedent.validate()
    return antecedent



def fit_fixed_k_antecedent(
    X_prior: np.ndarray,
    *,
    rule_count: int,
    max_active_rules: int,
    spread_floor: float = 0.15,
    spread_ceiling: float = 3.0,
    spread_multiplier: float = 1.0,
    refinement_iterations: int = 2,
    epsilon: float = 1e-12,
) -> SparseTSKAntecedent:
    """Build a deterministic fixed-rule antecedent from ``D_prior`` only.

    The initial centers use the same origin-seeded farthest-first traversal as
    :func:`fit_radius_antecedent`, but the traversal stops after exactly
    ``rule_count`` centers rather than at a radius threshold.  The stored
    ``radius`` is the *realized* covering radius after refinement; it is a
    diagnostic and is not a selected hyperparameter for this family.
    """
    X = _validate_matrix(X_prior, name="X_prior")
    if rule_count <= 0:
        raise SparseTSKError("rule_count must be strictly positive.")
    if rule_count > len(X):
        raise SparseTSKError(
            "rule_count cannot exceed the number of prior observations."
        )
    if max_active_rules <= 0:
        raise SparseTSKError("max_active_rules must be strictly positive.")
    if max_active_rules > rule_count:
        raise SparseTSKError("max_active_rules cannot exceed rule_count.")
    if not np.isfinite(spread_floor) or spread_floor <= 0.0:
        raise SparseTSKError("spread_floor must be finite and positive.")
    if not np.isfinite(spread_ceiling) or spread_ceiling < spread_floor:
        raise SparseTSKError(
            "spread_ceiling must be finite and >= spread_floor."
        )
    if not np.isfinite(spread_multiplier) or spread_multiplier <= 0.0:
        raise SparseTSKError(
            "spread_multiplier must be finite and positive."
        )
    if refinement_iterations < 0:
        raise SparseTSKError("refinement_iterations cannot be negative.")

    feature_mean, feature_scale, standardized = _standardizer(
        X, epsilon=epsilon
    )
    distance_to_origin = np.mean(np.square(standardized), axis=1)
    first_index = int(np.argmin(distance_to_origin))
    centers = [standardized[first_index].copy()]
    nearest_squared = _rms_squared_distances(
        standardized, np.asarray(centers, dtype=np.float64)
    )[:, 0]

    while len(centers) < int(rule_count):
        farthest_index = int(np.argmax(nearest_squared))
        farthest_distance = float(np.sqrt(nearest_squared[farthest_index]))
        if farthest_distance <= float(epsilon):
            raise SparseTSKError(
                "The prior data do not contain enough distinct points for "
                f"a fixed {rule_count}-rule antecedent."
            )
        centers.append(standardized[farthest_index].copy())
        new_squared = _rms_squared_distances(
            standardized,
            np.asarray(centers[-1:], dtype=np.float64),
        )[:, 0]
        nearest_squared = np.minimum(nearest_squared, new_squared)

    center_matrix = np.asarray(centers, dtype=np.float64)
    for _ in range(int(refinement_iterations)):
        assignment = _hard_assign(standardized, center_matrix)
        refined = center_matrix.copy()
        for rule_index in range(len(center_matrix)):
            members = standardized[assignment == rule_index]
            if len(members) > 0:
                refined[rule_index] = np.mean(members, axis=0)
        center_matrix = refined

    assignment = _hard_assign(standardized, center_matrix)
    support = np.bincount(
        assignment, minlength=len(center_matrix)
    ).astype(np.int64)
    if np.any(support <= 0):
        raise SparseTSKError(
            "Fixed-K refinement produced an empty rule; the candidate is "
            "ineligible rather than silently reducing K."
        )

    global_spread = np.std(standardized, axis=0, ddof=0)
    global_spread = np.where(global_spread > epsilon, global_spread, 1.0)
    spreads = np.empty_like(center_matrix, dtype=np.float64)
    for rule_index in range(len(center_matrix)):
        members = standardized[assignment == rule_index]
        if len(members) >= 2:
            rule_spread = np.std(members, axis=0, ddof=0)
            rule_spread = np.where(
                rule_spread > epsilon, rule_spread, global_spread
            )
        else:
            rule_spread = global_spread.copy()
        spreads[rule_index] = np.clip(
            rule_spread * float(spread_multiplier),
            float(spread_floor),
            float(spread_ceiling),
        )

    realized_nearest = np.min(
        _rms_squared_distances(standardized, center_matrix), axis=1
    )
    realized_radius = float(np.sqrt(np.max(realized_nearest)))
    realized_radius = max(realized_radius, float(epsilon))

    antecedent = SparseTSKAntecedent(
        feature_mean=np.asarray(feature_mean, dtype=np.float64),
        feature_scale=np.asarray(feature_scale, dtype=np.float64),
        centers=center_matrix,
        spreads=spreads,
        prior_support=support,
        radius=realized_radius,
        max_active_rules=int(max_active_rules),
        radius_cap_reached=False,
    )
    antecedent.validate()
    if antecedent.rule_count != int(rule_count):
        raise SparseTSKError("Fixed-K antecedent did not preserve rule_count.")
    return antecedent

def fit_sparse_tsk_consequents(
    antecedent: SparseTSKAntecedent,
    X_bound: np.ndarray,
    y_bound_scaled: np.ndarray,
    *,
    ridge_alpha: float,
) -> SparseTSKModel:
    """
    Fit first-order consequent parameters on D_bound only.
    """
    X = _validate_matrix(
        X_bound,
        name="X_bound",
        expected_features=(
            antecedent.n_features
        ),
    )
    y = _validate_target(
        y_bound_scaled,
        n_rows=len(X),
        name="y_bound_scaled",
    )
    if (
        not np.isfinite(ridge_alpha)
        or ridge_alpha < 0.0
    ):
        raise SparseTSKError(
            "ridge_alpha must be finite and non-negative."
        )

    design = antecedent.design_matrix(X)
    estimator = Ridge(
        alpha=float(ridge_alpha),
        fit_intercept=False,
        solver="auto",
    )
    estimator.fit(
        design,
        y,
    )
    coefficients = np.asarray(
        estimator.coef_,
        dtype=np.float64,
    ).reshape(-1)

    model = SparseTSKModel(
        antecedent=antecedent,
        coefficients=coefficients,
        ridge_alpha=float(
            ridge_alpha
        ),
    )
    model.validate()
    return model


def inverse_scale(
    values_scaled: np.ndarray,
    *,
    mean: float,
    scale: float,
) -> np.ndarray:
    values = np.asarray(
        values_scaled,
        dtype=np.float64,
    )
    if not np.isfinite(mean):
        raise SparseTSKError(
            "Scaler mean must be finite."
        )
    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise SparseTSKError(
            "Scaler scale must be finite and positive."
        )
    raw = (
        values * float(scale)
        + float(mean)
    )
    if not np.all(np.isfinite(raw)):
        raise SparseTSKError(
            "Inverse scaling produced non-finite values."
        )
    return raw
