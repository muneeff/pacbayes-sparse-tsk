from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True, order=True)
class ModelIndex:
    family: str
    lag: int
    ridge_alpha: float
    radius: float | None = None
    rule_count: int = 1


@dataclass(frozen=True)
class HierarchicalModelPrior:
    """Finite, data-independent structural prior for all compared families.

    ``fixed_k_dense_tsk`` may be evaluated on a predeclared finite set of
    rule counts and therefore does not pay a radius-grid charge. Each realized
    K still pays the same normalized geometric rule-count code as the
    radius-controlled TSK families, so selecting K is never free.
    """

    families: tuple[str, ...]
    lags: tuple[int, ...]
    radii: tuple[float, ...]
    ridge_alphas: tuple[float, ...]
    max_rules: int
    eta_rule: float = math.log(2)

    _ALLOWED_FAMILIES = frozenset(
        {"ridge", "dense_tsk", "sparse_tsk", "fixed_k_dense_tsk"}
    )

    def validate(self) -> None:
        if not self.families or not self.lags or not self.ridge_alphas:
            raise ValueError("The prior support cannot be empty.")
        if len(set(self.families)) != len(self.families):
            raise ValueError("The prior family support contains duplicates.")
        if set(self.families) - self._ALLOWED_FAMILIES:
            raise ValueError("Unknown model family in prior support.")
        radius_families = {"dense_tsk", "sparse_tsk"}
        if not self.radii and radius_families.intersection(self.families):
            raise ValueError("Radius-controlled TSK families require a radius grid.")
        if self.max_rules < 1 or self.eta_rule < 0:
            raise ValueError("Invalid rule prior.")
        if any(x <= 0 for x in self.lags + self.ridge_alphas + self.radii):
            raise ValueError("Grid values must be positive.")

    def _rule_penalty(self, rule_count: int) -> float:
        if not 1 <= int(rule_count) <= self.max_rules:
            raise ValueError("Rule count is outside frozen support.")
        ks = np.arange(1, self.max_rules + 1, dtype=float)
        log_normalizer = float(np.log(np.exp(-self.eta_rule * ks).sum()))
        return float(self.eta_rule * int(rule_count) + log_normalizer)

    def negative_log_mass(self, index: ModelIndex) -> float:
        self.validate()
        if index.family not in self.families:
            raise ValueError("Family is outside frozen support.")
        if index.lag not in self.lags or index.ridge_alpha not in self.ridge_alphas:
            raise ValueError("Lag or ridge alpha is outside frozen support.")

        penalty = (
            math.log(len(self.families))
            + math.log(len(self.lags))
            + math.log(len(self.ridge_alphas))
        )
        if index.family == "ridge":
            if index.radius is not None or index.rule_count != 1:
                raise ValueError("Ridge index must have radius=None and rule_count=1.")
            return float(penalty)

        penalty += self._rule_penalty(index.rule_count)
        if index.family == "fixed_k_dense_tsk":
            if index.radius is not None:
                raise ValueError("Fixed-K TSK must use radius=None in the model index.")
            # The prior remains normalized over K=1,...,Kmax. A particular
            # experiment may evaluate only a predeclared subset; unused mass
            # remains on non-executed/dummy hypotheses.
            return float(penalty)

        if index.radius not in self.radii:
            raise ValueError("TSK radius is outside frozen support.")
        penalty += math.log(len(self.radii))
        return float(penalty)


# Backward-compatible name for early V3 scripts.
FrozenModelGrid = HierarchicalModelPrior
