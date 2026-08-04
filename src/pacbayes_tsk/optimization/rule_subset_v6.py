"""Certificate-aware rule-subset primitives for V6.

This module is intentionally independent of the V3/V5 experiment engines.
It provides:
1. a normalized, data-independent prior over complete non-empty rule subsets;
2. deterministic validation/canonicalization of subset identities; and
3. safe construction of a reduced first-order TSK antecedent.

For a dictionary of K frozen rules, the prior first assigns a truncated
geometric mass to the subset size and then distributes that mass uniformly
over the C(K, k) identities of that size:

    pi(S) = exp(-eta * |S|) / (Z_K * C(K, |S|)).

Hence -log pi(S) explicitly charges both subset size and subset identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable, Iterator

import numpy as np

from pacbayes_tsk.models.sparse_tsk import SparseTSKAntecedent


class RuleSubsetV6Error(ValueError):
    """Raised when a V6 rule-subset object is invalid."""


def canonical_rule_subset(
    indices: Iterable[int],
    *,
    total_rule_count: int,
) -> tuple[int, ...]:
    """Return a sorted, unique, non-empty tuple of zero-based rule indices."""
    if int(total_rule_count) != total_rule_count or total_rule_count < 1:
        raise RuleSubsetV6Error(
            "total_rule_count must be a strictly positive integer."
        )
    try:
        raw = tuple(indices)
    except TypeError as error:
        raise RuleSubsetV6Error(
            "indices must be an iterable of integers."
        ) from error
    if not raw:
        raise RuleSubsetV6Error("A rule subset cannot be empty.")
    if any(
        isinstance(index, bool)
        or not isinstance(index, (int, np.integer))
        for index in raw
    ):
        raise RuleSubsetV6Error("Rule indices must be integers.")
    canonical = tuple(sorted(int(index) for index in raw))
    if len(set(canonical)) != len(canonical):
        raise RuleSubsetV6Error("Rule indices must be unique.")
    if canonical[0] < 0 or canonical[-1] >= int(total_rule_count):
        raise RuleSubsetV6Error(
            "A rule index lies outside the frozen dictionary."
        )
    return canonical


@dataclass(frozen=True)
class RuleSubsetPriorV6:
    """Normalized prior over complete rule-subset identities."""

    total_rule_count: int
    eta_rule: float = math.log(2.0)
    minimum_size: int = 1
    maximum_size: int | None = None

    def validate(self) -> None:
        if (
            isinstance(self.total_rule_count, bool)
            or not isinstance(self.total_rule_count, (int, np.integer))
            or self.total_rule_count < 1
        ):
            raise RuleSubsetV6Error(
                "total_rule_count must be a strictly positive integer."
            )
        if not np.isfinite(self.eta_rule) or self.eta_rule < 0.0:
            raise RuleSubsetV6Error(
                "eta_rule must be finite and nonnegative."
            )
        maximum = (
            self.total_rule_count
            if self.maximum_size is None
            else self.maximum_size
        )
        if (
            isinstance(self.minimum_size, bool)
            or not isinstance(self.minimum_size, (int, np.integer))
            or isinstance(maximum, bool)
            or not isinstance(maximum, (int, np.integer))
        ):
            raise RuleSubsetV6Error(
                "Subset-size limits must be integers."
            )
        if not (
            1
            <= int(self.minimum_size)
            <= int(maximum)
            <= int(self.total_rule_count)
        ):
            raise RuleSubsetV6Error(
                "Subset-size limits must satisfy "
                "1 <= minimum <= maximum <= K."
            )

    @property
    def resolved_maximum_size(self) -> int:
        self.validate()
        return (
            int(self.total_rule_count)
            if self.maximum_size is None
            else int(self.maximum_size)
        )

    def _log_size_normalizer(self) -> float:
        self.validate()
        sizes = np.arange(
            int(self.minimum_size),
            self.resolved_maximum_size + 1,
            dtype=np.float64,
        )
        terms = -float(self.eta_rule) * sizes
        maximum = float(np.max(terms))
        return float(
            maximum + np.log(np.exp(terms - maximum).sum())
        )

    def negative_log_mass(
        self,
        indices: Iterable[int],
    ) -> float:
        subset = canonical_rule_subset(
            indices,
            total_rule_count=int(self.total_rule_count),
        )
        size = len(subset)
        if not (
            int(self.minimum_size)
            <= size
            <= self.resolved_maximum_size
        ):
            raise RuleSubsetV6Error(
                "The subset size lies outside the declared prior support."
            )
        identity_penalty = (
            math.lgamma(int(self.total_rule_count) + 1)
            - math.lgamma(size + 1)
            - math.lgamma(
                int(self.total_rule_count) - size + 1
            )
        )
        return float(
            float(self.eta_rule) * size
            + self._log_size_normalizer()
            + identity_penalty
        )

    def iter_support(self) -> Iterator[tuple[int, ...]]:
        """Yield the complete declared support in deterministic order."""
        self.validate()
        rules = range(int(self.total_rule_count))
        for size in range(
            int(self.minimum_size),
            self.resolved_maximum_size + 1,
        ):
            yield from combinations(rules, size)


def subset_antecedent(
    antecedent: SparseTSKAntecedent,
    indices: Iterable[int],
    *,
    max_active_rules: int | None = None,
) -> SparseTSKAntecedent:
    """Create a reduced antecedent from a frozen rule dictionary.

    Rule centers, spreads, and prior supports are copied from ``antecedent``.
    Firing strengths are subsequently renormalized by the existing
    ``SparseTSKAntecedent`` implementation over the retained rules only.
    """
    antecedent.validate()
    subset = canonical_rule_subset(
        indices,
        total_rule_count=antecedent.rule_count,
    )
    active = (
        min(int(antecedent.max_active_rules), len(subset))
        if max_active_rules is None
        else max_active_rules
    )
    if (
        isinstance(active, bool)
        or not isinstance(active, (int, np.integer))
        or not 1 <= int(active) <= len(subset)
    ):
        raise RuleSubsetV6Error(
            "max_active_rules must be an integer "
            "in [1, subset size]."
        )

    selected = np.asarray(subset, dtype=np.int64)
    reduced = SparseTSKAntecedent(
        feature_mean=np.asarray(
            antecedent.feature_mean,
            dtype=np.float64,
        ).copy(),
        feature_scale=np.asarray(
            antecedent.feature_scale,
            dtype=np.float64,
        ).copy(),
        centers=np.asarray(
            antecedent.centers,
            dtype=np.float64,
        )[selected].copy(),
        spreads=np.asarray(
            antecedent.spreads,
            dtype=np.float64,
        )[selected].copy(),
        prior_support=np.asarray(
            antecedent.prior_support,
            dtype=np.int64,
        )[selected].copy(),
        radius=float(antecedent.radius),
        max_active_rules=int(active),
        radius_cap_reached=bool(
            antecedent.radius_cap_reached
        ),
    )
    reduced.validate()
    return reduced
