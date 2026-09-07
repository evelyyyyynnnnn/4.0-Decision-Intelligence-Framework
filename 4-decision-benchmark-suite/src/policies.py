"""Policies to benchmark, from trivial to well-specified."""

from __future__ import annotations

import numpy as np


class AlwaysAction:
    """Ignores the context entirely. The floor every policy must clear."""

    def __init__(self, idx: int, name: str | None = None):
        self.idx = idx
        self.name = name or f"always action {idx}"

    def fit(self, sample):
        return self

    def decide(self, sample) -> np.ndarray:
        return np.full(len(sample.context), self.idx, int)

    def confidence(self, sample) -> np.ndarray:
        return np.full(len(sample.context), 0.5)


class EmpiricalNewsvendor:
    """Order the critical-fractile quantile of OBSERVED demand.

    The textbook answer, and correct given enough data. Its regret comes from
    estimating the quantile from a finite sample, not from the rule.
    """

    name = "empirical fractile"

    def __init__(self, task):
        self.task = task
        self.order = None

    def fit(self, sample):
        demand = sample.context[:, 0]
        q = np.quantile(demand, self.task.critical_fractile())
        self.order = int(np.argmin(np.abs(self.task.actions - q)))
        return self

    def decide(self, sample) -> np.ndarray:
        return np.full(len(sample.context), self.order, int)

    def confidence(self, sample) -> np.ndarray:
        return np.full(len(sample.context), 0.7)


class MeanDemandNewsvendor:
    """Order the mean. Ignores the cost asymmetry, which is the whole problem."""

    name = "order the mean"

    def __init__(self, task):
        self.task = task
        self.order = None

    def fit(self, sample):
        m = float(sample.context[:, 0].mean())
        self.order = int(np.argmin(np.abs(self.task.actions - m)))
        return self

    def decide(self, sample) -> np.ndarray:
        return np.full(len(sample.context), self.order, int)

    def confidence(self, sample) -> np.ndarray:
        return np.full(len(sample.context), 0.7)


class ThresholdTriage:
    """Treat when the observed signal exceeds a fitted threshold."""

    name = "fitted threshold"

    def __init__(self, task):
        self.task = task
        self.thr = 0.0

    def fit(self, sample):
        x = sample.context[:, 0]
        best, best_cost = 0.0, np.inf
        for t in np.quantile(x, np.linspace(0.02, 0.98, 60)):
            a = (x >= t).astype(int)
            cost = -sample.outcomes[np.arange(len(x)), a].mean()
            if cost < best_cost:
                best, best_cost = float(t), cost
        self.thr = best
        return self

    def decide(self, sample) -> np.ndarray:
        return (sample.context[:, 0] >= self.thr).astype(int)

    def confidence(self, sample) -> np.ndarray:
        d = sample.context[:, 0] - self.thr
        return 1 / (1 + np.exp(-np.abs(d)))


class BayesTriage:
    """Treat when the posterior expected cost of waiting exceeds treating.

    Uses the generative model, so it is the best any policy could do given the
    noisy observation. It is NOT the oracle -- the oracle sees the latent truth
    -- and the gap between the two is irreducible uncertainty rather than a
    modelling failure. Separating those two is the point of including it.
    """

    name = "Bayes-optimal given the signal"

    def __init__(self, task):
        self.task = task

    def fit(self, sample):
        return self

    def _posterior(self, x):
        # latent ~ N(-1.4, 1); observed = latent + N(0, noise)
        s2, n2 = 1.0, self.task.noise ** 2
        post_mean = (-1.4 * n2 + x * s2) / (s2 + n2)
        post_var = s2 * n2 / (s2 + n2)
        # E[sigmoid(latent)] approximated by the probit-matched form.
        return 1 / (1 + np.exp(-post_mean / np.sqrt(1 + np.pi * post_var / 8)))

    def decide(self, sample) -> np.ndarray:
        p = self._posterior(sample.context[:, 0])
        return (p * self.task.miss_cost >= self.task.treat_cost).astype(int)

    def confidence(self, sample) -> np.ndarray:
        p = self._posterior(sample.context[:, 0])
        return np.maximum(p, 1 - p)
