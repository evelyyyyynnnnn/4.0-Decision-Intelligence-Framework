"""Benchmark tasks with known optimal decisions.

A decision benchmark needs an oracle. Without one, a policy can only be compared
to other policies, and a whole field can drift together without anyone noticing.
Each task here exposes the true parameters, so regret against the genuinely
optimal decision is computable rather than estimated.

Tasks also expose a `shift` method producing a distribution the policy was not
fitted on. Robustness is not a property of a policy; it is a property of a policy
against a specified shift, and a benchmark that does not name the shift is not
measuring robustness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TaskSample:
    context: np.ndarray        # N x d observed features
    outcomes: np.ndarray       # N x k payoff of each action
    optimal: np.ndarray        # N, index of the best action
    probs: np.ndarray | None = None   # N x k true P(success) where meaningful


class NewsvendorTask:
    """Order quantity under uncertain demand. The classic asymmetric-cost problem.

    The optimal order is the critical-fractile quantile of demand, which is a
    closed form -- so the oracle here is exact, not approximated by search.
    """

    name = "newsvendor"
    actions = np.arange(0, 101, 5)

    def __init__(self, underage: float = 7.0, overage: float = 2.0,
                 mean: float = 50.0, sd: float = 18.0):
        self.underage, self.overage = underage, overage
        self.mean, self.sd = mean, sd

    def critical_fractile(self) -> float:
        return self.underage / (self.underage + self.overage)

    def oracle_action(self) -> float:
        from math import erf, sqrt
        # Invert the normal CDF by bisection; avoids a scipy dependency here.
        target = self.critical_fractile()
        lo, hi = self.mean - 6 * self.sd, self.mean + 6 * self.sd
        for _ in range(200):
            mid = (lo + hi) / 2
            cdf = 0.5 * (1 + erf((mid - self.mean) / (self.sd * sqrt(2))))
            if cdf < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def cost(self, order: np.ndarray, demand: np.ndarray) -> np.ndarray:
        short = np.clip(demand - order, 0, None) * self.underage
        excess = np.clip(order - demand, 0, None) * self.overage
        return short + excess

    def sample(self, n: int = 2000, seed: int = 0, shift: float = 0.0):
        rng = np.random.default_rng(seed)
        demand = rng.normal(self.mean + shift, self.sd * (1 + 0.5 * abs(shift) / 20),
                            n)
        costs = np.stack([self.cost(a, demand) for a in self.actions], axis=1)
        return TaskSample(context=demand.reshape(-1, 1), outcomes=-costs,
                          optimal=np.argmin(costs, axis=1))


class TriageTask:
    """Treat or wait, given a noisy risk signal.

    The oracle knows the true probability of deterioration, which the policy sees
    only through a noisy observation. That gap is deliberate: a benchmark whose
    oracle is achievable measures nothing about decision quality.
    """

    name = "triage"
    actions = np.array([0, 1])       # 0 = wait, 1 = treat

    def __init__(self, treat_cost: float = 1.0, miss_cost: float = 8.0,
                 noise: float = 0.9):
        self.treat_cost, self.miss_cost, self.noise = treat_cost, miss_cost, noise

    def sample(self, n: int = 3000, seed: int = 0, shift: float = 0.0):
        rng = np.random.default_rng(seed)
        latent = rng.normal(-1.4 + shift, 1.0, n)
        p_true = 1 / (1 + np.exp(-latent))
        observed = latent + rng.normal(0, self.noise, n)
        cost_wait = p_true * self.miss_cost
        cost_treat = np.full(n, self.treat_cost)
        costs = np.stack([cost_wait, cost_treat], axis=1)
        return TaskSample(context=observed.reshape(-1, 1), outcomes=-costs,
                          optimal=np.argmin(costs, axis=1),
                          probs=np.stack([p_true, 1 - p_true], axis=1))


TASKS = {"newsvendor": NewsvendorTask, "triage": TriageTask}
