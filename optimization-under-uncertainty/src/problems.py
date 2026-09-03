"""Two allocation problems that share a structure and differ in what a bad
decision costs.

Portfolio allocation: choose weights over assets with uncertain returns. A bad
decision costs money.

Hospital resource allocation: choose staffing across units with uncertain
demand. A bad decision costs unmet demand, which is not symmetric with the cost
of over-staffing and cannot be netted against it.

They are in one module because the same solvers apply to both, and because the
asymmetry in the second is what makes expected-value optimisation the wrong
default rather than merely a simplification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scenario:
    """A sampled realisation of the uncertain parameters."""
    values: np.ndarray          # S x N
    probs: np.ndarray           # S

    def n_scenarios(self) -> int:
        return self.values.shape[0]

    def n_vars(self) -> int:
        return self.values.shape[1]

    def mean(self) -> np.ndarray:
        return self.probs @ self.values

    def cov(self) -> np.ndarray:
        c = self.values - self.mean()
        return (c * self.probs[:, None]).T @ c


def sample_returns(n_assets: int = 8, n_scenarios: int = 600,
                   seed: int = 0, regime_prob: float = 0.15) -> Scenario:
    """Asset returns with a crash regime.

    A mixture, not a Gaussian. The whole point of robust and CVaR formulations
    is the tail, and a single Gaussian has no tail worth protecting against --
    on Gaussian data every method here would look identical, which would make
    the comparison meaningless.
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.03, 0.11, n_assets)
    vol = rng.uniform(0.10, 0.28, n_assets)
    # Correlation rises in the crash regime; that is the mechanism that makes
    # diversification fail exactly when it is needed.
    base_corr = 0.25
    crash_corr = 0.80

    def draw(corr, shift, scale, n):
        C = np.full((n_assets, n_assets), corr)
        np.fill_diagonal(C, 1.0)
        L = np.linalg.cholesky(C)
        z = rng.normal(size=(n, n_assets)) @ L.T
        return mu + shift + z * vol * scale

    n_crash = int(n_scenarios * regime_prob)
    normal = draw(base_corr, 0.0, 1.0, n_scenarios - n_crash)
    crash = draw(crash_corr, -0.22, 1.9, n_crash)
    vals = np.vstack([normal, crash])
    rng.shuffle(vals)
    return Scenario(values=vals, probs=np.full(n_scenarios, 1.0 / n_scenarios))


def sample_demand(n_units: int = 6, n_scenarios: int = 600,
                  seed: int = 1, surge_prob: float = 0.12) -> Scenario:
    """Hospital unit demand, in patients per shift, with surge scenarios."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(14, 42, n_units)
    disp = rng.uniform(0.16, 0.34, n_units)
    n_surge = int(n_scenarios * surge_prob)
    normal = rng.lognormal(np.log(base), disp, (n_scenarios - n_surge, n_units))
    # A surge hits several units at once -- an epidemic or a mass-casualty
    # event is correlated across units, which is what breaks unit-by-unit
    # planning.
    shock = rng.lognormal(0.55, 0.20, (n_surge, 1))
    surge = rng.lognormal(np.log(base), disp, (n_surge, n_units)) * shock
    vals = np.vstack([normal, surge])
    rng.shuffle(vals)
    return Scenario(values=vals, probs=np.full(n_scenarios, 1.0 / n_scenarios))


# --- objectives -----------------------------------------------------------

def portfolio_loss(w: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Loss per scenario. Negative return, so lower is better throughout."""
    return -(returns @ w)


def staffing_cost(x: np.ndarray, demand: np.ndarray,
                  unmet_penalty: float = 6.0,
                  idle_cost: float = 1.0) -> np.ndarray:
    """Cost per scenario of a staffing vector against realised demand.

    The asymmetry is the whole problem. Unmet demand is charged at
    `unmet_penalty` per unit and idle capacity at `idle_cost`; with the default
    6:1 ratio, a plan optimised for average demand is systematically understaffed
    because it treats the two as interchangeable.
    """
    gap = demand - x                       # S x N
    unmet = np.clip(gap, 0, None)
    idle = np.clip(-gap, 0, None)
    return (unmet * unmet_penalty + idle * idle_cost).sum(axis=1)
