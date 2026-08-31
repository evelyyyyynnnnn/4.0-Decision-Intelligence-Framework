"""Out-of-sample evaluation.

Every method is fitted on one scenario sample and scored on a different one.
That split is the whole discipline: a stochastic program evaluated on the
scenarios it optimised over reports its own objective back to itself, and CVaR
in particular looks spectacular in-sample because it is fitted to that exact
tail.
"""

from __future__ import annotations

import numpy as np


def cvar(losses: np.ndarray, alpha: float = 0.90) -> float:
    tau = np.quantile(losses, alpha)
    tail = losses[losses >= tau]
    return float(tail.mean()) if len(tail) else float(losses.mean())


def var(losses: np.ndarray, alpha: float = 0.90) -> float:
    return float(np.quantile(losses, alpha))


def summarise_losses(losses: np.ndarray, alpha: float = 0.90) -> dict:
    return {
        "mean": round(float(losses.mean()), 6),
        "std": round(float(losses.std()), 6),
        "var": round(var(losses, alpha), 6),
        "cvar": round(cvar(losses, alpha), 6),
        "worst": round(float(losses.max()), 6),
        "p05_best": round(float(np.quantile(losses, 0.05)), 6),
    }


def regret(losses_by_method: dict) -> dict:
    """Regret against the best method on each scenario.

    Reported because a method can have a good average and still be the wrong
    choice in most individual states of the world.
    """
    names = list(losses_by_method)
    L = np.stack([losses_by_method[n] for n in names])   # M x S
    best = L.min(axis=0)
    out = {}
    for i, n in enumerate(names):
        r = L[i] - best
        out[n] = {"mean_regret": round(float(r.mean()), 6),
                  "max_regret": round(float(r.max()), 6),
                  "frac_best": round(float(np.mean(np.isclose(L[i], best))), 4)}
    return out
