"""Deterministic, stochastic (SAA), CVaR and robust formulations.

All four solve the same allocation problem and differ only in what they optimise
against: the mean scenario, the average over scenarios, the worst tail, or an
adversarially chosen parameter inside an uncertainty set.

Implemented with scipy's SLSQP rather than a modelling language so the
repository has no solver dependency. That limits problem size and is stated
rather than hidden -- a production version belongs in cvxpy or Pyomo.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize


def _simplex_constraints(n: int):
    return ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)


def _bounds(n: int, lo: float = 0.0, hi: float = 1.0):
    return [(lo, hi)] * n


# --- portfolio ------------------------------------------------------------

def deterministic_portfolio(scen, risk_aversion: float = 0.0) -> np.ndarray:
    """Optimise against the mean scenario only. The naive baseline."""
    mu = scen.mean()
    n = len(mu)
    Sigma = scen.cov()

    def obj(w):
        return -(mu @ w) + risk_aversion * (w @ Sigma @ w)

    res = minimize(obj, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=_bounds(n), constraints=_simplex_constraints(n),
                   options={"maxiter": 400, "ftol": 1e-10})
    return np.clip(res.x, 0, None) / max(1e-12, np.clip(res.x, 0, None).sum())


def saa_portfolio(scen, risk_aversion: float = 3.0) -> np.ndarray:
    """Sample-average approximation: mean-variance over the empirical scenarios."""
    R = scen.values
    p = scen.probs
    n = R.shape[1]

    def obj(w):
        losses = -(R @ w)
        return float(p @ losses + risk_aversion * np.sqrt(p @ (losses - p @ losses) ** 2))

    res = minimize(obj, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=_bounds(n), constraints=_simplex_constraints(n),
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0, None)
    return w / max(1e-12, w.sum())


def cvar_portfolio(scen, alpha: float = 0.90,
                   min_return: float | None = None,
                   max_weight: float = 0.35) -> np.ndarray:
    """Minimise Conditional Value-at-Risk via the Rockafellar-Uryasev LP.

    The reformulation is what makes this tractable: minimising the mean of the
    worst (1-alpha) tail looks combinatorial, and this turns it into a linear
    program in (w, tau, u).

    Variables: [w (n), tau (1), u (S)]
    minimise  tau + 1/((1-alpha) S) * sum(u)
    s.t.      u_s >= loss_s(w) - tau,  u >= 0,  sum(w) = 1,  w >= 0
              mu' w >= min_return                    (the return target)

    `max_weight` is the constraint that makes this behave, and its absence was a
    real error in the first version. Pure CVaR minimisation over a simplex is a
    linear program, so its optimum sits at a VERTEX: the solver put the whole
    portfolio into two assets and the resulting weights scored worse out of
    sample than equal weighting.

    The obvious diagnosis -- overfitting a tail estimated from 60 scenarios --
    was wrong, and a scenario-count sweep is what ruled it out: the disadvantage
    did not shrink as the sample grew to 6,400, and concentration stayed at
    1.000. That pointed at the formulation rather than the sample.

    Adding a return target did not help either, because the concentrated
    solution already had the HIGHEST expected return, so the constraint never
    bound. What was missing is a position limit, which every real mandate
    carries and without which an LP has no reason to diversify at all.
    """
    R = scen.values
    S, n = R.shape
    k = 1.0 / ((1.0 - alpha) * S)
    mu = scen.mean()
    if min_return is None:
        # Default target: the mean return of the equal-weight portfolio. Any
        # sensible target works; having none at all does not.
        min_return = float(mu.mean())

    c = np.concatenate([np.zeros(n), [1.0], np.full(S, k)])
    # -R w - tau - u <= 0   (loss = -Rw)
    A_ub = [np.hstack([-R, -np.ones((S, 1)), -np.eye(S)])]
    b_ub = [np.zeros(S)]
    # -mu' w <= -min_return
    A_ub.append(np.concatenate([-mu, [0.0], np.zeros(S)]).reshape(1, -1))
    b_ub.append(np.array([-min_return]))

    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(S)]).reshape(1, -1)
    b_eq = np.array([1.0])
    bounds = ([(0.0, max_weight)] * n + [(None, None)] + [(0.0, None)] * S)

    res = linprog(c, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return np.full(n, 1.0 / n)
    w = np.clip(res.x[:n], 0, None)
    return w / max(1e-12, w.sum())


def robust_portfolio(scen, gamma: float = 1.0) -> np.ndarray:
    """Box-uncertainty robust mean-variance.

    Worst case over mu in [mu_hat - gamma*se, mu_hat + gamma*se]. With a box
    set and long-only weights the inner maximisation is closed form -- the
    adversary sets every asset to its lower bound -- so no nested optimisation
    is needed.
    """
    R = scen.values
    S, n = R.shape
    mu = scen.mean()
    se = R.std(axis=0) / np.sqrt(S)
    mu_worst = mu - gamma * se
    Sigma = scen.cov()

    def obj(w):
        return -(mu_worst @ w) + 3.0 * (w @ Sigma @ w)

    res = minimize(obj, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=_bounds(n), constraints=_simplex_constraints(n),
                   options={"maxiter": 400, "ftol": 1e-10})
    w = np.clip(res.x, 0, None)
    return w / max(1e-12, w.sum())


# --- staffing -------------------------------------------------------------

def deterministic_staffing(scen, budget: float) -> np.ndarray:
    """Staff to mean demand, scaled to the budget. What a spreadsheet does."""
    mu = scen.mean()
    return mu * (budget / mu.sum())


def _staffing_lp(scen, budget: float, unmet_penalty: float, idle_cost: float,
                 alpha: float | None):
    """Exact LP for expected-cost or CVaR staffing.

    The cost is piecewise linear in the staffing vector, so both objectives have
    exact linear programs and neither needs a general-purpose optimiser. That
    matters: the first version minimised a numpy quantile inside SLSQP, which is
    non-smooth, and the optimiser returned a plan WORSE than staffing to mean
    demand -- a CVaR method that loses to the naive baseline on its own metric
    is a solver failure, not a finding.

    Variables: x (n), unmet (S*n), idle (S*n), and for CVaR also tau (1), u (S).
    """
    D = scen.values
    S, n = D.shape
    nx, nu = n, S * n
    off_un, off_id = nx, nx + nu
    n_base = nx + 2 * nu

    rows, cols, vals, b_ub = [], [], [], []
    r = 0
    # unmet_si >= d_si - x_i   ->  -unmet - x <= -d
    # idle_si  >= x_i - d_si   ->  -idle  + x <=  d
    for s in range(S):
        for i in range(n):
            rows += [r, r]; cols += [off_un + s * n + i, i]; vals += [-1.0, -1.0]
            b_ub.append(-D[s, i]); r += 1
            rows += [r, r]; cols += [off_id + s * n + i, i]; vals += [-1.0, 1.0]
            b_ub.append(D[s, i]); r += 1

    if alpha is None:
        n_var = n_base
        c = np.zeros(n_var)
        c[off_un:off_un + nu] = unmet_penalty / S
        c[off_id:off_id + nu] = idle_cost / S
    else:
        off_tau, off_u = n_base, n_base + 1
        n_var = n_base + 1 + S
        c = np.zeros(n_var)
        c[off_tau] = 1.0
        c[off_u:off_u + S] = 1.0 / ((1.0 - alpha) * S)
        # u_s >= cost_s - tau  ->  sum_i(p*unmet + c*idle) - tau - u_s <= 0
        for s in range(S):
            for i in range(n):
                rows += [r, r]
                cols += [off_un + s * n + i, off_id + s * n + i]
                vals += [unmet_penalty, idle_cost]
            rows += [r, r]; cols += [off_tau, off_u + s]; vals += [-1.0, -1.0]
            b_ub.append(0.0); r += 1

    from scipy.sparse import coo_matrix
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, n_var)).tocsr()
    A_eq = np.zeros((1, n_var)); A_eq[0, :n] = 1.0
    bounds = ([(0.0, budget)] * n + [(0.0, None)] * (2 * nu)
              + ([] if alpha is None else [(None, None)] + [(0.0, None)] * S))
    res = linprog(c, A_ub=A_ub, b_ub=np.array(b_ub), A_eq=A_eq,
                  b_eq=np.array([budget]), bounds=bounds, method="highs")
    if not res.success:
        return scen.mean() * (budget / scen.mean().sum())
    return np.clip(res.x[:n], 0, None)


def saa_staffing(scen, budget: float, unmet_penalty: float = 6.0,
                 idle_cost: float = 1.0) -> np.ndarray:
    """Minimise expected cost over the sampled scenarios, subject to a budget."""
    return _staffing_lp(scen, budget, unmet_penalty, idle_cost, alpha=None)


def cvar_staffing(scen, budget: float, alpha: float = 0.90,
                  unmet_penalty: float = 6.0,
                  idle_cost: float = 1.0) -> np.ndarray:
    """Minimise the mean cost of the worst (1-alpha) demand scenarios."""
    return _staffing_lp(scen, budget, unmet_penalty, idle_cost, alpha=alpha)
