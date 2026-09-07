import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.evaluate import cvar, regret, summarise_losses, var
from src.problems import (portfolio_loss, sample_demand, sample_returns,
                          staffing_cost)
from src.solvers import (cvar_portfolio, cvar_staffing, deterministic_portfolio,
                         deterministic_staffing, robust_portfolio, saa_portfolio,
                         saa_staffing)

ALPHA = 0.90


# --- risk measures on known inputs ---------------------------------------

def test_cvar_of_a_uniform_tail():
    losses = np.arange(100.0)
    assert abs(cvar(losses, 0.90) - np.mean(np.arange(90, 100.0))) < 1e-9


def test_cvar_is_at_least_var():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert cvar(x, 0.90) >= var(x, 0.90)


def test_cvar_of_a_constant_is_that_constant():
    assert abs(cvar(np.full(50, 3.0), 0.9) - 3.0) < 1e-12


def test_regret_is_zero_for_the_pointwise_best():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    out = regret({"a": a, "b": b})
    assert out["a"]["mean_regret"] == 0.0
    assert out["a"]["frac_best"] == 1.0
    assert out["b"]["mean_regret"] > 0


# --- scenario generators -------------------------------------------------

def test_returns_have_a_crash_regime():
    """A single Gaussian would make every method score identically."""
    s = sample_returns(n_assets=6, n_scenarios=4000, seed=0)
    port = s.values.mean(axis=1)
    assert np.quantile(port, 0.01) < np.quantile(port, 0.5) - 4 * port.std() / 2


def test_demand_is_positive_and_correlated():
    s = sample_demand(n_units=5, n_scenarios=2000, seed=1)
    assert (s.values > 0).all()
    C = np.corrcoef(s.values.T)
    assert np.mean(C[np.triu_indices(5, 1)]) > 0.1


def test_scenario_probabilities_sum_to_one():
    for s in (sample_returns(seed=0), sample_demand(seed=1)):
        assert abs(s.probs.sum() - 1.0) < 1e-12


# --- portfolio solvers ---------------------------------------------------

def test_all_portfolios_are_valid_simplex_points():
    tr = sample_returns(n_assets=8, n_scenarios=400, seed=0)
    for fn in (deterministic_portfolio, saa_portfolio, cvar_portfolio,
               robust_portfolio):
        w = fn(tr)
        assert abs(w.sum() - 1.0) < 1e-6, fn.__name__
        assert (w >= -1e-9).all(), fn.__name__


def test_cvar_lp_minimises_in_sample_cvar():
    """The LP must be optimal on the objective it solves.

    Tested UNCONSTRAINED. Comparing a position-limited CVaR portfolio against an
    unconstrained one from another method tests the constraint, not the solver:
    a limited optimum is allowed to be worse than an unlimited portfolio, and an
    assertion that ignores that is checking the wrong thing.
    """
    tr = sample_returns(n_assets=8, n_scenarios=600, seed=0)
    w_cvar = cvar_portfolio(tr, alpha=ALPHA, max_weight=1.0,
                            min_return=-1e9)          # objective only
    c_cvar = cvar(portfolio_loss(w_cvar, tr.values), ALPHA)
    for other in (deterministic_portfolio(tr), saa_portfolio(tr),
                  robust_portfolio(tr), np.full(8, 1 / 8)):
        assert c_cvar <= cvar(portfolio_loss(other, tr.values), ALPHA) + 1e-6


def test_position_limit_costs_in_sample_cvar():
    """Constraining the solution must not improve the unconstrained optimum."""
    tr = sample_returns(n_assets=8, n_scenarios=600, seed=0)
    free = cvar(portfolio_loss(
        cvar_portfolio(tr, ALPHA, max_weight=1.0, min_return=-1e9), tr.values), ALPHA)
    limited = cvar(portfolio_loss(
        cvar_portfolio(tr, ALPHA, max_weight=0.35), tr.values), ALPHA)
    assert free <= limited + 1e-9


def test_position_limit_is_respected():
    """Regression: without it the LP put everything in two assets."""
    tr = sample_returns(n_assets=8, n_scenarios=600, seed=0)
    w = cvar_portfolio(tr, alpha=ALPHA, max_weight=0.35)
    assert w.max() <= 0.35 + 1e-6


def test_position_limit_forces_diversification():
    tr = sample_returns(n_assets=8, n_scenarios=600, seed=0)
    loose = cvar_portfolio(tr, max_weight=1.0)
    tight = cvar_portfolio(tr, max_weight=0.35)
    assert float(np.sum(tight ** 2)) < float(np.sum(loose ** 2))
    assert int((tight > 0.01).sum()) > int((loose > 0.01).sum())


def test_cvar_beats_the_deterministic_plan_out_of_sample():
    tr = sample_returns(n_assets=8, n_scenarios=600, seed=0)
    te = sample_returns(n_assets=8, n_scenarios=4000, seed=99)
    c = cvar(portfolio_loss(cvar_portfolio(tr, ALPHA), te.values), ALPHA)
    d = cvar(portfolio_loss(deterministic_portfolio(tr), te.values), ALPHA)
    assert c < d


def test_robust_is_less_concentrated_than_deterministic():
    tr = sample_returns(n_assets=8, n_scenarios=600, seed=0)
    r = float(np.sum(robust_portfolio(tr) ** 2))
    d = float(np.sum(deterministic_portfolio(tr) ** 2))
    assert r < d


def test_higher_gamma_makes_the_robust_solution_more_defensive():
    tr = sample_returns(n_assets=8, n_scenarios=600, seed=0)
    lo = robust_portfolio(tr, gamma=0.0)
    hi = robust_portfolio(tr, gamma=8.0)
    assert not np.allclose(lo, hi)


# --- staffing solvers ----------------------------------------------------

def test_staffing_plans_respect_the_budget():
    tr = sample_demand(n_units=6, n_scenarios=300, seed=1)
    budget = float(tr.mean().sum()) * 1.25
    for fn in (deterministic_staffing, saa_staffing, cvar_staffing):
        x = fn(tr, budget)
        assert abs(x.sum() - budget) < 1e-3, fn.__name__
        assert (x >= -1e-9).all(), fn.__name__


def test_staffing_cost_penalises_unmet_more_than_idle():
    d = np.array([[10.0, 10.0]])
    understaffed = staffing_cost(np.array([8.0, 10.0]), d, unmet_penalty=6.0)
    overstaffed = staffing_cost(np.array([12.0, 10.0]), d, unmet_penalty=6.0)
    assert understaffed[0] > overstaffed[0]


def test_saa_beats_the_mean_demand_plan_out_of_sample():
    tr = sample_demand(n_units=6, n_scenarios=600, seed=1)
    te = sample_demand(n_units=6, n_scenarios=3000, seed=77)
    budget = float(tr.mean().sum()) * 1.25
    a = staffing_cost(saa_staffing(tr, budget), te.values).mean()
    b = staffing_cost(deterministic_staffing(tr, budget), te.values).mean()
    assert a < b


def test_cvar_staffing_lp_wins_in_sample_on_its_own_objective():
    """Regression: minimising a numpy quantile inside SLSQP returned a plan
    worse than the naive baseline. A CVaR method losing on CVaR in-sample is a
    solver failure, not a finding."""
    tr = sample_demand(n_units=6, n_scenarios=400, seed=1)
    budget = float(tr.mean().sum()) * 1.25
    c = cvar(staffing_cost(cvar_staffing(tr, budget, ALPHA), tr.values), ALPHA)
    d = cvar(staffing_cost(deterministic_staffing(tr, budget), tr.values), ALPHA)
    assert c <= d + 1e-6


def test_summarise_orders_the_risk_measures():
    rng = np.random.default_rng(3)
    x = rng.normal(size=4000)
    s = summarise_losses(x, 0.90)
    assert s["mean"] < s["var"] <= s["cvar"] <= s["worst"]
