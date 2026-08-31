import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.metrics import calibration, regret, robustness
from src.policies import (AlwaysAction, BayesTriage, EmpiricalNewsvendor,
                          MeanDemandNewsvendor, ThresholdTriage)
from src.tasks import NewsvendorTask, TriageTask


# --- oracles are exact ---------------------------------------------------

def test_critical_fractile_matches_the_cost_ratio():
    nv = NewsvendorTask(underage=7.0, overage=2.0)
    assert abs(nv.critical_fractile() - 7 / 9) < 1e-12


def test_oracle_order_is_the_fractile_quantile():
    nv = NewsvendorTask(underage=7.0, overage=2.0, mean=50.0, sd=18.0)
    demand = np.random.default_rng(0).normal(50, 18, 400000)
    empirical = np.quantile(demand, nv.critical_fractile())
    assert abs(nv.oracle_action() - empirical) < 0.5


def test_oracle_order_beats_the_mean_order_in_cost():
    """If this fails, the asymmetry is not being modelled at all."""
    nv = NewsvendorTask()
    d = np.random.default_rng(1).normal(nv.mean, nv.sd, 200000)
    assert nv.cost(nv.oracle_action(), d).mean() < nv.cost(nv.mean, d).mean()


def test_higher_underage_cost_raises_the_optimal_order():
    a = NewsvendorTask(underage=3.0, overage=2.0).oracle_action()
    b = NewsvendorTask(underage=20.0, overage=2.0).oracle_action()
    assert b > a


def test_triage_optimal_matches_the_cost_threshold():
    tr = TriageTask(treat_cost=1.0, miss_cost=8.0)
    s = tr.sample(n=5000, seed=0)
    p_true = s.probs[:, 0]
    expected = (p_true * tr.miss_cost >= tr.treat_cost).astype(int)
    assert (s.optimal == expected).mean() > 0.999


def test_the_oracle_is_not_achievable_from_the_signal():
    """A benchmark whose oracle is reachable measures nothing."""
    tr = TriageTask()
    s = tr.sample(n=4000, seed=0)
    best = BayesTriage(tr).fit(s).decide(s)
    assert (best == s.optimal).mean() < 0.98


# --- metrics -------------------------------------------------------------

def test_regret_is_zero_for_the_optimal_choice():
    out = np.array([[1.0, 2.0], [3.0, 1.0]])
    opt = np.array([1, 0])
    r = regret(opt, out, opt)
    assert r["mean"] == 0.0 and r["frac_optimal"] == 1.0


def test_regret_is_positive_for_a_wrong_choice():
    out = np.array([[1.0, 5.0]])
    r = regret(np.array([0]), out, np.array([1]))
    assert abs(r["mean"] - 4.0) < 1e-12


def test_regret_reports_a_distribution_not_just_a_mean():
    out = np.array([[0.0, 1.0]] * 9 + [[0.0, 100.0]])
    r = regret(np.zeros(10, int), out, np.ones(10, int))
    assert r["p90"] > r["median"]
    assert r["max"] == 100.0


def test_calibration_is_zero_for_a_perfect_forecast():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 60000)
    y = (rng.uniform(size=60000) < p).astype(float)
    assert calibration(p, y)["ece"] < 0.02


def test_calibration_catches_overconfidence():
    p = np.full(2000, 0.99)
    y = (np.arange(2000) % 2).astype(float)      # actually 50% correct
    assert calibration(p, y)["ece"] > 0.4


def test_robustness_slope_is_zero_for_a_flat_policy():
    out = robustness({-2.0: 1.0, 0.0: 1.0, 2.0: 1.0})
    assert abs(out["degradation_per_unit_shift"]) < 1e-9
    assert out["relative_worst"] == 1.0


def test_robustness_slope_is_positive_when_shift_hurts():
    out = robustness({-2.0: 3.0, 0.0: 1.0, 2.0: 3.0})
    assert out["degradation_per_unit_shift"] > 0
    assert out["worst_case_regret"] == 3.0


# --- policies ------------------------------------------------------------

def test_empirical_fractile_beats_ordering_the_mean():
    nv = NewsvendorTask()
    fit = nv.sample(n=3000, seed=0)
    ev = nv.sample(n=6000, seed=101)
    a = EmpiricalNewsvendor(nv).fit(fit)
    b = MeanDemandNewsvendor(nv).fit(fit)
    ra = regret(a.decide(ev), ev.outcomes, ev.optimal)["mean"]
    rb = regret(b.decide(ev), ev.outcomes, ev.optimal)["mean"]
    assert ra < rb


def test_empirical_fractile_sits_near_the_achievable_floor():
    """Its remaining regret is irreducible, not a deficiency."""
    nv = NewsvendorTask()
    ev = nv.sample(n=6000, seed=101)
    demand = ev.context[:, 0]
    floor = min(float(nv.cost(a, demand).mean()) for a in nv.actions)
    clair = float(np.min(np.stack([nv.cost(a, demand) for a in nv.actions], axis=1),
                         axis=1).mean())
    pol = EmpiricalNewsvendor(nv).fit(nv.sample(n=3000, seed=0))
    r = regret(pol.decide(ev), ev.outcomes, ev.optimal)["mean"]
    assert abs(r - (floor - clair)) < 2.0


def test_every_policy_beats_the_worst_constant_policy():
    tr = TriageTask()
    fit, ev = tr.sample(n=2000, seed=0), tr.sample(n=4000, seed=101)
    worst = regret(AlwaysAction(0).fit(fit).decide(ev),
                   ev.outcomes, ev.optimal)["mean"]
    for pol in (BayesTriage(tr), ThresholdTriage(tr), AlwaysAction(1)):
        r = regret(pol.fit(fit).decide(ev), ev.outcomes, ev.optimal)["mean"]
        assert r < worst


def test_fitted_threshold_matches_bayes_on_a_one_dimensional_signal():
    """Expected: a monotone posterior makes a threshold the Bayes rule."""
    tr = TriageTask()
    fit, ev = tr.sample(n=4000, seed=0), tr.sample(n=6000, seed=101)
    rb = regret(BayesTriage(tr).fit(fit).decide(ev), ev.outcomes, ev.optimal)["mean"]
    rt = regret(ThresholdTriage(tr).fit(fit).decide(ev),
                ev.outcomes, ev.optimal)["mean"]
    assert abs(rb - rt) < 0.02 * max(rb, rt) + 0.005


def _regret_at(pol, task, shift):
    ev = task.sample(4000, seed=101, shift=shift)
    return regret(pol.decide(ev), ev.outcomes, ev.optimal)["mean"]


def test_worst_case_shift_degrades_a_fitted_policy():
    """Degradation is a worst-case property, not a directional one.

    An earlier version asserted that a +1.2 shift makes things worse. It does
    not: shifting the population toward deterioration makes "treat" correct more
    often, which HELPS a policy whose threshold is already low. Robustness is
    about the worst shift in the set, and asserting a direction tests the
    generator's sign convention rather than the policy.
    """
    tr = TriageTask()
    pol = ThresholdTriage(tr).fit(tr.sample(n=3000, seed=0))
    base = _regret_at(pol, tr, 0.0)
    worst = max(_regret_at(pol, tr, s) for s in (-1.2, -0.6, 0.0, 0.6, 1.2))
    assert worst > base


def test_a_context_free_policy_degrades_more_than_a_fitted_one():
    """The comparison robustness is actually for."""
    tr = TriageTask()
    fit = tr.sample(n=3000, seed=0)
    fitted = ThresholdTriage(tr).fit(fit)
    constant = AlwaysAction(1).fit(fit)
    shifts = (-1.2, -0.6, 0.0, 0.6, 1.2)
    f = robustness({s: _regret_at(fitted, tr, s) for s in shifts})
    c = robustness({s: _regret_at(constant, tr, s) for s in shifts})
    assert f["degradation_per_unit_shift"] < c["degradation_per_unit_shift"]
