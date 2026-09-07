import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.frontier import (Operating, constrained_best, hypervolume, knee_point,
                          operating_points, pareto_front)
from src.triage import (evaluate_policy, expected_benefit_policy,
                        threshold_policy, top_k_policy)


def _data(n=4000, sep=0.6, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.15).astype(int)
    s = rng.uniform(size=n) + y * sep
    return y, s


# --- frontier ------------------------------------------------------------

def test_sensitivity_falls_as_the_threshold_rises():
    y, s = _data()
    pts = operating_points(y, s, n_points=40)
    sens = [p.sensitivity for p in pts]
    assert sens == sorted(sens, reverse=True)


def test_false_alerts_fall_as_the_threshold_rises():
    y, s = _data()
    fa = [p.false_alerts_per_100 for p in operating_points(y, s, n_points=40)]
    assert fa == sorted(fa, reverse=True)


def test_single_sweep_domination_comes_only_from_a_saturated_objective():
    """Two wrong claims preceded this one, and both were caught by a test.

    First: "sweeping one score's threshold can never produce a dominated point,
    because the trade-off is monotone." False -- once sensitivity plateaus at
    1.0, further loosening adds false alerts and catches nothing more.

    Second: "so every dominated point is dominated by one with the SAME
    sensitivity." Also false. On a well-separated score, false alerts reach zero
    while sensitivity is still climbing, and the domination then runs along the
    other axis: a point with sensitivity 0.83 and no false alerts is dominated
    by one with sensitivity 0.998 and no false alerts.

    The true statement covers both: a dominated point on a single sweep always
    has a dominator that TIES it on one objective. Verified here across a range
    of class separations, since the failure only appears at some of them.
    """
    total_dominated = 0
    for sep in (0.05, 0.35, 1.0, 2.0):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            n = 3000
            y = (rng.uniform(size=n) < 0.15).astype(int)
            s = rng.uniform(size=n) + y * sep
            pts = operating_points(y, s, n_points=40)
            front = pareto_front(pts)
            dominated = [p for p in pts if not any(p is f for f in front)]
            total_dominated += len(dominated)
            for p in dominated:
                a = p.objectives()
                tie = [q for q in pts if q is not p
                       and (abs(q.objectives()[0] - a[0]) < 1e-12
                            or abs(q.objectives()[1] - a[1]) < 1e-12)
                       and all(x <= y_ + 1e-12
                               for x, y_ in zip(q.objectives(), a))
                       and any(x < y_ - 1e-12
                               for x, y_ in zip(q.objectives(), a))]
                assert tie, (
                    f"sep={sep} seed={seed}: point {a} is dominated but no "
                    f"dominator ties it on either objective")
    assert total_dominated > 0, "no domination observed; the test proves nothing"


def test_perfect_separation_saturates_false_alerts_before_sensitivity():
    """The case that broke the second claim above."""
    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.uniform(size=n) < 0.15).astype(int)
    s = rng.uniform(size=n) + y * 1.0
    pts = operating_points(y, s, n_points=40)
    zero_fa = [p for p in pts if p.false_alerts_per_100 == 0.0]
    assert len(zero_fa) > 1, "expected a plateau at zero false alerts"
    assert max(p.sensitivity for p in zero_fa) > min(
        p.sensitivity for p in zero_fa)


def test_a_dominated_point_is_excluded():
    a = Operating(0.5, 0.9, 10.0, 20.0, 0.5, 2, 1.0)     # better on both
    b = Operating(0.4, 0.8, 20.0, 30.0, 0.4, 4, 2.0)     # worse on both
    front = pareto_front([a, b])
    assert len(front) == 1 and front[0] is a


def test_hypervolume_rewards_a_better_frontier():
    y, s_good = _data(sep=1.2)
    _, s_bad = _data(sep=0.05, seed=2)
    hv_good = hypervolume(pareto_front(operating_points(y, s_good, 40)))
    hv_bad = hypervolume(pareto_front(operating_points(y, s_bad, 40)))
    assert hv_good > hv_bad


def test_knee_point_lies_on_the_frontier():
    y, s = _data()
    front = pareto_front(operating_points(y, s, 40))
    assert knee_point(front) in front


def test_constrained_best_meets_the_requirement():
    y, s = _data()
    pts = operating_points(y, s, 60)
    for req in (0.7, 0.8, 0.9):
        b = constrained_best(pts, req)
        assert b is not None and b.sensitivity >= req


def test_constrained_best_picks_the_fewest_false_alerts():
    y, s = _data()
    pts = operating_points(y, s, 60)
    b = constrained_best(pts, 0.80)
    eligible = [p for p in pts if p.sensitivity >= 0.80]
    assert b.false_alerts_per_100 == min(p.false_alerts_per_100 for p in eligible)


def test_constrained_best_returns_none_when_unreachable():
    y, s = _data()
    assert constrained_best(operating_points(y, s, 20), 1.01) is None


def test_a_better_model_needs_fewer_false_alerts_at_matched_sensitivity():
    y, good = _data(sep=1.0)
    _, weak = _data(sep=0.1, seed=3)
    g = constrained_best(operating_points(y, good, 60), 0.80)
    w = constrained_best(operating_points(y, weak, 60), 0.80)
    assert g.false_alerts_per_100 < w.false_alerts_per_100


# --- triage --------------------------------------------------------------

def test_top_k_selects_exactly_k():
    s = np.random.default_rng(0).uniform(size=500)
    assert top_k_policy(s, 37).sum() == 37


def test_top_k_selects_the_highest_scores():
    s = np.array([0.1, 0.9, 0.5, 0.7])
    sel = top_k_policy(s, 2)
    assert list(np.where(sel)[0]) == [1, 3]


def test_threshold_policy_ignores_capacity():
    """The reason a threshold is the wrong instrument under a hard capacity."""
    s = np.random.default_rng(1).uniform(size=1000)
    assert threshold_policy(s, 0.1).sum() > 500


def test_benefit_ranking_differs_from_risk_ranking():
    s = np.array([0.9, 0.8, 0.2])
    b = np.array([0.05, 0.1, 5.0])
    assert not np.array_equal(top_k_policy(s, 1),
                              expected_benefit_policy(s, 1, b))


def test_benefit_ranking_captures_more_benefit():
    rng = np.random.default_rng(5)
    n = 3000
    y = (rng.uniform(size=n) < 0.15).astype(int)
    s = np.clip(rng.uniform(size=n) * 0.5 + y * 0.4, 0, 1)
    b = rng.beta(2, 2, n) * 2
    k = 150
    by_risk = evaluate_policy(top_k_policy(s, k), y, b)
    by_benefit = evaluate_policy(expected_benefit_policy(s, k, b), y, b)
    assert by_benefit["benefit_capture"] > by_risk["benefit_capture"]


def test_evaluate_policy_counts_are_consistent():
    y = np.array([1, 0, 1, 0])
    sel = np.array([True, True, False, False])
    out = evaluate_policy(sel, y)
    assert out["caught"] == 1 and out["missed"] == 1 and out["wasted_reviews"] == 1
    assert out["reviewed"] == 2


def test_benefit_capture_is_one_when_everything_is_reviewed():
    y = np.array([1, 0, 1])
    out = evaluate_policy(np.ones(3, bool), y, np.array([1.0, 1.0, 2.0]))
    assert abs(out["benefit_capture"] - 1.0) < 1e-9
