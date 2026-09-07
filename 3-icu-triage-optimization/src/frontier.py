"""Pareto frontiers over alert thresholds.

An ICU alerting policy has no single best setting, because the two things it
trades off are not commensurable: a missed deterioration and a false alarm are
different kinds of harm and no exchange rate between them is a technical fact.

So this module does not pick a threshold. It computes the frontier of
non-dominated options and leaves the choice where it belongs -- with the people
who carry the consequences. What it *can* do is show which settings are
dominated, and those can be discarded on technical grounds alone: another
setting is better on every axis at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Operating:
    threshold: float
    sensitivity: float
    false_alerts_per_100: float
    alerts_per_100: float
    ppv: float
    missed: int
    nurse_load: float          # alerts per nurse-shift, a workload proxy

    def objectives(self) -> tuple:
        """(missed rate, alert burden). Both to be minimised."""
        return (1.0 - self.sensitivity, self.false_alerts_per_100)

    def as_dict(self) -> dict:
        return {"threshold": round(self.threshold, 5),
                "sensitivity": round(self.sensitivity, 4),
                "false_alerts_per_100": round(self.false_alerts_per_100, 3),
                "alerts_per_100": round(self.alerts_per_100, 3),
                "ppv": round(self.ppv, 4), "missed": int(self.missed),
                "nurse_load": round(self.nurse_load, 3)}


def operating_points(y: np.ndarray, scores: np.ndarray,
                     n_points: int = 60, beds_per_nurse: int = 4) -> list:
    """Sweep the alert threshold across the score range."""
    y = np.asarray(y)
    s = np.asarray(scores, float)
    qs = np.linspace(0.001, 0.999, n_points)
    out = []
    for q in qs:
        thr = float(np.quantile(s, q))
        pred = s >= thr
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        n = len(y)
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        out.append(Operating(
            threshold=thr, sensitivity=sens,
            false_alerts_per_100=100.0 * fp / n,
            alerts_per_100=100.0 * (tp + fp) / n,
            ppv=tp / (tp + fp) if (tp + fp) else 0.0,
            missed=fn,
            nurse_load=(tp + fp) / max(1, n // beds_per_nurse)))
    return out


def pareto_front(points: list) -> list:
    """Non-dominated points. Both objectives minimised.

    A point is dominated when another is at least as good on both objectives and
    strictly better on one. Dominated settings can be discarded without any
    value judgement, which is the only part of this problem that is technical.
    """
    front = []
    for p in points:
        a = p.objectives()
        dominated = any(
            all(b <= ai + 1e-12 for b, ai in zip(q.objectives(), a))
            and any(b < ai - 1e-12 for b, ai in zip(q.objectives(), a))
            for q in points if q is not p)
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p.objectives()[0])


def hypervolume(front: list, ref: tuple = (1.0, 100.0)) -> float:
    """Area dominated by the frontier, relative to a reference point.

    One number for comparing whole frontiers, which is what is needed when
    asking whether a model gives better options than a baseline rather than
    whether it wins at one threshold.
    """
    pts = sorted(((p.objectives()[0], p.objectives()[1]) for p in front))
    area, prev_x = 0.0, 0.0
    best_y = ref[1]
    for x, y in pts:
        if y >= best_y:
            continue
        area += (x - prev_x) * (ref[1] - best_y)
        prev_x, best_y = x, y
    area += (ref[0] - prev_x) * (ref[1] - best_y)
    return float(area)


def knee_point(front: list) -> Operating | None:
    """The point furthest from the line joining the frontier's extremes.

    Offered as a starting point for a conversation, not an answer. Where the
    curve bends hardest is where a small change in one objective stops buying
    much of the other -- useful to look at, and not a substitute for someone
    deciding what a missed deterioration is worth.
    """
    if len(front) < 3:
        return front[0] if front else None
    P = np.array([p.objectives() for p in front], float)
    rng = P.max(0) - P.min(0)
    rng[rng == 0] = 1.0
    Q = (P - P.min(0)) / rng
    a, b = Q[0], Q[-1]
    ab = b - a
    denom = np.linalg.norm(ab) or 1.0
    # The 2-D cross product written out. np.cross on 2-vectors is deprecated in
    # NumPy 2.0 and will be removed; this is the same quantity, |ab x aq|.
    aq = Q - a
    d = np.abs(ab[0] * aq[:, 1] - ab[1] * aq[:, 0]) / denom
    return front[int(np.argmax(d))]


def constrained_best(points: list, min_sensitivity: float) -> Operating | None:
    """Fewest false alerts among settings meeting a required sensitivity.

    This is how an alerting policy is actually specified: clinical governance
    fixes the sensitivity it will accept, and the optimisation happens inside
    that constraint.
    """
    ok = [p for p in points if p.sensitivity >= min_sensitivity]
    return min(ok, key=lambda p: p.false_alerts_per_100) if ok else None
