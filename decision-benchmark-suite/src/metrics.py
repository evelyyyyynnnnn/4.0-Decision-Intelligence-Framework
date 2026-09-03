"""Regret, calibration and robustness — the three axes of the suite."""

from __future__ import annotations

import numpy as np


def regret(chosen: np.ndarray, outcomes: np.ndarray,
           optimal: np.ndarray) -> dict:
    """Payoff foregone against the oracle, per decision.

    Reported as a distribution rather than a mean. A policy with low mean regret
    and a long tail is a different proposition from a steady one, and averaging
    hides exactly the cases a decision-maker cares about.
    """
    n = len(chosen)
    got = outcomes[np.arange(n), chosen]
    best = outcomes[np.arange(n), optimal]
    r = best - got
    return {"mean": round(float(r.mean()), 6),
            "median": round(float(np.median(r)), 6),
            "p90": round(float(np.percentile(r, 90)), 6),
            "max": round(float(r.max()), 6),
            "frac_optimal": round(float(np.mean(r <= 1e-12)), 4),
            "total": round(float(r.sum()), 4)}


def calibration(pred_probs: np.ndarray, outcomes_binary: np.ndarray,
                bins: int = 10) -> dict:
    """Expected calibration error of a policy's stated confidence.

    Included because a decision system that cannot say how sure it is cannot be
    escalated from. Accuracy alone does not tell you when to ask a human.
    """
    p = np.asarray(pred_probs, float)
    y = np.asarray(outcomes_binary, float)
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not m.any():
            continue
        pred, obs = float(p[m].mean()), float(y[m].mean())
        rows.append({"bin": round((lo + hi) / 2, 3), "n": int(m.sum()),
                     "predicted": round(pred, 4), "observed": round(obs, 4)})
        ece += m.mean() * abs(pred - obs)
    return {"ece": round(float(ece), 5), "bins": rows}


def robustness(scores_by_shift: dict) -> dict:
    """How performance degrades as the distribution moves away from training.

    `degradation` is the slope of regret against shift magnitude, and
    `worst_case` is the largest regret over the shifts tested. Both are relative
    to the named shift set -- a robustness number without the shift set attached
    is not interpretable.
    """
    shifts = sorted(scores_by_shift)
    vals = [scores_by_shift[s] for s in shifts]
    base = scores_by_shift.get(0.0, vals[0])
    mags = [abs(s) for s in shifts]
    if len(set(mags)) > 1:
        slope = float(np.polyfit(mags, vals, 1)[0])
    else:
        slope = 0.0
    return {"baseline_regret": round(float(base), 6),
            "worst_case_regret": round(float(max(vals)), 6),
            "degradation_per_unit_shift": round(slope, 6),
            "relative_worst": round(float(max(vals) / base), 4) if base else None,
            "shifts": {str(s): round(float(v), 6)
                       for s, v in zip(shifts, vals)}}
