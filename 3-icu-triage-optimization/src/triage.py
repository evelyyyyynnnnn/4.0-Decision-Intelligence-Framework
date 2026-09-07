"""Allocating a limited number of review slots across patients.

The alerting question asks where to put a threshold. The triage question is
different and harder: given that only k patients can be reviewed this shift, who
are they? A threshold answers that badly, because it ignores how many slots
exist and produces however many alerts it produces.
"""

from __future__ import annotations

import numpy as np


def top_k_policy(scores: np.ndarray, k: int) -> np.ndarray:
    """Review the k highest-risk patients. The obvious baseline."""
    idx = np.argsort(-np.asarray(scores, float))[:k]
    out = np.zeros(len(scores), bool)
    out[idx] = True
    return out


def threshold_policy(scores: np.ndarray, thr: float) -> np.ndarray:
    return np.asarray(scores, float) >= thr


def expected_benefit_policy(scores: np.ndarray, k: int,
                            benefit_if_caught: np.ndarray | None = None) -> np.ndarray:
    """Review the k patients with the highest expected benefit.

    Risk and benefit are not the same quantity. A patient certain to deteriorate
    but for whom review changes nothing should rank below a patient at moderate
    risk whose course review would change. Ranking by risk alone is the standard
    approach and it is wrong whenever the two diverge.
    """
    s = np.asarray(scores, float)
    b = np.ones_like(s) if benefit_if_caught is None else np.asarray(
        benefit_if_caught, float)
    idx = np.argsort(-(s * b))[:k]
    out = np.zeros(len(s), bool)
    out[idx] = True
    return out


def evaluate_policy(selected: np.ndarray, y: np.ndarray,
                    benefit: np.ndarray | None = None) -> dict:
    y = np.asarray(y)
    b = np.ones(len(y)) if benefit is None else np.asarray(benefit, float)
    caught = int(np.sum(selected & (y == 1)))
    missed = int(np.sum(~selected & (y == 1)))
    wasted = int(np.sum(selected & (y == 0)))
    return {
        "reviewed": int(selected.sum()),
        "caught": caught, "missed": missed, "wasted_reviews": wasted,
        "catch_rate": round(caught / max(1, caught + missed), 4),
        "review_yield": round(caught / max(1, int(selected.sum())), 4),
        "benefit_realised": round(float(np.sum(b[selected & (y == 1)])), 4),
        "benefit_available": round(float(np.sum(b[y == 1])), 4),
        "benefit_capture": round(
            float(np.sum(b[selected & (y == 1)]) / max(1e-9, np.sum(b[y == 1]))), 4),
    }
