"""Which input drove this decision?

Two methods, because they answer different questions and disagree in an
informative way.

`occlusion` asks what happens if one feature is replaced by its typical value --
cheap, and blind to interactions.

`shapley` distributes the decision across features by averaging over orderings,
which is the only allocation satisfying efficiency, symmetry, dummy and
additivity. Exact for small feature counts; sampled above that, with the
sampling error reported rather than hidden.
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def occlusion(policy, inputs: dict, baseline: dict) -> dict:
    """Score change when each feature is replaced by its baseline value."""
    _, base_score = policy(inputs)
    out = {}
    for k in inputs:
        if k not in baseline:
            continue
        perturbed = {**inputs, k: baseline[k]}
        _, s = policy(perturbed)
        out[k] = round(float(base_score) - float(s), 6)
    return out


def shapley(policy, inputs: dict, baseline: dict, features: list | None = None,
            n_samples: int | None = None, seed: int = 0) -> dict:
    """Shapley values of the decision score.

    Exact enumeration up to 8 features (8! = 40,320 orderings is fine); above
    that, permutation sampling. The mode used is returned so a reader knows
    whether the numbers are exact.
    """
    feats = features or [k for k in inputs if k in baseline]
    n = len(feats)

    def value(subset) -> float:
        """Score with `subset` at its actual value and the rest at baseline."""
        x = {**baseline, **{k: inputs[k] for k in subset}}
        for k in inputs:
            x.setdefault(k, inputs[k])
        return float(policy(x)[1])

    phi = {f: 0.0 for f in feats}
    if n <= 8 and n_samples is None:
        for perm in itertools.permutations(feats):
            cur, prev = [], value([])
            for f in perm:
                cur.append(f)
                v = value(cur)
                phi[f] += v - prev
                prev = v
        total = math.factorial(n)
        return {"values": {k: round(v / total, 6) for k, v in phi.items()},
                "mode": "exact", "n_orderings": total}

    rng = np.random.default_rng(seed)
    m = n_samples or 400
    for _ in range(m):
        perm = list(rng.permutation(feats))
        cur, prev = [], value([])
        for f in perm:
            cur.append(f)
            v = value(cur)
            phi[f] += v - prev
            prev = v
    return {"values": {k: round(v / m, 6) for k, v in phi.items()},
            "mode": "sampled", "n_orderings": m}


def check_efficiency(policy, inputs: dict, baseline: dict, phi: dict) -> dict:
    """Shapley values must sum to the score difference. A correctness check."""
    full = float(policy(inputs)[1])
    base = float(policy({**baseline, **{k: v for k, v in inputs.items()
                                        if k not in baseline}})[1])
    total = sum(phi.values())
    return {"sum_of_values": round(total, 6),
            "score_difference": round(full - base, 6),
            "residual": round(total - (full - base), 9),
            "efficient": abs(total - (full - base)) < 1e-6}
