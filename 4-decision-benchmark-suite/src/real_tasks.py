"""A real-data triage track on the PhysioNet MIMIC-IV demo.

The closed-form tasks in ``tasks.py`` have an EXACT oracle, so regret is
computable. Real data does not: there is no latent truth to measure a decision
against. This track is therefore scored differently, and the difference is the
honest point of including it —

  * no oracle, so no oracle-regret. Policies are scored by *realised* cost, and
    compared to the best in-hindsight fixed threshold on the test split (the
    achievable floor), never to an unreachable clairvoyant;
  * calibration (ECE) of the predicted event probability against realised
    outcomes, which IS meaningful on real data;
  * degradation under a NAMED covariate shift (train on younger patients,
    evaluate on older, and vice-versa) — robustness reported against a stated
    shift, not in the abstract.

Honesty caveat, carried from the ICU project this data is shared with: the
label (a deterioration event inside the horizon) is a threshold on the same
MAP signal that also appears among the features. So this measures short-horizon
persistence of an observed vital, not an independent clinical outcome, and the
cohort (~130 ICU stays from the open demo) is a demonstration, not a study.
Features come from time <= t and the label from a strictly later window, and
decision points already in the event are dropped, so there is no look-ahead.
"""
from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Github-Website / 2.0-Healthcare-Ai-Systems / 2-icu-early-warning / data
_ICU_DATA = (ROOT.parents[1] / "2.0-Healthcare-Ai-Systems"
             / "2-icu-early-warning" / "data")

EVENT = "hypotension"          # MAP < 65 mmHg
HORIZON_HOURS = 2.0
STRIDE_STEPS = 4               # a decision every 2 h, to limit autocorrelation
TREAT_COST = 1.0
MISS_COST = 8.0

FEATURES = ["map_mmhg", "heart_rate_bpm", "spo2_pct", "resp_rate",
            "pulse_pressure", "temp_c"]


def mimic_data_root() -> pathlib.Path | None:
    """The MIMIC-IV demo `data` dir (holding raw/), or None if not cached.

    Order: $MIMIC_DATA_ROOT, the sibling icu-early-warning cache, then a local
    copy under this repo.
    """
    for cand in (os.environ.get("MIMIC_DATA_ROOT"), _ICU_DATA, ROOT / "data"):
        if not cand:
            continue
        p = pathlib.Path(cand)
        if (p / "raw" / "mimic-iv-demo" / "icu" / "chartevents.csv.gz").exists():
            return p
    return None


def available() -> bool:
    return mimic_data_root() is not None


def _stay_series(root: pathlib.Path):
    """Reuse the ICU project's shared resampling, without importing its src/."""
    icu_repo = _ICU_DATA.parent
    if str(icu_repo) not in sys.path:
        sys.path.insert(0, str(icu_repo))
    from data.mimicvitals import events_from_vitals, stay_series  # noqa: E402
    stays, prov = stay_series(root, min_hours=12.0, max_stays=0)
    return stays, prov, events_from_vitals


@dataclass
class RealTriageSample:
    X: np.ndarray            # N x d features, all from time <= t
    y: np.ndarray            # N realised event-in-horizon (0/1)
    subject: np.ndarray      # N subject id per row (split-integrity checks)
    age: np.ndarray          # N patient age per row (the named shift covariate)
    feature_names: list

    def __len__(self) -> int:
        return len(self.y)


def build_sample(root: pathlib.Path | None = None):
    """Turn the real ICU stays into (RealTriageSample, provenance).

    A decision point is a grid time t where the patient is NOT currently in the
    event and a full horizon lies ahead. Its label is whether the event occurs
    in (t, t+H]. Features are the current vitals plus their 1-hour trend.
    """
    root = root or mimic_data_root()
    if root is None:
        raise FileNotFoundError("MIMIC-IV demo cache not found; see data/README.")
    stays, prov, events_from_vitals = _stay_series(root)

    h_steps = int(round(HORIZON_HOURS / prov["grid_step_hours"]))
    trend = int(round(1.0 / prov["grid_step_hours"]))     # 1-hour lookback
    rows_X, rows_y, rows_s, rows_age = [], [], [], []
    n_points_in_event = 0
    for s in stays:
        v = s["vitals"]
        ev = events_from_vitals(v)[EVENT]                 # boolean per grid step
        n = len(s["times"])
        for t in range(trend, n - h_steps, STRIDE_STEPS):
            if ev[t]:                                      # already in the event
                n_points_in_event += 1
                continue
            cur = [float(v[f][t]) for f in FEATURES]
            dmap = float(v["map_mmhg"][t] - v["map_mmhg"][t - trend])
            dspo2 = float(v["spo2_pct"][t] - v["spo2_pct"][t - trend])
            dhr = float(v["heart_rate_bpm"][t] - v["heart_rate_bpm"][t - trend])
            rows_X.append(cur + [dmap, dspo2, dhr, float(s["age"])])
            rows_y.append(int(bool(ev[t + 1:t + 1 + h_steps].any())))
            rows_s.append(str(s["subject_id"]))
            rows_age.append(float(s["age"]))

    sample = RealTriageSample(
        X=np.asarray(rows_X, float), y=np.asarray(rows_y, int),
        subject=np.asarray(rows_s, object), age=np.asarray(rows_age, float),
        feature_names=FEATURES + ["d_map_1h", "d_spo2_1h", "d_hr_1h", "age"])
    prov = dict(prov)
    prov["decision_points"] = len(sample)
    prov["event_rate_in_horizon"] = round(float(sample.y.mean()), 4)
    prov["dropped_points_already_in_event"] = n_points_in_event
    prov["horizon_hours"] = HORIZON_HOURS
    prov["decision_stride_hours"] = STRIDE_STEPS * prov["grid_step_hours"]
    return sample, prov


# --- cost model -----------------------------------------------------------
def realised_cost(actions: np.ndarray, y: np.ndarray) -> float:
    """Mean realised cost. Waiting risks the miss; treating is a fixed premium.

    wait (0): MISS_COST if the event occurs, else 0.   treat (1): TREAT_COST.
    """
    a = np.asarray(actions, int)
    cost = np.where(a == 1, TREAT_COST, MISS_COST * y)
    return float(np.mean(cost))


def subject_split(sample: RealTriageSample, frac: float = 0.5, seed: int = 0):
    """Split rows by SUBJECT so no patient appears in both sides (no leakage)."""
    subs = np.array(sorted(set(sample.subject.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    cut = max(1, int(len(subs) * frac))
    train_subs = set(subs[:cut].tolist())
    tr = np.array([s in train_subs for s in sample.subject])
    return tr, ~tr


def _sub(sample: RealTriageSample, mask: np.ndarray) -> RealTriageSample:
    return RealTriageSample(sample.X[mask], sample.y[mask], sample.subject[mask],
                            sample.age[mask], sample.feature_names)


# --- policies -------------------------------------------------------------
def fit_calibrated_logistic(train: RealTriageSample):
    """Calibrated logistic risk -> treat when p * MISS >= TREAT (Bayes rule).

    Returns (predict_proba_fn, decide_fn). Isotonic calibration by default,
    falling back to Platt if a fold has too few positives.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    base = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    for method in ("isotonic", "sigmoid"):
        try:
            cal = CalibratedClassifierCV(base, method=method, cv=3)
            cal.fit(train.X, train.y)
            break
        except Exception:
            continue
    proba = lambda X: cal.predict_proba(X)[:, 1]
    threshold = TREAT_COST / MISS_COST          # treat when p >= this
    decide = lambda X: (proba(X) >= threshold).astype(int)
    return proba, decide


def fit_threshold_on_map(train: RealTriageSample):
    """A single threshold on current MAP: treat if MAP < tau. tau fit on train."""
    map_col = train.feature_names.index("map_mmhg")
    m = train.X[:, map_col]
    best_tau, best_cost = float(np.median(m)), np.inf
    for tau in np.quantile(m, np.linspace(0.02, 0.98, 60)):
        a = (m < tau).astype(int)
        c = realised_cost(a, train.y)
        if c < best_cost:
            best_tau, best_cost = float(tau), c

    def decide(X):
        return (X[:, map_col] < best_tau).astype(int)
    return decide, best_tau


def best_hindsight_threshold_cost(proba: np.ndarray, y: np.ndarray) -> dict:
    """Lowest realised cost achievable by ANY fixed probability threshold,
    chosen with hindsight on the test labels. The achievable floor."""
    order = np.argsort(proba)
    grid = np.unique(np.quantile(proba, np.linspace(0, 1, 101)))
    best_tau, best_cost = 1.1, MISS_COST * float(np.mean(y))   # never-treat
    for tau in grid:
        a = (proba >= tau).astype(int)
        c = realised_cost(a, y)
        if c < best_cost:
            best_tau, best_cost = float(tau), c
    return {"cost": round(best_cost, 5), "threshold": round(best_tau, 5)}
