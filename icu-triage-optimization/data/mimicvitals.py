"""Turning MIMIC-IV's irregular charting into regular vital-sign series.

Shared by every project here that consumes ICU records, so the resampling rule
is written down once. That rule is the part worth stating plainly:

  Values are carried FORWARD, never backwards, and only for a bounded window.
  A measurement taken at 14:00 must not appear at 13:00, because the clinician
  did not have it then; and a blood pressure from four hours ago is not a
  current blood pressure. Interpolating across a gap invents an observation
  that nobody made.

The single backwards step is the prefix before a stay's first observation,
where no alternative exists and no future information is involved.
"""
from __future__ import annotations

import numpy as np

from .datakit import FetchError, Fetcher
from .physionet import parse_ts, read_csv, to_float

PROJECT = "mimic-iv-demo"

# MIMIC-IV itemids. Both invasive (arterial line) and non-invasive cuff
# measurements are accepted; an ICU patient may have either or both.
ITEMS = {
    "heart_rate_bpm": [220045],
    "spo2_pct": [220277],
    "resp_rate": [220210, 224690],
    "map_mmhg": [220052, 220181, 225312],      # ABP mean, NIBP mean, cuff mean
    "sbp": [220050, 220179],                   # systolic, for pulse pressure
    "dbp": [220051, 220180],                   # diastolic
    "temp_c": [223762],                        # Celsius
    "temp_f": [223761],                        # Fahrenheit, converted below
}

PLAUSIBLE = {
    "heart_rate_bpm": (20, 220), "spo2_pct": (50, 100),
    "resp_rate": (4, 60), "map_mmhg": (30, 160),
    "sbp": (50, 260), "dbp": (20, 160), "temp_c": (30, 43),
}

STEP_H = 0.5
CARRY_FORWARD_H = 2.0      # a 4-hour-old blood pressure is not a current one


def ages(f) -> dict:
    p = f.raw / f"{PROJECT}/hosp/patients.csv.gz"
    if not p.exists():
        return {}
    out = {}
    for r in read_csv(p):
        a = to_float(r.get("anchor_age"))
        if a is not None:
            out[str(r.get("subject_id"))] = a
    return out


def collect_events(f, stay_ids: set) -> dict:
    """Stream chartevents into {stay_id: {vital: [(hours, value)]}}."""
    wanted = {}
    for vital, ids in ITEMS.items():
        for i in ids:
            wanted[i] = vital

    rows = read_csv(f.raw / f"{PROJECT}/icu/chartevents.csv.gz")
    by_stay: dict = {}
    t0: dict = {}
    for r in rows:
        try:
            sid = int(r["stay_id"])
            itemid = int(r["itemid"])
        except (KeyError, ValueError, TypeError):
            continue
        if sid not in stay_ids or itemid not in wanted:
            continue
        val = to_float(r.get("valuenum"))
        ts = parse_ts(r.get("charttime"))
        if val is None or ts is None:
            continue
        vital = wanted[itemid]
        if vital == "temp_f":
            val, vital = (val - 32.0) * 5.0 / 9.0, "temp_c"
        lo, hi = PLAUSIBLE.get(vital, (-1e9, 1e9))
        if not (lo <= val <= hi):
            continue        # charting errors are common and are not signal
        if sid not in t0 or ts < t0[sid]:
            t0[sid] = ts
        by_stay.setdefault(sid, {}).setdefault(vital, []).append((ts, val))

    out = {}
    for sid, vitals in by_stay.items():
        base = t0[sid]
        out[sid] = {v: sorted(((ts - base).total_seconds() / 3600.0, x)
                              for ts, x in pairs)
                    for v, pairs in vitals.items()}
    return out


def resample(pairs, grid, default):
    """Carry the last observation forward, but only for a bounded time.

    Never interpolate: a value between two observations is information the
    clinician did not have at that moment.
    """
    out = np.full(len(grid), np.nan)
    j = 0
    last_t, last_v = None, None
    for i, t in enumerate(grid):
        while j < len(pairs) and pairs[j][0] <= t:
            last_t, last_v = pairs[j]
            j += 1
        if last_v is not None and (t - last_t) <= CARRY_FORWARD_H:
            out[i] = last_v
    if np.isnan(out).all():
        return None
    # Leading gap before the first observation: back-fill with the first value.
    # This is the one backwards step, and it is bounded to the pre-observation
    # prefix, where no alternative exists.
    first = np.flatnonzero(~np.isnan(out))
    if len(first):
        out[:first[0]] = out[first[0]]
    idx = np.flatnonzero(np.isnan(out))
    if len(idx):
        good = np.flatnonzero(~np.isnan(out))
        out[idx] = np.interp(idx, good, out[good])
    return out


def require_cache(root):
    """Return a Fetcher over the real cache, or explain what to run.

    Every project that reads MIMIC goes through here, so the refusal to fall
    back to simulated data is implemented once rather than remembered five
    times.
    """
    f = Fetcher(root)
    need = [f"{PROJECT}/icu/chartevents.csv.gz", f"{PROJECT}/icu/icustays.csv.gz"]
    missing = [n for n in need if not (f.raw / n).exists()]
    if missing:
        raise FetchError(
            f"missing real MIMIC-IV files: {missing}. Run `python -m data.fetch` "
            f"in a networked environment first; this project will not report "
            f"results from simulated data as if they came from MIMIC.")
    return f


def stay_series(root, min_hours: float = 12.0, max_stays: int = 0):
    """Return (stays, provenance) where each stay carries gridded vitals.

    stays: [{stay_id, subject_id, age, times, vitals{name: array}}]
    """
    f = require_cache(root)
    man = f.load_manifest()
    rows = read_csv(f.raw / f"{PROJECT}/icu/icustays.csv.gz")
    age_by_subject = ages(f)
    collected = collect_events(f, {int(r["stay_id"]) for r in rows})

    defaults = {"map_mmhg": 85.0, "heart_rate_bpm": 80.0, "spo2_pct": 97.0,
                "resp_rate": 16.0, "temp_c": 36.8, "sbp": 120.0, "dbp": 70.0}

    stays, skipped = [], {"too_short": 0, "no_vitals": 0}
    for r in sorted(rows, key=lambda x: int(x["stay_id"])):
        sid = int(r["stay_id"])
        have = {v: p for v, p in (collected.get(sid) or {}).items() if p}
        if not have or ("spo2_pct" not in have and "map_mmhg" not in have):
            skipped["no_vitals"] += 1
            continue
        span = max(p[-1][0] for p in have.values())
        grid = np.arange(0.0, span, STEP_H)
        if span < min_hours or len(grid) < 10:
            skipped["too_short"] += 1
            continue

        cols = {}
        for v in defaults:
            got = resample(have[v], grid, defaults[v]) if have.get(v) else None
            cols[v] = got if got is not None else np.full(len(grid), defaults[v])
        if not have.get("map_mmhg") and have.get("sbp") and have.get("dbp"):
            # The standard approximation, used only when no mean was charted.
            cols["map_mmhg"] = cols["dbp"] + (cols["sbp"] - cols["dbp"]) / 3.0

        stays.append({
            "stay_id": sid, "subject_id": str(r.get("subject_id")),
            "age": age_by_subject.get(str(r.get("subject_id")), 65.0),
            "times": grid,
            "vitals": {
                "map_mmhg": cols["map_mmhg"],
                "heart_rate_bpm": cols["heart_rate_bpm"],
                "spo2_pct": cols["spo2_pct"], "resp_rate": cols["resp_rate"],
                "temp_c": cols["temp_c"],
                "pulse_pressure": cols["sbp"] - cols["dbp"],
            },
        })
        if max_stays and len(stays) >= max_stays:
            break

    if not stays:
        raise FetchError("no ICU stay in the demo yielded a usable vital-sign "
                         "series; check that chartevents downloaded completely")

    prov = {
        "database": f"PhysioNet {PROJECT} (open access)",
        "n_stays_in_file": len(rows), "n_stays_built": len(stays),
        "skipped": skipped, "grid_step_hours": STEP_H,
        "carry_forward_limit_hours": CARRY_FORWARD_H,
        "cohort_is_a_demonstration_not_a_study": True,
        "files": {k: {"sha256": v["sha256"][:16],
                      "retrieved_utc": v.get("retrieved_utc"),
                      "bytes": v["bytes"]}
                  for k, v in man["files"].items()},
    }
    return stays, prov


def events_from_vitals(vitals) -> dict:
    """Clinical thresholds, stated so a reader can disagree with them."""
    return {"hypoxemia": vitals["spo2_pct"] < 90.0,
            "hypotension": vitals["map_mmhg"] < 65.0}


EVENT_DEFINITIONS = {"hypoxemia": "SpO2 < 90%",
                     "hypotension": "mean arterial pressure < 65 mmHg"}
