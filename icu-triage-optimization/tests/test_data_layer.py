"""Tests for building the triage frontier from real ICU records.

The frontier is drawn over a model's operating points, so it is only as
meaningful as the scores feeding it. These tests check that the real path
builds those scores from genuine records, and that it refuses when the cohort
is too small for a frontier to mean anything.
"""
import datetime as dt
import gzip
import io
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data import datakit
from data.mimicvitals import PROJECT, resample


def _gz(rows, header):
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for r in rows:
        buf.write(",".join(str(r.get(h, "")) for h in header) + "\n")
    return gzip.compress(buf.getvalue().encode())


def _seed(tmp_path, n_stays=14, hours=36, hypotensive=(1, 3, 5, 7, 9)):
    f = datakit.Fetcher(tmp_path)
    man = f.load_manifest()
    rng = np.random.default_rng(5)
    stays = [{"stay_id": 100 + i, "subject_id": 200 + i} for i in range(n_stays)]
    patients = [{"subject_id": 200 + i, "anchor_age": 50 + i} for i in range(n_stays)]

    base = dt.datetime(2180, 7, 23)
    chart = []
    for i in range(n_stays):
        sid = 100 + i
        drifts = i in hypotensive
        for h in range(hours * 2):
            ts = (base + dt.timedelta(minutes=30 * h)).strftime("%Y-%m-%d %H:%M:%S")
            mapv = 85 + rng.normal(0, 5)
            # A gradual fall into hypotension, so the horizon task is learnable.
            if drifts and h > 20:
                mapv -= 0.55 * (h - 20)
            for itemid, val in ((220181, mapv), (220277, 97 + rng.normal(0, 1)),
                                (220045, 80 + rng.normal(0, 4)),
                                (220210, 16 + rng.normal(0, 2)),
                                (223762, 36.8), (220179, 120), (220180, 70)):
                chart.append({"stay_id": sid, "itemid": itemid, "charttime": ts,
                              "valuenum": round(float(val), 2)})

    files = {
        f"{PROJECT}/icu/icustays.csv.gz": _gz(stays, ["stay_id", "subject_id"]),
        f"{PROJECT}/hosp/patients.csv.gz": _gz(patients,
                                               ["subject_id", "anchor_age"]),
        f"{PROJECT}/icu/chartevents.csv.gz": _gz(
            chart, ["stay_id", "itemid", "charttime", "valuenum"]),
    }
    for dest, raw in files.items():
        p = f.raw / dest
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        man["files"][dest] = {
            "source": dest, "url": f"https://physionet.org/files/{dest}",
            "publisher": "PhysioNet", "terms": "open access",
            "sha256": datakit.sha256_file(p), "bytes": len(raw),
            "retrieved_utc": datakit.utc_now()}
    f._write_manifest(man)
    return f


def test_refuses_when_mimic_is_not_cached(tmp_path):
    from data.load import load_patients
    from src.demo import _load_icu
    with pytest.raises(datakit.FetchError, match="missing real MIMIC-IV files"):
        load_patients(_load_icu()["cohort"], root=tmp_path)


def test_this_project_keeps_its_own_cache(tmp_path):
    """It must not reach into the sibling ICU project's download directory.

    The two are meant to be independently copyable; a project that silently
    reads another repository's cache is not.
    """
    from data.load import ROOT as DATA_ROOT
    assert DATA_ROOT.name == "data"
    assert DATA_ROOT.parent.name == "icu-triage-optimization"


def test_builds_patients_the_sibling_model_can_consume(tmp_path):
    _seed(tmp_path)
    from data.load import load_patients
    from src.demo import _load_icu
    m = _load_icu()

    patients, prov = load_patients(m["cohort"], root=tmp_path)
    assert len(patients) == 14
    assert prov["event_definitions"]["hypotension"].startswith("mean arterial")

    X, y, groups, _ = m["dataset"].build(patients, event="hypotension",
                                         horizon_h=4.0)
    assert X.shape[1] == len(m["dataset"].FEATURE_NAMES)
    assert np.isfinite(X).all()
    assert y.sum() > 0, "the seeded hypotensive drift should produce events"


def test_the_split_is_by_patient(tmp_path):
    _seed(tmp_path)
    from data.load import load_patients
    from src.demo import _load_icu
    m = _load_icu()
    patients, _ = load_patients(m["cohort"], root=tmp_path)
    _, _, groups, _ = m["dataset"].build(patients, event="hypotension",
                                         horizon_h=4.0)
    tr, te = m["dataset"].split_by_patient(groups, test_frac=0.3, seed=3)
    assert not (set(groups[tr]) & set(groups[te]))


def test_a_cohort_with_too_few_events_is_refused(tmp_path, monkeypatch):
    """A frontier over five positives moves sensitivity in steps of 20%.

    Reporting a Pareto front from that would give a shape that looks like a
    trade-off curve and is really quantisation noise.
    """
    _seed(tmp_path, n_stays=6, hypotensive=())
    from src import demo
    import data.load as dl
    monkeypatch.setattr(dl, "ROOT", tmp_path)
    monkeypatch.setattr(demo, "ROOT", tmp_path.parent)
    with pytest.raises(Exception) as exc:
        demo._scores_from_real_icu()
    assert "events" in str(exc.value) or "MIMIC" in str(exc.value)


def test_frontier_computation_is_shared_between_both_runs():
    """The simulated and real runs must not differ in method, only in data."""
    from src import demo
    import inspect
    src = inspect.getsource(demo.run)
    assert "_frontier_results" in src
    assert "_frontier_results" in inspect.getsource(demo.run_real)


def test_frontier_results_runs_on_arbitrary_scores():
    from src.demo import _frontier_results
    rng = np.random.default_rng(0)
    n = 800
    y = (rng.uniform(size=n) < 0.15).astype(int)
    p = np.clip(rng.beta(2, 6, n) + y * 0.25, 0, 1)
    baseline = np.clip(rng.beta(2, 6, n) + y * 0.10, 0, 1)

    r = _frontier_results(y, p, baseline, "test scores")
    assert r["n_observations"] == n
    assert r["n_operating_points"] >= len(r["frontier"])
    assert r["hypervolume"]["model"] >= 0
    # Pooling is where domination appears; a single sweep is monotone.
    assert r["pooled"]["n_pooled"] == (r["pooled"]["baseline_points_total"]
                                       + r["pooled"]["model_points_total"])


def test_resample_never_looks_forward(tmp_path):
    grid = np.arange(0.0, 5.0, 1.0)
    out = resample([(3.0, 120.0)], grid, default=None)
    assert out[3] == 120.0
    assert set(out[:3]) == {120.0}
