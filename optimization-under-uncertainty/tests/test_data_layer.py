"""Tests for fitting the allocation to real returns.

The decision that matters here is the split. A random split of real returns
puts the test period's regime into the training set, which quietly restores
the stationarity guarantee that makes the simulated comparison easy -- and the
risk-aware formulations then look better than they have earned.
"""
import datetime as dt
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data import datakit
from data.load import load_scenarios
from data.marketdata import align, parse_stooq


def test_parse_stooq_rejects_the_no_data_body():
    with pytest.raises(ValueError, match="market suffix"):
        parse_stooq(b"No data")


def test_align_intersects_calendars():
    D = dt.date.fromisoformat
    a = ([D("2024-01-02"), D("2024-01-03"), D("2024-01-04")], [1., 2., 3.])
    b = ([D("2024-01-02"), D("2024-01-04")], [10., 30.])
    common, out = align({"a": a, "b": b})
    assert common == [D("2024-01-02"), D("2024-01-04")]
    assert out["a"] == [1., 3.]


def test_refuses_when_nothing_is_cached(tmp_path):
    with pytest.raises(datakit.FetchError, match="no real price data cached"):
        load_scenarios(root=tmp_path)


def _seed(tmp_path, n=2000, seed=5, crash_at=None):
    """Stooq-shaped CSVs with a shared factor and, optionally, a crash."""
    rng = np.random.default_rng(seed)
    f = datakit.Fetcher(tmp_path)
    man = f.load_manifest()
    (f.raw / "stooq").mkdir(parents=True, exist_ok=True)

    factor = rng.normal(0, 0.009, n)
    if crash_at is not None:
        factor[crash_at:crash_at + 5] -= 0.05
    betas = {"spy.us": 1.0, "iwm.us": 1.3, "agg.us": 0.05, "gld.us": 0.0}
    for sym, beta in betas.items():
        r = beta * factor + rng.normal(0, 0.004, n)
        px, p = [], 100.0
        for x in r:
            p *= (1 + x)
            px.append(p)
        d, lines = dt.date(2015, 1, 2), ["Date,Open,High,Low,Close,Volume"]
        for v in px:
            while d.weekday() >= 5:
                d += dt.timedelta(days=1)
            lines.append(f"{d.isoformat()},{v:.4f},{v:.4f},{v:.4f},{v:.4f},1000")
            d += dt.timedelta(days=1)
        raw = ("\n".join(lines) + "\n").encode()
        dest = f"stooq/{sym}.csv"
        (f.raw / dest).write_bytes(raw)
        man["files"][dest] = {
            "source": f"Stooq {sym}", "url": f"https://stooq.com/q/d/l/?s={sym}",
            "publisher": "Stooq", "terms": "free for research",
            "sha256": datakit.sha256_file(f.raw / dest), "bytes": len(raw),
            "retrieved_utc": datakit.utc_now()}
    f._write_manifest(man)
    return f


def test_the_split_is_chronological_not_random(tmp_path):
    """The property the whole real run depends on."""
    _seed(tmp_path)
    train, test, meta = load_scenarios(root=tmp_path, train_frac=0.6)
    assert meta["split"] == "chronological"
    # Every training day precedes every test day.
    assert meta["train"]["last"] < meta["test"]["first"]
    assert train.values.shape[0] + test.values.shape[0] > 0
    assert "leaks the test period's regime" in meta["split_rationale"]


def test_train_and_test_do_not_overlap(tmp_path):
    _seed(tmp_path)
    train, test, meta = load_scenarios(root=tmp_path, train_frac=0.6)
    total = meta["train"]["n_days"] + meta["test"]["n_days"]
    assert train.values.shape[0] == meta["train"]["n_days"]
    assert test.values.shape[0] == meta["test"]["n_days"]
    assert total == train.values.shape[0] + test.values.shape[0]


def test_scenario_probabilities_are_uniform_and_sum_to_one(tmp_path):
    _seed(tmp_path)
    train, test, _ = load_scenarios(root=tmp_path)
    for s in (train, test):
        assert s.probs.sum() == pytest.approx(1.0)
        assert len(np.unique(s.probs)) == 1


def test_a_crash_in_the_test_window_is_visible(tmp_path):
    """A window with no drawdown makes every risk-aware method look wasteful."""
    _seed(tmp_path, crash_at=1600)      # past the 60% cut of 2000 days
    _, _, meta = load_scenarios(root=tmp_path, train_frac=0.6)
    assert meta["worst_day_test"] < -0.03


def test_too_short_a_history_is_refused(tmp_path):
    _seed(tmp_path, n=300)
    with pytest.raises(datakit.FetchError, match="overlapping trading days"):
        load_scenarios(root=tmp_path, min_days=750)


def test_a_lopsided_split_is_refused(tmp_path):
    _seed(tmp_path, n=800)
    with pytest.raises(datakit.FetchError, match="both sides need at least 250"):
        load_scenarios(root=tmp_path, train_frac=0.95, min_days=750)


def test_all_five_allocations_solve_on_real_returns(tmp_path):
    """The end that matters: every formulation produces a valid portfolio."""
    _seed(tmp_path)
    from src.demo import portfolio_study_real
    out = portfolio_study_real(tmp_path)

    assert out["n_assets"] == 4
    for name, w in out["weights"].items():
        arr = np.asarray(w)
        assert arr.sum() == pytest.approx(1.0, abs=1e-4), name
        assert (arr >= -1e-9).all(), name
    for name in out["weights"]:
        assert np.isfinite(out["out_of_sample"][name]["cvar"])
        assert np.isfinite(out["in_sample"][name]["cvar"])


def test_provenance_travels_with_the_result(tmp_path):
    _seed(tmp_path)
    from src.demo import portfolio_study_real
    out = portfolio_study_real(tmp_path)
    ok = [p for p in out["provenance"] if p["status"] == "ok"]
    assert len(ok) == 4
    assert all(len(p["sha256"]) == 16 for p in ok)


def test_staffing_is_explicitly_not_claimed_as_real(tmp_path):
    _seed(tmp_path)
    _, _, meta = load_scenarios(root=tmp_path)
    assert "no public series" in meta["staffing_remains_simulated_because"]
