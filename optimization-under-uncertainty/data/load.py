"""Turn the real price tape into train and test Scenarios.

The split is the thing that changes, and it changes the meaning of the whole
comparison.

With simulated data you can draw a training sample and a test sample
independently from the same distribution. That is a fair test of estimation
error and nothing else: the future is guaranteed to look like the past, so a
method that overfits the training sample is penalised and a method that assumes
stationarity is not.

Real markets give you one history. The only honest split is in TIME: fit on the
earlier period, judge on the later one. That test is harder and it is the one
that matters, because it penalises the assumption every one of these
formulations makes -- that the distribution estimated from the past still holds.
A random split of real returns would leak the test period's regime into the
training set and quietly restore the guarantee that the simulation gives away.
"""
from __future__ import annotations

import pathlib
from datetime import date, timedelta

import numpy as np

from .datakit import Fetcher, FetchError
from .marketdata import align, parse_french_industries, parse_stooq, to_returns

ROOT = pathlib.Path(__file__).resolve().parent

# The optimiser builds a dense scenario matrix (the CVaR LP is S x S), so the
# full 1926-2026 history is neither tractable nor the point. A recent window
# long enough to contain more than one drawdown -- 2020 and 2022 both sit inside
# it -- is what makes the risk-aware comparison mean something.
REAL_WINDOW_YEARS = 12


def _french_industry_key(files):
    return next((k for k in files
                 if k.startswith("french/") and k.endswith("industry_daily.zip")),
                None)


def _load_french(f, key, rec, window_years):
    """Return (names, R, ret_dates, prov, source) from the industry zip.

    French quotes returns directly, so there is no price-to-return step. The
    ten industries share one calendar; the recent-window cut is applied here so
    the LP stays a size scipy's HiGHS can solve.
    """
    dates, data = parse_french_industries((f.raw / key).read_bytes())
    cutoff = date.today() - timedelta(days=window_years * 365)
    names = sorted(data)
    keep = [i for i, d in enumerate(dates) if d >= cutoff]
    if len(keep) < 2:
        raise FetchError(
            f"only {len(keep)} industry return days inside the last "
            f"{window_years} years; the file may be older than expected")
    ret_dates = [dates[i] for i in keep]
    R = np.column_stack([[data[n][i] for i in keep] for n in names])
    prov = [{"symbol": n, "status": "ok", "n_days": len(keep),
             "first": str(ret_dates[0]), "last": str(ret_dates[-1]),
             "sha256": rec["sha256"][:16], "url": rec["url"]} for n in names]
    source = (
        "real daily returns of the 10 Fama-French industry portfolios "
        "(value-weighted); see data/MANIFEST.json for the URL, hash and "
        "retrieval time")
    return names, R, ret_dates, prov, source


def _load_stooq(f, cached):
    """Return (names, R, ret_dates, prov, source) from cached Stooq price CSVs."""
    series, prov = {}, []
    for dest, rec in sorted(cached.items()):
        sym = pathlib.Path(dest).stem
        try:
            dates, closes = parse_stooq((f.raw / dest).read_bytes())
        except ValueError as exc:
            prov.append({"symbol": sym, "status": f"unusable: {exc}"})
            continue
        series[sym] = (dates, closes)
        prov.append({"symbol": sym, "status": "ok", "n_closes": len(closes),
                     "first": str(dates[0]), "last": str(dates[-1]),
                     "sha256": rec["sha256"][:16], "url": rec["url"]})

    if len(series) < 2:
        raise FetchError(f"only {len(series)} usable series; need at least 2")

    dates, aligned = align(series)
    names = sorted(aligned)
    R = np.column_stack([to_returns(aligned[n]) for n in names])
    ret_dates = dates[1:]
    source = ("real daily closes from Stooq; see data/MANIFEST.json for URLs, "
              "hashes and retrieval times")
    return names, R, ret_dates, prov, source


def load_scenarios(root=ROOT, train_frac: float = 0.6, min_days: int = 750):
    """Return (train, test, provenance) as chronologically split Scenarios.

    The real return series come from the Fama-French 10-industry daily zip when
    it is cached; a Stooq price cache is the fallback. Either way the split is
    chronological, which is the only honest split for one real history.
    """
    from src.problems import Scenario

    f = Fetcher(root)
    man = f.load_manifest()
    french_key = _french_industry_key(man["files"])
    stooq = {k: v for k, v in man["files"].items() if k.startswith("stooq/")}

    if french_key:
        names, R, ret_dates, prov, data_source = _load_french(
            f, french_key, man["files"][french_key], REAL_WINDOW_YEARS)
    elif stooq:
        names, R, ret_dates, prov, data_source = _load_stooq(f, stooq)
    else:
        raise FetchError(
            "no real return data cached. Run `python -m data.fetch` in a "
            "networked environment first; this project will not fit an "
            "allocation to simulated returns and report it as a real one.")

    if len(ret_dates) < min_days:
        raise FetchError(
            f"only {len(ret_dates)} overlapping trading days; need {min_days}. "
            f"A CVaR estimate at the 90th percentile from a short window is "
            f"fitted to a handful of tail observations.")

    cut = int(len(R) * train_frac)
    if cut < 250 or len(R) - cut < 250:
        raise FetchError(
            f"a {train_frac:.0%} split leaves {cut} training and "
            f"{len(R) - cut} test days; both sides need at least 250")

    def scen(block):
        return Scenario(values=block,
                        probs=np.full(len(block), 1.0 / len(block)))

    train, test = scen(R[:cut]), scen(R[cut:])
    meta = {
        "assets": names,
        "n_assets": len(names),
        "data_source": data_source,
        "split": "chronological",
        "split_rationale":
            "real markets give one history, so the only honest split is in "
            "time. A random split leaks the test period's regime into the "
            "training set and restores the stationarity guarantee that makes "
            "the simulated comparison easy.",
        "train": {"n_days": int(cut), "first": str(ret_dates[0]),
                  "last": str(ret_dates[cut - 1])},
        "test": {"n_days": int(len(R) - cut), "first": str(ret_dates[cut]),
                 "last": str(ret_dates[-1])},
        "worst_day_train": round(float(train.values.mean(axis=1).min()), 5),
        "worst_day_test": round(float(test.values.mean(axis=1).min()), 5),
        "series": prov,
        "staffing_remains_simulated_because":
            "there is no public series of per-unit hospital staffing demand to "
            "download; a proxy would not make it a staffing study.",
    }
    return train, test, meta
