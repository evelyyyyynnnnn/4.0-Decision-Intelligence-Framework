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

import numpy as np

from .datakit import Fetcher, FetchError
from .marketdata import align, parse_stooq, to_returns

ROOT = pathlib.Path(__file__).resolve().parent


def load_scenarios(root=ROOT, train_frac: float = 0.6, min_days: int = 750):
    """Return (train, test, provenance) as chronologically split Scenarios."""
    from src.problems import Scenario

    f = Fetcher(root)
    man = f.load_manifest()
    cached = {k: v for k, v in man["files"].items() if k.startswith("stooq/")}
    if not cached:
        raise FetchError(
            "no real price data cached. Run `python -m data.fetch` in a "
            "networked environment first; this project will not fit an "
            "allocation to simulated returns and report it as a real one.")

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
    if len(dates) < min_days:
        raise FetchError(
            f"only {len(dates)} overlapping trading days; need {min_days}. A "
            f"CVaR estimate at the 90th percentile from a short window is "
            f"fitted to a handful of tail observations.")

    names = sorted(aligned)
    R = np.column_stack([to_returns(aligned[n]) for n in names])
    ret_dates = dates[1:]

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
