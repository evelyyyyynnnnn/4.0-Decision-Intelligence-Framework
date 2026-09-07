"""Pull the real return history the portfolio allocation is fitted to.

    python -m data.fetch --list
    python -m data.fetch
    python -m data.fetch --verify

The ten Fama-French industry portfolios, daily: consumer non-durables,
durables, manufacturing, energy, tech, telecom, shops, health, utilities and a
residual. Breadth matters here more than in most places: the comparison between
deterministic, SAA, CVaR and robust allocation only has something to show when
the assets differ in their tails, and ten sectors that behave differently in a
crash give a risk-aware allocation something to earn.

These are the canonical academic series, published by Kenneth French and
retrievable by anyone with no account or API key, so the fetch is reproducible
in a way a bot-walled per-ticker price feed is not.

Only the portfolio problem becomes real. The hospital staffing problem stays
simulated, and it should: there is no public series of per-unit hospital
staffing demand to download. Saying so is better than substituting a proxy and
calling the result a staffing study.
"""
from __future__ import annotations

import pathlib
import sys

from .datakit import Fetcher, FetchError, NetworkBlocked
from .marketdata import french_industries_source

ROOT = pathlib.Path(__file__).resolve().parent

SOURCES = [french_industries_source()]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    f = Fetcher(ROOT)

    if args.list:
        for s in SOURCES:
            print(f"{s.name}\n  {s.url}\n  {s.note}")
        print(f"\n{len(SOURCES)} file: 10 daily industry return series")
        print("the staffing problem has no public demand series and stays "
              "simulated")
        return 0
    if args.verify:
        problems = f.verify()
        for p in problems:
            print("  " + p)
        print("VERIFICATION FAILED" if problems else
              f"all {len(f.load_manifest()['files'])} cached file(s) verified")
        return 1 if problems else 0

    print(f"fetching {len(SOURCES)} file: the 10-industry daily return series")
    try:
        f.get_all(SOURCES, refresh=args.refresh)
    except NetworkBlocked as e:
        print(f"\nBLOCKED: {e}", file=sys.stderr)
        return 2
    except FetchError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1
    print(f"\nwrote {f.manifest_path}")
    print("run `python -m src.demo --real` to fit on the past and judge on "
          "the future")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
