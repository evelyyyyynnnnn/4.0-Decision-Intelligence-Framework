"""Pull the Statlog German Credit dataset -- real credit applications.

    python -m data.fetch --list
    python -m data.fetch
    python -m data.fetch --verify

Chosen over the other public credit datasets for one reason: its attributes
line up with the decisions this framework audits without being bent to fit.
It records an installment rate as a percentage of disposable income (a real
debt-service ratio), a credit history category that encodes past delinquency,
employment duration, credit amount and age -- and a class label saying whether
the applicant turned out to be a good or bad credit risk.

That label is what the synthetic version cannot offer. A hash-chained ledger
and Shapley attribution can be demonstrated on invented applicants, but
"do the declines actually correspond to defaults" needs outcomes, and here
there are 1,000 of them.

The dataset ships two files. german.data is the categorical original;
german.data-numeric is a pre-encoded variant. The categorical file is used
because its codes carry meaning -- A34 is "critical account", not "3".
"""
from __future__ import annotations

import pathlib
import sys

from .datakit import Fetcher, FetchError, NetworkBlocked, Source

ROOT = pathlib.Path(__file__).resolve().parent

UCI = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"

SOURCES = [Source(
    name="Statlog German Credit Data", url=UCI, dest="uci/german-credit.zip",
    publisher="UCI Machine Learning Repository",
    terms="UCI ML Repository, free for research use; cite the repository",
    note="1,000 credit applications with attributes and a good/bad risk class",
)]


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
            print(f"{s.name}\n  {s.url}\n  -> raw/{s.dest}\n  {s.note}")
        return 0
    if args.verify:
        problems = f.verify()
        for p in problems:
            print("  " + p)
        print("VERIFICATION FAILED" if problems else
              f"all {len(f.load_manifest()['files'])} cached file(s) verified")
        return 1 if problems else 0

    print("fetching the Statlog German Credit dataset ...")
    try:
        f.get_all(SOURCES, refresh=args.refresh)
    except NetworkBlocked as e:
        print(f"\nBLOCKED: {e}", file=sys.stderr)
        return 2
    except FetchError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1
    print(f"\nwrote {f.manifest_path}")
    print("run `python -m src.demo --real` to audit decisions on real applications")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
