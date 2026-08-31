"""Pull the MIMIC-IV clinical demo -- real ICU vital-sign records.

    python -m data.fetch --list
    python -m data.fetch
    python -m data.fetch --verify

This is the OPEN-ACCESS demo of MIMIC-IV: about 100 ICU stays, same schema as
the full database, no credentialing and no data use agreement. The full
database needs a credentialed PhysioNet account and a signed DUA, which would
make this pipeline unreproducible for anyone reading the repository.

The size is the limitation and it is a real one. A hundred stays is enough to
show that the pipeline parses genuine clinical records, resamples irregular
charting onto a regular grid, and produces calibrated risk. It is nowhere near
enough to estimate a discrimination statistic anyone should act on, and the
results say so instead of quoting a confidence interval built on 100 people.

chartevents.csv.gz is the large one -- tens of megabytes even in the demo.
"""
from __future__ import annotations

import pathlib
import sys

from .datakit import Fetcher, FetchError, NetworkBlocked
from .physionet import mimic

ROOT = pathlib.Path(__file__).resolve().parent

SOURCES = [
    mimic("icu/icustays.csv.gz", "one row per ICU stay: admission and discharge times"),
    mimic("hosp/patients.csv.gz", "age and sex, for the age feature"),
    mimic("hosp/admissions.csv.gz", "admission context"),
    mimic("icu/d_items.csv.gz", "itemid dictionary -- what each charted number is"),
    mimic("icu/chartevents.csv.gz", "the vitals themselves, irregularly sampled"),
]


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
        print(f"\n{len(SOURCES)} files from PhysioNet's open-access MIMIC-IV demo")
        print("no credentialing required; the cache is gitignored, not redistributed")
        return 0
    if args.verify:
        problems = f.verify()
        for p in problems:
            print("  " + p)
        print("VERIFICATION FAILED" if problems else
              f"all {len(f.load_manifest()['files'])} cached file(s) verified")
        return 1 if problems else 0

    print(f"fetching {len(SOURCES)} files from the MIMIC-IV demo ...")
    try:
        f.get_all(SOURCES, refresh=args.refresh)
    except NetworkBlocked as e:
        print(f"\nBLOCKED: {e}", file=sys.stderr)
        return 2
    except FetchError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1
    print(f"\nwrote {f.manifest_path}")
    print("run `python -m src.demo --real` to build frontiers from real ICU stays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
