"""Parse German Credit applications into the policy's input shape.

Every mapping below is a judgement, so every one is written down. The rule
followed throughout: where the dataset has a genuine counterpart to a policy
feature, use it; where it does not, drop the feature rather than manufacture
a proxy and give it the real feature's name.

  installment_rate   attribute 8, "installment rate in percentage of
                     disposable income". This is a real debt-service ratio,
                     which is why this dataset was chosen over the credit-card
                     default data, where nothing corresponds to income at all.
  delinquencies      derived from attribute 3, the credit-history category.
                     A30/A31 (nothing outstanding, all paid) -> 0;
                     A32 (paid to date) -> 0; A33 (delay in the past) -> 1;
                     A34 (critical account) -> 2. An ordinal reading of an
                     ordered category, and the coarsest thing this dataset
                     supports.
  employment_years   attribute 7, banded in the source (<1y, 1-4, 4-7, 7+).
                     Band midpoints are used, so the value is a real quantity
                     recorded at a resolution the source chose.
  credit_amount_k    attribute 5, in thousands of Deutsche Mark.
  age                attribute 13.

DROPPED: the synthetic policy's `income_k` and `dti`. This dataset records no
income figure, and a debt-to-income ratio cannot be computed from an
installment rate alone. Renaming the installment rate to `dti` would put a
different quantity under a name readers already understand.
"""
from __future__ import annotations

import io
import pathlib
import zipfile

from .datakit import Fetcher, FetchError

ROOT = pathlib.Path(__file__).resolve().parent

DELINQUENCY = {"A30": 0.0, "A31": 0.0, "A32": 0.0, "A33": 1.0, "A34": 2.0}
EMPLOYMENT_YEARS = {"A71": 0.0, "A72": 0.5, "A73": 2.5, "A74": 5.5, "A75": 10.0}

FEATURES = ("installment_rate", "delinquencies", "employment_years",
            "credit_amount_k", "age")


def parse_german(text: str) -> list:
    """Return [(case_id, features, outcome)] from german.data.

    The file is space-separated with 21 fields: 20 attributes then the class,
    where 1 means a good credit risk and 2 means a bad one.
    """
    rows = []
    for i, line in enumerate(text.splitlines()):
        parts = line.split()
        if len(parts) < 21:
            continue
        try:
            feats = {
                "installment_rate": float(parts[7]),
                "delinquencies": DELINQUENCY.get(parts[2], 0.0),
                "employment_years": EMPLOYMENT_YEARS.get(parts[6], 0.0),
                "credit_amount_k": float(parts[4]) / 1000.0,
                "age": float(parts[12]),
            }
            klass = int(parts[20])
        except (ValueError, IndexError):
            continue
        if klass not in (1, 2):
            continue
        rows.append((f"GC-{i:04d}", feats, "bad" if klass == 2 else "good"))
    if not rows:
        raise ValueError("no usable rows parsed from german.data")
    return rows


def load_cases(root=ROOT):
    """Return (cases, outcomes, provenance). Refuses when nothing is cached."""
    f = Fetcher(root)
    man = f.load_manifest()
    dest = "uci/german-credit.zip"
    if dest not in man["files"] or not (f.raw / dest).exists():
        raise FetchError(
            "no real credit applications cached. Run `python -m data.fetch` in "
            "a networked environment first; this project will not audit "
            "invented applicants and describe them as filed applications.")

    raw = (f.raw / dest).read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next((n for n in z.namelist()
                     if n.endswith("german.data")), None)
        if name is None:
            raise FetchError(
                f"german.data not found in the archive; it contains "
                f"{z.namelist()[:6]}")
        text = z.read(name).decode("utf-8", errors="replace")

    rows = parse_german(text)
    cases = [(cid, feats) for cid, feats, _ in rows]
    outcomes = {cid: out for cid, _, out in rows}

    rec = man["files"][dest]
    n_bad = sum(1 for v in outcomes.values() if v == "bad")
    prov = {
        "dataset": "Statlog German Credit Data (UCI ML Repository)",
        "n_cases": len(cases),
        "n_bad_risk": n_bad,
        "bad_rate": round(n_bad / len(cases), 4),
        "features_used": list(FEATURES),
        "features_dropped": ["income_k", "dti"],
        "dropped_because":
            "this dataset records no income figure, and a debt-to-income ratio "
            "cannot be derived from an installment rate alone. Renaming the "
            "installment rate to dti would put a different quantity under a "
            "name readers already understand.",
        "sha256": rec["sha256"][:16], "url": rec["url"],
        "retrieved_utc": rec.get("retrieved_utc"),
    }
    return cases, outcomes, prov
