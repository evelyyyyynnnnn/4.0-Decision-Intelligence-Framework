"""Tests for auditing decisions on real credit applications.

Most of these pin mapping decisions. A mapping error here does not crash: it
produces a complete, hash-chained, replayable audit trail of decisions made on
the wrong numbers, which is worse than a crash.
"""
import io
import pathlib
import sys
import zipfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data import datakit
from data.load import DELINQUENCY, EMPLOYMENT_YEARS, FEATURES, load_cases, parse_german

# Real german.data lines: 20 space-separated attributes, then the class.
def _row(hist="A32", employ="A73", installment=2, amount=1500, age=35, klass=1):
    a = ["A11", "6", hist, "A43", str(amount), "A65", employ, str(installment),
         "A93", "A101", "4", "A121", str(age), "A143", "A152", "2", "A173",
         "1", "A192", "A201", str(klass)]
    return " ".join(a)


GOOD = _row(klass=1)
BAD = _row(hist="A34", employ="A71", installment=4, amount=9000, age=23, klass=2)


def test_parse_extracts_the_five_mapped_features():
    rows = parse_german(GOOD + "\n" + BAD)
    assert len(rows) == 2
    _, feats, outcome = rows[0]
    assert set(feats) == set(FEATURES)
    assert outcome == "good"
    assert rows[1][2] == "bad"


def test_class_2_means_bad_risk_not_good():
    """Inverting the class label would make every conclusion backwards."""
    assert parse_german(_row(klass=2))[0][2] == "bad"
    assert parse_german(_row(klass=1))[0][2] == "good"


def test_credit_amount_is_scaled_to_thousands():
    _, feats, _ = parse_german(_row(amount=9000))[0]
    assert feats["credit_amount_k"] == pytest.approx(9.0)


def test_credit_history_maps_to_an_ordinal_delinquency_count():
    """A34 is 'critical account', not the number 3."""
    assert DELINQUENCY["A30"] == 0.0
    assert DELINQUENCY["A32"] == 0.0
    assert DELINQUENCY["A33"] == 1.0
    assert DELINQUENCY["A34"] == 2.0
    _, feats, _ = parse_german(_row(hist="A34"))[0]
    assert feats["delinquencies"] == 2.0


def test_employment_bands_become_midpoints_in_order():
    vals = [EMPLOYMENT_YEARS[k] for k in ("A71", "A72", "A73", "A74", "A75")]
    assert vals == sorted(vals), "the employment bands must stay ordered"
    _, feats, _ = parse_german(_row(employ="A75"))[0]
    assert feats["employment_years"] == 10.0


def test_unknown_codes_do_not_crash_the_parse():
    _, feats, _ = parse_german(_row(hist="A99", employ="A99"))[0]
    assert feats["delinquencies"] == 0.0
    assert feats["employment_years"] == 0.0


def test_short_and_malformed_lines_are_skipped():
    assert parse_german("too few fields\n" + GOOD)[0][2] == "good"
    with pytest.raises(ValueError, match="no usable rows"):
        parse_german("nothing parseable here")


# --- end to end ------------------------------------------------------------

def test_refuses_when_nothing_is_cached(tmp_path):
    with pytest.raises(datakit.FetchError, match="no real credit applications"):
        load_cases(root=tmp_path)


def _seed(tmp_path, n_good=60, n_bad=40, inner="german.data"):
    lines = []
    for i in range(n_good):
        lines.append(_row(hist="A32", employ="A74", installment=1,
                          amount=1200 + i, age=45, klass=1))
    for i in range(n_bad):
        lines.append(_row(hist="A34", employ="A71", installment=4,
                          amount=9000 + i, age=24, klass=2))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(inner, "\n".join(lines))
        z.writestr("german.doc", "documentation")
    raw = buf.getvalue()

    f = datakit.Fetcher(tmp_path)
    man = f.load_manifest()
    dest = "uci/german-credit.zip"
    (f.raw / "uci").mkdir(parents=True, exist_ok=True)
    (f.raw / dest).write_bytes(raw)
    man["files"][dest] = {
        "source": "Statlog German Credit", "url": "https://archive.ics.uci.edu/x",
        "publisher": "UCI", "terms": "research use",
        "sha256": datakit.sha256_file(f.raw / dest), "bytes": len(raw),
        "retrieved_utc": datakit.utc_now()}
    f._write_manifest(man)
    return f


def test_a_zip_without_german_data_is_an_explicit_error(tmp_path):
    _seed(tmp_path, inner="something-else.txt")
    with pytest.raises(datakit.FetchError, match="german.data not found"):
        load_cases(root=tmp_path)


def test_load_cases_reports_the_base_rate_and_the_dropped_features(tmp_path):
    _seed(tmp_path)
    cases, outcomes, prov = load_cases(root=tmp_path)
    assert len(cases) == 100
    assert prov["n_bad_risk"] == 40
    assert prov["bad_rate"] == pytest.approx(0.40)
    # income_k and dti are dropped, not renamed onto another quantity.
    assert set(prov["features_dropped"]) == {"income_k", "dti"}
    assert "no income figure" in prov["dropped_because"]
    assert all(k not in cases[0][1] for k in ("dti", "income_k"))


def test_the_audit_trail_holds_on_real_applications(tmp_path, monkeypatch):
    """Ledger, replay and tamper detection, exercised on real-shaped input."""
    _seed(tmp_path)
    from src.demo import REAL_POLICY_V1, tamper_demo
    from src.ledger import DecisionLedger

    cases, outcomes, _ = load_cases(root=tmp_path)
    led = DecisionLedger()
    for cid, x in cases:
        led.decide(cid, x, REAL_POLICY_V1, context={"channel": "branch"})

    assert led.stats()["n_decisions"] == 100
    assert led.stats()["chain_intact"] is True
    rp = led.replay_all()
    assert rp["reproduced"] == rp["n"] == 100
    assert tamper_demo(led)["detected"] is True


def test_the_policy_declines_some_and_approves_some(tmp_path):
    """A policy that declines everyone or nobody makes attribution vacuous."""
    _seed(tmp_path)
    from src.demo import REAL_POLICY_V1
    from src.ledger import DecisionLedger
    cases, _, _ = load_cases(root=tmp_path)
    led = DecisionLedger()
    for cid, x in cases:
        led.decide(cid, x, REAL_POLICY_V1, context={})
    actions = {r.action for r in led.records}
    assert actions == {"approve", "decline"}


def test_occlusion_and_shapley_disagree_because_of_the_interaction(tmp_path):
    """The interaction term is why the two attribution methods are not the same;
    without it occlusion equals Shapley and the comparison teaches nothing."""
    _seed(tmp_path)
    from src.attribution import occlusion, shapley
    from src.demo import REAL_BASELINE, REAL_POLICY_V1
    cases, _, _ = load_cases(root=tmp_path)
    x = dict(cases[-1][1])          # a bad-risk applicant, far from baseline

    occ = occlusion(REAL_POLICY_V1, x, REAL_BASELINE)
    sh = shapley(REAL_POLICY_V1, x, REAL_BASELINE)
    diffs = [abs(occ[k] - sh["values"][k]) for k in sh["values"]]
    assert max(diffs) > 1e-6, "occlusion and Shapley should differ here"


def test_shapley_values_sum_to_the_score_difference(tmp_path):
    """Efficiency: the property that makes Shapley an attribution and not just
    a score."""
    _seed(tmp_path)
    from src.attribution import check_efficiency, shapley
    from src.demo import REAL_BASELINE, REAL_POLICY_V1
    cases, _, _ = load_cases(root=tmp_path)
    x = dict(cases[-1][1])
    sh = shapley(REAL_POLICY_V1, x, REAL_BASELINE)
    eff = check_efficiency(REAL_POLICY_V1, x, REAL_BASELINE, sh["values"])
    assert eff["efficient"] is True, eff
