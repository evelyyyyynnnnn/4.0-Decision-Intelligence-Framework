"""A credit-approval policy, logged, replayed, attributed and stress-tested."""
from __future__ import annotations
import json, pathlib, sys
from datetime import datetime, timezone
import numpy as np
from .ledger import DecisionLedger, Policy
from .attribution import check_efficiency, occlusion, shapley

ROOT = pathlib.Path(__file__).resolve().parent.parent

FEATURES = ("dti", "utilisation", "delinquencies", "tenure_years", "income_k")
BASELINE = {"dti": 0.30, "utilisation": 0.35, "delinquencies": 0.0,
            "tenure_years": 5.0, "income_k": 70.0}


# The interaction term is deliberate. With a purely additive score, occlusion
# and Shapley return IDENTICAL values -- correct, and completely uninformative
# about why anyone would pay for Shapley. Leverage genuinely interacts with
# revolving utilisation in credit risk, and once the score contains that term
# the two attribution methods separate, which is the case worth showing.
INTERACTION = -2.2
INTERCEPT = 1.05          # calibrated so approvals and declines are comparable


def _base_terms(x: dict) -> float:
    return (-2.6 * x["dti"] - 1.4 * x["utilisation"]
            + 0.06 * x["tenure_years"] + 0.011 * x["income_k"]
            + INTERACTION * x["dti"] * x["utilisation"] + INTERCEPT)


def _score_v1(x: dict) -> float:
    return _base_terms(x) - 0.55 * x["delinquencies"]


def _score_v2(x: dict) -> float:
    """v2 weights delinquencies harder. A real policy change, versioned."""
    return _base_terms(x) - 1.30 * x["delinquencies"]


def _decide(scorer):
    def fn(x: dict):
        s = scorer(x)
        return ("approve" if s >= 0.0 else "decline"), s
    return fn


POLICY_V1 = Policy("credit-approval", "1.0", _decide(_score_v1))
POLICY_V2 = Policy("credit-approval", "2.0", _decide(_score_v2))


def make_cases(n: int = 120, seed: int = 4) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append((f"APP-{i:04d}", {
            "dti": float(np.clip(rng.normal(0.34, 0.11), 0.02, 0.85)),
            "utilisation": float(np.clip(rng.normal(0.42, 0.20), 0.0, 1.0)),
            "delinquencies": float(rng.poisson(0.35)),
            "tenure_years": float(np.clip(rng.normal(6.0, 3.5), 0, 30)),
            "income_k": float(np.clip(rng.normal(78, 26), 18, 260)),
        }))
    return out


def tamper_demo(led: DecisionLedger) -> dict:
    """Editing a recorded input must break the chain at that record.

    The edited field is whichever input the record happens to carry, rather
    than a hard-coded name: this same check runs over the synthetic policy and
    over the real one, and those policies do not share a feature set.
    """
    ok_before, _ = led.verify()
    rec = led.records[3]
    field = next(iter(rec.inputs))
    original = rec.inputs[field]
    rec.inputs[field] = original + 1.0 if original == original else 0.01
    ok_after, idx = led.verify()
    rec.inputs[field] = original
    ok_restored, _ = led.verify()
    return {"intact_before": ok_before, "detected": not ok_after,
            "detected_at": idx, "field_edited": field,
            "intact_after_restore": ok_restored}


def version_demo(cases) -> dict:
    """The same case under two policy versions, replayed against each.

    A counterfactual must run the policy that actually ran. Re-scoring an old
    decision with today's policy answers "what would we do now", which is a
    different and usually less useful question than "what would we have done
    then" -- and conflating the two is how a review concludes the system behaved
    reasonably when it did not.
    """
    led = DecisionLedger()
    flipped = []
    for cid, x in cases[:40]:
        r1 = led.decide(cid, x, POLICY_V1)
        r2 = led.decide(cid, x, POLICY_V2)
        if r1.action != r2.action:
            flipped.append({"case": cid, "v1": r1.action, "v2": r2.action,
                            "delinquencies": x["delinquencies"],
                            "v1_score": round(r1.score, 4),
                            "v2_score": round(r2.score, 4)})
    replay = led.replay_all()
    return {"n_cases": 40, "n_flipped": len(flipped), "examples": flipped[:5],
            "replay": replay}


def run() -> dict:
    cases = make_cases()
    led = DecisionLedger()
    for cid, x in cases:
        led.decide(cid, x, POLICY_V1, context={"channel": "online"})

    # Pick the declined case where the interaction term is largest. Attribution
    # on a case sitting near the baseline shows a near-zero disagreement between
    # the two methods, which is true but tells the reader nothing.
    declined = max(
        (r for r in led.records if r.action == "decline"),
        key=lambda r: abs(r.inputs["dti"] - BASELINE["dti"])
        * abs(r.inputs["utilisation"] - BASELINE["utilisation"]))

    occ = occlusion(POLICY_V1, declined.inputs, BASELINE)
    sh = shapley(POLICY_V1, declined.inputs, BASELINE)
    disagreement = {k: round(occ[k] - sh["values"][k], 6) for k in sh["values"]}
    eff = check_efficiency(POLICY_V1, declined.inputs, BASELINE, sh["values"])

    # Counterfactuals on the declined case: what would have flipped it?
    cfs = []
    for k, v in [("dti", 0.20), ("utilisation", 0.15), ("delinquencies", 0.0),
                 ("income_k", 150.0), ("tenure_years", 20.0)]:
        cfs.append({"feature": k, "to": v,
                    **led.counterfactual(declined.seq, {k: v})})

    # Minimal single-feature change that flips the decision.
    flips = [c for c in cfs if c["changed"]]

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_synthetic": True,
        "data_source": "synthetic credit applications (src/demo.py)",
        "ledger": led.stats(),
        "replay": led.replay_all(),
        "tamper": tamper_demo(led),
        "versioning": version_demo(cases),
        "case": {"case_id": declined.case_id, "seq": declined.seq,
                 "action": declined.action, "score": round(declined.score, 5),
                 "inputs": {k: round(v, 4) for k, v in declined.inputs.items()}},
        "occlusion": occ,
        "shapley": sh,
        "attribution_disagreement": disagreement,
        "max_disagreement": round(max(abs(v) for v in disagreement.values()), 6),
        "efficiency": eff,
        "counterfactuals": cfs,
        "n_flipping_features": len(flips),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "latest.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf8")
    return results


# --- the real-data policy --------------------------------------------------
#
# A separate policy, not the synthetic one with different numbers. The German
# Credit dataset records no income, so `dti` and `income_k` have no counterpart
# and the policy is written over the features that do exist. Coefficients are
# authored -- this is a policy, not a fitted model -- and the interaction term
# is kept because it is what makes occlusion and Shapley disagree.

REAL_BASELINE = {"installment_rate": 2.0, "delinquencies": 0.0,
                 "employment_years": 5.5, "credit_amount_k": 2.0, "age": 40.0}
REAL_INTERACTION = -0.22


def _real_terms(x: dict) -> float:
    return (-0.42 * x["installment_rate"]
            - 0.30 * x["credit_amount_k"]
            + 0.06 * x["employment_years"]
            + 0.012 * x["age"]
            + REAL_INTERACTION * x["installment_rate"] * x["credit_amount_k"]
            + 1.55)


def _real_v1(x: dict) -> float:
    return _real_terms(x) - 0.45 * x["delinquencies"]


def _real_v2(x: dict) -> float:
    """v2 weights past delinquency harder. A real policy change, versioned."""
    return _real_terms(x) - 1.20 * x["delinquencies"]


REAL_POLICY_V1 = Policy("credit-approval-de", "1.0", _decide(_real_v1))
REAL_POLICY_V2 = Policy("credit-approval-de", "2.0", _decide(_real_v2))


def run_real() -> dict:
    """Audit decisions made on real credit applications, with real outcomes.

    The outcome column is what this run adds. Everything the framework does --
    hash-chained ledger, exact replay, versioned counterfactuals, Shapley
    attribution -- can be demonstrated on invented applicants. What cannot is
    the question a reader will actually ask: do the declines correspond to
    applicants who turned out to be bad credit risks? With 1,000 labelled
    applications, that can be answered.

    It is answered about the POLICY, which is authored, not fitted. A policy
    written by hand that separates good from bad risks at all is doing
    something; one that does not is worth knowing about too, and is reported
    either way.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from data.load import FEATURES, load_cases

    cases, outcomes, prov = load_cases(root=ROOT / "data")

    led = DecisionLedger()
    for cid, x in cases:
        led.decide(cid, x, REAL_POLICY_V1, context={"channel": "branch"})

    # Does the policy's decision correspond to the recorded outcome?
    tp = fp = tn = fn = 0
    for r in led.records:
        bad = outcomes[r.case_id] == "bad"
        if r.action == "decline":
            tp, fp = tp + int(bad), fp + int(not bad)
        else:
            fn, tn = fn + int(bad), tn + int(not bad)
    declined = tp + fp
    approved = tn + fn
    outcome_check = {
        "declined": declined, "approved": approved,
        "bad_rate_overall": prov["bad_rate"],
        "bad_rate_among_declined": round(tp / declined, 4) if declined else None,
        "bad_rate_among_approved": round(fn / approved, 4) if approved else None,
        "lift": round((tp / declined) / prov["bad_rate"], 3)
                if declined and prov["bad_rate"] else None,
        "note": "the policy is authored, not fitted to this data, so a lift "
                "near 1.0 would mean it separates nothing -- which is a "
                "finding about the policy, not a failure of the audit trail",
    }

    declined_records = [r for r in led.records if r.action == "decline"]
    if not declined_records:
        raise RuntimeError("the policy declined nobody; nothing to attribute")
    case = max(declined_records,
               key=lambda r: abs(r.inputs["installment_rate"]
                                 - REAL_BASELINE["installment_rate"])
               * abs(r.inputs["credit_amount_k"]
                     - REAL_BASELINE["credit_amount_k"]))

    occ = occlusion(REAL_POLICY_V1, case.inputs, REAL_BASELINE)
    sh = shapley(REAL_POLICY_V1, case.inputs, REAL_BASELINE)
    disagreement = {k: round(occ[k] - sh["values"][k], 6) for k in sh["values"]}
    eff = check_efficiency(REAL_POLICY_V1, case.inputs, REAL_BASELINE, sh["values"])

    cfs = []
    for k, v in [("installment_rate", 1.0), ("delinquencies", 0.0),
                 ("credit_amount_k", 1.0), ("employment_years", 10.0),
                 ("age", 55.0)]:
        cfs.append({"feature": k, "to": v,
                    **led.counterfactual(case.seq, {k: v})})

    # The same versioning comparison, on real applications.
    led2 = DecisionLedger()
    for cid, x in cases:
        led2.decide(cid, x, REAL_POLICY_V2, context={"channel": "branch"})
    changed = sum(1 for a, b in zip(led.records, led2.records)
                  if a.action != b.action)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_synthetic": False,
        "data_source": "Statlog German Credit Data (UCI); see data/MANIFEST.json "
                       "for the URL, hash and retrieval time",
        "policy_is_authored_not_fitted": True,
        "features_used": list(FEATURES),
        "features_dropped": prov["features_dropped"],
        "dropped_because": prov["dropped_because"],
        "provenance": prov,
        "ledger": led.stats(),
        "replay": led.replay_all(),
        "tamper": tamper_demo(led),
        "versioning": {"policy": "credit-approval-de", "from": "1.0", "to": "2.0",
                       "decisions_changed": changed,
                       "share_changed": round(changed / len(cases), 4)},
        "outcome_check": outcome_check,
        "case": {"case_id": case.case_id, "seq": case.seq,
                 "action": case.action, "score": round(case.score, 5),
                 "inputs": {k: round(v, 4) for k, v in case.inputs.items()}},
        "occlusion": occ,
        "shapley": sh,
        "attribution_disagreement": disagreement,
        "max_disagreement": round(max(abs(v) for v in disagreement.values()), 6),
        "efficiency": eff,
        "counterfactuals": cfs,
        "n_flipping_features": sum(1 for c in cfs if c["changed"]),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "latest-real.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf8")
    return results


def main_real() -> int:
    from data.datakit import FetchError
    try:
        r = run_real()
    except FetchError as exc:
        print(f"cannot run on real data: {exc}", file=sys.stderr)
        return 2
    pv, oc = r["provenance"], r["outcome_check"]
    print(f"source: {r['data_source']}")
    print(f"{pv['n_cases']} applications, {pv['n_bad_risk']} bad risks "
          f"({pv['bad_rate']:.1%})  [{pv['sha256']}]")
    print(f"features used: {', '.join(r['features_used'])}")
    print(f"features dropped: {', '.join(r['features_dropped'])} -- "
          f"{r['dropped_because'][:90]}...")

    led = r["ledger"]
    rp = r["replay"]
    exact = rp["reproduced"] == rp["n"]
    print(f"\nledger: {led['n_decisions']} decisions, "
          f"{rp['reproduced']}/{rp['n']} replayed exactly"
          f"{'' if exact else ' -- MISMATCH'}")
    print(f"tamper detected: {r['tamper']['detected']}")
    v = r["versioning"]
    print(f"policy {v['from']} -> {v['to']}: {v['decisions_changed']} of "
          f"{led['n_decisions']} decisions change ({v['share_changed']:.1%})")

    print(f"\ndeclined {oc['declined']}, approved {oc['approved']}")
    if oc["bad_rate_among_declined"] is not None:
        print(f"  bad-risk rate among declined: {oc['bad_rate_among_declined']:.1%}")
        print(f"  bad-risk rate among approved: {oc['bad_rate_among_approved']:.1%}")
        print(f"  lift over the base rate:       {oc['lift']}x")
    print(f"  {oc['note']}")

    c = r["case"]
    print(f"\nattribution on declined case {c['case_id']} (score {c['score']:+.4f}):")
    for k in sorted(r["shapley"]["values"]):
        print(f"  {k:<20} occlusion {r['occlusion'][k]:+.4f}   "
              f"shapley {r['shapley']['values'][k]:+.4f}   "
              f"diff {r['attribution_disagreement'][k]:+.4f}")
    eff = r["efficiency"]
    print(f"  efficiency: values sum to {eff['sum_of_values']:+.6f} against a "
          f"score difference of {eff['score_difference']:+.6f} "
          f"(residual {eff['residual']:+.2e}, "
          f"{'efficient' if eff['efficient'] else 'NOT EFFICIENT'})")
    print(f"\ncounterfactuals that flip the decision: {r['n_flipping_features']} "
          f"of {len(r['counterfactuals'])}")
    for cf in r["counterfactuals"]:
        if cf["changed"]:
            print(f"  {cf['feature']} -> {cf['to']}")
    print("\nwrote results/latest-real.json")
    return 0


def main() -> int:
    if "--real" in sys.argv[1:]:
        return main_real()
    r = run()
    L, rp, t = r["ledger"], r["replay"], r["tamper"]
    print(f"ledger: {L['n_decisions']} decisions, chain intact {L['chain_intact']}")
    print(f"  actions: {L['action_counts']}")
    print(f"replay: {rp['reproduced']}/{rp['n']} reproduced exactly")
    print(f"tamper: detected {t['detected']} at record {t['detected_at']}, "
          f"restored {t['intact_after_restore']}")

    v = r["versioning"]
    print(f"\npolicy v1 vs v2 on {v['n_cases']} cases: "
          f"{v['n_flipped']} decisions flip")
    for e in v["examples"][:3]:
        print(f"  {e['case']}: {e['v1']} -> {e['v2']} "
              f"(delinquencies={e['delinquencies']:.0f}, "
              f"{e['v1_score']:+.3f} -> {e['v2_score']:+.3f})")

    c = r["case"]
    print(f"\nattribution for {c['case_id']} ({c['action']}, score {c['score']:+.4f}):")
    print(f"  {'feature':<16}{'occlusion':>12}{'shapley':>12}{'diff':>12}")
    for k in r["shapley"]["values"]:
        print(f"  {k:<16}{r['occlusion'][k]:>12.4f}{r['shapley']['values'][k]:>12.4f}"
              f"{r['attribution_disagreement'][k]:>12.4f}")
    print(f"  max |occlusion - shapley| = {r['max_disagreement']:.4f}")
    e = r["efficiency"]
    print(f"  shapley mode: {r['shapley']['mode']} "
          f"({r['shapley']['n_orderings']} orderings)")
    print(f"  efficiency check: sum {e['sum_of_values']:.4f} vs "
          f"score difference {e['score_difference']:.4f} -> {e['efficient']}")

    print(f"\ncounterfactuals ({r['n_flipping_features']} single changes flip it):")
    for cf in r["counterfactuals"]:
        mark = "FLIPS" if cf["changed"] else "     "
        print(f"  {mark} {cf['feature']:<16}-> {cf['to']:<7} "
              f"score {cf['original_score']:+.4f} -> {cf['counterfactual_score']:+.4f}")
    try:
        from .site import build_site
        build_site(r); print("\nwebsite/ rebuilt from this run")
    except Exception as exc:
        print(f"\n(site not rebuilt: {exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
