import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.attribution import check_efficiency, occlusion, shapley
from src.ledger import DecisionLedger, Policy

BASE = {"a": 0.0, "b": 0.0, "c": 0.0}


def linear(x):
    s = 2.0 * x["a"] - 3.0 * x["b"] + 1.0 * x["c"]
    return ("yes" if s >= 0 else "no"), s


def interacting(x):
    s = 2.0 * x["a"] - 3.0 * x["b"] + 1.0 * x["c"] - 4.0 * x["a"] * x["b"]
    return ("yes" if s >= 0 else "no"), s


P_LIN = Policy("p", "1.0", linear)
P_INT = Policy("p", "2.0", interacting)


# --- ledger integrity ----------------------------------------------------

def test_chain_verifies_when_untouched():
    led = DecisionLedger()
    for i in range(5):
        led.decide(f"c{i}", {"a": i * 0.1, "b": 0.2, "c": 0.3}, P_LIN)
    intact, idx = led.verify()
    assert intact and idx == -1


def test_editing_an_input_breaks_the_chain_at_that_record():
    led = DecisionLedger()
    for i in range(5):
        led.decide(f"c{i}", {"a": 0.1, "b": 0.2, "c": 0.3}, P_LIN)
    led.records[2].inputs["a"] = 99.0
    intact, idx = led.verify()
    assert not intact and idx == 2


def test_editing_the_action_breaks_the_chain():
    led = DecisionLedger()
    led.decide("c", {"a": 1.0, "b": 0.0, "c": 0.0}, P_LIN)
    led.records[0].action = "no"
    assert not led.verify()[0]


def test_first_record_links_to_genesis():
    led = DecisionLedger()
    r = led.decide("c", {"a": 0.0, "b": 0.0, "c": 0.0}, P_LIN)
    assert set(r.prev_hash) == {"0"}


# --- replay --------------------------------------------------------------

def test_every_decision_replays_exactly():
    led = DecisionLedger()
    rng = np.random.default_rng(0)
    for i in range(40):
        led.decide(f"c{i}", {k: float(rng.normal()) for k in "abc"}, P_LIN)
    out = led.replay_all()
    assert out["reproduced"] == out["n"] == 40


def test_replay_fails_loudly_for_an_unregistered_policy():
    led = DecisionLedger()
    led.decide("c", {"a": 1.0, "b": 0.0, "c": 0.0}, P_LIN)
    led.policies.clear()
    out = led.replay(0)
    assert not out["reproduced"] and "not registered" in out["reason"]


# --- counterfactuals -----------------------------------------------------

def test_counterfactual_changes_the_action_when_it_should():
    led = DecisionLedger()
    r = led.decide("c", {"a": 0.0, "b": 1.0, "c": 0.0}, P_LIN)
    assert r.action == "no"
    cf = led.counterfactual(r.seq, {"a": 5.0})
    assert cf["changed"] and cf["counterfactual_action"] == "yes"


def test_counterfactual_leaves_the_record_untouched():
    led = DecisionLedger()
    r = led.decide("c", {"a": 0.0, "b": 1.0, "c": 0.0}, P_LIN)
    led.counterfactual(r.seq, {"a": 99.0})
    assert led.records[0].inputs["a"] == 0.0
    assert led.verify()[0]


def test_counterfactual_uses_the_recorded_policy_version_not_the_latest():
    """Re-scoring with today's policy answers a different question."""
    led = DecisionLedger()
    x = {"a": 0.5, "b": 0.4, "c": 0.0}
    r1 = led.decide("c", x, P_LIN)
    led.decide("c", x, P_INT)          # a newer version is now registered
    cf = led.counterfactual(r1.seq, {})
    assert abs(cf["counterfactual_score"] - r1.score) < 1e-9


# --- attribution ---------------------------------------------------------

def test_shapley_is_exact_for_small_feature_counts():
    out = shapley(P_LIN, {"a": 1.0, "b": 1.0, "c": 1.0}, BASE)
    assert out["mode"] == "exact" and out["n_orderings"] == 6


def test_shapley_satisfies_efficiency():
    x = {"a": 1.3, "b": -0.7, "c": 2.1}
    out = shapley(P_INT, x, BASE)
    chk = check_efficiency(P_INT, x, BASE, out["values"])
    assert chk["efficient"], chk


def test_shapley_recovers_linear_coefficients():
    x = {"a": 1.0, "b": 1.0, "c": 1.0}
    v = shapley(P_LIN, x, BASE)["values"]
    assert abs(v["a"] - 2.0) < 1e-9
    assert abs(v["b"] + 3.0) < 1e-9
    assert abs(v["c"] - 1.0) < 1e-9


def test_occlusion_equals_shapley_on_an_additive_policy():
    """True, and the reason an additive demo shows nothing."""
    x = {"a": 1.4, "b": 0.6, "c": -0.3}
    occ = occlusion(P_LIN, x, BASE)
    sh = shapley(P_LIN, x, BASE)["values"]
    for k in sh:
        assert abs(occ[k] - sh[k]) < 1e-9


def test_occlusion_double_counts_an_interaction():
    """Each interacting feature takes full credit for the shared term."""
    x = {"a": 1.0, "b": 1.0, "c": 0.0}
    occ = occlusion(P_INT, x, BASE)
    sh = shapley(P_INT, x, BASE)["values"]
    assert abs(occ["a"] - sh["a"]) > 1e-6
    assert abs(occ["b"] - sh["b"]) > 1e-6
    assert abs(occ["c"] - sh["c"]) < 1e-9      # not in the interaction


def test_shapley_splits_the_interaction_evenly():
    """A symmetric interaction must be shared equally. That is the axiom."""
    x = {"a": 1.0, "b": 1.0, "c": 0.0}
    occ = occlusion(P_INT, x, BASE)
    sh = shapley(P_INT, x, BASE)["values"]
    assert abs((occ["a"] - sh["a"]) - (occ["b"] - sh["b"])) < 1e-9


def test_dummy_feature_gets_zero():
    def ignores_c(x):
        s = 2.0 * x["a"] - 3.0 * x["b"]
        return ("yes" if s >= 0 else "no"), s
    pol = Policy("d", "1.0", ignores_c)
    v = shapley(pol, {"a": 1.0, "b": 1.0, "c": 5.0}, BASE)["values"]
    assert abs(v["c"]) < 1e-9


def test_sampled_shapley_approaches_the_exact_value():
    """Tolerance is relative to the spread of the exact values.

    An absolute tolerance is the wrong test here: these values span 5.5, so a
    fixed 0.05 bound demands 1% accuracy from a permutation sample and fails on
    ordinary sampling noise rather than on any defect.
    """
    x = {"a": 1.0, "b": 1.0, "c": 0.5}
    exact = shapley(P_INT, x, BASE)["values"]
    spread = max(exact.values()) - min(exact.values())
    sampled = shapley(P_INT, x, BASE, n_samples=3000, seed=1)
    assert sampled["mode"] == "sampled"
    for k in exact:
        assert abs(sampled["values"][k] - exact[k]) < 0.05 * spread


def test_sampled_shapley_is_unbiased_across_seeds():
    """Averaging independent samples must converge on the exact values."""
    x = {"a": 1.0, "b": 1.0, "c": 0.5}
    exact = shapley(P_INT, x, BASE)["values"]
    runs = [shapley(P_INT, x, BASE, n_samples=800, seed=s)["values"]
            for s in range(12)]
    for k in exact:
        mean = sum(r[k] for r in runs) / len(runs)
        assert abs(mean - exact[k]) < 0.02 * (
            max(exact.values()) - min(exact.values()))


def test_ledger_stats_report_action_counts():
    led = DecisionLedger()
    led.decide("a", {"a": 5.0, "b": 0.0, "c": 0.0}, P_LIN)
    led.decide("b", {"a": 0.0, "b": 5.0, "c": 0.0}, P_LIN)
    s = led.stats()
    assert s["action_counts"] == {"yes": 1, "no": 1}
    assert s["n_decisions"] == 2
