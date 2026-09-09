"""Run the suite across tasks, policies and distribution shifts."""
from __future__ import annotations
import json, pathlib, sys
from datetime import datetime, timezone
import numpy as np
from . import real_tasks as rt
from .metrics import calibration, regret, robustness
from .policies import (AlwaysAction, BayesTriage, EmpiricalNewsvendor,
                       MeanDemandNewsvendor, ThresholdTriage)
from .tasks import NewsvendorTask, TriageTask

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHIFTS = (-16.0, -8.0, 0.0, 8.0, 16.0)
TRIAGE_SHIFTS = (-1.2, -0.6, 0.0, 0.6, 1.2)


def run_real_triage() -> dict:
    """Score the real MIMIC-IV triage track. No oracle -> realised cost,
    calibration, and degradation under a named age shift.

    Returns an honest status dict if the real cache is absent, rather than
    reporting anything from data that was not loaded.
    """
    if not rt.available():
        return {"status": "data not fetched in this environment",
                "detail": "the PhysioNet MIMIC-IV demo cache was not found; set "
                          "MIMIC_DATA_ROOT or fetch it in 2-icu-early-warning. No "
                          "real-triage numbers are reported.",
                "is_synthetic": False}

    sample, cohort = rt.build_sample()
    tr_mask, te_mask = rt.subject_split(sample, frac=0.5, seed=0)
    train, test = rt._sub(sample, tr_mask), rt._sub(sample, te_mask)

    proba_fn, log_decide = rt.fit_calibrated_logistic(train)
    thr_decide, tau = rt.fit_threshold_on_map(train)

    p_test = proba_fn(test.X)
    floor = rt.best_hindsight_threshold_cost(p_test, test.y)
    policies = {
        "calibrated logistic (Bayes cost rule)": log_decide(test.X),
        "single threshold on MAP": thr_decide(test.X),
        "always treat": np.ones(len(test), int),
        "never treat": np.zeros(len(test), int),
    }
    pol_rows = {}
    for name, a in policies.items():
        c = rt.realised_cost(a, test.y)
        pol_rows[name] = {"realised_cost": round(c, 5),
                          "excess_over_hindsight_floor": round(c - floor["cost"], 5),
                          "treat_rate": round(float(np.mean(a)), 4)}

    cal = calibration(p_test, test.y.astype(float))

    # Named covariate shift: median-age split (whole subjects, so no leakage).
    med = float(np.median(sample.age))
    shift = {"shift": "patient age, median split; train on one age group, "
                      "evaluate realised cost on the other",
             "median_age": round(med, 1)}
    for label, keep in (("train_younger", sample.age < med),
                        ("train_older", sample.age >= med)):
        grp = rt._sub(sample, keep)
        other = rt._sub(sample, ~keep)
        g_tr, g_te = rt.subject_split(grp, frac=0.6, seed=1)
        gtr, gte = rt._sub(grp, g_tr), rt._sub(grp, g_te)
        if gtr.y.sum() < 5 or len(gte) == 0:
            continue
        _, dec = rt.fit_calibrated_logistic(gtr)
        c_in = rt.realised_cost(dec(gte.X), gte.y)
        c_shift = rt.realised_cost(dec(other.X), other.y)
        shift[label] = {"in_distribution_cost": round(c_in, 5),
                        "shifted_cost": round(c_shift, 5),
                        "degradation": round(c_shift - c_in, 5)}

    n_sub_tr = len(set(train.subject.tolist()))
    n_sub_te = len(set(test.subject.tolist()))
    return {
        "is_synthetic": False,
        "data_source": "PhysioNet MIMIC-IV demo (open access); file hashes, URLs "
                       "and retrieval times in 2-icu-early-warning/data/MANIFEST.json",
        "cohort_is_a_demonstration_not_a_study": True,
        "no_oracle": True,
        "scoring_note": "no oracle on real data; policies scored by realised cost "
                        "and against the best hindsight threshold on the test split",
        "label_caveat": "the label is a future threshold on MAP, which is also a "
                        "feature: this measures short-horizon persistence of an "
                        "observed vital, not an independent clinical outcome",
        "event": "hypotension (MAP < 65 mmHg) within a 2-hour horizon",
        "cost_model": {"treat_cost": rt.TREAT_COST, "miss_cost": rt.MISS_COST,
                       "decision": "treat when p * miss_cost >= treat_cost"},
        "cohort": cohort,
        "split": {"by": "subject", "subject_overlap": 0,
                  "n_subjects_train": n_sub_tr, "n_subjects_test": n_sub_te,
                  "n_decisions_train": len(train), "n_decisions_test": len(test)},
        "fitted_map_threshold_mmhg": round(tau, 2),
        "hindsight_floor": floor,
        "policies": pol_rows,
        "calibrated_logistic_calibration": cal,
        "robustness_named_shift": shift,
    }


def _oracle_gap(nv) -> dict:
    """Two oracles, and only one of them is a fair target.

    The regret column is measured against a CLAIRVOYANT oracle that picks the
    best order after seeing realised demand. No policy can approach it, and its
    level says nothing about decision quality -- it is dominated by the variance
    of demand itself. Quoting a policy's regret against it as though it were a
    performance gap would be misleading.

    The achievable benchmark is the distributional optimum: the critical-fractile
    order given the TRUE distribution. A policy that reaches that has nothing
    left to learn, and the remaining regret is irreducible.
    """
    ev = nv.sample(n=6000, seed=101, shift=0.0)
    demand = ev.context[:, 0]
    best_fixed_cost = min(float(nv.cost(a, demand).mean()) for a in nv.actions)
    clairvoyant = float(np.min(
        np.stack([nv.cost(a, demand) for a in nv.actions], axis=1), axis=1).mean())
    true_opt = nv.oracle_action()
    dist_opt_cost = float(nv.cost(
        nv.actions[int(np.argmin(np.abs(nv.actions - true_opt)))], demand).mean())
    return {
        "clairvoyant_cost": round(clairvoyant, 4),
        "best_fixed_action_cost": round(best_fixed_cost, 4),
        "distributional_optimum_cost": round(dist_opt_cost, 4),
        "irreducible_regret": round(best_fixed_cost - clairvoyant, 4),
        "note": "regret in the table is against the clairvoyant oracle; the "
                "achievable floor is the best fixed action",
    }


def run_task(task, policies, shifts, seed_fit=0, seed_eval=101) -> dict:
    fit_sample = task.sample(n=2000, seed=seed_fit, shift=0.0)
    fitted = {p.name: p.fit(fit_sample) for p in policies}

    rows, shift_regret = {}, {name: {} for name in fitted}
    for sh in shifts:
        ev = task.sample(n=4000, seed=seed_eval, shift=sh)
        for name, pol in fitted.items():
            a = pol.decide(ev)
            r = regret(a, ev.outcomes, ev.optimal)
            shift_regret[name][sh] = r["mean"]
            if sh == 0.0:
                correct = (a == ev.optimal).astype(float)
                cal = calibration(pol.confidence(ev), correct)
                rows[name] = {"regret": r, "calibration": cal,
                              "accuracy": round(float(correct.mean()), 4)}

    for name in rows:
        rows[name]["robustness"] = robustness(shift_regret[name])
    return {"task": task.name, "shifts": list(shifts), "policies": rows}


def run() -> dict:
    nv = NewsvendorTask()
    tr = TriageTask()
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_synthetic": True,
        "data_source": "closed-form benchmark tasks with exact oracles (src/tasks.py)",
        "newsvendor": run_task(nv, [
            EmpiricalNewsvendor(nv), MeanDemandNewsvendor(nv),
            AlwaysAction(int(np.argmin(np.abs(nv.actions - 50))), "always order 50"),
        ], SHIFTS),
        "triage": run_task(tr, [
            BayesTriage(tr), ThresholdTriage(tr),
            AlwaysAction(1, "always treat"), AlwaysAction(0, "never treat"),
        ], TRIAGE_SHIFTS),
        "oracle": {
            "newsvendor_critical_fractile": round(nv.critical_fractile(), 4),
            "newsvendor_optimal_order": round(nv.oracle_action(), 3),
            "triage_treat_threshold_prob": round(tr.treat_cost / tr.miss_cost, 4),
        },
        "oracle_gap": _oracle_gap(nv),
        "has_real_track": True,
        "tracks_note": "the newsvendor and triage tasks are synthetic BY DESIGN "
                       "(closed-form, so the oracle is exact); real_triage uses the "
                       "real PhysioNet MIMIC-IV demo and is scored without an oracle",
        "real_triage": run_real_triage(),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "latest.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf8")
    return results


def main() -> int:
    r = run()
    o = r["oracle"]
    print(f"oracle: newsvendor critical fractile {o['newsvendor_critical_fractile']}, "
          f"optimal order {o['newsvendor_optimal_order']:.1f}")
    print(f"        triage treats when P(deteriorate) >= "
          f"{o['triage_treat_threshold_prob']}")
    g = r["oracle_gap"]
    print(f"\nnewsvendor oracles: clairvoyant {g['clairvoyant_cost']:.2f}, "
          f"best fixed action {g['best_fixed_action_cost']:.2f}, "
          f"distributional optimum {g['distributional_optimum_cost']:.2f}")
    print(f"  irreducible regret against the clairvoyant: "
          f"{g['irreducible_regret']:.2f}")
    for key in ("newsvendor", "triage"):
        t = r[key]
        print(f"\n=== {t['task']} ===")
        print(f"{'policy':<32}{'regret':>9}{'p90':>9}{'% opt':>8}{'ECE':>8}"
              f"{'worst/base':>12}{'slope':>9}")
        for name, v in t["policies"].items():
            rb = v["robustness"]
            rel = rb["relative_worst"]
            print(f"{name:<32}{v['regret']['mean']:>9.4f}{v['regret']['p90']:>9.4f}"
                  f"{v['regret']['frac_optimal']:>8.1%}{v['calibration']['ece']:>8.4f}"
                  f"{(rel if rel else 0):>12.2f}"
                  f"{rb['degradation_per_unit_shift']:>9.4f}")
    rtr = r.get("real_triage", {})
    print("\n=== real_triage (MIMIC-IV demo -- no oracle) ===")
    if rtr.get("policies"):
        sp = rtr["split"]
        print(f"real cohort: {sp['n_subjects_train']}+{sp['n_subjects_test']} "
              f"subjects, {sp['n_decisions_train']}+{sp['n_decisions_test']} "
              f"decisions; event rate {rtr['cohort']['event_rate_in_horizon']:.1%}")
        print(f"hindsight-best fixed threshold cost: {rtr['hindsight_floor']['cost']}")
        print(f"{'policy':<40}{'cost':>9}{'excess':>9}{'treat%':>9}")
        for name, v in rtr["policies"].items():
            print(f"{name:<40}{v['realised_cost']:>9.4f}"
                  f"{v['excess_over_hindsight_floor']:>9.4f}"
                  f"{v['treat_rate']:>9.1%}")
        print(f"calibrated-logistic ECE: "
              f"{rtr['calibrated_logistic_calibration']['ece']}")
        sh = rtr["robustness_named_shift"]
        for k in ("train_younger", "train_older"):
            if k in sh:
                print(f"  {k}: in-dist {sh[k]['in_distribution_cost']:.4f} -> "
                      f"shifted {sh[k]['shifted_cost']:.4f} "
                      f"(degradation {sh[k]['degradation']:+.4f})")
    else:
        print(f"  {rtr.get('status', 'unavailable')}: {rtr.get('detail', '')}")

    try:
        from .site import build_site
        build_site(r); print("\nwebsite/ rebuilt from this run")
    except Exception as exc:
        print(f"\n(site not rebuilt: {exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
