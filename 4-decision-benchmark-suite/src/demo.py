"""Run the suite across tasks, policies and distribution shifts."""
from __future__ import annotations
import json, pathlib, sys
from datetime import datetime, timezone
import numpy as np
from .metrics import calibration, regret, robustness
from .policies import (AlwaysAction, BayesTriage, EmpiricalNewsvendor,
                       MeanDemandNewsvendor, ThresholdTriage)
from .tasks import NewsvendorTask, TriageTask

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHIFTS = (-16.0, -8.0, 0.0, 8.0, 16.0)
TRIAGE_SHIFTS = (-1.2, -0.6, 0.0, 0.6, 1.2)


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
    try:
        from .site import build_site
        build_site(r); print("\nwebsite/ rebuilt from this run")
    except Exception as exc:
        print(f"\n(site not rebuilt: {exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
