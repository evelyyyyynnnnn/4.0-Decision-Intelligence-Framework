"""Frontiers and triage policies, using the ICU models from repo 2.0 when present."""
from __future__ import annotations
import importlib.util, json, pathlib, sys, types
from datetime import datetime, timezone
import numpy as np
from .frontier import (constrained_best, hypervolume, knee_point,
                       operating_points, pareto_front)
from .triage import (evaluate_policy, expected_benefit_policy, threshold_policy,
                     top_k_policy)

ROOT = pathlib.Path(__file__).resolve().parent.parent
ICU = ROOT.parent.parent / "2.0-Healthcare-Ai-Systems" / "icu-early-warning"


def _load_icu():
    """Import the ICU project from repo 2.0 under its own namespace.

    Both projects ship a package called `src`, so a plain sys.path import would
    resolve to whichever was loaded first. Loading by file path keeps them
    independently copyable -- neither has to know it sits beside the other.
    """
    ns = "icu_early_warning"
    if ns not in sys.modules:
        pkg = types.ModuleType(ns)
        pkg.__path__ = [str(ICU / "src")]
        sys.modules[ns] = pkg
    mods = {}
    for name in ("cohort", "dataset", "models", "metrics"):
        full = f"{ns}.{name}"
        if full in sys.modules:
            mods[name] = sys.modules[full]
            continue
        spec = importlib.util.spec_from_file_location(full, ICU / "src" / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        mods[name] = m
    return mods


def _scores_from_icu():
    m = _load_icu()
    patients = m["cohort"].make_cohort(n_patients=300, hours=48.0, step_h=0.5, seed=11)
    X, y, groups, _ = m["dataset"].build(patients, event="hypotension", horizon_h=4.0)
    tr, te = m["dataset"].split_by_patient(groups, test_frac=0.3, seed=3)
    mdl = m["models"].boosted().fit(X[tr], y[tr])
    p = mdl.predict_proba(X[te])[:, 1]
    key = m["dataset"].FEATURE_NAMES.index("map_mmhg")
    baseline = -X[te][:, key]
    return y[te], p, baseline, "ICU early-warning model (repo 2.0)"


def _scores_synthetic():
    rng = np.random.default_rng(3)
    n = 6000
    y = (rng.uniform(size=n) < 0.12).astype(int)
    p = np.clip(rng.beta(2, 8, n) + y * rng.uniform(0.05, 0.35, n), 0, 1)
    baseline = np.clip(rng.beta(2, 8, n) + y * rng.uniform(0.0, 0.15, n), 0, 1)
    return y, p, baseline, "synthetic fallback (repo 2.0 not found)"


def run() -> dict:
    try:
        y, p, baseline, source = _scores_from_icu()
        linked = True
    except Exception:
        y, p, baseline, source = _scores_synthetic()
        linked = False

    pts = operating_points(y, p)
    front = pareto_front(pts)
    base_pts = operating_points(y, baseline)
    base_front = pareto_front(base_pts)

    # Pooling both policies' operating points is where domination actually
    # appears. Sweeping ONE score's threshold can never produce a dominated
    # point -- the trade-off is monotone by construction, so the "frontier" is
    # the whole sweep and reporting it as a Pareto result says nothing. The
    # meaningful question is how many of the baseline's settings are beaten
    # outright by some setting of the model.
    pooled = [(pt, "model") for pt in pts] + [(pt, "baseline") for pt in base_pts]
    pooled_front = pareto_front([pt for pt, _ in pooled])
    front_ids = {id(pt) for pt in pooled_front}
    baseline_dominated = sum(1 for pt, src_ in pooled
                             if src_ == "baseline" and id(pt) not in front_ids)
    model_dominated = sum(1 for pt, src_ in pooled
                          if src_ == "model" and id(pt) not in front_ids)
    pooled_stats = {
        "n_pooled": len(pooled),
        "n_on_front": len(pooled_front),
        "baseline_points_dominated": baseline_dominated,
        "baseline_points_total": len(base_pts),
        "model_points_dominated": model_dominated,
        "model_points_total": len(pts),
        "front_share_model": round(
            sum(1 for pt in pooled_front if any(pt is q for q in pts))
            / max(1, len(pooled_front)), 4),
    }

    knee = knee_point(front)
    governance = {}
    for req in (0.70, 0.80, 0.90, 0.95):
        b = constrained_best(pts, req)
        bb = constrained_best(base_pts, req)
        governance[f"{req:.2f}"] = {
            "model": b.as_dict() if b else None,
            "baseline": bb.as_dict() if bb else None,
            "false_alert_reduction_pct": (
                round(100.0 * (bb.false_alerts_per_100 - b.false_alerts_per_100)
                      / bb.false_alerts_per_100, 2)
                if b and bb and bb.false_alerts_per_100 else None),
        }

    # Triage: fixed review capacity, with benefit not equal to risk.
    rng = np.random.default_rng(9)
    benefit = np.clip(rng.beta(2, 2, len(y)) * 2.0, 0.05, None)
    capacity = max(1, int(0.05 * len(y)))
    policies = {
        "top-k by risk": top_k_policy(p, capacity),
        "top-k by expected benefit": expected_benefit_policy(p, capacity, benefit),
        "fixed threshold": threshold_policy(p, float(np.quantile(p, 0.95))),
    }
    triage = {k: evaluate_policy(v, y, benefit) for k, v in policies.items()}

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_synthetic": True,
        "data_source": source,
        "linked_to_repo_2": linked,
        "n_observations": int(len(y)),
        "event_rate": round(float(y.mean()), 4),
        "n_operating_points": len(pts),
        "frontier": [p_.as_dict() for p_ in front],
        "baseline_frontier": [p_.as_dict() for p_ in base_front],
        "n_dominated": len(pts) - len(front),
        "pooled": pooled_stats,
        "hypervolume": {"model": round(hypervolume(front), 3),
                        "baseline": round(hypervolume(base_front), 3)},
        "knee": knee.as_dict() if knee else None,
        "governance": governance,
        "triage": {"capacity": capacity, "policies": triage},
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "latest.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf8")
    return results


def main() -> int:
    r = run()
    print(f"source: {r['data_source']} (linked to repo 2.0: {r['linked_to_repo_2']})")
    print(f"{r['n_observations']:,} observations, event rate {r['event_rate']:.2%}")
    print(f"\n{r['n_operating_points']} thresholds swept -> "
          f"{len(r['frontier'])} on the Pareto front, "
          f"{r['n_dominated']} dominated and discardable")
    po = r["pooled"]
    print(f"pooling model + baseline settings ({po['n_pooled']} points): "
          f"{po['n_on_front']} non-dominated")
    print(f"  baseline settings dominated by some model setting: "
          f"{po['baseline_points_dominated']}/{po['baseline_points_total']}")
    print(f"  model settings dominated by some baseline setting: "
          f"{po['model_points_dominated']}/{po['model_points_total']}")
    hv = r["hypervolume"]
    print(f"hypervolume: model {hv['model']:.2f} vs baseline {hv['baseline']:.2f}")
    k = r["knee"]
    if k:
        print(f"knee point: sensitivity {k['sensitivity']:.3f}, "
              f"{k['false_alerts_per_100']:.2f} false alerts/100, "
              f"PPV {k['ppv']:.3f}")
    print("\nunder a governance-set sensitivity floor:")
    print(f"  {'required':>9}{'model FA/100':>14}{'baseline FA/100':>17}{'reduction':>11}")
    for req, g in r["governance"].items():
        if g["model"] and g["baseline"]:
            print(f"  {req:>9}{g['model']['false_alerts_per_100']:>14.2f}"
                  f"{g['baseline']['false_alerts_per_100']:>17.2f}"
                  f"{g['false_alert_reduction_pct']:>10.1f}%")
    t = r["triage"]
    print(f"\ntriage with {t['capacity']} review slots:")
    print(f"  {'policy':<28}{'reviewed':>9}{'caught':>8}{'yield':>8}"
          f"{'benefit capture':>17}")
    for name, v in t["policies"].items():
        print(f"  {name:<28}{v['reviewed']:>9}{v['caught']:>8}"
              f"{v['review_yield']:>8.3f}{v['benefit_capture']:>17.3f}")
    try:
        from .site import build_site
        build_site(r); print("\nwebsite/ rebuilt from this run")
    except Exception as exc:
        print(f"\n(site not rebuilt: {exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
