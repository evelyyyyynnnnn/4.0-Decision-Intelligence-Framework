"""Fit every method in-sample, score every method out-of-sample."""
from __future__ import annotations
import json, pathlib, sys
from datetime import datetime, timezone
import numpy as np
from .problems import (portfolio_loss, sample_demand, sample_returns,
                       staffing_cost)
from .solvers import (cvar_portfolio, cvar_staffing, deterministic_portfolio,
                      deterministic_staffing, robust_portfolio, saa_portfolio,
                      saa_staffing)
from .evaluate import regret, summarise_losses

ROOT = pathlib.Path(__file__).resolve().parent.parent
ALPHA = 0.90


def portfolio_study() -> dict:
    train = sample_returns(n_assets=8, n_scenarios=600, seed=0)
    test = sample_returns(n_assets=8, n_scenarios=4000, seed=99)

    weights = {
        "equal weight": np.full(8, 1 / 8),
        "deterministic (mean only)": deterministic_portfolio(train),
        "SAA mean-variance": saa_portfolio(train, risk_aversion=3.0),
        f"CVaR-{int(ALPHA*100)}": cvar_portfolio(train, alpha=ALPHA),
        "robust (box)": robust_portfolio(train, gamma=1.0),
    }

    in_sample, out_sample, losses = {}, {}, {}
    for name, w in weights.items():
        in_sample[name] = summarise_losses(portfolio_loss(w, train.values), ALPHA)
        L = portfolio_loss(w, test.values)
        losses[name] = L
        out_sample[name] = summarise_losses(L, ALPHA)

    return {
        "n_assets": 8, "n_train": 600, "n_test": 4000, "alpha": ALPHA,
        "weights": {k: [round(float(x), 4) for x in v] for k, v in weights.items()},
        "concentration": {k: round(float(np.sum(v ** 2)), 4)
                          for k, v in weights.items()},
        "in_sample": in_sample, "out_of_sample": out_sample,
        "regret": regret(losses),
    }


def staffing_study() -> dict:
    train = sample_demand(n_units=6, n_scenarios=600, seed=1)
    test = sample_demand(n_units=6, n_scenarios=4000, seed=77)
    # Budget set above mean demand. At exactly the mean, every method is
    # starved by the surge scenarios and the comparison becomes a comparison of
    # who fails least, which hides the allocation decision being studied.
    budget = float(train.mean().sum()) * 1.25

    plans = {
        "deterministic (mean demand)": deterministic_staffing(train, budget),
        "SAA expected cost": saa_staffing(train, budget),
        f"CVaR-{int(ALPHA*100)}": cvar_staffing(train, budget, alpha=ALPHA),
    }

    out, losses, service, insample = {}, {}, {}, {}
    for name, x in plans.items():
        insample[name] = summarise_losses(staffing_cost(x, train.values), ALPHA)
        C = staffing_cost(x, test.values)
        losses[name] = C
        out[name] = summarise_losses(C, ALPHA)
        gap = np.clip(test.values - x, 0, None)
        service[name] = {
            "unmet_rate": round(float((gap > 0).mean()), 4),
            "mean_unmet": round(float(gap.sum(axis=1).mean()), 4),
            "p95_unmet": round(float(np.percentile(gap.sum(axis=1), 95)), 4),
            "shifts_fully_covered": round(
                float((gap.sum(axis=1) == 0).mean()), 4),
        }

    return {
        "n_units": 6, "budget": round(budget, 2), "alpha": ALPHA,
        "unmet_penalty": 6.0,
        "plans": {k: [round(float(v), 2) for v in x] for k, x in plans.items()},
        "in_sample": insample,
        "out_of_sample": out, "service": service, "regret": regret(losses),
        # Is the surge a common shock across units? If it is, no reallocation of
        # a FIXED budget can protect against it, and the CVaR plan pays for tail
        # protection it cannot deliver.
        "surge_is_common": round(float(np.mean(
            np.corrcoef(test.values.T)[np.triu_indices(test.values.shape[1], 1)])), 4),
    }


def scenario_sweep() -> dict:
    """Does CVaR's out-of-sample disadvantage shrink with more scenarios?

    CVaR-90 fits its objective on the worst 10% of the training sample, so at
    600 scenarios it is estimating a tail from 60 points. If that is the cause
    of its poor out-of-sample showing, the gap should close as the sample grows
    -- and if it does not, the problem is the formulation rather than the sample
    size. This is the check that distinguishes the two.
    """
    test = sample_returns(n_assets=8, n_scenarios=6000, seed=99)
    rows = []
    for n in (200, 400, 800, 1600, 3200, 6400):
        train = sample_returns(n_assets=8, n_scenarios=n, seed=5)
        w_cvar = cvar_portfolio(train, alpha=ALPHA)
        w_saa = saa_portfolio(train, risk_aversion=3.0)
        c_cvar = summarise_losses(portfolio_loss(w_cvar, test.values), ALPHA)
        c_saa = summarise_losses(portfolio_loss(w_saa, test.values), ALPHA)
        rows.append({
            "n_train": n,
            "tail_scenarios": int(n * (1 - ALPHA)),
            "cvar_method_oos_cvar": c_cvar["cvar"],
            "saa_method_oos_cvar": c_saa["cvar"],
            "gap": round(c_cvar["cvar"] - c_saa["cvar"], 6),
            "cvar_concentration": round(float(np.sum(w_cvar ** 2)), 4),
        })
    return {"rows": rows,
            "gap_narrows": rows[-1]["gap"] < rows[0]["gap"]}


def in_vs_out(port: dict) -> list:
    """How much each method flatters itself when scored in-sample."""
    rows = []
    for name in port["in_sample"]:
        i = port["in_sample"][name]["cvar"]
        o = port["out_of_sample"][name]["cvar"]
        rows.append({"method": name, "in_sample_cvar": i, "out_of_sample_cvar": o,
                     "optimism": round(o - i, 6)})
    return sorted(rows, key=lambda r: -r["optimism"])


def run() -> dict:
    port = portfolio_study()
    staff = staffing_study()
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_synthetic": True,
        "data_source": "synthetic scenario samples (src/problems.py)",
        "portfolio": port,
        "staffing": staff,
        "optimism": in_vs_out(port),
        "scenario_sweep": scenario_sweep(),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "latest.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf8")
    return results


def portfolio_study_real(root):
    """The same five allocations, fitted to the past and judged on the future."""
    import numpy as _np
    from data.load import load_scenarios

    train, test, meta = load_scenarios(root=root)
    n = meta["n_assets"]

    weights = {
        "equal weight": _np.full(n, 1 / n),
        "deterministic (mean only)": deterministic_portfolio(train),
        "SAA mean-variance": saa_portfolio(train, risk_aversion=3.0),
        f"CVaR-{int(ALPHA*100)}": cvar_portfolio(train, alpha=ALPHA),
        "robust (box)": robust_portfolio(train, gamma=1.0),
    }

    in_sample, out_sample, losses = {}, {}, {}
    for name, w in weights.items():
        in_sample[name] = summarise_losses(portfolio_loss(w, train.values), ALPHA)
        L = portfolio_loss(w, test.values)
        losses[name] = L
        out_sample[name] = summarise_losses(L, ALPHA)

    return {
        "assets": meta["assets"],
        "n_assets": n,
        "n_train": meta["train"]["n_days"], "n_test": meta["test"]["n_days"],
        "alpha": ALPHA,
        "split": meta["split"], "split_rationale": meta["split_rationale"],
        "train_window": meta["train"], "test_window": meta["test"],
        "worst_day_train": meta["worst_day_train"],
        "worst_day_test": meta["worst_day_test"],
        "weights": {k: [round(float(x), 4) for x in v] for k, v in weights.items()},
        "concentration": {k: round(float(_np.sum(v ** 2)), 4)
                          for k, v in weights.items()},
        "in_sample": in_sample, "out_of_sample": out_sample,
        "regret": regret(losses),
        "provenance": meta["series"],
        "staffing_remains_simulated_because":
            meta["staffing_remains_simulated_because"],
    }


def run_real() -> dict:
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from data.load import ROOT as DATA_ROOT

    port = portfolio_study_real(DATA_ROOT)
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_synthetic": False,
        "data_source": "real daily closes from Stooq; see data/MANIFEST.json for "
                       "URLs, hashes and retrieval times",
        "portfolio": port,
        "optimism": in_vs_out(port),
        "staffing_reported": False,
        "staffing_withheld_because": port["staffing_remains_simulated_because"],
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
    p = r["portfolio"]
    print(f"source: {r['data_source']}")
    print(f"assets: {', '.join(p['assets'])}")
    print(f"split: {p['split']} -- fit {p['train_window']['first']}"
          f"..{p['train_window']['last']} ({p['n_train']} days), "
          f"judge {p['test_window']['first']}..{p['test_window']['last']} "
          f"({p['n_test']} days)")
    print(f"worst single day: {p['worst_day_train']:.2%} in the fitting window, "
          f"{p['worst_day_test']:.2%} in the judging window")
    print(f"\n{'method':<28}{'in CVaR':>10}{'out CVaR':>10}{'out mean':>10}"
          f"{'concen.':>9}")
    for name in p["weights"]:
        i, o = p["in_sample"][name], p["out_of_sample"][name]
        print(f"{name:<28}{i['cvar']:>10.4f}{o['cvar']:>10.4f}"
              f"{o['mean']:>10.4f}{p['concentration'][name]:>9.3f}")
    print("\noptimism (in-sample CVaR minus out-of-sample CVaR):")
    for row in r["optimism"]:
        print(f"  {row['method']:<28}{row['optimism']:+.4f}")
    print("\nwhy the split is chronological: " + p["split_rationale"])
    print("\nSTAFFING IS NOT REPORTED HERE: " + r["staffing_withheld_because"])
    print("wrote results/latest-real.json")
    return 0


def main() -> int:
    if "--real" in sys.argv[1:]:
        return main_real()
    r = run()
    p = r["portfolio"]
    print(f"portfolio: {p['n_assets']} assets, {p['n_train']} train / "
          f"{p['n_test']} test scenarios, alpha={p['alpha']}")
    print(f"\n{'method':<28}{'mean loss':>11}{'CVaR':>10}{'worst':>10}"
          f"{'concen.':>9}{'regret':>9}")
    for name in p["out_of_sample"]:
        o = p["out_of_sample"][name]
        print(f"{name:<28}{o['mean']:>11.4f}{o['cvar']:>10.4f}{o['worst']:>10.4f}"
              f"{p['concentration'][name]:>9.3f}"
              f"{p['regret'][name]['mean_regret']:>9.4f}")

    print("\nin-sample optimism (out-of-sample CVaR minus in-sample):")
    for row in r["optimism"]:
        print(f"  {row['method']:<28}{row['in_sample_cvar']:>9.4f} -> "
              f"{row['out_of_sample_cvar']:>8.4f}   {row['optimism']:+.4f}")

    sw = r["scenario_sweep"]
    print(f"\nCVaR vs SAA out-of-sample as the training sample grows "
          f"(gap narrows: {sw['gap_narrows']}):")
    print(f"  {'n_train':>8}{'tail pts':>10}{'CVaR method':>13}{'SAA method':>12}"
          f"{'gap':>9}{'concen.':>9}")
    for row in sw["rows"]:
        print(f"  {row['n_train']:>8}{row['tail_scenarios']:>10}"
              f"{row['cvar_method_oos_cvar']:>13.4f}"
              f"{row['saa_method_oos_cvar']:>12.4f}{row['gap']:>9.4f}"
              f"{row['cvar_concentration']:>9.3f}")

    s = r["staffing"]
    print(f"\nstaffing: {s['n_units']} units, budget {s['budget']}, "
          f"unmet penalty {s['unmet_penalty']}x")
    print(f"mean pairwise demand correlation across units: {s['surge_is_common']}")
    print(f"\n{'method':<28}{'CVaR in':>10}{'CVaR out':>10}{'mean cost':>11}"
          f"{'unmet':>8}{'covered':>9}")
    for name in s["out_of_sample"]:
        o, sv = s["out_of_sample"][name], s["service"][name]
        i = s["in_sample"][name]
        print(f"{name:<28}{i['cvar']:>10.1f}{o['cvar']:>10.1f}{o['mean']:>11.2f}"
              f"{sv['unmet_rate']:>8.1%}{sv['shifts_fully_covered']:>9.1%}")
    try:
        from .site import build_site
        build_site(r); print("\nwebsite/ rebuilt from this run")
    except Exception as exc:
        print(f"\n(site not rebuilt: {exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
