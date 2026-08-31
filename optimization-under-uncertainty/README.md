# Optimization-Under-Uncertainty Library

> **Status: scaffold.** Structure only — no method, data or result is claimed
> yet. Every "not yet measured" below is a real gap, not a placeholder to be
> filled in with an estimate.

**Repository:** `4.0-Decision-Intelligence-Framework`
**NIW pillar (Dhanasar prong 1):** Cross-cutting — the endeavor itself
**Evidence value:** CORE — the endeavor expressed as code

## Core idea

Stochastic and robust optimization applied to portfolio allocation and hospital resource allocation.

## Why it earns its place

This is your Cornell ORIE training expressed as code, and the petition's Version 3 rests entirely on that framing with nothing to point at.

## The petition claim it supports

> Endeavor framed as "optimization-driven, system-level decision frameworks" (Version 3 of the petition letter).

**What the portfolio shows today:** Repo 4.0 is named Decision-Intelligence-Framework but contained a Tang-poetry WebGIS project and two duplicated LLM utilities.

**Action required:** Build the library. It is the load-bearing artifact for the endeavor statement itself.

No prior work in the portfolio — this starts from scratch.

## Petition-grade checklist

A project counts as petition-grade only when all five are true. None are yet.

- [ ] Original work, authored here
- [ ] A stated method (`docs/METHOD.md`)
- [ ] Real data at a stated scale (`docs/DATA.md` — target: Portfolio allocation and hospital resource allocation instances)
- [ ] A measured result (`results/README.md`)
- [ ] A README a reviewer can follow, start to finish

## Measured results

Target scale: **Portfolio allocation and hospital resource allocation instances**

| Metric | Baseline | Result | Out-of-sample |
|---|---|---|---|
| Objective value vs. deterministic baseline | _not yet measured_ | _not yet measured_ | _pending_ |
| Regret under distribution shift | _not yet measured_ | _not yet measured_ | _pending_ |
| Robustness to worst-case scenarios | _not yet measured_ | _not yet measured_ | _pending_ |
| Solve time at problem scale | _not yet measured_ | _not yet measured_ | _pending_ |

Populate this from `results/`. Do not cite any number in the petition that does
not appear here with a run date behind it.

## Layout

```
optimization-under-uncertainty/
├── README.md        this file
├── docs/
│   ├── METHOD.md    what the method is and why it is non-obvious
│   ├── DATA.md      source, scale, licence, and how to reproduce the pull
│   └── EVIDENCE.md  the petition claim, the gap, and the exhibit it becomes
├── src/             implementation
├── data/            pointers and manifests — never raw licensed data
├── results/         measured results, run logs, and the baseline comparison
└── tests/           tests that establish the result is reproducible
```

---
Scaffold generated from `NIW_Project_Portfolio_and_Gap_Plan.xlsx` (sheets: Repo Build-Out Plan, Core Ideas at a Glance, NIW Claim vs Repo Evidence, Notion 创业 Alignment). Structure only — no results are claimed here yet.
