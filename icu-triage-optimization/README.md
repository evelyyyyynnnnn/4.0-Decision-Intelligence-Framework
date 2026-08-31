# Multi-Objective ICU Triage Optimization

> **Status: scaffold.** Structure only — no method, data or result is claimed
> yet. Every "not yet measured" below is a real gap, not a placeholder to be
> filled in with an estimate.

**Repository:** `4.0-Decision-Intelligence-Framework`
**NIW pillar (Dhanasar prong 1):** Healthcare Safety
**Evidence value:** CORE — joins the healthcare pillar to the OR framing

## Core idea

Multi-objective ICU triage and alert-threshold optimization.

## Why it earns its place

Connects the healthcare pillar to the operations-research framing, which is currently the weakest join in the petition.

## The petition claim it supports

> Optimization-driven decision frameworks applied to healthcare safety.

**What the portfolio shows today:** The healthcare and operations-research halves of the endeavor are argued separately and joined nowhere.

**Action required:** Build it against the ICU early-warning models in repo 2.0 so the join is demonstrated, not asserted.

Prior work to build on: `2.0-Healthcare-Ai-Systems — icu-early-warning`.

## Petition-grade checklist

A project counts as petition-grade only when all five are true. None are yet.

- [ ] Original work, authored here
- [ ] A stated method (`docs/METHOD.md`)
- [ ] Real data at a stated scale (`docs/DATA.md` — target: MIMIC-IV / eICU cohorts, shared with repo 2.0)
- [ ] A measured result (`results/README.md`)
- [ ] A README a reviewer can follow, start to finish

## Measured results

Target scale: **MIMIC-IV / eICU cohorts, shared with repo 2.0**

| Metric | Baseline | Result | Out-of-sample |
|---|---|---|---|
| Pareto frontier: false alerts vs. missed deterioration | _not yet measured_ | _not yet measured_ | _pending_ |
| Threshold policy vs. fixed-threshold baseline | _not yet measured_ | _not yet measured_ | _pending_ |
| Nurse-workload proxy at equal sensitivity | _not yet measured_ | _not yet measured_ | _pending_ |

Populate this from `results/`. Do not cite any number in the petition that does
not appear here with a run date behind it.

## Layout

```
icu-triage-optimization/
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
