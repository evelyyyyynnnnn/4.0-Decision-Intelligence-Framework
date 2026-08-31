# Decision-Audit Framework

> **Status: scaffold.** Structure only — no method, data or result is claimed
> yet. Every "not yet measured" below is a real gap, not a placeholder to be
> filled in with an estimate.

**Repository:** `4.0-Decision-Intelligence-Framework`
**NIW pillar (Dhanasar prong 1):** Cross-cutting
**Evidence value:** CORE — makes interpretability an artifact

## Core idea

Decision logging, counterfactual replay, and attribution for AI-assisted decisions.

## Why it earns its place

Makes "auditable, interpretable decision frameworks" an artifact rather than a phrase.

## The petition claim it supports

> Auditable, interpretable decision frameworks.

**What the portfolio shows today:** Nothing in any repository implements decision auditing.

**Action required:** Build logging, counterfactual replay and attribution, and wire it into the optimization library and the ICU triage work.

No prior work in the portfolio — this starts from scratch.

## Petition-grade checklist

A project counts as petition-grade only when all five are true. None are yet.

- [ ] Original work, authored here
- [ ] A stated method (`docs/METHOD.md`)
- [ ] Real data at a stated scale (`docs/DATA.md` — target: Decision traces from the other projects in this repository)
- [ ] A measured result (`results/README.md`)
- [ ] A README a reviewer can follow, start to finish

## Measured results

Target scale: **Decision traces from the other projects in this repository**

| Metric | Baseline | Result | Out-of-sample |
|---|---|---|---|
| Replay fidelity | _not yet measured_ | _not yet measured_ | _pending_ |
| Attribution faithfulness | _not yet measured_ | _not yet measured_ | _pending_ |
| Audit-log completeness | _not yet measured_ | _not yet measured_ | _pending_ |
| Counterfactual coverage | _not yet measured_ | _not yet measured_ | _pending_ |

Populate this from `results/`. Do not cite any number in the petition that does
not appear here with a run date behind it.

## Layout

```
decision-audit-framework/
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
