# Decision-Framework Benchmark Suite

> **Status: scaffold.** Structure only — no method, data or result is claimed
> yet. Every "not yet measured" below is a real gap, not a placeholder to be
> filled in with an estimate.

**Repository:** `4.0-Decision-Intelligence-Framework`
**NIW pillar (Dhanasar prong 1):** Cross-cutting
**Evidence value:** CORE — the most citable form of contribution

## Core idea

Regret, calibration and robustness benchmarks across the frameworks in this repository.

## Why it earns its place

A benchmark is the most citable form of contribution and the easiest for others to adopt.

## The petition claim it supports

> Independent adoption of open-source contributions.

**What the portfolio shows today:** No benchmark exists for any decision framework in the portfolio.

**Action required:** Build the suite last, once the three frameworks above have something to measure. Version it and give it a DOI, as with ChainTrust-Bench.

No prior work in the portfolio — this starts from scratch.

## Petition-grade checklist

A project counts as petition-grade only when all five are true. None are yet.

- [ ] Original work, authored here
- [ ] A stated method (`docs/METHOD.md`)
- [ ] Real data at a stated scale (`docs/DATA.md` — target: All frameworks in this repository)
- [ ] A measured result (`results/README.md`)
- [ ] A README a reviewer can follow, start to finish

## Measured results

Target scale: **All frameworks in this repository**

| Metric | Baseline | Result | Out-of-sample |
|---|---|---|---|
| Regret | _not yet measured_ | _not yet measured_ | _pending_ |
| Calibration | _not yet measured_ | _not yet measured_ | _pending_ |
| Robustness under shift | _not yet measured_ | _not yet measured_ | _pending_ |
| External adoption — documented, never inflated | _not yet measured_ | _not yet measured_ | _pending_ |

Populate this from `results/`. Do not cite any number in the petition that does
not appear here with a run date behind it.

## Layout

```
decision-benchmark-suite/
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
