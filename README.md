# 4.0 — Decision Intelligence Framework

[![tests](https://github.com/evelyyyyynnnnn/4.0-Decision-Intelligence-Framework/actions/workflows/tests.yml/badge.svg)](https://github.com/evelyyyyynnnnn/4.0-Decision-Intelligence-Framework/actions/workflows/tests.yml)

The repository whose name matches the endeavor phrase most closely is the one least aligned with it. This rebuild makes the name true: optimization-driven, system-level decision frameworks.

Part of a five-repository portfolio on one theme: **optimization-driven,
system-level decision frameworks** —
integrating operations research, mathematical optimization and applied AI — for
domains where a wrong decision carries systemic consequences. The three pillars
are financial stability, healthcare safety and secure digital infrastructure.

| | |
|---|---|
| Projects with measured results | **3 of 4** |
| Tests across the suites | **107** |
| Running on real third-party data | **3 of 4** |
| Most recent reportable run | 2026-09-07 |
| Verdict | **Built around its own name — three projects measured** |

> "Measured" means: original work, a stated method, real data at a stated
> scale, a measured result, and a README a reviewer can follow. Counts exclude
> duplicates, forks of third-party work, retired projects, and asset-only
> folders.

## Projects

| Folder | Project | Pillar | Evidence value |
|---|---|---|---|
| [`2-optimization-under-uncertainty/`](2-optimization-under-uncertainty/) | Optimization-Under-Uncertainty Library | Cross-cutting — the endeavor itself | CORE — the endeavor expressed as code |
| [`1-decision-audit-framework/`](1-decision-audit-framework/) | Decision-Audit Framework | Cross-cutting | CORE — makes interpretability an artifact |
| [`3-icu-triage-optimization/`](3-icu-triage-optimization/) | Multi-Objective ICU Triage Optimization | Healthcare Safety | CORE — joins the healthcare pillar to the OR framing |
| [`4-decision-benchmark-suite/`](4-decision-benchmark-suite/) | Decision-Framework Benchmark Suite | Cross-cutting | CORE — the most citable form of contribution |

## What each one is

### 1. Optimization-Under-Uncertainty Library — [`2-optimization-under-uncertainty/`](2-optimization-under-uncertainty/)

Stochastic and robust optimization applied to portfolio allocation and hospital resource allocation.

*Why it earns its place:* The usual comparison reports each formulation's objective value under its own distribution, which flatters whichever one generated the scenarios. The informative test is cross-evaluation: score the stochastic solution under the robust uncertainty set and the robust solution under the sampled distribution, and report both.

*Target scale:* Portfolio allocation and hospital resource allocation instances

### 2. Decision-Audit Framework — [`1-decision-audit-framework/`](1-decision-audit-framework/)

Decision logging, counterfactual replay, and attribution for AI-assisted decisions.

*Why it earns its place:* Makes "auditable, interpretable decision frameworks" an artifact rather than a phrase.

*Target scale:* Decision traces from the other projects in this repository

### 3. Multi-Objective ICU Triage Optimization — [`3-icu-triage-optimization/`](3-icu-triage-optimization/)

Multi-objective ICU triage and alert-threshold optimization.

*Why it earns its place:* Triage policy and alert threshold are normally optimised separately, but the threshold determines the arrival process the triage policy then faces. Optimising either one alone yields a policy that degrades as soon as the other moves, so the two have to be solved together to mean anything.

*Target scale:* MIMIC-IV / eICU cohorts, shared with repo 2.0

### 4. Decision-Framework Benchmark Suite — [`4-decision-benchmark-suite/`](4-decision-benchmark-suite/)

Regret, calibration and robustness benchmarks across the frameworks in this repository.

*Why it earns its place:* A benchmark is the most citable form of contribution and the easiest for others to adopt.

*Target scale:* All frameworks in this repository

## Repository layout

```
4.0-Decision-Intelligence-Framework/
├── 2-optimization-under-uncertainty/
├── 1-decision-audit-framework/
├── 3-icu-triage-optimization/
└── 4-decision-benchmark-suite/
```

Every rebuilt project carries the same skeleton: `README.md`, `src/`, `data/`, `results/`, `tests/`, `website/`.

> Earlier and off-theme prior work (dashboards, consumer apps, templates, and other non-endeavor folders) has been moved to a separate `archive-prior-work` repository to keep this repository focused on the endeavor. It remains recoverable from this repository's git history.

## Ground rules

1. **No number without a run log.** Any figure quoted anywhere in this
   repository must appear in that project's `results/README.md` with a run
   date behind it.
2. **No simulated data under a real claim.** Sample data lives in
   `data/sample/`, labelled, and is never the source of a cited figure.
3. **Adoption must be documentable** — named institutions, dated
   correspondence, registry statistics. Never an inflated count.
4. **Third-party and forked code stays labelled** and is never counted.

---
Structure only — no results are claimed here unless a project's `results/README.md` records a dated run.

## Licence

[Apache License 2.0](LICENSE). The code, benchmarks and documentation in this
repository may be reused, modified and redistributed under its terms, including
by projects that need a patent grant to incorporate them.

Third-party data and vendored code kept under `*/data/` and
`historical-archive/` carry their own terms and are labelled where they appear.
