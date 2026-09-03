# 4.0 — Decision Intelligence Framework

The repository whose name matches the endeavor phrase most closely is the one least aligned with it. This rebuild makes the name true: optimization-driven, system-level decision frameworks.

Part of a five-repository portfolio supporting the endeavor described in the
EB2-NIW petition: **optimization-driven, system-level decision frameworks** —
integrating operations research, mathematical optimization and applied AI — for
domains where a wrong decision carries systemic consequences. The three pillars
are financial stability, healthcare safety and secure digital infrastructure.

| | |
|---|---|
| Petition-grade projects today | 0 original, NIW-relevant projects |
| Verdict | **Rebuild around its own name** |

> "Petition-grade" means: original work, a stated method, real data at a stated
> scale, a measured result, and a README a reviewer can follow. Counts exclude
> duplicates, forks of third-party work, retired projects, and asset-only
> folders.

## Projects

| Folder | Project | Pillar | Evidence value |
|---|---|---|---|
| [`optimization-under-uncertainty/`](optimization-under-uncertainty/) | Optimization-Under-Uncertainty Library | Cross-cutting — the endeavor itself | CORE — the endeavor expressed as code |
| [`decision-audit-framework/`](decision-audit-framework/) | Decision-Audit Framework | Cross-cutting | CORE — makes interpretability an artifact |
| [`icu-triage-optimization/`](icu-triage-optimization/) | Multi-Objective ICU Triage Optimization | Healthcare Safety | CORE — joins the healthcare pillar to the OR framing |
| [`decision-benchmark-suite/`](decision-benchmark-suite/) | Decision-Framework Benchmark Suite | Cross-cutting | CORE — the most citable form of contribution |

## What each one is

### 1. Optimization-Under-Uncertainty Library — [`optimization-under-uncertainty/`](optimization-under-uncertainty/)

Stochastic and robust optimization applied to portfolio allocation and hospital resource allocation.

*Why it earns its place:* This is your Cornell ORIE training expressed as code, and the petition's Version 3 rests entirely on that framing with nothing to point at.

*Target scale:* Portfolio allocation and hospital resource allocation instances

### 2. Decision-Audit Framework — [`decision-audit-framework/`](decision-audit-framework/)

Decision logging, counterfactual replay, and attribution for AI-assisted decisions.

*Why it earns its place:* Makes "auditable, interpretable decision frameworks" an artifact rather than a phrase.

*Target scale:* Decision traces from the other projects in this repository

### 3. Multi-Objective ICU Triage Optimization — [`icu-triage-optimization/`](icu-triage-optimization/)

Multi-objective ICU triage and alert-threshold optimization.

*Why it earns its place:* Connects the healthcare pillar to the operations-research framing, which is currently the weakest join in the petition.

*Target scale:* MIMIC-IV / eICU cohorts, shared with repo 2.0

### 4. Decision-Framework Benchmark Suite — [`decision-benchmark-suite/`](decision-benchmark-suite/)

Regret, calibration and robustness benchmarks across the frameworks in this repository.

*Why it earns its place:* A benchmark is the most citable form of contribution and the easiest for others to adopt.

*Target scale:* All frameworks in this repository

## Repository layout

```
4.0-Decision-Intelligence-Framework/
├── optimization-under-uncertainty/
├── decision-audit-framework/
├── icu-triage-optimization/
├── decision-benchmark-suite/
│
│   ── earlier work, promoted out of previous/ ──
├── efficiency-toolkit/
├── geo-mapping-system/
└── llm-fine-tuning-template/
```

Every rebuilt project carries the same skeleton: `README.md`, `src/`, `data/`, `results/`, `tests/`, `website/`.

## Ground rules

1. **No number without a run log.** Anything cited in the petition must appear
   in that project's `results/README.md` with a run date behind it.
2. **No simulated data under a real claim.** Sample data lives in
   `data/sample/`, labelled, and is never the source of a cited figure.
3. **Adoption must be documentable** — named institutions, dated
   correspondence, registry statistics. Never an inflated count.
4. **Third-party and forked code stays labelled** and is never counted.

## Earlier work

There is no `previous/` folder any more. Everything that was in it has been promoted to the top level, so every piece of work in this repository is a first-class folder that can be opened, read and continued.

Nothing was deleted except items the rebuild genuinely supersedes; those remain in git history.

| Folder | What it is | How it may be used |
|---|---|---|
| [`efficiency-toolkit/`](efficiency-toolkit/) | Academic reference and citation automation (two scripts). | Prior original work. Duplicated as `repo4-llm-efficiency-reference-search/` in repo 5.0. |
| [`geo-mapping-system/`](geo-mapping-system/) | PoetMap / 寻踪·诗意盛唐 — Tang poet journeys as 3D geo-visualisation. Documents, meeting records and design files only; no code. | Documents only. Not citable as software. |
| [`llm-fine-tuning-template/`](llm-fine-tuning-template/) | Qwen DPO preference-tuning scaffold. | Prior original work. Duplicated as `repo3-fine-tuning-template/` in repo 5.0. |

**Read the third column before citing anything here.** Forks of third-party work, duplicates, retired projects and asset-only folders are labelled as such and are not part of the petition's evidence.


---
Scaffold generated from `NIW_Project_Portfolio_and_Gap_Plan.xlsx` (sheets: Repo Build-Out Plan, Core Ideas at a Glance, NIW Claim vs Repo Evidence, Notion 创业 Alignment). Structure only — no results are claimed here yet.
