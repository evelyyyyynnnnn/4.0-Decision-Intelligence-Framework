# Decision Benchmark Suite

> Regret against an exact oracle, calibration of stated confidence, and degradation under named distribution shifts — three axes on which a decision policy can be wrong in different ways.

**Repository:** `4.0-Decision-Intelligence-Framework` &middot; **Pillar:** Cross-cutting

## Status

Working code with a runnable demo and 24 passing tests, across two tracks: a
**synthetic** closed-form track (exact oracle) and a **real-data** track on the
PhysioNet MIMIC-IV demo (no oracle).

The closed-form tasks are synthetic *by design*: because they are closed-form,
the oracle is exact rather than searched, which is what makes regret computable
instead of estimated. That also means they are constructed problems, and a
score there says nothing about a messy real one — which is exactly why the
real-data track exists alongside them.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 24 tests (real-track tests skip if MIMIC absent)
python -m src.demo             # runs everything, rewrites results/ and website/
```

## Real-data track

`src/real_tasks.py` builds a triage decision problem from the real PhysioNet
MIMIC-IV demo (~130 ICU stays; open access, non-redistributable, so the cache
is gitignored). At each decision time the policy sees vitals from time ≤ t and
must choose *wait* or *treat/escalate*; the label is whether hypotension
(MAP < 65) occurs in the next two hours. Because real data has **no oracle**,
this track is scored honestly and differently from the synthetic tasks:

- **realised cost**, compared to the *best in-hindsight fixed threshold* on the
  test split (the achievable floor), never to an unreachable clairvoyant;
- **calibration (ECE)** of the predicted event probability against realised
  outcomes;
- **degradation under a named covariate shift** — train on one age group,
  evaluate realised cost on the other.

The split is by subject (no patient on both sides). Caveat, carried from the
ICU project this data is shared with: the label is a future threshold on MAP,
which is also a feature, so this measures short-horizon *persistence* of an
observed vital, not an independent clinical outcome, and the cohort is a
demonstration, not a study. Real numbers live in `results/latest.json` under
`real_triage`; if the cache is absent the demo records that honestly instead of
reporting anything.

## Layout

```
README.md
data/
  |-- README.md
  |-- manifests/
  |-- sample/
docs/
  |-- DATA.md
  |-- EVIDENCE.md
  |-- METHOD.md
requirements.txt
results/
  |-- README.md
  |-- latest.json
src/
  |-- .gitkeep
  |-- __init__.py
  |-- demo.py
  |-- metrics.py
  |-- policies.py
  |-- real_tasks.py
  |-- site.py
  |-- sitekit.py
  |-- tasks.py
tests/
  |-- .gitkeep
  |-- test_benchmark.py
  |-- test_real_triage.py
website/
  |-- README.md
  |-- index.html
  |-- results.json
  |-- vercel.json
```

- `src/` &mdash; the implementation.
- `tests/` &mdash; pytest suite. These guard behaviour, not just imports.
- `results/latest.json` &mdash; the output of the last demo run. Every figure quoted
  anywhere in this project traces back to this file.
- `website/` &mdash; a self-contained static site, deployable to Vercel by copying the
  folder into its own repository. See `website/README.md`.

## The website

`website/` has no build step. To deploy it independently:

```bash
cp -r website/ ../my-4-decision-benchmark-suite-site && cd ../my-4-decision-benchmark-suite-site
git init && git add -A && git commit -m "site"
vercel deploy --prod
```

The page is regenerated from `results.json` on every `python -m src.demo`, so the
figures on the site and the figures the code produces cannot drift apart. Do not edit
numbers on the page by hand.

## Honesty note

Two tracks, labelled as such. The closed-form tasks (`newsvendor`, `triage`) run
on synthetic data *by design* — that is the only way the oracle can be exact and
regret can be measured rather than estimated; their scores say nothing about a
real problem and should not be cited as measured real-world results. The
`real_triage` track runs on the real PhysioNet MIMIC-IV demo and is scored
without an oracle (realised cost vs the hindsight-best threshold, calibration,
and degradation under a named shift). Its cohort is a demonstration, not a study,
and its label is a threshold on an observed vital rather than an independent
outcome — the results file and the site both state this plainly.
