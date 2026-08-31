# ICU Triage Optimization

> Pareto frontiers over alert thresholds and review-capacity triage — the join between the healthcare models and the operations-research framing, demonstrated rather than asserted.

**Repository:** `4.0-Decision-Intelligence-Framework` &middot; **Pillar:** Healthcare Safety

## Status

This is working code with a runnable demo and 0 tests. It is **not** a
finished result.

Risk scores come from the ICU early-warning model in repo 2.0, trained on its synthetic cohort. No real patients, and no clinical conclusion follows. What is real is the optimisation: the frontiers, the domination analysis and the triage comparison are computed from those scores.

Last run: `2026-08-31T19:06:07+00:00`

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 0 tests
python -m src.demo             # runs everything, rewrites results/ and website/
```

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
  |-- frontier.py
  |-- site.py
  |-- sitekit.py
  |-- triage.py
tests/
  |-- .gitkeep
  |-- test_triage.py
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
cp -r website/ ../my-icu-triage-optimization-site && cd ../my-icu-triage-optimization-site
git init && git add -A && git commit -m "site"
vercel deploy --prod
```

The page is regenerated from `results.json` on every `python -m src.demo`, so the
figures on the site and the figures the code produces cannot drift apart. Do not edit
numbers on the page by hand.

## Honesty note

Everything in this project runs on clearly-labelled synthetic or authored data.
Swap in the real source and the same pipeline reports real numbers &mdash; that is
what the structure is for. Until that happens, nothing here should be cited as a
measured result, and the site's closing section states explicitly what the project
does not establish.
