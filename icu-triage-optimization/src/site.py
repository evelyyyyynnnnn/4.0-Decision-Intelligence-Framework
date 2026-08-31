"""Builds website/ from the last demo run."""
from __future__ import annotations
import pathlib
from . import sitekit as sk

ROOT = pathlib.Path(__file__).resolve().parent.parent
META = {
    "name": "ICU Triage Optimization",
    "slug": "icu-triage-optimization",
    "repo": "4.0-Decision-Intelligence-Framework",
    "pillar": "Healthcare Safety",
    "tagline": "Pareto frontiers over alert thresholds and review-capacity triage — "
               "the join between the healthcare models and the operations-research "
               "framing, demonstrated rather than asserted.",
    "tags": [("Pareto frontier", ""), ("hypervolume", ""),
             ("capacity-constrained triage", ""), ("links to repo 2.0", ""),
             ("synthetic cohort", "demo")],
    "banner": "Risk scores come from the ICU early-warning model in repo 2.0, trained "
              "on its synthetic cohort. No real patients, and no clinical conclusion "
              "follows. What is real is the optimisation: the frontiers, the "
              "domination analysis and the triage comparison are computed from those "
              "scores.",
}


def build_site(results: dict) -> pathlib.Path:
    po, hv, k = results["pooled"], results["hypervolume"], results["knee"]
    gov, tri = results["governance"], results["triage"]

    metrics = sk.metric_grid([
        ("Observations", f"{results['n_observations']:,}",
         f"event rate {results['event_rate']:.1%}"),
        ("Baseline settings dominated",
         f"{po['baseline_points_dominated']}/{po['baseline_points_total']}",
         "beaten outright by the model"),
        ("Hypervolume", f"{hv['model']:.1f} vs {hv['baseline']:.1f}",
         "model vs single-vital alarm"),
        ("Benefit capture gain",
         f"{tri['policies']['top-k by expected benefit']['benefit_capture'] - tri['policies']['top-k by risk']['benefit_capture']:+.3f}",
         "benefit ranking vs risk ranking"),
    ])

    front_chart = sk.line_chart(
        [("model", [(p["false_alerts_per_100"], p["sensitivity"])
                    for p in results["frontier"]]),
         ("single-vital alarm", [(p["false_alerts_per_100"], p["sensitivity"])
                                 for p in results["baseline_frontier"]])],
        xlabel="false alerts per 100 observations", ylabel="sensitivity")

    gov_rows = []
    for req, g in gov.items():
        if g["model"] and g["baseline"]:
            gov_rows.append([
                f"{float(req):.0%}", f"{g['model']['false_alerts_per_100']:.2f}",
                f"{g['baseline']['false_alerts_per_100']:.2f}",
                f"{g['false_alert_reduction_pct']:+.1f}%",
                f"{g['model']['ppv']:.3f}"])
    gov_tbl = sk.table(
        ["Required sensitivity", "Model FA/100", "Baseline FA/100",
         "Reduction", "Model PPV"], gov_rows, numeric_cols=(0, 1, 2, 3, 4))

    gov_chart = sk.bar_chart(
        [(f"{float(req):.0%} sensitivity", g["false_alert_reduction_pct"])
         for req, g in gov.items()
         if g["model"] and g["baseline"] and g["false_alert_reduction_pct"] > 0],
        fmt="{:.1f}%")

    tri_tbl = sk.table(
        ["Policy", "Reviewed", "Caught", "Missed", "Review yield", "Benefit capture"],
        [[name, v["reviewed"], v["caught"], v["missed"],
          f"{v['review_yield']:.3f}", f"{v['benefit_capture']:.3f}"]
         for name, v in tri["policies"].items()],
        numeric_cols=(1, 2, 3, 4, 5))

    body = f"""
<section>
  <h2>There is no best threshold, and that is not a limitation</h2>
  <div class="stack">
    <p>A missed deterioration and a false alarm are different kinds of harm, and no
    exchange rate between them is a technical fact. So this project does not pick a
    threshold. It computes the frontier of non-dominated options and leaves the choice
    with the people who carry the consequences.</p>
    <p>What optimisation <em>can</em> settle is which settings are dominated — another
    setting is better on every axis at once — and those can be discarded on technical
    grounds alone.</p>
  </div>
</section>

<section>
  <h2>This run</h2>
  <div class="stack-lg">
    {metrics}
    <p class="mono" style="color:var(--muted);font-size:12.5px">
      generated {sk.esc(results['generated_at'])} &middot;
      {sk.esc(results['data_source'])} &middot;
      {results['n_operating_points']} thresholds swept
    </p>
    {front_chart}
  </div>
</section>

<section>
  <h2>Where domination actually shows up</h2>
  <div class="stack-lg">
    {sk.table(["Comparison", "Points", "Non-dominated", "Dominated"],
              [["Model thresholds alone", results["n_operating_points"],
                len(results["frontier"]), results["n_dominated"]],
               ["Model + baseline pooled", po["n_pooled"], po["n_on_front"],
                po["n_pooled"] - po["n_on_front"]]],
              numeric_cols=(1, 2, 3))}
    <div class="note">
      <h3>A single sweep is almost all frontier, and that is not a result</h3>
      <p>All {results['n_operating_points']} model settings sit on their own frontier
      here, and reporting that as a Pareto result would say nothing: while both
      objectives are strictly moving, no setting can dominate another.</p>
      <p>The stronger version of that claim &mdash; that a single-score sweep can
      <em>never</em> produce a dominated point &mdash; is false, and two successive
      tests were needed to pin down why. Domination appears whenever one objective
      <em>saturates</em>: once sensitivity plateaus at 1.0, further loosening adds false
      alerts and catches nothing more; and on a well-separated score, false alerts reach
      zero while sensitivity is still climbing, so a setting with 83% sensitivity and no
      false alerts is beaten by one with 99.8% sensitivity and no false alerts. The
      correct statement is that a dominated point on a single sweep always has a
      dominator tying it on one objective, which is now verified across a range of class
      separations. This run shows none because neither objective saturates within the
      swept range.</p>
      <p>The question with content is what happens when the model's settings and the
      single-vital alarm's settings compete. Pooled,
      <strong>{po['baseline_points_dominated']} of {po['baseline_points_total']}</strong>
      baseline settings are beaten outright by some model setting — and
      <strong>{po['model_points_dominated']}</strong> model settings are beaten by some
      baseline setting. The model dominates mostly, not everywhere, and the second
      number is the one a paper would usually omit.</p>
    </div>
  </div>
</section>

<section>
  <h2>Under a governance-set sensitivity floor</h2>
  <div class="stack-lg">
    {gov_chart}
    {gov_tbl}
    <div class="note">
      <h3>The benefit disappears exactly where governance sits</h3>
      <p>Alerting policies are specified the way clinical governance specifies them: fix
      the sensitivity that will be accepted, then minimise alerts inside that
      constraint. Read the reduction column down. At 70% required sensitivity the model
      removes {gov['0.70']['false_alert_reduction_pct']:.1f}% of false alerts; at 80%,
      {gov['0.80']['false_alert_reduction_pct']:.1f}%; at 90%,
      {gov['0.90']['false_alert_reduction_pct']:.1f}%; and at 95% it is
      {gov['0.95']['false_alert_reduction_pct']:.1f}% — no better than thresholding a
      single vital sign.</p>
      <p>That matters more than the headline number, because 95% is the region real ICU
      governance tends to demand. A model evaluated only at a convenient operating point
      would look useful and would not be, and this is the analysis that shows it.</p>
    </div>
  </div>
</section>

<section>
  <h2>Triage: who gets the {tri['capacity']} review slots?</h2>
  <div class="stack-lg">
    {tri_tbl}
    <div class="note">
      <h3>Risk and benefit are not the same quantity</h3>
      <p>Ranking by expected benefit catches
      <strong>fewer</strong> events than ranking by risk
      ({tri['policies']['top-k by expected benefit']['caught']} against
      {tri['policies']['top-k by risk']['caught']}) and captures
      <strong>more</strong> benefit
      ({tri['policies']['top-k by expected benefit']['benefit_capture']:.3f} against
      {tri['policies']['top-k by risk']['benefit_capture']:.3f}).</p>
      <p>That is the intended behaviour, not a defect. A patient certain to deteriorate
      but for whom review changes nothing should rank below a patient at moderate risk
      whose course review would change. Ranking by risk is the standard approach and it
      is wrong wherever the two diverge — which is also why "events caught" is the wrong
      thing to optimise a triage policy against.</p>
      <p>The fixed threshold does the same work as top-k by risk here while producing an
      alert count nobody chose. Under a hard capacity constraint, a threshold is the
      wrong instrument.</p>
    </div>
  </div>
</section>

<section>
  <h2>Reproduce it</h2>
  <div class="stack">
    <pre>cd icu-triage-optimization
pip install -r requirements.txt
python -m pytest tests/ -q
python -m src.demo</pre>
    <p>The demo imports the ICU early-warning model from repo 2.0 by file path, under
    its own module namespace, so both projects stay independently copyable. If repo 2.0
    is absent it falls back to synthetic scores and says so on this page.</p>
  </div>
</section>

<section>
  <h2>What this does not establish</h2>
  <div class="stack">
    <ul class="tight">
      <li>Scores come from a model trained on a synthetic cohort. Nothing here is a
      clinical result.</li>
      <li>Benefit-if-caught is generated, not measured. In reality it would come from
      an intervention-effect estimate, which is a harder problem than the ranking
      built on top of it.</li>
      <li>Alert burden is counted per observation. Real alarm fatigue depends on
      clustering, timing and who is in the room.</li>
      <li>The knee point is offered as a place to start a conversation, not an
      answer. Where a curve bends hardest is a property of the axes' scaling.</li>
      <li>Triage assumes review slots are interchangeable and reviews are equally
      effective whoever performs them.</li>
    </ul>
  </div>
</section>
"""
    return sk.build(ROOT, META, body, results)
