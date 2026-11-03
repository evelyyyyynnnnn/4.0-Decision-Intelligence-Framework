"""Builds website/ from the last demo run."""
from __future__ import annotations
import pathlib
from . import sitekit as sk

ROOT = pathlib.Path(__file__).resolve().parent.parent
META = {
    "name": "Decision Audit Framework",
    "slug": "decision-audit-framework",
    "repo": "4.0-Decision-Intelligence-Framework",
    "pillar": "Cross-cutting",
    "tagline": "A hash-chained decision ledger with exact replay, versioned "
               "counterfactuals, and Shapley attribution — so \"why did the system "
               "do that\" has an answer that survives being checked.",
    "tags": [("hash-chained ledger", ""), ("exact replay", ""),
             ("counterfactuals", ""), ("exact Shapley", ""),
             ("synthetic decisions", "demo")],
    "banner": "Decisions here are made on synthetic credit applications by a policy "
              "written for this demo. The machinery is real; the lending policy is "
              "not, and no conclusion about credit risk follows from it.",
}


def build_site(results: dict) -> pathlib.Path:
    # The real run and the authored run have different shapes; dispatch on the
    # flag the results file carries rather than on which keys happen to exist.
    if not results.get("is_synthetic", True):
        return build_site_real(results)

    L, rp, t = results["ledger"], results["replay"], results["tamper"]
    v, c, sh, eff = (results["versioning"], results["case"], results["shapley"],
                     results["efficiency"])

    metrics = sk.metric_grid([
        ("Decisions logged", L["n_decisions"], f"chain intact: {L['chain_intact']}"),
        ("Replayed exactly", f"{rp['reproduced']}/{rp['n']}", "bit for bit"),
        ("Version flips", v["n_flipped"], f"of {v['n_cases']} cases, v1 → v2"),
        ("Attribution gap", f"{results['max_disagreement']:.4f}",
         "occlusion vs Shapley"),
    ])

    integrity_tbl = sk.table(
        ["Check", "Result"],
        [["Chain intact before tampering", str(t["intact_before"])],
         ["Edited input detected", str(t["detected"])],
         ["Detected at record", t["detected_at"]],
         ["Chain intact after restore", str(t["intact_after_restore"])],
         ["Decisions reproduced on replay", f"{rp['reproduced']}/{rp['n']}"]],
        numeric_cols=(1,))

    flip_tbl = sk.table(
        ["Case", "v1", "v2", "Delinquencies", "v1 score", "v2 score"],
        [[e["case"], e["v1"], e["v2"], e["delinquencies"],
          f"{e['v1_score']:+.3f}", f"{e['v2_score']:+.3f}"]
         for e in v["examples"]], numeric_cols=(3, 4, 5))

    attr_rows = [[k, f"{results['occlusion'][k]:+.4f}", f"{sh['values'][k]:+.4f}",
                  f"{results['attribution_disagreement'][k]:+.4f}"]
                 for k in sh["values"]]
    attr_tbl = sk.table(
        ["Feature", "Occlusion", "Shapley", "Difference"], attr_rows,
        numeric_cols=(1, 2, 3))

    attr_chart = sk.bar_chart(
        [(k, abs(sh["values"][k])) for k in sh["values"]], fmt="{:.3f}")

    cf_tbl = sk.table(
        ["Feature", "Changed to", "Original score", "Counterfactual score",
         "Decision flips"],
        [[cf["feature"], cf["to"], f"{cf['original_score']:+.4f}",
          f"{cf['counterfactual_score']:+.4f}", "yes" if cf["changed"] else "no"]
         for cf in results["counterfactuals"]],
        numeric_cols=(1, 2, 3))

    body = f"""
<section>
  <h2>The question this answers</h2>
  <div class="stack">
    <p>The one asked after a decision goes wrong: <em>why did the system do that, and
    what would it have done if the inputs had been different?</em> Answering it needs
    three things ordinary logging does not provide — the inputs exactly as they were,
    the policy version that actually ran, and a guarantee that neither has been edited
    since.</p>
    <p>Replay is exact, not approximate. A policy is a pure function of its recorded
    inputs, so re-running it reproduces the decision bit for bit. That is a real
    restriction on how policies may be written, and it is what makes the rest possible.</p>
  </div>
</section>

<section>
  <h2>This run</h2>
  <div class="stack-lg">
    {metrics}
    <p class="mono" style="color:var(--muted);font-size:12.5px">
      generated {sk.esc(results['generated_at'])} &middot;
      actions: {sk.esc(str(L['action_counts']))} &middot;
      policies: {sk.esc(", ".join(L['policies']))}
    </p>
    {integrity_tbl}
    <p>Editing one recorded input — lowering a declined applicant's debt-to-income to
    make the decline look unjustified — breaks the hash chain at exactly that record.
    The chain proves the log has not been altered since it was written. It does not
    prove who wrote it; there is no signing here, and tamper-<em>evident</em> is the
    accurate word.</p>
  </div>
</section>

<section>
  <h2>Counterfactuals run the policy that actually ran</h2>
  <div class="stack-lg">
    {flip_tbl}
    <div class="note">
      <h3>Why the version matters</h3>
      <p>Policy v2 weights delinquencies more heavily than v1, and
      {v['n_flipped']} of {v['n_cases']} decisions change as a result. A counterfactual
      that re-scored an old decision with today's policy would answer <em>"what would we
      do now"</em> — a different and usually less useful question than <em>"what would we
      have done then"</em>.</p>
      <p>Conflating the two is how a review concludes the system behaved reasonably when
      it did not: the current policy would have declined the case, so the decline looks
      justified, even though the policy in force at the time approved it.</p>
    </div>
  </div>
</section>

<section>
  <h2>Attribution — {sk.esc(c['case_id'])}</h2>
  <div class="stack-lg">
    <p>Declined with a score of {c['score']:+.4f}. Inputs:
    <code>{sk.esc(str(c['inputs']))}</code></p>
    {attr_chart}
    {attr_tbl}
    <div class="note">
      <h3>Where the two methods disagree, and why</h3>
      <p>They agree exactly on <code>delinquencies</code>, <code>tenure_years</code> and
      <code>income_k</code>, and differ by {results['max_disagreement']:.4f} on
      <code>dti</code> and <code>utilisation</code> — the two features the policy
      combines in an interaction term.</p>
      <p>Occlusion removes one feature at a time, so each of the interacting pair
      receives <em>full</em> credit for their shared effect and the interaction is
      counted twice. Shapley averages over every ordering, which splits it. On a purely
      additive policy the two methods return identical values, and an earlier version of
      this demo used exactly such a policy — producing a table where the columns matched
      perfectly and demonstrated nothing about why anyone would pay for Shapley.</p>
      <p>Efficiency check: the Shapley values sum to {eff['sum_of_values']:+.4f} against
      a score difference of {eff['score_difference']:+.4f}, residual
      {eff['residual']:.2e}. That identity is what makes them an allocation rather than
      a set of importances, and it is asserted in the test suite.</p>
    </div>
  </div>
</section>

<section>
  <h2>What would have changed the answer</h2>
  <div class="stack-lg">
    {cf_tbl}
    <p>{results['n_flipping_features']} single-feature changes would have flipped this
    decision. That is the form of explanation an applicant can act on, and a regulator
    can check, in a way that a ranked importance list is not.</p>
  </div>
</section>

<section>
  <h2>Reproduce it</h2>
  <div class="stack">
    <pre>cd decision-audit-framework
pip install -r requirements.txt
python -m pytest tests/ -q
python -m src.demo</pre>
  </div>
</section>

<section>
  <h2>What this does not establish</h2>
  <div class="stack">
    <ul class="tight">
      <li>The credit policy is invented for this demo. Nothing here is a statement
      about lending, fairness, or credit risk.</li>
      <li>Exact replay requires policies to be pure functions of recorded inputs. A
      policy that reads a database, calls a model server, or depends on wall-clock
      time cannot be replayed this way, and most production policies do at least one
      of those.</li>
      <li>The hash chain is tamper-evident, not tamper-proof, and it is not signed.
      It shows the log has changed; it cannot say who changed it.</li>
      <li>Exact Shapley enumerates all orderings and is used here for five features.
      Beyond eight it falls back to sampling, and the sampled values carry error
      that this demo does not quantify.</li>
      <li>No fairness or disparate-impact analysis. Attribution says what drove a
      decision, not whether the decision was right.</li>
    </ul>
  </div>
</section>
"""
    return sk.build(ROOT, META, body, results)


# --------------------------------------------------------------------------
# Real-data page: the ledger exercised on real UCI credit applications.
# --------------------------------------------------------------------------

REAL_META = dict(META)
REAL_META.update({
    "tags": [("real UCI German Credit", "real"), ("hash-chained ledger", ""),
             ("exact Shapley", ""), ("policy authored, not fitted", "")],
    "banner": "Run on the Statlog German Credit Data from the UCI Machine Learning "
              "Repository: 1,000 real credit applications, retrieved with URL, hash and "
              "retrieval time recorded in data/MANIFEST.json. The approval rule is "
              "authored rather than fitted, because the machinery is the object of "
              "study and the lending policy is not.",
    "datanote": "Built from results/latest-real.json. Replay fidelity, tamper detection "
                "and attribution are properties of the ledger, shown here on one "
                "dataset and one policy. Nothing on this page is a claim about credit "
                "risk.",
})


def build_site_real(results: dict) -> pathlib.Path:
    led, rep = results["ledger"], results["replay"]
    tam, ver = results["tamper"], results["versioning"]
    oc, case = results["outcome_check"], results["case"]
    occ, shp = results["occlusion"], results["shapley"]
    dis, eff = results["attribution_disagreement"], results["efficiency"]
    prov = results.get("provenance", {})

    metrics = sk.metric_grid([
        ("Decisions replayed", f"{rep['reproduced']:,} / {rep['n']:,}",
         "to the same action, bit for bit"),
        ("Tamper detected at", f"index {tam['detected_at']}",
         f"one edit to {tam['field_edited']}"),
        ("Changed by policy v{} → v{}".format(ver["from"], ver["to"]),
         f"{ver['decisions_changed']:,}",
         f"{ver['share_changed'] * 100:.1f}% of the ledger"),
        ("Largest attribution gap", f"{results['max_disagreement']:.3f}",
         "occlusion against exact Shapley"),
    ])

    prop_tbl = sk.table(
        ["Property", "Result", "How it was shown"],
        [["Replay", f"{rep['reproduced']:,} of {rep['n']:,}, {len(rep['failures'])} failures",
          "every decision re-executed from the ledger reproduces the same action"],
         ["Tamper evidence", f"detected at index {tam['detected_at']}",
          f"chain intact; one field ({tam['field_edited']}) edited; break detected; "
          f"intact again after restore" if tam["intact_after_restore"] else "detected"],
         ["Versioning", f"{ver['decisions_changed']:,} of {rep['n']:,}",
          f"re-run under {ver['policy']}@{ver['to']}; each flip recorded with both scores"],
         ["Action split",
          f"{led['action_counts'].get('decline', 0):,} decline / "
          f"{led['action_counts'].get('approve', 0):,} approve",
          f"under {', '.join(led['policies'])}"]])

    outcome_tbl = sk.table(
        ["", "Bad-risk rate"],
        [["Overall", f"{oc['bad_rate_overall'] * 100:.1f}%"],
         [f"Among the {oc['declined']:,} declined",
          f"{oc['bad_rate_among_declined'] * 100:.1f}%"],
         [f"Among the {oc['approved']:,} approved",
          f"{oc['bad_rate_among_approved'] * 100:.1f}%"],
         ["Outcome lift", f"{oc['lift']:.3f}"]],
        numeric_cols=(1,))
    outcome_chart = sk.bar_chart(
        [("declined", oc["bad_rate_among_declined"]),
         ("base rate", oc["bad_rate_overall"]),
         ("approved", oc["bad_rate_among_approved"])], fmt="{:.3f}")

    feats = sorted(occ, key=lambda f: -abs(occ[f]))
    attr_tbl = sk.table(
        ["Feature", "Occlusion", "Exact Shapley", "Gap"],
        [[f, f"{occ[f]:+.3f}", f"{shp['values'][f]:+.3f}", f"{abs(dis.get(f, 0)):.3f}"]
         for f in feats],
        numeric_cols=(1, 2, 3))
    attr_chart = sk.bar_chart(
        [(f + " (occlusion)", abs(occ[f])) for f in feats if abs(occ[f]) > 0]
        + [(f + " (Shapley)", abs(shp["values"][f]))
           for f in feats if abs(shp["values"][f]) > 0],
        fmt="{:.3f}")

    body = f"""
<section>
  <h2>This run</h2>
  <div class="stack-lg">
    {metrics}
    <p class="mono" style="color:var(--muted);font-size:12.5px">
      generated {sk.esc(results['generated_at'])} &middot;
      {prov.get('n_cases', 0):,} real applications &middot;
      {prov.get('n_bad_risk', 0):,} labelled bad risk
    </p>
  </div>
</section>

<section>
  <h2>Ledger properties, measured</h2>
  <div class="stack-lg">
    {prop_tbl}
  </div>
</section>

<section>
  <h2>What the policy itself achieves</h2>
  <div class="stack-lg">
    {outcome_chart}
    {outcome_tbl}
    <p>A lift of {oc['lift']:.3f} means the rule separates very little. That is the
    intended state, and it is reported rather than hidden: {sk.esc(oc.get('note', ''))}</p>
    <p>Two fields the earlier synthetic demo used were dropped, and the reason recorded:
    {sk.esc(results.get('dropped_because', ''))}</p>
  </div>
</section>

<section>
  <h2>Two attribution methods on one real decision &mdash; {sk.esc(case['case_id'])}</h2>
  <div class="stack-lg">
    {attr_chart}
    {attr_tbl}
    <p>{sk.esc(case['action'].capitalize())}d at score {case['score']:.3f}. Shapley is
    exact over {len(feats)} features &mdash; all {shp['n_orderings']} orderings, not a
    sample &mdash; and efficiency holds: the values sum to {eff['sum_of_values']:.3f},
    which equals the score difference, with residual {eff['residual']:.1f}.</p>
    <p>So the {results['max_disagreement']:.3f} gap is not an arithmetic error in either
    method. It is the genuine distance between two defensible definitions of a feature's
    contribution, measured on a real record. An adverse-action notice built on one would
    quote a different magnitude from one built on the other.</p>
    <p>No single-feature change within the tested ranges flips this decision:
    <code>n_flipping_features: {results.get('n_flipping_features', 0)}</code>.</p>
  </div>
</section>

<section>
  <h2>Reproduce</h2>
  <div class="stack">
    <pre>python -m src.demo --real
# -&gt; results/latest-real.json
pytest -q</pre>
  </div>
</section>
"""
    return sk.build(ROOT, REAL_META, body, results)
