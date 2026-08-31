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
