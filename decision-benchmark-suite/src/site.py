"""Builds website/ from the last demo run."""
from __future__ import annotations
import pathlib
from . import sitekit as sk

ROOT = pathlib.Path(__file__).resolve().parent.parent
META = {
    "name": "Decision Benchmark Suite",
    "slug": "decision-benchmark-suite",
    "repo": "4.0-Decision-Intelligence-Framework",
    "pillar": "Cross-cutting",
    "tagline": "Regret against an exact oracle, calibration of stated confidence, and "
               "degradation under named distribution shifts — three axes on which a "
               "decision policy can be wrong in different ways.",
    "tags": [("exact oracles", ""), ("regret distribution", ""),
             ("named shifts", ""), ("closed-form tasks", "demo")],
    "banner": "Tasks are closed-form so the oracle is exact rather than searched. That "
              "is what makes regret computable instead of estimated — but it also "
              "means these are constructed problems, and a policy's score here says "
              "nothing about a messy real one.",
}


def _task_section(title: str, t: dict, floor: float | None = None) -> str:
    rows = []
    for name, v in t["policies"].items():
        rb, rg = v["robustness"], v["regret"]
        excess = f"{rg['mean'] - floor:+.3f}" if floor is not None else "—"
        rows.append([name, f"{rg['mean']:.4f}", excess, f"{rg['p90']:.3f}",
                     f"{rg['frac_optimal']:.1%}", f"{v['calibration']['ece']:.4f}",
                     f"{rb['degradation_per_unit_shift']:.4f}",
                     f"{rb['relative_worst']:.2f}×" if rb["relative_worst"] else "—"])
    tbl = sk.table(
        ["Policy", "Mean regret", "Excess over floor", "p90 regret", "% optimal",
         "ECE", "Degradation / unit shift", "Worst / baseline"],
        rows, numeric_cols=(1, 2, 3, 4, 5, 6, 7))

    first = next(iter(t["policies"]))
    shift_series = []
    for name, v in t["policies"].items():
        pts = [(float(s), val) for s, val in v["robustness"]["shifts"].items()]
        shift_series.append((name[:22], sorted(pts)))
    chart = sk.line_chart(shift_series[:4], xlabel="distribution shift",
                          ylabel="mean regret")
    return f"""
<section>
  <h2>{title}</h2>
  <div class="stack-lg">
    {tbl}
    {chart}
  </div>
</section>
"""


def build_site(results: dict) -> pathlib.Path:
    o, g = results["oracle"], results["oracle_gap"]
    nv, tr = results["newsvendor"], results["triage"]
    floor = g["irreducible_regret"]

    best_nv = min(nv["policies"], key=lambda k: nv["policies"][k]["regret"]["mean"])
    best_tr = min(tr["policies"], key=lambda k: tr["policies"][k]["regret"]["mean"])

    metrics = sk.metric_grid([
        ("Tasks", 2, "both with exact oracles"),
        ("Policies scored", len(nv["policies"]) + len(tr["policies"]),
         "across both tasks"),
        ("Shifts per task", len(nv["shifts"]), "named, not generic"),
        ("Best newsvendor policy", f"{nv['policies'][best_nv]['regret']['mean']:.1f}",
         best_nv),
    ])

    oracle_tbl = sk.table(
        ["Reference", "Mean cost", "What it means"],
        [["Clairvoyant oracle", f"{g['clairvoyant_cost']:.2f}",
          "Picks the best order AFTER seeing realised demand. Unreachable."],
         ["Best fixed action", f"{g['best_fixed_action_cost']:.2f}",
          "The achievable floor — the best single order over the whole sample."],
         ["Distributional optimum", f"{g['distributional_optimum_cost']:.2f}",
          "Critical-fractile order given the TRUE distribution."]],
        numeric_cols=(1,))

    body = f"""
<section>
  <h2>Three ways a decision policy is wrong</h2>
  <div class="stack">
    <p><strong>Regret</strong>: how much payoff it gave up against the best available
    decision. <strong>Calibration</strong>: whether its stated confidence means
    anything, which is what decides when a human should be asked.
    <strong>Robustness</strong>: how fast it degrades as the world moves away from what
    it was fitted on.</p>
    <p>A policy can be excellent on one and useless on another, and the table below has
    a case of exactly that.</p>
  </div>
</section>

<section>
  <h2>This run</h2>
  <div class="stack-lg">
    {metrics}
    <p class="mono" style="color:var(--muted);font-size:12.5px">
      generated {sk.esc(results['generated_at'])} &middot;
      newsvendor critical fractile {o['newsvendor_critical_fractile']} &middot;
      optimal order {o['newsvendor_optimal_order']:.1f} &middot;
      triage treats above P={o['triage_treat_threshold_prob']}
    </p>
  </div>
</section>

<section>
  <h2>Which oracle you measure against decides what the number means</h2>
  <div class="stack-lg">
    {oracle_tbl}
    <div class="note">
      <h3>The clairvoyant gap is not a performance gap</h3>
      <p>Regret in the newsvendor table is measured against an oracle that chooses after
      seeing realised demand. No policy can approach it: the gap is
      <strong>{floor:.1f}</strong> and it is dominated by the variance of demand itself,
      not by anything a decision-maker could do better.</p>
      <p>So the column that carries information is <em>excess over floor</em>. The
      empirical-fractile policy scores
      {nv['policies']['empirical fractile']['regret']['mean'] - floor:+.2f} — it is
      sitting on the achievable optimum with nothing left to learn. Ordering the mean
      scores {nv['policies']['order the mean']['regret']['mean'] - floor:+.2f}, and that
      number is the real cost of ignoring the cost asymmetry.</p>
      <p>Reporting the raw regret alone would make a perfect policy look like it had a
      44-point problem.</p>
    </div>
  </div>
</section>
{_task_section("Newsvendor — ordering under asymmetric cost", nv, floor)}
<section>
  <div class="stack">
    <div class="note">
      <h3>Two policies scoring identically is a sanity check, not a coincidence</h3>
      <p>"Order the mean" and "always order 50" produce the same regret because mean
      demand in this task is 50. If those two rows ever diverged, the fitting code would
      be doing something other than what it claims.</p>
    </div>
  </div>
</section>
{_task_section("Triage — treat or wait on a noisy signal", tr)}
<section>
  <div class="stack">
    <div class="note">
      <h3>The Bayes policy wins on regret and loses badly on calibration</h3>
      <p>The Bayes-optimal policy and the fitted threshold are separated by
      {abs(tr['policies']['Bayes-optimal given the signal']['regret']['mean'] - tr['policies']['fitted threshold']['regret']['mean']):.4f}
      in mean regret — effectively tied, which is the expected result: with a
      one-dimensional signal and a monotone posterior, a well-fitted threshold
      <em>is</em> the Bayes rule. Nothing is gained from the generative model here, and
      that is worth knowing before building one.</p>
      <p>They separate sharply on calibration. The threshold policy's ECE is
      {tr['policies']['fitted threshold']['calibration']['ece']:.4f}; the Bayes policy's
      is {tr['policies']['Bayes-optimal given the signal']['calibration']['ece']:.4f},
      an order of magnitude worse. Its confidence is a well-calibrated estimate of
      <em>the patient's risk</em>, and it is being scored here as a claim about
      <em>its own correctness</em>. Those are different quantities, and conflating them
      is how a system ends up escalating the wrong cases to a human.</p>
      <p>On robustness the ordering flips back: the Bayes policy degrades at
      {tr['policies']['Bayes-optimal given the signal']['robustness']['degradation_per_unit_shift']:.4f}
      per unit of shift against
      {tr['policies']['fitted threshold']['robustness']['degradation_per_unit_shift']:.4f}
      for the threshold, because a threshold fitted to one distribution is fitted to
      that distribution. Three axes, three different winners.</p>
    </div>
  </div>
</section>

<section>
  <h2>Reproduce it</h2>
  <div class="stack">
    <pre>cd decision-benchmark-suite
pip install -r requirements.txt
python -m pytest tests/ -q
python -m src.demo</pre>
    <p>To benchmark your own policy, implement <code>fit(sample)</code>,
    <code>decide(sample) -&gt; action indices</code> and
    <code>confidence(sample) -&gt; [0,1]</code>, then add it to the list in
    <code>src/demo.py</code>.</p>
  </div>
</section>

<section>
  <h2>What this does not establish</h2>
  <div class="stack">
    <ul class="tight">
      <li>Two constructed tasks with closed-form solutions. That is what makes exact
      regret possible and also what makes the results unrepresentative of real
      decision problems.</li>
      <li>Robustness is measured against the shifts named in the table. A policy robust
      to a mean shift can be fragile to a variance or correlation shift, and neither is
      tested here.</li>
      <li>The Bayes policy uses the true generative model. Real systems estimate it,
      and that estimation error is not represented.</li>
      <li>No sequential or feedback effects: every decision is independent, and no
      decision changes the distribution of later ones.</li>
      <li>The suite has never been run against the other projects in this repository,
      which is the obvious next step and has not been done.</li>
    </ul>
  </div>
</section>
"""
    return sk.build(ROOT, META, body, results)
