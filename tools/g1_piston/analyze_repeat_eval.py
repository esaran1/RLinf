"""The run-to-run noise floor: how much does one checkpoint's score move between runs?

Two identical-protocol evaluations of ``rlpd_ckpt_step415140`` produced lifting-condition
sets with ZERO overlap at lift rates 0.20 and 0.32, while grasp was 25/25 in both. This
aggregates every available repeat of that checkpoint and reports, per metric:

* the spread across runs -- the honest error bar for any rate in this study;
* how it compares with the binomial (Wilson) interval, which assumes a fixed
  per-condition success probability and therefore understates the true uncertainty;
* the per-condition reproducibility, as mean pairwise Jaccard overlap of the sets of
  conditions that achieved each stage.

The last number is the important one. A metric whose *rate* is stable but whose
*condition set* is not is measuring a property of the policy in aggregate only -- matched
per-condition comparison of two policies is invalid for it.

Any algorithmic difference smaller than the spread reported here is unresolvable by a
single evaluation, however many conditions that evaluation contains, because the variance
is between runs rather than between conditions.

Usage:  python analyze_repeat_eval.py
"""
import glob
import itertools
import json
import os
import statistics as st
import sys

sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
from rlinf.envs.isaaclab.tasks import g1_piston_metrics as M  # noqa: E402

S = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
     "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")
V = "/home/jren313/research/starvla_rl/RLinf/verified_results"

#: Every identical-protocol deterministic evaluation of rlpd_ckpt_step415140 on the
#: frozen 25-condition prefix, whatever produced it. BLEND_STEPS must be 0: a blended run
#: is a different execution mode and would confound the noise estimate.
SOURCES = [
    (f"{S}/runs/n50_rlpd_415140.json", "n50 eval (first 25)", 25),
    (f"{S}/runs/blend_ab/rlpd_s1_415140_blend0_n25.json", "blend A/B baseline", None),
]


def load_rows(path, limit=None):
    d = json.load(open(path))
    if d.get("_status") != "OK":
        return None
    if d.get("blend_steps", 0):
        return None
    rows = d.get("modes", {}).get("deterministic", {}).get("per_condition", [])
    if not rows:
        return None
    return rows[:limit] if limit else rows


def main():
    runs = []
    for path, label, limit in SOURCES:
        if os.path.exists(path):
            r = load_rows(path, limit)
            if r:
                runs.append((label, r))
    for path in sorted(glob.glob(f"{S}/runs/repeat_eval/rep*_n*.json")):
        r = load_rows(path)
        if r:
            runs.append((os.path.basename(path).replace(".json", ""), r))

    if len(runs) < 2:
        print(f"{len(runs)} run(s) available; need at least 2")
        return 0

    n = min(len(r) for _, r in runs)
    runs = [(lbl, r[:n]) for lbl, r in runs]
    print(f"metrics {M.METRICS_VERSION}")
    print(f"ONE checkpoint, identical protocol, {len(runs)} runs, n={n} frozen "
          f"conditions each\n")

    keys = ["grasp_rate", "lift_rate", "carry_rate", "throw_rate",
            "full_success_rate", "mean_return"]
    print("%-24s %s" % ("run", "  ".join("%9s" % k.replace("_rate", "")
                                         for k in keys)))
    vals = {k: [] for k in keys}
    for lbl, rows in runs:
        c = M.classify(rows)
        for k in keys:
            vals[k].append(c[k])
        print("%-24s %s" % (lbl[:24],
                            "  ".join("%9.3f" % c[k] for k in keys)))

    print("\n%-24s %s" % ("SPREAD (max-min)",
                          "  ".join("%9.3f" % (max(vals[k]) - min(vals[k]))
                                    for k in keys)))
    print("%-24s %s" % ("mean", "  ".join("%9.3f" % st.mean(vals[k]) for k in keys)))
    if len(runs) > 2:
        print("%-24s %s" % ("stdev", "  ".join("%9.3f" % st.stdev(vals[k])
                                               for k in keys)))

    # Per-condition reproducibility: does the SAME condition succeed every run?
    print("\nPER-CONDITION REPRODUCIBILITY (mean pairwise Jaccard of the condition sets)")
    repro = {}
    for stage in ("grasp", "lift", "success"):
        sets = [{r["condition"] for r in rows if r["stages"].get(stage)}
                for _, rows in runs]
        js = []
        for a, b in itertools.combinations(sets, 2):
            if a or b:
                js.append(len(a & b) / len(a | b))
        carry_sets = None
        repro[stage] = round(st.mean(js), 3) if js else None
        sizes = [len(s) for s in sets]
        print("  %-8s sets sized %-22s  mean Jaccard %s"
              % (stage, str(sizes), repro[stage] if js else "n/a (never fired)"))
    carry_sets = [{r["condition"] for r in rows if M.is_carry(r)} for _, rows in runs]
    js = [len(a & b) / len(a | b) for a, b in itertools.combinations(carry_sets, 2)
          if (a or b)]
    repro["carry"] = round(st.mean(js), 3) if js else None
    print("  %-8s sets sized %-22s  mean Jaccard %s"
          % ("carry", str([len(s) for s in carry_sets]),
             repro["carry"] if js else "n/a"))

    # The headline: run-to-run spread against the binomial interval a single run reports.
    print("\nRUN-TO-RUN SPREAD vs THE BINOMIAL INTERVAL A SINGLE RUN WOULD REPORT")
    for k, stage in (("carry_rate", "carry"), ("lift_rate", "lift"),
                     ("grasp_rate", "grasp")):
        mean = st.mean(vals[k])
        spread = max(vals[k]) - min(vals[k])
        ci = M.wilson95(round(mean * n), n)
        print("  %-6s mean %.3f   run-to-run spread %.3f   Wilson95 width %.3f   %s"
              % (stage, mean, spread, ci[1] - ci[0],
                 "SPREAD EXCEEDS THE CI" if spread > (ci[1] - ci[0]) else
                 "within the CI"))

    print("\nINTERPRETATION")
    gr = max(vals["grasp_rate"]) - min(vals["grasp_rate"])
    cr = max(vals["carry_rate"]) - min(vals["carry_rate"])
    print("  grasp spread %.3f, carry spread %.3f" % (gr, cr))
    print("  Any algorithmic difference smaller than the carry spread is unresolvable")
    print("  by a single evaluation, because the variance is BETWEEN RUNS, not between")
    print("  conditions -- adding conditions does not shrink it.")

    out = {"metrics_version": M.METRICS_VERSION, "n_runs": len(runs),
           "n_conditions": n,
           "rates": {k: vals[k] for k in keys},
           "spread": {k: round(max(vals[k]) - min(vals[k]), 4) for k in keys},
           "mean": {k: round(st.mean(vals[k]), 4) for k in keys},
           "per_condition_jaccard": repro,
           "runs": [lbl for lbl, _ in runs]}
    os.makedirs(f"{V}/tables", exist_ok=True)
    json.dump(out, open(f"{V}/tables/repeat_eval_noise_floor.json", "w"), indent=2)
    print(f"\nwrote {V}/tables/repeat_eval_noise_floor.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
