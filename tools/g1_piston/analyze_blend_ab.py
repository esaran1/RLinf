"""Did removing the chunk-boundary jitter improve the task, or only its appearance?

Reads the paired A/B evaluations -- same checkpoint, same frozen conditions, same
deterministic mode, ``BLEND_STEPS`` the only variable -- and recomputes every rate with
the frozen metric code. Both halves were measured in the same session, so the comparison
is paired rather than against stored numbers from an earlier protocol.

Because the conditions are identical and paired, per-condition changes are reported too:
a rate can stay flat while individual conditions flip in both directions, and that is a
different (weaker) result than a rate that moves because nothing regressed.

Usage:  python analyze_blend_ab.py
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
from rlinf.envs.isaaclab.tasks import g1_piston_metrics as M  # noqa: E402

S = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
     "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")
V = "/home/jren313/research/starvla_rl/RLinf/verified_results"


def label(row):
    if row["stages"].get("success"):
        return "SUCCESS"
    if M.is_carry(row):
        return "carry"
    if M.is_throw(row):
        return "throw"
    if row["stages"].get("grasp"):
        return "grasp"
    if row["stages"].get("reach"):
        return "reach"
    return "-"


def main():
    arms = defaultdict(dict)
    for p in sorted(glob.glob(f"{S}/runs/blend_ab/*_blend*_n*.json")):
        m = re.match(r"(.+)_blend(\d+)_n(\d+)\.json", os.path.basename(p))
        if not m:
            continue
        name, blend, n = m.group(1), int(m.group(2)), int(m.group(3))
        d = json.load(open(p))
        if d.get("_status") != "OK":
            continue
        rows = d.get("modes", {}).get("deterministic", {}).get("per_condition", [])
        if rows:
            arms[(name, n)][blend] = rows

    if not arms:
        print("no completed A/B arms yet")
        return 0

    print(f"metrics {M.METRICS_VERSION}\n")
    print("PAIRED A/B -- same checkpoint, same frozen conditions, blend is the only "
          "variable")
    print("%-18s %3s %6s %6s %6s %6s %6s %6s %8s" %
          ("arm", "n", "blend", "succ", "carry", "throw", "lift", "grasp", "return"))
    out = {}
    for (name, n), by_blend in sorted(arms.items()):
        for blend in sorted(by_blend):
            c = M.classify(by_blend[blend])
            print("%-18s %3d %6d %6.2f %6.2f %6.2f %6.2f %6.2f %8.3f" %
                  (name, n, blend, c["full_success_rate"], c["carry_rate"],
                   c["throw_rate"], c["lift_rate"], c["grasp_rate"], c["mean_return"]))
        if 0 in by_blend and 6 in by_blend:
            a, b = M.classify(by_blend[0]), M.classify(by_blend[6])
            deltas = {k: round(b[k] - a[k], 4) for k in
                      ("full_success_rate", "carry_rate", "throw_rate", "lift_rate",
                       "grasp_rate", "mean_return")}
            print("%-18s %3s %6s %6.2f %6.2f %6.2f %6.2f %6.2f %8.3f   <- delta" %
                  ("", "", "6-0", deltas["full_success_rate"], deltas["carry_rate"],
                   deltas["throw_rate"], deltas["lift_rate"], deltas["grasp_rate"],
                   deltas["mean_return"]))

            # Paired per-condition movement. A flat rate can hide churn.
            ra = {r["condition"]: r for r in by_blend[0]}
            rb = {r["condition"]: r for r in by_blend[6]}
            shared = sorted(set(ra) & set(rb))
            rank = {"-": 0, "reach": 1, "grasp": 2, "throw": 2, "carry": 3, "SUCCESS": 4}
            better = worse = same = 0
            moves = []
            for k in shared:
                la, lb = label(ra[k]), label(rb[k])
                if la == lb:
                    same += 1
                elif rank[lb] > rank[la]:
                    better += 1; moves.append((k, la, lb, "+"))
                else:
                    worse += 1; moves.append((k, la, lb, "-"))
            print("    per-condition: %d improved, %d regressed, %d unchanged (n=%d)"
                  % (better, worse, same, len(shared)))
            for k, la, lb, sign in moves[:12]:
                print("      cond %-3d %-8s -> %-8s  %s" % (k, la, lb, sign))
            n_succ_a = sum(1 for r in by_blend[0] if r["stages"].get("success"))
            n_succ_b = sum(1 for r in by_blend[6] if r["stages"].get("success"))
            print("    success CI95  blend0 %s   blend6 %s"
                  % (M.wilson95(n_succ_a, len(by_blend[0])),
                     M.wilson95(n_succ_b, len(by_blend[6]))))
            out[name] = {"n": n, "deltas": deltas, "improved": better,
                         "regressed": worse, "unchanged": same,
                         "blend0": {k: a[k] for k in a if k.endswith("_rate")
                                    or k.startswith("mean_")},
                         "blend6": {k: b[k] for k in b if k.endswith("_rate")
                                    or k.startswith("mean_")}}
        print()

    if out:
        os.makedirs(f"{V}/tables", exist_ok=True)
        dest = f"{V}/tables/blend_ab.json"
        json.dump({"metrics_version": M.METRICS_VERSION, "arms": out},
                  open(dest, "w"), indent=2)
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
