"""Quantify the deterministic-vs-stochastic execution gap from rendered rollouts.

The finding under test: the untrained SFT policy completes the task under its own
sampling noise but is inert under the deterministic mean action. Two independent sweeps
agreed on the RATE (0.16, 0.12) but succeeded on disjoint conditions, so the claim is
about a rate, and only repeated draws can test it.

This reads the verified rollout manifest, groups by (method, checkpoint, condition,
mode), and reports the per-condition and pooled outcome rates with Wilson intervals.
It uses only rollouts that passed verification.

Usage:  python analyze_stochastic_gap.py
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
from rlinf.envs.isaaclab.tasks import g1_piston_metrics as M  # noqa: E402

V = "/home/jren313/research/starvla_rl/verified_results"
VERIF = f"{V}/manifests/rollout_verification.json"


def main():
    if not os.path.exists(VERIF):
        print("no verification manifest; run verify_rollouts.py first")
        return 1
    data = json.load(open(VERIF))
    recs = [r for r in data["rollouts"] if r["verified"]]
    if not recs:
        print("no verified rollouts")
        return 1

    groups = defaultdict(list)
    for r in recs:
        groups[(r["method"], r["checkpoint_env_steps"], r["mode"])].append(r)

    print(f"metrics {M.METRICS_VERSION}   {len(recs)} verified rollouts\n")
    print("POOLED OUTCOME RATES BY MODE (verified rendered rollouts only)")
    print("%-10s %9s %-14s %4s %7s %7s %7s %7s  %s" %
          ("method", "steps", "mode", "n", "succ", "carry", "throw", "grasp",
           "succ CI95"))
    for k in sorted(groups, key=lambda k: (k[0], k[1] or 0, k[2])):
        rs = groups[k]
        n = len(rs)

        def cnt(field):
            return sum(1 for r in rs
                       if r["checks"]["recomputed_label"].get(field))
        print("%-10s %9s %-14s %4d %7.2f %7.2f %7.2f %7.2f  %s" %
              (k[0], k[1], k[2], n, cnt("success") / n, cnt("carry") / n,
               cnt("throw") / n, cnt("grasp") / n, M.wilson95(cnt("success"), n)))

    # Per-condition detail wherever a condition was drawn more than once: this is where a
    # rate claim either survives or dies.
    per = defaultdict(list)
    for r in recs:
        per[(r["method"], r["checkpoint_env_steps"], r["mode"],
             r["condition_index"])].append(r)
    multi = {k: v for k, v in per.items() if len(v) > 1}
    if multi:
        print("\nREPEATED DRAWS PER CONDITION (independent samples, same reset)")
        print("%-10s %9s %-13s %5s %5s  %s" %
              ("method", "steps", "mode", "cond", "n", "outcomes"))
        for k in sorted(multi, key=lambda k: (k[0], k[1] or 0, k[2], k[3])):
            rs = multi[k]
            outs = []
            for r in rs:
                lab = r["checks"]["recomputed_label"]
                outs.append("SUCC" if lab.get("success") else
                            "carry" if lab.get("carry") else
                            "throw" if lab.get("throw") else
                            "grasp" if lab.get("grasp") else
                            "reach" if lab.get("reach") else ".")
            print("%-10s %9s %-13s %5d %5d  %s" %
                  (k[0], k[1], k[2], k[3], len(rs), " ".join(outs)))

    # The headline comparison: same weights, same conditions, mode is the only difference.
    print("\nEXECUTION GAP (same checkpoint and conditions, mode is the only difference)")
    for method in sorted({r["method"] for r in recs}):
        for steps in sorted({r["checkpoint_env_steps"] for r in recs
                             if r["method"] == method}):
            det = [r for r in recs if r["method"] == method
                   and r["checkpoint_env_steps"] == steps
                   and r["mode"] == "deterministic"]
            sto = [r for r in recs if r["method"] == method
                   and r["checkpoint_env_steps"] == steps
                   and r["mode"] == "stochastic"]
            if not det or not sto:
                continue
            shared = ({r["condition_index"] for r in det}
                      & {r["condition_index"] for r in sto})
            if not shared:
                continue
            det = [r for r in det if r["condition_index"] in shared]
            sto = [r for r in sto if r["condition_index"] in shared]

            def rate(rs, f):
                return (sum(1 for r in rs
                            if r["checks"]["recomputed_label"].get(f)) / len(rs))
            print(f"  {method} @ {steps:,} on {len(shared)} shared conditions:")
            for f in ("success", "carry", "grasp"):
                print("    %-8s deterministic %.2f (n=%d)   stochastic %.2f (n=%d)"
                      % (f, rate(det, f), len(det), rate(sto, f), len(sto)))

    dest = f"{V}/tables/stochastic_gap.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump({"metrics_version": M.METRICS_VERSION,
               "groups": {f"{k[0]}|{k[1]}|{k[2]}": {
                   "n": len(v),
                   "success": sum(1 for r in v
                                  if r["checks"]["recomputed_label"].get("success")),
                   "carry": sum(1 for r in v
                                if r["checks"]["recomputed_label"].get("carry")),
                   "videos": [r["video"] for r in v]} for k, v in groups.items()}},
              open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
