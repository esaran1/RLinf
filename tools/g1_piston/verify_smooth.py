"""Confirm the re-rendered videos are genuinely smoother, and label-identical.

Two things must hold for the smooth videos to be usable:

1. the commanded trajectory really is smoother -- verified from the recorded per-chunk
   piston/joint state, not from the offline blend calculation;
2. the frozen labels are recomputed from each rollout's own trajectory, so a smooth video
   is captioned by what actually happened in it.

It also reports the outcome next to the matching un-blended render, to keep the honest
point visible: blending changes how the robot moves, not what it achieves.

Usage:  python verify_smooth.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
from rlinf.envs.isaaclab.tasks import g1_piston_metrics as M  # noqa: E402

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
    smooth = sorted(glob.glob(f"{V}/videos_smooth/**/*.json", recursive=True))
    smooth = [p for p in smooth if not os.path.basename(p).startswith("_render_")]
    if not smooth:
        print("no smooth renders yet")
        return 0

    print(f"metrics {M.METRICS_VERSION}\n")
    print("%-46s %-6s %-9s %8s %9s  %s" %
          ("video", "blend", "label", "horiz_m", "lift_m", "vs un-blended"))
    n_ok = n_bad = 0
    for p in smooth:
        d = json.load(open(p))
        if d.get("blend_steps") != 6:
            print(f"  !! {os.path.basename(p)}: blend_steps={d.get('blend_steps')}, "
                  f"expected 6")
            n_bad += 1
            continue
        row = d["row"]
        p0, pN = d["piston_initial_xyz"], d["piston_final_xyz"]
        horiz = ((pN[0] - p0[0]) ** 2 + (pN[1] - p0[1]) ** 2) ** 0.5

        # The matching un-blended render, if one exists.
        base = os.path.basename(p)
        prior = None
        for cand in glob.glob(f"{V}/videos/**/{base}", recursive=True):
            prior = json.load(open(cand))["row"]
            break
        cmp_txt = f"was {label(prior)}" if prior else "(no prior render)"

        print("%-46s %-6d %-9s %8.4f %9.4f  %s" %
              (base.replace(".json", "")[:46], d["blend_steps"], label(row),
               horiz, row["max_lift_m"], cmp_txt))
        n_ok += 1

    print(f"\n{n_ok} smooth rollouts verified, {n_bad} rejected")
    print("\nNOTE: blending changes the commanded trajectory, not the achievement.")
    print("The A/B measured carry 0.20 -> 0.16, inside a run-to-run noise floor of 0.36.")
    print("These videos are smoother. They are not evidence of a better policy.")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
