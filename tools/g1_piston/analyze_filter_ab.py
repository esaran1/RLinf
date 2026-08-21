"""Judge the demo-envelope filter against the characterised baseline distribution.

The unfiltered baseline for rlpd_ckpt_step415140 is EIGHT identical-protocol runs
(carry 0.12/0.20/0.00/0.36/0.20/0.20/0.12/0.36, grasp 199/200), so a single filtered
run can be placed against a real distribution instead of a single noisy point:

* grasp: baseline spread is 0.04 -- a filtered grasp below 0.92 is a real regression,
  not noise (filter lag breaking grasp timing).
* carry: baseline range [0.00, 0.36] -- only a filtered carry ABOVE 0.36 is evidence of
  improvement; anything inside the range is indistinguishable from noise.
* SFT stochastic success: baseline 0.16 (two independent draws: 4/25 and 4/25) -- the
  filter must not destroy the only behaviour in the study that completes the task.

Usage:  python analyze_filter_ab.py
"""
import json
import os
import sys

sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
from rlinf.envs.isaaclab.tasks import g1_piston_metrics as M  # noqa: E402

S = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
     "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")
V = "/home/jren313/research/starvla_rl/RLinf/verified_results"

BASE_CARRY = [0.12, 0.20, 0.00, 0.36, 0.20, 0.20, 0.12, 0.36]
BASE_GRASP_MIN = 0.92   # worst single baseline run was 0.96; allow one extra miss


def load(path, mode):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    rows = d.get("modes", {}).get(mode, {}).get("per_condition", [])
    if d.get("_status") not in ("OK", "RUNNING") or not rows:
        return None
    if d.get("filter_hz", 0) <= 0:
        print(f"  !! {os.path.basename(path)} has filter_hz={d.get('filter_hz')}")
        return None
    return rows, d.get("_status")


def main():
    print(f"metrics {M.METRICS_VERSION}\n")
    got = load(f"{S}/runs/filter_ab/rlpd_415k_filter12_n25.json", "deterministic")
    if got:
        rows, status = got
        c = M.classify(rows)
        print(f"RLPD@415k FILTERED (f_c=1.2 Hz), n={len(rows)}, status={status}")
        print("  grasp %.2f  lift %.2f  carry %.2f  throw %.2f  succ %.2f  ret %.3f"
              % (c["grasp_rate"], c["lift_rate"], c["carry_rate"], c["throw_rate"],
                 c["full_success_rate"], c["mean_return"]))
        print(f"  baseline carry distribution (8 runs): {BASE_CARRY}")
        cr, gr = c["carry_rate"], c["grasp_rate"]
        if gr < BASE_GRASP_MIN:
            print(f"  VERDICT grasp: REGRESSION -- {gr:.2f} below every baseline run; "
                  "the filter lag plausibly breaks grasp timing")
        else:
            print(f"  VERDICT grasp: preserved ({gr:.2f}, baseline min 0.96)")
        if cr > max(BASE_CARRY):
            print(f"  VERDICT carry: ABOVE the entire baseline range -- first evidence "
                  "the pump was binding")
        elif cr < min(BASE_CARRY):
            print(f"  VERDICT carry: below the baseline range")
        else:
            print(f"  VERDICT carry: inside the baseline noise range [0.00, 0.36] -- "
                  "no effect resolvable from one run")
    else:
        print("RLPD filtered eval: not available yet")

    got = load(f"{S}/runs/filter_ab/sft_stoch_filter12_n25.json", "stochastic")
    if got:
        rows, status = got
        c = M.classify(rows)
        n_succ = sum(1 for r in rows if r["stages"].get("success"))
        print(f"\nSFT STOCHASTIC FILTERED, n={len(rows)}, status={status}")
        print("  succ %.2f (%d/25)  carry %.2f  grasp %.2f  ret %.3f"
              % (c["full_success_rate"], n_succ, c["carry_rate"], c["grasp_rate"],
                 c["mean_return"]))
        print("  baseline: 0.16 success in each of two independent unfiltered draws")
        if c["full_success_rate"] >= 0.08:
            print("  VERDICT: the filter preserves the only working behaviour")
        elif c["full_success_rate"] == 0.0:
            print("  VERDICT: the filter DESTROYS the SFT successes -- do not use it "
                  "for stochastic execution")
        else:
            print("  VERDICT: reduced but present; more draws needed")
    else:
        print("\nSFT filtered eval: not available yet")

    p = f"{S}/handdyn_rlpd_filt.json"
    if os.path.exists(p):
        d = json.load(open(p))
        if d.get("_status") == "OK":
            print(f"\nIN-SIM SMOOTHNESS, filtered (from the command probe):")
            print("  boundary %.4f  intra %.4f  boundary_max %.4f"
                  % (d["boundary_mean"], d["intra_chunk_mean"], d["boundary_max"]))
            print("  unfiltered baseline: boundary 0.1140, intra 0.0055 "
                  "(hand 0.0201), boundary_max 0.9026")
    return 0


if __name__ == "__main__":
    sys.exit(main())
