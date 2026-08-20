"""Does frame capture explain the render/stored divergence?

Rendered RLPD@415140 produced ZERO lifts where the stored n=50 evaluation of the same
checkpoint and conditions recorded five carries; only the LIFTING conditions disagreed.
The leading hypothesis was that the renderer's mid-chunk camera reads perturb contact
physics.

The blend A/B's ``blend=0`` arm is an independent test of exactly that: it runs
``eval_checkpoint.py`` (no mid-chunk camera reads, blending disabled) on the same
checkpoint and the same frozen conditions. So:

  * if blend=0 reproduces the stored lifts  -> capture perturbs the physics
  * if blend=0 also shows no lifts          -> the stored result is not reproducible
                                               across processes at all, and the cause is
                                               nondeterminism, not capture

Usage:  python check_capture_hypothesis.py
"""
import json
import os
import sys

sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
from rlinf.envs.isaaclab.tasks import g1_piston_metrics as M  # noqa: E402

S = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
     "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")
V = "/home/jren313/research/starvla_rl/RLinf/verified_results"
AB = f"{S}/runs/blend_ab/rlpd_s1_415140_blend0_n25.json"
STORED = f"{S}/runs/n50_rlpd_415140.json"


def main():
    if not os.path.exists(AB):
        print("blend=0 arm not finished yet")
        return 0
    d = json.load(open(AB))
    if d.get("_status") != "OK":
        print(f"blend=0 arm status={d.get('_status')}, progress={d.get('progress')}")
        return 0

    ab = {r["condition"]: r for r in d["modes"]["deterministic"]["per_condition"]}
    st = {r["condition"]: r
          for r in json.load(open(STORED))["modes"]["deterministic"]["per_condition"]}
    rendered = {}
    for c in sorted(ab):
        p = (f"{V}/videos/matched/rlpd_s1_step415140_cond{c}_deterministic.json")
        if os.path.exists(p):
            rendered[c] = json.load(open(p))["row"]

    shared = sorted(set(ab) & set(st))
    print(f"metrics {M.METRICS_VERSION}")
    print("RLPD@415140, frozen conditions, deterministic. Three independent measurements:")
    print("  STORED   = eval_checkpoint, no capture, earlier session")
    print("  A/B b=0  = eval_checkpoint, no capture, this session")
    print("  RENDER   = render_rollouts, WITH mid-chunk camera reads\n")
    print("%-5s | %-22s | %-22s | %-22s" % ("cond", "STORED", "A/B blend=0", "RENDER"))
    agree_ab = agree_render = 0
    lifting = []
    for c in shared:
        def fmt(r):
            if r is None:
                return "%-22s" % "(not rendered)"
            return "lift=%-5s disp %.4f" % (bool(r["stages"].get("lift")), r["disp_m"])
        s, a, r = st[c], ab[c], rendered.get(c)
        print("%-5d | %s | %s | %s" % (c, fmt(s), fmt(a), fmt(r)))
        if bool(s["stages"].get("lift")):
            lifting.append(c)
            if bool(a["stages"].get("lift")):
                agree_ab += 1
            if r is not None and bool(r["stages"].get("lift")):
                agree_render += 1

    print()
    n = len(lifting)
    print(f"conditions the STORED evaluation reported as lifting: {n} ({lifting})")
    if n:
        print(f"  reproduced by A/B blend=0 (no capture):  {agree_ab}/{n}")
        print(f"  reproduced by the RENDER (with capture): {agree_render}/{n}")
        print()
        if agree_ab > agree_render:
            print("VERDICT: capture is implicated -- the no-capture rerun reproduces "
                  "lifts the\n         capture-enabled render does not.")
        elif agree_ab == 0:
            print("VERDICT: capture is NOT the cause -- a no-capture rerun in a fresh "
                  "process\n         also fails to reproduce the stored lifts, so the "
                  "stored result is not\n         reproducible across processes. The "
                  "cause is nondeterminism in a\n         contact-sensitive task, not "
                  "frame capture.")
        else:
            print("VERDICT: inconclusive -- both reproduce the lifts partially.")

    ca, cs = M.classify(list(ab.values())), M.classify([st[c] for c in shared])
    print(f"\nrates on the shared {len(shared)} conditions:")
    print("  STORED    carry %.2f lift %.2f grasp %.2f"
          % (cs["carry_rate"], cs["lift_rate"], cs["grasp_rate"]))
    print("  A/B b=0   carry %.2f lift %.2f grasp %.2f"
          % (ca["carry_rate"], ca["lift_rate"], ca["grasp_rate"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
