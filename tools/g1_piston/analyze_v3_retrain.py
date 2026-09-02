#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Report the v3 retrain against its pre-registered decision rules.

The rules were fixed before the data existed, in
``docs/contracts/g1_piston_v3_retrain_preregistration.json``:

* the primary readout is v3 stage rates on the frozen 25-condition suite, for the
  warm-start checkpoint AND the retrained checkpoint under the SAME predicate;
* a run counts as progress only if a stage rate moves beyond the noise floor already
  measured for this setup (8 identical evaluations of one unchanged checkpoint);
* ``dispense`` staying at 0.00 is a reportable finding about the task and the data, not
  a failure to hide;
* v3's thresholds are fixed by measurement and are not tuned after seeing results.

This script therefore prints the comparison and applies those rules verbatim rather than
inviting a fresh interpretation. It refuses to compare arms scored under different
reward versions, which is the mistake most likely to produce a flattering number.

Usage:  python analyze_v3_retrain.py
"""
import json
import os

S = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
     "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")

#: The 8-run noise floor measured on rlpd_ckpt_step415140 under the v1 predicate.
#: Post-grasp metrics are not resolvable from a single evaluation; grasp is.
NOISE = {
    "grasp_rate": (0.96, 1.00),
    "lift_rate": (0.20, 0.56),
    "carry_rate": (0.00, 0.36),
    "tube_rate": (0.00, 0.24),
    "mean_return": (3.21, 6.07),
}

STAGES = ("reach", "grasp", "lift", "tube", "press", "dispense", "plate", "success")


def load(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if d.get("_status") != "OK":
        return {"_pending": d.get("_status"), "progress": d.get("progress")}
    return d


def rates(d):
    m = (d.get("modes") or {}).get("deterministic") or {}
    return {k: v for k, v in m.items() if isinstance(v, (int, float))}


def main():
    base = load(f"{S}/runs/filter_ab/rlpd415k_v3baseline_n25.json")
    # Successive retrain attempts wrote different filenames (run 1 died on throughput,
    # run 2 on an import, run 3 is the first with enough gradient updates to test the
    # fixes). Take the NEWEST result that exists so the analysis follows the experiment
    # instead of silently reporting "not available" against a stale path.
    import glob as _glob
    _cands = sorted(
        (f for f in _glob.glob(f"{S}/runs/filter_ab/rlpd_v3*_n25.json")
         if "baseline" not in f),
        key=os.path.getmtime)
    new = load(_cands[-1]) if _cands else None
    new_path = _cands[-1] if _cands else None

    print("v3 RETRAIN vs PRE-REGISTERED RULES")
    print("=" * 68)

    if new_path:
        print(f"\n(retrained arm read from {os.path.basename(new_path)})")
    for name, d in (("warm-start (rlpd@415k)", base), ("v3-retrained", new)):
        if d is None:
            print(f"\n{name}: not available yet")
            continue
        if "_pending" in d:
            print(f"\n{name}: {d['_pending']} {d.get('progress') or ''}")
            continue
        rv = d.get("reward_version")
        print(f"\n{name}  [reward_version={rv}]")
        if rv != "v3_review_fixed":
            print("  REFUSING to report: this arm was not scored under v3. Comparing "
                  "arms across reward versions is meaningless.")
            continue
        r = rates(d)
        for k in sorted(r):
            print(f"    {k:24} {r[k]:.4f}")

    if not (base and new) or "_pending" in (base or {}) or "_pending" in (new or {}):
        print("\n(comparison pending both arms)")
        return 0
    if base.get("reward_version") != "v3_review_fixed" or \
            new.get("reward_version") != "v3_review_fixed":
        print("\nREFUSING the comparison: both arms must be scored under v3.")
        return 1

    rb, rn = rates(base), rates(new)
    print("\n" + "=" * 68)
    print("PRE-REGISTERED READOUT (same predicate, same frozen suite, n=25 det)")
    print(f"  {'metric':24} {'warm-start':>11} {'retrained':>10} {'delta':>9}   verdict")
    for k in ("reach_rate", "grasp_rate", "lift_rate", "tube_rate",
              "plate_rate", "full_success_rate", "mean_return"):
        if k not in rb or k not in rn:
            continue
        a, b = rb[k], rn[k]
        d = b - a
        lo, hi = NOISE.get(k, (None, None))
        if lo is None:
            verdict = "no noise floor on record"
        elif b > hi:
            verdict = f"ABOVE the 8-run range [{lo:.2f}, {hi:.2f}]"
        elif b < lo:
            verdict = f"BELOW the 8-run range [{lo:.2f}, {hi:.2f}]"
        else:
            verdict = f"inside noise [{lo:.2f}, {hi:.2f}] -- not resolvable"
        print(f"  {k:24} {a:>11.4f} {b:>10.4f} {d:>+9.4f}   {verdict}")

    # The functional act is the reason v3 exists; report it explicitly either way.
    print("\nFUNCTIONAL ACT (the reason v3 exists):")
    for k in ("press_rate", "dispense_rate"):
        a, b = rb.get(k), rn.get(k)
        if a is None and b is None:
            continue
        print(f"  {k:24} {a if a is None else f'{a:.4f}':>11} "
              f"{b if b is None else f'{b:.4f}':>10}")
    dis = rn.get("dispense_rate")
    if dis is not None:
        if dis > 0:
            print("  VERDICT: the policy performs the functional act. Report the rate "
                  "with the noise-floor caveat -- one evaluation cannot resolve a small "
                  "post-grasp rate.")
        else:
            print("  VERDICT: dispense 0.00. Per the pre-registration this is a "
                  "reportable finding: the functional act is not reachable from this "
                  "initialisation within the compute budget. Context: exactly 1 of 22 "
                  "executable demonstrations shows a genuine grasped press, and none "
                  "shows a dispense.")
    print("\nCaveat carried from the noise-floor study: post-grasp outcomes are not "
          "reproducible from a single evaluation in this setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
