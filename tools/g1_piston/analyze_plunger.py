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

"""Report plunger actuation across demonstrations, and camera shake across policies.

Two measurements that share one theme: things the v1 pipeline never looked at.

1. **Plunger** -- the scene object is an articulation whose prismatic ``PistonJoint``
   is the pipette plunger (travel 0-0.04 m, spring-loaded). Replaying every
   demonstration while logging ``joint_pos`` answers the question the v2 reward
   depends on: do the demonstrations actually press it, and how far?

2. **Camera shake** -- the ego camera rides on ``d435_link``. The waist is frozen in
   the action space, so any camera motion is physical wobble transmitted from arm
   reaction forces through the fixed-base robot. Comparing the head pose under SFT,
   RLPD and the smoothness-trained policy says whether RL made the robot shake its
   own camera, which would feed motion blur straight back into the observation.

Usage:  python analyze_plunger.py
"""
import glob
import json
import os

import numpy as np

S = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
     "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")

#: Full travel of the plunger joint, metres (measured from the scene).
TRAVEL = 0.04
#: Fraction of travel the v2 reward counts as a deliberate press.
PRESS_FRAC = 0.50


def plunger_report():
    fp = f"{S}/piston_joint_probe.json"
    if not os.path.exists(fp):
        print("plunger probe: not available yet")
        return None
    d = json.load(open(fp))
    per = d.get("per_episode", {})
    if not per:
        print(f"plunger probe: status {d.get('_status')}, no episodes yet")
        return None
    print(f"\nPLUNGER ACTUATION IN THE DEMONSTRATIONS (status {d.get('_status')})")
    print(f"  joint travel {TRAVEL} m; v2 press threshold "
          f"{PRESS_FRAC * TRAVEL:.3f} m ({PRESS_FRAC:.0%} of travel)")
    fracs = {}
    for ep, v in sorted(per.items(), key=lambda kv: int(kv[0])):
        f = v["joint_max"] / TRAVEL
        fracs[int(ep)] = f
        flag = "PRESS" if f >= PRESS_FRAC else "compliance only"
        print(f"  ep{int(ep):<3d} max {v['joint_max']:.4f} m  {f:5.0%} of travel"
              f"  range {v['joint_range']:.4f}  -> {flag}")
    vals = np.array(list(fracs.values()))
    n_press = int((vals >= PRESS_FRAC).sum())
    print(f"\n  {n_press}/{len(vals)} episodes press past {PRESS_FRAC:.0%} of travel")
    print(f"  median {np.median(vals):.0%}, min {vals.min():.0%}, max {vals.max():.0%}")
    if n_press == 0:
        print("  VERDICT: the demonstrations never deliberately press the plunger. The "
              "dataset is transport-only, so RLPD demo replay supplies no pressing "
              "behaviour and the v2 press reward must induce it de novo.")
    elif n_press == len(vals):
        print("  VERDICT: every demonstration presses the plunger. Demo replay already "
              "contains the functional act; the v2 reward simply scores it.")
    else:
        print(f"  VERDICT: the demonstrations are SPLIT -- {n_press} press, "
              f"{len(vals) - n_press} show only passive compliance from gripping. "
              "Demo replay contains the act but at low density, so the v2 press reward "
              "matters for making it reliable.")
    return fracs


def _cam_stats(rows):
    C = np.asarray(rows, dtype=float)
    pos, quat = C[:, :3], C[:, 3:]
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 1000.0      # mm
    qstep = np.linalg.norm(np.diff(quat, axis=0), axis=1)
    return {
        "n": len(C),
        "pos_ptp_mm": float(np.linalg.norm(pos.max(0) - pos.min(0)) * 1000.0),
        "step_mm_mean": float(step.mean()),
        "step_mm_p95": float(np.percentile(step, 95)),
        "step_mm_max": float(step.max()),
        "quat_step_mean": float(qstep.mean()),
        "quat_step_max": float(qstep.max()),
    }


def camera_report():
    rows = {}
    for tag in ("sft", "rlpd415k", "smooth"):
        hits = glob.glob(f"{S}/camshake/{tag}/*cond*.json")
        hits = [h for h in hits if not os.path.basename(h).startswith("_render")]
        if not hits:
            continue
        d = json.load(open(hits[0]))
        log = d.get("camera_log")
        if log:
            rows[tag] = _cam_stats(log)
    if not rows:
        print("\ncamera-shake logs: not available yet")
        return None
    print("\nHEAD-CAMERA MOTION (ego camera rides on d435_link; waist is frozen,")
    print("so this is physical wobble from arm reaction forces)")
    print(f"  {'policy':<12} {'steps':>6} {'travel mm':>10} {'mean mm':>9} "
          f"{'p95 mm':>8} {'max mm':>8} {'rot/step':>10}")
    for tag, s in rows.items():
        print(f"  {tag:<12} {s['n']:>6} {s['pos_ptp_mm']:>10.2f} "
              f"{s['step_mm_mean']:>9.4f} {s['step_mm_p95']:>8.4f} "
              f"{s['step_mm_max']:>8.4f} {s['quat_step_mean']:>10.6f}")
    if "sft" in rows and "rlpd415k" in rows:
        r = rows["rlpd415k"]["step_mm_mean"] / max(rows["sft"]["step_mm_mean"], 1e-9)
        print(f"\n  RLPD shakes the camera {r:.2f}x more per step than SFT")
        if "smooth" in rows:
            r2 = (rows["smooth"]["step_mm_mean"]
                  / max(rows["sft"]["step_mm_mean"], 1e-9))
            print(f"  smoothness-trained: {r2:.2f}x SFT "
                  f"({'improved' if r2 < r else 'no better'} vs RLPD's {r:.2f}x)")
    return rows


def main():
    print("plunger + camera-shake report")
    fr = plunger_report()
    cam = camera_report()
    out = {"plunger_frac_of_travel": fr, "camera": cam}
    dst = f"{S}/plunger_camera_report.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
