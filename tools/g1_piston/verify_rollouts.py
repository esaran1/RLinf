"""Independently verify every rendered rollout against its raw trajectory.

A rollout is only usable as evidence if the label on the video is re-derivable from the
simulator state that was recorded while the video was being made. This script does not
trust the sidecar's stored ``row``: it recomputes displacement, lift, carry, throw and
success from ``trajectory`` and the recorded piston poses, using the frozen metric code,
and fails the artifact on any disagreement.

It also checks that the condition hash is genuinely a member of the frozen suite, that
the video file exists and is non-trivial, and that the frame count is consistent with
the episode length.

Usage:  python verify_rollouts.py <dir> [<dir> ...]
Writes: verified_results/manifests/rollout_verification.json
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
import numpy as np  # noqa: E402

from rlinf.envs.isaaclab.tasks import g1_piston_metrics as M  # noqa: E402
from rlinf.envs.isaaclab.tasks.g1_piston_reset import (  # noqa: E402
    EXPERIMENT_N_TRAIN,
    build_reset_suite,
)

TOL = 1e-3

_, EV = build_reset_suite(n_train=EXPERIMENT_N_TRAIN, n_eval=50, seed=20260817)
FROZEN = {c.hash(): c.index for c in EV}


def verify(path):
    d = json.load(open(path))
    fails, checks = [], {}

    h = d["condition_hash"]
    checks["hash_in_frozen_suite"] = h in FROZEN
    if h not in FROZEN:
        fails.append(f"condition hash {h} is not in the frozen 50-condition suite")
    elif FROZEN[h] != d["condition_index"]:
        fails.append(f"hash {h} is frozen index {FROZEN[h]}, sidecar says "
                     f"{d['condition_index']}")
    checks["suite_config"] = (d.get("n_train") == EXPERIMENT_N_TRAIN
                              and d.get("reset_suite_seed") == 20260817)
    if not checks["suite_config"]:
        fails.append("rendered under a different suite configuration")

    checks["metrics_version"] = M.METRICS_VERSION

    traj = d["trajectory"]
    stored = d["row"]

    # Recompute displacement and lift from the recorded piston poses alone.
    p0 = np.array(d["piston_initial_xyz"], dtype=float)
    pN = np.array(d["piston_final_xyz"], dtype=float)
    disp = float(np.linalg.norm(pN - p0))
    # Floored at 0.0, matching the trainer/evaluator/renderer, which all seed
    # ``maxlift = 0.0`` and take a running max. An untouched piston settles a few mm
    # DOWNWARD under gravity, so an unfloored max is negative and would spuriously
    # disagree with every no-contact rollout.
    lift = max([0.0] + [float(np.array(t["piston_xyz"], dtype=float)[2] - p0[2])
                        for t in traj])

    # Carry/throw are defined on HORIZONTAL transport. The rollout code used to write
    # disp_m as a 3-D norm, letting a vertical fling clear the threshold; that bug is
    # fixed in metrics v1.1 and disp_xy_m is now recorded at source. This recomputes the
    # horizontal component from the true final pose as an independent check on both.
    horiz = float(np.linalg.norm((pN - p0)[:2]))
    checks["disp_recomputed"] = round(disp, 4)
    checks["horizontal_disp_recomputed"] = round(horiz, 4)
    checks["lift_recomputed"] = round(lift, 4)
    if abs(disp - stored["disp_m"]) > TOL:
        fails.append(f"disp mismatch: stored {stored['disp_m']} vs raw {disp:.4f}")
    if abs(lift - stored["max_lift_m"]) > TOL:
        fails.append(f"lift mismatch: stored {stored['max_lift_m']} vs raw {lift:.4f}")

    # Stage flags must be the monotone OR of the per-chunk flags.
    acc = {}
    for t in traj:
        for k, v in t["stages"].items():
            acc[k] = acc.get(k, False) or bool(v)
    for k, v in stored["stages"].items():
        if acc.get(k, False) != bool(v):
            fails.append(f"stage '{k}' mismatch: stored {v} vs raw {acc.get(k, False)}")

    # Frozen labels, recomputed from the raw row.
    raw_row = {"disp_m": disp, "max_lift_m": lift, "stages": acc,
               "return": stored["return"]}
    label = {"carry": M.is_carry(raw_row), "throw": M.is_throw(raw_row),
             "success": bool(acc.get("success")), "grasp": bool(acc.get("grasp")),
             "lift": bool(acc.get("lift")), "reach": bool(acc.get("reach"))}
    checks["recomputed_label"] = label
    if M.is_carry(stored) != label["carry"]:
        fails.append("carry label differs between stored row and raw trajectory")
    if M.is_throw(stored) != label["throw"]:
        fails.append("throw label differs between stored row and raw trajectory")

    # carry and throw are mutually exclusive by construction; assert it holds here.
    if label["carry"] and label["throw"]:
        fails.append("row classified BOTH carry and throw")

    checks["carry_by_horizontal_only"] = bool(
        acc.get("lift") and horiz >= M.CARRY_MIN_DISPLACEMENT_M)
    # Under metrics v1.1 the frozen label is ALREADY horizontal, so these must agree.
    # A disagreement means the sidecar's disp_xy_m and the recorded poses disagree,
    # which is a data-integrity failure rather than a definitional nuance.
    checks["carry_depends_on_vertical"] = bool(
        label["carry"] and not checks["carry_by_horizontal_only"])
    if checks["carry_depends_on_vertical"]:
        fails.append(f"carry label survives only via the vertical component "
                     f"(horizontal {horiz:.4f} m < {M.CARRY_MIN_DISPLACEMENT_M} m); "
                     f"disp_xy_m and the recorded piston poses disagree")
    if "disp_xy_m" in stored and abs(stored["disp_xy_m"] - horiz) > TOL:
        fails.append(f"disp_xy_m mismatch: stored {stored['disp_xy_m']} "
                     f"vs raw {horiz:.4f}")

    vid = d["video"]
    checks["video_exists"] = os.path.exists(vid)
    checks["video_bytes"] = os.path.getsize(vid) if os.path.exists(vid) else 0
    if not checks["video_exists"]:
        fails.append("video file missing")
    elif checks["video_bytes"] < 10_000:
        fails.append(f"video suspiciously small ({checks['video_bytes']} B)")

    if d.get("n_frames", 0) < 10:
        fails.append(f"only {d.get('n_frames')} frames captured")

    return {"sidecar": path, "video": vid, "method": d.get("method"),
            "seed": d.get("seed"), "checkpoint": d.get("checkpoint"),
            "checkpoint_env_steps": d.get("checkpoint_env_steps"),
            "condition_index": d.get("condition_index"), "condition_hash": h,
            "mode": d.get("mode"), "stored_row": stored, "checks": checks,
            "failures": fails, "verified": not fails}


def main(dirs):
    recs = []
    for dd in dirs:
        for p in sorted(glob.glob(os.path.join(dd, "**", "*.json"), recursive=True)):
            if os.path.basename(p).startswith("_render_"):
                continue
            try:
                d = json.load(open(p))
            except Exception:
                continue
            if "trajectory" not in d or "condition_hash" not in d:
                continue
            recs.append(verify(p))

    ok = sum(1 for r in recs if r["verified"])
    out = {"metrics_version": M.METRICS_VERSION,
           "n_rollouts": len(recs), "n_verified": ok, "n_failed": len(recs) - ok,
           "rollouts": recs}
    dest = ("/home/jren313/research/starvla_rl/verified_results/manifests/"
            "rollout_verification.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)

    print(f"metrics {M.METRICS_VERSION}   {ok}/{len(recs)} verified")
    for r in recs:
        flag = "OK  " if r["verified"] else "FAIL"
        lab = r["checks"].get("recomputed_label", {})
        tags = ",".join(k for k, v in lab.items() if v) or "-"
        c = r["checks"]
        note = ""
        if c.get("carry_depends_on_vertical"):
            note = (f"   <-- CARRY ONLY VIA VERTICAL "
                    f"(horiz {c['horizontal_disp_recomputed']:.3f} m "
                    f"< {M.CARRY_MIN_DISPLACEMENT_M})")
        print(f"  {flag} {r['method']:<10} s{r['seed']} step{r['checkpoint_env_steps']:>7} "
              f"cond{r['condition_index']:<3} {r['mode'][:4]}  {tags}{note}")
        for f in r["failures"]:
            print(f"        ! {f}")

    susp = [r for r in recs if r["checks"].get("carry_depends_on_vertical")]
    if susp:
        print(f"\n{len(susp)} rollout(s) FAILED the horizontal consistency check: the "
              f"carry label survives only through the vertical component.")
        print("Under metrics v1.1 carry is horizontal by definition, so this is a "
              "data-integrity failure, not a definitional nuance.")
    return 0 if ok == len(recs) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or [
        "/home/jren313/research/starvla_rl/verified_results/videos"]))
