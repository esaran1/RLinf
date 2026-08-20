"""Recompute every stored result under the corrected horizontal carry/throw definition.

The frozen definition of carry has always been HORIZONTAL transport >= 0.05 m, but the
rollout code recorded ``disp_m`` as a full 3-D norm, so a purely vertical fling could be
scored as a carry -- the exploit counted as its own opposite. This recomputes carry and
throw from the raw per-condition rows.

Historical rows predate ``disp_xy_m``. They are corrected where the geometry allows:
``max_lift_m`` is a RUNNING max rather than the final height, so it only bounds the
vertical component. The correction is therefore reported in three tiers:

  exact      the row carries disp_xy_m (or the piston poses) -- no ambiguity
  bounded    only the running max lift is available: horizontal lies in a known interval,
             and the label flips only if the WHOLE interval falls below the threshold
  unchanged  no lift, so carry/throw cannot apply either way

Nothing is overwritten. The report is written alongside the originals.

Usage:  python recompute_horizontal.py
"""
import glob
import json
import math
import os
import sys

sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
from rlinf.envs.isaaclab.tasks import g1_piston_metrics as M  # noqa: E402

S = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
     "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")
V = "/home/jren313/research/starvla_rl/verified_results"
T = M.CARRY_MIN_DISPLACEMENT_M


def horiz_interval(row):
    """(low, high, tier) bounds on horizontal displacement for one row.

    With only a 3-D norm d and a running max lift L, the final vertical offset z is
    unknown but bounded by |z| <= L, so horizontal h = sqrt(d^2 - z^2) lies in
    [sqrt(max(d^2 - L^2, 0)), d].
    """
    if "disp_xy_m" in row:
        h = float(row["disp_xy_m"])
        return h, h, "exact"
    p0, pN = row.get("piston_initial_xyz"), row.get("piston_final_xyz")
    if p0 is not None and pN is not None:
        h = math.hypot(pN[0] - p0[0], pN[1] - p0[1])
        return h, h, "exact"
    if row.get("final_dz_m") is not None:
        h = math.sqrt(max(row["disp_m"] ** 2 - float(row["final_dz_m"]) ** 2, 0.0))
        return h, h, "exact"
    d = float(row["disp_m"])
    lift = float(row.get("max_lift_m", 0.0))
    return math.sqrt(max(d * d - lift * lift, 0.0)), d, "bounded"


def reclassify(rows):
    """Carry/throw counts under the corrected definition, with the uncertainty made explicit."""
    n = len(rows)
    carry_old = sum(1 for r in rows if bool(r["stages"].get("lift"))
                    and r["disp_m"] >= T)
    carry_sure = carry_flip = carry_maybe = 0
    flipped = []
    for r in rows:
        if not r["stages"].get("lift"):
            continue
        lo, hi, tier = horiz_interval(r)
        was = r["disp_m"] >= T
        if not was:
            continue
        if lo >= T:
            carry_sure += 1                       # still a carry on any reading
        elif hi < T:
            carry_flip += 1                       # definitely a throw now
            flipped.append((r, lo, hi, tier, "definite"))
        else:
            carry_maybe += 1                      # interval straddles the threshold
            flipped.append((r, lo, hi, tier, "ambiguous"))
    return {"n": n, "carry_3d": carry_old, "carry_certain": carry_sure,
            "flipped_to_throw": carry_flip, "ambiguous": carry_maybe,
            "flipped_rows": flipped}


def report(name, rows, out):
    r = reclassify(rows)
    if r["carry_3d"] == 0 and r["carry_certain"] == 0:
        return
    lo_rate = r["carry_certain"] / r["n"]
    hi_rate = (r["carry_certain"] + r["ambiguous"]) / r["n"]
    band = (f"{lo_rate:.2f}" if lo_rate == hi_rate
            else f"{lo_rate:.2f}-{hi_rate:.2f}")
    print("%-34s n=%-3d carry 3-D %.2f -> horizontal %s   "
          "(certain %d, flipped %d, ambiguous %d)" %
          (name, r["n"], r["carry_3d"] / r["n"], band, r["carry_certain"],
           r["flipped_to_throw"], r["ambiguous"]))
    for row, lo, hi, tier, kind in r["flipped_rows"]:
        print("      cond %-3d disp3d %.4f lift %.4f -> horiz [%.4f, %.4f] %s (%s)" %
              (row["condition"], row["disp_m"], row.get("max_lift_m", 0.0),
               lo, hi, kind, tier))
    out[name] = {k: v for k, v in r.items() if k != "flipped_rows"}
    out[name]["carry_rate_3d"] = round(r["carry_3d"] / r["n"], 4)
    out[name]["carry_rate_horizontal_low"] = round(lo_rate, 4)
    out[name]["carry_rate_horizontal_high"] = round(hi_rate, 4)
    out[name]["flipped"] = [
        {"condition": row["condition"], "hash": row.get("hash"),
         "disp_m_3d": row["disp_m"], "max_lift_m": row.get("max_lift_m"),
         "horizontal_low": round(lo, 4), "horizontal_high": round(hi, 4),
         "tier": tier, "verdict": kind}
        for row, lo, hi, tier, kind in r["flipped_rows"]]


def main():
    out = {}
    print(f"corrected metric: {M.METRICS_VERSION}")
    print(f"carry threshold: horizontal >= {T} m\n")

    print("POST-HOC n=50 EVALUATIONS")
    for p in sorted(glob.glob(f"{S}/runs/n50_*.json")):
        d = json.load(open(p))
        if d.get("_status") != "OK":
            continue
        for mode, v in d.get("modes", {}).items():
            rows = v.get("per_condition", [])
            if rows:
                report(f"{os.path.basename(p)[:-5]}/{mode[:4]}", rows, out)

    print("\nTRAINING-TIME SWEEPS")
    for f in ["sac.json", "sac_s2.json", "sac_s3.json", "rlpd.json", "rlpd_s2.json",
              "rlpd_s3.json", "sft_v2.json"]:
        p = f"{S}/runs/{f}"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if d.get("_status") != "OK":
            continue
        for e in d.get("evals", []):
            rows = e.get("per_condition", [])
            if rows:
                report(f"{f[:-5]}/{e['tag']}@{e['env_steps']}", rows, out)

    dest = f"{V}/tables/horizontal_recompute.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump({"metrics_version": M.METRICS_VERSION,
               "carry_threshold_m": T, "results": out}, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")
    if not out:
        print("no evaluation contained a carry-labelled rollout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
