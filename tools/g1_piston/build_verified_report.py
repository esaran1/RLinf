"""Build the verified results tables and figures from suite-checked evaluations only.

Every number here is recomputed from per-condition raw rows with the frozen metric code.
An evaluation is admitted only if it is complete, its status is OK, and its condition
hashes are EXACTLY the frozen 25-prefix or the frozen 50-suite. Anything else -- partial
files, wrong-suite outputs, pilot runs, aborted runs -- is listed as excluded, with the
reason, and never enters a table or a plot.

Usage:  python build_verified_report.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
from rlinf.envs.isaaclab.tasks import g1_piston_metrics as M  # noqa: E402
from rlinf.envs.isaaclab.tasks.g1_piston_reset import (  # noqa: E402
    EXPERIMENT_N_TRAIN,
    build_reset_suite,
)

S = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
     "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")
V = "/home/jren313/research/starvla_rl/verified_results"

_, EV = build_reset_suite(n_train=EXPERIMENT_N_TRAIN, n_eval=50, seed=20260817)
H50 = [c.hash() for c in EV]
H25 = H50[:25]

admitted, excluded = [], []


def admit(rows, **meta):
    """Admit an evaluation only if its hashes are exactly a frozen suite."""
    hs = [r["hash"] for r in rows]
    if hs == H25:
        suite = "n25"
    elif hs == H50:
        suite = "n50"
    else:
        excluded.append(dict(meta, reason=f"condition hashes are not a frozen suite "
                                          f"(n={len(hs)})"))
        return
    c = M.classify(rows)
    admitted.append(dict(meta, suite=suite, n=len(rows), **{
        k: c[k] for k in ("carry_rate", "throw_rate", "full_success_rate", "grasp_rate",
                          "lift_rate", "reach_rate", "mean_return", "mean_disp_m",
                          "mean_max_lift_m")},
        success_ci95=M.wilson95(sum(1 for r in rows if r["stages"].get("success")),
                                len(rows)),
        carry_ci95=M.wilson95(sum(1 for r in rows if M.is_carry(r)), len(rows))))


def load_training_runs():
    for f, method, seed in [("sac.json", "SAC", 1), ("sac_s2.json", "SAC", 2),
                            ("sac_s3.json", "SAC", 3), ("rlpd.json", "RLPD", 1),
                            ("rlpd_s2.json", "RLPD", 2), ("rlpd_s3.json", "RLPD", 3)]:
        p = f"{S}/runs/{f}"
        if not os.path.exists(p):
            excluded.append({"source": f, "reason": "run not present on disk"})
            continue
        d = json.load(open(p))
        if d.get("_status") != "OK":
            excluded.append({"source": f, "reason": f"_status={d.get('_status')}"})
            continue
        for e in d.get("evals", []):
            rows = e.get("per_condition", [])
            if not rows:
                continue
            admit(rows, source=f, method=method, seed=seed, tag=e["tag"],
                  env_steps=e["env_steps"],
                  mode=("stochastic" if "stoch" in e.get("tag", "") + e.get("mode", "")
                        else "deterministic"),
                  origin="training")


def load_posthoc():
    for p in sorted(glob.glob(f"{S}/runs/n50_*.json")):
        b = os.path.basename(p)
        d = json.load(open(p))
        if d.get("_status") != "OK":
            excluded.append({"source": b, "reason": f"_status={d.get('_status')}"})
            continue
        method = "RLPD" if "rlpd" in b else "SAC"
        seed = 2 if "sac_s2" in b else 1
        for mode, v in d.get("modes", {}).items():
            rows = v.get("per_condition", [])
            if not rows:
                continue
            admit(rows, source=b, method=method, seed=seed, tag="posthoc_n50",
                  env_steps=d.get("checkpoint_env_steps"), mode=mode, origin="posthoc")
    for p in sorted(glob.glob(f"{S}/runs/*.wrongsuite")) + \
            sorted(glob.glob(f"{S}/runs/*.partial.*")) + \
            sorted(glob.glob(f"{S}/runs/*.failed.*")):
        excluded.append({"source": os.path.basename(p),
                         "reason": "quarantined (wrong suite / partial / failed)"})


def load_sft():
    p = f"{S}/runs/sft_v2.json"
    if os.path.exists(p):
        d = json.load(open(p))
        for e in d.get("evals", []):
            rows = e.get("per_condition", [])
            if rows:
                admit(rows, source="sft_v2.json", method="SFT", seed=0,
                      tag=e["tag"], env_steps=0, mode="deterministic", origin="baseline")


def carry_auc(curve):
    """Trapezoidal mean carry rate over interaction count -- average carry while training."""
    if len(curve) < 2:
        return None
    x = np.array([c[0] for c in curve], float)
    y = np.array([c[1] for c in curve], float)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy<2 names it trapz
    return float(trapz(y, x) / (x[-1] - x[0])) if x[-1] > x[0] else None


def main():
    load_training_runs(); load_posthoc(); load_sft()
    os.makedirs(f"{V}/tables", exist_ok=True)
    os.makedirs(f"{V}/figures", exist_ok=True)
    os.makedirs(f"{V}/manifests", exist_ok=True)

    json.dump({"metrics_version": M.METRICS_VERSION,
               "n_admitted": len(admitted), "n_excluded": len(excluded),
               "admitted": admitted, "excluded": excluded},
              open(f"{V}/manifests/admitted_evaluations.json", "w"), indent=2)

    det = [a for a in admitted if a["mode"] == "deterministic"]

    # ---- learning curves, n=25 matched prefix, deterministic ----
    print(f"metrics {M.METRICS_VERSION}   admitted {len(admitted)}   "
          f"excluded {len(excluded)}\n")
    print("LEARNING CURVES (frozen n=25 prefix, deterministic, training-time)")
    print("%-6s %-4s %8s %6s %6s %6s %6s %8s" %
          ("meth", "seed", "steps", "carry", "throw", "succ", "grasp", "return"))
    curves = {}
    for a in sorted(det, key=lambda a: (a["method"], a["seed"], a["env_steps"] or 0)):
        if a["origin"] != "training" or a["suite"] != "n25":
            continue
        k = (a["method"], a["seed"])
        curves.setdefault(k, []).append((a["env_steps"], a["carry_rate"]))
        print("%-6s %-4s %8s %6.2f %6.2f %6.2f %6.2f %8.3f" %
              (a["method"], a["seed"], a["env_steps"], a["carry_rate"], a["throw_rate"],
               a["full_success_rate"], a["grasp_rate"], a["mean_return"]))

    print("\nCARRY AUC (normalised mean carry rate over training, n=25 curve)")
    aucs = {}
    for k, c in sorted(curves.items()):
        v = carry_auc(sorted(c))
        aucs[k] = v
        if v is not None:
            print("  %-6s seed %s  AUC %.4f   final %.2f   best %.2f" %
                  (k[0], k[1], v, sorted(c)[-1][1], max(y for _, y in c)))

    # ---- within-method variance vs between-method difference ----
    print("\nWITHIN-METHOD vs BETWEEN-METHOD (the decisive quantity)")
    for metric in ["carry_rate"]:
        by = {}
        for k, c in curves.items():
            by.setdefault(k[0], []).append(max(y for _, y in c))
        for m, vals in sorted(by.items()):
            if len(vals) > 1:
                print("  %-5s best-%s across %d seeds: %s  spread %.2f" %
                      (m, metric, len(vals), [round(v, 2) for v in vals],
                       max(vals) - min(vals)))
            else:
                print("  %-5s best-%s: %s  (ONE seed -- no within-method variance "
                      "estimate)" % (m, metric, [round(v, 2) for v in vals]))
        if len(by) == 2 and all(len(v) > 1 for v in by.values()):
            sac, rlpd = by.get("SAC", []), by.get("RLPD", [])
            within = max(max(sac) - min(sac), max(rlpd) - min(rlpd))
            between = abs(np.mean(sac) - np.mean(rlpd))
            print("  within-method spread %.3f   between-method difference %.3f" %
                  (within, between))
            print("  VERDICT:", "between-seed variance EXCEEDS the algorithm effect; "
                  "no ordering is supportable" if within >= between else
                  "algorithm effect exceeds between-seed variance")

    # ---- final n=50 table ----
    print("\nFINAL EVALUATION (frozen n=50)")
    print("%-6s %-4s %8s %-13s %6s %6s %6s %6s %8s  %s" %
          ("meth", "seed", "steps", "mode", "carry", "throw", "succ", "grasp",
           "return", "succ CI95"))
    for a in sorted(admitted, key=lambda a: (a["method"], a["seed"],
                                             a["env_steps"] or 0, a["mode"])):
        if a["suite"] != "n50":
            continue
        print("%-6s %-4s %8s %-13s %6.2f %6.2f %6.2f %6.2f %8.3f  %s" %
              (a["method"], a["seed"], a["env_steps"], a["mode"], a["carry_rate"],
               a["throw_rate"], a["full_success_rate"], a["grasp_rate"],
               a["mean_return"], a["success_ci95"]))

    # ---- the SFT stochastic anomaly ----
    print("\nSTEP-0 SFT POLICY (identical weights, mode is the only difference)")
    for a in sorted(admitted, key=lambda a: a["mode"]):
        if a["env_steps"] not in (0, None) or a["origin"] not in ("training", "baseline"):
            continue
        print("  %-10s seed %s %-13s n=%-3d carry %.2f succ %.2f grasp %.2f  "
              "succCI %s" % (a["source"][:10], a["seed"], a["mode"], a["n"],
                             a["carry_rate"], a["full_success_rate"], a["grasp_rate"],
                             a["success_ci95"]))

    with open(f"{V}/tables/verified_results.json", "w") as f:
        json.dump({"metrics_version": M.METRICS_VERSION, "curves":
                   {f"{k[0]}_s{k[1]}": sorted(v) for k, v in curves.items()},
                   "carry_auc": {f"{k[0]}_s{k[1]}": v for k, v in aucs.items()},
                   "admitted": admitted, "excluded": excluded}, f, indent=2)

    if excluded:
        print("\nEXCLUDED")
        for e in excluded:
            print("  %-34s %s" % (e.get("source", "?"), e["reason"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
