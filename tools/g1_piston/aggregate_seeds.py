"""Aggregate SAC/RLPD across independent training seeds and emit the paper plots.

Two sources of variability must be kept distinct:

* the n=25 held-out initial conditions measure ENVIRONMENT-condition variability
  (already summarised per run as Wilson intervals);
* independent training seeds measure ALGORITHMIC variability.

This script reports the second: mean and spread ACROSS seeds at each matched online
interaction checkpoint. It never pools conditions from different seeds into one
interval, which would conflate the two.

Usage:
    python aggregate_seeds.py <out_dir> <runs_dir> [--sft <sft.json>]

Discovers ``sac.json`` / ``rlpd.json`` (seed 1) and ``{sac,rlpd}_s<N>.json``.
"""

from __future__ import annotations

import glob
import json
import os
import sys

STAGES = ["reach_rate", "grasp_rate", "lift_rate", "plate_rate", "full_success_rate"]

#: Thresholds for the sample-efficiency table. "Reliable" grasp is deliberately below
#: 1.0 so a single unlucky condition does not move the crossing point.
THRESHOLDS = [
    ("reliable_grasp", "grasp_rate", 0.80),
    ("first_lift", "lift_rate", 0.001),
    ("lift_20pct", "lift_rate", 0.20),
    ("lift_40pct", "lift_rate", 0.40),
    ("first_success", "full_success_rate", 0.001),
    ("sustained_success", "full_success_rate", 0.20),
]


def load_runs(runs_dir):
    """Return {"SAC": {seed: run}, "RLPD": {seed: run}}."""
    out = {"SAC": {}, "RLPD": {}}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        base = os.path.basename(path)[:-5]
        if base in ("sac", "rlpd"):
            algo, seed = base.upper(), 1
        elif "_s" in base and base.split("_s")[0] in ("sac", "rlpd"):
            algo, seed = base.split("_s")[0].upper(), int(base.split("_s")[1])
        else:
            continue
        try:
            with open(path) as f:
                run = json.load(f)
        except Exception:
            continue
        if run.get("evals"):
            out[algo][seed] = run
    return out


def deterministic_curve(run):
    """Evaluation points from the deterministic protocol, in interaction order."""
    evs = [e for e in run.get("evals", [])
           if e.get("mode", "deterministic") == "deterministic"]
    return sorted(evs, key=lambda e: e["env_steps"])


def bucket(env_steps, tol=20000):
    """Snap to the nominal matched checkpoints so seeds line up on one x axis.

    ``env_steps == 0`` is matched exactly: the spurious ~1380-step eval (see the
    off-by-one note in train_sac.py) must not be pooled with the step-0 point, or a
    single seed appears to have a non-zero spread at initialisation.
    """
    if env_steps == 0:
        return 0
    for nominal in (138000, 276000, 414000, 552000, 690000):
        if abs(env_steps - nominal) <= tol:
            return nominal
    return env_steps


def across_seeds(runs_by_seed):
    """{bucket: {metric: [value per seed]}}"""
    agg = {}
    for _seed, run in sorted(runs_by_seed.items()):
        for e in deterministic_curve(run):
            b = agg.setdefault(bucket(e["env_steps"]), {})
            for m in STAGES + ["mean_return", "mean_disp_m", "mean_max_lift_m"]:
                if m in e:
                    b.setdefault(m, []).append(e[m])
    return agg


def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return None
    m = sum(xs) / n
    if n == 1:
        return {"mean": round(m, 4), "sd": None, "n_seeds": 1, "values": xs}
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return {"mean": round(m, 4), "sd": round(var ** 0.5, 4), "n_seeds": n,
            "values": xs}


def interactions_to(runs_by_seed, metric, thresh):
    """Per-seed online interactions to first exceed ``thresh``; None if never."""
    out = {}
    for seed, run in sorted(runs_by_seed.items()):
        hit = None
        for e in deterministic_curve(run):
            if e.get(metric, 0.0) > thresh:
                hit = e["env_steps"]
                break
        out[seed] = hit
    return out


def plots(out_dir, agg, sft, dual=None, eff=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    colors = {"SAC": "tab:blue", "RLPD": "tab:orange"}

    def series(a, metric):
        xs = sorted(a)
        ys = [mean_sd(a[x][metric])["mean"] for x in xs if metric in a[x]]
        sds = [(mean_sd(a[x][metric])["sd"] or 0.0) for x in xs if metric in a[x]]
        return [x for x in xs if metric in a[x]], ys, sds

    def draw(ax, metric, title):
        for algo, a in agg.items():
            if not a:
                continue
            xs, ys, sds = series(a, metric)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=sds, marker="o", capsize=3,
                        color=colors.get(algo), label=algo)
        if sft and metric in sft:
            ax.axhline(sft[metric], ls="--", color="grey", label="SFT (no RL)")
        ax.set_title(title)
        ax.set_xlabel("online environment interactions")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Plot 1: headline success
    f1, a1 = plt.subplots(figsize=(7, 4.5))
    draw(a1, "full_success_rate", "Full task success vs online interactions")
    f1.tight_layout(); f1.savefig(os.path.join(out_dir, "plot1_success.png"), dpi=130)

    # Plot 2: every stage
    f2, ax2 = plt.subplots(2, 2, figsize=(12, 8))
    for ax, m, t in zip(ax2.ravel(), STAGES[:4],
                        ["reach", "grasp", "lift", "plate"]):
        draw(ax, m, t)
    f2.suptitle("Stage rates vs online interactions (mean +/- sd across training seeds)")
    f2.tight_layout(); f2.savefig(os.path.join(out_dir, "plot2_stages.png"), dpi=130)

    # Plot 3: physical progress
    f3, ax3 = plt.subplots(1, 2, figsize=(11, 4))
    draw(ax3[0], "mean_disp_m", "mean piston displacement (m)")
    draw(ax3[1], "mean_max_lift_m", "mean max lift (m)")
    f3.tight_layout(); f3.savefig(os.path.join(out_dir, "plot3_physical.png"), dpi=130)

    # Plot 4: deterministic vs stochastic execution (from eval_checkpoint.py outputs)
    if dual:
        f4, ax4 = plt.subplots(1, 2, figsize=(11, 4))
        metrics = ["grasp_rate", "lift_rate", "full_success_rate"]
        labels = ["grasp", "lift", "success"]
        width = 0.35
        for ax, (algo, dd) in zip(ax4, sorted(dual.items())):
            xs = range(len(metrics))
            det = [dd["modes"]["deterministic"][m] for m in metrics]
            sto = [dd["modes"]["stochastic"][m] for m in metrics]
            ax.bar([x - width / 2 for x in xs], det, width, label="deterministic")
            ax.bar([x + width / 2 for x in xs], sto, width, label="stochastic")
            ax.set_xticks(list(xs)); ax.set_xticklabels(labels)
            ax.set_title(f"{algo}: execution mode (std={dd.get('action_std_active_mean', 0):.3f})")
            ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
        f4.suptitle("Deterministic vs stochastic execution on identical conditions")
        f4.tight_layout()
        f4.savefig(os.path.join(out_dir, "plot4_exec_mode.png"), dpi=130)

    # Plot 5: RLPD sample-efficiency gain over SAC
    if agg.get("SAC") and agg.get("RLPD"):
        f5, a5 = plt.subplots(figsize=(7, 4.5))
        names, gains = [], []
        for name, metric, th in THRESHOLDS:
            s = [v for v in eff[name]["SAC"].values() if v is not None]
            r = [v for v in eff[name]["RLPD"].values() if v is not None]
            if not s or not r:
                continue
            names.append(name)
            # positive => RLPD needed FEWER interactions
            gains.append((sum(s) / len(s) - sum(r) / len(r)) / 1000.0)
        if names:
            colors_b = ["tab:orange" if g > 0 else "tab:blue" for g in gains]
            a5.barh(names, gains, color=colors_b)
            a5.axvline(0, color="black", lw=1)
            a5.set_xlabel("thousand interactions SAC needed minus RLPD\n"
                          "(positive = RLPD reached the threshold earlier)")
            a5.set_title("Sample-efficiency gain from the demonstration prior")
            a5.grid(alpha=0.3, axis="x")
        f5.tight_layout()
        f5.savefig(os.path.join(out_dir, "plot5_sample_efficiency.png"), dpi=130)
    return True


def main():
    out_dir, runs_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    sft = None
    if "--sft" in sys.argv:
        with open(sys.argv[sys.argv.index("--sft") + 1]) as f:
            s = json.load(f)
        if s.get("evals"):
            sft = s["evals"][0]

    dual = {}
    for dp in sorted(glob.glob(os.path.join(runs_dir, "dualmode_*.json"))):
        try:
            with open(dp) as f:
                dd = json.load(f)
        except Exception:
            continue
        if dd.get("modes", {}).get("stochastic"):
            algo = os.path.basename(dp)[len("dualmode_"):].split("_")[0].upper()
            dual[algo] = dd

    runs = load_runs(runs_dir)
    agg = {algo: across_seeds(by_seed) for algo, by_seed in runs.items()}

    report = {
        "seeds_per_method": {a: sorted(b) for a, b in runs.items()},
        "note": ("stage rates are mean +/- sd ACROSS TRAINING SEEDS; the per-run Wilson "
                 "intervals in each run's own json measure environment-condition "
                 "variability and are a different quantity"),
        "sft_baseline": ({k: sft[k] for k in STAGES + ["mean_return", "n_eval_episodes"]}
                         if sft else None),
        "curves": {a: {str(k): {m: mean_sd(v) for m, v in b.items()}
                       for k, b in sorted(x.items())} for a, x in agg.items()},
        "dual_mode": {a: {"gap": d.get("gap"),
                          "action_std_active_mean": d.get("action_std_active_mean"),
                          "deterministic": {k: d["modes"]["deterministic"][k] for k in STAGES},
                          "stochastic": {k: d["modes"]["stochastic"][k] for k in STAGES}}
                      for a, d in dual.items()},
        "sample_efficiency": {
            name: {algo: interactions_to(runs[algo], metric, th)
                   for algo in ("SAC", "RLPD")}
            for name, metric, th in THRESHOLDS
        },
    }
    with open(os.path.join(out_dir, "seed_comparison.json"), "w") as f:
        json.dump(report, f, indent=2)

    for algo in ("SAC", "RLPD"):
        if not runs[algo]:
            continue
        print(f"--- {algo} (seeds {sorted(runs[algo])}) ---")
        for k in sorted(agg[algo]):
            b = agg[algo][k]
            g, li, sc = (mean_sd(b.get(m, [])) for m in
                         ("grasp_rate", "lift_rate", "full_success_rate"))
            fmt = lambda d: ("  n/a" if d is None else
                             f"{d['mean']:.2f}" + (f"+/-{d['sd']:.2f}" if d["sd"] is not None else ""))
            print(f"  steps={k:>7}  grasp {fmt(g)}  lift {fmt(li)}  succ {fmt(sc)}")
    print()
    print("sample efficiency (online interactions to threshold, per seed):")
    for name, _m, _t in THRESHOLDS:
        print(f"  {name:<20} SAC {report['sample_efficiency'][name]['SAC']}"
              f"  RLPD {report['sample_efficiency'][name]['RLPD']}")
    print("\nplots:", "written" if plots(out_dir, agg, sft, dual, report["sample_efficiency"])
          else "matplotlib unavailable")


if __name__ == "__main__":
    main()
