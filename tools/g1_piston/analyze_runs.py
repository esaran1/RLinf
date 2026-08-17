"""Compare SFT / SAC / RLPD on the fixed behavioural evaluation suite.

Selection is on behaviour -- stage rates, full success, return, displacement -- not on
optimizer statistics. Loss curves are plotted only as a debugging aid and are explicitly
labelled as such.

Usage:
    python analyze_runs.py <out_dir> <sft.json> <sac.json> [rlpd.json]
"""

from __future__ import annotations

import json
import os
import sys

STAGE_KEYS = [
    ("reach_rate", "reach"),
    ("grasp_rate", "grasp"),
    ("lift_rate", "lift"),
    ("tube_rate", "tube"),
    ("plate_rate", "plate"),
    ("full_success_rate", "success"),
]


def load(path):
    with open(path) as f:
        return json.load(f)


def curve(run):
    """Behavioural evaluations in interaction order."""
    return sorted(run.get("evals", []), key=lambda e: e["env_steps"])


def summarize(name, run):
    evs = curve(run)
    if not evs:
        return {"algo": name, "status": run.get("_status"), "evals": 0}
    best = max(evs, key=lambda e: (e["full_success_rate"], e["mean_return"]))
    final = evs[-1]
    tot = run.get("totals", {})
    return {
        "algo": name,
        "status": run.get("_status"),
        "n_evals": len(evs),
        "final": {k: final[k] for k, _ in STAGE_KEYS},
        "final_mean_return": final["mean_return"],
        "final_env_steps": final["env_steps"],
        "best_by_behaviour": {
            "env_steps": best["env_steps"],
            **{k: best[k] for k, _ in STAGE_KEYS},
            "mean_return": best["mean_return"],
        },
        "env_interactions": tot.get("env_steps"),
        "gradient_updates": tot.get("grad_updates"),
        "online_samples": tot.get("n_online_samples"),
        "demo_samples": tot.get("n_demo_samples"),
        "demo_fraction": tot.get("demo_fraction"),
        "wall_clock_s": tot.get("wall_clock_s"),
        "gpu_hours": tot.get("gpu_hours"),
        "peak_vram_mib": tot.get("peak_vram_mib"),
        "replay_buffer": tot.get("buffer_size"),
    }


def text_curve(name, run):
    """A plain-text learning curve, so the result survives without a plot viewer."""
    lines = [f"--- {name} ---",
             f"{'env_steps':>10} {'succ':>5} {'reach':>6} {'grasp':>6} "
             f"{'lift':>5} {'plate':>6} {'return':>8} {'disp_m':>7}"]
    for e in curve(run):
        lines.append(
            f"{e['env_steps']:>10} {e['full_success_rate']:>5.2f} {e['reach_rate']:>6.2f} "
            f"{e['grasp_rate']:>6.2f} {e['lift_rate']:>5.2f} {e['plate_rate']:>6.2f} "
            f"{e['mean_return']:>8.3f} {e['mean_disp_m']:>7.4f}")
    return "\n".join(lines)


def plot(out_dir, runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    panels = [("full_success_rate", "full success"), ("grasp_rate", "grasp"),
              ("lift_rate", "lift"), ("reach_rate", "reach"),
              ("mean_return", "mean return"), ("mean_disp_m", "piston displacement (m)")]
    for ax, (key, title) in zip(axes.ravel(), panels):
        for name, run in runs.items():
            evs = curve(run)
            if not evs:
                continue
            xs = [e["env_steps"] for e in evs]
            ys = [e[key] for e in evs]
            if name == "SFT":
                # SFT gets no interaction budget: draw it as a reference line.
                ax.axhline(ys[-1], ls="--", color="grey", label="SFT (no RL)")
            else:
                ax.plot(xs, ys, marker="o", label=name)
        ax.set_title(title)
        ax.set_xlabel("online env interactions")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("G1 piston: behavioural evaluation on fixed seeds "
                 "(selection metric -- not optimizer statistics)")
    fig.tight_layout()
    p = os.path.join(out_dir, "behaviour_curves.png")
    fig.savefig(p, dpi=120)

    # Optimizer statistics, explicitly secondary.
    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))
    for name, run in runs.items():
        tl = run.get("train_log", [])
        if not tl:
            continue
        xs = [m["env_steps"] for m in tl]
        ax2[0].plot(xs, [m["critic_loss"] for m in tl], label=name)
        ax2[1].plot(xs, [m["alpha"] for m in tl], label=name)
        ax2[2].plot(xs, [m["logprob"] for m in tl], label=name)
    for a, t in zip(ax2, ["critic loss", "alpha", "chunk log-prob"]):
        a.set_title(t + " (debug only)")
        a.set_xlabel("online env interactions")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "optimizer_debug.png"), dpi=120)
    return p


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    runs = {}
    labels = ["SFT", "SAC", "RLPD"]
    for label, path in zip(labels, sys.argv[2:]):
        if os.path.exists(path):
            runs[label] = load(path)

    report = {
        "selection_criterion": "behavioural metrics on the fixed eval suite; optimizer "
                               "statistics are debugging only",
        "eval_suite": {
            "seeds": runs.get("SAC", runs.get("SFT", {})).get("config", {}).get("eval_seeds"),
            "deterministic_policy": True,
        },
        "summaries": [summarize(k, v) for k, v in runs.items()],
    }
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(report, f, indent=2)

    text = "\n\n".join(text_curve(k, v) for k, v in runs.items())
    with open(os.path.join(out_dir, "curves.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print()
    for s in report["summaries"]:
        print(json.dumps(s, indent=1))
    p = plot(out_dir, runs)
    print("\nplots:", p or "matplotlib unavailable")


if __name__ == "__main__":
    main()
