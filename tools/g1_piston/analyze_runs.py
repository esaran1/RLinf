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


def first_reaching(run, stage_key, threshold=0.0):
    """Online interactions at which a stage first exceeds ``threshold``.

    This is the sample-efficiency statistic for the SAC-vs-RLPD comparison:
    interactions to first lift, to first success, to X% success.
    """
    for e in curve(run):
        if e.get(stage_key, 0.0) > threshold:
            return e["env_steps"]
    return None


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
        "n_eval_episodes": final.get("n_eval_episodes"),
        "interactions_to_first_grasp": first_reaching(run, "grasp_rate"),
        "interactions_to_first_lift": first_reaching(run, "lift_rate"),
        "interactions_to_first_success": first_reaching(run, "full_success_rate"),
        "interactions_to_20pct_success": first_reaching(run, "full_success_rate", 0.20),
        "interactions_to_50pct_success": first_reaching(run, "full_success_rate", 0.50),
        "final_ci95": final.get("ci95"),
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

    # Same behavioural metrics against wall-clock.
    fig3, ax3 = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, title) in zip(ax3, [("full_success_rate", "full success"),
                                      ("lift_rate", "lift"),
                                      ("mean_return", "mean return")]):
        for name, run in runs.items():
            evs = curve(run)
            if not evs or name == "SFT":
                continue
            ax.plot([e.get("wall_clock_s", 0) / 3600.0 for e in evs],
                    [e[key] for e in evs], marker="o", label=name)
        ax.set_title(title)
        ax.set_xlabel("wall-clock (hours)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "behaviour_vs_wallclock.png"), dpi=120)

    # Optimizer statistics, explicitly secondary.
    keys = [("critic_loss", "critic loss"), ("alpha", "alpha"),
            ("logprob_per_step", "log-prob per control action"),
            ("actor_q_term", "actor Q term"),
            ("actor_entropy_term", "actor entropy term"),
            ("entropy_to_q_ratio", "|alpha*logp| / |Q|")]
    fig2, ax2 = plt.subplots(2, 3, figsize=(16, 7))
    for ax, (key, title) in zip(ax2.ravel(), keys):
        for name, run in runs.items():
            tl = run.get("train_log", [])
            if not tl or key not in tl[0]:
                continue
            ax.plot([m["env_steps"] for m in tl], [m[key] for m in tl], label=name)
        ax.set_title(title + " (debug only)")
        ax.set_xlabel("online env interactions")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
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

    any_run = runs.get("SAC") or runs.get("RLPD") or runs.get("SFT") or {}
    report = {
        "selection_criterion": "behavioural metrics on the held-out initial-condition "
                               "suite; optimizer statistics are debugging only",
        "eval_suite": {
            "n_conditions": (any_run.get("reset_suite") or {}).get("n_eval"),
            "disjoint_from_train": (any_run.get("reset_suite") or {}).get("disjoint"),
            "reset_suite_seed": (any_run.get("config") or {}).get("reset_suite_seed"),
            "deterministic_policy": True,
            "note": "the task's own reset is deterministic; all variation comes from the "
                    "initial-condition suite (g1_piston_reset)",
        },
        "canonical_diagnostic": {
            k: v.get("canonical_diagnostic") for k, v in runs.items()
            if v.get("canonical_diagnostic")
        },
        "canonical_note": "single fixed reset; one binary observation per checkpoint, "
                          "never a success rate",
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
