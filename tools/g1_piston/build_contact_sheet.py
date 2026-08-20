"""Contact sheet + final manifest, built only from verified rollouts.

Panels are drawn from the rendered videos, and every caption is taken from the sidecar's
recomputed label -- never from a summary JSON and never from a hand-written description.
A rollout that failed ``verify_rollouts.py`` is skipped and reported, not shown.

Usage:  python build_contact_sheet.py
"""
import json
import os
import sys

import numpy as np

V = "/home/jren313/research/starvla_rl/RLinf/verified_results"
VERIF = f"{V}/manifests/rollout_verification.json"


def frame_at(video, frac=0.72):
    """Pull one real frame from the rendered video (no synthesis)."""
    try:
        import imageio.v2 as imageio
        rd = imageio.get_reader(video)
        frames = [f for f in rd]
        rd.close()
        if not frames:
            return None
        return frames[min(int(len(frames) * frac), len(frames) - 1)]
    except Exception:
        return None


def label_of(r):
    lab = r["checks"].get("recomputed_label", {})
    for k in ("success", "carry", "throw", "lift", "grasp", "reach"):
        if lab.get(k):
            return k.upper() if k in ("success", "carry", "throw") else k
    return "no-contact"


def main():
    if not os.path.exists(VERIF):
        print(f"no verification manifest at {VERIF}; run verify_rollouts.py first")
        return 1
    data = json.load(open(VERIF))
    recs = [r for r in data["rollouts"] if r["verified"]]
    skipped = [r for r in data["rollouts"] if not r["verified"]]
    if not recs:
        print("no verified rollouts to draw")
        return 1

    # One panel per distinct behaviour class, preferring the strongest evidence:
    # a full success outranks a carry, which outranks a throw, and so on.
    rank = {"SUCCESS": 0, "CARRY": 1, "THROW": 2, "lift": 3, "grasp": 4, "reach": 5,
            "no-contact": 6}
    chosen, seen = [], set()
    for r in sorted(recs, key=lambda r: (rank.get(label_of(r), 9),
                                         -(r["stored_row"]["return"]))):
        key = (r["method"], label_of(r))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(r)
    chosen = chosen[:12]

    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("PIL unavailable; cannot build a contact sheet")
        return 1

    panels = []
    for r in chosen:
        f = frame_at(r["video"])
        if f is None:
            continue
        panels.append((r, f))
    if not panels:
        print("no frames could be read from the verified videos")
        return 1

    cols = min(4, len(panels))
    rows = (len(panels) + cols - 1) // cols
    ph, pw = panels[0][1].shape[0], panels[0][1].shape[1]
    cap = 46
    sheet = Image.new("RGB", (cols * pw, rows * (ph + cap)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (r, f) in enumerate(panels):
        x, y = (i % cols) * pw, (i // cols) * (ph + cap)
        sheet.paste(Image.fromarray(f.astype(np.uint8)), (x, y))
        lines = [
            f"{r['method']} seed {r['seed']}  {label_of(r)}",
            f"{r['checkpoint_env_steps']:,} steps  cond {r['condition_index']} "
            f"{r['condition_hash'][:8]}  {r['mode'][:4]}",
            f"ret {r['stored_row']['return']:.2f}  disp "
            f"{r['stored_row']['disp_m']:.3f}m  lift {r['stored_row']['max_lift_m']:.3f}m",
        ]
        d.rectangle([x, y + ph, x + pw, y + ph + cap], fill=(20, 20, 20))
        for j, t in enumerate(lines):
            d.text((x + 5, y + ph + 4 + 13 * j), t, fill=(255, 255, 255))

    os.makedirs(f"{V}/contact_sheets", exist_ok=True)
    dest = f"{V}/contact_sheets/verified_contact_sheet.png"
    sheet.save(dest)
    print(f"wrote {dest}  ({len(panels)} verified panels)")

    manifest = {
        "metrics_version": data.get("metrics_version"),
        "n_rollouts_rendered": data["n_rollouts"],
        "n_verified": data["n_verified"], "n_failed": data["n_failed"],
        "contact_sheet": dest,
        "panels": [{"video": r["video"], "method": r["method"], "seed": r["seed"],
                    "checkpoint": r["checkpoint"],
                    "checkpoint_env_steps": r["checkpoint_env_steps"],
                    "condition_index": r["condition_index"],
                    "condition_hash": r["condition_hash"], "mode": r["mode"],
                    "label": label_of(r), "metrics": r["stored_row"],
                    "verified": True} for r, _ in panels],
        "skipped_unverified": [{"video": r["video"], "failures": r["failures"]}
                               for r in skipped],
    }
    os.makedirs(f"{V}/manifests", exist_ok=True)
    json.dump(manifest, open(f"{V}/manifests/verified_visual_results.json", "w"),
              indent=2)
    print(f"wrote {V}/manifests/verified_visual_results.json")
    for r in skipped:
        print(f"  SKIPPED (unverified) {os.path.basename(r['video'])}: {r['failures']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
