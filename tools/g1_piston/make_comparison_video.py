"""Stack two verified rollouts side by side into one comparison video.

The point of the study is a contrast -- untrained policy completes the task, trained
policy grasps and stops -- and that reads far better as one file than as two the viewer
must alternate between.

Each panel keeps its own overlay (method, checkpoint, condition hash, live state), and a
banner names what the pair is showing. Frames come from the already-rendered, already-
verified videos; nothing is re-simulated, so the comparison inherits their verification.

Usage:
    python make_comparison_video.py LEFT.mp4 RIGHT.mp4 OUT.mp4 "Left caption" "Right caption"
"""
import json
import os
import sys

import numpy as np


def load(path):
    import imageio.v2 as imageio
    rd = imageio.get_reader(path)
    frames = [f for f in rd]
    rd.close()
    return frames


def sidecar(path):
    p = path[:-4] + ".json"
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    left, right, out = sys.argv[1], sys.argv[2], sys.argv[3]
    cap_l = sys.argv[4] if len(sys.argv) > 4 else ""
    cap_r = sys.argv[5] if len(sys.argv) > 5 else ""

    for p in (left, right):
        if not os.path.exists(p):
            print(f"missing: {p}")
            return 1
        if sidecar(p) is None:
            print(f"refusing {p}: no sidecar, so the rollout is unverified")
            return 1

    a, b = load(left), load(right)
    # Hold the shorter clip's last frame so both panels run the full length; an episode
    # that terminated early really did stop there.
    n = max(len(a), len(b))
    a += [a[-1]] * (n - len(a))
    b += [b[-1]] * (n - len(b))

    h = max(a[0].shape[0], b[0].shape[0])
    w = a[0].shape[1] + b[0].shape[1]
    banner = 22

    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("PIL unavailable")
        return 1

    frames = []
    for i in range(n):
        canvas = Image.new("RGB", (w, h + banner), (0, 0, 0))
        canvas.paste(Image.fromarray(a[i].astype(np.uint8)), (0, banner))
        canvas.paste(Image.fromarray(b[i].astype(np.uint8)),
                     (a[0].shape[1], banner))
        d = ImageDraw.Draw(canvas)
        d.text((6, 6), cap_l, fill=(255, 255, 255))
        d.text((a[0].shape[1] + 6, 6), cap_r, fill=(255, 255, 255))
        frames.append(np.asarray(canvas))

    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    try:
        imageio.mimsave(out, frames, fps=20, codec="libx264", quality=8)
    except Exception:
        imageio.mimsave(out, frames, fps=20)

    meta = {"output": out, "left": {"video": left, "sidecar": sidecar(left)["row"],
                                    "blend_steps": sidecar(left).get("blend_steps")},
            "right": {"video": right, "sidecar": sidecar(right)["row"],
                      "blend_steps": sidecar(right).get("blend_steps")},
            "n_frames": len(frames)}
    json.dump(meta, open(out[:-4] + ".json", "w"), indent=2)
    print(f"wrote {out}  ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
