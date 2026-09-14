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

"""Assemble the behaviour-cloning buffer for the pressing policy, plus its normaliser.

Inputs are (a) the certified synthetic episodes written by ``make_press_demos.py`` (human
transport up to the plate stage + scripted dispense) from one or more directories, and
(b) optionally the original executable demonstrations, TRUNCATED at their plate chunk so
"arrive over the plate" is never paired with "hold still" in one place and "start the
dispense" in another. Files are hard-linked into ``OUT`` under unique names.

The normaliser: the SFT ``dataset_statistics.json`` spans only the demonstrations, in which
the left arm never moves; its q01/q99 on the left-arm dims (0-6) are 0.1-0.2 rad wide and
the dispense primitive moves them by up to 2.2 rad, which the squashed head could not
express (measured: normalised values of -24 to +19 against a +/-2.2 squash). This tool
writes ``OUT/dataset_statistics.json``: the SFT file with dims 0-6 widened to the buffer's
actual range plus a margin; every other dim is left bit-identical. The trained checkpoint
records this path and every evaluator maps through it (``NORM_STATS``).

Environment:
    OUT           destination directory (required)
    SYNTH_DIRS    comma-separated directories of certified synthetic episodes (required)
    ORIG_DIR      original buffer (default demo_buffer_v3); "" to exclude
    ORIG_EPISODES comma-separated ids (default: the 16 lifting episodes)
    MARGIN_RAD    range margin on the widened dims (default 0.05)
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

OUT = os.environ["OUT"]
SYNTH_DIRS = [d for d in os.environ["SYNTH_DIRS"].split(",") if d.strip()]
ORIG_DIR = os.environ.get("ORIG_DIR", "/home/jren313/research/starvla_rl/demo_buffer_v3")
ORIG_EPISODES = [int(x) for x in os.environ.get("ORIG_EPISODES", "0,4,5,39,40,43,46,47,48,49,50,51,52,53,54,59").split(",") if x.strip()]
MARGIN = float(os.environ.get("MARGIN_RAD", "0.05"))
BASE_STATS = "/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/dataset_statistics.json"
WIDEN_DIMS = list(range(0, 7))
#: cumulative v3 return at which the plate stage has certainly fired (reach 1 + grasp 2 +
#: lift 4 + plate 8 = 15 in bonuses; before the plate the cumulative stays below ~10).
PLATE_CUM_RETURN = 14.0

os.makedirs(OUT, exist_ok=True)
manifest = {"out": OUT, "synth_dirs": SYNTH_DIRS, "orig_dir": ORIG_DIR, "episodes": [], "widen_dims": WIDEN_DIMS}


def link(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


all_actions = []
n = 0
for d in SYNTH_DIRS:
    status = json.load(open(os.path.join(d, "_build_status.json")))
    certified = {e["file"]: e for e in status["episodes"] if e.get("outcome") == "press_certified" and e.get("file")}
    for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
        if f not in certified:
            continue
        z = np.load(f)
        dst = os.path.join(OUT, f"ep{n:03d}.npz"); link(f, dst); n += 1
        all_actions.append(z["actions"][:-1].reshape(-1, 30))
        e = certified[f]
        manifest["episodes"].append({"file": dst, "source": f, "kind": "synthetic_dispense", "episode": e["episode"],
                                     "cond": e["cond"], "n_chunks": int(len(z["actions"]) - 1),
                                     "synthetic_from_chunk": e.get("synthetic_from_chunk"), "return_v5": e.get("return_v5"),
                                     "press_max_m": e.get("press_max_m"), "dispense_steps": e.get("max_dispense_steps")})
n_synth = n
if ORIG_DIR:
    for ep in ORIG_EPISODES:
        f = os.path.join(ORIG_DIR, f"ep{ep:03d}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        rew = np.asarray(z["rewards"], dtype=np.float64)
        cum = np.cumsum(rew)
        hit = np.nonzero(cum >= PLATE_CUM_RETURN)[0]
        if len(hit) == 0:
            manifest["episodes"].append({"source": f, "kind": "original_skipped_no_plate", "episode": ep}); continue
        pc = int(hit[0])
        keep = pc + 1                       # chunks 0..pc, then the trailing observation
        imgs = z["images"][:keep + 1]; acts = np.concatenate([z["actions"][:keep], z["actions"][pc:pc + 1]])
        rews = np.concatenate([z["rewards"][:keep], np.zeros(1, dtype=np.float32)])
        cs = z["critic_state"][:keep + 1] if "critic_state" in z else None
        dst = os.path.join(OUT, f"ep{n:03d}.npz")
        kw = {"images": imgs, "actions": acts.astype(np.float32), "rewards": rews.astype(np.float32)}
        if cs is not None:
            kw["critic_state"] = cs
        np.savez_compressed(dst, **kw); n += 1
        all_actions.append(acts[:-1].reshape(-1, 30))
        manifest["episodes"].append({"file": dst, "source": f, "kind": "original_truncated_at_plate", "episode": ep,
                                     "plate_chunk": pc, "n_chunks": keep, "orig_chunks": int(len(z["actions"]) - 1)})

A = np.concatenate(all_actions) if all_actions else np.zeros((0, 30))
base = json.load(open(BASE_STATS))
stats = json.loads(json.dumps(base))
key = "new_embodiment"
q01 = list(stats[key]["action"]["q01"]); q99 = list(stats[key]["action"]["q99"])
widened = {}
for d in WIDEN_DIMS:
    dmin, dmax = float(A[:, d].min()), float(A[:, d].max())
    # widen ONLY where the buffer actually exceeds the SFT range; a primitive that does not
    # move the left arm (the adopted palm press) keeps the normaliser bit-identical, so the
    # pressing policy and the working BC policy map their outputs through the same file
    lo = min(dmin - MARGIN, q01[d]) if dmin < q01[d] else q01[d]
    hi = max(dmax + MARGIN, q99[d]) if dmax > q99[d] else q99[d]
    if lo != q01[d] or hi != q99[d]:
        widened[d] = {"q01_old": q01[d], "q99_old": q99[d], "q01": lo, "q99": hi}
    q01[d], q99[d] = lo, hi
stats[key]["action"]["q01"] = q01; stats[key]["action"]["q99"] = q99
for f_ in ("min", "max"):
    if f_ in stats[key]["action"]:
        v = list(stats[key]["action"][f_])
        for d in WIDEN_DIMS:
            v[d] = min(v[d], q01[d]) if f_ == "min" else max(v[d], q99[d])
        stats[key]["action"][f_] = v
stats["_provenance"] = {"base": BASE_STATS, "widened_dims": widened, "margin_rad": MARGIN, "buffer": OUT,
                        "note": ("dims 0-6 (left arm) widened where the pressing buffer exceeds the SFT range; all other dims identical"
                                 if widened else "IDENTICAL to the SFT statistics: the buffer stays inside its range on every dim")}
json.dump(stats, open(os.path.join(OUT, "dataset_statistics.json"), "w"), indent=1)
manifest.update({"n_synthetic": n_synth, "n_original": n - n_synth, "n_episodes": n, "n_action_rows": int(len(A)),
                 "widened": widened, "stats_file": os.path.join(OUT, "dataset_statistics.json")})
json.dump(manifest, open(os.path.join(OUT, "_manifest.json"), "w"), indent=1)
print(json.dumps({k: manifest[k] for k in ("n_synthetic", "n_original", "n_episodes", "n_action_rows")}))
for d, w in widened.items():
    print(f"dim {d}: q01 {w['q01_old']:+.3f}->{w['q01']:+.3f}  q99 {w['q99_old']:+.3f}->{w['q99']:+.3f}")
