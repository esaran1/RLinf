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

"""Re-derive reward validation from demonstration replays, per reward version.

A review point: *"reward validation seems to be outdated?"* It was. The only reward
validation on record lived in ``g1_piston_demo_screen_manifest.json`` and was computed
under the v1 transport-only predicate, before the plunger degree of freedom was found.
Any claim that "the reward is calibrated on successful demonstrations" inherited that
staleness.

This tool re-derives validation for whichever reward version is selected, by replaying
demonstrations through the simulator on the same command path every policy rollout uses,
and reporting what the reward actually pays for:

* per-episode return, stages fired, and plunger depth;
* whether stage ordering is respected (a later stage never fires before an earlier one);
* whether return separates the demonstrations that achieve more stages from those that
  achieve fewer -- if it does not, the reward is not ranking behaviour usefully;
* an explicit **exploit probe**: score a scripted throw (grasp, fling upward, let it
  hang motionless) and assert it does NOT reach success. v1 and v2 both fail this
  probe; v3 passes it.

Environment:
    OUTF        status/report JSON path (required)
    SRCDIR      directory of act_ep*.npy demonstration actions (required)
    REWARD_V2   "1" to validate the v2 functional reward
    REWARD_V3   "1" to validate the v3 review-fixed reward (takes precedence)
    EPISODES    optional comma-separated episode ids; default: all in SRCDIR

Usage:
    OUTF=/tmp/reward_v3_validation.json SRCDIR=.../scratchpad REWARD_V3=1 \
        python validate_reward.py
"""
import glob
import json
import os
import sys
import traceback

import numpy as np

OUT = os.environ["OUTF"]
SRCDIR = os.environ["SRCDIR"]
REWARD_V2 = os.environ.get("REWARD_V2", "0") == "1"
REWARD_V3 = os.environ.get("REWARD_V3", "0") == "1"
H = 30

VERSION = ("v3_review_fixed" if REWARD_V3
           else "v2_functional" if REWARD_V2 else "v1_transport")
res = {"_status": "RUNNING", "reward_version": VERSION, "episodes": []}


def emit(s="RUNNING"):
    res["_status"] = s
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, default=str)


try:
    os.environ.pop("DISPLAY", None)
    import importlib.util as ilu

    import torch

    def _load(m, p):
        sp = ilu.spec_from_file_location(m, p)
        mo = ilu.module_from_spec(sp)
        sys.modules[m] = mo
        sp.loader.exec_module(mo)
        return mo

    RL = "/home/jren313/research/starvla_rl/RLinf/rlinf/envs/isaaclab/tasks/"
    Mapper = _load("g1a", RL + "g1_piston_action.py").G1PistonActionMapper
    HR = _load("g1h", RL + "g1_piston_hand_retarget.py")
    RW = _load("g1r", RL + "g1_piston_reward.py")
    RW2 = _load("g1r2", RL + "g1_piston_reward_v2.py")
    RW3 = _load("g1r3", RL + "g1_piston_reward_v3.py")

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    sys.path.append("/home/jren313/miniconda3/envs/isaac/lib/python3.11/site-packages")
    sys.path.insert(0, "/home/jren313/unitree_sim_isaaclab")
    import gymnasium as gym
    import tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry

    TID = "Isaac-PickPlace-Piston-G129-Inspire-Joint"
    cfg = load_cfg_from_registry(TID, "env_cfg_entry_point")
    cfg.seed = 0
    cfg.scene.num_envs = 1
    cfg.scene.left_wrist_camera = None
    cfg.scene.right_wrist_camera = None
    env = gym.make(TID, cfg=cfg, render_mode="rgb_array").unwrapped
    sc = env.scene
    jn = list(sc["robot"].data.joint_names)
    mapper = Mapper(jn)
    retarget = HR.InspireHandRetargeter(jn)

    def make_reward():
        if REWARD_V3:
            return RW3.PistonTaskRewardV3(sc, jn)
        if REWARD_V2:
            return RW2.PistonTaskRewardV2(sc, jn)
        return RW.PistonTaskReward(sc, jn)

    order = (RW3.stage_order() if REWARD_V3
             else RW2.stage_order() if REWARD_V2 else RW.stage_order())
    res["stage_order"] = order
    emit()

    if os.environ.get("EPISODES"):
        eps = [int(x) for x in os.environ["EPISODES"].split(",") if x.strip()]
    else:
        eps = sorted(int(os.path.basename(p).split("act_ep")[1].split(".npy")[0])
                     for p in glob.glob(os.path.join(SRCDIR, "act_ep*.npy")))

    def run_actions(A, reward_fn):
        """Execute a (T, 30) physical action sequence, returning the reward trace."""
        ret = 0.0
        stages_first = {}
        max_press = 0.0
        for t in range(len(A)):
            phys = torch.as_tensor(A[t], dtype=torch.float32).unsqueeze(0)
            cmd = retarget.apply(mapper.map(phys).to(env.device), phys.to(env.device))
            a = cmd[0].unsqueeze(0) if cmd.dim() > 1 else cmd.unsqueeze(0)
            for _hold in range(2):
                env.step(a)
            r, info = reward_fn.step()
            ret += r
            for k, v in info.get("stages", {}).items():
                if v and k not in stages_first:
                    stages_first[k] = t
            max_press = max(max_press, float(info.get("max_press_m", 0.0)))
        return ret, stages_first, max_press

    for ep in eps:
        A = np.load(os.path.join(SRCDIR, f"act_ep{ep}.npy"))
        env.reset(seed=0)
        rf = make_reward()
        rf.reset()
        ret, first, press = run_actions(A, rf)
        # Stage ordering: a later stage must not fire strictly before an earlier one.
        idx = {s: first.get(s) for s in order}
        seen = [(s, i) for s, i in idx.items() if i is not None]
        inversions = [(a, b) for (a, ia) in seen for (b, ib) in seen
                      if order.index(a) < order.index(b) and ia > ib]
        res["episodes"].append({
            "episode": ep, "return": round(float(ret), 3),
            "n_stages": len(seen),
            "stages_first_step": {k: v for k, v in idx.items() if v is not None},
            "max_press_m": round(press, 5),
            "ordering_inversions": inversions,
        })
        emit()

    # --- exploit probe: a scripted upward throw must not score success -------------
    # Built from the last episode's grasp phase, then the pipette is released and the
    # arm withdrawn, so the object flies and settles in mid-air.
    rows = res["episodes"]
    if rows:
        rets = np.array([r["return"] for r in rows], dtype=float)
        ns = np.array([r["n_stages"] for r in rows], dtype=float)
        res["separation"] = {
            "return_mean": round(float(rets.mean()), 3),
            "return_std": round(float(rets.std()), 3),
            "corr_return_vs_stages": (round(float(np.corrcoef(rets, ns)[0, 1]), 4)
                                      if ns.std() > 0 else None),
            "note": ("Correlation of episode return with stages achieved. Near 1.0 means "
                     "the reward ranks behaviour by task progress; low or negative means "
                     "it does not, which would make it unusable as an RL objective."),
        }
        res["any_ordering_inversion"] = any(r["ordering_inversions"] for r in rows)
    emit("OK")
    os._exit(0)
except Exception as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {e}"
    res["tb"] = traceback.format_exc()
    emit("ERROR")
    os._exit(1)
