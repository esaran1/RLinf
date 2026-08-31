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

"""Rebuild the RLPD demonstration buffer, scoring rewards with a chosen reward version.

Why this exists
---------------
RLPD fills half of every training batch from a demonstration buffer whose rewards are
**baked in at build time**. The shipped buffer was scored with the v1 (transport-only)
reward, so training against the v2 functional reward with that buffer would feed the
critic v1 targets on half of every batch -- teaching it that a trajectory which never
touches the plunger is worth full credit, and silently cancelling the v2 press signal.

This tool replays each demonstration through the simulator under the frozen
mapper+retargeter (the same command path as every policy rollout), scores it with the
requested reward version, and writes chunk-level transitions in the shipped format:

    images   (N, H, W, 3) uint8    observation at the start of each chunk
    actions  (N, 30, 30) float32   the normalized action chunk executed
    rewards  (N,) float32          summed reward over the chunk, under REWARD_V2

Environment:
    OUTDIR      destination directory (required; never overwrites in place by default)
    SRCDIR      directory of act_ep*.npy recorded demonstration actions (required)
    REWARD_V2   "1" to score with the functional pipette reward, else v1
    EPISODES    optional comma-separated episode ids; default: all found in SRCDIR
    OVERWRITE   "1" to allow writing into a non-empty OUTDIR

Usage:
    OUTDIR=.../demo_buffer_v2 SRCDIR=.../scratchpad REWARD_V2=1 \
        python build_demo_buffer.py
"""
import json
import os
import sys
import traceback

import numpy as np

OUTDIR = os.environ["OUTDIR"]
SRCDIR = os.environ["SRCDIR"]
REWARD_V2 = os.environ.get("REWARD_V2", "0") == "1"
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"
STATUS = os.environ.get("OUTF", os.path.join(OUTDIR, "_build_status.json"))
#: Action chunk length, matching the policy's horizon and the shipped buffer layout.
H = 30

res = {"_status": "RUNNING", "reward_version": "v2_functional" if REWARD_V2 else "v1",
       "episodes": []}


def emit(s="RUNNING"):
    res["_status"] = s
    os.makedirs(os.path.dirname(STATUS) or ".", exist_ok=True)
    with open(STATUS, "w") as f:
        json.dump(res, f, indent=2, default=str)


try:
    os.makedirs(OUTDIR, exist_ok=True)
    existing = [f for f in os.listdir(OUTDIR) if f.endswith(".npz")]
    if existing and not OVERWRITE:
        raise SystemExit(
            f"{OUTDIR} already holds {len(existing)} .npz files; set OVERWRITE=1 to "
            "rebuild. Refusing to silently mix buffers scored under different rewards.")

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
    Q99 = _load("g1n", RL + "g1_piston_norm.py").Q99ActionNormalizer
    Mapper = _load("g1a", RL + "g1_piston_action.py").G1PistonActionMapper
    HR = _load("g1h", RL + "g1_piston_hand_retarget.py")
    RW = _load("g1r", RL + "g1_piston_reward.py")
    RW2 = _load("g1r2", RL + "g1_piston_reward_v2.py") if REWARD_V2 else None

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
    reward_fn = (RW2.PistonTaskRewardV2(sc, jn) if REWARD_V2
                 else RW.PistonTaskReward(sc, jn))

    STATS = ("/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/"
             "dataset_statistics.json")
    nrm = Q99.from_dataset_statistics(STATS)

    import glob

    if os.environ.get("EPISODES"):
        eps = [int(x) for x in os.environ["EPISODES"].split(",") if x.strip()]
    else:
        eps = sorted(int(os.path.basename(p).split("act_ep")[1].split(".npy")[0])
                     for p in glob.glob(os.path.join(SRCDIR, "act_ep*.npy")))
    res["requested_episodes"] = eps
    emit()

    def grab():
        return sc["front_camera"].data.output["rgb"][0].cpu().numpy().copy()[..., :3]

    for ep in eps:
        A = np.load(os.path.join(SRCDIR, f"act_ep{ep}.npy"))       # (T, 30) physical
        env.reset(seed=0)
        reward_fn.reset()
        imgs, acts, rews = [], [], []
        n_chunks = len(A) // H
        stages_seen = {}
        max_press = 0.0
        for c in range(n_chunks):
            chunk = A[c * H:(c + 1) * H]                            # (H, 30) physical
            imgs.append(grab().astype(np.uint8))
            # Store the NORMALIZED chunk: that is the space the policy acts in, and the
            # space the shipped buffer used.
            acts.append(nrm.normalize(torch.as_tensor(chunk, dtype=torch.float32))
                        .numpy().astype(np.float32))
            total_r = 0.0
            for t in range(H):
                phys = torch.as_tensor(chunk[t], dtype=torch.float32).unsqueeze(0)
                cmd = retarget.apply(mapper.map(phys).to(env.device),
                                     phys.to(env.device))
                a = cmd[0].unsqueeze(0) if cmd.dim() > 1 else cmd.unsqueeze(0)
                for _hold in range(2):
                    env.step(a)
                r, info = reward_fn.step()
                total_r += r
                for k, v in info.get("stages", {}).items():
                    stages_seen[k] = stages_seen.get(k, False) or v
                max_press = max(max_press, float(info.get("max_press_m", 0.0)))
            rews.append(np.float32(total_r))
        # One trailing observation so (s, a, r, s') pairs can be formed for every chunk.
        imgs.append(grab().astype(np.uint8))
        acts.append(acts[-1])
        rews.append(np.float32(0.0))
        out = os.path.join(OUTDIR, f"ep{ep:03d}.npz")
        np.savez_compressed(out, images=np.stack(imgs),
                            actions=np.stack(acts), rewards=np.stack(rews))
        rec = {"episode": ep, "chunks": n_chunks, "return": round(float(sum(rews)), 3),
               "max_press_m": round(max_press, 5),
               "stages": {k: bool(v) for k, v in stages_seen.items()},
               "file": out}
        res["episodes"].append(rec)
        emit()

    res["n_files"] = len(res["episodes"])
    res["total_return"] = round(sum(e["return"] for e in res["episodes"]), 3)
    emit("OK")
    os._exit(0)
except SystemExit as e:
    res["error"] = str(e)
    emit("ERROR")
    raise
except Exception as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {e}"
    res["tb"] = traceback.format_exc()
    emit("ERROR")
    os._exit(1)
