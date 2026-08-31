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
    actions  (N, 30, 30) float32   the PHYSICAL action chunk executed (matching the
                                   shipped buffer's convention -- verified, not assumed)
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
REWARD_V3 = os.environ.get("REWARD_V3", "0") == "1"
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"
STATUS = os.environ.get("OUTF", os.path.join(OUTDIR, "_build_status.json"))
#: Action chunk length, matching the policy's horizon and the shipped buffer layout.
H = 30

res = {"_status": "RUNNING",
       "reward_version": ("v3_review_fixed" if REWARD_V3
                          else "v2_functional" if REWARD_V2 else "v1"),
       "has_critic_state": True,
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
    RW3 = _load("g1r3", RL + "g1_piston_reward_v3.py") if REWARD_V3 else None
    CST = _load("g1cs", RL + "g1_piston_critic_state.py")

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
    reward_fn = (RW3.PistonTaskRewardV3(sc, jn) if REWARD_V3
                 else RW2.PistonTaskRewardV2(sc, jn) if REWARD_V2
                 else RW.PistonTaskReward(sc, jn))
    # Record the privileged critic state alongside each transition, so a run with
    # CRITIC_STATE=1 can use RLPD's offline half instead of being blocked by it.
    # max_chunks normalises episode_progress. It MUST match the trainer's EP_CHUNKS, or
    # demonstration transitions carry a different progress scale than online ones and the
    # critic sees two conventions for one feature. H is the chunk LENGTH (30), not the
    # episode length -- using it here was a latent mismatch.
    EP_CHUNKS = int(os.environ.get("EP_CHUNKS", "23"))
    csb = CST.CriticStateBuilder(sc, max_chunks=EP_CHUNKS)

    STATS = ("/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/"
             "dataset_statistics.json")
    nrm = Q99.from_dataset_statistics(STATS)

    import glob

    # SRCDIR may be either a directory of act_ep*.npy recordings or an existing demo
    # buffer of ep*.npz files. This matters: only 3 of the 22 EXECUTABLE episodes have
    # a local act_ep recording, so sourcing from act_ep*.npy alone would rebuild an
    # 11-episode buffer, 8 of them non-executable failures, and would silently drop
    # ep46 -- the only demonstration of a genuine grasped press in the whole dataset.
    npz_src = sorted(glob.glob(os.path.join(SRCDIR, "ep*.npz")))
    npy_src = sorted(glob.glob(os.path.join(SRCDIR, "act_ep*.npy")))
    if npz_src:
        SOURCE = "buffer"
        avail = {int(os.path.basename(f)[2:5]): f for f in npz_src}
    elif npy_src:
        SOURCE = "recordings"
        avail = {int(os.path.basename(f).split("act_ep")[1].split(".npy")[0]): f
                 for f in npy_src}
    else:
        raise SystemExit(f"no ep*.npz or act_ep*.npy found in {SRCDIR}")
    if os.environ.get("EPISODES"):
        eps = [int(x) for x in os.environ["EPISODES"].split(",") if x.strip()]
        missing = [e for e in eps if e not in avail]
        if missing:
            raise SystemExit(f"requested episodes absent from {SRCDIR}: {missing}")
    else:
        eps = sorted(avail)
    res["source_kind"] = SOURCE
    res["source_dir"] = SRCDIR
    res["requested_episodes"] = eps
    emit()

    def grab():
        return sc["front_camera"].data.output["rgb"][0].cpu().numpy().copy()[..., :3]

    for ep in eps:
        if SOURCE == "buffer":
            # Stored chunks are PHYSICAL (verified); flatten back to a step sequence.
            _a = np.load(avail[ep])["actions"]
            A = _a.reshape(-1, _a.shape[-1])
        else:
            A = np.load(avail[ep])                                  # (T, 30) physical
        env.reset(seed=0)
        reward_fn.reset()
        imgs, acts, rews, states = [], [], [], []
        csb.reset()
        n_chunks = len(A) // H
        stages_seen = {}
        max_press = 0.0
        for c in range(n_chunks):
            chunk = A[c * H:(c + 1) * H]                            # (H, 30) physical
            imgs.append(grab().astype(np.uint8))
            states.append(csb.build(stages=stages_seen, chunk=c))
            # Store the PHYSICAL chunk. Verified against the shipped buffer: its
            # ep000.npz actions match act_ep0.npy exactly (max deviation 0.0000),
            # whereas treating them as normalized and denormalizing deviates by up to
            # 0.86 rad. A rebuilt buffer must use the same convention or every RLPD
            # offline sample is silently wrong.
            acts.append(np.asarray(chunk, dtype=np.float32))
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
        states.append(csb.build(stages=stages_seen, chunk=n_chunks))
        acts.append(acts[-1])
        rews.append(np.float32(0.0))
        out = os.path.join(OUTDIR, f"ep{ep:03d}.npz")
        np.savez_compressed(out, images=np.stack(imgs),
                            actions=np.stack(acts), rewards=np.stack(rews),
                            critic_state=np.stack(states).astype(np.float32))
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
