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

"""Where is the rod top in the HAND frame when the thumb press works, and when it fails?

The thumb-yaw press reached 24.5 mm from ep40's end pose but 4-19 mm on other episodes
and other truncation points. The thumb sweeps a fixed arc in the hand frame, so the
outcome must be a function of where the rod top sits in that frame. This probe logs the
rod top, barrel top and thumb links in the right_hand_base_link frame before and after a
press, per episode and per grip variant, so the primitive can be designed to register
the pipette where the thumb lands.

Environment:  OUTDIR (required), EPISODES (default "0,4,40,43,47,51")
"""
import json
import os
import sys
import traceback

import numpy as np

OUTDIR = os.environ["OUTDIR"]
EPISODES = [int(x) for x in os.environ.get("EPISODES", "0,4,40,43,47,51").split(",") if x.strip()]
DEMO_DIR = os.environ.get("DEMO_DIR", "/home/jren313/research/starvla_rl/demo_buffer_v3")
os.makedirs(OUTDIR, exist_ok=True)
summary = {"_status": "RUNNING", "trials": []}
H = 30


def emit(s="RUNNING"):
    summary["_status"] = s
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


try:
    os.environ.pop("DISPLAY", None)
    import torch
    import importlib.util as ilu

    def _load(m, p):
        sp = ilu.spec_from_file_location(m, p); mo = ilu.module_from_spec(sp)
        sys.modules[m] = mo; sp.loader.exec_module(mo); return mo

    RL = "/home/jren313/research/starvla_rl/RLinf/rlinf/envs/isaaclab/tasks/"
    Mapper = _load("g1a", RL + "g1_piston_action.py").G1PistonActionMapper
    HR = _load("g1h", RL + "g1_piston_hand_retarget.py")
    RW5 = _load("g1r5", RL + "g1_piston_reward_v5.py")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    sys.path.append("/home/jren313/miniconda3/envs/isaac/lib/python3.11/site-packages")
    sys.path.insert(0, "/home/jren313/unitree_sim_isaaclab")
    import gymnasium as gym
    import tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    TID = "Isaac-PickPlace-Piston-G129-Inspire-Joint"
    ecfg = load_cfg_from_registry(TID, "env_cfg_entry_point")
    ecfg.seed = 0; ecfg.scene.num_envs = 1
    ecfg.scene.left_wrist_camera = None; ecfg.scene.right_wrist_camera = None
    env = gym.make(TID, cfg=ecfg, render_mode="rgb_array").unwrapped
    sc = env.scene; robot = sc["robot"]
    jn = list(robot.data.joint_names); bn = list(robot.data.body_names)
    mapper = Mapper(jn); retarget = HR.InspireHandRetargeter(jn)
    reward_fn = RW5.PistonTaskRewardV5(sc, jn)
    obj = sc["object"]; pj = list(obj.data.joint_names).index("PistonJoint")
    hb = bn.index("right_hand_base_link")
    links = {n: bn.index(n) for n in ("R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal",
                                      "R_index_intermediate", "R_middle_intermediate", "R_ring_intermediate", "R_pinky_intermediate")}

    def step_phys(a):
        phys = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)
        cmd = retarget.apply(mapper.map(phys).to(env.device), phys.to(env.device))
        for _ in range(2):
            env.step(cmd)
        return reward_fn.step()

    def geo(info):
        t = robot.data.body_pos_w[0, hb].cpu().numpy(); R = quat_to_R(robot.data.body_quat_w[0, hb].cpu().numpy())
        def loc(p): return [round(float(v), 4) for v in (R.T @ (np.asarray(p) - t))]
        rod = obj.data.body_pos_w[0, 0].cpu().numpy(); bar = obj.data.body_pos_w[0, 1].cpu().numpy()
        Rb = quat_to_R(obj.data.body_quat_w[0, 1].cpu().numpy()); axis = Rb[:, 2]
        rod_top = rod + 0.09 * (quat_to_R(obj.data.body_quat_w[0, 0].cpu().numpy())[:, 2])
        bar_top = bar + 0.095 * axis
        rb = robot.data.body_pos_w[0].cpu().numpy()
        return {"press": float(obj.data.joint_pos[0, pj]), "lift": float(info["lift"]), "finger_hold": bool(info["finger_hold"]),
                "finger_r": round(info["finger_radial"], 4), "barrel_tilt_deg": round(float(np.degrees(np.arccos(np.clip(axis[2], -1, 1)))), 1),
                "rod_top_hand": loc(rod_top), "barrel_top_hand": loc(bar_top), "barrel_axis_hand": [round(float(v), 3) for v in (R.T @ axis)],
                **{k + "_hand": loc(rb[i]) for k, i in links.items()}}

    def run(ep, variant):
        A = np.load(f"{DEMO_DIR}/ep{ep:03d}.npz")["actions"].reshape(-1, 30)
        env.reset(seed=0); reward_fn.reset()
        stages = {}
        for t in range(len(A)):
            r, info = step_phys(A[t])
            for k, v in info["stages"].items(): stages[k] = stages.get(k, False) or v
        last = A[-1].copy()
        rec = {"episode": ep, "variant": variant, "transport_stages": {k: bool(v) for k, v in stages.items()}, "before": geo(info)}
        if variant == "loosen_reclose":
            for k in range(15):
                a = last.copy(); a[20:24] = 1.0; r, info = step_phys(a)
            rec["after_loosen"] = geo(info)
            for k in range(15):
                r, info = step_phys(last)
            rec["after_reclose"] = geo(info)
        if variant == "raise3":
            # same IK raise the generator uses
            ee_idx = bn.index("right_wrist_yaw_link")
            arm_j = [jn.index(n) for n in ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                           "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")]
            for k in range(30):
                J = robot.root_physx_view.get_jacobians()[0]
                bi = ee_idx - 1 if J.shape[0] == len(bn) - 1 else ee_idx
                J = J[bi].cpu().numpy()[:, arm_j]; tt = np.array([0, 0, 0.001, 0, 0, 0.0])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), tt)
                last = last.copy(); last[7:14] += dq.astype(np.float32)
                r, info = step_phys(last)
            rec["after_raise"] = geo(info)
        a = last.copy(); a[25] = 0.42
        log = []
        for k in range(60):
            r, info = step_phys(a); log.append(geo(info))
        rec["after_press"] = log[-1]
        rec["press_max"] = max(g["press"] for g in log); rec["press_end"] = log[-1]["press"]
        rec["press_t10"] = log[9]["press"]
        summary["trials"].append(rec); emit()

    for ep in EPISODES:
        for v in ("instant", "loosen_reclose", "raise3"):
            run(ep, v)
    emit("OK"); os._exit(0)
except Exception as e:  # noqa: BLE001
    summary["error"] = f"{type(e).__name__}: {e}"; summary["tb"] = traceback.format_exc(); emit("FAIL"); os._exit(1)
