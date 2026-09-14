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

"""Make the thumb press robust: thumb curl during the press x pipette seating.

The hand-frame probe showed the thumb sweep lands at a fixed point in the hand frame while
the rod top sits 1-2 cm away from it, varying per grasp, so the press depth is a coin flip
(4-24 mm). Two levers, swept together here from each demonstration's end pose:

* CURL: extra thumb intermediate/distal flexion ramped in with the press channel
  (InspireRetargetParams.thumb_press_inter_gain / thumb_press_distal_gain), so the thumb
  tip hooks over the rod top instead of passing beside it.
* SEAT: before pressing, lower the pipette tip onto the pot and push 2 cm so the barrel
  slides up into the hand, then raise 8 cm -- a mechanical registration of where the rod
  top sits relative to the thumb.

Environment: OUTDIR (required), EPISODES (default "0,4,40,43")
"""
import dataclasses
import json
import os
import sys
import traceback

import numpy as np

OUTDIR = os.environ["OUTDIR"]
EPISODES = [int(x) for x in os.environ.get("EPISODES", "0,4,40,43").split(",") if x.strip()]
DEMO_DIR = os.environ.get("DEMO_DIR", "/home/jren313/research/starvla_rl/demo_buffer_v3")
CURLS = [(0.0, 0.0), (0.4, 0.6), (0.4, 0.0), (0.0, 0.6), (0.2, 0.3)]
os.makedirs(OUTDIR, exist_ok=True)
summary = {"_status": "RUNNING", "trials": []}


def emit(s="RUNNING"):
    summary["_status"] = s
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)


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
    mapper = Mapper(jn)
    reward_fn = RW5.PistonTaskRewardV5(sc, jn)
    obj = sc["object"]; pj = list(obj.data.joint_names).index("PistonJoint")
    ee_idx = bn.index("right_wrist_yaw_link")
    arm_j = [jn.index(n) for n in ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                   "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")]
    retargeters = {c: HR.InspireHandRetargeter(jn, dataclasses.replace(HR.DEFAULT_PARAMS, thumb_press_inter_gain=c[0], thumb_press_distal_gain=c[1])) for c in CURLS}

    def step_phys(a, rt):
        phys = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)
        cmd = rt.apply(mapper.map(phys).to(env.device), phys.to(env.device))
        for _ in range(2):
            env.step(cmd)
        return reward_fn.step()

    def ik_step(last, dz):
        J = robot.root_physx_view.get_jacobians()[0]
        bi = ee_idx - 1 if J.shape[0] == len(bn) - 1 else ee_idx
        J = J[bi].cpu().numpy()[:, arm_j]; tt = np.array([0, 0, dz, 0, 0, 0.0])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), tt)
        new = last.copy(); new[7:14] += dq.astype(np.float32); return new

    def ee_z():
        return robot.data.body_pos_w[0, ee_idx, 2].item()

    for ep in EPISODES:
        A = np.load(f"{DEMO_DIR}/ep{ep:03d}.npz")["actions"].reshape(-1, 30)
        for seat in (False, True):
            for curl in CURLS:
                rt = retargeters[curl]; rt0 = retargeters[(0.0, 0.0)]
                env.reset(seed=0); reward_fn.reset()
                for t in range(len(A)):
                    r, info = step_phys(A[t], rt0)
                last = A[-1].copy()
                rec = {"episode": ep, "seat": seat, "curl": list(curl), "lift_start": round(info["lift"], 4), "finger_hold_start": bool(info["finger_hold"])}
                if not info["finger_hold"]:
                    rec["outcome"] = "transport_failed"; summary["trials"].append(rec); emit(); continue
                if seat:
                    # descend until the barrel stops moving, push 2 cm more, raise 8 cm
                    bz_hist = []; contact = None
                    for k in range(300):
                        last = ik_step(last, -0.0008); r, info = step_phys(last, rt0)
                        bz = float(obj.data.body_pos_w[0, 1, 2]); bz_hist.append(bz)
                        if contact is None and k > 5 and abs(bz_hist[-1] - bz_hist[-4]) < 2e-4:
                            contact = k; ez0 = ee_z()
                        if contact is not None and ez0 - ee_z() > 0.02:
                            break
                    rec["seat_contact_k"] = contact; rec["seat_barrel_bottom_z"] = round(bz - 0.095, 4)
                    for k in range(80):
                        last = ik_step(last, 0.001); r, info = step_phys(last, rt0)
                    rec["lift_after_seat"] = round(info["lift"], 4); rec["finger_hold_after_seat"] = bool(info["finger_hold"])
                    rec["press_after_seat"] = round(info["press_m"], 4)
                a = last.copy(); a[25] = 0.42
                log = []
                for k in range(60):
                    r, info = step_phys(a, rt); log.append((info["press_m"], info["lift"], info["finger_hold"], info["press_sustain_steps"]))
                rec.update({"press_max": round(max(x[0] for x in log), 4), "press_end": round(log[-1][0], 4),
                            "lift_min": round(min(x[1] for x in log), 4), "finger_hold_frac": round(float(np.mean([x[2] for x in log])), 3),
                            "sustain_max": int(max(x[3] for x in log)), "outcome": "press" if max(x[3] for x in log) >= 15 else "no_press"})
                summary["trials"].append(rec); emit()
    emit("OK"); os._exit(0)
except Exception as e:  # noqa: BLE001
    summary["error"] = f"{type(e).__name__}: {e}"; summary["tb"] = traceback.format_exc(); emit("FAIL"); os._exit(1)
