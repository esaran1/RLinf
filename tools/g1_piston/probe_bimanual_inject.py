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

"""Feasibility of a BIMANUAL plunger press: the tube in the left fist pushes the rod top.

Every single-hand press mechanism measured so far is a coin flip, because the rod top
(radius 1 cm, 2.5 cm proud of the barrel) has to meet a small rigid part of the same hand
that holds the barrel. The task description itself says "inject it into the tube held by
the left hand": the left fist, holding the tube, is a large rigid body under position
control, and the rod top is the highest point of the held pipette. Bringing the tube's
bottom down onto the rod top turns the press into a problem with centimetres of tolerance.

Per episode: replay to the end (pipette held over the plate); right arm presents the
pipette upright at chest height near the body centre; left arm brings the tube bottom
above the rod top; left arm descends until the plunger reaches the target depth; hold.

Environment: OUTDIR (required), EPISODES (default "0,4,40,43")
"""
import json
import os
import sys
import traceback

import numpy as np

OUTDIR = os.environ["OUTDIR"]
EPISODES = [int(x) for x in os.environ.get("EPISODES", "0,4,40,43").split(",") if x.strip()]
DEMO_DIR = os.environ.get("DEMO_DIR", "/home/jren313/research/starvla_rl/demo_buffer_v3")
PRESENT = np.array([float(v) for v in os.environ.get("PRESENT", "-0.15,0.32,1.05").split(",")])
TARGET = float(os.environ.get("TARGET", "0.0215"))
os.makedirs(OUTDIR, exist_ok=True)
summary = {"_status": "RUNNING", "present_xyz": PRESENT.tolist(), "trials": []}


def emit(s="RUNNING"):
    summary["_status"] = s
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)


def axis_z(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


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
    sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
    from rlinf.envs.isaaclab.tasks.g1_piston_reset import CANONICAL, apply_reset_condition
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
    tube = sc["tube"]
    R_EE = bn.index("right_wrist_yaw_link"); L_EE = bn.index("left_wrist_yaw_link")
    R_J = [jn.index(n) for n in ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                 "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")]
    L_J = [jn.index(n) for n in ("left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                                 "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint")]
    L_LIM = [(float(robot.data.soft_joint_pos_limits[0, j, 0]), float(robot.data.soft_joint_pos_limits[0, j, 1])) for j in L_J]
    summary["left_arm_limits"] = L_LIM

    def step_phys(a):
        phys = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)
        cmd = retarget.apply(mapper.map(phys).to(env.device), phys.to(env.device))
        for _ in range(2):
            env.step(cmd)
        return reward_fn.step()

    def ik(last, ee_idx, joint_ids, action_slice, dx, lam=1e-3, hold_rot=True):
        J = robot.root_physx_view.get_jacobians()[0]
        bi = ee_idx - 1 if J.shape[0] == len(bn) - 1 else ee_idx
        J = J[bi].cpu().numpy()[:, joint_ids]
        if hold_rot:
            t = np.concatenate([dx, np.zeros(3)])
        else:
            J = J[:3]; t = np.asarray(dx)
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), t)
        new = last.copy(); new[action_slice] += dq.astype(np.float32); return new

    def rod_top():
        return obj.data.body_pos_w[0, 0].cpu().numpy() + 0.09 * axis_z(obj.data.body_quat_w[0, 0].cpu().numpy())

    def tube_bottom():
        return tube.data.root_pos_w[0].cpu().numpy() - 0.05 * axis_z(tube.data.root_quat_w[0].cpu().numpy())

    def tube_in_fist():
        return float(np.linalg.norm(tube.data.root_pos_w[0].cpu().numpy() - robot.data.body_pos_w[0, L_EE].cpu().numpy()))

    for ep in EPISODES:
        A = np.load(f"{DEMO_DIR}/ep{ep:03d}.npz")["actions"].reshape(-1, 30)
        env.reset(seed=0); apply_reset_condition(env, CANONICAL); reward_fn.reset()
        for t in range(len(A)):
            r, info = step_phys(A[t])
        last = A[-1].copy()
        rec = {"episode": ep, "finger_hold_start": bool(info["finger_hold"]), "lift_start": round(info["lift"], 4),
               "tube_fist_dist_start": round(tube_in_fist(), 4)}
        if not info["finger_hold"]:
            rec["outcome"] = "transport_failed"; summary["trials"].append(rec); emit(); continue
        # ---- A. present the pipette near the body centre, higher -----------------------
        for t in range(300):
            err = PRESENT - obj.data.body_pos_w[0, 1].cpu().numpy()
            if np.linalg.norm(err) < 0.01:
                break
            step = err / max(np.linalg.norm(err), 1e-9) * min(0.003, float(np.linalg.norm(err)))
            last = ik(last, R_EE, R_J, slice(7, 14), step)
            r, info = step_phys(last)
        rec["present_steps"] = t; rec["present_err"] = round(float(np.linalg.norm(err)), 4)
        rec["present_finger_hold"] = bool(info["finger_hold"]); rec["present_lift"] = round(info["lift"], 4)
        rec["rod_top_after_present"] = [round(float(v), 4) for v in rod_top()]
        if not info["finger_hold"]:
            rec["outcome"] = "dropped_in_present"; summary["trials"].append(rec); emit(); continue
        # ---- B. left fist: tube bottom 4 cm above the rod top ---------------------------
        for t in range(400):
            target = rod_top() + np.array([0.0, 0.0, 0.04])
            err = target - tube_bottom()
            if np.linalg.norm(err) < 0.012:
                break
            step = err / max(np.linalg.norm(err), 1e-9) * min(0.003, float(np.linalg.norm(err)))
            last = ik(last, L_EE, L_J, slice(0, 7), step, hold_rot=False)
            r, info = step_phys(last)
        rec["approach_steps"] = t; rec["approach_err"] = round(float(np.linalg.norm(err)), 4)
        rec["tube_fist_dist_after_approach"] = round(tube_in_fist(), 4)
        rec["left_arm_q"] = [round(float(v), 3) for v in robot.data.joint_pos[0, L_J].cpu().numpy()]
        rec["approach_finger_hold"] = bool(info["finger_hold"])
        # ---- C. descend the left arm until the target depth -----------------------------
        z0 = float(robot.data.body_pos_w[0, L_EE, 2]); trace = []
        for t in range(200):
            err_xy = (rod_top() - tube_bottom())[:2]
            lat = err_xy / max(np.linalg.norm(err_xy), 1e-9) * min(0.002, float(np.linalg.norm(err_xy)))
            last = ik(last, L_EE, L_J, slice(0, 7), np.array([lat[0], lat[1], -0.0008]), hold_rot=False)
            r, info = step_phys(last)
            pressed = info["press_m"]
            if t % 10 == 0:
                trace.append([t, round(pressed, 4), round(float(np.linalg.norm((rod_top() - tube_bottom())[:2])), 4),
                              round(float(rod_top()[2] - tube_bottom()[2]), 4), round(tube_in_fist(), 4), bool(info["finger_hold"])])
            if pressed >= TARGET or z0 - float(robot.data.body_pos_w[0, L_EE, 2]) > 0.10:
                break
        rec["descend_steps"] = t; rec["descent_m"] = round(z0 - float(robot.data.body_pos_w[0, L_EE, 2]), 4)
        rec["press_reached"] = round(pressed, 4); rec["trace [t, press, xy_err, rod_top_minus_tube_bottom_z, tube_fist_dist, finger_hold]"] = trace
        # ---- D. hold ----------------------------------------------------------------------
        log = []
        for t in range(45):
            r, info = step_phys(last); log.append((info["press_m"], info["press_sustain_steps"], info["finger_hold"], info["lift"]))
        rec.update({"hold_press_min": round(min(x[0] for x in log), 4), "hold_press_max": round(max(x[0] for x in log), 4),
                    "press_sustain_max": int(max(x[1] for x in log)), "finger_hold_frac": round(float(np.mean([x[2] for x in log])), 3),
                    "lift_min": round(min(x[3] for x in log), 4), "stage_press": bool(info["stages"]["press"]),
                    "tube_fist_dist_end": round(tube_in_fist(), 4)})
        rec["outcome"] = "press" if info["stages"]["press"] else "no_press"
        summary["trials"].append(rec); emit()
    emit("OK"); os._exit(0)
except Exception as e:  # noqa: BLE001
    summary["error"] = f"{type(e).__name__}: {e}"; summary["tb"] = traceback.format_exc(); emit("FAIL"); os._exit(1)
