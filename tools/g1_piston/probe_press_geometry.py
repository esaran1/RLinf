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

"""Measure the geometry of a plunger press, before scripting one.

Three questions the scripted press primitive depends on, answered from the simulator
rather than from the object's documentation:

1. Which part of the pipette does the demonstrated grasp hold -- the rod (plunger, the
   articulation root) or the barrel -- and where along it do the fingers and thumb sit?
2. What was the barrel bottom touching during the one demonstrated press transient
   (ep46, chunk 13)?
3. If a held pipette is pushed tip-first onto a rigid surface, does the plunger
   compress (the hand holds the rod) or does the whole pipette slide in the hand?

The third is tested directly: after a demonstration ends with the pipette held over the
plate, the right arm is driven by damped-least-squares differential IK on the live
Jacobian to lower the pipette onto the pot, then to push a further ``PUSH_M`` down,
while the plunger joint is logged.

Environment:
    OUTDIR      directory for the per-step npz logs and summary.json (required)
    EPISODES    demonstration episodes to replay (default "46,0")
    PUSH_M      extra descent commanded after first contact, metres (default 0.05)
"""
import json
import os
import sys
import time
import traceback

import numpy as np

OUTDIR = os.environ["OUTDIR"]
EPISODES = [int(x) for x in os.environ.get("EPISODES", "46,0").split(",") if x.strip()]
PUSH_M = float(os.environ.get("PUSH_M", "0.05"))
DEMO_DIR = os.environ.get("DEMO_DIR", "/home/jren313/research/starvla_rl/demo_buffer_v3")
os.makedirs(OUTDIR, exist_ok=True)
summary = {"_status": "RUNNING", "episodes": {}}
H = 30


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
    RW3 = _load("g1r3", RL + "g1_piston_reward_v3.py")

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
    sc = env.scene
    robot = sc["robot"]
    jn = list(robot.data.joint_names)
    bn = list(robot.data.body_names)
    mapper = Mapper(jn); retarget = HR.InspireHandRetargeter(jn)
    reward_fn = RW3.PistonTaskRewardV3(sc, jn)

    # ---- scene inventory -------------------------------------------------------------
    inv = {}
    for key in sc.keys():
        try:
            ent = sc[key]
            d = {"type": type(ent).__name__}
            data = getattr(ent, "data", None)
            if data is not None and hasattr(data, "root_pos_w"):
                d["root_pos_w"] = [round(float(v), 4) for v in data.root_pos_w[0]]
            if data is not None and hasattr(data, "body_names"):
                d["body_names"] = list(data.body_names)[:60]
            inv[key] = d
        except Exception as e:  # noqa: BLE001
            inv[key] = {"error": str(e)}
    summary["scene"] = inv
    summary["robot_joint_names"] = jn
    summary["robot_body_names"] = bn
    emit()

    # Right-hand bodies of interest.
    fing_idx = [bn.index(f"R_{f}_intermediate") for f in ("index", "middle", "pinky", "ring")]
    thumb_idx = bn.index("R_thumb_distal")
    ee_name = next(n for n in ("right_wrist_yaw_link", "R_hand_base_link", "right_hand_palm_link") if n in bn)
    ee_idx = bn.index(ee_name)
    arm_j = [jn.index(n) for n in (
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")]
    summary["ee_body"] = ee_name
    obj = sc["object"]
    pj = list(obj.data.joint_names).index("PistonJoint")

    def read_state(info=None):
        rod = obj.data.body_pos_w[0, 0].cpu().numpy(); barrel = obj.data.body_pos_w[0, 1].cpu().numpy()
        q = obj.data.body_quat_w[0, 1].cpu().numpy()  # barrel orientation (w,x,y,z)
        rb = robot.data.body_pos_w[0].cpu().numpy()
        return {"press": float(obj.data.joint_pos[0, pj]), "rod": rod.tolist(), "barrel": barrel.tolist(),
                "barrel_quat": q.tolist(), "fingers": rb[fing_idx].tolist(), "thumb": rb[thumb_idx].tolist(),
                "ee": rb[ee_idx].tolist(), "pot": sc["pot"].data.root_pos_w[0].cpu().numpy().tolist(),
                "tube": sc["tube"].data.root_pos_w[0].cpu().numpy().tolist(),
                "grasped": bool(info["grasped"]) if info else None, "lift": float(info["lift"]) if info else None}

    def step_phys(phys30):
        phys = torch.as_tensor(phys30, dtype=torch.float32).unsqueeze(0)
        cmd = retarget.apply(mapper.map(phys).to(env.device), phys.to(env.device))
        for _ in range(2):
            env.step(cmd)
        return reward_fn.step()

    def jacobian_lin_ang():
        J = robot.root_physx_view.get_jacobians()[0]  # [nb or nb-1, 6, ndof]
        bi = ee_idx - 1 if J.shape[0] == len(bn) - 1 else ee_idx
        Jb = J[bi].cpu().numpy()  # [6, ndof]
        return Jb[:, arm_j]       # [6, 7]

    def ik_step(last_phys, dx_world, lam=1e-3, rot_weight=1.0):
        """One damped-least-squares step: move the EE by dx (m), hold orientation."""
        J = jacobian_lin_ang()
        target = np.concatenate([dx_world, np.zeros(3)])
        W = np.diag([1, 1, 1, rot_weight, rot_weight, rot_weight])
        Jw = W @ J; tw = W @ target
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam * np.eye(6), tw)
        new = np.array(last_phys, dtype=np.float32).copy()
        new[7:14] += dq.astype(np.float32)
        return new

    for ep in EPISODES:
        fp = f"{DEMO_DIR}/ep{ep:03d}.npz"
        A = np.load(fp)["actions"].reshape(-1, 30)
        env.reset(seed=0); reward_fn.reset()
        log = []
        stages = {}
        for t in range(len(A)):
            r, info = step_phys(A[t])
            st = read_state(info); st["t"] = t; st["chunk"] = t // H; st["reward"] = r
            st["finger_r"] = info["finger_radial"]; st["thumb_r"] = info["thumb_radial"]
            st["finger_dz"] = info["finger_dz"]; st["thumb_dz"] = info["thumb_dz"]
            log.append(st)
            for k, v in info["stages"].items():
                stages[k] = stages.get(k, False) or v
        rec = {"n_steps": len(A), "stages": stages,
               "max_press": max(s["press"] for s in log),
               "t_max_press": int(np.argmax([s["press"] for s in log]))}
        tm = rec["t_max_press"]; s = log[tm]
        rec["at_max_press"] = {k: s[k] for k in ("press", "rod", "barrel", "fingers", "thumb", "ee", "grasped", "lift", "chunk")}
        bz = s["barrel"][2]
        rec["at_max_press"]["barrel_bottom_z"] = bz - 0.095
        rec["at_max_press"]["rod_top_z"] = s["rod"][2] + 0.09
        rec["at_max_press"]["finger_z_minus_barrel_z"] = [f[2] - bz for f in s["fingers"]]
        rec["at_max_press"]["thumb_z_minus_barrel_z"] = s["thumb"][2] - bz
        # grasp anatomy at the moment lift first exceeds 0.05 while grasped
        lifted = [x for x in log if x["grasped"] and x["lift"] is not None and x["lift"] > 0.05]
        if lifted:
            g = lifted[0]; bz = g["barrel"][2]
            rec["at_first_lift"] = {"t": g["t"], "finger_z_minus_barrel_z": [f[2] - bz for f in g["fingers"]],
                                    "thumb_z_minus_barrel_z": g["thumb"][2] - bz,
                                    "ee_z_minus_barrel_z": g["ee"][2] - bz,
                                    "rod_top_minus_barrel_top": (g["rod"][2] + 0.09) - (bz + 0.095),
                                    "press": g["press"]}
        final = log[-1]
        rec["final"] = {k: final[k] for k in ("press", "barrel", "rod", "grasped", "lift")}

        # ---- push test: from the held pose, descend onto the pot and push ----------------
        push = {"ran": False}
        if final["grasped"] and final["lift"] and final["lift"] > 0.05:
            push["ran"] = True
            last = A[-1].copy()
            plog = []
            contact_z = None
            descended = 0.0
            phase = "descend"
            for k in range(400):   # up to 8 s at 50 Hz
                dx = np.array([0.0, 0.0, -0.0006])   # 3 cm/s
                last = ik_step(last, dx)
                r, info = step_phys(last)
                st = read_state(info); st["k"] = k; st["phase"] = phase
                plog.append(st)
                bz = st["barrel"][2]
                if phase == "descend":
                    # contact: barrel stops descending although the hand keeps going
                    if k > 5 and abs(plog[-1]["barrel"][2] - plog[-4]["barrel"][2]) < 2e-4 and st["ee"][2] < plog[-4]["ee"][2] - 1e-4:
                        phase = "push"; contact_z = bz; push["contact_k"] = k
                        push["contact_barrel_bottom_z"] = bz - 0.095
                        push["contact_pot_z"] = st["pot"][2]
                elif phase == "push":
                    descended = plog[push["contact_k"]]["ee"][2] - st["ee"][2]
                    if descended > PUSH_M or not st["grasped"]:
                        break
            push["steps"] = len(plog)
            push["max_press"] = max(s["press"] for s in plog)
            push["press_at_end"] = plog[-1]["press"]
            push["grasped_at_end"] = plog[-1]["grasped"]
            push["lift_at_end"] = plog[-1]["lift"]
            push["ee_descended_after_contact"] = descended
            push["barrel_z_start_end"] = [plog[0]["barrel"][2], plog[-1]["barrel"][2]]
            push["rod_minus_barrel_z_start_end"] = [plog[0]["rod"][2] - plog[0]["barrel"][2], plog[-1]["rod"][2] - plog[-1]["barrel"][2]]
            push["ee_minus_barrel_z_start_end"] = [plog[0]["ee"][2] - plog[0]["barrel"][2], plog[-1]["ee"][2] - plog[-1]["barrel"][2]]
            np.savez_compressed(os.path.join(OUTDIR, f"push_ep{ep:03d}.npz"),
                                press=np.array([s["press"] for s in plog]), barrel=np.array([s["barrel"] for s in plog]),
                                rod=np.array([s["rod"] for s in plog]), ee=np.array([s["ee"] for s in plog]),
                                grasped=np.array([s["grasped"] for s in plog]))
        rec["push_test"] = push
        np.savez_compressed(os.path.join(OUTDIR, f"replay_ep{ep:03d}.npz"),
                            press=np.array([s["press"] for s in log]), barrel=np.array([s["barrel"] for s in log]),
                            rod=np.array([s["rod"] for s in log]), ee=np.array([s["ee"] for s in log]),
                            fingers=np.array([s["fingers"] for s in log]), thumb=np.array([s["thumb"] for s in log]),
                            grasped=np.array([s["grasped"] for s in log]), lift=np.array([s["lift"] for s in log]),
                            finger_r=np.array([s["finger_r"] for s in log]), thumb_r=np.array([s["thumb_r"] for s in log]))
        summary["episodes"][str(ep)] = rec
        emit()
    emit("OK")
    os._exit(0)
except Exception as e:  # noqa: BLE001
    summary["error"] = f"{type(e).__name__}: {e}"; summary["tb"] = traceback.format_exc()
    emit("FAIL")
    os._exit(1)
