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

"""Which hand configuration actually depresses the plunger? A controlled sweep.

probe_press_geometry.py showed the demonstrated grasp holds the UPPER BARREL with the
thumb at the barrel top, 3.3 cm off the axis, so pushing the pipette onto the pot slid
the hand down the barrel (5 cm) while compressing the plunger only 1.7 cm. The rod
protrudes 2.5 cm above the barrel at rest and the press threshold is 2.8 cm, so the
press has to come from a hand part that sits over the rod and pushes it to, and a few
millimetres past, the barrel top.

This probe replays one demonstration to a chosen phase, applies a hand/wrist variant to
the last demonstrated action, then pushes the pipette down with differential IK against
a support (the socket/table before lift, the pot after transport) or squeezes in the
air, logging the plunger depth and which hand bodies sit over the rod top.

Environment:
    OUTDIR   output directory (required)
    EPISODE  demonstration to replay (default 40)
"""
import json
import os
import sys
import traceback

import numpy as np

OUTDIR = os.environ["OUTDIR"]
EP = int(os.environ.get("EPISODE", "40"))
DEMO_DIR = os.environ.get("DEMO_DIR", "/home/jren313/research/starvla_rl/demo_buffer_v3")
os.makedirs(OUTDIR, exist_ok=True)
summary = {"_status": "RUNNING", "episode": EP, "trials": []}


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
    sc = env.scene; robot = sc["robot"]
    jn = list(robot.data.joint_names); bn = list(robot.data.body_names)
    mapper = Mapper(jn); retarget = HR.InspireHandRetargeter(jn)
    reward_fn = RW3.PistonTaskRewardV3(sc, jn)
    obj = sc["object"]; pj = list(obj.data.joint_names).index("PistonJoint")
    ee_idx = bn.index("right_wrist_yaw_link")
    hand_bodies = [b for b in bn if b.startswith("R_") or b in ("right_hand_base_link", "right_wrist_yaw_link")]
    hand_idx = [bn.index(b) for b in hand_bodies]
    arm_j = [jn.index(n) for n in (
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")]
    A = np.load(f"{DEMO_DIR}/ep{EP:03d}.npz")["actions"].reshape(-1, 30)

    def step_phys(phys30):
        phys = torch.as_tensor(phys30, dtype=torch.float32).unsqueeze(0)
        cmd = retarget.apply(mapper.map(phys).to(env.device), phys.to(env.device))
        for _ in range(2):
            env.step(cmd)
        return reward_fn.step()

    def press():
        return float(obj.data.joint_pos[0, pj])

    def geometry(info):
        rod = obj.data.body_pos_w[0, 0].cpu().numpy(); bar = obj.data.body_pos_w[0, 1].cpu().numpy()
        rod_top = rod.copy(); rod_top[2] += 0.09
        bar_top_z = bar[2] + 0.095
        rb = robot.data.body_pos_w[0].cpu().numpy()
        over = {}
        for name, i in zip(hand_bodies, hand_idx):
            p = rb[i]; r_xy = float(np.linalg.norm((p - rod_top)[:2])); dz = float(p[2] - rod_top[2])
            over[name] = [round(r_xy, 4), round(dz, 4)]
        return {"press": press(), "rod_top_minus_barrel_top": float(rod_top[2] - bar_top_z),
                "barrel": bar.tolist(), "lift": float(info["lift"]), "grasped": bool(info["grasped"]),
                "finger_r": info["finger_radial"], "thumb_r": info["thumb_radial"], "hand_over_rod_top": over}

    def jac():
        J = robot.root_physx_view.get_jacobians()[0]
        bi = ee_idx - 1 if J.shape[0] == len(bn) - 1 else ee_idx
        return J[bi].cpu().numpy()[:, arm_j]

    def ik_step(last, dx, lam=1e-3):
        J = jac(); t = np.concatenate([dx, np.zeros(3)])
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), t)
        new = np.array(last, dtype=np.float32).copy(); new[7:14] += dq.astype(np.float32); return new

    def replay_to(phase):
        """Reset and replay the demo to 'grasp' (+0.6 s) or 'end'. Returns (last_action, t)."""
        env.reset(seed=0); reward_fn.reset()
        grasp_t = None
        for t in range(len(A)):
            r, info = step_phys(A[t])
            if phase == "grasp":
                if grasp_t is None and info["stages"]["grasp"]:
                    grasp_t = t
                if grasp_t is not None and t >= grasp_t + 30:
                    return A[t].copy(), t, info
        return A[-1].copy(), len(A) - 1, info

    def apply_variant(a, v):
        a = a.copy()
        for dim, val in v.get("set", {}).items():
            a[int(dim)] = val
        for dim, dv in v.get("add", {}).items():
            a[int(dim)] += dv
        return a

    def run_trial(name, phase, variant, mode, push_m=0.04, settle=25, hold=60):
        last, t, info = replay_to(phase)
        base_geo = geometry(info)
        last = apply_variant(last, variant)
        log = []
        for k in range(settle):
            r, info = step_phys(last); log.append(geometry(info))
        ee0 = robot.data.body_pos_w[0, ee_idx, 2].item()
        if mode == "push":
            moved = 0.0
            for k in range(400):
                last = ik_step(last, np.array([0.0, 0.0, -0.0006]))
                r, info = step_phys(last); g = geometry(info); log.append(g)
                moved = ee0 - robot.data.body_pos_w[0, ee_idx, 2].item()
                if moved > push_m:
                    break
        for k in range(hold):
            r, info = step_phys(last); log.append(geometry(info))
        presses = [g["press"] for g in log]
        im = int(np.argmax(presses)); gm = log[im]
        # which hand body is closest to being over the rod top when pressing hardest
        near = sorted(gm["hand_over_rod_top"].items(), key=lambda kv: kv[1][0])[:4]
        rec = {"name": name, "phase": phase, "variant": variant, "mode": mode, "t_start": int(t),
               "start": {k: base_geo[k] for k in ("press", "lift", "grasped", "rod_top_minus_barrel_top", "finger_r", "thumb_r")},
               "max_press": max(presses), "press_end": presses[-1],
               "grasped_at_max": gm["grasped"], "lift_at_max": gm["lift"], "rod_top_minus_barrel_top_at_max": gm["rod_top_minus_barrel_top"],
               "grasped_frac": float(np.mean([g["grasped"] for g in log])),
               "ee_moved_m": float(ee0 - robot.data.body_pos_w[0, ee_idx, 2].item()),
               "hand_bodies_nearest_rod_top_at_max [r_xy, dz]": near,
               "final_hand_over_rod_top": log[-1]["hand_over_rod_top"]}
        summary["trials"].append(rec); emit()
        return rec

    # ---- grasp phase, in the socket: what pushes the rod? ---------------------------
    V0 = {}
    run_trial("socket_push_baseline", "grasp", V0, "push")
    run_trial("socket_push_thumb_yaw_1.3", "grasp", {"set": {"25": 1.3}}, "push")
    run_trial("socket_push_thumb_yaw_0.0", "grasp", {"set": {"25": 0.0}}, "push")
    run_trial("socket_push_thumb_pitch_0.0", "grasp", {"set": {"24": 0.0}}, "push")
    run_trial("socket_push_fingers_1.7", "grasp", {"set": {"20": 1.7, "21": 1.7, "22": 1.7, "23": 1.7}}, "push")
    run_trial("socket_push_fingers_1.2", "grasp", {"set": {"20": 1.2, "21": 1.2, "22": 1.2, "23": 1.2}}, "push")
    run_trial("socket_push_wrist_pitch_+0.3", "grasp", {"add": {"12": 0.3}}, "push")
    run_trial("socket_push_wrist_pitch_-0.3", "grasp", {"add": {"12": -0.3}}, "push")
    run_trial("socket_push_wrist_roll_+0.3", "grasp", {"add": {"11": 0.3}}, "push")
    run_trial("socket_push_wrist_roll_-0.3", "grasp", {"add": {"11": -0.3}}, "push")
    run_trial("socket_push_wrist_yaw_+0.3", "grasp", {"add": {"13": 0.3}}, "push")
    run_trial("socket_push_wrist_yaw_-0.3", "grasp", {"add": {"13": -0.3}}, "push")
    run_trial("socket_push_8cm", "grasp", V0, "push", push_m=0.08)
    # ---- in the air, after transport: squeeze variants --------------------------------
    run_trial("air_hold_baseline", "end", V0, "hold", hold=100)
    run_trial("air_fingers_1.7", "end", {"set": {"20": 1.7, "21": 1.7, "22": 1.7, "23": 1.7}}, "hold", hold=100)
    run_trial("air_thumb_yaw_1.3", "end", {"set": {"25": 1.3}}, "hold", hold=100)
    run_trial("air_thumb_yaw_0.0_pitch_0.0", "end", {"set": {"24": 0.0, "25": 0.0}}, "hold", hold=100)
    # ---- on the pot, after transport ----------------------------------------------------
    run_trial("pot_push_baseline", "end", V0, "push", push_m=0.06)
    run_trial("pot_push_fingers_1.7", "end", {"set": {"20": 1.7, "21": 1.7, "22": 1.7, "23": 1.7}}, "push", push_m=0.06)
    run_trial("pot_push_thumb_yaw_1.3", "end", {"set": {"25": 1.3}}, "push", push_m=0.06)
    run_trial("pot_push_wrist_pitch_+0.3", "end", {"add": {"12": 0.3}}, "push", push_m=0.06)
    run_trial("pot_push_wrist_pitch_-0.3", "end", {"add": {"12": -0.3}}, "push", push_m=0.06)
    emit("OK"); os._exit(0)
except Exception as e:  # noqa: BLE001
    summary["error"] = f"{type(e).__name__}: {e}"; summary["tb"] = traceback.format_exc(); emit("FAIL"); os._exit(1)
