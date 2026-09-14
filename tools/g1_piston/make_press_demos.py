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

"""Synthesise pressing demonstrations: human transport + scripted thumb press.

The demonstrations transport the pipette to the plate but never press the plunger
(g1_piston_plunger_dof.json); RL found the press once in ~300 rollouts. This tool adds
the missing behaviour to the data instead of waiting for exploration to find it:

1. Replay an executable demonstration (frozen mapper + retargeter, the deployment path)
   until its ``plate`` stage fires -- the pipette is held over the plate.
2. Append a scripted primitive, three 30-step chunks at the policy's 50 Hz:
     P1  raise the pipette 3 cm (differential IK on the live Jacobian, arm dims only),
         so the lift gate has margin -- a pressing hand sags ~1.5 cm;
     P2  hold the arm, ramp the thumb-yaw dim (25) to ``YAW_CMD`` (0.42 rad: the press
         channel of the retargeter drives the thumb to its 1.3 rad limit, over the rod top);
     P3  hold -- the press is sustained.
3. Score with reward v5 and KEEP the episode only if ``dispense`` fired (sustained 20 mm
   press while lifted and over the plate). Failures are recorded, not saved.

Output is a demonstration buffer in the shipped format (images at chunk starts, PHYSICAL
action chunks, rewards, critic_state), so ``train_bc.py`` consumes it unchanged. The
transport chunks are the demonstration's own; only the press chunks are synthetic, and the
manifest marks where each episode's synthetic part begins.

Environment:
    OUTDIR      destination (required)
    EPISODES    comma-separated demo ids (default: the 16 lifting episodes)
    CONDS       reset conditions per episode: "canonical" and/or train-split indices,
                e.g. "canonical,0,1" (default "canonical")
    RAISE_M     P1 raise height (default 0.03)
    YAW_CMD     thumb-yaw command in P2/P3 (default 0.42)
    KEEP_FAILED "1" to also save episodes whose press did not certify (default 0)
    VARIANTS    JSON list of primitive variants to run per episode/condition, each a dict
                with optional keys: raise_m, yaw_cmd, ramp_steps, wrist (dict of action dim
                -> offset, ramped in with the thumb during P2), name. Default: one baseline.
"""
import json
import os
import sys
import time
import traceback

import numpy as np

OUTDIR = os.environ["OUTDIR"]
DEMO_DIR = os.environ.get("DEMO_DIR", "/home/jren313/research/starvla_rl/demo_buffer_v3")
LIFTING = [0, 4, 5, 39, 40, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 59]
EPISODES = [int(x) for x in os.environ.get("EPISODES", ",".join(map(str, LIFTING))).split(",") if x.strip()]
CONDS = [c.strip() for c in os.environ.get("CONDS", "canonical").split(",") if c.strip()]
RAISE_M = float(os.environ.get("RAISE_M", "0.03"))
YAW_CMD = float(os.environ.get("YAW_CMD", "0.42"))
KEEP_FAILED = os.environ.get("KEEP_FAILED", "0") == "1"
VARIANTS = json.loads(os.environ.get("VARIANTS", "[{}]"))
EP_CHUNKS = int(os.environ.get("EP_CHUNKS", "23"))
H = 30
os.makedirs(OUTDIR, exist_ok=True)
summary = {"_status": "RUNNING", "config": {"episodes": EPISODES, "conds": CONDS, "raise_m": RAISE_M,
                                             "yaw_cmd": YAW_CMD, "reward": "v5_geometric_press"},
           "episodes": []}


def emit(s="RUNNING"):
    summary["_status"] = s
    with open(os.path.join(OUTDIR, "_build_status.json"), "w") as f:
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
    CST = _load("g1cs", RL + "g1_piston_critic_state.py")

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    sys.path.append("/home/jren313/miniconda3/envs/isaac/lib/python3.11/site-packages")
    sys.path.insert(0, "/home/jren313/unitree_sim_isaaclab")
    import gymnasium as gym
    import tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
    from rlinf.envs.isaaclab.tasks.g1_piston_reset import CANONICAL, apply_reset_condition, build_reset_suite

    TID = "Isaac-PickPlace-Piston-G129-Inspire-Joint"
    ecfg = load_cfg_from_registry(TID, "env_cfg_entry_point")
    ecfg.seed = 0; ecfg.scene.num_envs = 1
    ecfg.scene.left_wrist_camera = None; ecfg.scene.right_wrist_camera = None
    env = gym.make(TID, cfg=ecfg, render_mode="rgb_array").unwrapped
    sc = env.scene; robot = sc["robot"]
    jn = list(robot.data.joint_names); bn = list(robot.data.body_names)
    mapper = Mapper(jn); retarget = HR.InspireHandRetargeter(jn)
    reward_fn = RW5.PistonTaskRewardV5(sc, jn)
    csb = CST.CriticStateBuilder(sc, max_chunks=EP_CHUNKS)
    obj = sc["object"]; pj = list(obj.data.joint_names).index("PistonJoint")
    ee_idx = bn.index("right_wrist_yaw_link")
    arm_j = [jn.index(n) for n in (
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")]
    TRAIN_CONDS, _ = build_reset_suite(n_train=400, n_eval=50, seed=20260817)
    summary["retarget_params"] = retarget.params.as_dict()
    emit()

    def grab():
        return sc["front_camera"].data.output["rgb"][0].cpu().numpy().copy()[..., :3].astype(np.uint8)

    def step_phys(phys30):
        phys = torch.as_tensor(phys30, dtype=torch.float32).unsqueeze(0)
        cmd = retarget.apply(mapper.map(phys).to(env.device), phys.to(env.device))
        for _ in range(2):
            env.step(cmd)
        return reward_fn.step()

    def jac():
        J = robot.root_physx_view.get_jacobians()[0]
        bi = ee_idx - 1 if J.shape[0] == len(bn) - 1 else ee_idx
        return J[bi].cpu().numpy()[:, arm_j]

    L_EE = bn.index("left_wrist_yaw_link")
    L_J = [jn.index(n) for n in (
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint")]
    # HARD limits: the soft limits plus a 0.05 rad margin clamped the commanded target
    # while the measured joint, sagging under gravity, still read as free, and the servo
    # stalled 1.5-2 cm short with no joint reported at a limit.
    _hard = getattr(robot.data, "joint_pos_limits", None)
    LIM = (_hard if _hard is not None else robot.data.soft_joint_pos_limits)[0].cpu().numpy()   # [ndof, 2]
    R_LIM = LIM[arm_j]; L_LIM = LIM[L_J]
    tube = sc["tube"]

    def _axis_z(q):
        w_, x_, y_, z_ = [float(v) for v in q]
        return np.array([2 * (x_ * z_ + y_ * w_), 2 * (y_ * z_ - x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)])

    def ik_arm(last, ee_idx, joint_ids, sl, lim, dx, lam=1e-3, hold_rot=True, margin=0.01, rot_weight=1.0, point=None):
        """One damped-least-squares step for an arm; commanded targets clamped to the soft
        joint limits (minus a margin) so a blocked motion cannot wind the arm up.
        ``dx`` is a 3-vector (translation; rotation held or free) or a 6-vector twist.
        ``point``: world position whose translation the linear rows describe (Jacobian
        transferred from the link origin: v_p = v_ee + w x r). Servoing a pusher 15-20 cm
        from the wrist origin with the wrist Jacobian never converged: the free rotation
        the solver picks moves the pusher unpredictably."""
        dx = np.asarray(dx, dtype=np.float64)
        J = robot.root_physx_view.get_jacobians()[0]
        bi = ee_idx - 1 if J.shape[0] == len(bn) - 1 else ee_idx
        J = J[bi].cpu().numpy()[:, joint_ids].copy()
        if point is not None:
            r = np.asarray(point, dtype=np.float64) - robot.data.body_pos_w[0, ee_idx].cpu().numpy()
            rx = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            J[:3] = J[:3] - rx @ J[3:]
        if dx.shape[0] == 6:
            t = dx
        elif hold_rot:
            t = np.concatenate([dx, np.zeros(3)])
        else:
            J = J[:3]; t = dx
        if J.shape[0] == 6 and rot_weight != 1.0:
            W = np.diag([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight]); J = W @ J; t = W @ t
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), t)
        new = np.array(last, dtype=np.float32).copy()
        new[sl] = np.clip(new[sl] + dq.astype(np.float32), lim[:, 0] + margin, lim[:, 1] - margin)
        return new

    def ik_step(last, dx, lam=1e-3):
        return ik_arm(last, ee_idx, arm_j, slice(7, 14), R_LIM, dx, lam)

    def ik_left(last, dx, lam=1e-3, point=None):
        # translation of ``point`` with the wrist rotation softly held (weight 0.3). Free
        # rotation let the solver put the motion into wrist rotations the PD-driven arm did
        # not realise, and the servo stalled 1.5-2 cm short with no joint at a limit.
        return ik_arm(last, L_EE, L_J, slice(0, 7), L_LIM, dx, lam, hold_rot=True, rot_weight=0.3, point=point)

    def ik_left_twist(last, v, w, lam=1e-3):
        return ik_arm(last, L_EE, L_J, slice(0, 7), L_LIM, np.concatenate([v, w]), lam)

    def tube_axis():
        return _axis_z(tube.data.root_quat_w[0].cpu().numpy())

    def tube_line_error(pt):
        """(perpendicular xy distance from pt to the tube's axis line, signed offset of pt
        along the axis from the tube centre, xy unit normal pointing from the line to pt)."""
        c = tube.data.root_pos_w[0].cpu().numpy(); a = tube_axis()
        a_xy = a[:2] / max(np.linalg.norm(a[:2]), 1e-9)
        d = pt[:2] - c[:2]
        along = float(d @ a_xy)
        perp = d - along * a_xy
        n = np.linalg.norm(perp)
        return float(n), along, (perp / n if n > 1e-9 else np.zeros(2)), a_xy

    def rod_top():
        return obj.data.body_pos_w[0, 0].cpu().numpy() + 0.09 * _axis_z(obj.data.body_quat_w[0, 0].cpu().numpy())

    def tube_bottom():
        return tube.data.root_pos_w[0].cpu().numpy() - 0.05 * _axis_z(tube.data.root_quat_w[0].cpu().numpy())

    def tube_fist_dist():
        return float(np.linalg.norm(tube.data.root_pos_w[0].cpu().numpy() - robot.data.body_pos_w[0, L_EE].cpu().numpy()))

    L_HAND_IDX = [i for i, b in enumerate(bn) if b.startswith("L_") or b == "left_hand_base_link"]
    L_BASE = bn.index("left_hand_base_link")
    L_INTER = [bn.index(f"L_{f}_intermediate") for f in ("index", "middle", "ring", "pinky")]

    def palm_dir():
        """Unit vector from the hand base toward the curled fingers: the palm side of the
        fist. The BACK of the hand is the opposite face."""
        v = robot.data.body_pos_w[0, L_INTER].cpu().numpy().mean(0) - robot.data.body_pos_w[0, L_BASE].cpu().numpy()
        return v / max(np.linalg.norm(v), 1e-9)

    def back_of_hand():
        """Pusher point: the back-of-hand surface, ~2 cm behind the base frame."""
        return robot.data.body_pos_w[0, L_BASE].cpu().numpy() - 0.02 * palm_dir()

    def palm_up_twist(gain=0.006):
        """Angular velocity that keeps the palm facing up (a translation-only step lets the
        wrist drift and undid the flip within 100 steps)."""
        v = palm_dir(); w = np.cross(v, np.array([0.0, 0.0, 1.0])); nw = np.linalg.norm(w)
        return w / max(nw, 1e-9) * min(gain, nw)

    def pusher_pd():
        """Palm-down pusher: the tube bottom while the tube is held, else the lowest fist link."""
        if tube_fist_dist() < 0.20:
            return tube_bottom()
        pts = [robot.data.body_pos_w[0, i].cpu().numpy() for i in L_HAND_IDX]
        return min(pts, key=lambda p_: p_[2])

    def left_arm_state():
        q = robot.data.joint_pos[0, L_J].cpu().numpy()
        return {"q": [round(float(v), 3) for v in q],
                "at_limit": [bool(q[i] <= L_LIM[i, 0] + 0.06 or q[i] >= L_LIM[i, 1] - 0.06) for i in range(7)]}

    def ik_left_keep_up(last, dx, lam=1e-3, point=None):
        return ik_arm(last, L_EE, L_J, slice(0, 7), L_LIM, np.concatenate([np.asarray(dx, dtype=np.float64), palm_up_twist()]), lam, rot_weight=0.5, point=point)

    def pusher():
        """Lowest point of the left fist (tube bottom if the tube is still in the fist,
        else the lowest hand link): whatever will touch the rod top first."""
        if tube_fist_dist() < 0.20:
            return tube_bottom()          # the tube sticks 5 cm out of the palm: it touches first
        pts = [robot.data.body_pos_w[0, i].cpu().numpy() for i in L_HAND_IDX]
        return min(pts, key=lambda p_: p_[2])

    def merge(stages, info):
        for k, v in info["stages"].items():
            stages[k] = stages.get(k, False) or v

    for ep in EPISODES:
        A = np.load(f"{DEMO_DIR}/ep{ep:03d}.npz")["actions"].reshape(-1, 30)
        n_demo_chunks = len(A) // H
        for cond_name, var in [(c, v) for c in CONDS for v in VARIANTS]:
            raise_m = float(var.get("raise_m", RAISE_M)); yaw_cmd = float(var.get("yaw_cmd", YAW_CMD))
            ramp_steps = int(var.get("ramp_steps", 10)); wrist = {int(k): float(v) for k, v in var.get("wrist", {}).items()}
            extra_chunks = int(var.get("extra_chunks", 0))      # demo chunks to keep after the plate stage
            to_end = bool(var.get("to_end", False))              # keep the whole demonstration
            settle_chunks = int(var.get("settle_chunks", 0))     # hold-still chunks before the raise
            seat = bool(var.get("seat", False))                  # seat the pipette on the pot before pressing
            seat_push_m = float(var.get("seat_push_m", 0.02))    # extra descent after contact
            seat_raise_m = float(var.get("seat_raise_m", 0.09))  # raise after seating (replaces raise_m)
            potpress = bool(var.get("potpress", False))          # palm press with the tip on the plate
            pp_target = float(var.get("pp_target", 0.0215))      # stop pushing at this plunger depth
            pp_max_m = float(var.get("pp_max_m", 0.10))          # max hand descent after contact
            pp_rate = float(var.get("pp_rate", 0.0008))          # descent per control step (m)
            pp_raise_m = float(var.get("pp_raise_m", 0.05))      # raise after the press
            pp_upright_gain = float(var.get("pp_upright_gain", 0.004))  # lateral m/step per unit axis_xy
            pp_thumb = bool(var.get("pp_thumb", True))           # sweep the thumb over the rod top at contact
            inject = bool(var.get("inject", False))              # bimanual: tube in the left fist presses the rod
            inj_present = np.array(var.get("inj_present", [-0.15, 0.32, 1.05]), dtype=np.float64)  # barrel centre target
            inj_target = float(var.get("inj_target", 0.0215))    # plunger depth to reach
            inj_clear = float(var.get("inj_clear", 0.06))        # pusher height above the rod top before descending
            inj_attempts = int(var.get("inj_attempts", 3))       # re-approach from another side if the press fails
            lean_deg = float(var.get("lean_deg", 30.0))          # lean of the supported pipette toward the left shoulder
            flip = bool(var.get("flip", True))                   # palm-up back-of-hand press (else palm-down fist/tube)
            inj_squeeze = float(var.get("inj_squeeze", 1.3))     # right-finger command during the inject (demo grip 1.3;
                                                                 # 1.7 measured to eject the barrel from the palm)
            dispense = bool(var.get("dispense", False))          # tip on the plate + left fist presses the rod: no friction
            inj_speed = float(var.get("inj_speed", 0.0015))      # left-arm approach speed, m per control step
            vname = var.get("name", "base")
            t0 = time.time()
            env.reset(seed=0)
            cond = None
            if cond_name != "canonical":
                cond = TRAIN_CONDS[int(cond_name)]
            # also restores the pot and tube, which the upstream reset leaves where the
            # previous episode's press pushed them
            apply_reset_condition(env, cond if cond is not None else CANONICAL)
            reward_fn.reset(); csb.reset()
            pot0 = sc["pot"].data.root_pos_w[0].cpu().numpy()
            imgs, acts, rews, states = [], [], [], []
            stages = {}
            rec = {"episode": ep, "cond": cond_name, "cond_hash": cond.hash() if cond else None,
                   "pot_start": [round(float(v), 4) for v in pot0],
                   "variant": vname, "variant_cfg": {"raise_m": raise_m, "yaw_cmd": yaw_cmd, "ramp_steps": ramp_steps, "wrist": wrist,
                                                     "extra_chunks": extra_chunks, "to_end": to_end, "settle_chunks": settle_chunks,
                                                     "seat": seat, "seat_push_m": seat_push_m, "seat_raise_m": seat_raise_m,
                                                     "potpress": potpress, "pp_target": pp_target, "pp_max_m": pp_max_m,
                                                     "pp_rate": pp_rate, "pp_raise_m": pp_raise_m, "pp_upright_gain": pp_upright_gain, "pp_thumb": pp_thumb,
                                                     "inject": inject, "inj_present": inj_present.tolist(), "inj_target": inj_target, "inj_clear": inj_clear,
                                                     "inj_squeeze": inj_squeeze, "dispense": dispense, "inj_speed": inj_speed,
                                                     "lean_deg": lean_deg, "flip": flip}}
            # ---- 1. human transport, truncated at the plate stage ----------------------
            plate_chunk = None
            for c in range(n_demo_chunks):
                chunk = A[c * H:(c + 1) * H]
                imgs.append(grab()); states.append(csb.build(stages=stages, chunk=c))
                acts.append(np.asarray(chunk, dtype=np.float32))
                tot = 0.0
                for t in range(H):
                    r, info = step_phys(chunk[t]); tot += r; merge(stages, info)
                rews.append(np.float32(tot))
                if plate_chunk is None and stages.get("plate") and stages.get("lift"):
                    plate_chunk = c
                if plate_chunk is not None and not to_end and c >= plate_chunk + extra_chunks:
                    break
            rec["transport_chunks"] = len(acts)
            rec["plate_chunk"] = plate_chunk
            rec["transport_stages"] = {k: bool(v) for k, v in stages.items()}
            if plate_chunk is None or not info["finger_hold"]:
                rec["outcome"] = "transport_failed"; rec["wall_s"] = round(time.time() - t0, 1)
                summary["episodes"].append(rec); emit(); continue
            # ---- 2. scripted press primitive ------------------------------------------
            st = {"last": A[c * H + H - 1].copy()}
            press_log = []

            def phase_steps(kind, n=None):
                """Yield (kind, action) step by step for one primitive phase."""
                last = st["last"]
                if kind == "hold":
                    for t in range(n):
                        yield kind, last
                elif kind == "raise":
                    for t in range(n):
                        last = ik_step(last, np.array([0.0, 0.0, raise_m / H])); st["last"] = last
                        yield kind, last
                elif kind == "seat":
                    # descend until the barrel stops while the hand keeps going, then push
                    bz_hist = []; contact_ee = None
                    for t in range(300):
                        last = ik_step(last, np.array([0.0, 0.0, -0.0008])); st["last"] = last
                        yield kind, last
                        bz_hist.append(float(obj.data.body_pos_w[0, 1, 2]))
                        ee_z = float(robot.data.body_pos_w[0, ee_idx, 2])
                        if contact_ee is None and t > 5 and abs(bz_hist[-1] - bz_hist[-4]) < 2e-4:
                            contact_ee = ee_z; rec["seat_contact_step"] = t
                            rec["seat_barrel_bottom_z"] = round(bz_hist[-1] - 0.095, 4)
                        if contact_ee is not None and contact_ee - ee_z > seat_push_m:
                            break
                    for t in range(int(round(seat_raise_m / 0.001))):
                        last = ik_step(last, np.array([0.0, 0.0, 0.001])); st["last"] = last
                        yield kind, last
                elif kind == "centre":
                    # bring the TIP (barrel bottom, 9.5 cm down the barrel axis; the pipette
                    # hangs tilted 10-40 deg) over the plate centre so it lands inside the pot
                    for t in range(120):
                        q = obj.data.body_quat_w[0, 1].cpu().numpy(); w_, x_, y_, z_ = [float(v) for v in q]
                        axis = np.array([2 * (x_ * z_ + y_ * w_), 2 * (y_ * z_ - x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)])
                        tip = obj.data.body_pos_w[0, 1].cpu().numpy() - 0.095 * axis
                        # tip 2 cm to the +x side of the plate centre: leaning 30 deg toward -x
                        # displaces the barrel centre 4.75 cm toward -x, so it lands ~3 cm from
                        # the plate centre, inside the 6 cm dispense gate
                        err = sc["pot"].data.root_pos_w[0, :2].cpu().numpy() + np.array([0.02, 0.0]) - tip[:2]
                        if np.linalg.norm(err) < 0.008:
                            break
                        step = err / max(np.linalg.norm(err), 1e-9) * min(0.002, float(np.linalg.norm(err)))
                        last = ik_step(last, np.array([step[0], step[1], 0.0])); st["last"] = last
                        yield kind, last
                    rec["centre_steps"] = t; rec["centre_err_m"] = round(float(np.linalg.norm(err)), 4)
                    rec["tilt_deg_at_centre"] = round(float(np.degrees(np.arccos(np.clip(axis[2], -1, 1)))), 1)
                elif kind == "potpress":
                    # descend until the barrel stops (tip on the plate) while the hand keeps
                    # going, then keep descending so the hand slides down the barrel until the
                    # palm drives the rod in; stop at the target depth (closed loop on the
                    # measured plunger) or at the descent cap
                    # contact = the barrel bottom has reached the plate floor (geometric, not a
                    # stall test: a tip sliding on the floor never stalls cleanly)
                    # the "pot" is a 12 x 8 x 4 cm kinematic hole plate: the tip rests on its
                    # TOP face (pot z + 0.04); its 6 mm holes cannot admit the 19 mm barrel
                    plate_top = float(sc["pot"].data.root_pos_w[0, 2]) + 0.04
                    ez0 = float(robot.data.body_pos_w[0, ee_idx, 2]); contact_ee = None
                    trace = []
                    for t in range(400):
                        # once the tip rests on the plate, move the hand horizontally so the
                        # grip point comes above the tip: the pipette rotates upright and the
                        # rod top ends up under the palm. Free descent before contact.
                        lat = np.zeros(2)
                        if contact_ee is not None:
                            q = obj.data.body_quat_w[0, 1].cpu().numpy(); w_, x_, y_, z_ = [float(v) for v in q]
                            ax = np.array([2 * (x_ * z_ + y_ * w_), 2 * (y_ * z_ - x_ * w_)])
                            lat = -pp_upright_gain * ax
                            n = np.linalg.norm(lat)
                            if n > 0.002:
                                lat = lat / n * 0.002
                        last = ik_step(last, np.array([lat[0], lat[1], -pp_rate]))
                        if pp_thumb and contact_ee is not None:
                            # thumb over the rod top (press channel): a rigid pusher above the
                            # rod, where the index finger alone deflects and lets the rod slip
                            y_now = float(last[25]); last[25] = min(yaw_cmd, y_now + 0.02)
                        st["last"] = last
                        yield kind, last
                        bz = float(obj.data.body_pos_w[0, 1, 2]) - 0.095
                        ee_z = float(robot.data.body_pos_w[0, ee_idx, 2])
                        pressed = float(obj.data.joint_pos[0, pj])
                        if t % 10 == 0:
                            q = obj.data.body_quat_w[0, 1].cpu().numpy(); w_, x_, y_, z_ = [float(v) for v in q]
                            tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (x_ * x_ + y_ * y_), -1, 1))))
                            trace.append([t, round(ee_z, 4), round(bz, 4), round(pressed, 4), round(float(robot.data.body_pos_w[0, ee_idx, 2] - obj.data.body_pos_w[0, 1, 2]), 4), round(tilt, 1)])
                        rec["pp_trace [t, ee_z, barrel_bottom_z, press, ee_minus_barrel_z, tilt_deg]"] = trace
                        if contact_ee is None and (bz <= plate_top + 0.012 or pressed > 0.014):
                            contact_ee = ee_z; rec["pp_contact_step"] = t; rec["pp_barrel_bottom_z"] = round(bz, 4)
                        if contact_ee is None and ez0 - ee_z > 0.30:
                            rec["pp_no_contact"] = True; break
                        if contact_ee is not None and (pressed >= pp_target or contact_ee - ee_z > pp_max_m):
                            rec["pp_descent_after_contact"] = round(contact_ee - ee_z, 4); rec["pp_press_reached"] = round(pressed, 4)
                            break
                elif kind == "present":
                    # right arm: bring the barrel centre to the presentation point (closed loop);
                    # tighten the grip first so the press reaction cannot slide the barrel
                    for t in range(15):
                        last = last.copy(); last[20:24] = np.minimum(last[20:24] + (inj_squeeze - 1.3) / 15.0, inj_squeeze); st["last"] = last
                        yield kind, last
                    for t in range(300):
                        err = inj_present - obj.data.body_pos_w[0, 1].cpu().numpy()
                        if np.linalg.norm(err) < 0.01:
                            break
                        step = err / max(np.linalg.norm(err), 1e-9) * min(0.003, float(np.linalg.norm(err)))
                        last = ik_step(last, step); st["last"] = last
                        yield kind, last
                    rec["present_steps"] = t; rec["present_err_m"] = round(float(np.linalg.norm(err)), 4)
                elif kind == "approach":
                    # left arm, three legs so the tube never sweeps through the right hand:
                    # up to clearance height, over the rod top, then the descend phase
                    d0 = tube_fist_dist()
                    # UP: fist bottom to clearance height above the rod top
                    for t in range(400):
                        goal_z = rod_top()[2] + inj_clear + 0.03
                        err = np.array([0.0, 0.0, goal_z - pusher()[2]])
                        if abs(err[2]) < 0.012:
                            break
                        speed = inj_speed * min(1.0, (t + 1) / 20.0)
                        step = err / max(np.linalg.norm(err), 1e-9) * min(speed, float(np.linalg.norm(err)))
                        last = ik_left(last, step, point=pusher()); st["last"] = last
                        yield kind, last
                    rec["approach_up_steps"] = t; rec["approach_up_err_m"] = round(float(np.linalg.norm(err)), 4)
                    # FLIP: palm up. The tube, held only by friction, slides out of the fist
                    # under any press load or motion along its axis (measured in every
                    # orientation). Palm up, it is cradled in the channel on top of the fist and
                    # out of the way, and the BACK of the hand -- flat, rigid, 5-6 cm wide --
                    # becomes the pusher. Rotation about the fist's own centre.
                    for t in range(400 if flip else 0):
                        v = palm_dir()
                        if v[2] > 0.90:
                            break
                        w = np.cross(v, np.array([0.0, 0.0, 1.0])); nw = np.linalg.norm(w)
                        w = w / max(nw, 1e-9) * min(0.006, nw)
                        last = ik_left_twist(last, np.zeros(3), w); st["last"] = last
                        yield kind, last
                    rec["flip_steps"] = t; rec["palm_up_after_flip"] = round(float(palm_dir()[2]), 3)
                    rec["tube_fist_dist_after_flip"] = round(tube_fist_dist(), 4)
                    # UP again (the flip moves the pusher), then OVER: back of the hand above the rod top
                    push_pt = back_of_hand if flip else pusher_pd
                    move = ik_left_keep_up if flip else ik_left
                    side = st.get("side", np.array([-0.04, 0.0]))
                    for leg in ("up2", "side", "over"):
                        for t in range(400):
                            goal = rod_top() + np.array([0.0, 0.0, inj_clear])
                            cur = push_pt()
                            if leg == "up2":
                                err = np.array([0.0, 0.0, goal[2] - cur[2]])
                            elif leg == "side":
                                # waypoint beside the rod top, so the tube does not come in
                                # through the right hand (the over leg stalled at 2 cm otherwise)
                                err = np.array([goal[0] + side[0] - cur[0], goal[1] + side[1] - cur[1], 0.0])
                            else:
                                err = np.array([goal[0] - cur[0], goal[1] - cur[1], 0.0])
                            if np.linalg.norm(err) < 0.008:
                                break
                            e = float(np.linalg.norm(err)); speed = inj_speed * min(1.0, (t + 1) / 20.0)
                            step = err / max(e, 1e-9) * min(speed, e)
                            last = move(last, step, point=cur); st["last"] = last
                            yield kind, last
                        rec[f"approach_{leg}_steps"] = t; rec[f"approach_{leg}_err_m"] = round(float(np.linalg.norm(err)), 4)
                        rec[f"left_arm_after_{leg}"] = left_arm_state()
                    rec["palm_up_before_inject"] = round(float(palm_dir()[2]), 3)
                elif kind == "attempt":
                    if st.get("pressed_ok"):
                        return
                    sides = [np.array([-0.04, 0.0]), np.array([0.0, 0.04]), np.array([0.0, -0.04]), np.array([0.04, 0.0])]
                    st["side"] = sides[n % len(sides)]
                    if n > 0:
                        for t in range(50):                          # back off 5 cm before re-approaching
                            last = ik_left(last, np.array([0.0, 0.0, 0.001]), point=pusher_pd()); st["last"] = last
                            yield kind, last
                    for sub in ("approach", "inject"):
                        for item in phase_steps(sub):
                            yield item
                        last = st["last"]
                    rec.setdefault("attempts", []).append({"n": n, "side": st["side"].tolist(), "press": rec.get("inject_press_reached"),
                                                           "over_err": rec.get("approach_over_err_m")})
                elif kind == "inject":
                    # left arm descends, servoing the tube bottom over the rod top, until the
                    # plunger reaches the target depth (closed loop) or the descent cap
                    z0 = float(robot.data.body_pos_w[0, L_EE, 2]); trace = []
                    for t in range(300):
                        # back of the hand over the rod top; descend while aligned within 2 cm
                        pp = back_of_hand() if flip else pusher_pd()
                        err_xy = (rod_top() - pp)[:2]; e = float(np.linalg.norm(err_xy))
                        lat = err_xy / max(e, 1e-9) * min(0.0015, e)
                        dz = -(0.0006 if dispense else 0.0008) if e < 0.025 else 0.0
                        last = (ik_left_keep_up if flip else ik_left)(last, np.array([lat[0], lat[1], dz]), point=pp)
                        st["last"] = last
                        yield kind, last
                        pressed = float(obj.data.joint_pos[0, pj])
                        if t % 10 == 0:
                            trace.append([t, round(pressed, 4), round(e, 4), round(float(palm_dir()[2]), 3), round(tube_fist_dist(), 4)])
                        if pressed >= inj_target or z0 - float(robot.data.body_pos_w[0, L_EE, 2]) > 0.12:
                            break
                    rec["inject_steps"] = t; rec["inject_descent_m"] = round(z0 - float(robot.data.body_pos_w[0, L_EE, 2]), 4)
                    rec["inject_press_reached"] = round(pressed, 4); rec["inject_trace [t, press, xy_err, palm_up, tube_fist_dist]"] = trace
                    st["pressed_ok"] = bool(pressed >= inj_target)
                elif kind == "upright":
                    # tip on the plate: ROTATE the hand about the tip until the barrel leans
                    # ``lean_deg`` toward the left shoulder (-x). Upright, the rod top sits at
                    # the plate centre, which is at the flipped left fist's reach limit (it
                    # stalled 12-15 cm short); leaning 30 deg brings the rod top ~10 cm closer
                    # and 3 cm lower while the barrel centre stays inside the plate gate.
                    goal_ax = np.array([-np.sin(np.radians(lean_deg)), 0.0, np.cos(np.radians(lean_deg))])
                    for t in range(250):
                        ax = _axis_z(obj.data.body_quat_w[0, 1].cpu().numpy())
                        tilt = float(np.degrees(np.arccos(np.clip(ax @ goal_ax, -1, 1))))
                        if tilt < 3.0:
                            break
                        w = np.cross(ax, goal_ax)                              # rotation bringing ax to the goal
                        nw = np.linalg.norm(w)
                        w = w / max(nw, 1e-9) * min(0.004, nw)                # rad per step
                        tip = obj.data.body_pos_w[0, 1].cpu().numpy() - 0.095 * ax
                        ee = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
                        v = np.cross(w, ee - tip)                              # keep the tip fixed
                        last = ik_step(last, np.concatenate([v, w])); st["last"] = last
                        yield kind, last
                    rec["upright_steps"] = t; rec["upright_tilt_deg"] = round(tilt, 1)
                    rec["barrel_axis_after_lean"] = [round(float(v), 3) for v in _axis_z(obj.data.body_quat_w[0, 1].cpu().numpy())]
                elif kind == "release":
                    for t in range(int(round(0.05 / 0.001))):
                        last = ik_left(last, np.array([0.0, 0.0, 0.001])); st["last"] = last
                        yield kind, last
                elif kind == "raise_after":
                    for t in range(int(round(pp_raise_m / 0.001))):
                        last = ik_step(last, np.array([0.0, 0.0, 0.001])); st["last"] = last
                        yield kind, last
                elif kind == "press":
                    y0 = float(last[25]); w0 = {d: float(last[d]) for d in wrist}
                    for t in range(n):
                        frac = min(1.0, (t + 1) / ramp_steps)
                        last = last.copy(); last[25] = y0 + (yaw_cmd - y0) * frac
                        for d, off in wrist.items():
                            last[d] = w0[d] + off * frac
                        st["last"] = last
                        yield kind, last

            def run_primitive(phases):
                """Execute phases, recording fixed 30-step chunks; the last chunk is padded with holds."""
                stream = (x for ph in phases for x in phase_steps(*ph))
                chunk_acts = []; tot = 0.0; c_index = len(acts)
                def flush():
                    nonlocal chunk_acts, tot, c_index
                    acts.append(np.asarray(chunk_acts, dtype=np.float32)); rews.append(np.float32(tot))
                    chunk_acts = []; tot = 0.0; c_index += 1
                done = False
                while not done:
                    imgs.append(grab()); states.append(csb.build(stages=stages, chunk=c_index))
                    for t in range(H):
                        try:
                            kind, a = next(stream)
                        except StopIteration:
                            kind, a = "hold", st["last"]; done = True
                        chunk_acts.append(np.array(a, dtype=np.float32).copy())
                        r, info = step_phys(a); tot += r; merge(stages, info)
                        press_log.append({"kind": kind, "press": info["press_m"], "lift": info["lift"],
                                          "finger_hold": info["finger_hold"], "sustain": info["press_sustain_steps"],
                                          "dsustain": info.get("dispense_sustain_steps", 0), "d_pot_xy": info["d_pot_xy"]})
                    flush()

            c_next = len(acts)
            phases = [("hold", H)] * settle_chunks
            if dispense:
                # tip rested on the plate (support from below), pipette uprighted, left fist
                # presses the rod from above: the force path never loads the right grip.
                # Up to inj_attempts approach+inject rounds, each from another side.
                pp_max_m = 0.005; pp_thumb = False
                phases += [("centre", None), ("potpress", None), ("upright", None), ("centre", None)]
                for _k in range(inj_attempts):
                    phases += [("attempt", _k)]
                phases += [("hold", H), ("release", None), ("hold", H)]
            elif inject:
                phases += [("present", None), ("approach", None), ("inject", None), ("hold", H), ("hold", H)]
            elif potpress:
                phases += [("centre", None), ("potpress", None), ("hold", H), ("raise_after", None), ("hold", H)]
            else:
                phases += [("seat", None)] if seat else [("raise", H)]
                phases += [("press", H), ("hold", H)]
            run_primitive(phases)
            # trailing observation so (s, a, r, s') pairs exist for every chunk
            imgs.append(grab()); states.append(csb.build(stages=stages, chunk=len(acts)))
            acts.append(acts[-1]); rews.append(np.float32(0.0))
            pl = [x for x in press_log if x["kind"] in ("press", "hold", "potpress", "raise_after", "inject", "release")] or press_log
            pre = [x for x in press_log if x["kind"] in ("raise", "seat", "centre", "present", "approach", "upright")]
            rec.update({
                "stages": {k: bool(v) for k, v in stages.items()},
                "return_v5": round(float(sum(rews)), 3),
                "n_chunks": len(acts) - 1,
                "synthetic_from_chunk": c_next,
                "press_max_m": round(max(x["press"] for x in pl), 5),
                "press_end_m": round(pl[-1]["press"], 5),
                "lift_min_during_press": round(min(x["lift"] for x in pl), 4),
                "finger_hold_frac_during_press": round(float(np.mean([x["finger_hold"] for x in pl])), 3),
                "max_sustain_steps": int(max(x["sustain"] for x in pl)),
                "max_dispense_steps": int(max(x.get("dsustain", 0) for x in pl)),
                "press_max_at_plate_m": round(max(x["press"] for x in pl if x["kind"] in ("potpress", "hold")), 5) if any(x["kind"] == "potpress" for x in pl) else None,
                "d_pot_xy_end": round(pl[-1]["d_pot_xy"], 4),
                "lift_after_raise": round(pre[-1]["lift"], 4) if pre else None,
                "finger_hold_after_raise": bool(pre[-1]["finger_hold"]) if pre else None,
                "wall_s": round(time.time() - t0, 1),
            })
            ok = bool(stages.get("press")) if inject else bool(stages.get("dispense"))
            if inject or dispense:
                rec["tube_fist_dist_end"] = round(tube_fist_dist(), 4)
            rec["outcome"] = "press_certified" if ok else "press_failed"
            if ok or KEEP_FAILED:
                tag = "c" if cond_name == "canonical" else f"t{int(cond_name):03d}"
                vtag = "" if len(VARIANTS) == 1 else f"_{vname}"
                out = os.path.join(OUTDIR, f"ep{ep:03d}_{tag}{vtag}.npz")
                np.savez_compressed(out, images=np.stack(imgs), actions=np.stack(acts),
                                    rewards=np.stack(rews), critic_state=np.stack(states).astype(np.float32))
                rec["file"] = out
            summary["episodes"].append(rec); emit()
    eps = summary["episodes"]
    summary["n_certified"] = sum(1 for e in eps if e.get("outcome") == "press_certified")
    summary["n_press_failed"] = sum(1 for e in eps if e.get("outcome") == "press_failed")
    summary["n_transport_failed"] = sum(1 for e in eps if e.get("outcome") == "transport_failed")
    emit("OK"); os._exit(0)
except Exception as e:  # noqa: BLE001
    summary["error"] = f"{type(e).__name__}: {e}"; summary["tb"] = traceback.format_exc(); emit("FAIL"); os._exit(1)
