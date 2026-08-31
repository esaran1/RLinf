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

"""Privileged critic conditioning: proprioception, object state, and phase.

A review point: *"Critic should have access to proprioceptive state and phase label.
Pure RGB figure is too weak."* That is correct, and it is a standard asymmetric
actor-critic argument (Pinto et al., "Asymmetric Actor Critic", 2017): the actor must
stay deployable from the sensors a real robot has (here, one RGB image), but the critic
is discarded at deployment, so conditioning it on privileged simulator state is free and
strictly reduces value-estimation variance.

It matters more than usual here. The critic must fit a Q-function over a **900-D** action
chunk from roughly 7k transitions; every bit of state that removes ambiguity from the
value target is leverage. A single 424x240 RGB frame does not disambiguate:

* joint configuration behind occlusion (the hand is often hidden by the pipette/arm),
* the plunger's depression, which is a few millimetres of travel inside the grip,
* which phase of a long-horizon multi-stage task the episode is in.

The last point is the sharpest: reward is stage-gated, so the same image can carry very
different value depending on which stages have already fired. Without a phase label the
critic must infer episode history from a single frame, which is impossible in principle,
not merely hard.

What this module builds (all read directly off the simulator, none of it learned):

``proprio``   right-arm and right-hand joint positions and velocities, plus fingertip
              and thumb positions in the barrel's frame.
``object``    barrel and rod position/velocity, plunger joint position and velocity,
              barrel pose relative to the tube and pot.
``phase``     one-hot of stages already achieved, plus normalized episode progress.

The vector is **fixed-width and ordered**; :data:`FEATURE_NAMES` documents every slot so
a saved critic can be interpreted later. Everything is finite-checked, because a single
NaN in a value target silently poisons a whole run.
"""

from __future__ import annotations

import numpy as np

#: Stages, in task order. Must match the reward module's ``stage_order()``.
PHASE_STAGES = ("reach", "grasp", "lift", "tube", "press", "dispense", "plate", "success")

#: Right-arm joints in the 53-DoF ordering are selected by name at build time; these are
#: the names the G1 uses. Kept explicit so a robot change fails loudly.
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
RIGHT_HAND_JOINTS = (
    "R_index_proximal_joint", "R_middle_proximal_joint", "R_pinky_proximal_joint",
    "R_ring_proximal_joint", "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint", "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint", "R_ring_intermediate_joint",
    "R_thumb_intermediate_joint", "R_thumb_distal_joint",
)


def feature_names() -> list[str]:
    """Ordered names of every slot in the critic-state vector."""
    names = []
    for j in RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS:
        names.append(f"qpos:{j}")
    for j in RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS:
        names.append(f"qvel:{j}")
    for tag in ("finger_cx", "finger_cy", "finger_cz", "thumb_x", "thumb_y", "thumb_z"):
        names.append(f"rel_barrel:{tag}")
    names += ["barrel_x", "barrel_y", "barrel_z", "barrel_vx", "barrel_vy", "barrel_vz"]
    names += ["rod_dz", "plunger_pos", "plunger_vel"]
    names += ["barrel_minus_tube_x", "barrel_minus_tube_y", "barrel_minus_tube_z"]
    names += ["barrel_minus_pot_x", "barrel_minus_pot_y", "barrel_minus_pot_z"]
    names += [f"phase:{s}" for s in PHASE_STAGES]
    names += ["episode_progress"]
    return names


#: Every slot, in order. Length is :data:`STATE_DIM`.
FEATURE_NAMES = feature_names()
#: Width of the critic-state vector.
STATE_DIM = len(FEATURE_NAMES)


class CriticStateBuilder:
    """Builds the privileged critic-state vector from a live scene.

    Args:
        scene: the IsaacLab scene.
        max_chunks: episode length in chunks, used to normalize progress to [0, 1].
    """

    def __init__(self, scene, max_chunks: int = 23):
        self.scene = scene
        self.max_chunks = int(max_chunks)
        robot = scene["robot"]
        jn = list(robot.data.joint_names)
        missing = [j for j in RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS if j not in jn]
        if missing:
            raise ValueError(f"joints absent from this robot: {missing}")
        self.joint_idx = [jn.index(j) for j in RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS]
        bn = list(robot.data.body_names)
        self.finger_idx = [bn.index(f"R_{f}_intermediate")
                           for f in ("index", "middle", "pinky", "ring")]
        self.thumb_idx = bn.index("R_thumb_distal")
        obj = scene["object"]
        ojn = list(getattr(obj.data, "joint_names", []) or [])
        self.piston_idx = ojn.index("PistonJoint") if "PistonJoint" in ojn else None
        self._prev_barrel = None

    def reset(self):
        """Clear cross-step state (the finite-difference velocity)."""
        self._prev_barrel = None

    def build(self, stages: dict | None = None, chunk: int = 0,
              dt: float = 0.02) -> np.ndarray:
        """Return the critic-state vector for the current simulator state.

        Args:
            stages: mapping of stage name -> bool for stages already achieved.
            chunk: index of the current chunk, for episode progress.
            dt: control period, for the finite-difference barrel velocity.
        """
        sc = self.scene
        robot, obj = sc["robot"], sc["object"]
        qpos = robot.data.joint_pos[0].detach().float().cpu().numpy()[self.joint_idx]
        qvel = robot.data.joint_vel[0].detach().float().cpu().numpy()[self.joint_idx]
        bp = robot.data.body_pos_w[0].detach().float().cpu().numpy()
        fing_c = bp[self.finger_idx].mean(axis=0)
        thumb = bp[self.thumb_idx]

        obp = obj.data.body_pos_w[0].detach().float().cpu().numpy()
        rod, barrel = obp[0], obp[1]
        tube = sc["tube"].data.root_pos_w[0].detach().float().cpu().numpy()
        pot = sc["pot"].data.root_pos_w[0].detach().float().cpu().numpy()

        if self._prev_barrel is None:
            bvel = np.zeros(3)
        else:
            bvel = (barrel - self._prev_barrel) / max(dt, 1e-6)
        self._prev_barrel = barrel.copy()

        if self.piston_idx is not None:
            ppos = float(obj.data.joint_pos[0, self.piston_idx].detach().float().cpu())
            pvel_t = getattr(obj.data, "joint_vel", None)
            pvel = (float(pvel_t[0, self.piston_idx].detach().float().cpu())
                    if pvel_t is not None else 0.0)
        else:
            ppos, pvel = 0.0, 0.0

        stages = stages or {}
        phase = [1.0 if stages.get(s) else 0.0 for s in PHASE_STAGES]
        progress = float(np.clip(chunk / max(self.max_chunks, 1), 0.0, 1.0))

        vec = np.concatenate([
            qpos, qvel,
            (fing_c - barrel), (thumb - barrel),
            barrel, bvel,
            [float(rod[2] - barrel[2]), ppos, pvel],
            (barrel - tube), (barrel - pot),
            phase, [progress],
        ]).astype(np.float32)

        if vec.shape[0] != STATE_DIM:
            raise ValueError(f"critic state width {vec.shape[0]} != {STATE_DIM}")
        # A single NaN in a value target poisons the run silently; fail loudly instead.
        if not np.all(np.isfinite(vec)):
            bad = [FEATURE_NAMES[i] for i in np.where(~np.isfinite(vec))[0]]
            raise ValueError(f"non-finite critic state at: {bad}")
        return vec
