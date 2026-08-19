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

"""Initial-state randomization for the G1 piston task.

The upstream task has exactly **one** initial state: ``reset(seed=s)`` returns a
bit-identical scene for every seed (barrel ``[-0.15, 0.40, 0.89]``, pot
``[0.15, 0.30, 0.80]``, joint-position sum ``0.107856``), so a seed-indexed evaluation
suite has an effective sample size of 1 regardless of how many seeds it names. See
``docs/contracts/g1_piston_eval_suite_defect.json``.

Why the perturbation is where it is
-----------------------------------

*Piston xy* is nearly pinned. Four kinematic socket walls hold the barrel upright: wall
centres sit ``_SOCKET_HALF = 0.032`` m from the piston axis with ``_WALL_T = 0.02`` m
thickness, so the inner faces are 0.022 m from the axis against a 0.020 m barrel radius
-- **2 mm** of radial clearance. The inherited cylinder task randomised the object by
+/-0.05 m, which upstream deliberately zeroed for the piston because it jams the barrel
into a socket corner. This module therefore keeps piston jitter to a fraction of that
clearance and never approaches a wall.

*Robot joint configuration* is the variable that actually changes the task: it moves the
hand's starting pose relative to a fixed piston, which is exactly the reach problem the
policy has to solve. The 67 demonstrations are effectively identical at reset (max std
0.021 rad across all 63 state dims, and that only on the left-hand fingers already
gripping the tube), so there is no empirical range to inherit and the perturbation is
introduced deliberately and conservatively.

Nothing here modifies scene geometry, object dimensions, camera calibration, the hand
retargeter, the reward, the action mapping, or control timing. It only perturbs the
*state* the simulator is left in immediately after ``env.reset()``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

#: Socket geometry (upstream ``base_scene_pickplace_pistoncfg``), used to bound jitter.
SOCKET_HALF = 0.032
WALL_THICKNESS = 0.02
BARREL_RADIUS = 0.020
#: Radial gap between the barrel surface and a socket wall's inner face: 2 mm.
SOCKET_CLEARANCE = SOCKET_HALF - WALL_THICKNESS / 2.0 - BARREL_RADIUS

#: Fraction of the socket clearance the piston may be displaced by. Deliberately well
#: inside the wall so a reset can never spawn the barrel in contact.
PISTON_JITTER_FRACTION = 0.5
PISTON_JITTER_M = SOCKET_CLEARANCE * PISTON_JITTER_FRACTION  # 1.0 mm

#: Right-arm joints. These set where the grasping hand starts, so perturbing them varies
#: the reach without touching the left hand's grip on the tube or the object poses.
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

#: Waist joints: they move the whole upper body relative to the table.
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")

#: Per-joint perturbation (rad), uniform +/-. ~2.9 deg on the arm is a visible change in
#: hand pose (order centimetres at the fingertip) while staying well inside the joint
#: limits and far from self-collision.
ARM_JITTER_RAD = 0.05
WAIST_JITTER_RAD = 0.02


@dataclass(frozen=True)
class ResetCondition:
    """One reproducible initial condition."""

    index: int
    split: str  # "train" | "eval"
    joint_delta: dict = field(default_factory=dict)   # joint name -> radians
    piston_dxy: tuple = (0.0, 0.0)                    # metres

    def state_vector(self):
        """Flat vector of everything this condition perturbs, in a stable order."""
        names = sorted(self.joint_delta)
        return np.array([self.joint_delta[n] for n in names]
                        + list(self.piston_dxy), dtype=np.float64)

    def hash(self) -> str:
        payload = json.dumps(
            {"joint_delta": {k: round(v, 9) for k, v in sorted(self.joint_delta.items())},
             "piston_dxy": [round(v, 9) for v in self.piston_dxy]},
            sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_dict(self):
        return {"index": self.index, "split": self.split, "hash": self.hash(),
                "joint_delta": self.joint_delta, "piston_dxy": list(self.piston_dxy)}


def _draw(rng, joints):
    d = {}
    for j in joints:
        scale = WAIST_JITTER_RAD if "waist" in j else ARM_JITTER_RAD
        d[j] = float(rng.uniform(-scale, scale))
    return d


#: The n_train the experiment actually uses. build_reset_suite draws TRAIN then EVAL
#: from one generator, so the eval conditions depend on n_train: rebuilding the suite
#: with a different n_train silently yields DIFFERENT eval conditions. Every consumer
#: (trainer and post-hoc evaluators alike) must pass this value.
EXPERIMENT_N_TRAIN = 400


def build_reset_suite(n_train: int = EXPERIMENT_N_TRAIN, n_eval: int = 50,
                      seed: int = 20260817):
    """Build disjoint TRAIN and EVAL initial-condition sets.

    The two splits are drawn from one generator in sequence, so they are disjoint by
    construction and reproducible from ``seed`` alone.
    """
    rng = np.random.default_rng(seed)
    joints = list(RIGHT_ARM_JOINTS) + list(WAIST_JOINTS)

    train, eval_ = [], []
    for i in range(n_train):
        train.append(ResetCondition(
            index=i, split="train", joint_delta=_draw(rng, joints),
            piston_dxy=(float(rng.uniform(-PISTON_JITTER_M, PISTON_JITTER_M)),
                        float(rng.uniform(-PISTON_JITTER_M, PISTON_JITTER_M)))))
    for i in range(n_eval):
        eval_.append(ResetCondition(
            index=i, split="eval", joint_delta=_draw(rng, joints),
            piston_dxy=(float(rng.uniform(-PISTON_JITTER_M, PISTON_JITTER_M)),
                        float(rng.uniform(-PISTON_JITTER_M, PISTON_JITTER_M)))))
    return train, eval_


#: The canonical, unperturbed reset -- kept as a named diagnostic condition. Its outcome
#: is a single binary observation and must never be reported as a success *rate*.
CANONICAL = ResetCondition(index=-1, split="canonical", joint_delta={}, piston_dxy=(0.0, 0.0))


def apply_reset_condition(env, condition: ResetCondition):
    """Apply ``condition`` to a freshly reset env, in place.

    Must be called immediately after ``env.reset()``. Writes joint positions and the
    piston root pose through the simulation views, then steps physics zero times -- the
    caller's first ``env.step`` picks the new state up.
    """
    import torch

    scene = env.scene
    robot = scene["robot"]
    if condition.joint_delta:
        names = list(robot.data.joint_names)
        q = robot.data.joint_pos.clone()
        for jname, delta in condition.joint_delta.items():
            if jname not in names:
                raise KeyError(f"joint {jname!r} not in articulation")
            q[:, names.index(jname)] += delta
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))

    dx, dy = condition.piston_dxy
    if dx or dy:
        obj = scene["object"]
        root = obj.data.root_state_w.clone()
        root[:, 0] += dx
        root[:, 1] += dy
        obj.write_root_state_to_sim(root)

    scene.write_data_to_sim()
    return condition


def suite_manifest(train, eval_):
    """Serialisable description of the suite, for the evaluation contract."""
    return {
        "n_train": len(train),
        "n_eval": len(eval_),
        "disjoint": len({c.hash() for c in train} & {c.hash() for c in eval_}) == 0,
        "arm_jitter_rad": ARM_JITTER_RAD,
        "waist_jitter_rad": WAIST_JITTER_RAD,
        "piston_jitter_m": PISTON_JITTER_M,
        "socket_clearance_m": SOCKET_CLEARANCE,
        "joints": list(RIGHT_ARM_JOINTS) + list(WAIST_JOINTS),
        "eval_hashes": [c.hash() for c in eval_],
        "train_hashes_head": [c.hash() for c in train[:10]],
    }
