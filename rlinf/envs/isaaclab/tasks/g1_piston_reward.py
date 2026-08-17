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

"""Task-grounded reward for the G1 piston long-horizon task.

Replaces the inherited cylinder-region predicate, which was authored for a different
scene and never fired in 2000 env steps of evaluation.

Every term is a quantity the simulator measures directly: rigid-body positions of the
piston (Rod/Barrel), the tube, and the pot, plus right-hand fingertip positions. There
is no visual/language reward model and no term that cannot be read off the scene.

Shaping is **potential-based on best-so-far progress**: each stage contributes a bounded
one-off bonus the first time it is reached, and the continuous term only pays for
*improvement* on the closest approach achieved so far. That means an agent cannot farm
reward by oscillating (approach/retreat/approach), which is the main reward-hacking risk
for a reach-and-place task.

Stage structure, in task order::

    reach    right fingertips within GRASP_RADIUS of the barrel axis
    grasp    fingers AND thumb both near the barrel (opposing contact)
    lift     barrel raised LIFT_H above its resting height
    tube     barrel brought near the tube (the insertion target)
    plate    barrel brought over the pot in xy
    success  barrel over the pot, above plate height, and no longer rising

``success`` is the explicit terminal predicate; everything else is shaping.
"""

from __future__ import annotations

import numpy as np
import torch

#: Barrel radius (scene cfg). Fingertips must come within ~this to contact.
BARREL_RADIUS = 0.020
#: Fingertip radial distance counted as "at the barrel".
GRASP_RADIUS = 0.045
#: Lift above resting height that counts as the piston being picked up.
LIFT_H = 0.05
#: Distance from the barrel to the tube counted as "brought to the tube".
TUBE_NEAR = 0.10
#: xy distance from the pot counted as "over the plate".
PLATE_NEAR = 0.06
#: Height above the table the piston must clear when over the plate.
PLATE_Z = 0.95

#: One-off bonus per stage, in task order. Later stages are worth strictly more, so a
#: trajectory that reaches a later stage always outranks one that stalls earlier.
STAGE_BONUS = {
    "reach": 1.0,
    "grasp": 2.0,
    "lift": 4.0,
    "tube": 6.0,
    "plate": 8.0,
    "success": 20.0,
}

#: Weight on best-so-far approach progress (continuous shaping).
W_APPROACH = 2.0
#: Weight on best-so-far transport progress once the piston is held.
W_TRANSPORT = 4.0
#: Small per-step cost so dithering is never free.
STEP_COST = 0.001


class PistonTaskReward:
    """Stateful reward. Call :meth:`reset` at episode start, :meth:`step` per env step."""

    def __init__(self, scene, joint_names, device="cpu"):
        self.scene = scene
        self.device = device
        body_names = list(scene["robot"].data.body_names)
        self.finger_idx = [
            body_names.index(f"R_{f}_intermediate")
            for f in ("index", "middle", "pinky", "ring")
        ]
        self.thumb_idx = body_names.index("R_thumb_distal")
        self.reset()

    def reset(self):
        self._stages = {k: False for k in STAGE_BONUS}
        self._rest_z = None
        self._best_approach = None
        self._best_transport = None
        self._prev_z = None

    def _read(self):
        sc = self.scene
        barrel = sc["object"].data.body_pos_w[0, 1].detach().float().cpu().numpy()
        tube = sc["tube"].data.root_pos_w[0].detach().float().cpu().numpy()
        pot = sc["pot"].data.root_pos_w[0].detach().float().cpu().numpy()
        rb = sc["robot"].data.body_pos_w[0]
        fing = rb[self.finger_idx].detach().float().cpu().numpy()
        thumb = rb[self.thumb_idx].detach().float().cpu().numpy()
        return barrel, tube, pot, fing, thumb

    def step(self):
        """Return ``(reward, info)`` for the current simulator state."""
        barrel, tube, pot, fing, thumb = self._read()
        if self._rest_z is None:
            self._rest_z = float(barrel[2])

        finger_r = float(np.mean(np.linalg.norm((fing - barrel)[:, :2], axis=1)))
        thumb_r = float(np.linalg.norm((thumb - barrel)[:2]))
        lift = float(barrel[2] - self._rest_z)
        d_tube = float(np.linalg.norm(barrel - tube))
        d_pot_xy = float(np.linalg.norm((barrel - pot)[:2]))

        reward = -STEP_COST

        # --- continuous, best-so-far approach (cannot be farmed by oscillating) ---
        if self._best_approach is None:
            self._best_approach = finger_r
        if finger_r < self._best_approach:
            reward += W_APPROACH * (self._best_approach - finger_r)
            self._best_approach = finger_r

        grasped = (finger_r < GRASP_RADIUS) and (thumb_r < GRASP_RADIUS)

        # --- transport progress, only credited while the piston is actually held ---
        if grasped and lift > LIFT_H:
            if self._best_transport is None:
                self._best_transport = d_pot_xy
            if d_pot_xy < self._best_transport:
                reward += W_TRANSPORT * (self._best_transport - d_pot_xy)
                self._best_transport = d_pot_xy

        # --- one-off stage bonuses ---
        def fire(name, cond):
            nonlocal reward
            if cond and not self._stages[name]:
                self._stages[name] = True
                reward += STAGE_BONUS[name]

        fire("reach", finger_r < GRASP_RADIUS)
        fire("grasp", grasped)
        fire("lift", grasped and lift > LIFT_H)
        fire("tube", self._stages["lift"] and d_tube < TUBE_NEAR)
        fire("plate", self._stages["lift"] and d_pot_xy < PLATE_NEAR)

        settled = self._prev_z is not None and abs(barrel[2] - self._prev_z) < 2e-3
        success = bool(
            self._stages["lift"]
            and d_pot_xy < PLATE_NEAR
            and barrel[2] > PLATE_Z
            and settled
        )
        fire("success", success)
        self._prev_z = float(barrel[2])

        info = {
            "stages": dict(self._stages),
            "grasped": bool(grasped),
            "finger_radial": finger_r,
            "thumb_radial": thumb_r,
            "lift": lift,
            "d_tube": d_tube,
            "d_pot_xy": d_pot_xy,
            "success": bool(self._stages["success"]),
        }
        return float(reward), info

    @property
    def succeeded(self) -> bool:
        """Explicit terminal success predicate."""
        return self._stages["success"]


def stage_order():
    """Stages in task order, for reporting progression."""
    return ["reach", "grasp", "lift", "tube", "plate", "success"]
