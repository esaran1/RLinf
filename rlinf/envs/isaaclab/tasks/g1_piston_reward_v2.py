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

"""Functional pipette reward (v2): scores the act the task is actually named for.

Why a v2 at all
---------------
The v1 reward (:mod:`g1_piston_reward`) scores the task purely from rigid-body
*positions*: reach, grasp, lift, tube-proximity, plate-proximity, settle. It never
reads the object's articulated degree of freedom. But the scene's ``object`` is not a
passive cylinder -- it is an ``Articulation`` with a real prismatic ``PistonJoint``:

    Rod      the PLUNGER, and the articulation ROOT (the part the robot grasps)
    Barrel   the pipette body; the rod slides through its bore
    joint    prismatic, travel [0.0, 0.04] m, passive spring (150 N/m) + damper
             (12 N-s/m) with rest target 0.0 -- it springs back when released

The visual asset is ``Micropipette01.obj``. Depressing the plunger is therefore the
functional act the task is named for, and under v1 it is neither measured nor rewarded.
A v1 "success" means only: the pipette was carried over the pot and allowed to settle.
This was found by direct scene inspection (see
``docs/contracts/g1_piston_plunger_dof.json``); every v1 number in the study remains
valid *under the v1 predicate*, which is why v1 is frozen rather than edited.

What v2 adds
------------
Two measured quantities v1 ignores, both read straight off the simulator:

* ``press``  -- ``joint_pos`` of ``PistonJoint``, i.e. how far the plunger is depressed.
* ``insert`` -- barrel-to-tube distance, tightened to an actual insertion tolerance and
  gated on relative *height*, so "near the tube" cannot be satisfied by hovering past it.

Design constraints carried over from v1 (they were load-bearing):

* Shaping is **potential-based on best-so-far progress**: continuous terms pay only for
  improving on the best value achieved so far, so an agent cannot farm reward by
  oscillating. This matters more in v2, not less: a spring-loaded joint that returns to
  rest is exactly the kind of thing a naive per-step reward would let a policy pump.
* Stage bonuses are one-off and monotone in task order, so reaching a later stage always
  outranks stalling at an earlier one.
* Every term is a direct simulator reading; no learned or visual reward model.

The v1 lift exploit is also closed here. Under v1, ``lift`` paid for height alone, so
flinging the pipette banked reward without transport (one whole evaluation run consisted
of throws). v2 pays the lift bonus only when the piston is *under control*: lifted, and
not moving ballistically.
"""

from __future__ import annotations

import numpy as np

# --- geometry, in metres -------------------------------------------------------------
#: Fingertip radial distance counted as "at the barrel" (unchanged from v1).
GRASP_RADIUS = 0.045
#: Lift above resting height that counts as picked up (unchanged from v1).
LIFT_H = 0.05
#: Speed above which a lifted piston is considered thrown, not carried [m/s].
BALLISTIC_SPEED = 0.60
#: xy distance from the tube axis counted as "aligned over the tube".
TUBE_ALIGN_XY = 0.035
#: Height of the pipette tip above the tube mouth for a valid insertion approach.
TUBE_INSERT_DZ = 0.06
#: xy distance from the pot counted as "over the plate" (unchanged from v1).
PLATE_NEAR = 0.06
#: Height above the table the piston must clear when over the plate (unchanged).
PLATE_Z = 0.95

# --- plunger -------------------------------------------------------------------------
#: Full travel of the prismatic joint, measured from the scene (metres).
PLUNGER_TRAVEL = 0.04
#: Fraction of travel that counts as a deliberate press. Passive compliance from merely
#: gripping the pipette registers well below this (measured on the demonstrations, see
#: the plunger-dof contract); a press must clear it to be credited.
PRESS_FRAC = 0.50
#: Absolute press threshold in metres.
PRESS_DEPTH = PLUNGER_TRAVEL * PRESS_FRAC
#: A press only counts as *dispensing* if it happens while aligned over the tube.
#: Pressing the plunger in mid-air is a distinct (uncredited) behaviour.

#: One-off bonus per stage, in task order. Later stages are worth strictly more.
STAGE_BONUS = {
    "reach": 1.0,
    "grasp": 2.0,
    "lift": 4.0,
    "tube": 6.0,
    "press": 12.0,      # the functional act: plunger depressed
    "dispense": 18.0,   # plunger depressed WHILE aligned over the tube
    "plate": 8.0,
    "success": 30.0,
}

#: Weight on best-so-far approach progress (continuous shaping).
W_APPROACH = 2.0
#: Weight on best-so-far transport progress once the piston is held.
W_TRANSPORT = 4.0
#: Weight on best-so-far tube-alignment progress once the piston is lifted.
W_ALIGN = 4.0
#: Weight on best-so-far plunger depression, per metre of travel. Scaled so full travel
#: is worth ~6.0, comparable to a mid-tier stage bonus, and only paid while the pipette
#: is actually held (otherwise "press the plunger on the table" would be free reward).
W_PRESS = 150.0
#: Small per-step cost so dithering is never free.
STEP_COST = 0.001


def stage_order():
    """Stages in task order, for reporting progression."""
    return ["reach", "grasp", "lift", "tube", "press", "dispense", "plate", "success"]


class PistonTaskRewardV2:
    """Stateful functional-pipette reward.

    Call :meth:`reset` at episode start and :meth:`step` once per env step. The
    ``info`` dict it returns is a superset of v1's, so v1 consumers keep working and
    v1 metrics can be recomputed from a v2 rollout for direct comparison.
    """

    def __init__(self, scene, joint_names=None, device="cpu", dt=0.02):
        self.scene = scene
        self.device = device
        self.dt = float(dt)
        body_names = list(scene["robot"].data.body_names)
        self.finger_idx = [
            body_names.index(f"R_{f}_intermediate")
            for f in ("index", "middle", "pinky", "ring")
        ]
        self.thumb_idx = body_names.index("R_thumb_distal")
        # The plunger DoF. Absent only if the scene is not the articulated piston.
        obj = scene["object"]
        jn = list(getattr(obj.data, "joint_names", []) or [])
        self.piston_joint_idx = jn.index("PistonJoint") if "PistonJoint" in jn else None
        self.reset()

    def reset(self):
        self._stages = {k: False for k in STAGE_BONUS}
        self._rest_z = None
        self._best_approach = None
        self._best_transport = None
        self._best_align = None
        self._best_press = None
        self._prev_z = None
        self._prev_barrel = None
        self._max_press = 0.0
        self._max_press_aligned = 0.0

    def _read(self):
        sc = self.scene
        obj = sc["object"]
        barrel = obj.data.body_pos_w[0, 1].detach().float().cpu().numpy()
        rod = obj.data.body_pos_w[0, 0].detach().float().cpu().numpy()
        tube = sc["tube"].data.root_pos_w[0].detach().float().cpu().numpy()
        pot = sc["pot"].data.root_pos_w[0].detach().float().cpu().numpy()
        rb = sc["robot"].data.body_pos_w[0]
        fing = rb[self.finger_idx].detach().float().cpu().numpy()
        thumb = rb[self.thumb_idx].detach().float().cpu().numpy()
        if self.piston_joint_idx is not None:
            press = float(
                obj.data.joint_pos[0, self.piston_joint_idx].detach().float().cpu()
            )
        else:
            press = 0.0
        return barrel, rod, tube, pot, fing, thumb, press

    def step(self):
        """Return ``(reward, info)`` for the current simulator state."""
        barrel, rod, tube, pot, fing, thumb, press = self._read()
        if self._rest_z is None:
            self._rest_z = float(barrel[2])

        finger_r = float(np.mean(np.linalg.norm((fing - barrel)[:, :2], axis=1)))
        thumb_r = float(np.linalg.norm((thumb - barrel)[:2]))
        lift = float(barrel[2] - self._rest_z)
        d_tube_xy = float(np.linalg.norm((barrel - tube)[:2]))
        dz_tube = float(barrel[2] - tube[2])
        d_pot_xy = float(np.linalg.norm((barrel - pot)[:2]))

        # Speed, for the ballistic test that closes the v1 lift exploit.
        if self._prev_barrel is None:
            speed = 0.0
        else:
            speed = float(np.linalg.norm(barrel - self._prev_barrel) / max(self.dt, 1e-6))
        self._prev_barrel = barrel.copy()

        reward = -STEP_COST

        # --- continuous, best-so-far approach ------------------------------------
        if self._best_approach is None:
            self._best_approach = finger_r
        if finger_r < self._best_approach:
            reward += W_APPROACH * (self._best_approach - finger_r)
            self._best_approach = finger_r

        grasped = (finger_r < GRASP_RADIUS) and (thumb_r < GRASP_RADIUS)
        held = self._stages["grasp"] and lift > LIFT_H

        # --- transport progress, credited once the piston is picked up -----------
        if held:
            if self._best_transport is None:
                self._best_transport = d_pot_xy
            if d_pot_xy < self._best_transport:
                reward += W_TRANSPORT * (self._best_transport - d_pot_xy)
                self._best_transport = d_pot_xy

            # --- tube alignment progress -----------------------------------------
            if self._best_align is None:
                self._best_align = d_tube_xy
            if d_tube_xy < self._best_align:
                reward += W_ALIGN * (self._best_align - d_tube_xy)
                self._best_align = d_tube_xy

        # --- plunger depression, best-so-far, only while genuinely held ----------
        # Paid on improvement only: the joint is spring-loaded and returns to rest, so
        # a per-step term would let a policy pump it for unbounded reward.
        if self._stages["grasp"]:
            if self._best_press is None:
                self._best_press = press
            if press > self._best_press:
                reward += W_PRESS * (press - self._best_press)
                self._best_press = press
        self._max_press = max(self._max_press, press)

        aligned_over_tube = (
            self._stages["lift"]
            and d_tube_xy < TUBE_ALIGN_XY
            and 0.0 < dz_tube < TUBE_INSERT_DZ
        )
        if aligned_over_tube:
            self._max_press_aligned = max(self._max_press_aligned, press)

        # --- one-off stage bonuses ------------------------------------------------
        def fire(name, cond):
            nonlocal reward
            if cond and not self._stages[name]:
                self._stages[name] = True
                reward += STAGE_BONUS[name]

        fire("reach", finger_r < GRASP_RADIUS)
        fire("grasp", grasped)
        # v1 rationale retained: gate on the grasp having *happened* plus height, since
        # the contact predicate flickers during a carry. v2 adds the ballistic test so
        # a throw cannot bank the lift bonus (the v1 exploit).
        fire("lift", self._stages["grasp"] and lift > LIFT_H and speed < BALLISTIC_SPEED)
        fire("tube", aligned_over_tube)
        fire("press", self._stages["grasp"] and press > PRESS_DEPTH)
        fire("dispense", aligned_over_tube and press > PRESS_DEPTH)
        fire("plate", self._stages["lift"] and d_pot_xy < PLATE_NEAR)

        settled = self._prev_z is not None and abs(barrel[2] - self._prev_z) < 2e-3
        # v2 success is the FULL task: the pipette was used (plunger dispensed over the
        # tube) and then placed over the plate. v1 success is reported alongside, so a
        # single rollout can be scored under both predicates.
        success_v2 = bool(
            self._stages["dispense"]
            and self._stages["lift"]
            and d_pot_xy < PLATE_NEAR
            and barrel[2] > PLATE_Z
            and settled
        )
        fire("success", success_v2)
        success_v1 = bool(
            self._stages["lift"]
            and d_pot_xy < PLATE_NEAR
            and barrel[2] > PLATE_Z
            and settled
        )
        self._prev_z = float(barrel[2])

        info = {
            "stages": dict(self._stages),
            "grasped": bool(grasped),
            "finger_radial": finger_r,
            "thumb_radial": thumb_r,
            "lift": lift,
            "speed": speed,
            "d_tube_xy": d_tube_xy,
            "dz_tube": dz_tube,
            "d_pot_xy": d_pot_xy,
            "press_m": press,
            "press_frac": press / PLUNGER_TRAVEL,
            "max_press_m": self._max_press,
            "max_press_aligned_m": self._max_press_aligned,
            "aligned_over_tube": bool(aligned_over_tube),
            "success": bool(self._stages["success"]),
            "success_v1_predicate": success_v1,
        }
        return float(reward), info

    @property
    def succeeded(self) -> bool:
        """Explicit terminal success predicate (v2: full functional task)."""
        return self._stages["success"]
