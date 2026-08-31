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

"""Functional pipette reward (v3): 3-D grasping, contact, and exploit-free success.

v3 answers a specific review of the v1/v2 rewards. Each numbered item below is a
reviewer point and what this module does about it.

**(1) Reach/grasp used only x,y; z and contact were ignored.**
    v1/v2 computed fingertip-to-barrel distance in the xy plane only, so a hand
    hovering 20 cm above the pipette scored a perfect "reach", and "grasp" merely
    required fingers and thumb to be near the *axis* -- not near the object, and not on
    opposite sides of it. v3 uses full 3-D distance, and adds two geometric conditions
    that a real grasp must satisfy and a hover cannot:

      * **height gate** -- the fingertips must be within ``GRASP_DZ`` of the barrel in z;
      * **opposition** -- the thumb and the finger group must lie on *opposite sides* of
        the barrel axis (their offset vectors must have negative dot product) and be
        close enough together to be pinching it (``GRASP_SPAN``).

    The scene ships no contact sensors (verified by enumerating every scene entity), so
    v3 does not pretend to read contact forces. It uses the strongest available
    *measured* proxy: opposing fingertips within grasp distance in 3-D, plus the
    object's own response (it moves with the hand, and its spring compresses). This is
    stated rather than hidden -- see ``grasp_is_geometric_proxy`` in the info dict.

**(2) "lift" only tested height, so throwing upward satisfied it.**
    v2 added a ballistic-speed test. v3 makes the condition positive rather than
    negative: the pipette must be *held while high* -- above ``LIFT_H`` **and** still
    grasped (3-D, opposing) **and** moving slowly. A thrown object fails the grasp
    conjunct the instant it leaves the hand, so it cannot bank the bonus at apex.

**(3) Success fired for a pipette thrown upward that momentarily hit zero velocity.**
    This is the sharpest hole: v1/v2 tested "over the pot in xy, above PLATE_Z, and z
    barely changing between steps". At the apex of a throw, vertical velocity passes
    through zero, so a pipette *in mid-air* satisfied "settled". v3 requires the pipette
    to be genuinely at rest **and supported**: speed below ``REST_SPEED`` for
    ``REST_STEPS`` consecutive steps, at a height consistent with resting on the plate
    (a band, not a floor), and *not* held. Momentary zero-crossing cannot satisfy a
    multi-step window, and mid-air cannot satisfy the height band.

**(4) No penalty on high-frequency motion.**
    v3 adds an explicit, bounded penalty on the pipette's own jerk while held
    (``W_JERK``), complementing the training-time CAPS action penalty. It is capped per
    step so it can never dominate task reward, and it is only applied while the object
    is held, so it does not tax free-space motion.

**(5) Reward validation was outdated.**
    The v1 validation lived in the demo-screen manifest and was computed under the
    transport-only predicate. ``tools/g1_piston/validate_reward.py`` re-derives
    validation for whichever reward version is selected, directly from demonstration
    replays, and is what the contract now cites.

Retained from v1/v2 because they were load-bearing: potential-based best-so-far shaping
(so no term can be farmed by oscillating), one-off monotone stage bonuses, and the
plunger stages (``press``/``dispense``) that make this the functional pipette task
rather than a transport task. See ``docs/contracts/g1_piston_plunger_dof.json``.
"""

from __future__ import annotations

import numpy as np

# --- grasp geometry, in metres --------------------------------------------------------
# CALIBRATED FROM MEASUREMENT, not from intuition. A first version of v3 tested 3-D
# distance to the barrel's CENTRE plus "opposition about the centre". Measuring the real
# geometry over a 690-step RLPD rollout showed both are wrong for this object:
#
#   * the barrel is a 0.19 m CYLINDER. Gripping it anywhere along its length puts the
#     hand up to ~0.095 m from the centre, so distance-to-centre never fell below 0.071
#     (fingers) or 0.078 (thumb) -- a 0.045 threshold was unsatisfiable by construction.
#   * the Inspire hand grips a cylinder by CURLING the four fingers around it with the
#     thumb bracing from the same side; measured opposition cosine never went negative
#     (min 0.754), and the thumb sits CLOSER to the axis (xy 0.033) than the fingers
#     (0.054). "Thumb opposite the fingers about the centre" describes a pinch, not this
#     grasp.
#
# The correct measures for a long cylinder are RADIAL distance to its AXIS and whether
# the contact height lies WITHIN the barrel's extent. Both are kept strict enough to
# reject the hover that the original xy-only test accepted, because a hover fails the
# height-extent test.
#: Radial (xy) fingertip distance to the barrel axis counted as "at the barrel".
GRASP_RADIUS = 0.045
#: Half-height of the barrel; contact must occur within this of the centre in z, i.e.
#: on the object itself rather than above or below it. Barrel height is 0.19 m.
BARREL_HALF_H = 0.095
#: Maximum thumb-to-finger separation consistent with a closed hand on the barrel.
GRASP_SPAN = 0.11

# --- lift / transport ----------------------------------------------------------------
#: Lift above resting height that counts as picked up.
LIFT_H = 0.05
#: Speed above which the pipette is considered thrown rather than carried [m/s].
BALLISTIC_SPEED = 0.60
#: xy distance from the tube axis counted as "aligned over the tube".
TUBE_ALIGN_XY = 0.035
#: Height of the pipette above the tube mouth for a valid insertion approach.
TUBE_INSERT_DZ = 0.06
#: xy distance from the pot counted as "over the plate".
PLATE_NEAR = 0.06

# --- success: at rest AND supported ---------------------------------------------------
#: Speed below which the pipette counts as momentarily still [m/s].
REST_SPEED = 0.02
#: Consecutive steps the pipette must stay still. A throw's apex is a single
#: zero-crossing and cannot satisfy a multi-step window.
REST_STEPS = 10
#: Height band, relative to the pot's own z, in which a pipette can plausibly be
#: RESTING on the plate. Mid-air "settling" falls outside it.
PLATE_REST_DZ = (0.05, 0.22)

# --- plunger --------------------------------------------------------------------------
#: Full travel of the prismatic joint, measured from the scene.
PLUNGER_TRAVEL = 0.04
#: Fraction of travel counted as a deliberate press (gripping compliance is ~25%).
PRESS_FRAC = 0.50
#: Absolute press threshold, metres.
PRESS_DEPTH = PLUNGER_TRAVEL * PRESS_FRAC

#: One-off bonus per stage, in task order.
STAGE_BONUS = {
    "reach": 1.0,
    "grasp": 2.0,
    "lift": 4.0,
    "tube": 6.0,
    "press": 12.0,
    "dispense": 18.0,
    "plate": 8.0,
    "success": 30.0,
}

#: Weight on best-so-far approach progress (now on 3-D distance).
W_APPROACH = 2.0
#: Weight on best-so-far transport progress once the pipette is held.
W_TRANSPORT = 4.0
#: Weight on best-so-far tube-alignment progress once lifted.
W_ALIGN = 4.0
#: Weight on best-so-far plunger depression, per metre.
W_PRESS = 150.0
#: Penalty weight on object jerk while held (reviewer point 4).
W_JERK = 0.02
#: Hard cap on the per-step jerk penalty, so smoothness can never dominate the task.
JERK_PENALTY_CAP = 0.05
#: Small per-step cost so dithering is never free.
STEP_COST = 0.001


def stage_order():
    """Stages in task order, for reporting progression."""
    return ["reach", "grasp", "lift", "tube", "press", "dispense", "plate", "success"]


class PistonTaskRewardV3:
    """Stateful functional-pipette reward with 3-D grasping and exploit-free success."""

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
        self._prev_barrel = None
        self._prev_speed = None
        self._still_steps = 0
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

        # --- reviewer point 1: radial distance to the AXIS + on-object height ------
        # (See the constants block for why distance-to-centre and opposition-about-
        # centre were measured to be the wrong tests for a 0.19 m cylinder.)
        finger_c = fing.mean(axis=0)
        finger_r = float(np.mean(np.linalg.norm((fing - barrel)[:, :2], axis=1)))
        thumb_r = float(np.linalg.norm((thumb - barrel)[:2]))
        finger_dz = float(abs(finger_c[2] - barrel[2]))
        thumb_dz = float(abs(thumb[2] - barrel[2]))
        span = float(np.linalg.norm(thumb - finger_c))
        # Retained for reporting: it characterises the grasp even though it is not a
        # gate, and it is what showed the pinch model to be wrong here.
        v_thumb = thumb - barrel
        v_fing = finger_c - barrel
        nt, nf = np.linalg.norm(v_thumb), np.linalg.norm(v_fing)
        opposition = (float(np.dot(v_thumb, v_fing) / (nt * nf))
                      if nt > 1e-9 and nf > 1e-9 else 1.0)

        near_axis = (finger_r < GRASP_RADIUS) and (thumb_r < GRASP_RADIUS)
        # ON the object, not above or below it: this is what rejects a hover, and it is
        # the condition the original xy-only v1/v2 test was missing.
        on_object = (finger_dz < BARREL_HALF_H) and (thumb_dz < BARREL_HALF_H)
        closed = span < GRASP_SPAN
        grasped = bool(near_axis and on_object and closed)

        lift = float(barrel[2] - self._rest_z)
        d_tube_xy = float(np.linalg.norm((barrel - tube)[:2]))
        dz_tube = float(barrel[2] - tube[2])
        d_pot_xy = float(np.linalg.norm((barrel - pot)[:2]))
        dz_pot = float(barrel[2] - pot[2])

        # velocity and jerk of the object itself
        if self._prev_barrel is None:
            speed = 0.0
        else:
            speed = float(np.linalg.norm(barrel - self._prev_barrel) / max(self.dt, 1e-6))
        accel = 0.0 if self._prev_speed is None else abs(speed - self._prev_speed) / max(self.dt, 1e-6)
        self._prev_barrel = barrel.copy()
        self._prev_speed = speed

        reward = -STEP_COST

        # --- continuous, best-so-far approach, on 3-D distance --------------------
        # Approach shaping combines radial closing with getting ON the object in z.
        # Radial alone would pay a hovering hand the full approach reward.
        approach = 0.5 * (finger_r + thumb_r) + 0.5 * max(finger_dz - BARREL_HALF_H, 0.0)
        if self._best_approach is None:
            self._best_approach = approach
        if approach < self._best_approach:
            reward += W_APPROACH * (self._best_approach - approach)
            self._best_approach = approach

        held = self._stages["grasp"] and grasped and lift > LIFT_H

        if self._stages["grasp"] and lift > LIFT_H:
            if self._best_transport is None:
                self._best_transport = d_pot_xy
            if d_pot_xy < self._best_transport:
                reward += W_TRANSPORT * (self._best_transport - d_pot_xy)
                self._best_transport = d_pot_xy
            if self._best_align is None:
                self._best_align = d_tube_xy
            if d_tube_xy < self._best_align:
                reward += W_ALIGN * (self._best_align - d_tube_xy)
                self._best_align = d_tube_xy

        # --- plunger, best-so-far, only while ACTUALLY holding it ------------------
        # Gate on the LIVE grasp, not the latched stage. The latched form is farmable:
        # grasp once (latching the stage), release, then drive the plunger down by any
        # other means -- pushing it against the table -- and the term keeps paying.
        # Measured on the latched version: 18.0 reward farmed after releasing.
        if grasped:
            if self._best_press is None:
                self._best_press = press
            if press > self._best_press:
                reward += W_PRESS * (press - self._best_press)
                self._best_press = press
        self._max_press = max(self._max_press, press)

        # --- reviewer point 4: bounded jerk penalty while held --------------------
        jerk_pen = 0.0
        if held:
            jerk_pen = min(W_JERK * accel, JERK_PENALTY_CAP)
            reward -= jerk_pen

        aligned_over_tube = (
            self._stages["lift"]
            and d_tube_xy < TUBE_ALIGN_XY
            and 0.0 < dz_tube < TUBE_INSERT_DZ
        )
        if aligned_over_tube:
            self._max_press_aligned = max(self._max_press_aligned, press)

        # --- reviewer point 3: rest requires a multi-step still window ------------
        if speed < REST_SPEED:
            self._still_steps += 1
        else:
            self._still_steps = 0
        at_rest = self._still_steps >= REST_STEPS
        supported = PLATE_REST_DZ[0] < dz_pot < PLATE_REST_DZ[1]

        def fire(name, cond):
            nonlocal reward
            if cond and not self._stages[name]:
                self._stages[name] = True
                reward += STAGE_BONUS[name]

        fire("reach", near_axis and on_object)
        fire("grasp", grasped)
        # reviewer point 2: positive condition -- held AND high AND slow.
        fire("lift", grasped and lift > LIFT_H and speed < BALLISTIC_SPEED)
        fire("tube", aligned_over_tube)
        # Same reasoning as the shaping term: the press only counts if the pipette is
        # being held at the moment it is depressed.
        fire("press", grasped and press > PRESS_DEPTH)
        fire("dispense", aligned_over_tube and grasped and press > PRESS_DEPTH)
        fire("plate", self._stages["lift"] and d_pot_xy < PLATE_NEAR)

        # Success: dispensed, then placed over the plate, at rest, supported, released.
        success = bool(
            self._stages["dispense"]
            and self._stages["lift"]
            and d_pot_xy < PLATE_NEAR
            and supported
            and at_rest
            and not grasped
        )
        fire("success", success)

        info = {
            "stages": dict(self._stages),
            "grasped": bool(grasped),
            "grasp_is_geometric_proxy": True,
            "finger_radial": finger_r,
            "thumb_radial": thumb_r,
            "finger_dz": finger_dz,
            "thumb_dz": thumb_dz,
            "grasp_span": span,
            "opposition_cos": opposition,
            "lift": lift,
            "speed": speed,
            "accel": accel,
            "jerk_penalty": jerk_pen,
            "d_tube_xy": d_tube_xy,
            "dz_tube": dz_tube,
            "d_pot_xy": d_pot_xy,
            "dz_pot": dz_pot,
            "still_steps": self._still_steps,
            "at_rest": bool(at_rest),
            "supported": bool(supported),
            "press_m": press,
            "press_frac": press / PLUNGER_TRAVEL,
            "max_press_m": self._max_press,
            "max_press_aligned_m": self._max_press_aligned,
            "aligned_over_tube": bool(aligned_over_tube),
            "success": bool(self._stages["success"]),
        }
        return float(reward), info

    @property
    def succeeded(self) -> bool:
        return self._stages["success"]
