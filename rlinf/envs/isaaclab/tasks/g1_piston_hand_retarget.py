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

"""Inspire 6-DoF -> 12-DoF hand retargeting (embodiment adapter).

The demonstrations record 6 DoF per hand (4 finger proximals + thumb pitch + thumb
yaw). The simulator's Inspire hand exposes 12 independently driven joints per hand: the
6 recorded ones plus 4 finger intermediates, a thumb intermediate and a thumb distal.
Those 6 extra joints were previously commanded 0.0.

Measured consequence (``docs/contracts/g1_piston_grasp_diagnosis.json``): at the
demonstrated grasp the four finger links reach radial 0.018 m -- inside the 0.020 m
barrel radius -- while the thumb sits at 0.084 m, roughly 4x too far to oppose. The
fingers close *through* the barrel and nothing is grasped.

This module is the missing embodiment adapter: a fixed, interpretable rule mapping the
recorded 6 DoF onto 12 simulator targets. It is **not** a per-episode fit and it does
not move the robot, the objects, or the arm command.

Structure (5 scalars, all fixed after calibration):

    intermediate_j   = finger_coupling * proximal_j        (4 fingers)
    thumb_pitch      = max(recorded_pitch, thumb_pitch_floor) * grip_gate
    thumb_yaw        = max(recorded_yaw,   thumb_yaw_target) * grip_gate
    thumb_intermed   = thumb_intermediate * grip_gate
    thumb_distal     = thumb_distal_gain  * grip_gate

``grip_gate`` in [0, 1] ramps the thumb in with the recorded finger closure, so the
thumb only opposes when the demonstration is actually closing the hand. Without it the
thumb would sit folded across the palm during the reach and collide with the barrel on
approach.

STATUS: **INFERRED**, not recovered. The original ``inspire6_to_urdf12`` is absent from
this machine and from the dataset repository. These values were chosen on a calibration
split and then frozen and tested on held-out demonstrations; see
``docs/contracts/g1_piston_hand_retarget.json``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Recorded policy dims for each hand's 6 DoF (see the 30-D action contract).
LEFT_HAND_DIMS = (14, 15, 16, 17, 18, 19)  # pinky ring middle index, thumb pitch, yaw
RIGHT_HAND_DIMS = (20, 21, 22, 23, 24, 25)

#: Joint-limit ceilings from the USD (documented in the SFT handoff).
THUMB_PITCH_LIMIT = 0.6
THUMB_YAW_LIMIT = 1.3
THUMB_INTERMEDIATE_LIMIT = 0.8
THUMB_DISTAL_LIMIT = 1.2
FINGER_LIMIT = 1.7


@dataclass(frozen=True)
class InspireRetargetParams:
    """The five fixed scalars. Frozen after calibration; do not fit per episode."""

    #: Selected on the calibration split (episodes 0/10/20/33/45), then frozen.
    #: Chosen from the *centre* of the stable region rather than the single best
    #: score: yaw 0.6-1.2 x pitch 0.4-0.6 all give 0.29-0.32 m displacement with
    #: 59/69 grasp steps, so the result is not sensitive to the exact value.
    finger_coupling: float = 1.0
    thumb_pitch_floor: float = 0.6
    thumb_yaw_target: float = 0.9
    thumb_intermediate: float = 0.4
    thumb_distal_gain: float = 0.6
    #: Finger closure (rad) at which the thumb is fully engaged.
    grip_gate_full: float = 0.8
    #: Finger closure below which the thumb stays at its recorded pose.
    grip_gate_start: float = 0.2

    def as_dict(self):
        return {
            "finger_coupling": self.finger_coupling,
            "thumb_pitch_floor": self.thumb_pitch_floor,
            "thumb_yaw_target": self.thumb_yaw_target,
            "thumb_intermediate": self.thumb_intermediate,
            "thumb_distal_gain": self.thumb_distal_gain,
            "grip_gate_full": self.grip_gate_full,
            "grip_gate_start": self.grip_gate_start,
        }


#: Frozen parameters selected on the calibration split.
DEFAULT_PARAMS = InspireRetargetParams()


def _grip_gate(finger_mean: torch.Tensor, p: InspireRetargetParams) -> torch.Tensor:
    """Ramp in [0,1]: 0 while the hand is open, 1 once it is closing."""
    span = max(p.grip_gate_full - p.grip_gate_start, 1e-6)
    return ((finger_mean - p.grip_gate_start) / span).clamp(0.0, 1.0)


class InspireHandRetargeter:
    """Fixed 6->12 hand adapter applied to the 53-D simulator command.

    Consumes the already-mapped 53-D joint command plus the 30-D policy action, and
    overwrites only the 12 Inspire joints. The arm command is never touched.
    """

    def __init__(self, joint_names, params: InspireRetargetParams = DEFAULT_PARAMS):
        self.params = params
        self.joint_names = list(joint_names)
        idx = {n: i for i, n in enumerate(self.joint_names)}

        def group(side):
            return {
                "prox": [idx[f"{side}_{f}_proximal_joint"] for f in ("pinky", "ring", "middle", "index")],
                "inter": [idx[f"{side}_{f}_intermediate_joint"] for f in ("pinky", "ring", "middle", "index")],
                "t_pitch": idx[f"{side}_thumb_proximal_pitch_joint"],
                "t_yaw": idx[f"{side}_thumb_proximal_yaw_joint"],
                "t_inter": idx[f"{side}_thumb_intermediate_joint"],
                "t_distal": idx[f"{side}_thumb_distal_joint"],
            }

        self.groups = {"L": group("L"), "R": group("R")}
        self.hand_dims = {"L": LEFT_HAND_DIMS, "R": RIGHT_HAND_DIMS}

    def apply(self, command53: torch.Tensor, action30: torch.Tensor) -> torch.Tensor:
        """Return ``command53`` with the 12 Inspire joints per hand retargeted.

        Args:
            command53: ``[..., 53]`` joint command from the validated 30->53 mapper.
            action30: ``[..., 30]`` policy action the command was built from.
        """
        p = self.params
        out = command53.clone()
        for side in ("L", "R"):
            g = self.groups[side]
            dims = self.hand_dims[side]
            fingers = action30[..., list(dims[:4])]  # pinky ring middle index
            rec_pitch = action30[..., dims[4]]
            rec_yaw = action30[..., dims[5]]

            finger_mean = fingers.mean(dim=-1)
            gate = _grip_gate(finger_mean, p)

            # Finger intermediates follow their proximal joint.
            for k, ji in enumerate(g["inter"]):
                out[..., ji] = (fingers[..., k] * p.finger_coupling).clamp(0.0, FINGER_LIMIT)

            # Thumb: ramp toward an opposing pose as the hand closes, never below the
            # recorded command.
            pitch = torch.maximum(rec_pitch, gate * p.thumb_pitch_floor)
            yaw = torch.maximum(rec_yaw, gate * p.thumb_yaw_target)
            out[..., g["t_pitch"]] = pitch.clamp(-0.1, THUMB_PITCH_LIMIT)
            out[..., g["t_yaw"]] = yaw.clamp(-0.1, THUMB_YAW_LIMIT)
            out[..., g["t_inter"]] = (gate * p.thumb_intermediate).clamp(0.0, THUMB_INTERMEDIATE_LIMIT)
            out[..., g["t_distal"]] = (gate * p.thumb_distal_gain).clamp(0.0, THUMB_DISTAL_LIMIT)
        return out
