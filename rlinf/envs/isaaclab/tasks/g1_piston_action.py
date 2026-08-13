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

"""StarVLA 30-D action -> IsaacLab 53-D joint command, by joint name.

The SFT policy emits 30 absolute joint-position targets; the piston task's
``JointPositionActionCfg(joint_names=[".*"], ...)`` expects 53. The conversion is
defined here in terms of **named joints**, never bare scatter indices, so a
mis-ordered articulation fails loudly instead of driving the wrong limb.

Contract (from RLINF_INTERFACE.json, cross-checked against the live articulation in
``docs/contracts/g1_piston_joint_contract.json``)::

    left_arm    0:7    right_arm   7:14
    left_hand  14:20   right_hand 20:26
    base_height  26    navigate   27:30

``use_default_offset=True`` and every default joint position is 0.0, so a commanded
value equals the absolute target in radians; no offset arithmetic is needed.

Three groups are not driven by the policy:

* **Dropped (dims 26-29)** -- base height and the 3 navigation velocities have no
  articulation DOF in this fixed-base task, and are constant in the dataset.
* **Legs and waist (15 joints)** -- held at the default pose. This is the fixed-base
  piston task; the demonstrations never moved them.
* **Inspire intermediate/distal (12 joints)** -- held at 0.0. See
  ``INSPIRE_FIDELITY_CAVEAT``: 0.0 is a *valid* command, but whether the original
  demonstrations were recorded with a 6->12 coupling expansion is **unproven**. This
  is a deployment/contact-fidelity assumption, not reproduced demonstrator behaviour.
"""

from __future__ import annotations

import torch

#: Policy action dimensionality.
STARVLA_ACTION_DIM = 30

#: Simulator articulation dimensionality.
ISAACLAB_ACTION_DIM = 53

#: Policy dims that are deliberately dropped, with the reason.
DROPPED_POLICY_DIMS = {
    26: "base_height - no articulation DOF (fixed-base task); constant 0.76 in data",
    27: "nav_vx - no DOF; exactly 0.0 in all dataset frames",
    28: "nav_vy - no DOF; exactly 0.0 in all dataset frames",
    29: "nav_yaw - no DOF; exactly 0.0 in all dataset frames",
}

#: policy dim -> simulator joint name. Authoritative, name-based form of the handoff's
#: ``ds_dim_to_articulation_index``. Verified against the live articulation.
POLICY_DIM_TO_JOINT_NAME = {
    # left_arm 0:7
    0: "left_shoulder_pitch_joint",
    1: "left_shoulder_roll_joint",
    2: "left_shoulder_yaw_joint",
    3: "left_elbow_joint",
    4: "left_wrist_roll_joint",
    5: "left_wrist_pitch_joint",
    6: "left_wrist_yaw_joint",
    # right_arm 7:14
    7: "right_shoulder_pitch_joint",
    8: "right_shoulder_roll_joint",
    9: "right_shoulder_yaw_joint",
    10: "right_elbow_joint",
    11: "right_wrist_roll_joint",
    12: "right_wrist_pitch_joint",
    13: "right_wrist_yaw_joint",
    # left_hand 14:20 (Inspire 6-DoF proximal)
    14: "L_pinky_proximal_joint",
    15: "L_ring_proximal_joint",
    16: "L_middle_proximal_joint",
    17: "L_index_proximal_joint",
    18: "L_thumb_proximal_pitch_joint",
    19: "L_thumb_proximal_yaw_joint",
    # right_hand 20:26 (Inspire 6-DoF proximal)
    20: "R_pinky_proximal_joint",
    21: "R_ring_proximal_joint",
    22: "R_middle_proximal_joint",
    23: "R_index_proximal_joint",
    24: "R_thumb_proximal_pitch_joint",
    25: "R_thumb_proximal_yaw_joint",
}

#: Joints held at the default pose because the policy does not drive them.
HELD_LEG_AND_WAIST_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

#: Inspire joints the 6-DoF dataset never commanded. Held at 0.0 -- see the caveat.
HELD_INSPIRE_INTERMEDIATE_DISTAL_JOINTS = (
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
)

INSPIRE_FIDELITY_CAVEAT = (
    "The 12 Inspire intermediate/distal joints are commanded 0.0. The USD proves they "
    "are independently driven (no mimic/tendon coupling), so 0.0 is a VALID command "
    "that reproduces the default pose. It is NOT proven that the original replay used "
    "0.0: a 6->12 coupling expansion (Dex3InspireMapper.inspire6_to_urdf12) exists in "
    "this ecosystem but its implementation is unavailable on this machine. If the "
    "demonstrations used coupling, the deployed hand curls less at the middle "
    "phalanges, changing fingertip pose, grasp stability, and contact timing during "
    "insertion. Treat grasp/slip failures as possibly caused by this, not by the "
    "arm trajectory. The 30-D contract is unaffected either way."
)


class G1PistonActionMapper:
    """Maps StarVLA 30-D actions onto the 53-D articulation command by joint name.

    Built once against a concrete ``joint_names`` ordering (from
    ``env.scene["robot"].data.joint_names``), so the index arithmetic is derived from
    the live simulator rather than hard-coded.
    """

    def __init__(self, joint_names, *, strict: bool = True):
        self.joint_names = list(joint_names)
        self.num_joints = len(self.joint_names)

        if strict and self.num_joints != ISAACLAB_ACTION_DIM:
            raise ValueError(
                f"Expected {ISAACLAB_ACTION_DIM} articulation joints, got "
                f"{self.num_joints}. The action contract was derived for the "
                "G1 29-DOF + Inspire piston articulation."
            )

        name_to_index = {name: i for i, name in enumerate(self.joint_names)}

        missing = [
            n for n in POLICY_DIM_TO_JOINT_NAME.values() if n not in name_to_index
        ]
        if missing:
            raise ValueError(
                f"Articulation is missing joints required by the policy: {missing}"
            )

        # policy dim -> articulation index, resolved through names.
        self.policy_dim_to_joint_index = {
            dim: name_to_index[name]
            for dim, name in POLICY_DIM_TO_JOINT_NAME.items()
        }

        # Dense gather/scatter pair, ordered by policy dim for a single index_copy.
        driven_dims = sorted(self.policy_dim_to_joint_index)
        self._src_dims = torch.tensor(driven_dims, dtype=torch.long)
        self._dst_idx = torch.tensor(
            [self.policy_dim_to_joint_index[d] for d in driven_dims],
            dtype=torch.long,
        )

        self.held_joint_indices = sorted(
            name_to_index[n]
            for n in HELD_LEG_AND_WAIST_JOINTS + HELD_INSPIRE_INTERMEDIATE_DISTAL_JOINTS
            if n in name_to_index
        )

    def map(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert ``[..., 30]`` policy actions to ``[..., 53]`` joint commands.

        Undriven joints are 0.0, which under ``use_default_offset=True`` with an
        all-zero default pose commands exactly the default pose.
        """
        if actions.shape[-1] != STARVLA_ACTION_DIM:
            raise ValueError(
                f"Expected trailing dim {STARVLA_ACTION_DIM}, got {actions.shape[-1]}"
            )

        out = torch.zeros(
            *actions.shape[:-1],
            self.num_joints,
            dtype=actions.dtype,
            device=actions.device,
        )
        src = self._src_dims.to(actions.device)
        dst = self._dst_idx.to(actions.device)
        out.index_copy_(-1, dst, actions.index_select(-1, src))
        return out

    def describe(self) -> dict:
        """Human-readable mapping, for logging into a run's provenance record."""
        return {
            "policy_dim_to_joint": {
                d: self.joint_names[i]
                for d, i in sorted(self.policy_dim_to_joint_index.items())
            },
            "dropped_policy_dims": DROPPED_POLICY_DIMS,
            "held_at_default_joints": [
                self.joint_names[i] for i in self.held_joint_indices
            ],
            "inspire_fidelity_caveat": INSPIRE_FIDELITY_CAVEAT,
        }
