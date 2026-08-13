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

"""StarVLA <-> IsaacLab contract tests: normalization, 30->53 mapping, camera, hold.

CPU-only; no Isaac Sim and no checkpoint required. The pieces these cover are the
ones where a silent error would be invisible in a rollout and would look like poor
policy quality instead of a wiring bug.
"""

import json

import pytest
import torch
import yaml

from rlinf.envs.isaaclab.tasks.g1_piston import (
    EGO_HFOV_DEG,
    EGO_NATIVE_HW,
    EGO_VFOV_DEG,
    compute_fov_deg,
)
from rlinf.envs.isaaclab.tasks.g1_piston_action import (
    DROPPED_POLICY_DIMS,
    HELD_INSPIRE_INTERMEDIATE_DISTAL_JOINTS,
    HELD_LEG_AND_WAIST_JOINTS,
    ISAACLAB_ACTION_DIM,
    POLICY_DIM_TO_JOINT_NAME,
    STARVLA_ACTION_DIM,
    G1PistonActionMapper,
)
from rlinf.envs.isaaclab.tasks.g1_piston_norm import Q99ActionNormalizer

JOINT_CONTRACT = "docs/contracts/g1_piston_joint_contract.json"
ENV_YAML = "examples/embodiment/config/env/isaaclab_g1_piston.yaml"

DATASET_STATS = (
    "/home/jren313/research/starvla_rl/checkpoints/"
    "g1-longhorizon-oft-v1/dataset_statistics.json"
)
HANDOFF = (
    "/home/jren313/research/starvla_rl/starVLA/examples/realRobots/"
    "G1InspirePistonLongHorizon/RLINF_INTERFACE.json"
)

TASK_STRING = (
    "pick up the piston with the right hand, inject it into the tube held by "
    "the left hand, then move it over the hole plate."
)
TASK_STRING_SHA256 = (
    "23c5ab23b8e8f7a903580bb14b32706c438100989ed45293e2415d87238e8062"
)


@pytest.fixture(scope="module")
def joint_names():
    with open(JOINT_CONTRACT) as f:
        return json.load(f)["joint_names_in_articulation_order"]


@pytest.fixture(scope="module")
def mapper(joint_names):
    return G1PistonActionMapper(joint_names)


# --------------------------------------------------------------------------
# Task instruction
# --------------------------------------------------------------------------


def test_task_string_matches_recorded_sha256():
    """The instruction must reach the policy byte-for-byte; any rewording is OOD."""
    import hashlib

    assert hashlib.sha256(TASK_STRING.encode()).hexdigest() == TASK_STRING_SHA256
    assert len(TASK_STRING) == 120


def test_env_yaml_task_description_is_verbatim():
    with open(ENV_YAML) as f:
        cfg = yaml.safe_load(f)
    assert cfg["init_params"]["task_description"] == TASK_STRING


# --------------------------------------------------------------------------
# Phase 4: 30 -> 53 action mapping
# --------------------------------------------------------------------------


def test_mapper_output_dimension(mapper):
    out = mapper.map(torch.zeros(1, STARVLA_ACTION_DIM))
    assert out.shape == (1, ISAACLAB_ACTION_DIM)


def test_every_policy_dim_lands_on_its_named_joint(mapper, joint_names):
    """The core anti-miswiring test: one-hot each policy dim, find where it landed.

    Proves no policy dimension maps to the wrong simulator joint -- the failure mode
    that would silently drive the wrong limb and look like bad policy quality.
    """
    for dim, expected_joint in POLICY_DIM_TO_JOINT_NAME.items():
        action = torch.zeros(1, STARVLA_ACTION_DIM)
        action[0, dim] = 1.234
        out = mapper.map(action)[0]

        nonzero = (out != 0).nonzero(as_tuple=True)[0].tolist()
        assert len(nonzero) == 1, (
            f"policy dim {dim} touched {len(nonzero)} joints, expected exactly 1"
        )
        landed = joint_names[nonzero[0]]
        assert landed == expected_joint, (
            f"policy dim {dim} drove {landed!r}, expected {expected_joint!r}"
        )
        assert out[nonzero[0]] == pytest.approx(1.234)


def test_all_53_dims_accounted_for(mapper, joint_names):
    """Driven + held joints partition the full articulation with no gaps/overlap."""
    driven = set(mapper.policy_dim_to_joint_index.values())
    held = set(mapper.held_joint_indices)

    assert len(driven) == 26, f"expected 26 driven joints, got {len(driven)}"
    assert len(held) == 27, f"expected 27 held joints, got {len(held)}"
    assert not (driven & held), "a joint is both driven and held"
    assert driven | held == set(range(ISAACLAB_ACTION_DIM)), "not all 53 dims covered"


def test_dropped_dims_do_not_reach_the_simulator(mapper):
    """base_height and the 3 nav velocities have no DOF and must be discarded."""
    action = torch.zeros(1, STARVLA_ACTION_DIM)
    for dim in DROPPED_POLICY_DIMS:
        action[0, dim] = 99.0
    out = mapper.map(action)
    assert torch.all(out == 0), "a dropped policy dim leaked into the joint command"


def test_left_and_right_never_cross(mapper, joint_names):
    """A left-arm/hand dim must never drive a right-side joint, and vice versa."""
    for dim, name in POLICY_DIM_TO_JOINT_NAME.items():
        idx = mapper.policy_dim_to_joint_index[dim]
        landed = joint_names[idx]
        if dim < 7 or 14 <= dim < 20:  # left arm / left hand
            assert landed.startswith("left_") or landed.startswith("L_"), (
                f"left policy dim {dim} drove {landed!r}"
            )
        elif 7 <= dim < 14 or 20 <= dim < 26:  # right arm / right hand
            assert landed.startswith("right_") or landed.startswith("R_"), (
                f"right policy dim {dim} drove {landed!r}"
            )


def test_held_joints_are_legs_waist_and_inspire_secondaries(mapper, joint_names):
    held = {joint_names[i] for i in mapper.held_joint_indices}
    assert held == set(HELD_LEG_AND_WAIST_JOINTS) | set(
        HELD_INSPIRE_INTERMEDIATE_DISTAL_JOINTS
    )
    assert len(HELD_LEG_AND_WAIST_JOINTS) == 15
    assert len(HELD_INSPIRE_INTERMEDIATE_DISTAL_JOINTS) == 12


def test_mapper_matches_handoff_index_table(mapper):
    """Name-based mapping agrees with the handoff's numeric ds_dim -> art_idx table."""
    with open(HANDOFF) as f:
        table = json.load(f)["isaaclab_action_mapping"]["ds_dim_to_articulation_index"]
    for dim_str, art_idx in table.items():
        assert mapper.policy_dim_to_joint_index[int(dim_str)] == art_idx


def test_mapper_rejects_wrong_action_dim(mapper):
    with pytest.raises(ValueError, match="Expected trailing dim 30"):
        mapper.map(torch.zeros(1, 41))


def test_mapper_rejects_wrong_articulation(joint_names):
    with pytest.raises(ValueError, match="Expected 53 articulation joints"):
        G1PistonActionMapper(joint_names[:40])


def test_mapper_preserves_batch_and_chunk_dims(mapper):
    out = mapper.map(torch.zeros(4, 30, STARVLA_ACTION_DIM))
    assert out.shape == (4, 30, ISAACLAB_ACTION_DIM)


# --------------------------------------------------------------------------
# Phase 3: normalization
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def normalizer():
    return Q99ActionNormalizer.from_dataset_statistics(DATASET_STATS)


def test_degenerate_dims_match_handoff(normalizer):
    assert normalizer.degenerate_dims == [14, 15, 16, 17, 18, 19, 26, 27, 28, 29]


def test_normalization_round_trip(normalizer):
    """physical -> normalize -> denormalize -> same physical, within float32 eps."""
    torch.manual_seed(0)
    t = torch.rand(256, STARVLA_ACTION_DIM)
    physical = normalizer.q01 + t * (normalizer.q99 - normalizer.q01)

    recovered = normalizer.denormalize(normalizer.normalize(physical))
    torch.testing.assert_close(recovered, physical, rtol=0, atol=1e-5)


def test_degenerate_dims_pass_through_unnormalized(normalizer):
    """Model output on degenerate dims is physical, not in [-1, 1]."""
    physical = normalizer.q99.clone().unsqueeze(0)
    normalized = normalizer.normalize(physical)
    for dim in normalizer.degenerate_dims:
        assert normalized[0, dim] == pytest.approx(float(normalizer.q99[dim])), (
            f"dim {dim} was normalized but should pass through raw"
        )
    # e.g. the left hand grips the tube at a constant 1.7 rad.
    assert normalized[0, 14] == pytest.approx(1.7)
    assert normalized[0, 26] == pytest.approx(0.76)


def test_denormalize_ignores_model_output_on_degenerate_dims(normalizer):
    """Upstream's unmasked inverse emits the dataset constant regardless of input."""
    a = torch.zeros(1, STARVLA_ACTION_DIM)
    b = torch.full((1, STARVLA_ACTION_DIM), 7.0)
    da, db = normalizer.denormalize(a), normalizer.denormalize(b)
    for dim in normalizer.degenerate_dims:
        assert da[0, dim] == pytest.approx(db[0, dim])
        assert da[0, dim] == pytest.approx(float(normalizer.q01[dim]))


def test_normalize_clamps_outliers(normalizer):
    """Out-of-range physical values clamp to [-2.2, 2.2], not [-1, 1]."""
    physical = normalizer.q99 + 1000.0
    normalized = normalizer.normalize(physical.unsqueeze(0))
    varying = normalizer.mask
    assert normalized[0, varying].max() <= 2.2 + 1e-6
    assert normalized[0][varying].max() == pytest.approx(2.2)


def test_normalizer_matches_starvla_reference_implementation(normalizer):
    """Re-implement StarVLA's Normalizer.forward inline and require agreement."""
    q01, q99 = normalizer.q01, normalizer.q99
    mask = q01 != q99

    def starvla_forward(x):
        normalized = torch.zeros_like(x)
        normalized[..., mask] = (x[..., mask] - q01[mask]) / (q99[mask] - q01[mask])
        normalized[..., mask] = 2 * normalized[..., mask] - 1
        normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
        return torch.clamp(normalized, -2.2, 2.2)

    def starvla_inverse(x):
        return (x + 1) / 2 * (q99 - q01) + q01

    torch.manual_seed(1)
    x = torch.rand(64, STARVLA_ACTION_DIM) * 4 - 2
    torch.testing.assert_close(normalizer.normalize(x), starvla_forward(x))
    torch.testing.assert_close(normalizer.denormalize(x), starvla_inverse(x))


def test_unknown_unnorm_key_raises():
    with pytest.raises(KeyError, match="not in dataset statistics"):
        Q99ActionNormalizer.from_dataset_statistics(
            DATASET_STATS, unnorm_key="does_not_exist"
        )


# --------------------------------------------------------------------------
# Phase 2: camera optics
# --------------------------------------------------------------------------


def test_d435_optics_reproduce_dataset_fov():
    hfov, vfov = compute_fov_deg(14.55, 20.0, 424, 240)
    assert hfov == pytest.approx(EGO_HFOV_DEG, abs=0.1)
    assert vfov == pytest.approx(EGO_VFOV_DEG, abs=0.1)
    assert EGO_NATIVE_HW == (240, 424)


def test_stock_preset_would_be_out_of_distribution():
    """The un-fixed preset renders 105.5 deg -- documents why the fix is required."""
    hfov, vfov = compute_fov_deg(7.6, 20.0, 640, 480)
    assert hfov == pytest.approx(105.5, abs=0.1)
    assert abs(hfov - EGO_HFOV_DEG) > 30


def test_resizing_cannot_fix_fov():
    """A square resize keeps the wrong HFOV and additionally distorts VFOV.

    Guards against the tempting mistake of 'fixing' the camera by changing the
    resolution: FOV is baked into the render.
    """
    hfov_256, vfov_256 = compute_fov_deg(7.6, 20.0, 256, 256)
    assert hfov_256 == pytest.approx(105.5, abs=0.1)  # unchanged by resolution
    assert vfov_256 == pytest.approx(105.5, abs=0.1)  # worse than the 89.2 stock VFOV


def test_env_yaml_does_not_override_camera_resolution():
    """Overriding height/width from RLinf would silently re-break the VFOV."""
    with open(ENV_YAML) as f:
        cfg = yaml.safe_load(f)
    init = cfg["init_params"]
    for key in ("front_cam", "left_wrist_cam", "right_wrist_cam"):
        assert key not in init, f"{key} override would re-break the ego FOV"
    assert init["enable_wrist_cameras"] is False
    assert init["verify_ego_optics"] is True


# --------------------------------------------------------------------------
# Phase 5: 50 Hz -> 100 Hz zero-order hold config
# --------------------------------------------------------------------------


def test_episode_budget_is_in_env_steps():
    """2000 env steps = 1000 policy actions at 50 Hz; episode_length_s stays 20.0."""
    with open(ENV_YAML) as f:
        cfg = yaml.safe_load(f)
    assert cfg["max_episode_steps"] == 2000
    assert cfg["init_params"]["action_hold_steps"] == 2
    # 30 policy actions per chunk -> 60 env steps; 2000 // 30 chunk iterations.
    assert cfg["max_episode_steps"] % 2 == 0
