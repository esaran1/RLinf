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

"""Static tests for the G1 + Inspire piston IsaacLab adapter.

These run without Isaac Sim: they cover registration, the observation reshaping
contract, and the guard rails around the documented upstream limitations. The
simulator-backed reset/step smoke test lives in
``tests/e2e_tests/embodied/test_isaaclab_g1_piston_smoke.py`` because it needs a
GPU and an Isaac Sim install.
"""

import types

import pytest
import torch
import yaml

from rlinf.envs import SupportedEnvType, get_env_cls
from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS
from rlinf.envs.isaaclab.tasks.g1_piston import (
    CAMERA_SCENE_KEYS,
    NUM_BODY_JOINTS,
    NUM_INSPIRE_JOINTS,
    PENDING_STARVLA_HANDOFF,
    IsaaclabG1PistonEnv,
)

PISTON_TASK_ID = "Isaac-PickPlace-Piston-G129-Inspire-Joint"
STANDTEST_TASK_ID = "Isaac-PickPlace-Piston-StandTest-G129-Inspire-Joint"

ENV_YAML = "examples/embodiment/config/env/isaaclab_g1_piston.yaml"


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task_id", [PISTON_TASK_ID, STANDTEST_TASK_ID])
def test_task_registered(task_id):
    """Both piston variants resolve to the adapter through the RLinf registry."""
    assert task_id in REGISTER_ISAACLAB_ENVS
    assert REGISTER_ISAACLAB_ENVS[task_id] is IsaaclabG1PistonEnv


def test_get_env_cls_dispatches_to_piston_adapter():
    """The public env factory routes the piston task id to our adapter."""
    env_cfg = types.SimpleNamespace(
        init_params=types.SimpleNamespace(id=PISTON_TASK_ID)
    )
    assert (
        get_env_cls(SupportedEnvType.ISAACLAB.value, env_cfg) is IsaaclabG1PistonEnv
    )


def test_registering_piston_does_not_displace_stack_cube():
    """The pre-existing IsaacLab task still resolves (no registry regression)."""
    from rlinf.envs.isaaclab.tasks.stack_cube import IsaaclabStackCubeEnv

    assert (
        REGISTER_ISAACLAB_ENVS["Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0"]
        is IsaaclabStackCubeEnv
    )


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_env_yaml_matches_adapter_contract():
    """The shipped env YAML stays consistent with the adapter's constraints."""
    with open(ENV_YAML) as f:
        cfg = yaml.safe_load(f)

    assert cfg["env_type"] == SupportedEnvType.ISAACLAB.value
    assert cfg["init_params"]["id"] == PISTON_TASK_ID
    # num_envs must stay 1 while the upstream termination term is scalar-valued.
    assert cfg["total_num_envs"] == 1
    assert cfg["init_params"]["num_envs"] == 1
    # The external simulator path must be provided or the task ids never register.
    assert cfg["init_params"]["unitree_sim_path"]


# --------------------------------------------------------------------------
# Guard rails around documented upstream limitations
# --------------------------------------------------------------------------


def test_multi_env_is_rejected():
    """num_envs > 1 fails fast rather than tripping the scalar termination term."""
    with pytest.raises(AssertionError, match="num_envs == 1"):
        IsaaclabG1PistonEnv.__init__(
            object.__new__(IsaaclabG1PistonEnv),
            cfg=None,
            num_envs=4,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )


def test_recorded_joint_contract_matches_adapter_constants():
    """The empirically captured articulation contract stays in sync with the code.

    Captured from a live sim boot; see docs/contracts/g1_piston_joint_contract.json.
    Guards the non-obvious fact that the action space is 53 DOFs while the
    observation terms only expose 29 + 12 = 41.
    """
    import json

    with open("docs/contracts/g1_piston_joint_contract.json") as f:
        contract = json.load(f)

    assert contract["num_articulation_dofs"] == 53
    assert contract["action_space"] == [1, 53]
    assert len(contract["joint_names_in_articulation_order"]) == 53
    assert len(contract["body_obs_gather_order_names"]) == NUM_BODY_JOINTS
    assert len(contract["inspire_obs_gather_order_names"]) == NUM_INSPIRE_JOINTS

    # A naive concat of the observed joints would be 41 -- and therefore wrong.
    assert NUM_BODY_JOINTS + NUM_INSPIRE_JOINTS != contract["num_articulation_dofs"]

    # The Inspire gather order is right hand first, contrary to the upstream comment.
    assert contract["inspire_obs_gather_order_names"][0].startswith("R_")
    assert contract["inspire_obs_gather_order_names"][6].startswith("L_")


def test_handoff_fields_are_declared_pending():
    """We never silently invent StarVLA-side control semantics."""
    for field in (
        "action_dim",
        "action_horizon",
        "normalization_key",
        "camera_key_order",
        "state_enabled",
    ):
        assert field in PENDING_STARVLA_HANDOFF


# --------------------------------------------------------------------------
# Observation mapping (no simulator required)
# --------------------------------------------------------------------------


def _make_stub_env(num_envs=1):
    """An adapter instance with just enough state to exercise ``_wrap_obs``."""
    env = object.__new__(IsaaclabG1PistonEnv)
    env.num_envs = num_envs
    env.task_description = "Pick up the piston and place it in the pot."
    return env


def _make_raw_obs(num_envs=1, with_cameras=True, height=256, width=256):
    """The observation as it arrives from the simulator subprocess.

    Camera frames are already inside the ``policy`` group: ``_CameraInjectingEnv``
    puts them there on the simulator side, since the parent process has no scene.
    """
    policy = {
        "robot_joint_state": torch.arange(
            num_envs * 3 * NUM_BODY_JOINTS, dtype=torch.float32
        ).reshape(num_envs, 3 * NUM_BODY_JOINTS),
        "robot_inspire_state": torch.arange(
            num_envs * NUM_INSPIRE_JOINTS, dtype=torch.float32
        ).reshape(num_envs, NUM_INSPIRE_JOINTS),
        # The upstream placeholder: constant zeros, deliberately ignored.
        "camera_image": torch.zeros(1, 480, 640, 3),
    }
    if with_cameras:
        for key in CAMERA_SCENE_KEYS:
            policy[key] = torch.zeros(num_envs, height, width, 3, dtype=torch.uint8)
    return {"policy": policy}


def test_wrap_obs_state_dimensions():
    """Joint telemetry is split into pos/vel/torque at the documented offsets."""
    env = _make_stub_env()
    out = env._wrap_obs(_make_raw_obs())

    assert out["robot_joint_state"].shape == (1, 3 * NUM_BODY_JOINTS)
    assert out["robot_inspire_state"].shape == (1, NUM_INSPIRE_JOINTS)
    assert out["body_joint_pos"].shape == (1, NUM_BODY_JOINTS)
    assert out["body_joint_vel"].shape == (1, NUM_BODY_JOINTS)
    assert out["body_joint_torque"].shape == (1, NUM_BODY_JOINTS)
    assert out["inspire_joint_pos"].shape == (1, NUM_INSPIRE_JOINTS)


def test_wrap_obs_split_is_contiguous_and_ordered():
    """pos | vel | torque concatenate back to the original 87-vector."""
    env = _make_stub_env()
    raw = _make_raw_obs()
    out = env._wrap_obs(raw)

    recombined = torch.cat(
        [out["body_joint_pos"], out["body_joint_vel"], out["body_joint_torque"]],
        dim=1,
    )
    torch.testing.assert_close(recombined, raw["policy"]["robot_joint_state"])


def test_wrap_obs_camera_shim_is_bypassed():
    """Images come from the injected sensor frames, not the zero placeholder term."""
    env = _make_stub_env()
    raw = _make_raw_obs()
    # Make a sensor frame distinguishable from the zero placeholder.
    raw["policy"]["front_camera"].fill_(7)

    out = env._wrap_obs(raw)

    for key in CAMERA_SCENE_KEYS:
        assert key in out, f"missing camera {key}"
        assert out[key].shape == (1, 256, 256, 3)
    assert out["front_camera"].max() == 7, "adapter fell back to the zero placeholder"
    # The useless upstream term must not leak into the policy observation.
    assert "camera_image" not in out


def test_wrap_obs_task_description_batched():
    """Instruction is one string per env."""
    env = _make_stub_env()
    out = env._wrap_obs(_make_raw_obs())
    assert out["task_descriptions"] == [env.task_description]


def test_wrap_obs_missing_camera_is_omitted_not_zeroed():
    """A missing sensor drops the key so shape checks fail loudly downstream."""
    env = _make_stub_env()
    out = env._wrap_obs(_make_raw_obs(with_cameras=False))
    for key in CAMERA_SCENE_KEYS:
        assert key not in out


# --------------------------------------------------------------------------
# Camera injection across the subprocess boundary
# --------------------------------------------------------------------------


def _make_fake_sim_env(height=256, width=256):
    """A stand-in for the IsaacLab env living inside the simulator subprocess."""
    scene = {
        key: types.SimpleNamespace(
            data=types.SimpleNamespace(
                output={"rgb": torch.full((1, height, width, 3), 5, dtype=torch.uint8)}
            )
        )
        for key in CAMERA_SCENE_KEYS
    }

    class _FakeEnv:
        def __init__(self):
            self.scene = scene
            self.device = "cpu"

        def reset(self, **kwargs):
            return _make_raw_obs(with_cameras=False), {}

        def step(self, action):
            return (
                _make_raw_obs(with_cameras=False),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {},
            )

    return _FakeEnv()


def test_camera_injection_on_reset_and_step():
    """The wrapper attaches sensor frames so they survive the process boundary."""
    from rlinf.envs.isaaclab.tasks.g1_piston import _CameraInjectingEnv

    wrapped = _CameraInjectingEnv(_make_fake_sim_env())

    obs, _ = wrapped.reset()
    for key in CAMERA_SCENE_KEYS:
        assert key in obs["policy"], f"{key} not injected on reset"
        assert obs["policy"][key].shape == (1, 256, 256, 3)

    obs, _, _, _, _ = wrapped.step(torch.zeros(1, 53))
    for key in CAMERA_SCENE_KEYS:
        assert key in obs["policy"], f"{key} not injected on step"


def test_camera_injection_delegates_unknown_attributes():
    """Non-overridden attributes still resolve to the wrapped env."""
    from rlinf.envs.isaaclab.tasks.g1_piston import _CameraInjectingEnv

    wrapped = _CameraInjectingEnv(_make_fake_sim_env())
    assert wrapped.device == "cpu"


def test_wrap_obs_all_values_finite():
    """No NaN/Inf is introduced by the reshaping."""
    env = _make_stub_env()
    out = env._wrap_obs(_make_raw_obs())
    for key, value in out.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value.float()).all(), f"non-finite in {key}"
