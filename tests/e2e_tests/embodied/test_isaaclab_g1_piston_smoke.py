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

"""Environment-only smoke test for the G1 + Inspire piston task.

No policy, no StarVLA checkpoint, no RL: this boots the external IsaacLab task
through the RLinf adapter, resets, applies a *known-safe* hold action (the
articulation default pose, which under ``use_default_offset=True`` is the
all-zeros action), steps, and validates shapes/finiteness.

Requires Isaac Sim + the external ``unitree_sim_isaaclab`` checkout and a GPU, so
it is skipped unless ``RLINF_G1_PISTON_SMOKE=1`` is set. Run it directly with the
Isaac conda interpreter::

    RLINF_G1_PISTON_SMOKE=1 UNITREE_SIM_PATH=$HOME/unitree_sim_isaaclab \\
        ~/miniconda3/envs/isaac/bin/python -m pytest \\
        tests/e2e_tests/embodied/test_isaaclab_g1_piston_smoke.py -s
"""

import os

import pytest
import torch
from omegaconf import OmegaConf

pytestmark = pytest.mark.skipif(
    os.environ.get("RLINF_G1_PISTON_SMOKE") != "1",
    reason="needs Isaac Sim + unitree_sim_isaaclab; set RLINF_G1_PISTON_SMOKE=1",
)

UNITREE_SIM_PATH = os.environ.get(
    "UNITREE_SIM_PATH", os.path.expanduser("~/unitree_sim_isaaclab")
)

#: Steps to run. Kept tiny: this is a smoke test, not a rollout.
NUM_STEPS = 5

#: Articulation DOF count, measured from a live sim boot. Note this is NOT
#: 29 + 12 = 41: the Inspire hands carry coupled/passive joints that the
#: observation terms do not expose. See docs/contracts/g1_piston_joint_contract.json.
ACTION_DIM = 53


def _make_cfg():
    return OmegaConf.create(
        {
            "env_type": "isaaclab",
            "seed": 0,
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": True,
            "reward_coef": 1.0,
            "max_episode_steps": 64,
            "video_cfg": {"save_video": False},
            "init_params": {
                "id": "Isaac-PickPlace-Piston-G129-Inspire-Joint",
                "num_envs": 1,
                "task_description": "Pick up the piston and place it in the pot.",
                "unitree_sim_path": UNITREE_SIM_PATH,
                # Camera optics come from the piston scene cfg (D435, 240x424,
                # 69 deg HFOV) and are deliberately not overridden here.
                "enable_wrist_cameras": False,
                "verify_ego_optics": True,
                "action_hold_steps": 2,
            },
        }
    )


@pytest.fixture(scope="module")
def env():
    from rlinf.envs.isaaclab.tasks.g1_piston import IsaaclabG1PistonEnv

    e = IsaaclabG1PistonEnv(
        cfg=_make_cfg(),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    yield e
    e.close()


def _check_obs(obs):
    assert "task_descriptions" in obs
    assert len(obs["task_descriptions"]) == 1

    # State vectors: 29 body joints (pos/vel/torque) + 12 Inspire joints.
    assert obs["robot_joint_state"].shape == (1, 87)
    assert obs["robot_inspire_state"].shape == (1, 12)
    assert obs["body_joint_pos"].shape == (1, 29)

    # The ego camera reproduces the dataset's D435 colour stream: 240x424, 69 deg
    # HFOV. Wrist cameras are disabled for evaluation (the policy never sees them).
    assert "front_camera" in obs, "ego camera missing from observation"
    frame = obs["front_camera"]
    assert frame.shape == (1, 240, 424, 3), f"front_camera has shape {frame.shape}"

    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value.float()).all(), f"non-finite obs in {key}"


def test_reset_returns_valid_observation(env):
    obs, _ = env.reset()
    _check_obs(obs)
    # The camera shim would hand back an all-zero frame; a real render should not
    # be uniformly zero across every camera.
    assert obs["front_camera"].float().abs().sum() > 0, (
        "ego camera frame is identically zero -- the sensor did not render"
    )


def test_step_with_hold_action(env):
    """A zero action == hold the articulation default pose (use_default_offset)."""
    obs, _ = env.reset()

    hold_action = torch.zeros(1, ACTION_DIM, device=env.device)

    for i in range(NUM_STEPS):
        obs, reward, terminations, truncations, infos = env.step(hold_action)

        _check_obs(obs)
        assert reward.shape == (1,), f"reward shape {reward.shape}"
        assert torch.isfinite(reward).all(), f"non-finite reward at step {i}"
        # Upstream reward is a 3-valued predicate: -1 / 0 / +1.
        assert reward.abs().max() <= 1.0 + 1e-6
        assert terminations.shape == (1,)
        assert truncations.shape == (1,)


def test_reset_after_step(env):
    """The env can be reset again after stepping (episode boundary path)."""
    env.reset()
    env.step(torch.zeros(1, ACTION_DIM, device=env.device))
    obs, _ = env.reset()
    _check_obs(obs)
