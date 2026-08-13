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

"""RLinf adapter for the external Unitree G1 (29 DOF) + Inspire piston pick-place task.

The simulator itself lives outside RLinf, in ``unitree_sim_isaaclab`` (plus the
``g1_redball_eval`` overlay that registers the piston variants). This module is a
*thin* adapter: it launches that already-registered Gym task and reshapes its
observations into the dict the RLinf embodied workers consume. No scene, reward, or
robot configuration is redefined here.

Task IDs provided by the overlay::

    Isaac-PickPlace-Piston-G129-Inspire-Joint            (primary)
    Isaac-PickPlace-Piston-StandTest-G129-Inspire-Joint  (collision-test variant)

Upstream contract, as read from the task sources
------------------------------------------------
Observation group ``policy`` (``concatenate_terms=False``, so a dict):

* ``robot_joint_state``   ``[B, 87]`` float32 -- 29 body joint positions, then 29
  velocities, then 29 applied torques, gathered in the DDS joint order
  ``[0,3,6,9,13,17, 1,4,7,10,14,18, 2,5,8,11,15,19,21,23,25,27, 12,16,20,22,24,26,28]``.
* ``robot_inspire_state`` ``[B, 12]`` float32 -- Inspire hand joint *positions* only,
  gathered in the order ``[36,37,35,34,48,38, 31,32,30,29,43,33]``. Resolved against
  the live articulation, this is **right hand first**, then left -- each as
  ``(pinky, ring, middle, index, thumb_pitch, thumb_yaw)`` proximal joints. Note the
  upstream comment claims left-first; the joint names say otherwise. See
  ``docs/contracts/g1_piston_joint_contract.json``.
* ``camera_image``        ``[1, 480, 640, 3]`` float32 -- see ``CAMERA_SHIM`` below.

Actions: ``mdp.JointPositionActionCfg(joint_names=[".*"], scale=1.0,
use_default_offset=True)`` over **all** articulation DOFs, i.e. *absolute* target
joint positions expressed as offsets from the articulation default pose, in radians,
in **IsaacLab articulation order** -- which is NOT the DDS observation order above.

The measured action space is ``Box(-inf, inf, (1, 53), float32)``: **53** DOFs, not
the 29 + 12 = 41 that the observation terms report. The articulation carries more
joints than the observation gathers expose (the Inspire hands have passive/coupled
joints that ``get_robot_inspire_joint_states`` does not sample). Any StarVLA action
vector must therefore be *mapped* into this 53-DOF ordering -- a straight concat of
29 body + 12 hand targets is wrong. The joint-name ordering needed for that mapping
is ``env.scene["robot"].data.joint_names``; deriving it is part of the pending
action-contract work, not guessed here.

Timing: ``sim.dt = 0.005`` s, ``decimation = 2`` -> 100 Hz control,
``episode_length_s = 20.0`` -> 2000 control steps per episode.

Known upstream limitations (deliberately surfaced, not silently patched)
-----------------------------------------------------------------------
1. ``CAMERA_SHIM``: the upstream ``get_camera_image`` observation term writes camera
   frames to *shared memory* for the DDS teleop stack and returns a constant
   ``torch.zeros((1, 480, 640, 3))`` placeholder. It is therefore useless as a policy
   observation. Until that is addressed, this adapter reads the camera sensors
   directly off the scene (``_grab_images``) instead of trusting that term.
2. ``TERMINATION_SCALAR``: ``reset_object_estimate`` combines per-env tensors with
   Python ``and``/``not``, which collapses to a single Python bool. It only works for
   ``num_envs == 1`` and raises for more. This adapter therefore pins ``num_envs`` to
   1 and asserts it.
3. ``REWARD_IS_NOT_SUCCESS``: ``compute_reward`` returns ``-1`` (object out of the
   workspace box), ``0`` (in workspace, not at target), or ``+1`` (inside the target
   "post" box). ``+1`` is a *placement region* predicate, not a verified task success,
   and the thresholds were authored for the cylinder scene, not the piston. Treat
   ``+1`` as the provisional success signal and re-validate the boxes against the
   piston scene before using it as an RL objective.
4. ``DDS_SIDE_EFFECTS``: the observation and reward terms try to publish to DDS. The
   calls are wrapped in try/except upstream, so they degrade to log noise when the
   DDS master is absent, but they do cost host-side copies each step.

Pending the StarVLA handoff
---------------------------
The mapping from this environment's observation into the StarVLA input contract
(camera ordering, whether ``states`` is fed at all and in what layout, action
dimension/horizon, normalization key) is **not** decided here. ``_wrap_obs``
deliberately exposes every camera under an explicit name and passes the raw joint
vectors through, so the handoff can pick without this file having guessed. See
``PENDING_STARVLA_HANDOFF``.
"""

from typing import Dict

import gymnasium as gym
import torch

from ..isaaclab_env import IsaaclabBaseEnv

#: Fields that cannot be fixed until the StarVLA agent delivers the SFT checkpoint
#: and its action contract. Kept as data so tests can assert we never silently
#: invented them.
PENDING_STARVLA_HANDOFF = (
    "action_dim",
    "action_horizon",
    "state_dim",
    "state_enabled",
    "camera_key_order",
    "image_preprocessing",
    "normalization_key",
    "dataset_statistics",
    "action_physical_semantics",
    "controller_expectation",
)

#: Number of G1 body joints reported by ``get_robot_boy_joint_states`` (pos/vel/torque).
NUM_BODY_JOINTS = 29

#: Number of Inspire hand joints reported by ``get_robot_inspire_joint_states``.
NUM_INSPIRE_JOINTS = 12

#: Camera scene keys, in the order the upstream task declares them. This is the
#: *environment's* order and is NOT asserted to be StarVLA's expected input order.
CAMERA_SCENE_KEYS = ("front_camera", "left_wrist_camera", "right_wrist_camera")


class _CameraInjectingEnv:
    """Copies rendered camera frames into the observation dict, in-process.

    RLinf runs Isaac Sim in a subprocess (``SubProcIsaacLabEnv``) and ships only the
    returned observation across the pipe; the parent holds a proxy with no ``scene``.
    Because the upstream ``camera_image`` observation term returns a constant zero
    placeholder (see ``CAMERA_SHIM``), the frames have to be read off the sensors and
    attached to the observation *here*, on the simulator side of the boundary.

    Everything else is delegated to the wrapped IsaacLab env.
    """

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        # Only called for attributes this wrapper does not define itself.
        return getattr(self._env, name)

    def _inject(self, obs):
        if not isinstance(obs, dict) or "policy" not in obs:
            return obs
        policy = obs["policy"]
        for key in CAMERA_SCENE_KEYS:
            try:
                policy[key] = self._env.scene[key].data.output["rgb"]
            except (KeyError, TypeError, AttributeError):
                # Leave the key absent so the adapter's shape checks fail loudly
                # rather than silently feeding a policy black frames.
                continue
        return obs

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        return self._inject(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._inject(obs), reward, terminated, truncated, info

    def close(self):
        return self._env.close()


class IsaaclabG1PistonEnv(IsaaclabBaseEnv):
    """Thin RLinf wrapper around ``Isaac-PickPlace-Piston-G129-Inspire-Joint``.

    Reuses the external task unchanged. Because the upstream termination term is
    scalar-valued (see ``TERMINATION_SCALAR`` in the module docstring), this env
    supports ``num_envs == 1`` only.
    """

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        assert num_envs == 1, (
            "IsaaclabG1PistonEnv supports num_envs == 1 only: the upstream "
            "termination term 'reset_object_estimate' combines per-env tensors with "
            "Python 'and'/'not', which raises for batched tensors. Increase "
            "num_envs only after that term is vectorized upstream."
        )
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

    def _make_env_function(self):
        """Build the callable that boots Isaac Sim inside the subprocess worker."""
        init_params = self.cfg.init_params
        env_id = self.isaaclab_env_id
        seed = self.seed
        num_envs = init_params.num_envs
        unitree_sim_path = init_params.get("unitree_sim_path", None)

        def make_env_isaaclab():
            import os
            import sys

            # Force headless: the RLinf worker has no display, and a stray DISPLAY
            # makes Isaac Sim try GLX and fail.
            os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            # The piston task is registered by the external unitree_sim_isaaclab
            # package, whose modules import as top-level 'tasks.*'. It must be on
            # sys.path *before* the registry import below.
            if unitree_sim_path is not None and unitree_sim_path not in sys.path:
                sys.path.insert(0, unitree_sim_path)

            # The scene configs resolve USD asset paths from $PROJECT_ROOT. Upstream
            # this is set by unitree_sim_isaaclab's own sim_main.py entry point,
            # which RLinf bypasses; without it the paths render as the literal
            # string "None/assets/..." and spawning fails with FileNotFoundError.
            if unitree_sim_path is not None:
                os.environ.setdefault("PROJECT_ROOT", unitree_sim_path)

            import tasks  # noqa: F401  (registers the gym task IDs as a side effect)

            from isaaclab_tasks.utils import load_cfg_from_registry

            isaac_env_cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
            isaac_env_cfg.seed = seed
            isaac_env_cfg.scene.num_envs = num_envs

            # Camera resolution is configurable; the upstream default is 480x640 per
            # camera, which is heavy. Only override when the config asks.
            for cam_key, cfg_key in (
                ("front_camera", "front_cam"),
                ("left_wrist_camera", "left_wrist_cam"),
                ("right_wrist_camera", "right_wrist_cam"),
            ):
                cam_cfg = init_params.get(cfg_key, None)
                if cam_cfg is None:
                    continue
                scene_cam = getattr(isaac_env_cfg.scene, cam_key, None)
                if scene_cam is None:
                    continue
                scene_cam.height = cam_cfg.height
                scene_cam.width = cam_cfg.width

            env = gym.make(
                env_id, cfg=isaac_env_cfg, render_mode="rgb_array"
            ).unwrapped
            return _CameraInjectingEnv(env), sim_app

        return make_env_isaaclab

    def _grab_images(self, obs) -> Dict[str, torch.Tensor]:
        """Return each camera's RGB frame, keyed by scene name.

        The frames are placed into the ``policy`` observation group by
        ``_CameraInjectingEnv`` on the simulator side of the subprocess boundary,
        because the parent process holds only a proxy with no ``scene`` and the
        upstream ``camera_image`` term is a zero placeholder (``CAMERA_SHIM``).

        A camera that failed to render is simply absent, so downstream shape checks
        fail loudly instead of a policy silently consuming black frames.
        """
        policy_obs = obs.get("policy", {}) if isinstance(obs, dict) else {}
        images: Dict[str, torch.Tensor] = {}
        for key in CAMERA_SCENE_KEYS:
            frame = policy_obs.get(key, None)
            if frame is not None:
                # IsaacLab yields [B, H, W, 3] uint8 RGB. dtype/layout conversion is
                # left to the (pending) StarVLA preprocessing contract.
                images[key] = frame
        return images

    def _wrap_obs(self, obs):
        """Reshape the upstream ``policy`` obs group into the RLinf env-obs dict.

        The joint vectors are passed through *unsplit and unreordered*: the upstream
        DDS gather order is documented in the module docstring, and remapping it into
        a policy state layout is part of the StarVLA action contract, which has not
        been delivered. Splitting it here would be a guess.
        """
        policy_obs = obs["policy"]

        body_state = policy_obs["robot_joint_state"]
        inspire_state = policy_obs["robot_inspire_state"]

        images = self._grab_images(obs)

        env_obs = {
            "task_descriptions": [self.task_description] * self.num_envs,
            # Raw, documented-order joint telemetry. See PENDING_STARVLA_HANDOFF:
            # which slice becomes the policy 'states' input is not decided here.
            "robot_joint_state": body_state,
            "robot_inspire_state": inspire_state,
            "body_joint_pos": body_state[:, :NUM_BODY_JOINTS],
            "body_joint_vel": body_state[:, NUM_BODY_JOINTS : 2 * NUM_BODY_JOINTS],
            "body_joint_torque": body_state[:, 2 * NUM_BODY_JOINTS :],
            "inspire_joint_pos": inspire_state,
        }

        # Cameras under explicit names; no 'main'/'wrist' aliasing until the handoff
        # fixes camera identity and ordering.
        for key, frame in images.items():
            env_obs[key] = frame

        return env_obs
