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

"""Inspire 6->12 retargeter: it must touch only hand joints, and only when closing."""

import json

import pytest
import torch

from rlinf.envs.isaaclab.tasks.g1_piston_action import (
    HELD_LEG_AND_WAIST_JOINTS,
    G1PistonActionMapper,
)
from rlinf.envs.isaaclab.tasks.g1_piston_hand_retarget import (
    DEFAULT_PARAMS,
    InspireHandRetargeter,
)

JOINT_CONTRACT = "docs/contracts/g1_piston_joint_contract.json"


def _joint_names():
    with open(JOINT_CONTRACT) as f:
        return json.load(f)["joint_names_in_articulation_order"]


def _mapped(action30, names):
    return G1PistonActionMapper(names).map(action30)


def test_retargeter_leaves_arm_and_leg_joints_untouched():
    """The adapter is a HAND adapter: it must not perturb the validated arm mapping."""
    names = _joint_names()
    rt = InspireHandRetargeter(names)
    a = torch.zeros(1, 30)
    a[0, 7:14] = 0.5  # right arm
    cmd = _mapped(a, names)
    out = rt.apply(cmd, a)

    arm_and_leg = [
        i
        for i, n in enumerate(names)
        if not (n.startswith("L_") or n.startswith("R_"))
    ]
    assert torch.allclose(out[0, arm_and_leg], cmd[0, arm_and_leg])
    for n in HELD_LEG_AND_WAIST_JOINTS:
        assert out[0, names.index(n)] == cmd[0, names.index(n)]


def test_open_hand_leaves_thumb_at_recorded_pose():
    """With the hand open the gate is 0, so the thumb must not swing into opposition."""
    names = _joint_names()
    rt = InspireHandRetargeter(names)
    a = torch.zeros(1, 30)  # all fingers open
    out = rt.apply(_mapped(a, names), a)
    assert out[0, names.index("R_thumb_proximal_yaw_joint")].abs() < 1e-6
    assert out[0, names.index("R_thumb_proximal_pitch_joint")].abs() < 1e-6


def test_closed_hand_drives_thumb_into_opposition():
    """Once the fingers close, the thumb must reach the calibrated opposing pose."""
    names = _joint_names()
    rt = InspireHandRetargeter(names)
    a = torch.zeros(1, 30)
    a[0, 20:24] = 1.3  # right hand fully closing
    out = rt.apply(_mapped(a, names), a)
    assert out[0, names.index("R_thumb_proximal_yaw_joint")].item() == pytest.approx(
        DEFAULT_PARAMS.thumb_yaw_target, abs=1e-6
    )
    assert out[0, names.index("R_thumb_proximal_pitch_joint")].item() == pytest.approx(
        DEFAULT_PARAMS.thumb_pitch_floor, abs=1e-6
    )
    # finger intermediates follow their proximal joints
    assert out[0, names.index("R_index_intermediate_joint")].item() == pytest.approx(1.3, abs=1e-6)


def test_retarget_never_exceeds_joint_limits():
    names = _joint_names()
    rt = InspireHandRetargeter(names)
    a = torch.full((1, 30), 5.0)  # absurd command
    out = rt.apply(_mapped(a, names), a)
    assert out[0, names.index("R_thumb_proximal_yaw_joint")] <= 1.3 + 1e-6
    assert out[0, names.index("R_thumb_proximal_pitch_joint")] <= 0.6 + 1e-6
    assert out[0, names.index("R_thumb_distal_joint")] <= 1.2 + 1e-6
    assert out[0, names.index("R_index_intermediate_joint")] <= 1.7 + 1e-6


def test_output_shape_preserved():
    names = _joint_names()
    rt = InspireHandRetargeter(names)
    a = torch.zeros(2, 30, 30)
    out = rt.apply(_mapped(a, names), a)
    assert out.shape == (2, 30, 53)


# ------------------------------------------------------------ press channel ----
from rlinf.envs.isaaclab.tasks.g1_piston_hand_retarget import InspireRetargetParams  # noqa: E402
import dataclasses  # noqa: E402


def _closed_right_hand(yaw):
    a = torch.zeros((1, 30))
    a[0, 20:24] = 1.3          # the demonstrations' closed grip
    a[0, 24] = 0.0
    a[0, 25] = yaw
    a[0, 14:18] = 1.7; a[0, 18] = 0.35; a[0, 19] = 0.25   # frozen left-hand grip
    return a


def test_press_channel_is_inert_over_the_whole_demonstrated_yaw_range():
    names = _joint_names()
    rt = InspireHandRetargeter(names)
    rt0 = InspireHandRetargeter(names, dataclasses.replace(DEFAULT_PARAMS, thumb_press_gain=0.0))
    for yaw in (-0.10, 0.0, 0.10, 0.25, 0.30):
        a = _closed_right_hand(yaw)
        assert torch.equal(rt.apply(_mapped(a, names), a), rt0.apply(_mapped(a, names), a)), yaw


def test_press_channel_drives_thumb_yaw_to_its_limit_at_0_42():
    names = _joint_names()
    rt = InspireHandRetargeter(names)
    a = _closed_right_hand(0.42)
    out = rt.apply(_mapped(a, names), a)
    assert out[0, names.index("R_thumb_proximal_yaw_joint")].item() == pytest.approx(1.30, abs=1e-6)
    # the left hand's frozen yaw (0.25) is below the channel start: untouched
    assert out[0, names.index("L_thumb_proximal_yaw_joint")].item() == pytest.approx(0.9, abs=1e-6)
    # nothing but the thumb yaw differs from the pre-channel output
    rt0 = InspireHandRetargeter(names, dataclasses.replace(DEFAULT_PARAMS, thumb_press_gain=0.0))
    diff = (out - rt0.apply(_mapped(a, names), a)).abs()[0]
    assert diff.nonzero().flatten().tolist() == [names.index("R_thumb_proximal_yaw_joint")]


def test_press_channel_is_monotone_and_reaches_limit_inside_the_policy_range():
    names = _joint_names()
    rt = InspireHandRetargeter(names)
    j = names.index("R_thumb_proximal_yaw_joint")
    ys = [rt.apply(_mapped(_closed_right_hand(y), names), _closed_right_hand(y))[0, j].item()
          for y in (0.30, 0.33, 0.36, 0.39, 0.42, 0.46)]
    assert all(b >= a for a, b in zip(ys, ys[1:]))
    assert ys[-1] == pytest.approx(1.3, abs=1e-6)   # 0.46 = the policy's ceiling, clamped
    assert InspireRetargetParams().as_dict()["thumb_press_start"] == 0.30


def test_press_curl_gains_default_off_and_ramp_with_the_channel():
    names = _joint_names()
    rt = InspireHandRetargeter(names)
    rt_curl = InspireHandRetargeter(names, dataclasses.replace(DEFAULT_PARAMS, thumb_press_inter_gain=0.4, thumb_press_distal_gain=0.6))
    ji, jd = names.index("R_thumb_intermediate_joint"), names.index("R_thumb_distal_joint")
    a = _closed_right_hand(0.25)                        # below the channel: identical
    assert torch.equal(rt.apply(_mapped(a, names), a), rt_curl.apply(_mapped(a, names), a))
    a = _closed_right_hand(0.42)                        # full press: +0.4 / +0.6, at the limits
    o0 = rt.apply(_mapped(a, names), a); o1 = rt_curl.apply(_mapped(a, names), a)
    assert o0[0, ji].item() == pytest.approx(0.4, abs=1e-6) and o0[0, jd].item() == pytest.approx(0.6, abs=1e-6)
    assert o1[0, ji].item() == pytest.approx(0.8, abs=1e-6) and o1[0, jd].item() == pytest.approx(1.2, abs=1e-6)
    a = _closed_right_hand(0.36)                        # half way up the channel
    o = rt_curl.apply(_mapped(a, names), a)
    assert o[0, ji].item() == pytest.approx(0.6, abs=1e-6) and o[0, jd].item() == pytest.approx(0.9, abs=1e-6)
