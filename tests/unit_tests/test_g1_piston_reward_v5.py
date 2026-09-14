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

"""v5: a geometry-grounded press. Transport stages stay identical to v3.

Measured facts the predicate rests on (runs_g1_piston/diag/press_*): the rod protrudes
25 mm above the barrel, so 25 mm is the deepest clean press (v3 demanded 28 mm); a thumb
press holds 24.5 mm steadily; the demonstrations' only 'press' is a ~6-step snatch.
"""

import importlib.util
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tests.unit_tests.test_g1_piston_reward_v3 import FakeScene  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


V5 = _load("g1r5_t", "rlinf/envs/isaaclab/tasks/g1_piston_reward_v5.py")
V3 = _load("g1r3_t5", "rlinf/envs/isaaclab/tasks/g1_piston_reward_v3.py")


def test_constants_are_geometry_grounded():
    assert V5.REWARD_VERSION == "v5_geometric_press"
    assert V5.PRESS_DEPTH == pytest.approx(0.020)
    assert V5.ROD_PROTRUSION == pytest.approx(0.025)
    assert V5.PRESS_DEPTH < V5.ROD_PROTRUSION < V3.PRESS_DEPTH, "v3's threshold is past the ceiling"
    assert V5.PRESS_SUSTAIN_STEPS == 15 > 6
    # transport constants untouched
    for k in ("GRASP_RADIUS", "BARREL_HALF_H", "GRASP_SPAN", "LIFT_H", "PLATE_NEAR", "W_TRANSPORT", "STAGE_BONUS"):
        assert getattr(V5, k) == getattr(V3, k), k


def _transport(sc, rf):
    rf.reset(); rf.step()
    out = []
    sc.grip(); out.append(rf.step())
    # raise and translate slowly (0.3 m/s at dt 0.02): a fast move fails the ballistic test
    for k in range(1, 21):
        sc.place((0.0, 0.0), 0.89 + 0.12 * k / 20); out.append(rf.step())
    for k in range(1, 61):
        sc.place((0.30 * k / 60, -0.10 * k / 60), 1.01); out.append(rf.step())
    out.append(rf.step())
    return out


def test_transport_rewards_and_stages_identical_to_v3_when_no_press():
    a, b = FakeScene(), FakeScene()
    o5 = _transport(a, V5.PistonTaskRewardV5(a, dt=0.02))
    o3 = _transport(b, V3.PistonTaskRewardV3(b, dt=0.02))
    for (r5, i5), (r3, i3) in zip(o5, o3):
        assert r5 == pytest.approx(r3)
        assert i5["stages"] == i3["stages"]
    assert o5[-1][1]["stages"]["plate"] is True


def _thumb_press_pose(sc, depth):
    """Fingers on the barrel; thumb ABOVE the barrel top, on the rod. v3's thumb test fails."""
    sc.fingers = np.array([sc.barrel + np.array([0.02, 0.0, 0.03])] * 4)
    sc.thumb = sc.barrel + np.array([0.016, 0.0, V5.BARREL_HALF_H + 0.002])
    sc.press = depth
    sc.rod = sc.barrel + np.array([0.0, 0.0, 0.03 - depth])


def test_sustained_thumb_press_while_lifted_counts_and_is_not_a_v3_grasp():
    sc = FakeScene(); rf = V5.PistonTaskRewardV5(sc, dt=0.02)
    _transport(sc, rf)
    total = 0.0
    for _ in range(V5.PRESS_SUSTAIN_STEPS - 1):
        _thumb_press_pose(sc, 0.0245); r, info = rf.step(); total += r
        assert info["grasped"] is False and info["finger_hold"] is True
        assert info["stages"]["press"] is False
    _thumb_press_pose(sc, 0.0245); r, info = rf.step(); total += r
    assert info["stages"]["press"] is True
    assert info["stages"]["dispense"] is True          # over the plate after plate stage
    assert total > V5.STAGE_BONUS["press"] + V5.STAGE_BONUS["dispense"]
    assert info["max_press_held_m"] == pytest.approx(0.0245)


def test_same_pose_never_fires_under_v3():
    sc = FakeScene(); rf = V3.PistonTaskRewardV3(sc, dt=0.02)
    _transport(sc, rf)
    for _ in range(40):
        _thumb_press_pose(sc, 0.0245); _, info = rf.step()
    assert info["stages"]["press"] is False


def test_inertial_transient_does_not_count():
    sc = FakeScene(); rf = V5.PistonTaskRewardV5(sc, dt=0.02)
    _transport(sc, rf)
    for _ in range(6):
        _thumb_press_pose(sc, 0.024); _, info = rf.step()
    for _ in range(20):
        _thumb_press_pose(sc, 0.008); _, info = rf.step()
    assert info["stages"]["press"] is False and info["press_sustain_steps"] == 0


def test_table_press_pays_nothing():
    sc = FakeScene(); rf = V5.PistonTaskRewardV5(sc, dt=0.02)
    rf.reset(); rf.step(); sc.grip(); rf.step()
    total = 0.0
    for _ in range(40):
        _thumb_press_pose(sc, 0.035); r, info = rf.step(); total += r
    assert info["stages"]["press"] is False and info["stages"]["dispense"] is False
    assert info["lift"] < V5.LIFT_H
    assert total < 0.5                                  # no dense press term either


def test_tip_on_plate_press_after_transport_is_a_dispense_without_the_lift_gate():
    """The robust mechanism: pipette carried to the plate, tip rested on it (lift ~0.01),
    hand pushed down until the palm meets the rod top. Dispense fires; 'press' (lifted) does not."""
    sc = FakeScene(); rf = V5.PistonTaskRewardV5(sc, dt=0.02)
    _transport(sc, rf)
    for k in range(1, 21):                               # lower onto the plate, slowly
        sc.place((0.30, -0.10), 1.01 - 0.11 * k / 20); rf.step()
    total = 0.0
    for _ in range(V5.PRESS_SUSTAIN_STEPS + 1):
        sc.grip(); sc.press = 0.024; sc.rod = sc.barrel + np.array([0, 0, 0.03 - 0.024]); r, info = rf.step(); total += r
    assert info["lift"] < V5.LIFT_H
    assert info["stages"]["dispense"] is True and info["stages"]["press"] is False
    assert total > V5.STAGE_BONUS["dispense"]
    assert info["max_press_plate_m"] == pytest.approx(0.024)


def test_press_away_from_plate_is_not_dispense():
    sc = FakeScene(); rf = V5.PistonTaskRewardV5(sc, dt=0.02)
    rf.reset(); rf.step(); sc.grip(); rf.step()
    for k in range(1, 21):
        sc.place((0.0, 0.0), 0.89 + 0.12 * k / 20); rf.step()
    for _ in range(V5.PRESS_SUSTAIN_STEPS + 2):
        _thumb_press_pose(sc, 0.0245); _, info = rf.step()
    assert info["stages"]["press"] is True and info["stages"]["dispense"] is False


def test_finger_hold_follows_the_barrel_axis_when_the_pipette_tilts():
    """Fingers wrapped round the barrel 8 cm above its centre, pipette tilted 30 deg: the xy
    test to the centre says 4 cm (flickering at the 4.5 cm gate); the axis test says 2 cm."""
    import math

    class TiltedScene(FakeScene):
        def __getitem__(self, k):
            e = super().__getitem__(k)
            if k == "object":
                # barrel tilted 30 deg about +y: quaternion (w, x, y, z)
                half = math.radians(15.0)
                e.data.body_quat_w = torch.tensor([[[1.0, 0, 0, 0], [math.cos(half), 0.0, math.sin(half), 0.0]]])
            return e

    sc = TiltedScene(); rf = V5.PistonTaskRewardV5(sc, dt=0.02)
    rf.reset(); rf.step()
    axis = np.array([math.sin(math.radians(30)), 0.0, math.cos(math.radians(30))])
    grip = sc.barrel + 0.08 * axis
    side = np.array([math.cos(math.radians(30)), 0.0, -math.sin(math.radians(30))])  # perpendicular
    sc.fingers = np.array([grip + 0.02 * side] * 4)
    sc.thumb = grip - 0.02 * side
    _, info = rf.step()
    assert info["finger_radial"] > 0.035                     # v3's xy-to-centre measure
    assert info["finger_radial_axis"] == pytest.approx(0.02, abs=1e-3)
    assert info["finger_along_axis"] == pytest.approx(0.08, abs=1e-3)
    assert info["finger_hold"] is True
    assert info["barrel_tilt_deg"] == pytest.approx(30.0, abs=0.1)
