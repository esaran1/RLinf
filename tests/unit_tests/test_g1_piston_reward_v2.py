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

"""The v2 reward must score the functional pipette task and resist reward hacking.

Guards the plunger-DoF finding: the scene object is an articulation whose prismatic
``PistonJoint`` (travel 0-0.04 m, spring-loaded) is the plunger, and v1 never read it.
"""

import numpy as np
import pytest

from rlinf.envs.isaaclab.tasks.g1_piston_reward_v2 import (
    BALLISTIC_SPEED,
    PLUNGER_TRAVEL,
    PRESS_DEPTH,
    PistonTaskRewardV2,
    stage_order,
)


class _FakeTensor:
    """Minimal stand-in for the torch tensors the reward reads off the scene."""

    def __init__(self, arr):
        self._a = np.asarray(arr, dtype=np.float64)

    def __getitem__(self, idx):
        return _FakeTensor(self._a[idx])

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def __float__(self):
        return float(self._a)


class _Data:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Entity:
    def __init__(self, **kw):
        self.data = _Data(**kw)


class FakeScene:
    """Scriptable scene: set positions and the plunger joint, then step the reward."""

    BODY_NAMES = (
        [f"filler{i}" for i in range(10)]
        + ["R_index_intermediate", "R_middle_intermediate",
           "R_pinky_intermediate", "R_ring_intermediate", "R_thumb_distal"]
    )

    def __init__(self):
        self.barrel = np.array([0.0, 0.0, 0.89])
        self.rod = np.array([0.0, 0.0, 0.92])
        self.tube = np.array([-0.13, -0.03, 0.87])
        self.pot = np.array([0.30, -0.10, 0.80])
        self.fingers = np.array([[1.0, 1.0, 1.0]] * 4)
        self.thumb = np.array([1.0, 1.0, 1.0])
        self.press = 0.0

    def __getitem__(self, key):
        if key == "object":
            return _Entity(
                body_pos_w=_FakeTensor(np.stack([self.rod, self.barrel])[None]),
                joint_pos=_FakeTensor(np.array([[self.press]])),
                joint_names=["PistonJoint"],
            )
        if key == "tube":
            return _Entity(root_pos_w=_FakeTensor(self.tube[None]))
        if key == "pot":
            return _Entity(root_pos_w=_FakeTensor(self.pot[None]))
        if key == "robot":
            rows = [np.array([9.0, 9.0, 9.0])] * 10 + list(self.fingers) + [self.thumb]
            return _Entity(body_pos_w=_FakeTensor(np.stack(rows)[None]),
                           body_names=list(self.BODY_NAMES))
        raise KeyError(key)

    # -- scripting helpers -------------------------------------------------------
    def hold(self):
        """Put the hand on the barrel (fires reach + grasp)."""
        self.fingers = np.array([self.barrel + np.array([0.005, 0.0, 0.0])] * 4)
        self.thumb = self.barrel + np.array([-0.005, 0.0, 0.0])

    def move_to(self, xy, z=None):
        z = self.barrel[2] if z is None else z
        self.barrel = np.array([xy[0], xy[1], z])
        self.rod = self.barrel + np.array([0.0, 0.0, 0.03])
        self.hold()


def glide(scene, rw, xy, z, steps=25):
    """Move the pipette to a pose at carry speed, stepping the reward each tick.

    Teleporting between poses registers as several m/s, which the ballistic guard
    correctly rejects; a real carry is well under BALLISTIC_SPEED.
    """
    start = scene.barrel.copy()
    target = np.array([xy[0], xy[1], z], dtype=float)
    out = None
    for k in range(1, steps + 1):
        p = start + (target - start) * (k / steps)
        scene.move_to((p[0], p[1]), z=p[2])
        out = rw.step()
    return out


def _rw(scene):
    return PistonTaskRewardV2(scene, dt=0.02)


def _settle(scene, rw, n=3):
    """Step a few times with no motion so the 'settled' test can pass."""
    out = None
    for _ in range(n):
        out = rw.step()
    return out


def test_stage_order_includes_the_functional_stages():
    order = stage_order()
    assert "press" in order and "dispense" in order
    assert order.index("tube") < order.index("dispense")
    assert order.index("dispense") < order.index("success")


def test_plunger_travel_matches_the_scene():
    """Pinned to the measured scene value; a scene change must break this test."""
    assert PLUNGER_TRAVEL == pytest.approx(0.04)
    assert PRESS_DEPTH == pytest.approx(0.02)


def test_v1_style_transport_is_not_a_v2_success():
    """The whole point: carrying the pipette to the plate without ever pressing the
    plunger scored success under v1. Under v2 it must not."""
    sc = FakeScene()
    rw = _rw(sc)
    rw.step()
    sc.hold()
    rw.step()
    glide(sc, rw, (0.0, 0.0), 0.89 + 0.10)     # lifted, at carry speed
    glide(sc, rw, sc.pot[:2], 0.96)            # over the plate, above PLATE_Z
    _, info = _settle(sc, rw)
    assert info["stages"]["lift"] and info["stages"]["plate"]
    assert info["success_v1_predicate"] is True     # v1 would have called this success
    assert info["stages"]["success"] is False       # v2 does not
    assert info["stages"]["press"] is False


def test_full_functional_sequence_reaches_v2_success():
    sc = FakeScene()
    rw = _rw(sc)
    rw.step()
    sc.hold()
    rw.step()
    glide(sc, rw, (0.0, 0.0), 0.89 + 0.10)                      # lift
    glide(sc, rw, sc.tube[:2], sc.tube[2] + 0.03)               # aligned over tube
    sc.press = 0.03                                             # depress plunger
    _, info = rw.step()
    assert info["stages"]["tube"] and info["stages"]["press"]
    assert info["stages"]["dispense"] is True
    sc.press = 0.0                                              # spring returns
    glide(sc, rw, sc.pot[:2], 0.96)                             # carry to plate
    _, info = _settle(sc, rw)
    assert info["stages"]["success"] is True


def test_pressing_the_plunger_away_from_the_tube_is_not_dispensing():
    sc = FakeScene()
    rw = _rw(sc)
    rw.step()
    sc.hold()
    rw.step()
    glide(sc, rw, (0.20, 0.20), 0.89 + 0.10)    # lifted, far from the tube
    sc.press = 0.035
    _, info = rw.step()
    assert info["stages"]["press"] is True
    assert info["stages"]["dispense"] is False


def test_pumping_the_spring_loaded_plunger_cannot_farm_reward():
    """The joint springs back to rest, so a per-step press term would be farmable.
    Best-so-far shaping must make repeated pumping pay nothing after the first press."""
    sc = FakeScene()
    rw = _rw(sc)
    rw.step()
    sc.hold()
    rw.step()
    sc.press = 0.03
    first, _ = rw.step()
    total_pumped = 0.0
    for _ in range(20):
        sc.press = 0.0
        r_a, _ = rw.step()
        sc.press = 0.03
        r_b, _ = rw.step()
        total_pumped += r_a + r_b
    # Twenty full pump cycles must not pay more than the single genuine press did.
    assert total_pumped < first, (total_pumped, first)
    assert total_pumped <= 0.0 + 1e-9


def test_partial_compliance_press_is_not_credited_as_a_press():
    """Merely gripping the pipette compresses the spring slightly (measured ~9 mm on
    demonstrations). That must stay below the press threshold."""
    sc = FakeScene()
    rw = _rw(sc)
    rw.step()
    sc.hold()
    sc.press = 0.0093           # the measured demonstration value
    _, info = rw.step()
    assert info["stages"]["press"] is False
    assert info["press_frac"] < 0.5


def test_a_thrown_piston_does_not_bank_the_lift_bonus():
    """Closes the v1 exploit: height alone paid, so flinging banked reward."""
    sc = FakeScene()
    rw = _rw(sc)
    rw.step()
    sc.hold()
    rw.step()
    # Teleport upward between steps -> speed far above the ballistic threshold.
    sc.move_to((0.0, 0.0), z=0.89 + 0.30)
    _, info = rw.step()
    assert info["speed"] > BALLISTIC_SPEED
    assert info["stages"]["lift"] is False


def test_a_carried_piston_does_bank_the_lift_bonus():
    sc = FakeScene()
    rw = _rw(sc)
    rw.step()
    sc.hold()
    rw.step()
    # Rise slowly: 0.006 m per 0.02 s step = 0.3 m/s, below the ballistic threshold.
    z = 0.89
    for _ in range(12):
        z += 0.006
        sc.move_to((0.0, 0.0), z=z)
        _, info = rw.step()
    assert info["lift"] > 0.05
    assert info["speed"] < BALLISTIC_SPEED
    assert info["stages"]["lift"] is True


def test_reset_makes_episodes_independent():
    sc = FakeScene()
    rw = _rw(sc)
    rw.step()
    sc.hold()
    sc.press = 0.035
    rw.step()
    rw.reset()
    sc.press = 0.0
    sc.fingers = np.array([[1.0, 1.0, 1.0]] * 4)
    sc.thumb = np.array([1.0, 1.0, 1.0])
    _, info = rw.step()
    assert not any(info["stages"].values())
    assert info["max_press_m"] == 0.0


def test_info_is_a_superset_of_v1_fields():
    """v1 consumers must keep working against a v2 rollout."""
    sc = FakeScene()
    rw = _rw(sc)
    _, info = rw.step()
    for k in ("stages", "grasped", "finger_radial", "thumb_radial", "lift",
              "d_pot_xy", "success"):
        assert k in info


def test_press_reward_requires_holding_the_pipette():
    """Pressing the plunger without having grasped it (e.g. mashing it on the table)
    must pay nothing."""
    sc = FakeScene()
    rw = _rw(sc)
    rw.step()
    sc.press = 0.04
    r, info = rw.step()
    assert info["stages"]["press"] is False
    assert r <= 0.0
