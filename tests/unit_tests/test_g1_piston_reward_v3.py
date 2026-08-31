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

"""v3 must close the reward holes a review found in v1/v2.

Each test names the reviewer point it guards:
  (1) reach/grasp ignored z and contact -> 3-D distance, height gate, opposition
  (2) lift could be satisfied by throwing -> held AND high AND slow
  (3) success fired for a pipette thrown into the sky at apex -> multi-step rest,
      a supported height band, and released
  (4) no smoothness term -> bounded jerk penalty while held
"""

import numpy as np
import pytest

SCRATCH = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
           "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")

from rlinf.envs.isaaclab.tasks.g1_piston_reward_v3 import (
    BARREL_HALF_H,
    GRASP_RADIUS,
    GRASP_SPAN,
    JERK_PENALTY_CAP,
    PLATE_REST_DZ,
    REST_STEPS,
    PistonTaskRewardV3,
    stage_order,
)


class _FT:
    def __init__(self, a):
        self._a = np.asarray(a, dtype=np.float64)

    def __getitem__(self, i):
        return _FT(self._a[i])

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


class _D:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _E:
    def __init__(self, **kw):
        self.data = _D(**kw)


class FakeScene:
    BODY_NAMES = ([f"f{i}" for i in range(10)]
                  + ["R_index_intermediate", "R_middle_intermediate",
                     "R_pinky_intermediate", "R_ring_intermediate", "R_thumb_distal"])

    def __init__(self):
        self.barrel = np.array([0.0, 0.0, 0.89])
        self.rod = np.array([0.0, 0.0, 0.92])
        self.tube = np.array([-0.13, -0.03, 0.87])
        self.pot = np.array([0.30, -0.10, 0.80])
        self.fingers = np.array([[1.0, 1.0, 1.0]] * 4)
        self.thumb = np.array([1.0, 1.0, 1.0])
        self.press = 0.0

    def __getitem__(self, k):
        if k == "object":
            return _E(body_pos_w=_FT(np.stack([self.rod, self.barrel])[None]),
                      joint_pos=_FT(np.array([[self.press]])),
                      joint_names=["PistonJoint"])
        if k == "tube":
            return _E(root_pos_w=_FT(self.tube[None]))
        if k == "pot":
            return _E(root_pos_w=_FT(self.pot[None]))
        if k == "robot":
            rows = [np.array([9.0, 9.0, 9.0])] * 10 + list(self.fingers) + [self.thumb]
            return _E(body_pos_w=_FT(np.stack(rows)[None]),
                      body_names=list(self.BODY_NAMES))
        raise KeyError(k)

    # --- scripting helpers ---------------------------------------------------
    def grip(self):
        """A real pinch: fingers and thumb on OPPOSITE sides, at barrel height."""
        self.fingers = np.array([self.barrel + np.array([0.02, 0.0, 0.0])] * 4)
        self.thumb = self.barrel + np.array([-0.02, 0.0, 0.0])

    def release(self):
        self.fingers = np.array([self.barrel + np.array([0.4, 0.0, 0.3])] * 4)
        self.thumb = self.barrel + np.array([0.4, 0.02, 0.3])

    def hover_above(self, dz=0.20):
        """Same xy as the barrel but far above it: an xy-only test called this a grasp."""
        self.fingers = np.array([self.barrel + np.array([0.02, 0.0, dz])] * 4)
        self.thumb = self.barrel + np.array([-0.02, 0.0, dz])

    def same_side(self):
        """Fingers and thumb both on one side: touching, but not opposing."""
        self.fingers = np.array([self.barrel + np.array([0.02, 0.0, 0.0])] * 4)
        self.thumb = self.barrel + np.array([0.03, 0.0, 0.0])

    def place(self, xy, z, gripping=True):
        self.barrel = np.array([xy[0], xy[1], z])
        self.rod = self.barrel + np.array([0.0, 0.0, 0.03])
        if gripping:
            self.grip()
        else:
            self.release()


def rw(sc):
    return PistonTaskRewardV3(sc, dt=0.02)


def glide(sc, r, xy, z, steps=25, gripping=True):
    start = sc.barrel.copy()
    tgt = np.array([xy[0], xy[1], z], dtype=float)
    out = None
    for k in range(1, steps + 1):
        p = start + (tgt - start) * (k / steps)
        sc.place((p[0], p[1]), p[2], gripping=gripping)
        out = r.step()
    return out


def hold_still(sc, r, n):
    out = None
    for _ in range(n):
        out = r.step()
    return out


# ---------------------------------------------------------------- point 1 ----
def test_hovering_above_the_pipette_is_not_a_grasp():
    """An xy-only distance test scored this as reach+grasp; the on-object height test
    is what rejects it. On the real rollout this case occurs on 169 of 690 steps,
    every one of which v1/v2 counted as a grasp."""
    sc = FakeScene()
    r = rw(sc)
    sc.hover_above(0.20)
    _, info = r.step()
    assert info["stages"]["grasp"] is False
    assert info["stages"]["reach"] is False


def test_a_same_side_hand_on_the_barrel_is_a_grasp_because_that_is_how_it_grips():
    """Calibrated from measurement, and it overturned an assumption.

    An earlier v3 required the thumb to OPPOSE the fingers about the barrel centre --
    a pinch. Measuring a real 690-step RLPD rollout showed the Inspire hand never does
    that on this object: the opposition cosine never went negative (min 0.754) and the
    thumb sits CLOSER to the axis (xy 0.033) than the fingers (0.054). It grips a
    cylinder by curling the fingers around it with the thumb bracing from the same
    side. Requiring opposition made the grasp predicate unsatisfiable -- 0 of 690 steps.
    So same-side digits ON the barrel must count as a grasp.
    """
    sc = FakeScene()
    r = rw(sc)
    sc.same_side()
    _, info = r.step()
    assert info["opposition_cos"] > 0.0        # same side, as the real hand does
    assert info["stages"]["grasp"] is True


def test_a_hand_on_the_barrel_at_barrel_height_is_a_grasp():
    sc = FakeScene()
    r = rw(sc)
    sc.grip()
    _, info = r.step()
    assert info["finger_radial"] < GRASP_RADIUS
    assert info["finger_dz"] < BARREL_HALF_H
    assert info["stages"]["grasp"] is True


def test_the_measured_geometry_satisfies_the_calibrated_predicate():
    """Guards against re-introducing an unsatisfiable threshold: replay the real
    measured geometry (RLPD@415k, 690 steps) through the predicate and require that it
    fires on some steps and is rejected on the off-object ones."""
    import os

    import numpy as np
    rows = os.path.join(SCRATCH, "graspgeom_rlpd415k_rows.npy")
    if not os.path.exists(rows):
        pytest.skip("measured grasp geometry not on this machine")
    R = np.load(rows)                       # fd3, td3, fxy, txy, dz, span, opp
    fxy, txy, dz, span = R[:, 2], R[:, 3], R[:, 4], R[:, 5]
    fires = (fxy < GRASP_RADIUS) & (txy < GRASP_RADIUS) & (dz < BARREL_HALF_H) & (span < GRASP_SPAN)
    assert fires.sum() > 0, "grasp predicate is unsatisfiable on real measured geometry"
    # And the hover case -- near the axis but off the object in z -- must be rejected.
    hover = (fxy < GRASP_RADIUS) & (dz >= BARREL_HALF_H)
    assert hover.sum() > 0                  # the data contains such steps
    assert not (hover & fires).any()         # none of them are scored as a grasp


def test_grasp_is_labelled_a_geometric_proxy():
    """The scene ships no contact sensors; the reward must say so rather than imply
    it reads contact forces."""
    sc = FakeScene()
    r = rw(sc)
    _, info = r.step()
    assert info["grasp_is_geometric_proxy"] is True


# ---------------------------------------------------------------- point 2 ----
def test_a_thrown_pipette_cannot_bank_the_lift_bonus():
    sc = FakeScene()
    r = rw(sc)
    sc.grip()
    r.step()
    # Released and flying upward fast.
    sc.place((0.0, 0.0), 1.20, gripping=False)
    _, info = r.step()
    assert info["stages"]["lift"] is False


def test_a_carried_pipette_banks_the_lift_bonus():
    sc = FakeScene()
    r = rw(sc)
    sc.grip()
    r.step()
    _, info = glide(sc, r, (0.0, 0.0), 0.89 + 0.10)
    assert info["stages"]["lift"] is True


# ---------------------------------------------------------------- point 3 ----
def test_a_pipette_thrown_into_the_sky_is_not_a_success():
    """The sharpest v1/v2 hole: at the apex of a throw vertical velocity passes
    through zero, so a single-step 'settled' test fired in mid-air."""
    sc = FakeScene()
    r = rw(sc)
    # Legitimately grasp, lift, align, dispense first, so only the final predicate
    # is under test.
    sc.grip()
    r.step()
    glide(sc, r, (0.0, 0.0), 0.89 + 0.10)
    glide(sc, r, sc.tube[:2], sc.tube[2] + 0.03)
    sc.press = 0.03
    _, info = r.step()
    assert info["stages"]["dispense"] is True
    sc.press = 0.0
    # Now throw it: high above the pot, released, momentarily motionless at apex.
    sc.place(sc.pot[:2], 1.60, gripping=False)
    info = hold_still(sc, r, REST_STEPS + 5)[1]
    assert info["at_rest"] is True          # it IS momentarily still ...
    assert info["supported"] is False       # ... but far outside the resting band
    assert info["stages"]["success"] is False


def test_still_while_held_is_not_a_success():
    """Holding the pipette motionless over the plate is not placing it."""
    sc = FakeScene()
    r = rw(sc)
    sc.grip()
    r.step()
    glide(sc, r, (0.0, 0.0), 0.89 + 0.10)
    glide(sc, r, sc.tube[:2], sc.tube[2] + 0.03)
    sc.press = 0.03
    r.step()
    sc.press = 0.0
    z = sc.pot[2] + 0.5 * (PLATE_REST_DZ[0] + PLATE_REST_DZ[1])
    glide(sc, r, sc.pot[:2], z, gripping=True)
    info = hold_still(sc, r, REST_STEPS + 5)[1]
    assert info["at_rest"] and info["supported"]
    assert info["grasped"] is True
    assert info["stages"]["success"] is False


def test_a_single_zero_crossing_cannot_satisfy_rest():
    """Rest requires a multi-step window, not one motionless frame."""
    sc = FakeScene()
    r = rw(sc)
    sc.grip()
    r.step()
    for i in range(REST_STEPS + 4):
        # Alternate moving and still: the still counter must keep resetting.
        z = 1.2 + (0.05 if i % 2 else 0.0)
        sc.place(sc.pot[:2], z, gripping=False)
        _, info = r.step()
        assert info["still_steps"] < REST_STEPS


def test_the_full_functional_sequence_succeeds():
    """Grasp, lift, align, dispense, place on the plate, release, settle."""
    sc = FakeScene()
    r = rw(sc)
    sc.grip()
    r.step()
    glide(sc, r, (0.0, 0.0), 0.89 + 0.10)
    glide(sc, r, sc.tube[:2], sc.tube[2] + 0.03)
    sc.press = 0.03
    _, info = r.step()
    assert info["stages"]["dispense"] is True
    sc.press = 0.0
    z = sc.pot[2] + 0.5 * (PLATE_REST_DZ[0] + PLATE_REST_DZ[1])
    glide(sc, r, sc.pot[:2], z, gripping=True)
    sc.release()                                    # let go
    _, info = hold_still(sc, r, REST_STEPS + 5)
    assert info["supported"] and info["at_rest"] and not info["grasped"]
    assert info["stages"]["success"] is True


# ---------------------------------------------------------------- point 4 ----
def test_jerk_is_penalised_while_held_and_bounded():
    sc = FakeScene()
    r = rw(sc)
    sc.grip()
    r.step()
    glide(sc, r, (0.0, 0.0), 0.89 + 0.10)
    # Violent alternating motion while holding it.
    pen = []
    for i in range(12):
        sc.place((0.0, 0.0), 0.99 + (0.03 if i % 2 else -0.03), gripping=True)
        _, info = r.step()
        pen.append(info["jerk_penalty"])
    assert max(pen) > 0.0
    assert max(pen) <= JERK_PENALTY_CAP + 1e-12


def test_jerk_is_not_penalised_in_free_space():
    """Fast motion with nothing in hand must not be taxed."""
    sc = FakeScene()
    r = rw(sc)
    for i in range(10):
        sc.barrel = np.array([0.0, 0.0, 0.89])
        sc.fingers = np.array([[1.0 + 0.1 * i, 1.0, 1.0]] * 4)
        sc.thumb = np.array([1.0 + 0.1 * i, 1.02, 1.0])
        _, info = r.step()
        assert info["jerk_penalty"] == 0.0


# ---------------------------------------------------------------- general ----
def test_plunger_stages_are_retained_from_v2():
    assert "press" in stage_order() and "dispense" in stage_order()


def test_pumping_the_plunger_still_cannot_farm_reward():
    sc = FakeScene()
    r = rw(sc)
    sc.grip()
    r.step()
    sc.press = 0.03
    first, _ = r.step()
    total = 0.0
    for _ in range(20):
        sc.press = 0.0
        a, _ = r.step()
        sc.press = 0.03
        b, _ = r.step()
        total += a + b
    assert total < first
    assert total <= 0.0 + 1e-9


def test_reset_makes_episodes_independent():
    sc = FakeScene()
    r = rw(sc)
    sc.grip()
    sc.press = 0.035
    r.step()
    r.reset()
    sc.press = 0.0
    sc.release()
    _, info = r.step()
    assert not any(info["stages"].values())
    assert info["still_steps"] in (0, 1)


@pytest.mark.parametrize("dz,expected", [
    (0.02, False),   # sunk into / below the plate
    (0.13, True),    # resting on it
    (0.60, False),   # in the air above it
])
def test_supported_band_rejects_midair_and_sunken(dz, expected):
    sc = FakeScene()
    r = rw(sc)
    sc.place(sc.pot[:2], sc.pot[2] + dz, gripping=False)
    _, info = r.step()
    assert info["supported"] is expected


# ------------------------------------------------- self-audit exploit probes ----
def test_pressing_after_release_cannot_farm_reward():
    """Found by adversarial self-audit of v3, not by review.

    ``grasp`` is a latched stage, so gating the plunger term on the LATCHED stage let a
    policy grasp once, drop the pipette, and then drive the plunger down by any other
    means (pushing it against the table) while the term kept paying. Measured on the
    latched version: 18.0 reward farmed after releasing. The term must gate on the LIVE
    grasp instead.
    """
    sc = FakeScene()
    r = rw(sc)
    sc.grip()
    r.step()
    assert r._stages["grasp"] is True          # latched, as intended
    sc.release()
    r.step()
    total = 0.0
    for p in (0.01, 0.02, 0.03, 0.04):
        sc.press = p
        rew, info = r.step()
        total += rew
    assert info["grasped"] is False
    assert info["stages"]["press"] is False
    assert total <= 0.0, total


def test_dispense_requires_actually_holding_the_pipette():
    """A pipette resting in the tube with the plunger pushed by something else is not
    a dispense."""
    sc = FakeScene()
    r = rw(sc)
    sc.grip()
    r.step()
    glide(sc, r, (0.0, 0.0), 0.89 + 0.10)
    glide(sc, r, sc.tube[:2], sc.tube[2] + 0.03)
    sc.release()
    r.step()
    sc.press = 0.035
    _, info = r.step()
    assert info["stages"]["dispense"] is False


def test_press_still_credited_when_genuinely_held():
    """The gate must not break the legitimate path."""
    sc = FakeScene()
    r = rw(sc)
    sc.grip()
    r.step()
    sc.press = 0.03
    rew, info = r.step()
    assert info["stages"]["press"] is True
    assert rew > 0.0
