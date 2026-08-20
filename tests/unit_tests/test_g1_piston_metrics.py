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

"""The behavioural metric definitions are frozen and must not drift.

Fixed before any RLPD evaluation existed. If a later change makes these tests fail,
the comparison is no longer measuring what it measured when the thresholds were set.
"""

import pytest

from rlinf.envs.isaaclab.tasks.g1_piston_metrics import (
    CARRY_MIN_DISPLACEMENT_M,
    DEMO_DISPLACEMENT_RANGE_M,
    FAILURE_METRICS,
    METRICS_VERSION,
    PRIMARY_METRICS,
    classify,
    is_carry,
    is_throw,
    wilson95,
)


def _row(lift, disp, success=False, ret=0.0, maxlift=0.0):
    return {
        "return": ret, "disp_m": disp, "max_lift_m": maxlift,
        "stages": {"reach": True, "grasp": True, "lift": lift,
                   "plate": False, "success": success},
    }


def test_threshold_is_frozen():
    """The threshold never moves. The version moved once, for a bugfix, not a retune."""
    assert CARRY_MIN_DISPLACEMENT_M == 0.05
    assert METRICS_VERSION == "v1.1-frozen-2026-08-17-horizontal-fix"


def test_carry_uses_horizontal_not_three_dimensional_displacement():
    """The bug this fixes: a purely vertical fling scored as a carry.

    ``disp_m`` was recorded as a 3-D norm, so height alone could clear the horizontal
    threshold -- the exploit counted as its own opposite.
    """
    from rlinf.envs.isaaclab.tasks.g1_piston_metrics import horizontal_disp_m

    # 0.30 m straight up, no horizontal travel whatsoever.
    fling = {"return": 6.0, "disp_m": 0.30, "disp_xy_m": 0.0, "max_lift_m": 0.30,
             "stages": {"reach": True, "grasp": True, "lift": True,
                        "plate": False, "success": False}}
    assert horizontal_disp_m(fling) == 0.0
    assert is_throw(fling), "a vertical fling must be a throw, never a carry"
    assert not is_carry(fling)

    # Same 3-D magnitude, but genuinely horizontal.
    slide = dict(fling, disp_xy_m=0.30, max_lift_m=0.06)
    assert is_carry(slide)
    assert not is_throw(slide)


def test_horizontal_disp_is_recovered_from_piston_geometry():
    """Rows predating disp_xy_m are corrected from the recorded poses."""
    from rlinf.envs.isaaclab.tasks.g1_piston_metrics import (
        disp_is_3d_fallback,
        horizontal_disp_m,
    )

    row = {"disp_m": 0.5, "max_lift_m": 0.4,
           "piston_initial_xyz": [0.0, 0.0, 0.9],
           "piston_final_xyz": [0.03, 0.04, 1.3],   # 0.05 horizontal, 0.40 vertical
           "stages": {"lift": True}}
    assert horizontal_disp_m(row) == pytest.approx(0.05, abs=1e-9)
    assert not disp_is_3d_fallback(row)
    # Exactly at the threshold, so it remains a carry.
    assert is_carry(row)

    # A row with neither xy nor geometry falls back to the 3-D value and says so.
    bare = {"disp_m": 0.5, "max_lift_m": 0.4, "stages": {"lift": True}}
    assert horizontal_disp_m(bare) == 0.5
    assert disp_is_3d_fallback(bare)


def test_a_non_lifting_episode_is_neither_carry_nor_throw():
    """Both labels are defined only among lifting episodes."""
    row = {"disp_m": 0.9, "disp_xy_m": 0.9, "max_lift_m": 0.0,
           "stages": {"reach": True, "grasp": True, "lift": False}}
    assert not is_carry(row)
    assert not is_throw(row)


def test_threshold_is_permissive_against_every_demonstration():
    """A carry threshold must not be so strict that real demonstrations fail it."""
    assert CARRY_MIN_DISPLACEMENT_M < DEMO_DISPLACEMENT_RANGE_M[0]
    # Deliberately 4x below the least-transporting demonstration.
    assert DEMO_DISPLACEMENT_RANGE_M[0] / CARRY_MIN_DISPLACEMENT_M >= 4.0


def test_carry_and_throw_partition_the_lifting_episodes():
    lifting = [_row(True, 0.0), _row(True, 0.049), _row(True, 0.05), _row(True, 0.3)]
    assert sum(is_carry(r) for r in lifting) + sum(is_throw(r) for r in lifting) == 4
    assert not any(is_carry(r) and is_throw(r) for r in lifting)


def test_non_lifting_episodes_are_neither():
    """A large horizontal push without a lift is not a carry."""
    r = _row(False, 0.4)
    assert not is_carry(r) and not is_throw(r)


def test_the_measured_sac_exploit_is_classified_as_a_throw():
    """The 552k rollouts: 0.50 m of lift with under 0.01 m of transport."""
    assert is_throw(_row(True, 0.0100, maxlift=0.497))
    assert is_throw(_row(True, 0.0081, maxlift=0.477))


def test_a_demonstration_like_rollout_is_classified_as_a_carry():
    assert is_carry(_row(True, 0.296, maxlift=0.202))
    assert is_carry(_row(True, DEMO_DISPLACEMENT_RANGE_M[0], maxlift=0.129))


def test_boundary_is_inclusive_for_carry():
    assert is_carry(_row(True, CARRY_MIN_DISPLACEMENT_M))
    assert is_throw(_row(True, CARRY_MIN_DISPLACEMENT_M - 1e-9))


def test_classify_reproduces_the_measured_552k_checkpoint():
    """6 lifting conditions: 3 carries, 3 throws, out of 25."""
    rows = ([_row(True, 0.25) for _ in range(3)]
            + [_row(True, 0.009) for _ in range(3)]
            + [_row(False, 0.01) for _ in range(19)])
    m = classify(rows)
    assert m["n_eval_episodes"] == 25
    assert m["lift_rate"] == pytest.approx(6 / 25)
    assert m["carry_rate"] == pytest.approx(3 / 25)
    assert m["throw_rate"] == pytest.approx(3 / 25)
    # lift_rate overstates progress: carry is half of it here
    assert m["carry_rate"] < m["lift_rate"]


def test_classify_reproduces_the_measured_414k_checkpoint():
    """10 lifting conditions: 9 carries, 1 throw -- mostly genuine."""
    rows = ([_row(True, 0.15) for _ in range(9)]
            + [_row(True, 0.01)]
            + [_row(False, 0.01) for _ in range(15)])
    m = classify(rows)
    assert m["lift_rate"] == pytest.approx(10 / 25)
    assert m["carry_rate"] == pytest.approx(9 / 25)
    assert m["throw_rate"] == pytest.approx(1 / 25)


def test_metric_roles_are_declared():
    assert "full_success_rate" in PRIMARY_METRICS
    assert "carry_rate" in PRIMARY_METRICS
    assert "mean_return" in PRIMARY_METRICS
    assert "throw_rate" in FAILURE_METRICS
    # lift_rate is explicitly NOT primary any more
    assert "lift_rate" not in PRIMARY_METRICS


def test_classify_emits_every_reported_field():
    m = classify([_row(True, 0.2, success=True, ret=35.4, maxlift=0.13)])
    for k in ("full_success_rate", "reach_rate", "grasp_rate", "lift_rate",
              "carry_rate", "throw_rate", "mean_disp_m", "max_lift_m", "mean_return"):
        assert k in m, k


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson95(3, 25)
    assert lo <= 3 / 25 <= hi
    assert wilson95(0, 25)[0] == 0.0
    assert wilson95(25, 25)[1] == 1.0
