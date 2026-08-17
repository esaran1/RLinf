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

"""The evaluation suite must be varied, reproducible, disjoint from train, and safe.

Guards ``docs/contracts/g1_piston_eval_suite_defect.json``: the upstream task has one
initial state, so a seed-indexed suite silently had an effective sample size of 1.
"""

import pytest

from rlinf.envs.isaaclab.tasks.g1_piston_reset import (
    ARM_JITTER_RAD,
    BARREL_RADIUS,
    CANONICAL,
    PISTON_JITTER_M,
    RIGHT_ARM_JOINTS,
    SOCKET_CLEARANCE,
    WAIST_JITTER_RAD,
    WAIST_JOINTS,
    build_reset_suite,
    suite_manifest,
)


def test_eval_suite_is_large_enough_to_report_rates():
    _, ev = build_reset_suite()
    assert len(ev) >= 25, "a rate needs at least 25 distinct conditions to be meaningful"
    assert len(ev) == 50


def test_train_and_eval_conditions_are_disjoint():
    train, ev = build_reset_suite()
    assert not ({c.hash() for c in train} & {c.hash() for c in ev})


def test_conditions_within_a_split_are_distinct():
    train, ev = build_reset_suite()
    assert len({c.hash() for c in ev}) == len(ev)
    assert len({c.hash() for c in train}) == len(train)


def test_suite_is_reproducible_from_the_seed():
    a_train, a_eval = build_reset_suite()
    b_train, b_eval = build_reset_suite()
    assert [c.hash() for c in a_eval] == [c.hash() for c in b_eval]
    assert [c.hash() for c in a_train] == [c.hash() for c in b_train]


def test_a_different_seed_gives_a_different_suite():
    _, a_eval = build_reset_suite(seed=1)
    _, b_eval = build_reset_suite(seed=2)
    assert [c.hash() for c in a_eval] != [c.hash() for c in b_eval]


def test_piston_jitter_cannot_reach_a_socket_wall():
    """The socket has 2 mm of radial clearance; a reset must never spawn in contact."""
    assert SOCKET_CLEARANCE == pytest.approx(0.002, abs=1e-9)
    assert PISTON_JITTER_M < SOCKET_CLEARANCE
    # Worst case is a diagonal draw on both axes.
    worst = (2 * PISTON_JITTER_M**2) ** 0.5
    assert worst < SOCKET_CLEARANCE, (worst, SOCKET_CLEARANCE)


def test_piston_jitter_is_far_smaller_than_the_upstream_range_that_jammed_it():
    """Upstream zeroed a +/-0.05 m randomisation because it jammed the barrel."""
    assert PISTON_JITTER_M < 0.05 / 20


def test_perturbed_joints_are_the_reach_relevant_ones():
    _, ev = build_reset_suite()
    perturbed = set(ev[0].joint_delta)
    assert perturbed == set(RIGHT_ARM_JOINTS) | set(WAIST_JOINTS)
    # The left hand grips the tube and its dims are frozen in the action space; it must
    # not be perturbed at reset either.
    assert not any(j.startswith("left_") or j.startswith("L_") for j in perturbed)


def test_joint_perturbations_respect_the_declared_bounds():
    _, ev = build_reset_suite()
    for c in ev:
        for j, d in c.joint_delta.items():
            bound = WAIST_JITTER_RAD if "waist" in j else ARM_JITTER_RAD
            assert abs(d) <= bound + 1e-12, (j, d)
        for v in c.piston_dxy:
            assert abs(v) <= PISTON_JITTER_M + 1e-12


def test_canonical_condition_is_unperturbed():
    """Kept as a named diagnostic; its outcome is one binary observation, not a rate."""
    assert CANONICAL.joint_delta == {}
    assert CANONICAL.piston_dxy == (0.0, 0.0)
    assert CANONICAL.split == "canonical"


def test_state_vector_and_hash_track_the_condition():
    _, ev = build_reset_suite()
    assert ev[0].state_vector().shape == (len(RIGHT_ARM_JOINTS) + len(WAIST_JOINTS) + 2,)
    assert ev[0].hash() != ev[1].hash()
    assert ev[0].hash() == ev[0].hash()


def test_manifest_records_what_reviewers_need():
    train, ev = build_reset_suite()
    m = suite_manifest(train, ev)
    assert m["disjoint"] is True
    assert m["n_eval"] == len(ev)
    assert len(m["eval_hashes"]) == len(ev)
    assert m["piston_jitter_m"] < m["socket_clearance_m"]


def test_barrel_radius_and_clearance_are_consistent():
    assert BARREL_RADIUS == 0.020
    assert SOCKET_CLEARANCE > 0
