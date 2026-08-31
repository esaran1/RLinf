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

"""RLPD must refuse a demonstration buffer scored under the wrong reward version.

The failure this guards against is silent and expensive: RLPD fills half of every
batch from a buffer whose rewards were computed at BUILD time. Training against the v2
functional reward with the v1 buffer teaches the critic that a trajectory which never
touches the plunger is worth full credit, cancelling the v2 press signal -- while the
run looks completely normal for hours.

The guard lives in ``tools/g1_piston/train_sac.py``. These tests pin the decision logic
and the on-disk marker contract it depends on, without importing the trainer (which
needs a GPU and the simulator).
"""

import json

import pytest


def buffer_is_v2(marker_path, read_json=json.load):
    """Mirror of the trainer's marker check (see train_sac.py, RLPD demo load)."""
    import os

    if not os.path.exists(marker_path):
        return False
    try:
        with open(marker_path) as f:
            return read_json(f).get("reward_version") == "v2_functional"
    except Exception:  # noqa: BLE001 - a corrupt marker must not read as v2
        return False


#: Normalisation the trainer applies: build_demo_buffer writes "v1" but older markers
#: may say "v1_transport".
_NORM = {"v1_transport": "v1"}


def wanted_version(reward_v2: bool, reward_v3: bool) -> str:
    """Mirror of the trainer's version selection (v3 takes precedence over v2)."""
    return ("v3_review_fixed" if reward_v3
            else "v2_functional" if reward_v2 else "v1")


def should_refuse(reward_v2: bool, reward_v3: bool, built: str) -> bool:
    """The trainer refuses whenever the built version differs from the trained one."""
    return _NORM.get(built, built) != wanted_version(reward_v2, reward_v3)


@pytest.mark.parametrize("reward_v2,reward_v3,built,refuse", [
    # the original bug: v2 training against the shipped v1 buffer
    (True, False, "v1", True),
    (False, False, "v2_functional", True),      # v1 training against a v2 buffer
    # the bug this audit found: a v2-only guard mishandles v3 in BOTH directions
    (False, True, "v3_review_fixed", False),    # matched v3 must be ALLOWED
    (False, True, "v1", True),                  # v3 training on a v1 buffer must refuse
    (False, True, "v2_functional", True),       # v3 training on a v2 buffer must refuse
    (False, False, "v3_review_fixed", True),    # v1 training on a v3 buffer must refuse
    (True, False, "v3_review_fixed", True),     # v2 training on a v3 buffer must refuse
    (True, True, "v3_review_fixed", False),     # v3 wins when both flags are set
    (True, False, "v2_functional", False),      # matched v2
    (False, False, "v1", False),                # matched v1 (the default path)
    (False, False, "v1_transport", False),      # legacy marker spelling
])
def test_version_mismatch_is_refused(reward_v2, reward_v3, built, refuse):
    assert should_refuse(reward_v2, reward_v3, built) is refuse


def test_marker_absent_reads_as_v1(tmp_path):
    """The shipped buffer predates the marker, so 'no marker' must mean v1."""
    assert buffer_is_v2(str(tmp_path / "missing.json")) is False


def test_trainer_guard_covers_every_version_not_just_v2():
    """Regression: the first guard hard-coded 'v2_functional', so a v3 run on a v3
    buffer was wrongly refused and a v3 buffer under v1 training passed silently."""
    src = open("tools/g1_piston/train_sac.py").read()
    assert "_want" in src and "v3_review_fixed" in src
    assert "_built_v2" not in src, "guard still hard-codes the v2-only check"


def test_marker_v2_is_detected(tmp_path):
    m = tmp_path / "_build_status.json"
    m.write_text(json.dumps({"_status": "OK", "reward_version": "v2_functional"}))
    assert buffer_is_v2(str(m)) is True


def test_marker_v1_is_detected(tmp_path):
    m = tmp_path / "_build_status.json"
    m.write_text(json.dumps({"_status": "OK", "reward_version": "v1"}))
    assert buffer_is_v2(str(m)) is False


def test_corrupt_marker_does_not_read_as_v2(tmp_path):
    """A truncated or unparseable marker must fail closed, not open."""
    m = tmp_path / "_build_status.json"
    m.write_text("{not json")
    assert buffer_is_v2(str(m)) is False


def test_trainer_contains_the_guard():
    """Pin that the guard is actually wired into the trainer, not just tested here."""
    src = open("tools/g1_piston/train_sac.py").read()
    assert "_build_status.json" in src
    assert "reward version mismatch" in src.lower()


def test_demo_dir_is_configurable():
    """A v2 run must be able to point at a different buffer directory."""
    src = open("tools/g1_piston/train_sac.py").read()
    assert 'DEMO_DIR = os.environ.get("DEMO_DIR"' in src
