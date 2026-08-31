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


def should_refuse(reward_v2: bool, built_v2: bool) -> bool:
    """The trainer refuses exactly when the two versions disagree."""
    return reward_v2 != built_v2


@pytest.mark.parametrize("reward_v2,built_v2,refuse", [
    (True, False, True),    # v2 training against the shipped v1 buffer -- the bug
    (False, True, True),    # v1 training against a v2 buffer -- also wrong
    (True, True, False),    # matched v2
    (False, False, False),  # matched v1 (the default path)
])
def test_version_mismatch_is_refused(reward_v2, built_v2, refuse):
    assert should_refuse(reward_v2, built_v2) is refuse


def test_marker_absent_reads_as_v1(tmp_path):
    """The shipped buffer predates the marker, so 'no marker' must mean v1."""
    assert buffer_is_v2(str(tmp_path / "missing.json")) is False


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
    assert "REWARD_V2 and not _built_v2" in src
    assert "_built_v2 and not REWARD_V2" in src


def test_demo_dir_is_configurable():
    """A v2 run must be able to point at a different buffer directory."""
    src = open("tools/g1_piston/train_sac.py").read()
    assert 'DEMO_DIR = os.environ.get("DEMO_DIR"' in src
