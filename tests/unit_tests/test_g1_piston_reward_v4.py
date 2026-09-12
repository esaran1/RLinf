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

"""v4 must pay for a press only while the pipette is held in the air.

Critic-free RL under v3 learned to press the plunger against the table while grasping,
never lifting, for a return above a full transport. Every certified press in runs 9b/9c
had no lift. v4 differs from v3 in exactly two gates; v3 stays the scorer of record.
"""

import importlib.util
import sys


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def test_v4_differs_from_v3_in_exactly_the_two_press_gates():
    v3 = open("rlinf/envs/isaaclab/tasks/g1_piston_reward_v3.py").read()
    v4 = open("rlinf/envs/isaaclab/tasks/g1_piston_reward_v4.py").read()
    assert "if grasped and lift > LIFT_H:" in v4 and "if grasped:\n            if self._best_press is None" in v3
    assert 'fire("press", grasped and lift > LIFT_H and press > PRESS_DEPTH)' in v4
    assert 'fire("press", grasped and press > PRESS_DEPTH)' in v3
    # thresholds and bonuses are untouched
    for tok in ("PRESS_FRAC = 0.70", "W_PRESS = 150.0", '"press": 12.0', '"dispense": 18.0', "LIFT_H = 0.05"):
        assert tok in v4 and tok in v3, tok
    assert "class PistonTaskRewardV4" in v4 and "PistonTaskRewardV3" not in v4.split("Original v3 documentation follows.")[1].split("class")[0]


def test_v4_module_loads_by_path_and_reports_its_version():
    m = _load("g1r4_t", "rlinf/envs/isaaclab/tasks/g1_piston_reward_v4.py")
    assert hasattr(m, "PistonTaskRewardV4")
    assert m.REWARD_VERSION == "v4_held_press"
    assert m.PRESS_DEPTH == 0.04 * 0.70


def test_table_press_exploit_is_on_the_record():
    import json
    c = json.load(open("docs/contracts/g1_piston_table_press_exploit.json"))
    assert all(p["lift"] is False for p in c["certified_presses"])
