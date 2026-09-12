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


# ------------------------------------------------------------ functional gate ----
import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
from tests.unit_tests.test_g1_piston_reward_v3 import FakeScene  # noqa: E402


def _v4(sc):
    m = _load("g1r4_f", "rlinf/envs/isaaclab/tasks/g1_piston_reward_v4.py")
    return m.PistonTaskRewardV4(sc, dt=0.02), m


def _v3(sc):
    m = _load("g1r3_f", "rlinf/envs/isaaclab/tasks/g1_piston_reward_v3.py")
    return m.PistonTaskRewardV3(sc, dt=0.02)


def _press_sequence(sc, rf, lift_dz):
    """Grasp, optionally raise the barrel by lift_dz, then depress the plunger to 35 mm
    over several steps (keeping the rod/fingers/thumb attached). Returns (total reward,
    stages)."""
    rf.reset(); rf.step()                       # rest pose registered
    sc.grip(); rf.step()
    total = 0.0
    if lift_dz:
        for k in range(1, 6):                    # raise slowly (below the ballistic limit)
            dz = lift_dz * k / 5
            sc.barrel = np.array([0.0, 0.0, 0.89 + dz]); sc.rod = sc.barrel + np.array([0, 0, 0.03])
            sc.grip(); r, _ = rf.step(); total += r
    for p in (0.010, 0.020, 0.030, 0.035):
        sc.press = p; sc.grip(); r, info = rf.step(); total += r
    return total, info["stages"]


def test_v4_pays_nothing_for_a_grasped_table_press():
    """The exploit: plunger pressed to 35 mm while grasped but never lifted."""
    sc = FakeScene(); rf4, _ = _v4(sc)
    r4, st4 = _press_sequence(sc, rf4, lift_dz=0.0)
    assert st4["press"] is False
    sc3 = FakeScene(); rf3 = _v3(sc3)
    r3, st3 = _press_sequence(sc3, rf3, lift_dz=0.0)
    assert st3["press"] is True                 # v3 paid for it -- that is the exploit
    assert r3 - r4 > 10.0                        # the 12.0 bonus plus the dense term


def test_v4_pays_for_a_press_while_lifted():
    sc = FakeScene(); rf4, m = _v4(sc)
    r_lift, st = _press_sequence(sc, rf4, lift_dz=0.12)
    assert st["lift"] is True and st["press"] is True
    sc2 = FakeScene(); rf4b, _ = _v4(sc2)
    r_table, _ = _press_sequence(sc2, rf4b, lift_dz=0.0)
    assert r_lift > r_table + 10.0
