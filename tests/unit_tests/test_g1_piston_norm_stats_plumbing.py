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

"""A checkpoint's normaliser statistics must travel with it through every tool.

The dispense primitive moves the left arm by up to 2.2 rad; the SFT statistics give those
dims a 0.1-0.2 rad range, so the squashed head could not express the motion (normalised
-24..+19 against +/-2.2). The pressing policy therefore trains against a widened,
versioned statistics file, recorded in its checkpoint as ``norm_stats``.
"""


def _src(name):
    return open(f"tools/g1_piston/{name}").read()


def test_trainer_records_and_tools_honour_norm_stats():
    bc = _src("train_bc.py")
    assert 'os.environ.get("NORM_STATS", "")' in bc and '"norm_stats": STATS' in bc
    for tool in ("eval_checkpoint.py", "render_rollouts.py"):
        s = _src(tool)
        assert '.get("norm_stats", "")' in s, tool
        # the early read is the ONLY checkpoint load: no second, possibly different, load
        assert s.count("torch.load(CKPT_PATH") == 1, tool
    g = _src("train_grpo.py")
    assert '_bc_early.get("norm_stats", "")' in g and '"norm_stats": STATS' in g


def test_buffer_builder_widens_only_the_left_arm_dims():
    s = _src("build_press_bc_buffer.py")
    assert "WIDEN_DIMS = list(range(0, 7))" in s
    assert "PLATE_CUM_RETURN = 14.0" in s
    # widening can only grow the range, never shrink it
    assert "if dmin < q01[d] - EXCEED_TOL else q01[d]" in s
