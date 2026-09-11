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

"""The H10 probe must work on any ensemble size and apply a residual if present."""

import re


def _src():
    with open("tools/g1_piston/probe_critic_ranking.py") as f:
        return f.read()


def test_ensemble_size_is_inferred_not_hardcoded():
    src = _src()
    assert "num_q_heads=NQ" in src and "num_q_heads=2" not in src
    assert "def infer_num_q(sd)" in src


def test_infer_num_q_logic():
    """Replicates the helper on a synthetic state dict."""
    def infer_num_q(sd):
        return 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("qs."))
    sd = {f"qs.{i}.net.0.weight": None for i in range(10)}
    sd["other"] = None
    assert infer_num_q(sd) == 10
    assert infer_num_q({"qs.0.x": None, "qs.1.x": None}) == 2


def test_h10_criteria_are_the_registered_ones():
    src = _src()
    assert '"H10_sensitivity":frac_bc_over_pert>=0.70' in src
    assert '"H10_gap":gap>0.10' in src


def test_probe_applies_a_residual_and_uses_the_run_aggregation():
    src = _src()
    assert 'if "residual" in R6:' in src and "residual.act(AQ,a_pol,None,am,deterministic=True)" in src
    assert 'qa.mean(dim=1) if AGG=="mean" else qa.min(dim=1)[0]' in src
    assert "os.environ[\"CKPT\"]" in src and "rl6/rlpd_ckpt_step60840" not in src
