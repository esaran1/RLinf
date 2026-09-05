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

"""The residual arm must be wired identically in the trainer and in every scorer.

Every train/deploy mismatch in this project had matching tensor shapes and no error.
So the composition the trainer optimises (policy_action) and the composition the
evaluation and rendering tools execute (sample_action with a residual) are replicated
here against the shared module, and the trainer's direct arm is required to be
byte-for-byte unaffected when RESIDUAL is off.
"""

import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")

TRAINER = "tools/g1_piston/train_sac.py"
SCORERS = ("tools/g1_piston/eval_checkpoint.py", "tools/g1_piston/render_rollouts.py",
           "tools/g1_piston/measure_deployed_action_error.py")


def _src(p):
    with open(p) as f:
        return f.read()


def _mods():
    def load(n, p):
        spec = importlib.util.spec_from_file_location(n, p)
        m = importlib.util.module_from_spec(spec); sys.modules[n] = m
        spec.loader.exec_module(m); return m
    rlsp = load("g1s_rw", "rlinf/envs/isaaclab/tasks/g1_piston_rl_space.py")
    resp = load("g1rp_rw", "rlinf/envs/isaaclab/tasks/g1_piston_residual_policy.py")
    return rlsp, resp


def test_trainer_routes_every_action_through_policy_action():
    src = _src(TRAINER)
    assert src.count("policy_action(") == 5, src.count("policy_action(")
    # No remaining direct sample_action call sites outside policy_action itself.
    block = src.split("def policy_action(")[1]
    assert "sample_action(mean, logstd, deterministic)" in block
    assert "= sample_action(" not in src.split("def policy_action(")[1].split("def act_to_command")[1]


def test_base_head_is_frozen_only_under_the_residual_arm():
    src = _src(TRAINER)
    assert "p.requires_grad_(not RESIDUAL)" in src
    assert 'RESIDUAL = os.environ.get("RESIDUAL", "0") == "1"' in src


def test_residual_arm_trains_the_residual_not_the_head():
    src = _src(TRAINER)
    assert "torch.optim.Adam(list(res_actor.parameters()) + [actor_logstd]" in src
    assert "(list(res_actor.parameters()) if RESIDUAL" in src
    assert 'res["oft_unchanged_required"] = bool(RESIDUAL)' in src


def test_residual_checkpoint_carries_what_the_scorers_need():
    src = _src(TRAINER)
    assert '"residual": res_actor.state_dict()' in src
    assert '"residual_cfg": {"feat_dim": HID, "hidden": list(RES_HIDDEN)' in src
    for p in SCORERS:
        s = _src(p)
        assert 'if "residual" in ck' in s, p
        assert "residual_cfg" in s, p


def test_scorers_apply_the_residual_on_the_deterministic_base():
    for p in SCORERS[:2]:
        s = _src(p)
        assert "residual = None" in s
        assert "residual.act(aq, a_base, None if deterministic else actor_logstd" in s, p
        assert "mean, aq = vlm_encode(img)" in s, p


def test_trainer_and_scorer_compositions_agree():
    """Executable replication: the trainer's policy_action and the scorer's
    sample_action must produce the SAME deterministic chunk from the same inputs."""
    rlsp, resp = _mods()
    torch.manual_seed(0)
    mask = rlsp.build_active_mask()
    frozen = torch.zeros(30)
    LOW, HIGH = -2.2, 2.2
    res = resp.ResidualPolicy(feat_dim=64, hidden=(32, 32), r_max=0.15)
    torch.nn.init.normal_(res.out.weight, std=0.5)
    aq = torch.randn(2, 30, 64); mean = torch.randn(2, 30, 30)
    # trainer: policy_action(deterministic)
    b, c, d = mean.shape
    flat = mean.reshape(b * c, d)
    a_base = torch.tanh(flat) * (HIGH - LOW) / 2 + (HIGH + LOW) / 2
    a_base = torch.where(mask, a_base, frozen.expand_as(a_base)).reshape(b, c, d)
    a_tr, _, _ = res.act(aq, a_base, None, mask, deterministic=True)
    a_tr = torch.where(mask, a_tr, frozen.expand_as(a_tr))
    # scorer: sample_action(deterministic, aq)
    a_base2 = torch.tanh(flat) * (HIGH - LOW) / 2 + (HIGH + LOW) / 2
    a_base2 = torch.where(mask, a_base2, frozen.expand_as(a_base2)).reshape(b, c, d)
    a_sc, _, _ = res.act(aq, a_base2, None, mask, deterministic=True)
    a_sc = torch.where(mask, a_sc, frozen.expand_as(a_sc))
    assert torch.equal(a_tr, a_sc)
    # and with a ZERO residual both equal the base exactly
    res0 = resp.ResidualPolicy(feat_dim=64, hidden=(32, 32), r_max=0.15)
    a0, _, _ = res0.act(aq, a_base, None, mask, deterministic=True)
    assert torch.equal(torch.where(mask, a0, frozen.expand_as(a0)), a_base)


def test_residual_target_is_the_closed_form_at_the_init_std():
    src = _src(TRAINER)
    assert "TARGET_ENTROPY = RESP.latent_target_entropy(RES_INIT_STD, N_ACTIVE)" in src
    _, resp = _mods()
    import math
    t = resp.latent_target_entropy(0.15, 20)
    assert t == pytest.approx((-0.5 - 0.5 * math.log(2 * math.pi) - math.log(0.15)) * 6 * 20 / 30)


def test_smoothness_penalty_targets_the_composed_chunk_under_the_residual_arm():
    src = _src(TRAINER)
    assert 'det = (RES_LAST["det"] if RESIDUAL else' in src
    assert 'RES_LAST["det"] = res_actor.compose(a_base, c_mean, ACT_MASK)' in src
