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

"""The GRPO trainer must compose actions exactly as the scorers do, and stay critic-free."""


def _src():
    with open("tools/g1_piston/train_grpo.py") as f:
        return f.read()


def test_base_head_is_frozen_and_loaded_from_the_bc_checkpoint():
    src = _src()
    assert 'model.action_model.load_state_dict(bc["action_model"])' in src
    assert "for p in model.parameters(): p.requires_grad_(False)" in src
    assert "opt = torch.optim.Adam(residual.parameters(), lr=LR)" in src


def test_action_composition_matches_the_scorers():
    """Deterministic base = squash + frozen dims; residual composes on top; frozen dims
    re-masked -- the same three steps eval_checkpoint.py and render_rollouts.py apply."""
    src = _src()
    assert "torch.tanh(mean.reshape(b * c, d)) * (ACTION_HIGH - ACTION_LOW) / 2 + (ACTION_HIGH + ACTION_LOW) / 2" in src
    assert "torch.where(ACT_MASK, a, FROZEN_V.expand_as(a))" in src
    assert "residual.compose(a_base, c, ACT_MASK)" in src


def test_no_critic_anywhere():
    src = _src()
    for forbidden in ("MultiQHead", "critic(", "EntropyTemperature", "target("):
        assert forbidden not in src, forbidden


def test_objective_is_the_tested_module():
    src = _src()
    for fn in ("GRPO.returns_to_go(", "GRPO.group_advantages(", "GRPO.gaussian_logp_mean(",
               "GRPO.ppo_clipped_loss(", "GRPO.kl_to_base_mean("):
        assert fn in src, fn
    assert "if np.mean(kls) > KL_STOP" in src


def test_checkpoint_format_is_loadable_by_the_scorers():
    src = _src()
    for key in ('"action_model": model.action_model.state_dict()', '"residual": residual.state_dict()',
                '"residual_cfg"', '"actor_logstd": torch.log(SIG.detach().cpu())'):
        assert key in src, key


def test_best_checkpoint_is_selected_by_deterministic_evaluation():
    src = _src()
    assert 'if ev["mean_return"] > best["mean_return"]:' in src
    assert 'save_ckpt("grpo_ckpt_best.pt", it, n_updates)' in src


def test_continuation_loads_only_the_residual():
    src = _src()
    assert 'residual.load_state_dict(_prev["residual"])' in src
    assert '_prev["action_model"]' not in src
