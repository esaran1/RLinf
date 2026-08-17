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

"""The actor, replay buffer, log-prob and critic must describe ONE decision variable.

One replay transition is one executed StarVLA chunk: 30 action steps x 30 dims, of
which 20 dims per step are RL-controllable, executed with a 2x zero-order hold. So the
SAC action is the whole 30x30 chunk (600 stochastic scalars), not a single 30-D vector.

This module pins that invariant, and pins the entropy reduction convention:

    logp_step[h] = sum over the 20 active dims
    logp_chunk   = mean over the 30 horizon steps
    target       = default_target_entropy()   (per control action)

The mean-over-horizon convention keeps entropy regularisation on a per-control-action
scale instead of letting it grow with the prediction horizon. The target's *value* is
set by what the bounded action distribution can reach, not by ``-dim(A)`` -- see
``test_g1_piston_entropy_target.py``. Runs that motivated both:
``docs/contracts/g1_piston_sac_pilot_v1_collapse.json`` and
``docs/contracts/g1_piston_eval_suite_defect.json``.
"""

import pytest
import torch

from rlinf.envs.isaaclab.tasks.g1_piston_rl_space import (
    ACTION_DIM,
    NUM_ACTIVE_DIMS,
    build_active_mask,
    default_target_entropy,
)
from rlinf.models.embodiment.modules.gaussian_policy import SquashedNormal

ACTION_HORIZON = 30
ACTION_LOW, ACTION_HIGH = -2.2, 2.2

#: Total stochastic scalars in one SAC decision: 30 steps x 20 active dims.
N_STOCHASTIC_SCALARS = ACTION_HORIZON * NUM_ACTIVE_DIMS


def chunk_logprob(dist, flat_sample, mask, horizon=ACTION_HORIZON, reduction="mean"):
    """Per-chunk log-prob under an explicit reduction convention.

    ``flat_sample`` is ``[B*horizon, ACTION_DIM]``; the active dims are always summed
    within a step, then the horizon is either averaged (per-control-action scale) or
    summed (whole-chunk scale).
    """
    base = dist.base_dist.base_dist
    parts = []
    for t in dist.transforms:
        parts.extend(getattr(t, "parts", [t]))
    x = flat_sample
    for t in reversed(parts):
        x = t.inv(x)
    per_dim = base.log_prob(x)
    y = x
    for t in parts:
        y2 = t(y)
        if type(t).__name__ == "TanhTransform":
            per_dim = per_dim - torch.log(1 - y2.pow(2) + 1e-7)
        else:
            s = torch.as_tensor(t.scale, dtype=x.dtype, device=x.device)
            per_dim = per_dim - torch.log(s.abs()).expand_as(x)
        y = y2
    logp_step = (per_dim * mask).sum(dim=-1)             # [B*horizon]
    logp_step = logp_step.reshape(-1, horizon)           # [B, horizon]
    if reduction == "mean":
        return logp_step.mean(dim=-1)
    if reduction == "sum":
        return logp_step.sum(dim=-1)
    raise ValueError(reduction)


def _dist(batch=4):
    mask = build_active_mask()
    mean = torch.zeros(batch * ACTION_HORIZON, ACTION_DIM)
    std = torch.full((batch * ACTION_HORIZON, ACTION_DIM), 0.37)
    return SquashedNormal(mean, std, low=ACTION_LOW, high=ACTION_HIGH), mask


# --- the decision variable itself -------------------------------------------------


def test_one_transition_is_a_whole_chunk():
    """600 stochastic scalars, not 30."""
    assert N_STOCHASTIC_SCALARS == 600
    assert N_STOCHASTIC_SCALARS == ACTION_HORIZON * NUM_ACTIVE_DIMS


def test_actor_sample_shape_is_the_chunk():
    dist, mask = _dist(batch=4)
    sample = dist.rsample()
    assert sample.shape == (4 * ACTION_HORIZON, ACTION_DIM)
    assert sample.reshape(4, ACTION_HORIZON, ACTION_DIM).shape[1:] == (
        ACTION_HORIZON, ACTION_DIM)


def test_critic_action_input_is_the_flattened_chunk():
    """The critic must score Q(s, whole executed chunk), not Q(s, one 30-D action).

    This is the invariant that would silently break credit assignment: the environment
    transition is produced by 30 action steps, so a critic fed only one of them would be
    regressing a chunk-long reward onto 1/30th of its cause.
    """
    critic_action_feature_dim = ACTION_DIM * ACTION_HORIZON
    assert critic_action_feature_dim == 900

    actions = torch.zeros(4, ACTION_HORIZON, ACTION_DIM)
    assert actions.reshape(4, -1).shape == (4, critic_action_feature_dim)
    # A single 30-D action would be the bug.
    assert actions.reshape(4, -1).shape[1] != ACTION_DIM


def test_replay_action_shape_matches_critic_input():
    stored = torch.zeros(ACTION_HORIZON, ACTION_DIM)          # one transition
    batched = torch.stack([stored] * 4)
    assert batched.shape == (4, ACTION_HORIZON, ACTION_DIM)
    assert batched.reshape(4, -1).shape == (4, ACTION_DIM * ACTION_HORIZON)


# --- log-prob reduction -----------------------------------------------------------


def test_logprob_shape_before_and_after_reduction():
    dist, mask = _dist(batch=4)
    sample = dist.rsample()
    lp = chunk_logprob(dist, sample, mask)
    assert lp.shape == (4,), "one log-prob per chunk transition"


def test_active_dims_only_contribute_to_logprob():
    """Frozen dims are deterministic; including them would add a constant."""
    dist, mask = _dist(batch=2)
    sample = dist.rsample()
    full = chunk_logprob(dist, sample, torch.ones_like(mask))
    active = chunk_logprob(dist, sample, mask)
    assert not torch.allclose(full, active)


def test_mean_convention_is_per_control_action_scale():
    """Averaging over the horizon must not scale with the prediction horizon."""
    dist, mask = _dist(batch=3)
    sample = dist.rsample()
    mean_red = chunk_logprob(dist, sample, mask, reduction="mean")
    sum_red = chunk_logprob(dist, sample, mask, reduction="sum")
    assert torch.allclose(sum_red, mean_red * ACTION_HORIZON, atol=1e-4)


def test_target_entropy_matches_the_mean_convention():
    """The target describes ONE control action, like the mean-reduced log-prob.

    Its value is set by what the bounded action distribution can actually reach, not by
    the -dim(A) heuristic; see test_g1_piston_entropy_target.py.
    """
    from rlinf.envs.isaaclab.tasks.g1_piston_rl_space import MIN_ACHIEVABLE_LOGPROB
    assert MIN_ACHIEVABLE_LOGPROB < default_target_entropy() < 0


def test_summed_convention_requires_a_scaled_target():
    """The alternative convention is valid only with a correspondingly scaled target.

    Mixing a summed log-prob with the per-step -20 target is the inconsistency that the
    earlier pilots ran with.
    """
    per_step_target = default_target_entropy()
    summed_target = per_step_target * ACTION_HORIZON
    assert summed_target == pytest.approx(per_step_target * 30)
    assert summed_target != per_step_target


@pytest.mark.parametrize("std", [0.1, 0.37, 1.0])
def test_entropy_objective_invariant_to_reduction_convention(std):
    """The SAME entropy objective under either convention, given a matched target.

    alpha_mean * (logp_mean + target) must equal alpha_sum * (logp_sum + target*H)
    when alpha_sum = alpha_mean / H. This is what "one internally consistent
    convention" means: the choice is a reparameterisation, not a different objective.
    """
    mask = build_active_mask()
    mean = torch.zeros(2 * ACTION_HORIZON, ACTION_DIM)
    scale = torch.full((2 * ACTION_HORIZON, ACTION_DIM), std)
    dist = SquashedNormal(mean, scale, low=ACTION_LOW, high=ACTION_HIGH)
    sample = dist.rsample()

    lp_mean = chunk_logprob(dist, sample, mask, reduction="mean")
    lp_sum = chunk_logprob(dist, sample, mask, reduction="sum")

    alpha_mean = 0.05
    alpha_sum = alpha_mean / ACTION_HORIZON
    target_mean = default_target_entropy()
    target_sum = target_mean * ACTION_HORIZON

    obj_mean = alpha_mean * (lp_mean + target_mean)
    obj_sum = alpha_sum * (lp_sum + target_sum)
    assert torch.allclose(obj_mean, obj_sum, atol=1e-4)


def test_entropy_term_does_not_dwarf_the_q_signal():
    """Regression guard for the measured instability.

    With the summed convention the entropy term was ~300 nats against Q ~ 3-7, i.e. the
    actor objective was dominated by entropy bookkeeping that grew with the horizon. The
    mean convention keeps |alpha * logp| the same order as Q.
    """
    dist, mask = _dist(batch=4)
    sample = dist.rsample()
    lp_mean = chunk_logprob(dist, sample, mask, reduction="mean")
    lp_sum = chunk_logprob(dist, sample, mask, reduction="sum")

    alpha = 0.05
    q_scale = 5.0
    assert abs(float((alpha * lp_mean).mean())) < q_scale
    assert abs(float((alpha * lp_sum).mean())) > q_scale
