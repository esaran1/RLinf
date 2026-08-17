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

"""The SAC entropy target must be REACHABLE by the bounded action distribution.

The usual ``-dim(A)`` heuristic assumes an unbounded Gaussian, whose log-density is
unbounded below. This action space is tanh-squashed and rescaled to ``[-2.2, 2.2]``, so
its log-density is bounded **below** (measured floor -13.68 at std ~ 0.40, rising again
for both smaller and larger std as the tanh Jacobian dominates).

``-dim(A) = -20`` therefore lies below the floor. With the standard alpha loss
``-alpha * (logp + target)``, an unreachable target makes the gradient positive for every
attainable policy, so alpha is driven monotonically to zero, the entropy regulariser
vanishes, and the actor drifts unregularised. Both SAC pilots did exactly this: alpha
0.049 -> 0.039 while behaviour oscillated between 0 and 1.0. See
``docs/contracts/g1_piston_sac_pilot_v1_collapse.json``.
"""

import pytest
import torch

from rlinf.envs.isaaclab.tasks.g1_piston_rl_space import (
    MIN_ACHIEVABLE_LOGPROB,
    NUM_ACTIVE_DIMS,
    TARGET_ENTROPY_STD,
    build_active_mask,
    default_target_entropy,
)
from rlinf.models.embodiment.modules.gaussian_policy import SquashedNormal

ACTION_DIM = 30
ACTION_LOW, ACTION_HIGH = -2.2, 2.2


def measured_logprob(std, n=4096, seed=0):
    """Mean per-control-action log-prob of the bounded policy at a given std."""
    torch.manual_seed(seed)
    mask = build_active_mask()
    mean = torch.zeros(n, ACTION_DIM)
    scale = torch.full((n, ACTION_DIM), float(std))
    dist = SquashedNormal(mean, scale, low=ACTION_LOW, high=ACTION_HIGH)
    x = dist.rsample()
    base = dist.base_dist.base_dist
    parts = []
    for t in dist.transforms:
        parts.extend(getattr(t, "parts", [t]))
    y = x
    for t in reversed(parts):
        y = t.inv(y)
    per_dim = base.log_prob(y)
    z = y
    for t in parts:
        z2 = t(z)
        if type(t).__name__ == "TanhTransform":
            per_dim = per_dim - torch.log(1 - z2.pow(2) + 1e-7)
        else:
            s = torch.as_tensor(t.scale, dtype=y.dtype)
            per_dim = per_dim - torch.log(s.abs()).expand_as(y)
        z = z2
    return float((per_dim * mask).sum(dim=-1).mean())


def alpha_gradient(logp, target):
    """d/d(alpha) of ``-alpha * (logp + target)``. Positive => alpha is pushed DOWN."""
    return -(logp + target)


def test_target_is_reachable():
    assert default_target_entropy() > MIN_ACHIEVABLE_LOGPROB


def test_dim_heuristic_would_be_unreachable():
    """The bug: -dim(A) sits below the floor of the squashed distribution."""
    assert -float(NUM_ACTIVE_DIMS) < MIN_ACHIEVABLE_LOGPROB


def test_measured_floor_matches_the_recorded_constant():
    best = min(measured_logprob(s) for s in (0.3, 0.35, 0.4, 0.45, 0.5))
    assert best == pytest.approx(MIN_ACHIEVABLE_LOGPROB, abs=0.6), best


def test_logprob_is_bounded_below_and_non_monotonic_in_std():
    """Small AND large std both raise log-prob; the minimum is interior."""
    low, mid, high = (measured_logprob(s) for s in (0.05, 0.40, 2.0))
    assert mid < low and mid < high
    assert mid >= MIN_ACHIEVABLE_LOGPROB - 0.6


def test_unreachable_target_collapses_alpha_over_the_operating_range():
    """With target -20 alpha is pushed DOWN everywhere a real policy lives.

    The -20 target is below the log-density floor, so no policy can ever satisfy it by
    adding entropy in the useful regime. (Very diffuse policies, std >~ 1.5, do push the
    log-prob back up above -20 via the tanh Jacobian, but that regime is far outside the
    q99 action range the SFT policy occupies and is never reached from initialisation --
    exploration collapses long before.)
    """
    wrong = -float(NUM_ACTIVE_DIMS)
    for std in (0.05, 0.1, 0.2, 0.37, 0.6, 1.0):
        assert alpha_gradient(measured_logprob(std), wrong) > 0, std
    # The floor itself -- the most entropic reachable policy -- still reads as "too
    # stochastic" against -20, which is what makes the target unsatisfiable.
    assert alpha_gradient(MIN_ACHIEVABLE_LOGPROB, wrong) > 0


def test_reachable_target_lets_alpha_correct_in_both_directions():
    target = default_target_entropy()
    # Too little entropy (tight policy) -> alpha must rise.
    assert alpha_gradient(measured_logprob(0.05), target) < 0
    # Too much entropy (near the floor) -> alpha must fall.
    assert alpha_gradient(measured_logprob(0.40), target) > 0


def test_target_corresponds_to_the_declared_std():
    assert measured_logprob(TARGET_ENTROPY_STD) == pytest.approx(
        default_target_entropy(), abs=1.0)


def test_target_std_keeps_exploration_near_the_sft_policy():
    """Meaningful exploration without leaving the action range the SFT policy uses."""
    assert 0.1 < TARGET_ENTROPY_STD < 0.5
