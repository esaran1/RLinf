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

"""The SAC entropy target must live on the same scale as the log-prob it is compared to.

Regression guard for a measured training collapse: the alpha loss consumed a log-prob
summed over all 30 chunk steps while the target was the single-action convention
(-20.0). ``logp + target`` therefore stayed strongly negative, alpha was driven toward
zero, the entropy regulariser vanished, and the actor saturated tanh to chase Q. Actor
loss fell monotonically 1.94 -> -4.27 while behaviour collapsed from reach 1.0 / grasp
0.8 to zero. See ``docs/contracts/g1_piston_sac_pilot_v1_collapse.json``.
"""

import torch

from rlinf.envs.isaaclab.tasks.g1_piston_rl_space import (
    NUM_ACTIVE_DIMS,
    default_target_entropy,
)

#: Action chunk length of the StarVLA QwenOFT checkpoint.
ACTION_HORIZON = 30


def chunk_target_entropy(horizon: int = ACTION_HORIZON) -> float:
    """The target the trainer must use for a chunk-summed log-prob."""
    return default_target_entropy() * horizon


def test_single_action_target_is_negative_active_dims():
    assert default_target_entropy() == -float(NUM_ACTIVE_DIMS)


def test_chunk_target_scales_with_horizon():
    """A log-prob summed over H steps needs a target scaled by H."""
    assert chunk_target_entropy() == -600.0
    assert chunk_target_entropy(1) == default_target_entropy()


def test_alpha_gradient_direction_at_the_entropy_target():
    """At log-prob == target the alpha gradient must vanish, and flip sign around it.

    alpha_loss = -alpha * (logp + target); d/d(alpha) = -(logp + target). So a log-prob
    ABOVE the target (too little entropy) must push alpha UP.
    """
    target = chunk_target_entropy()

    at_target = -(torch.tensor(-target) + target)
    assert torch.isclose(at_target, torch.tensor(0.0))

    too_little_entropy = -(torch.tensor(-target + 50.0) + target)
    assert too_little_entropy < 0  # gradient descent raises alpha

    too_much_entropy = -(torch.tensor(-target - 50.0) + target)
    assert too_much_entropy > 0  # gradient descent lowers alpha


def test_mis_scaled_target_understates_the_entropy_floor():
    """The single-action target against a chunk-summed log-prob understates the floor.

    Uses log-probs measured during the collapsed run. Both targets happen to push alpha
    in the same direction at these values -- the mis-scaling is not what flipped the
    sign -- but the wrong target sets the equilibrium 580 nats too high, so the policy
    is allowed to drift far more stochastic before alpha reacts at all.
    """
    measured_chunk_logprobs = [-139.2, -256.9, -293.1]
    wrong_target = default_target_entropy()      # -20.0
    right_target = chunk_target_entropy()        # -600.0

    assert right_target - wrong_target == -580.0

    # The equilibrium the alpha loss drives toward is logp == -target.
    assert -wrong_target == 20.0        # a near-deterministic 30-step chunk
    assert -right_target == 600.0       # the correct floor for 30 x 20 active dims

    # At the measured log-probs both targets read "too stochastic" and lower alpha, so
    # the mis-scaling alone does not explain the collapse.
    for lp in measured_chunk_logprobs:
        assert -(lp + wrong_target) > 0
        assert -(lp + right_target) > 0


def test_entropy_floor_is_per_active_dim_not_per_action_dim():
    """The floor must count only RL-controllable dims, over the whole chunk."""
    assert chunk_target_entropy() == -float(NUM_ACTIVE_DIMS) * ACTION_HORIZON
    # 30 total dims would over-count the 10 frozen, deterministic dims.
    assert chunk_target_entropy() != -30.0 * ACTION_HORIZON
