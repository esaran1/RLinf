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

"""The critic-free objective must push toward higher-return actions and nowhere else."""

import math

import pytest

torch = pytest.importorskip("torch")

from rlinf.envs.isaaclab.tasks.g1_piston_grpo import (  # noqa: E402
    gaussian_logp_mean,
    group_advantages,
    kl_to_base_mean,
    ppo_clipped_loss,
    returns_to_go,
)

MASK = torch.tensor([True] * 20 + [False] * 10)


def test_logp_is_a_per_entry_mean_and_peaks_at_the_mean():
    mu = torch.zeros(1, 6, 30)
    lp0 = gaussian_logp_mean(mu, mu, 0.15, MASK)
    lp1 = gaussian_logp_mean(mu + 0.15, mu, 0.15, MASK)
    assert lp0 > lp1
    assert lp0.item() == pytest.approx(-math.log(0.15) - 0.5 * math.log(2 * math.pi))
    # frozen dims do not contribute
    mu2 = mu.clone(); mu2[..., 25] = 5.0
    assert torch.allclose(gaussian_logp_mean(mu, mu2, 0.15, MASK), lp0)


def test_kl_to_base_is_zero_at_zero_residual_and_grows_quadratically():
    assert float(kl_to_base_mean(torch.zeros(2, 6, 30), 0.15, MASK).abs().max()) == 0.0
    a = kl_to_base_mean(torch.full((1, 6, 30), 0.1), 0.15, MASK)
    b = kl_to_base_mean(torch.full((1, 6, 30), 0.2), 0.15, MASK)
    assert float(b / a) == pytest.approx(4.0)


def test_returns_to_go():
    r = torch.tensor([1.0, 0.0, 2.0])
    assert torch.allclose(returns_to_go(r), torch.tensor([3.0, 2.0, 2.0]))
    assert torch.allclose(returns_to_go(r, 0.5), torch.tensor([1.5, 1.0, 2.0]))


def test_group_advantage_ranks_the_better_episode_and_is_zero_without_contrast():
    good = torch.tensor([10.0, 8.0, 5.0]); bad = torch.tensor([2.0, 1.0, 0.0])
    a_good, a_bad = group_advantages([good, bad])
    assert (a_good > 0).all() and (a_bad < 0).all()
    same = [torch.tensor([3.0, 2.0]), torch.tensor([3.0, 2.0])]
    assert all(float(a.abs().max()) == 0.0 for a in group_advantages(same))
    # a single episode carries no signal
    assert float(group_advantages([good])[0].abs().max()) == 0.0


def test_group_advantage_handles_unequal_lengths():
    a, b = group_advantages([torch.tensor([5.0, 4.0, 3.0]), torch.tensor([1.0])])
    assert a.shape == (3,) and b.shape == (1,)
    assert a[0] > 0 and b[0] < 0 and float(a[1]) == 0.0 and float(a[2]) == 0.0


def test_ppo_loss_gradient_direction_and_clipping():
    logp_old = torch.zeros(4)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    logp_new = torch.zeros(4, requires_grad=True)
    loss, info = ppo_clipped_loss(logp_new, logp_old, adv, clip=0.2)
    loss.backward()
    # increasing logp where advantage is positive decreases the loss
    assert (logp_new.grad[:2] < 0).all() and (logp_new.grad[2:] > 0).all()
    assert info["clipfrac"] == 0.0
    # far outside the clip range, positive-advantage terms are clipped (zero gradient)
    lp = torch.full((2,), 1.0, requires_grad=True)
    loss2, info2 = ppo_clipped_loss(lp, torch.zeros(2), torch.ones(2))
    loss2.backward()
    assert float(lp.grad.abs().max()) == 0.0 and info2["clipfrac"] == 1.0
