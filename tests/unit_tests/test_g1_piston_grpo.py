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
# The real active set: both arms (0-13) and the right hand (20-25).
REAL_MASK = torch.tensor([i < 14 or 20 <= i < 26 for i in range(30)])


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


def test_per_dimension_sigma_reduces_to_the_scalar_case_and_targets_dims():
    mu = torch.zeros(2, 6, 30); c = mu + 0.1
    scalar = gaussian_logp_mean(c, mu, 0.15, MASK)
    vec = gaussian_logp_mean(c, mu, torch.full((30,), 0.15), MASK)
    assert torch.allclose(scalar, vec)
    # a larger sigma on the hand dims (20-25) changes the density there -- and ONLY there:
    # with a mask that excludes the hand dims the result is identical.
    sig = torch.full((30,), 0.15); sig[20:26] = 0.35
    assert torch.allclose(gaussian_logp_mean(c, mu, sig, MASK), vec)
    vec_real = gaussian_logp_mean(c, mu, torch.full((30,), 0.15), REAL_MASK)
    lp_hand = gaussian_logp_mean(c, mu, sig, REAL_MASK)
    assert not torch.allclose(lp_hand, vec_real)
    kl_s = kl_to_base_mean(torch.full((1, 6, 30), 0.1), 0.15, REAL_MASK)
    kl_v = kl_to_base_mean(torch.full((1, 6, 30), 0.1), torch.full((30,), 0.15), REAL_MASK)
    assert torch.allclose(kl_s, kl_v)
    kl_h = kl_to_base_mean(torch.full((1, 6, 30), 0.1), sig, REAL_MASK)
    assert float(kl_h) < float(kl_v)          # wider hand sigma -> smaller KL for the same mean


def test_trainer_wires_targeted_exploration():
    src = open("tools/g1_piston/train_grpo.py").read()
    assert 'SIGMA_HAND = float(os.environ.get("SIGMA_HAND", str(SIGMA)))' in src
    assert "SIG[20:26] = SIGMA_HAND" in src
    assert "SIG.view(1, 1, -1) * torch.randn_like(c_mean)" in src
    assert '"actor_logstd": torch.log(SIG.detach().cpu())' in src
