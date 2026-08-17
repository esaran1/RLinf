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

"""SAC distribution/critic correctness for the StarVLA OFT actor."""

import pytest
import torch

from rlinf.envs.isaaclab.tasks.g1_piston_rl_space import (
    ACTIVE_ACTION_DIMS,
    FROZEN_ACTION_DIMS,
    build_active_mask,
    load_frozen_values,
)
from rlinf.models.embodiment.modules.gaussian_policy import SquashedNormal
from rlinf.models.embodiment.starvla.sac import (
    ACTION_HIGH,
    ACTION_LOW,
    StarVLASACHeads,
    StarVLASACMixin,
)

STATS = (
    "/home/jren313/research/starvla_rl/checkpoints/"
    "g1-longhorizon-oft-v1/dataset_statistics.json"
)


class _Stub(StarVLASACMixin):
    """Actor stub: distribution + masking only, no VLM."""

    def __init__(self):
        self.action_dim = 30
        self.num_action_chunks = 30
        self.actor_logstd = torch.nn.Parameter(torch.full((30,), -2.5))
        self._active_mask = build_active_mask()
        self._frozen_values = load_frozen_values(STATS)


@pytest.fixture
def stub():
    return _Stub()


def test_squashed_normal_respects_action_bounds():
    d = SquashedNormal(
        torch.zeros(8, 30), torch.full((8, 30), 1.0), low=ACTION_LOW, high=ACTION_HIGH
    )
    s = d.rsample()
    assert s.min() >= ACTION_LOW - 1e-5
    assert s.max() <= ACTION_HIGH + 1e-5


def test_rescale_jacobian_accepts_scalar_bounds():
    """Regression: log_abs_det_jacobian used to call torch.abs on a Python float."""
    d = SquashedNormal(
        torch.zeros(4, 30), torch.full((4, 30), 0.1), low=ACTION_LOW, high=ACTION_HIGH
    )
    lp = d.log_prob(d.rsample())
    assert torch.isfinite(lp).all()


def test_sac_logprob_shape_and_finiteness(stub):
    mu = torch.zeros(2, 30, 30)
    dist, (b, c, _) = stub._distribution(mu)
    lp = stub._masked_log_prob(dist, dist.rsample()).reshape(b, c)
    assert lp.shape == (2, 30)
    assert torch.isfinite(lp).all()


def test_frozen_dims_are_held_at_dataset_constants(stub):
    mu = torch.randn(3, 30, 30)
    dist, (b, c, d) = stub._distribution(mu)
    action = stub._apply_mask(dist.rsample().reshape(b, c, d))
    fv = load_frozen_values(STATS)
    for dim in FROZEN_ACTION_DIMS:
        assert torch.allclose(action[..., dim], fv[dim].expand(b, c), atol=1e-6)


def test_frozen_dims_receive_no_gradient(stub):
    """The invariant that keeps SAC exploration out of the degenerate dims."""
    mu = torch.zeros(2, 30, 30, requires_grad=True)
    dist, (b, c, _) = stub._distribution(mu)
    lp = stub._masked_log_prob(dist, dist.rsample()).reshape(b, c)
    g = torch.autograd.grad(lp.sum(), mu)[0].abs().sum(dim=(0, 1))

    assert float(g[list(FROZEN_ACTION_DIMS)].sum()) == pytest.approx(0.0, abs=1e-9)
    assert float(g[list(ACTIVE_ACTION_DIMS)].sum()) > 0.0


def test_reparameterized_sample_is_differentiable(stub):
    mu = torch.zeros(2, 30, 30, requires_grad=True)
    dist, _ = stub._distribution(mu)
    g = torch.autograd.grad(dist.rsample().sum(), mu)[0]
    assert torch.isfinite(g).all()
    assert g.abs().sum() > 0


def test_critic_shapes_and_finiteness():
    h = StarVLASACHeads(hidden_size=512, action_dim=30, num_action_chunks=30)
    q = h(torch.randn(4, 512), torch.randn(4, 30, 30))
    assert q.shape == (4, 2)  # two Q heads
    assert torch.isfinite(q).all()


def test_critic_is_small_enough_to_coexist_with_the_vla():
    h = StarVLASACHeads(hidden_size=2048, action_dim=30, num_action_chunks=30)
    n = sum(p.numel() for p in h.parameters())
    assert n < 20e6, f"critic has {n / 1e6:.1f}M params; must stay light on a 16 GB card"
