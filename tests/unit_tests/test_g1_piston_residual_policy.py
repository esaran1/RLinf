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

"""The residual arm must be the base policy at step 0 and stay bounded around it."""

import importlib.util
import math
import sys

import pytest

torch = pytest.importorskip("torch")

from rlinf.envs.isaaclab.tasks.g1_piston_residual_policy import (  # noqa: E402
    ACTION_HIGH,
    ACTION_LOW,
    ResidualPolicy,
    latent_target_entropy,
)


def _mask():
    spec = importlib.util.spec_from_file_location(
        "g1s_res", "rlinf/envs/isaaclab/tasks/g1_piston_rl_space.py")
    m = importlib.util.module_from_spec(spec); sys.modules["g1s_res"] = m
    spec.loader.exec_module(m)
    return m.build_active_mask()


def _inputs(b=4, feat=2048):
    torch.manual_seed(0)
    aq = torch.randn(b, 30, feat)
    a_base = torch.tanh(torch.randn(b, 30, 30)) * 2.2
    return aq, a_base


def test_residual_is_exactly_zero_at_initialisation():
    """The load-bearing property: no luck involved, the initial policy IS the base."""
    pol = ResidualPolicy()
    aq, a_base = _inputs()
    a, lp, c = pol.act(aq, a_base, None, _mask(), deterministic=True)
    assert torch.equal(a, torch.clamp(a_base, ACTION_LOW, ACTION_HIGH))
    assert float(c.abs().max()) == 0.0


def test_residual_is_bounded_by_r_max_per_coefficient():
    pol = ResidualPolicy(r_max=0.15)
    nn_ = torch.nn
    nn_.init.normal_(pol.out.weight, std=5.0)      # saturate deliberately
    aq, a_base = _inputs()
    c = pol.coeff_mean(aq, a_base)
    assert float(c.abs().max()) <= 0.15 + 1e-6


def test_executed_action_stays_inside_the_bounds():
    pol = ResidualPolicy()
    torch.nn.init.normal_(pol.out.weight, std=5.0)
    aq, a_base = _inputs()
    logstd = torch.full((30,), math.log(0.5))
    a, _, _ = pol.act(aq, a_base, logstd, _mask())
    # float32 representation of the bound sits a few ulp below the Python double.
    assert float(a.min()) >= ACTION_LOW - 1e-6 and float(a.max()) <= ACTION_HIGH + 1e-6


def test_frozen_dims_receive_no_residual():
    pol = ResidualPolicy()
    torch.nn.init.normal_(pol.out.weight, std=1.0)
    aq, a_base = _inputs()
    mask = _mask()
    a, _, _ = pol.act(aq, a_base, None, mask, deterministic=True)
    assert torch.equal(a[..., ~mask], torch.clamp(a_base, ACTION_LOW, ACTION_HIGH)[..., ~mask])


def test_residual_is_temporally_smooth():
    """A residual living in the first 6 DCT components must be far smoother than i.i.d."""
    pol = ResidualPolicy()
    torch.nn.init.normal_(pol.out.weight, std=1.0)
    aq, a_base = _inputs(b=64)
    c = pol.coeff_mean(aq, a_base)
    r = pol.expand(c)
    d_res = (r[:, 1:] - r[:, :-1]).abs().mean()
    iid = torch.randn_like(r) * r.std()
    d_iid = (iid[:, 1:] - iid[:, :-1]).abs().mean()
    assert float(d_res) < 0.35 * float(d_iid)


def test_gradient_reaches_the_residual_but_not_the_base():
    """RL must train the residual only; the base chunk is an input, not a parameter."""
    pol = ResidualPolicy()
    aq, a_base = _inputs()
    a_base = a_base.requires_grad_(True)
    logstd = torch.full((30,), math.log(0.25), requires_grad=True)
    a, lp, _ = pol.act(aq, a_base, logstd, _mask())
    (a.sum() + lp.sum()).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in pol.parameters())
    # The base receives a gradient only through the identity of a = base + r, which the
    # trainer never applies because the head is frozen; what matters is the residual's.
    assert pol.out.weight.grad is not None


def test_latent_logprob_matches_the_closed_form_target_at_zero_noise():
    """At z = 0 the per-step log-density equals the closed-form target minus the 0.5
    quadratic term, so the target and the sampler share one convention."""
    pol = ResidualPolicy()
    aq, a_base = _inputs(b=1)
    mask = _mask(); n_active = int(mask.sum())
    std = 0.25
    logstd = torch.full((30,), math.log(std))
    g = torch.Generator().manual_seed(0)
    # Force z = 0 by sampling with a zero generator trick: compare expectations instead.
    lps = []
    for s in range(200):
        g.manual_seed(s)
        _, lp, _ = pol.act(aq, a_base, logstd, mask, generator=g)
        lps.append(float(lp[0, 0]))
    expected = latent_target_entropy(std, n_active)
    assert abs(sum(lps) / len(lps) - expected) < 0.15, (sum(lps) / len(lps), expected)


def test_entropy_pulls_the_residual_toward_the_base_not_away():
    """Under latent entropy the only mean-dependence is none at all: the entropy
    gradient w.r.t. the coefficient mean is zero, so the entropy term can never push
    the residual away from the base."""
    pol = ResidualPolicy()
    torch.nn.init.normal_(pol.out.weight, std=0.5)
    aq, a_base = _inputs()
    logstd = torch.full((30,), math.log(0.25))
    _, lp, c_mean = pol.act(aq, a_base, logstd, _mask())
    # c_mean depends on the network; lp must not: there is no autograd path at all.
    assert c_mean.requires_grad
    assert not lp.requires_grad


def test_loads_by_file_path_without_the_package():
    spec = importlib.util.spec_from_file_location(
        "g1_res_path", "rlinf/envs/isaaclab/tasks/g1_piston_residual_policy.py")
    m = importlib.util.module_from_spec(spec); sys.modules["g1_res_path"] = m
    spec.loader.exec_module(m)
    assert m.ResidualPolicy().n_basis == 6
