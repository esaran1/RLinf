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

"""Residual policy over a FROZEN behaviour-cloning base (ResFiT-style).

Why
---
Direct SAC fine-tuning of the OFT action head destroyed a working policy twice, and the
mechanism is now measured (``g1_piston_entropy_mean_force.json``). Even with that fixed,
the literature is consistent that directly optimising a large chunked BC network with
off-policy RL is unstable, and that the robust alternative is to freeze the base and
learn a small per-step correction with RL: ResFiT (arXiv 2509.19301), the first
real-world RL on a bimanual five-fingered humanoid, used exactly this with sparse binary
reward. Classical residual RL: Silver et al. 2018, Johannink et al. 2019.

What
----
    a = clamp(a_base + r,  LOW, HIGH)
    a_base = squash(frozen_head(aq))                 # the BC policy's executed chunk
    r      = expand(c),  c = c_mean + std * z        # temporally smooth residual
    c_mean = R_MAX * tanh(MLP(pool(aq), dct(a_base)))

* ``r`` lives in the same truncated DCT basis as the correlated exploration noise, so the
  correction is smooth in time like the demonstrations (real chunks carry 99.7% of their
  energy in the first 6 components).
* The MLP's last layer is initialised to ZERO, so at step 0 the policy IS the BC policy:
  grasp 1.00 is the starting point, not something to be preserved by luck.
* ``R_MAX`` bounds the coefficients, so the residual cannot leave the base's neighbourhood
  in one step ("safety since we can control the magnitude of this residual", ResFiT).
* Exploration is Gaussian on the LATENT coefficients with no squash, so the entropy is
  closed-form and its only pull on the mean is toward zero residual, i.e. toward the base
  policy -- the one direction in which an inward force is desirable.
* The actor input is built from ``aq`` alone (pooled VLM action queries + the DCT of the
  base chunk), so evaluation and rendering tools can apply the residual with nothing
  but the checkpoint and the image. No privileged state reaches the actor.

The module loads both as a package member and by file path, because the training and
evaluation tools load task modules by path.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

try:  # package import
    from rlinf.envs.isaaclab.tasks.g1_piston_action_basis import (
        ACTION_DIMS,
        HORIZON,
        N_BASIS,
        dct_basis,
    )
    from rlinf.envs.isaaclab.tasks.g1_piston_correlated_policy import (
        correlated_noise_scales,
    )
except Exception:  # loaded by file path: import siblings the same way
    import importlib.util as _ilu
    import sys as _sys

    def _sib(name, fname):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        spec = _ilu.spec_from_file_location(name, p)
        mod = _ilu.module_from_spec(spec)
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _ab = _sib("g1_piston_action_basis_res", "g1_piston_action_basis.py")
    _cp = _sib("g1_piston_correlated_policy_res", "g1_piston_correlated_policy.py")
    ACTION_DIMS, HORIZON, N_BASIS, dct_basis = (_ab.ACTION_DIMS, _ab.HORIZON,
                                                _ab.N_BASIS, _ab.dct_basis)
    correlated_noise_scales = _cp.correlated_noise_scales

#: Bound on each residual DCT coefficient, in the executed (normalised, [-2.2, 2.2])
#: action units. With unit-variance basis weights the per-step residual is then bounded
#: by roughly R_MAX in those units -- about 7% of the action range, comparable to the
#: exploration std, so the residual starts as a correction, not a replacement.
R_MAX_DEFAULT = 0.15
ACTION_LOW, ACTION_HIGH = -2.2, 2.2


class ResidualPolicy(nn.Module):
    """Small MLP emitting residual DCT coefficients over a frozen base chunk."""

    def __init__(self, feat_dim: int = 2048, hidden=(512, 512), r_max: float = R_MAX_DEFAULT,
                 horizon: int = HORIZON, dims: int = ACTION_DIMS, n_basis: int = N_BASIS,
                 device="cpu"):
        super().__init__()
        self.horizon, self.dims, self.n_basis, self.r_max = horizon, dims, n_basis, r_max
        B = torch.as_tensor(dct_basis(horizon, n_basis), dtype=torch.float32)   # [H,K]
        w = torch.as_tensor(correlated_noise_scales(horizon, n_basis), dtype=torch.float32)
        self.register_buffer("basis", B)
        self.register_buffer("scales", w)
        in_dim = feat_dim + n_basis * dims
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.Tanh()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(d, n_basis * dims)
        # ZERO init: the residual is exactly zero at step 0, so the initial policy is the
        # base policy. This is the property that makes the arm safe by construction.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.to(device)

    # ----------------------------------------------------------------- features --
    def project(self, chunk: torch.Tensor) -> torch.Tensor:
        """[B,H,D] -> [B,K,D] DCT coefficients."""
        return torch.einsum("hk,bhd->bkd", self.basis, chunk)

    def expand(self, coeffs: torch.Tensor) -> torch.Tensor:
        """[B,K,D] -> [B,H,D], with the unit-per-step-variance weights."""
        return torch.einsum("hk,bkd->bhd", self.basis, coeffs * self.scales.view(1, -1, 1))

    def features(self, aq: torch.Tensor, a_base: torch.Tensor) -> torch.Tensor:
        """Actor input from the VLM action queries and the base chunk only."""
        pooled = aq.mean(dim=1)                                         # [B,feat]
        return torch.cat([pooled, self.project(a_base).flatten(1)], dim=-1)

    # -------------------------------------------------------------------- policy --
    def coeff_mean(self, aq: torch.Tensor, a_base: torch.Tensor) -> torch.Tensor:
        """Bounded residual coefficient means, [B,K,D]."""
        x = self.out(self.trunk(self.features(aq, a_base)))
        return self.r_max * torch.tanh(x).view(-1, self.n_basis, self.dims)

    def compose(self, a_base: torch.Tensor, coeffs: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Executed chunk: base plus expanded residual, clamped to the action bounds.
        Frozen dims (mask False) receive no residual."""
        r = self.expand(coeffs)
        if mask is not None:
            r = r * mask.to(r.dtype).view(1, 1, -1)
        return torch.clamp(a_base + r, ACTION_LOW, ACTION_HIGH)

    def act(self, aq: torch.Tensor, a_base: torch.Tensor, logstd: torch.Tensor | None,
            mask: torch.Tensor | None = None, deterministic: bool = False,
            generator=None):
        """Sample (or take the mean of) the residual policy.

        Returns ``(action [B,H,D], logp_per_step [B,H], coeff_mean [B,K,D])``.
        The log-density is that of the LATENT Gaussian on the coefficients (no squash),
        summed over active dims and spread over the horizon, matching the scale of the
        trainer's per-control-action convention.
        """
        c_mean = self.coeff_mean(aq, a_base)
        B_, K, D = c_mean.shape
        if deterministic or logstd is None:
            a = self.compose(a_base, c_mean, mask)
            return a, torch.zeros(B_, self.horizon, device=a.device), c_mean
        std = torch.exp(logstd).view(1, 1, -1)
        z = torch.randn(B_, K, D, device=c_mean.device, generator=generator)
        c = c_mean + std * z
        a = self.compose(a_base, c, mask)
        m = (mask.to(c.dtype) if mask is not None
             else torch.ones(D, device=c.device)).view(1, 1, -1)
        lp = ((-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) - logstd.view(1, 1, -1)) * m
              ).sum(dim=(-2, -1))                                        # [B]
        lp = (lp / self.horizon).unsqueeze(-1).expand(-1, self.horizon)  # [B,H]
        return a, lp, c_mean


def latent_target_entropy(std: float, n_active: int, n_basis: int = N_BASIS,
                          horizon: int = HORIZON) -> float:
    """Closed-form per-control-action log-density of the residual's latent Gaussian at
    exploration std ``std``: the target the temperature should regulate toward."""
    per_coeff = -0.5 - 0.5 * math.log(2 * math.pi) - math.log(std)
    return per_coeff * n_basis * n_active / horizon
