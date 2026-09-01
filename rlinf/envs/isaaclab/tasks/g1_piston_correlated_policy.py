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

"""Temporally correlated action-chunk policy: stop sampling 900 independent scalars.

The review point this answers: *"Each dimension of the action seems to be independent.
That's not correct. The action dimensions depend heavily on each other given the
configuration of the hand, the arm and the temporal dependency."*

Measured on the real demonstrations, the factorised diagonal Gaussian is not a mild
approximation -- it is the dominant source of what the robot actually executes:

* real demonstration chunks move **0.0011 rad** between consecutive steps;
* i.i.d. noise at the training std (0.20) adds **0.2257 rad** per step -- **209x** the
  real motion;
* in the DCT basis, real chunks carry **99.7%** of their energy in the first 6 temporal
  components, while i.i.d. noise spreads energy **flat across all 30** -- so **80% of the
  injected exploration is high-frequency jitter the task never contains**.

So the policy explores almost entirely in directions the demonstrations show are
irrelevant, and the executed trajectory is dominated by noise rather than by the mean the
network predicts. That also explains the measured oscillation pathology
(``g1_piston_rl_induced_oscillation.json``) at its source: the behaviour policy *is*
jittery by construction, and the actor is trained on those jittery samples.

What this module does
---------------------
Keep the same network and the same squashed-Gaussian machinery, but draw the exploration
noise in a **temporally correlated** way: sample low-dimensional coefficients in the DCT
basis and expand them back into a chunk. Concretely, for each action dimension the noise
is

    eps = B[:, :K] @ (w * z),        z ~ N(0, I_K)

where ``B`` is the orthonormal DCT-II basis, ``K = N_BASIS`` (6), and ``w`` scales each
component so the resulting per-step marginal standard deviation matches the requested
``std``. The result is a Gaussian with a **structured covariance** rather than a diagonal
one: samples are smooth in time by construction, and the policy explores the subspace the
demonstrations actually occupy.

Three properties that make this defensible rather than a trick:

* **It is still an exact Gaussian**, so the log-probability is computable in closed form
  (in the whitened coefficient space) and SAC's entropy term remains correct. No
  approximation is introduced into the objective.
* **It is a reparameterised sample**, so ``rsample``-style gradients flow to the mean
  exactly as before.
* **It does not restrict the mean.** The network can still emit any chunk it likes; only
  the *exploration* is shaped. Deterministic evaluation is bit-identical to before.

Relation to the alternatives the review named (PPO, diffusion policy): those replace the
policy class outright and are a larger change. This is the smallest change that removes
the specific defect measured here -- independent per-scalar exploration -- inside the
existing off-policy algorithm. It is not a claim that a diffusion policy would not do
better; that remains recorded as an open architectural question.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# The tools in tools/g1_piston load these task modules BY FILE PATH (importlib), where
# the `rlinf` package is not on sys.path -- every other g1_piston task module is
# self-contained for exactly this reason. Support both: a normal package import when
# available, and a path-relative load when this module is executed standalone.
try:  # package context (tests, library use)
    from rlinf.envs.isaaclab.tasks.g1_piston_action_basis import (
        ACTION_DIMS,
        HORIZON,
        N_BASIS,
        dct_basis,
    )
except ModuleNotFoundError:  # path-loaded context (the training/eval tools)
    import importlib.util as _ilu
    import os as _os

    _spec = _ilu.spec_from_file_location(
        "_g1_piston_action_basis",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                      "g1_piston_action_basis.py"))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ACTION_DIMS = _mod.ACTION_DIMS
    HORIZON = _mod.HORIZON
    N_BASIS = _mod.N_BASIS
    dct_basis = _mod.dct_basis


def correlated_noise_scales(horizon: int = HORIZON, n_basis: int = N_BASIS,
                            flat: bool = False) -> np.ndarray:
    """Per-component weights ``w`` giving unit per-step marginal variance.

    With ``eps = B[:, :K] @ (w * z)`` and ``z ~ N(0, I_K)``, the marginal variance at
    step ``t`` is ``sum_k (B[t, k] * w_k)**2``. Choosing ``w_k`` constant makes that
    vary across ``t``; instead we solve for the constant ``c`` that makes the *average*
    per-step variance equal to 1, which keeps the interpretation of ``std`` unchanged
    (it remains the per-step exploration magnitude) while redistributing where the noise
    lives in frequency.

    Args:
        horizon: chunk length.
        n_basis: number of temporal components the noise is allowed to occupy.
        flat: if True, weight every component equally before normalisation; otherwise
            weight by ``1/(k+1)``, a pink-noise profile that puts more energy in slower
            components, matching the measured demonstration spectrum.

    Returns:
        ``(n_basis,)`` positive weights.
    """
    if not 1 <= n_basis <= horizon:
        raise ValueError(f"n_basis must be in [1, {horizon}], got {n_basis}")
    k = np.arange(n_basis, dtype=np.float64)
    w = np.ones(n_basis) if flat else 1.0 / (k + 1.0)
    B = dct_basis(horizon, n_basis)
    # Mean per-step variance induced by these weights, before normalisation.
    var = float(((B * w[None, :]) ** 2).sum(axis=1).mean())
    return w / math.sqrt(var)


class CorrelatedChunkNoise:
    """Draws temporally correlated exploration noise for action chunks.

    The noise for one chunk is ``B[:, :K] @ (scales * z)`` per action dimension, with
    ``z`` standard normal. Because the transform is linear and ``B`` is orthonormal, the
    sample is Gaussian with a known covariance and the entropy of ``z`` is what SAC's
    temperature should regulate -- see :meth:`logprob_z`.
    """

    def __init__(self, horizon: int = HORIZON, dims: int = ACTION_DIMS,
                 n_basis: int = N_BASIS, device: str = "cpu", flat: bool = False):
        self.horizon = int(horizon)
        self.dims = int(dims)
        self.n_basis = int(n_basis)
        self.scales = torch.as_tensor(
            correlated_noise_scales(horizon, n_basis, flat=flat), dtype=torch.float32,
            device=device)
        self.basis = torch.as_tensor(
            dct_basis(horizon, n_basis), dtype=torch.float32, device=device)

    def to(self, device):
        self.scales = self.scales.to(device)
        self.basis = self.basis.to(device)
        return self

    def sample_eps(self, batch: int, std, generator=None) -> torch.Tensor:
        """Return correlated noise ``[batch, horizon, dims]`` scaled by ``std``.

        Args:
            batch: number of chunks.
            std: scalar or ``[dims]`` tensor of per-step exploration magnitudes.
            generator: optional torch generator for reproducibility.
        """
        z = torch.randn(batch, self.n_basis, self.dims, device=self.basis.device,
                        generator=generator)
        return self.expand(z, std)

    def expand(self, z: torch.Tensor, std) -> torch.Tensor:
        """Expand whitened coefficients ``[batch, n_basis, dims]`` into a chunk.

        Kept separate from sampling so a caller can reparameterise: draw ``z`` once and
        differentiate through this expansion.
        """
        if z.shape[-2:] != (self.n_basis, self.dims):
            raise ValueError(
                f"expected z with shape [..., {self.n_basis}, {self.dims}], "
                f"got {tuple(z.shape)}")
        scaled = z * self.scales.view(1, -1, 1)
        eps = torch.einsum("hk,bkd->bhd", self.basis, scaled)
        s = torch.as_tensor(std, dtype=eps.dtype, device=eps.device)
        return eps * (s.view(1, 1, -1) if s.dim() == 1 else s)

    @staticmethod
    def logprob_z(z: torch.Tensor) -> torch.Tensor:
        """Log-density of the whitened coefficients, summed over ``(n_basis, dims)``.

        The chunk noise is a fixed linear map of ``z``, so up to a constant Jacobian term
        (identical for every sample and therefore irrelevant to gradients) this is the
        chunk's log-density. Returned per batch element.
        """
        return (-0.5 * (z ** 2) - 0.5 * math.log(2 * math.pi)).sum(dim=(-2, -1))

    def per_step_std(self, std=1.0) -> torch.Tensor:
        """Realised per-step marginal std, for verifying the calibration."""
        w = (self.basis * self.scales.view(1, -1)) ** 2
        return torch.sqrt(w.sum(dim=1)) * float(std)

    def effective_dim(self) -> int:
        """Number of independent scalars actually sampled per chunk (compare 900)."""
        return self.n_basis * self.dims
