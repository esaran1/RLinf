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

"""Critic-free policy optimisation for the chunked residual policy (GRPO + PPO + KL anchor).

Why: three actor-critic arms collapsed a working policy because the critic could not rank
actions (``g1_piston_critic_exploitation.json``, replicated on runs 7 and 8). Here the
learning signal is the simulator's own return, so a perturbation that breaks the grasp is
pushed down by what actually happened, and a trust region around the known-good policy
replaces a hard bound. This is the published recipe for RL fine-tuning of OFT-style VLA
heads (GRPO, Shao et al. 2024; SimpleVLA-RL, 2025; RLinf's embodied PPO), applied to a
ResFiT-style frozen base + residual (arXiv 2509.19301).

Pure functions over tensors, so they are tested without a simulator. Conventions:
* a "chunk" is one policy decision; an episode has T chunks; a group is G episodes from
  the same initial condition.
* log-densities are MEANS over the active latent dimensions (K coefficients x active
  dims), so PPO's clip range and the KL coefficient have the same meaning regardless of
  dimensionality.
"""

from __future__ import annotations

import math

import torch


def gaussian_logp_mean(c: torch.Tensor, mu: torch.Tensor, sigma: float,
                       mask: torch.Tensor) -> torch.Tensor:
    """Per-chunk log-density of latent coefficients ``c`` under N(mu, sigma^2), averaged
    over the active (k, d) entries. Shapes: c, mu [B, K, D]; mask [D] bool -> [B]."""
    m = mask.to(c.dtype).view(1, 1, -1)
    per = -0.5 * ((c - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
    n = m.sum() * c.shape[1]
    return (per * m).sum(dim=(-2, -1)) / n


def kl_to_base_mean(mu: torch.Tensor, sigma: float, mask: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, s^2) || N(0, s^2)) per chunk, averaged over active entries: the distance
    of the residual policy from the frozen base, which has zero residual. [B]."""
    m = mask.to(mu.dtype).view(1, 1, -1)
    n = m.sum() * mu.shape[1]
    return ((mu ** 2) * m).sum(dim=(-2, -1)) / (2 * sigma ** 2) / n


def returns_to_go(rewards: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """[T] per-chunk rewards -> [T] discounted return-to-go."""
    out = torch.zeros_like(rewards)
    acc = 0.0
    for t in range(rewards.shape[0] - 1, -1, -1):
        acc = float(rewards[t]) + gamma * acc
        out[t] = acc
    return out


def group_advantages(rtg_by_episode: list[torch.Tensor], eps: float = 1e-6) -> list[torch.Tensor]:
    """GRPO advantages for one group (G episodes of the same initial condition).

    Each episode supplies its per-chunk return-to-go [T_i]. At chunk index t the advantage
    is the episode's return-to-go standardised across the group members that reached
    chunk t (mean subtracted, divided by std + eps). A group whose members all got the
    same return carries no signal and yields zero advantage, by construction.
    """
    G = len(rtg_by_episode)
    if G < 2:
        return [torch.zeros_like(r) for r in rtg_by_episode]
    T = max(r.shape[0] for r in rtg_by_episode)
    out = [torch.zeros_like(r) for r in rtg_by_episode]
    for t in range(T):
        vals = torch.stack([r[t] for r in rtg_by_episode if r.shape[0] > t])
        if vals.numel() < 2:
            continue
        mu, sd = vals.mean(), vals.std(unbiased=False)
        for i, r in enumerate(rtg_by_episode):
            if r.shape[0] > t:
                out[i][t] = (r[t] - mu) / (sd + eps)
    return out


def ppo_clipped_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor,
                     clip: float = 0.2):
    """Clipped surrogate (to MINIMISE) and diagnostics. All inputs [B]."""
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        approx_kl = (logp_old - logp_new).mean()
        clipfrac = ((ratio - 1).abs() > clip).float().mean()
    return loss, {"approx_kl": float(approx_kl), "clipfrac": float(clipfrac),
                  "ratio_mean": float(ratio.mean())}
