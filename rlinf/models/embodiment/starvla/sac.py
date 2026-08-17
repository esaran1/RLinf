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

"""SAC support for the StarVLA QwenOFT policy.

The existing OFT rollout path builds an **unsquashed** ``Normal`` (PPO convention:
unbounded support, log-prob needs no Jacobian). SAC needs a bounded, reparameterized
sample whose log-prob carries the tanh log-det correction, so this module wraps the OFT
mean action in RLinf's ``SquashedNormal`` instead of adding a second ad-hoc distribution.

Two things are specific to this task and are enforced here rather than left to the
worker:

* **Active-dimension masking.** Only 20 of the 30 action dims are RL-controllable; the
  other 10 are degenerate in the demonstration data. Frozen dims are re-injected at
  their dataset constants after sampling, and are excluded from the log-prob sum, so
  exploration cannot perturb them and alpha tuning cannot chase entropy in them.
* **Frozen backbone.** Only the OFT action head and ``actor_logstd`` carry gradients;
  the Qwen VLM is run under ``torch.no_grad`` for the representation used by the critic.

The critic consumes the pooled VLM hidden state plus the 30-D **actor-space** action.
The 30->53 mapper and the Inspire hand retargeter stay on the environment side and are
never seen by the critic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rlinf.envs.isaaclab.tasks.g1_piston_rl_space import (
    ACTION_DIM,
    build_active_mask,
)
from rlinf.models.embodiment.modules.gaussian_policy import SquashedNormal
from rlinf.models.embodiment.modules.q_head import MultiQHead

#: q99-normalized actions are clamped to this range, so the squash bounds match it.
#: Using [-1, 1] would saturate tanh on in-distribution SFT actions.
ACTION_LOW = -2.2
ACTION_HIGH = 2.2


def pool_hidden(hidden: torch.Tensor, attention_mask=None) -> torch.Tensor:
    """Mean-pool the VLM hidden states into a single state feature vector."""
    if attention_mask is None:
        return hidden.mean(dim=1)
    m = attention_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


class StarVLASACHeads(nn.Module):
    """Critic (+ target) heads bolted onto a frozen StarVLA representation.

    Deliberately small: the state feature is already a 2B-model embedding, so the Q
    functions only need a light MLP on top. Running a second VLA as the critic would not
    fit alongside Isaac Sim on a 16 GB card.
    """

    def __init__(
        self,
        hidden_size: int,
        action_dim: int = ACTION_DIM,
        num_action_chunks: int = 30,
        hidden_dims=(256, 256),
        num_q_heads: int = 2,
        dtype=torch.float32,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        # The critic scores a whole chunk, because one replay transition is one chunk.
        self.action_feature_dim = action_dim * num_action_chunks
        self.q_head = MultiQHead(
            hidden_size=hidden_size,
            action_feature_dim=self.action_feature_dim,
            hidden_dims=list(hidden_dims),
            num_q_heads=num_q_heads,
        ).to(dtype=dtype)

    def forward(self, state_features, actions):
        """``actions``: ``[B, chunk, action_dim]`` -> flattened for the Q head."""
        if actions.dim() == 3:
            actions = actions.reshape(actions.shape[0], -1)
        return self.q_head(state_features, actions)


class StarVLASACMixin:
    """Adds ``sac_forward`` / ``sac_q_forward`` to the StarVLA action model.

    Mixed into ``StarVLAForRLActionPrediction`` so the existing RLinf SAC/RLPD workers
    can drive it unchanged.
    """

    def init_sac(
        self,
        hidden_size: int,
        frozen_action_values: torch.Tensor,
        num_q_heads: int = 2,
        critic_hidden_dims=(256, 256),
        dtype=torch.float32,
    ):
        self.sac_heads = StarVLASACHeads(
            hidden_size=hidden_size,
            action_dim=self.action_dim,
            num_action_chunks=self.num_action_chunks,
            hidden_dims=critic_hidden_dims,
            num_q_heads=num_q_heads,
            dtype=dtype,
        )
        self.register_buffer("_active_mask", build_active_mask(), persistent=False)
        self.register_buffer(
            "_frozen_values", frozen_action_values.to(dtype=torch.float32), persistent=False
        )

    # -- helpers -----------------------------------------------------------------

    def _oft_mean_and_hidden(self, obs):
        """Run the (frozen) backbone + OFT head; return mean action and pooled state.

        The backbone runs under ``no_grad``; only the action head is differentiable, so
        the actor gradient reaches the OFT head without ever touching the VLM.
        """
        from .action_heads.oft import _run_oft_backbone_and_head

        model_inputs = obs["model_inputs"] if "model_inputs" in obs else obs
        mean_actions, last_hidden, _ = _run_oft_backbone_and_head(
            self, model_inputs=model_inputs, use_cache=False
        )
        state = pool_hidden(last_hidden, model_inputs.get("attention_mask"))
        return mean_actions, state.detach()

    def _distribution(self, mean_actions):
        """SquashedNormal over ``[B*chunk, action_dim]``.

        ``SquashedNormal`` wraps ``Independent(Normal, 1)``, so it treats exactly the
        last dim as the event and returns one log-prob per batch row. Folding the chunk
        into the batch dim keeps the per-timestep factorization (and lets the worker sum
        over the chunk itself), rather than mis-broadcasting a 3-D input.
        """
        b, c, d = mean_actions.shape
        flat = mean_actions.reshape(b * c, d)
        std = torch.exp(self.actor_logstd).view(1, -1).expand_as(flat)
        return SquashedNormal(flat, std, low=ACTION_LOW, high=ACTION_HIGH), (b, c, d)

    def _apply_mask(self, sampled):
        m = self._active_mask.to(sampled.device)
        fv = self._frozen_values.to(device=sampled.device, dtype=sampled.dtype)
        return torch.where(m, sampled, fv.expand_as(sampled))

    # -- BasePolicy SAC interface -------------------------------------------------

    def sac_forward(self, obs=None, **kwargs):
        """Sample a bounded action chunk. Returns ``(action, logprob, state_feature)``.

        ``logprob`` is summed over active dims only and over the chunk, matching what
        ``fsdp_sac_policy_worker`` expects (it sums over the trailing dim itself).
        """
        mean_actions, state = self._oft_mean_and_hidden(obs)
        dist, (b, c, d) = self._distribution(mean_actions)
        flat = dist.rsample()  # reparameterized: gradient flows to the OFT head

        # Zero the log-prob contribution of frozen dims: they are deterministic, so
        # including them would add a constant that alpha tuning would chase.
        logprob = self._masked_log_prob(dist, flat).reshape(b, c)

        action = self._apply_mask(flat.reshape(b, c, d))
        return action, logprob, state

    def _masked_log_prob(self, dist, flat_sample):
        """Per-row log-prob counting active dims only.

        ``SquashedNormal.log_prob`` already sums over the event dim, so recompute the
        per-dimension terms and drop the frozen ones instead of subtracting them out.
        """
        base = dist.base_dist.base_dist  # Normal(loc, scale)
        # dist.transforms is a single ComposeTransform; flatten to its parts so each
        # one's per-dimension Jacobian can be applied before the event-dim sum.
        transform = []
        for t in dist.transforms:
            transform.extend(getattr(t, "parts", [t]))
        x = flat_sample
        # Invert the transform chain to recover the pre-squash sample.
        for t in reversed(transform):
            x = t.inv(x)
        per_dim = base.log_prob(x)
        # Jacobian corrections are per-dim before the event sum; recompute them.
        y = x
        for t in transform:
            y_next = t(y)
            per_dim = per_dim - self._per_dim_log_det(t, y, y_next)
            y = y_next
        m = self._active_mask.to(per_dim.device)
        return (per_dim * m).sum(dim=-1)

    @staticmethod
    def _per_dim_log_det(transform, x, y):
        """Per-dimension log|det J| for the transforms used by SquashedNormal."""
        name = type(transform).__name__
        if name == "TanhTransform":
            return torch.log(1 - y.pow(2) + 1e-7)
        # RescaleTransform: constant scale
        scale = torch.as_tensor(transform.scale, dtype=x.dtype, device=x.device)
        return torch.log(scale.abs()).expand_as(x)

    def sac_q_forward(self, obs=None, actions=None, shared_feature=None,
                      detach_encoder=False, **kwargs):
        """Q-values for ``actions`` in **actor space** (30-D), not simulator space."""
        if shared_feature is None:
            _, shared_feature = self._oft_mean_and_hidden(obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
        return self.sac_heads(shared_feature, actions)

    def sac_deterministic_action(self, obs=None):
        """Evaluation action: the OFT mean, squashed and masked. No exploration."""
        mean_actions, _ = self._oft_mean_and_hidden(obs)
        dist = self._distribution(mean_actions)
        # The post-tanh mean is not tanh(mu); use the transformed base mean.
        squashed = dist.transforms[-1](dist.transforms[0](mean_actions)) \
            if len(dist.transforms) > 1 else torch.tanh(mean_actions)
        return self._apply_mask(squashed)

    def trainable_sac_parameters(self):
        """OFT head + exploration log-std + critics. The VLM is excluded by design."""
        groups = {
            "oft_action_head": list(self.starvla_model.action_model.parameters()),
            "actor_logstd": [self.actor_logstd],
            "critic": list(self.sac_heads.parameters()),
        }
        return groups

    def freeze_vlm(self):
        """Freeze every VLM parameter; leave the OFT head and RL params trainable."""
        for p in self.starvla_model.parameters():
            p.requires_grad_(False)
        for p in self.starvla_model.action_model.parameters():
            p.requires_grad_(True)
        self.actor_logstd.requires_grad_(True)
        for p in self.sac_heads.parameters():
            p.requires_grad_(True)
