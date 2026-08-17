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

"""Replay entries are stored in half precision; the round trip must be lossless enough.

The action queries dominate the buffer footprint -- ``H x HID = 30 x 2048`` floats per
state, twice per transition -- so a 500-episode run holds ~5.5 GB of CPU RAM at float32
alongside a resident Isaac Sim. They are a deterministic function of the FROZEN VLM, so
half precision is safe, but only if the error stays far below the exploration noise the
policy adds on top.
"""

import torch

ACTION_HORIZON = 30
HIDDEN = 2048
ACTION_DIM = 30

#: Exploration std at the initial ``actor_logstd`` of -1.0.
EXPLORATION_STD = float(torch.exp(torch.tensor(-1.0)))


def _store(feat, aq, action, reward, nfeat, naq, done):
    """Mirror of the trainer's ``store``: big tensors to CPU float16."""
    h = torch.float16
    return (feat.squeeze(0).to("cpu", dtype=h), aq.squeeze(0).to("cpu", dtype=h),
            action.detach().to("cpu", dtype=h), float(reward),
            nfeat.squeeze(0).to("cpu", dtype=h), naq.squeeze(0).to("cpu", dtype=h),
            bool(done))


def _collate(items):
    def cat(i):
        return torch.stack([x[i] for x in items]).to(dtype=torch.float32)
    return cat(0), cat(1), cat(2)


def _entry(seed=0):
    g = torch.Generator().manual_seed(seed)
    feat = torch.randn(1, HIDDEN, generator=g)
    aq = torch.randn(1, ACTION_HORIZON, HIDDEN, generator=g)
    action = torch.randn(ACTION_HORIZON, ACTION_DIM, generator=g) * 2.2
    return feat, aq, action


def test_round_trip_shapes_survive_storage():
    feat, aq, action = _entry()
    e = _store(feat, aq, action, 1.0, feat, aq, False)
    feats, aqs, acts = _collate([e, e])
    assert feats.shape == (2, HIDDEN)
    assert aqs.shape == (2, ACTION_HORIZON, HIDDEN)
    assert acts.shape == (2, ACTION_HORIZON, ACTION_DIM)
    assert feats.dtype == torch.float32, "consumers expect float32"


def test_round_trip_error_is_far_below_exploration_noise():
    feat, aq, action = _entry()
    e = _store(feat, aq, action, 0.0, feat, aq, False)
    feats, aqs, acts = _collate([e])
    assert torch.max(torch.abs(feats[0] - feat[0])) < EXPLORATION_STD / 100
    assert torch.max(torch.abs(aqs[0] - aq[0])) < EXPLORATION_STD / 100
    assert torch.max(torch.abs(acts[0] - action)) < EXPLORATION_STD / 100


def test_relative_error_is_small_across_activation_scales():
    for scale in (1.0, 10.0, 100.0):
        x = torch.randn(4096) * scale
        rel = (x.to(torch.float16).float() - x).abs() / x.abs().clamp(min=1e-6)
        assert float(rel.median()) < 1e-3, scale


def test_scalars_are_not_stored_in_half_precision():
    """Rewards and done flags stay exact: they drive the bootstrap target."""
    feat, aq, action = _entry()
    e = _store(feat, aq, action, 3.14159265, feat, aq, True)
    assert isinstance(e[3], float) and e[3] == 3.14159265
    assert e[6] is True


def test_half_precision_halves_the_footprint():
    per_f32 = (HIDDEN + ACTION_HORIZON * HIDDEN) * 2 * 4 + ACTION_HORIZON * ACTION_DIM * 4
    per_f16 = (HIDDEN + ACTION_HORIZON * HIDDEN) * 2 * 2 + ACTION_HORIZON * ACTION_DIM * 2
    assert per_f16 * 2 == per_f32
    # A 500-episode run at 23 chunks per episode must stay well inside available RAM.
    assert per_f16 * 500 * 23 / 1024**3 < 4.0
