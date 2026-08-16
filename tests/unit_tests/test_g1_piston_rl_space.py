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

"""Active RL action-space tests: degenerate dims must be unreachable by exploration.

These guard the SAC-side invariant that matters most for this task -- that entropy-driven
exploration can never perturb the left-hand grip on the tube, the base height, or the
navigation velocities, all of which are constant in the demonstration data.
"""

import pytest
import torch

from rlinf.envs.isaaclab.tasks.g1_piston_rl_space import (
    ACTION_DIM,
    ACTIVE_ACTION_DIMS,
    FROZEN_ACTION_DIMS,
    NUM_ACTIVE_DIMS,
    apply_action_mask,
    build_active_mask,
    default_target_entropy,
    load_frozen_values,
    masked_sum,
    verify_mask_against_statistics,
)

DATASET_STATS = (
    "/home/jren313/research/starvla_rl/checkpoints/"
    "g1-longhorizon-oft-v1/dataset_statistics.json"
)


def test_active_and_frozen_partition_the_action_space():
    assert set(ACTIVE_ACTION_DIMS) | set(FROZEN_ACTION_DIMS) == set(range(ACTION_DIM))
    assert not (set(ACTIVE_ACTION_DIMS) & set(FROZEN_ACTION_DIMS))
    assert NUM_ACTIVE_DIMS == 20


def test_frozen_dims_match_the_checkpoint_statistics():
    """The declared split must equal the dataset's degenerate dims, not approximate it."""
    report = verify_mask_against_statistics(DATASET_STATS)
    assert report["match"], report
    assert report["statistics_degenerate"] == list(FROZEN_ACTION_DIMS)


def test_frozen_values_are_the_demonstration_constants():
    fv = load_frozen_values(DATASET_STATS)
    # left hand grips the tube fully curled; thumb pitch/yaw; base height
    assert fv[14] == pytest.approx(1.70)
    assert fv[17] == pytest.approx(1.70)
    assert fv[18] == pytest.approx(0.35)
    assert fv[19] == pytest.approx(0.25)
    assert fv[26] == pytest.approx(0.76)
    # navigation velocities are exactly zero in every frame
    for d in (27, 28, 29):
        assert fv[d] == pytest.approx(0.0)


def test_mask_shape_and_count():
    mask = build_active_mask()
    assert mask.shape == (ACTION_DIM,)
    assert int(mask.sum()) == NUM_ACTIVE_DIMS
    for d in FROZEN_ACTION_DIMS:
        assert not bool(mask[d])
    for d in ACTIVE_ACTION_DIMS:
        assert bool(mask[d])


def test_exploration_noise_cannot_reach_frozen_dims():
    """The core safety property: arbitrary sampled noise never survives on frozen dims."""
    fv = load_frozen_values(DATASET_STATS)
    mask = build_active_mask()

    torch.manual_seed(0)
    noisy = torch.randn(64, ACTION_DIM) * 10.0  # deliberately violent exploration
    out = apply_action_mask(noisy, fv, mask)

    for d in FROZEN_ACTION_DIMS:
        assert torch.allclose(out[:, d], fv[d].expand(64)), f"dim {d} drifted"
    # active dims must pass through untouched
    act = list(ACTIVE_ACTION_DIMS)
    assert torch.allclose(out[:, act], noisy[:, act])


def test_mask_preserves_chunk_and_batch_dims():
    fv = load_frozen_values(DATASET_STATS)
    mask = build_active_mask()
    out = apply_action_mask(torch.randn(4, 30, ACTION_DIM), fv, mask)
    assert out.shape == (4, 30, ACTION_DIM)
    for d in FROZEN_ACTION_DIMS:
        assert torch.allclose(out[..., d], fv[d].expand(4, 30))


def test_masked_sum_excludes_frozen_dims():
    """log-prob / entropy must not accumulate contributions from deterministic dims."""
    mask = build_active_mask()
    per_dim = torch.ones(8, ACTION_DIM)
    assert torch.allclose(masked_sum(per_dim, mask), torch.full((8,), float(NUM_ACTIVE_DIMS)))

    # a huge value on a frozen dim must not leak into the sum
    per_dim[:, FROZEN_ACTION_DIMS[0]] = 1e6
    assert torch.allclose(masked_sum(per_dim, mask), torch.full((8,), float(NUM_ACTIVE_DIMS)))


def test_target_entropy_uses_active_dims_only():
    """Alpha tuning must not chase entropy in dims that cannot move."""
    assert default_target_entropy() == -20.0
    assert default_target_entropy() != -float(ACTION_DIM)


def test_apply_mask_rejects_wrong_dim():
    fv = load_frozen_values(DATASET_STATS)
    mask = build_active_mask()
    with pytest.raises(ValueError, match="expected trailing dim 30"):
        apply_action_mask(torch.zeros(2, 53), fv, mask)


def test_declared_active_dims_have_demonstration_variation():
    """Guards against asking SAC to explore a dim with no data support."""
    # verify_mask_against_statistics raises if an active dim is degenerate
    verify_mask_against_statistics(DATASET_STATS)
