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

"""Chunk-boundary blending must remove the discontinuity without rewriting the policy.

Guards ``docs/contracts/g1_piston_chunk_boundary_jitter.json``: the commanded joint
trajectory stepped 21x further at every 30-step chunk boundary than within a chunk, with
a worst single-step jump of 1.0 rad on the Inspire hand joints.
"""

import numpy as np
import pytest

from rlinf.envs.isaaclab.tasks.g1_piston_chunk_blend import (
    DEFAULT_BLEND_STEPS,
    blend_chunk,
    boundary_stats,
)


def test_first_chunk_of_an_episode_is_untouched():
    """With no previous command there is no discontinuity to remove."""
    chunk = np.random.default_rng(0).normal(size=(30, 53))
    assert np.array_equal(blend_chunk(chunk, None), chunk)


def test_blending_is_disabled_by_zero_steps():
    """The default must be reproducible-off, so existing results cannot change silently."""
    chunk = np.random.default_rng(1).normal(size=(30, 53))
    prev = np.zeros(53)
    assert np.array_equal(blend_chunk(chunk, prev, blend_steps=0), chunk)


def test_the_boundary_jump_is_reduced():
    """The defect itself: a 1.0 rad single-step jump must be spread out."""
    prev = np.zeros(4)
    chunk = np.ones((30, 4))          # a full 1.0 rad step at the boundary
    out = blend_chunk(chunk, prev, blend_steps=6)
    first_step = np.abs(out[0] - prev).max()
    assert first_step < 0.2, first_step
    # and every step inside the ramp stays below the observed intra-chunk maximum
    traj = np.vstack([prev[None, :], out])
    assert np.abs(np.diff(traj, axis=0)).max() < 0.29


def test_the_policy_trajectory_is_executed_after_the_ramp():
    """Blending must not rewrite the policy's intent, only its approach."""
    prev = np.zeros(4)
    chunk = np.random.default_rng(2).normal(size=(30, 4))
    out = blend_chunk(chunk, prev, blend_steps=6)
    assert np.array_equal(out[6:], chunk[6:])
    assert not np.array_equal(out[:6], chunk[:6])


def test_a_continuous_chunk_is_barely_modified():
    """If the policy is already continuous, blending must be close to a no-op."""
    prev = np.zeros(3)
    chunk = np.linspace(0.0, 0.03, 30)[:, None] * np.ones((1, 3))  # smooth from prev
    out = blend_chunk(chunk, prev, blend_steps=6)
    assert np.abs(out - chunk).max() < 0.02


def test_weights_are_monotone_and_strictly_inside_the_endpoints():
    """Weight 0 would repeat a stale command; weight 1 would leave the jump in place."""
    prev = np.zeros(1)
    chunk = np.ones((30, 1))
    out = blend_chunk(chunk, prev, blend_steps=6).ravel()[:6]
    assert np.all(np.diff(out) > 0)
    assert out[0] > 0.0 and out[-1] < 1.0


def test_blend_steps_longer_than_the_chunk_is_clamped():
    chunk = np.ones((4, 2))
    out = blend_chunk(chunk, np.zeros(2), blend_steps=99)
    assert out.shape == (4, 2)
    assert np.all(np.isfinite(out))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        blend_chunk(np.ones((30, 53)), np.zeros(7))


def test_the_input_chunk_is_not_mutated():
    chunk = np.ones((30, 3))
    before = chunk.copy()
    blend_chunk(chunk, np.zeros(3), blend_steps=6)
    assert np.array_equal(chunk, before)


def test_boundary_stats_detects_the_defect_and_the_fix():
    """The diagnostic must report the real ratio, so a fix is checkable against it."""
    rng = np.random.default_rng(3)
    # Two chunks: smooth inside, with a large jump between them.
    a = np.cumsum(rng.normal(scale=0.005, size=(30, 2)), axis=0)
    b = a[-1] + 1.0 + np.cumsum(rng.normal(scale=0.005, size=(30, 2)), axis=0)
    broken = np.vstack([a, b])
    st = boundary_stats(broken, chunk_starts=[0, 30])
    assert st["ratio"] > 10, st
    fixed = np.vstack([a, blend_chunk(b, a[-1], blend_steps=6)])
    st2 = boundary_stats(fixed, chunk_starts=[0, 30])
    assert st2["boundary_max"] < st["boundary_max"] / 4
    assert st2["ratio"] < st["ratio"] / 4


def test_default_blend_window_is_short_relative_to_the_chunk():
    """The ramp must not consume the chunk: most steps are the policy's own."""
    assert 0 < DEFAULT_BLEND_STEPS <= 30 // 4
