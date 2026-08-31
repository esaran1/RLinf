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

"""The temporal basis must shrink the critic's input without discarding behaviour.

Guards the review point that a 900-D action chunk is far too wide for the ~7k
transitions available: the critic cannot fit it, and SAC's actor maximises whatever
error the critic makes.
"""

import glob
import os

import numpy as np
import pytest

from rlinf.envs.isaaclab.tasks.g1_piston_action_basis import (
    ACTION_DIMS,
    BASIS,
    HORIZON,
    N_BASIS,
    critic_input_dim,
    dct_basis,
    flatten,
    project,
    reconstruct,
    reconstruction_report,
    retained_energy,
)

SCRATCH = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
           "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")


def _demo_chunks():
    out = []
    for fp in sorted(glob.glob(os.path.join(SCRATCH, "act_ep*.npy"))):
        A = np.load(fp)
        for c in range(len(A) // HORIZON):
            out.append(A[c * HORIZON:(c + 1) * HORIZON])
    return np.stack(out) if out else None


def test_basis_is_orthonormal():
    B = dct_basis()
    G = B.T @ B
    assert np.allclose(G, np.eye(B.shape[1]), atol=1e-10), G


def test_full_basis_is_lossless():
    """With all 30 components the transform must be exactly invertible."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, HORIZON, ACTION_DIMS))
    B = dct_basis(HORIZON, HORIZON)
    assert np.allclose(reconstruct(project(x, B), B), x, atol=1e-9)


def test_truncation_shrinks_the_critic_input_fivefold():
    assert critic_input_dim() == N_BASIS * ACTION_DIMS == 180
    assert HORIZON * ACTION_DIMS == 900
    assert 900 / critic_input_dim() == pytest.approx(5.0)


def test_projection_shapes():
    x = np.zeros((7, HORIZON, ACTION_DIMS))
    c = project(x)
    assert c.shape == (7, N_BASIS, ACTION_DIMS)
    assert flatten(c).shape == (7, N_BASIS * ACTION_DIMS)


def test_low_order_coefficient_is_the_chunk_mean():
    """Coefficient 0 must be (a scaling of) the mean posture, so the critic's first
    180/30 numbers carry the chunk's average configuration."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, HORIZON, ACTION_DIMS))
    c = project(x)[0, 0]
    expected = x[0].mean(axis=0) * np.sqrt(HORIZON)
    assert np.allclose(c, expected, atol=1e-9)


def test_smooth_trajectories_are_captured_by_few_components():
    """A ramp plus a slow sine -- demonstration-like motion -- must survive truncation."""
    t = np.linspace(0, 1, HORIZON)[:, None]
    x = (0.5 * t + 0.2 * np.sin(2 * np.pi * 0.5 * t)) * np.ones((1, HORIZON, ACTION_DIMS))
    assert retained_energy(x, N_BASIS) > 0.999


def test_truncation_keeps_task_bandwidth_and_drops_only_fast_jitter():
    """Pins the basis's actual frequency behaviour, measured rather than assumed.

    A 30-sample chunk at 50 Hz is a 0.6 s window, so DCT index k carries k/2 cycles per
    window and the N_BASIS=6 cutoff sits near 5 Hz. Two consequences, both wanted:

    * behaviour the task contains -- including the ~1.7 Hz oscillation the smoothness
      work characterised -- is RETAINED, so the critic still sees it. This module is
      critic conditioning, not a smoother; smoothing is SMOOTH_LAMBDA's job, and a
      critic blind to the pathology could not penalise it.
    * genuinely fast content (>~6 Hz) is discarded, which is where sampling noise lives.
    """
    t = np.arange(HORIZON) * 0.02

    def amp(f):
        x = np.sin(2 * np.pi * f * t)[None, :, None] * np.ones((1, HORIZON, ACTION_DIMS))
        return retained_energy(x, N_BASIS)

    assert amp(0.5) > 0.99      # demonstration bandwidth: kept
    assert amp(1.7) > 0.99      # the measured oscillation: kept, deliberately
    assert amp(6.0) < 0.30      # above the cutoff: mostly discarded
    assert amp(12.0) < 0.10     # fast noise: gone


def test_torch_and_numpy_paths_agree():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, HORIZON, ACTION_DIMS))
    cn = project(x)
    ct = project(torch.as_tensor(x, dtype=torch.float64)).numpy()
    assert np.allclose(cn, ct, atol=1e-10)


def test_torch_projection_is_differentiable():
    """The basis sits inside the actor->critic gradient path, so it must carry grad."""
    torch = pytest.importorskip("torch")
    x = torch.zeros(1, HORIZON, ACTION_DIMS, requires_grad=True)
    project(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_bad_basis_size_is_rejected():
    with pytest.raises(ValueError):
        dct_basis(HORIZON, 0)
    with pytest.raises(ValueError):
        dct_basis(HORIZON, HORIZON + 1)


@pytest.mark.skipif(not glob.glob(os.path.join(SCRATCH, "act_ep*.npy")),
                    reason="demonstration files not on this machine")
def test_real_demonstrations_survive_truncation():
    """The load-bearing measurement: on real demonstration chunks, 6 components must
    retain essentially all energy, so shrinking the critic's input discards jitter
    rather than behaviour."""
    X = _demo_chunks()
    assert X is not None and X.shape[1:] == (HORIZON, ACTION_DIMS)
    e = retained_energy(X, N_BASIS)
    assert e > 0.999, e
    rep = {r["n_basis"]: r for r in reconstruction_report(X, N_BASIS)}
    assert rep[N_BASIS]["rmse"] < 0.01, rep[N_BASIS]


@pytest.mark.skipif(not glob.glob(os.path.join(SCRATCH, "act_ep*.npy")),
                    reason="demonstration files not on this machine")
def test_retained_energy_is_monotone_in_n_basis():
    X = _demo_chunks()
    es = [retained_energy(X, n) for n in range(1, 9)]
    assert all(b >= a - 1e-12 for a, b in zip(es, es[1:])), es


def test_basis_constant_matches_module_default():
    assert BASIS.shape == (HORIZON, N_BASIS)
    assert np.allclose(BASIS, dct_basis())
