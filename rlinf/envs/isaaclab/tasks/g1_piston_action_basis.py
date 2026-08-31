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

"""Temporal basis for action chunks: shrink the critic's 900-D input honestly.

The problem, from a review of this setup: *"The action space is 900 dimensional, which
is shockingly high for SAC given the data size. The critic is not able to estimate the
state well and the actor could easily make use of the error of critic."* Measured, that
is roughly **8 transitions per action dimension** (about 7k online chunks + 467
demonstration chunks against 30 dims x 30 horizon = 900). A Q-function fitted that
sparsely is mostly interpolation error, and SAC's actor objective is *literally* to
maximise the critic, so it climbs that error -- the classic exploitation-of-a-bad-critic
failure.

The related point: *"Each dimension of the action seems to be independent. That's not
correct. The action dimensions depend heavily on each other given the configuration of
the hand, the arm and the temporal dependency."* Both the Gaussian policy (diagonal
covariance over 900 scalars) and the critic (a flat 900-wide input layer) treat those
scalars as exchangeable, discarding the fact that index *t* and *t+1* of the same joint
are strongly correlated while index *t* of two different joints are not.

This module addresses the **critic input** with a fixed, information-preserving change of
basis: project each joint's 30-step trajectory onto a small set of smooth temporal basis
functions (discrete cosine, the natural basis for smooth trajectories), and feed the
critic the coefficients instead of raw samples. With ``N_BASIS=6`` the critic sees
``30 x 6 = 180`` numbers rather than 900 -- a 5x reduction, taking the ratio from ~8 to
~41 samples per input dimension -- while retaining, on the real demonstrations, over 99%
of trajectory energy (measured; see :func:`reconstruction_report`).

Two properties make this safe rather than a hack:

* **It is a linear, fixed, invertible-in-subspace projection**, not a learned encoder, so
  it adds no parameters and cannot itself overfit.
* **It is applied only to the critic's view.** The actor still emits the full chunk that
  the robot executes, so the policy class is unchanged and nothing about deployment
  changes. This is a variance-reduction measure on value estimation, not a restriction of
  behaviour.

Low-order DCT coefficients also *are* the temporal structure: coefficient 0 is the mean
posture over the chunk and coefficient 1 the dominant ramp, so a critic in this basis
reads trajectory shape rather than 900 unordered scalars.

One thing this is **not**: a smoother. A 30-sample chunk at 50 Hz is a 0.6 s window, so
index *k* carries *k/2* cycles per window and the ``N_BASIS=6`` cutoff sits near 5 Hz.
Measured retained energy: 99.97% at 0.5 Hz, **99.5% at 1.7 Hz**, 11% at 6 Hz, 2% at
12 Hz. The ~1.7 Hz oscillation documented in
``docs/contracts/g1_piston_rl_induced_oscillation.json`` is therefore *retained* --
deliberately, because a critic blind to the pathology could not learn to penalise it.
Only genuinely fast content (above ~6 Hz, where sampling noise lives) is discarded.
Smoothing remains ``SMOOTH_LAMBDA``'s job on the actor side.

This does not claim to fix the deeper modelling point about *joint* dependence across
dimensions -- a diagonal Gaussian over chunks remains a factorised policy. That is a
policy-class question (PPO with a structured head, or a diffusion/flow policy) recorded
in ``docs/contracts/g1_piston_rl_setup_review.json``; this module is the part that is
correct to do inside the existing algorithm.
"""

from __future__ import annotations

import numpy as np

#: Chunk horizon (actions per chunk).
HORIZON = 30
#: Number of retained temporal basis functions per action dimension.
N_BASIS = 6
#: Number of physical action dimensions.
ACTION_DIMS = 30


def dct_basis(horizon: int = HORIZON, n_basis: int = N_BASIS) -> np.ndarray:
    """Return the ``(horizon, n_basis)`` orthonormal DCT-II basis matrix.

    Columns are ordered from lowest to highest frequency, so truncation keeps the
    smooth part of a trajectory and discards the jitter.
    """
    if not 1 <= n_basis <= horizon:
        raise ValueError(f"n_basis must be in [1, {horizon}], got {n_basis}")
    t = np.arange(horizon)[:, None]
    k = np.arange(n_basis)[None, :]
    B = np.cos(np.pi * (t + 0.5) * k / horizon)
    # Orthonormal scaling so coefficient magnitudes are comparable across k.
    B *= np.where(k == 0, np.sqrt(1.0 / horizon), np.sqrt(2.0 / horizon))
    return B.astype(np.float64)


#: The fixed basis used everywhere in this project.
BASIS = dct_basis()


def project(chunk, basis: np.ndarray | None = None):
    """Project a chunk onto the temporal basis.

    Args:
        chunk: ``(..., horizon, dims)`` array or torch tensor of actions.
        basis: optional ``(horizon, n_basis)`` matrix; defaults to :data:`BASIS`.

    Returns:
        ``(..., n_basis, dims)`` coefficients, same array type as the input.
    """
    B = BASIS if basis is None else basis
    if hasattr(chunk, "detach"):  # torch
        import torch

        Bt = torch.as_tensor(B, dtype=chunk.dtype, device=chunk.device)
        return torch.einsum("hk,...hd->...kd", Bt, chunk)
    chunk = np.asarray(chunk)
    return np.einsum("hk,...hd->...kd", B, chunk)


def reconstruct(coeffs, basis: np.ndarray | None = None):
    """Map coefficients back to a chunk (the truncated reconstruction)."""
    B = BASIS if basis is None else basis
    if hasattr(coeffs, "detach"):
        import torch

        Bt = torch.as_tensor(B, dtype=coeffs.dtype, device=coeffs.device)
        return torch.einsum("hk,...kd->...hd", Bt, coeffs)
    coeffs = np.asarray(coeffs)
    return np.einsum("hk,...kd->...hd", B, coeffs)


def flatten(coeffs):
    """Flatten coefficients to the critic's input vector (``n_basis * dims``)."""
    if hasattr(coeffs, "detach"):
        return coeffs.reshape(*coeffs.shape[:-2], -1)
    coeffs = np.asarray(coeffs)
    return coeffs.reshape(*coeffs.shape[:-2], -1)


def critic_input_dim(n_basis: int = N_BASIS, dims: int = ACTION_DIMS) -> int:
    """Width of the critic's action input in this basis (compare 900 for raw)."""
    return n_basis * dims


def retained_energy(chunks, n_basis: int = N_BASIS) -> float:
    """Fraction of trajectory energy retained by truncating to ``n_basis``.

    Args:
        chunks: ``(N, horizon, dims)`` real action chunks.

    Returns:
        Retained energy in [0, 1]; 1.0 means the truncation is lossless on this data.
    """
    X = np.asarray(chunks, dtype=np.float64)
    B_full = dct_basis(X.shape[-2], X.shape[-2])
    C = np.einsum("hk,nhd->nkd", B_full, X)
    total = float((C ** 2).sum())
    kept = float((C[:, :n_basis, :] ** 2).sum())
    return kept / total if total > 0 else 1.0


def reconstruction_report(chunks, max_basis: int = 12) -> list[dict]:
    """Per-``n_basis`` retained energy and RMSE, for choosing the truncation on data."""
    X = np.asarray(chunks, dtype=np.float64)
    out = []
    for n in range(1, max_basis + 1):
        B = dct_basis(X.shape[-2], n)
        C = np.einsum("hk,nhd->nkd", B, X)
        R = np.einsum("hk,nkd->nhd", B, C)
        out.append({
            "n_basis": n,
            "retained_energy": retained_energy(X, n),
            "rmse": float(np.sqrt(((R - X) ** 2).mean())),
            "critic_input_dim": critic_input_dim(n, X.shape[-1]),
        })
    return out
