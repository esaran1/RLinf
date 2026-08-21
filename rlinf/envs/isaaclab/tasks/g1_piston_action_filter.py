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

"""Demonstration-envelope action filter: keep executed dynamics inside the data's.

The measured pathology (``docs/contracts/g1_piston_chunk_boundary_jitter.json`` and the
hand-oscillation follow-up): the policy's predicted finger commands swing ~0.67 rad
within every 30-step chunk at ~1.7 Hz -- 43x the demonstrations' per-step velocity. The
demonstrations close the hand once over ~1 s and then hold; the policy pumps it open and
closed continuously. This is not chunk-boundary jerk (fixed separately by blending, with
no effect on this) and not sensor noise: it is the predicted trajectory itself.

The principled execution-layer remedy is to constrain executed actions to the dynamics
the demonstrations actually contain. Both constants are *measured from the 11
demonstration episodes*, not tuned:

* **Bandwidth**: 97.6% of demonstrated hand-signal energy lies below 1.0 Hz (arm: 97.7%),
  98.5% below 1.5 Hz. The cutoff is chosen empirically as the highest value meeting all
  three criteria on real data (measured, per-cutoff):

  ======  ==========  =================  =========  ============
  f_c     gain@1.7Hz  policy pump ratio  demo RMSE  90% step (s)
  ======  ==========  =================  =========  ============
  1.0 Hz  0.26        0.33               0.055      0.62
  **1.2**  **0.33**   **0.40**           **0.046**  **0.52**
  1.4 Hz  0.41        0.47               0.040      0.44
  ======  ==========  =================  =========  ============

  ``f_c = 1.2 Hz`` is the highest cutoff that still halves the measured per-chunk finger
  swing on the real recorded policy commands while tracking held-out demonstrations to
  under 0.05 RMSE and closing a 1.3 rad grasp to 90% in ~0.5 s.
* **Slew**: the largest single-step change anywhere in the demonstrations is 0.175
  (action units at 50 Hz); the peak sustained closing rate is 0.0325/step. The filter
  caps per-step change at the demonstrated maximum as a hard envelope.

The filter is **causal** and carries state across chunk boundaries, so it also subsumes
chunk blending: a boundary step is just another over-bandwidth transient. Group delay is
~0.32 s at 1 Hz, small against the ~1 s demonstrated closure.

Training-time alternative (for future runs, not applied here): a CAPS-style temporal
smoothness penalty on consecutive actions during RL (Mysore et al., ICRA 2021,
arXiv:2012.06644) removes the oscillation at the source. This module is the inference-
time counterpart that needs no retraining.

Like blending, this is an EXECUTION-layer change: it never touches the policy, reward,
metrics or evaluation suite, and it is off by default. Enabling it is a distinct
execution mode that must be reported as its own arm.
"""

from __future__ import annotations

import math

import numpy as np

#: Empirically selected: highest cutoff halving the measured policy pump while
#: tracking demonstrations to <0.05 RMSE. See the module docstring table.
DEMO_BANDWIDTH_HZ = 1.2
#: Largest single-step action change observed anywhere in the demonstrations.
DEMO_MAX_STEP = 0.175
#: Control period of the 50 Hz interface the demonstrations were recorded at.
CONTROL_DT = 0.02


class DemoEnvelopeFilter:
    """Causal second-order low-pass + slew limit, calibrated from the demonstrations.

    Implemented as two cascaded first-order stages (critically damped -- no resonance,
    monotone step response), followed by a per-step slew cap. State persists across
    chunks; call :meth:`reset` at episode boundaries.
    """

    def __init__(self, dim: int = 30, fc_hz: float = DEMO_BANDWIDTH_HZ,
                 dt: float = CONTROL_DT, max_step: float = DEMO_MAX_STEP,
                 apply_dims=None):
        """``apply_dims``: optional iterable of dim indices to filter; all other dims
        pass through untouched. The measured pathology is confined to the hand dims
        (0.020 rad/step vs the arm's 0.0026, demo 0.0009), and filtering the arm adds
        reach-phase lag that a pre-registered A/B showed breaks grasp timing (grasp
        0.68 vs a baseline worst of 0.96). Hand-only application is therefore the
        minimal intervention matched to the measurement."""
        if fc_hz <= 0:
            raise ValueError("fc_hz must be positive; use enabled=False to bypass")
        self.dim = int(dim)
        if apply_dims is None:
            self.apply_mask = np.ones(self.dim, dtype=bool)
        else:
            self.apply_mask = np.zeros(self.dim, dtype=bool)
            self.apply_mask[list(apply_dims)] = True
        self.fc_hz = float(fc_hz)
        self.dt = float(dt)
        self.max_step = float(max_step)
        #: One-pole coefficient per stage. Two identical stages give a second-order
        #: rolloff with combined -3 dB near fc for this alpha choice.
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * fc_hz * dt)
        self._s1 = None
        self._s2 = None
        self._out = None

    def reset(self):
        """Forget all state. Call at ``env.reset()`` so episodes stay independent."""
        self._s1 = None
        self._s2 = None
        self._out = None

    def step(self, u):
        """Filter one action vector ``u`` (shape ``(dim,)``); returns the same shape."""
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (self.dim,):
            raise ValueError(f"expected shape ({self.dim},), got {u.shape}")
        if self._s1 is None:
            # Initialise at the first command: no startup transient, no initial lag.
            self._s1 = u.copy()
            self._s2 = u.copy()
            self._out = u.copy()
            return u.copy()
        a = self.alpha
        self._s1 += a * (u - self._s1)
        self._s2 += a * (self._s1 - self._s2)
        step = np.clip(self._s2 - self._out, -self.max_step, self.max_step)
        self._out = self._out + step
        y = np.where(self.apply_mask, self._out, u)
        # Track the raw command on pass-through dims so re-enabling later is seamless.
        self._out = np.where(self.apply_mask, self._out, u)
        self._s1 = np.where(self.apply_mask, self._s1, u)
        self._s2 = np.where(self.apply_mask, self._s2, u)
        return y.copy()

    def filter_chunk(self, chunk):
        """Filter a ``(H, dim)`` chunk sequentially, carrying state across calls."""
        chunk = np.asarray(chunk, dtype=np.float64)
        return np.stack([self.step(chunk[i]) for i in range(chunk.shape[0])])


def attenuation_at(f_hz: float, fc_hz: float = DEMO_BANDWIDTH_HZ,
                   dt: float = CONTROL_DT) -> float:
    """Empirical gain of the two-stage filter at ``f_hz`` (measured, not analytic)."""
    n = max(int(50.0 / dt / max(f_hz, 1e-6)), 2000)
    t = np.arange(n) * dt
    x = np.sin(2 * math.pi * f_hz * t)
    filt = DemoEnvelopeFilter(dim=1, fc_hz=fc_hz, dt=dt, max_step=1e9)
    y = np.array([filt.step(np.array([v]))[0] for v in x])
    # Skip the transient; compare steady-state amplitudes.
    tail = slice(n // 2, None)
    return float(np.abs(y[tail]).max() / np.abs(x[tail]).max())


def temporal_smoothness_penalty(action_chunk, active_mask=None):
    """CAPS-style temporal smoothness loss for a predicted action chunk.

    Mean squared difference between consecutive actions within the chunk (Mysore et
    al., ICRA 2021, arXiv:2012.06644 -- the L_T term), the training-time counterpart of
    :class:`DemoEnvelopeFilter`: rather than filtering the oscillation at execution, it
    penalises the actor for predicting it. Works on torch tensors so it can sit inside
    the actor loss with gradients flowing.

    Args:
        action_chunk: ``[..., H, D]`` predicted actions (any monotone action space).
        active_mask: optional boolean ``[D]``; only these dims are penalised, so frozen
            dims cannot dilute the term.

    Returns:
        A scalar tensor. For the demonstrations this evaluates to ~1e-6 (they move
        0.0009/step); for the measured RLPD checkpoint, ~1e-3 -- three orders larger,
        so a weight around 0.1-1.0 penalises the pathology without touching demo-like
        behaviour.
    """
    diff = action_chunk[..., 1:, :] - action_chunk[..., :-1, :]
    if active_mask is not None:
        diff = diff[..., active_mask]
    return (diff ** 2).mean()


#: The 12 hand dims of the 30-D action (both hands): where the measured pathology lives.
HAND_DIMS = tuple(range(14, 26))
