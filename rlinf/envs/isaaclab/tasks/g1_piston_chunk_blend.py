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

"""Continuity at action-chunk boundaries.

The controller is open-loop inside a chunk: the VLA predicts ``H`` actions from one
observation, all ``H`` execute, then a fresh observation produces a new chunk. Nothing
constrains the new chunk's first action to be continuous with the previous chunk's last,
so the command teleports. Measured on ``rlpd_ckpt_step415140`` over one full episode:

===========================  ==========
commanded step, intra-chunk  0.0054 rad
commanded step, at boundary  0.1161 rad   (21x)
worst single-step jump       1.0000 rad
===========================  ==========

The worst offenders are the Inspire hand joints, which snap ~0.9 rad in one 50 Hz step,
23 times per episode. The measured joints do not track the spike -- command error triples
at boundaries -- so the arm chases a target it cannot reach, and a finger snap of that
size during contact can knock or drop the object.

This is **not** how the data behaves. Across the demonstrations, the mean action step at
every-30th index is 0.94x the step elsewhere: the demonstrated trajectories are smooth
and have no chunk structure at all, because chunking is imposed at inference time.
Blending therefore *restores* a property of the training data rather than inventing one.

What this module is and is not
------------------------------

It is an **execution-layer** change: it rewrites the commanded trajectory only. It does
not touch the policy, the reward, the frozen metrics, the evaluation suite, or the action
space, and it is disabled by default so no existing result changes silently. Enabling it
produces a *different execution mode*, and must be reported as its own arm.
"""

from __future__ import annotations

import numpy as np

#: Control steps over which a new chunk ramps in from the previous chunk's last command.
#: 6 steps at 50 Hz is 120 ms -- long enough to spread a 1 rad jump into ~0.17 rad steps
#: (below the observed intra-chunk maximum of 0.29), short enough that the policy's
#: intent for the chunk is still executed over the remaining 24 of 30 steps.
DEFAULT_BLEND_STEPS = 6


def blend_chunk(new_chunk, last_command, blend_steps: int = DEFAULT_BLEND_STEPS):
    """Ramp ``new_chunk`` in from ``last_command`` over ``blend_steps``.

    Parameters
    ----------
    new_chunk:
        ``(H, D)`` commanded trajectory for the upcoming chunk.
    last_command:
        ``(D,)`` command actually issued at the final step of the previous chunk, or
        ``None`` for the first chunk of an episode (returned unchanged).
    blend_steps:
        Number of leading steps to blend. ``0`` disables blending entirely.

    Returns
    -------
    A ``(H, D)`` array. Step ``i < blend_steps`` is a convex combination
    ``(1 - w) * last_command + w * new_chunk[i]`` with ``w`` rising linearly from
    ``1/(blend_steps+1)`` to ``blend_steps/(blend_steps+1)``; later steps are untouched,
    so the policy's own trajectory is executed exactly for the rest of the chunk.

    The weights deliberately never reach 0 or 1 at the endpoints: weight 0 would repeat
    the stale command for a step, and the ramp is meant to *approach* the policy's
    trajectory, which it meets at step ``blend_steps``.
    """
    out = np.array(new_chunk, dtype=np.float64, copy=True)
    if last_command is None or blend_steps <= 0:
        return out
    n = min(int(blend_steps), out.shape[0])
    prev = np.asarray(last_command, dtype=np.float64)
    if prev.shape != out.shape[1:]:
        raise ValueError(f"last_command shape {prev.shape} != chunk dim {out.shape[1:]}")
    for i in range(n):
        w = (i + 1) / (n + 1)
        out[i] = (1.0 - w) * prev + w * out[i]
    return out


def boundary_stats(commands, chunk_starts):
    """Diagnostic: mean absolute command step within chunks vs at chunk boundaries.

    ``commands`` is ``(T, D)``; ``chunk_starts`` are indices where a new chunk begins.
    Returns a dict with both means, their ratio, and the worst boundary step -- the same
    quantities used to characterise the defect, so a fix can be checked against them.
    """
    c = np.asarray(commands, dtype=np.float64)
    if c.shape[0] < 2:
        return {}
    d = np.abs(np.diff(c, axis=0))
    starts = set(int(s) for s in chunk_starts)
    inter = [i for i in range(c.shape[0] - 1) if (i + 1) in starts]
    intra = [i for i in range(c.shape[0] - 1) if (i + 1) not in starts]
    m_inter = float(d[inter].mean()) if inter else 0.0
    m_intra = float(d[intra].mean()) if intra else 0.0
    return {
        "intra_chunk_mean": round(m_intra, 6),
        "boundary_mean": round(m_inter, 6),
        "ratio": round(m_inter / m_intra, 3) if m_intra > 0 else None,
        "boundary_max": round(float(d[inter].max()), 6) if inter else 0.0,
        "intra_max": round(float(d[intra].max()), 6) if intra else 0.0,
        "n_boundaries": len(inter),
    }
