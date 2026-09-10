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

"""n-step returns for chunk-level replay (ResFiT uses n=3; RLPD's recipe is 1-step).

Why: run 6's critic was nearly insensitive to the action (2% of Q for a 6.5 degree change
that destroys the task; ``g1_piston_critic_exploitation.json``). The reward that matters
arrives late in a 23-chunk episode, and 1-step bootstrapping through a fresh critic
propagates it slowly. n-step targets carry the actual reward further back per update.

This module is a PURE function over an episode's transitions, so it is tested directly
and the trainer only calls it. ``n = 1`` reproduces the 1-step items exactly.

An episode is a list of transitions ``(s_t, a_t, r_t, s_{t+1}, done_t)`` where ``s``
may be any object (the trainer passes feature tuples). The n-step item for index ``t``
is ``(s_t, a_t, R_t, s_{t+m}, done_{t+m-1}, gamma**m)`` with
``m = min(n, T - t)`` and ``R_t = sum_{k<m} gamma**k r_{t+k}``, so an item that reaches
the end of the episode bootstraps from the terminal state with the episode's own done
flag and a correspondingly smaller discount power.
"""

from __future__ import annotations


def nstep_items(transitions, n: int, gamma: float):
    """Return the list of n-step items for one episode (see module docstring)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    T = len(transitions)
    out = []
    for t in range(T):
        m = min(n, T - t)
        R = 0.0
        for k in range(m):
            R += (gamma ** k) * float(transitions[t + k][2])
        s_t, a_t = transitions[t][0], transitions[t][1]
        s_next = transitions[t + m - 1][3]
        done = bool(transitions[t + m - 1][4])
        out.append((s_t, a_t, R, s_next, done, gamma ** m))
    return out
