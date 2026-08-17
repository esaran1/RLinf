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

"""Active RL action space for the G1 piston task.

The SFT policy emits 30 dims, but only 20 of them are RL-controllable. The other 10 are
degenerate in the demonstration data (``q01 == q99``): the left hand holds the tube at a
constant grip, base height is constant, and the three navigation velocities are exactly
zero. Letting SAC explore those dims would be wrong in three separate ways:

* the left hand would let go of the tube the policy is supposed to be inserting into;
* base/navigate dims are dropped by the 30->53 mapper, so noise there is invisible to
  the environment but still pollutes ``log_prob``, the entropy term, and the critic's
  input distribution;
* alpha tuning targets an entropy computed over all 30 dims, so it would chase entropy
  in dims that cannot affect the return.

The model keeps emitting all 30 dims (checkpoint compatibility is non-negotiable); this
module supplies the mask and the frozen values that are stitched back in.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

#: Policy action dimensionality (StarVLA output).
ACTION_DIM = 30

#: Dims SAC may explore: both arms (0-13) and the right hand (20-25).
ACTIVE_ACTION_DIMS = tuple(list(range(0, 14)) + list(range(20, 26)))

#: Dims held at their demonstration constant. 14-19 is the left-hand grip on the tube;
#: 26 is base height; 27-29 are the navigation velocities (dropped by the mapper).
FROZEN_ACTION_DIMS = tuple(list(range(14, 20)) + list(range(26, 30)))

#: Number of dims SAC actually controls. Target entropy should be derived from this,
#: not from ACTION_DIM.
NUM_ACTIVE_DIMS = len(ACTIVE_ACTION_DIMS)

#: Human-readable reason per frozen dim, for provenance in run logs.
FROZEN_DIM_REASONS = {
    14: "left_hand pinky - constant 1.70 rad grip on the tube",
    15: "left_hand ring - constant 1.70 rad grip on the tube",
    16: "left_hand middle - constant 1.70 rad grip on the tube",
    17: "left_hand index - constant 1.70 rad grip on the tube",
    18: "left_hand thumb pitch - constant 0.35 rad",
    19: "left_hand thumb yaw - constant 0.25 rad",
    26: "base_height - constant 0.76 m; no articulation DOF (fixed base)",
    27: "navigate vx - exactly 0.0 in all frames; dropped by the mapper",
    28: "navigate vy - exactly 0.0 in all frames; dropped by the mapper",
    29: "navigate vyaw - exactly 0.0 in all frames; dropped by the mapper",
}


def build_active_mask(device=None, dtype=torch.bool) -> torch.Tensor:
    """Return a ``[30]`` boolean mask, True where SAC may explore."""
    mask = torch.zeros(ACTION_DIM, dtype=dtype, device=device)
    mask[list(ACTIVE_ACTION_DIMS)] = True
    return mask


def load_frozen_values(dataset_statistics_path, unnorm_key: str = "new_embodiment"):
    """Return the ``[30]`` vector of demonstration constants for the frozen dims.

    Read from the SFT checkpoint's own statistics: for a degenerate dim ``q01 == q99``,
    and that shared value *is* the physical constant the demonstrations commanded. Dims
    that are not frozen come back as 0.0 and are never used.
    """
    stats = json.loads(Path(dataset_statistics_path).read_text())
    if unnorm_key not in stats:
        raise KeyError(
            f"unnorm_key {unnorm_key!r} not in dataset statistics; available: {list(stats)}"
        )
    action_stats = stats[unnorm_key]["action"]
    q01 = torch.tensor(action_stats["q01"], dtype=torch.float32)
    q99 = torch.tensor(action_stats["q99"], dtype=torch.float32)

    frozen = torch.zeros(ACTION_DIM, dtype=torch.float32)
    for dim in FROZEN_ACTION_DIMS:
        if not torch.isclose(q01[dim], q99[dim], atol=1e-8):
            raise ValueError(
                f"dim {dim} is declared frozen but is NOT degenerate in the dataset "
                f"statistics (q01={q01[dim]:.6f}, q99={q99[dim]:.6f}). The frozen-dim "
                "list and the checkpoint's statistics disagree; do not guess."
            )
        frozen[dim] = q99[dim]
    return frozen


def verify_mask_against_statistics(dataset_statistics_path, unnorm_key="new_embodiment"):
    """Check the declared active/frozen split against the checkpoint's statistics.

    Returns a dict describing the comparison. Raises if a dim declared *active* is
    actually degenerate, which would mean SAC is being asked to explore a dim with no
    demonstration support.
    """
    stats = json.loads(Path(dataset_statistics_path).read_text())
    action_stats = stats[unnorm_key]["action"]
    q01 = torch.tensor(action_stats["q01"], dtype=torch.float32)
    q99 = torch.tensor(action_stats["q99"], dtype=torch.float32)
    degenerate = torch.isclose(q01, q99, atol=1e-8)

    declared_frozen = torch.zeros(ACTION_DIM, dtype=torch.bool)
    declared_frozen[list(FROZEN_ACTION_DIMS)] = True

    active_but_degenerate = [
        d for d in ACTIVE_ACTION_DIMS if bool(degenerate[d])
    ]
    if active_but_degenerate:
        raise ValueError(
            f"dims {active_but_degenerate} are declared RL-active but are degenerate in "
            "the dataset statistics: there is no demonstration variation to learn from."
        )

    return {
        "declared_frozen": list(FROZEN_ACTION_DIMS),
        "statistics_degenerate": degenerate.nonzero(as_tuple=True)[0].tolist(),
        "match": bool(torch.equal(declared_frozen, degenerate)),
        "num_active": NUM_ACTIVE_DIMS,
    }


def apply_action_mask(
    sampled: torch.Tensor, frozen_values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Stitch frozen constants back into a sampled ``[..., 30]`` action.

    Active dims keep the sampled value; frozen dims are overwritten with the
    demonstration constant, so exploration noise can never reach them.
    """
    if sampled.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected trailing dim {ACTION_DIM}, got {sampled.shape[-1]}")
    m = mask.to(sampled.device)
    fv = frozen_values.to(device=sampled.device, dtype=sampled.dtype)
    return torch.where(m, sampled, fv.expand_as(sampled))


def masked_sum(per_dim: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum a per-dimension quantity (log-prob, entropy) over active dims only.

    Frozen dims are deterministic, so they contribute no entropy and no log-probability;
    including them would add a constant that alpha tuning would then try to optimize.
    """
    m = mask.to(per_dim.device)
    return (per_dim * m).sum(dim=-1)


#: Log-density floor of the bounded (tanh-squashed, rescaled to +/-2.2) 20-dim action
#: distribution, measured by sampling: log-prob is minimised at std ~= 0.40 and rises
#: again for both smaller and larger std, because the tanh Jacobian correction dominates
#: once the pre-squash mass runs into the saturation region.
MIN_ACHIEVABLE_LOGPROB = -13.68

#: Exploration std the target corresponds to (log-prob ~= -8.4). Chosen inside the
#: achievable range with headroom on both sides so alpha can push entropy up *or* down.
TARGET_ENTROPY_STD = 0.20


def default_target_entropy() -> float:
    """Target log-density per control action, for the SQUASHED 20-dim action space.

    **Not** the usual ``-dim(A)`` heuristic. That heuristic assumes an unbounded
    Gaussian, where log-density is unbounded below; here the action distribution is
    tanh-squashed and rescaled to ``[-2.2, 2.2]``, so its log-density is bounded below
    by ``MIN_ACHIEVABLE_LOGPROB`` (-13.68, measured).

    ``-dim(A) = -20`` is therefore **unreachable**: ``logp + target`` is negative for
    every attainable policy, so the alpha loss ``-alpha * (logp + target)`` has a
    positive gradient always and drives alpha monotonically to zero. The entropy
    regulariser dies, the actor becomes unregularised, and the policy drifts -- which is
    what both SAC pilots did (alpha 0.049 -> 0.039 while behaviour oscillated; see
    ``docs/contracts/g1_piston_sac_pilot_v1_collapse.json``).

    -8.4 corresponds to an exploration std of ~0.20: enough to explore, close enough to
    the SFT policy to stay on-distribution, and comfortably inside the achievable range
    so alpha can correct in either direction.
    """
    return -8.4
