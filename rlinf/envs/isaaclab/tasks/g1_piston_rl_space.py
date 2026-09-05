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

import math

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

    -8.4 was chosen as "the log-density at exploration std ~0.20". Measured against the
    trainer's own log-probability computation, that is **wrong for both noise branches**,
    and wrong in different directions:

    ==================  ==========================  ==========================
    exploration std     correlated-noise branch     i.i.d. branch
    ==================  ==========================  ==========================
    0.05                -8.46                       +31.76
    0.20 (documented)   -7.73                       +7.36
    0.30                -6.79                       +3.12
    ==================  ==========================  ==========================

    So -8.4 needs std ~0.05 under correlated noise, and is unreachable at any std under
    i.i.d. noise, where this quantity is positive throughout. The branches differ by
    ~15 nats at the same std, so **no single scalar can be correct for both**.

    Worse, -8.4 sits in the correlated branch's FLAT region: below std 0.10 the curve
    moves only 3 nats per unit std, so the target barely constrains std and alpha must
    grow without bound to move entropy at all. Run 4 measured exactly that -- alpha rose
    57%, the entropy term averaged 28% of the actor objective for the whole run, and a
    policy with v3 grasp 1.00 was flattened to 0.00. See
    ``docs/contracts/g1_piston_entropy_target_miscalibrated.json``.

    ``default_target_entropy`` is kept for callers that want the historical constant,
    but new runs should use :func:`calibrated_target_entropy`, which MEASURES the target
    for the branch actually in use.
    """
    return -8.4


#: Exploration std the calibrated target aims at. 0.25 rather than 0.20: measured slope
#: at 0.20 is 6.8 nats per unit std and at 0.25 it is 8.5, so alpha has more leverage
#: while the policy stays close to the demonstrations. Below 0.10 the curve is flat
#: (slope ~3) and the target stops constraining std, which is the run-4 failure.
CALIBRATED_TARGET_STD = 0.25

#: Minimum slope, in nats per unit std, for a target to actually constrain exploration.
#: Below this the alpha update has no leverage and diverges instead of regulating.
MIN_TARGET_SLOPE = 5.0


def iid_logprob(dist, flat_sample, mask, entropy_space="latent"):
    """Per-control-action log-density for the i.i.d. (SquashedNormal) branch.

    Shared by the trainer and the calibration. ``latent`` is the density of the
    pre-squash Gaussian sample; ``executed`` applies the affine and tanh change of
    variables, exactly as the trainer's original ``masked_logprob`` did.
    """
    import torch

    base = dist.base_dist.base_dist
    parts = []
    for t in dist.transforms:
        parts.extend(getattr(t, "parts", [t]))
    x = flat_sample
    for t in reversed(parts):
        x = t.inv(x)
    per_dim = base.log_prob(x)
    if entropy_space == "executed":
        y = x
        for t in parts:
            y2 = t(y)
            if type(t).__name__ == "TanhTransform":
                per_dim = per_dim - torch.log(1 - y2.pow(2) + 1e-7)
            else:
                sc = torch.as_tensor(t.scale, dtype=x.dtype, device=x.device)
                per_dim = per_dim - torch.log(sc.abs()).expand_as(x)
            y = y2
    elif entropy_space != "latent":
        raise ValueError(f"entropy_space must be 'latent' or 'executed', got "
                         f"{entropy_space!r}")
    return (per_dim * mask.to(per_dim.dtype)).sum(dim=-1)


def measure_logprob_at_std(std, correlated=True, n=400, seed=0, horizon=30, dims=30,
                           entropy_space="latent", legacy=False):
    """Measure log-density per control action at a given exploration std, mean 0.

    Uses the SAME implementation the trainer uses (:meth:`CorrelatedChunkNoise.
    logprob_chunk` / :func:`iid_logprob`), so the number is comparable to the
    ``logprob_per_step`` a run reports. ``legacy=True`` reproduces the formula runs 1-5
    trained under, for reproduction only.
    """
    import torch

    mask = build_active_mask()
    torch.manual_seed(seed)
    logstd = torch.full((dims,), float(math.log(float(std))))
    if correlated:
        from rlinf.envs.isaaclab.tasks.g1_piston_correlated_policy import (
            CorrelatedChunkNoise,
            N_BASIS,
        )

        noise = CorrelatedChunkNoise(horizon=horizon, dims=dims, device="cpu")
        z = torch.randn(n, N_BASIS, dims)
        pre = noise.expand(z, torch.exp(logstd))
        if legacy:
            lp = noise.logprob_chunk_legacy(z, pre, mask)
        else:
            lp = noise.logprob_chunk(z, pre, logstd, mask, entropy_space)
        return float(lp.mean())

    from rlinf.models.embodiment.modules.gaussian_policy import SquashedNormal

    dist = SquashedNormal(torch.zeros(n, dims), torch.exp(logstd).expand(n, dims),
                          low=-2.2, high=2.2)
    sample = dist.rsample()
    return float(iid_logprob(dist, sample, mask,
                             "executed" if legacy else entropy_space).mean())


def calibrated_target_entropy(correlated=True, std=None, check_slope=True,
                              entropy_space="latent"):
    """Target log-density MEASURED for the noise branch actually in use.

    The scalar in :func:`default_target_entropy` was calibrated against neither branch
    and cost run 4 a working policy. This measures the target instead of asserting it,
    so the target and the exploration std it claims to represent cannot drift apart.

    Raises ``ValueError`` when the requested std sits in a flat region of the curve,
    where the target would not constrain exploration and alpha would diverge. That is a
    real failure mode, not a hypothetical: it is what run 4 did.
    """
    std = float(CALIBRATED_TARGET_STD if std is None else std)
    target = measure_logprob_at_std(std, correlated=correlated,
                                    entropy_space=entropy_space)
    if check_slope:
        delta = max(0.05 * std, 0.01)
        lo = measure_logprob_at_std(std - delta, correlated=correlated,
                                    entropy_space=entropy_space)
        hi = measure_logprob_at_std(std + delta, correlated=correlated,
                                    entropy_space=entropy_space)
        slope = (hi - lo) / (2 * delta)
        if abs(slope) < MIN_TARGET_SLOPE:
            raise ValueError(
                f"target at std {std} sits on slope {slope:.1f} nats per unit std, "
                f"below MIN_TARGET_SLOPE={MIN_TARGET_SLOPE}. The alpha update would "
                "have no leverage over exploration and would diverge, as in run 4. "
                "Choose an std on a steeper part of the curve.")
    return target
