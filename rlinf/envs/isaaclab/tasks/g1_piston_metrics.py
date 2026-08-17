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

"""FROZEN behavioural metric definitions for the G1 piston comparison.

Fixed on 2026-08-17, **before any RLPD evaluation existed** (SAC seed 1 was at episode
450/500 and ``runs/rlpd.json`` had not been created). The thresholds are derived only
from simulator trajectories and the demonstration reference -- never from which method
produced a rollout, and never from comparative performance. They must not be retuned
later in light of SAC/RLPD results.

Metric roles
------------

===============  ==========================================================
``full_success`` PRIMARY. The task's own terminal predicate.
``carry_rate``   PRIMARY. Lift accompanied by real horizontal transport.
``return``       PRIMARY, but reward-dependent: it measures optimisation of
                 a frozen reward, which is exactly what may drift out of
                 alignment with task progress.
``reach_rate``   Context.
``grasp_rate``   Context.
``lift_rate``    AUXILIARY. Superseded by ``carry_rate``: it counts vertical
                 throws as progress.
``throw_rate``   FAILURE / exploit metric. Higher is worse.
===============  ==========================================================

Why these thresholds
--------------------

The 22 executable demonstrations bound what genuine transport looks like:

* horizontal displacement 0.200-0.322 m (minimum 0.200 m),
* max lift 0.129-0.232 m,
* lift/displacement ratio 0.50-0.81.

``CARRY_MIN_DISPLACEMENT_M = 0.05`` therefore sits a factor of four below the least
transporting demonstration -- deliberately permissive, so a rollout only fails it by
barely moving the piston horizontally at all. A throw is the complement among lifting
episodes: the piston went up but did not go anywhere.
"""

from __future__ import annotations

#: Frozen on 2026-08-17, before any RLPD result existed.
METRICS_VERSION = "v1-frozen-2026-08-17"

#: Horizontal displacement (m) at or above which a lifting episode counts as genuine
#: transport. Every executable demonstration displaces at least 0.200 m, so this is 4x
#: permissive.
CARRY_MIN_DISPLACEMENT_M = 0.05

#: Demonstration reference, recorded so the threshold's provenance stays auditable.
DEMO_DISPLACEMENT_RANGE_M = (0.200, 0.322)
DEMO_MAX_LIFT_RANGE_M = (0.129, 0.232)
DEMO_LIFT_DISP_RATIO_RANGE = (0.50, 0.81)

#: Metric roles, fixed with the thresholds.
PRIMARY_METRICS = ("full_success_rate", "carry_rate", "mean_return")
AUXILIARY_METRICS = ("reach_rate", "grasp_rate", "lift_rate")
FAILURE_METRICS = ("throw_rate",)


def is_carry(row) -> bool:
    """A lifting episode that also transported the piston horizontally."""
    return bool(row["stages"].get("lift")) and row["disp_m"] >= CARRY_MIN_DISPLACEMENT_M


def is_throw(row) -> bool:
    """A lifting episode with essentially no horizontal transport.

    The reward-exploiting failure mode: the ``lift`` stage bonus pays for height alone,
    so flinging the piston upward banks it without earning any transport credit.
    """
    return bool(row["stages"].get("lift")) and row["disp_m"] < CARRY_MIN_DISPLACEMENT_M


def classify(rows):
    """Per-condition rows -> the frozen metric set.

    ``rows`` are the ``per_condition`` entries an evaluation sweep produces.
    """
    n = len(rows)
    if n == 0:
        return {}

    def rate(pred):
        return round(sum(1 for r in rows if pred(r)) / n, 4)

    def stage(name):
        return rate(lambda r: r["stages"].get(name))

    def mean(key):
        return round(sum(r[key] for r in rows) / n, 4)

    return {
        "metrics_version": METRICS_VERSION,
        "n_eval_episodes": n,
        # primary
        "full_success_rate": stage("success"),
        "carry_rate": rate(is_carry),
        "mean_return": mean("return"),
        # auxiliary
        "reach_rate": stage("reach"),
        "grasp_rate": stage("grasp"),
        "lift_rate": stage("lift"),
        "plate_rate": stage("plate"),
        # failure / exploit
        "throw_rate": rate(is_throw),
        # physical
        "mean_disp_m": mean("disp_m"),
        "max_disp_m": round(max(r["disp_m"] for r in rows), 4),
        "mean_max_lift_m": mean("max_lift_m"),
        "max_lift_m": round(max(r["max_lift_m"] for r in rows), 4),
    }


def wilson95(count, n):
    """95% Wilson interval, for rates over evaluation conditions within one seed."""
    import math

    if n == 0:
        return [0.0, 0.0]
    z = 1.96
    p = count / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]
