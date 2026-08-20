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

Horizontal means horizontal
---------------------------

The threshold is applied to XY displacement. This was always the definition, but the
rollout code recorded ``disp_m`` as a 3-D norm, so a vertical fling could clear 0.05 m
on height alone -- scoring the exploit as its own opposite. ``horizontal_disp_m``
recovers the XY component, preferring an explicit ``disp_xy_m`` and falling back to the
recorded piston geometry. Reward, training and the stage predicates are unaffected: this
module is measurement only and is never in the optimisation path.
"""

from __future__ import annotations

#: Frozen on 2026-08-17, before any RLPD result existed.
#:
#: ``v1.1`` corrects an IMPLEMENTATION BUG, not the definition: carry/throw always meant
#: HORIZONTAL transport (see the module docstring and the demonstration reference, which
#: is a horizontal range), but the ``disp_m`` written by the rollout code was a full 3-D
#: norm, so a purely vertical fling could be scored as a carry. Thresholds, roles and
#: demonstration references are untouched; only the axis the threshold is applied to is
#: brought back in line with the frozen text. Results computed under v1 must be
#: recomputed from raw trajectories -- see tools/g1_piston/recompute_horizontal.py.
METRICS_VERSION = "v1.1-frozen-2026-08-17-horizontal-fix"
#: The pre-fix identifier, for reading historical result files.
METRICS_VERSION_V1 = "v1-frozen-2026-08-17"

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


def horizontal_disp_m(row) -> float:
    """Horizontal (XY) piston displacement for ``row``, in metres.

    The frozen definition of carry has always been HORIZONTAL transport -- see the module
    docstring, the ``carry_rate`` role, and the demonstration reference range, which is a
    horizontal range. Some producers nevertheless wrote ``disp_m`` as a full 3-D norm,
    which let a purely vertical move clear the threshold: precisely the throw the metric
    exists to catch. That was an implementation bug, not an alternative definition.

    Rows carrying an explicit ``disp_xy_m`` are used directly. Older rows are corrected
    from the recorded geometry where possible (``piston_initial_xyz`` /
    ``piston_final_xyz``, or the 3-D norm and the final vertical offset). Where neither is
    available the 3-D value is returned unchanged and ``disp_is_3d_fallback`` reports it,
    so a caller can never silently mistake an uncorrected row for a corrected one.
    """
    if "disp_xy_m" in row:
        return float(row["disp_xy_m"])
    p0, pN = row.get("piston_initial_xyz"), row.get("piston_final_xyz")
    if p0 is not None and pN is not None:
        return float(((pN[0] - p0[0]) ** 2 + (pN[1] - p0[1]) ** 2) ** 0.5)
    dz = row.get("final_dz_m")
    if dz is not None:
        return float(max(row["disp_m"] ** 2 - float(dz) ** 2, 0.0) ** 0.5)
    return float(row["disp_m"])


def disp_is_3d_fallback(row) -> bool:
    """True when ``horizontal_disp_m`` could not correct the row and returned the 3-D norm."""
    return not ("disp_xy_m" in row
                or (row.get("piston_initial_xyz") is not None
                    and row.get("piston_final_xyz") is not None)
                or row.get("final_dz_m") is not None)


def is_carry(row) -> bool:
    """A lifting episode that also transported the piston horizontally."""
    return (bool(row["stages"].get("lift"))
            and horizontal_disp_m(row) >= CARRY_MIN_DISPLACEMENT_M)


def is_throw(row) -> bool:
    """A lifting episode with essentially no horizontal transport.

    The reward-exploiting failure mode: the ``lift`` stage bonus pays for height alone,
    so flinging the piston upward banks it without earning any transport credit.
    """
    return (bool(row["stages"].get("lift"))
            and horizontal_disp_m(row) < CARRY_MIN_DISPLACEMENT_M)


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
        # physical. disp_xy is what carry/throw test; disp_m is kept as recorded so a
        # stored row can still be reconciled against the value its producer wrote.
        "mean_disp_xy_m": round(sum(horizontal_disp_m(r) for r in rows) / n, 4),
        "max_disp_xy_m": round(max(horizontal_disp_m(r) for r in rows), 4),
        "mean_disp_m": mean("disp_m"),
        "max_disp_m": round(max(r["disp_m"] for r in rows), 4),
        "mean_max_lift_m": mean("max_lift_m"),
        "max_lift_m": round(max(r["max_lift_m"] for r in rows), 4),
        # True if ANY row lacked the geometry to correct its 3-D displacement.
        "disp_3d_fallback_rows": sum(1 for r in rows if disp_is_3d_fallback(r)),
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
