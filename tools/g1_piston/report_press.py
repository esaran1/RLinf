#!/usr/bin/env python3
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

"""Certification table for the pressing policy: transport under v3, press under v5.

Reads the fresh-process sweeps written by chain_bc_press.sh for TAG (default bc_press)
and the BC baseline, and prints a markdown table plus a paired dispense comparison. Rates
are re-derived from the per-condition rows, never read off a summary.
"""
import glob
import json
import os
import sys

D = "/home/jren313/research/starvla_rl/runs_g1_piston/filter_ab"
TAG = os.environ.get("TAG", "bc_press")


def rows(fp):
    return json.load(open(fp))["modes"]["deterministic"]["per_condition"]


def rate(rs, k):
    return sum(1 for r in rs if r["stages"].get(k)) / len(rs)


def summ(fp):
    rs = rows(fp)
    return {"n": len(rs), "grasp": rate(rs, "grasp"), "lift": rate(rs, "lift"), "plate": rate(rs, "plate"),
            "press": rate(rs, "press"), "dispense": rate(rs, "dispense"),
            "ret": sum(r["return"] for r in rs) / len(rs),
            "max_press_mean": sum(r.get("max_press_m", 0.0) for r in rs) / len(rs)}


def main():
    out = []
    out.append(f"| policy | reward | sweep | grasp | lift | plate | press | dispense | return | mean max press (mm) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    table = {}
    for label, pat in (("BC (working)", f"{D}/bc_working_v3_*n25.json"), (TAG, f"{D}/{TAG}_s?_v3_n25.json"),
                       ("BC (working)", f"{D}/bc_working_v5_s?_n25.json"), (TAG, f"{D}/{TAG}_s?_v5_n25.json")):
        for fp in sorted(glob.glob(pat)):
            s = summ(fp); rv = "v5" if "_v5_" in fp else "v3"
            table.setdefault((label, rv), []).append((os.path.basename(fp), s))
            out.append(f"| {label} | {rv} | {os.path.basename(fp)} | {s['grasp']:.2f} | {s['lift']:.2f} | {s['plate']:.2f} | "
                       f"{s['press']:.2f} | {s['dispense']:.2f} | {s['ret']:.2f} | {1000 * s['max_press_mean']:.1f} |")
    # paired dispense comparison under v5, same conditions, sweep by sweep
    a = table.get((TAG, "v5"), []); b = table.get(("BC (working)", "v5"), [])
    if a and b:
        pa = [rows(f"{D}/{f}") for f, _ in a]; pb = [rows(f"{D}/{f}") for f, _ in b]
        wins = losses = ties = 0; na = nb = 0; n = 0
        for ra, rb in zip(pa, pb):
            for x, y in zip(ra, rb):
                dx, dy = int(bool(x["stages"].get("dispense"))), int(bool(y["stages"].get("dispense")))
                na += dx; nb += dy; n += 1
                wins += dx > dy; losses += dx < dy; ties += dx == dy
        out.append("")
        out.append(f"**Paired dispense under v5 ({n} condition-evaluations per policy):** {TAG} {na / max(n, 1):.3f} vs BC {nb / max(n, 1):.3f}; "
                   f"per-condition wins {wins}, losses {losses}, ties {ties}.")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
