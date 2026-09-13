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

"""Append the 50-condition certification (run 9 best vs BC) to FINAL_REPORT.md."""
import json, os

D = "/home/jren313/research/starvla_rl/runs_g1_piston"
REPORT = "verified_results/FINAL_REPORT.md"
rows = {}
for tag in ("rl9_best", "bc"):
    d = json.load(open(f"{D}/filter_ab/{tag}_v3_n50.json")); m = d["modes"]["deterministic"]
    rows[tag] = {k: m[k] for k in ("grasp_rate", "lift_rate", "plate_rate", "press_rate", "dispense_rate", "mean_return", "n_eval_episodes")}
    rows[tag]["ci_lift"] = m["ci95"]["lift"]; rows[tag]["ci_plate"] = m["ci95"]["plate"]
a, b = rows["rl9_best"], rows["bc"]
block = f"""
### 50-condition certification (all frozen eval conditions, fresh processes)

| | BC (n={b['n_eval_episodes']}) | **run 9 best** (n={a['n_eval_episodes']}) |
|---|---|---|
| grasp | {b['grasp_rate']:.2f} | **{a['grasp_rate']:.2f}** |
| lift | {b['lift_rate']:.2f} (CI {b['ci_lift'][0]:.2f}–{b['ci_lift'][1]:.2f}) | **{a['lift_rate']:.2f}** (CI {a['ci_lift'][0]:.2f}–{a['ci_lift'][1]:.2f}) |
| plate | {b['plate_rate']:.2f} (CI {b['ci_plate'][0]:.2f}–{b['ci_plate'][1]:.2f}) | **{a['plate_rate']:.2f}** (CI {a['ci_plate'][0]:.2f}–{a['ci_plate'][1]:.2f}) |
| press / dispense | {b['press_rate']:.2f} / {b['dispense_rate']:.2f} | {a['press_rate']:.2f} / {a['dispense_rate']:.2f} |
| mean return | {b['mean_return']:.2f} | **{a['mean_return']:.2f}** |

Manifests: `verified_results/manifests/grpo_rl9_best_v3_n50.json`, `bc_v3_n50.json`.
"""
rep = open(REPORT).read()
marker = "## What did not work, and why (all measured)"
if "### 50-condition certification" in rep:
    s = rep.index("### 50-condition certification"); e = rep.index(marker)
    rep = rep[:s] + block.lstrip("\n") + "\n" + rep[e:]
else:
    rep = rep.replace(marker, block.lstrip("\n") + "\n" + marker, 1)
open(REPORT, "w").write(rep)
print(block)
