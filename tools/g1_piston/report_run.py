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

"""Fill a run's section of verified_results/FINAL_REPORT.md from certified files (RUN=rl10|rl11).

Reads every rl10_iter*_s?_v3_n25.json (two fresh-process v3 sweeps per iteration), the
selection written by chain_rl10.sh, and the training record; prints the markdown block and
replaces the placeholder in the report. Numbers come only from those files.
"""
import glob, json, os, re, sys

D = "/home/jren313/research/starvla_rl/runs_g1_piston"
REPORT = "verified_results/FINAL_REPORT.md"
RUN = os.environ.get("RUN", "rl10")
TITLE = {"rl10": "## Run 10 (training reward v4: press pays only while lifted)",
         "rl11": "## Run 11 (targeted exploration: hand sigma 0.35, arm 0.10; v4)"}[RUN]

def m(p):
    d = json.load(open(p)); x = d["modes"]["deterministic"]
    pc = x["per_condition"]
    x = {k: x[k] for k in ("grasp_rate", "lift_rate", "plate_rate", "press_rate", "mean_return")}
    x["press_without_lift"] = sum(1 for r in pc if r["stages"].get("press") and not r["stages"].get("lift"))
    x["press_with_lift"] = sum(1 for r in pc if r["stages"].get("press") and r["stages"].get("lift"))
    return x

rows = {}
for p in sorted(glob.glob(f"{D}/filter_ab/{RUN}_iter*_s?_v3_n25.json")):
    it = re.search(RUN + r"_(iter\d+)_s(\d)", p)
    try: rows.setdefault(it.group(1), {})[int(it.group(2))] = m(p)
    except Exception: pass
train = json.load(open(f"{D}/{RUN}/run.json"))
sel = ""
if os.path.exists(f"{D}/chain_{RUN}.log"):
    for line in open(f"{D}/chain_{RUN}.log"):
        if "selected:" in line: sel = line.split("selected:")[-1].strip()
lines = [TITLE + " — certified", "",
         f"Training: {train['totals']['iterations'] if train.get('totals') else len(train.get('iterations', []))} iterations from the BC base, run 9's configuration, SEED 3. Collected press rate per iteration: "
         + ", ".join(f"{it['collect']['press_rate']:.3f}" for it in train.get("iterations", [])) + " (v4 pays nothing for a table press).", "",
         "| checkpoint | sweep | grasp | lift | plate | press (w/ lift, w/o lift) | return |", "|---|---|---|---|---|---|---|"]
for it in sorted(rows, key=lambda s: int(s[4:])):
    for s_ in sorted(rows[it]):
        x = rows[it][s_]
        lines.append(f"| {RUN} {it} | {s_} | {x['grasp_rate']:.2f} | {x['lift_rate']:.2f} | {x['plate_rate']:.2f} | {x['press_rate']:.2f} ({x['press_with_lift']}, {x['press_without_lift']}) | {x['mean_return']:.2f} |")
lines += ["", f"Selected by the registered rule (grasp ≥ 0.87 in both sweeps, highest mean return): **{sel or 'none qualified'}**.",
          "Reference: run 9 iteration 1 paired mean lift 0.87 / return 12.9; BC 0.71 / 10.7.", ""]
block = "\n".join(lines)
print(block)
rep = open(REPORT).read()
if TITLE in rep:
    start = rep.index(TITLE); end = rep.index("## Evaluation caveats")
else:
    start = end = rep.index("## Evaluation caveats")
open(REPORT, "w").write(rep[:start] + block + "\n" + rep[end:])
