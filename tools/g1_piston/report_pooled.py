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

"""Pooled paired comparison of run 9 best vs BC over every paired fresh-process sweep.

Pairs are (sweep k of run 9, sweep k of BC) on the same 25 conditions; the 50-condition
sweep contributes its two halves as two more pairs. Reports pooled lift/plate/return and a
paired bootstrap 95% CI on the lift difference (resampling pairs). Appends to the report.
"""
import glob, json, os, random

D = "/home/jren313/research/starvla_rl/runs_g1_piston"
REPORT = "verified_results/FINAL_REPORT.md"
BCM = "verified_results/manifests/bc_squashfixed_v3_n25.json"
R9M = "verified_results/manifests/grpo_run9_best_v3_n25.json"
CAND = os.environ.get("CAND", "rl9_best")   # rl9_best | rl11_iter2

def pc(p): return {r["condition"]: r for r in json.load(open(p))["modes"]["deterministic"]["per_condition"]}
if CAND == "rl9_best":
    pairs = [(R9M, BCM)]
    for k in range(2, 20):
        a, b = f"{D}/filter_ab/rl9_best_rep{k}_v3_n25.json", f"{D}/filter_ab/bc_rep{k}_v3_n25.json"
        if os.path.exists(a) and os.path.exists(b): pairs.append((a, b))
    n50 = (f"{D}/filter_ab/rl9_best_v3_n50.json", f"{D}/filter_ab/bc_v3_n50.json")
else:
    # run 11 iteration 2: its two chain sweeps pair with BC's first two, its rep sweeps with BC's reps.
    cand = [f"{D}/filter_ab/rl11_iter2_s1_v3_n25.json", f"{D}/filter_ab/rl11_iter2_s2_v3_n25.json"] + \
           [f"{D}/filter_ab/rl11_iter2_rep{k}_v3_n25.json" for k in range(4, 20)]
    bcs = [BCM, f"{D}/filter_ab/bc_rep2_v3_n25.json"] + [f"{D}/filter_ab/bc_rep{k}_v3_n25.json" for k in range(4, 20)]
    pairs = [(a, b) for a, b in zip(cand, bcs) if os.path.exists(a) and os.path.exists(b)]
    n50 = ("", "")
rows = []   # per pair: (lift_a, lift_b, ret_a, ret_b, n)
for a, b in pairs:
    A, B = pc(a), pc(b); cs = sorted(set(A) & set(B))
    rows.append((sum(A[c]["stages"].get("lift", False) for c in cs), sum(B[c]["stages"].get("lift", False) for c in cs),
                 sum(A[c]["return"] for c in cs), sum(B[c]["return"] for c in cs), len(cs), os.path.basename(a)))
if os.path.exists(n50[0]) and os.path.exists(n50[1]):
    A, B = pc(n50[0]), pc(n50[1])
    for lo, hi, tag in ((0, 25, "n50 first half"), (25, 50, "n50 second half")):
        cs = [c for c in sorted(A) if lo <= c < hi]
        rows.append((sum(A[c]["stages"].get("lift", False) for c in cs), sum(B[c]["stages"].get("lift", False) for c in cs),
                     sum(A[c]["return"] for c in cs), sum(B[c]["return"] for c in cs), len(cs), tag))
N = sum(r[4] for r in rows)
la, lb = sum(r[0] for r in rows) / N, sum(r[1] for r in rows) / N
ra, rb = sum(r[2] for r in rows) / N, sum(r[3] for r in rows) / N
random.seed(0); diffs = []
for _ in range(20000):
    s = [rows[random.randrange(len(rows))] for _ in rows]
    n = sum(r[4] for r in s); diffs.append((sum(r[0] for r in s) - sum(r[1] for r in s)) / n)
diffs.sort(); ci = (diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))])
wins = sum(1 for r in rows if r[0] > r[1]); ties = sum(1 for r in rows if r[0] == r[1])
TITLE = {"rl9_best": "### Pooled paired comparison: run 9 best vs BC (every fresh-process sweep)", "rl11_iter2": "### Pooled paired comparison: run 11 iteration 2 vs BC (every fresh-process sweep)"}[CAND]
lines = [TITLE, "",
         "| pair | candidate lift | BC lift | candidate return | BC return |", "|---|---|---|---|---|"]
for r in rows: lines.append(f"| {r[5]} | {r[0]}/{r[4]} | {r[1]}/{r[4]} | {r[2]/r[4]:.2f} | {r[3]/r[4]:.2f} |")
lines += ["", f"**Pooled over {N} paired condition-evaluations per policy:** lift {la:.3f} vs {lb:.3f} "
          f"(difference {la-lb:+.3f}, paired-bootstrap 95% CI {ci[0]:+.3f} to {ci[1]:+.3f}); return {ra:.2f} vs {rb:.2f}. "
          f"Candidate won {wins} of {len(rows)} pairs on lift ({ties} ties).", ""]
verdict = ("**Established** (CI excludes zero)." if ci[0] > 0 else "**Not established** (CI includes zero).")
lines += [f"Verdict: {verdict}", ""]
block = "\n".join(lines)
print(block)
rep = open(REPORT).read(); marker = "## What did not work, and why (all measured)"
if TITLE in rep:
    s = rep.index(TITLE); e = rep.index(marker)
    nxt = rep.find("### Pooled paired comparison", s + 10)
    e = nxt if (nxt != -1 and nxt < e) else e
    rep = rep[:s] + block + "\n" + rep[e:]
else:
    rep = rep.replace(marker, block + "\n" + marker, 1)
open(REPORT, "w").write(rep)
