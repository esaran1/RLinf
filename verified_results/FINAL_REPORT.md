# G1 piston RL fine-tuning: final report (living document, updated as runs certify)

## Deliverable — with the certified verdict

The pipeline works end to end and is reproducible (`docs/g1_piston_rl_pipeline.md`), and two
RL checkpoints have scored above behaviour cloning on individual sweeps. Whether either is a
real improvement is a question this benchmark's evaluation variance makes expensive to
answer, and both answers are now in:

**Run 9, iteration 1 vs BC — 9 paired fresh-process sweeps, 225 condition-evaluations each:**
lift **0.742 vs 0.698** (+0.044, paired-bootstrap 95 % CI −0.133 to +0.191); return
**11.37 vs 10.79**; 6 wins, 2 losses, 1 tie. **Not established.** The same deterministic
checkpoint scored lift anywhere from 0.36 to 0.96 across sweeps; BC from 0.48 to 0.88.

**Run 11, iteration 2 vs BC — 6 paired fresh-process sweeps, 150 condition-evaluations each:**
lift **0.747 vs 0.680** (+0.067, paired-bootstrap 95 % CI −0.087 to +0.207); return
**11.70 vs 10.58**; 4 wins, 2 losses. **Not established.** This is the iteration trained on the
project's first lifted press (targeted hand exploration, v4 training reward); its sweeps ranged
lift 0.56 to 0.96, BC's 0.48 to 0.88 on the same conditions.

**Bottom line.** Transport (reach, grasp, lift, carry to the plate) is solved by behaviour
cloning and is not improved beyond evaluation noise by any RL run (both critic-free GRPO
candidates are ahead by +0.04 to +0.07 lift; neither margin clears the noise at 150 to 225
paired evaluations). The plunger press, the task's functional act, was blocked by two measured
defects: the v3 press threshold (28 mm) lies 3 mm past what the object's geometry permits
(25 mm), and the thumb was uncommandable through the frozen action normaliser. With a
geometry-grounded press scorer (v5), a scripted palm press synthesised into the demonstration
data, and a 40-chunk evaluation horizon (the human demonstrations are 23 chunks; the press adds
~20), the pressing policy `bc_press` dispenses on 12 % of conditions and completes the whole
task (dispense, place, release) on 6 % across 50 condition-evaluations, against 0 of 25 for the
baseline, with transport intact (lift 0.92). Critic-free GRPO from that policy (run 12) is
scored at the end of this report.

* checkpoints: `checkpoints/g1_piston_bc_press/bc_ckpt_latest.pt` (pressing policy), `checkpoints/g1_piston_grpo_working/run9_best.pt`, `run11_iter2.pt`
* videos: `verified_results/videos_bc_press_h40/` (pressing policy, 40 chunks), `verified_results/videos_rl_grpo/run9_best/`, `run11_iter2/` (12 clips each; run 11
  iteration 2 renders at 12/12 grasp, 11/12 lift), and BC's `verified_results/videos_bc_working/`
* recipe: `docs/g1_piston_rl_pipeline.md`

### Pooled paired comparison: run 9 best vs BC (every fresh-process sweep)

| pair | candidate lift | BC lift | candidate return | BC return |
|---|---|---|---|---|
| grpo_run9_best_v3_n25.json | 24/25 | 20/25 | 14.05 | 11.93 |
| rl9_best_rep2_v3_n25.json | 19/25 | 18/25 | 11.60 | 10.66 |
| rl9_best_rep3_v3_n25.json | 22/25 | 15/25 | 13.08 | 9.35 |
| rl9_best_rep4_v3_n25.json | 20/25 | 12/25 | 10.99 | 7.80 |
| rl9_best_rep5_v3_n25.json | 9/25 | 22/25 | 6.63 | 14.00 |
| rl9_best_rep6_v3_n25.json | 15/25 | 15/25 | 9.65 | 9.89 |
| rl9_best_rep7_v3_n25.json | 21/25 | 15/25 | 13.27 | 9.17 |
| n50 first half | 16/25 | 21/25 | 10.03 | 12.51 |
| n50 second half | 21/25 | 19/25 | 12.98 | 11.77 |

**Pooled over 225 paired condition-evaluations per policy:** lift 0.742 vs 0.698 (difference +0.044, paired-bootstrap 95% CI -0.133 to +0.191); return 11.37 vs 10.79. Candidate won 6 of 9 pairs on lift (1 ties).

Verdict: **Not established** (CI includes zero).

### 50-condition certification (all frozen eval conditions, fresh processes)

| | BC (n=50) | **run 9 best** (n=50) |
|---|---|---|
| grasp | 1.00 | **0.94** |
| lift | 0.80 (CI 0.67–0.89) | **0.74** (CI 0.60–0.84) |
| plate | 0.76 (CI 0.63–0.86) | **0.74** (CI 0.60–0.84) |
| press / dispense | 0.00 / 0.00 | 0.00 / 0.00 |
| mean return | 12.14 | **11.51** |

Manifests: `verified_results/manifests/grpo_rl9_best_v3_n50.json`, `bc_v3_n50.json`.

### Pooled paired comparison: run 11 iteration 2 vs BC (every fresh-process sweep)

| pair | candidate lift | BC lift | candidate return | BC return |
|---|---|---|---|---|
| rl11_iter2_s1_v3_n25.json | 23/25 | 20/25 | 13.88 | 11.93 |
| rl11_iter2_s2_v3_n25.json | 21/25 | 18/25 | 12.87 | 10.66 |
| rl11_iter2_rep4_v3_n25.json | 14/25 | 12/25 | 9.06 | 7.80 |
| rl11_iter2_rep5_v3_n25.json | 16/25 | 22/25 | 10.80 | 14.00 |
| rl11_iter2_rep6_v3_n25.json | 24/25 | 15/25 | 14.35 | 9.89 |
| rl11_iter2_rep7_v3_n25.json | 14/25 | 15/25 | 9.24 | 9.17 |

**Pooled over 150 paired condition-evaluations per policy:** lift 0.747 vs 0.680 (difference +0.067, paired-bootstrap 95% CI -0.087 to +0.207); return 11.70 vs 10.58. Candidate won 4 of 6 pairs on lift (0 ties).

Verdict: **Not established** (CI includes zero).

## What did not work, and why (all measured)

1. **SAC/RLPD on the head (runs 1–5)**: the entropy term flattened the policy
   (tanh-Jacobian mean force; std-sign defect). Reproduced offline without a simulator; fixed.
2. **Actor-critic after the fix (runs 6–8)**: the critic cannot rank actions — two independent
   critics rated exploration-scale perturbations of the BC action at chance (0.46, 0.49), and a
   10-critic RLPD ensemble with 3-step returns was no better. The actor followed a meaningless
   gradient off the task. This is a property of the data (demo actions within 0.4° of the
   policy's own), not of the architecture.
3. **Continuing GRPO (runs 9 iter 2, 9b, 9c)**: no later update beat the first; the group
   signal is dominated by contact nondeterminism, and later updates random-walk the residual.
4. **The press**: every certified press was a *table* press (grasped, not lifted) — a v3
   reward exploit worth more than a full transport. Reward v4 gates the press on a lifted
   pipette; run 10 trains against it (certification stays on v3). Status below.

## Run 10 (training reward v4: press pays only while lifted) — certified

Training: 3 iterations from the BC base, run 9's configuration, SEED 3. Collected press rate per iteration: 0.000, 0.000, 0.000 (v4 pays nothing for a table press).

| checkpoint | sweep | grasp | lift | plate | press (w/ lift, w/o lift) | return |
|---|---|---|---|---|---|---|
| rl10 iter1 | 1 | 0.96 | 0.68 | 0.60 | 0.00 (0, 0) | 9.68 |
| rl10 iter1 | 2 | 0.88 | 0.52 | 0.48 | 0.04 (0, 1) | 8.79 |
| rl10 iter2 | 1 | 0.96 | 0.48 | 0.48 | 0.00 (0, 0) | 7.72 |
| rl10 iter2 | 2 | 0.96 | 0.40 | 0.36 | 0.00 (0, 0) | 7.17 |
| rl10 iter3 | 1 | 1.00 | 0.72 | 0.72 | 0.00 (0, 0) | 11.32 |
| rl10 iter3 | 2 | 0.96 | 0.76 | 0.76 | 0.00 (0, 0) | 11.29 |

Selected by the registered rule (grasp ≥ 0.87 in both sweeps, highest mean return): **iter3**.
Reference: run 9 iteration 1 paired mean lift 0.87 / return 12.9; BC 0.71 / 10.7.

## Run 11 (targeted exploration: hand sigma 0.35, arm 0.10; v4) — certified

Training: 3 iterations from the BC base, run 9's configuration, SEED 3. Collected press rate per iteration: 0.000, 0.021, 0.000 (v4 pays nothing for a table press).

| checkpoint | sweep | grasp | lift | plate | press (w/ lift, w/o lift) | return |
|---|---|---|---|---|---|---|
| rl11 iter1 | 1 | 1.00 | 0.76 | 0.72 | 0.00 (0, 0) | 11.08 |
| rl11 iter1 | 2 | 1.00 | 0.88 | 0.84 | 0.00 (0, 0) | 11.76 |
| rl11 iter2 | 1 | 1.00 | 0.92 | 0.92 | 0.00 (0, 0) | 13.88 |
| rl11 iter2 | 2 | 1.00 | 0.84 | 0.84 | 0.00 (0, 0) | 12.87 |
| rl11 iter3 | 1 | 0.96 | 0.68 | 0.68 | 0.00 (0, 0) | 10.73 |
| rl11 iter3 | 2 | 1.00 | 0.80 | 0.80 | 0.00 (0, 0) | 12.64 |

Selected by the registered rule (grasp ≥ 0.87 in both sweeps, highest mean return): **iter2**.
Reference: run 9 iteration 1 paired mean lift 0.87 / return 12.9; BC 0.71 / 10.7.

## Evaluation caveats carried throughout

* A single n=25 sweep is one sample: the scorer flips lift on ~6/25 conditions between
  fresh-process repeats. Never select a checkpoint by in-trainer evaluation after rollouts.
* Post-grasp outcomes are not reproducible per condition across processes.

## Pressing policy (BC on human transport + scripted palm press), certified

Transport under v3 (scorer of record) and the press under v5 (geometry-grounded press scorer); fresh-process sweeps; BC baseline paired on the same conditions.

| policy | reward | sweep | grasp | lift | plate | press | dispense | return | mean max press (mm) |
|---|---|---|---|---|---|---|---|---|---|
| BC (working) | v3 | bc_rep2_v3_n25.json | 1.00 | 0.72 | 0.72 | 0.00 | 0.00 | 10.66 | 14.1 |
| BC (working) | v3 | bc_rep3_v3_n25.json | 0.88 | 0.60 | 0.56 | 0.04 | 0.00 | 9.35 | 16.0 |
| BC (working) | v3 | bc_rep4_v3_n25.json | 0.96 | 0.48 | 0.40 | 0.00 | 0.00 | 7.80 | 14.9 |
| BC (working) | v3 | bc_rep5_v3_n25.json | 1.00 | 0.88 | 0.88 | 0.04 | 0.00 | 14.00 | 13.3 |
| BC (working) | v3 | bc_rep6_v3_n25.json | 1.00 | 0.60 | 0.60 | 0.00 | 0.00 | 9.89 | 14.4 |
| BC (working) | v3 | bc_rep7_v3_n25.json | 0.96 | 0.60 | 0.56 | 0.00 | 0.00 | 9.17 | 14.3 |
| bc_press | v3 | bc_press_s1_v3_n25.json | 1.00 | 0.88 | 0.88 | 0.00 | 0.00 | 12.35 | 13.1 |
| bc_press | v3 | bc_press_s2_v3_n25.json | 1.00 | 0.84 | 0.84 | 0.00 | 0.00 | 12.19 | 16.3 |
| BC (working) | v5 | bc_working_v5_s1_n25.json | 1.00 | 0.80 | 0.80 | 0.04 | 0.04 | 14.29 | 13.8 |
| BC (working) | v5 | bc_working_v5_s2_n25.json | 0.96 | 0.80 | 0.76 | 0.00 | 0.00 | 11.57 | 12.4 |
| bc_press | v5 | bc_press_s1_v5_n25.json | 1.00 | 0.88 | 0.88 | 0.00 | 0.00 | 13.44 | 13.5 |
| bc_press | v5 | bc_press_s2_v5_n25.json | 1.00 | 0.88 | 0.88 | 0.00 | 0.00 | 12.81 | 12.6 |

**Paired dispense under v5 (50 condition-evaluations per policy):** bc_press 0.000 vs BC 0.020; per-condition wins 0, losses 1, ties 49.

## The press, delivered: pressing policy at the 40-chunk horizon (2026-09-14)

The pressing policy (`bc_press`: behaviour cloning on 40 scripted palm-press episodes plus the
human transport, `docs/contracts/g1_piston_press_primitive.json`) presses the plunger onto the
plate. The 23-chunk sweeps above could not show it: the human demonstrations are 22-23 chunks
long, the press adds ~20 chunks after the plate, and the task has no time-out termination, so
the horizon, not the policy, produced the zeros. At 40 chunks, two fresh-process v5 sweeps on
the frozen 25-condition suite:

| policy | horizon | sweep | grasp | lift | plate | press | dispense | full success | return | mean max press (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| bc_press | 40 | s1 | 1.00 | 0.88 | 0.88 | 0.08 | 0.16 | 0.04 | 16.96 | 16.4 |
| bc_press | 40 | s2 | 1.00 | 0.96 | 0.96 | 0.08 | 0.08 | 0.08 | 17.26 | 17.9 |
| bc_press | 40 | pooled | 1.00 | 0.92 | 0.92 | 0.08 | 0.12 | 0.06 | 17.11 | 17.1 |
| BC (working) | 23 | s1, s2 | 1.00, 0.96 | 0.80, 0.80 | 0.80, 0.76 | 0.04, 0.00 | 0.04, 0.00 | 0.00, 0.00 | 14.29, 11.57 | 12.1, - |

`dispense` (v5) = plunger held past 20 mm for 0.3 s with the fingers on the barrel and the
pipette over the plate after transport; `full success` = dispensed, then placed on the plate at
rest and released -- the task's own terminal predicate, reached for the first time in this
project (3 of 50 condition-evaluations). The working BC policy's single 23-chunk "dispense" was
a contact accident (31 mm, interpenetration). A paired 40-chunk baseline sweep is appended
below when it lands, and run 12 (critic-free GRPO under v5 from this policy, 40-chunk
episodes) follows.

Manifests: `verified_results/manifests/bc_press_s{1,2}_v5_h40_n25.json`; checkpoint
`checkpoints/g1_piston_bc_press/bc_ckpt_latest.pt`; videos `verified_results/videos_bc_press_h40/`.

**Paired 40-chunk baseline (same 25 conditions, fresh process):** the working BC policy scores
grasp 1.00, lift 0.88, plate 0.88, press 0.04, dispense 0.00,
full success 0.00, return 11.72, mean max press 12.4 mm. Against the pressing
policy's first sweep on the same conditions: dispense 0.16 vs 0.00 (4 conditions won, 0 lost);
pooled over both pressing-policy sweeps, dispense 0.12 and full success 0.06 on 50
condition-evaluations versus 0 of 25 for the baseline. Transport is not degraded (lift 0.92 vs 0.88).
Manifest: `verified_results/manifests/bc_working_s1_v5_h40_n25.json`.

## Run 12: critic-free GRPO under v5 from the pressing policy, 40-chunk horizon (auto)

Every iteration scored in fresh processes on the frozen 25-condition suite at EP_CHUNKS=40: twice under v5 (press scorer), once under v3 (transport scorer of record).

| policy | reward | sweep | grasp | lift | plate | press | dispense | full success | return | mean max press (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| BC (working) | v5 | s1 | 1.00 | 0.88 | 0.88 | 0.04 | 0.00 | 0.00 | 11.72 | 12.4 |
| bc_press | v5 | s1 | 1.00 | 0.88 | 0.88 | 0.08 | 0.16 | 0.04 | 16.96 | 16.4 |
| bc_press | v5 | s2 | 1.00 | 0.96 | 0.96 | 0.08 | 0.08 | 0.08 | 17.26 | 17.9 |
| rl12_iter1 | v3 | s1 | 0.96 | 0.68 | 0.64 | 0.00 | 0.00 | 0.00 | 10.02 | 19.4 |
| rl12_iter1 | v5 | s1 | 1.00 | 0.64 | 0.64 | 0.00 | 0.04 | 0.00 | 11.48 | 17.8 |
| rl12_iter1 | v5 | s2 | 0.92 | 0.80 | 0.80 | 0.04 | 0.16 | 0.00 | 15.38 | 16.7 |
| rl12_iter2 | v3 | s1 | 1.00 | 0.76 | 0.76 | 0.08 | 0.00 | 0.00 | 11.95 | 18.6 |
| rl12_iter2 | v5 | s1 | 1.00 | 0.68 | 0.60 | 0.12 | 0.12 | 0.08 | 15.70 | 17.8 |
| rl12_iter2 | v5 | s2 | 0.96 | 0.48 | 0.48 | 0.20 | 0.28 | 0.16 | 21.16 | 20.9 |
| rl12_iter3 | v3 | s1 | 1.00 | 0.76 | 0.76 | 0.04 | 0.00 | 0.00 | 11.46 | 18.5 |
| rl12_iter3 | v5 | s1 | 0.96 | 0.76 | 0.76 | 0.04 | 0.12 | 0.00 | 13.31 | 17.4 |
| rl12_iter3 | v5 | s2 | 1.00 | 0.56 | 0.52 | 0.16 | 0.12 | 0.00 | 13.06 | 19.8 |

Training (exploration rollouts, 48 per iteration, 40-chunk episodes): iteration 1: dispense 0.146, lift 0.75, return 12.99, KL-to-base 0.0089; iteration 2: dispense 0.021, lift 0.46, return 7.75, KL-to-base 0.0131; iteration 3: dispense 0.000, lift 0.25, return 4.84, KL-to-base 0.0465.

## Paired 40-chunk comparison: run 12 iteration 2 vs the pressing policy (auto)

| sweep | bc_press dispense | rl12 iter2 dispense | bc_press success | rl12 iter2 success | bc_press lift | rl12 iter2 lift |
|---|---|---|---|---|---|---|
| s1 | 0.16 | 0.12 | 0.04 | 0.08 | 0.88 | 0.68 |
| s2 | 0.08 | 0.28 | 0.08 | 0.16 | 0.96 | 0.48 |
| s3 | 0.04 | 0.12 | 0.00 | 0.04 | 0.84 | 0.64 |
| s4 | 0.04 | 0.16 | 0.04 | 0.16 | 0.80 | 0.72 |

**Pooled over 100 paired condition-evaluations per policy (v5, 40 chunks):** dispense rl12 iter2 0.170 vs bc_press 0.080 (difference +0.090, paired-bootstrap 95% CI +0.000 to +0.190); full success 0.110 vs 0.040; lift 0.630 vs 0.870.

Verdict: **not established** on dispense (CI includes zero).
