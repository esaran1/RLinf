# G1 piston RL fine-tuning: final report (living document, updated as runs certify)

## Deliverable — and an honest correction

**Run 9, iteration 1** (critic-free GRPO on a frozen BC head + residual) is the only RL run
that ever scored above behaviour cloning, and the pipeline that produced it is reproducible
(`docs/g1_piston_rl_pipeline.md`). But the improvement is **small and not established**:

| paired sweep (first 25 conditions) | BC lift | run 9 lift |
|---|---|---|
| 1 | 0.80 | 0.96 |
| 2 | 0.72 | 0.76 |
| 3 | 0.60 | 0.88 |
| 4 (first half of the 50-condition sweep) | 0.84 | 0.64 |
| conditions 25–49 | 0.76 | 0.84 |
| **pooled, 125 paired condition-evaluations each** | **0.744** | **0.816** |

A deterministic policy's lift rate on the same fixed conditions swings by ~0.2 between
fresh processes, so three paired wins were never strong evidence and the fourth sweep
reversed them. The pooled point estimate favours run 9 by +0.07 lift; the full
50-condition sweep favoured BC (0.80 vs 0.74 lift, 12.14 vs 11.51 return). Four more paired
sweeps are queued to reach ~300 condition-evaluations per policy. Contract:
`g1_piston_eval_variance.json`.

* checkpoint: `checkpoints/g1_piston_grpo_working/run9_best.pt`
* videos: `verified_results/videos_rl_grpo/run9_best/` (12 clips) and BC's
  `verified_results/videos_bc_working/`
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
