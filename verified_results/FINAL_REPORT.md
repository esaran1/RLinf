# G1 piston RL fine-tuning: final report (living document, updated as runs certify)

## Deliverable

**Run 9, iteration 1** — critic-free GRPO on a frozen BC head + residual — is the first RL
fine-tuning in this project that improves the policy, certified by the scorer of record
(`eval_checkpoint.py`, v3 predicate, fresh process) with paired repeats:

| sweep | BC lift | run 9 lift | BC return | run 9 return |
|---|---|---|---|---|
| 1 | 0.80 | **0.96** | 11.94 | **14.05** |
| 2 | 0.72 | **0.76** | 10.6 | **12.3** |
| 3 | 0.60 | **0.88** | 9.4 | **12.4** |
| **mean** | 0.71 | **0.87** | 10.65 | **12.91** |

Run 9 wins all three pairs on lift and on return; grasp 0.97 vs 0.96; plate 0.85 vs 0.69.
A 50-condition certification of both is queued.

* checkpoint: `checkpoints/g1_piston_grpo_working/run9_best.pt`
* videos: `verified_results/videos_rl_grpo/run9_best/` (12 clips; best: cond 13, 14, 18)
* recipe: `docs/g1_piston_rl_pipeline.md`

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

## Run 10 (training reward v4) — in progress

(filled in when certified)

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

## Evaluation caveats carried throughout

* A single n=25 sweep is one sample: the scorer flips lift on ~6/25 conditions between
  fresh-process repeats. Never select a checkpoint by in-trainer evaluation after rollouts.
* Post-grasp outcomes are not reproducible per condition across processes.
