# RL fine-tuning that improved the policy (critic-free GRPO, run 9)

First RL result in this project that beats behaviour cloning on the frozen 25-condition
suite under the strict v3 predicate, scored by `eval_checkpoint.py` in a fresh process:

| | BC | **run 9 best** | run 9b best |
|---|---|---|---|
| grasp | 1.00 | **1.00** | 1.00 |
| lift | 0.80 | **0.96** | 0.64 |
| plate | 0.80 | **0.96** | 0.60 |
| press / dispense | 0.00 | 0.00 | 0.00 |
| mean return | 11.935 | **14.054** | 10.125 |

Method: frozen BC head + zero-initialised residual (ResFiT); GRPO advantages (per-chunk
return-to-go standardised within groups of 6 rollouts per initial condition); PPO clipped
surrogate on the latent Gaussian with a KL anchor to the base. **No critic** — two
independent critics had rated exploration-scale perturbations of the BC action at chance,
which is why every actor-critic arm collapsed. Contracts: `g1_piston_run9_result.json`,
`g1_piston_critic_exploitation.json`, `g1_piston_run9_grpo_preregistration.json`.

## Clips

`run9_best/` — 12 rollouts from conditions the evaluation actually scored (11/12 grasp,
9/12 lift). Best: cond 13 (return 15.35, lift 0.21 m), cond 14, cond 18. Two clips grasp
without lifting and are kept deliberately: post-grasp outcomes are not reproducible per
condition across processes.

`run9b_best/` — the continuation's in-trainer "best"; its certified score is below BC
(selection by 8 post-rollout conditions is noisy). Kept for the record.

The plunger is never pressed in any clip: that behaviour is absent from the data and was
not discovered within the budget.

Checkpoints: `checkpoints/g1_piston_grpo_working/run9_best.pt` (and `run9b_best.pt`).
Score or render with the same commands as the BC policy (the tools apply the residual).
