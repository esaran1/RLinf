# G1 piston: the RL fine-tuning pipeline that works

End-to-end recipe that took the StarVLA QwenOFT policy from behaviour cloning
(grasp 1.00 / lift 0.80) to **grasp 1.00 / lift 0.96 / plate 0.96** on the frozen
25-condition suite under the strict v3 predicate (`verified_results/REVIEW_RESPONSE.md`).

All commands use `PY=$HOME/miniconda3/envs/env_isaaclab/bin/python` and
`MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab`, from the RLinf root.
Never write run artifacts to the session scratchpad (wiped three times); use
`/home/jren313/research/starvla_rl/runs_g1_piston/`.

## 1. Behaviour cloning (the base policy)

    OUTF=<json> RUN_DIR=<dir> EPOCHS=300 BC_LR=1e-4 BATCH=16 \
      EPISODES=0,4,5,39,40,43,46,47,48,49,50,51,52,53,54,59 $PY tools/g1_piston/train_bc.py

Two defects had to be fixed for this to work (features gathered at the action-token
positions; loss on the *squashed* action). Backed-up result:
`checkpoints/g1_piston_bc_working/bc_ckpt_latest.pt`.

## 2. Critic-free RL (GRPO + PPO + KL anchor on a frozen base + residual)

    OUTF=<json> RUN_DIR=<dir> SIGMA=0.15 R_MAX=0.15 N_GROUPS=8 GROUP=6 PPO_EPOCHS=3 MB=64 \
      LR=1e-4 CLIP=0.2 KL_BETA=0.1 KL_STOP=0.02 WALL_BUDGET=6000 N_EVAL_PERIODIC=8 \
      RESET_SUITE_SEED=20260817 EP_CHUNKS=23 $PY tools/g1_piston/train_grpo.py

`BASE_CKPT` defaults to the BC checkpoint above; `INIT_RESIDUAL=<ckpt>` continues from a
previous run's residual. Every iteration is saved as `grpo_ckpt_iter<k>.pt`.

Why this and not SAC/RLPD: every actor-critic arm collapsed the policy because the critic
could not rank actions (two independent critics rated exploration-scale perturbations of
the BC action at chance). GRPO learns from the simulator's return directly. Contracts:
`g1_piston_critic_exploitation.json`, `g1_piston_run9_result.json`.

## 3. Select by the scorer of record, never by in-trainer evaluation

In-trainer evaluations after stochastic rollouts are noisy and mis-score (a BC-identical
policy scored 0.76 grasp there vs 1.00 certified). Score every saved iteration:

    OUTF=<json> CKPT=<iter ckpt> RUN_DIR=<dir> N_EVAL=25 MODES=det REWARD_V3=1 \
      RESET_SUITE_SEED=20260817 EP_CHUNKS=23 $PY tools/g1_piston/eval_checkpoint.py

and pick the iteration with the highest mean return among those with grasp >= 0.87.
(`runs_g1_piston/scripts/score_intermediates.sh`, `chain_rl9c.sh` do this.)

## 4. Render

    OUTDIR=<dir> CKPT=<ckpt> CONDS=<from the scored per_condition> MODE=deterministic TAG=<tag> \
      REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 FPS=20 LOG_STEPS=1 \
      $PY tools/g1_piston/render_rollouts.py

Both scorers apply the residual automatically when the checkpoint carries one.

## Operational notes

* Throughput: ~50 s per rollout (grasping episodes have sustained contact); a scorer-of-
  record sweep is ~23 min; the trainer needs ~9.5 GB and a scorer ~7.6 GB on the 16 GB card.
* Pipeline scripts live in `tools/g1_piston/pipeline/` (operational copies in
  `runs_g1_piston/scripts/`). Their GPU busy check is built so a launching shell cannot
  match it; start them with `setsid nohup ... </dev/null &`.
* Pre-register before running (`docs/contracts/*_preregistration.json`); every number in
  the writeup is re-derived from its source file.
