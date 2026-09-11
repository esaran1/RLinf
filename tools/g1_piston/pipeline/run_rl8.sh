#!/usr/bin/env bash
# Run 8 (residual arm + critic repair), pre-registered in
# docs/contracts/g1_piston_v3_retrain_run7_preregistration.json. Second launch: the first
# was destroyed at ~43k env steps by a scratchpad wipe. All artifacts now live on durable
# storage under runs_g1_piston/.
set -uo pipefail
D=/home/jren313/research/starvla_rl/runs_g1_piston
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab
cd /home/jren313/research/starvla_rl/RLinf
OUTF=$D/rl8/run.json RUN_DIR=$D/rl8 \
  ALGO=rlpd DEMO_FRAC=0.5 \
  WARM_CKPT=/home/jren313/research/starvla_rl/checkpoints/g1_piston_bc_working/bc_ckpt_latest.pt \
  DEMO_DIR=/home/jren313/research/starvla_rl/demo_buffer_v3 \
  REWARD_V3=1 ACTION_BASIS=1 CRITIC_STATE=1 CORRELATED_NOISE=1 \
  ENTROPY_SPACE=latent ACTOR_WARMUP_UPDATES=300 \
  RESIDUAL=1 R_MAX=0.15 RES_INIT_STD=0.15 RES_ACTOR_LR=3e-4 \
  NUM_Q=10 TARGET_SUBSET=2 ACTOR_Q_AGG=mean N_STEP=3 \
  GAMMA_CHUNK=0.98 SMOOTH_LAMBDA=100 UTD=0.5 BATCH=8 \
  MAX_ENV_STEPS=90000 EVAL_EVERY=15000 N_EVAL_PERIODIC=25 \
  RUN_INIT_EVAL=0 FIRST_EVAL_AT_ZERO=0 EVAL_STOCHASTIC=0 \
  PREWARM_DEMO_CACHE=1 SAVE_EVERY=200 \
  timeout 28800 $PY tools/g1_piston/train_sac.py > $D/rl8/train.log 2>&1
echo "rl8 exit=$?" >> $D/rl8/train.log
