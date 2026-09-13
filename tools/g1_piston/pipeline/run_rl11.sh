#!/usr/bin/env bash
# Run 11: targeted exploration (hand sigma 0.35, arm 0.10), v4 training reward.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
OUTF=$D/rl11/run.json RUN_DIR=$D/rl11 TRAIN_REWARD=v4 \
  SIGMA=0.10 SIGMA_HAND=0.35 R_MAX=0.15 N_GROUPS=8 GROUP=6 PPO_EPOCHS=3 MB=64 LR=1e-4 CLIP=0.2 KL_BETA=0.1 KL_STOP=0.02 \
  WALL_BUDGET=7200 N_EVAL_PERIODIC=8 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 SEED=4 \
  timeout 14400 $PY $T/train_grpo.py > $D/rl11/train.log 2>&1
echo "rl11 exit=$?" >> $D/rl11/train.log
