#!/usr/bin/env bash
# Run 10: run 9's exact GRPO configuration, TRAIN reward v4 (press pays only while lifted).
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
OUTF=$D/rl10/run.json RUN_DIR=$D/rl10 TRAIN_REWARD=v4 \
  SIGMA=0.15 R_MAX=0.15 N_GROUPS=8 GROUP=6 PPO_EPOCHS=3 MB=64 LR=1e-4 CLIP=0.2 KL_BETA=0.1 KL_STOP=0.02 \
  WALL_BUDGET=7200 N_EVAL_PERIODIC=8 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 SEED=3 \
  timeout 14400 $PY $T/train_grpo.py > $D/rl10/train.log 2>&1
echo "rl10 exit=$?" >> $D/rl10/train.log
