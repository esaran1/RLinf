#!/usr/bin/env bash
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
OUTF=$D/rl9c/run.json RUN_DIR=$D/rl9c INIT_RESIDUAL=$D/rl9/grpo_ckpt_best.pt \
  SIGMA=0.15 R_MAX=0.15 N_GROUPS=6 GROUP=6 PPO_EPOCHS=3 MB=64 LR=5e-5 CLIP=0.2 KL_BETA=0.1 KL_STOP=0.02 \
  WALL_BUDGET=10800 N_EVAL_PERIODIC=8 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 SEED=2 \
  timeout 18000 $PY $T/train_grpo.py > $D/rl9c/train.log 2>&1
echo "rl9c exit=$?" >> $D/rl9c/train.log
