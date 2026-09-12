#!/usr/bin/env bash
# Run 9: critic-free GRPO on the frozen BC head + residual (run9_grpo_preregistration.json).
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
OUTF=$D/rl9/run.json RUN_DIR=$D/rl9 SIGMA=0.15 R_MAX=0.15 N_GROUPS=8 GROUP=6 PPO_EPOCHS=3 MB=64 LR=1e-4 CLIP=0.2 KL_BETA=0.1 KL_STOP=0.02 \
  WALL_BUDGET=6000 N_EVAL_PERIODIC=8 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 \
  timeout 12000 $PY $T/train_grpo.py > $D/rl9/train.log 2>&1
echo "rl9 exit=$?" >> $D/rl9/train.log
