#!/usr/bin/env bash
# Run 9b: continuation of run 9 from its best residual, smaller groups (6 x 5) so an
# iteration takes ~30 min at the measured 50 s/episode; 2 h budget.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
OUTF=$D/rl9b/run.json RUN_DIR=$D/rl9b INIT_RESIDUAL=$D/rl9/grpo_ckpt_best.pt \
  SIGMA=0.15 R_MAX=0.15 N_GROUPS=6 GROUP=5 PPO_EPOCHS=3 MB=64 LR=1e-4 CLIP=0.2 KL_BETA=0.1 KL_STOP=0.02 \
  WALL_BUDGET=7200 N_EVAL_PERIODIC=8 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 SEED=1 \
  timeout 14400 $PY $T/train_grpo.py > $D/rl9b/train.log 2>&1
echo "rl9b exit=$?" >> $D/rl9b/train.log
