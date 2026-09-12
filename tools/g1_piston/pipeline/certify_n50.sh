#!/usr/bin/env bash
# Final certification on ALL 50 frozen eval conditions: run 9 best vs BC, fresh processes.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/certify_n50.log
while ! grep -q "CHAIN RL10 DONE" $D/chain_rl10.log 2>/dev/null; do sleep 120; done
score() { wait_gpu; OUTF=$D/filter_ab/$1_v3_n50.json CKPT=$2 RUN_DIR=$D/filter_ab/$1_n50 N_EVAL=50 MODES=det REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 timeout 12000 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_n50_eval.log 2>&1; echo "$(date +%H:%M:%S) scored $1 n50 exit=$?" >> $LOG; }
score rl9_best $D/rl9/grpo_ckpt_best.pt
score bc $BC
echo "$(date +%H:%M:%S) CERTIFY N50 DONE" >> $LOG
