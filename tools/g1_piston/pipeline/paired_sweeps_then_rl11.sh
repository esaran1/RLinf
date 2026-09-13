#!/usr/bin/env bash
# Four more paired fresh-process sweeps (first 25 conditions) of run 9 best and BC, so the
# pooled comparison has ~300 condition-evaluations per policy; then hand over to run 11.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/paired_sweeps.log
score() { wait_gpu; OUTF=$D/filter_ab/$1_v3_n25.json CKPT=$2 RUN_DIR=$D/filter_ab/$1 N_EVAL=25 MODES=det REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 timeout 9000 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_eval.log 2>&1; echo "$(date +%H:%M:%S) scored $1 exit=$?" >> $LOG; }
for k in 4 5 6 7; do score rl9_best_rep$k $D/rl9/grpo_ckpt_best.pt; score bc_rep$k $BC; done
echo "$(date +%H:%M:%S) PAIRED SWEEPS DONE" >> $LOG
exec $D/scripts/chain_rl11.sh
