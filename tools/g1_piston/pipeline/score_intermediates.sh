#!/usr/bin/env bash
# Scorer of record on every saved iteration of runs 9 and 9b, so the continuation
# protocol is decided by certified numbers rather than in-trainer selection.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/score_intermediates.log
score() { wait_gpu; OUTF=$D/filter_ab/$1_v3_n25.json CKPT=$2 RUN_DIR=$D/filter_ab/$1 N_EVAL=25 MODES=det REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 timeout 9000 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_eval.log 2>&1; echo "$(date +%H:%M:%S) scored $1 exit=$?" >> $LOG; }
score rl9_iter2 $D/rl9/grpo_ckpt_iter2.pt
for i in 1 2 3 4; do [ -f $D/rl9b/grpo_ckpt_iter$i.pt ] && score rl9b_iter$i $D/rl9b/grpo_ckpt_iter$i.pt; done
echo "$(date +%H:%M:%S) SCORE INTERMEDIATES DONE" >> $LOG
