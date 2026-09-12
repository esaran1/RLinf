#!/usr/bin/env bash
# Paired repeats of the scorer of record: run 9 iteration 1 vs BC, two more fresh-process
# scorings each, so the +0.16 lift is judged against the scorer's own repeat noise.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/repeat_scoring.log
while ! grep -q "CHAIN RL9C DONE" $D/chain_rl9c.log 2>/dev/null; do sleep 120; done
score() { wait_gpu; OUTF=$D/filter_ab/$1_v3_n25.json CKPT=$2 RUN_DIR=$D/filter_ab/$1 N_EVAL=25 MODES=det REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 timeout 9000 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_eval.log 2>&1; echo "$(date +%H:%M:%S) scored $1 exit=$?" >> $LOG; }
for k in 2 3; do score rl9_best_rep$k $D/rl9/grpo_ckpt_best.pt; score bc_rep$k $BC; done
echo "$(date +%H:%M:%S) REPEATS DONE" >> $LOG
