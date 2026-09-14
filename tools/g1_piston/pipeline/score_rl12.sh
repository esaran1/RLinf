#!/usr/bin/env bash
# Re-score run 12 (the chain's score step passed the reward flag as a bare word). Every
# iteration: v5 twice, v3 once, 40-chunk horizon, fresh processes. v5 first on all
# iterations so the press verdict lands before the transport sweeps.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/chain_rl12.log
score() { wait_gpu; OUTF=$D/filter_ab/$1_n25.json CKPT=$2 RUN_DIR=$D/filter_ab/$1 N_EVAL=25 MODES=det RESET_SUITE_SEED=20260817 EP_CHUNKS=40 timeout 9000 env $3=1 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_eval.log 2>&1; echo "$(date +%H:%M:%S) rescored $1 exit=$?" >> $LOG; }
for s in s1 s2; do for n in iter2 iter1 iter3; do score rl12_${n}_${s}_v5_h40 $D/rl12/grpo_ckpt_$n.pt REWARD_V5; done; done
for n in iter2 iter1 iter3; do score rl12_${n}_s1_v3_h40 $D/rl12/grpo_ckpt_$n.pt REWARD_V3; done
echo "$(date +%H:%M:%S) SCORE RL12 DONE" >> $LOG
