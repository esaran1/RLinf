#!/usr/bin/env bash
# Paired repeats for run 11 iteration 2 (the strongest two-sweep candidate) against BC's
# existing rep sweeps, after the current queue; then pooled reports for both candidates.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/rl11_iter2_repeats.log
while ! grep -q "autoreport pooled done" $D/autoreport.log 2>/dev/null; do sleep 180; done
score() { wait_gpu; OUTF=$D/filter_ab/$1_v3_n25.json CKPT=$2 RUN_DIR=$D/filter_ab/$1 N_EVAL=25 MODES=det REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 timeout 9000 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_eval.log 2>&1; echo "$(date +%H:%M:%S) scored $1 exit=$?" >> $LOG; }
for k in 4 5 6 7; do score rl11_iter2_rep$k $D/rl11/grpo_ckpt_iter2.pt; done
for k in 4 5 6 7; do cp $D/filter_ab/rl11_iter2_rep${k}_v3_n25.json verified_results/manifests/grpo_rl11_iter2_rep${k}_v3_n25.json; done
CAND=rl11_iter2 $PY tools/g1_piston/report_pooled.py > $D/report_pooled_rl11.txt 2>&1
git add verified_results/FINAL_REPORT.md verified_results/manifests && git commit -q -s -m "docs(g1_piston): pooled paired comparison run 11 iter 2 vs BC (auto)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || true
echo "$(date +%H:%M:%S) RL11 ITER2 REPEATS DONE" >> $LOG
