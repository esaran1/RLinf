#!/usr/bin/env bash
# run 9 ends -> score+render its best -> run 9b (continuation) -> score+render its best
# -> deferred run-8 rescoring and H10. Sequential on one GPU.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/chain_rl9.log
score() { wait_gpu; env $3 OUTF=$D/filter_ab/$1_v3_n25.json CKPT=$2 RUN_DIR=$D/filter_ab/$1 N_EVAL=25 MODES=det REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 timeout 9000 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_eval.log 2>&1; echo "$(date +%H:%M:%S) scored $1 exit=$?" >> $LOG; }
render_if_pass() {  # $1 tag $2 ckpt
  PASS=$($PY -c "
import json
try: d=json.load(open('$D/filter_ab/$1_v3_n25.json')); m=d['modes']['deterministic']; print('1' if d.get('_status')=='OK' and m['grasp_rate']>=0.87 else '0')
except Exception: print('0')")
  if [ "$PASS" = "1" ]; then
    CONDS=$($PY -c "
import json;pc=json.load(open('$D/filter_ab/$1_v3_n25.json'))['modes']['deterministic']['per_condition']
best=sorted(pc,key=lambda r:(-int(bool(r['stages'].get('press'))),-int(bool(r['stages'].get('lift'))),-r['return']))[:12]
print(','.join(str(r['condition']) for r in best))")
    mkdir -p $D/videos/$1; wait_gpu
    OUTDIR=$D/videos/$1 CKPT=$2 CONDS=$CONDS MODE=deterministic TAG=$1 REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 FPS=20 LOG_STEPS=1 timeout 9000 $PY $T/render_rollouts.py > $D/videos/$1_render.log 2>&1
    echo "$(date +%H:%M:%S) rendered $1 conds $CONDS exit=$?" >> $LOG
  else echo "$(date +%H:%M:%S) $1 did not pass H4; no videos" >> $LOG; fi
}
while ! grep -q "rl9 exit=" $D/rl9/train.log 2>/dev/null; do sleep 120; done
echo "$(date +%H:%M:%S) run 9 ended" >> $LOG
score rl9_best $D/rl9/grpo_ckpt_best.pt; render_if_pass rl9_best $D/rl9/grpo_ckpt_best.pt
wait_gpu; echo "$(date +%H:%M:%S) LAUNCHING run 9b" >> $LOG
$D/scripts/run_rl9b.sh; echo "$(date +%H:%M:%S) run 9b ended" >> $LOG
[ -f $D/rl9b/grpo_ckpt_best.pt ] && { score rl9b_best $D/rl9b/grpo_ckpt_best.pt; render_if_pass rl9b_best $D/rl9b/grpo_ckpt_best.pt; }
score rl8 $D/rl8/rlpd_ckpt_latest.pt
wait_gpu; OUTF=$D/diag/h10_probe_rl8.json CKPT=$D/rl8/rlpd_ckpt_latest.pt RUNJSON=$D/rl8/run.json SCRATCH=$D timeout 1800 $PY $T/probe_critic_ranking.py > $D/diag/h10_probe_rl8.log 2>&1; echo "$(date +%H:%M:%S) H10 rl8 exit=$?" >> $LOG
echo "$(date +%H:%M:%S) CHAIN RL9 DONE" >> $LOG
