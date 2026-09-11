#!/usr/bin/env bash
# After run 8: score it (scorer of record), render if H4 passes, then run the two jobs
# the stuck finish_rl7 never did: score run 7's final checkpoint, and the reversed-order
# BC probe. Sequential on the GPU.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/finish_rl8.log
while ! grep -q "rl8 exit=" $D/rl8/train.log 2>/dev/null; do sleep 180; done
echo "$(date +%H:%M:%S) run 8 ended" >> $LOG
score() {  # $1 tag  $2 ckpt  [$3 extra env]
  wait_gpu
  env $3 OUTF=$D/filter_ab/$1_v3_n25.json CKPT=$2 RUN_DIR=$D/filter_ab/$1 N_EVAL=25 MODES=det REWARD_V3=1 \
    RESET_SUITE_SEED=20260817 EP_CHUNKS=23 timeout 9000 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_eval.log 2>&1
  echo "$(date +%H:%M:%S) scored $1 exit=$?" >> $LOG
}
[ -f $D/rl8/rlpd_ckpt_latest.pt ] && score rl8 $D/rl8/rlpd_ckpt_latest.pt || echo "$(date) run 8 has no checkpoint" >> $LOG
PASS=$($PY -c "
import json
try: d=json.load(open('$D/filter_ab/rl8_v3_n25.json')); m=d['modes']['deterministic']; print('1' if d.get('_status')=='OK' and m['grasp_rate']>=0.87 else '0')
except Exception: print('0')")
if [ "$PASS" = "1" ]; then
  CONDS=$($PY -c "
import json;pc=json.load(open('$D/filter_ab/rl8_v3_n25.json'))['modes']['deterministic']['per_condition']
best=sorted(pc,key=lambda r:(-int(bool(r['stages'].get('press'))),-int(bool(r['stages'].get('lift'))),-r['return']))[:12]
print(','.join(str(r['condition']) for r in best))")
  mkdir -p $D/videos/rl8; wait_gpu
  OUTDIR=$D/videos/rl8 CKPT=$D/rl8/rlpd_ckpt_latest.pt CONDS=$CONDS MODE=deterministic TAG=rl8 REWARD_V3=1 \
    RESET_SUITE_SEED=20260817 EP_CHUNKS=23 FPS=20 LOG_STEPS=1 timeout 9000 $PY $T/render_rollouts.py > $D/videos/rl8_render.log 2>&1
  echo "$(date +%H:%M:%S) rendered $CONDS exit=$?" >> $LOG
else
  echo "$(date +%H:%M:%S) run 8 did not pass H4; no videos" >> $LOG
fi
score rl7 $D/rl7/rlpd_ckpt_latest.pt
score bc_reverse $BC "COND_ORDER=reverse"
echo "$(date +%H:%M:%S) FINISH RL8 DONE" >> $LOG
