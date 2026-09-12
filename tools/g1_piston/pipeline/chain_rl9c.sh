#!/usr/bin/env bash
# After the intermediate scoring: run 9c, then score EVERY saved iteration with the scorer
# of record, pick the certified best (grasp >= 0.87, max return), render it.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/chain_rl9c.log
while ! grep -q "SCORE INTERMEDIATES DONE" $D/score_intermediates.log 2>/dev/null; do sleep 120; done
wait_gpu; echo "$(date +%H:%M:%S) LAUNCHING run 9c" >> $LOG
$D/scripts/run_rl9c.sh; echo "$(date +%H:%M:%S) run 9c ended" >> $LOG
score() { wait_gpu; OUTF=$D/filter_ab/$1_v3_n25.json CKPT=$2 RUN_DIR=$D/filter_ab/$1 N_EVAL=25 MODES=det REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 timeout 9000 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_eval.log 2>&1; echo "$(date +%H:%M:%S) scored $1 exit=$?" >> $LOG; }
for f in $(ls $D/rl9c/grpo_ckpt_iter*.pt 2>/dev/null | grep -v iter0 | sort -t r -k3 -n); do n=$(basename $f .pt | sed 's/grpo_ckpt_//'); score rl9c_$n $f; done
BEST=$($PY -c "
import json,glob
best=None
for p in glob.glob('$D/filter_ab/rl9c_iter*_v3_n25.json'):
    try: d=json.load(open(p)); m=d['modes']['deterministic']
    except Exception: continue
    if d.get('_status')=='OK' and m['grasp_rate']>=0.87 and (best is None or m['mean_return']>best[1]): best=(p.split('/')[-1].replace('_v3_n25.json',''),m['mean_return'])
print(best[0] if best else '')")
echo "$(date +%H:%M:%S) certified best: $BEST" >> $LOG
if [ -n "$BEST" ]; then
  CK=$D/rl9c/grpo_ckpt_${BEST#rl9c_}.pt
  CONDS=$($PY -c "
import json;pc=json.load(open('$D/filter_ab/${BEST}_v3_n25.json'))['modes']['deterministic']['per_condition']
best=sorted(pc,key=lambda r:(-int(bool(r['stages'].get('press'))),-int(bool(r['stages'].get('lift'))),-r['return']))[:12]
print(','.join(str(r['condition']) for r in best))")
  mkdir -p $D/videos/$BEST; wait_gpu
  OUTDIR=$D/videos/$BEST CKPT=$CK CONDS=$CONDS MODE=deterministic TAG=$BEST REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 FPS=20 LOG_STEPS=1 timeout 9000 $PY $T/render_rollouts.py > $D/videos/${BEST}_render.log 2>&1
  echo "$(date +%H:%M:%S) rendered $BEST conds $CONDS exit=$?" >> $LOG
fi
echo "$(date +%H:%M:%S) CHAIN RL9C DONE" >> $LOG
