#!/usr/bin/env bash
# After run 10: score every saved iteration TWICE under v3 (scorer of record), select the
# iteration with the highest mean return whose grasp >= 0.87 in both sweeps, render it.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/chain_rl11.log
while ! grep -q "CERTIFY N50 DONE" $D/certify_n50.log 2>/dev/null; do sleep 120; done
wait_gpu; echo "$(date +%H:%M:%S) LAUNCHING run 11" >> $LOG; $D/scripts/run_rl11.sh
echo "$(date +%H:%M:%S) run 10 ended" >> $LOG
score() { wait_gpu; OUTF=$D/filter_ab/$1_v3_n25.json CKPT=$2 RUN_DIR=$D/filter_ab/$1 N_EVAL=25 MODES=det REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 timeout 9000 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_eval.log 2>&1; echo "$(date +%H:%M:%S) scored $1 exit=$?" >> $LOG; }
for f in $(ls $D/rl11/grpo_ckpt_iter*.pt 2>/dev/null | grep -v iter0 | sort -t r -k3 -n); do n=$(basename $f .pt | sed 's/grpo_ckpt_//'); score rl11_${n}_s1 $f; score rl11_${n}_s2 $f; done
BEST=$($PY -c "
import json,glob,re
cand={}
for p in glob.glob('$D/filter_ab/rl11_iter*_s?_v3_n25.json'):
    it=re.search(r'rl11_(iter\d+)_s',p).group(1)
    try: d=json.load(open(p)); m=d['modes']['deterministic']
    except Exception: continue
    if d.get('_status')=='OK': cand.setdefault(it,[]).append((m['grasp_rate'],m['mean_return']))
ok=[(sum(r for g,r in v)/len(v),k) for k,v in cand.items() if len(v)==2 and all(g>=0.87 for g,r in v)]
print(max(ok)[1] if ok else '')")
echo "$(date +%H:%M:%S) selected: $BEST" >> $LOG
if [ -n "$BEST" ]; then
  CK=$D/rl11/grpo_ckpt_$BEST.pt
  CONDS=$($PY -c "
import json;pc=json.load(open('$D/filter_ab/rl11_${BEST}_s1_v3_n25.json'))['modes']['deterministic']['per_condition']
best=sorted(pc,key=lambda r:(-int(bool(r['stages'].get('press'))),-int(bool(r['stages'].get('lift'))),-r['return']))[:12]
print(','.join(str(r['condition']) for r in best))")
  mkdir -p $D/videos/rl11_$BEST; wait_gpu
  OUTDIR=$D/videos/rl11_$BEST CKPT=$CK CONDS=$CONDS MODE=deterministic TAG=rl11_$BEST REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 FPS=20 LOG_STEPS=1 timeout 9000 $PY $T/render_rollouts.py > $D/videos/rl11_${BEST}_render.log 2>&1
  echo "$(date +%H:%M:%S) rendered rl11_$BEST conds $CONDS exit=$?" >> $LOG
fi
echo "$(date +%H:%M:%S) CHAIN RL11 DONE" >> $LOG
