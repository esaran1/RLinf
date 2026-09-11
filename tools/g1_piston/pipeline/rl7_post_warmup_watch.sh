#!/usr/bin/env bash
# H8 of record: first checkpoint saved AFTER the residual starts updating.
D=/home/jren313/research/starvla_rl/runs_g1_piston
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
cd /home/jren313/research/starvla_rl/RLinf
while true; do
  N=$($PY -c "
import json,glob
try:
    d=json.load(open('$D/rl7/run.json')); tl=d.get('train_log') or []
    first=[r['env_steps'] for r in tl if r.get('actor_updated')]
    cks=sorted(glob.glob('$D/rl7/rlpd_ckpt_step*.pt'), key=lambda p:int(p.split('step')[-1][:-3]))
    ok=[p for p in cks if first and int(p.split('step')[-1][:-3])>=first[0]]
    print(ok[0] if ok else '')
except Exception: print('')" 2>/dev/null)
  if [ -n "$N" ]; then
    echo "$(date +%H:%M:%S) first post-warmup checkpoint: $N" > $D/diag/rl7_post_warmup.log
    OUTF=$D/diag/rl7_post_warmup.json CKPT=$N $PY tools/g1_piston/measure_deployed_action_error.py >> $D/diag/rl7_post_warmup.log 2>&1
    echo "$(date +%H:%M:%S) exit=$?" >> $D/diag/rl7_post_warmup.log; exit 0
  fi
  sleep 180
done
