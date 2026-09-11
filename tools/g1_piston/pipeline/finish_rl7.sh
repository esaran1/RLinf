#!/usr/bin/env bash
# After run 7 ends: score with the scorer of record; render videos if H4 (grasp>=0.87) passes.
set -uo pipefail
D=/home/jren313/research/starvla_rl/runs_g1_piston
T=/home/jren313/research/starvla_rl/RLinf/tools/g1_piston
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
export MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/finish_rl7.log
while ! grep -q "rl7 exit=" $D/rl7/train.log 2>/dev/null; do sleep 180; done
CK=$D/rl7/rlpd_ckpt_latest.pt; [ -f "$CK" ] || { echo "$(date) no checkpoint" >> $LOG; exit 1; }
while pgrep -f "bin/python tools/g1_piston/train_sac.py" >/dev/null; do sleep 60; done
OUTF=$D/filter_ab/rl7_v3_n25.json CKPT=$CK RUN_DIR=$D/filter_ab/rl7 N_EVAL=25 MODES=det REWARD_V3=1 \
  RESET_SUITE_SEED=20260817 EP_CHUNKS=23 timeout 9000 $PY $T/eval_checkpoint.py > $D/filter_ab/rl7_eval.log 2>&1
echo "$(date +%H:%M:%S) scored exit=$?" >> $LOG
PASS=$($PY -c "
import json;d=json.load(open('$D/filter_ab/rl7_v3_n25.json'));m=d['modes']['deterministic'];print('1' if d.get('_status')=='OK' and m['grasp_rate']>=0.87 else '0')")
if [ "$PASS" = "1" ]; then
  CONDS=$($PY -c "
import json;pc=json.load(open('$D/filter_ab/rl7_v3_n25.json'))['modes']['deterministic']['per_condition']
best=sorted(pc,key=lambda r:(-int(bool(r['stages'].get('press'))),-int(bool(r['stages'].get('lift'))),-r['return']))[:12]
print(','.join(str(r['condition']) for r in best))")
  mkdir -p $D/videos/rl7
  OUTDIR=$D/videos/rl7 CKPT=$CK CONDS=$CONDS MODE=deterministic TAG=rl7 REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 FPS=20 LOG_STEPS=1 \
    timeout 9000 $PY $T/render_rollouts.py > $D/videos/rl7_render.log 2>&1
  echo "$(date +%H:%M:%S) rendered $CONDS exit=$?" >> $LOG
else
  echo "$(date +%H:%M:%S) H4 not passed; no videos" >> $LOG
fi
# order-dependence probe on BC, fresh process, reversed conditions
OUTF=$D/filter_ab/bc_v3_n25_reverse.json CKPT=/home/jren313/research/starvla_rl/checkpoints/g1_piston_bc_working/bc_ckpt_latest.pt \
  RUN_DIR=$D/filter_ab/bc_rev N_EVAL=25 MODES=det REWARD_V3=1 COND_ORDER=reverse RESET_SUITE_SEED=20260817 EP_CHUNKS=23 \
  timeout 9000 $PY $T/eval_checkpoint.py > $D/filter_ab/bc_rev.log 2>&1
echo "$(date +%H:%M:%S) reverse-order probe exit=$?" >> $LOG
echo "$(date +%H:%M:%S) FINISH RL7 DONE" >> $LOG
