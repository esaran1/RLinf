#!/usr/bin/env bash
# After the 40-chunk sweeps: render the pressing policy at 40 chunks, then run 12 = critic-free
# GRPO under v5 from the pressing policy with 40-chunk episodes; score every iteration twice
# under v5 and once under v3 at 40 chunks; report. Unattended.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/chain_rl12.log
while ! grep -q "H40 BASELINE DONE" $D/chain_bc_press.log 2>/dev/null; do sleep 60; done
CK=$D/bc_press/bc_ckpt_latest.pt
CONDS=$($PY -c "
import json
rows=json.load(open('$D/filter_ab/bc_press_s1_v5_h40_n25.json'))['modes']['deterministic']['per_condition']
best=sorted(rows,key=lambda r:(-int(bool(r['stages'].get('success'))),-int(bool(r['stages'].get('dispense'))),-int(bool(r['stages'].get('press'))),-r['max_press_grasped_m']))[:12]
print(','.join(str(r['condition']) for r in best))")
mkdir -p $D/videos/bc_press_h40; wait_gpu; echo "$(date +%H:%M:%S) rendering bc_press h40 conds $CONDS" >> $LOG
OUTDIR=$D/videos/bc_press_h40 CKPT=$CK CONDS=$CONDS MODE=deterministic TAG=bc_press_h40 REWARD_V5=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=40 FPS=20 LOG_STEPS=1 timeout 9000 $PY $T/render_rollouts.py > $D/videos/bc_press_h40_render.log 2>&1
echo "$(date +%H:%M:%S) rendered exit=$?" >> $LOG
mkdir -p verified_results/videos_bc_press_h40 && cp $D/videos/bc_press_h40/*.mp4 verified_results/videos_bc_press_h40/ 2>/dev/null
git add verified_results/videos_bc_press_h40 && git commit -q -s -m "feat(g1_piston): 40-chunk videos of the pressing policy (auto)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || true
mkdir -p $D/rl12; wait_gpu; echo "$(date +%H:%M:%S) LAUNCHING run 12" >> $LOG
OUTF=$D/rl12/run.json RUN_DIR=$D/rl12 BASE_CKPT=$CK TRAIN_REWARD=v5 \
  SIGMA=0.10 SIGMA_HAND=0.10 R_MAX=0.15 N_GROUPS=8 GROUP=6 PPO_EPOCHS=3 MB=64 LR=1e-4 CLIP=0.2 KL_BETA=0.1 KL_STOP=0.02 \
  WALL_BUDGET=10800 N_EVAL_PERIODIC=8 RESET_SUITE_SEED=20260817 EP_CHUNKS=40 SEED=5 \
  timeout 18000 $PY $T/train_grpo.py > $D/rl12/train.log 2>&1
echo "$(date +%H:%M:%S) run 12 ended exit=$?" >> $LOG
score() { wait_gpu; OUTF=$D/filter_ab/$1_n25.json CKPT=$2 RUN_DIR=$D/filter_ab/$1 N_EVAL=25 MODES=det $3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=40 timeout 9000 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_eval.log 2>&1; echo "$(date +%H:%M:%S) scored $1 exit=$?" >> $LOG; }
for f in $(ls $D/rl12/grpo_ckpt_iter*.pt 2>/dev/null | grep -v iter0 | sort -t r -k3 -n); do n=$(basename $f .pt | sed 's/grpo_ckpt_//'); score rl12_${n}_s1_v5_h40 $f REWARD_V5; score rl12_${n}_s2_v5_h40 $f REWARD_V5; score rl12_${n}_s1_v3_h40 $f REWARD_V3; done
echo "$(date +%H:%M:%S) CHAIN RL12 DONE" >> $LOG
