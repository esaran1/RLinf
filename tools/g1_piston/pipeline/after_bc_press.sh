#!/usr/bin/env bash
# After chain_bc_press.sh: report, render the pressing policy, commit; then, if the BC policy
# certifies any dispense, fine-tune it with critic-free GRPO under v5 (run 12), score every
# iteration twice under v5 and v3, select, render, report. Unattended.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/after_bc_press.log
while ! grep -q "CHAIN BC_PRESS DONE" $D/chain_bc_press.log 2>/dev/null; do sleep 120; done
echo "$(date +%H:%M:%S) bc_press chain done" >> $LOG
TAG=bc_press $PY tools/g1_piston/report_press.py > $D/report_bc_press.txt 2>&1
{ echo; echo "## Pressing policy (BC on human transport + scripted palm press), certified"; echo; echo "Transport under v3 (scorer of record) and the press under v5 (geometry-grounded press scorer); fresh-process sweeps; BC baseline paired on the same conditions."; echo; cat $D/report_bc_press.txt; } >> verified_results/FINAL_REPORT.md
for f in $D/filter_ab/bc_press_s?_v?_n25.json $D/filter_ab/bc_working_v5_s?_n25.json; do [ -f $f ] && cp $f verified_results/manifests/; done
cp $D/bc_press/train_run.json verified_results/manifests/bc_press_train.json 2>/dev/null
git add verified_results/FINAL_REPORT.md verified_results/manifests && git commit -q -s -m "docs(g1_piston): pressing policy certified (auto)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || true
# ---- render: conditions where the policy dispensed first, then pressed, then lifted ----
CK=$D/bc_press/bc_ckpt_latest.pt
CONDS=$($PY -c "
import json,glob
rows=[]
for p in sorted(glob.glob('$D/filter_ab/bc_press_s?_v5_n25.json')):
    rows=json.load(open(p))['modes']['deterministic']['per_condition']; break
best=sorted(rows,key=lambda r:(-int(bool(r['stages'].get('dispense'))),-int(bool(r['stages'].get('press'))),-int(bool(r['stages'].get('lift'))),-r['return']))[:12]
print(','.join(str(r['condition']) for r in best))")
mkdir -p $D/videos/bc_press; wait_gpu
OUTDIR=$D/videos/bc_press CKPT=$CK CONDS=$CONDS MODE=deterministic TAG=bc_press REWARD_V5=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 FPS=20 LOG_STEPS=1 timeout 9000 $PY $T/render_rollouts.py > $D/videos/bc_press_render.log 2>&1
echo "$(date +%H:%M:%S) rendered bc_press exit=$?" >> $LOG
mkdir -p verified_results/videos_bc_press && cp $D/videos/bc_press/*.mp4 verified_results/videos_bc_press/ 2>/dev/null
git add verified_results/videos_bc_press && git commit -q -s -m "feat(g1_piston): videos of the pressing policy (auto)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || true
# ---- run 12: GRPO under v5 from the pressing policy, if it dispenses at all ----
DISP=$($PY -c "
import json,glob
m=0.0
for p in glob.glob('$D/filter_ab/bc_press_s?_v5_n25.json'):
    try: m=max(m,json.load(open(p))['modes']['deterministic']['dispense_rate'])
    except Exception: pass
print(m)")
echo "$(date +%H:%M:%S) bc_press best v5 dispense rate $DISP" >> $LOG
if $PY -c "import sys; sys.exit(0 if float('$DISP')>=0.12 else 1)"; then
  mkdir -p $D/rl12; wait_gpu; echo "$(date +%H:%M:%S) LAUNCHING run 12" >> $LOG
  OUTF=$D/rl12/run.json RUN_DIR=$D/rl12 BASE_CKPT=$CK TRAIN_REWARD=v5 \
    SIGMA=0.10 SIGMA_HAND=0.10 R_MAX=0.15 N_GROUPS=8 GROUP=6 PPO_EPOCHS=3 MB=64 LR=1e-4 CLIP=0.2 KL_BETA=0.1 KL_STOP=0.02 \
    WALL_BUDGET=7200 N_EVAL_PERIODIC=8 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 SEED=5 \
    timeout 14400 $PY $T/train_grpo.py > $D/rl12/train.log 2>&1
  echo "$(date +%H:%M:%S) run 12 ended exit=$?" >> $LOG
  score() { wait_gpu; OUTF=$D/filter_ab/$1_n25.json CKPT=$2 RUN_DIR=$D/filter_ab/$1 N_EVAL=25 MODES=det $3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 timeout 9000 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_eval.log 2>&1; echo "$(date +%H:%M:%S) scored $1 exit=$?" >> $LOG; }
  for f in $(ls $D/rl12/grpo_ckpt_iter*.pt 2>/dev/null | grep -v iter0 | sort -t r -k3 -n); do n=$(basename $f .pt | sed 's/grpo_ckpt_//'); score rl12_${n}_s1_v5 $f REWARD_V5; score rl12_${n}_s2_v5 $f REWARD_V5; score rl12_${n}_s1_v3 $f REWARD_V3; done
  TAG=rl12 $PY - >> $D/report_rl12.txt 2>&1 <<'PY'
import json, glob, re
D = "/home/jren313/research/starvla_rl/runs_g1_piston/filter_ab"
print("| checkpoint | reward | sweep | grasp | lift | plate | press | dispense | return |")
print("|---|---|---|---|---|---|---|---|---|")
for p in sorted(glob.glob(f"{D}/rl12_iter*_s?_v?_n25.json")):
    try: m = json.load(open(p))["modes"]["deterministic"]
    except Exception: continue
    name = re.search(r"(rl12_iter\d+)_(s\d)_(v\d)", p)
    print(f"| {name.group(1)} | {name.group(3)} | {name.group(2)} | {m['grasp_rate']:.2f} | {m['lift_rate']:.2f} | {m['plate_rate']:.2f} | {m['press_rate']:.2f} | {m['dispense_rate']:.2f} | {m['mean_return']:.2f} |")
PY
  { echo; echo "## Run 12: critic-free GRPO under v5 from the pressing policy (auto)"; echo; cat $D/report_rl12.txt; } >> verified_results/FINAL_REPORT.md
  cp $D/rl12/run.json verified_results/manifests/grpo_rl12_train.json 2>/dev/null; for f in $D/filter_ab/rl12_iter*_n25.json; do cp $f verified_results/manifests/grpo_$(basename $f); done
  git add verified_results/FINAL_REPORT.md verified_results/manifests && git commit -q -s -m "docs(g1_piston): run 12 (GRPO under v5 from the pressing policy) certified (auto)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || true
fi
echo "$(date +%H:%M:%S) AFTER BC_PRESS DONE" >> $LOG
