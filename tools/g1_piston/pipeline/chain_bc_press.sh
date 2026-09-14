#!/usr/bin/env bash
# BC retrain on human transport + scripted press demonstrations, then certification.
#   BUF   demonstration buffer directory (default demo_buffer_v5_press)
#   TAG   run tag (default bc_press)
# Scores: 2 fresh-process sweeps under v3 (transport scorer of record) and 2 under v5
# (press scorer), plus a second BC-baseline v5 sweep so the comparison is paired.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
BUF=${BUF:-/home/jren313/research/starvla_rl/demo_buffer_v5_press}
TAG=${TAG:-bc_press}
LOG=$D/chain_$TAG.log
mkdir -p $D/$TAG
echo "$(date +%H:%M:%S) chain $TAG start buf=$BUF" >> $LOG
wait_gpu
OUTF=$D/$TAG/train_run.json RUN_DIR=$D/$TAG DEMO_DIR=$BUF NORM_STATS=${NORM_STATS:-$BUF/dataset_statistics.json} EPOCHS=${EPOCHS:-300} $PY tools/g1_piston/train_bc.py > $D/$TAG/train.log 2>&1
echo "$(date +%H:%M:%S) trained exit=$?" >> $LOG
CK=$D/$TAG/bc_ckpt_latest.pt
for s in 1 2; do
  wait_gpu
  OUTF=$D/filter_ab/${TAG}_s${s}_v3_n25.json CKPT=$CK RUN_DIR=$D/$TAG N_EVAL=25 MODES=det REWARD_V3=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 $PY tools/g1_piston/eval_checkpoint.py > $D/$TAG/eval_v3_s$s.log 2>&1
  echo "$(date +%H:%M:%S) scored ${TAG}_s${s} v3 exit=$?" >> $LOG
  wait_gpu
  OUTF=$D/filter_ab/${TAG}_s${s}_v5_n25.json CKPT=$CK RUN_DIR=$D/$TAG N_EVAL=25 MODES=det REWARD_V5=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 $PY tools/g1_piston/eval_checkpoint.py > $D/$TAG/eval_v5_s$s.log 2>&1
  echo "$(date +%H:%M:%S) scored ${TAG}_s${s} v5 exit=$?" >> $LOG
done
for s in 1 2; do if [ -f $D/filter_ab/bc_working_v5_s${s}_n25.json ]; then continue; fi
  wait_gpu
  OUTF=$D/filter_ab/bc_working_v5_s${s}_n25.json CKPT=$BC RUN_DIR=$D/diag N_EVAL=25 MODES=det REWARD_V5=1 RESET_SUITE_SEED=20260817 EP_CHUNKS=23 $PY tools/g1_piston/eval_checkpoint.py > $D/diag/bc_working_v5_s$s.log 2>&1
  echo "$(date +%H:%M:%S) scored bc_working v5 s$s exit=$?" >> $LOG
done
echo "$(date +%H:%M:%S) CHAIN $(echo $TAG | tr a-z A-Z) DONE" >> $LOG
