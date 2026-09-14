#!/usr/bin/env bash
# After the dispense generators finish: assemble the pressing-policy buffer, then train
# and certify the BC policy (chain_bc_press.sh). Unattended.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
R=/home/jren313/research/starvla_rl
GENS="$R/demo_buffer_v5_palm_canon $R/demo_buffer_v5_palm_t01 $R/demo_buffer_v5_palm_t23"
LOG=$D/press_pipeline.log
echo "$(date +%H:%M:%S) waiting for generators: $GENS" >> $LOG
for g in $GENS; do
  while ! grep -q '"_status": "OK"\|"_status": "FAIL"' $g/_build_status.json 2>/dev/null; do sleep 120; done
  echo "$(date +%H:%M:%S) $g finished: $(grep -o '"n_certified": [0-9]*' $g/_build_status.json)" >> $LOG
done
OUT=$R/demo_buffer_v5_press SYNTH_DIRS=$(echo $GENS | tr ' ' ',') $PY tools/g1_piston/build_press_bc_buffer.py >> $LOG 2>&1
echo "$(date +%H:%M:%S) buffer built: $(grep -o '"n_episodes": [0-9]*' $R/demo_buffer_v5_press/_manifest.json)" >> $LOG
BUF=$R/demo_buffer_v5_press TAG=bc_press bash /home/jren313/research/starvla_rl/runs_g1_piston/scripts/chain_bc_press.sh
echo "$(date +%H:%M:%S) PRESS PIPELINE DONE" >> $LOG
