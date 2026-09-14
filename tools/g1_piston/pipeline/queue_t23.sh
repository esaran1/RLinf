#!/usr/bin/env bash
# Start the conditions 2,3 generator when the canonical one has finished (two sims at a time).
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
R=/home/jren313/research/starvla_rl
while ! grep -q '"_status": "OK"\|"_status": "FAIL"' $R/demo_buffer_v5_dispense_canon/_build_status.json 2>/dev/null; do sleep 60; done
B=$R/demo_buffer_v5_dispense_t23; mkdir -p $B
OUTDIR=$B CONDS=2,3 KEEP_FAILED=0 VARIANTS='[{"name":"dispense","dispense":true}]' $PY tools/g1_piston/make_press_demos.py > $B/build.log 2>&1
