#!/usr/bin/env bash
# H10 (run8_preregistration.json): run the critic-ranking probe on run 8's final
# checkpoint once the scorer of record has finished with it.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
while ! grep -q "scored rl8" $D/finish_rl8.log 2>/dev/null; do sleep 180; done
while gpu_busy; do sleep 60; done
OUTF=$D/diag/h10_probe_rl8.json CKPT=$D/rl8/rlpd_ckpt_latest.pt RUNJSON=$D/rl8/run.json SCRATCH=$D \
  timeout 1800 $PY $T/probe_critic_ranking.py > $D/diag/h10_probe_rl8.log 2>&1
echo "$(date +%H:%M:%S) H10 probe exit=$?" >> $D/finish_rl8.log
