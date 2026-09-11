#!/usr/bin/env bash
D=/home/jren313/research/starvla_rl/runs_g1_piston
while ! grep -q "FINISH RL7 DONE" $D/finish_rl7.log 2>/dev/null; do sleep 300; done
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 9500 ] || pgrep -f "bin/python tools/g1_piston/(train_sac|eval_checkpoint|render_rollouts)\.py" >/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) LAUNCHING run 8" >> $D/rl8/chain.log
cd $D && ./scripts/run_rl8.sh >> $D/rl8/nohup.log 2>&1
echo "$(date +%H:%M:%S) run 8 finished" >> $D/rl8/chain.log
