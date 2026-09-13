#!/usr/bin/env bash
# When a run's chain completes, fill its report section and commit. RUN given as $1.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
RUN=$1; UP=$(echo $RUN | tr a-z A-Z)
while ! grep -q "CHAIN $UP DONE" $D/chain_$RUN.log 2>/dev/null; do sleep 180; done
RUN=$RUN $PY tools/g1_piston/report_run.py > $D/report_$RUN.txt 2>&1
for f in $D/filter_ab/${RUN}_iter*_s?_v3_n25.json; do cp $f verified_results/manifests/grpo_$(basename $f); done
cp $D/$RUN/run.json verified_results/manifests/grpo_${RUN}_train.json
git add verified_results/FINAL_REPORT.md verified_results/manifests && git commit -q -s -m "docs(g1_piston): certified $RUN section of the final report (auto)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || true
echo "$(date +%H:%M:%S) autoreport $RUN done" >> $D/autoreport.log
