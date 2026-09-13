#!/usr/bin/env bash
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
while ! grep -q "CERTIFY N50 DONE" $D/certify_n50.log 2>/dev/null; do sleep 180; done
cp $D/filter_ab/rl9_best_v3_n50.json verified_results/manifests/grpo_rl9_best_v3_n50.json; cp $D/filter_ab/bc_v3_n50.json verified_results/manifests/bc_v3_n50.json
$PY tools/g1_piston/report_n50.py > $D/report_n50.txt 2>&1
git add verified_results/FINAL_REPORT.md verified_results/manifests && git commit -q -s -m "docs(g1_piston): 50-condition certification of run 9 best vs BC (auto)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || true
echo "$(date +%H:%M:%S) autoreport n50 done" >> $D/autoreport.log
