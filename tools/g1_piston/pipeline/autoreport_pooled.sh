#!/usr/bin/env bash
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
while ! grep -q "PAIRED SWEEPS DONE" $D/paired_sweeps.log 2>/dev/null; do sleep 180; done
for k in 4 5 6 7; do cp $D/filter_ab/rl9_best_rep${k}_v3_n25.json verified_results/manifests/grpo_rl9_best_rep${k}_v3_n25.json; cp $D/filter_ab/bc_rep${k}_v3_n25.json verified_results/manifests/bc_rep${k}_v3_n25.json; done
$PY tools/g1_piston/report_pooled.py > $D/report_pooled.txt 2>&1
git add verified_results/FINAL_REPORT.md verified_results/manifests && git commit -q -s -m "docs(g1_piston): pooled paired comparison run 9 vs BC over all sweeps (auto)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || true
echo "$(date +%H:%M:%S) autoreport pooled done" >> $D/autoreport.log
