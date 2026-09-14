#!/usr/bin/env bash
# After run 12's scoring: two more paired 40-chunk v5 sweeps each of the pressing policy and
# run 12 iteration 2, then a pooled paired comparison appended to the final report.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
LOG=$D/chain_rl12.log
while ! grep -q "REPORT RL12 DONE" $LOG 2>/dev/null; do sleep 120; done
score() { wait_gpu; OUTF=$D/filter_ab/$1_n25.json CKPT=$2 RUN_DIR=$D/filter_ab/$1 N_EVAL=25 MODES=det RESET_SUITE_SEED=20260817 EP_CHUNKS=40 timeout 9000 env REWARD_V5=1 $PY $T/eval_checkpoint.py > $D/filter_ab/$1_eval.log 2>&1; echo "$(date +%H:%M:%S) extra $1 exit=$?" >> $LOG; }
for s in s3 s4; do score bc_press_${s}_v5_h40 $D/bc_press/bc_ckpt_latest.pt; score rl12_iter2_${s}_v5_h40 $D/rl12/grpo_ckpt_iter2.pt; done
$PY - > $D/report_extra_h40.txt <<'PY'
import json, glob
D = "/home/jren313/research/starvla_rl/runs_g1_piston/filter_ab"
def rows(p): return json.load(open(p))["modes"]["deterministic"]["per_condition"]
A = {s: rows(f"{D}/bc_press_{s}_v5_h40_n25.json") for s in ("s1", "s2", "s3", "s4") if glob.glob(f"{D}/bc_press_{s}_v5_h40_n25.json")}
B = {s: rows(f"{D}/rl12_iter2_{s}_v5_h40_n25.json") for s in ("s1", "s2", "s3", "s4") if glob.glob(f"{D}/rl12_iter2_{s}_v5_h40_n25.json")}
print("| sweep | bc_press dispense | rl12 iter2 dispense | bc_press success | rl12 iter2 success | bc_press lift | rl12 iter2 lift |")
print("|---|---|---|---|---|---|---|")
def rate(rs, k): return sum(1 for x in rs if x["stages"].get(k)) / len(rs)
pa = pb = []
for s in sorted(set(A) & set(B)):
    print(f"| {s} | {rate(A[s],'dispense'):.2f} | {rate(B[s],'dispense'):.2f} | {rate(A[s],'success'):.2f} | {rate(B[s],'success'):.2f} | {rate(A[s],'lift'):.2f} | {rate(B[s],'lift'):.2f} |")
    pa = pa + A[s]; pb = pb + B[s]
n = len(pa)
if n:
    import random
    da = [int(bool(x["stages"].get("dispense"))) for x in pa]; db = [int(bool(x["stages"].get("dispense"))) for x in pb]
    diffs = [b - a for a, b in zip(da, db)]
    random.seed(0); boots = []
    for _ in range(4000):
        smp = [diffs[random.randrange(n)] for _ in range(n)]; boots.append(sum(smp) / n)
    boots.sort(); lo, hi = boots[int(0.025 * n and 100)], boots[int(0.975 * 4000) - 1]; lo = boots[100]
    print(); print(f"**Pooled over {n} paired condition-evaluations per policy (v5, 40 chunks):** dispense rl12 iter2 {sum(db)/n:.3f} vs bc_press {sum(da)/n:.3f} (difference {sum(diffs)/n:+.3f}, paired-bootstrap 95% CI {lo:+.3f} to {hi:+.3f}); "
          f"full success {rate(pb,'success'):.3f} vs {rate(pa,'success'):.3f}; lift {rate(pb,'lift'):.3f} vs {rate(pa,'lift'):.3f}.")
    print(); print("Verdict: **" + ("established" if lo > 0 or hi < 0 else "not established") + "** on dispense (CI " + ("excludes" if lo > 0 or hi < 0 else "includes") + " zero).")
PY
{ echo; echo "## Paired 40-chunk comparison: run 12 iteration 2 vs the pressing policy (auto)"; echo; cat $D/report_extra_h40.txt; } >> verified_results/FINAL_REPORT.md
for f in $D/filter_ab/bc_press_s?_v5_h40_n25.json $D/filter_ab/rl12_iter2_s?_v5_h40_n25.json; do cp $f verified_results/manifests/; done
git add verified_results/FINAL_REPORT.md verified_results/manifests && git commit -q -s -m "docs(g1_piston): paired 40-chunk comparison, run 12 iteration 2 vs the pressing policy (auto)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || true
echo "$(date +%H:%M:%S) EXTRA H40 DONE" >> $LOG
