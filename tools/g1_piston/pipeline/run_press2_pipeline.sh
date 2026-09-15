#!/usr/bin/env bash
# Press data scaled: wait for the t47/t811 generators, assemble demo_buffer_v5_press2 from all
# five palm-press generators + truncated originals, train bc_press2, certify at 40 chunks
# (v5 x4 on the same conditions as bc_press s1-s4, v3 x2), append a paired report, commit.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
R=/home/jren313/research/starvla_rl
LOG=$D/press2_pipeline.log
GENS="$R/demo_buffer_v5_palm_t47 $R/demo_buffer_v5_palm_t811"
for g in $GENS; do while ! grep -q '"_status": "OK"\|"_status": "FAIL"' $g/_build_status.json 2>/dev/null; do sleep 120; done; echo "$(date +%H:%M:%S) $g: $(grep -o '"n_certified": [0-9]*' $g/_build_status.json)" >> $LOG; done
ALL="$R/demo_buffer_v5_palm_canon,$R/demo_buffer_v5_palm_t01,$R/demo_buffer_v5_palm_t23,$R/demo_buffer_v5_palm_t47,$R/demo_buffer_v5_palm_t811"
OUT=$R/demo_buffer_v5_press2 SYNTH_DIRS=$ALL $PY tools/g1_piston/build_press_bc_buffer.py >> $LOG 2>&1
echo "$(date +%H:%M:%S) buffer: $(grep -o '"n_synthetic": [0-9]*' $R/demo_buffer_v5_press2/_manifest.json) $(grep -o '"n_episodes": [0-9]*' $R/demo_buffer_v5_press2/_manifest.json)" >> $LOG
TAG=bc_press2; mkdir -p $D/$TAG; wait_gpu
OUTF=$D/$TAG/train_run.json RUN_DIR=$D/$TAG DEMO_DIR=$R/demo_buffer_v5_press2 NORM_STATS=$R/demo_buffer_v5_press2/dataset_statistics.json EPOCHS=300 $PY tools/g1_piston/train_bc.py > $D/$TAG/train.log 2>&1
echo "$(date +%H:%M:%S) trained exit=$?" >> $LOG
CK=$D/$TAG/bc_ckpt_latest.pt
score() { wait_gpu; OUTF=$D/filter_ab/$1_n25.json CKPT=$CK RUN_DIR=$D/$TAG N_EVAL=25 MODES=det RESET_SUITE_SEED=20260817 EP_CHUNKS=40 timeout 9000 env $2=1 $PY $T/eval_checkpoint.py > $D/$TAG/$1.log 2>&1; echo "$(date +%H:%M:%S) scored $1 exit=$?" >> $LOG; }
for s in s1 s2 s3 s4; do score ${TAG}_${s}_v5_h40 REWARD_V5; done
for s in s1 s2; do score ${TAG}_${s}_v3_h40 REWARD_V3; done
$PY - > $D/report_press2.txt <<'PY'
import json, glob, random
D = "/home/jren313/research/starvla_rl/runs_g1_piston/filter_ab"
def rows(p): return json.load(open(p))["modes"]["deterministic"]["per_condition"]
def rate(rs, k): return sum(1 for x in rs if x["stages"].get(k)) / len(rs)
print("| policy | reward | sweep | grasp | lift | plate | press | dispense | full success | return |")
print("|---|---|---|---|---|---|---|---|---|---|")
A = {}; B = {}
for s in ("s1", "s2", "s3", "s4"):
    for name, store in (("bc_press", A), ("bc_press2", B)):
        p = f"{D}/{name}_{s}_v5_h40_n25.json"
        if glob.glob(p):
            rs = rows(p); store[s] = rs; m = json.load(open(p))["modes"]["deterministic"]
            print(f"| {name} | v5 | {s} | {rate(rs,'grasp'):.2f} | {rate(rs,'lift'):.2f} | {rate(rs,'plate'):.2f} | {rate(rs,'press'):.2f} | {rate(rs,'dispense'):.2f} | {rate(rs,'success'):.2f} | {m['mean_return']:.2f} |")
for s in ("s1", "s2"):
    p = f"{D}/bc_press2_{s}_v3_h40_n25.json"
    if glob.glob(p):
        rs = rows(p); m = json.load(open(p))["modes"]["deterministic"]
        print(f"| bc_press2 | v3 | {s} | {rate(rs,'grasp'):.2f} | {rate(rs,'lift'):.2f} | {rate(rs,'plate'):.2f} | {rate(rs,'press'):.2f} | {rate(rs,'dispense'):.2f} | {rate(rs,'success'):.2f} | {m['mean_return']:.2f} |")
pa = sum((A[s] for s in sorted(set(A) & set(B))), []); pb = sum((B[s] for s in sorted(set(A) & set(B))), [])
n = len(pa)
if n:
    da = [int(bool(x["stages"].get("dispense"))) for x in pa]; db = [int(bool(x["stages"].get("dispense"))) for x in pb]
    diffs = [b - a for a, b in zip(da, db)]; random.seed(0); boots = sorted(sum(random.choice(diffs) for _ in range(n)) / n for _ in range(4000))
    lo, hi = boots[100], boots[3899]
    print(); print(f"**Pooled over {n} paired condition-evaluations (v5, 40 chunks):** dispense bc_press2 {sum(db)/n:.3f} vs bc_press {sum(da)/n:.3f} (difference {sum(diffs)/n:+.3f}, paired-bootstrap 95% CI {lo:+.3f} to {hi:+.3f}); full success {rate(pb,'success'):.3f} vs {rate(pa,'success'):.3f}; lift {rate(pb,'lift'):.3f} vs {rate(pa,'lift'):.3f}.")
    print(); print("Verdict: **" + ("established" if lo > 0 or hi < 0 else "not established") + "** on dispense.")
try:
    r = json.load(open("/home/jren313/research/starvla_rl/runs_g1_piston/bc_press2/train_run.json")); m = json.load(open("/home/jren313/research/starvla_rl/demo_buffer_v5_press2/_manifest.json"))
    print(); print(f"Training: {m['n_synthetic']} certified palm-press episodes + {m['n_original']} truncated originals, {r['n_transitions']} transitions, final MSE {r['final_mse']}.")
except Exception: pass
PY
{ echo; echo "## Press data scaled: bc_press2 (auto)"; echo; cat $D/report_press2.txt; } >> verified_results/FINAL_REPORT.md
for f in $D/filter_ab/bc_press2_s?_v?_h40_n25.json; do cp $f verified_results/manifests/; done; cp $D/$TAG/train_run.json verified_results/manifests/bc_press2_train.json 2>/dev/null
git add verified_results/FINAL_REPORT.md verified_results/manifests && git commit -q -s -m "docs(g1_piston): bc_press2 (scaled press data) certified at 40 chunks (auto)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || true
echo "$(date +%H:%M:%S) PRESS2 PIPELINE DONE" >> $LOG
