#!/usr/bin/env bash
# After chain_rl12.sh: table of every run-12 iteration at the 40-chunk horizon against the
# pressing policy and the working BC, appended to the final report; manifests copied; commit.
source /home/jren313/research/starvla_rl/runs_g1_piston/scripts/common.sh
cd /home/jren313/research/starvla_rl/RLinf
while ! grep -q "SCORE RL12 DONE" $D/chain_rl12.log 2>/dev/null; do sleep 120; done
$PY - > $D/report_rl12.txt <<'PY'
import json, glob, re
D = "/home/jren313/research/starvla_rl/runs_g1_piston/filter_ab"
def row(fp):
    m = json.load(open(fp))["modes"]["deterministic"]; rs = m["per_condition"]; n = len(rs)
    r = lambda k: sum(1 for x in rs if x["stages"].get(k)) / n
    return r("grasp"), r("lift"), r("plate"), r("press"), r("dispense"), r("success"), m["mean_return"], 1000 * m["mean_max_press_m"]
print("| policy | reward | sweep | grasp | lift | plate | press | dispense | full success | return | mean max press (mm) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
rows = [("BC (working)", "v5", "s1", f"{D}/bc_working_s1_v5_h40_n25.json")]
rows += [("bc_press", "v5", s, f"{D}/bc_press_{s}_v5_h40_n25.json") for s in ("s1", "s2")]
for fp in sorted(glob.glob(f"{D}/rl12_iter*_h40_n25.json")):
    mm = re.search(r"(rl12_iter\d+)_(s\d)_(v\d)_h40", fp); rows.append((mm.group(1), mm.group(3), mm.group(2), fp))
for name, rv, s, fp in rows:
    try: g, l, p, pr, d, su, ret, mp = row(fp)
    except Exception: continue
    print(f"| {name} | {rv} | {s} | {g:.2f} | {l:.2f} | {p:.2f} | {pr:.2f} | {d:.2f} | {su:.2f} | {ret:.2f} | {mp:.1f} |")
try:
    r = json.load(open("/home/jren313/research/starvla_rl/runs_g1_piston/rl12/run.json"))
    print(); print("Training (exploration rollouts, 48 per iteration, 40-chunk episodes): " + "; ".join(
        f"iteration {it['iteration']}: dispense {it['collect'].get('dispense_rate', 0):.3f}, lift {it['collect'].get('lift_rate', 0):.2f}, return {it['collect'].get('mean_return', 0):.2f}, KL-to-base {it['ppo'].get('kl_base', 0):.4f}" for it in r.get("iterations", [])) + ".")
except Exception: pass
PY
{ echo; echo "## Run 12: critic-free GRPO under v5 from the pressing policy, 40-chunk horizon (auto)"; echo; echo "Every iteration scored in fresh processes on the frozen 25-condition suite at EP_CHUNKS=40: twice under v5 (press scorer), once under v3 (transport scorer of record)."; echo; cat $D/report_rl12.txt; } >> verified_results/FINAL_REPORT.md
cp $D/rl12/run.json verified_results/manifests/grpo_rl12_train.json 2>/dev/null; for f in $D/filter_ab/rl12_iter*_h40_n25.json; do cp $f verified_results/manifests/grpo_$(basename $f); done
git add verified_results/FINAL_REPORT.md verified_results/manifests && git commit -q -s -m "docs(g1_piston): run 12 certified at the 40-chunk horizon (auto)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || true
echo "$(date +%H:%M:%S) REPORT RL12 DONE" >> $D/chain_rl12.log
