#!/usr/bin/env bash
# Render the verified visual-evidence set from saved checkpoints.
#
# Strictly post-hoc and GPU-polite: every job waits for an idle GPU, so this cannot
# contend with the training arms in run_missing_seeds.sh. Training always wins -- this
# queue simply blocks until the GPU is free, including between jobs.
#
# Hardened the same way as the other runners:
#  - helpers end in an explicit `return 0`; the loop never uses `|| exit`, so one failed
#    render cannot abort the queue;
#  - the script signals no PIDs; it owns its child directly;
#  - a job counts as complete only if _status == OK AND every requested rollout produced
#    a video and a sidecar; anything else is quarantined rather than left looking valid;
#  - idempotent: complete jobs are skipped, so a re-run resumes where it stopped.
set -uo pipefail

S="${SCRATCH:-/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad}"
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/env_isaaclab/bin/python}"
RUNS="$S/runs"
V="${VERIFIED:-/home/jren313/research/starvla_rl/verified_results}"
export MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab

echo "$$" > "$S/render_queue.pid"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# A job is complete when its status file says OK and each condition has both artifacts.
is_complete(){
  local outdir="$1" tag="$2" mode="$3" conds="$4" repeats="${5:-1}"
  "$PY" - "$outdir" "$tag" "$mode" "$conds" "$repeats" <<'PYEOF' >/dev/null 2>&1
import json,os,sys,glob
outdir,tag,mode,conds=sys.argv[1:5]
repeats=int(sys.argv[5]) if len(sys.argv)>5 else 1
st=os.path.join(outdir,f"_render_{tag}_{mode}.json")
if not os.path.exists(st): sys.exit(1)
d=json.load(open(st))
if d.get("_status")!="OK": sys.exit(1)
want={int(c) for c in conds.split(",") if c.strip()}
got={}
for r in d.get("rollouts",[]):
    v=r.get("video","")
    j=v[:-4]+".json"
    if os.path.exists(v) and os.path.exists(j) and os.path.getsize(v)>10000:
        got[int(r["cond"])]=got.get(int(r["cond"]),0)+1
sys.exit(0 if all(got.get(c,0)>=repeats for c in want) else 1)
PYEOF
}

wait_gpu(){
  for _ in $(seq 1 2160); do   # up to ~12 h: training arms are ~11 h each
    local used total
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0)
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] \
       && [ $((total-used)) -ge 12000 ]; then return 0; fi
    sleep 20
  done
  log "WARNING: gpu still busy after 12h; skipping this job"
  return 1
}

# render <subdir> <tag> <seed> <ckpt> <mode> <conds> [repeats]
# `repeats` renders N independent draws per condition -- only meaningful for stochastic
# mode, where a single draw proves nothing about a rate.
render(){
  local sub="$1" tag="$2" seed="$3" ckpt="$4" mode="$5" conds="$6"
  local repeats="${7:-1}"
  local outdir="$V/videos/$sub"
  mkdir -p "$outdir"

  if is_complete "$outdir" "$tag" "$mode" "$conds" "$repeats"; then
    log "SKIP  $tag/$mode conds=$conds (already complete)"; return 0
  fi
  if [ "$ckpt" != "sft" ] && [ ! -f "$ckpt" ]; then
    log "MISS  $tag/$mode -- checkpoint absent: $ckpt"; return 0
  fi

  wait_gpu || return 0
  log "RUN   $tag/$mode conds=$conds -> $sub"
  OUTDIR="$outdir" CKPT="$ckpt" CONDS="$conds" MODE="$mode" \
  TAG="$tag" SEED_LABEL="$seed" FPS=20 REPEATS="$repeats" \
  RESET_SUITE_SEED=20260817 EP_CHUNKS=23 \
  "$PY" "$T/render_rollouts.py" > "$outdir/${tag}_${mode}.log" 2>&1
  local rc=$?

  if is_complete "$outdir" "$tag" "$mode" "$conds" "$repeats"; then
    log "DONE  $tag/$mode"
  else
    log "FAIL  $tag/$mode (exit $rc) -- continuing"
    local st="$outdir/_render_${tag}_${mode}.json"
    [ -f "$st" ] && mv -f "$st" "${st}.failed.$(date +%s)"
  fi
  return 0
}

RL="$RUNS/rlpd"; S2="$RUNS/sac_s2"; S1="$RUNS/sac"

# ---- selections come from manifests/render_plan.json, by explicit rule ----
# Matched-budget disagreement set: every condition whose frozen label differs between
# RLPD@415140 and SAC_s2@414000. Rendering the SAME conditions for both arms gives the
# matched comparison; conditions 46 and 32 add the typical and worst-return cases.
DISAGREE="3,16,19,20,21,31,39,41"
CONTEXT="46,32"                       # median-return and worst-return
SHOWCASE="21,19,3"                    # best carry, throw exploit, carry-vs-no-reach

# 1. SFT baseline on the same conditions used for the comparison.
render sft  sft  0 sft "deterministic" "$DISAGREE,$CONTEXT"

# 1b. THE MOST IMPORTANT ARTIFACTS IN THE SET.
# The untrained SFT policy produces the only full task successes anywhere in this
# experiment, and only under its own sampling noise (conds 11,14,16,18 -> reach, grasp,
# lift, plate, success; disp and lift inside the demonstration ranges). Under the
# deterministic mean action the same weights are inert across 125 rollouts. See
# docs/contracts/g1_piston_sft_stochastic_success.json. Both modes are rendered on the
# SAME conditions so the difference is attributable to sampling alone.
#
# IMPORTANT: a stochastic render is an INDEPENDENT DRAW, not a replay -- the trainer's
# RNG had advanced through a deterministic sweep before its stochastic pass.
#
# Two independent stochastic sweeps of the (essentially) SFT policy succeeded on
# DISJOINT condition sets -- step 0 on {11,14,16,18}, step 1380 on {5,9,20} -- with no
# overlap, while the RATE held (0.16 then 0.12). Success is therefore driven by the
# sampled noise, not by the initial condition, so rendering {11,14,16,18} once would
# most likely show four failures and prove nothing.
#
# Instead: 8 independent draws on each of 4 conditions drawn from BOTH observed success
# sets. Across 32 stochastic rollouts a rate near 0.12-0.16 should yield ~4-5 successes.
# The matched deterministic renders of the same conditions are the control -- identical
# weights, identical resets, sampling the only difference.
SFT_STOCH_CONDS="11,16,5,20"
render sft sft 0 sft "stochastic"    "$SFT_STOCH_CONDS" 8
render sft sft 0 sft "deterministic" "$SFT_STOCH_CONDS,14,18,9"

# 2. Matched pair at the ~414k budget, identical conditions.
render matched rlpd_s1 1 "$RL/rlpd_ckpt_step415140.pt"  "deterministic" "$DISAGREE,$CONTEXT"
render matched sac_s2  2 "$S2/sac_ckpt_step414000.pt"   "deterministic" "$DISAGREE,$CONTEXT"

# 3. Progression on the showcase conditions, every retained RLPD checkpoint.
render progression rlpd_s1 1 "$RL/rlpd_ckpt_step1380.pt"   "deterministic" "$SHOWCASE"
render progression rlpd_s1 1 "$RL/rlpd_ckpt_step139320.pt" "deterministic" "$SHOWCASE"
render progression rlpd_s1 1 "$RL/rlpd_ckpt_step277320.pt" "deterministic" "$SHOWCASE"
render progression rlpd_s1 1 "$RL/rlpd_ckpt_step552000.pt" "deterministic" "$SHOWCASE"
render progression rlpd_s1 1 "$RL/rlpd_ckpt_step690180.pt" "deterministic" "$SHOWCASE"

# 4. SAC seed 2 progression on the same showcase conditions.
render sac sac_s2 2 "$S2/sac_ckpt_step276000.pt" "deterministic" "$SHOWCASE"
render sac sac_s2 2 "$S2/sac_ckpt_step552240.pt" "deterministic" "$SHOWCASE"
render sac sac_s2 2 "$S2/sac_ckpt_step690780.pt" "deterministic" "$SHOWCASE"

# 5. SAC seed 1: ONLY the final checkpoint survives (see
#    docs/contracts/g1_piston_sac_s1_checkpoint_loss.json). Its 414k checkpoint, where
#    that seed's carry rate peaked, was overwritten and cannot be rendered.
render sac sac_s1 1 "$S1/sac_ckpt_latest.pt" "deterministic" "$SHOWCASE,$CONTEXT"

# 6. Deterministic vs stochastic on the final checkpoints, same conditions.
render deterministic_vs_stochastic rlpd_s1 1 "$RL/rlpd_ckpt_step690180.pt" "stochastic" "$SHOWCASE"
render deterministic_vs_stochastic sac_s2  2 "$S2/sac_ckpt_step690780.pt"  "stochastic" "$SHOWCASE"
render deterministic_vs_stochastic sac_s1  1 "$S1/sac_ckpt_latest.pt"      "stochastic" "$SHOWCASE"
# The deterministic halves of that comparison, rendered at the same conditions.
render deterministic_vs_stochastic rlpd_s1 1 "$RL/rlpd_ckpt_step690180.pt" "deterministic" "$SHOWCASE"
render deterministic_vs_stochastic sac_s2  2 "$S2/sac_ckpt_step690780.pt"  "deterministic" "$SHOWCASE"

# 7. The strongest single carry, rendered at the checkpoint that produced it.
render rlpd rlpd_s1 1 "$RL/rlpd_ckpt_step415140.pt" "deterministic" "$SHOWCASE"

log "RENDER_QUEUE_DONE -- verifying"
"$PY" "$T/verify_rollouts.py" "$V/videos" 2>&1 | tail -60
rm -f "$S/render_queue.pid"
