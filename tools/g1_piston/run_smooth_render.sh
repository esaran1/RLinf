#!/usr/bin/env bash
# Re-render the presentation videos with chunk-boundary blending enabled.
#
# The commanded trajectory in the original videos contains a step discontinuity at every
# 30-step chunk boundary -- up to 1.0 rad in a single 50 Hz control step, worst on the
# Inspire hand joints, which is the visible shaking. See
# docs/contracts/g1_piston_chunk_boundary_jitter.json.
#
# BLEND_STEPS=6 ramps each chunk in from the previous command over 120 ms, cutting the
# worst jump to 0.14 rad while leaving the last 24 of 30 steps bit-identical to the
# policy's own trajectory.
#
# HONESTY REQUIREMENT: blending did NOT improve task outcomes (A/B: carry 0.20 -> 0.16,
# inside the run-to-run noise floor of 0.36). These videos are smoother, not better. They
# go in a SEPARATE directory so they can never be confused with the frozen-protocol
# renders, and every sidecar records blend_steps=6.
set -uo pipefail
S="${SCRATCH:-/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad}"
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/env_isaaclab/bin/python}"
RUNS="$S/runs"
V="${VERIFIED:-/home/jren313/research/starvla_rl/RLinf/verified_results}"
export MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab
echo "$$" > "$S/smooth_render.pid"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

wait_gpu(){
  for _ in $(seq 1 2160); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0
    sleep 20
  done
  return 1
}

# render <subdir> <tag> <seed> <ckpt> <mode> <conds> [repeats]
render(){
  local sub="$1" tag="$2" seed="$3" ckpt="$4" mode="$5" conds="$6" repeats="${7:-1}"
  local outdir="$V/videos_smooth/$sub"
  local cid; cid=$([ "$ckpt" = "sft" ] && echo sft || basename "$ckpt" .pt)
  local st="$outdir/_render_${tag}_${cid}_${mode}.json"
  mkdir -p "$outdir"
  if [ -f "$st" ] && "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$st')).get('_status')=='OK' else 1)"; then
    log "SKIP  $tag/$mode $conds"; return 0
  fi
  [ "$ckpt" = "sft" ] || [ -f "$ckpt" ] || { log "MISS  $tag -- no checkpoint"; return 0; }
  wait_gpu || return 0
  log "RUN   $tag/$mode conds=$conds blend=6 -> $sub"
  OUTDIR="$outdir" CKPT="$ckpt" CONDS="$conds" MODE="$mode" \
  TAG="$tag" SEED_LABEL="$seed" FPS=20 REPEATS="$repeats" BLEND_STEPS=6 \
  RESET_SUITE_SEED=20260817 EP_CHUNKS=23 \
  "$PY" "$T/render_rollouts.py" > "$outdir/${tag}_${mode}.log" 2>&1
  log "DONE  $tag/$mode (exit $?)"
  return 0
}

RL="$RUNS/rlpd"; S2="$RUNS/sac_s2"; S1="$RUNS/sac"

# The presentation set: the two videos an advisor should actually watch, plus the
# supporting comparison. Conditions 0/4/6/9 are where the SFT policy completed the task.
render sft     sft     0 sft                          "stochastic"    "0,4,6,9" 
render sft     sft     0 sft                          "deterministic" "0,4,6,9"
render matched rlpd_s1 1 "$RL/rlpd_ckpt_step415140.pt" "deterministic" "21,3,19"
render matched sac_s2  2 "$S2/sac_ckpt_step414000.pt"  "deterministic" "21,3,19"
render final   sac_s1  1 "$S1/sac_ckpt_latest.pt"      "deterministic" "21,3"
render final   rlpd_s1 1 "$RL/rlpd_ckpt_step690180.pt" "deterministic" "21,3"

log "SMOOTH_RENDER_DONE"
rm -f "$S/smooth_render.pid"
