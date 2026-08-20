#!/usr/bin/env bash
# Does removing the chunk-boundary jitter actually improve the task, or only look better?
#
# A/B on the SAME checkpoints and the SAME frozen 50-condition suite, deterministic,
# with BLEND_STEPS as the only variable. BLEND_STEPS=0 reproduces the study's execution,
# so the paired baseline is re-run here rather than compared against stored numbers -- a
# fair A/B needs both halves measured the same way in the same session.
#
# This is an EXECUTION-mode arm. It never touches the policy, reward, metrics or suite,
# and its results are reported separately and never merged into the frozen comparison.
set -uo pipefail
S="${SCRATCH:-/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad}"
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/env_isaaclab/bin/python}"
RUNS="$S/runs"; OUT="$S/runs/blend_ab"; mkdir -p "$OUT"
N="${N_EVAL_AB:-25}"
export MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab
echo "$$" > "$S/blend_ab.pid"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

is_done(){ "$PY" - "$1" <<'PYEOF' >/dev/null 2>&1
import json,sys
d=json.load(open(sys.argv[1]))
rows=d.get("modes",{}).get("deterministic",{}).get("per_condition",[])
import os
need=int(os.environ.get("N_EVAL_AB","25"))
sys.exit(0 if d.get("_status")=="OK" and len(rows)>=need else 1)
PYEOF
}

wait_gpu(){
  for _ in $(seq 1 4320); do   # up to 24 h: training arms are ~11 h each
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0
    sleep 20
  done
  return 1
}

# arm <name> <ckpt> <blend>
arm(){
  local name="$1" ckpt="$2" blend="$3"
  local out="$OUT/${name}_blend${blend}_n${N}.json"
  if is_done "$out"; then log "SKIP  $name blend=$blend"; return 0; fi
  # "sft" is a sentinel the evaluator accepts, meaning the raw SFT policy.
  [ "$ckpt" = "sft" ] || [ -f "$ckpt" ] || { log "MISS  $name -- no checkpoint"; return 0; }
  wait_gpu || { log "gpu never freed"; return 0; }
  log "RUN   $name blend=$blend (n=$N, deterministic)"
  OUTF="$out" CKPT="$ckpt" RUN_DIR="$OUT/${name}_b${blend}" \
  N_EVAL="$N" MODES=det BLEND_STEPS="$blend" \
  RESET_SUITE_SEED=20260817 EP_CHUNKS=23 \
  "$PY" "$T/eval_checkpoint.py" > "$OUT/${name}_b${blend}.log" 2>&1
  is_done "$out" && log "DONE  $name blend=$blend" || log "FAIL  $name blend=$blend"
  return 0
}

# Ordered so the decisive pair completes first and every later arm is a bonus.
# SFT leads: it is the ONLY policy that ever solved the task, so if jitter removal
# helps anywhere it should help there most, and its 0.16 stochastic success gives a
# non-degenerate baseline to move. Then the two RL checkpoints that reached lifting.
#
# N_EVAL is 25 (the frozen prefix) rather than 50: it halves each arm to ~20 min and the
# prefix is the matched-curve suite used throughout. A promising result can be upgraded
# to n=50 afterwards.
N="${N_EVAL_AB:-25}"
# ORDERING (corrected): the RL checkpoints go first.
#
# SFT led originally because it is the only policy that ever solved the task -- but that
# was its STOCHASTIC behaviour, and this A/B is deterministic. Measured, SFT-deterministic
# grasps 1/25: a near floor, where boundary smoothing has almost nothing to preserve,
# because it can only help a rollout that already makes contact.
#
# The RL checkpoints grasp ~1.00 and then lose the object before transport. That is
# exactly the population in which a 0.6 rad finger snap during contact would be the
# binding constraint, so they are the informative arms.
arm rlpd_s1_415140 "$RUNS/rlpd/rlpd_ckpt_step415140.pt"     0
arm rlpd_s1_415140 "$RUNS/rlpd/rlpd_ckpt_step415140.pt"     6
arm sac_s2_690780  "$RUNS/sac_s2/sac_ckpt_step690780.pt"    0
arm sac_s2_690780  "$RUNS/sac_s2/sac_ckpt_step690780.pt"    6
# SFT last, retained as a floor control: if smoothing "improves" a policy that barely
# makes contact, that would signal a confound rather than a real effect.
arm sft            "sft"                                   0
arm sft            "sft"                                   6

log "BLEND_AB_DONE"
rm -f "$S/blend_ab.pid"
