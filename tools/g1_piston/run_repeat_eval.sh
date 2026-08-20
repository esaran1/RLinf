#!/usr/bin/env bash
# How noisy is a single evaluation of a single checkpoint?
#
# Two identical-protocol runs of rlpd_ckpt_step415140 produced lifting-condition sets
# with ZERO overlap (Jaccard 0.00) at lift rates 0.20 and 0.32. Grasp was 25/25 in both.
# So per-condition post-grasp outcomes are noise, and the aggregate rate itself moves by
# 0.12 between runs -- the same magnitude as every SAC-vs-RLPD difference in the study.
#
# This measures that noise floor directly: the SAME checkpoint, the SAME frozen
# conditions, the SAME deterministic protocol, repeated N times. The spread across
# repeats is the honest error bar for every rate reported in this study, and no
# algorithmic claim smaller than it can be supported.
#
# Nothing here changes the policy, reward, metrics or suite.
set -uo pipefail
S="${SCRATCH:-/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad}"
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/env_isaaclab/bin/python}"
RUNS="$S/runs"; OUT="$RUNS/repeat_eval"; mkdir -p "$OUT"
CKPT="${CKPT_OVERRIDE:-$RUNS/rlpd/rlpd_ckpt_step415140.pt}"
N="${N_EVAL_REPEAT:-25}"
REPEATS="${REPEATS:-6}"
export MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab
echo "$$" > "$S/repeat_eval.pid"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

is_done(){ N_NEED="$N" "$PY" - "$1" <<'PYEOF' >/dev/null 2>&1
import json,sys,os
d=json.load(open(sys.argv[1]))
rows=d.get("modes",{}).get("deterministic",{}).get("per_condition",[])
sys.exit(0 if d.get("_status")=="OK" and len(rows)>=int(os.environ["N_NEED"]) else 1)
PYEOF
}

wait_gpu(){
  for _ in $(seq 1 4320); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0
    sleep 20
  done
  return 1
}

for i in $(seq 1 "$REPEATS"); do
  out="$OUT/rep${i}_n${N}.json"
  if is_done "$out"; then log "SKIP  repeat $i"; continue; fi
  wait_gpu || { log "gpu never freed"; break; }
  log "RUN   repeat $i/$REPEATS (n=$N, deterministic, identical protocol)"
  # SEED varies only the torch/np/random seeds, NOT the frozen conditions, which are
  # fixed by RESET_SUITE_SEED. Deterministic execution ignores the policy sampler, so
  # any difference between repeats is genuine run-to-run nondeterminism.
  OUTF="$out" CKPT="$CKPT" RUN_DIR="$OUT/rep$i" \
  N_EVAL="$N" MODES=det BLEND_STEPS=0 SEED="$i" \
  RESET_SUITE_SEED=20260817 EP_CHUNKS=23 \
  "$PY" "$T/eval_checkpoint.py" > "$OUT/rep${i}.log" 2>&1
  is_done "$out" && log "DONE  repeat $i" || log "FAIL  repeat $i"
done

log "REPEAT_EVAL_DONE"
rm -f "$S/repeat_eval.pid"
