#!/usr/bin/env bash
# Chain: wait for the render queue to finish, then resume the three missing training arms.
#
# Rendering the verified visual-evidence set takes ~2 h; the training arms take ~32 h.
# Ordering rendering first delays the 3-seed result by ~2 h and delivers the auditable
# videos tonight instead of in ~34 h.
#
# run_missing_seeds.sh is idempotent -- complete arms are skipped -- so resuming it here
# is safe regardless of how far it had progressed (it had not started an arm when paused).
set -uo pipefail

S="${SCRATCH:-/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad}"
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for the render queue to finish"
for _ in $(seq 1 1440); do   # up to 12 h
  p=$(cat "$S/render_queue.pid" 2>/dev/null || echo "")
  if [ -z "$p" ] || ! ps -p "$p" -o cmd= 2>/dev/null | grep -q run_render_queue; then
    log "render queue finished"; break
  fi
  sleep 30
done

# Never start training on top of a live GPU job.
for _ in $(seq 1 360); do
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && break
  sleep 20
done

log "resuming the missing training arms"
exec bash "$T/run_missing_seeds.sh"
