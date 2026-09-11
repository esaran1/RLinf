# Durable run pipeline for the G1 piston RL runs

Copies of the scripts that live in `/home/jren313/research/starvla_rl/runs_g1_piston/scripts/`
(the operational location; absolute paths are deliberate). Versioned here because the
session scratchpad has been wiped three times.

* `common.sh` — shared paths and the GPU busy check. The check is anchored to the start
  of the command line and uses bracketed names, so no shell whose command line merely
  contains this text can match it. A plain `pgrep -f "train_sac.py"` matched the shell
  wrapper that launched the script and idled the GPU for five days.
* `run_rl7.sh`, `run_rl8.sh` — the pre-registered commands (contracts
  `g1_piston_v3_retrain_run7/8_preregistration.json`).
* `rl*_post_warmup_watch.sh` — measures the first checkpoint saved AFTER the actor starts
  updating, through the full deployment path (`measure_deployed_action_error.py`).
* `finish_rl8.sh` — scores run 8 with the scorer of record, renders videos if grasp >= 0.87,
  then scores run 7 and runs the reversed-order BC probe. Sequential on one GPU.
* `chain_rl8.sh` — superseded (run 8 was launched manually on 2026-09-11).

Start scripts with `setsid nohup ./scripts/<name>.sh >/dev/null 2>&1 </dev/null &` from a
shell whose command line does not contain tool names.
