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

## Press pipeline (2026-09-14)

* `queue_t23.sh` — starts the conditions-2,3 generator (`make_press_demos.py`, palm-press
  variant) when the canonical generator has finished, so two simulators share the GPU.
* `run_press_pipeline.sh` — waits for `demo_buffer_v5_palm_{canon,t01,t23}`, assembles
  `demo_buffer_v5_press` (`build_press_bc_buffer.py`), then runs `chain_bc_press.sh`.
* `chain_bc_press.sh` — BC retrain on the pressing buffer, then fresh-process sweeps: v3 x2
  (transport scorer of record), v5 x2 (press scorer), plus the BC-baseline v5 sweeps.
* `after_bc_press.sh` — appends `report_press.py` to the final report, renders the pressing
  policy, and, if it dispenses at all, runs GRPO under v5 from it (run 12) and certifies every
  iteration.
