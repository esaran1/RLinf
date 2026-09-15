# Run 12 iteration 2 (GRPO under v5 from the pressing policy), 40-chunk horizon, deterministic

12 clips on the frozen evaluation conditions that scored best in the certified sweeps. A render
is a fresh process, so per-condition outcomes differ from the sweeps (documented evaluation
variance). In this render:

* `cond21`, `cond23` — full **dispense** (v5): carried to the plate, tip rested on it, plunger
  held past 20 mm for 0.3 s; return 34.7 each.
* `cond13` — lifted **press** away from the plate; return 16.8.
* `cond22`, `cond20`, `cond19`, `cond14`, `cond5`, `cond16` — transport to the plate, press
  attempted but not certified within the horizon.
* `cond0`, `cond1`, `cond24` — grasp without lift (this checkpoint's transport loss: lift 0.63
  over 100 evaluations against the pressing policy's 0.87).

Certified over four paired 40-chunk v5 sweeps (100 condition-evaluations): dispense 0.17, full
task success 0.11, lift 0.63; the pressing policy on the same conditions: 0.08, 0.04, 0.87.
Checkpoint: `checkpoints/g1_piston_grpo_working/run12_iter2.pt`. Re-render:

    OUTDIR=<dir> CKPT=<ckpt> CONDS=21,23,... MODE=deterministic REWARD_V5=1 \
      RESET_SUITE_SEED=20260817 EP_CHUNKS=40 FPS=20 python tools/g1_piston/render_rollouts.py
