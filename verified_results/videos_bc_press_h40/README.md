# Pressing policy (`bc_press`), 40-chunk horizon, deterministic, reward v5 overlay

12 clips on the frozen evaluation conditions that scored best in the certified sweep
(`manifests/bc_press_s1_v5_h40_n25.json`). A render is a fresh process, so per-condition
outcomes differ from the sweep (documented evaluation variance, `g1_piston_eval_variance.json`).
In this render:

* `cond7` — full **dispense** (v5): pipette carried to the plate, tip rested on it, hand slides
  down the barrel, plunger held past 20 mm for 0.3 s; return 33.7.
* `cond6` — lifted **press** at the socket (plunger past 20 mm while held above the stand),
  not over the plate; return 30.6.
* the other ten — transport to the plate (reach, grasp, lift, plate) with the press attempt
  started but not certified within the horizon (plunger 10-19 mm).

In the certified sweeps the same checkpoint dispensed on conditions 4, 7, 17, 20 (sweep 1)
and 10, 11 (sweep 2), with full task success (dispense, place, release) on 17 (sweep 1) and
10, 11 (sweep 2). Checkpoint: `checkpoints/g1_piston_bc_press/bc_ckpt_latest.pt`. Re-render:

    OUTDIR=<dir> CKPT=<ckpt> CONDS=7,6,... MODE=deterministic REWARD_V5=1 \
      RESET_SUITE_SEED=20260817 EP_CHUNKS=40 FPS=20 python tools/g1_piston/render_rollouts.py
