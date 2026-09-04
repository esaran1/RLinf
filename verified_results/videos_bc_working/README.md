# Working policy: behaviour cloning after two train/deploy fixes

First policy in this project that performs the transport task. Trained by behaviour
cloning on 16 demonstrations, VLM frozen, OFT action head only.

## What these videos show

12 rollouts, deterministic, frozen reset suite (seed 20260817), scored under the
**corrected v3 predicate** — the geometrically correct grasp test, not the xy-only
metric the earlier RL result was measured against.

| | rendered (n=12) | evaluation (n=25) |
|---|---|---|
| grasp | 12/12 = 1.00 | 1.00 |
| lift  | 10/12 = 0.83 | 0.80 |
| plate | 10/12 = 0.83 | 0.80 |
| press / dispense | 0.00 | 0.00 |

The rendered rate agrees with the evaluation, so these clips represent the measured
behaviour rather than a selected best case.

**Two of the twelve (cond 2, cond 18) grasp but do not lift.** They are included
deliberately. Post-grasp outcomes in this task are not reproducible per condition across
processes (`docs/contracts/g1_piston_post_grasp_nondeterminism.json`): grasp is stable,
which condition lifts is not. Publishing only the ten that lift would overstate
per-condition reliability.

## Best single clip

`bc_sq_step0_cond20_deterministic.mp4` — highest return (16.09), clean
reach → grasp → lift → plate.

## What these videos do NOT show

The plunger is never pressed. That is expected and was pre-registered, not a defect:
1 of 22 executable demonstrations shows a genuine grasped press and none shows a
dispense, so imitation cannot produce the functional act. Reaching it requires
reward-driven discovery.

## Provenance

- checkpoint: BC, 300 epochs, 358 transitions from 16 episodes, final MSE 0.001005
  measured on the **executed** (post-squash) action
- evaluation of record: `verified_results/manifests/bc_squashfixed_v3_n25.json`
- render summary: `_render_summary.json` in this directory
- the two defects that had to be fixed first:
  `docs/contracts/g1_piston_bc_feature_mismatch.json`,
  `docs/contracts/g1_piston_bc_squash_mismatch.json`
