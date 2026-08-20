# Verified visual results — G1 piston SAC/RLPD study

Rendered simulator rollouts and the audit trail that backs them. Every video here is a
**real closed-loop IsaacLab rollout**: the policy under test actually controlled the
simulator while the frames were recorded. Nothing is replayed from a logged action array,
animated from joint values, or reused from demonstration footage.

## Layout

```
manifests/    audit trail (tracked in git)
tables/       recomputed metrics (tracked in git)
videos/       rendered rollouts + per-rollout state traces (NOT tracked — see below)
figures/
contact_sheets/
```

`videos/` is git-ignored: ~150 MB of MP4 that `tools/g1_piston/run_render_queue.sh`
regenerates from the saved checkpoints. The manifests and tables that index and summarise
them are small and *are* tracked, so a reviewer gets the evidence index from git and the
footage from the working tree.

## Start here

The strongest result in the study. The **untrained SFT policy** completes the full task
under its own sampling noise, and never does so under the deterministic mean action:

```
videos/sft/sft_step0_cond0_stochastic.mp4    full success, return 35.9
videos/sft/sft_step0_cond4_stochastic.mp4    full success
videos/sft/sft_step0_cond6_stochastic.mp4    full success
videos/sft/sft_step0_cond9_stochastic.mp4    full success
videos/sft/sft_step0_cond20_deterministic.mp4   same weights, piston never touched
```

Watch one success and one deterministic rollout back to back: identical weights,
identical reset, sampling is the only difference.

Verified rates: stochastic success **0.16** (n=25, Wilson 95% [0.064, 0.347]) against
deterministic **0.00** (n=15). True horizontal transport on the successes is
0.299–0.307 m, inside the demonstration range of 0.200–0.322 m.

## Reading a video

Each `.mp4` has a same-named `.json` beside it carrying the checkpoint path, the frozen
condition hash, and the full per-chunk state trajectory, so any frame traces back to
simulator state. The overlay shows method, seed, mode, env steps, condition + hash,
cumulative return, stage, displacement and lift — all read from real simulator state.

Filenames are `{method}_step{env_steps}_cond{condition}_{mode}[_draw{n}].mp4`.

## Caveat on per-condition outcomes

The rollouts in `videos/matched/` do not reproduce the carries the stored n=50
evaluations recorded for the same checkpoints and conditions. This was first attributed
to rendering perturbing the physics; that hypothesis is **refuted**. A no-capture rerun
using the evaluator reproduces 0 of 5 stored lifts too, and produces lifts on entirely
different conditions.

The real finding: for a fixed checkpoint under deterministic execution, **grasp is
perfectly reproducible (25/25 in both runs) while the post-grasp outcome is not** — the
lifting condition sets had zero overlap (Jaccard 0.00) at a similar aggregate rate. See
`docs/contracts/g1_piston_post_grasp_nondeterminism.json`.

So no video is privileged: these are as faithful as the stored evaluation. But **no
per-condition claim is supportable** for lift or carry, and aggregate rates need an
interval that covers run-to-run variation, not just binomial error. Grasp-level claims
are solid. The `sft/` stochastic videos are unaffected — that finding replicated at the
rate level, which is the level it was ever claimed at.

## Regenerating

```bash
bash tools/g1_piston/run_render_queue.sh      # renders (waits for an idle GPU)
python tools/g1_piston/verify_rollouts.py     # recomputes every label from raw state
python tools/g1_piston/build_verified_report.py
python tools/g1_piston/build_contact_sheet.py
```

`verify_rollouts.py` is the gate: it re-derives displacement, lift, carry, throw and
success from each rollout's own trajectory using the frozen metric code and fails any
artifact whose stored label disagrees. Only rollouts it passes may be used as evidence.

Metrics version: `v1.1-frozen-2026-08-17-horizontal-fix`.
