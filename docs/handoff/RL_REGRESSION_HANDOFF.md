# Handoff: RL keeps destroying a working policy on the G1 piston task

You are picking up a robot-learning debugging task. Read this whole file before running
anything. Everything below is measured, and where something is a hypothesis it says so.

## The goal

Produce videos of a policy performing a micropipette ("piston") task in IsaacLab, with a
StarVLA QwenOFT VLA policy on a Unitree G1 + Inspire hands. ICRA target. A working policy
and 12 videos already exist. The open problem is that **every reinforcement-learning run
destroys that policy**, and we want RL to improve it (specifically to discover the
plunger press, which the demonstrations cannot teach).

## Repository and environment

* repo: `/home/jren313/research/starvla_rl/RLinf`, branch `starvla-g1-isaaclab-rl`
* python: `$HOME/miniconda3/envs/env_isaaclab/bin/python` (NOT the default python)
* env vars every sim tool needs: `MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab`
* tests: `PYTHONPATH=/home/jren313/research/starvla_rl/starVLA <py> -m pytest tests/unit_tests/test_g1_piston*.py -q`
  Expect **282 passing, 1 failing**. The failure (`test_denormalizing_the_buffer_would_corrupt_it`)
  is pre-existing: it needs `act_ep0.npy`, destroyed by a scratchpad wipe. Not yours.
* GPU: one RTX 4080, 16 GB, shared. **Never kill another user's process.** Check
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and the owner with
  `ps -o user= -p <pid>` first. A trainer needs ~7.6 GB *including* non-PyTorch memory
  (`peak_vram_mib` in run records counts only torch allocations and will mislead you).
* Isaac Sim traps SIGTERM and SIGINT. A stuck trainer of your own may need SIGQUIT.

## Hard constraints

Never use `git reset --hard`, `git clean -fd`, force push, broad `rm -rf`, `sudo`, or
system-wide CUDA changes. Do not modify the user's IsaacLab repo. Do not delete caches or
environments to free space. Commit with `git commit -s`.

## The working policy (do not lose this)

Backed up at `/home/jren313/research/starvla_rl/checkpoints/g1_piston_bc_working/`
(the scratchpad has been wiped twice, destroying every RL checkpoint both times).

Behaviour cloning on 358 transitions from 16 executable demonstrations, VLM frozen, OFT
action head only. On the frozen 25-condition suite under the corrected v3 reward,
deterministic:

    reach 1.00   grasp 1.00   lift 0.80   plate 0.80   mean return 11.935
    press 0.00   dispense 0.00

Videos: `verified_results/videos_bc_working/` (12 clips, grasp 12/12, lift 10/12).

## The problem you are solving

Two RL runs warm-started from that policy. Both destroyed it.

| | BC | run 4 | run 5 |
|---|---|---|---|
| eval grasp | **1.00** | 0.00 | 0.00 |
| deployed action error | **0.403°** | 13.773° | 11.542° |
| raw head output magnitude | **0.4151** | 0.2364 | 0.2543 |
| critic Q (end) | n/a | 2.58 | 2.45 |

The signature is consistent: **RL flattens the head's pre-squash output toward zero**, and
because deployment executes `tanh(head_out) * 2.2`, a small weight change becomes a large
executed-action error. Head weights barely move (mean |Δ| ≈ 0.0005 per tensor) while the
executed action degrades 30-fold.

Meanwhile every training diagnostic looks healthy: critic loss converges, Q rises,
log-probability moves toward its target, demo replay is genuinely active at 0.5.

## What has been ruled out, by measurement

1. **In-trainer eval disagreeing with the scoring tool.** `eval_checkpoint.py` — the tool
   that scores BC at 1.00 — scores run 4 at 0.00, matching. The regression is real.
2. **The entropy target's value.** The old target −8.4 was genuinely miscalibrated (it
   needs std ≈ 0.05 under correlated noise; it is unreachable at any std under i.i.d.
   noise; the branches differ ~15 nats at the same std). It was fixed by measuring the
   target for the branch in use (`calibrated_target_entropy()`, −7.30 at std 0.25). Run 5
   used the fix: the residual closed 0.85 → 0.21 and alpha's rise fell 57% → 36%, **and
   the policy still collapsed identically**. So this was not the cause.
   Contract: `g1_piston_run5_refutes_entropy_diagnosis.json`.
3. **The smoothness penalty.** `SMOOTH_LAMBDA=100` contributes 0.011 against Q ≈ 2.45,
   under 0.5% of the actor objective.
4. **The entropy update's sign.** Fixed earlier and confirmed working: log-probability now
   moves toward the target, not away.

## The cause — found and reproduced offline (2026-09-05)

The entropy term, as implemented, flattens the policy by itself. Reproduced on the
working BC head with **no critic, no reward, no simulator** (`tools/g1_piston/` probe
recorded in `docs/contracts/g1_piston_entropy_mean_force.json`):

| arm (650 Adam steps, lr 3e-6, alpha 0.06) | deployed error | raw magnitude |
|---|---|---|
| executed-action entropy (trainer's formula) | 0.40° → **15.04°** | 0.415 → 0.196 |
| latent entropy (no tanh Jacobian)           | 0.40° → 0.40°     | unchanged |
| no entropy                                  | 0.40° → 0.40°     | unchanged |

Two defects in one formula:

1. **Std sign.** The correlated-noise log-prob omitted the change-of-variables term for
   `std` (`-n_basis * Σ log std_d`), so `d lp/d logstd` was *positive*: more noise reported
   a higher density. The "entropy" term shrank exploration instead of regulating it.
2. **Mean force.** Entropy measured on the executed action includes the tanh Jacobian,
   whose mean-gradient is `+2α·E[tanh(u)]` — an inward pull on every pre-squash
   component ([arXiv 2608.24488](https://arxiv.org/html/2608.24488)). A competent policy
   has large |means| (0.415), so it is eroded toward zero regardless of the target value,
   which is why run 5's recalibrated target changed nothing.

**Fix (commit `c50dd658`):** one shared log-prob implementation for trainer and
calibration (`CorrelatedChunkNoise.logprob_chunk`, `rl_space.iid_logprob`) with the
change-of-variables term; `ENTROPY_SPACE=latent` by default (closed-form target, slope
−4/std, never flat); `ACTOR_WARMUP_UPDATES=300` critic-only updates before the actor
moves (WSRL [arXiv 2412.07762](https://arxiv.org/abs/2412.07762); ResFiT
[arXiv 2509.19301](https://arxiv.org/abs/2509.19301)). `ENTROPY_LP_LEGACY=1` reproduces
the old formula.

**Run 6** is pre-registered in `g1_piston_v3_retrain_run6_preregistration.json` with a
falsification rule that measures the flattening signature directly at the first
checkpoint (H6: raw magnitude > 0.35 and ep046 deployed error < 3°). It was blocked on GPU
memory at handoff time (`scratchpad/autolaunch_rl6.sh` starts it when ≥ 9.5 GB is free).

**Run 7 (residual arm, commit `5c990200`)** is implemented and pre-registered
(`g1_piston_v3_retrain_run7_preregistration.json`) as a *comparison* from the same
initialisation, not a rescue: `RESIDUAL=1` freezes the BC head and trains a zero-initialised,
temporally smooth residual (ResFiT). H8 verified before training: the zero residual
reproduces BC's deployed error exactly (0.403000°). `scratchpad/chain_rl7.sh` starts it
after run 6 finishes; `run_rl7.sh` holds the exact command. Every scorer applies the
residual when the checkpoint carries one.

## Durable storage (2026-09-05, 23:50)

The scratchpad was wiped a **third** time, destroying run 6's checkpoints and run 7's first
attempt at ~43k steps. Every run artifact now lives in
`/home/jren313/research/starvla_rl/runs_g1_piston/` (`scripts/`, `rl7/`, `diag/`,
`filter_ab/`, `videos/`). Run 7 was relaunched there with its pre-registered command;
`scripts/finish_rl7.sh` scores it with the scorer of record and renders videos if it passes
H4. Never put anything that must survive in the scratchpad. Contract:
`g1_piston_scratchpad_wipe_3.json`.

## Run 6 result and the second mechanism (2026-09-05, evening)

Run 6 **regressed** (in-trainer grasp 0.80 → 0.48 → 0.04 over 150 actor updates) but with a
**different signature**: raw head magnitude 0.446 (not shrunk), alpha +6%, log-probability
at its target. The entropy fix is verified live — the flattening mechanism is gone — and
the policy still drifted 7.3° off the demonstrations. Stopped early per its pre-registered
rule. Contract: `g1_piston_run6_result.json`.

The remaining driver was then confirmed with run 6's own critic, no simulator
(`tools/g1_piston/probe_critic_ranking.py`): it rates the drifted chunk **above** the BC
chunk on 100% of demonstration frames, Q rises monotonically along the line between them,
and the environment ranks them the other way (return 11.9 vs ≈ −0.6). The preference is 2%
of Q for a 6.5° change that destroys the task — the critic is nearly **action-insensitive**,
and once entropy is fixed its gradient is the only term the actor follows. Contract:
`g1_piston_critic_exploitation.json`, which also registers a prediction for run 7 before
its data: bounded residual ⇒ grasp preserved, but no press, because a bound does not repair
the critic. Candidate critic fixes (untested): 10-critic LayerNorm ensemble (RLPD), n-step
returns (ResFiT n=3), contrastive negatives near the BC action, Cal-QL calibration.

## Read the in-trainer evaluations carefully (2026-09-05)

The trainer's periodic evaluation **under-scores a working policy**: run 6's first
checkpoint is bit-identical to BC (deployed error 0.403°, no actor update yet) and the
in-trainer eval gave grasp 0.76 / lift 0.32, where the scorer of record gives 1.00 / 0.80.
The two loops are identical in code; what differs is process history (the in-trainer
eval runs after noisy training episodes). So: judge in-trainer curves relative to their
own baseline (0.76), and treat only `eval_checkpoint.py` in a fresh process as the number
of record. Contract: `g1_piston_in_trainer_eval_underscores.json`. A reversed-order
scoring of BC (`COND_ORDER=reverse`) is queued after the chain to test whether simulator
state carries across `env.reset`.

## Other things worth suspecting

* The demonstration-conditioned critic (Q ≈ 2.45 tracking demo return 11.88 while the
  policy returned −0.66). Not required to explain the collapse, but a `DEMO_FRAC=0`
  ablation would show whether it *also* harms the policy.
* The actor maximises `Q(s, a_sampled)` where `a_sampled` comes from the squashed
  distribution. If the critic is inaccurate off the demonstration manifold, the gradient
  through the squash may systematically pull toward the linear centre.
* The action space is 900-D (30 dims × 30-step horizon) with only ~8 transitions per
  dimension. The critic sees a 180-D DCT projection, but the actor is still a diagonal
  Gaussian over 900 scalars. A structured or diffusion policy is recommended in the
  original review and remains untested.

## How to run things

Training (RLPD from the BC policy). **`ALGO=rlpd` is load-bearing** — without it the
trainer silently runs plain SAC with no demo replay; it now fails closed, but know why:

    OUTF=<json> RUN_DIR=<dir> ALGO=rlpd DEMO_FRAC=0.5 \
      WARM_CKPT=/home/jren313/research/starvla_rl/checkpoints/g1_piston_bc_working/bc_ckpt_latest.pt \
      DEMO_DIR=/home/jren313/research/starvla_rl/demo_buffer_v3 \
      REWARD_V3=1 ACTION_BASIS=1 CRITIC_STATE=1 CORRELATED_NOISE=1 \
      GAMMA_CHUNK=0.98 SMOOTH_LAMBDA=100 UTD=0.5 BATCH=8 \
      MAX_ENV_STEPS=90000 EVAL_EVERY=30000 N_EVAL_PERIODIC=25 \
      RUN_INIT_EVAL=0 FIRST_EVAL_AT_ZERO=0 EVAL_STOCHASTIC=0 \
      python tools/g1_piston/train_sac.py

Scoring a checkpoint (this is the number of record):

    OUTF=<json> CKPT=<ckpt> RUN_DIR=<dir> N_EVAL=25 MODES=det REWARD_V3=1 \
      RESET_SUITE_SEED=20260817 EP_CHUNKS=23 python tools/g1_piston/eval_checkpoint.py

Rendering videos: same plus `CONDS=17,16,18,2` and `tools/g1_piston/render_rollouts.py`.

**Timing:** training is ~1.5 s per gradient update. **Evaluation dominates**: ~150 s per
condition, so a 25-condition sweep is ~40 min and a 50-condition one ~100 min. A full
90k-step run is about 2.3 GPU-hours. **The failure is visible at the first 30k evaluation**,
so kill a run early rather than waiting for it to finish.

Diagnostics that localised the problem, reusable:

* `tools/g1_piston/measure_deployed_action_error.py` — measures deployed action error in degrees against
  a demonstration, through the **full deployment path including the squash**. This is the
  instrument that found both BC bugs. `OUTF=<json> CKPT=<ckpt> python measure_deployed_action_error.py`.
* `REPLAY_DEMO=46` in `eval_checkpoint.py` — a policy-free control arm that feeds a
  demonstration's own actions through the identical env, mapper, retargeter and reward.
  It gives reach 1.00, grasp 1.00, lift 0.67, which proves the execution path works. Use
  it whenever a zero needs attributing.

## Method notes that matter here

This project has produced several "negative results" that turned out to be tooling
defects. What actually worked:

* **Measure at the point of execution.** An open-loop probe that omitted the deployment
  squash reported 1.15° for a policy that was 12° off where it counted, and its conclusion
  ("the policy needs proprioception") was wrong.
* **Keep a policy-free control arm.** A zero cannot be attributed to the policy unless the
  execution path is independently known to be capable.
* **Distinguish configured from actual.** One run reported `demo_frac: 0.5` in its config
  while drawing zero demo samples for its entire length, and exited OK.
* **Pre-register.** `docs/contracts/g1_piston_v3_retrain_run5_preregistration.json` fixed
  in advance what would falsify its own diagnosis, which is why run 5 is reported as a
  refutation rather than a mystery. Follow that pattern.
* **Quote marginal, end-to-end rates.** Naive wall clock said 81.6 s per gradient update;
  98% of it was one evaluation.

## Definition of done

Either:

1. An RL run that **preserves** grasp ≥ 0.87 on the frozen 25-condition suite and ideally
   improves lift or reaches press > 0, with fresh videos rendered from conditions the
   evaluation actually scored; or
2. A **measured, verified explanation** of why RL cannot preserve this policy in this
   setup, with the BC result and its videos written up as the deliverable.

Both are acceptable outcomes. Do not tune reward thresholds or the entropy target to make
numbers look better; those are fixed by measurement and any change needs a new measurement
recorded alongside it.

## Where the record lives

`docs/contracts/` (53 JSON contracts, one per finding), `verified_results/REVIEW_RESPONSE.md`
(the single-page writeup), and the git log on this branch. Start with:

* `g1_piston_run5_refutes_entropy_diagnosis.json` — the current open problem
* `g1_piston_bc_squash_mismatch.json` — how the working policy was obtained
* `g1_piston_rl_learned_the_metric_not_the_task.json` — why the reward predicate is strict
* `g1_piston_post_grasp_nondeterminism.json` — why single rollouts cannot resolve a rate
