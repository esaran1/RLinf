# G1 piston off-policy RL harness

Single-process SAC / RLPD fine-tuning of the StarVLA QwenOFT policy on the IsaacLab G1
piston task, plus the throughput benchmark and correctness smoke test that justify the
design.

These are research drivers, not library code: they are run directly with the
`env_isaaclab` interpreter and take their paths from the environment. They deliberately
live outside `rlinf/` because they orchestrate an external simulator and an external
checkpoint.

## Why single-process

RLinf's `fsdp_sac_policy_worker` assumes Ray-distributed env / actor / rollout nodes
(see `examples/embodiment/config/*_rlpd_*.yaml`, which allocate a GPU node group and a
separate robot node group). Two measured facts make that the wrong shape here:

* Isaac Sim must own the process — it boots a Kit kernel and its USD stage is not
  fork-safe — and the piston scene does **not** clone, so `num_envs > 1` fails at reset
  (`docs/contracts/g1_piston_throughput_benchmark.json`).
* The whole experiment fits on one 16 GB card: 9.46 GB peak with 6.9 GB headroom
  (simulator 4.8 GB, StarVLA 4.2 GB, SAC heads + gradients ~0.4 GB).

So these scripts reuse RLinf's SAC *math* components — `MultiQHead`,
`EntropyTemperature`, `SquashedNormal` — and the project's own action-space contract,
rather than its multi-node plumbing.

## Throughput (`bench_vec.py`)

Measured on one RTX 4080, front camera only, batched policy forward:

| num_envs | result |
| --- | --- |
| 1 | 2.34 s/chunk, 25.7 env steps/s, ~135 episodes/hour, 4.3 GB peak |
| 2 | fails at `env.reset` (upstream scene does not clone) |
| 4 | not run |

The chunk cost is **98% simulator**: batched warm inference is 0.044 s. Batching the
policy would therefore not have helped even if the scene had cloned.

## Correctness (`smoke_sac.py`)

Eighteen checks, all of which must pass before a long run. The load-bearing ones:

* `split_forward_matches_predict_action` — the differentiable path used for the actor
  gradient is numerically **identical** (max abs diff 0.0) to the frozen
  `QwenOFT.predict_action`. This matters because `predict_action` is decorated
  `@torch.inference_mode()` and detaches to numpy, so it cannot carry a gradient; the
  trainer splits the forward and re-runs only the OFT head with grad.
* `vlm_params_unchanged` / `oft_params_changed` — the 2B backbone stays frozen.
* `frozen_dims_exact` — the 10 degenerate action dims receive exactly 0.0 exploration.
* `old_transitions_reused` — transitions collected before any gradient update are
  consumed by 15 later updates with finite losses. This is the off-policy milestone.
* `entropy_target_is_reachable` / `alpha_moves_in_the_correcting_direction` — the two
  checks that would have caught the pilots' failure: an entropy target below the
  squashed distribution's log-density floor makes alpha slide monotonically to zero and
  leaves the actor unregularised.

Recorded in `docs/contracts/g1_piston_sac_smoke.json`.

## Training (`train_sac.py`)

Use `run_experiments.sh <scratch_dir>` for the matched SAC/RLPD comparison — it pins
every shared setting in one place so the arms cannot drift apart. To drive one arm
directly:

```bash
OUTF=<json> ALGO=sac|rlpd|sft_eval RUN_DIR=<dir> \
MAX_ENV_STEPS=690000 EVAL_EVERY=138000 \
N_EVAL_CONDITIONS=50 N_EVAL_PERIODIC=25 EP_CHUNKS=23 \
UTD=0.5 BATCH=8 SEED=0 DEMO_FRAC=0.5 \
ALPHA_INIT=0.05 ALPHA_LR=1e-3 ACTOR_LR=3e-6 \
MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab \
~/miniconda3/envs/env_isaaclab/bin/python train_sac.py
```

`sac` uses online replay only. `rlpd` mixes the frozen verified-executable demo buffer
(22 episodes, 467 usable transitions, 18 successful) at `DEMO_FRAC`. `sft_eval` runs the
evaluation suite once against the unmodified SFT checkpoint.

Held identical across algorithms by construction: SFT initialization, reset
distribution, reward, 20-D active action mask, held-out evaluation suite, and interaction budget.
Online and offline sample counts are logged separately so sample efficiency and compute
efficiency can be compared independently.

### The decision variable, and entropy scaling

One replay transition is one executed StarVLA chunk: 30 action steps × 30 dims, 20 of
them RL-controllable per step, executed under a 2× zero-order hold. So the SAC action is
the **whole chunk** — 600 stochastic scalars — and the critic scores
`Q(s, 900-D flattened chunk)`, matching the stored replay action. A critic fed a single
30-D action would be regressing a chunk-long reward onto 1/30th of its cause;
`tests/unit_tests/test_g1_piston_sac_decision_variable.py` asserts that it is not.

The entropy reduction uses one internally consistent convention:

```
logp_step[h] = sum over the 20 active dims      # per control action
logp_chunk   = mean over the 30 horizon steps   # per control action
target       = -20                              # per control action
```

The earlier pilots summed over all 600 scalars while keeping the per-step −20 target, so
the entropy term reached ~300 nats against Q ≈ 3–7 and dominated the actor objective
purely because the policy emits 30 steps at once. Averaging over the horizon keeps
regularisation on a per-control-action scale that does not grow with the prediction
horizon. Summing is equally valid *with* a target scaled to −600 — the tests prove the
two are the same objective up to `alpha_sum = alpha_mean / H` — but mixing them is the
bug. `ALPHA_INIT`, `ALPHA_LR` and `ACTOR_LR` are exposed because alpha must also start
large enough to bind against the Q scale. See
`docs/contracts/g1_piston_sac_pilot_v1_collapse.json`.

The actor logs mean Q, log-prob per control action, alpha, `|alpha·logp|`, the actor Q
term and the actor entropy term separately, so their relative magnitudes are visible.

### Replay storage

Entries are held in CPU float16. Action queries dominate the footprint (`H × HID` =
30 × 2048 per state, twice per transition): a 500-episode run would otherwise hold
~5.5 GB alongside a resident Isaac Sim. They are a deterministic function of the frozen
VLM, so the ~1.7e-4 relative error is three orders of magnitude below the exploration
std. Rewards and done flags stay exact — they drive the bootstrap target.

### Evaluation suite

The upstream task has **one** initial state: `reset(seed=s)` is bit-identical for every
seed, so a seed-indexed suite has an effective sample size of 1 no matter how many seeds
it names (`docs/contracts/g1_piston_eval_suite_defect.json`). All variation therefore
comes from `rlinf/envs/isaaclab/tasks/g1_piston_reset.py`.

The 67 demonstrations are effectively identical at reset too — max std 0.021 rad across
all 63 state dims, and only on the left-hand fingers already gripping the tube — so
there is no empirical range to inherit and the perturbation is introduced deliberately:

* **Robot joint configuration** is the variable that actually changes the task: right
  arm ±0.05 rad, waist ±0.02 rad. Measured 52 mm of fingertip spread against a 45 mm
  grasp radius.
* **Piston xy** is nearly pinned. Four kinematic socket walls leave **2 mm** of radial
  clearance (that is why upstream zeroed the inherited ±0.05 m randomisation, which
  jammed the barrel into a socket corner), so jitter is capped at ±1 mm.

400 train / 50 eval conditions, disjoint by construction, reproducible from
`RESET_SUITE_SEED`, and hashed so every SFT/SAC/RLPD checkpoint is scored on identical
conditions. Verified in-simulator: `docs/contracts/g1_piston_reset_suite_verification.json`.

Stage rates carry 95% Wilson intervals and report `n_eval_episodes`. The original single
fixed reset is retained as a **canonical diagnostic**, stored separately from the suite
so its binary outcome can never be reported as a success rate.

Optimizer statistics are logged for debugging but are never the selection criterion.

## Environment note

`env_isaaclab` needs a few VLA dependencies that were installed with `--no-deps` to
protect the working stack (verified unchanged: torch 2.7.0+cu128, transformers 4.57.6,
numpy 1.26.0): `diffusers`, `qwen_vl_utils`, `accelerate`, `timm`, `peft`,
`sentencepiece`, `av`, `decord`. `unitree_sdk2py` and `cyclonedds` are picked up from
the `isaac` env's site-packages (same Python 3.11 ABI) by appending it to `sys.path`
before `import tasks`, since the upstream task package imports DDS at import time.
