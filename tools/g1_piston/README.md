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

Sixteen checks, all of which must pass before a long run. The load-bearing ones:

* `split_forward_matches_predict_action` — the differentiable path used for the actor
  gradient is numerically **identical** (max abs diff 0.0) to the frozen
  `QwenOFT.predict_action`. This matters because `predict_action` is decorated
  `@torch.inference_mode()` and detaches to numpy, so it cannot carry a gradient; the
  trainer splits the forward and re-runs only the OFT head with grad.
* `vlm_params_unchanged` / `oft_params_changed` — the 2B backbone stays frozen.
* `frozen_dims_exact` — the 10 degenerate action dims receive exactly 0.0 exploration.
* `old_transitions_reused` — transitions collected before any gradient update are
  consumed by 15 later updates with finite losses. This is the off-policy milestone.

Recorded in `docs/contracts/g1_piston_sac_smoke.json`.

## Training (`train_sac.py`)

```bash
OUTF=<json> ALGO=sac|rlpd|sft_eval RUN_DIR=<dir> \
MAX_ENV_STEPS=24000 EVAL_EVERY=6000 EVAL_SEEDS=0,1,2,3,4 \
UTD=0.5 BATCH=8 SEED=0 DEMO_FRAC=0.5 \
MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab \
~/miniconda3/envs/env_isaaclab/bin/python train_sac.py
```

`sac` uses online replay only. `rlpd` mixes the frozen verified-executable demo buffer
(22 episodes / 489 transitions / 18 successful) at `DEMO_FRAC`. `sft_eval` runs the
evaluation suite once against the unmodified SFT checkpoint.

Held identical across algorithms by construction: SFT initialization, reset
distribution, reward, 20-D active action mask, evaluation seeds, and interaction budget.
Online and offline sample counts are logged separately so sample efficiency and compute
efficiency can be compared independently.

### Evaluation suite

Fixed at creation and reused throughout: deterministic policy (no exploration) on seeds
0-4, scored on full-success / reach / grasp / lift / plate rates, mean return, piston
displacement, and max lift. Optimizer statistics are logged for debugging but are never
the selection criterion.

## Environment note

`env_isaaclab` needs a few VLA dependencies that were installed with `--no-deps` to
protect the working stack (verified unchanged: torch 2.7.0+cu128, transformers 4.57.6,
numpy 1.26.0): `diffusers`, `qwen_vl_utils`, `accelerate`, `timm`, `peft`,
`sentencepiece`, `av`, `decord`. `unitree_sdk2py` and `cyclonedds` are picked up from
the `isaac` env's site-packages (same Python 3.11 ABI) by appending it to `sys.path`
before `import tasks`, since the upstream task package imports DDS at import time.
