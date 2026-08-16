# Off-policy RL fine-tuning design — StarVLA QwenOFT + IsaacLab G1 piston

Algorithm: **SAC**, with **RLPD (demonstration-seeded SAC)** as the complete configuration.
No GRPO.

This records the decisions that must be settled *before* any training code runs, and the
evidence for each. Everything here was read out of the current repository, not assumed.

## What already exists in RLinf (reused, not rebuilt)

| Component | Location | Status |
|---|---|---|
| SAC policy worker (critic/actor/alpha, target nets, ckpt) | `rlinf/workers/actor/fsdp_sac_policy_worker.py` | reuse |
| `TrajectoryReplayBuffer` (cache, persistence, windowed sampling) | `rlinf/data/storage/replay/buffer.py` | reuse |
| Demo buffer + mixed sampling (RLPD) | same worker, `algorithm.demo_buffer` | reuse |
| `SquashedNormal` / `GaussianPolicy` (tanh Jacobian correct) | `rlinf/models/embodiment/modules/gaussian_policy.py` | reuse |
| `MultiQHead` / `CrossQHead` | `rlinf/models/embodiment/modules/q_head.py` | reuse |
| `EntropyTemperature` (alpha tuning) | `rlinf/models/embodiment/modules/entropy_tunning.py` | reuse |
| Transition collection from rollouts | `rlinf/workers/env/env_worker.py` (`collect_transitions`) | reuse |
| 30->53 mapper, q99 norm, 2x hold | `rlinf/envs/isaaclab/tasks/g1_piston*.py` | **frozen, do not touch** |

The gap: `StarVLAForRLActionPrediction` implements `default_forward` (PPO) but **not**
`sac_forward` / `sac_q_forward`. That is the integration work.

## PHASE 2 — the off-policy transition (decision interval)

**Recovered, not assumed.** In `env_worker.py`, `append_transitions(curr_obs, next_obs)`
is called once per `chunk_step_idx`, inside a loop of
`n_train_chunk_steps = max_episode_steps // num_action_chunks`. Each `chunk_step`
executes `num_action_chunks` policy actions before the next inference.

So for this task:

```
action horizon          = 30   (StarVLA predicts 30 actions)
executed per inference  = 30   (whole chunk executed, then re-plan)
replan interval         = 30 policy actions
policy inference freq   = 50 Hz / 30 = 1.667 Hz
env steps per decision  = 30 * 2 = 60   (2x zero-order hold, 100 Hz sim)
wall-clock per decision = 0.6 s
```

One replay transition therefore spans **60 IsaacLab control steps / 0.6 s**, not one step.

Confirmed by the closed-loop eval: 1000 policy actions ran as 34 replans
(1000/30 ≈ 33.3), and each replan produced one `predict_action` call.

### Reward aggregation and discounting

`fsdp_sac_policy_worker.forward_critic` already implements both conventions:

```python
if use_dsrl:                                    # chunked-decision convention
    discount            = gamma ** num_action_chunks
    rewards_for_bootstrap = rewards[:, 0:1]
else:                                           # one-step convention
    discount            = gamma
    rewards_for_bootstrap = rewards.sum(dim=-1, keepdim=True)
```

We use the **chunked convention**, because our transition spans 30 policy actions:

- reward = **sum** of the per-action rewards over the executed chunk (undiscounted
  within the chunk — an intentional, documented simplification; the chunk is 0.6 s and
  our reward is a potential-style progress signal, so intra-chunk discounting is
  second-order);
- bootstrap discount = `gamma ** num_action_chunks`, so the temporal span of the
  transition is respected rather than silently treated as one step;
- `next_obs` is the observation at the **end** of the executed chunk, which is exactly
  the observation the policy re-plans from.

`bootstrap_type: standard` masks the bootstrap on termination. Truncation (budget
exhausted) is *not* a termination and must still bootstrap — the buffer stores
`terminations` and `truncations` separately, so this is representable.

## PHASE 3 — actor space

The actor operates in **StarVLA's 30-D normalized action space**. The replay buffer
stores 30-D actor-space actions; the critic consumes 30-D actions. The validated 30->53
mapper stays purely an environment adapter, applied after sampling, on the way to
IsaacLab. The critic never sees 53-D actions.

Rationale: the actor only controls 30 dims; training a critic on 53 dims would make 27
of its input dims constant and uncontrollable, which is both wasteful and misleading.

## PHASE 4 — active action mask

From `dataset_statistics.json` (verified empirically last session), 10 of 30 dims are
degenerate (`q01 == q99`):

| dims | meaning | dataset value | RL treatment |
|---|---|---|---|
| 14-19 | left hand (grips the tube) | 1.7/1.7/1.7/1.7/0.35/0.25 | **frozen** — hold SFT constant, no exploration, no entropy |
| 26 | base height | 0.76 | **dropped** before sim; frozen |
| 27-29 | navigate vx/vy/vyaw | 0.0 | **dropped** before sim; frozen |
| 0-13 | left+right arm | varies | **active** |
| 20-25 | right hand | varies | **active** |

**Active RL dims = 20** (0-13 arms, 20-25 right hand).

Design: keep the model's full 30-D output tensor (checkpoint compatibility) and apply an
explicit `active_action_mask`. On inactive dims the policy emits the SFT/dataset constant
with **zero exploration noise** and **zero entropy contribution**. Target entropy is set
from the active count (20), not 30 — otherwise alpha tuning chases entropy in dims that
cannot move.

Note dims 26-29 are dropped by the mapper anyway, so exploration there could never reach
the simulator; freezing them additionally keeps them out of the log-prob and the critic's
input distribution.

## PHASE 5 — trainable scope

Frozen: Qwen3-VL vision encoder + language backbone (~2.44 B).
Trainable: OFT action head (~42.11 M), `actor_logstd`, critics + target critics, alpha.

## PHASE 12 — distribution convention

The existing OFT rollout path builds an **unsquashed** `Normal(mean, exp(actor_logstd))`
(PPO convention — unbounded, log-prob needs no Jacobian). SAC needs a bounded,
reparameterized sample with a correct tanh log-det correction, which is what
`SquashedNormal` in `gaussian_policy.py` provides.

Convention chosen, matching the rest of RLinf's SAC path: the distribution lives in the
**normalized** action space, log-prob/entropy are computed there, and de-normalization to
physical units happens once afterwards on the execution path only. No log-prob is
computed after a nonlinear transform.

Because the q99 clamp is `[-2.2, 2.2]`, the squash bounds are set to that range rather
than `[-1, 1]`, so in-distribution SFT actions are representable without saturating tanh.

## Definition of "off-policy" for this pipeline

The proof obligation is not "an optimizer ran". It is:

```
experience collected at rollout i
    -> stored in TrajectoryReplayBuffer
    -> sampled again at update j, where j corresponds to rollout k > i
    -> used to update critic + OFT actor
```

A dedicated test asserts a transition from an earlier rollout is sampled after later
rollouts have been appended, and that the same transition can participate in more than
one update.
