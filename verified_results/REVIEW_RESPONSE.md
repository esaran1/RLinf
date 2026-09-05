# Review response: what was wrong, what was measured, what changed

A review raised six points about this project's RL setup. Every one was correct. Acting
on them uncovered a defect large enough to reframe the study's headline claim, so this
page records the response in one place: the claim, the measurement that settled it, and
the code change.

Everything below is measured on the simulator and reproducible from the repository.
Per-finding evidence lives in `docs/contracts/`.

---

## The headline: RL learned the grasp *metric*, not the grasp

The study reported that RL fine-tuning took grasp from ~0.02 to ~1.00. That number came
from a reward that tested **only xy distance to the barrel axis** and never checked
whether the hand was at the object's height.

Measured per simulator step over a full episode, same frozen condition:

| | SFT | RLPD @415k | |
|---|---|---|---|
| finger radial distance to axis | 0.068 m | **0.054** | closer |
| thumb radial distance to axis | 0.072 m | **0.033** | much closer |
| hand span (closure) | 0.118 m | **0.039** | 3× tighter |
| **height offset from barrel** | **0.021 m** | **0.102** | **5× worse** |

RL drove the hand onto the barrel axis and closed it completely, while moving it about
**8 cm along the barrel's length, off the body**. Under an xy-only test that is a perfect
grasp.

Scored on the **same 25 frozen conditions**, same checkpoint, under a geometrically
correct predicate:

| metric | old predicate (8-run range) | corrected predicate |
|---|---|---|
| grasp | 0.96 – 1.00 | **0.20** |
| lift | 0.20 – 0.56 | **0.00** |
| mean return | 3.21 – 6.07 | **0.55** |

Roughly four in five "grasps" were the hand off the object, and with a valid grasp
required the policy never lifts the pipette at all. The improvement was real *as an
improvement in satisfying the metric*; it was not an improvement at the task.

**Restate the claim as:** *RL took xy-axis alignment with a closed hand from ~0 to ~1.00,
while moving the hand off the object.*

Contract: `g1_piston_rl_learned_the_metric_not_the_task.json`.

---

## The task was never fully scored

The scene object is a **micropipette**: an articulation whose prismatic `PistonJoint` is
the plunger (0–40 mm travel, spring-loaded, returns to rest). The original reward read
only rigid-body positions and never read `joint_pos`, so **depressing the plunger — the
act the task is named for — was never measured, rewarded, or required.**

That is also why every screened demonstration records `success: True` with `tube: False`:
the prompt asks the robot to inject the pipette into the tube, and only the transport
clause was ever scored.

The policies optimised exactly what was measured. Absence of pressing was the expected
outcome, not a training failure.

Contract: `g1_piston_plunger_dof.json`.

---

## The six review points

| # | point | verified | response |
|---|---|---|---|
| 1 | 900-D action space is far too large for the data | **Yes** — measured 8.2 transitions per action dimension | Fixed DCT basis: critic input 900 → 180 (41 per dim), 99.9994% energy retained on real demos |
| 2 | Action dimensions are not independent | **Yes** — chunks live on a ~16-dim manifold; 2 components explain 90% of variance | Critic side fixed; **actor remains factorised** — a structured/diffusion policy is recommended, untested here |
| 3 | Discount too small | **Yes, worse than stated** — 0.99³⁰ = 0.74/chunk gave terminal credit **0.001** at episode start | Discount now set at chunk level: 0.98 → **0.63** terminal credit |
| 4 | Critic needs proprioception and phase | **Yes** — RGB alone cannot disambiguate stage, and reward is stage-gated | Privileged critic state: joint pos/vel, object and plunger state, phase one-hot |
| 5 | Dataset does not succeed | **Yes** — only 22 of 67 episodes are executable; SFT scores 0.00 deterministic success | Recorded; expectations set accordingly |
| 6 | Reward is unreasonable | **Yes, on every sub-point** | Reward v3, below |

Contract: `g1_piston_rl_setup_review.json`.

---

## Reward defects, and the one that was reproduced

The sharpest sub-point: **success fired for a pipette thrown into the air.** The old test
was "over the pot, above a height, and z barely changing between two steps" — and at a
throw's apex, vertical velocity passes through zero.

Reproduced directly: a pipette placed 0.8 m above the pot, released and motionless,
scores **success = True under both v1 and v2**, and **False under v3**.

Reward v3 also fixes:

- **grasp** — radial distance to the barrel *axis* plus contact height *within the
  barrel's extent*, instead of an xy projection that credited hovering;
- **lift** — requires the object to be *held*, not merely high;
- **success** — requires rest over a 10-step window, inside a supported height band, with
  the object *released*;
- **smoothness** — a bounded penalty on object jerk while held.

Validated on demonstration replays: v3 ranks task progress monotonically (corr 0.933, no
stage-ordering inversions) with **zero false positives** against ground truth — it never
credits a lift to a non-executable episode.

Contract: `g1_piston_reward_v3_review_fixes.json`.

---

## A reward plateau that plausibly explains the stall

Under the old reward, failed episodes — including ones that crush the plunger while the
pipette stays socketed — scored **~2.97 with reach and grasp credited**. Under v3 they
score **~0.00**.

So the old reward paid ~3.0 to any rollout that brought the hand near the barrel axis and
then offered nothing until full transport at ~36: a wide flat plateau ending in a cliff.
That is a concrete mechanism for why the policies stalled at "grasp", and it matches the
independently measured bimodal return distribution (≈2.95 vs ≈36, almost nothing between).

---

## Corrections to our own claims

Findings that measurement overturned, kept on the record rather than quietly dropped:

- **"Camera shake is a real physical pathology"** — refuted. The ego camera moves
  0.014 mm per step, ~12 mm over a whole episode. The original pixel measurement sampled
  a region the robot's arm passes through. The apparent shake is arm and hand motion in a
  stable frame. (`g1_piston_camera_shake_refuted.json`)
- **"3 of 11 demonstrations press the plunger"** — the sample spanned all 67 dataset
  episodes, only 3 of which are executable.
- **"ep46 is the one genuine grasped press"** — replaying the same episode a second time
  gave 19.4 mm instead of 23.9 mm and the press stage did not fire. The press is a
  single-chunk transient straddling the threshold.
- **The press threshold itself** — plunger peaks across all 22 executable episodes form
  one continuous 8.4–23.9 mm band with no separable mode, and 8 episodes sat within 5 mm
  of the old 20 mm threshold. It is now set from the object (28 mm), above anything the
  demonstrations reach, so firing it requires behaviour the data does not contain.

**Consequence:** there is no reproducible demonstration of a deliberate press anywhere in
the dataset. The imitation prior for the functional act is effectively **zero**, so any
press must be discovered from reward.

---

## Defects caught before they corrupted a run

Each of these would have produced a plausible-looking result rather than an error:

1. **Units** — the demo buffer stores *physical* actions; a probe denormalised them,
   corrupting every rollout into a convincing "zero stages everywhere". The buffer
   *builder* had the same bug on the write side, which would have fed RLPD corrupted
   actions on half of every batch for a whole run.
2. **Buffer source** — the v3 rebuild would have produced 11 episodes, 8 of them
   non-executable failures, dropping the entire executable set's structure.
3. **Progress scale** — builder and trainer normalised `episode_progress` by different
   values, so the critic would have seen two conventions for one feature.
4. **Unsatisfiable grasp predicate** — v3's first grasp test scored 0 of 690 real steps
   because it assumed a pinch; the Inspire hand grips a cylinder with the thumb bracing
   from the *same* side.
5. **Reward exploit in v3 itself** — 18.0 reward farmable by grasping once, releasing,
   then pressing the plunger against the table.
6. **Critic width on warm start** — caught by a fast, correct crash rather than silent
   padding.

---

## Training runs: four attempts, four distinct failures

The review's fixes are implemented and tested, but **none has yet been shown to improve
the policy**, because every training run failed for a different reason. Recording them
plainly, because "the fixes did not help" and "the run never tested the fixes" are
different claims:

| run | updates | outcome | cause |
|---|---|---|---|
| 1 | 150 | grasp 0.20 → 0.00 | throughput: 72 s/update, critic never fit (loss 0.15 → 8.39) |
| 2 | 0 | crashed in 27 s | a package import in a module the tools load by file path |
| 2b | 125 | no checkpoint | raising UTD *slowed* collection 16×: demo VLM cache misses |
| 3 | 2,700 | grasp 0.20 → 0.00 | entropy-target sign: policy driven to near-determinism |

Run 3 is the informative one. Its critic was **healthy** — the loss median improved
monotonically (ratio 1.99 → 0.64, final 0.091) and Q plateaued near the value a grasping
policy is actually worth. That **refutes** run 1's explanation that an unfitted critic
was dragging the policy down.

The measured cause is an inconsistency between the alpha update and its documented
target. The update is `−α(logp + TARGET_ENTROPY)`, whose equilibrium sits at
**logp = +8.4**, while `default_target_entropy()` documents −8.4 as a target log-density
("exploration std ~0.20"). Run 3's log-probability climbed −6.13 → +7.79 toward that
equilibrium, α fell 73%, the entropy term reached **0.3%** of the actor objective, and
the policy stopped moving from episode 4 onward (`mean_disp_m` 0.0091).

The original 415k checkpoint was trained under the same formula, but started at
logp 22.95 — far *above* the equilibrium — and was descending toward it. So the same
update that paralysed run 3 was, for the original run, correctly reducing an
over-dispersed policy. Contract: `g1_piston_entropy_target_sign.json`.

**Throughput is no longer a constraint**: three avoidable costs (demonstration VLM cache
misses, step-0 initial evaluations, a step-0 periodic sweep) took a run from 72 s per
gradient update to 1.23 s. Contract: `g1_piston_utd_throughput_analysis.json`.

## Behaviour cloning: two train/deploy defects, then a working policy

With every RL checkpoint destroyed by a scratchpad wipe, the remaining route to a working
policy was to imitate the demonstrations directly. 16 of the 22 executable episodes reach
grasp, lift and plate when replayed, so the behaviour is present in the data.

The first two attempts scored **0.00 on every stage** of the frozen 25-condition suite.
Both were defects in our own tooling, not properties of the task, and both had the same
shape: a training tool and the deployment path disagreeing about what a tensor means,
with matching shapes so nothing errored.

| defect | what BC did | what deployment does | effect |
|---|---|---|---|
| **features** | sliced the last 30 hidden states | gathers hidden states *at the action-token positions* | prompt ends with `<action>.` after the tokens, so the slice is off by 8 positions; 49% relative difference |
| **squash** | regressed the head's **raw** output | executes `tanh(head_out) * 2.2` | head trained in a space the simulator never runs |

The squash defect is the one that mattered, measured end to end on demonstration ep046:

| checkpoint | error before squash | **error after deployment squash** |
|---|---|---|
| feature-fixed | 0.376° | **12.393°** |
| squash-fixed | 8.068° | **0.403°** |

The errors swap, which is the signature of a correct fix: accurate in the space that is
executed, inaccurate in a space nothing uses.

**A policy-free control arm localised the failure.** `REPLAY_DEMO` feeds a recorded
demonstration's own physical actions through the identical env, mapper, retargeter,
reward and reset conditions. Replaying ep046 gives reach 1.00, grasp 1.00, lift 0.67,
which proves the execution path is capable and pins the failure on what the policy
commands. Without that control, a zero cannot be attributed to anything.

### Result on the frozen 25-condition suite, v3 reward, deterministic

| stage | before (either defect) | **after both fixes** | 95% CI |
|---|---|---|---|
| reach | 0.00 | **1.00** | [0.87, 1.00] |
| grasp | 0.00 | **1.00** | [0.87, 1.00] |
| lift | 0.00 | **0.80** | [0.61, 0.91] |
| plate | 0.00 | **0.80** | [0.61, 0.91] |
| press | 0.00 | 0.00 | [0.00, 0.13] |
| dispense | 0.00 | 0.00 | [0.00, 0.13] |
| mean return | −0.611 | **11.935** | |
| mean displacement | 0.0096 m | **0.2742 m** | |

This is the first policy in the project to perform the transport task, and it is scored
under the **corrected** v3 predicate — the one that costs the old RL checkpoint four in
five of its "grasps". Every rate was re-derived from the 25 per-condition records rather
than read from the summary.

**The functional act is still absent.** Press and dispense are 0.00, exactly as
pre-registered. That is expected and is not a defect: only 1 of 22 executable
demonstrations shows a genuine grasped press, none shows a dispense, and behaviour
cloning cannot produce behaviour the data does not contain. Reaching the plunger stages
requires reward-driven discovery, which is what the corrected RL setup exists to test.

Contracts: `g1_piston_bc_feature_mismatch.json`, `g1_piston_bc_squash_mismatch.json`.

**A correction to our own record.** The earlier explanation for BC's failure — that the
policy cannot learn the task from one RGB frame without proprioception — was inferred
from these defects and is withdrawn. It was recorded as the leading explanation on the
strength of an open-loop probe that measured the head *without* the deployment squash,
so it scored a function that is never executed. A fidelity measurement is only meaningful
at the point of execution.

## Why RL destroyed the working policy, and the fix

Two RLPD runs warm-started from the behaviour-cloning policy (v3 grasp 1.00, lift 0.80)
and both evaluated at grasp **0.00**, while every training diagnostic looked healthy:
critic loss converging, Q rising, demonstration replay active at 0.5, log-probability
moving toward its target. Two confident diagnoses were refuted by measurement first
(a proprioception gap; a miscalibrated entropy target — real, fixed, and not the cause).

What located the cause was that the action head barely moved (mean |Δ| ≈ 0.0005 per
tensor) while the **executed** action error went 0.40° → 12–14°, and the head's raw
output shrank 0.415 → 0.24. That is a policy being pulled toward the tanh's centre, not
one learning something different.

**Reproduced offline, with no critic, no reward and no simulator.** Optimising the
working head against the entropy term alone (Adam 3e-6, α = 0.06, 650 updates):

| entropy measured on | deployed error | raw magnitude |
|---|---|---|
| the executed action (trainer's formula) | 0.40° → **15.04°** | 0.415 → 0.196 |
| the latent (pre-squash) chunk | 0.40° → 0.40° | unchanged |
| none | 0.40° → 0.40° | unchanged |

Two defects in one formula:

1. **Std sign.** The correlated-noise log-prob omitted the change-of-variables term for
   the learned std (`−n_basis · Σ log std_d`), so `∂ log p / ∂ log std` was *positive* at
   every std — more noise reported a higher density. The entropy term was shrinking
   exploration, not regulating it.
2. **Mean force.** Entropy measured on the executed action includes the tanh Jacobian,
   whose mean-gradient is `+2α · E[tanh(u)]`: an inward pull on every pre-squash
   component ([arXiv 2608.24488](https://arxiv.org/html/2608.24488), measured cosine
   +0.987 there, 90 % of components here). A competent policy has large |means|, so it is
   eroded toward zero at any positive α, regardless of the target — which is why run 5's
   recalibrated target changed nothing.

**Fix.** One shared log-probability implementation for the trainer and the target
calibration, with the change-of-variables term; entropy measured in latent space by
default (closed-form target, slope −4/std, never flat); and 300 critic-only warm-up
updates before the actor moves, since the warm start reinitialises the critic
([WSRL](https://arxiv.org/abs/2412.07762); [ResFiT](https://arxiv.org/abs/2509.19301)).
Run 6 is pre-registered with a rule that measures the flattening signature directly at
the first checkpoint. Contract: `g1_piston_entropy_mean_force.json`.

A correction to our own record: one test had asserted the defective curve's increase
with std as a sanity property. It is relabelled as the defect it pins.

## Status and honest expectations

**A policy now performs the transport task**: reach 1.00, grasp 1.00, lift 0.80,
plate 0.80 on the frozen 25-condition suite under the corrected v3 predicate, from
behaviour cloning on 16 demonstrations once two train/deploy defects were fixed. Videos
are rendered from conditions the evaluation actually scored.

**No policy has performed the full pipette task**, and the gap is specific rather than
mysterious: press and dispense are 0.00, because 1 of 22 executable demonstrations shows
a grasped press and none shows a dispense. Imitation cannot supply behaviour the data
does not contain, so the plunger stages must be discovered from reward. Every RL attempt
so far destroyed the starting policy for a reason that is now measured and fixed; run 6
is the first that can fairly test discovery from a competent initialisation.

Two cautions carried forward. Post-grasp outcomes are not reproducible from a single
rollout in this setup (`g1_piston_post_grasp_nondeterminism.json`), so a rendered video
can differ from its evaluation row; the 25-condition evaluation is the number of record.
And both defects fixed here were invisible to every test in the suite until the failure
was measured end to end at the point of execution — the parity tests in
`test_g1_piston_feature_extraction_parity.py` now pin that agreement, but the general
lesson is to keep a policy-free control arm that proves the execution path can do the
task.
