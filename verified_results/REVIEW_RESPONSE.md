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

## Status and honest expectations

A retrain is running with every fix applied (v3 reward, corrected discount, DCT critic
input, privileged critic state, smoothness penalty, rebuilt 22-episode buffer). Its
configuration, hypotheses and decision rules were **pre-registered before the data
existed** (`g1_piston_v3_retrain_preregistration.json`), including the commitment that a
dispense rate of 0.00 is a reportable finding about the task and the data rather than
something to hide.

**No policy in this project has performed the full pipette task**, and that is now
measured rather than assumed. What the corrected setup makes possible is a fair test of
whether removing the plateau and fixing the discount converts into genuine grasping and
lifting.
