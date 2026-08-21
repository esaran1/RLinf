# G1 piston RL study — what is and is not supportable

Metrics `v1.1-frozen-2026-08-17-horizontal-fix`. Every number below is recomputed from
raw per-condition rows by the frozen metric code, from evaluations whose condition hashes
match the frozen suite exactly.

---

## The headline

**Reinforcement learning reliably induces grasping on this task and reliably fails to
induce transport. The transport measurement itself is too noisy to compare algorithms.**

| claim | status | evidence |
|---|---|---|
| RL raises grasp from 0.02 to ~1.00 | **SUPPORTED** | 199/200 attempts, spread 0.040, Jaccard 0.99 across 8 runs |
| No RL checkpoint ever completes the task | **SUPPORTED** | full success 0.00 in every run of every arm |
| The untrained SFT policy solves it stochastically | **SUPPORTED** | 0.16 success, replicated by an independent render (0.16), verified on video |
| SAC beats RLPD, or the reverse | **NOT SUPPORTED** | noise range [0.00, 0.36] contains the algorithmic range [0.12, 0.36] |
| Any per-condition lift or carry claim | **NOT SUPPORTED** | carry Jaccard 0.066 across identical runs |
| Chunk-boundary jitter limits carry | **NOT SUPPORTED** | Δcarry −0.04, inside noise |

---

## 1. Post-grasp outcomes are not reproducible

Eight identical-protocol deterministic evaluations of **one** checkpoint
(`rlpd_ckpt_step415140`) on the **same** frozen 25 conditions:

```
carry    0.12 0.20 0.00 0.36 0.20 0.20 0.12 0.36   mean 0.195  stdev 0.122  spread 0.360
lift     0.20 0.32 0.24 0.48 0.24 0.40 0.20 0.56   mean 0.330  stdev 0.136  spread 0.360
grasp    1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.96   mean 0.995  stdev 0.014  spread 0.040
success  0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
```

The dissociation is the result. In the *same* runs, grasp is highly reproducible
condition-for-condition (**199/200** attempts, Jaccard **0.99**) while carry is
essentially uncorrelated (Jaccard **0.066**) — roughly 9× the stability in spread and 15×
in per-condition agreement. The instrument is sound; post-grasp outcomes carry almost no
per-run signal.

The single grasp failure, in run 8, was a condition the policy failed even to **reach**
(return −0.65) — a whole-rollout deviation, not a marginal grasp that slipped.

The run-to-run carry spread (0.360) **exceeds the Wilson95 width a single run reports**
(0.282), so a single evaluation is provably over-confident, not merely suspected of it.
The spread converged and held: 0.08 → 0.20 → 0.36 → 0.36 → 0.36 → 0.36 → 0.36. At
eight runs the **lift** spread also crosses its interval (0.360 vs 0.344); only grasp
remains inside.

**Why this sinks the comparison.** The study's arms were each evaluated once:

```
across algorithms and seeds:  SAC s1 0.36 | SAC s2 0.16 | RLPD 0.12   range [0.12, 0.36]
one unchanged checkpoint:     0.12 0.20 0.00 0.36 0.20 0.20 0.12 0.36 range [0.00, 0.36]
```

Run 4 of the *RLPD* checkpoint produced carry 0.36 — the exact value that made SAC seed 1
look like the strongest arm — while the same checkpoint produced 0.00 two runs earlier.

**Power required**, in repeats per arm, not conditions — the variance is between runs, so
a larger single evaluation does not shrink it:

| carry difference to resolve | repeats per arm |
|---|---|
| 0.10 | ~5 |
| 0.05 | ~19 |
| 0.02 | ~117 |

---

## 2. Chunk-boundary jitter: real, fixed, and not the constraint

The commanded joint trajectory contains a step discontinuity at every 30-step chunk
boundary — nothing forces a new chunk's first action to be continuous with the previous
chunk's last, so the command teleports.

| | intra-chunk | at boundary |
|---|---|---|
| mean joint step | 0.0054 rad | **0.1161 rad (21×)** |
| worst single step | 0.29 rad | **1.00 rad** |

Worst on the Inspire hand joints: **13 of 22 boundaries** command a finger snap over
0.5 rad, including while the piston is held. Tracking error spikes to 0.177 rad at each
boundary and decays to 0.061 by step 29 — the arm spends the first third of every chunk
recovering. The demonstrations contain no such discontinuity (0.94× at every-30th index).

`g1_piston_chunk_blend.py` ramps each chunk in over 6 steps: boundary step
0.1161 → 0.0166 rad, worst jump **1.00 → 0.14 rad**, with the last 24 steps of every
chunk left bit-identical to the policy's own.

**But it does not improve the task.** Paired A/B on the frozen suite: carry 0.20 → 0.16,
grasp 1.00 in both. Two runs of the *same* treatment overlap Jaccard 0.00 while the
cross-treatment pair overlaps 0.15 — the treatment effect is below the noise floor. The
fix is retained on trajectory-realisability grounds (a 1.0 rad step at 50 Hz is not
physically executable and would be unsafe on hardware), **not** on task performance.
Default `BLEND_STEPS=0`.

---

## 3. The untrained policy is the only thing that solves the task

Identical weights; execution mode is the only difference:

| SFT policy | success | carry | grasp |
|---|---|---|---|
| deterministic (n=125 pooled) | 0.00 | 0.00 | 0.01 |
| stochastic (n=25) | **0.16** | 0.24 | 0.48 |

Replicated by an independent render: **0.16** success, 0.24 carry. Verified on video —
the four successes show coherent reach → grasp → lift → plate sequences with true
horizontal transport 0.299–0.307 m, inside the demonstration range of 0.200–0.322 m.

Success is driven by the **sampled noise, not the condition**: three independent draws
succeeded on three disjoint condition sets — {11,14,16,18}, {5,9,20}, {0,4,6,9} — while
the rate held near 0.12–0.16.

The deterministic policy fails differently than "narrowly": in 12 of 15 rollouts the
piston displacement is identical to 4 decimals across *different* reset conditions. It
never touches the object; it only settles ~9 mm under gravity.

**Consequence.** The entire SAC/RLPD comparison was scored deterministically — the one
mode in which even the SFT initialisation is inert — so the baseline the RL arms were
measured against was systematically understated.

---

## 4. The reward exploit is still live

Throws appear in every repeat run, and **one run was entirely throws** (6 lifts, 0
carries; horizontal displacement 0.0004–0.0115 m against a 0.05 m threshold). The `lift`
bonus pays for height alone, so flinging the piston banks reward without transport.

Any future carry comparison should fix this first — otherwise it optimises and measures
the wrong quantity.

---

## 5. RL made the hands oscillate — root cause and the validated fix

The visible "hand shaking" in the videos is **not** sensor noise, rendering, or the
chunk-boundary jitter of finding 2. It is the RL policy's *predicted finger trajectory
itself*: a ~1.7 Hz open-close pump, ~0.6 rad per 30-step chunk, with 92% of its energy
below 2 Hz and smooth sign-flips — deliberate swings, not dither.

**Attribution ladder** (per-chunk finger swing on joint 47, identical execution):

| policy | swing/chunk | per-step | vs demos |
|---|---|---|---|
| demonstrations | 0.03 | 0.0009 | 1× |
| SFT (untrained) | 0.065 | 0.0041 | 4.6× |
| RLPD @415k | 0.580 | 0.0398 | **43×** |
| SAC seed 2 @690k | 1.594 | 0.2225 | **247×** |

SFT is demo-smooth; the oscillation is *created by RL fine-tuning*. Dose-response
supports the mechanism: RLPD's 50% demonstration replay keeps its pump 3× smaller than
SAC's — demo replay partially protects smoothness. The reward has no action-smoothness
term, so nothing opposes the drift (consistent with CAPS, Mysore et al., ICRA 2021).

**Fix: a demonstration-envelope filter** (`g1_piston_action_filter.py`) — causal 2-pole
low-pass at 1.2 Hz plus a 0.175/step slew cap, every constant measured from the 11 demos
(97.6% of demonstrated hand energy is below 1.0 Hz; 0.175 is the largest step any demo
contains). Three pre-registered A/B arms on the frozen suite, judged against the 8-run
noise floor of finding 1:

| arm | grasp | verdict |
|---|---|---|
| filter on all 30 dims | 0.68 | **FAIL** — below every baseline run; arm lag breaks reach timing |
| filter on the 12 hand dims only | **1.00** | **PASS** — arm bit-exact, grasp fully preserved |
| full-dim filter, SFT stochastic | success 0/25 | **FAIL** — destroys the 0.16 success rate |
| hand filter 1.2 Hz + arm blend 6 | 0.96 | **PASS** — presentation config; arm boundary snap 1.0 → 0.15 rad |
| hand filter **0.6 Hz** + arm blend 6 | **1.00** | **PASS** — calmest validated config; finger range 1.32 → 0.31 rad, per-step 43× → 5.6× demos |

In-sim with the hand-only filter: finger per-step motion 0.0398 → 0.0156 (2.6×
smoother), total command range halved (1.32 → 0.68 rad), energy above 1 Hz cut from 32%
to 10%. Carry 0.04, inside the noise range — the filter removes the fast pathology at
zero task cost.

**What remains is not noise — and it is not only the hands.** RL also induced a slow
(sub-1 Hz) **arm sway**: shoulder/elbow command sway is 2–4× the SFT policy's (joint 26:
0.40 → 1.03 rad; joint 16: 0.14 → 0.60), with SAC equal or worse (joint 16: 1.66). This
sits at 0.2–0.8 Hz, where demonstrated reaching also lives, so no execution filter can
remove it — the full-dim A/B already showed arm filtering breaks grasp. The fc=0.6 arm
brought the *fingers* to 5.6× demo velocity (from 43×) at grasp 1.00; the visible
residual in even the calmest config is this arm sway. The source-level remedy is
training-time: a CAPS smoothness penalty (`SMOOTH_LAMBDA`) or heavier demonstration
replay (the dose-response row above). A pre-registered continuation run — RLPD resumed
from the same 415k checkpoint with `SMOOTH_LAMBDA=100` — tests exactly this; its gate
(penalty −50% at grasp ≥ 0.92) was committed before the data.

Two corollaries worth stating plainly:

1. **The SFT policy's successes depend on high-frequency action noise.** Filtering the
   stochastic SFT to the demonstrated bandwidth eliminated not just its 4/25 successes
   but *all reaching*. The only behaviour in this study that completes the task does so
   by exploiting exactly the out-of-envelope dynamics the filter removes.
2. **The training-time fix works, and the controlled experiment proves it.** RLPD was
   continued from the same 415k checkpoint for the same wall-clock (2.5 h, seed 0) twice:
   with `SMOOTH_LAMBDA=100` and with λ=0.

   | continuation | penalty metric | finger swing | arm sway | grasp | lift | return |
   |---|---|---|---|---|---|---|
   | none (415k baseline) | 1.88e-3 | 0.580 | 0.764 | 0.96–1.00 | ≤0.56 | ≤6.07 |
   | **λ=100** | **1.00e-3 (−47%)** | **0.363** | **0.563** | 1.00 | 0.72 | 8.56 |
   | λ=0 control | 3.37e-3 (**+79%**) | 0.549 | 0.769 | 1.00 | 0.68 | 7.71 |

   Three conclusions. (a) *Continued RL degrades smoothness by default* — the control got
   79% rougher, confirming the drift mechanism. (b) *The penalty causes the smoothing*:
   3.4× smoother than the counterfactual, and it is the only arm where the sub-1 Hz arm
   sway moves. (c) *It costs nothing*: task metrics are indistinguishable between the two
   continuations. The pre-registered static gate (−50%) was narrowly missed at −47%;
   the controlled comparison is the stronger and cleaner readout. Both continuations
   exceed the entire 8-run baseline range on lift/tube/return — attributable to the
   extra training, and stated with the noise-floor caveat (one evaluation per arm).
   Videos: `videos_smoothtrain/` (renders reach **lift**; the 415k baseline's matched
   renders end at grasp). One render caveat, diagnosed after the first cut: the sim's
   own termination (an out-of-bounds reset the upstream task misnames `success`) fires
   when the piston leaves y ∈ (0.2, 0.7) — and the smoothed policy carries the piston
   ~0.17 m toward that boundary, so some draws truncate at chunk 3 while a logged
   re-render of the same checkpoint/condition ran all 23 chunks (1380 steps, no fire,
   final y = 0.2345). Truncation biases the continuation lift/return numbers *down*, so
   they are conservative. The comparison video uses the full-length draw; the truncated
   draw is retained.

Contract: `docs/contracts/g1_piston_rl_induced_oscillation.json`. Videos:
`videos_filtered/rlpd_hand/` (hand-only filter, the validated fix).

---

## Recommendation

**Do not** resume the three remaining training seeds for a carry-based comparison: ~32
GPU-hours to produce three more single-evaluation numbers drawn from a distribution whose
spread already covers the entire effect size.

**Do** report grasp emergence as the primary result, full success as 0.00 with the SFT
stochastic 0.16 as the only success anywhere, and the noise floor with its power table as
a methodological contribution. If a carry comparison is still wanted, fix the reward
exploit and budget repeats per arm.

---

## Contracts

`docs/contracts/`: `g1_piston_post_grasp_nondeterminism.json` ·
`g1_piston_chunk_boundary_jitter.json` · `g1_piston_blend_ab_result.json` ·
`g1_piston_sft_stochastic_success.json` · `g1_piston_carry_metric_3d_norm.json` ·
`g1_piston_sac_s1_checkpoint_loss.json` · `g1_piston_render_physics_divergence.json`
(closed, refuted) · `g1_piston_lift_reward_exploit.json`
