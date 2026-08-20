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
| RL raises grasp from 0.02 to ~1.00 | **SUPPORTED** | spread 0.000, Jaccard 1.0 across 6 identical runs |
| No RL checkpoint ever completes the task | **SUPPORTED** | full success 0.00 in every run of every arm |
| The untrained SFT policy solves it stochastically | **SUPPORTED** | 0.16 success, replicated by an independent render (0.16), verified on video |
| SAC beats RLPD, or the reverse | **NOT SUPPORTED** | noise range [0.00, 0.36] contains the algorithmic range [0.12, 0.36] |
| Any per-condition lift or carry claim | **NOT SUPPORTED** | carry Jaccard 0.074 across identical runs |
| Chunk-boundary jitter limits carry | **NOT SUPPORTED** | Δcarry −0.04, inside noise |

---

## 1. Post-grasp outcomes are not reproducible

Six identical-protocol deterministic evaluations of **one** checkpoint
(`rlpd_ckpt_step415140`) on the **same** frozen 25 conditions:

```
carry    0.12  0.20  0.00  0.36  0.20  0.20     mean 0.180  stdev 0.118  spread 0.360
lift     0.20  0.32  0.24  0.48  0.24  0.40
grasp    1.00  1.00  1.00  1.00  1.00  1.00     spread 0.000
success  0.00  0.00  0.00  0.00  0.00  0.00
```

The dissociation is the result. In the *same* runs, grasp is perfectly reproducible
condition-for-condition (Jaccard **1.0**) while carry is essentially uncorrelated
(Jaccard **0.074**). The instrument is sound; post-grasp outcomes carry almost no per-run
signal.

The run-to-run carry spread (0.360) **exceeds the Wilson95 width a single run reports**
(0.282), so a single evaluation is provably over-confident, not merely suspected of it.
The spread converged: 0.08 → 0.20 → 0.36 → 0.36 → 0.36.

**Why this sinks the comparison.** The study's arms were each evaluated once:

```
across algorithms and seeds:  SAC s1 0.36 | SAC s2 0.16 | RLPD 0.12   range [0.12, 0.36]
one unchanged checkpoint:     0.12 0.20 0.00 0.36 0.20 0.20           range [0.00, 0.36]
```

Run 4 of the *RLPD* checkpoint produced carry 0.36 — the exact value that made SAC seed 1
look like the strongest arm — while the same checkpoint produced 0.00 two runs earlier.

**Power required**, in repeats per arm, not conditions — the variance is between runs, so
a larger single evaluation does not shrink it:

| carry difference to resolve | repeats per arm |
|---|---|
| 0.10 | ~6 |
| 0.05 | ~22 |
| 0.02 | ~134 |

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

Throws appear in all six repeat runs, and **run 3 was entirely throws** (6 lifts, 0
carries; horizontal displacement 0.0004–0.0115 m against a 0.05 m threshold). The `lift`
bonus pays for height alone, so flinging the piston banks reward without transport.

Any future carry comparison should fix this first — otherwise it optimises and measures
the wrong quantity.

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
