# What to show, and what to say

A one-page guide for presenting this work. Every claim below is backed by a file in this
directory; nothing here is stronger than the data supports.

---

## Show this one video

```
verified_results/videos_smooth/comparison/untrained_success_vs_rl_grasp_only.mp4
```

Two rollouts side by side, same task, same scene, same camera:

| | left — untrained SFT | right — after 415k RL steps |
|---|---|---|
| start | `stage -`, disp 0.008 m | `stage -`, disp 0.007 m |
| middle | `stage lift`, disp 0.182 m | `stage grasp`, disp 0.008 m |
| end | **`stage success`, disp 0.298 m, return 35.88** | **`stage grasp`, disp 0.006 m, return 2.52** |

The right panel holds the piston for the whole second half without moving it. The
contrast is legible without any statistics.

**What to say:** *"The untrained policy completes the task 16% of the time under its own
sampling noise. After 415k steps of RL the policy grasps on essentially every condition
and completes the task zero times, in every run. RL learned the first stage of the
sequence and eliminated the rest."*

---

## Then show the number that matters more

One checkpoint, eight identical deterministic evaluations, same frozen conditions:

```
carry    0.12  0.20  0.00  0.36  0.20  0.20  0.12  0.36
grasp    1.00  1.00  1.00  1.00  1.00  1.00  1.00  0.96
success  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
```

Run 4 produced carry **0.36** — the exact value that made SAC seed 1 look like the best
arm in the study — from a checkpoint that produced **0.00** two runs earlier.

**What to say:** *"Post-grasp outcomes in this setup are not reproducible. Grasp is
199/200 attempts with per-condition agreement of 0.99; carry has per-condition agreement
of 0.066 and a rate spanning 0.00 to 0.36. The noise range from re-running one unchanged
checkpoint contains the entire range we observed across both algorithms and all seeds. So
the SAC-versus-RLPD comparison we set out to make is not resolvable from single
evaluations — it needs about 19 repeats per arm to detect a 0.05 difference."*

This is the stronger contribution. It is a methodological result that applies to anyone
comparing manipulation policies from one evaluation run.

---

## Questions you will get

**"Can you re-run that specific rollout?"**
For grasp, yes. For lift or carry, no — a different run succeeds on different conditions.
Say this before being asked; it is the finding, not a limitation of the tooling.

**"Why do the hands shake in the earlier videos?"**
Two real, separately measured defects. (1) Chunk-boundary jerk: the command jumped up to
1.0 rad in one 50 Hz step at every 30-step boundary — fixed by blending. (2) The larger
one: **RL fine-tuning itself induced a ~1.7 Hz finger oscillation** — the RLPD policy's
predicted finger trajectory pumps 43× the demonstrations' per-step velocity (SAC: 247×),
while the untrained SFT policy is demo-smooth. Fixed by a demonstration-envelope filter
on the 12 hand dims only (`videos_filtered/rlpd_hand/`): every constant measured from
the demos, grasp preserved at 1.00 on the frozen suite, arm passed through bit-exactly.
**Neither fix changed task outcomes** — carry stays inside the noise range — which rules
out the shaking as the reason grasps do not survive into transport. The calmest validated config is
`videos_filtered/rlpd_handfc06/` (grasp 1.00); the before/after is
`videos_filtered/comparison/hand_oscillation_raw_vs_calmest_config.mp4`. The residual
slow arm sway is also RL-induced and only fixable in training. Full story: FINDINGS.md §5.

**"So RL made things worse?"**
On full task success, RL went from 0.16 (stochastic SFT) to 0.00, and it never recovers
at any checkpoint in any seed. On grasping, RL went from 0.02 to ~1.00 and that is
reproducible. Both are true. The reward pays for grasp and lift, and the policy optimised
exactly that — including flinging the piston for the lift bonus, which one whole
evaluation run consisted of.

**"Is the evaluation itself trustworthy?"**
Grasp reproduces at 199/200 across eight runs, so the instrument works. That is the
control that makes the carry result a measurement rather than a shrug.

---

## Do not say

- "SAC beats RLPD" or the reverse — the noise range covers the entire difference.
- "Our best checkpoint reaches 36% carry" — that number is a single draw from a
  distribution spanning 0.00–0.36.
- "RL improves manipulation on this task" — it improves grasping and eliminates
  transport.
- Anything about a specific condition succeeding — per-condition agreement is 0.066.

---

## Supporting files

| file | contents |
|---|---|
| `FINDINGS.md` | every claim tagged SUPPORTED / NOT SUPPORTED with evidence |
| `tables/repeat_eval_noise_floor.json` | the eight-run reproducibility measurement |
| `tables/blend_ab.json` | the jitter-fix A/B |
| `videos_smooth/` | presentation videos, chunk blending on |
| `videos/` | frozen-protocol videos, backing every reported number |
| `docs/contracts/*.json` | one file per finding, including the refuted hypotheses |
