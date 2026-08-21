# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The demo-envelope filter must kill the measured pump and pass the demonstrations.

Guards the hand-oscillation finding: the policy's finger commands swing ~0.67 rad per
30-step chunk at ~1.7 Hz, 43x the demonstrations' per-step velocity, while 97.6% of
demonstrated signal energy lies below 1.0 Hz.
"""

import glob
import os

import numpy as np
import pytest

from rlinf.envs.isaaclab.tasks.g1_piston_action_filter import (
    CONTROL_DT,
    DEMO_BANDWIDTH_HZ,
    DEMO_MAX_STEP,
    DemoEnvelopeFilter,
    attenuation_at,
)

SCRATCH = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
           "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")


def test_first_sample_has_no_startup_transient():
    f = DemoEnvelopeFilter(dim=3)
    u = np.array([0.4, -0.2, 1.1])
    assert np.allclose(f.step(u), u)


def test_the_measured_pump_is_strongly_attenuated():
    """The pathology: ~1.7 Hz oscillation. The filter must cut it to under half."""
    g = attenuation_at(1.7)
    assert g < 0.45, g


def test_demonstrated_bandwidth_is_preserved():
    """Behaviour the demos actually contain (<= ~0.5 Hz dominant) must pass."""
    assert attenuation_at(0.3) > 0.9
    assert attenuation_at(0.5) > 0.8


def test_step_response_closes_a_hand_fast_enough():
    """Demos close the hand over ~1 s; the filtered step must reach 90% within 0.6 s
    (measured 0.52 s at the calibrated cutoff)."""
    f = DemoEnvelopeFilter(dim=1)
    f.step(np.zeros(1))
    y = [f.step(np.array([1.3]))[0] for _ in range(int(0.6 / CONTROL_DT))]
    assert y[-1] >= 0.9 * 1.3, y[-1]


def test_slew_cap_is_the_demonstrated_maximum():
    f = DemoEnvelopeFilter(dim=1)
    f.step(np.zeros(1))
    out = f.step(np.array([100.0]))
    assert abs(out[0]) <= DEMO_MAX_STEP + 1e-12


def test_output_step_never_exceeds_the_envelope():
    rng = np.random.default_rng(0)
    f = DemoEnvelopeFilter(dim=4)
    prev = f.step(rng.normal(size=4))
    for _ in range(500):
        cur = f.step(rng.normal(scale=3.0, size=4))
        assert np.all(np.abs(cur - prev) <= DEMO_MAX_STEP + 1e-9)
        prev = cur


def test_state_carries_across_chunk_boundaries():
    """Feeding one long sequence or the same sequence chunked must be identical,
    so the filter subsumes chunk blending rather than re-introducing boundary steps."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(90, 5))
    a = DemoEnvelopeFilter(dim=5)
    whole = a.filter_chunk(x)
    b = DemoEnvelopeFilter(dim=5)
    parts = np.vstack([b.filter_chunk(x[i:i + 30]) for i in range(0, 90, 30)])
    assert np.allclose(whole, parts)


def test_reset_makes_episodes_independent():
    f = DemoEnvelopeFilter(dim=2)
    f.filter_chunk(np.ones((30, 2)) * 5.0)
    f.reset()
    u = np.array([0.1, 0.2])
    assert np.allclose(f.step(u), u)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        DemoEnvelopeFilter(dim=30).step(np.zeros(7))
    with pytest.raises(ValueError):
        DemoEnvelopeFilter(dim=30, fc_hz=0.0)


@pytest.mark.skipif(not glob.glob(os.path.join(SCRATCH, "act_ep*.npy")),
                    reason="demonstration files not on this machine")
def test_real_demonstrations_pass_nearly_unchanged():
    """The filter must not rewrite the behaviour it is calibrated on."""
    for fp in sorted(glob.glob(os.path.join(SCRATCH, "act_ep*.npy")))[:3]:
        A = np.load(fp)
        f = DemoEnvelopeFilter(dim=A.shape[1])
        Y = f.filter_chunk(A)
        rmse = float(np.sqrt(np.mean((Y - A) ** 2)))
        # Demos span ~1.3 in the hand dims; a few-percent tracking error is passing.
        assert rmse < 0.05, (fp, rmse)


@pytest.mark.skipif(not os.path.exists(os.path.join(SCRATCH, "blendcmd_b6_cmd.npy")),
                    reason="measured policy command stream not on this machine")
def test_the_actual_measured_policy_pump_is_flattened():
    """Apply the filter to the real recorded policy commands: the per-chunk finger
    swing must drop by at least half, toward the demonstrated hold behaviour."""
    C = np.load(os.path.join(SCRATCH, "blendcmd_b6_cmd.npy"))  # (T, 53) joint cmds
    x = C[:, 47:48]  # the worst finger joint, action-equivalent (coupling 1.0)
    f = DemoEnvelopeFilter(dim=1)
    y = f.filter_chunk(x)[:, 0]
    def swing(sig):
        return np.mean([sig[b:b + 30].max() - sig[b:b + 30].min()
                        for b in range(0, len(sig) - 30, 30)])
    assert swing(y) < 0.5 * swing(x[:, 0]), (swing(x[:, 0]), swing(y))


def test_smoothness_penalty_separates_demos_from_the_pathology():
    """The training-time penalty must score the measured oscillation orders of
    magnitude above demonstration-like motion, so a single weight covers both."""
    import torch

    from rlinf.envs.isaaclab.tasks.g1_piston_action_filter import (
        temporal_smoothness_penalty,
    )

    t = torch.arange(30, dtype=torch.float64) * CONTROL_DT
    demo_like = (0.0009 * 50 * t).unsqueeze(-1).expand(30, 4).unsqueeze(0)
    pump = (0.33 * torch.sin(2 * torch.pi * 1.7 * t)).unsqueeze(-1).expand(30, 4)
    pump = pump.unsqueeze(0)
    p_demo = temporal_smoothness_penalty(demo_like)
    p_pump = temporal_smoothness_penalty(pump)
    assert p_pump / max(p_demo, torch.tensor(1e-12)) > 100, (p_demo, p_pump)


def test_smoothness_penalty_respects_the_active_mask():
    import torch

    from rlinf.envs.isaaclab.tasks.g1_piston_action_filter import (
        temporal_smoothness_penalty,
    )

    chunk = torch.zeros(1, 30, 4)
    chunk[..., 3] = torch.linspace(0, 10, 30)     # violent motion on dim 3 only
    mask = torch.tensor([True, True, True, False])
    assert temporal_smoothness_penalty(chunk, mask).item() == 0.0
    assert temporal_smoothness_penalty(chunk).item() > 0.0


def test_hand_only_filtering_leaves_the_arm_bit_exact():
    """The regression fix: filtering all dims lagged the reach and broke grasp
    (0.68 vs baseline worst 0.96). Hand-only mode must pass arm dims through
    bit-exactly while still smoothing the hands."""
    from rlinf.envs.isaaclab.tasks.g1_piston_action_filter import HAND_DIMS

    rng = np.random.default_rng(2)
    x = rng.normal(size=(120, 30))
    f = DemoEnvelopeFilter(dim=30, apply_dims=HAND_DIMS)
    y = f.filter_chunk(x)
    arm = [i for i in range(30) if i not in HAND_DIMS]
    assert np.array_equal(y[:, arm], x[:, arm])
    assert not np.array_equal(y[:, list(HAND_DIMS)], x[:, list(HAND_DIMS)])
    # and the hands are genuinely smoother
    dh = np.abs(np.diff(y[:, list(HAND_DIMS)], axis=0)).mean()
    dx = np.abs(np.diff(x[:, list(HAND_DIMS)], axis=0)).mean()
    assert dh < 0.5 * dx
