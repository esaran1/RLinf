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

"""n-step items must equal 1-step items at n=1 and carry the right return otherwise."""

import pytest

from rlinf.envs.isaaclab.tasks.g1_piston_nstep import nstep_items


def _ep(T, done_last=True):
    # (s_t, a_t, r_t, s_{t+1}, done_t)
    return [(f"s{t}", f"a{t}", float(t + 1), f"s{t+1}", (t == T - 1) and done_last)
            for t in range(T)]


def test_n1_is_the_identity_on_1_step_items():
    ep = _ep(5)
    out = nstep_items(ep, 1, 0.98)
    for (s, a, r, sn, d), (s2, a2, R, sn2, d2, g) in zip(ep, out):
        assert (s, a, r, sn, d) == (s2, a2, R, sn2, d2)
        assert g == pytest.approx(0.98)


def test_n3_return_and_bootstrap_state():
    ep = _ep(6)
    out = nstep_items(ep, 3, 0.5)
    s, a, R, sn, d, g = out[0]
    assert (s, a) == ("s0", "a0")
    assert R == pytest.approx(1 + 0.5 * 2 + 0.25 * 3)
    assert sn == "s3" and d is False and g == pytest.approx(0.125)


def test_tail_items_shrink_and_inherit_the_terminal_done():
    ep = _ep(4, done_last=True)
    out = nstep_items(ep, 3, 0.9)
    # t=2: m=2 -> r2 + 0.9 r3, next s4, done True, gamma^2
    s, a, R, sn, d, g = out[2]
    assert R == pytest.approx(3 + 0.9 * 4) and sn == "s4" and d is True and g == pytest.approx(0.81)
    # t=3: m=1
    s, a, R, sn, d, g = out[3]
    assert R == pytest.approx(4.0) and sn == "s4" and d is True and g == pytest.approx(0.9)


def test_truncated_episode_does_not_invent_a_done():
    ep = _ep(3, done_last=False)
    out = nstep_items(ep, 5, 0.98)
    assert all(d is False for *_, d, g in out)
    assert out[0][5] == pytest.approx(0.98 ** 3)


def test_rejects_n_below_1():
    with pytest.raises(ValueError):
        nstep_items(_ep(2), 0, 0.98)


def test_trainer_wires_nstep_and_ensemble_flags():
    src = open("tools/g1_piston/train_sac.py").read()
    assert 'N_STEP = int(os.environ.get("N_STEP", "1"))' in src
    assert 'NUM_Q = int(os.environ.get("NUM_Q", "2"))' in src
    assert "num_q_heads=NUM_Q" in src
    assert "tq = rews + (~dones) * gpow * qmin" in src
    assert "NST.nstep_items(" in src


def test_rlpd_aggregation_flags_default_to_todays_behaviour():
    """RLPD: target = min over a random subset (2 of 10); actor = MEAN over all.
    Defaults must reproduce every earlier run: subset = all heads, actor = min."""
    src = open("tools/g1_piston/train_sac.py").read()
    assert 'TARGET_SUBSET = int(os.environ.get("TARGET_SUBSET", str(NUM_Q)))' in src
    assert 'ACTOR_Q_AGG = os.environ.get("ACTOR_Q_AGG", "min").lower()' in src
    assert "torch.randperm(qn.shape[1], device=qn.device)[:TARGET_SUBSET]" in src
    assert '_qall.mean(dim=1, keepdim=True) if ACTOR_Q_AGG == "mean"' in src


def test_demo_tuple_has_six_fields_everywhere():
    """The n-step change grew the demo tuple to 6 fields; every unpack must agree, or a
    run dies at the prewarm loop three minutes after launch."""
    src = open("tools/g1_piston/train_sac.py").read()
    assert "img, act, r, nimg, d, gp_ = demo[idx]" in src
    assert "_img, _act, _r, _nimg, _d, _gp = demo[_idx]" in src
    assert "= demo[_idx]\n" not in src.replace("_img, _act, _r, _nimg, _d, _gp = demo[_idx]\n", "")
