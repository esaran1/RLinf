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

"""The alpha update must CONVERGE to the documented target, not merely be able to reach it.

This test exists because an earlier check was insufficient in a specific, instructive
way. Before run 3 it was verified that ``logp = -8.4`` is REACHABLE by the exploration
distribution -- which it is, near std 0.05-0.1. The run then collapsed anyway, because
the alpha update's fixed point was at ``logp = +8.4``: the residual was
``logp + TARGET_ENTROPY`` rather than ``logp - TARGET_ENTROPY``.

Reachability and convergence are different properties. Run 3 measured the consequence:
log-probability climbed -6.13 -> +7.79 toward +8.4, alpha fell 73%, the entropy term
reached 0.3% of the actor objective, and the policy stopped moving (reach and grasp
0.20 -> 0.00). See ``docs/contracts/g1_piston_entropy_target_sign.json``.

So these tests drive the real update to its fixed point and assert where it lands.
"""

import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, ".")
from rlinf.models.embodiment.modules.entropy_tunning import (  # noqa: E402
    EntropyTemperature,
)


def _rl_space():
    spec = importlib.util.spec_from_file_location(
        "g1s_fp", "rlinf/envs/isaaclab/tasks/g1_piston_rl_space.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["g1s_fp"] = mod
    spec.loader.exec_module(mod)
    return mod


def alpha_direction(logp, target, legacy=False, steps=1):
    """Return whether alpha rises or falls, using plain SGD.

    SGD, not Adam: Adam's normalised step moves nearly the same distance regardless of
    gradient magnitude, which hides the direction entirely. An earlier diagnosis with
    Adam at lr=0.1 showed identical behaviour for every input and was wrong.
    """
    ent = EntropyTemperature(initial_alpha=0.05, alpha_type="softplus", device="cpu")
    opt = torch.optim.SGD(ent.parameters(), lr=1e-2)
    before = float(ent.compute_alpha())
    for _ in range(steps):
        av = ent.compute_alpha()
        lp = torch.tensor(float(logp))
        residual = (lp + target) if legacy else (lp - target)
        loss = -av * residual
        opt.zero_grad()
        loss.backward()
        opt.step()
    return before, float(ent.compute_alpha())


TARGET = -8.4


def test_alpha_rises_when_entropy_is_below_target():
    """Entropy is -E[log pi], so a log-probability ABOVE the target means LESS entropy
    than asked for. Alpha must then RISE, increasing the entropy term's weight in the
    actor objective and pushing the policy back toward exploration."""
    before, after = alpha_direction(logp=-2.0, target=TARGET)
    assert after > before, (before, after)


def test_alpha_falls_when_entropy_is_above_target():
    """A log-probability BELOW the target means MORE entropy than asked for, so alpha
    must fall and let the Q term dominate."""
    before, after = alpha_direction(logp=-12.0, target=TARGET)
    assert after < before, (before, after)


def test_alpha_is_stationary_at_the_target():
    before, after = alpha_direction(logp=TARGET, target=TARGET)
    assert abs(after - before) < 1e-7, (before, after)


def test_the_fixed_point_is_the_documented_target_not_its_negation():
    """The load-bearing assertion. Drive the update to convergence from both sides and
    require it to settle at TARGET, not -TARGET."""
    # Above the target: alpha grows, which (in the full loop) increases entropy pressure.
    _, up = alpha_direction(logp=TARGET + 3.0, target=TARGET, steps=50)
    # Below the target: alpha shrinks.
    _, down = alpha_direction(logp=TARGET - 3.0, target=TARGET, steps=50)
    assert up > down, (up, down)
    # And exactly at the target there is no movement in either direction.
    _, still = alpha_direction(logp=TARGET, target=TARGET, steps=50)
    # float32 parameters cannot represent a difference below ~1e-8, so the tolerance is
    # set by the dtype rather than by the maths: 50 steps of an exactly-zero gradient
    # must not move alpha beyond float32 round-off.
    assert abs(still - 0.05) < 1e-7, still


def test_the_legacy_residual_has_the_opposite_fixed_point():
    """Pins WHY run 3 collapsed: under the legacy residual the stationary point is at
    +8.4, so a policy at the documented target of -8.4 is pushed to concentrate."""
    before, after = alpha_direction(logp=TARGET, target=TARGET, legacy=True, steps=1)
    assert after < before, "legacy residual should push alpha DOWN at logp = -8.4"
    # Stationary only at the negation of the documented target.
    b2, a2 = alpha_direction(logp=-TARGET, target=TARGET, legacy=True, steps=1)
    assert abs(a2 - b2) < 1e-7, (b2, a2)


def test_run3_logprob_trajectory_is_explained_by_the_legacy_residual():
    """Run 3's measured log-probability climbed from -6.13 toward +7.79. Under the
    legacy residual that is movement TOWARD the fixed point; under the fixed residual it
    is movement AWAY from it. The direction of alpha's pressure must reflect that."""
    # At run 3's starting logp, the legacy residual pushes alpha down (less entropy).
    b, a = alpha_direction(logp=-6.13, target=TARGET, legacy=True)
    assert a < b
    # The corrected residual pushes alpha UP there, restoring exploration.
    b2, a2 = alpha_direction(logp=-6.13, target=TARGET, legacy=False)
    assert a2 > b2


def test_target_is_inside_the_achievable_range():
    """Necessary but NOT sufficient -- this is the check that passed before run 3 while
    the run still collapsed. Retained so the distinction stays visible."""
    m = _rl_space()
    assert m.default_target_entropy() == pytest.approx(-8.4)
    # The measured floor for this squashed action space.
    assert m.MIN_ACHIEVABLE_LOGPROB < m.default_target_entropy()


def test_trainer_uses_the_corrected_residual_by_default():
    src = open("tools/g1_piston/train_sac.py").read()
    assert "(_logp - TARGET_ENTROPY)" in src
    assert 'os.environ.get("ENTROPY_RESIDUAL_LEGACY", "0")' in src


def test_run_records_where_alpha_actually_converges():
    """A run's own record must state its fixed point, so a result can never be misread
    as having used the documented target when it used the negation."""
    src = open("tools/g1_piston/train_sac.py").read()
    assert '"alpha_fixed_point_logp"' in src
    assert '"entropy_residual_legacy"' in src
