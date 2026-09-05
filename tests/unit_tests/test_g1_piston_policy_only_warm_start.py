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

"""RL warm-started from a behaviour-cloning checkpoint.

The BC policy is the first in this project to perform the transport task (grasp 1.00,
lift 0.80 under v3), so it is the initialisation the plunger-discovery run should start
from. But a BC checkpoint is POLICY-ONLY: it has no critic, no target and no alpha, and
its ``actor_logstd`` is a placeholder written at save time (std 0.05), not a learned
exploration scale.

Two things must therefore be true, and both are load-bearing:

1. The trainer must not KeyError on the missing critic/alpha.
2. It must NOT inherit the placeholder std. The corrected entropy target (-8.4) is
   defined for std 0.20; starting at 0.05 begins far above the alpha equilibrium, which
   is the exact condition that collapsed run 3
   (docs/contracts/g1_piston_entropy_target_sign.json).

A silent inheritance of 0.05 would look like a working run for hundreds of updates and
then collapse the same way, so it is pinned here rather than left to inspection.
"""

import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")

SRC = "tools/g1_piston/train_sac.py"


def _src():
    with open(SRC) as f:
        return f.read()


def _rl_space():
    spec = importlib.util.spec_from_file_location(
        "g1s_ws", "rlinf/envs/isaaclab/tasks/g1_piston_rl_space.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["g1s_ws"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_policy_only_checkpoints_are_detected_rather_than_crashing():
    src = _src()
    assert 'POLICY_ONLY = "critic" not in wc or "alpha" not in wc' in src, (
        "the trainer must detect a policy-only checkpoint explicitly; otherwise it "
        "KeyErrors on wc['alpha'] when warm-started from behaviour cloning."
    )
    assert '"policy_only_checkpoint"' in src, (
        "the run's own record must state whether it started from a policy-only "
        "checkpoint, so a result cannot be misread as having inherited a critic."
    )


def test_exploration_std_is_reset_not_inherited_from_a_bc_checkpoint():
    """The load-bearing assertion: BC's placeholder std must not become the run's
    exploration scale."""
    src = _src()
    # The PROPERTY, not a literal constant: exploration is reset to the std the entropy
    # target is calibrated for. The constant itself changed when the target was
    # recalibrated (TARGET_ENTROPY_STD -> CALIBRATED_TARGET_STD, see
    # docs/contracts/g1_piston_entropy_target_miscalibrated.json); asserting the old
    # literal would have forced the fix to break this test for no reason.
    assert "actor_logstd.fill_(math.log(_reset_std))" in src, (
        "a policy-only warm start must reset exploration to the std the entropy target "
        "is calibrated for, not inherit the checkpoint's placeholder."
    )
    assert "RLSP.CALIBRATED_TARGET_STD" in src, (
        "the reset must aim at the CALIBRATED std, or the run starts off-equilibrium."
    )
    assert '"actor_logstd_source"' in src, (
        "the run must record where its exploration std came from."
    )


def test_the_reset_std_matches_the_entropy_target_it_is_defined_for():
    """Reset and target must stay consistent; if one is changed without the other the
    run starts away from the alpha equilibrium again."""
    m = _rl_space()
    assert m.TARGET_ENTROPY_STD == pytest.approx(0.20)
    assert m.default_target_entropy() == pytest.approx(-8.4)
    assert m.MIN_ACHIEVABLE_LOGPROB < m.default_target_entropy()


def test_bc_placeholder_std_would_have_been_wrong():
    """Pins WHY the reset exists, in numbers rather than prose.

    BC writes log(0.05) as a placeholder. The calibrated target aims at
    CALIBRATED_TARGET_STD, so inheriting the placeholder would start the run at a
    materially smaller exploration scale than the target is defined for.
    """
    import math
    m = _rl_space()
    bc_placeholder = 0.05
    correct = m.CALIBRATED_TARGET_STD
    assert bc_placeholder < correct
    assert correct / bc_placeholder >= 4.0, (correct, bc_placeholder)
    assert math.log(correct) > math.log(bc_placeholder)


def test_critic_is_reinitialised_and_the_reason_recorded():
    src = _src()
    assert "policy-only checkpoint (behaviour cloning): no critic to transfer" in src
    assert '"critic_reinit_reason"' in src


def test_the_policy_always_transfers():
    """Whatever else happens, the learned behaviour must carry over -- that is the whole
    point of warm-starting from BC."""
    src = _src()
    assert 'model.action_model.load_state_dict(wc["action_model"])' in src
