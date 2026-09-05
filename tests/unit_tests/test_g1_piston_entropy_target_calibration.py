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

"""The entropy target must be reachable by the log-probability it is compared against.

``default_target_entropy()`` returns -8.4 and ``TARGET_ENTROPY_STD`` documents that as
"exploration std ~0.20". Measured against the trainer's own log-probability computation,
those two statements disagree, and they disagree DIFFERENTLY in each noise branch:

    correlated branch:  std 0.20 -> logp -7.76,  -8.4 needs std ~0.05
    i.i.d. branch:      std 0.20 -> logp +7.40,  -8.4 unreachable at any std

Run 4 measured the consequence. Warm-started from a policy with v3 grasp 1.00, it sat at
logp -7.55 against the -8.4 target, so the residual never closed, alpha rose 57%, the
entropy term averaged 28% of the actor objective for the whole run, and the head's output
was flattened (raw magnitude 0.4151 -> 0.2364). Deployed action error went 0.403 -> 13.773
degrees and grasp fell 1.00 -> 0.00, confirmed by an independent evaluation tool.

This is a DIFFERENT defect from the sign bug in test_g1_piston_entropy_alpha_fixed_point.
That fix was to the DIRECTION of the alpha update and is confirmed working here: logp
moved toward the target rather than away. This is about the target's VALUE.

See docs/contracts/g1_piston_entropy_target_miscalibrated.json.
"""

import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")

RL = "rlinf/envs/isaaclab/tasks/"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rl_space():
    return _load("g1s_cal", RL + "g1_piston_rl_space.py")


def _corr():
    return _load("g1c_cal", RL + "g1_piston_correlated_policy.py")


def correlated_logp(std_value, n=200, seed=0):
    """The trainer's correlated-noise log-probability, verbatim, at mean 0.

    Reproduces train_sac.py's branch:
        jac = (log(1 - tanh(pre)^2) * ACT_MASK).sum(-1)
        lp  = logprob_z(z) / horizon - jac
    """
    rlsp, cnz = _rl_space(), _corr()
    am = rlsp.build_active_mask()
    h, d = cnz.HORIZON, cnz.ACTION_DIMS
    torch.manual_seed(seed)
    corr = cnz.CorrelatedChunkNoise(horizon=h, dims=d, device="cpu")
    std = torch.full((d,), float(std_value))
    z = torch.randn(n, cnz.N_BASIS, d)
    pre = corr.expand(z, std)                       # mean 0
    jac = (torch.log(1 - torch.tanh(pre).pow(2) + 1e-7) * am).sum(dim=-1)
    lp = (cnz.CorrelatedChunkNoise.logprob_z(z).unsqueeze(-1) / h) - jac
    return float(lp.mean())


def test_the_documented_std_does_not_produce_the_documented_target():
    """The defect itself, pinned as a measurement.

    If a future change makes these agree, this test SHOULD fail and be updated together
    with the contract -- that is the point of pinning it.
    """
    rlsp = _rl_space()
    at_documented_std = correlated_logp(rlsp.TARGET_ENTROPY_STD)
    target = rlsp.default_target_entropy()
    assert at_documented_std > target, (
        f"logp at the documented std ({at_documented_std:.2f}) should be ABOVE the "
        f"target ({target}); that gap is what drove alpha up for all of run 4."
    )
    assert abs(at_documented_std - target) > 0.4, (
        "the mismatch measured in run 4 was ~0.65 nats; a much smaller gap means the "
        "calibration changed and the contract needs updating."
    )


def test_the_target_is_reachable_only_near_the_bottom_of_the_std_range():
    """-8.4 corresponds to std ~0.05, not the documented 0.20."""
    lo = correlated_logp(0.05)
    hi = correlated_logp(0.30)
    target = _rl_space().default_target_entropy()
    assert lo < target, (lo, target)
    assert hi > target, (hi, target)


def test_logp_decreases_monotonically_as_exploration_shrinks():
    """Sanity on the instrument: less noise must mean higher density, i.e. LOWER logp
    under this branch's sign convention. If this inverts, the measurement above is
    meaningless."""
    vals = [correlated_logp(s) for s in (0.05, 0.10, 0.20, 0.30, 0.50)]
    assert vals == sorted(vals), vals


def test_the_two_noise_branches_disagree_about_the_same_std():
    """The load-bearing structural point: one target cannot serve both branches.

    The i.i.d. branch's log-probability at the documented std is positive, while the
    correlated branch's is around -7.8. A single target of -8.4 cannot be correct for
    both, so the target must be branch-aware or documented per branch.
    """
    from rlinf.models.embodiment.modules.gaussian_policy import SquashedNormal
    rlsp = _rl_space()
    am = rlsp.build_active_mask()
    torch.manual_seed(0)
    n, d = 4000, 30
    std_v = rlsp.TARGET_ENTROPY_STD
    dist = SquashedNormal(torch.zeros(n, d), torch.full((n, d), std_v),
                          low=-2.2, high=2.2)
    a = dist.rsample()
    base = dist.base_dist.base_dist
    parts = []
    for t in dist.transforms:
        parts.extend(getattr(t, "parts", [t]))
    x = a
    for t in reversed(parts):
        x = t.inv(x)
    per = base.log_prob(x)
    y = x
    for t in parts:
        y2 = t(y)
        if type(t).__name__ == "TanhTransform":
            per = per - torch.log(1 - torch.tanh(y).pow(2) + 1e-7)
        y = y2
    iid = float((per * am).sum(-1).mean())
    corr = correlated_logp(std_v)
    assert iid > 0 > corr, (iid, corr)
    assert abs(iid - corr) > 10.0, (
        f"the branches differ by {abs(iid - corr):.1f} nats at the same std; a single "
        "scalar target cannot be calibrated for both."
    )


def test_run4_regression_is_recorded_with_its_cause():
    """The pre-registered H4 rule fired; the record must carry the mechanism, not just
    the number, so the run is not later misread as 'RL does not work here'."""
    import json
    with open("docs/contracts/g1_piston_entropy_target_miscalibrated.json") as f:
        c = json.load(f)
    assert c["measurement"]["target"] == -8.4
    assert "0.05" in str(c["measurement"]["correlated_noise_branch_used_by_run_4"])
    assert "REGRESSION" in c["status_of_run_4"].upper()


# --------------------------------------------- the calibrated target ----
"""``calibrated_target_entropy`` MEASURES the target for the branch in use.

The historical scalar was calibrated against neither branch. These tests pin that the
replacement is self-consistent: the target it returns is exactly the log-density its
own std produces, in the same branch, under the trainer's own computation.
"""


def _space():
    import importlib
    sys.path.insert(0, ".")
    m = importlib.import_module("rlinf.envs.isaaclab.tasks.g1_piston_rl_space")
    return importlib.reload(m)


def test_calibrated_target_equals_the_logprob_at_its_own_std():
    """Self-consistency: this is the property the old constant lacked."""
    m = _space()
    for correlated in (True, False):
        target = m.calibrated_target_entropy(correlated=correlated)
        measured = m.measure_logprob_at_std(m.CALIBRATED_TARGET_STD,
                                            correlated=correlated)
        assert abs(target - measured) < 1e-6, (correlated, target, measured)


def test_the_alpha_residual_is_zero_at_the_calibrated_target():
    """The fixed point of the alpha update must sit exactly where the policy is asked
    to be. Run 4's residual never closed because it did not."""
    m = _space()
    target = m.calibrated_target_entropy(correlated=True)
    logp_there = m.measure_logprob_at_std(m.CALIBRATED_TARGET_STD, correlated=True)
    assert abs(logp_there - target) < 1e-6


def test_alpha_has_leverage_at_the_calibrated_target():
    """A target on a flat part of the curve does not constrain exploration; alpha then
    grows without bound. Run 4's target sat on slope ~2."""
    m = _space()
    std = m.CALIBRATED_TARGET_STD
    d = 0.05 * std
    lo = m.measure_logprob_at_std(std - d, correlated=True)
    hi = m.measure_logprob_at_std(std + d, correlated=True)
    slope = abs((hi - lo) / (2 * d))
    assert slope >= m.MIN_TARGET_SLOPE, slope


def test_flat_region_is_rejected():
    """The std the OLD target implied (~0.05) must now raise rather than silently
    produce a target with no leverage."""
    m = _space()
    with pytest.raises(ValueError, match="slope"):
        m.calibrated_target_entropy(correlated=True, std=0.05)


def test_the_old_target_would_still_be_rejected_today():
    """Guards against someone reinstating -8.4 by adjusting CALIBRATED_TARGET_STD: the
    std that produces -8.4 under the correlated branch is in the flat region."""
    m = _space()
    old = m.default_target_entropy()
    at_old_std = m.measure_logprob_at_std(0.05, correlated=True)
    assert abs(at_old_std - old) < 0.2, (at_old_std, old)
    with pytest.raises(ValueError):
        m.calibrated_target_entropy(correlated=True, std=0.05)


def test_branches_get_different_targets():
    """One scalar cannot serve both; the calibrated function must not pretend it can."""
    m = _space()
    c = m.calibrated_target_entropy(correlated=True)
    i = m.calibrated_target_entropy(correlated=False)
    assert abs(c - i) > 10.0, (c, i)


def test_measurement_is_deterministic():
    """A target that moves between calls would make runs incomparable."""
    m = _space()
    a = m.calibrated_target_entropy(correlated=True)
    b = m.calibrated_target_entropy(correlated=True)
    assert a == b


def test_trainer_uses_the_calibrated_target_by_default():
    with open("tools/g1_piston/train_sac.py") as f:
        src = f.read()
    assert "RLSP.calibrated_target_entropy(correlated=CORRELATED_NOISE)" in src
    assert 'os.environ.get("ENTROPY_TARGET_LEGACY", "0")' in src
    assert '"entropy_target_source"' in src


def test_trainer_starts_exploration_at_the_calibrated_std():
    """Both the cold start and the warm-start reset must aim at the std the target is
    calibrated for, or the run begins off-equilibrium by construction."""
    with open("tools/g1_piston/train_sac.py") as f:
        src = f.read()
    assert "_init_std = (RLSP.TARGET_ENTROPY_STD" in src
    assert "_reset_std = (RLSP.TARGET_ENTROPY_STD" in src
    assert src.count("RLSP.CALIBRATED_TARGET_STD") >= 2
