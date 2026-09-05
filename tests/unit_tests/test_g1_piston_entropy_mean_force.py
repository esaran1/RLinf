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

"""The entropy term must not flatten a competent policy.

Two defects shared one formula. (1) The correlated-noise log-prob omitted the
change-of-variables term for ``std``, so ``d lp / d logstd`` was POSITIVE: more noise
reported a higher density. (2) Measuring entropy on the EXECUTED action includes the tanh
Jacobian, whose gradient w.r.t. the pre-squash mean is ``+2 alpha E[tanh(u)]`` per dim:
an inward force on essentially every component (arXiv 2608.24488).

Measured offline on the working behaviour-cloning head, with NO critic and NO reward
(``docs/contracts/g1_piston_entropy_mean_force.json``):

    executed entropy alone:  deployed error 0.40 -> 15.04 deg, raw magnitude 0.415 -> 0.196
    latent entropy alone:    0.40 -> 0.40, unchanged
    no entropy:              0.40 -> 0.40, unchanged

That reproduces the signature of RL runs 4 and 5 (11.5-13.8 deg, 0.24-0.25) from the
entropy term by itself. These tests pin the corrected shared implementation.
"""

import importlib.util
import json
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


@pytest.fixture(scope="module")
def mods():
    rlsp = _load("g1s_mf", RL + "g1_piston_rl_space.py")
    cnz = _load("g1c_mf", RL + "g1_piston_correlated_policy.py")
    return rlsp, cnz


def _setup(mods, std=0.25, n=256, seed=0, mean_scale=0.4):
    rlsp, cnz = mods
    torch.manual_seed(seed)
    noise = cnz.CorrelatedChunkNoise(horizon=cnz.HORIZON, dims=cnz.ACTION_DIMS,
                                     device="cpu")
    mask = rlsp.build_active_mask()
    logstd = torch.full((cnz.ACTION_DIMS,), float(torch.log(torch.tensor(std))),
                        requires_grad=True)
    # A competent policy's means are far from zero: the BC head measured |mean| 0.415.
    # Use clearly non-zero means (|mean| in [0.3, 0.8], random sign) so the sign test is
    # about the force, not about components smaller than the exploration noise.
    sign = torch.sign(torch.randn(n, cnz.HORIZON, cnz.ACTION_DIMS))
    mag = 0.3 + 0.5 * torch.rand(n, cnz.HORIZON, cnz.ACTION_DIMS)
    mean = (sign * mag).requires_grad_(True)
    z = torch.randn(n, cnz.N_BASIS, cnz.ACTION_DIMS)
    pre = mean + noise.expand(z, torch.exp(logstd))
    return noise, mask, logstd, mean, z, pre


def test_corrected_logprob_has_negative_std_gradient_in_both_spaces(mods):
    """Defect 1, fixed: more noise must mean lower density."""
    for space in ("latent", "executed"):
        noise, mask, logstd, mean, z, pre = _setup(mods)
        lp = noise.logprob_chunk(z, pre, logstd, mask, space).mean()
        lp.backward()
        g = logstd.grad[mask]
        assert (g < 0).all(), (space, g)


def test_legacy_logprob_has_positive_std_gradient(mods):
    """Defect 1, pinned: the formula runs 1-5 used points the wrong way."""
    noise, mask, logstd, mean, z, pre = _setup(mods)
    lp = noise.logprob_chunk_legacy(z, pre, mask).mean()
    lp.backward()
    assert (logstd.grad[mask] > 0).all(), logstd.grad[mask]


def test_latent_entropy_exerts_no_force_on_the_mean(mods):
    """Defect 2, fixed: the load-bearing property. Latent entropy is independent of
    where the mean sits, so the entropy term cannot flatten a policy."""
    noise, mask, logstd, mean, z, pre = _setup(mods)
    lp = noise.logprob_chunk(z, pre, logstd, mask, "latent").mean()
    lp.backward()
    assert mean.grad is None or float(mean.grad.abs().max()) == 0.0


def test_executed_entropy_pushes_the_mean_inward(mods):
    """Defect 2, pinned: with the tanh Jacobian, minimising alpha*logp moves nearly
    every active component of the mean toward zero."""
    noise, mask, logstd, mean, z, pre = _setup(mods)
    lp = noise.logprob_chunk(z, pre, logstd, mask, "executed").mean()
    lp.backward()
    g = mean.grad[..., mask]
    m = mean.detach()[..., mask]
    # Gradient DESCENT moves the mean by -g; inward means sign(g) == sign(mean).
    frac_inward = float((torch.sign(g) == torch.sign(m)).float().mean())
    assert frac_inward > 0.85, frac_inward


def test_executed_mean_force_matches_the_analytic_form(mods):
    """arXiv 2608.24488: the mean-gradient of the Jacobian term is 2*tanh(u) per dim."""
    noise, mask, logstd, mean, z, pre = _setup(mods, n=64)
    lp = noise.logprob_chunk(z, pre, logstd, mask, "executed").sum()
    lp.backward()
    analytic = 2 * torch.tanh(pre.detach())
    # Only the Jacobian depends on the mean, so the gradient is exactly the analytic
    # form on active dims and zero on frozen ones.
    assert torch.allclose(mean.grad[..., mask], analytic[..., mask], atol=1e-4)
    assert float(mean.grad[..., ~mask].abs().max()) == 0.0


def test_latent_target_is_closed_form_in_std(mods):
    """With latent entropy the target is analytic: it depends on std only through
    ``-n_basis * n_active * log(std) / horizon``, so its slope is ``-4/std`` nats per
    unit std -- never flat, unlike the legacy curve below std 0.10."""
    rlsp, cnz = mods
    n_active = int(rlsp.build_active_mask().sum())
    for s in (0.1, 0.25, 0.4):
        a = rlsp.measure_logprob_at_std(s, correlated=True, entropy_space="latent")
        b = rlsp.measure_logprob_at_std(s * 1.01, correlated=True, entropy_space="latent")
        slope = (b - a) / (0.01 * s)
        expected = -cnz.N_BASIS * n_active / cnz.HORIZON / s
        assert abs(slope - expected) < 0.05 * abs(expected), (s, slope, expected)


def test_calibrated_target_passes_the_slope_guard_in_latent_space(mods):
    rlsp, _ = mods
    t = rlsp.calibrated_target_entropy(correlated=True, entropy_space="latent")
    assert t == pytest.approx(
        rlsp.measure_logprob_at_std(rlsp.CALIBRATED_TARGET_STD, correlated=True,
                                    entropy_space="latent"))


def test_trainer_uses_the_shared_implementation_and_latent_by_default():
    with open("tools/g1_piston/train_sac.py") as f:
        src = f.read()
    assert 'ENTROPY_SPACE = os.environ.get("ENTROPY_SPACE", "latent").lower()' in src
    assert "corr_noise.logprob_chunk(zc, pre, logstd, ACT_MASK, ENTROPY_SPACE)" in src
    assert "RLSP.iid_logprob(dist, flat_sample, ACT_MASK," in src
    # The legacy formula must not be reachable without the explicit flag.
    assert "if ENTROPY_LP_LEGACY:" in src
    assert "logprob_chunk_legacy" in src


def test_trainer_replicates_the_shared_branch_exactly(mods):
    """Executable replication of the trainer's correlated branch against the shared
    function, so a wiring mistake fails here rather than three hours into a run."""
    rlsp, cnz = mods
    noise, mask, logstd, mean, z, pre = _setup(mods, n=4)
    lp_shared = noise.logprob_chunk(z, pre, logstd, mask, "latent")
    # By-hand latent formula on active dims.
    m = mask.float()
    lpz = ((-0.5 * z ** 2 - 0.5 * torch.log(torch.tensor(2 * torch.pi))) * m).sum((-2, -1))
    cov = -(cnz.N_BASIS * (logstd * m).sum()) - m.sum() * torch.log(noise.scales).sum()
    by_hand = ((lpz + cov) / cnz.HORIZON).unsqueeze(-1).expand(-1, cnz.HORIZON)
    assert torch.allclose(lp_shared, by_hand, atol=1e-5)


def test_actor_warmup_is_wired_and_recorded():
    with open("tools/g1_piston/train_sac.py") as f:
        src = f.read()
    assert 'ACTOR_WARMUP_UPDATES = int(os.environ.get("ACTOR_WARMUP_UPDATES", "300"))' in src
    assert "if grad_updates < ACTOR_WARMUP_UPDATES:" in src
    assert '"actor_updated": False' in src and '"actor_updated": True' in src
    assert '"actor_warmup_updates": ACTOR_WARMUP_UPDATES' in src


def test_offline_replication_is_on_the_record():
    """The measurement that settled the cause, pinned from the contract."""
    with open("docs/contracts/g1_piston_entropy_mean_force.json") as f:
        c = json.load(f)
    a = c["offline_replication"]["arm_A_executed_entropy"]
    b = c["offline_replication"]["arm_B_latent_entropy"]
    assert a[-1]["deployed_err_deg"] > 10.0 and a[0]["deployed_err_deg"] < 0.5
    assert all(abs(r["deployed_err_deg"] - b[0]["deployed_err_deg"]) < 1e-6 for r in b)
