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

"""Correlated chunk noise must explore where the demonstrations live.

Guards the review point that the action dimensions are not independent. Measured on the
real demonstrations, i.i.d. exploration at std 0.20 adds 0.2257 rad between consecutive
steps against the demonstrations' 0.0011 rad -- 209x -- and spreads 80% of its energy
into temporal components the task never uses.
"""

import glob
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rlinf.envs.isaaclab.tasks.g1_piston_action_basis import (  # noqa: E402
    ACTION_DIMS,
    HORIZON,
    N_BASIS,
    dct_basis,
)
from rlinf.envs.isaaclab.tasks.g1_piston_correlated_policy import (  # noqa: E402
    CorrelatedChunkNoise,
    correlated_noise_scales,
)

SCRATCH = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
           "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")


def _demo_chunks():
    out = []
    for fp in sorted(glob.glob(os.path.join(SCRATCH, "act_ep*.npy"))):
        A = np.load(fp)
        for c in range(len(A) // HORIZON):
            out.append(A[c * HORIZON:(c + 1) * HORIZON])
    return np.stack(out) if out else None


def test_effective_dimension_is_far_below_900():
    n = CorrelatedChunkNoise()
    assert n.effective_dim() == N_BASIS * ACTION_DIMS == 180
    assert HORIZON * ACTION_DIMS == 900


def test_per_step_std_is_calibrated_to_the_requested_value():
    """`std` must keep meaning "per-step exploration magnitude", so the change is a
    redistribution in frequency rather than a silent change of exploration scale."""
    n = CorrelatedChunkNoise()
    realised = n.per_step_std(0.20).numpy()
    assert abs(float(realised.mean()) - 0.20) < 0.02, realised.mean()


def test_empirical_std_matches_the_analytic_value():
    n = CorrelatedChunkNoise()
    g = torch.Generator().manual_seed(0)
    eps = n.sample_eps(4000, 0.20, generator=g)
    assert abs(float(eps.std()) - 0.20) < 0.01, float(eps.std())


def test_samples_are_smooth_in_time_unlike_iid_noise():
    """The load-bearing property: consecutive steps must be close, as in the data."""
    n = CorrelatedChunkNoise()
    g = torch.Generator().manual_seed(1)
    corr = n.sample_eps(2000, 0.20, generator=g)
    iid = torch.randn(2000, HORIZON, ACTION_DIMS, generator=g) * 0.20
    d_corr = (corr[:, 1:] - corr[:, :-1]).abs().mean().item()
    d_iid = (iid[:, 1:] - iid[:, :-1]).abs().mean().item()
    assert d_corr < 0.35 * d_iid, (d_corr, d_iid)


def test_energy_sits_in_the_components_the_demonstrations_use():
    """Real chunks carry 99.7% of energy in the first 6 DCT components; i.i.d. noise
    carries 20%. Correlated noise must live in the same subspace as the data."""
    n = CorrelatedChunkNoise()
    g = torch.Generator().manual_seed(2)
    eps = n.sample_eps(3000, 0.20, generator=g).numpy()
    B = dct_basis(HORIZON, HORIZON)
    C = np.einsum("hk,nhd->nkd", B, eps)
    e = (C ** 2).mean(axis=(0, 2))
    assert e[:N_BASIS].sum() / e.sum() > 0.99


def test_gradients_flow_through_the_expansion():
    """Reparameterised sampling: SAC differentiates the actor through the sample."""
    n = CorrelatedChunkNoise()
    z = torch.zeros(2, N_BASIS, ACTION_DIMS, requires_grad=True)
    n.expand(z, 0.20).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert (z.grad.abs() > 0).any()


def test_logprob_is_a_proper_gaussian_density():
    n = CorrelatedChunkNoise()
    z = torch.zeros(1, N_BASIS, ACTION_DIMS)
    lp = n.logprob_z(z)
    expected = -0.5 * math.log(2 * math.pi) * N_BASIS * ACTION_DIMS
    assert abs(float(lp) - expected) < 1e-4


def test_logprob_decreases_for_less_likely_samples():
    n = CorrelatedChunkNoise()
    near = torch.zeros(1, N_BASIS, ACTION_DIMS)
    far = torch.full((1, N_BASIS, ACTION_DIMS), 2.0)
    assert float(n.logprob_z(near)) > float(n.logprob_z(far))


def test_per_dimension_std_is_supported():
    """Exploration std is learned per action dimension, so the noise must accept it."""
    n = CorrelatedChunkNoise()
    std = torch.linspace(0.05, 0.4, ACTION_DIMS)
    g = torch.Generator().manual_seed(3)
    eps = n.sample_eps(3000, std, generator=g)
    realised = eps.std(dim=(0, 1))
    ratio = (realised / std).numpy()
    assert np.allclose(ratio, 1.0, atol=0.12), ratio


def test_shape_errors_are_rejected():
    n = CorrelatedChunkNoise()
    with pytest.raises(ValueError):
        n.expand(torch.zeros(1, N_BASIS + 1, ACTION_DIMS), 0.2)
    with pytest.raises(ValueError):
        correlated_noise_scales(HORIZON, 0)


@pytest.mark.skipif(not glob.glob(os.path.join(SCRATCH, "act_ep*.npy")),
                    reason="demonstration files not on this machine")
def test_correlated_noise_is_closer_to_demonstration_dynamics():
    """The measurement that motivates the whole module, asserted end to end.

    i.i.d. noise perturbs consecutive steps by ~209x the demonstrations' own per-step
    motion. Correlated noise at the same std must be substantially closer to the data's
    temporal statistics.
    """
    X = _demo_chunks()
    demo_step = float(np.abs(np.diff(X, axis=1)).mean())
    n = CorrelatedChunkNoise()
    g = torch.Generator().manual_seed(4)
    corr = n.sample_eps(2000, 0.20, generator=g)
    iid = torch.randn(2000, HORIZON, ACTION_DIMS, generator=g) * 0.20
    corr_step = float((corr[:, 1:] - corr[:, :-1]).abs().mean())
    iid_step = float((iid[:, 1:] - iid[:, :-1]).abs().mean())
    # Both still exceed the demonstrations (they are exploration, after all), but the
    # correlated version must close a large part of the gap.
    assert corr_step < iid_step
    assert (iid_step / demo_step) > 100          # the documented pathology
    assert (corr_step / demo_step) < 0.4 * (iid_step / demo_step)


def test_deterministic_path_is_unaffected():
    """Only exploration is shaped; a zero-noise sample must be exactly the mean."""
    n = CorrelatedChunkNoise()
    z = torch.zeros(3, N_BASIS, ACTION_DIMS)
    assert torch.allclose(n.expand(z, 0.20), torch.zeros(3, HORIZON, ACTION_DIMS))


# ------------------------------------------------------- trainer integration ----
def test_trainer_sampling_block_produces_valid_actions_and_gradients():
    """Replicates the trainer's correlated branch exactly, so a wiring mistake fails
    here rather than three hours into a run."""
    import importlib.util as ilu
    import sys

    spec = ilu.spec_from_file_location(
        "g1s_t", "rlinf/envs/isaaclab/tasks/g1_piston_rl_space.py")
    rlsp = ilu.module_from_spec(spec)
    sys.modules["g1s_t"] = rlsp
    spec.loader.exec_module(rlsp)
    act_mask = rlsp.build_active_mask()

    low, high = -2.2, 2.2
    B = 4
    corr = CorrelatedChunkNoise(horizon=HORIZON, dims=ACTION_DIMS, device="cpu")
    mean = torch.zeros(B, HORIZON, ACTION_DIMS, requires_grad=True)
    flat = mean.reshape(B * HORIZON, ACTION_DIMS)
    b, c, d = B, HORIZON, ACTION_DIMS
    std_d = torch.exp(torch.full((ACTION_DIMS,), -1.6))

    zc = torch.randn(b, corr.n_basis, d)
    eps = corr.expand(zc, std_d)
    pre = flat.reshape(b, c, d) + eps
    scale, shift = (high - low) / 2.0, (high + low) / 2.0
    a = (torch.tanh(pre) * scale + shift).reshape(b * c, d)
    jac = (torch.log(1 - torch.tanh(pre).pow(2) + 1e-7) * act_mask).sum(dim=-1)
    lp = ((CorrelatedChunkNoise.logprob_z(zc).unsqueeze(-1) / c) - jac).reshape(b * c)

    assert a.shape == (B * HORIZON, ACTION_DIMS)
    assert lp.shape == (B * HORIZON,)
    assert bool((a >= low - 1e-5).all() and (a <= high + 1e-5).all())
    assert torch.isfinite(lp).all()
    a.sum().backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert (mean.grad.abs() > 0).any()


def test_logprob_scale_is_compatible_with_the_entropy_target():
    """SAC's temperature is tuned against TARGET_ENTROPY = -8.4 per control action.
    The correlated path must return a log-prob on that same scale, or alpha would chase
    a target it can never reach -- the failure documented in the earlier SAC pilots."""
    corr = CorrelatedChunkNoise()
    b, c = 8, HORIZON
    zc = torch.randn(b, corr.n_basis, ACTION_DIMS)
    pre = corr.expand(zc, torch.exp(torch.full((ACTION_DIMS,), -1.6)))
    jac = torch.log(1 - torch.tanh(pre).pow(2) + 1e-7).sum(dim=-1)
    lp = (CorrelatedChunkNoise.logprob_z(zc).unsqueeze(-1) / c) - jac
    assert -20.0 < float(lp.mean()) < 0.0, float(lp.mean())


def test_trainer_exposes_the_flag_and_records_it():
    src = open("tools/g1_piston/train_sac.py").read()
    assert 'CORRELATED_NOISE = os.environ.get("CORRELATED_NOISE", "0") == "1"' in src
    assert '"correlated_noise": bool(CORRELATED_NOISE),' in src
    assert "corr_noise.expand(zc, std_d)" in src


def test_deterministic_branch_precedes_the_correlated_branch():
    """Evaluation must stay bit-identical: `deterministic` is checked first."""
    src = open("tools/g1_piston/train_sac.py").read()
    block = src.split("def sample_action(")[1].split("def masked_logprob")[0]
    assert block.index("if deterministic:") < block.index("elif corr_noise is not None:")


def test_module_loads_by_file_path_without_the_rlinf_package():
    """The training and evaluation tools load task modules BY FILE PATH, where `rlinf`
    is not importable. Run 2 died instantly on this: the module used a package import
    and every other g1_piston task module is self-contained for exactly that reason.

    Loading with importlib from a directory outside the repo reproduces the tool's
    environment closely enough to catch a regression.
    """
    import importlib.util as ilu
    import os
    import subprocess
    import sys

    path = os.path.abspath(
        "rlinf/envs/isaaclab/tasks/g1_piston_correlated_policy.py")
    # Run in a subprocess with cwd outside the repo so `rlinf` cannot be found.
    code = (
        "import sys, importlib.util as ilu\n"
        f"sp = ilu.spec_from_file_location('g1cn', {path!r})\n"
        "mo = ilu.module_from_spec(sp); sys.modules['g1cn'] = mo\n"
        "sp.loader.exec_module(mo)\n"
        "print(mo.CorrelatedChunkNoise().effective_dim())\n")
    out = subprocess.run([sys.executable, "-c", code], cwd="/tmp",
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-600:]
    assert out.stdout.strip() == "180", out.stdout


def test_no_other_task_module_gained_a_package_import():
    """Keep the convention: these modules are loaded by path, so a bare `from rlinf...`
    import in any of them is a latent instant-crash in the tools."""
    import glob
    import os

    offenders = []
    for fp in glob.glob("rlinf/envs/isaaclab/tasks/g1_piston_*.py"):
        src = open(fp).read()
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith(("from rlinf", "import rlinf")):
                # Guarded imports inside a try/except are fine.
                if "    " not in line[:4]:
                    offenders.append((os.path.basename(fp), stripped))
    assert not offenders, offenders


def test_the_entropy_target_is_reachable_under_correlated_noise():
    """The earlier SAC pilots collapsed because TARGET_ENTROPY sat OUTSIDE the
    achievable log-prob range: alpha then falls monotonically, the entropy regulariser
    dies, and the actor drifts unregularised
    (docs/contracts/g1_piston_sac_pilot_v1_collapse.json).

    Changing the noise distribution changes that range, so it must be rechecked. Under
    correlated noise the target -8.4 is reached near std 0.05-0.1, i.e. inside the
    operating range, so alpha has a genuine equilibrium.
    """
    import importlib.util as ilu
    import sys

    spec = ilu.spec_from_file_location(
        "g1s_e", "rlinf/envs/isaaclab/tasks/g1_piston_rl_space.py")
    rlsp = ilu.module_from_spec(spec)
    sys.modules["g1s_e"] = rlsp
    spec.loader.exec_module(rlsp)
    target = rlsp.default_target_entropy()
    act_mask = rlsp.build_active_mask()

    corr = CorrelatedChunkNoise()

    def logp(std):
        zc = torch.randn(400, corr.n_basis, ACTION_DIMS)
        pre = corr.expand(zc, torch.full((ACTION_DIMS,), float(std)))
        jac = (torch.log(1 - torch.tanh(pre).pow(2) + 1e-7) * act_mask).sum(-1)
        return float(((CorrelatedChunkNoise.logprob_z(zc).unsqueeze(-1) / HORIZON)
                      - jac).mean())

    tight, loose = logp(0.05), logp(0.4)
    # The target must lie inside the range the policy can actually produce.
    assert tight <= target <= loose, (tight, target, loose)
