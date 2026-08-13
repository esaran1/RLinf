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

"""Behavioural tests for the 50 Hz -> 100 Hz zero-order hold in chunk_step.

Driven by a recording stub env, so the hold semantics are checked without Isaac Sim:
how many env steps run, which action each receives, how reward accumulates, and how
terminations that fire on the first vs second held step are handled.
"""

import pytest
import torch

from rlinf.envs.isaaclab.tasks.g1_piston import IsaaclabG1PistonEnv


class _RecordingEnv(IsaaclabG1PistonEnv):
    """Bypasses __init__/Isaac Sim; records every step() the hold performs."""

    def __init__(self, *, hold=2, num_envs=1, terminate_at=None, reward=1.0):
        self.action_hold_steps = hold
        self.num_envs = num_envs
        self.device = "cpu"
        self.auto_reset = False
        self.ignore_terminations = False
        self.stepped_actions = []
        self._terminate_at = terminate_at
        self._reward = reward
        self._n = 0

    def step(self, actions, auto_reset=True):
        self.stepped_actions.append(actions.clone())
        self._n += 1
        reward = torch.full((self.num_envs,), self._reward)
        term = torch.zeros(self.num_envs, dtype=torch.bool)
        trunc = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._terminate_at is not None and self._n == self._terminate_at:
            term[:] = True
        obs = {"step_index": self._n}
        return obs, reward, term, trunc, {}


def _chunk(n_actions=30, action_dim=53):
    """[num_envs, chunk, action_dim] with a distinct value per chunk index."""
    a = torch.zeros(1, n_actions, action_dim)
    for i in range(n_actions):
        a[0, i, :] = float(i + 1)
    return a


def test_30_policy_actions_produce_60_env_steps():
    env = _RecordingEnv(hold=2)
    env.chunk_step(_chunk(30))
    assert len(env.stepped_actions) == 60


def test_each_action_is_applied_exactly_twice_in_order():
    """Zero-order hold: a[t], a[t], a[t+1], a[t+1], ... -- never reordered or skipped."""
    env = _RecordingEnv(hold=2)
    env.chunk_step(_chunk(5))

    values = [float(a[0, 0]) for a in env.stepped_actions]
    assert values == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_returned_chunk_length_stays_at_policy_horizon():
    """Bookkeeping stays per-policy-action, so downstream chunk accounting holds."""
    env = _RecordingEnv(hold=2)
    _, rewards, terms, truncs, infos = env.chunk_step(_chunk(30))

    assert rewards.shape == (1, 30), "reward must be per policy action, not per env step"
    assert terms.shape == (1, 30)
    assert truncs.shape == (1, 30)
    assert len(infos) == 30


def test_reward_accumulates_across_held_steps():
    """A chunk entry is the true environment return over the held interval."""
    env = _RecordingEnv(hold=2, reward=1.0)
    _, rewards, _, _, _ = env.chunk_step(_chunk(4))
    assert torch.all(rewards == 2.0), "expected 1.0 + 1.0 per held action"


def test_policy_rate_is_preserved():
    """60 env steps at 100 Hz == 30 policy actions at 50 Hz == 0.6 s."""
    env = _RecordingEnv(hold=2)
    env.chunk_step(_chunk(30))
    sim_dt, decimation = 0.005, 2
    env_step_dt = sim_dt * decimation
    assert env_step_dt == pytest.approx(0.01)  # 100 Hz
    assert len(env.stepped_actions) * env_step_dt == pytest.approx(0.6)
    assert 30 * (1 / 50) == pytest.approx(0.6)


def test_termination_on_first_held_step_survives():
    """A done on the first of the two held steps must not be lost by the second."""
    env = _RecordingEnv(hold=2, terminate_at=1)
    _, _, terms, _, _ = env.chunk_step(_chunk(3))
    assert bool(terms[0, 0]), "termination on the first held step was dropped"


def test_termination_on_second_held_step_recorded():
    env = _RecordingEnv(hold=2, terminate_at=2)
    _, _, terms, _, _ = env.chunk_step(_chunk(3))
    assert bool(terms[0, 0])


def test_hold_of_one_matches_base_behaviour():
    """hold=1 delegates to the base implementation: one env step per action."""
    env = _RecordingEnv(hold=1)
    env.chunk_step(_chunk(10))
    assert len(env.stepped_actions) == 10


def test_hold_does_not_change_action_horizon():
    """The hold must not inflate the number of actions the policy is asked for."""
    env = _RecordingEnv(hold=2)
    chunk = _chunk(30)
    env.chunk_step(chunk)
    assert chunk.shape[1] == 30, "action_horizon must remain 30"
    unique = {float(a[0, 0]) for a in env.stepped_actions}
    assert len(unique) == 30, "expected 30 distinct actions, each held twice"


def test_invalid_hold_rejected_by_constructor(monkeypatch):
    """hold < 1 must fail at construction, not silently execute zero env steps."""
    from omegaconf import OmegaConf

    from rlinf.envs.isaaclab import tasks as _tasks  # noqa: F401

    cfg = OmegaConf.create({"init_params": {"action_hold_steps": 0}})

    # Skip the IsaacLab base __init__ (needs a live simulator); exercise only the
    # hold-validation branch that follows it.
    monkeypatch.setattr(
        IsaaclabG1PistonEnv.__bases__[0],
        "__init__",
        lambda self, *a, **k: None,
        raising=False,
    )

    with pytest.raises(ValueError, match="action_hold_steps must be >= 1"):
        IsaaclabG1PistonEnv(
            cfg=cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
