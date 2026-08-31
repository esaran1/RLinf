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

"""The demo buffer stores PHYSICAL actions; a rebuilt buffer must match.

Found the expensive way: a probe assumed the shipped buffer held NORMALIZED chunks and
denormalized them before replay. Every rollout was corrupted (deviating up to 0.86 rad
from the recorded demonstrations) and the whole 22-episode measurement came back as
"zero stages everywhere" -- which reads exactly like a real finding.

``build_demo_buffer.py`` had the same latent bug on the write side: it normalized chunks
on the way in. A buffer rebuilt that way would have fed RLPD corrupted actions on half of
every batch for an entire training run, with no error and no obvious symptom.
"""

import glob
import os

import numpy as np
import pytest

SCRATCH = ("/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/"
           "c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad")
BUFFER = "/home/jren313/research/starvla_rl/demo_buffer"
STATS = ("/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/"
         "dataset_statistics.json")


@pytest.mark.skipif(not os.path.exists(f"{BUFFER}/ep000.npz")
                    or not os.path.exists(f"{SCRATCH}/act_ep0.npy"),
                    reason="demo buffer or recorded demonstrations not on this machine")
def test_buffer_actions_are_physical_not_normalized():
    """The shipped buffer's chunks must equal the recorded physical actions exactly."""
    A = np.load(f"{SCRATCH}/act_ep0.npy")                 # (T, 30) physical
    B = np.load(f"{BUFFER}/ep000.npz")["actions"]          # (chunks, H, 30)
    flat = B.reshape(-1, B.shape[-1])
    n = min(len(A), len(flat))
    assert np.abs(A[:n] - flat[:n]).max() < 1e-5, "buffer actions are not physical"


@pytest.mark.skipif(not os.path.exists(f"{BUFFER}/ep000.npz")
                    or not os.path.exists(STATS),
                    reason="demo buffer or dataset statistics not on this machine")
def test_denormalizing_the_buffer_would_corrupt_it():
    """Guards the specific mistake: treating stored chunks as normalized."""
    torch = pytest.importorskip("torch")
    import importlib.util as ilu
    import sys

    spec = ilu.spec_from_file_location(
        "g1n", "rlinf/envs/isaaclab/tasks/g1_piston_norm.py")
    mod = ilu.module_from_spec(spec)
    sys.modules["g1n"] = mod
    spec.loader.exec_module(mod)
    nrm = mod.Q99ActionNormalizer.from_dataset_statistics(STATS)

    A = np.load(f"{SCRATCH}/act_ep0.npy")
    B = np.load(f"{BUFFER}/ep000.npz")["actions"].reshape(-1, 30)
    wrong = nrm.denormalize(torch.as_tensor(B, dtype=torch.float32)).numpy()
    n = min(len(A), len(wrong))
    # The corruption must be large enough that this test is meaningful.
    assert np.abs(A[:n] - wrong[:n]).max() > 0.1


def test_builder_writes_physical_actions():
    """Pin the write side: a rebuilt buffer must not normalize on the way in."""
    src = open("tools/g1_piston/build_demo_buffer.py").read()
    assert "nrm.normalize(" not in src, "builder still normalizes stored actions"
    assert "PHYSICAL" in src


@pytest.mark.skipif(not glob.glob(f"{BUFFER}/*.npz"),
                    reason="demo buffer not on this machine")
def test_every_buffer_episode_is_in_the_expected_range():
    """Physical joint commands live in roughly [-1.3, 1.8] for this embodiment;
    normalized ones would sit near [-1, 1] with a different distribution."""
    for fp in sorted(glob.glob(f"{BUFFER}/*.npz"))[:5]:
        a = np.load(fp)["actions"]
        assert a.min() > -3.0 and a.max() < 3.0
        # Physical hand commands reach the 1.7 rad finger limit; normalized would not.
        assert a.max() > 1.0, os.path.basename(fp)


def test_builder_and_trainer_agree_on_episode_progress_scale():
    """episode_progress is chunk/max_chunks. The builder and the trainer must use the
    SAME max_chunks, or demonstration transitions carry a different progress scale than
    online ones and the critic sees two conventions for one feature.

    The builder originally passed the chunk LENGTH (H=30) where the episode length
    (EP_CHUNKS=23) was meant.
    """
    b = open("tools/g1_piston/build_demo_buffer.py").read()
    t = open("tools/g1_piston/train_sac.py").read()
    assert "CriticStateBuilder(sc, max_chunks=EP_CHUNKS)" in b
    assert "CriticStateBuilder(sc, max_chunks=EP_CHUNKS)" in t
    assert 'EP_CHUNKS = int(os.environ.get("EP_CHUNKS", "23"))' in b
    assert 'EP_CHUNKS = int(os.environ.get("EP_CHUNKS", "23"))' in t
