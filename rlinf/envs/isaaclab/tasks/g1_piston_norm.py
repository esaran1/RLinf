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

"""q99 action normalization for the G1 long-horizon SFT checkpoint.

A faithful re-implementation of StarVLA's
``starVLA/dataloader/gr00t_lerobot/transform/state_action.py::Normalizer`` for
``mode="q99"``, restricted to what the deployed policy needs. Kept as a small,
self-contained module so RLinf can de-normalize without importing the training
dataloader stack, and so the behaviour can be tested against StarVLA directly.

The subtlety that makes a naive affine transform wrong
------------------------------------------------------
For dims where ``q01 == q99`` (no variation in the dataset), StarVLA's ``forward``
does **not** normalize: it passes the raw physical value through. For this checkpoint
that is 10 of 30 dims -- the entire left hand (14-19, constant because it grips the
tube), base height (26), and the 3 navigation velocities (27-29).

So the model's output on those dims is **already physical**, not in [-1, 1].

``Normalizer.inverse`` has no such mask -- it applies ``(x+1)/2*(q99-q01)+q01`` to
every dim. That is nonetheless correct here, because ``q99-q01 == 0`` collapses the
expression to ``q01``, which *is* the constant physical value. The consequence worth
knowing: on degenerate dims the inverse **ignores the model's output entirely** and
emits the dataset constant. That is intended, and is what ``denormalize`` reproduces.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

#: Clamp applied by StarVLA's q99 forward. Wider than [-1, 1] so in-distribution
#: values are untouched while true outliers stay bounded.
Q99_CLAMP = (-2.2, 2.2)

#: Normalization key for this checkpoint's dataset statistics.
DEFAULT_UNNORM_KEY = "new_embodiment"


class Q99ActionNormalizer:
    """q99 normalize/de-normalize matching StarVLA's ``Normalizer`` bit for bit."""

    def __init__(self, q01, q99, *, clamp=Q99_CLAMP):
        self.q01 = torch.as_tensor(q01, dtype=torch.float32)
        self.q99 = torch.as_tensor(q99, dtype=torch.float32)
        if self.q01.shape != self.q99.shape:
            raise ValueError(
                f"q01 {tuple(self.q01.shape)} and q99 {tuple(self.q99.shape)} "
                "must have the same shape"
            )
        self.clamp = clamp
        #: True where the dim actually varies and is therefore normalized.
        self.mask = self.q01 != self.q99

    @classmethod
    def from_dataset_statistics(
        cls, path, *, unnorm_key: str = DEFAULT_UNNORM_KEY, field: str = "action"
    ):
        """Load q01/q99 from a checkpoint's ``dataset_statistics.json``."""
        stats = json.loads(Path(path).read_text())
        if unnorm_key not in stats:
            raise KeyError(
                f"unnorm_key {unnorm_key!r} not in dataset statistics; "
                f"available: {list(stats)}"
            )
        entry = stats[unnorm_key][field]
        return cls(entry["q01"], entry["q99"])

    @property
    def degenerate_dims(self) -> list[int]:
        """Dims with ``q01 == q99``: passed through raw by ``normalize``."""
        return (~self.mask).nonzero(as_tuple=True)[0].tolist()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Physical -> normalized, matching ``Normalizer.forward`` for q99."""
        q01 = self.q01.to(device=x.device, dtype=x.dtype)
        q99 = self.q99.to(device=x.device, dtype=x.dtype)
        mask = self.mask.to(x.device)

        out = torch.zeros_like(x)
        out[..., mask] = (
            2 * ((x[..., mask] - q01[mask]) / (q99[mask] - q01[mask])) - 1
        )
        # Degenerate dims pass through unnormalized -- they carry physical units.
        out[..., ~mask] = x[..., ~mask]
        return torch.clamp(out, self.clamp[0], self.clamp[1])

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized -> physical, matching ``Normalizer.inverse`` for q99.

        Deliberately unmasked, exactly as upstream: on degenerate dims ``q99-q01`` is
        zero, so this returns the dataset constant regardless of ``x``.
        """
        q01 = self.q01.to(device=x.device, dtype=x.dtype)
        q99 = self.q99.to(device=x.device, dtype=x.dtype)
        return (x + 1) / 2 * (q99 - q01) + q01
