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

"""Every tool must feed the action head the SAME tensor.

This test exists because behaviour cloning produced a checkpoint that fit the
demonstrations open-loop to 1.15 degrees and then scored 0.00 on all 25 frozen
conditions. The cause was not the observation space, as first hypothesised, but a
feature-extraction mismatch:

    train_bc.py        lh[:, -chunk_len:, :]                       <- WRONG
    train_sac.py       _gather_action_token_embeddings(...)
    eval_checkpoint.py _gather_action_token_embeddings(...)
    render_rollouts.py _gather_action_token_embeddings(...)

The prompt ends with ``<action>.`` AFTER the action tokens, so the tokens occupy
positions 110-139 of 148 while the slice takes 118-147. The tensors differ by 49%
in relative magnitude, so the head was trained on one input distribution and scored on
another. Nothing errored; the result was simply wrong.

A training tool and an evaluation tool that disagree about the model's input produce a
silently invalid result, so the agreement is pinned here rather than left to review.
See ``docs/contracts/g1_piston_bc_feature_mismatch.json``.
"""

import re

TOOLS = (
    "tools/g1_piston/train_bc.py",
    "tools/g1_piston/train_sac.py",
    "tools/g1_piston/eval_checkpoint.py",
    "tools/g1_piston/render_rollouts.py",
)


def _src(path):
    with open(path) as f:
        return f.read()


def test_every_tool_gathers_action_token_embeddings():
    """The load-bearing assertion: one shared way of building the head's input."""
    for path in TOOLS:
        assert "_gather_action_token_embeddings" in _src(path), (
            f"{path} does not gather action-token embeddings; it will feed the head a "
            "different tensor than the other tools and its results will not be "
            "comparable to theirs."
        )


def test_no_tool_slices_the_trailing_hidden_states():
    """The specific defect: taking the last chunk_len positions by slicing.

    Matches the slice on the hidden-state variable regardless of its local name
    (``hs`` in train_bc.py, ``lh``/``last_hidden`` elsewhere), so renaming the
    variable cannot reintroduce the bug unnoticed.
    """
    pattern = re.compile(
        r"^(?!\s*#).*?\w+\[\s*:\s*,\s*-\s*[\w.]*chunk_len\s*:\s*,\s*:\s*\]",
        re.MULTILINE)
    for path in TOOLS:
        hits = pattern.findall(_src(path))
        assert not hits, (
            f"{path} slices the trailing hidden states {hits}. The action tokens are "
            "NOT the final positions of the prompt, which ends with '<action>.' after "
            "them. Use _gather_action_token_embeddings instead."
        )


def test_action_token_id_is_used_to_locate_the_tokens():
    """Gathering must be keyed on the action token id, not a positional assumption."""
    for path in TOOLS:
        assert "action_token_id" in _src(path), path


def test_train_bc_records_why_the_slice_was_wrong():
    """The correction stays explained at the call site, so the next person editing this
    function does not 'simplify' it back into the bug."""
    src = _src("tools/g1_piston/train_bc.py")
    assert "g1_piston_bc_feature_mismatch" in src
