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


# --------------------------------------------------------- squash parity ----
"""The head's output is PRE-SQUASH, and every consumer must treat it that way.

Second defect, same shape as the first: train_bc.py regressed the head's raw output
onto the normalised demonstration action, while train_sac.py, eval_checkpoint.py and
render_rollouts.py all execute tanh(head_out) * (HIGH-LOW)/2 + (HIGH+LOW)/2. Measured on
demonstration ep046, the feature-corrected head matched the demonstration to 0.376 deg
before the squash and 12.393 deg after it -- a 33x degradation, and 0.00 on all 25
conditions. See docs/contracts/g1_piston_bc_squash_mismatch.json.
"""


def test_training_tools_regress_through_the_deployment_squash():
    """train_bc.py must apply the squash before computing its loss, or it optimises a
    function the simulator never runs."""
    src = _src("tools/g1_piston/train_bc.py")
    assert "def squash(" in src, (
        "train_bc.py does not define the deployment squash; its loss is then computed "
        "on the head's raw output, which is not what the simulator executes."
    )
    assert "squash(head_mean(" in src, (
        "train_bc.py computes its loss on an unsquashed prediction."
    )


def test_squash_constants_agree_across_tools():
    """The bounds are the contract between the head and the simulator; a tool using
    different ones silently rescales every action."""
    for path in ("tools/g1_piston/train_bc.py", "tools/g1_piston/train_sac.py",
                 "tools/g1_piston/eval_checkpoint.py"):
        src = _src(path)
        assert "ACTION_LOW, ACTION_HIGH = -2.2, 2.2" in src, path


def test_demo_targets_are_reachable_through_the_squash():
    """tanh saturates, so a target beyond the bounds is unreachable and its gradient
    vanishes. Measured on the 16 training episodes: max |x| = 1.25 against a bound of
    2.2, nothing beyond 2.0. Pinned so a future renormalisation that pushes targets into
    saturation fails here rather than as a mysteriously bad policy."""
    import json
    with open("docs/contracts/g1_piston_bc_squash_mismatch.json") as f:
        c = json.load(f)
    r = c["target_reachability_checked_before_retraining"]["measured_on_the_16_training_episodes"]
    assert r["fraction_at_or_beyond_2.2"] == 0.0
    assert r["fraction_beyond_2.0"] == 0.0
    assert abs(r["max"]) < 2.0 and abs(r["min"]) < 2.0


def test_eval_has_a_policy_free_control_arm():
    """A zero from the evaluation path can only be attributed to the policy if the path
    itself is known to be capable. REPLAY_DEMO provides that control."""
    src = _src("tools/g1_piston/eval_checkpoint.py")
    assert "REPLAY_DEMO" in src
    assert "physical=True" in src, (
        "the replay arm must pass demonstration actions as PHYSICAL; denormalising them "
        "would corrupt them by the whole transform."
    )
