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

"""Throughput guards: the two costs that consumed entire training runs.

Run 1 completed 150 gradient updates in 3 hours; run 2, after raising UTD to 8 on the
strength of a 4.66 ms critic-MLP timing, completed 125 updates and collected 16x LESS
data. The timing had excluded the demonstration VLM encodes, which a measurement put at
107.5 ms each -- and RLPD draws demo transitions on every update, so at UTD 8 the cache
misses dominate the loop until the cache fills.
"""

import os


SRC = open("tools/g1_piston/train_sac.py").read()


def test_demo_cache_is_prewarmed_before_training():
    """The fix: encode all demonstration transitions once, before the update loop,
    instead of paying misses inside it."""
    assert "PREWARM_DEMO_CACHE" in SRC
    assert "demo_cache_prewarm_s" in SRC
    # It must run BEFORE the training loop, not inside it.
    assert SRC.index("PREWARM_DEMO_CACHE") < SRC.index("while env_steps < MAX_ENV_STEPS")


def test_prewarm_populates_the_same_cache_the_update_loop_reads():
    """A separate cache would leave the misses in place while looking fixed."""
    prewarm = SRC.split("if demo and PREWARM_DEMO_CACHE:")[1].split(
        "if RUN_INIT_EVAL:")[0]
    assert "demo_feat_cache[_idx] = store(" in prewarm
    assert "demo_feat_cache" in SRC.split("def demo_batch")[1][:800]


def test_prewarm_respects_the_critic_state_pairing():
    """Pre-warmed entries must carry the same privileged state the lazy path attaches,
    or demo transitions would silently lose their critic conditioning."""
    prewarm = SRC.split("if demo and PREWARM_DEMO_CACHE:")[1].split(
        "if RUN_INIT_EVAL:")[0]
    assert "demo_state.get(_idx)" in prewarm
    assert "demo_next_state.get(_idx)" in prewarm
    assert "cs=_dcs, ncs=_dncs" in prewarm


def test_init_evaluation_can_be_skipped():
    """The step-0 evaluations cost 2 x 25 x 23 = 1150 chunk rollouts before a single
    gradient step. When the warm-start checkpoint has already been scored under the same
    predicate, that is pure duplication and the dominant cost of a short run."""
    assert "RUN_INIT_EVAL" in SRC
    block = SRC.split("if RUN_INIT_EVAL:")[1][:400]
    assert 'evaluate("init"' in block
    assert 'evaluate("init_stochastic"' in block


def test_both_flags_are_recorded_in_the_run_config():
    """A run's own record must say whether it skipped these, or later comparisons
    between runs are not interpretable."""
    assert '"prewarm_demo_cache": bool(PREWARM_DEMO_CACHE),' in SRC
    assert '"run_init_eval": bool(RUN_INIT_EVAL),' in SRC


def test_defaults_preserve_existing_behaviour():
    """Both default to the previous behaviour except the prewarm, which is strictly a
    reordering of work that would happen anyway."""
    assert 'os.environ.get("PREWARM_DEMO_CACHE", "1")' in SRC
    assert 'os.environ.get("RUN_INIT_EVAL", "1")' in SRC


def test_measured_costs_are_documented_at_the_call_site():
    """The projection failed because a component timing was presented as a system cost.
    The measured numbers must live next to the code that depends on them."""
    assert "107.5 ms" in SRC
    assert "UTD 8" in SRC


def test_first_periodic_eval_no_longer_fires_at_step_zero_by_default():
    """`next_eval = 0` made the first training episode trip an extra 25-condition sweep
    (575 chunk rollouts, ~24 min). That was deliberately retained while the original
    SAC and RLPD arms were running -- matched evaluation cadence mattered more than the
    time -- and the note in the code said to change it once they finished. They have.

    With the prewarm and skipped init evals in place, a 3-hour run is dominated by
    simulator time, so this sweep is now a large fraction of the budget.
    """
    src = open("tools/g1_piston/train_sac.py").read()
    assert "next_eval = 0 if FIRST_EVAL_AT_ZERO else EVAL_EVERY" in src
    assert 'os.environ.get("FIRST_EVAL_AT_ZERO", "0")' in src


def test_the_original_cadence_is_still_reproducible():
    """Anyone reproducing the original arms must be able to restore the old behaviour."""
    src = open("tools/g1_piston/train_sac.py").read()
    assert "FIRST_EVAL_AT_ZERO=1 restores" in src
    assert '"first_eval_at_zero": bool(FIRST_EVAL_AT_ZERO),' in src


def test_run_records_a_rolling_critic_health_summary():
    """critic_loss is spiky on a small replay buffer, so a single large value says
    little. The run must record the MEDIAN trend and the Q trend, which together are
    what separate high-variance TD learning from runaway overestimation -- the
    pre-registered primary risk for the high-UTD configuration.
    """
    src = open("tools/g1_piston/train_sac.py").read()
    assert '"critic_health"' in src
    for key in ("median_first_half", "median_second_half", "median_ratio",
                "q_median_second_half", "max_seen"):
        assert key in src, key


def test_critic_health_is_computed_from_the_medians_not_the_extremes():
    """Guards the reasoning: using max would flag every ordinary spike as divergence."""
    src = open("tools/g1_piston/train_sac.py").read()
    block = src.split('res["critic_health"]')[0][-1200:]
    assert "sorted(" in block and "len(_cl) // 2" in block
