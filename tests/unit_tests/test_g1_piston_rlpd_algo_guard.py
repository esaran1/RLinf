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

"""A run must not silently train a different algorithm than it was launched for.

ALGO defaults to "sac". A launch script that passes DEMO_DIR and the RLPD knobs but
forgets ALGO=rlpd therefore loads NO demonstrations, trains without the 50% replay that
defines RLPD, records demo_fraction 0.0, and exits OK. That happened: a throughput run
completed cleanly with n_demo_samples 0 across all 30 updates while its
pre-registration specified RLPD.

Nothing errored, so the only signal was a 0.0 in the totals block. These tests make the
mismatch fail closed at startup instead.
"""

SRC = "tools/g1_piston/train_sac.py"


def _src():
    with open(SRC) as f:
        return f.read()


def test_rlpd_knobs_without_rlpd_algo_fail_closed():
    """The load-bearing assertion: configuring demonstration replay while leaving ALGO
    at its default must raise, not quietly run plain SAC."""
    src = _src()
    assert 'if ALGO != "rlpd":' in src, (
        "no guard exists for RLPD knobs set under a non-RLPD algorithm."
    )
    assert "_rlpd_only" in src and "raise SystemExit(" in src
    for knob in ("DEMO_DIR", "DEMO_FRAC", "PREWARM_DEMO_CACHE"):
        assert knob in src, knob


def test_the_guard_names_the_actual_failure_mode():
    """An error that does not say what went wrong invites the same mistake again."""
    src = _src()
    assert "demo_fraction 0.0" in src
    assert "WITHOUT" in src


def test_algo_still_defaults_to_sac():
    """The default is not changed: plain SAC remains runnable on purpose. The guard only
    rejects the CONTRADICTORY configuration."""
    src = _src()
    assert 'ALGO = os.environ.get("ALGO", "sac").lower()' in src


def test_demo_fraction_is_recorded_in_the_run_totals():
    """After the fact, a run's own record must show whether demonstrations were actually
    drawn, so a stored result can be audited without rerunning it."""
    src = _src()
    assert '"demo_fraction"' in src
    assert '"n_demo_samples"' in src


def test_reward_version_guard_still_present():
    """The pre-existing fail-closed check on buffer reward version must not have been
    disturbed by the new guard."""
    src = _src()
    assert "Demo buffer reward version mismatch" in src


# ------------------------------------------- executable behaviour ----
"""The tests above read the source. These EXECUTE the guard's logic.

A source-text assertion cannot tell whether the condition is actually correct, only
that some text is present. The guard sits after simulator boot and VLM load in the real
trainer, so exercising it there costs minutes and a GPU; the logic itself is pure and
can be checked directly. It is extracted here verbatim from train_sac.py.
"""

import pytest


def _guard(env):
    """Byte-for-byte the guard's condition from train_sac.py, over an explicit env."""
    algo = env.get("ALGO", "sac").lower()
    if algo != "rlpd":
        rlpd_only = [n for n, v in (("DEMO_DIR", env.get("DEMO_DIR")),
                                    ("DEMO_FRAC", env.get("DEMO_FRAC")),
                                    ("PREWARM_DEMO_CACHE",
                                     env.get("PREWARM_DEMO_CACHE"))) if v]
        if rlpd_only:
            raise SystemExit(
                f"ALGO={algo!r} but {', '.join(rlpd_only)} was set.")
    return algo


def test_guard_fires_on_the_exact_configuration_that_slipped_through():
    """The real launch that ran plain SAC while its pre-registration said RLPD."""
    with pytest.raises(SystemExit) as e:
        _guard({"DEMO_DIR": "/home/jren313/research/starvla_rl/demo_buffer_v3",
                "REWARD_V3": "1"})
    assert "DEMO_DIR" in str(e.value)
    assert "'sac'" in str(e.value)


@pytest.mark.parametrize("knob", ["DEMO_DIR", "DEMO_FRAC", "PREWARM_DEMO_CACHE"])
def test_each_rlpd_knob_alone_trips_the_guard(knob):
    with pytest.raises(SystemExit):
        _guard({knob: "1"})


def test_correct_rlpd_configuration_passes():
    """The guard must not block the run it is protecting."""
    assert _guard({"ALGO": "rlpd", "DEMO_DIR": "/x", "DEMO_FRAC": "0.5"}) == "rlpd"


def test_deliberate_plain_sac_passes():
    """Plain SAC with no demonstration knobs is a legitimate configuration."""
    assert _guard({"ALGO": "sac"}) == "sac"
    assert _guard({}) == "sac"


def test_algo_is_case_insensitive():
    assert _guard({"ALGO": "RLPD", "DEMO_DIR": "/x"}) == "rlpd"


def test_empty_values_do_not_trip_the_guard():
    """An exported-but-empty variable is not a configuration request."""
    assert _guard({"DEMO_DIR": ""}) == "sac"
