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
