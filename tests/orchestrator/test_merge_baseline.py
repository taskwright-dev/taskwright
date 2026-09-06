"""The merge measures its own baseline: no new red.

Spec: ``docs/merge-word-on-host-spec-2026-09-06.md`` (ai-transition), rule 6 —
``guardkit autobuild merge`` runs the resolved test command on the target
branch BEFORE merging, and charges the merged result only for failures the
target branch did not already have.

Everything here is driven on real code paths: a real git repository in
``tmp_path``, a real pytest layout with a genuinely failing test, and the
repository's own ``.guardkit/config.yaml`` declaring a test command that runs
this test session's own interpreter. Nothing under test is mocked — only the
feature-file validation is injected, because a fixture repo has no feature
YAML and the validation is a different check with its own tests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from guardkit.orchestrator.merge_executor import (
    OUTCOME_CONFLICT,
    OUTCOME_MERGED,
    execute_merge,
    measure_pre_merge_baseline,
    merge_verdict_from_run,
    passed_count_from_output,
)

PRE_EXISTING_RED = "tests/test_already_red.py::test_already_red"
NEW_RED = "tests/test_new_red.py::test_new_red"


# ---------------------------------------------------------------------------
# A real repository with a real (partly red) test suite
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def declared_command(python: str = sys.executable) -> str:
    """The repository's declared test command, run by a named interpreter."""
    return f'"{python}" -m pytest -q -p no:cacheprovider tests/'


def _write_toolchain(repo: Path, command: str) -> None:
    config = repo / ".guardkit" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "toolchain:\n" f'  test: {json.dumps(command)}\n', encoding="utf-8"
    )


def _base_repo(tmp_path: Path, command: str) -> Path:
    """main: two passing tests and one that was already failing."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    (repo / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n.pytest_cache/\n", encoding="utf-8"
    )
    # An ini file of its own so the fixture suite never inherits guardkit's.
    (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_green.py").write_text(
        "def test_green_one():\n    assert True\n\n\n"
        "def test_green_two():\n    assert True\n",
        encoding="utf-8",
    )
    (tests / "test_already_red.py").write_text(
        "def test_already_red():\n"
        "    assert False, 'red on main before anyone merged anything'\n",
        encoding="utf-8",
    )
    _write_toolchain(repo, command)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "the repository as it already stands")
    return repo


def _branch_adding(repo: Path, feature_id: str, filename: str, body: str) -> None:
    _git(repo, "checkout", "-q", "-b", f"autobuild/{feature_id}")
    (repo / "tests" / filename).write_text(body, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"{feature_id} work")
    _git(repo, "checkout", "-q", "main")


@pytest.fixture
def repo_with_clean_branch(tmp_path: Path) -> Path:
    repo = _base_repo(tmp_path, declared_command())
    _branch_adding(
        repo,
        "FEAT-CLEAN",
        "test_new_green.py",
        "def test_new_green():\n    assert True\n",
    )
    return repo


@pytest.fixture
def repo_with_new_red_branch(tmp_path: Path) -> Path:
    repo = _base_repo(tmp_path, declared_command())
    _branch_adding(
        repo,
        "FEAT-NEWRED",
        "test_new_red.py",
        "def test_new_red():\n    assert False, 'this merge broke it'\n",
    )
    return repo


VALIDATE_OK = [
    sys.executable,
    "-c",
    "import json; print(json.dumps({'valid': True, 'errors': []}))",
]


def _branch_exists(repo: Path, branch: str) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Rule 6 — the four cases the spec names
# ---------------------------------------------------------------------------


class TestMeasuredBaseline:
    def test_clean_branch_passes_with_the_old_red_in_the_baseline(
        self, repo_with_clean_branch: Path
    ):
        """A branch that adds nothing red passes, though main is red."""
        report = execute_merge(
            repo_with_clean_branch,
            "FEAT-CLEAN",
            validate_command=VALIDATE_OK,
        )

        assert report.outcome == OUTCOME_MERGED
        assert report.baseline_measured is not None
        assert list(report.baseline_measured.failing) == [PRE_EXISTING_RED]
        assert report.baseline_measured.passed == 2
        assert report.baseline_measured.command == declared_command()
        # Nothing new is red, so nothing is charged and the merge passes —
        # even though the suite itself still exits non-zero.
        assert report.charged_failures == ()
        assert report.verify_suite_status == "failed"
        assert report.verify_status == "passed"
        assert report.verify_ok is True
        assert report.verify_source == "repository toolchain declaration"
        assert report.verify_command == declared_command()

    def test_the_receipt_says_why_a_red_suite_still_passed(
        self, repo_with_clean_branch: Path
    ):
        report = execute_merge(
            repo_with_clean_branch,
            "FEAT-CLEAN",
            validate_command=VALIDATE_OK,
        )
        receipt = "\n".join(report.receipt_lines())
        assert "1 test(s) were already failing there" in receipt
        assert "nothing is charged to this merge" in receipt
        assert "The tests were run with:" in receipt

    def test_a_new_red_is_charged_and_only_that_one(
        self, repo_with_new_red_branch: Path
    ):
        report = execute_merge(
            repo_with_new_red_branch,
            "FEAT-NEWRED",
            validate_command=VALIDATE_OK,
        )

        assert report.outcome == OUTCOME_MERGED
        assert list(report.baseline_measured.failing) == [PRE_EXISTING_RED]
        assert list(report.charged_failures) == [NEW_RED]
        assert report.verify_status == "failed"
        assert report.verify_ok is False

    def test_no_measure_baseline_charges_the_old_red_as_well(
        self, repo_with_new_red_branch: Path
    ):
        report = execute_merge(
            repo_with_new_red_branch,
            "FEAT-NEWRED",
            validate_command=VALIDATE_OK,
            measure_baseline=False,
        )

        assert report.outcome == OUTCOME_MERGED
        assert report.baseline_measured is None
        assert sorted(report.charged_failures) == sorted(
            [PRE_EXISTING_RED, NEW_RED]
        )
        assert report.verify_status == "failed"
        assert report.verify_ok is False
        assert any("no pre-merge baseline" in n for n in report.notes)

    def test_a_runner_that_cannot_start_is_unverified_and_has_no_baseline(
        self, tmp_path: Path
    ):
        """Never an empty baseline: a check that could not run says so."""
        repo = _base_repo(
            tmp_path, declared_command("/nowhere/there-is-no-python")
        )
        _branch_adding(
            repo,
            "FEAT-CLEAN",
            "test_new_green.py",
            "def test_new_green():\n    assert True\n",
        )
        pre = _git(repo, "rev-parse", "main")

        report = execute_merge(repo, "FEAT-CLEAN", validate_command=VALIDATE_OK)

        assert report.outcome == OUTCOME_MERGED
        assert report.baseline_measured is None
        assert report.verify_ran is True
        assert report.verify_status == "unverified"
        assert report.verify_suite_status == "unverified"
        assert report.verify_detail == "test runner could not start"
        assert report.charged_failures == ()
        assert report.verify_ok is False
        # The merge itself still happened, and the branch still survives.
        assert _git(repo, "rev-parse", "main") != pre
        assert _branch_exists(repo, "autobuild/FEAT-CLEAN")
        receipt = "\n".join(report.receipt_lines())
        assert "could not be verified" in receipt
        assert "not a pass" in receipt

    def test_a_supplied_baseline_is_used_and_nothing_is_measured(
        self, repo_with_new_red_branch: Path
    ):
        report = execute_merge(
            repo_with_new_red_branch,
            "FEAT-NEWRED",
            validate_command=VALIDATE_OK,
            baseline_failing=[PRE_EXISTING_RED],
        )

        assert report.baseline_measured is None
        assert list(report.charged_failures) == [NEW_RED]

    def test_the_baseline_is_measured_on_the_target_branch(
        self, repo_with_new_red_branch: Path
    ):
        """HEAD sitting on the feature branch must not become the baseline."""
        _git(repo_with_new_red_branch, "checkout", "-q", "autobuild/FEAT-NEWRED")

        report = execute_merge(
            repo_with_new_red_branch,
            "FEAT-NEWRED",
            validate_command=VALIDATE_OK,
        )

        # Measured on main, so the branch's own new red is NOT excused.
        assert list(report.baseline_measured.failing) == [PRE_EXISTING_RED]
        assert list(report.charged_failures) == [NEW_RED]

    def test_the_json_report_carries_the_three_baseline_fields(
        self, repo_with_clean_branch: Path
    ):
        report = execute_merge(
            repo_with_clean_branch,
            "FEAT-CLEAN",
            validate_command=VALIDATE_OK,
        )
        data = report.to_dict()
        assert set(data["baseline_measured"]) == {"failing", "passed", "command"}
        assert data["baseline_measured"]["failing"] == [PRE_EXISTING_RED]
        assert data["baseline_measured"]["passed"] == 2
        assert data["verify_command"] == declared_command()
        assert data["verify_source"] == "repository toolchain declaration"
        assert data["verify_suite_status"] == "failed"
        assert data["verify_ok"] is True
        # Still JSON, still one shape.
        json.loads(json.dumps(data))

    def test_no_baseline_run_when_verification_is_off(
        self, repo_with_clean_branch: Path
    ):
        report = execute_merge(
            repo_with_clean_branch, "FEAT-CLEAN", verify=False
        )
        assert report.outcome == OUTCOME_MERGED
        assert report.baseline_measured is None
        assert report.verify_ran is False

    def test_a_conflict_still_reports_what_the_baseline_found(
        self, tmp_path: Path
    ):
        repo = _base_repo(tmp_path, declared_command())
        _git(repo, "checkout", "-q", "-b", "autobuild/FEAT-CLASH")
        (repo / "tests" / "test_green.py").write_text(
            "def test_green_one():\n    assert True  # branch side\n",
            encoding="utf-8",
        )
        _git(repo, "commit", "-aqm", "branch edit")
        _git(repo, "checkout", "-q", "main")
        (repo / "tests" / "test_green.py").write_text(
            "def test_green_one():\n    assert True  # main side\n",
            encoding="utf-8",
        )
        _git(repo, "commit", "-aqm", "main edit")

        report = execute_merge(repo, "FEAT-CLASH", validate_command=VALIDATE_OK)

        assert report.outcome == OUTCOME_CONFLICT
        assert report.baseline_measured is not None
        assert list(report.baseline_measured.failing) == [PRE_EXISTING_RED]
        assert _git(repo, "status", "--porcelain") == ""
        assert _branch_exists(repo, "autobuild/FEAT-CLASH")


# ---------------------------------------------------------------------------
# The measurement on its own
# ---------------------------------------------------------------------------


class TestMeasurePreMergeBaseline:
    def test_it_runs_the_declared_command_and_counts_both_sides(
        self, repo_with_clean_branch: Path
    ):
        measured = measure_pre_merge_baseline(repo_with_clean_branch, "main")

        assert measured.ran is True
        assert measured.command == declared_command()
        assert measured.source == "repository toolchain declaration"
        assert list(measured.failing) == [PRE_EXISTING_RED]
        assert measured.passed == 2

    def test_it_switches_to_the_target_branch_first(
        self, repo_with_new_red_branch: Path
    ):
        _git(repo_with_new_red_branch, "checkout", "-q", "autobuild/FEAT-NEWRED")

        measured = measure_pre_merge_baseline(repo_with_new_red_branch, "main")

        assert measured.ran is True
        assert list(measured.failing) == [PRE_EXISTING_RED]

    def test_a_missing_target_branch_is_a_run_that_never_started(
        self, repo_with_clean_branch: Path
    ):
        measured = measure_pre_merge_baseline(
            repo_with_clean_branch, "no-such-branch"
        )
        assert measured.ran is False
        assert "never started" in measured.detail
        assert measured.failing == ()

    def test_a_green_target_branch_measures_an_honest_empty_baseline(
        self, tmp_path: Path
    ):
        repo = _base_repo(tmp_path, declared_command())
        (repo / "tests" / "test_already_red.py").write_text(
            "def test_already_red():\n    assert True\n", encoding="utf-8"
        )
        _git(repo, "commit", "-aqm", "the old red was fixed")

        measured = measure_pre_merge_baseline(repo, "main")

        assert measured.ran is True
        assert measured.failing == ()
        assert measured.passed == 3


# ---------------------------------------------------------------------------
# The verdict rule, stated once
# ---------------------------------------------------------------------------


class TestMergeVerdictFromRun:
    def _verdict(self, **kwargs):
        args = dict(
            suite_status="failed",
            suite_detail="test run failed (exit 1)",
            observed_failures=[PRE_EXISTING_RED],
            charged_failures=[],
            baseline_known=True,
            target_branch="main",
        )
        args.update(kwargs)
        return merge_verdict_from_run(**args)

    def test_only_old_failures_is_a_pass(self):
        status, detail = self._verdict()
        assert status == "passed"
        assert "already failing on main before the merge" in detail

    def test_a_charged_failure_is_a_fail(self):
        status, detail = self._verdict(charged_failures=[NEW_RED])
        assert status == "failed"
        assert detail == "test run failed (exit 1)"

    def test_no_baseline_is_a_fail(self):
        status, _ = self._verdict(baseline_known=False)
        assert status == "failed"

    def test_a_failure_that_named_no_tests_is_never_excused(self):
        """A collection error charges nothing — it must not read as a pass."""
        status, detail = self._verdict(observed_failures=[])
        assert status == "failed"
        assert "named no failing tests" in detail

    def test_a_run_that_could_not_start_stays_unverified(self):
        status, detail = self._verdict(
            suite_status="unverified",
            suite_detail="test runner could not start",
            observed_failures=[],
        )
        assert status == "unverified"
        assert detail == "test runner could not start"

    def test_a_passing_run_is_passed_verbatim(self):
        status, detail = self._verdict(
            suite_status="passed", suite_detail="12 passed", observed_failures=[]
        )
        assert status == "passed"
        assert detail == "12 passed"


class TestPassedCount:
    @pytest.mark.parametrize(
        "output,expected",
        [
            ("1 failed, 2 passed in 0.30s\n", 2),
            ("12 passed in 1.10s\n", 12),
            ("no summary here", 0),
            ("", 0),
            (None, 0),
            ("collected 3 items\n\n1 failed, 0 passed in 0.1s\n", 0),
        ],
    )
    def test_reads_the_summary_count(self, output, expected):
        assert passed_count_from_output(output) == expected
