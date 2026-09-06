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

from guardkit.orchestrator.baseline import failing_node_ids
from guardkit.orchestrator.completion_verification import (
    resolve_verify_command,
    run_completion_verification,
)
from guardkit.orchestrator.merge_executor import (
    OUTCOME_CONFLICT,
    OUTCOME_MERGED,
    OUTCOME_REFUSED,
    execute_merge,
    measure_pre_merge_baseline,
    merge_verdict_from_run,
    named_failures_are_complete,
    passed_count_from_output,
    reported_failure_count,
    resolve_verify_command_on_target,
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
        # No post-merge run was attempted, so nothing ran (rule 12).
        assert report.verify_ran is False
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


# ---------------------------------------------------------------------------
# A badly failing suite must not be read as a clean one (2026-09-06 fix)
#
# pytest's end-of-run summary lists one line per failing test. Past roughly
# twenty failures that block is longer than the 2000-character excerpt the
# verification result keeps for receipts, so a merge that read test names out
# of that excerpt would lose names from BOTH the pre-merge baseline and the
# post-merge run — and a genuinely new failure whose name fell off the end
# would be charged to nobody. Measured on this very fixture before the fix:
# 8 pre-existing reds charged the new one correctly, 24 charged nothing at all
# and reported the merge as clean.
# ---------------------------------------------------------------------------


ADDED_RED = "tests/test_added_new_red.py::test_added_new_red"


def _repo_with_many_reds(tmp_path: Path, how_many: int) -> Path:
    """main carries ``how_many`` failing tests; the branch adds one more."""
    repo = _base_repo(tmp_path, declared_command())
    # test_already_red.py is red already; add the rest with names long enough
    # to be realistic (real node ids are paths, not two letters).
    for index in range(how_many - 1):
        name = f"test_pre_existing_failure_number_{index:02d}"
        (repo / "tests" / f"{name}.py").write_text(
            f"def {name}():\n"
            f"    assert False, 'this one was red on main all along'\n",
            encoding="utf-8",
        )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main was already this red")
    _branch_adding(
        repo,
        "FEAT-NEWRED",
        "test_added_new_red.py",
        "def test_added_new_red():\n    assert False, 'this merge broke it'\n",
    )
    return repo


class TestABadlyFailingSuiteIsNotReadAsClean:
    @pytest.mark.parametrize("how_many", [8, 16, 25])
    def test_the_one_new_failure_is_charged_however_red_main_was(
        self, tmp_path: Path, how_many: int
    ):
        repo = _repo_with_many_reds(tmp_path, how_many)

        report = execute_merge(repo, "FEAT-NEWRED", validate_command=VALIDATE_OK)

        assert report.outcome == OUTCOME_MERGED
        assert len(report.baseline_measured.failing) == how_many
        assert list(report.charged_failures) == [ADDED_RED]
        assert report.verify_status == "failed"
        assert report.verify_ok is False

    def test_the_receipts_excerpt_really_does_lose_names(self, tmp_path: Path):
        """The fixture genuinely overflows the excerpt — otherwise the test
        above would prove nothing."""
        repo = _repo_with_many_reds(tmp_path, 25)
        command, source, profile = resolve_verify_command(repo)
        result = run_completion_verification(
            repo, command, source, stack_profile=profile
        )

        from_excerpt = failing_node_ids(result.output_tail)
        from_whole_run = failing_node_ids(result.output_for_parsing)

        assert len(result.output_tail) == 2000
        assert len(from_excerpt) < len(from_whole_run) == 25


# ---------------------------------------------------------------------------
# The cross-check: never excuse more failures than the output could name
# ---------------------------------------------------------------------------


def _repo_that_under_reports(root: Path) -> Path:
    """A repository whose declared test command names two of thirty failures.

    Not a mock: a genuine script, on disk, in the repository, resolved and run
    through the same declaration, runner and parser as any other test command.
    It stands in for every way a reading can come up short — a log cut off, a
    reporter guardkit does not know, a run that died part-way through its
    summary.
    """
    repo = _base_repo(root, "placeholder, replaced below")
    runner = repo / ".guardkit" / "under_reporting_runner.py"
    runner.write_text(
        "print('FAILED tests/test_one.py::test_one - boom')\n"
        "print('FAILED tests/test_two.py::test_two - boom')\n"
        "print('30 failed, 2 passed in 1.00s')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    _write_toolchain(
        repo, f'"{sys.executable}" .guardkit/under_reporting_runner.py'
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "a runner that cannot name every failure")
    return repo


class TestFailuresThatCouldNotAllBeNamed:
    def test_nothing_is_excused_when_the_counts_disagree(self, tmp_path: Path):
        repo = _repo_that_under_reports(tmp_path)
        _branch_adding(
            repo,
            "FEAT-CLEAN",
            "test_new_green.py",
            "def test_new_green():\n    assert True\n",
        )

        report = execute_merge(repo, "FEAT-CLEAN", validate_command=VALIDATE_OK)

        # The two named failures cancel out, so nothing is charged — and yet
        # the merge must NOT pass, because twenty-eight failures went unnamed
        # and any one of them could be this merge's doing.
        assert report.outcome == OUTCOME_MERGED
        assert report.charged_failures == ()
        assert report.verify_status == "failed"
        assert report.verify_ok is False
        assert "more tests failed than could be named" in report.verify_detail
        assert any("more failures than its output named" in n for n in report.notes)

    def test_the_count_the_runner_gave_is_read(self):
        assert reported_failure_count("30 failed, 2 passed in 1.0s") == 30
        assert reported_failure_count("1 failed, 3 errors in 1.0s") == 4
        # A run that says nothing about failures gives no count to check
        # against — that is not the same as a count of zero.
        assert reported_failure_count("12 passed in 1.0s") is None
        assert reported_failure_count("no summary at all") is None
        assert reported_failure_count(None) is None

    @pytest.mark.parametrize(
        "output,named,complete",
        [
            ("2 failed in 1s", ["a::b", "c::d"], True),
            ("3 failed in 1s", ["a::b", "c::d"], False),
            ("1 failed, 1 errors in 1s", ["a::b", "c::d"], True),
            ("nothing the parser knows", ["a::b"], True),
            ("30 failed in 1s", [], False),
        ],
    )
    def test_a_reading_is_complete_only_when_it_names_them_all(
        self, output, named, complete
    ):
        assert named_failures_are_complete(output, named) is complete

    def test_the_verdict_refuses_to_excuse_a_partial_reading(self):
        status, detail = merge_verdict_from_run(
            suite_status="failed",
            suite_detail="test run failed (exit 1)",
            observed_failures=[PRE_EXISTING_RED],
            charged_failures=[],
            baseline_known=True,
            target_branch="main",
            reading_complete=False,
        )
        assert status == "failed"
        assert "more tests failed than could be named" in detail

    def test_a_short_baseline_reading_excuses_nothing_either(
        self, tmp_path: Path
    ):
        """The baseline's own reading has to be complete too."""
        repo = _repo_with_many_reds(tmp_path, 8)
        measured = measure_pre_merge_baseline(repo, "main")
        assert measured.ran is True
        assert measured.names_complete is True

        short_root = tmp_path / "short"
        short_root.mkdir()
        short = _repo_that_under_reports(short_root)
        measured_short = measure_pre_merge_baseline(short, "main")
        assert measured_short.ran is True
        assert measured_short.names_complete is False


# ---------------------------------------------------------------------------
# The receipt must not say the tests ran when they did not
# ---------------------------------------------------------------------------


class TestTheReceiptSaysWhetherTheTestsRan:
    def test_a_check_that_never_ran_does_not_claim_it_did(self, tmp_path: Path):
        repo = _base_repo(
            tmp_path, declared_command("/nowhere/there-is-no-python")
        )
        _branch_adding(
            repo,
            "FEAT-CLEAN",
            "test_new_green.py",
            "def test_new_green():\n    assert True\n",
        )

        report = execute_merge(repo, "FEAT-CLEAN", validate_command=VALIDATE_OK)
        receipt = "\n".join(report.receipt_lines())

        assert report.verify_status == "unverified"
        assert "The tests would have been run with:" in receipt
        assert "The tests were run with:" not in receipt

    def test_a_check_that_did_run_still_says_so(self, repo_with_clean_branch: Path):
        report = execute_merge(
            repo_with_clean_branch, "FEAT-CLEAN", validate_command=VALIDATE_OK
        )
        receipt = "\n".join(report.receipt_lines())
        assert "The tests were run with:" in receipt
        assert "would have been run" not in receipt


# ---------------------------------------------------------------------------
# A refusal costs nothing: no branch switch, no suite run
# ---------------------------------------------------------------------------


def _marker_command(marker: Path) -> str:
    """A real command that records the fact that it ran, then passes."""
    script = f"open({json.dumps(str(marker))}, 'w').write('ran')\n"
    return f'"{sys.executable}" -c {json.dumps(script)}'


class TestARefusalTouchesNothing:
    def test_an_unresolvable_target_refuses_before_any_test_runs(
        self, tmp_path: Path
    ):
        marker = tmp_path / "the-suite-ran"
        repo = _base_repo(tmp_path, _marker_command(marker))
        _branch_adding(
            repo,
            "FEAT-CLEAN",
            "test_new_green.py",
            "def test_new_green():\n    assert True\n",
        )
        _git(repo, "checkout", "-q", "-b", "somewhere-else")

        report = execute_merge(
            repo,
            "FEAT-CLEAN",
            target_branch="no-such-branch",
            validate_command=VALIDATE_OK,
        )

        assert report.outcome == OUTCOME_REFUSED
        assert "could not resolve" in report.refusal_reason
        # Nothing was run and nothing was moved.
        assert not marker.exists()
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "somewhere-else"
        assert report.baseline_measured is None
        assert report.notes == ()


# ---------------------------------------------------------------------------
# Rule 11 — a branch never chooses the command that judges it
#
# The declared test command lives in .guardkit/config.yaml, INSIDE the
# repository being merged. Resolve it after the merge and the branch under
# judgement has just written the command that judges it: set toolchain.test
# to "true", add as many failing tests as you like, and the merge reads as
# clean because nothing was ever run. So the command is resolved on the
# TARGET branch, before the merge, on every path — the same answer
# toolchain_declaration.py's snapshot law gives inside a build, taken at the
# merge's own scale.
#
# The fixture below is a real branch doing exactly that, and every case here
# is driven through execute_merge on a real repository.
# ---------------------------------------------------------------------------


SELF_JUDGE = "FEAT-SELFJUDGE"


def _repo_with_a_branch_that_rewrites_the_command(tmp_path: Path) -> Path:
    """main declares a real pytest run; the branch declares ``true``.

    The branch also adds a genuinely failing test. Judged by main's declared
    command the merged tree is red; judged by the branch's own it is green,
    because ``true`` runs no tests at all and exits 0.
    """
    repo = _base_repo(tmp_path, declared_command())
    _git(repo, "checkout", "-q", "-b", f"autobuild/{SELF_JUDGE}")
    _write_toolchain(repo, "true")
    (repo / "tests" / "test_new_red.py").write_text(
        "def test_new_red():\n    assert False, 'this merge broke it'\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"{SELF_JUDGE} greens itself")
    _git(repo, "checkout", "-q", "main")
    return repo


class TestTheBranchCannotChooseItsOwnJudge:
    def test_the_fixture_really_would_green_itself(self, tmp_path: Path):
        """Without this, the three cases below would prove nothing.

        On main the declaration is a real test run. After the merge it is
        ``true`` — so a command resolved on the merged tree runs no tests and
        reports a pass, with the branch's new failure never executed.
        """
        repo = _repo_with_a_branch_that_rewrites_the_command(tmp_path)
        assert resolve_verify_command(repo)[0] == declared_command()

        execute_merge(
            repo,
            SELF_JUDGE,
            validate_command=VALIDATE_OK,
            measure_baseline=False,
        )

        command, source, profile = resolve_verify_command(repo)
        assert command == "true"
        after = run_completion_verification(
            repo, command, source, stack_profile=profile
        )
        assert after.status == "passed"

    def test_a_supplied_baseline_still_runs_the_target_branchs_command(
        self, tmp_path: Path
    ):
        """--baseline-json: the command comes from main, so the new red bites."""
        repo = _repo_with_a_branch_that_rewrites_the_command(tmp_path)

        report = execute_merge(
            repo,
            SELF_JUDGE,
            validate_command=VALIDATE_OK,
            baseline_failing=[PRE_EXISTING_RED],
        )

        assert report.outcome == OUTCOME_MERGED
        assert report.verify_command == declared_command()
        assert report.verify_source == "repository toolchain declaration"
        assert list(report.charged_failures) == [NEW_RED]
        assert report.verify_status == "failed"
        assert report.verify_ok is False

    def test_no_measure_baseline_still_runs_the_target_branchs_command(
        self, tmp_path: Path
    ):
        """--no-measure-baseline: same command, and every red is charged."""
        repo = _repo_with_a_branch_that_rewrites_the_command(tmp_path)

        report = execute_merge(
            repo,
            SELF_JUDGE,
            validate_command=VALIDATE_OK,
            measure_baseline=False,
        )

        assert report.outcome == OUTCOME_MERGED
        assert report.verify_command == declared_command()
        assert report.verify_source == "repository toolchain declaration"
        assert report.baseline_measured is None
        assert sorted(report.charged_failures) == sorted(
            [PRE_EXISTING_RED, NEW_RED]
        )
        assert report.verify_status == "failed"
        assert report.verify_ok is False

    def test_the_measured_baseline_path_resists_it_too(self, tmp_path: Path):
        repo = _repo_with_a_branch_that_rewrites_the_command(tmp_path)

        report = execute_merge(
            repo, SELF_JUDGE, validate_command=VALIDATE_OK
        )

        assert report.verify_command == declared_command()
        assert report.baseline_measured.command == declared_command()
        assert list(report.charged_failures) == [NEW_RED]
        assert report.verify_status == "failed"


class TestResolveVerifyCommandOnTarget:
    def test_it_reads_the_target_branch_not_wherever_head_is(
        self, tmp_path: Path
    ):
        repo = _repo_with_a_branch_that_rewrites_the_command(tmp_path)
        _git(repo, "checkout", "-q", f"autobuild/{SELF_JUDGE}")

        (command, source, _profile), problem = (
            resolve_verify_command_on_target(repo, "main")
        )

        assert problem is None
        assert command == declared_command()
        assert source == "repository toolchain declaration"
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"

    def test_a_target_it_cannot_reach_is_reported_never_guessed(
        self, repo_with_clean_branch: Path
    ):
        _resolution, problem = resolve_verify_command_on_target(
            repo_with_clean_branch, "no-such-branch"
        )
        assert problem is not None
        assert "could not switch to branch no-such-branch" in problem


# ---------------------------------------------------------------------------
# Rule 12 — verify_ran says whether a post-merge run was attempted
# ---------------------------------------------------------------------------


class TestVerifyRanSaysWhetherARunWasAttempted:
    def test_a_check_that_could_not_run_reports_no_run(self, tmp_path: Path):
        repo = _base_repo(
            tmp_path, declared_command("/nowhere/there-is-no-python")
        )
        _branch_adding(
            repo,
            "FEAT-CLEAN",
            "test_new_green.py",
            "def test_new_green():\n    assert True\n",
        )

        report = execute_merge(repo, "FEAT-CLEAN", validate_command=VALIDATE_OK)

        assert report.verify_ran is False
        assert report.verify_status == "unverified"
        assert report.to_dict()["verify_ran"] is False
        # And the receipt still explains itself, rather than claiming that
        # verification had been turned off.
        receipt = "\n".join(report.receipt_lines())
        assert "could not be verified" in receipt
        assert "not a pass" in receipt
        assert "turned off" not in receipt

    def test_a_check_that_did_run_reports_a_run(
        self, repo_with_clean_branch: Path
    ):
        report = execute_merge(
            repo_with_clean_branch, "FEAT-CLEAN", validate_command=VALIDATE_OK
        )
        assert report.verify_ran is True
        assert report.to_dict()["verify_ran"] is True

    def test_verification_turned_off_reads_differently(
        self, repo_with_clean_branch: Path
    ):
        report = execute_merge(
            repo_with_clean_branch, "FEAT-CLEAN", verify=False
        )
        assert report.verify_ran is False
        assert report.verify_status is None
        receipt = "\n".join(report.receipt_lines())
        assert "turned off for this run" in receipt
        assert "could not be verified" not in receipt
