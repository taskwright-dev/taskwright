"""Hermetic tests for the merge primitive (merge_executor).

Spec: ``docs/make-merge-work-build-spec-2026-08-24.md`` (ai-transition) — the
merge word as a mechanism. Modelled on ``test_machine_verify.py``: real git in
``tmp_path``, no mocks of git, no network, no seats.

The load-bearing assertions:

* the branch ``autobuild/<FEATURE_ID>`` survives EVERY path (it is the
  rollback path);
* the merge commit message is the exact template, filled only from records;
* refusals happen before anything is touched;
* a conflict aborts and leaves the tree clean;
* verification never invents a clean.
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
    OUTCOME_REFUSED,
    MergeReport,
    charged_failures_from_output,
    conflicted_files_from_status,
    execute_merge,
    merge_commit_message,
    parse_validate_stdout,
    preflight_refusal,
    run_feature_validate,
)


# ---------------------------------------------------------------------------
# git fixture repos (real git in tmp_path — the machine_verify pattern)
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


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "shared.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


@pytest.fixture
def merge_repo(tmp_path: Path) -> Path:
    """main + a clean-merging branch autobuild/FEAT-X, HEAD back on main."""
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "autobuild/FEAT-X")
    (repo / "feature.txt").write_text("built by the factory\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-q", "-m", "feature work")
    _git(repo, "checkout", "-q", "main")
    return repo


@pytest.fixture
def conflict_repo(tmp_path: Path) -> Path:
    """main and autobuild/FEAT-X both edit shared.txt — a genuine conflict."""
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "autobuild/FEAT-X")
    (repo / "shared.txt").write_text("branch side\n", encoding="utf-8")
    _git(repo, "commit", "-aqm", "branch edit")
    _git(repo, "checkout", "-q", "main")
    (repo / "shared.txt").write_text("main side\n", encoding="utf-8")
    _git(repo, "commit", "-aqm", "main edit")
    return repo


def _branch_exists(repo: Path, branch: str) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# happy merge
# ---------------------------------------------------------------------------


class TestHappyMerge:
    def test_merges_records_shas_and_keeps_branch(self, merge_repo: Path):
        pre = _git(merge_repo, "rev-parse", "main")

        report = execute_merge(merge_repo, "FEAT-X", verify=False)

        assert report.outcome == OUTCOME_MERGED
        assert report.pre_sha == pre
        assert report.post_sha == _git(merge_repo, "rev-parse", "main")
        assert report.post_sha != pre
        # A real --no-ff merge commit: two parents.
        parents = _git(merge_repo, "rev-list", "--parents", "-1", "main").split()
        assert len(parents) == 3  # commit + 2 parents
        # The feature's work arrived on main.
        assert (merge_repo / "feature.txt").exists()
        # THE branch-survival law: the rollback path is kept.
        assert _branch_exists(merge_repo, "autobuild/FEAT-X")

    def test_merge_commit_message_is_the_exact_template(self, merge_repo: Path):
        pre = _git(merge_repo, "rev-parse", "main")
        branch_sha = _git(merge_repo, "rev-parse", "autobuild/FEAT-X")

        report = execute_merge(merge_repo, "FEAT-X", verify=False)
        assert report.outcome == OUTCOME_MERGED

        expected = (
            f"merge(FEAT-X): merged on the merge word\n\n"
            f"{pre[:12]}..{branch_sha[:12]} — branch autobuild/FEAT-X "
            f"retained as the rollback path"
        )
        assert merge_commit_message("FEAT-X", pre, branch_sha) == expected
        actual = _git(merge_repo, "log", "-1", "--format=%B", "main").strip()
        assert actual == expected

    def test_no_verify_report_says_so_plainly(self, merge_repo: Path):
        report = execute_merge(merge_repo, "FEAT-X", verify=False)
        assert report.verify_ran is False
        assert report.verify_status is None
        assert report.verify_ok is False
        receipt = "\n".join(report.receipt_lines())
        assert "NOT verified" in receipt
        assert "rollback path" in receipt


# ---------------------------------------------------------------------------
# refusals — each before anything is touched
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_target_moved_refuses_loudly(self, merge_repo: Path):
        # The checks ran against the BRANCH sha — main points elsewhere now.
        stale = _git(merge_repo, "rev-parse", "autobuild/FEAT-X")
        pre = _git(merge_repo, "rev-parse", "main")

        report = execute_merge(
            merge_repo, "FEAT-X", expect_target_sha=stale, verify=False
        )

        assert report.outcome == OUTCOME_REFUSED
        assert "has moved since the checks ran" in report.refusal_reason
        # Nothing merged, nothing touched.
        assert _git(merge_repo, "rev-parse", "main") == pre
        assert _branch_exists(merge_repo, "autobuild/FEAT-X")

    def test_expected_sha_prefix_accepted(self, merge_repo: Path):
        """A >=7-char prefix of the true sha is accepted (CLI convenience)."""
        pre = _git(merge_repo, "rev-parse", "main")
        report = execute_merge(
            merge_repo, "FEAT-X", expect_target_sha=pre[:12], verify=False
        )
        assert report.outcome == OUTCOME_MERGED

    def test_dirty_tree_refuses(self, merge_repo: Path):
        (merge_repo / "app.py").write_text("x = 99  # uncommitted\n")
        pre = _git(merge_repo, "rev-parse", "main")

        report = execute_merge(merge_repo, "FEAT-X", verify=False)

        assert report.outcome == OUTCOME_REFUSED
        assert "dirty" in report.refusal_reason
        assert _git(merge_repo, "rev-parse", "main") == pre
        assert _branch_exists(merge_repo, "autobuild/FEAT-X")

    def test_missing_branch_refuses(self, merge_repo: Path):
        report = execute_merge(merge_repo, "FEAT-NONE", verify=False)
        assert report.outcome == OUTCOME_REFUSED
        assert "autobuild/FEAT-NONE does not exist" in report.refusal_reason

    def test_not_a_git_repo_refuses(self, tmp_path: Path):
        plain = tmp_path / "not-a-repo"
        plain.mkdir()
        report = execute_merge(plain, "FEAT-X", verify=False)
        assert report.outcome == OUTCOME_REFUSED
        assert "not a git repository" in report.refusal_reason

    def test_preflight_is_pure_reason_or_none(self, merge_repo: Path):
        assert preflight_refusal(merge_repo, "FEAT-X") is None
        assert preflight_refusal(merge_repo, "FEAT-NONE") is not None


# ---------------------------------------------------------------------------
# conflict — abort, clean tree, branch survives
# ---------------------------------------------------------------------------


class TestConflict:
    def test_conflict_aborts_cleanly_and_names_files(self, conflict_repo: Path):
        pre = _git(conflict_repo, "rev-parse", "main")

        report = execute_merge(conflict_repo, "FEAT-X", verify=False)

        assert report.outcome == OUTCOME_CONFLICT
        assert "shared.txt" in report.conflict_files
        # The tree is clean after the abort.
        assert _git(conflict_repo, "status", "--porcelain") == ""
        # main did not move; the branch survives.
        assert _git(conflict_repo, "rev-parse", "main") == pre
        assert _branch_exists(conflict_repo, "autobuild/FEAT-X")

    def test_conflict_receipt_speaks_plainly(self, conflict_repo: Path):
        report = execute_merge(conflict_repo, "FEAT-X", verify=False)
        receipt = "\n".join(report.receipt_lines())
        assert "aborted" in receipt
        assert "shared.txt" in receipt
        assert "rollback path" in receipt

    def test_uu_row_parsing(self):
        porcelain = "UU shared.txt\nM  other.txt\nUU deep/dir/file.py\n"
        assert conflicted_files_from_status(porcelain) == [
            "shared.txt",
            "deep/dir/file.py",
        ]


# ---------------------------------------------------------------------------
# validate: stdout-only parse
# ---------------------------------------------------------------------------


class TestValidateStdoutParse:
    def test_valid_true_parses(self):
        stdout = json.dumps(
            {"feature_id": "FEAT-X", "valid": True, "errors": []}
        )
        assert parse_validate_stdout(stdout) == (True, "")

    def test_valid_false_carries_errors(self):
        stdout = json.dumps(
            {"feature_id": "FEAT-X", "valid": False, "errors": ["task missing"]}
        )
        valid, detail = parse_validate_stdout(stdout)
        assert valid is False
        assert "task missing" in detail

    def test_non_json_gives_no_verdict(self):
        valid, detail = parse_validate_stdout("INFO: something\nnot json")
        assert valid is None
        assert "not JSON" in detail

    def test_empty_stdout_gives_no_verdict(self):
        valid, _ = parse_validate_stdout("")
        assert valid is None

    def test_json_without_valid_field_gives_no_verdict(self):
        valid, _ = parse_validate_stdout(json.dumps({"feature_id": "F"}))
        assert valid is None

    def test_subprocess_stderr_noise_never_reaches_the_parser(
        self, tmp_path: Path
    ):
        """An INFO line on stderr must not corrupt the stdout-only parse."""
        script = (
            "import sys, json\n"
            "sys.stderr.write('INFO: worktree state loaded\\n')\n"
            "print(json.dumps({'feature_id': 'FEAT-X', 'valid': True,"
            " 'errors': []}))\n"
        )
        valid, detail = run_feature_validate(
            tmp_path,
            "FEAT-X",
            validate_command=[sys.executable, "-c", script],
        )
        assert valid is True
        assert detail == ""


# ---------------------------------------------------------------------------
# charged failures — with and without a baseline
# ---------------------------------------------------------------------------


_SUITE_OUTPUT = (
    "FAILED tests/a.py::test_x - AssertionError\n"
    "FAILED tests/b.py::test_y - ValueError\n"
    "2 failed, 10 passed in 1.2s\n"
)


class TestChargedFailures:
    def test_baseline_excuses_preexisting_reds(self, tmp_path: Path):
        charged, notes = charged_failures_from_output(
            tmp_path, _SUITE_OUTPUT, baseline_failing=["tests/a.py::test_x"]
        )
        assert charged == ["tests/b.py::test_y"]
        assert notes == []

    def test_no_baseline_reports_full_observed_set_with_note(
        self, tmp_path: Path
    ):
        charged, notes = charged_failures_from_output(
            tmp_path, _SUITE_OUTPUT, baseline_failing=None
        )
        # Never an invented clean: everything observed is reported.
        assert charged == ["tests/a.py::test_x", "tests/b.py::test_y"]
        assert any("no pre-merge baseline" in n for n in notes)
        assert any("diff unavailable" in n for n in notes)

    def test_empty_baseline_list_is_a_real_baseline_no_note(
        self, tmp_path: Path
    ):
        charged, notes = charged_failures_from_output(
            tmp_path, _SUITE_OUTPUT, baseline_failing=[]
        )
        assert charged == ["tests/a.py::test_x", "tests/b.py::test_y"]
        assert notes == []

    def test_known_failures_ledger_excuses(self, tmp_path: Path):
        qa = tmp_path / "qa"
        qa.mkdir()
        (qa / "known-failures.yaml").write_text(
            "known_failures:\n  - test_id: tests/a.py::test_x\n",
            encoding="utf-8",
        )
        charged, _ = charged_failures_from_output(
            tmp_path, _SUITE_OUTPUT, baseline_failing=[]
        )
        assert charged == ["tests/b.py::test_y"]

    def test_green_suite_charges_nothing(self, tmp_path: Path):
        charged, notes = charged_failures_from_output(
            tmp_path, "12 passed in 0.4s\n", baseline_failing=None
        )
        assert charged == []
        # A green suite with no baseline needs no scary note.
        assert notes == []


# ---------------------------------------------------------------------------
# execute_merge with verification on (hermetic: no runner in the fixture repo)
# ---------------------------------------------------------------------------


class TestExecuteMergeWithVerify:
    def test_merged_but_unverified_is_never_a_pass(self, merge_repo: Path):
        # The fixture repo has no pyproject/tests -> no runner is detected;
        # the injected validate command answers valid=true on stdout.
        script = (
            "import json\n"
            "print(json.dumps({'feature_id': 'FEAT-X', 'valid': True,"
            " 'errors': []}))\n"
        )
        report = execute_merge(
            merge_repo,
            "FEAT-X",
            verify=True,
            validate_command=[sys.executable, "-c", script],
        )
        assert report.outcome == OUTCOME_MERGED
        # No runner could be found, so no post-merge run was attempted
        # (2026-09-06 spec, rule 12) — and absence is still not a pass.
        assert report.verify_ran is False
        assert report.validate_valid is True
        assert report.verify_status == "unverified"
        assert report.verify_ok is False  # absence of evidence is not a pass
        assert _branch_exists(merge_repo, "autobuild/FEAT-X")
        receipt = "\n".join(report.receipt_lines())
        assert "could not be verified" in receipt
        assert "not a pass" in receipt


# ---------------------------------------------------------------------------
# report shape
# ---------------------------------------------------------------------------


class TestReportShape:
    def test_to_dict_round_trips_the_fields(self):
        report = MergeReport(
            outcome=OUTCOME_MERGED,
            feature_id="FEAT-X",
            target_branch="main",
            branch="autobuild/FEAT-X",
            pre_sha="a" * 40,
            post_sha="b" * 40,
            verify_ran=True,
            verify_status="passed",
            verify_detail="12 passed",
            validate_valid=True,
            charged_failures=(),
            notes=("a note",),
        )
        d = report.to_dict()
        assert d["outcome"] == "merged"
        assert d["branch"] == "autobuild/FEAT-X"
        assert d["pre_sha"] == "a" * 40
        assert d["post_sha"] == "b" * 40
        assert d["verify_ok"] is True
        assert d["charged_failures"] == []
        assert d["conflict_files"] == []
        assert d["notes"] == ["a note"]
        assert isinstance(d["charged_failures"], list)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"verify_status": "failed"},
            {"verify_status": "unverified"},
            {"validate_valid": False},
            {"validate_valid": None},
            {"charged_failures": ("tests/a.py::test_x",)},
            {"verify_ran": False},
        ],
    )
    def test_verify_ok_requires_every_positive(self, kwargs):
        base = dict(
            outcome=OUTCOME_MERGED,
            feature_id="FEAT-X",
            target_branch="main",
            branch="autobuild/FEAT-X",
            verify_ran=True,
            verify_status="passed",
            validate_valid=True,
        )
        base.update(kwargs)
        assert MergeReport(**base).verify_ok is False

    def test_refused_receipt_names_the_reason(self):
        report = MergeReport(
            outcome=OUTCOME_REFUSED,
            feature_id="FEAT-X",
            target_branch="main",
            branch="autobuild/FEAT-X",
            refusal_reason="main has moved since the checks ran",
        )
        receipt = "\n".join(report.receipt_lines())
        assert "refused" in receipt
        assert "has moved since the checks ran" in receipt
