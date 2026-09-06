"""CLI tests for ``guardkit autobuild merge`` (the merge word as a verb).

Fixture pattern: real git in tmp_path (test_cli_autobuild.py fixture_git_repo
style), CliRunner for the Click layer, no mocks of git.

Exit codes under test:
    0 merged (verification off, or on and passed)
    2 refused (dirty / moved / missing branch)
    3 conflict (aborted, tree clean)
    4 merged but verification failed-or-unverified
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from guardkit.cli.autobuild import _load_baseline_failing, merge


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
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    (repo / "shared.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def merge_repo(tmp_path: Path, monkeypatch) -> Path:
    """A repo with a clean-merging autobuild/FEAT-C1 branch; cwd inside it."""
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "autobuild/FEAT-C1")
    (repo / "feature.txt").write_text("the feature\n", encoding="utf-8")
    _git(repo, "add", "feature.txt")
    _git(repo, "commit", "-q", "-m", "feature work")
    _git(repo, "checkout", "-q", "main")
    monkeypatch.chdir(repo)
    return repo


@pytest.fixture
def conflict_repo(tmp_path: Path, monkeypatch) -> Path:
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "autobuild/FEAT-C1")
    (repo / "shared.txt").write_text("branch side\n", encoding="utf-8")
    _git(repo, "commit", "-aqm", "branch edit")
    _git(repo, "checkout", "-q", "main")
    (repo / "shared.txt").write_text("main side\n", encoding="utf-8")
    _git(repo, "commit", "-aqm", "main edit")
    monkeypatch.chdir(repo)
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
# exit code 0 — merged
# ---------------------------------------------------------------------------


class TestMergeHappyPath:
    def test_no_verify_merges_and_exits_zero(self, cli_runner, merge_repo):
        result = cli_runner.invoke(merge, ["FEAT-C1", "--no-verify"])

        assert result.exit_code == 0, result.output
        assert "merged into main" in result.output
        assert "rollback path" in result.output
        assert _branch_exists(merge_repo, "autobuild/FEAT-C1")
        assert (merge_repo / "feature.txt").exists()

    def test_expect_main_sha_matching_merges(self, cli_runner, merge_repo):
        sha = _git(merge_repo, "rev-parse", "main")
        result = cli_runner.invoke(
            merge, ["FEAT-C1", "--no-verify", "--expect-main-sha", sha]
        )
        assert result.exit_code == 0, result.output

    def test_json_prints_the_report_dict(self, cli_runner, merge_repo):
        pre = _git(merge_repo, "rev-parse", "main")
        result = cli_runner.invoke(merge, ["FEAT-C1", "--no-verify", "--json"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["outcome"] == "merged"
        assert data["feature_id"] == "FEAT-C1"
        assert data["branch"] == "autobuild/FEAT-C1"
        assert data["pre_sha"] == pre
        assert data["post_sha"] == _git(merge_repo, "rev-parse", "main")
        assert data["verify_ran"] is False
        assert data["verify_ok"] is False
        assert data["conflict_files"] == []


# ---------------------------------------------------------------------------
# exit code 2 — refused
# ---------------------------------------------------------------------------


class TestMergeRefusals:
    def test_dirty_tree_exits_two(self, cli_runner, merge_repo):
        (merge_repo / "README.md").write_text("dirtied\n", encoding="utf-8")
        pre = _git(merge_repo, "rev-parse", "main")

        result = cli_runner.invoke(merge, ["FEAT-C1", "--no-verify"])

        assert result.exit_code == 2, result.output
        assert "refused" in result.output
        assert _git(merge_repo, "rev-parse", "main") == pre
        assert _branch_exists(merge_repo, "autobuild/FEAT-C1")

    def test_target_moved_exits_two(self, cli_runner, merge_repo):
        stale = _git(merge_repo, "rev-parse", "autobuild/FEAT-C1")
        result = cli_runner.invoke(
            merge, ["FEAT-C1", "--no-verify", "--expect-main-sha", stale]
        )
        assert result.exit_code == 2, result.output
        assert "has moved since the checks ran" in result.output

    def test_missing_branch_exits_two(self, cli_runner, merge_repo):
        result = cli_runner.invoke(merge, ["FEAT-NOPE", "--no-verify"])
        assert result.exit_code == 2, result.output
        assert "does not exist" in result.output

    def test_refusal_json_shape(self, cli_runner, merge_repo):
        result = cli_runner.invoke(merge, ["FEAT-NOPE", "--no-verify", "--json"])
        assert result.exit_code == 2, result.output
        data = json.loads(result.output)
        assert data["outcome"] == "refused"
        assert "does not exist" in data["refusal_reason"]


# ---------------------------------------------------------------------------
# exit code 3 — conflict
# ---------------------------------------------------------------------------


class TestMergeConflict:
    def test_conflict_exits_three_tree_clean_branch_kept(
        self, cli_runner, conflict_repo
    ):
        pre = _git(conflict_repo, "rev-parse", "main")

        result = cli_runner.invoke(merge, ["FEAT-C1", "--no-verify"])

        assert result.exit_code == 3, result.output
        assert "shared.txt" in result.output
        assert _git(conflict_repo, "status", "--porcelain") == ""
        assert _git(conflict_repo, "rev-parse", "main") == pre
        assert _branch_exists(conflict_repo, "autobuild/FEAT-C1")

    def test_conflict_json_names_the_files(self, cli_runner, conflict_repo):
        result = cli_runner.invoke(merge, ["FEAT-C1", "--json", "--no-verify"])
        assert result.exit_code == 3, result.output
        data = json.loads(result.output)
        assert data["outcome"] == "conflict"
        assert "shared.txt" in data["conflict_files"]


# ---------------------------------------------------------------------------
# exit code 4 — merged, but verification gave no pass
# ---------------------------------------------------------------------------


class TestMergeVerifyExitCode:
    def test_merged_but_unverified_exits_four(self, cli_runner, merge_repo):
        # The fixture repo has no tests/ and no runner: verification resolves
        # to no command -> "unverified"; feature validate finds no feature.
        # Both are honestly non-passes, so the exit is 4 — never 0.
        result = cli_runner.invoke(merge, ["FEAT-C1"])

        assert result.exit_code == 4, result.output
        assert _branch_exists(merge_repo, "autobuild/FEAT-C1")
        # The merge itself happened; the report says the verify part plainly.
        assert (merge_repo / "feature.txt").exists()
        assert "not a pass" in result.output or "NOT" in result.output


# ---------------------------------------------------------------------------
# --baseline-json parsing
# ---------------------------------------------------------------------------


class TestBaselineJsonLoading:
    def test_bare_list(self, tmp_path: Path):
        p = tmp_path / "baseline.json"
        p.write_text(json.dumps(["tests/a.py::test_x"]), encoding="utf-8")
        assert _load_baseline_failing(p) == ["tests/a.py::test_x"]

    def test_baseline_object_with_failing_node_ids(self, tmp_path: Path):
        p = tmp_path / "baseline.json"
        p.write_text(
            json.dumps(
                {
                    "command": "pytest",
                    "passed": False,
                    "failing_node_ids": ["tests/a.py::test_x", "tests/b.py::test_y"],
                }
            ),
            encoding="utf-8",
        )
        assert _load_baseline_failing(p) == [
            "tests/a.py::test_x",
            "tests/b.py::test_y",
        ]

    def test_object_with_a_failing_list(self, tmp_path: Path):
        """The shape the deploy sidecar hands over."""
        p = tmp_path / "merge-baseline-FEAT-C1.json"
        p.write_text(
            json.dumps({"failing": ["tests/a.py::test_x"]}), encoding="utf-8"
        )
        assert _load_baseline_failing(p) == ["tests/a.py::test_x"]

    def test_object_without_failing_node_ids_raises(self, tmp_path: Path):
        p = tmp_path / "baseline.json"
        p.write_text(json.dumps({"command": "pytest"}), encoding="utf-8")
        with pytest.raises(ValueError):
            _load_baseline_failing(p)

    def test_scalar_raises(self, tmp_path: Path):
        p = tmp_path / "baseline.json"
        p.write_text(json.dumps(42), encoding="utf-8")
        with pytest.raises(ValueError):
            _load_baseline_failing(p)

    def test_missing_file_is_a_cli_usage_error(
        self, cli_runner, merge_repo
    ):
        result = cli_runner.invoke(
            merge,
            ["FEAT-C1", "--no-verify", "--baseline-json", "absent.json"],
        )
        # click.Path(exists=True) rejects it before the executor runs.
        assert result.exit_code == 2
        assert "absent.json" in result.output


# ---------------------------------------------------------------------------
# --measure-baseline / --no-measure-baseline
# ---------------------------------------------------------------------------


def _declare_toolchain_test(repo: Path, command: str) -> None:
    config = repo / ".guardkit" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "toolchain:\n" f"  test: {json.dumps(command)}\n", encoding="utf-8"
    )


@pytest.fixture
def red_main_repo(tmp_path: Path, monkeypatch) -> Path:
    """main carries one already-failing test; the branch adds a passing one."""
    repo = _init_repo(tmp_path)
    (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n.pytest_cache/\n", encoding="utf-8"
    )
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_green.py").write_text(
        "def test_green():\n    assert True\n", encoding="utf-8"
    )
    (tests / "test_already_red.py").write_text(
        "def test_already_red():\n    assert False\n", encoding="utf-8"
    )
    _declare_toolchain_test(
        repo, f'"{sys.executable}" -m pytest -q -p no:cacheprovider tests/'
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "already red on main")
    _git(repo, "checkout", "-q", "-b", "autobuild/FEAT-C1")
    (tests / "test_new_green.py").write_text(
        "def test_new_green():\n    assert True\n", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feature work")
    _git(repo, "checkout", "-q", "main")
    monkeypatch.chdir(repo)
    return repo


class TestMeasureBaselineOption:
    def test_the_baseline_is_measured_by_default(self, cli_runner, red_main_repo):
        result = cli_runner.invoke(merge, ["FEAT-C1", "--json"])

        data = json.loads(result.output)
        assert data["outcome"] == "merged"
        assert data["baseline_measured"]["failing"] == [
            "tests/test_already_red.py::test_already_red"
        ]
        assert data["baseline_measured"]["passed"] == 1
        assert data["verify_source"] == "repository toolchain declaration"
        # The already-red test is the target branch's, not this merge's.
        assert data["charged_failures"] == []

    def test_no_measure_baseline_charges_the_old_red(
        self, cli_runner, red_main_repo
    ):
        result = cli_runner.invoke(
            merge, ["FEAT-C1", "--json", "--no-measure-baseline"]
        )

        data = json.loads(result.output)
        assert data["outcome"] == "merged"
        assert data["baseline_measured"] is None
        assert data["charged_failures"] == [
            "tests/test_already_red.py::test_already_red"
        ]
        assert data["verify_status"] == "failed"
        assert result.exit_code == 4

    def test_a_supplied_baseline_file_turns_the_measurement_off(
        self, cli_runner, red_main_repo, tmp_path
    ):
        baseline = tmp_path / "baseline.json"
        baseline.write_text(
            json.dumps({"failing": ["tests/test_already_red.py::test_already_red"]}),
            encoding="utf-8",
        )

        result = cli_runner.invoke(
            merge, ["FEAT-C1", "--json", "--baseline-json", str(baseline)]
        )

        data = json.loads(result.output)
        assert data["baseline_measured"] is None
        assert data["charged_failures"] == []


# ---------------------------------------------------------------------------
# help text
# ---------------------------------------------------------------------------


class TestMergeHelp:
    def test_help_names_the_contract(self, cli_runner):
        result = cli_runner.invoke(merge, ["--help"])
        assert result.exit_code == 0
        assert "rollback path" in result.output
        assert "--expect-main-sha" in result.output
        assert "--no-verify" in result.output
        assert "--measure-baseline" in result.output
        assert "--no-measure-baseline" in result.output
