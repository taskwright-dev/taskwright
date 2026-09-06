"""Regression suite for TASK-AB-VERIFYCLI01 — ``--verify`` on
``guardkit autobuild complete``.

Pins AC-006:
(a) ``--verify`` triggers the post-completion test run;
(b) verify-failure → exit 4 + explicit failure text;
(c) no ``--verify`` → no test run, unchanged behaviour;
(d) a run that cannot start (no runner / zero tests) is UNVERIFIED, never a
    pass (absence-of-failure-is-not-success).
Plus: command resolution precedence (override > smoke command > stack
default), timeout = ran-and-failed, and display derives from the result
object.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from guardkit.orchestrator.completion_verification import (
    DEFAULT_VERIFY_TIMEOUT,
    VerificationResult,
    resolve_verify_command,
    run_completion_verification,
)


# ============================================================================
# Command resolution
# ============================================================================


class TestResolveVerifyCommand:
    def test_explicit_override_wins(self, tmp_path):
        command, source, profile = resolve_verify_command(
            tmp_path, smoke_command="pytest -q", override="make check"
        )
        assert command == "make check"
        assert "override" in source
        assert profile is None

    def test_smoke_command_beats_stack_default(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "tests").mkdir()
        command, source, _ = resolve_verify_command(
            tmp_path, smoke_command="pytest tests/smoke -q"
        )
        assert command == "pytest tests/smoke -q"
        assert "smoke_gates" in source

    def test_python_stack_default_uses_project_venv(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "tests").mkdir()
        venv_python = tmp_path / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.touch()
        command, source, _ = resolve_verify_command(tmp_path)
        # AC-003: the project's own interpreter, not guardkit's.
        assert str(venv_python) in command
        assert "-m pytest tests/" in command
        assert "venv" in source

    def test_non_python_stack_uses_registry_profile(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/x\n")
        command, source, profile = resolve_verify_command(tmp_path)
        assert command == "go test ./..."
        assert profile is not None and profile.stack == "go"

    def test_no_runner_detected_returns_none(self, tmp_path):
        command, source, profile = resolve_verify_command(tmp_path)
        assert command is None
        assert "no test runner" in source
        assert profile is None


# ============================================================================
# The repository's own declared test command (2026-09-06 spec, rule 7)
# ============================================================================


def _declare_toolchain_test(repo_root: Path, command: str) -> None:
    """Write the repository's ``toolchain: test:`` declaration for real."""
    import json as _json

    config = repo_root / ".guardkit" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "toolchain:\n" f"  test: {_json.dumps(command)}\n", encoding="utf-8"
    )


def _python_repo(repo_root: Path, with_venv: bool = True) -> None:
    """A repo that would otherwise resolve to the venv-pinned pytest default."""
    (repo_root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (repo_root / "tests").mkdir(exist_ok=True)
    if with_venv:
        venv_python = repo_root / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.touch()


class TestDeclaredToolchainTestCommand:
    def test_the_declaration_beats_the_venv_default(self, tmp_path):
        _python_repo(tmp_path)
        _declare_toolchain_test(tmp_path, "uv run --frozen pytest -q")

        command, source, profile = resolve_verify_command(tmp_path)

        assert command == "uv run --frozen pytest -q"
        assert source == "repository toolchain declaration"
        assert profile is None

    def test_the_declaration_beats_a_non_python_stack_default(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/x\n")
        _declare_toolchain_test(tmp_path, "go test ./... -count=1")

        command, source, _ = resolve_verify_command(tmp_path)

        assert command == "go test ./... -count=1"
        assert source == "repository toolchain declaration"

    def test_the_feature_smoke_command_still_beats_the_declaration(
        self, tmp_path
    ):
        _python_repo(tmp_path)
        _declare_toolchain_test(tmp_path, "uv run --frozen pytest -q")

        command, source, _ = resolve_verify_command(
            tmp_path, smoke_command="pytest tests/smoke -q"
        )

        assert command == "pytest tests/smoke -q"
        assert "smoke_gates" in source

    def test_the_override_still_beats_the_declaration(self, tmp_path):
        _python_repo(tmp_path)
        _declare_toolchain_test(tmp_path, "uv run --frozen pytest -q")

        command, source, _ = resolve_verify_command(
            tmp_path, override="make check"
        )

        assert command == "make check"
        assert "override" in source

    def test_no_declaration_falls_through_to_the_venv_default(self, tmp_path):
        _python_repo(tmp_path)
        (tmp_path / ".guardkit").mkdir()
        (tmp_path / ".guardkit" / "config.yaml").write_text(
            "autobuild:\n  coach:\n    contract: v4\n", encoding="utf-8"
        )

        command, source, _ = resolve_verify_command(tmp_path)

        assert "-m pytest tests/" in command
        assert "venv" in source

    def test_a_malformed_declaration_falls_through_rather_than_crashing(
        self, tmp_path
    ):
        _python_repo(tmp_path)
        (tmp_path / ".guardkit").mkdir()
        (tmp_path / ".guardkit" / "config.yaml").write_text(
            "toolchain:\n  tests: 'a typo for test'\n", encoding="utf-8"
        )

        command, source, _ = resolve_verify_command(tmp_path)

        assert "venv" in source
        assert command is not None

    def test_an_empty_declared_command_is_no_declaration(self, tmp_path):
        _python_repo(tmp_path)
        (tmp_path / ".guardkit").mkdir()
        (tmp_path / ".guardkit" / "config.yaml").write_text(
            "toolchain:\n  install: 'uv sync'\n", encoding="utf-8"
        )

        command, source, _ = resolve_verify_command(tmp_path)

        assert "venv" in source


class TestDeclaredCommandClassification:
    """Driven for real: the declared command actually runs."""

    def test_a_declared_pytest_command_is_read_by_the_pytest_rules(
        self, tmp_path
    ):
        _python_repo(tmp_path, with_venv=False)
        _declare_toolchain_test(
            tmp_path, '"/nowhere/there-is-no-python" -m pytest -q tests/'
        )
        command, source, profile = resolve_verify_command(tmp_path)

        result = run_completion_verification(
            tmp_path, command, source, stack_profile=profile, timeout=60
        )

        assert result.status == "unverified"
        assert result.detail == "test runner could not start"
        assert result.source == "repository toolchain declaration"

    def test_a_declared_non_pytest_command_is_read_by_the_exit_code_rule(
        self, tmp_path
    ):
        _python_repo(tmp_path, with_venv=False)
        _declare_toolchain_test(tmp_path, "true")
        command, source, profile = resolve_verify_command(tmp_path)

        result = run_completion_verification(
            tmp_path, command, source, stack_profile=profile, timeout=60
        )

        assert result.status == "passed"
        assert result.detail == "verification command exited 0"

    def test_a_declared_non_pytest_command_that_fails_is_failed(self, tmp_path):
        _python_repo(tmp_path, with_venv=False)
        _declare_toolchain_test(tmp_path, "false")
        command, source, profile = resolve_verify_command(tmp_path)

        result = run_completion_verification(
            tmp_path, command, source, stack_profile=profile, timeout=60
        )

        assert result.status == "failed"

    def test_a_declared_pytest_run_reports_a_real_pass(self, tmp_path):
        import sys as _sys

        _python_repo(tmp_path, with_venv=False)
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        (tmp_path / "tests" / "test_ok.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
        _declare_toolchain_test(
            tmp_path,
            f'"{_sys.executable}" -m pytest -q -p no:cacheprovider tests/',
        )
        command, source, profile = resolve_verify_command(tmp_path)

        result = run_completion_verification(
            tmp_path, command, source, stack_profile=profile, timeout=300
        )

        assert result.status == "passed"
        assert "1 passed" in result.detail


class TestVerificationResultCarriesItsSource:
    def test_the_source_rides_the_result(self, tmp_path):
        result = run_completion_verification(
            tmp_path, None, "no test runner detected"
        )
        assert result.source == "no test runner detected"


# ============================================================================
# Run classification (absence-of-failure safety)
# ============================================================================


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args="cmd", returncode=returncode, stdout=stdout, stderr=stderr
    )


class TestRunCompletionVerification:
    def test_no_command_is_unverified_never_pass(self, tmp_path):
        result = run_completion_verification(
            tmp_path, None, "no test runner detected"
        )
        assert result.status == "unverified"
        assert result.returncode is None

    def test_pytest_pass_requires_positive_passed_count(self, tmp_path):
        with patch("subprocess.run", return_value=_completed(0, "34 passed in 0.18s")):
            result = run_completion_verification(
                tmp_path, "pytest tests/", "python stack default"
            )
        assert result.status == "passed"
        assert "34 passed" in result.detail

    def test_pytest_clean_exit_with_no_evidence_is_unverified(self, tmp_path):
        with patch("subprocess.run", return_value=_completed(0, "warning only")):
            result = run_completion_verification(
                tmp_path, "pytest tests/", "python stack default"
            )
        assert result.status == "unverified"

    def test_pytest_exit_5_zero_collected_is_unverified(self, tmp_path):
        with patch("subprocess.run", return_value=_completed(5, "no tests ran")):
            result = run_completion_verification(
                tmp_path, "pytest tests/", "python stack default"
            )
        assert result.status == "unverified"

    def test_pytest_failure_is_failed(self, tmp_path):
        with patch(
            "subprocess.run", return_value=_completed(1, "1 failed, 33 passed")
        ):
            result = run_completion_verification(
                tmp_path, "pytest tests/", "python stack default"
            )
        assert result.status == "failed"

    def test_command_not_found_is_unverified(self, tmp_path):
        with patch(
            "subprocess.run", return_value=_completed(127, "", "sh: pytest: not found")
        ):
            result = run_completion_verification(
                tmp_path, "pytest tests/", "python stack default"
            )
        assert result.status == "unverified"

    def test_timeout_is_ran_and_failed(self, tmp_path):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=600),
        ):
            result = run_completion_verification(
                tmp_path, "pytest tests/", "python stack default"
            )
        # Runtime-parity L3 precedent: a suite that hangs is a real defect.
        assert result.status == "failed"
        assert "timed out" in result.detail

    def test_oserror_is_unverified(self, tmp_path):
        with patch("subprocess.run", side_effect=OSError("no shell")):
            result = run_completion_verification(
                tmp_path, "pytest tests/", "python stack default"
            )
        assert result.status == "unverified"

    def test_stack_profile_absent_signal_is_unverified(self, tmp_path):
        from guardkit.orchestrator.quality_gates.stack_test_execution import (
            STACK_TEST_PROFILES,
        )

        dotnet = next(p for p in STACK_TEST_PROFILES if p.stack == "dotnet")
        with patch(
            "subprocess.run",
            return_value=_completed(1, "No test is available in the project"),
        ):
            result = run_completion_verification(
                tmp_path, "dotnet test", "dotnet stack default", stack_profile=dotnet
            )
        assert result.status == "unverified"

    def test_custom_command_exit_zero_passes(self, tmp_path):
        with patch("subprocess.run", return_value=_completed(0, "ok")):
            result = run_completion_verification(
                tmp_path, "make check", "--verify-cmd override"
            )
        assert result.status == "passed"

    def test_custom_command_nonzero_fails(self, tmp_path):
        with patch("subprocess.run", return_value=_completed(2, "", "boom")):
            result = run_completion_verification(
                tmp_path, "make check", "--verify-cmd override"
            )
        assert result.status == "failed"


# ============================================================================
# CLI wiring (AC-001, AC-002, AC-006 a/b/c)
# ============================================================================


def _mock_complete_orchestrator(success: bool = True):
    result = SimpleNamespace(
        feature_id="FEAT-TEST",
        success=success,
        status="completed" if success else "failed",
        tasks_completed=1,
        total_tasks=1,
        worktree_path=None,
        error=None,
    )
    orchestrator = MagicMock()
    orchestrator.complete.return_value = result
    return orchestrator


class TestCompleteCliVerifyFlag:
    def _invoke(self, args, verification=None, orchestrator=None):
        from contextlib import ExitStack

        from guardkit.cli import autobuild as cli_mod

        runner = CliRunner()
        orchestrator = orchestrator or _mock_complete_orchestrator()
        run_mock = MagicMock(return_value=verification)
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    cli_mod,
                    "FeatureCompleteOrchestrator",
                    return_value=orchestrator,
                )
            )
            stack.enter_context(
                patch.object(
                    cli_mod,
                    "_resolve_completion_verification",
                    return_value=("pytest tests/", "python stack default", None),
                )
            )
            stack.enter_context(
                patch(
                    "guardkit.orchestrator.completion_verification."
                    "run_completion_verification",
                    run_mock,
                )
            )
            result = runner.invoke(cli_mod.complete, args, catch_exceptions=False)
        return result, run_mock

    def test_verify_triggers_post_completion_test_run(self):
        verification = VerificationResult(
            status="passed",
            command="pytest tests/",
            cwd="/repo",
            returncode=0,
            detail="34 passed",
        )
        result, run_mock = self._invoke(
            ["FEAT-TEST", "--verify"], verification=verification
        )
        assert run_mock.called  # AC-006(a): the test run actually happened
        assert result.exit_code == 0
        assert "verification passed" in result.output.lower()

    def test_verify_failure_exits_4_with_failure_text(self):
        verification = VerificationResult(
            status="failed",
            command="pytest tests/",
            cwd="/repo",
            returncode=1,
            detail="test run failed (exit 1)",
        )
        result, _ = self._invoke(["FEAT-TEST", "--verify"], verification=verification)
        assert result.exit_code == 4  # AC-006(b)
        assert "FAILED" in result.output
        assert "verification passed" not in result.output.lower()

    def test_unverified_exits_4_never_prints_success(self):
        verification = VerificationResult(
            status="unverified",
            command="",
            cwd="/repo",
            returncode=None,
            detail="UNVERIFIED: no test runner detected",
        )
        result, _ = self._invoke(["FEAT-TEST", "--verify"], verification=verification)
        assert result.exit_code == 4  # AC-006(d): never a pass
        assert "UNVERIFIED" in result.output
        assert "verification passed" not in result.output.lower()

    def test_without_verify_no_test_run_and_exit_zero(self):
        result, run_mock = self._invoke(["FEAT-TEST"])
        assert result.exit_code == 0  # AC-002: additive flag
        assert not run_mock.called  # AC-006(c): no test run
        assert "verification" not in result.output.lower()

    def test_verify_cmd_implies_verify(self):
        verification = VerificationResult(
            status="passed",
            command="make check",
            cwd="/repo",
            returncode=0,
            detail="verification command exited 0",
        )
        result, run_mock = self._invoke(
            ["FEAT-TEST", "--verify-cmd", "make check"], verification=verification
        )
        assert run_mock.called
        assert result.exit_code == 0

    def test_dry_run_previews_but_does_not_run(self):
        result, run_mock = self._invoke(["FEAT-TEST", "--verify", "--dry-run"])
        assert result.exit_code == 0
        assert not run_mock.called
        assert "would verify" in result.output.lower()
