"""Post-completion test verification for ``guardkit autobuild complete --verify``.

TASK-AB-VERIFYCLI01. R3 of the 2026-07-04 retro cross-reference showed
post-merge verification is a real aperture: ``after_wave: [2,3,4]`` left the
final waves ungated and ``tests/`` outside every gate, so nothing re-ran the
suite at completion. The slash-command workflow documented a ``--verify`` step
the Python CLI never implemented; this module is the ONE implementation both
entry points share (``cli-wrapper-shares-client-acquisition-path.md`` — a
CLI-only second implementation is exactly the divergence that rule documents).

Verification semantics (absence-of-failure-safe):

- ``passed`` requires POSITIVE evidence tests ran (``tests_run > 0``) on the
  runners this module understands: a pytest run must show a passed-count; a
  stack-profile run must not classify as absent. A runner that could not
  start, collected zero tests, or produced no evidence any test executed is
  ``unverified`` — never a pass (``absence-of-failure-is-not-success.md``).
  An operator-supplied CUSTOM command (``--verify-cmd`` / a non-pytest smoke
  command) is trusted on exit code alone — its output shape is unknowable
  here, matching the smoke-gate exit-code semantics.
- A timeout is ``failed`` (ran-and-hung is a real defect — the
  runtime-parity L3 precedent), never absent.
- The command itself is resolved in one place (``resolve_verify_command``):
  an explicit override, then the feature's smoke command, then the
  repository's own declared test command (``.guardkit/config.yaml``,
  ``toolchain.test``), then a stack default.
- The verification runs as a SUBPROCESS in the merge-target repo with the
  project's own interpreter/venv where one exists — not guardkit's — so a
  merged-in missing dependency cannot be masked by guardkit's environment
  (``namespace-hygiene.md``).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from guardkit.orchestrator.quality_gates.stack_test_execution import (
    StackTestProfile,
    classify_absent_for_stack,
    detect_stack_profile,
)

logger = logging.getLogger(__name__)

__all__ = [
    "VerificationResult",
    "declared_toolchain_test_command",
    "resolve_verify_command",
    "run_completion_verification",
    "sweep_completion_against_ledger",
]

# Full-suite verification needs more headroom than the 120s between-wave
# smoke default; 600s matches the smoke-gate schema's upper bound.
DEFAULT_VERIFY_TIMEOUT = 600

_ABSENT_RETURNCODES = frozenset({126, 127})

# pytest's "N passed" summary — the positive tests-ran evidence.
_PYTEST_PASSED_RE = re.compile(r"\b[1-9]\d*\s+passed\b")


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of a post-completion verification run.

    ``status`` is the enforcement source every display line must derive from
    (``display-must-derive-from-enforcement-source-not-proxy.md``): "verified"
    output is only legitimate when ``status == "passed"``, never inferred
    from "the completion succeeded".
    """

    status: str  # "passed" | "failed" | "unverified"
    command: str
    cwd: str
    returncode: Optional[int]
    detail: str
    output_tail: str = ""
    # Where the command came from — the same words ``resolve_verify_command``
    # returns, carried through so a receipt can say what was run and why that
    # command and not another.
    source: str = ""


def _project_python(repo_root: Path) -> Optional[Path]:
    """Return the project's own venv interpreter, if one exists."""
    for candidate in (
        repo_root / ".venv" / "bin" / "python",
        repo_root / "venv" / "bin" / "python",
        repo_root / ".venv" / "Scripts" / "python.exe",
        repo_root / "venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return candidate
    return None


def _looks_python(repo_root: Path) -> bool:
    return any(
        (repo_root / marker).exists()
        for marker in ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini")
    ) and (repo_root / "tests").is_dir()


def declared_toolchain_test_command(repo_root: Path) -> Optional[str]:
    """The repository's own declared test command, or ``None``.

    Reads ``<repo_root>/.guardkit/config.yaml``, the top-level ``toolchain:``
    block, through the one reader the rest of guardkit uses
    (``toolchain_declaration.load_toolchain_declaration``) — so a repository
    states how its tests are run in exactly one place, and this module does
    not grow a second, drifting copy of that rule.

    Never raises: a missing file, a malformed block, or a missing dependency
    all mean "no declaration", and resolution carries on down the list.
    """
    try:
        from guardkit.orchestrator.toolchain_declaration import (
            load_toolchain_declaration,
        )
    except ImportError:  # pragma: no cover - guardkit always ships this module
        return None
    try:
        declaration = load_toolchain_declaration(Path(repo_root))
    except Exception as exc:  # noqa: BLE001 - resolution never crashes a merge
        logger.warning(
            "Could not read the toolchain declaration in %s: %s", repo_root, exc
        )
        return None
    if declaration is None:
        return None
    command = (declaration.test or "").strip()
    return command or None


def resolve_verify_command(
    repo_root: Path,
    smoke_command: Optional[str] = None,
    override: Optional[str] = None,
) -> Tuple[Optional[str], str, Optional[StackTestProfile]]:
    """Resolve the verification command for ``repo_root``.

    Precedence: explicit ``--verify-cmd`` override > the feature YAML's
    ``smoke_gates.command`` > the repository's own declared test command
    (``.guardkit/config.yaml``, ``toolchain.test``) > a stack-aware default
    (venv-pinned pytest for Python; the ``stack_test_execution`` registry for
    .NET/JS-TS/Go).

    The declared command sits above the stack defaults because a repository
    that has said how to run its tests has said it for a reason — the venv
    default is a guess, and guessing is what sent a merge check looking for an
    interpreter that was not there (2026-09-06 spec, rule 7).

    Returns ``(command, source, stack_profile)``; ``command`` is ``None``
    when no runner could be determined (the caller reports UNVERIFIED —
    never a pass). A declared command carries no stack profile, so its exit
    code is read by the pytest rules when the command mentions pytest and by
    the plain exit-code rule otherwise.
    """
    if override:
        return override, "--verify-cmd override", None
    if smoke_command:
        return smoke_command, "feature smoke_gates.command", None
    declared = declared_toolchain_test_command(repo_root)
    if declared:
        return declared, "repository toolchain declaration", None
    if _looks_python(repo_root):
        python = _project_python(repo_root)
        if python is not None:
            return f'"{python}" -m pytest tests/', "python stack default (project venv)", None
        logger.warning(
            "TASK-AB-VERIFYCLI01: no project venv found at %s — falling back "
            "to PATH pytest, which may resolve from guardkit's environment",
            repo_root,
        )
        return "pytest tests/", "python stack default (PATH pytest)", None
    profile = detect_stack_profile(repo_root)
    if profile is not None:
        return (
            profile.whole_suite_command,
            f"{profile.stack} stack default",
            profile,
        )
    return None, "no test runner detected", None


def _classify_pytest(returncode: int, output: str) -> Tuple[str, str]:
    if returncode in _ABSENT_RETURNCODES:
        return "unverified", "test runner could not start"
    if returncode == 5:
        return "unverified", "zero tests collected (pytest exit 5)"
    if returncode == 4:
        return "unverified", "pytest usage error (exit 4)"
    if returncode == 0:
        match = _PYTEST_PASSED_RE.search(output)
        if match:
            return "passed", match.group(0)
        # Clean exit with no evidence any test ran — absent, never a pass.
        return "unverified", "clean exit but no evidence any test ran"
    return "failed", f"test run failed (exit {returncode})"


def _classify_stack(
    profile: StackTestProfile, returncode: int, output: str
) -> Tuple[str, str]:
    if classify_absent_for_stack(profile, returncode, output):
        return "unverified", f"{profile.stack} run produced no test verdict"
    if returncode == 0:
        return "passed", f"{profile.stack} suite passed"
    return "failed", f"{profile.stack} suite failed (exit {returncode})"


def _classify_custom(returncode: int) -> Tuple[str, str]:
    if returncode in _ABSENT_RETURNCODES:
        return "unverified", "verification command could not start"
    if returncode == 0:
        return "passed", "verification command exited 0"
    return "failed", f"verification command failed (exit {returncode})"


def run_completion_verification(
    repo_root: Path,
    command: Optional[str],
    source: str,
    stack_profile: Optional[StackTestProfile] = None,
    timeout: int = DEFAULT_VERIFY_TIMEOUT,
) -> VerificationResult:
    """Run the resolved verification command in the merge-target repo.

    Never raises: every outcome (including a missing command, a timeout, or a
    subprocess error) is a :class:`VerificationResult`. Only ``passed`` may
    ever be rendered as success.
    """
    cwd = str(repo_root)
    if not command:
        return VerificationResult(
            status="unverified",
            command="",
            cwd=cwd,
            returncode=None,
            detail=f"UNVERIFIED: {source}",
            source=source,
        )

    logger.info(
        "Post-completion verification (%s): %s (cwd=%s, timeout=%ds)",
        source,
        command,
        cwd,
        timeout,
    )
    try:
        # TS-lane D.1b (design §B.5): the second declared-command shell site.
        # ``env=`` is now EXPLICIT (a verbatim copy of the daemon's own
        # environment — byte-equivalent to the previous implicit inherit) so
        # the resolution rule is visible where the command runs: worktree/repo
        # cwd + the daemon PATH, on which node resolves ONLY via the D.0
        # symlink fence. Never source nvm here.
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        # Ran-and-hung is a genuine defect (runtime-parity L3 precedent):
        # a suite that hangs post-merge must block, never read as absent.
        stdout = exc.stdout or b""
        tail = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else str(stdout)
        return VerificationResult(
            status="failed",
            command=command,
            cwd=cwd,
            returncode=None,
            detail=f"verification timed out after {timeout}s",
            output_tail=tail[-2000:],
            source=source,
        )
    except OSError as exc:
        return VerificationResult(
            status="unverified",
            command=command,
            cwd=cwd,
            returncode=None,
            detail=f"verification command could not start: {exc}",
            source=source,
        )

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if stack_profile is not None:
        status, detail = _classify_stack(stack_profile, proc.returncode, output)
    elif "pytest" in command:
        status, detail = _classify_pytest(proc.returncode, output)
    else:
        status, detail = _classify_custom(proc.returncode)

    return VerificationResult(
        status=status,
        command=command,
        cwd=cwd,
        returncode=proc.returncode,
        detail=detail,
        output_tail=output[-2000:],
        source=source,
    )


# ---------------------------------------------------------------------------
# Post-merge known-failure ledger sweep (WS2 session B2)
# ---------------------------------------------------------------------------
#
# WS2 build-plan §B2 ownership note (2026-07-07): the ledger-sweep enforcement
# lives in THIS module and is built ONCE, here (WS3-S4 is the same deliverable —
# no double-dispatch). It diffs the completion suite result against the F2
# ledger ``qa/known-failures.yaml``: an un-ledgered failure fails the gate; an
# unexpectedly-passing unconditional ledger entry fails the gate. The pure diff
# + pytest-output parsing live in ``guardkit.qa.enforcement`` (kept there so the
# schema/enforcement library is unit-testable without the CLI); this function is
# the completion-time consumer. Flag-gated by ``qa.enforce_tier1`` at the CALL
# site (``cli/autobuild.py``), default OFF — this function itself is pure and
# does not read the flag, so it stays testable in isolation.


def sweep_completion_against_ledger(
    repo_root: Path,
    verification: "VerificationResult",
    ledger_path: Optional[Path] = None,
):
    """Diff a completion :class:`VerificationResult` against the F2 ledger.

    ``ledger_path`` defaults to ``<repo_root>/qa/known-failures.yaml``. Returns
    a ``guardkit.qa.enforcement.LedgerSweepResult``:

    - The suite output is parsed for failing node ids + summary counts.
    - A run with no positive evidence it executed (``unverified``) yields
      ``status="unverified"`` — never a fail (absence-of-failure-is-not-success).
    - Otherwise the parsed failures are diffed against the ledger (un-ledgered
      failure or stale unconditional entry ⇒ ``status="fail"``).

    Custom / non-pytest verification commands emit output shapes this parser
    does not understand; when the verification status is ``passed``/``failed``
    on a custom command, the sweep only has the exit-code verdict to go on and
    returns ``unverified`` for the ledger dimension (the exit-code verdict from
    ``run_completion_verification`` still stands on its own).
    """
    from guardkit.qa.enforcement import (
        LedgerSweepResult,
        diff_failures_against_ledger,
        parse_pytest_outcome,
    )

    ledger_path = ledger_path or (repo_root / "qa" / "known-failures.yaml")

    # A non-pytest / custom command's output is not ledger-diffable here (its
    # per-test shape is unknowable — the same reason its verdict is exit-code
    # only). Report the ledger dimension as unverified, never a fail.
    is_pytest_shaped = "pytest" in (verification.command or "") or _PYTEST_PASSED_RE.search(
        verification.output_tail or ""
    )
    if not is_pytest_shaped:
        return LedgerSweepResult(
            status="unverified",
            unledgered_failures=(),
            stale_ledger_entries=(),
            detail=(
                "ledger sweep not applicable: verification used a non-pytest "
                "command whose per-test output shape is unknowable"
            ),
        )

    outcome = parse_pytest_outcome(verification.output_tail)
    return diff_failures_against_ledger(outcome, ledger_path)
