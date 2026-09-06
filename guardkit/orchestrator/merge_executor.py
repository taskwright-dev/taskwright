"""The merge primitive behind the merge word (make-merge-work, 2026-08-24).

Spec: ``docs/make-merge-work-build-spec-2026-08-24.md`` (ai-transition). Today
"merge" is a phrase — the build-complete message says the merge word is Rich's,
but nothing listens. This module is the mechanism: code — never an AI session —
merges ``autobuild/<FEATURE_ID>`` into the target branch, refuses loudly when
anything is off, and re-checks the merged result.

House pattern: ``machine_verify.py`` — pure functions, explicit inputs, one
frozen dataclass report with ``to_dict()`` and ``receipt_lines()``.

The three laws this module enforces:

* **Refuse, never half-do.** A dirty tree, a missing branch, or a target that
  has moved since the checks ran each refuse the merge before anything is
  touched.
* **The branch survives EVERY path.** ``autobuild/<FEATURE_ID>`` is the
  rollback path. This module never calls ``manager.cleanup()``, never deletes
  a branch, and on conflict aborts the merge and leaves the tree exactly as it
  found it.
* **Never invent a clean.** Post-merge verification only ever reports what it
  observed; with no pre-merge baseline the full observed failure set is
  reported with a note saying the diff is unavailable.
* **Judge no new red, not no red.** When no baseline is handed in, the merge
  measures one itself: the SAME resolved test command is run on the target
  branch before the merge, and its failures are what the merged result is
  excused for (2026-09-06 spec, rule 6). A baseline run that could not start
  makes the whole check "could not run" — it never becomes an empty baseline,
  because an empty baseline would charge this merge for red the branch never
  caused.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from guardkit.orchestrator.baseline import (
    compute_charged_failures,
    failing_node_ids,
    load_known_failure_ids,
)
from guardkit.orchestrator.completion_verification import (
    DEFAULT_VERIFY_TIMEOUT,
    VerificationResult,
    resolve_verify_command,
    run_completion_verification,
)
from guardkit.orchestrator.quality_gates.stack_test_execution import (
    StackTestProfile,
)
from guardkit.worktrees.manager import (
    Worktree,
    WorktreeManager,
    WorktreeMergeError,
)

logger = logging.getLogger(__name__)

# --- outcome constants (module-level so callers compare without the dataclass)
OUTCOME_MERGED = "merged"
OUTCOME_REFUSED = "refused"
OUTCOME_CONFLICT = "conflict"

_GIT_TIMEOUT_SECONDS = 60
_VALIDATE_TIMEOUT_SECONDS = 120

# The tail of a pytest summary line: "... 118 passed, 8 failed in 4.21s".
_PASSED_COUNT_RE = re.compile(r"\b(\d+)\s+passed\b")


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasuredBaseline:
    """What the target branch's own tests did BEFORE this merge touched it.

    Measured by running the SAME resolved test command on the target branch,
    so the merged result is judged on "no new red" rather than "no red".
    ``ran`` is False when the command could not start at all — in that case
    there is no baseline, and the whole check is reported as "could not run"
    rather than being quietly treated as a clean baseline.
    """

    command: str
    source: str
    failing: Tuple[str, ...] = ()
    passed: int = 0
    ran: bool = False
    detail: str = ""
    stack_profile: Optional[StackTestProfile] = None

    def to_dict(self) -> dict:
        """The three fields the merge report publishes."""
        return {
            "failing": list(self.failing),
            "passed": self.passed,
            "command": self.command,
        }


@dataclass(frozen=True)
class MergeReport:
    """The one receipt-bearing report the merge primitive emits."""

    outcome: str  # OUTCOME_MERGED | OUTCOME_REFUSED | OUTCOME_CONFLICT
    feature_id: str
    target_branch: str
    branch: str  # autobuild/<FEATURE_ID> — retained on every path
    refusal_reason: Optional[str] = None
    pre_sha: Optional[str] = None
    post_sha: Optional[str] = None
    conflict_files: Tuple[str, ...] = ()
    verify_ran: bool = False
    # The MERGE's verdict on the tests, not the raw exit code of the run:
    # "passed" means nothing is charged to this merge, "failed" means at
    # least one failure is, and "unverified" means the tests could not be
    # run at all. The raw run's own word is kept in ``verify_suite_status``.
    verify_status: Optional[str] = None  # "passed" | "failed" | "unverified"
    verify_detail: str = ""
    # The test run's own verdict, before the baseline was subtracted.
    verify_suite_status: Optional[str] = None
    validate_valid: Optional[bool] = None
    charged_failures: Tuple[str, ...] = ()
    # The target branch's failures measured before the merge, or None when no
    # baseline was measured (one was handed in, measuring was turned off, or
    # the tests could not be run).
    baseline_measured: Optional[MeasuredBaseline] = None
    verify_command: Optional[str] = None
    verify_source: Optional[str] = None
    notes: Tuple[str, ...] = ()

    @property
    def verify_ok(self) -> bool:
        """True only when every post-merge check positively passed.

        Requires the merge's test verdict to be ``passed``, the feature YAML
        to validate, and zero charged failures. Absence of evidence is never
        a pass (absence-of-failure-is-not-success): a run that could not
        start is ``unverified`` and fails this test, and a run that failed on
        something this merge is charged for is ``failed``. A run that failed
        only on failures the target branch already had is ``passed`` — that
        is the "no new red" rule, and it is stated once, in
        :func:`merge_verdict_from_run`.
        """
        return (
            self.verify_ran
            and self.verify_status == "passed"
            and self.validate_valid is True
            and not self.charged_failures
        )

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "feature_id": self.feature_id,
            "target_branch": self.target_branch,
            "branch": self.branch,
            "refusal_reason": self.refusal_reason,
            "pre_sha": self.pre_sha,
            "post_sha": self.post_sha,
            "conflict_files": list(self.conflict_files),
            "verify_ran": self.verify_ran,
            "verify_status": self.verify_status,
            "verify_detail": self.verify_detail,
            "verify_suite_status": self.verify_suite_status,
            "validate_valid": self.validate_valid,
            "charged_failures": list(self.charged_failures),
            "verify_ok": self.verify_ok,
            "baseline_measured": (
                self.baseline_measured.to_dict()
                if self.baseline_measured is not None
                else None
            ),
            "verify_command": self.verify_command,
            "verify_source": self.verify_source,
            "notes": list(self.notes),
        }

    def receipt_lines(self) -> List[str]:
        """Plain sentences describing what happened — no jargon, no colour."""
        lines: List[str] = []
        if self.outcome == OUTCOME_REFUSED:
            lines.append(
                f"The merge of {self.feature_id} was refused before anything "
                f"was touched."
            )
            if self.refusal_reason:
                lines.append(f"Reason: {self.refusal_reason}")
        elif self.outcome == OUTCOME_CONFLICT:
            lines.append(
                f"The merge of branch {self.branch} into {self.target_branch} "
                f"hit conflicts, so it was aborted and the tree was left clean."
            )
            if self.conflict_files:
                lines.append("Files in conflict:")
                lines.extend(f"  - {p}" for p in self.conflict_files)
            lines.append(
                f"Branch {self.branch} is untouched and remains the rollback "
                f"path."
            )
        else:
            lines.append(
                f"{self.feature_id} merged into {self.target_branch}: "
                f"{(self.pre_sha or '?')[:12]} -> {(self.post_sha or '?')[:12]}."
            )
            lines.append(
                f"Branch {self.branch} is kept as the rollback path."
            )
            if not self.verify_ran:
                lines.append(
                    "The merged result was NOT verified (verification was "
                    "turned off for this run)."
                )
            else:
                if self.validate_valid is True:
                    lines.append("The feature file validates.")
                elif self.validate_valid is False:
                    lines.append("The feature file does NOT validate.")
                else:
                    lines.append(
                        "The feature file validation gave no usable answer."
                    )
                if self.verify_command:
                    lines.append(
                        f"The tests were run with: {self.verify_command} "
                        f"(that command came from the {self.verify_source})."
                    )
                if self.baseline_measured is not None:
                    measured = self.baseline_measured
                    if measured.failing:
                        lines.append(
                            f"Before the merge, the same command was run on "
                            f"{self.target_branch}: "
                            f"{len(measured.failing)} test(s) were already "
                            f"failing there and {measured.passed} passed. "
                            f"Those are not counted against this merge."
                        )
                    else:
                        lines.append(
                            f"Before the merge, the same command was run on "
                            f"{self.target_branch}: nothing was failing there "
                            f"({measured.passed} test(s) passed)."
                        )
                if self.verify_status == "passed":
                    if self.verify_suite_status == "failed":
                        lines.append(
                            f"The merged result fails only tests that "
                            f"{self.target_branch} was already failing, so "
                            f"nothing is charged to this merge "
                            f"({self.verify_detail})."
                        )
                    else:
                        lines.append(
                            f"The test suite passed on the merged result "
                            f"({self.verify_detail})."
                        )
                elif self.verify_status == "failed":
                    lines.append(
                        f"The test suite FAILED on the merged result "
                        f"({self.verify_detail})."
                    )
                else:
                    lines.append(
                        f"The test suite could not be verified "
                        f"({self.verify_detail}). This is not a pass."
                    )
                if self.charged_failures:
                    lines.append(
                        f"{len(self.charged_failures)} failing test(s) are "
                        f"charged to this merge (not excused by the baseline "
                        f"or the known-failures ledger):"
                    )
                    lines.extend(f"  - {n}" for n in self.charged_failures)
        for note in self.notes:
            lines.append(f"Note: {note}")
        return lines


# ---------------------------------------------------------------------------
# git plumbing (explicit, timeout-bounded, never raises past the boundary)
# ---------------------------------------------------------------------------


def _run_git(
    repo_root: Path, *args: str
) -> subprocess.CompletedProcess:
    """Run one git command in ``repo_root``; never raises on non-zero exit."""
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
    )


def _rev_parse(repo_root: Path, ref: str) -> Optional[str]:
    proc = _run_git(repo_root, "rev-parse", "--verify", "--quiet", ref)
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def _porcelain_status(repo_root: Path) -> Optional[str]:
    """``git status --porcelain`` output, or None when git itself failed."""
    proc = _run_git(repo_root, "status", "--porcelain")
    if proc.returncode != 0:
        return None
    return proc.stdout


def conflicted_files_from_status(porcelain: str) -> List[str]:
    """Paths from the ``UU`` rows of ``git status --porcelain`` output."""
    files: List[str] = []
    for line in porcelain.splitlines():
        if line.startswith("UU "):
            files.append(line[3:].strip())
    return files


def current_branch(repo_root: Path) -> Optional[str]:
    """The branch HEAD is on, or ``None`` when HEAD is detached/unreadable."""
    proc = _run_git(repo_root, "symbolic-ref", "--quiet", "--short", "HEAD")
    if proc.returncode != 0:
        return None
    name = proc.stdout.strip()
    return name or None


def checkout_branch(repo_root: Path, branch: str) -> Optional[str]:
    """Put HEAD on ``branch``. Returns a plain reason on failure, else None.

    A no-op when HEAD is already there. The caller has already refused a
    dirty tree, so this moves nothing a person had not committed.
    """
    if current_branch(repo_root) == branch:
        return None
    proc = _run_git(repo_root, "checkout", branch)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or "(no output)"
        return f"could not switch to branch {branch}: {detail}"
    return None


def passed_count_from_output(output: Optional[str]) -> int:
    """How many tests the runner said passed; 0 when it did not say.

    Reads the last ``N passed`` in the output — pytest's summary line. A
    runner whose output shape we do not read simply reports 0, which is a
    count, not a claim about the run.
    """
    matches = _PASSED_COUNT_RE.findall(output or "")
    if not matches:
        return 0
    try:
        return int(matches[-1])
    except ValueError:  # pragma: no cover - the regex only matches digits
        return 0


# ---------------------------------------------------------------------------
# Refusal preflight
# ---------------------------------------------------------------------------


def preflight_refusal(
    repo_root: Path,
    feature_id: str,
    target_branch: str = "main",
    expect_target_sha: Optional[str] = None,
) -> Optional[str]:
    """Return the refusal reason, or None when the merge may proceed.

    Refusals (each checked before anything is touched):

    * ``repo_root`` is not a git repository;
    * the working tree is dirty (``git status --porcelain`` non-empty);
    * branch ``autobuild/<FEATURE_ID>`` does not exist;
    * ``expect_target_sha`` is given and the target branch no longer resolves
      to it — the checks were run against a target that has since moved.
    """
    proc = _run_git(repo_root, "rev-parse", "--git-dir")
    if proc.returncode != 0:
        return f"{repo_root} is not a git repository"

    porcelain = _porcelain_status(repo_root)
    if porcelain is None:
        return "git status could not be read"
    if porcelain.strip():
        dirty = [ln for ln in porcelain.splitlines() if ln.strip()]
        shown = "; ".join(dirty[:5])
        more = f" (and {len(dirty) - 5} more)" if len(dirty) > 5 else ""
        return (
            f"the working tree is dirty — refuse to merge over uncommitted "
            f"changes: {shown}{more}"
        )

    branch = f"autobuild/{feature_id}"
    if _rev_parse(repo_root, f"refs/heads/{branch}") is None:
        return f"branch {branch} does not exist"

    if expect_target_sha:
        actual = _rev_parse(repo_root, target_branch)
        if actual is None:
            return f"branch {target_branch} does not exist"
        expected = expect_target_sha.strip()
        matches = actual == expected or (
            len(expected) >= 7 and actual.startswith(expected)
        )
        if not matches:
            return (
                f"{target_branch} has moved since the checks ran "
                f"(expected {expected}, found {actual})"
            )

    return None


# ---------------------------------------------------------------------------
# The merge itself
# ---------------------------------------------------------------------------


def merge_commit_message(
    feature_id: str, pre_sha: str, branch_sha: str
) -> str:
    """The template message, filled ONLY from the build's own records.

    ``pre_sha`` is the target branch head before the merge; ``branch_sha`` is
    the tip of ``autobuild/<FEATURE_ID>`` being merged (the merge commit
    cannot carry its own sha, so the range in the message is
    target-before..branch-tip). No model writes this.
    """
    return (
        f"merge({feature_id}): merged on the merge word\n\n"
        f"{pre_sha[:12]}..{branch_sha[:12]} — branch autobuild/{feature_id} "
        f"retained as the rollback path"
    )


def perform_merge(
    repo_root: Path,
    feature_id: str,
    target_branch: str = "main",
    manager: Optional[WorktreeManager] = None,
) -> MergeReport:
    """Merge ``autobuild/<FEATURE_ID>`` into ``target_branch``.

    On :class:`WorktreeMergeError`: capture the conflicted files, run
    ``git merge --abort`` (its own failure is ignored), re-verify the tree is
    clean, and report outcome ``conflict``. The branch is NEVER deleted on any
    path — no ``cleanup()``, no ``auto_merge_if_graduated``, no
    preserve-then-delete.
    """
    branch = f"autobuild/{feature_id}"
    notes: List[str] = []

    pre_sha = _rev_parse(repo_root, target_branch)
    branch_sha = _rev_parse(repo_root, branch)
    if pre_sha is None or branch_sha is None:
        return MergeReport(
            outcome=OUTCOME_REFUSED,
            feature_id=feature_id,
            target_branch=target_branch,
            branch=branch,
            refusal_reason=(
                f"could not resolve {target_branch} and {branch} to commits"
            ),
        )

    if manager is None:
        manager = WorktreeManager(repo_root=Path(repo_root))

    # Only task_id and branch_name are read by ``manager.merge``; the path
    # need not exist (the ``_find_worktree`` reconstruction pattern —
    # cli/autobuild.py). The feature file may already be archived, so nothing
    # here reads the feature YAML either.
    worktree = Worktree(
        task_id=feature_id,
        branch_name=branch,
        path=Path(repo_root) / ".guardkit" / "worktrees" / feature_id,
        base_branch=target_branch,
    )

    message = merge_commit_message(feature_id, pre_sha, branch_sha)

    try:
        manager.merge(worktree, target_branch=target_branch, message=message)
    except WorktreeMergeError as exc:
        # Conflict path: capture the UU rows BEFORE aborting (the abort wipes
        # them), then abort and re-verify the tree is clean.
        status_before_abort = _porcelain_status(repo_root) or ""
        conflict_files = conflicted_files_from_status(status_before_abort)

        abort = _run_git(repo_root, "merge", "--abort")
        if abort.returncode != 0:
            # Ignored by design (there may be nothing to abort), but recorded.
            notes.append(
                f"git merge --abort exited {abort.returncode}: "
                f"{abort.stderr.strip() or '(no stderr)'}"
            )

        status_after = _porcelain_status(repo_root)
        if status_after is None or status_after.strip():
            notes.append(
                "the working tree is NOT clean after the abort — "
                "look before touching anything"
            )
        notes.append(f"merge error: {exc}")

        return MergeReport(
            outcome=OUTCOME_CONFLICT,
            feature_id=feature_id,
            target_branch=target_branch,
            branch=branch,
            pre_sha=pre_sha,
            conflict_files=tuple(conflict_files),
            notes=tuple(notes),
        )

    post_sha = _rev_parse(repo_root, target_branch)
    return MergeReport(
        outcome=OUTCOME_MERGED,
        feature_id=feature_id,
        target_branch=target_branch,
        branch=branch,
        pre_sha=pre_sha,
        post_sha=post_sha,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# The pre-merge baseline — what the target branch was already failing
# ---------------------------------------------------------------------------


def measure_pre_merge_baseline(
    repo_root: Path,
    target_branch: str = "main",
    timeout: int = DEFAULT_VERIFY_TIMEOUT,
) -> MeasuredBaseline:
    """Run the resolved test command on ``target_branch`` before any merge.

    HEAD is put on the target branch first (a no-op when it is already
    there), because a baseline measured on the wrong branch would excuse the
    very failures the merge is being judged on. The command is resolved AFTER
    the switch, so the target branch's own declaration is the one that
    speaks, and the same resolution is then reused for the post-merge run.

    ``ran`` is False when the command could not start (or no runner could be
    found at all). There is then no baseline: the caller must report the
    whole check as "could not run", never as a clean baseline.
    """
    repo_root = Path(repo_root)

    switch_problem = checkout_branch(repo_root, target_branch)
    if switch_problem is not None:
        return MeasuredBaseline(
            command="",
            source="",
            ran=False,
            detail=(
                f"the pre-merge test run never started: {switch_problem}"
            ),
        )

    command, source, profile = resolve_verify_command(repo_root)
    result = run_completion_verification(
        repo_root, command, source, stack_profile=profile, timeout=timeout
    )
    if result.status == "unverified":
        return MeasuredBaseline(
            command=command or "",
            source=source,
            ran=False,
            detail=result.detail,
            stack_profile=profile,
        )

    return MeasuredBaseline(
        command=result.command,
        source=source,
        failing=tuple(failing_node_ids(result.output_tail)),
        passed=passed_count_from_output(result.output_tail),
        ran=True,
        detail=result.detail,
        stack_profile=profile,
    )


def merge_verdict_from_run(
    suite_status: str,
    suite_detail: str,
    observed_failures: Sequence[str],
    charged_failures: Sequence[str],
    baseline_known: bool,
    target_branch: str,
) -> Tuple[str, str]:
    """The MERGE's verdict on the test run — "no new red", stated once.

    Returns ``(status, detail)``:

    * a run that could not start stays ``unverified`` — absence of evidence
      is never a pass;
    * a run that failed on something charged to this merge stays ``failed``;
    * a run that failed ONLY on failures the target branch was already
      failing is ``passed``, and the detail says so in plain words;
    * a run that failed but named no failing tests stays ``failed`` — there
      was nothing to compare, so nothing may be excused.
    """
    if suite_status != "failed":
        return suite_status, suite_detail
    if charged_failures or not baseline_known:
        return "failed", suite_detail
    if not observed_failures:
        return (
            "failed",
            f"the test run failed but named no failing tests, so nothing "
            f"could be compared with {target_branch} ({suite_detail})",
        )
    return (
        "passed",
        f"{len(observed_failures)} failing test(s), every one of them "
        f"already failing on {target_branch} before the merge "
        f"({suite_detail})",
    )


# ---------------------------------------------------------------------------
# Post-merge verification
# ---------------------------------------------------------------------------


def default_validate_command(feature_id: str) -> List[str]:
    """The ``guardkit feature validate <fid> --json`` argv.

    Prefers the installed ``guardkit`` console script; falls back to
    ``python -m guardkit.cli.main`` (the documented module entry point) so the
    validation runs in exactly the environment running this executor.
    """
    exe = shutil.which("guardkit")
    if exe:
        return [exe, "feature", "validate", feature_id, "--json"]
    return [
        sys.executable,
        "-m",
        "guardkit.cli.main",
        "feature",
        "validate",
        feature_id,
        "--json",
    ]


def parse_validate_stdout(stdout: str) -> Tuple[Optional[bool], str]:
    """Parse the ``feature validate --json`` STDOUT into ``(valid, detail)``.

    STDOUT ONLY — an INFO line rides stderr and must never reach this parser.
    Returns ``(None, reason)`` when no verdict could be read (never a pass).
    """
    text = stdout.strip()
    if not text:
        return None, "feature validate printed nothing on stdout"
    try:
        data = json.loads(text)
    except ValueError:
        return None, "feature validate stdout was not JSON"
    if not isinstance(data, dict) or "valid" not in data:
        return None, "feature validate JSON carried no 'valid' field"
    detail = ""
    errors = data.get("errors") or []
    if errors:
        detail = "; ".join(str(e) for e in errors[:5])
    return bool(data["valid"]), detail


def run_feature_validate(
    repo_root: Path,
    feature_id: str,
    validate_command: Optional[Sequence[str]] = None,
) -> Tuple[Optional[bool], str]:
    """Run ``guardkit feature validate <fid> --json`` and parse STDOUT only."""
    argv = list(validate_command or default_validate_command(feature_id))
    try:
        proc = subprocess.run(
            argv,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=_VALIDATE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"feature validate could not run: {exc}"
    return parse_validate_stdout(proc.stdout)


def charged_failures_from_output(
    repo_root: Path,
    verification_output: Optional[str],
    baseline_failing: Optional[Sequence[str]],
) -> Tuple[List[str], List[str]]:
    """``(charged, notes)`` from the post-merge suite output.

    ``observed - (baseline ∪ ledger)`` via the same primitives the Coach loop
    uses. ``baseline_failing=None`` means no pre-merge baseline exists: the
    FULL observed set is reported (minus only the human-triaged ledger) with a
    note that the diff is unavailable — never an invented clean.
    """
    notes: List[str] = []
    observed = failing_node_ids(verification_output)
    ledger = load_known_failure_ids(Path(repo_root))
    charged = compute_charged_failures(
        observed_node_ids=observed,
        baseline_node_ids=list(baseline_failing or []),
        ledger_ids=ledger,
    )
    if baseline_failing is None and observed:
        notes.append(
            "no pre-merge baseline — diff unavailable; the full observed "
            "failure set is reported"
        )
    return charged, notes


def validate_with_notes(
    repo_root: Path,
    feature_id: str,
    validate_command: Optional[Sequence[str]] = None,
) -> Tuple[Optional[bool], List[str]]:
    """Run the feature-file validation and turn its verdict into notes."""
    notes: List[str] = []
    validate_valid, validate_detail = run_feature_validate(
        repo_root, feature_id, validate_command=validate_command
    )
    if validate_valid is None:
        notes.append(f"feature validation gave no verdict: {validate_detail}")
    elif validate_valid is False and validate_detail:
        notes.append(f"feature validation errors: {validate_detail}")
    return validate_valid, notes


def verify_merged(
    repo_root: Path,
    feature_id: str,
    baseline_failing: Optional[Sequence[str]] = None,
    timeout: int = DEFAULT_VERIFY_TIMEOUT,
    validate_command: Optional[Sequence[str]] = None,
    resolved_command: Optional[
        Tuple[Optional[str], str, Optional[StackTestProfile]]
    ] = None,
) -> Tuple[Optional[bool], VerificationResult, List[str], List[str]]:
    """The three post-merge checks, in order.

    Returns ``(validate_valid, verification, charged_failures, notes)``:

    * (a) ``guardkit feature validate <fid> --json`` as a subprocess with
      ``cwd=repo_root``, STDOUT parsed exclusively;
    * (b) the resolved verification command run in the merged repo — only
      ``status == "passed"`` is ever success (the precedence lives in
      ``resolve_verify_command``);
    * (c) the failing node ids charged to this merge, diffed against the
      pre-merge baseline and the known-failures ledger.

    ``resolved_command`` passes in a resolution already made — the one the
    pre-merge baseline used — so the baseline and the merged result are
    provably judged by the SAME command. Omit it and the command is resolved
    here, exactly as before.
    """
    validate_valid, notes = validate_with_notes(
        repo_root, feature_id, validate_command=validate_command
    )

    command, source, profile = resolved_command or resolve_verify_command(
        Path(repo_root)
    )
    verification = run_completion_verification(
        Path(repo_root),
        command,
        source,
        stack_profile=profile,
        timeout=timeout,
    )

    # The runner output available here is the recorded tail (capped at 2000
    # characters by run_completion_verification); a very long failure list may
    # be under-reported — the status verdict above is unaffected.
    charged, charge_notes = charged_failures_from_output(
        repo_root, verification.output_tail, baseline_failing
    )
    notes.extend(charge_notes)

    return validate_valid, verification, charged, notes


# ---------------------------------------------------------------------------
# Assembly — preflight, merge, verify, one report
# ---------------------------------------------------------------------------


def execute_merge(
    repo_root: Path,
    feature_id: str,
    target_branch: str = "main",
    expect_target_sha: Optional[str] = None,
    verify: bool = True,
    baseline_failing: Optional[Sequence[str]] = None,
    verify_timeout: int = DEFAULT_VERIFY_TIMEOUT,
    manager: Optional[WorktreeManager] = None,
    validate_command: Optional[Sequence[str]] = None,
    measure_baseline: bool = True,
) -> MergeReport:
    """Refusal preflight, the pre-merge baseline, the merge, the checks.

    Every outcome is a :class:`MergeReport`. The ``autobuild/<FEATURE_ID>``
    branch survives every path.

    When no ``baseline_failing`` is handed in, verification is on, and
    ``measure_baseline`` is left at its default, the same resolved test
    command is run on the target branch BEFORE the merge and its failures
    become the baseline. The merged result is then charged only for
    ``observed - (baseline union ledger)`` — the same subtraction the Coach
    loop uses, through the same primitives.

    If that pre-merge run cannot start, there is no baseline: the merge still
    happens (the branch is sound; it is the checking that is broken), and the
    whole test check is reported as "could not run" with that reason. It is
    never turned into an empty baseline, which would charge this merge for
    red the target branch already had.
    """
    repo_root = Path(repo_root)
    branch = f"autobuild/{feature_id}"

    reason = preflight_refusal(
        repo_root, feature_id, target_branch, expect_target_sha
    )
    if reason is not None:
        return MergeReport(
            outcome=OUTCOME_REFUSED,
            feature_id=feature_id,
            target_branch=target_branch,
            branch=branch,
            refusal_reason=reason,
        )

    baseline: Optional[MeasuredBaseline] = None
    resolved_command = None
    pre_merge_notes: List[str] = []
    if verify and measure_baseline and baseline_failing is None:
        baseline = measure_pre_merge_baseline(
            repo_root, target_branch=target_branch, timeout=verify_timeout
        )
        resolved_command = (
            baseline.command or None,
            baseline.source,
            baseline.stack_profile,
        )
        if baseline.ran:
            baseline_failing = list(baseline.failing)
            pre_merge_notes.append(
                f"the pre-merge baseline was measured on {target_branch}: "
                f"{len(baseline.failing)} failing, {baseline.passed} passing"
            )
        else:
            pre_merge_notes.append(
                f"the pre-merge test run on {target_branch} could not start, "
                f"so there is no baseline and the post-merge test run was "
                f"not attempted: {baseline.detail}"
            )

    report = perform_merge(
        repo_root, feature_id, target_branch, manager=manager
    )
    if report.outcome != OUTCOME_MERGED or not verify:
        return replace(
            report,
            baseline_measured=(
                baseline if (baseline is not None and baseline.ran) else None
            ),
            verify_command=(
                (baseline.command or None) if baseline is not None else None
            ),
            verify_source=(
                (baseline.source or None) if baseline is not None else None
            ),
            notes=tuple(list(report.notes) + pre_merge_notes),
        )

    # The pre-merge run could not start, so the post-merge run cannot either:
    # say "could not run" once, in the run's own words, and charge nothing.
    if baseline is not None and not baseline.ran:
        validate_valid, validate_notes = validate_with_notes(
            repo_root, feature_id, validate_command=validate_command
        )
        return MergeReport(
            outcome=report.outcome,
            feature_id=report.feature_id,
            target_branch=report.target_branch,
            branch=report.branch,
            pre_sha=report.pre_sha,
            post_sha=report.post_sha,
            verify_ran=True,
            verify_status="unverified",
            verify_detail=baseline.detail,
            verify_suite_status="unverified",
            validate_valid=validate_valid,
            charged_failures=(),
            baseline_measured=None,
            verify_command=baseline.command or None,
            verify_source=baseline.source or None,
            notes=tuple(
                list(report.notes) + pre_merge_notes + validate_notes
            ),
        )

    validate_valid, verification, charged, notes = verify_merged(
        repo_root,
        feature_id,
        baseline_failing=baseline_failing,
        timeout=verify_timeout,
        validate_command=validate_command,
        resolved_command=resolved_command,
    )

    observed = failing_node_ids(verification.output_tail)
    verdict_status, verdict_detail = merge_verdict_from_run(
        suite_status=verification.status,
        suite_detail=verification.detail,
        observed_failures=observed,
        charged_failures=charged,
        baseline_known=baseline_failing is not None,
        target_branch=target_branch,
    )

    return MergeReport(
        outcome=report.outcome,
        feature_id=report.feature_id,
        target_branch=report.target_branch,
        branch=report.branch,
        pre_sha=report.pre_sha,
        post_sha=report.post_sha,
        verify_ran=True,
        verify_status=verdict_status,
        verify_detail=verdict_detail,
        verify_suite_status=verification.status,
        validate_valid=validate_valid,
        charged_failures=tuple(charged),
        baseline_measured=(
            baseline if (baseline is not None and baseline.ran) else None
        ),
        verify_command=verification.command or None,
        verify_source=verification.source or None,
        notes=tuple(list(report.notes) + pre_merge_notes + notes),
    )


__all__ = [
    "OUTCOME_MERGED",
    "OUTCOME_REFUSED",
    "OUTCOME_CONFLICT",
    "MeasuredBaseline",
    "MergeReport",
    "conflicted_files_from_status",
    "current_branch",
    "checkout_branch",
    "passed_count_from_output",
    "preflight_refusal",
    "merge_commit_message",
    "perform_merge",
    "measure_pre_merge_baseline",
    "merge_verdict_from_run",
    "default_validate_command",
    "parse_validate_stdout",
    "run_feature_validate",
    "charged_failures_from_output",
    "validate_with_notes",
    "verify_merged",
    "execute_merge",
]
