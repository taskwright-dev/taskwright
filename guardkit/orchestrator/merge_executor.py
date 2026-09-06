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
  reported with a note saying the diff is unavailable. Test names are read
  from the WHOLE output of each run, never from the 2000-character excerpt
  kept for receipts, and every run is cross-checked against the number of
  failures the runner itself reported: if more tests failed than could be
  named, nothing is excused.
* **Judge no new red, not no red.** When no baseline is handed in, the merge
  measures one itself: the SAME resolved test command is run on the target
  branch before the merge, and its failures are what the merged result is
  excused for (2026-09-06 spec, rule 6). A baseline run that could not start
  makes the whole check "could not run" — it never becomes an empty baseline,
  because an empty baseline would charge this merge for red the branch never
  caused.
* **A branch never chooses the command that judges it.** The test command is
  resolved on the TARGET branch, before the merge, on every path — baseline
  measured, baseline handed in, or baseline turned off — and that one
  resolution runs both the pre-merge baseline and the post-merge check
  (2026-09-06 spec, rule 11). See
  :func:`resolve_verify_command_on_target` for why.
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


def reported_failure_count(output: Optional[str]) -> Optional[int]:
    """How many tests the RUNNER itself said failed, or None when it did not.

    Read from pytest's own count line ("8 failed, 3 errors, 118 passed"),
    which is one short line at the very end of a run and is therefore always
    present even in a heavily truncated excerpt. The per-test names are not:
    pytest's summary block lists one line per failure and easily runs to tens
    of thousands of characters. That gap is exactly what
    :func:`named_failures_are_complete` guards.
    """
    from guardkit.qa.enforcement import parse_pytest_outcome

    outcome = parse_pytest_outcome(output or "")
    if outcome.failed is None and outcome.errored is None:
        return None
    return (outcome.failed or 0) + (outcome.errored or 0)


def named_failures_are_complete(
    output: Optional[str], named: Sequence[str]
) -> bool:
    """True when every failure the runner counted could also be named.

    A run whose output names fewer failing tests than the runner said failed
    has been read incompletely — truncated output, an unfamiliar reporter, a
    crashed run. Nothing may be excused on such a reading, because the very
    failure this merge caused could be one of the ones that went unnamed.
    When the runner gave no count of its own there is nothing to check
    against, and the reading is taken at face value.
    """
    reported = reported_failure_count(output)
    if reported is None:
        return True
    return len(named) >= reported


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
    # True when every failure the pre-merge runner counted could also be
    # named. A baseline that could not name them all is still subtracted (a
    # short baseline only ever charges this merge with MORE), but it may
    # never be the grounds for excusing a failed suite.
    names_complete: bool = True

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
    # True only when a post-merge test run was actually ATTEMPTED. It is
    # False both when verification was turned off and when the checks could
    # not run at all — nothing was run after the merge on either path
    # (2026-09-06 spec, rule 12). ``verify_status`` tells the two apart:
    # None when verification was off, "unverified" when it could not run.
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
            if self.verify_status is None:
                # Verification was turned off. A check that was ATTEMPTED and
                # could not run says "unverified" instead, and is reported
                # below in full — it must never be mistaken for a run nobody
                # asked for.
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
                    # "would have been" on the unverified path: there the
                    # report itself says the run was never made, and a
                    # receipt must never imply tests ran that did not.
                    ran_words = (
                        "The tests would have been run with"
                        if self.verify_status == "unverified"
                        else "The tests were run with"
                    )
                    lines.append(
                        f"{ran_words}: {self.verify_command} "
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


def resolve_verify_command_on_target(
    repo_root: Path,
    target_branch: str = "main",
    smoke_command: Optional[str] = None,
) -> Tuple[
    Tuple[Optional[str], str, Optional[StackTestProfile]], Optional[str]
]:
    """Choose the test command on the TARGET branch, before any merge.

    Returns ``(resolution, problem)``: the ``(command, source, profile)``
    triple :func:`resolve_verify_command` gives, and a plain reason when HEAD
    could not be put on the target branch first — in which case the
    resolution was made against whatever tree was in front of us and the
    caller must not trust it.

    Why the target branch, and why before the merge (2026-09-06 spec, rule
    11). ``resolve_verify_command`` reads the repository's declared test
    command out of ``.guardkit/config.yaml`` in the WORKING TREE, and that
    file lives inside the very repository being merged. Resolve it after the
    merge and the branch under judgement has just written the command that
    judges it: a branch that sets ``toolchain.test`` to ``true`` and adds any
    number of failing tests would merge green, its own new red never run.

    ``toolchain_declaration.py`` names this danger in its second law — "the
    declaration is snapshotted before the model's first turn ... without a
    snapshot, a Player turn could rewrite ``toolchain.test`` to ``true`` and
    green itself" — and answers it by pinning a copy of the declaration
    outside the worktree before turn 1. This function is that same law at the
    merge's own scale. The merge's snapshot is the resolution made here, on
    the target branch, before anything lands; it is what runs the pre-merge
    baseline AND, handed on unchanged, what verifies the merged tree. Both
    ends of the comparison are therefore judged by a command the branch could
    not touch.
    """
    repo_root = Path(repo_root)
    problem = checkout_branch(repo_root, target_branch)
    resolution = resolve_verify_command(
        repo_root, smoke_command=smoke_command
    )
    return resolution, problem


def measure_pre_merge_baseline(
    repo_root: Path,
    target_branch: str = "main",
    timeout: int = DEFAULT_VERIFY_TIMEOUT,
    resolved_command: Optional[
        Tuple[Optional[str], str, Optional[StackTestProfile]]
    ] = None,
) -> MeasuredBaseline:
    """Run the resolved test command on ``target_branch`` before any merge.

    HEAD is put on the target branch first (a no-op when it is already
    there), because a baseline measured on the wrong branch would excuse the
    very failures the merge is being judged on.

    ``resolved_command`` is the resolution already made on the target branch
    by :func:`resolve_verify_command_on_target` — the SAME one that will
    verify the merged tree. Omit it and the command is resolved here, after
    the switch, which comes to the same thing.

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

    command, source, profile = (
        resolved_command
        if resolved_command is not None
        else resolve_verify_command(repo_root)
    )
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

    text = result.output_for_parsing
    failing = tuple(failing_node_ids(text))
    return MeasuredBaseline(
        command=result.command,
        source=source,
        failing=failing,
        passed=passed_count_from_output(text),
        ran=True,
        detail=result.detail,
        stack_profile=profile,
        names_complete=named_failures_are_complete(text, failing),
    )


def merge_verdict_from_run(
    suite_status: str,
    suite_detail: str,
    observed_failures: Sequence[str],
    charged_failures: Sequence[str],
    baseline_known: bool,
    target_branch: str,
    reading_complete: bool = True,
) -> Tuple[str, str]:
    """The MERGE's verdict on the test run — "no new red", stated once.

    Returns ``(status, detail)``:

    * a run that could not start stays ``unverified`` — absence of evidence
      is never a pass;
    * a run that failed on something charged to this merge stays ``failed``;
    * a run whose failures could not all be read stays ``failed``
      (``reading_complete=False``): either the merged run named fewer
      failures than the runner counted, or the baseline it would be excused
      against did. Excusing on a partial reading is how a merge invents a
      clean, so it is refused outright;
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
    if not reading_complete:
        return (
            "failed",
            f"more tests failed than could be named from the test output, so "
            f"none of them were excused against {target_branch} "
            f"({suite_detail})",
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

    ``resolved_command`` passes in a resolution already made on the target
    branch before the merge (:func:`resolve_verify_command_on_target`), so
    the baseline and the merged result are judged by the SAME command and the
    merged branch cannot have chosen it (2026-09-06 spec, rule 11).
    :func:`execute_merge` always passes it. Omit it — a direct caller
    verifying a tree it already trusts — and the command is resolved here,
    against the tree as it now stands.
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

    # The WHOLE output of the run, not the 2000-character excerpt kept for
    # receipts: a suite with roughly twenty or more failures overflows that
    # excerpt, and a failure whose name fell off the end would be charged to
    # nobody. ``execute_merge`` cross-checks the names read here against the
    # number of failures the runner itself counted.
    charged, charge_notes = charged_failures_from_output(
        repo_root, verification.output_for_parsing, baseline_failing
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

    With verification on, the test command is resolved on the TARGET branch
    before the merge on EVERY path — a baseline measured, a baseline handed
    in, or measuring turned off — and that one resolution is what verifies
    the merged tree. A branch cannot choose the command that judges it
    (2026-09-06 spec, rule 11; see
    :func:`resolve_verify_command_on_target`). ``verify_command`` and
    ``verify_source`` in the report are that pre-merge resolution.

    When no ``baseline_failing`` is handed in, verification is on, and
    ``measure_baseline`` is left at its default, that same command is then
    run on the target branch BEFORE the merge and its failures become the
    baseline. The merged result is then charged only for
    ``observed - (baseline union ledger)`` — the same subtraction the Coach
    loop uses, through the same primitives.

    If that pre-merge run cannot start, there is no baseline: the merge still
    happens (the branch is sound; it is the checking that is broken), and the
    whole test check is reported as "could not run" with that reason. It is
    never turned into an empty baseline, which would charge this merge for
    red the target branch already had.

    The baseline run comes before the merge attempt, so a branch that turns
    out to conflict pays for one suite run that is then thrown away. That
    cost is accepted deliberately: the only way to know whether a merge
    conflicts is to try it, and guessing first (with ``git merge-tree``, say)
    would mean a wrong guess skips the baseline and reports a clean merge as
    red. A wasted run on a conflicting branch is cheaper than a false red on
    a good one, and a conflict already needs a person.
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

    # Resolve both ends of the merge before anything moves. ``perform_merge``
    # checks this too, but it does so AFTER the baseline run has already put
    # HEAD on the target branch and spent a whole suite; refusing here leaves
    # HEAD exactly where the caller had it, as a refusal always should.
    if _rev_parse(repo_root, target_branch) is None or _rev_parse(
        repo_root, branch
    ) is None:
        return MergeReport(
            outcome=OUTCOME_REFUSED,
            feature_id=feature_id,
            target_branch=target_branch,
            branch=branch,
            refusal_reason=(
                f"could not resolve {target_branch} and {branch} to commits"
            ),
        )

    # Rule 11. The command that judges this merge is chosen HERE: on the
    # target branch, before anything lands, whether or not a baseline is
    # measured or handed in. Resolving it later would let the merged branch's
    # own .guardkit/config.yaml name the command that judges it.
    resolution: Optional[
        Tuple[Optional[str], str, Optional[StackTestProfile]]
    ] = None
    resolution_problem: Optional[str] = None
    if verify:
        resolution, resolution_problem = resolve_verify_command_on_target(
            repo_root, target_branch
        )
    verify_command = (resolution[0] or None) if resolution else None
    verify_source = (resolution[1] or None) if resolution else None

    baseline: Optional[MeasuredBaseline] = None
    resolved_command = resolution
    pre_merge_notes: List[str] = []
    if resolution_problem is not None:
        # No trustworthy command could be chosen, so no check may be run:
        # falling back to whatever the merged tree declares is exactly the
        # hole rule 11 closes.
        pre_merge_notes.append(
            f"the test command could not be chosen on {target_branch} "
            f"before the merge, so no test run was attempted: "
            f"{resolution_problem}"
        )
    elif verify and measure_baseline and baseline_failing is None:
        baseline = measure_pre_merge_baseline(
            repo_root,
            target_branch=target_branch,
            timeout=verify_timeout,
            resolved_command=resolution,
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
            verify_command=verify_command,
            verify_source=verify_source,
            notes=tuple(list(report.notes) + pre_merge_notes),
        )

    # Nothing could be run before the merge — either no command could be
    # chosen on the target branch, or the chosen one would not start — so
    # nothing is run after it either. Say "could not run" once, in the run's
    # own words, and charge nothing. ``verify_ran`` is False because no
    # post-merge run was attempted (2026-09-06 spec, rule 12).
    could_not_run_detail: Optional[str] = None
    if resolution_problem is not None:
        could_not_run_detail = resolution_problem
    elif baseline is not None and not baseline.ran:
        could_not_run_detail = baseline.detail
    if could_not_run_detail is not None:
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
            verify_ran=False,
            verify_status="unverified",
            verify_detail=could_not_run_detail,
            verify_suite_status="unverified",
            validate_valid=validate_valid,
            charged_failures=(),
            baseline_measured=None,
            verify_command=verify_command,
            verify_source=verify_source,
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

    merged_output = verification.output_for_parsing
    observed = failing_node_ids(merged_output)
    # Two readings have to be complete before a failed suite may be excused:
    # the merged run's own (the new red could be one of the names that fell
    # off the end) and the baseline's (a short baseline cannot vouch for a
    # failure it never named). Either one short and nothing is excused.
    reading_complete = named_failures_are_complete(merged_output, observed)
    if not reading_complete:
        notes.append(
            f"the test runner reported more failures than its output named "
            f"({reported_failure_count(merged_output)} counted, "
            f"{len(observed)} named), so none of them were excused"
        )
    if baseline is not None and not baseline.names_complete:
        reading_complete = False
        notes.append(
            f"the pre-merge run on {target_branch} reported more failures "
            f"than its output named, so it was not used to excuse anything"
        )
    verdict_status, verdict_detail = merge_verdict_from_run(
        suite_status=verification.status,
        suite_detail=verification.detail,
        observed_failures=observed,
        charged_failures=charged,
        baseline_known=baseline_failing is not None,
        target_branch=target_branch,
        reading_complete=reading_complete,
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
        # The pre-merge resolution, not whatever the merged tree would say
        # now: this is the command that actually ran, on both ends.
        verify_command=verify_command,
        verify_source=verify_source,
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
    "reported_failure_count",
    "named_failures_are_complete",
    "preflight_refusal",
    "merge_commit_message",
    "perform_merge",
    "resolve_verify_command_on_target",
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
