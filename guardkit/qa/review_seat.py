"""F14 emission for the R-b code-review seat (S3, 2026-07-13).

Design source (binding): ``ai-transition/docs/factory-code-quality-seat-options-
2026-07.md`` — R-b build-lane sketch stage **S-3**: "Command emits an F14
review-findings record; wired **default-OFF** behind a flag mirroring
``qa.enforce_tier1``; advisory placement first." This module is that emission
layer, and only that.

The chain it closes:

  S-1 (``diff_ingest.ReviewPayload``)  ─┐
                                        ├─▶  local seat  ─▶  F14 ``ReviewFindings``
  S-2 (review dimensions + discipline) ─┘   (llama-swap)      (``review_findings.py``)

Four invariants this stage pins:

- **Default-OFF, advisory-only (mirrors ``enforcement.is_tier1_enforced``).**
  The emitter is gated on ``qa.review_seat`` (``.guardkit/config.yaml``, default
  ``False``; env override ``GUARDKIT_QA_REVIEW_SEAT``). When OFF it is a *provable
  no-op*: it never touches the seat, never writes a record, never fires a reader.
  When ON it EMITS a record but **never fails a flow** — a seat outage or a parse
  error is a NAMED result field, never a raise. Promotion to *blocking* is the
  S-4 calibration gate (Rich's bar), outside this module.

- **F14's honesty rules bind (``review_findings.py`` rules, LPA-14/LPA-15).** A
  ``confirmed`` verdict REQUIRES an executed reproduction — and **this seat has no
  execution channel** (the S-2 player prompt forbids tool use), so a
  reading-only seat can never *earn* ``confirmed``. Every emitted finding is
  therefore ``refuted`` (the not-yet-confirmed bucket): "reading is not a
  verdict". A seat that *claims* a reproduction it could not have run is not
  trusted (``trust_seat_reproduction=False``, the lane default). Critical/high
  severity is default-refuted: a finding the seat did not defend with ≥2 refuters
  is DOWNGRADED to ``medium`` (we never fabricate refuters), never dropped
  silently — the downgrade is annotated on the finding.

- **Local seats only, single-slot-checked (DF-001 + the ``-np 1`` law).** Every
  seat call goes to llama-swap (``localhost:9000``, OpenAI-compatible); the
  allowed seats are ``qwen36-workhorse`` / ``gemma4-coach``. Before a call the
  emitter probes ``/running`` and, if a factory drive is mid-generation on the
  single slot, WAITS (bounded) rather than colliding. Calls are bounded
  (``temperature=0.0``, capped tokens, a timeout).

- **One rule for the address and the key (``guardkit/lib/client_env.py``).**
  The address is, in order: an explicit ``base_url`` from the caller, then
  ``GUARDKIT_REVIEW_SEAT_URL``, then ``OPENAI_BASE_URL``, then
  ``http://localhost:9000/v1``. The key is ``OPENAI_API_KEY`` when it is set and
  not blank, else the placeholder ``not-needed`` llama-swap has always ignored —
  never logged or printed.

- **Honesty-to-state.** A measured emission is recorded as measured; a seat that
  could not be reached or whose output could not be parsed produces an outcome
  with ``record=None`` and a named ``error`` — never a fabricated empty-green
  review.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from guardkit.lib.client_env import resolve_api_key, resolve_base_url
from guardkit.qa.diff_ingest import (
    DiffIngestError,
    FileDiff,
    ReviewPayload,
    ingest_commit,
    ingest_merge,
)
from guardkit.qa.formats.review_findings import (
    Finding,
    ReviewFindings,
    ReviewStats,
    ReviewSubject,
)

logger = logging.getLogger(__name__)

# TASK-SBHO-001: env-tunable ceiling for the ASSEMBLED review-seat user message.
# Mirrors the _trim_synthesis_prompt pattern: loud truncation marker, WARNING log,
# degrade-never-raise. Default 300k chars ≈ 85k tokens.
REVIEW_SEAT_MAX_CHARS_ENV = "GUARDKIT_REVIEW_SEAT_MAX_CHARS"
REVIEW_SEAT_MAX_CHARS: int = int(
    os.environ.get(REVIEW_SEAT_MAX_CHARS_ENV, "300000")
)

# Protected section markers that must NEVER be trimmed.
_REVIEW_INSTRUCTION_HEADER = "## Review subject"
_REVIEW_FINDING_SCHEMA = "## Diff under review"

__all__ = [
    "REVIEW_SEAT_ENV",
    "REVIEW_SEAT_URL_ENV",
    "REVIEW_SEAT_MAX_CHARS_ENV",
    "REVIEW_SEAT_MAX_CHARS",
    "is_review_seat_enabled",
    "ALLOWED_SEATS",
    "DEFAULT_SEAT",
    "DEFAULT_BASE_URL",
    "CANONICAL_DIMENSIONS",
    "SeatCall",
    "RunningProbe",
    "ReviewOutcome",
    "render_payload_for_seat",
    "build_seat_messages",
    "check_single_slot",
    "emit_review_findings",
    "run_advisory_review",
    "PayloadFactory",
    "FindingRouter",
    "default_merge_candidate_payload",
    "run_review_gate_step",
]

# ---------------------------------------------------------------------------
# The flag (mirrors qa.enforce_tier1 — enforcement.is_tier1_enforced, exactly).
# ---------------------------------------------------------------------------

#: Env override for the review-seat flag. Truthy wins over config; falsy forces
#: OFF; anything unrecognised is treated as OFF (loud warning).
REVIEW_SEAT_ENV = "GUARDKIT_QA_REVIEW_SEAT"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


def _load_config(repo_root: Path) -> dict:
    """Read ``<repo_root>/.guardkit/config.yaml``; empty dict if absent/unreadable."""
    path = repo_root / ".guardkit" / "config.yaml"
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        logger.warning(
            "qa.review_seat: could not read %s (%s) — treating as OFF", path, exc
        )
        return {}
    return data if isinstance(data, dict) else {}


def is_review_seat_enabled(repo_root: Path) -> bool:
    """Return whether the advisory review seat is ON for ``repo_root``.

    Precedence (identical to ``enforcement.is_tier1_enforced``):
    ``GUARDKIT_QA_REVIEW_SEAT`` env (truthy/falsy) > ``.guardkit/config.yaml``
    ``qa.review_seat`` > ``False``.

    Default OFF everywhere: a repo flips this as an explicit step; nothing runs
    the review seat fleet-wide by default, and even when ON the seat is advisory
    (it never blocks) until the S-4 calibration bar promotes it.
    """
    env = os.environ.get(REVIEW_SEAT_ENV)
    if env is not None:
        token = env.strip().lower()
        if token in _TRUTHY:
            return True
        if token in _FALSY:
            return False
        logger.warning(
            "%s=%r is not a recognised boolean — treating as OFF",
            REVIEW_SEAT_ENV,
            env,
        )
        return False
    qa = _load_config(repo_root).get("qa")
    if not isinstance(qa, dict):
        return False
    return bool(qa.get("review_seat", False))


# ---------------------------------------------------------------------------
# Seat constants (DF-001 · local seats only).
# ---------------------------------------------------------------------------

#: The only seats this lane may call (the options paper's operator-cost note 2).
ALLOWED_SEATS = ("qwen36-workhorse", "gemma4-coach")

#: Default reviewer seat — the general workhorse (131072 ctx) reads diffs best.
DEFAULT_SEAT = "qwen36-workhorse"

#: llama-swap OpenAI-compatible base URL (the ``/v1`` root) — the last resort,
#: after a caller's own value, ``GUARDKIT_REVIEW_SEAT_URL`` and ``OPENAI_BASE_URL``.
DEFAULT_BASE_URL = "http://localhost:9000/v1"

#: Env override for this client's endpoint, consulted after a caller's explicit
#: ``base_url`` and before the shared ``OPENAI_BASE_URL``.
REVIEW_SEAT_URL_ENV = "GUARDKIT_REVIEW_SEAT_URL"


def resolve_seat_base_url(base_url: Optional[str] = None) -> str:
    """The seat address, by the one shared rule
    (:func:`guardkit.lib.client_env.resolve_base_url`): an explicit ``base_url``,
    then ``GUARDKIT_REVIEW_SEAT_URL``, then ``OPENAI_BASE_URL``, then
    :data:`DEFAULT_BASE_URL`."""
    return resolve_base_url(
        explicit=base_url,
        env_vars=(REVIEW_SEAT_URL_ENV, "OPENAI_BASE_URL"),
        default=DEFAULT_BASE_URL,
    )


#: The four review dimensions scored by S-2's ``code_review.yaml``. The record's
#: ``dimensions`` list is always exactly these (that is what the seat scored);
#: each finding's ``dimension`` is normalised into one of them.
CANONICAL_DIMENSIONS = ("correctness", "simplification", "efficiency", "test_coverage")

_DIMENSION_ALIASES = {
    "correctness": "correctness",
    "simplification": "simplification",
    "simplicity": "simplification",
    "reuse": "simplification",
    "efficiency": "efficiency",
    "performance": "efficiency",
    "test_coverage": "test_coverage",
    "test-coverage": "test_coverage",
    "testcoverage": "test_coverage",
    "tests": "test_coverage",
    "coverage": "test_coverage",
}

_VALID_SEVERITIES = ("critical", "high", "medium", "low")

# Bounded-call defaults.
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_TIMEOUT_S = 180.0

# Single-slot guard.
_BUSY_STATES = frozenset({"processing", "busy", "generating"})
_SLOT_RETRY_MAX = 6
_SLOT_RETRY_SLEEP_S = 10.0


# ---------------------------------------------------------------------------
# Injectable edges (so unit tests never touch the network).
# ---------------------------------------------------------------------------

#: A seat call: (system_prompt, user_prompt, model) -> the raw completion text.
SeatCall = Callable[[str, str, str], str]

#: A probe of llama-swap ``/running`` -> the parsed "running" list (or None on
#: any failure — an unreachable probe is not a busy signal).
RunningProbe = Callable[[], Optional[List[Dict[str, Any]]]]


# ===========================================================================
# 1. Prompt assembly (the S-2 review contract, self-contained in guardkit).
# ===========================================================================

_REVIEW_SYSTEM = """\
You are the factory's INSPECTOR: a senior code reviewer who reads a finished \
change and writes up what is wrong, with reasons. You are advisory — you do NOT \
gate the build; you emit findings. Review ONLY the supplied diff; never invent \
defects on code the diff does not touch.

Score against exactly these four dimensions:
1. correctness    — real defects the change introduces (logic errors, broken \
contracts, unhandled edge cases, races, security holes).
2. simplification — genuine reuse / dead-code / over-abstraction / duplication \
the change could have avoided, with the simpler alternative named.
3. efficiency     — avoidable cost the change introduces (needless O(n^2), \
repeated I/O in a loop, unbounded allocation), with the observable impact.
4. test_coverage  — new branches / error paths / boundaries in the diff that no \
test exercises; name the uncovered behaviour and the test that would cover it.

Finding discipline (the F14 finding record):
- Anchor every finding to a concrete file:line INSIDE the diff and state a \
failing-input -> wrong-output scenario. A finding with no anchor and no scenario \
is not a finding.
- You CANNOT run code. You have no tools and no execution channel. Therefore \
you must NEVER claim a reproduction you did not run: leave "executed_reproduction" \
null. Reading is not a verdict.
- For critical or high severity, supply at least TWO refuters — concrete reasons \
the finding might be wrong — each with your answer. Severity is default-refuted: \
a finding you cannot defend with two refuters is medium at most.
- Do not over-flag. A cosmetic or stylistic nit (naming taste, import order, \
formatting a linter would catch) is "low" severity or omitted — never a blocker.
- Propose-never-elicit: never address a question to a human.

OUTPUT: a single JSON object and nothing else (no prose, no markdown fences). \
Shape:
{
  "summary": "<plain-English summary of what the change does>",
  "findings": [
    {
      "id": "F1",
      "dimension": "correctness|simplification|efficiency|test_coverage",
      "severity": "critical|high|medium|low",
      "file": "path/from/the/diff",
      "line": 123,
      "summary": "<one-sentence defect statement>",
      "failing_scenario": "<failing-input -> wrong-output>",
      "executed_reproduction": null,
      "refuters": [
        {"who": "refuter-1", "verdict": "refuted|not_refuted", "note": "<why>"}
      ]
    }
  ]
}
If the change is clean, emit {"summary": "...", "findings": []}. Never \
manufacture a nit to fill the list.
"""


def render_payload_for_seat(payload: ReviewPayload, *, max_chars: int = 60000) -> str:
    """Render an S-1 ``ReviewPayload`` as the unified diff text the seat reads.

    Deterministic and lossless within budget: reconstructs each file's header
    line, change kind, and hunks (with ``@@`` headers + ``+``/``-``/context
    markers). A binary or hunk-less file is named with its change kind so the
    reviewer sees it exists. Truncated at ``max_chars`` with an explicit marker
    (honesty-to-state: a clipped diff says so, it is never silently short).
    """
    lines: List[str] = []
    for fd in payload.files:
        lines.append(_render_file_header(fd))
        if fd.is_binary:
            lines.append("(binary file — not shown)")
            continue
        for hunk in fd.hunks:
            lines.append(hunk.header if hunk.header else _synth_hunk_header(hunk))
            for dl in hunk.lines:
                marker = {"added": "+", "removed": "-", "context": " "}[dl.kind]
                lines.append(f"{marker}{dl.content}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = (
            text[:max_chars]
            + f"\n... [diff truncated at {max_chars} chars for the review seat "
            f"— {len(text) - max_chars} more chars not shown]"
        )
    return text


def _render_file_header(fd: FileDiff) -> str:
    old = fd.old_path or "/dev/null"
    new = fd.path or "/dev/null"
    return f"diff --git a/{old} b/{new}  [{fd.change_kind}]"


def _synth_hunk_header(hunk: Any) -> str:
    head = (
        f"@@ -{hunk.old_start},{hunk.old_count} "
        f"+{hunk.new_start},{hunk.new_count} @@"
    )
    return head + (f" {hunk.section_heading}" if hunk.section_heading else "")


def build_seat_messages(
    payload: ReviewPayload,
    *,
    repo_context: Optional[str] = None,
    max_chars: int | None = None,
) -> Tuple[str, str]:
    """Assemble the (system, user) messages for the reviewer seat.

    The user message carries the review subject, an optional repo-context
    section, and the rendered diff. The system message is the self-contained
    S-2 review contract (dimensions + F14 discipline + JSON shape).

    When *max_chars* is set (default from ``REVIEW_SEAT_MAX_CHARS``), the
    assembled user message is bounded: ``repo_context`` is trimmed first,
    then the diff tail. The instruction header (review subject) and the
    diff section itself are never trimmed — only the content within
    ``repo_context`` and the tail of the diff are eligible.

    Truncation is **loud**: a visible notice is inserted inside the prompt
    naming what was cut and by how much, and a WARNING is logged.
    """
    if max_chars is None:
        max_chars = REVIEW_SEAT_MAX_CHARS

    diff_text = render_payload_for_seat(payload)
    subject_section = (
        f"## Review subject\n\nkind: {payload.subject_kind} · ref: {payload.ref}"
    )
    repo_section: str | None = None
    if repo_context and repo_context.strip():
        repo_section = "## Repository context (reference)\n\n" + repo_context.strip()

    diff_section = (
        "## Diff under review (review ONLY these changes)\n\n"
        + (diff_text if diff_text.strip() else "(empty diff — nothing changed)")
    )

    user_message = _trim_review_seat_user(
        subject_section,
        repo_section,
        diff_section,
        max_chars=max_chars,
    )
    return _REVIEW_SYSTEM, user_message


def _trim_review_seat_user(
    subject_section: str,
    repo_section: str | None,
    diff_section: str,
    *,
    max_chars: int,
) -> str:
    """Trim the assembled user message to *max_chars*.

    Priority: never trim the subject (instruction header) or the diff
    section header. Trim ``repo_context`` first, then the diff tail.

    Returns the assembled user message, possibly truncated with a loud
    truncation marker.
    """
    # Build the full message to check if it's over budget.
    parts: List[str] = [subject_section]
    if repo_section:
        parts.append(repo_section)
    parts.append(diff_section)
    full = "\n\n".join(parts)

    if len(full) <= max_chars:
        return full

    # We need to trim. Strategy:
    # 1. Trim repo_context first (lowest signal for the review).
    # 2. If still over, trim the diff tail.
    # Never trim the subject or the diff section header.
    trimmed = full
    total_elided = 0

    # Step 1: Trim repo_context if present.
    if repo_section:
        repo_len = len(repo_section)
        # Reserve space for subject + diff_section + separators.
        # Truncation marker space is handled in Step 2 if needed.
        reserved = len(subject_section) + len(diff_section) + 4  # 2x "\n\n"
        available_for_repo = max(0, max_chars - reserved)
        if available_for_repo < 100:
            # Not enough room for even a minimal repo section — drop it.
            elided = repo_len
            total_elided += elided
            trimmed = subject_section + "\n\n" + diff_section
            logger.warning(
                "review seat: repo_context truncated (%d chars elided) "
                "to fit within %d-char budget",
                elided,
                max_chars,
            )
        else:
            # Trim the repo_context content (keep the section header).
            header_end = repo_section.find("\n\n")
            if header_end == -1:
                header_end = len(repo_section)
            header = repo_section[:header_end]
            content = repo_section[header_end:]
            max_repo_content = max(0, available_for_repo - len(header))
            if len(content) > max_repo_content:
                elided = len(content) - max_repo_content
                total_elided += elided
                repo_section_trimmed = header + content[:max_repo_content]
                trimmed = (
                    subject_section
                    + "\n\n"
                    + repo_section_trimmed
                    + "\n\n... [repository context truncated: "
                    f"{elided} more chars elided to fit budget.] ..."
                    + "\n\n"
                    + diff_section
                )
                logger.warning(
                    "review seat: repo_context truncated (%d chars elided) "
                    "to fit within %d-char budget",
                    elided,
                    max_chars,
                )
            else:
                trimmed = repo_section

    # Step 2: If still over budget, trim the diff tail.
    # Reserve ~100 chars for the truncation marker.
    _DIFF_MARKER_RESERVATION = 100
    if len(trimmed) > max_chars:
        # Find the diff section and trim its content.
        diff_header = "## Diff under review (review ONLY these changes)\n\n"
        diff_start = trimmed.find(diff_header)
        if diff_start != -1:
            diff_content_start = diff_start + len(diff_header)
            diff_content = trimmed[diff_content_start:]
            max_diff_tail = max(
                0, max_chars - diff_content_start - _DIFF_MARKER_RESERVATION
            )
            if len(diff_content) > max_diff_tail:
                elided = len(diff_content) - max_diff_tail
                total_elided += elided
                trimmed = (
                    trimmed[:diff_content_start]
                    + diff_content[:max_diff_tail]
                    + f"\n... [diff truncated: {elided} more chars elided "
                    f"to fit within {max_chars}-char budget.] ..."
                )
                logger.warning(
                    "review seat: diff tail truncated (%d chars elided) "
                    "to fit within %d-char budget",
                    elided,
                    max_chars,
                )
        else:
            # Fallback: hard trim at budget.
            trimmed = (
                trimmed[:max_chars]
                + f"\n... [review-seat user message truncated at "
                f"{max_chars} chars — {len(trimmed) - max_chars} more chars not shown.]"
            )
            logger.warning(
                "review seat: user message hard-trimmed at %d chars "
                "(%d chars elided)",
                max_chars,
                len(trimmed) - max_chars,
            )

    return trimmed


# ===========================================================================
# 2. Single-slot guard (the -np 1 law — never collide with a live drive).
# ===========================================================================


def _default_running_probe(base_url: str) -> RunningProbe:
    """Build a ``/running`` probe from the seat base URL (``.../v1`` -> ``/running``)."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    running_url = root + "/running"

    def _probe() -> Optional[List[Dict[str, Any]]]:
        try:
            with urllib.request.urlopen(running_url, timeout=5) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            return None
        running = data.get("running") if isinstance(data, dict) else None
        return running if isinstance(running, list) else None

    return _probe


def check_single_slot(
    running: Optional[Sequence[Dict[str, Any]]],
) -> Tuple[bool, str]:
    """Decide whether the single seat slot is free to call.

    ``running`` is the parsed ``/running`` list. Returns ``(free, reason)``.
    An unreachable probe (``None``) is NOT read as busy — we proceed and let the
    seat call itself surface any outage. A seat entry whose state names active
    generation (``processing`` / ``busy`` / ``generating``) means a factory
    drive holds the single slot: not free.
    """
    if running is None:
        return True, "running-probe unreachable — proceeding (seat call will report any outage)"
    busy = [
        str(e.get("model", "?"))
        for e in running
        if isinstance(e, dict) and str(e.get("state", "")).strip().lower() in _BUSY_STATES
    ]
    if busy:
        return False, f"seat slot held by a live drive: {', '.join(busy)}"
    return True, "seat slot free"


def _await_free_slot(
    running_probe: RunningProbe,
    *,
    retry_max: int = _SLOT_RETRY_MAX,
    retry_sleep_s: float = _SLOT_RETRY_SLEEP_S,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Wait (bounded) for the single slot, honouring the ``-np 1`` law.

    Returns a human note describing the outcome. If the slot never frees within
    the retry budget we PROCEED anyway (llama-swap serialises the request on the
    slot — the worst case is queueing, never corruption) and say so; this is an
    advisory lane, it must not hang a flow forever.
    """
    for attempt in range(retry_max + 1):
        free, reason = check_single_slot(running_probe())
        if free:
            return reason if attempt == 0 else f"{reason} (after {attempt} wait(s))"
        if attempt < retry_max:
            logger.info("review seat: %s — waiting %.0fs", reason, retry_sleep_s)
            sleep(retry_sleep_s)
    return (
        f"seat slot still contended after {retry_max} wait(s) — proceeding "
        f"(request will queue on the single slot)"
    )


# ===========================================================================
# 3. Default seat call (OpenAI-compatible llama-swap — the impure edge).
# ===========================================================================


def _default_seat_call(
    base_url: Optional[str] = None,
    *,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> SeatCall:
    """Build a bounded OpenAI-compatible seat call against llama-swap.

    Imported lazily so the module has no hard ``openai`` dependency for the
    pure/flag paths (unit tests inject their own ``SeatCall`` and never import
    ``openai``).
    """

    resolved_base_url = resolve_seat_base_url(base_url)

    def _call(system_prompt: str, user_prompt: str, model: str) -> str:
        from openai import OpenAI  # lazy — only the real-seat path needs it

        # OPENAI_API_KEY when it is set, else the placeholder llama-swap has
        # always ignored. Never logged — it goes into the request and nowhere else.
        client = OpenAI(
            base_url=resolved_base_url, api_key=resolve_api_key(), timeout=timeout_s
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    return _call


# ===========================================================================
# 4. Parse + honesty transform (seat text -> F14 findings).
# ===========================================================================


class ReviewSeatError(Exception):
    """The seat output could not be parsed into findings — surfaced as a named
    outcome error, never raised into a flow."""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Pull the review JSON object out of a raw completion, tolerant of reasoning
    ``<think>`` blocks and ``` fences. Raises :class:`ReviewSeatError` if no
    balanced object is found."""
    cleaned = _THINK_RE.sub("", text or "").strip()
    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ReviewSeatError("seat output contains no JSON object")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = cleaned[start : i + 1]
                try:
                    obj = json.loads(blob)
                except json.JSONDecodeError as exc:
                    raise ReviewSeatError(
                        f"seat output is not valid JSON: {exc}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise ReviewSeatError("seat JSON root is not an object")
                return obj
    raise ReviewSeatError("seat output has an unterminated JSON object")


def _normalise_dimension(raw: Any) -> str:
    token = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _DIMENSION_ALIASES.get(token, "correctness")


def _anchor(raw: Dict[str, Any]) -> Optional[str]:
    file = str(raw.get("file", "") or "").strip()
    line = raw.get("line")
    if not file:
        return None
    if isinstance(line, int) or (isinstance(line, str) and str(line).strip().isdigit()):
        return f"{file}:{line}"
    return file


def _coerce_refuters(raw: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            continue
        who = str(r.get("who", "") or f"refuter-{i + 1}").strip() or f"refuter-{i + 1}"
        verdict = str(r.get("verdict", "refuted") or "refuted").strip().lower()
        if verdict not in ("refuted", "not_refuted"):
            verdict = "refuted"
        note = r.get("note")
        entry: Dict[str, str] = {"who": who, "verdict": verdict}
        if note is not None and str(note).strip():
            entry["note"] = str(note).strip()
        out.append(entry)
    return out


@dataclass(frozen=True)
class _MappedFinding:
    finding: Finding
    downgraded: bool


def _to_finding(
    raw: Dict[str, Any],
    index: int,
    *,
    trust_seat_reproduction: bool,
) -> Optional[_MappedFinding]:
    """Map one seat finding dict onto an F14 :class:`Finding`, honesty rules applied.

    - status: ``confirmed`` only if reproduction is trusted AND present (never in
      this lane) — otherwise ``refuted`` (reading is not a verdict, LPA-15).
    - severity: critical/high with <2 refuters is downgraded to ``medium`` and
      annotated (default-refuted, LPA-14) — we never fabricate refuters.
    - a finding with no diff anchor at all is dropped (not a finding).
    """
    fid = str(raw.get("id", "") or f"F{index + 1}").strip() or f"F{index + 1}"
    dimension = _normalise_dimension(raw.get("dimension"))

    severity = str(raw.get("severity", "medium") or "medium").strip().lower()
    if severity not in _VALID_SEVERITIES:
        severity = "medium"

    anchor = _anchor(raw)
    summary = str(raw.get("summary", "") or "").strip()
    scenario = str(raw.get("failing_scenario", "") or "").strip()
    if not summary:
        summary = scenario
    if anchor is None or not summary:
        # No anchor or no statement — not an actionable finding; drop it.
        return None

    refuters = _coerce_refuters(raw.get("refuters"))

    # Honesty rule (LPA-15): reading is not a verdict. This seat cannot execute,
    # so a claimed reproduction is not trusted -> the finding stays refuted and
    # the reproduction is stripped (never a fabricated confirmed).
    claimed_repro = raw.get("executed_reproduction")
    if trust_seat_reproduction and claimed_repro and str(claimed_repro).strip():
        status = "confirmed"
        executed_reproduction: Optional[str] = str(claimed_repro).strip()
    else:
        status = "refuted"
        executed_reproduction = None

    # Honesty rule (LPA-14): critical/high is default-refuted — needs >=2 refuters.
    downgraded = False
    if severity in ("critical", "high") and len(refuters) < 2:
        severity = "medium"
        downgraded = True

    full_summary = f"[{anchor}] {summary}"
    if scenario and scenario not in full_summary:
        full_summary += f" — failing scenario: {scenario}"
    if downgraded:
        full_summary += (
            " [severity downgraded to medium: <2 refuters, F14 default-refuted]"
        )

    finding = Finding(
        id=fid,
        dimension=dimension,
        severity=severity,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        summary=full_summary,
        executed_reproduction=executed_reproduction,
        refuters=refuters,  # type: ignore[arg-type]
    )
    return _MappedFinding(finding=finding, downgraded=downgraded)


def _slugify(text: str, *, limit: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:limit] or "review"


def emit_review_findings(
    payload: ReviewPayload,
    seat_output: str,
    *,
    review_id: Optional[str] = None,
    trust_seat_reproduction: bool = False,
) -> ReviewFindings:
    """Build a schema-valid F14 ``ReviewFindings`` record from a seat completion.

    Pure (no I/O, no seat) — takes the raw seat text and the S-1 payload,
    returns the validated record. Raises :class:`ReviewSeatError` only if the
    output cannot be parsed at all; individual malformed findings are dropped or
    honesty-corrected, never faked.
    """
    obj = _extract_json_object(seat_output)
    raw_findings = obj.get("findings")
    if raw_findings is None:
        raw_findings = []
    if not isinstance(raw_findings, list):
        raise ReviewSeatError("seat JSON 'findings' is not a list")

    findings: List[Finding] = []
    for i, raw in enumerate(raw_findings):
        if not isinstance(raw, dict):
            continue
        mapped = _to_finding(raw, i, trust_seat_reproduction=trust_seat_reproduction)
        if mapped is not None:
            findings.append(mapped.finding)

    confirmed = sum(1 for f in findings if f.status == "confirmed")
    refuted = sum(1 for f in findings if f.status == "refuted")
    refutations_attempted = sum(len(f.refuters) for f in findings)

    rid = review_id or (
        f"codereview-{payload.subject_kind}-{_slugify(payload.ref)}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    return ReviewFindings(
        format_version=ReviewFindings.CURRENT_FORMAT_VERSION,
        review_id=rid,
        subject=ReviewSubject(kind=payload.subject_kind, ref=payload.ref),
        dimensions=list(CANONICAL_DIMENSIONS),
        findings=findings,
        stats=ReviewStats(
            findings_total=len(findings),
            confirmed=confirmed,
            refuted=refuted,
            refutations_attempted=refutations_attempted,
        ),
    )


# ===========================================================================
# 5. The advisory entrypoint (flag-gated no-op; never fails a flow).
# ===========================================================================


@dataclass(frozen=True)
class ReviewOutcome:
    """The result of an advisory review pass.

    - ``enabled=False``  → the flag was OFF: a *provable no-op*. No seat call, no
      record, no write. ``record`` / ``emitted_path`` are ``None``.
    - ``enabled=True`` + ``record`` set → the seat ran and an F14 record was
      emitted (``emitted_path`` set if written). ``error`` is ``None``.
    - ``enabled=True`` + ``record=None`` + ``error`` set → the seat could not be
      reached or its output could not be parsed. This is a NAMED failure, never a
      raise — advisory placement means a review never fails the surrounding flow.

    ``blocking`` is always ``False`` in this lane (promotion to blocking is the
    S-4 calibration gate, Rich's bar, outside this module).
    """

    enabled: bool
    record: Optional[ReviewFindings] = None
    emitted_path: Optional[str] = None
    error: Optional[str] = None
    seat_note: Optional[str] = None
    blocking: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def emitted(self) -> bool:
        return self.record is not None


def run_advisory_review(
    repo_root: Path,
    payload: ReviewPayload,
    *,
    review_id: Optional[str] = None,
    model: str = DEFAULT_SEAT,
    base_url: Optional[str] = None,
    repo_context: Optional[str] = None,
    write: bool = False,
    trust_seat_reproduction: bool = False,
    seat_call: Optional[SeatCall] = None,
    running_probe: Optional[RunningProbe] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ReviewOutcome:
    """Run the advisory code review — a provable no-op when the flag is OFF.

    Flow when ``qa.review_seat`` is ON:

      1. single-slot guard (bounded wait; never collide with a live drive);
      2. bounded local-seat call (``temperature=0.0``);
      3. parse + honesty-correct into a schema-valid F14 record;
      4. optionally write ``qa/review-<id>.yaml``.

    Advisory contract: this NEVER raises and NEVER returns ``blocking=True``. A
    seat outage or parse failure returns an outcome with ``record=None`` and a
    named ``error``. When the flag is OFF neither ``seat_call`` nor
    ``running_probe`` is ever invoked (the provable no-op).
    """
    if not is_review_seat_enabled(repo_root):
        return ReviewOutcome(
            enabled=False,
            notes=("review seat flag OFF (qa.review_seat) — no-op, no seat call",),
        )

    if model not in ALLOWED_SEATS:
        # Refuse an off-policy seat, but advisory: name it, never raise.
        return ReviewOutcome(
            enabled=True,
            error=(
                f"seat {model!r} is not an allowed local seat "
                f"(DF-001 — allowed: {', '.join(ALLOWED_SEATS)})"
            ),
        )

    resolved_base_url = resolve_seat_base_url(base_url)
    probe = running_probe or _default_running_probe(resolved_base_url)
    seat_note = _await_free_slot(probe, sleep=sleep)

    call = seat_call or _default_seat_call(resolved_base_url)
    system_prompt, user_prompt = build_seat_messages(payload, repo_context=repo_context)

    try:
        raw = call(system_prompt, user_prompt, model)
    except Exception as exc:  # noqa: BLE001 — advisory: a seat outage is named, not raised
        return ReviewOutcome(
            enabled=True,
            error=f"seat call failed ({type(exc).__name__}): {exc}",
            seat_note=seat_note,
        )

    try:
        record = emit_review_findings(
            payload,
            raw,
            review_id=review_id,
            trust_seat_reproduction=trust_seat_reproduction,
        )
    except ReviewSeatError as exc:
        return ReviewOutcome(
            enabled=True,
            error=f"seat output could not be parsed into F14 findings: {exc}",
            seat_note=seat_note,
        )

    emitted_path: Optional[str] = None
    notes: List[str] = []
    if write:
        try:
            emitted_path = _write_record(repo_root, record)
        except OSError as exc:
            # The record IS valid; only the write failed. Advisory: name it.
            notes.append(f"record built but could not be written: {exc}")

    return ReviewOutcome(
        enabled=True,
        record=record,
        emitted_path=emitted_path,
        seat_note=seat_note,
        notes=tuple(notes),
    )


def _write_record(repo_root: Path, record: ReviewFindings) -> str:
    """Write the F14 record to ``qa/review-<id>.yaml`` (its canonical instance path)."""
    out_dir = repo_root / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"review-{_slugify(record.review_id, limit=80)}.yaml"
    path.write_text(
        yaml.safe_dump(record.model_dump(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return str(path)


# ===========================================================================
# 6. The gate-flow advisory step (S5 — the pre-merge advisory placement).
# ===========================================================================
#
# S5 wires the S3 advisory review as a STEP IN THE GATE FLOW: an advisory stage,
# behind the same default-OFF flag, that emits an F14 record as a flow artifact
# and NEVER fails the flow. It mirrors ``qa.enforce_tier1``'s stage discipline
# (the tier-1 ledger sweep wired into ``autobuild complete --verify``): a
# provable no-op when the flag is OFF, informative and non-blocking when ON.
#
# Scope note (options paper R-b, §"Where it attaches"): this is the *pre-merge
# advisory* placement — emit-and-attach. The *post-merge* placement (a finding
# as a DF-021 trust-ledger demotion signal) is deliberately OUT OF SCOPE here:
# DF-021 is designed-only today (STATE-ANCHORS WC-7). The ``route`` hook below is
# the seam a future DF-021 co-lane injects; in this placement it defaults to None
# and the F14 record IS the routing (attached as a flow artifact).

#: Builds the review payload for the gate step. Injectable so the wiring can pick
#: the merge-candidate subject and tests can supply a fixture payload without
#: touching git.
PayloadFactory = Callable[[], ReviewPayload]

#: The DF-017 disposition seam "where it exists": routes an emitted F14 record
#: once the review has run. There is no live-gate envelope to bin a review
#: finding against in the pre-merge advisory placement, so this defaults to None
#: (attaching the record as an artifact is the routing). A future DF-021
#: post-merge co-lane injects a real router here. A router that raises is caught
#: and named — an advisory step never fails a flow, not even on a routing bug.
FindingRouter = Callable[[ReviewFindings], None]


def default_merge_candidate_payload(
    repo_root: Path,
    *,
    ref: str = "HEAD",
    git_run: Optional[Callable[..., Any]] = None,
) -> ReviewPayload:
    """Ingest the delivered merge candidate at ``ref`` as the review subject.

    Tries the merge view first — the branch's contribution vs its first parent
    (:func:`ingest_merge`, the reviewable surface for a delivered change). If
    ``ref`` is not a merge commit that git call fails, so it falls back to the
    single-commit diff (:func:`ingest_commit`). Either way the subject is "what
    this change delivered".
    """
    try:
        return ingest_merge(repo_root, ref, git_run=git_run)
    except DiffIngestError:
        return ingest_commit(repo_root, ref, git_run=git_run)


def run_review_gate_step(
    repo_root: Path,
    *,
    payload_factory: Optional[PayloadFactory] = None,
    review_id: Optional[str] = None,
    model: str = DEFAULT_SEAT,
    base_url: Optional[str] = None,
    write: bool = True,
    route: Optional[FindingRouter] = None,
    repo_context: Optional[str] = None,
    trust_seat_reproduction: bool = False,
    seat_call: Optional[SeatCall] = None,
    running_probe: Optional[RunningProbe] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ReviewOutcome:
    """Run the advisory review as a STEP in the gate flow (S5, pre-merge placement).

    Flag-gated on ``qa.review_seat`` (default OFF). When OFF this is a **provable
    no-op**: ``payload_factory`` is never called (no git), no seat call, nothing
    written — the tier-1-stage discipline applied to the review seat.

    When ON it builds the merge-candidate payload (default:
    :func:`default_merge_candidate_payload`), runs :func:`run_advisory_review`
    (which writes the F14 record as a flow artifact when ``write=True``), and —
    if a ``route`` hook is supplied (the DF-017 seam "where it exists") — routes
    the emitted record. It ALWAYS returns ``blocking=False`` and NEVER raises: an
    advisory step can inform a flow but must never fail it. Promotion to blocking
    is the S-4 calibration gate (Rich's bar), outside this function.
    """
    if not is_review_seat_enabled(repo_root):
        return ReviewOutcome(
            enabled=False,
            notes=(
                "review gate step: flag OFF (qa.review_seat) — no-op, "
                "payload not built, no seat call",
            ),
        )

    factory = payload_factory or (lambda: default_merge_candidate_payload(repo_root))
    try:
        payload = factory()
    except Exception as exc:  # noqa: BLE001 — advisory: an unreadable subject is named, not raised
        return ReviewOutcome(
            enabled=True,
            error=(
                f"review gate step: could not build the review subject "
                f"({type(exc).__name__}): {exc}"
            ),
        )

    outcome = run_advisory_review(
        repo_root,
        payload,
        review_id=review_id,
        model=model,
        base_url=base_url,
        repo_context=repo_context,
        write=write,
        trust_seat_reproduction=trust_seat_reproduction,
        seat_call=seat_call,
        running_probe=running_probe,
        sleep=sleep,
    )

    if route is not None and outcome.record is not None:
        try:
            route(outcome.record)
        except Exception as exc:  # noqa: BLE001 — advisory: a routing bug is named, never raised
            outcome = replace(
                outcome,
                notes=outcome.notes
                + (f"finding router raised (ignored, advisory): {exc}",),
            )

    return outcome
