"""QAV shadow mode — a log-only SECOND OPINION beside every coach verdict.

Design of record: ``ai-transition/docs/qav-shadow-mode-design-2026-07-25.md``
(§7 claim ``1cec8f8``). The QAV seat logs a training-envelope judgment beside
EVERY coach verdict on every factory build — **log-only, never blocks,
absent-not-fail**.

**The one law over everything (design §"The one law"):** the shadow can never
change, delay, or fail a build. This module copies the estate's proven
compile-shadow Fallback law (from the since-deleted DCL capture lane —
"neither function EVER raises, changes a verdict, blocks a run"):

- flag **default OFF** (``autobuild.coach.qav_shadow.enabled`` /
  ``GUARDKIT_QAV_SHADOW``), a bad config reads as OFF;
- every error path swallows to a ``logger.warning`` with the verdict untouched;
- a down/absent seat writes an ``absent`` record (it is never a failure);
- when the flag is OFF :func:`schedule_qav_shadow` returns after the config
  read — no thread, no ``/running`` probe, no seat call, no file.

**Self-contained (fence: no adf import).** The training envelope the tune's
Phase-5.2 gate validated lives in ``agentic-dataset-factory``'s
``src/qav/contracts.py`` (``SYSTEM_PROMPT`` + ``build_user_message``). guardkit
must NOT import adf, so the envelope constants are COPIED here verbatim with
their sha256s pinned in comments (``test_qav_shadow`` asserts they still match).

**The seat call idiom** mirrors ``guardkit/qa/review_seat.py``: a fresh
single-slot ``:9000/running`` probe before the call (the held-out-runner law),
injectable ``SeatCall`` / ``RunningProbe`` edges so unit tests never touch the
network, and a bounded OpenAI-compatible call against llama-swap (lazy
``openai`` import — the pure/flag paths carry no ``openai`` dependency).

**The address and the key** come from the one shared rule in
``guardkit/lib/client_env.py``. Address, in order: the ``qav_shadow`` config
block's ``endpoint``, then ``GUARDKIT_QAV_SHADOW_URL``, then ``OPENAI_BASE_URL``,
then ``http://localhost:9000/v1``. Key: ``OPENAI_API_KEY`` when it is set and
not blank, else the placeholder ``not-needed`` — never logged or printed.

**The receipt** (design §"The receipt") is written beside the verdict it
shadows at ``.guardkit/autobuild/{task_id}/qav_shadow_turn_{turn}.json`` and the
same object is appended (``sort_keys=True``) to
``.guardkit/qav-shadow/queue.jsonl`` (the DCL sink convention) — resolved at
the MAIN checkout root, never a nested build worktree, so each build has ONE
unambiguous queue file (:func:`_main_checkout_root`). ``agree`` is
precomputed so burn-in tallies are one-liners.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from guardkit.lib.client_env import resolve_api_key, resolve_base_url

logger = logging.getLogger(__name__)

__all__ = [
    "QAV_SHADOW_ENV",
    "QAV_SHADOW_URL_ENV",
    "QAV_SYSTEM_PROMPT",
    "QAV_SYSTEM_PROMPT_SHA256",
    "PINNED_BUNDLE_SCHEMA_SHA",
    "LIVE_GATE_ABSENT_MARKER",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_S",
    "QAV_SHADOW_QUEUE",
    "EXCLUSIVE_SET_TOKENS",
    "ABSENT_REASONS",
    "SeatResult",
    "SeatCall",
    "RunningProbe",
    "ShadowOutcome",
    "build_user_message",
    "load_qav_shadow_config",
    "is_qav_shadow_enabled",
    "run_qav_shadow",
    "schedule_qav_shadow",
]

# ---------------------------------------------------------------------------
# Flag (mirrors capture.CAPTURE_ENV / review_seat.REVIEW_SEAT_ENV precedence).
# ---------------------------------------------------------------------------

#: Env override for the shadow flag. Truthy wins over config; falsy forces OFF;
#: anything unrecognised is treated as OFF (loud warning). Default OFF.
QAV_SHADOW_ENV = "GUARDKIT_QAV_SHADOW"

#: Env override for this client's endpoint, consulted after the config block's
#: own ``endpoint`` and before the shared ``OPENAI_BASE_URL``.
QAV_SHADOW_URL_ENV = "GUARDKIT_QAV_SHADOW_URL"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})

# ---------------------------------------------------------------------------
# The adf training envelope — COPIED verbatim (fence: guardkit must not import
# adf). Source of truth: ``agentic-dataset-factory/src/qav/contracts.py``.
#
#   * ``QAV_SYSTEM_PROMPT`` == ``contracts.SYSTEM_PROMPT`` verbatim.
#     sha256(QAV_SYSTEM_PROMPT.encode("utf-8")) ==
#       d107290370b0e21f3037081894442af46273429626737e2f9db9452cc14950f1
#   * ``build_user_message`` == ``contracts.build_user_message`` verbatim
#     (bundle re-serialized indent=2, sort_keys=True, ensure_ascii=False).
#   * ``PINNED_BUNDLE_SCHEMA_SHA`` == ``contracts.PINNED_BUNDLE_SCHEMA_SHA``.
#
# ``test_qav_shadow`` recomputes these sha256s so a silent drift in the copied
# constant is caught by CI (the pins are enforced, not conventional).
# ---------------------------------------------------------------------------
QAV_SYSTEM_PROMPT = (
    "You are an expert QA verification judge for an autonomous software factory. You read a "
    "structured evidence bundle gathered about one task's implementation — honesty "
    "verification, quality gates, test results, independent test runs, BDD oracle and "
    "authoring-sweep results, wiring/mocked-seam/stub-scan/coverage/behavioural-oracle "
    "analyses, plan audit, and runtime parity — and you decide whether the evidence supports "
    "approving the work.\n\n"
    "Your core belief: **per-task green is not feature green, and absence of failure is never "
    "success.** Passing unit tests that inject dependencies directly tell you nothing about "
    "production call sites. A guard with no wired producer protects nothing. A green suite over "
    "a soft-failed TypeError is a dead feature with good manners. Evidence that was never "
    "gathered is absent signal, not clean signal — you read every null field against "
    "gathering_status before interpreting it.\n\n"
    "You are equally calibrated in both directions. You approve honest work that carries "
    "advisory blemishes, demoted discrepancies, profile-legitimate gate opt-outs, or "
    "infrastructure-classified failures — a judge that rejects every imperfection is as useless "
    "as one that approves everything. A false approval ships a broken feature; a false block "
    "burns the factory's throughput; you are measured on both.\n\n"
    "You render exactly one verdict per bundle: approve, or reject with named findings. Every "
    "finding carries its defect class from the documented taxonomy and the locus in the "
    "evidence where your judgment anchors. You reason from the evidence in front of you — you "
    "never invent evidence that is not in the bundle, and you never let a confident "
    "implementation narrative outweigh a discrepancy the honesty verification actually recorded."
)

#: sha256 of :data:`QAV_SYSTEM_PROMPT` — pinned so the verbatim copy cannot drift
#: from adf's ``contracts.SYSTEM_PROMPT`` unnoticed (asserted in tests).
QAV_SYSTEM_PROMPT_SHA256 = (
    "d107290370b0e21f3037081894442af46273429626737e2f9db9452cc14950f1"
)

#: adf ``contracts.PINNED_BUNDLE_SCHEMA_SHA`` — the CoachEvidenceBundle field set
#: pinned at guardkit ``41a0ebe457`` (recorded in every receipt for provenance).
PINNED_BUNDLE_SCHEMA_SHA = "41a0ebe457"

#: adf ``contracts.LIVE_GATE_ABSENT_MARKER`` — the shadow has no live-gate
#: channel, so the training envelope's live-gate section is always this marker.
LIVE_GATE_ABSENT_MARKER = "(none available)"


def build_user_message(
    bundle: Dict[str, Any], live_gate: str = LIVE_GATE_ABSENT_MARKER
) -> str:
    """Serialize the evidence input exactly per OUTPUT-CONTRACT §2.

    Verbatim copy of ``contracts.build_user_message`` (deterministic order:
    ``indent=2, sort_keys=True, ensure_ascii=False``). The tune trained on this
    exact layout, so the shadow must reproduce it byte-for-byte.
    """
    bundle_json = json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False)
    return (
        "## Evidence bundle\n"
        "```json\n"
        f"{bundle_json}\n"
        "```\n\n"
        "## Live-gate results\n"
        f"{live_gate}\n"
    )


# ---------------------------------------------------------------------------
# Seat + serving constants (design §"Serving posture" + §"The hook").
# ---------------------------------------------------------------------------

#: llama-swap OpenAI-compatible base URL (the ``/v1`` root) — review_seat's default.
DEFAULT_ENDPOINT = "http://localhost:9000/v1"

#: The shadow seat's llama-swap entry (B2 coordinator wires ``qav-shadow`` on the
#: same v4 GGUF; design §"Serving posture").
DEFAULT_MODEL = "qav-shadow"

#: Hard per-call timeout. Sized to clear ONE ~18.5 s cold load with margin
#: (warm judgments are ~1.3–1.8 s); a call that exceeds it is ``absent(timeout)``.
DEFAULT_TIMEOUT_S = 30.0

_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 4096

#: The shadow sink (the DCL ``.guardkit/<lane>/queue.jsonl`` convention).
#: Resolved against the MAIN checkout root (:func:`_main_checkout_root`), NOT
#: the worktree the turn ran in — autobuild builds inside a nested, gitignored,
#: transient worktree, and a queue appended there is a queue nobody ever reads.
QAV_SHADOW_QUEUE = ".guardkit/qav-shadow/queue.jsonl"

#: llama-swap set/model name tokens whose presence in ``/running`` means an
#: EXCLUSIVE primary workload holds the box — the shadow SKIPS rather than evict
#: it (design §"Serving posture" eligibility gate: "when /running shows the qav
#: teacher set, autobuild_go, coach31, or po_eval active, the shadow SKIPS and
#: logs it — evicting a primary workload is never acceptable for a log line").
#: Matched case-insensitively as substrings of each running entry's model name.
#: The exact set ids are the B2 coordinator's llama-swap config; these are the
#: recon-named members. Overridable via config ``exclusive_sets``.
EXCLUSIVE_SET_TOKENS: Tuple[str, ...] = (
    "qav-teacher",
    "qav_teacher",
    "autobuild_go",
    "coach31",
    "po_eval",
)

#: llama-swap ``/running`` states that name a live generation on the single slot.
_BUSY_STATES = frozenset({"processing", "busy", "generating"})

#: The closed absent-reason enum (design §"The receipt").
ABSENT_REASONS = frozenset(
    {
        "probe_refused",
        "slot_busy",
        "transport_aborted",
        "timeout",
        "no_bundle",
        "skipped_set",
        # A guard-caught crash inside the shadow itself. Before 2026-08-26 a
        # swallowed internal error left NO row at all — indistinguishable from
        # the lane being off. Now it still leaves exactly one absent row.
        "internal_error",
    }
)


# ---------------------------------------------------------------------------
# Injectable edges (so unit tests never touch the network — review_seat idiom).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeatResult:
    """One seat completion: the raw text plus provenance the receipt records."""

    text: str
    usage: Optional[Dict[str, Any]] = None
    truncated: bool = False


#: A seat call: (system_prompt, user_prompt, model, timeout_s) -> SeatResult.
SeatCall = Callable[[str, str, str, float], SeatResult]

#: A probe of llama-swap ``/running`` -> the parsed "running" list (or None on
#: any failure — an unreachable probe means the seat/model is absent).
RunningProbe = Callable[[], Optional[List[Dict[str, Any]]]]


@dataclass(frozen=True)
class ShadowOutcome:
    """The result of a shadow pass (returned for tests / callers; never raised).

    - ``enabled=False`` → the flag was OFF: a provable no-op. No probe, no seat
      call, no file. Everything else is ``None``.
    - ``enabled=True`` + ``status="ok"`` → the seat was reached and a receipt
      was written. ``verdict`` / ``agree`` set (``verdict`` may be ``None`` when
      the seat answered but emitted no parseable JSON — ``record["shadow"]``
      keeps the raw bytes honestly).
    - ``enabled=True`` + ``status="absent"`` → the seat could not judge this
      turn; ``absent_reason`` names why. Still a *receipt*, never a failure.
    """

    enabled: bool
    status: Optional[str] = None  # "ok" | "absent" | None (disabled)
    absent_reason: Optional[str] = None
    verdict: Optional[str] = None
    agree: Optional[bool] = None
    record: Optional[Dict[str, Any]] = None
    receipt_path: Optional[Path] = None
    error: Optional[str] = None
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Config (the capture.py / _load_coach_config idiom: a bad config reads as OFF).
# ---------------------------------------------------------------------------


def _load_config(repo_root: Path) -> dict:
    """Read ``<repo_root>/.guardkit/config.yaml``; empty dict if absent/unreadable
    (a bad config is treated as OFF, never a crash)."""
    path = repo_root / ".guardkit" / "config.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — a bad config never breaks a run
        logger.warning(
            "qav_shadow: could not read %s (%r) — treating as OFF", path, exc
        )
        return {}
    return data if isinstance(data, dict) else {}


def load_qav_shadow_config(repo_root: Path) -> dict:
    """The ``autobuild.coach.qav_shadow`` mapping (rides the coach config section).

    Empty dict when absent or malformed at any level (fail-open, like
    ``_load_coach_config``). Never raises.
    """
    data = _load_config(Path(repo_root))
    autobuild = data.get("autobuild")
    if not isinstance(autobuild, dict):
        return {}
    coach = autobuild.get("coach")
    if not isinstance(coach, dict):
        return {}
    qav = coach.get("qav_shadow")
    return qav if isinstance(qav, dict) else {}


def is_qav_shadow_enabled(repo_root: Path) -> bool:
    """Whether the QAV shadow lane is ON for ``repo_root``.

    Precedence (the capture.py idiom): ``GUARDKIT_QAV_SHADOW`` env
    (truthy/falsy) > ``.guardkit/config.yaml`` ``autobuild.coach.qav_shadow.enabled``
    > ``False``. **Default OFF everywhere.**
    """
    env = os.environ.get(QAV_SHADOW_ENV)
    if env is not None:
        token = env.strip().lower()
        if token in _TRUTHY:
            return True
        if token in _FALSY:
            return False
        logger.warning(
            "%s=%r is not a recognised boolean — treating as OFF", QAV_SHADOW_ENV, env
        )
        return False
    return bool(load_qav_shadow_config(Path(repo_root)).get("enabled", False))


def _endpoint(cfg: dict) -> str:
    """The seat address, by the one shared rule
    (:func:`guardkit.lib.client_env.resolve_base_url`): the config block's own
    ``endpoint``, then ``GUARDKIT_QAV_SHADOW_URL``, then ``OPENAI_BASE_URL``,
    then :data:`DEFAULT_ENDPOINT`."""
    v = cfg.get("endpoint")
    return resolve_base_url(
        explicit=v if isinstance(v, str) else None,
        env_vars=(QAV_SHADOW_URL_ENV, "OPENAI_BASE_URL"),
        default=DEFAULT_ENDPOINT,
    )


def _model(cfg: dict) -> str:
    v = cfg.get("model")
    return v.strip() if isinstance(v, str) and v.strip() else DEFAULT_MODEL


def _timeout_s(cfg: dict) -> float:
    v = cfg.get("timeout_seconds")
    try:
        t = float(v)
        return t if t > 0 else DEFAULT_TIMEOUT_S
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S


def _exclusive_tokens(cfg: dict) -> Tuple[str, ...]:
    v = cfg.get("exclusive_sets")
    if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v) and v:
        return tuple(v)
    return EXCLUSIVE_SET_TOKENS


# ---------------------------------------------------------------------------
# Single-slot / eligibility probe (the -np 1 held-out-runner law).
# ---------------------------------------------------------------------------


def _default_running_probe(endpoint: str) -> RunningProbe:
    """Build a ``/running`` probe from the seat endpoint.

    The endpoint may be configured as a base (``http://host:9000/v1``) or as the full
    completions URL (``http://host:9000/v1/chat/completions`` — the held-out runner's
    convention, and the shape the B3 live smoke used when this derivation's original
    trailing-``/v1``-strip produced ``.../chat/completions/running`` and fail-opened as
    ``probe_refused``). llama-swap serves ``/running`` at the server root, so derive from
    scheme+netloc and ignore the path entirely."""
    parts = urllib.parse.urlsplit(endpoint)
    running_url = f"{parts.scheme}://{parts.netloc}/running"

    def _probe() -> Optional[List[Dict[str, Any]]]:
        try:
            with urllib.request.urlopen(running_url, timeout=5) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            return None
        running = data.get("running") if isinstance(data, dict) else None
        return running if isinstance(running, list) else None

    return _probe


def _running_model_names(running: Sequence[Dict[str, Any]]) -> List[str]:
    return [
        str(e.get("model", "")).strip()
        for e in running
        if isinstance(e, dict) and str(e.get("model", "")).strip()
    ]


def _exclusive_set_hit(
    running: Sequence[Dict[str, Any]], tokens: Sequence[str]
) -> Optional[str]:
    """The first running model name matching an exclusive-set token (or None)."""
    low_tokens = [t.lower() for t in tokens]
    for name in _running_model_names(running):
        low = name.lower()
        if any(tok in low for tok in low_tokens):
            return name
    return None


def _slot_busy(running: Sequence[Dict[str, Any]]) -> Optional[str]:
    """The first running model whose state names a live generation (or None)."""
    for e in running:
        if not isinstance(e, dict):
            continue
        if str(e.get("state", "")).strip().lower() in _BUSY_STATES:
            return str(e.get("model", "?"))
    return None


def _probe_eligibility(
    running: Sequence[Dict[str, Any]], tokens: Sequence[str]
) -> Tuple[bool, Optional[str], str]:
    """Decide whether the shadow may call the seat given a parsed ``/running`` list.

    Returns ``(eligible, absent_reason, note)``. An exclusive-set member present
    means a primary workload holds the box (``skipped_set``); any other live
    generation on the single slot means we do not collide (``slot_busy``).
    """
    hit = _exclusive_set_hit(running, tokens)
    if hit:
        return False, "skipped_set", f"exclusive set active on the box: {hit}"
    busy = _slot_busy(running)
    if busy:
        return False, "slot_busy", f"single slot held by a live drive: {busy}"
    return True, None, "seat slot free"


# ---------------------------------------------------------------------------
# Default seat call (OpenAI-compatible llama-swap — the impure edge, lazy openai).
# ---------------------------------------------------------------------------


def _default_seat_call(
    endpoint: str,
    *,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> SeatCall:
    """Build a bounded OpenAI-compatible seat call against llama-swap.

    ``openai`` is imported lazily so the pure/flag paths (and every unit test,
    which injects its own ``SeatCall``) carry no ``openai`` dependency.
    """

    # The endpoint may be configured as an SDK base (…/v1) or as the full completions
    # URL (…/v1/chat/completions — the held-out runner's convention; the B3 live smoke
    # 404'd as absent(transport_aborted) when the full form was passed straight through
    # as base_url). Normalize to the base the SDK expects, accepting both shapes.
    base_url = endpoint.rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]

    def _call(
        system_prompt: str, user_prompt: str, model: str, timeout_s: float
    ) -> SeatResult:
        from openai import OpenAI  # lazy — only the real-seat path needs it

        # OPENAI_API_KEY when it is set, else the placeholder llama-swap has
        # always ignored. Never logged — it goes into the request and nowhere else.
        client = OpenAI(base_url=base_url, api_key=resolve_api_key(), timeout=timeout_s)
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        truncated = getattr(choice, "finish_reason", None) == "length"
        usage: Optional[Dict[str, Any]] = None
        raw_usage = getattr(resp, "usage", None)
        if raw_usage is not None:
            usage = {
                "prompt_tokens": getattr(raw_usage, "prompt_tokens", None),
                "completion_tokens": getattr(raw_usage, "completion_tokens", None),
                "total_tokens": getattr(raw_usage, "total_tokens", None),
            }
        return SeatResult(text=text, usage=usage, truncated=truncated)

    return _call


def _classify_seat_exc(exc: BaseException) -> str:
    """Map a seat-call exception to ``timeout`` or ``transport_aborted``."""
    if isinstance(exc, TimeoutError):
        return "timeout"
    name = type(exc).__name__.lower()
    if "timeout" in name or "timedout" in name:
        return "timeout"
    return "transport_aborted"


# ---------------------------------------------------------------------------
# Verdict extraction (first balanced JSON; no-JSON keeps raw bytes honestly).
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced JSON object from a raw completion (or None).

    Tolerant of a reasoning ``<think>`` block and ``` fences (the assistant
    format the tune emits). Never raises — a completion with no parseable
    object returns ``None`` so the caller keeps the raw bytes honestly.
    """
    cleaned = _THINK_RE.sub("", text or "").strip()
    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    if start == -1:
        return None
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
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _coerce_verdict(raw: Any) -> Optional[str]:
    token = str(raw or "").strip().lower()
    return token if token in ("approve", "reject") else None


def _coerce_findings(raw: Any) -> List[Dict[str, str]]:
    """Coerce the verdict's findings into ``[{class, locus}]`` (best-effort)."""
    out: List[Dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for f in raw:
        if not isinstance(f, dict):
            continue
        cls = str(f.get("class", "") or "").strip()
        locus = str(f.get("locus", "") or "").strip()
        if cls or locus:
            out.append({"class": cls, "locus": locus})
    return out


def _normalize_coach(decision: Any) -> str:
    """Normalize the coach's final decision to the QAV verdict axis.

    Autobuild's post-override decision is ``approve`` or ``feedback`` (and, on a
    hard coach failure, ``error``). The QAV judge speaks ``approve`` / ``reject``.
    Anything that is not an explicit ``approve`` is a non-approval, i.e. reject.
    """
    return "approve" if str(decision).strip().lower() == "approve" else "reject"


# ---------------------------------------------------------------------------
# Receipt (design §"The receipt") — one shape for ok + absent records.
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_record(
    *,
    task_id: str,
    turn: int,
    ts: str,
    coach_decision: str,
    status: str,
    absent_reason: Optional[str],
    agree: Optional[bool],
    verdict: Optional[str],
    findings: List[Dict[str, str]],
    json_extracted: bool,
    raw: Optional[str],
    model: str,
    endpoint: str,
    bundle_sha256: Optional[str],
    prompt_sha256: Optional[str],
    sampling: Dict[str, Any],
    usage: Optional[Dict[str, Any]],
    wall_time_s: Optional[float],
    truncated: bool,
) -> Dict[str, Any]:
    return {
        # identity
        "task_id": task_id,
        "turn": turn,
        "ts": ts,
        "coach_decision": coach_decision,
        "status": status,  # "ok" | "absent"
        "absent_reason": absent_reason,  # None on ok
        "agree": agree,  # precomputed; None when there is no verdict to compare
        # the shadow verdict (raw-bytes-on-no-JSON honesty)
        "shadow": {
            "verdict": verdict,
            "findings": findings,
            "json_extracted": json_extracted,
            "raw": raw,
        },
        # provenance
        "provenance": {
            "model": model,
            "endpoint": endpoint,
            "bundle_sha256": bundle_sha256,
            "prompt_sha256": prompt_sha256,
            "system_sha256": QAV_SYSTEM_PROMPT_SHA256,
            "bundle_schema_sha": PINNED_BUNDLE_SCHEMA_SHA,
            "sampling": sampling,
            "usage": usage,
            "wall_time_s": wall_time_s,
            "truncated": truncated,
        },
    }


def _receipt_path(repo: Path, task_id: str, turn: int) -> Path:
    """The per-turn receipt path (beside ``coach_turn_{turn}.json``).

    Uses the ``paths.py`` template constant when importable (source of truth),
    with a literal fallback so an import quirk can never break the shadow.
    """
    rel = ".guardkit/autobuild/{task_id}/qav_shadow_turn_{turn}.json"
    try:  # lazy — avoids importing the heavy orchestrator package at flag time
        from guardkit.orchestrator.paths import TaskArtifactPaths

        rel = TaskArtifactPaths.QAV_SHADOW
    except Exception:  # noqa: BLE001 — never let an import quirk break the shadow
        pass
    return repo / rel.format(task_id=task_id, turn=turn)


def _main_checkout_root(repo: Path) -> Path:
    """The queue sink root: the OUTERMOST checkout this build runs in.

    Autobuild runs each feature inside a nested git worktree
    (``<build checkout>/.guardkit/worktrees/FEAT-X/``). A queue row appended
    inside that nested worktree is a row nobody ever reads: the worktree is
    gitignored, transient, and every reader (the receipts harvest included)
    looks at ``<build checkout>/.guardkit/qav-shadow/queue.jsonl``. Exactly this
    made the shadow look silent across four builds on 2026-08-25/26 — the rows
    WERE written, but only inside the soon-deleted nested worktrees.

    Resolution: when ``repo`` is a linked git worktree — its ``.git`` is a
    *file* reading ``gitdir: <main>/.git/worktrees/<name>`` — return
    ``<main>``; otherwise return ``repo`` itself. Never raises; any surprise
    in the layout falls back to ``repo``.
    """
    try:
        gitfile = repo / ".git"
        if not gitfile.is_file():
            return repo
        text = gitfile.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^gitdir:\s*(.+?)\s*$", text, flags=re.MULTILINE)
        if not m:
            return repo
        gitdir = Path(m.group(1))
        if not gitdir.is_absolute():
            gitdir = (repo / gitdir).resolve()
        # <main>/.git/worktrees/<name> -> <main>
        if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
            main_root = gitdir.parent.parent.parent
            if main_root.is_dir():
                return main_root
        return repo
    except Exception:  # noqa: BLE001 — path resolution can never break the shadow
        return repo


def _write_receipt(repo: Path, task_id: str, turn: int, record: Dict[str, Any]) -> Optional[Path]:
    """Write the per-turn receipt + append the queue row. A failed write itself
    swallows to WARNING (design §"The one law"); returns the path or None."""
    path = _receipt_path(repo, task_id, turn)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        logger.warning(
            "qav_shadow: unwritable receipt %s (%r) — record dropped", path, exc
        )
        path = None  # type: ignore[assignment]

    # The queue append is independent — a receipt that wrote must still try the
    # sink, and a sink failure must not lose the receipt. The sink lives at the
    # MAIN checkout root, not the (possibly nested) worktree this turn ran in:
    # one build = one checkout = one unambiguous queue file (2026-08-26 fix —
    # see _main_checkout_root for the silent-four-builds story).
    line = json.dumps(record, sort_keys=True) + "\n"
    root = _main_checkout_root(repo)
    qpath = root / QAV_SHADOW_QUEUE
    try:
        qpath.parent.mkdir(parents=True, exist_ok=True)
        with qpath.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError as exc:
        if root != repo:
            # Last resort: better a row beside the worktree receipt than none.
            fallback = repo / QAV_SHADOW_QUEUE
            try:
                fallback.parent.mkdir(parents=True, exist_ok=True)
                with fallback.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                logger.warning(
                    "qav_shadow: unwritable queue %s (%r) — row appended to %s instead",
                    qpath,
                    exc,
                    fallback,
                )
            except OSError as exc2:
                logger.warning(
                    "qav_shadow: unwritable queue %s (%r after %r) — row dropped",
                    fallback,
                    exc2,
                    exc,
                )
        else:
            logger.warning(
                "qav_shadow: unwritable queue %s (%r) — row dropped", qpath, exc
            )
    return path


def _read_bundle(bundle_path: Path) -> Optional[Dict[str, Any]]:
    """Read the coach evidence bundle; None if missing/corrupt (⇒ no_bundle)."""
    if not bundle_path.is_file():
        return None
    try:
        obj = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "qav_shadow: unreadable coach evidence bundle %s (%r) — absent(no_bundle)",
            bundle_path,
            exc,
        )
        return None
    return obj if isinstance(obj, dict) else None


def _emit_absent_last_resort(
    repo: Path, task_id: str, turn: int, coach_decision: str
) -> None:
    """Write an ``absent(internal_error)`` row after a guard-caught crash.

    The contract this protects: with the lane ON, EVERY coach verdict leaves
    exactly one queue row — a real comparison when the shadow answered, an
    absent row when it could not. Before 2026-08-26 a crash swallowed by the
    never-raise guards left no row at all, indistinguishable from the lane
    being off. Never raises; does nothing when the lane is OFF or when this
    turn's receipt already landed (never two rows for one verdict).
    """
    try:
        repo = Path(repo)
        if not is_qav_shadow_enabled(repo):
            return
        if _receipt_path(repo, task_id, turn).exists():
            return  # a record for this verdict already landed
        cfg = load_qav_shadow_config(repo)
        record = _build_record(
            task_id=task_id,
            turn=turn,
            ts=_utc_now_iso(),
            coach_decision=str(coach_decision),
            status="absent",
            absent_reason="internal_error",
            agree=None,
            verdict=None,
            findings=[],
            json_extracted=False,
            raw=None,
            model=_model(cfg),
            endpoint=_endpoint(cfg),
            bundle_sha256=None,
            prompt_sha256=None,
            sampling={
                "temperature": _DEFAULT_TEMPERATURE,
                "max_tokens": _DEFAULT_MAX_TOKENS,
            },
            usage=None,
            wall_time_s=None,
            truncated=False,
        )
        _write_receipt(repo, task_id, turn, record)
    except Exception as exc:  # noqa: BLE001 — the last resort must never raise
        logger.warning(
            "qav_shadow: last-resort absent row also failed (%r) for %s turn %s",
            exc,
            task_id,
            turn,
        )


# ---------------------------------------------------------------------------
# The run (synchronous, NEVER raises) + the fire-and-forget scheduler.
# ---------------------------------------------------------------------------


def run_qav_shadow(
    repo_root: Path,
    task_id: str,
    turn: int,
    coach_decision: str,
    *,
    seat_call: Optional[SeatCall] = None,
    running_probe: Optional[RunningProbe] = None,
    now: Optional[Callable[[], str]] = None,
) -> ShadowOutcome:
    """Log a QAV second opinion beside the coach verdict for one turn.

    Synchronous and **never raises** (belt-and-suspenders guard, the
    compile-shadow Fallback law). When the flag is OFF this is a provable
    no-op: it returns after the config read with NO bundle read, NO probe, NO
    seat call, and NO file. When ON it reads the bundle from
    ``coach_evidence_turn_{turn}.json`` (missing/corrupt ⇒ ``absent(no_bundle)``),
    probes the single slot, calls the seat (bounded), extracts the verdict, and
    writes the receipt + queue row. ``seat_call`` / ``running_probe`` are
    injectable so tests never touch the network.
    """
    try:
        return _run_inner(
            Path(repo_root),
            task_id,
            turn,
            coach_decision,
            seat_call=seat_call,
            running_probe=running_probe,
            now=now or _utc_now_iso,
        )
    except Exception as exc:  # noqa: BLE001 — the shadow can never touch the build
        logger.warning(
            "qav_shadow: run guard swallowed %r for %s turn %s (verdict untouched)",
            exc,
            task_id,
            turn,
        )
        # Even a swallowed crash must leave its one queue row (absent, named).
        _emit_absent_last_resort(Path(repo_root), task_id, turn, coach_decision)
        return ShadowOutcome(enabled=True, error=f"guard:{type(exc).__name__}")


def _run_inner(
    repo: Path,
    task_id: str,
    turn: int,
    coach_decision: str,
    *,
    seat_call: Optional[SeatCall],
    running_probe: Optional[RunningProbe],
    now: Callable[[], str],
) -> ShadowOutcome:
    if not is_qav_shadow_enabled(repo):
        return ShadowOutcome(
            enabled=False, note="qav_shadow flag OFF — no-op, no probe, no seat call"
        )

    cfg = load_qav_shadow_config(repo)
    endpoint = _endpoint(cfg)
    model = _model(cfg)
    timeout_s = _timeout_s(cfg)
    tokens = _exclusive_tokens(cfg)
    sampling = {"temperature": _DEFAULT_TEMPERATURE, "max_tokens": _DEFAULT_MAX_TOKENS}
    ts = now()

    def _emit_absent(
        reason: str,
        *,
        bundle_sha256: Optional[str] = None,
        prompt_sha256: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        wall_time_s: Optional[float] = None,
    ) -> ShadowOutcome:
        record = _build_record(
            task_id=task_id,
            turn=turn,
            ts=ts,
            coach_decision=coach_decision,
            status="absent",
            absent_reason=reason,
            agree=None,
            verdict=None,
            findings=[],
            json_extracted=False,
            raw=None,
            model=model,
            endpoint=endpoint,
            bundle_sha256=bundle_sha256,
            prompt_sha256=prompt_sha256,
            sampling=sampling,
            usage=usage,
            wall_time_s=wall_time_s,
            truncated=False,
        )
        path = _write_receipt(repo, task_id, turn, record)
        return ShadowOutcome(
            enabled=True,
            status="absent",
            absent_reason=reason,
            record=record,
            receipt_path=path,
        )

    # 1. The bundle already exists in the exact QAV 25-field shape.
    # TASK-SBHO-002: read from private dir with legacy fallback.
    from guardkit.orchestrator.paths import TaskArtifactPaths

    bundle_path = TaskArtifactPaths.coach_evidence_path(task_id, turn, repo)
    bundle = _read_bundle(bundle_path)
    if bundle is None:
        return _emit_absent("no_bundle")

    user_message = build_user_message(bundle)
    bundle_json = json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False)
    bundle_sha = hashlib.sha256(bundle_json.encode("utf-8")).hexdigest()
    prompt_sha = hashlib.sha256(user_message.encode("utf-8")).hexdigest()

    # 2. Fresh single-slot probe before the call (the held-out-runner law).
    probe = running_probe or _default_running_probe(endpoint)
    try:
        running = probe()
    except Exception as exc:  # noqa: BLE001 — an unreachable probe is not a busy signal
        logger.warning("qav_shadow: /running probe raised %r — absent(probe_refused)", exc)
        running = None
    if running is None:
        # swap down / model absent — do not attempt the call.
        return _emit_absent(
            "probe_refused", bundle_sha256=bundle_sha, prompt_sha256=prompt_sha
        )
    eligible, reason, note = _probe_eligibility(running, tokens)
    if not eligible:
        logger.info("qav_shadow: %s turn %s — %s (%s)", task_id, turn, reason, note)
        return _emit_absent(
            reason or "slot_busy",
            bundle_sha256=bundle_sha,
            prompt_sha256=prompt_sha,
        )

    # 3. The bounded seat call (hard timeout ⇒ absent(timeout)).
    call = seat_call or _default_seat_call(endpoint)
    t0 = time.monotonic()
    try:
        seat = call(QAV_SYSTEM_PROMPT, user_message, model, timeout_s)
    except Exception as exc:  # noqa: BLE001 — a seat outage is a named absent, never a raise
        wall = time.monotonic() - t0
        reason = _classify_seat_exc(exc)
        logger.warning(
            "qav_shadow: seat call failed (%s) — absent(%s)", type(exc).__name__, reason
        )
        return _emit_absent(
            reason,
            bundle_sha256=bundle_sha,
            prompt_sha256=prompt_sha,
            wall_time_s=wall,
        )
    wall = time.monotonic() - t0

    # 4. Verdict extraction (first balanced JSON; raw bytes kept on no-JSON).
    obj = _extract_first_json(seat.text)
    if obj is None:
        verdict: Optional[str] = None
        findings: List[Dict[str, str]] = []
        json_extracted = False
        raw: Optional[str] = seat.text
    else:
        verdict = _coerce_verdict(obj.get("verdict"))
        findings = _coerce_findings(obj.get("findings"))
        json_extracted = True
        raw = None

    agree: Optional[bool] = None
    if verdict in ("approve", "reject"):
        # An errored coach made no substantive call — comparing against it would
        # conflate "coach crashed" with "coach rejected" and pollute the burn-in
        # tallies the graduation decision reads (coach advisory, B1 review).
        if str(coach_decision).strip().lower() == "error":
            agree = None
        else:
            agree = _normalize_coach(coach_decision) == verdict

    record = _build_record(
        task_id=task_id,
        turn=turn,
        ts=ts,
        coach_decision=coach_decision,
        status="ok",
        absent_reason=None,
        agree=agree,
        verdict=verdict,
        findings=findings,
        json_extracted=json_extracted,
        raw=raw,
        model=model,
        endpoint=endpoint,
        bundle_sha256=bundle_sha,
        prompt_sha256=prompt_sha,
        sampling=sampling,
        usage=seat.usage,
        wall_time_s=wall,
        truncated=seat.truncated,
    )
    path = _write_receipt(repo, task_id, turn, record)
    return ShadowOutcome(
        enabled=True,
        status="ok",
        verdict=verdict,
        agree=agree,
        record=record,
        receipt_path=path,
    )


def schedule_qav_shadow(
    repo_root: Path,
    *,
    task_id: str,
    turn: int,
    coach_decision: str,
    runner: Optional[Callable[..., Any]] = None,
) -> Optional[threading.Thread]:
    """Fire-and-forget the shadow off the turn's critical path (the ``_safe_emit``
    spirit — schedule, don't block, swallow all).

    The coach seam runs synchronously (``invoke_coach`` completes via an internal
    ``asyncio.run`` before returning), so there is no running loop to
    ``create_task`` onto; the fire-and-forget vehicle here is a non-daemon thread.
    A warm judgment is ~1.3–1.8 s while a build runs for minutes, so the thread
    finishes well within the build. The existing 60s seat timeout is the natural
    upper bound — the thread will never block shutdown past that ceiling.

    **Provable no-op when OFF:** the flag is read first; if OFF this returns
    ``None`` immediately — no thread, no ``/running`` probe, no seat call, no
    file (the DCL capture Fallback law). Returns the spawned thread (or None).
    NEVER raises.
    """
    try:
        repo = Path(repo_root)
        if not is_qav_shadow_enabled(repo):
            return None
        run = runner or run_qav_shadow

        def _body() -> None:
            try:
                run(repo, task_id, turn, coach_decision)
            except Exception as exc:  # noqa: BLE001 — the thread can never surface
                logger.warning("qav_shadow: threaded run swallowed %r", exc)
                # A crashed run must still leave its one (absent) queue row.
                _emit_absent_last_resort(repo, task_id, turn, coach_decision)

        thread = threading.Thread(
            target=_body, name=f"qav-shadow-{task_id}-t{turn}", daemon=False
        )
        thread.start()
        return thread
    except Exception as exc:  # noqa: BLE001 — scheduling must never touch the turn
        logger.warning(
            "qav_shadow: schedule guard swallowed %r for %s turn %s", exc, task_id, turn
        )
        # Best effort: a scheduling failure must still leave its one queue row
        # (the helper re-checks the flag itself and never raises).
        _emit_absent_last_resort(Path(repo_root), task_id, turn, coach_decision)
        return None
