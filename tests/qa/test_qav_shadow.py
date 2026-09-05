"""Unit tests for the QAV shadow mode lane (§7, log-only second opinion).

Proves the design's one law on the shadow seam (mirrors the Fallback-law tests
of the since-deleted DCL capture lane):

* **default-OFF** — with the flag absent, ``run_qav_shadow`` / ``schedule_qav_shadow``
  do NOTHING: no receipt, no queue, no ``/running`` probe, no seat call (both are
  injected as raise-if-called spies to assert they are never touched), and the
  coach path is byte-identical (proven by monkeypatched-absence of the run);
* **ON, happy path** — a mocked seat writes the full receipt with the correct
  precomputed ``agree`` + provenance sha256s + queue row;
* **every absent path** writes the right ``absent_reason`` (no_bundle,
  probe_refused, skipped_set, slot_busy, timeout, transport_aborted);
* **never-raise** — the run swallows an internal write failure to WARNING and
  the scheduler swallows a throwing runner; neither ever raises;
* **envelope pins** — the copied adf ``SYSTEM_PROMPT`` sha256 still matches;
* **review-summary** renders the QAV section from receipts (and is omitted when
  there are none); the **checkpoint pattern** archives a shadow receipt on rollback.

Zero network: the seat call and the ``/running`` probe are always injected.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from guardkit.qa import qav_shadow as q


# ===========================================================================
# Fixtures / helpers.
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_shadow_env(monkeypatch):
    """The env override must not leak in from the ambient shell."""
    monkeypatch.delenv(q.QAV_SHADOW_ENV, raising=False)


def _write_config(repo: Path, **qav) -> None:
    cfg = repo / ".guardkit"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "config.yaml").write_text(
        yaml.safe_dump({"autobuild": {"coach": {"qav_shadow": qav}}}),
        encoding="utf-8",
    )


def _write_bundle(repo: Path, task_id: str, turn: int, bundle: dict) -> Path:
    d = repo / ".guardkit" / "autobuild" / task_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"coach_evidence_turn_{turn}.json"
    # The orchestrator writes indent=2, default=str (NOT sort_keys); mirror it.
    p.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    return p


def _sample_bundle() -> dict:
    return {
        "honesty": {"verified": True, "discrepancies": []},
        "task_type": "implementation",
        "quality_gates": {"tests_passed": True},
        "profile_name": "default",
    }


def _assistant(verdict: str, findings=None) -> str:
    obj = {"verdict": verdict, "findings": findings or []}
    return f"<think>weighing the evidence</think>\n\n```json\n{json.dumps(obj)}\n```"


def _seat(text: str, *, usage=None, truncated=False):
    """A seat spy returning a fixed SeatResult, recording its call args."""
    calls = []

    def _call(system, user, model, timeout_s):
        calls.append((system, user, model, timeout_s))
        return q.SeatResult(text=text, usage=usage, truncated=truncated)

    _call.calls = calls  # type: ignore[attr-defined]
    return _call


def _raise_if_called(*_a, **_k):
    raise AssertionError("must not be called when the flag is OFF")


def _free_probe(models=None):
    """A probe returning a free running list (no exclusive set, no busy slot)."""
    running = models if models is not None else [{"model": "qav-shadow", "state": "ready"}]
    return lambda: running


def _read_receipt(repo: Path, task_id: str, turn: int) -> dict:
    p = repo / ".guardkit" / "autobuild" / task_id / f"qav_shadow_turn_{turn}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _read_queue(repo: Path) -> list:
    p = repo / q.QAV_SHADOW_QUEUE
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ===========================================================================
# Envelope pins (fence: the copied adf constants must not drift).
# ===========================================================================


def test_system_prompt_sha_pinned():
    """The copied ``SYSTEM_PROMPT`` sha256 matches its pin (drift caught by CI)."""
    recomputed = hashlib.sha256(q.QAV_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert recomputed == q.QAV_SYSTEM_PROMPT_SHA256


def test_system_prompt_asks_for_the_json_format():
    """The 2026-09-05 sentence is in the prompt (the seat was never asked before)."""
    assert q.QAV_SYSTEM_PROMPT.endswith(
        "Answer with the verdict JSON object only: no prose, no markdown, no code fences."
    )


def test_build_user_message_is_sorted_and_stable():
    """The training envelope re-serializes the bundle sort_keys=True, deterministically."""
    a = q.build_user_message({"b": 1, "a": 2})
    b = q.build_user_message({"a": 2, "b": 1})
    assert a == b  # key order in the input does not change the envelope
    # keys appear sorted in the embedded JSON
    assert a.index('"a"') < a.index('"b"')
    assert "## Live-gate results\n(none available)\n" in a


# ===========================================================================
# Config / flag precedence (the capture.py idiom).
# ===========================================================================


def test_is_enabled_default_false(tmp_path):
    assert q.is_qav_shadow_enabled(tmp_path) is False


def test_is_enabled_config_true(tmp_path):
    _write_config(tmp_path, enabled=True)
    assert q.is_qav_shadow_enabled(tmp_path) is True


def test_env_truthy_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(q.QAV_SHADOW_ENV, "yes")
    assert q.is_qav_shadow_enabled(tmp_path) is True


def test_env_falsy_overrides_config_on(tmp_path, monkeypatch):
    _write_config(tmp_path, enabled=True)
    monkeypatch.setenv(q.QAV_SHADOW_ENV, "0")
    assert q.is_qav_shadow_enabled(tmp_path) is False


def test_bad_config_is_off(tmp_path):
    cfg = tmp_path / ".guardkit"
    cfg.mkdir(parents=True)
    (cfg / "config.yaml").write_text("autobuild: [not, a, mapping\n", encoding="utf-8")
    assert q.is_qav_shadow_enabled(tmp_path) is False


def test_config_defaults(tmp_path):
    """Endpoint/model/timeout fall back to the module defaults when unset."""
    cfg = q.load_qav_shadow_config(tmp_path)
    assert cfg == {}
    assert q._endpoint(cfg) == q.DEFAULT_ENDPOINT
    assert q._model(cfg) == q.DEFAULT_MODEL
    assert q._timeout_s(cfg) == q.DEFAULT_TIMEOUT_S


# ===========================================================================
# DEFAULT-OFF: provable no-op — zero files, zero probe, zero seat.
# ===========================================================================


def test_off_run_is_noop(tmp_path):
    """Flag OFF ⇒ run returns disabled, no receipt/queue, probe+seat untouched."""
    repo = tmp_path
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())  # bundle PRESENT but flag off

    outcome = q.run_qav_shadow(
        repo,
        "TASK-1",
        1,
        "approve",
        seat_call=_raise_if_called,
        running_probe=_raise_if_called,
    )

    assert outcome.enabled is False
    assert not (repo / ".guardkit" / "autobuild" / "TASK-1" / "qav_shadow_turn_1.json").exists()
    assert not (repo / q.QAV_SHADOW_QUEUE).exists()


def test_off_schedule_is_noop_coach_path_untouched(tmp_path, monkeypatch):
    """Flag OFF ⇒ schedule spawns no thread and never invokes the run.

    Proves the coach path is byte-identical by monkeypatched-absence: the run is
    replaced with a raise-if-called sentinel and must never fire.
    """
    repo = tmp_path
    monkeypatch.setattr(q, "run_qav_shadow", _raise_if_called)

    thread = q.schedule_qav_shadow(repo, task_id="TASK-1", turn=1, coach_decision="approve")

    assert thread is None
    assert not (repo / ".guardkit" / "autobuild").exists()
    assert not (repo / q.QAV_SHADOW_QUEUE).exists()


# ===========================================================================
# ON, happy path: the full receipt + queue row + precomputed agree.
# ===========================================================================


def test_on_happy_approve_agrees(tmp_path):
    repo = tmp_path
    _write_config(repo, enabled=True)
    bundle = _sample_bundle()
    _write_bundle(repo, "TASK-1", 1, bundle)
    seat = _seat(_assistant("approve"), usage={"total_tokens": 42})

    outcome = q.run_qav_shadow(
        repo, "TASK-1", 1, "approve",
        seat_call=seat, running_probe=_free_probe(),
        now=lambda: "2026-07-25T00:00:00Z",
    )

    assert outcome.status == "ok"
    assert outcome.verdict == "approve"
    assert outcome.agree is True

    record = _read_receipt(repo, "TASK-1", 1)
    assert record["task_id"] == "TASK-1"
    assert record["turn"] == 1
    assert record["ts"] == "2026-07-25T00:00:00Z"
    assert record["coach_decision"] == "approve"
    assert record["status"] == "ok"
    assert record["absent_reason"] is None
    assert record["agree"] is True
    assert record["shadow"] == {
        "verdict": "approve",
        "findings": [],
        "json_extracted": True,
        "extraction_method": "json",
        "absent_reason": None,
        "reasoning": None,
        "raw": None,
    }
    prov = record["provenance"]
    assert prov["model"] == q.DEFAULT_MODEL
    assert prov["endpoint"] == q.DEFAULT_ENDPOINT
    assert prov["system_sha256"] == q.QAV_SYSTEM_PROMPT_SHA256
    assert prov["bundle_schema_sha"] == q.PINNED_BUNDLE_SCHEMA_SHA
    assert len(prov["bundle_sha256"]) == 64
    assert len(prov["prompt_sha256"]) == 64
    assert prov["usage"] == {"total_tokens": 42}
    assert prov["truncated"] is False

    # bundle sha256 is over the re-serialized (sort_keys) bundle the seat saw.
    expected = hashlib.sha256(
        json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert prov["bundle_sha256"] == expected

    # queue.jsonl carries the same object (sort_keys bytes).
    assert _read_queue(repo) == [record]

    # the seat was called with the pinned system prompt + the training envelope.
    (system, user, model, _timeout) = seat.calls[0]  # type: ignore[attr-defined]
    assert system == q.QAV_SYSTEM_PROMPT
    assert user == q.build_user_message(bundle)
    assert model == q.DEFAULT_MODEL


def test_on_disagreement_coach_feedback_shadow_approve(tmp_path):
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-2", 1, _sample_bundle())

    outcome = q.run_qav_shadow(
        repo, "TASK-2", 1, "feedback",
        seat_call=_seat(_assistant("approve")), running_probe=_free_probe(),
    )

    assert outcome.status == "ok"
    assert outcome.verdict == "approve"
    assert outcome.agree is False  # coach feedback (=reject) vs shadow approve


def test_on_disagreement_coach_approve_shadow_reject(tmp_path):
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-3", 2, _sample_bundle())
    findings = [{"class": "DC-05", "locus": "wiring"}]

    outcome = q.run_qav_shadow(
        repo, "TASK-3", 2, "approve",
        seat_call=_seat(_assistant("reject", findings)), running_probe=_free_probe(),
    )

    assert outcome.status == "ok"
    assert outcome.verdict == "reject"
    assert outcome.agree is False
    record = _read_receipt(repo, "TASK-3", 2)
    assert record["shadow"]["findings"] == findings


def test_on_no_json_keeps_raw_bytes_honestly(tmp_path):
    """The seat answered in a shape neither reader could read ⇒ status ok,
    verdict None, raw bytes kept, agree None (nothing to compare) — and the
    receipt NAMES it: extraction_method "none", absent_reason "unparseable"."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-4", 1, _sample_bundle())

    outcome = q.run_qav_shadow(
        repo, "TASK-4", 1, "approve",
        seat_call=_seat("I could not decide; here is prose with no object."),
        running_probe=_free_probe(),
    )

    assert outcome.status == "ok"
    assert outcome.verdict is None
    assert outcome.agree is None
    record = _read_receipt(repo, "TASK-4", 1)
    assert record["shadow"]["json_extracted"] is False
    assert record["shadow"]["verdict"] is None
    assert record["shadow"]["extraction_method"] == "none"
    assert record["shadow"]["absent_reason"] == "unparseable"
    assert record["shadow"]["raw"] == "I could not decide; here is prose with no object."


# ===========================================================================
# Absent paths: each writes the right absent_reason.
# ===========================================================================


def test_absent_no_bundle(tmp_path):
    repo = tmp_path
    _write_config(repo, enabled=True)  # no coach_evidence file written

    outcome = q.run_qav_shadow(
        repo, "TASK-1", 1, "approve",
        seat_call=_raise_if_called, running_probe=_raise_if_called,
    )

    assert outcome.status == "absent"
    assert outcome.absent_reason == "no_bundle"
    record = _read_receipt(repo, "TASK-1", 1)
    assert record["status"] == "absent"
    assert record["absent_reason"] == "no_bundle"
    assert record["agree"] is None
    assert record["shadow"]["verdict"] is None
    assert _read_queue(repo) == [record]


def test_absent_probe_refused(tmp_path):
    """An unreachable ``/running`` probe (None) ⇒ absent(probe_refused), no seat call."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())

    outcome = q.run_qav_shadow(
        repo, "TASK-1", 1, "approve",
        seat_call=_raise_if_called, running_probe=lambda: None,
    )

    assert outcome.absent_reason == "probe_refused"
    record = _read_receipt(repo, "TASK-1", 1)
    # bundle was read, so its sha is recorded even on this absent.
    assert len(record["provenance"]["bundle_sha256"]) == 64


def test_absent_probe_raises_is_probe_refused(tmp_path):
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())

    def _boom():
        raise ConnectionError("swap down")

    outcome = q.run_qav_shadow(
        repo, "TASK-1", 1, "approve",
        seat_call=_raise_if_called, running_probe=_boom,
    )

    assert outcome.absent_reason == "probe_refused"


def test_absent_skipped_set(tmp_path):
    """An exclusive-set member present in /running ⇒ absent(skipped_set)."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())

    outcome = q.run_qav_shadow(
        repo, "TASK-1", 1, "approve",
        seat_call=_raise_if_called,
        running_probe=_free_probe([{"model": "coach31", "state": "ready"}]),
    )

    assert outcome.absent_reason == "skipped_set"


def test_absent_slot_busy(tmp_path):
    """A live generation on the single slot ⇒ absent(slot_busy)."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())

    outcome = q.run_qav_shadow(
        repo, "TASK-1", 1, "approve",
        seat_call=_raise_if_called,
        running_probe=_free_probe([{"model": "some-workhorse", "state": "processing"}]),
    )

    assert outcome.absent_reason == "slot_busy"


def test_absent_timeout(tmp_path):
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())

    def _timeout(*_a, **_k):
        raise TimeoutError("seat took too long")

    outcome = q.run_qav_shadow(
        repo, "TASK-1", 1, "approve",
        seat_call=_timeout, running_probe=_free_probe(),
    )

    assert outcome.absent_reason == "timeout"


def test_absent_transport_aborted(tmp_path):
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())

    def _conn(*_a, **_k):
        raise ConnectionResetError("connection dropped")

    outcome = q.run_qav_shadow(
        repo, "TASK-1", 1, "approve",
        seat_call=_conn, running_probe=_free_probe(),
    )

    assert outcome.absent_reason == "transport_aborted"


# ===========================================================================
# Never-raise (the one law): the shadow can never touch the build.
# ===========================================================================


def test_run_never_raises_when_write_fails(tmp_path, monkeypatch, caplog):
    """An unexpected internal failure is swallowed to WARNING; run never raises."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())

    def _boom(*_a, **_k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(q, "_write_receipt", _boom)

    with caplog.at_level(logging.WARNING, logger="guardkit.qa.qav_shadow"):
        outcome = q.run_qav_shadow(
            repo, "TASK-1", 1, "approve",
            seat_call=_seat(_assistant("approve")), running_probe=_free_probe(),
        )

    assert outcome.enabled is True  # did not raise
    assert any("run guard swallowed" in r.message for r in caplog.records)


def test_unwritable_receipt_swallowed_to_warning(tmp_path, caplog):
    """A receipt whose parent cannot be created is swallowed-to-log, never raised."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())
    # The TASK-1 receipt dir exists (bundle write); block the queue sink instead
    # with a FILE where its parent dir must be, so only the queue append fails.
    (repo / ".guardkit" / "qav-shadow").write_text("i am a file", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="guardkit.qa.qav_shadow"):
        outcome = q.run_qav_shadow(
            repo, "TASK-1", 1, "approve",
            seat_call=_seat(_assistant("approve")), running_probe=_free_probe(),
        )

    # The per-turn receipt still wrote (its dir exists); only the queue failed.
    assert outcome.status == "ok"
    assert any("unwritable queue" in r.message for r in caplog.records)


def test_schedule_never_raises_when_runner_throws(tmp_path, caplog):
    """The scheduled thread swallows a throwing runner; nothing surfaces."""
    repo = tmp_path
    _write_config(repo, enabled=True)

    def _boom(*_a, **_k):
        raise RuntimeError("run exploded")

    with caplog.at_level(logging.WARNING, logger="guardkit.qa.qav_shadow"):
        thread = q.schedule_qav_shadow(
            repo, task_id="TASK-1", turn=1, coach_decision="approve", runner=_boom
        )
        assert thread is not None
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert any("threaded run swallowed" in r.message for r in caplog.records)


def test_schedule_never_raises_when_flag_read_throws(tmp_path, monkeypatch):
    """Even a fault reading the flag is swallowed — schedule returns None."""
    monkeypatch.setattr(q, "is_qav_shadow_enabled", _raise_if_called)
    assert q.schedule_qav_shadow(tmp_path, task_id="T", turn=1, coach_decision="approve") is None


def test_schedule_on_runs_via_thread(tmp_path):
    """Flag ON ⇒ schedule spawns a non-daemon thread that runs the injected runner."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    seen = {}

    def _runner(r, task_id, turn, decision):
        seen.update(task_id=task_id, turn=turn, decision=decision)

    thread = q.schedule_qav_shadow(
        repo, task_id="TASK-9", turn=3, coach_decision="feedback", runner=_runner
    )
    assert thread is not None
    thread.join(timeout=5)
    assert seen == {"task_id": "TASK-9", "turn": 3, "decision": "feedback"}


# ===========================================================================
# Durable shadow receipts (non-daemon thread survives scheduling scope exit).
# ===========================================================================


def test_durable_receipt_after_scope_exit(tmp_path):
    """AC-001: scheduling scope ends immediately — receipt still lands.

    The thread is non-daemon, so it survives the function return. A slow
    injected seat simulates the real-world delay; the receipt file is
    verified to exist after the thread completes.
    """
    import time as _time

    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-DUR", 1, _sample_bundle())

    # A seat that sleeps briefly to simulate a real network call.
    slow_seat_calls = []

    def _slow_seat(system, user, model, timeout_s):
        slow_seat_calls.append((system, user, model, timeout_s))
        _time.sleep(0.2)  # small delay — well under the 60s bound
        return q.SeatResult(text=_assistant("approve"), usage={"total_tokens": 10})

    # Schedule and let the scheduling scope exit immediately.
    thread = q.schedule_qav_shadow(
        repo,
        task_id="TASK-DUR",
        turn=1,
        coach_decision="approve",
        runner=lambda r, tid, t, d: q.run_qav_shadow(
            r, tid, t, d, seat_call=_slow_seat, running_probe=_free_probe()
        ),
    )
    assert thread is not None
    assert not thread.daemon  # the core fix: non-daemon

    # The scheduling scope has exited; the thread is still alive.
    assert thread.is_alive()

    # Wait for the thread to finish (bounded by seat timeout).
    thread.join(timeout=10)
    assert not thread.is_alive()

    # The receipt file must exist — this is the durability guarantee.
    receipt_path = repo / ".guardkit" / "autobuild" / "TASK-DUR" / "qav_shadow_turn_1.json"
    assert receipt_path.exists(), "receipt file must survive scope exit"
    record = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert record["task_id"] == "TASK-DUR"
    assert record["turn"] == 1
    assert record["status"] == "ok"
    assert record["agree"] is True

    # The seat was called exactly once.
    assert len(slow_seat_calls) == 1


def test_hanging_seat_cannot_block_beyond_timeout(tmp_path, monkeypatch):
    """AC-003: a hanging seat cannot extend process shutdown past the bound.

    The seat call has a timeout parameter (DEFAULT_TIMEOUT_S = 60). A thread
    that is blocked on a hanging seat will be unblocked after that timeout.
    The thread is non-daemon, so the receipt still lands — but the process
    shutdown is never extended beyond the bound.

    We inject a seat that respects the timeout by raising TimeoutError
    after the timeout period elapses, simulating a hung LLM swap.
    Uses a short timeout (2s) for test speed.
    """
    import time as _time

    repo = tmp_path
    _write_config(repo, enabled=True, timeout_seconds=2.0)
    _write_bundle(repo, "TASK-HANG", 1, _sample_bundle())

    def _hang_seat(system, user, model, timeout_s):
        # Block for the full timeout duration, then raise TimeoutError.
        # This simulates a hung LLM swap that finally times out.
        _time.sleep(timeout_s)
        raise TimeoutError("seat took too long")

    # Schedule the shadow. The runner wraps run_qav_shadow with our hanging seat.
    thread = q.schedule_qav_shadow(
        repo,
        task_id="TASK-HANG",
        turn=1,
        coach_decision="approve",
        runner=lambda r, tid, t, d: q.run_qav_shadow(
            r, tid, t, d, seat_call=_hang_seat, running_probe=_free_probe()
        ),
    )
    assert thread is not None
    assert not thread.daemon

    # Wait up to the seat timeout + margin. The thread must finish within
    # the timeout bound, not block indefinitely.
    thread.join(timeout=10)  # 2s timeout + 8s margin

    # The thread should have finished (seat call timed out).
    assert not thread.is_alive(), "thread must not block beyond seat timeout"

    # The receipt must still have been written (absent due to timeout).
    receipt_path = repo / ".guardkit" / "autobuild" / "TASK-HANG" / "qav_shadow_turn_1.json"
    assert receipt_path.exists(), "receipt must land even on timeout"
    record = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert record["status"] == "absent"
    assert record["absent_reason"] == "timeout"


def test_flag_off_no_thread_spawned(tmp_path):
    """AC-002: Flag-OFF is a provable no-op — no thread spawned.

    With the flag OFF, schedule_qav_shadow returns None and spawns no thread.
    """
    repo = tmp_path
    # No config written — flag defaults to OFF.

    thread = q.schedule_qav_shadow(
        repo,
        task_id="TASK-OFF",
        turn=1,
        coach_decision="approve",
    )

    assert thread is None

    # No receipt directory was created.
    assert not (repo / ".guardkit" / "autobuild").exists()
    assert not (repo / q.QAV_SHADOW_QUEUE).exists()


# ===========================================================================
# Review-summary section (design item 3).
# ===========================================================================


def _make_feature_result():
    from unittest.mock import MagicMock as _MM

    from guardkit.orchestrator.feature_orchestrator import (
        FeatureOrchestrationResult,
        TaskExecutionResult,
        WaveExecutionResult,
    )

    task = TaskExecutionResult(
        task_id="TASK-A",
        success=True,
        total_turns=1,
        final_decision="approved",
        sdk_total_invocations=1,
        sdk_turns_per_invocation=[1],
    )
    wave = WaveExecutionResult(
        wave_number=1, task_ids=["TASK-A"], results=[task], all_succeeded=True
    )
    return FeatureOrchestrationResult(
        feature_id="FEAT-X",
        success=True,
        status="completed",
        total_tasks=1,
        tasks_completed=1,
        tasks_failed=0,
        wave_results=[wave],
        worktree=_MM(),
    )


def _write_shadow_receipt(autobuild: Path, task_id: str, turn: int, **fields) -> None:
    d = autobuild / task_id
    d.mkdir(parents=True, exist_ok=True)
    record = {"task_id": task_id, "turn": turn, "status": "ok", "agree": True}
    record.update(fields)
    (d / f"qav_shadow_turn_{turn}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


def test_review_summary_section_renders(tmp_path):
    from guardkit.orchestrator.review_summary import ReviewSummaryGenerator

    autobuild = tmp_path / ".guardkit" / "autobuild"
    _write_shadow_receipt(autobuild, "TASK-A", 1, status="ok", agree=True)
    _write_shadow_receipt(autobuild, "TASK-B", 1, status="ok", agree=False)
    _write_shadow_receipt(autobuild, "TASK-C", 1, status="absent", agree=None)

    gen = ReviewSummaryGenerator(output_dir=autobuild / "FEAT-X")
    section = gen._render_qav_shadow(_make_feature_result())

    assert "## QAV Shadow" in section
    assert "2 judged" in section
    assert "1 disagreements" in section
    assert "TASK-B turn 1" in section
    assert "1 absent" in section


def test_review_summary_section_omitted_when_no_receipts(tmp_path):
    """No shadow receipts ⇒ section omitted, so the summary stays byte-identical."""
    from guardkit.orchestrator.review_summary import ReviewSummaryGenerator

    autobuild = tmp_path / ".guardkit" / "autobuild"
    gen = ReviewSummaryGenerator(output_dir=autobuild / "FEAT-X")
    result = _make_feature_result()

    assert gen._render_qav_shadow(result) == ""

    out = gen.generate(result)
    assert out.success
    assert "## QAV Shadow" not in out.output_path.read_text(encoding="utf-8")


# ===========================================================================
# Checkpoint pattern: a shadow receipt is archived on rollback (design item 3).
# ===========================================================================


def test_checkpoint_archives_qav_shadow_receipt(tmp_path):
    from guardkit.orchestrator.worktree_checkpoints import WorktreeCheckpointManager

    worktree = tmp_path / "worktree"
    autobuild = worktree / ".guardkit" / "autobuild" / "TASK-RB"
    autobuild.mkdir(parents=True)
    # Seed per-turn shadow receipts at turns 1 and 2.
    for turn in (1, 2):
        (autobuild / f"qav_shadow_turn_{turn}.json").write_text(
            json.dumps({"turn": turn, "status": "ok"}), encoding="utf-8"
        )

    manager = WorktreeCheckpointManager(
        worktree_path=worktree, task_id="TASK-RB", git_executor=MagicMock()
    )
    # Archive everything strictly after target turn 1 — the turn-2 shadow receipt.
    archived = manager._archive_post_target_audit_files(1)

    assert archived >= 1
    archive_root = autobuild / "_rollback_archive"
    snapshots = [d for d in archive_root.iterdir() if d.is_dir()]
    assert len(snapshots) == 1
    assert (snapshots[0] / "qav_shadow_turn_2.json").is_file()
    # The target turn's receipt is preserved in place (not archived).
    assert not (snapshots[0] / "qav_shadow_turn_1.json").exists()


def test_qav_shadow_audit_pattern_registered():
    from guardkit.orchestrator.worktree_checkpoints import WorktreeCheckpointManager

    kinds = {k for k, _ in WorktreeCheckpointManager._AUDIT_FILE_PATTERNS}
    assert "qav_shadow" in kinds


def test_agree_is_none_when_coach_errored(tmp_path):
    """An errored coach made no substantive call: shadow reject vs coach 'error'
    must record agree=None, never True (burn-in tally purity — B1 coach advisory)."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())
    seat = _seat(_assistant("reject", findings=[{"class": "DC-03", "locus": "x"}]))

    outcome = q.run_qav_shadow(
        repo, "TASK-1", 1, "error",
        seat_call=seat, running_probe=_free_probe(),
        now=lambda: "2026-07-25T00:00:00Z",
    )

    assert outcome.status == "ok"
    assert outcome.verdict == "reject"
    assert outcome.agree is None
    record = _read_receipt(repo, "TASK-1", 1)
    assert record["agree"] is None
    assert record["shadow"]["verdict"] == "reject"


def test_default_probe_url_derivation_handles_both_endpoint_shapes():
    """The /running URL must derive from scheme+host for BOTH endpoint conventions —
    the base-/v1 shape and the full completions URL (the B3 live-smoke catch)."""
    import guardkit.qa.qav_shadow as qs
    seen = []
    real_urlopen = qs.urllib.request.urlopen

    class _Resp:
        def read(self):
            return b'{"running": []}'
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=None):
        seen.append(url)
        return _Resp()

    qs.urllib.request.urlopen = fake_urlopen
    try:
        for ep in ("http://localhost:9000/v1",
                   "http://localhost:9000/v1/chat/completions",
                   "http://localhost:9000/v1/"):
            probe = qs._default_running_probe(ep)
            assert probe() == []
    finally:
        qs.urllib.request.urlopen = real_urlopen
    assert seen == ["http://localhost:9000/running"] * 3


def test_default_seat_base_url_accepts_both_endpoint_shapes(monkeypatch):
    """The SDK base_url must normalize from either the base-/v1 shape or the full
    completions URL (the B3 live-smoke 404 catch)."""
    import guardkit.qa.qav_shadow as qs
    captured = {}

    class _FakeClient:
        def __init__(self, base_url=None, api_key=None, timeout=None, max_retries=None):
            captured.setdefault("bases", []).append(base_url)
            captured.setdefault("retries", []).append(max_retries)
            raise RuntimeError("stop before any network")

    import sys, types
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    for ep in ("http://h:9000/v1", "http://h:9000/v1/chat/completions",
               "http://h:9000/v1/chat/completions/"):
        call = qs._default_seat_call(ep)
        try:
            call("s", "u", "m", 1.0)
        except RuntimeError:
            pass
    assert captured["bases"] == ["http://h:9000/v1"] * 3


# ===========================================================================
# 2026-08-26 repair — the silent-queue defect (four builds, zero visible rows).
#
# The shadow ran and wrote rows the whole time, but the queue path resolved
# against the NESTED autobuild worktree (`.guardkit/worktrees/FEAT-*` —
# gitignored, transient, unread), so four builds' rows died with their
# worktrees. These tests pin the two guarantees of the fix:
#   * the queue row lands at the MAIN checkout root — per-build-unambiguous;
#   * with the lane ON, every coach verdict leaves exactly one queue row,
#     even when the shadow call fails or the shadow itself crashes.
# ===========================================================================


def _make_linked_worktree(tmp_path):
    """A build checkout with a nested linked worktree, laid out exactly as
    autobuild lays them out on disk (no git binary needed — just the shape:
    the nested worktree's ``.git`` is a FILE pointing into the build
    checkout's ``.git/worktrees/<name>``)."""
    build_root = tmp_path / "build-checkout"
    (build_root / ".git" / "worktrees" / "FEAT-X").mkdir(parents=True)
    nested = build_root / ".guardkit" / "worktrees" / "FEAT-X"
    nested.mkdir(parents=True)
    (nested / ".git").write_text(
        f"gitdir: {build_root / '.git' / 'worktrees' / 'FEAT-X'}\n",
        encoding="utf-8",
    )
    return build_root, nested


def test_failed_seat_call_still_writes_absent_queue_row(tmp_path):
    """A failed shadow call must still leave exactly one queue row (absent)."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())

    def _die(*_a, **_k):
        raise TimeoutError("seat unreachable from the build context")

    outcome = q.run_qav_shadow(
        repo, "TASK-1", 1, "approve", seat_call=_die, running_probe=_free_probe(),
    )

    assert outcome.status == "absent"
    rows = _read_queue(repo)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "absent"
    assert row["absent_reason"] == "timeout"
    assert row["task_id"] == "TASK-1"
    assert row["turn"] == 1
    assert row["coach_decision"] == "approve"


def test_internal_crash_still_writes_absent_queue_row(tmp_path, monkeypatch, caplog):
    """A crash INSIDE the shadow (swallowed by the never-raise guard) must no
    longer mean a missing row: exactly one absent(internal_error) row lands."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())

    def _boom(*_a, **_k):
        raise RuntimeError("eligibility check exploded")

    monkeypatch.setattr(q, "_probe_eligibility", _boom)

    with caplog.at_level(logging.WARNING, logger="guardkit.qa.qav_shadow"):
        outcome = q.run_qav_shadow(
            repo, "TASK-1", 1, "approve",
            seat_call=_raise_if_called, running_probe=_free_probe(),
        )

    assert outcome.error == "guard:RuntimeError"  # still never raises
    rows = _read_queue(repo)
    assert len(rows) == 1
    assert rows[0]["status"] == "absent"
    assert rows[0]["absent_reason"] == "internal_error"
    # The per-turn receipt landed beside the coach verdict too.
    record = _read_receipt(repo, "TASK-1", 1)
    assert record["absent_reason"] == "internal_error"


def test_thread_crash_still_writes_absent_queue_row(tmp_path):
    """A runner that explodes inside the fire-and-forget thread still leaves
    its one absent row."""
    repo = tmp_path
    _write_config(repo, enabled=True)

    def _boom(*_a, **_k):
        raise RuntimeError("run exploded")

    thread = q.schedule_qav_shadow(
        repo, task_id="TASK-1", turn=1, coach_decision="approve", runner=_boom
    )
    assert thread is not None
    thread.join(timeout=5)

    rows = _read_queue(repo)
    assert len(rows) == 1
    assert rows[0]["status"] == "absent"
    assert rows[0]["absent_reason"] == "internal_error"


def test_queue_row_lands_in_build_checkout_not_nested_worktree(tmp_path):
    """PINS THE WRITE PATH: run in a nested autobuild worktree, the queue row
    lands at ``<build checkout>/.guardkit/qav-shadow/queue.jsonl`` — the one
    per-build location the receipts harvest reads — while the per-turn receipt
    stays beside the coach verdict inside the worktree."""
    build_root, nested = _make_linked_worktree(tmp_path)
    _write_config(nested, enabled=True)
    _write_bundle(nested, "TASK-1", 1, _sample_bundle())

    outcome = q.run_qav_shadow(
        nested, "TASK-1", 1, "approve",
        seat_call=_seat(_assistant("approve")), running_probe=_free_probe(),
    )

    assert outcome.status == "ok"
    # The queue is at the OUTER build checkout, not inside the worktree.
    assert not (nested / q.QAV_SHADOW_QUEUE).exists()
    rows = _read_queue(build_root)
    assert len(rows) == 1
    assert rows[0]["agree"] is True
    assert rows[0]["task_id"] == "TASK-1"
    # The per-turn receipt still lands beside the coach verdict.
    record = _read_receipt(nested, "TASK-1", 1)
    assert record["status"] == "ok"


def test_queue_path_pinned_for_plain_checkout(tmp_path):
    """PINS THE WRITE PATH for a normal checkout (``.git`` is a directory):
    the queue stays at ``<repo>/.guardkit/qav-shadow/queue.jsonl``."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-1", 1, _sample_bundle())

    q.run_qav_shadow(
        repo, "TASK-1", 1, "approve",
        seat_call=_seat(_assistant("approve")), running_probe=_free_probe(),
    )

    assert q._main_checkout_root(repo) == repo
    assert (repo / ".guardkit" / "qav-shadow" / "queue.jsonl").is_file()
    assert len(_read_queue(repo)) == 1


def test_queue_falls_back_to_worktree_when_build_root_unwritable(tmp_path, caplog):
    """When the build-checkout sink cannot be written, the row falls back to
    the worktree queue rather than being dropped."""
    build_root, nested = _make_linked_worktree(tmp_path)
    _write_config(nested, enabled=True)
    _write_bundle(nested, "TASK-1", 1, _sample_bundle())
    # Block the build-root sink: a FILE where the qav-shadow dir must be.
    (build_root / ".guardkit" / "qav-shadow").write_text("i am a file", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="guardkit.qa.qav_shadow"):
        outcome = q.run_qav_shadow(
            nested, "TASK-1", 1, "approve",
            seat_call=_seat(_assistant("approve")), running_probe=_free_probe(),
        )

    assert outcome.status == "ok"
    rows = _read_queue(nested)  # the fallback location: the worktree itself
    assert len(rows) == 1
    assert any("row appended to" in r.message for r in caplog.records)


def test_main_checkout_root_falls_back_on_odd_layouts(tmp_path):
    """Any surprise in the layout resolves to the worktree itself — never a raise."""
    # No .git at all.
    assert q._main_checkout_root(tmp_path) == tmp_path
    # A .git file with no gitdir line.
    odd = tmp_path / "odd"
    odd.mkdir()
    (odd / ".git").write_text("not a pointer", encoding="utf-8")
    assert q._main_checkout_root(odd) == odd
    # A gitdir that does not match the worktree layout.
    odd2 = tmp_path / "odd2"
    odd2.mkdir()
    (odd2 / ".git").write_text("gitdir: /nonexistent/somewhere/else\n", encoding="utf-8")
    assert q._main_checkout_root(odd2) == odd2


# ===========================================================================
# 2026-09-05 repair — the verdict-or-error contract.
#
# On the shared vLLM adapter host the seat answers in prose, so the JSON-only
# reader recorded verdict null on 46 of 53 turns and nothing said why. These
# tests pin the four parts of the fix: a text read beside the JSON read, an
# unreadable answer NAMED (never a silent null), the separated thinking kept
# out of the verdict, and the SDK's own retries turned off so the configured
# timeout means what it says.
# ===========================================================================


@pytest.mark.parametrize(
    "text,verdict,findings_count,method",
    [
        # (a) the trained shape: a balanced JSON object.
        ('{"verdict": "reject", "findings": [{"class": "DC-05", "locus": "wiring"}]}',
         "reject", 1, "json"),
        # (b) the shape the vLLM host actually emits.
        ("**Verdict: reject**\n\n**Findings:**\n- a\n- b", "reject", 2, "text"),
        ("Verdict: approve", "approve", 0, "text"),
        ("approve", "approve", 0, "text"),
        ("**reject**", "reject", 0, "text"),
        # (c) nothing readable.
        ("I weighed the evidence and could not settle it.", None, 0, "none"),
    ],
)
def test_extract_verdict_reads_json_then_text_then_gives_up(
    text, verdict, findings_count, method
):
    """The reading order: balanced JSON, then a labelled/bare-word text read."""
    extraction = q.extract_verdict(text)
    assert extraction.verdict == verdict
    assert len(extraction.findings) == findings_count
    assert extraction.method == method


def test_extract_verdict_keeps_the_findings_sentences():
    """A prose finding carries no class and no locus — the sentence is kept."""
    extraction = q.extract_verdict("Verdict: reject\nFindings:\n1. the guard has no producer\n2) the seam is mocked")
    assert [f["text"] for f in extraction.findings] == [
        "the guard has no producer",
        "the seam is mocked",
    ]
    assert all(f["class"] == "" and f["locus"] == "" for f in extraction.findings)


def test_prose_answer_now_records_a_verdict(tmp_path):
    """The live failure: a prose answer used to record verdict null. It reads now."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-PROSE", 1, _sample_bundle())

    outcome = q.run_qav_shadow(
        repo, "TASK-PROSE", 1, "approve",
        seat_call=_seat("**Verdict: reject**\n\n**Findings:**\n- no wired producer"),
        running_probe=_free_probe(),
    )

    assert outcome.status == "ok"
    assert outcome.verdict == "reject"
    assert outcome.agree is False
    record = _read_receipt(repo, "TASK-PROSE", 1)
    assert record["shadow"]["extraction_method"] == "text"
    assert record["shadow"]["absent_reason"] is None
    assert record["shadow"]["findings"] == [
        {"class": "", "locus": "", "text": "no wired producer"}
    ]
    # the raw bytes are kept whenever the trained JSON shape did not carry it
    assert record["shadow"]["raw"].startswith("**Verdict: reject**")


@pytest.mark.parametrize("field_name", ["reasoning_content", "reasoning"])
def test_reasoning_is_captured_under_either_field_name(tmp_path, field_name):
    """llama.cpp says reasoning_content, vLLM says reasoning — both are read,
    truncated, and NEVER used as the verdict."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-R", 1, _sample_bundle())

    class _Msg:
        content = "Verdict: approve"

    setattr(_Msg, field_name, "the seat's long think, which says reject a lot")

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_Choice()]
        usage = None

    class _FakeClient:
        def __init__(self, **kwargs):
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            return _Resp()

    import sys
    import types

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    sys.modules["openai"] = fake_openai
    try:
        outcome = q.run_qav_shadow(
            repo, "TASK-R", 1, "approve",
            seat_call=None, running_probe=_free_probe(),
        )
    finally:
        del sys.modules["openai"]

    assert outcome.verdict == "approve"  # the thinking never becomes the verdict
    record = _read_receipt(repo, "TASK-R", 1)
    assert record["shadow"]["reasoning"] == "the seat's long think, which says reject a lot"


def test_reasoning_is_truncated(tmp_path):
    """A very long think is capped at MAX_REASONING_CHARS in the receipt."""
    repo = tmp_path
    _write_config(repo, enabled=True)
    _write_bundle(repo, "TASK-RT", 1, _sample_bundle())

    def _call(system, user, model, timeout_s):
        return q.SeatResult(text=_assistant("approve"), reasoning="x" * 5000)

    q.run_qav_shadow(
        repo, "TASK-RT", 1, "approve", seat_call=_call, running_probe=_free_probe()
    )
    record = _read_receipt(repo, "TASK-RT", 1)
    assert len(record["shadow"]["reasoning"]) == q.MAX_REASONING_CHARS == 2000


def test_client_is_built_with_no_sdk_retries(monkeypatch):
    """max_retries=0 — the SDK's default 2 retries tripled the 60 s timeout."""
    import sys
    import types

    captured = {}

    class _FakeClient:
        def __init__(self, base_url=None, api_key=None, timeout=None, max_retries=None):
            captured["max_retries"] = max_retries
            captured["timeout"] = timeout
            raise RuntimeError("stop before any network")

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    call = q._default_seat_call("http://h:9000/v1")
    with pytest.raises(RuntimeError):
        call("s", "u", "m", 60.0)

    assert captured["max_retries"] == 0
    assert captured["timeout"] == 60.0
