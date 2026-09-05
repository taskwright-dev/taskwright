"""The QAV verdict backfill: reading verdicts back out of receipts that recorded none.

While the shadow seat answered in prose (2026-09-03 → 09-05) the JSON-only
reader wrote ``shadow.verdict: null`` and kept the prose in ``shadow.raw``. This
covers the recovery tool:

* **dry run is the default** — it names what it could recover and writes nothing
  (proven byte-for-byte and by mtime);
* **--write** updates exactly the recoverable receipts, preserves ``raw`` and
  every other field, and is idempotent on a second run.

Zero live receipts: every test builds its own fixture tree under ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from guardkit.cli.qa import qa
from guardkit.qa import qav_backfill as b


def _receipt(
    root: Path,
    task_id: str,
    turn: int,
    *,
    verdict=None,
    raw=None,
    method="none",
) -> Path:
    """One receipt on disk in the shape the shadow writes."""
    d = root / task_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"qav_shadow_turn_{turn}.json"
    record = {
        "task_id": task_id,
        "turn": turn,
        "ts": "2026-09-04T10:00:00Z",
        "coach_decision": "approve",
        "status": "ok",
        "absent_reason": None,
        "agree": None,
        "shadow": {
            "verdict": verdict,
            "findings": [],
            "json_extracted": verdict is not None,
            "extraction_method": method,
            "absent_reason": None if verdict is not None else "unparseable",
            "reasoning": None,
            "raw": raw,
        },
        "provenance": {"model": "qav-shadow", "endpoint": "http://h:9000/v1"},
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture_tree(tmp_path: Path) -> dict:
    """Six receipts: four recoverable, one already parsed, one with empty raw."""
    root = tmp_path / "receipts"
    return {
        "root": root,
        "bold": _receipt(root, "TASK-A", 1, raw="**Verdict: reject**\n\n**Findings:**\n- a\n- b"),
        "labelled": _receipt(root, "TASK-B", 1, raw="Verdict: approve"),
        "bare": _receipt(root, "TASK-C", 1, raw="approve"),
        "bold_bare": _receipt(root, "TASK-D", 1, raw="**reject**"),
        "parsed": _receipt(root, "TASK-E", 1, verdict="approve", method="json"),
        "empty": _receipt(root, "TASK-F", 1, raw=""),
    }


def _snapshot(root: Path) -> dict:
    return {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("qav_shadow_turn_*.json"))
    }


def test_dry_run_names_the_four_and_writes_nothing(tmp_path):
    tree = _fixture_tree(tmp_path)
    before = _snapshot(tree["root"])

    report = b.run_backfill(tree["root"])

    assert report.scanned == 6
    assert len(report.recoveries) == 4
    assert [r.verdict for r in report.recoveries] == ["reject", "approve", "approve", "reject"]
    assert [len(r.findings) for r in report.recoveries] == [2, 0, 0, 0]
    lines = report.lines()
    assert len(lines) == 5  # four receipts + the total
    assert "4 of 6 receipts recoverable" in lines[-1]
    assert "nothing written" in lines[-1]
    assert _snapshot(tree["root"]) == before  # bytes AND mtimes untouched


def test_write_updates_exactly_the_four_and_preserves_everything_else(tmp_path):
    tree = _fixture_tree(tmp_path)
    untouched_before = {
        k: tree[k].read_bytes() for k in ("parsed", "empty")
    }

    report = b.run_backfill(tree["root"], write=True, now=lambda: "2026-09-05T09:00:00Z")

    assert report.written == 4
    assert "4 of 6 receipts updated" in report.lines()[-1]

    bold = json.loads(tree["bold"].read_text(encoding="utf-8"))
    shadow = bold["shadow"]
    assert shadow["verdict"] == "reject"
    assert [f["text"] for f in shadow["findings"]] == ["a", "b"]
    assert shadow["extraction_method"] == "text-backfill"
    assert shadow["verdict_backfilled_at"] == "2026-09-05T09:00:00Z"
    # raw and every other field survive untouched
    assert shadow["raw"] == "**Verdict: reject**\n\n**Findings:**\n- a\n- b"
    assert shadow["reasoning"] is None
    assert shadow["json_extracted"] is False
    assert bold["task_id"] == "TASK-A"
    assert bold["turn"] == 1
    assert bold["ts"] == "2026-09-04T10:00:00Z"
    assert bold["coach_decision"] == "approve"
    assert bold["status"] == "ok"
    assert bold["provenance"] == {"model": "qav-shadow", "endpoint": "http://h:9000/v1"}

    assert json.loads(tree["bare"].read_text(encoding="utf-8"))["shadow"]["verdict"] == "approve"
    assert json.loads(tree["bold_bare"].read_text(encoding="utf-8"))["shadow"]["verdict"] == "reject"

    # the already-parsed receipt and the empty-raw one are byte-identical
    for k, before in untouched_before.items():
        assert tree[k].read_bytes() == before


def test_write_is_idempotent(tmp_path):
    tree = _fixture_tree(tmp_path)
    b.run_backfill(tree["root"], write=True, now=lambda: "2026-09-05T09:00:00Z")
    after_first = _snapshot(tree["root"])

    second = b.run_backfill(tree["root"], write=True, now=lambda: "2026-09-05T23:00:00Z")

    assert second.recoveries == []
    assert second.written == 0
    assert _snapshot(tree["root"]) == after_first  # nothing rewritten, no new timestamp


def test_unreadable_and_odd_files_are_skipped(tmp_path):
    """A corrupt receipt, a non-dict record and a missing shadow block are skipped."""
    root = tmp_path / "receipts"
    (root / "TASK-X").mkdir(parents=True)
    (root / "TASK-X" / "qav_shadow_turn_1.json").write_text("{not json", encoding="utf-8")
    (root / "TASK-X" / "qav_shadow_turn_2.json").write_text("[]", encoding="utf-8")
    (root / "TASK-X" / "qav_shadow_turn_3.json").write_text('{"status": "ok"}', encoding="utf-8")

    report = b.run_backfill(root)

    assert report.scanned == 3
    assert report.recoveries == []


def test_cli_dry_run_prints_the_lines_and_writes_nothing(tmp_path):
    """The command is registered under ``guardkit qa`` and defaults to dry run."""
    tree = _fixture_tree(tmp_path)
    before = _snapshot(tree["root"])

    result = CliRunner().invoke(qa, ["backfill-verdicts", str(tree["root"])])

    assert result.exit_code == 0, result.output
    assert "4 of 6 receipts recoverable" in result.output
    assert "reject" in result.output and "approve" in result.output
    assert _snapshot(tree["root"]) == before
