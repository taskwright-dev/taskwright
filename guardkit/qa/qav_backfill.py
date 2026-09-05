"""Recover verdicts from QAV shadow receipts the JSON-only reader could not read.

Between 2026-09-03 and 2026-09-05 the shadow seat ran on the shared vLLM adapter
host and answered in prose ("**Verdict: reject**", or the bare word "approve").
The reader of the day looked only for a balanced JSON object, so it wrote
``shadow.verdict: null`` and kept the prose in ``shadow.raw``. The verdicts are
therefore still on disk — this tool reads them back out.

It is **dry-run by default**: it prints one line per receipt it could recover
and a total, and writes nothing. ``--write`` updates each receipt in place,
adding the recovered ``verdict`` and ``findings``, ``extraction_method:
"text-backfill"`` and ``verdict_backfilled_at``, and preserving ``raw`` and
every other field. Running it with ``--write`` against the live receipts root is
an attended act — nothing here ever runs itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from guardkit.qa.qav_shadow import extract_verdict

__all__ = [
    "RECEIPT_GLOB",
    "BACKFILL_METHOD",
    "Recovery",
    "BackfillReport",
    "find_recoverable",
    "run_backfill",
]

#: The receipt filename shape the shadow writes (searched recursively).
RECEIPT_GLOB = "qav_shadow_turn_*.json"

#: What a backfilled receipt records as its ``extraction_method`` — distinct
#: from the live reader's "text" so a re-read verdict is always identifiable.
BACKFILL_METHOD = "text-backfill"


@dataclass(frozen=True)
class Recovery:
    """One receipt whose verdict can be read back out of its raw bytes."""

    path: Path
    verdict: str
    findings: List[Dict[str, str]] = field(default_factory=list)

    def line(self) -> str:
        """The one line the dry run prints for this receipt."""
        return f"{self.path}  {self.verdict}  {len(self.findings)} findings"


@dataclass(frozen=True)
class BackfillReport:
    """What a run found (and, with ``written``, what it changed)."""

    root: Path
    recoveries: List[Recovery]
    scanned: int
    written: int = 0
    write: bool = False

    def lines(self) -> List[str]:
        """The report as plain lines, one per recovered receipt plus a total."""
        out = [r.line() for r in self.recoveries]
        if self.write:
            out.append(f"{self.written} of {self.scanned} receipts updated")
        else:
            out.append(
                f"{len(self.recoveries)} of {self.scanned} receipts recoverable "
                "(dry run — nothing written)"
            )
        return out


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_record(path: Path) -> Optional[dict]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def find_recoverable(root: Path) -> tuple[List[Recovery], int]:
    """Walk ``root`` for shadow receipts whose verdict can be re-read.

    A receipt qualifies when its ``shadow.verdict`` is null and its
    ``shadow.raw`` holds text the labelled/bare-word read can turn into a
    verdict. Returns ``(recoveries, receipts scanned)``. Never raises — an
    unreadable file is skipped.
    """
    recoveries: List[Recovery] = []
    scanned = 0
    for path in sorted(Path(root).rglob(RECEIPT_GLOB)):
        if not path.is_file():
            continue
        scanned += 1
        record = _read_record(path)
        if record is None:
            continue
        shadow = record.get("shadow")
        if not isinstance(shadow, dict):
            continue
        if shadow.get("verdict") is not None:
            continue
        raw = shadow.get("raw")
        if not isinstance(raw, str) or not raw.strip():
            continue
        extraction = extract_verdict(raw)
        if extraction.verdict is None:
            continue
        recoveries.append(
            Recovery(path=path, verdict=extraction.verdict, findings=extraction.findings)
        )
    return recoveries, scanned


def _apply(recovery: Recovery, now: Callable[[], str]) -> bool:
    """Write one recovered verdict back into its receipt. True when written."""
    record = _read_record(recovery.path)
    if record is None:
        return False
    shadow = record.get("shadow")
    if not isinstance(shadow, dict):
        return False
    shadow["verdict"] = recovery.verdict
    shadow["findings"] = recovery.findings
    shadow["extraction_method"] = BACKFILL_METHOD
    shadow["verdict_backfilled_at"] = now()
    try:
        recovery.path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError:
        return False
    return True


def run_backfill(
    root: Path,
    *,
    write: bool = False,
    now: Callable[[], str] = _utc_now_iso,
) -> BackfillReport:
    """Find (and with ``write=True`` apply) every recoverable verdict under ``root``.

    Dry run is the default and touches nothing on disk. A second ``write`` run
    over the same tree finds nothing left to do — a backfilled receipt has a
    verdict, so it no longer qualifies.
    """
    recoveries, scanned = find_recoverable(root)
    written = 0
    if write:
        for recovery in recoveries:
            if _apply(recovery, now):
                written += 1
    return BackfillReport(
        root=Path(root),
        recoveries=recoveries,
        scanned=scanned,
        written=written,
        write=write,
    )
