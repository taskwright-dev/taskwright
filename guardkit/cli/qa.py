"""``guardkit qa`` — QA format validation + the live-gate runner (WS2 B1 + B3).

v1 surface (scope-design §3 CLI table — ``walk`` is B5):

    guardkit qa validate <kind> <path>     # exit 0 valid, 1 invalid (loud)
    guardkit qa schema <kind> [--out F]    # JSON-Schema export
    guardkit qa kinds                      # list known kinds
    guardkit qa live-gate --feature <id> --target <env> [--gates ..] [--campaign]
    guardkit qa mutate --task <id> ...            # ST-05 mutation stage (B6)
    guardkit qa probe-boundaries --seam <id> ...  # ST-06 boundary probes (B6)
    guardkit qa review [range selectors]          # R-b advisory code review (S5)

``validate``/``schema``/``kinds`` are on-demand format tools (no enforcement —
that is B2). ``live-gate`` runs the repo's registered F4 gates and emits the
results envelope on stdout for the forge adapter (scope-design §3). ``mutate``
and ``probe-boundaries`` are the B6 deeper stages — standalone (the Coach does
not consume them in v1), and **advisory by default**: findings file as
task-shaped records and do NOT block (exit 0). ``--strict`` opts a run into a
non-zero exit when findings exist (the gate-promotion path; see WS2 §B6 STATUS
for the gate-vs-advisory verdict).

``review`` (R-b, options paper ``factory-code-quality-seat-options-2026-07.md``
stage S-5) is the **on-demand advisory code-review entry**: the coordinator runs
it over any range — the R-b inspector reads a git diff on a local seat and emits
an F14 review-findings record with reasons. It is **advisory-only and flag-gated
default-OFF** (``qa.review_seat`` / ``GUARDKIT_QA_REVIEW_SEAT``): when the flag is
OFF the command is a provable no-op (exit 0, no git, no seat); when ON it emits
and exits 0 — it NEVER blocks. Promotion to a blocking gate is the S-4
calibration bar (Rich's numbers), not this command. The same advisory review is
wired as a step in the ``autobuild complete --verify`` gate flow (behind the
same flag), where it attaches its F14 record as a flow artifact and never
changes the completion's exit code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from guardkit.qa.formats import (
    FORMAT_KINDS,
    KIND_ALIASES,
    MARKDOWN_KINDS,
    MarkdownFormat,
    QAFormatError,
    export_json_schema,
    resolve_kind,
    validate_instance,
)

console = Console()

#: Exit codes for ``live-gate`` — distinct non-1 codes for the attribution
#: verdicts so the forge adapter can tell a feature failure (1) from an
#: instrument/environment fault (3/4), which per DF-017 never count against the
#: feature. (The authoritative verdict is the envelope's ``verdict`` field on
#: stdout; the exit code is a convenience mirror.)
_VERDICT_EXIT_CODES = {
    "pass": 0,
    "fail": 1,
    "instrument_fail": 3,
    "environment_fail": 4,
}


@click.group()
def qa() -> None:
    """QA verification formats (tier-1 F1–F5, tier-2/3 F6–F15 + deploy-profile)."""


@qa.command()
@click.argument("kind")
@click.argument("path", type=click.Path(path_type=Path))
def validate(kind: str, path: Path) -> None:
    """Validate a QA format instance file against its schema.

    KIND is any canonical kind (pass-bar, known-failures, leak-sweep,
    gate-registry, results-envelope, evidence-index, seam-manifest,
    deploy-record, disposition-record, attempts-ledger, live-matrix,
    deploy-profile, runbook, discovery-gates, kickoff-prompt, review-findings,
    walk-checkpoints) or an f1..f15 alias. ``runbook`` (F11) is validated as a
    markdown-convention document; all others as YAML/JSON.
    """
    try:
        instance = validate_instance(kind, path)
    except QAFormatError as exc:
        console.print("[bold red]✗ VALIDATION FAILED[/bold red]")
        # Print the full error verbatim — loud, field-level, never summarized.
        console.print(str(exc), highlight=False)
        sys.exit(1)
    console.print(
        f"[bold green]✓ VALID[/bold green] {path} "
        f"({instance.FORMAT_KIND} v{instance.format_version})"
    )


@qa.command()
@click.argument("kind")
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the JSON-Schema to a file instead of stdout.",
)
def schema(kind: str, out_path: Path | None) -> None:
    """Export the JSON-Schema for a QA format kind."""
    try:
        model = resolve_kind(kind)
    except QAFormatError as exc:
        console.print(f"[bold red]✗[/bold red] {exc}", highlight=False)
        sys.exit(1)
    # F11 (runbook) is a markdown-convention format — it has no JSON-Schema;
    # print its human-readable convention description instead.
    if isinstance(model, type) and issubclass(model, MarkdownFormat):
        text = model.describe_schema()
    else:
        text = export_json_schema(model)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        console.print(f"[green]Wrote[/green] {out_path}")
    else:
        click.echo(text)


@qa.command(name="live-gate")
@click.option("--feature", "feature_id", required=True, help="Feature id under test.")
@click.option("--target", "target_env", required=True, help="Target environment id.")
@click.option(
    "--gates",
    "gates",
    default=None,
    help="Comma-separated subset of registered gate ids (default: all).",
)
@click.option(
    "--campaign",
    is_flag=True,
    default=False,
    help="Campaign mode (attempts ledger is B4; accepted here, single run in v1).",
)
@click.option(
    "--repo",
    "repo_root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("."),
    help="Target repo root (default: cwd). Reads qa/gates/registry.yaml under it.",
)
def live_gate(
    feature_id: str,
    target_env: str,
    gates: str | None,
    campaign: bool,
    repo_root: Path,
) -> None:
    """Run the repo's registered F4 gates and emit the results envelope.

    Deterministic runner (DF-015 clause 1). Prints the F4 results envelope as
    JSON on stdout (the forge adapter parses its ``verdict``); exit code mirrors
    the verdict (0 pass, 1 fail, 3 instrument_fail, 4 environment_fail).

    ``--campaign`` records the run as attempt 1 of an F9 attempts ledger
    (``qa/attempts-<feature>.yaml``) and stamps the envelope's
    ``attempts_ledger_ref``. v1 has no live multi-attempt driver, so a single
    unattended run must be green or a pre-flight short-circuit — a run with reds
    that no arbiter has binned is honestly reported UNCLOSED (exit 2, DF-017
    §2.1), never a silent green.
    """
    # Imported lazily so `guardkit qa validate` has no orchestrator import cost.
    from guardkit.orchestrator.live_gate import (
        LiveGateError,
        LiveGateRunner,
        UndispositionedRedError,
    )

    requested = [g.strip() for g in gates.split(",") if g.strip()] if gates else None
    runner = LiveGateRunner(repo_root)
    try:
        envelope = runner.run(
            feature_id,
            target_env,
            requested_gate_ids=requested,
            campaign=campaign,
        )
        if campaign:
            envelope = _record_single_run_campaign(envelope, repo_root)
    except UndispositionedRedError as exc:
        # The run has reds no arbiter binned — UNCLOSED (never a silent green).
        console.print("[bold red]✗ live-gate run is UNCLOSED[/bold red]", highlight=False)
        console.print(str(exc), highlight=False)
        sys.exit(2)
    except (QAFormatError, LiveGateError) as exc:
        # A missing/invalid registry or an unknown gate id is a loud config
        # error — never a silent green.
        console.print("[bold red]✗ live-gate could not run[/bold red]", highlight=False)
        console.print(str(exc), highlight=False)
        sys.exit(2)

    # The envelope on stdout is the contract for the forge adapter.
    click.echo(json.dumps(envelope.model_dump(mode="json"), indent=2))
    sys.exit(_VERDICT_EXIT_CODES.get(envelope.verdict, 1))


@qa.command(name="normalize-stamps")
@click.option("--feature", "feature_id", required=True, help="Feature id (FEAT-X) to stamp.")
@click.option(
    "--repo",
    "repo_root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("."),
    help="Target repo root (default: cwd). Reads .guardkit/features/<id>.yaml under it.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Classify and print the result; write nothing.",
)
@click.option(
    "--ignore-existing",
    is_flag=True,
    default=False,
    help="Classify EVERY title as if unstamped (dry-run only; the reproduction proof).",
)
@click.option(
    "--http-surface/--no-http-surface",
    "http_surface",
    default=None,
    help=(
        "Override the repo HTTP-surface detection that gates R9 (hurl). Without "
        "it the surface is STRUCTURAL: a hurl gate in qa/gates/registry.yaml, "
        "`surface: http` in .guardkit/config.yaml, or an exact web-framework "
        "dependency in pyproject/package.json — never free text."
    ),
)
def normalize_stamps(
    feature_id: str,
    repo_root: Path,
    dry_run: bool,
    ignore_existing: bool,
    http_surface: bool | None,
) -> None:
    """THE STAMP NORMALIZER — mint ``verifier:`` stamps by rule (R1–R10) and WRITE them.

    Rules doc: ai-transition/docs/routing-law-stamp-normalizer-rules-2026-08-15.md.
    Rich's conditions: it WRITES the feature YAML's ``scenarios:`` map (only titles
    lacking a stamp — never overwrites). Titles no rule can decide are handed to a
    model (ruled 2026-08-31; only those titles, answer checked against the closed
    list, listed under `model_stamped`); anything still undecided REFUSES LOUD by
    name and nothing is stamped for it. Prints the result as JSON on stdout (forge's hook
    parses it); exit 0 = all decided (stamped/nothing to do), 3 = PARTIAL (decided stamps written, `refused` names the rest), 2 = cannot run.
    Already-stamped titles the rules would home DIFFERENTLY are listed under
    `disagreements` (advisory: echoed on stderr, never overwritten, exit unchanged).
    """
    from guardkit.orchestrator.stamp_normalizer import (
        StampNormalizerError,
        StampNormalizerRefusal,
        normalize_feature,
    )

    features_dir = repo_root / ".guardkit" / "features"
    yaml_path = features_dir / f"{feature_id}.yaml"
    if not yaml_path.exists():
        alt = features_dir / f"{feature_id}.yml"
        if alt.exists():
            yaml_path = alt
    if not yaml_path.exists():
        payload = {
            "feature_id": feature_id,
            "error": f"feature file not found: {yaml_path}",
            "refused": [],
        }
        click.echo(json.dumps(payload, indent=2))
        console.print(f"[bold red]✗ normalize-stamps: {payload['error']}[/bold red]", highlight=False)
        sys.exit(2)

    try:
        result = normalize_feature(
            yaml_path,
            None,
            repo_root,
            dry_run=dry_run,
            ignore_existing=ignore_existing,
            repo_has_http_surface=http_surface,
        )
    except StampNormalizerRefusal as exc:
        payload = {
            "feature_id": exc.feature_id,
            "error": str(exc),
            "refused": list(exc.refused),
            "written": False,
        }
        click.echo(json.dumps(payload, indent=2))
        console.print("[bold red]✗ normalize-stamps REFUSED (undecidable titles)[/bold red]", highlight=False)
        console.print(str(exc), highlight=False)
        sys.exit(2)
    except StampNormalizerError as exc:
        payload = {"feature_id": feature_id, "error": str(exc), "refused": [], "written": False}
        click.echo(json.dumps(payload, indent=2))
        console.print("[bold red]✗ normalize-stamps could not run[/bold red]", highlight=False)
        console.print(str(exc), highlight=False)
        sys.exit(2)

    if result.model_stamped:
        # THE MODEL FALLBACK (RULED 2026-08-31): the titles no rule could
        # decide, and what the model decided for each. Named in the JSON
        # (`model_stamped`) AND echoed on stderr ahead of the JSON, so nobody
        # mistakes a model-decided stamp for a rule-decided one.
        err_console = Console(stderr=True)
        err_console.print(
            f"[bold yellow]! normalize-stamps: the model decided "
            f"{len(result.model_stamped)} scenario(s) no rule could decide:[/bold yellow]",
            highlight=False,
        )
        for title in result.model_stamped:
            err_console.print(
                f"  - {title} -> {result.stamped.get(title, '?')}",
                highlight=False,
                soft_wrap=True,
            )
    if result.operator_stamped:
        # L3: an operator stamp is never silent — it is attended human work
        # handed to Rich. Named in the JSON (`operator_stamped`) AND in the
        # human echo (stderr, ahead of the JSON, so the hook's stdout parse
        # stays clean). Each title says whether a rule or the model decided it.
        err_console = Console(stderr=True)
        err_console.print(
            f"[bold yellow]! normalize-stamps minted `operator` (attended human work) "
            f"for {len(result.operator_stamped)} scenario(s):[/bold yellow]",
            highlight=False,
        )
        for title in result.operator_stamped:
            how = "decided by the model" if title in result.model_stamped else "rule R10"
            err_console.print(f"  - {title} ({how})", highlight=False, soft_wrap=True)
    if result.disagreements:
        # (2) RULED 2026-08-18: ADVISORY disagreements — already-stamped titles
        # the rules would home differently. Named in the JSON (`disagreements`)
        # AND echoed on stderr ahead of the JSON. NEVER overwritten; the exit
        # code is UNCHANGED (0/3/2 — a disagreement is advisory).
        err_console = Console(stderr=True)
        err_console.print(
            f"[bold yellow]! normalize-stamps: {len(result.disagreements)} stamp "
            f"DISAGREEMENT(s) (advisory, not overwritten):[/bold yellow]",
            highlight=False,
        )
        for d in result.disagreements:
            err_console.print(
                f"  - '{d['title']}' is stamped {d['stamped']} but the rules say "
                f"{d['rule_home']} ({d['rule']}: {d['evidence']})",
                highlight=False,
                soft_wrap=True,  # one line per disagreement, never re-flowed
            )
    if result.refused:
        # PARTIAL (coordinator review 2026-08-16): every DECIDED stamp was
        # written; the undecidable titles are named in the JSON `refused` list
        # and echoed on stderr. Exit 3 is DISTINCT from 2 (cannot run) so the
        # caller decides stop-vs-proceed — the normalizer never decides that.
        err_console = Console(stderr=True)
        err_console.print(
            f"[bold red]! normalize-stamps PARTIAL: {len(result.refused)} scenario(s) "
            f"undecidable by rule (no home invented; decided stamps written):[/bold red]",
            highlight=False,
        )
        for title in result.refused:
            err_console.print(f"  - {title}", highlight=False)
        click.echo(json.dumps(result.to_dict(), indent=2))
        sys.exit(3)
    click.echo(json.dumps(result.to_dict(), indent=2))
    sys.exit(0)


def _record_single_run_campaign(envelope, repo_root: Path):
    """Wrap a single B3 run as attempt 1 of an F9 ledger and stamp the envelope.

    Returns the finalized envelope (with ``attempts_ledger_ref`` /
    ``dispositions_ref`` set). Raises ``UndispositionedRedError`` if the run has
    unbinned reds — the CLI has no arbiter to bin them unattended.
    """
    from guardkit.orchestrator.live_gate import (
        finalize_envelope,
        single_run_campaign,
        write_campaign,
    )

    # The started timestamp's date part (YYYY-MM-DD) anchors the ledger entry.
    run_date = envelope.started[:10]
    result = single_run_campaign(envelope, date=run_date)
    refs = write_campaign(result, repo_root, run_id=envelope.run_id)
    return finalize_envelope(envelope, result, refs)


#: Advisory-by-default: findings do NOT block in v1. ``--strict`` maps a run
#: that produced findings to this non-zero code (the gate-promotion path).
_FINDINGS_STRICT_EXIT = 3


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


@qa.command()
@click.option("--task", "task_id", required=True, help="Task id under mutation (finding subject).")
@click.option(
    "--repo",
    "repo_root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("."),
    help="Repo root the throwaway sandbox is copied from (default: cwd).",
)
@click.option(
    "--files",
    "files",
    default=None,
    help="Comma-separated deliverable source files to mutate (default: derive from --base diff).",
)
@click.option(
    "--test-command",
    "test_command",
    required=True,
    help="Shell test command that must go RED when the behaviour breaks "
    "(e.g. 'python -m pytest -q tests/unit').",
)
@click.option(
    "--operator",
    "operators",
    multiple=True,
    type=click.Choice(["strip-auth-header", "revert-hunk"]),
    default=("strip-auth-header",),
    help="Mutation operator(s). revert-hunk requires --base.",
)
@click.option("--base", default=None, help="Git base ref for revert-hunk / file derivation.")
@click.option("--timeout", default=600, show_default=True, help="Per-mutant test timeout (s).")
@click.option("--no-file", "no_file", is_flag=True, default=False, help="Do not write finding files.")
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit non-zero (3) if any mutant survived (gate mode; default is advisory exit 0).",
)
def mutate(
    task_id: str,
    repo_root: Path,
    files: str | None,
    test_command: str,
    operators: tuple[str, ...],
    base: str | None,
    timeout: int,
    no_file: bool,
    strict: bool,
) -> None:
    """ST-05 mutation stage — break the key behaviour, require the tests to go red.

    Each mutation runs in a THROWAWAY sandbox (never the task branch). A mutant
    that survives its own test suite is a proven coverage hole, filed as a
    non-blocking finding. Advisory by default (exit 0); ``--strict`` exits 3 on
    any survivor.
    """
    import shlex

    from guardkit.orchestrator.qa_stages import write_findings
    from guardkit.orchestrator.qa_stages.assembly import assemble_mutation_campaign
    from guardkit.orchestrator.qa_stages.errors import MutationError

    source_files = [f.strip() for f in files.split(",") if f.strip()] if files else None
    try:
        assembly = assemble_mutation_campaign(
            Path(repo_root),
            task_id,
            source_files=source_files,
            test_command=shlex.split(test_command),
            operators=list(operators),
            base=base,
            timeout=timeout,
        )
    except MutationError as exc:
        console.print("[bold red]✗ mutation stage could not run[/bold red]", highlight=False)
        console.print(str(exc), highlight=False)
        sys.exit(2)

    res = assembly.result
    console.print(
        f"[bold]qa mutate[/bold] {task_id}: {assembly.mutant_count} mutant(s) — "
        f"[green]{len(res.killed)} killed[/green], "
        f"[yellow]{len(assembly.findings)} survived[/yellow], "
        f"{len(res.errored)} errored"
    )
    for finding in assembly.findings:
        console.print(f"  [yellow]⚠ SURVIVOR[/yellow] {finding.site} — {finding.summary}")
    if assembly.findings and not no_file:
        paths = write_findings(assembly.findings, Path(repo_root), date=_today())
        for path in paths:
            console.print(f"  [dim]filed[/dim] {path}")
    if strict and assembly.findings:
        sys.exit(_FINDINGS_STRICT_EXIT)
    sys.exit(0)


@qa.command(name="probe-boundaries")
@click.option("--seam", "seam_id", required=True, help="Seam id from the F6 manifest to probe.")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    default=None,
    help="F6 seam-manifest file (validates --seam is a declared seam id).",
)
@click.option(
    "--target",
    "target_spec",
    default=None,
    help="ProbeTarget as 'module.path:attr' (instance or zero-arg factory). "
    "Omit to run the loud unconfigured target (honest 'not wired').",
)
@click.option(
    "--repo",
    "repo_root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("."),
    help="Repo root findings are written under (default: cwd).",
)
@click.option("--no-file", "no_file", is_flag=True, default=False, help="Do not write finding files.")
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit non-zero (3) if any raw-error leak / junk-accept was found "
    "(gate mode; default is advisory exit 0).",
)
def probe_boundaries(
    seam_id: str,
    manifest_path: Path | None,
    target_spec: str | None,
    repo_root: Path,
    no_file: bool,
    strict: bool,
) -> None:
    """ST-06 boundary probes — feed non-conforming inputs at an F6 seam.

    A raw error leaking past the seam's sealed error set (or a garbage input
    silently accepted) is a non-blocking finding. Advisory by default (exit 0);
    ``--strict`` exits 3 on any finding. An unconfigured target raises loudly
    (honest "not wired") rather than reporting a clean posture.
    """
    from guardkit.orchestrator.qa_stages import (
        Finding,
        run_boundary_probes,
        write_findings,
    )
    from guardkit.orchestrator.qa_stages.assembly import resolve_probe_target
    from guardkit.orchestrator.qa_stages.boundary import load_seam_ids
    from guardkit.orchestrator.qa_stages.errors import BoundaryProbeError, QAStageStubError

    if manifest_path is not None:
        try:
            declared = load_seam_ids(manifest_path)
        except QAFormatError as exc:
            console.print("[bold red]✗ invalid seam manifest[/bold red]", highlight=False)
            console.print(str(exc), highlight=False)
            sys.exit(2)
        if seam_id not in declared:
            console.print(
                f"[bold red]✗[/bold red] seam {seam_id!r} not in manifest "
                f"(declared: {', '.join(declared) or '<none>'})",
                highlight=False,
            )
            sys.exit(2)

    try:
        target = resolve_probe_target(target_spec)
        result = run_boundary_probes(seam_id, target)
    except QAStageStubError as exc:
        console.print(
            "[bold yellow]⚠ boundary probe target not wired[/bold yellow]", highlight=False
        )
        console.print(str(exc), highlight=False)
        # Honest "not configured" — never a silent green. Non-blocking (exit 0).
        sys.exit(0)
    except BoundaryProbeError as exc:
        console.print("[bold red]✗ boundary probe could not run[/bold red]", highlight=False)
        console.print(str(exc), highlight=False)
        sys.exit(2)

    findings = [
        Finding(
            kind="boundary-leak" if o.classification == "leak" else "boundary-accept",
            subject=seam_id,
            site=o.input_label,
            summary=o.detail,
            evidence=f"input={o.input_label} classification={o.classification} "
            f"exc={o.exception_type}",
            suggested_pin=(
                "Fold this input class into the seam's sealed error set (reject "
                "with the seam's own error type), or reject garbage before decode."
            ),
        )
        for o in result.findings
    ]
    console.print(
        f"[bold]qa probe-boundaries[/bold] {seam_id}: {len(result.outcomes)} probe(s) — "
        f"[yellow]{len(result.leaks)} raw-error leak(s)[/yellow], "
        f"{len(result.findings)} total finding(s)"
    )
    for o in result.findings:
        console.print(f"  [yellow]⚠ {o.classification.upper()}[/yellow] {o.input_label} — {o.detail}")
    if findings and not no_file:
        paths = write_findings(findings, Path(repo_root), date=_today())
        for path in paths:
            console.print(f"  [dim]filed[/dim] {path}")
    if strict and findings:
        sys.exit(_FINDINGS_STRICT_EXIT)
    sys.exit(0)


@qa.command(name="review")
@click.option(
    "--repo",
    "repo_root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("."),
    help="Repo root to review (default: cwd).",
)
@click.option(
    "--base",
    default=None,
    help="Review a range: 'git diff <base> [<head>]' (any range, e.g. main, "
    "or a 'main..feature' expression).",
)
@click.option(
    "--head",
    default=None,
    help="Range head, paired with --base. Omit for a single-ended range/expr.",
)
@click.option("--commit", "commit", default=None, help="Review one commit's diff (git show).")
@click.option("--merge", "merge", default=None, help="Review a merge commit vs its first parent.")
@click.option(
    "--staged",
    is_flag=True,
    default=False,
    help="Review the staged working tree only (git diff --cached).",
)
@click.option(
    "--unstaged",
    is_flag=True,
    default=False,
    help="Review unstaged working-tree changes only (git diff).",
)
@click.option(
    "--seat",
    "model",
    default=None,
    help="Local reviewer seat (qwen36-workhorse | gemma4-coach; default: workhorse).",
)
@click.option(
    "--write/--no-write",
    "write",
    default=True,
    help="Write the F14 record to qa/review-<id>.yaml (default: write).",
)
@click.option(
    "--advisory/--blocking",
    "advisory",
    default=True,
    help="Advisory (default; exit 0 always). Blocking mode is gated on the S-4 "
    "calibration bar and not yet available.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Also emit the F14 record as JSON on stdout.",
)
def review(
    repo_root: Path,
    base: str | None,
    head: str | None,
    commit: str | None,
    merge: str | None,
    staged: bool,
    unstaged: bool,
    model: str | None,
    write: bool,
    advisory: bool,
    as_json: bool,
) -> None:
    """R-b advisory code review over any range — emit F14 findings, never block.

    The inspector: reads a git diff on a local seat and writes up findings with
    reasons (F14 review-findings). Advisory-only and flag-gated default-OFF
    (``qa.review_seat`` / ``GUARDKIT_QA_REVIEW_SEAT``) — when OFF it is a provable
    no-op; when ON it emits and exits 0. Promotion to a blocking gate is the S-4
    calibration bar, not this command.

    \b
    Subject (choose one; default = the whole working tree):
        --base main [--head feat]   an arbitrary range
        --commit <sha>              one commit
        --merge <sha>               a merge vs its first parent
        --staged / --unstaged       the working tree, narrowed

    \b
    Exit codes:
        0  advisory outcome (no-op / clean / findings / a named seat error)
        2  the requested range could not be read, or bad options
    """
    from guardkit.qa.diff_ingest import (
        DiffIngestError,
        ingest_commit,
        ingest_merge,
        ingest_range,
        ingest_working_tree,
    )
    from guardkit.qa.review_seat import (
        DEFAULT_SEAT,
        is_review_seat_enabled,
        run_advisory_review,
    )

    if not advisory:
        console.print(
            "[bold red]✗[/bold red] --blocking is not available: a blocking "
            "review gate is gated on the S-4 calibration bar (Rich's catch-rate "
            "floor + over-flag ceiling). This lane ships advisory-only.",
            highlight=False,
        )
        sys.exit(2)

    # Exactly one subject selector (working-tree scope flags are one group).
    selectors = [
        ("--base", base is not None),
        ("--commit", commit is not None),
        ("--merge", merge is not None),
        ("--staged/--unstaged", staged or unstaged),
    ]
    chosen = [name for name, on in selectors if on]
    if len(chosen) > 1:
        console.print(
            f"[bold red]✗[/bold red] choose one review subject, got: "
            f"{', '.join(chosen)}",
            highlight=False,
        )
        sys.exit(2)
    if staged and unstaged:
        console.print(
            "[bold red]✗[/bold red] --staged and --unstaged are mutually exclusive.",
            highlight=False,
        )
        sys.exit(2)
    if head is not None and base is None:
        console.print(
            "[bold red]✗[/bold red] --head requires --base.", highlight=False
        )
        sys.exit(2)

    seat = model or DEFAULT_SEAT

    # Flag-gate FIRST: default-OFF ⇒ a provable no-op (no git, no seat). The
    # coordinator opts in per run with GUARDKIT_QA_REVIEW_SEAT=1 (or the repo's
    # qa.review_seat), mirroring qa.enforce_tier1.
    if not is_review_seat_enabled(Path(repo_root)):
        console.print(
            "[yellow]○ review seat OFF[/yellow] (qa.review_seat / "
            "GUARDKIT_QA_REVIEW_SEAT) — no-op. Set the flag to run the advisory "
            "review. This never blocks; it is advisory until the S-4 bar."
        )
        sys.exit(0)

    # Build the review subject (the one genuinely-new S-1 piece). A range that
    # cannot be read is a loud config error (exit 2) — never a faked empty review.
    try:
        if base is not None:
            payload = ingest_range(Path(repo_root), base, head)
        elif commit is not None:
            payload = ingest_commit(Path(repo_root), commit)
        elif merge is not None:
            payload = ingest_merge(Path(repo_root), merge)
        elif staged:
            payload = ingest_working_tree(Path(repo_root), scope="staged")
        elif unstaged:
            payload = ingest_working_tree(Path(repo_root), scope="unstaged")
        else:
            payload = ingest_working_tree(Path(repo_root), scope="all")
    except DiffIngestError as exc:
        console.print("[bold red]✗ could not read the review subject[/bold red]", highlight=False)
        console.print(str(exc), highlight=False)
        sys.exit(2)

    outcome = run_advisory_review(
        Path(repo_root), payload, model=seat, write=write
    )
    _display_review_outcome(outcome, as_json=as_json)
    # Advisory: ALWAYS exit 0. A seat outage / parse failure is a named result,
    # not a failure of this command.
    sys.exit(0)


def _display_review_outcome(outcome, as_json: bool = False) -> None:
    """Render an advisory :class:`ReviewOutcome` — used by the CLI and the
    autobuild gate-flow step. Advisory throughout: an error is a NAMED line,
    never a non-zero verdict."""
    import json as _json

    from rich.markup import escape

    console.print()
    if not outcome.enabled:
        note = outcome.notes[0] if outcome.notes else "review seat OFF — no-op"
        console.print(f"[yellow]○ {escape(note)}[/yellow]")
        return
    if outcome.error is not None:
        # ON but the seat could not be reached / its output could not be parsed.
        console.print(
            Panel(
                f"[yellow]○ advisory review did not emit[/yellow]\n\n"
                f"{escape(outcome.error)}\n\n"
                "Advisory: this never blocks. The seat outage/parse failure is "
                "recorded as-is (honesty-to-state), not a faked clean review.",
                title="Review Advisory (no record)",
                border_style="yellow",
            )
        )
        return

    record = outcome.record
    total = record.stats.findings_total
    header = (
        f"[green]✓ clean[/green]" if total == 0 else f"[yellow]⚠ {total} finding(s)[/yellow]"
    )
    lines = [
        f"{header}  ·  subject: {escape(record.subject.kind)} "
        f"{escape(record.subject.ref)}",
        f"confirmed {record.stats.confirmed} · refuted {record.stats.refuted} "
        f"· refutations attempted {record.stats.refutations_attempted}",
    ]
    if outcome.emitted_path:
        lines.append(f"[dim]F14 record →[/dim] {escape(outcome.emitted_path)}")
    for note in outcome.notes:
        lines.append(f"[dim]note:[/dim] {escape(note)}")
    console.print(
        Panel(
            "\n".join(lines),
            title="Advisory Code Review (R-b) — advisory, never blocks",
            border_style="green" if total == 0 else "yellow",
        )
    )
    for f in record.findings:
        console.print(
            f"  [yellow]•[/yellow] [{escape(f.severity)}/{escape(f.dimension)}/"
            f"{escape(f.status)}] {escape(f.summary)}"
        )
    if as_json:
        click.echo(_json.dumps(record.model_dump(mode="json"), indent=2))


@qa.command()
def kinds() -> None:
    """List the known QA format kinds and their aliases."""
    alias_by_kind = {v: k for k, v in KIND_ALIASES.items()}
    all_kinds = {**FORMAT_KINDS, **MARKDOWN_KINDS}
    for name, model in all_kinds.items():
        alias = alias_by_kind.get(name, "")
        alias_txt = f"  (alias: {alias})" if alias else ""
        medium = " [markdown]" if name in MARKDOWN_KINDS else ""
        console.print(
            f"  {name:<18} v{model.CURRENT_FORMAT_VERSION}{alias_txt}{medium}"
        )


@qa.command(name="backfill-verdicts")
@click.argument(
    "receipts_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--write",
    is_flag=True,
    default=False,
    help="Update the receipts in place (default: dry run, nothing is written).",
)
def backfill_verdicts(receipts_root: Path, write: bool) -> None:
    """Re-read verdicts out of QAV shadow receipts that recorded none.

    While the shadow seat answered in prose the JSON-only reader recorded
    ``verdict: null`` and kept the prose in ``raw``. This walks RECEIPTS_ROOT
    for ``qav_shadow_turn_*.json`` files in that state and reads the verdict
    back out. It is a dry run by default: one line per recoverable receipt
    (path, verdict, findings count) and a total, nothing written. ``--write``
    updates each file in place, preserving ``raw`` and every other field.
    """
    from guardkit.qa.qav_backfill import run_backfill

    report = run_backfill(Path(receipts_root), write=write)
    for line in report.lines():
        click.echo(line)
