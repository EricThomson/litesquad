"""Command-line interface: one fixed-flow run, then interactive follow-ups."""

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.text import Text

from . import paths
from .config import RunConfig, SquadConfig, ensure_starter, load_config
from .llm import (
    OPENROUTER_KEY_VAR,
    LLMError,
    MissingKeyError,
    call_model,
    catalog_entry,
    key_info,
    load_env,
    mock_call_model,
    model_catalog,
    preflight,
)
from .models import Conversation, Stage, TranscriptEvent, Turn
from .squad import TooFewWorkersError, run_quick, run_turn

app = typer.Typer(add_completion=False, help="Ask a small ensemble of LLMs anything.")
console = Console()

STAGE_LABEL: dict[Stage, str] = {
    "propose": "responding",
    "critique": "critiquing",
    "revise": "revising",
    "extract": "de-stylizing",
    "cluster": "clustering",
    "judge": "judging",
    "reply": "answering",
}
QUIT_WORDS = {":quit", ":q", "quit", "exit"}
SMOKE_PROMPT = "What is 1 + 1? Answer in one short sentence."


class ConsoleReporter:
    """Renders each stage as a Rich panel and appends events to a JSONL file.

    Worker chains run in parallel, so their stages complete interleaved. Printing them
    as they land is unreadable, so chain stages (propose/critique/revise) are buffered
    per worker and printed as one coherent block the moment that worker's chain finishes
    (or dies) -- blocks appear in chain-completion order. Everything else (extract,
    cluster, judge, reply) prints as it completes, and the spinner always shows everyone
    currently in flight. The transcript JSONL is still appended per event in completion
    order: it is the durable stream, and the web UI regroups it by worker anyway.

    Per the Reporter contract, calls arrive serialized, so no lock is needed here;
    printing while the Status runs is safe (rich renders it above the spinner).
    """

    def __init__(self, transcript_path: Path | None) -> None:
        self.transcript_path = transcript_path
        self._status: Status | None = None
        self._in_flight: dict[tuple[str, Stage], str] = {}  # (role, stage) -> spinner text
        self._pending_chains: dict[str, list[Panel]] = {}  # worker key -> panels so far

    @staticmethod
    def _worker_key(role: str) -> str:
        """propose/revise carry ``worker_N``; critique carries ``critic->worker_N``."""
        return role.split("->")[-1]

    @staticmethod
    def _panel(event: TranscriptEvent) -> Panel:
        title = f"{event.role} | {event.model}"
        if event.error:
            # Text, not a str: error messages start with "[role]", which Rich would
            # otherwise read as a markup tag and silently drop.
            return Panel(Text(event.error), title=f"{title} (error)", border_style="red")
        border = "green" if event.stage in ("judge", "reply") else "cyan"
        return Panel(Markdown(event.output), title=title, border_style=border)

    def _spinner_text(self) -> str:
        return "  |  ".join(self._in_flight.values())

    def _flush_chain(self, worker: str) -> None:
        for panel in self._pending_chains.pop(worker, []):
            console.print(panel)

    def stage_start(self, stage: Stage, role: str, model: str) -> None:
        self._in_flight[(role, stage)] = f"[bold]{role}[/] ({model}) - {STAGE_LABEL[stage]}..."
        if self._status is None:
            self._status = console.status(self._spinner_text(), spinner="dots")
            self._status.start()
        else:
            self._status.update(self._spinner_text())

    def stage_done(self, event: TranscriptEvent) -> None:
        self._in_flight.pop((event.role, event.stage), None)
        if self.transcript_path is not None:
            with self.transcript_path.open("a", encoding="utf-8") as fh:
                fh.write(event.to_jsonl() + "\n")
        if event.stage in ("propose", "critique", "revise"):
            worker = self._worker_key(event.role)
            self._pending_chains.setdefault(worker, []).append(self._panel(event))
            if event.stage == "revise" or event.error:  # chain finished, or died here
                self._flush_chain(worker)
        else:
            console.print(self._panel(event))
        if self._status is not None:
            if self._in_flight:
                self._status.update(self._spinner_text())
            else:
                self._status.stop()
                self._status = None

    def close(self) -> None:
        """Flush any unfinished chains and stop the spinner. Call in a ``finally``: an
        aborted turn must still show what its surviving chains produced, and must not
        leave a live display running (hidden cursor, endless spinner)."""
        for worker in list(self._pending_chains):
            self._flush_chain(worker)
        if self._status is not None:
            self._status.stop()
            self._status = None
        self._in_flight.clear()


def _transcript_path(save: bool) -> Path | None:
    if not save:
        return None
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return paths.transcripts_dir() / f"{stamp}.jsonl"


def _model_roles(config: SquadConfig) -> list[tuple[str, str]]:
    """Distinct configured models in pipeline order (workers first, then critic ->
    extractor -> clusterer -> judge), each with the role(s) it holds -- so --check
    doubles as a who-does-what roster card."""
    assignments = [
        *(("worker", worker.model) for worker in config.workers),
        ("critic", config.critic.model),
        ("extractor", config.extractor.model),
        ("clusterer", config.clusterer.model),
        ("judge", config.judge.model),
    ]
    roles_by_model: dict[str, list[str]] = {}
    for role, model in assignments:
        roles = roles_by_model.setdefault(model, [])
        if role not in roles:  # two workers on one model still read as one "worker"
            roles.append(role)
    return [(model, ", ".join(roles)) for model, roles in roles_by_model.items()]


def _failed_stages(turn: Turn) -> list[str]:
    """The stages of a turn that failed but did not abort it (dropped workers), as text."""
    return [f"{event.role} at {event.stage}" for event in turn.events if event.error]


def _check_key() -> bool:
    """Confirm the OpenRouter key is valid. Free: spends no tokens. Returns ok."""
    try:
        info = key_info()
    except LLMError as exc:
        console.print(f"[red]✗[/] {escape(str(exc))}")
        return False
    remaining = info.get("limit_remaining")
    limit_note = "no spending limit" if remaining is None else f"${remaining:.2f} left on its limit"
    console.print(
        f"[green]✓[/] {OPENROUTER_KEY_VAR} - valid, ${info.get('usage', 0):.2f} spent, {limit_note}"
    )
    return True


def _catalog_notes(entry: dict) -> str:
    """Catalog facts worth flagging beside a model: an alias's current target, an expiry date."""
    notes = []
    if entry.get("alias_target"):
        notes.append(f"alias for {entry['alias_target']['slug']}")
    if entry.get("expiration_date"):
        notes.append(f"[yellow]expires {entry['expiration_date']}[/]")
    return f" [dim]|[/] {', '.join(notes)}" if notes else ""


def _check_models(config: SquadConfig) -> bool:
    """Look up each distinct configured model in OpenRouter's catalog, then ping it. Returns all-ok.

    The catalog lookup is free and catches typos and retired models without spending a
    call; it also flags announced expiry dates and shows what a ``~...-latest`` alias
    currently points to. The ping's ``max_tokens`` is generous (not 5) because reasoning
    models spend output budget on hidden reasoning before any visible text, and too small
    a cap returns empty content. Actual usage stays tiny since the model stops after "ok".
    """
    try:
        catalog = model_catalog()
    except LLMError as exc:
        console.print(f"[red]✗[/] {escape(str(exc))}")
        return False
    run_cfg = RunConfig(max_tokens=1024, save_transcript=False)
    messages = [{"role": "user", "content": "Reply with the single word: ok"}]
    all_ok = True
    for model, roles in _model_roles(config):
        label = f"{model} [dim]({roles})[/]"
        entry = catalog_entry(model, catalog)
        if entry is None:
            all_ok = False
            console.print(f"[red]✗[/] {label} - not in the OpenRouter catalog (typo, or retired?)")
            continue
        notes = _catalog_notes(entry)
        try:
            reply = call_model(model, messages, run_cfg, role="check").strip()
        except LLMError as exc:
            all_ok = False
            console.print(f"[red]✗[/] {label} - {escape(str(exc))}{notes}")
            continue
        console.print(f"[green]✓[/] {label} - {escape(reply[:40])}{notes}")
    return all_ok


@app.command()
def run(
    task: str = typer.Argument(None, help="Your question or task for the ensemble."),
    quick: bool = typer.Option(
        False, "--quick", help="Talk to just the judge, skipping the ensemble."
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Use canned offline responses; no API keys needed."
    ),
    check: bool = typer.Option(
        False, "--check",
        help="Verify the OpenRouter key, look up each configured model in the catalog, ping it, and exit.",
    ),
    smoke: bool = typer.Option(
        False, "--smoke", help="Run one real turn on a fixed tiny prompt and exit (cheap end-to-end)."
    ),
    web: bool = typer.Option(
        False, "--web", help="Serve the web UI instead of the terminal (needs the 'web' extra)."
    ),
    port: int = typer.Option(8050, help="Port for the web UI (only used with --web)."),
) -> None:
    """Ask the ensemble (or, with --quick, just the judge), then take follow-ups."""
    if web:
        if task is not None:
            console.print("[red]With --web, submit tasks in the browser, not on the command line.[/]")
            raise typer.Exit(2)
        from .web import serve  # lazy: keeps dash's import cost off plain CLI runs

        serve(port=port, mock=mock)
        return

    load_env()

    cfg_path = paths.config_path()
    if ensure_starter(cfg_path):
        console.print(
            f"[dim]Wrote a starter config (all defaults, commented) you can edit at {cfg_path}[/]"
        )
    config = load_config(cfg_path)

    if check:
        try:
            preflight()
        except MissingKeyError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        if not _check_key():
            raise typer.Exit(1)
        console.print("Checking configured models against the OpenRouter catalog, then pinging each...")
        raise typer.Exit(0 if _check_models(config) else 1)

    if smoke:
        caller = mock_call_model if mock else call_model
        if not mock:
            try:
                preflight()
            except MissingKeyError as exc:
                console.print(f"[red]{exc}[/]")
                raise typer.Exit(1) from exc
        console.print(f'Smoke test: one turn on [dim]"{SMOKE_PROMPT}"[/]\n')
        transcript_path = _transcript_path(config.run.save_transcript)
        reporter = ConsoleReporter(transcript_path)
        try:
            turn = run_turn(Conversation(), SMOKE_PROMPT, config, reporter, caller=caller)
        except Exception as exc:  # noqa: BLE001 - smoke surfaces ANY failure (call or save)
            console.print(f"[red]Smoke test FAILED: {escape(str(exc))}[/]")
            raise typer.Exit(1) from exc
        finally:
            reporter.close()
        if transcript_path is not None:
            console.print(f"[dim]Transcript: {transcript_path}[/]")
        # Stricter than a normal run, which carries on without failed workers: the smoke
        # test exists to prove every stage works, so any dropped worker fails it.
        failed = _failed_stages(turn)
        if failed:
            console.print(f"[red]Smoke test FAILED: workers dropped ({', '.join(failed)})[/]")
            raise typer.Exit(1)
        console.print("[green]Smoke test passed - all stages produced output.[/]")
        raise typer.Exit(0)

    if task is None:
        console.print('[red]Provide a task, e.g. litesquad "Plan my week", or use --check.[/]')
        raise typer.Exit(2)

    caller = call_model
    if mock:
        caller = mock_call_model
        console.print("[yellow]Running in --mock mode: canned responses, no API calls.[/]")
    else:
        try:
            preflight()
        except MissingKeyError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc

    conversation = Conversation()
    transcript_path = _transcript_path(config.run.save_transcript)
    reporter = ConsoleReporter(transcript_path)
    run_one = run_quick if quick else run_turn
    if quick:
        console.print("[dim]Quick mode: just the judge, no ensemble.[/]")

    current_task = task
    while True:
        try:
            turn = run_one(conversation, current_task, config, reporter, caller=caller)
            failed = _failed_stages(turn)
            if failed:
                console.print(f"[yellow]Carried on without failed workers: {', '.join(failed)}[/]")
        except (LLMError, TooFewWorkersError) as exc:
            console.print(f"[red]Turn aborted: {escape(str(exc))}[/]")
        finally:
            reporter.close()

        if transcript_path is not None:
            console.print(f"[dim]Transcript: {transcript_path}[/]")

        try:
            reply = Prompt.ask(
                "\n[bold magenta]Follow-up[/] ([bold yellow]:quit to exit[/])"
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not reply or reply.lower() in QUIT_WORDS:
            break
        current_task = reply
