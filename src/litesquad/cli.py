"""Command-line interface: one fixed-flow run, then interactive follow-ups."""

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status

from . import paths
from .config import RunConfig, SquadConfig, ensure_starter, load_config
from .llm import LLMError, MissingKeysError, call_model, load_env, mock_call_model, preflight
from .models import Conversation, Stage, TranscriptEvent
from .squad import run_quick, run_turn

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
            return Panel(event.error, title=f"{title} (error)", border_style="red")
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


def _check_models(config: SquadConfig) -> bool:
    """Ping each distinct configured model with a tiny request. Returns all-ok.

    ``max_tokens`` is generous (not 5) because reasoning models (GPT-5,
    Gemini 2.5 Pro) spend output budget on hidden reasoning before any visible
    text — too small a cap returns empty content. The cap only prevents
    truncation; actual usage stays tiny since the model stops after "ok".
    """
    run_cfg = RunConfig(max_tokens=1024, save_transcript=False)
    messages = [{"role": "user", "content": "Reply with the single word: ok"}]
    all_ok = True
    for model, roles in _model_roles(config):
        try:
            reply = call_model(model, messages, run_cfg, role="check").strip()
            console.print(f"[green]✓[/] {model} [dim]({roles})[/] — {reply[:40]}")
        except LLMError as exc:
            all_ok = False
            console.print(f"[red]✗[/] {model} [dim]({roles})[/] — {exc}")
    return all_ok


@app.command()
def run(
    task: str = typer.Argument(None, help="Your question or task for the ensemble."),
    quick: bool = typer.Option(
        False, "--quick", help="Talk to just the judge (Opus), skipping the ensemble."
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Use canned offline responses; no API keys needed."
    ),
    check: bool = typer.Option(
        False, "--check", help="Ping each configured model with a tiny request and exit."
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
        from .web import serve  # lazy: dash is an optional extra

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
            preflight(config)
        except MissingKeysError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        console.print("Pinging configured models…")
        raise typer.Exit(0 if _check_models(config) else 1)

    if smoke:
        caller = mock_call_model if mock else call_model
        if not mock:
            try:
                preflight(config)
            except MissingKeysError as exc:
                console.print(f"[red]{exc}[/]")
                raise typer.Exit(1) from exc
        console.print(f'Smoke test: one turn on [dim]"{SMOKE_PROMPT}"[/]\n')
        transcript_path = _transcript_path(config.run.save_transcript)
        reporter = ConsoleReporter(transcript_path)
        try:
            run_turn(Conversation(), SMOKE_PROMPT, config, reporter, caller=caller)
        except Exception as exc:  # noqa: BLE001 - smoke surfaces ANY failure (call or save)
            console.print(f"[red]Smoke test FAILED: {exc}[/]")
            raise typer.Exit(1) from exc
        finally:
            reporter.close()
        if transcript_path is not None:
            console.print(f"[dim]Transcript: {transcript_path}[/]")
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
            preflight(config)
        except MissingKeysError as exc:
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
            run_one(conversation, current_task, config, reporter, caller=caller)
        except LLMError as exc:
            console.print(f"[red]Turn aborted: {exc}[/]")
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
