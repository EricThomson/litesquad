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
    """Renders each completed stage as a Rich panel and appends events to a JSONL file.

    Worker chains run in parallel, so stages interleave: the spinner shows everyone
    currently in flight, and panels print in completion order (each panel's title carries
    its role and model). Per the Reporter contract, calls arrive serialized, so no lock is
    needed here; printing while the Status runs is safe (rich renders it above the spinner).
    """

    def __init__(self, transcript_path: Path | None) -> None:
        self.transcript_path = transcript_path
        self._status: Status | None = None
        self._in_flight: dict[tuple[str, Stage], str] = {}  # (role, stage) -> spinner text

    def _spinner_text(self) -> str:
        return "  |  ".join(self._in_flight.values())

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
        title = f"{event.role} | {event.model}"
        if event.error:
            console.print(Panel(event.error, title=f"{title} (error)", border_style="red"))
        else:
            border = "green" if event.stage in ("judge", "reply") else "cyan"
            console.print(Panel(Markdown(event.output), title=title, border_style=border))
        if self._status is not None:
            if self._in_flight:
                self._status.update(self._spinner_text())
            else:
                self._status.stop()
                self._status = None

    def close(self) -> None:
        """Stop the spinner. Call in a ``finally``: an aborted turn must not leave a live
        display running (it would hide the terminal cursor and keep spinning)."""
        if self._status is not None:
            self._status.stop()
            self._status = None
        self._in_flight.clear()


def _transcript_path(save: bool) -> Path | None:
    if not save:
        return None
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return paths.transcripts_dir() / f"{stamp}.jsonl"


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
    for model in config.models():
        try:
            reply = call_model(model, messages, run_cfg, role="check").strip()
            console.print(f"[green]✓[/] {model} — {reply[:40]}")
        except LLMError as exc:
            all_ok = False
            console.print(f"[red]✗[/] {model} — {exc}")
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
