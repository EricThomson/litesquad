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
from .config import RunConfig, SquadConfig, ensure_config, load_config
from .llm import LLMError, MissingKeysError, call_model, load_env, mock_call_model, preflight
from .models import Conversation, Stage, TranscriptEvent
from .squad import run_turn

app = typer.Typer(add_completion=False, help="Run a small squad of LLMs on a planning task.")
console = Console()

STAGE_LABEL: dict[Stage, str] = {
    "frame": "framing",
    "propose": "proposing",
    "critique": "critiquing",
    "revise": "revising",
    "synthesize": "synthesizing",
}
QUIT_WORDS = {":quit", ":q", "quit", "exit"}


class ConsoleReporter:
    """Renders each stage as a Rich panel and appends events to a JSONL file."""

    def __init__(self, transcript_path: Path | None) -> None:
        self.transcript_path = transcript_path
        self._status: Status | None = None

    def stage_start(self, stage: Stage, role: str, model: str) -> None:
        self._status = console.status(
            f"[bold]{role}[/] ({model}) — {STAGE_LABEL[stage]}…", spinner="dots"
        )
        self._status.start()

    def stage_done(self, event: TranscriptEvent) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
        if self.transcript_path is not None:
            with self.transcript_path.open("a", encoding="utf-8") as fh:
                fh.write(event.to_jsonl() + "\n")
        title = f"{event.role} · {event.model}"
        if event.error:
            console.print(Panel(event.error, title=f"{title} — error", border_style="red"))
        else:
            border = "green" if event.stage == "synthesize" else "cyan"
            console.print(Panel(Markdown(event.output), title=title, border_style=border))


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
    task: str = typer.Argument(None, help="The task for the squad to work on."),
    mock: bool = typer.Option(
        False, "--mock", help="Use canned offline responses; no API keys needed."
    ),
    check: bool = typer.Option(
        False, "--check", help="Ping each configured model with a tiny request and exit."
    ),
) -> None:
    """Run the squad on TASK, then take interactive follow-ups."""
    load_env()

    cfg_path = paths.config_path()
    first_run = not cfg_path.exists()
    ensure_config(cfg_path)
    if first_run and not mock:
        console.print(
            f"Created a default config at [bold]{cfg_path}[/].\n"
            "Edit your models there and make sure your API keys are set "
            "(environment or a .env file), then re-run."
        )
        raise typer.Exit(0)

    config = load_config(cfg_path)

    if check:
        try:
            preflight(config)
        except MissingKeysError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        console.print("Pinging configured models…")
        raise typer.Exit(0 if _check_models(config) else 1)

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

    current_task = task
    while True:
        try:
            run_turn(conversation, current_task, config, reporter, caller=caller)
        except LLMError as exc:
            console.print(f"[red]Turn aborted: {exc}[/]")

        if transcript_path is not None:
            console.print(f"[dim]Transcript: {transcript_path}[/]")

        try:
            reply = Prompt.ask("\n[bold]Follow-up[/] ([dim]:quit to exit[/])").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not reply or reply.lower() in QUIT_WORDS:
            break
        current_task = reply
