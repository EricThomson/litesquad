"""Run ensemble turns on a background thread for the web UI.

No Dash imports here: this module bridges the synchronous core (squad.run_turn /
run_quick) and any polling UI. Completed stages land in the transcript JSONL,
which is the durable event stream the UI reads. The few mutable bits a file
cannot carry -- which stages are in flight (several at once, since worker
chains run in parallel), whether the run is alive, how it ended -- live on the
:class:`TurnRunner` behind a lock.
"""

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .. import paths
from ..config import SquadConfig
from ..llm import call_model
from ..models import Conversation, Stage, TranscriptEvent, load_events_lenient
from ..squad import Caller, run_quick, run_turn


def expected_stage_count(config: SquadConfig, quick: bool) -> int:
    """How many stage events one turn emits if nothing aborts.

    Deep: propose/critique/revise per worker, one extract per worker, then
    cluster and judge. Quick: a single reply.
    """
    if quick:
        return 1
    workers = len(config.workers)
    return workers * 3 + workers + 2


def _fresh_transcript_path(folder: Path) -> Path:
    """A new, unused ``<timestamp>_web.jsonl`` path in ``folder``."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = folder / f"{stamp}_web.jsonl"
    suffix = 2
    while path.exists():  # a new conversation started within the same second
        path = folder / f"{stamp}_web-{suffix}.jsonl"
        suffix += 1
    return path


@dataclass(frozen=True)
class RunState:
    """Point-in-time snapshot of the runner, safe to hand to a UI callback."""

    running: bool
    stages_done: int
    stages_expected: int
    # Everything currently in flight, oldest first: parallel worker chains mean
    # several stages can run at once. Each entry is (stage, role, model).
    in_flight: tuple[tuple[Stage, str, str], ...]
    error: str | None  # set when the last turn aborted
    task: str = ""  # the message the current (or last) turn is answering
    quick: bool = False  # whether that turn bypasses the squad (judge only)


class WebReporter:
    """Reporter for web runs: mirrors stage progress into the runner's shared
    state and appends each completed event to the transcript JSONL."""

    def __init__(self, runner: "TurnRunner") -> None:
        self._runner = runner

    def stage_start(self, stage: Stage, role: str, model: str) -> None:
        self._runner._note_stage_start(stage, role, model)

    def stage_done(self, event: TranscriptEvent) -> None:
        self._runner._note_stage_done(event)


class TurnRunner:
    """Owns the background thread, the conversation, and the shared run state.

    One runner per server process. The conversation persists across turns so
    follow-ups carry context, exactly like the CLI loop, until :meth:`reset` starts
    a new one (in a new transcript file). ``start`` takes the
    config per call: today every call passes the same loaded config, but this
    is the seam that later lets the UI hand in a modified roster per run
    (roadmap shift 2) without touching anything here.

    The transcript JSONL is always written, regardless of run.save_transcript:
    for the web UI it is not a log but the event stream the page renders from.
    """

    def __init__(self, caller: Caller = call_model, transcript_path: Path | None = None) -> None:
        self._caller = caller
        if transcript_path is None:
            transcript_path = _fresh_transcript_path(paths.transcripts_dir())
        self.transcript_path = transcript_path
        self.conversation = Conversation()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._stages_done = 0
        self._stages_expected = 0
        self._in_flight: dict[tuple[str, str], tuple[Stage, str, str]] = {}
        self._error: str | None = None
        self._task = ""
        self._quick = False

    def start(self, task: str, config: SquadConfig, *, quick: bool = False) -> None:
        """Kick off one turn in the background. Raises RuntimeError if one is running."""
        with self._lock:
            if self._running:
                raise RuntimeError("a turn is already running")
            self._running = True
            self._stages_done = 0
            self._stages_expected = expected_stage_count(config, quick)
            self._in_flight.clear()
            self._error = None
            self._task = task
            self._quick = quick
        self._thread = threading.Thread(
            target=self._work, args=(task, config, quick), daemon=True, name="litesquad-turn"
        )
        self._thread.start()

    def _work(self, task: str, config: SquadConfig, quick: bool) -> None:
        run_one = run_quick if quick else run_turn
        try:
            run_one(self.conversation, task, config, WebReporter(self), caller=self._caller)
        except Exception as exc:  # noqa: BLE001 - a thread that dies silently would leave the UI spinning
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                self._running = False
                self._in_flight.clear()

    def reset(self) -> Path:
        """Start a new conversation, in a new transcript file next to the current one.

        Returns the previous transcript path (the old conversation stays saved there).
        Raises RuntimeError if a turn is running.
        """
        with self._lock:
            if self._running:
                raise RuntimeError("a turn is still running; start a new conversation after it finishes")
            previous = self.transcript_path
            self.conversation = Conversation()
            self.transcript_path = _fresh_transcript_path(previous.parent)
            self._stages_done = 0
            self._stages_expected = 0
            self._error = None
            self._task = ""
            self._quick = False
        return previous

    def has_history(self) -> bool:
        """True once a turn has finished with an answer: the next turn is then a follow-up
        (squad.run_turn hands earlier answered turns to the workers and judge)."""
        return any(turn.final_answer for turn in self.conversation.turns)

    def _note_stage_start(self, stage: Stage, role: str, model: str) -> None:
        with self._lock:
            self._in_flight[(role, stage)] = (stage, role, model)

    def _note_stage_done(self, event: TranscriptEvent) -> None:
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with self.transcript_path.open("a", encoding="utf-8") as fh:
            fh.write(event.to_jsonl() + "\n")
        with self._lock:
            self._in_flight.pop((event.role, event.stage), None)
            if not event.error:
                self._stages_done += 1

    def snapshot(self) -> RunState:
        with self._lock:
            return RunState(
                running=self._running,
                stages_done=self._stages_done,
                stages_expected=self._stages_expected,
                in_flight=tuple(self._in_flight.values()),
                error=self._error,
                task=self._task,
                quick=self._quick,
            )

    def events(self) -> list[TranscriptEvent]:
        """Everything completed so far this session, from the JSONL.

        Lenient load: a poll can catch the writer mid-append, so a torn final
        line is skipped this tick and picked up whole on the next one.
        """
        if not self.transcript_path.exists():
            return []
        events, _ = load_events_lenient(self.transcript_path)
        return events

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current turn finishes (used by tests)."""
        if self._thread is not None:
            self._thread.join(timeout)
