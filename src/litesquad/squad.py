"""Fixed-flow orchestration: frame -> propose -> critique -> (revise) -> synthesize.

Rendering-agnostic: progress is reported through a :class:`Reporter` so this
module never imports ``rich``.
"""

from typing import Callable, Protocol

from . import prompts
from .config import RunConfig, SquadConfig
from .llm import call_model
from .models import Conversation, Stage, TranscriptEvent, Turn

# A model caller: same signature as litellm-backed ``call_model``. Injectable so
# the CLI can run a mock (offline) squad without provider credentials.
Caller = Callable[..., str]


class Reporter(Protocol):
    def stage_start(self, stage: Stage, role: str, model: str) -> None: ...
    def stage_done(self, event: TranscriptEvent) -> None: ...


class NullReporter:
    """Reporter that does nothing (default for tests)."""

    def stage_start(self, stage: Stage, role: str, model: str) -> None:
        pass

    def stage_done(self, event: TranscriptEvent) -> None:
        pass


def _run_stage(
    turn: Turn,
    reporter: Reporter,
    run_cfg: RunConfig,
    caller: Caller,
    *,
    stage: Stage,
    role: str,
    model: str,
    system: str,
    prompt: str,
) -> str:
    """Run one model call, record an event, report it. Re-raises on failure."""
    reporter.stage_start(stage, role, model)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        output = caller(model, messages, run_cfg, role=role)
    except Exception as exc:  # noqa: BLE001 - record then re-raise to abort the turn
        event = turn.add(
            TranscriptEvent(
                turn_index=turn.index, stage=stage, role=role, model=model,
                prompt=prompt, error=str(exc),
            )
        )
        reporter.stage_done(event)
        raise
    event = turn.add(
        TranscriptEvent(
            turn_index=turn.index, stage=stage, role=role, model=model,
            prompt=prompt, output=output,
        )
    )
    reporter.stage_done(event)
    return output


def run_turn(
    conversation: Conversation,
    task: str,
    config: SquadConfig,
    reporter: Reporter | None = None,
    *,
    caller: Caller = call_model,
) -> Turn:
    """Run one full squad turn, appending it to ``conversation``.

    ``caller`` defaults to the real LiteLLM-backed model call; inject a stand-in
    (e.g. ``mock_call_model``) to run offline.
    """
    reporter = reporter or NullReporter()
    run_cfg = config.run
    history = conversation.history_digest()
    turn = conversation.new_turn(task)

    framing = _run_stage(
        turn, reporter, run_cfg, caller,
        stage="frame", role="pm", model=config.pm.model,
        system=prompts.PM_SYSTEM, prompt=prompts.frame_prompt(task, history),
    )

    worker_roles = [f"worker_{i + 1}" for i in range(len(config.workers))]
    proposals = [
        _run_stage(
            turn, reporter, run_cfg, caller,
            stage="propose", role=role, model=worker.model,
            system=prompts.WORKER_SYSTEM, prompt=prompts.propose_prompt(task, framing),
        )
        for role, worker in zip(worker_roles, config.workers)
    ]

    critique = _run_stage(
        turn, reporter, run_cfg, caller,
        stage="critique", role="critic", model=config.critic.model,
        system=prompts.CRITIC_SYSTEM,
        prompt=prompts.critique_prompt(task, framing, proposals),
    )

    if run_cfg.include_revision:
        proposals = [
            _run_stage(
                turn, reporter, run_cfg, caller,
                stage="revise", role=role, model=worker.model,
                system=prompts.WORKER_SYSTEM,
                prompt=prompts.revise_prompt(task, framing, proposal, critique),
            )
            for role, worker, proposal in zip(worker_roles, config.workers, proposals)
        ]

    _run_stage(
        turn, reporter, run_cfg, caller,
        stage="synthesize", role="pm", model=config.pm.model,
        system=prompts.PM_SYSTEM,
        prompt=prompts.synthesize_prompt(task, framing, proposals, critique),
    )

    return turn
