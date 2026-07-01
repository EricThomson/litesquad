"""Fixed-flow orchestration: frame -> propose -> critique -> (revise) -> synthesize.

Rendering-agnostic: progress is reported through a :class:`Reporter` so this
module never imports ``rich``.
"""

from typing import Callable, Protocol

from . import prompts
from .config import AgentConfig, RunConfig, SquadConfig
from .llm import call_model
from .models import Conversation, Stage, TranscriptEvent, Turn


def _system(base: str, agent: AgentConfig) -> str:
    """Append an agent's optional per-config instructions to its base prompt."""
    return f"{base} {agent.instructions}" if agent.instructions else base

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
        system=_system(prompts.PM_SYSTEM, config.pm), prompt=prompts.frame_prompt(task, history),
    )

    # Each worker runs an independent propose -> critique -> revise loop: it
    # proposes blind to the others, the critic critiques that one proposal, and
    # the worker revises against its own critique. The PM synthesizes the revised set.
    revised: list[str] = []
    for i, worker in enumerate(config.workers):
        role = f"worker_{i + 1}"
        proposal = _run_stage(
            turn, reporter, run_cfg, caller,
            stage="propose", role=role, model=worker.model,
            system=_system(prompts.WORKER_SYSTEM, worker),
            prompt=prompts.propose_prompt(task, framing),
        )
        critique = _run_stage(
            turn, reporter, run_cfg, caller,
            stage="critique", role=f"critic->{role}", model=config.critic.model,
            system=_system(prompts.CRITIC_SYSTEM, config.critic),
            prompt=prompts.critique_prompt(task, framing, proposal),
        )
        revision = _run_stage(
            turn, reporter, run_cfg, caller,
            stage="revise", role=role, model=worker.model,
            system=_system(prompts.WORKER_SYSTEM, worker),
            prompt=prompts.revise_prompt(task, framing, proposal, critique),
        )
        revised.append(revision)

    _run_stage(
        turn, reporter, run_cfg, caller,
        stage="synthesize", role="pm", model=config.pm.model,
        system=_system(prompts.PM_SYSTEM, config.pm),
        prompt=prompts.synthesize_prompt(task, framing, revised),
    )

    return turn
