"""Orchestration.

Deep turn: each worker responds independently, the critic critiques each response,
the worker revises, and the judge renders the single final answer. Quick turn: the
judge answers directly, no ensemble. Rendering-agnostic: progress is reported
through a :class:`Reporter` so this module never imports ``rich``.
"""

import random
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
                prompt=prompt, task=turn.task, error=str(exc),
            )
        )
        reporter.stage_done(event)
        raise
    event = turn.add(
        TranscriptEvent(
            turn_index=turn.index, stage=stage, role=role, model=model,
            prompt=prompt, task=turn.task, output=output,
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
    """Run one full ensemble turn, appending it to ``conversation``.

    ``caller`` defaults to the real LiteLLM-backed model call; inject a stand-in
    (e.g. ``mock_call_model``) to run offline.
    """
    reporter = reporter or NullReporter()
    run_cfg = config.run
    history = conversation.history_digest()
    turn = conversation.new_turn(task)

    # Each worker responds independently (blind to the others), the critic critiques
    # that one response, and the worker revises against its own critique. The judge
    # then weighs the revised responses and renders the final answer.
    revised: list[str] = []
    for i, worker in enumerate(config.workers):
        role = f"worker_{i + 1}"
        proposal = _run_stage(
            turn, reporter, run_cfg, caller,
            stage="propose", role=role, model=worker.model,
            system=_system(prompts.WORKER_SYSTEM, worker),
            prompt=prompts.propose_prompt(task, history),
        )
        critique = _run_stage(
            turn, reporter, run_cfg, caller,
            stage="critique", role=f"critic->{role}", model=config.critic.model,
            system=_system(prompts.CRITIC_SYSTEM, config.critic),
            prompt=prompts.critique_prompt(task, proposal),
        )
        revision = _run_stage(
            turn, reporter, run_cfg, caller,
            stage="revise", role=role, model=worker.model,
            system=_system(prompts.WORKER_SYSTEM, worker),
            prompt=prompts.revise_prompt(task, proposal, critique),
        )
        revised.append(revision)

    # Shuffle the responses before the judge sees them (unless disabled) so no worker
    # is permanently "Response 1": LLM judges have a primacy bias toward whatever comes
    # first. The judge is already blind to which model wrote which response, so this
    # just removes the position advantage. (Provenance stays recoverable by content,
    # since each worker's revised text is also recorded verbatim in its own stage.)
    ordered = revised[:]
    if run_cfg.shuffle:
        random.shuffle(ordered)
    _run_stage(
        turn, reporter, run_cfg, caller,
        stage="synthesize", role="judge", model=config.judge.model,
        system=_system(prompts.JUDGE_SYSTEM, config.judge),
        prompt=prompts.synthesize_prompt(task, ordered, history),
    )

    return turn


def run_quick(
    conversation: Conversation,
    task: str,
    config: SquadConfig,
    reporter: Reporter | None = None,
    *,
    caller: Caller = call_model,
) -> Turn:
    """Run one quick turn: the judge model answers directly, no ensemble."""
    reporter = reporter or NullReporter()
    history = conversation.history_digest()
    turn = conversation.new_turn(task)

    _run_stage(
        turn, reporter, config.run, caller,
        stage="reply", role="judge", model=config.judge.model,
        system=_system(prompts.QUICK_SYSTEM, config.judge),
        prompt=prompts.quick_prompt(task, history),
    )

    return turn
