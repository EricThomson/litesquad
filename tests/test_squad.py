"""Orchestration tests. Model calls are injected; no network or API keys."""

import pytest

from litesquad import squad
from litesquad.config import AgentConfig, RunConfig, SquadConfig
from litesquad.llm import LLMError
from litesquad.models import Conversation


def make_config(include_revision: bool = False) -> SquadConfig:
    return SquadConfig(
        run=RunConfig(include_revision=include_revision, save_transcript=False),
        pm=AgentConfig(model="anthropic/pm-model"),
        critic=AgentConfig(model="openai/critic-model"),
        workers=[AgentConfig(model="anthropic/worker-a"), AgentConfig(model="gemini/worker-b")],
    )


@pytest.fixture
def fake_call():
    """A deterministic caller that records prompts; inject via ``caller=``."""
    calls: list[dict] = []

    def _fake(model, messages, run_cfg, *, role=""):
        calls.append({"model": model, "role": role, "prompt": messages[-1]["content"]})
        return f"{role}::{model} output"

    _fake.calls = calls
    return _fake


def test_basic_flow_sequence(fake_call):
    conv = Conversation()
    turn = squad.run_turn(conv, "Plan my 12-week year", make_config(), caller=fake_call)

    stages = [e.stage for e in turn.events]
    assert stages == ["frame", "propose", "propose", "critique", "synthesize"]
    assert turn.final_answer == "pm::anthropic/pm-model output"
    assert conv.turns == [turn]


def test_revision_adds_two_stages(fake_call):
    conv = Conversation()
    turn = squad.run_turn(
        conv, "Plan my 12-week year", make_config(include_revision=True), caller=fake_call
    )

    stages = [e.stage for e in turn.events]
    assert stages == ["frame", "propose", "propose", "critique", "revise", "revise", "synthesize"]
    # worker roles are stable between propose and revise
    revise_roles = [e.role for e in turn.events if e.stage == "revise"]
    assert revise_roles == ["worker_1", "worker_2"]


def test_followup_includes_prior_context(fake_call):
    conv = Conversation()
    config = make_config()
    squad.run_turn(conv, "First task", config, caller=fake_call)
    fake_call.calls.clear()
    squad.run_turn(conv, "Tighten weeks 1-4", config, caller=fake_call)

    frame_prompt = next(c["prompt"] for c in fake_call.calls if c["role"] == "pm")
    assert "Prior context" in frame_prompt
    assert "First task" in frame_prompt


def test_failed_stage_aborts_and_is_recorded():
    def _boom(model, messages, run_cfg, *, role=""):
        if role == "critic":
            raise LLMError(role, model, "rate limited")
        return f"{role} output"

    conv = Conversation()
    with pytest.raises(LLMError):
        squad.run_turn(conv, "Plan", make_config(), caller=_boom)

    # the failed turn is still recorded, with the error event and no final answer
    turn = conv.turns[0]
    assert turn.events[-1].stage == "critique"
    assert turn.events[-1].error and "rate limited" in turn.events[-1].error
    assert turn.final_answer is None
