"""Orchestration tests. Model calls are injected; no network or API keys."""

import pytest

from litesquad import squad
from litesquad.config import AgentConfig, RunConfig, SquadConfig
from litesquad.llm import LLMError
from litesquad.models import Conversation


def make_config() -> SquadConfig:
    return SquadConfig(
        run=RunConfig(save_transcript=False),
        judge=AgentConfig(model="anthropic/judge-model"),
        critic=AgentConfig(model="openai/critic-model"),
        extractor=AgentConfig(model="openai/extractor-model"),
        clusterer=AgentConfig(model="anthropic/clusterer-model"),
        workers=[AgentConfig(model="anthropic/worker-a"), AgentConfig(model="gemini/worker-b")],
    )


@pytest.fixture
def fake_call():
    """A deterministic caller that records prompts; inject via ``caller=``."""
    calls: list[dict] = []

    def _fake(model, messages, run_cfg, *, role="", structured=False):
        calls.append({"model": model, "role": role, "prompt": messages[-1]["content"]})
        if structured:
            if "extract" in role:
                return '{"units": [{"kind": "claim", "text": "unit from ' + role + '"}]}'
            return '{"clusters": []}'
        return f"{role}::{model} output"

    _fake.calls = calls
    return _fake


def test_deep_flow_sequence(fake_call):
    conv = Conversation()
    turn = squad.run_turn(conv, "Help me think this through", make_config(), caller=fake_call)

    # propose->critique->revise per worker, then extract each, cluster, and the judge writes
    stages = [e.stage for e in turn.events]
    assert stages == [
        "propose", "critique", "revise",
        "propose", "critique", "revise",
        "extract", "extract",
        "cluster",
        "judge",
    ]
    assert turn.final_answer == "judge::anthropic/judge-model output"
    assert conv.turns == [turn]


def test_critique_is_per_worker(fake_call):
    conv = Conversation()
    turn = squad.run_turn(conv, "anything", make_config(), caller=fake_call)

    critique_roles = [e.role for e in turn.events if e.stage == "critique"]
    assert critique_roles == ["critic->worker_1", "critic->worker_2"]
    critique_models = {e.model for e in turn.events if e.stage == "critique"}
    assert critique_models == {"openai/critic-model"}
    revise = [(e.role, e.model) for e in turn.events if e.stage == "revise"]
    assert revise == [("worker_1", "anthropic/worker-a"), ("worker_2", "gemini/worker-b")]


def test_judge_writes_from_the_content_map(fake_call):
    conv = Conversation()
    squad.run_turn(conv, "anything", make_config(), caller=fake_call)

    # the judge sees a de-stylized content map built from the extracted units, not the
    # workers' polished prose -- that is what prevents cloning the most fluent draft
    judge_prompt = next(c["prompt"] for c in fake_call.calls if c["role"] == "judge")
    assert "content map" in judge_prompt.lower()
    assert "unit from extractor->worker_1" in judge_prompt
    assert "unit from extractor->worker_2" in judge_prompt
    # extract and cluster ran on their configured models
    assert any(c["role"].startswith("extractor") and c["model"] == "openai/extractor-model"
               for c in fake_call.calls)
    assert any(c["role"] == "clusterer" and c["model"] == "anthropic/clusterer-model"
               for c in fake_call.calls)


def test_no_frame_workers_get_raw_message(fake_call):
    conv = Conversation()
    squad.run_turn(conv, "the raw thing", make_config(), caller=fake_call)

    # first worker sees the user's message directly, not a PM framing
    worker_prompt = next(c["prompt"] for c in fake_call.calls if c["role"] == "worker_1")
    assert "the raw thing" in worker_prompt
    assert "framing" not in worker_prompt.lower()


def test_quick_is_a_single_judge_call(fake_call):
    conv = Conversation()
    turn = squad.run_quick(conv, "quick question", make_config(), caller=fake_call)

    assert [(e.stage, e.role, e.model) for e in turn.events] == [
        ("reply", "judge", "anthropic/judge-model")
    ]
    assert turn.final_answer == "judge::anthropic/judge-model output"


def test_followup_includes_prior_context(fake_call):
    conv = Conversation()
    config = make_config()
    squad.run_turn(conv, "First task", config, caller=fake_call)
    fake_call.calls.clear()
    squad.run_turn(conv, "Tighten it", config, caller=fake_call)

    worker_prompt = next(c["prompt"] for c in fake_call.calls if c["role"] == "worker_1")
    assert "Earlier in this conversation" in worker_prompt
    assert "First task" in worker_prompt


def test_per_agent_instructions_scoped_to_that_agent():
    captured: list[tuple[str, str]] = []

    def cap(model, messages, run_cfg, *, role="", structured=False):
        captured.append((role, messages[0]["content"]))  # system message
        if structured:
            if "extract" in role:
                return '{"units": [{"kind": "claim", "text": "u"}]}'
            return '{"clusters": []}'
        return f"{role} output"

    config = make_config()
    config.workers[0].instructions = "BE TERSE PLEASE"
    squad.run_turn(Conversation(), "task", config, caller=cap)

    worker1_systems = [sys for r, sys in captured if r == "worker_1"]
    worker2_systems = [sys for r, sys in captured if r == "worker_2"]
    assert worker1_systems and all("BE TERSE PLEASE" in s for s in worker1_systems)
    assert worker2_systems and all("BE TERSE PLEASE" not in s for s in worker2_systems)


def test_failed_stage_aborts_and_is_recorded():
    def _boom(model, messages, run_cfg, *, role="", structured=False):
        if role.startswith("critic"):
            raise LLMError(role, model, "rate limited")
        return f"{role} output"

    conv = Conversation()
    with pytest.raises(LLMError):
        squad.run_turn(conv, "anything", make_config(), caller=_boom)

    turn = conv.turns[0]
    assert [e.stage for e in turn.events] == ["propose", "critique"]
    assert turn.events[-1].error and "rate limited" in turn.events[-1].error
    assert turn.final_answer is None
