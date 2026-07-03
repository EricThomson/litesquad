"""TurnRunner tests: background thread, JSONL event stream, state snapshots.

No Dash needed here (runner.py never imports it) and no network: model calls
are injected, same pattern as test_squad.
"""

import threading

import pytest

from litesquad.config import AgentConfig, RunConfig, SquadConfig
from litesquad.llm import LLMError
from litesquad.web.runner import TurnRunner, expected_stage_count


def make_config() -> SquadConfig:
    return SquadConfig(
        run=RunConfig(save_transcript=False),
        judge=AgentConfig(model="anthropic/judge-model"),
        critic=AgentConfig(model="openai/critic-model"),
        extractor=AgentConfig(model="openai/extractor-model"),
        clusterer=AgentConfig(model="anthropic/clusterer-model"),
        workers=[AgentConfig(model="anthropic/worker-a"), AgentConfig(model="gemini/worker-b")],
    )


def fake_call(model, messages, run_cfg, *, role="", structured=False):
    if structured:
        if "extract" in role:
            return '{"units": [{"kind": "claim", "text": "unit from ' + role + '"}]}'
        return '{"clusters": []}'
    return f"{role}::{model} output"


def test_expected_stage_count():
    # two workers: 3 chain stages each, 2 extracts, cluster, judge
    assert expected_stage_count(make_config(), quick=False) == 10
    assert expected_stage_count(make_config(), quick=True) == 1


def test_turn_runs_in_background_and_streams_jsonl(tmp_path):
    runner = TurnRunner(fake_call, transcript_path=tmp_path / "t.jsonl")
    runner.start("Help me plan", make_config())
    runner.wait(timeout=10)

    state = runner.snapshot()
    assert not state.running
    assert state.error is None
    assert state.stages_done == 10

    events = runner.events()
    assert [e.stage for e in events] == [
        "propose", "critique", "revise",
        "propose", "critique", "revise",
        "extract", "extract",
        "cluster",
        "judge",
    ]
    assert events[-1].output == "judge::anthropic/judge-model output"


def test_followup_turn_appends_to_same_stream(tmp_path):
    runner = TurnRunner(fake_call, transcript_path=tmp_path / "t.jsonl")
    config = make_config()
    runner.start("First task", config)
    runner.wait(timeout=10)
    runner.start("Tighten it", config, quick=True)
    runner.wait(timeout=10)

    events = runner.events()
    assert {e.turn_index for e in events} == {0, 1}
    assert events[-1].stage == "reply"
    assert len(runner.conversation.turns) == 2


def test_only_one_turn_at_a_time(tmp_path):
    release = threading.Event()

    def slow_call(model, messages, run_cfg, *, role="", structured=False):
        release.wait(timeout=5)
        return fake_call(model, messages, run_cfg, role=role, structured=structured)

    runner = TurnRunner(slow_call, transcript_path=tmp_path / "t.jsonl")
    runner.start("task", make_config())
    with pytest.raises(RuntimeError, match="already running"):
        runner.start("another", make_config())
    release.set()
    runner.wait(timeout=10)
    assert not runner.snapshot().running


def test_aborted_turn_records_error_and_runner_recovers(tmp_path):
    def boom(model, messages, run_cfg, *, role="", structured=False):
        if role.startswith("critic"):
            raise LLMError(role, model, "rate limited")
        return fake_call(model, messages, run_cfg, role=role, structured=structured)

    runner = TurnRunner(boom, transcript_path=tmp_path / "t.jsonl")
    config = make_config()
    runner.start("anything", config)
    runner.wait(timeout=10)

    state = runner.snapshot()
    assert not state.running
    assert state.error and "rate limited" in state.error
    events = runner.events()
    assert events[-1].stage == "critique" and events[-1].error

    # the thread died cleanly, so a follow-up run still works
    runner._caller = fake_call
    runner.start("try again", config)
    runner.wait(timeout=10)
    assert runner.snapshot().error is None
    assert runner.events()[-1].stage == "judge"
