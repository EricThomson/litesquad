"""ConsoleReporter tests: chain-grouped panel output under interleaved completion."""

import re

from rich.console import Console

from litesquad import cli
from litesquad.config import AgentConfig, SquadConfig
from litesquad.models import TranscriptEvent


def test_model_roles_orders_pipeline_and_merges_shared_models():
    config = SquadConfig(
        judge=AgentConfig(model="prov/big"),
        critic=AgentConfig(model="prov/sharp"),
        extractor=AgentConfig(model="prov/sharp"),
        clusterer=AgentConfig(model="prov/big"),
        workers=[AgentConfig(model="prov/a"), AgentConfig(model="prov/b")],
    )
    assert cli._model_roles(config) == [
        ("prov/a", "worker"),
        ("prov/b", "worker"),
        ("prov/sharp", "critic, extractor"),
        ("prov/big", "clusterer, judge"),
    ]


def _event(stage: str, role: str, **kwargs) -> TranscriptEvent:
    return TranscriptEvent(
        turn_index=0, stage=stage, role=role, model="prov/m", prompt="p", task="t",
        output=kwargs.pop("output", "out"), **kwargs,
    )


def _panel_roles(recorder: Console) -> list[str]:
    """The role part of every printed panel title, in print order."""
    return re.findall(r"([\w>-]+) \| prov/m", recorder.export_text(clear=False))


def test_interleaved_chains_print_as_coherent_blocks(monkeypatch):
    recorder = Console(record=True, width=100)
    monkeypatch.setattr(cli, "console", recorder)
    reporter = cli.ConsoleReporter(transcript_path=None)

    # two chains completing interleaved; worker_2's chain finishes first
    for stage, role in [
        ("propose", "worker_1"), ("propose", "worker_2"),
        ("critique", "critic->worker_2"), ("critique", "critic->worker_1"),
        ("revise", "worker_2"), ("revise", "worker_1"),
    ]:
        reporter.stage_done(_event(stage, role))
    reporter.close()

    # each block reads propose -> critique -> revise, in chain-completion order
    assert _panel_roles(recorder) == [
        "worker_2", "critic->worker_2", "worker_2",
        "worker_1", "critic->worker_1", "worker_1",
    ]


def test_dead_and_unfinished_chains_still_print(monkeypatch):
    recorder = Console(record=True, width=100)
    monkeypatch.setattr(cli, "console", recorder)
    reporter = cli.ConsoleReporter(transcript_path=None)

    reporter.stage_done(_event("propose", "worker_1"))
    reporter.stage_done(_event("propose", "worker_2"))
    # worker_1's critique dies -> its chain flushes immediately, error panel last
    reporter.stage_done(_event("critique", "critic->worker_1", output="", error="rate limited"))
    assert _panel_roles(recorder) == ["worker_1", "critic->worker_1"]
    assert "rate limited" in recorder.export_text(clear=False)

    # worker_2's chain never finishes (turn aborted); close() must still show its work
    reporter.close()
    assert _panel_roles(recorder) == ["worker_1", "critic->worker_1", "worker_2"]
