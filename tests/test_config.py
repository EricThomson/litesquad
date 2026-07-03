"""Config tests: authoritative defaults, override merge, commented starter."""

import pytest

from litesquad.config import (
    AgentConfig,
    RunConfig,
    SquadConfig,
    default_config,
    ensure_starter,
    load_config,
)
from litesquad.llm import MissingKeysError, preflight

ALL_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY")


def test_default_squad():
    config = default_config()
    assert config.judge.model == "anthropic/claude-opus-4-8"
    assert config.critic.model == "openai/gpt-5"
    assert config.extractor.model == "openai/gpt-5"
    assert config.clusterer.model == "anthropic/claude-opus-4-8"
    assert [w.model for w in config.workers] == [
        "anthropic/claude-sonnet-4-6",
        "gemini/gemini-2.5-pro",
        "openrouter/deepseek/deepseek-chat",
        "openrouter/mistralai/mistral-large",
        "openrouter/meta-llama/llama-3.3-70b-instruct",
    ]
    # clustering scales with roster width, so the clusterer ships with headroom
    assert config.clusterer.max_tokens == 24000
    assert all(w.max_tokens is None for w in config.workers)


def test_max_parallel_class_default_matches_toml_default():
    # RunConfig() is constructed directly in a few places (e.g. --check), so the
    # class default must not drift from the authoritative TOML value
    assert default_config().run.max_parallel == RunConfig().max_parallel == 4


def test_missing_file_uses_defaults(tmp_path):
    config = load_config(tmp_path / "nope.toml")
    assert config.model_dump() == default_config().model_dump()


def test_starter_is_all_defaults(tmp_path):
    cfg_path = tmp_path / "config.toml"
    assert ensure_starter(cfg_path) is True
    assert ensure_starter(cfg_path) is False  # already there, not rewritten
    # a fresh starter is fully commented, so it resolves to the defaults
    assert load_config(cfg_path).model_dump() == default_config().model_dump()


def test_override_replaces_workers(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[[agents.workers]]\nmodel = "anthropic/only-me"\n', encoding="utf-8")
    config = load_config(cfg_path)
    assert [w.model for w in config.workers] == ["anthropic/only-me"]
    # unspecified pieces still track the default
    assert config.judge.model == default_config().judge.model


def test_agent_max_tokens_override_parses(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[agents.clusterer]\nmodel = "openai/gpt-5"\nmax_tokens = 24000\n', encoding="utf-8"
    )
    config = load_config(cfg_path)
    assert config.clusterer.max_tokens == 24000
    assert config.judge.max_tokens is None  # untouched agents keep the run default


def test_override_run_key_merges(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[run]\nmax_tokens = 123\n", encoding="utf-8")
    config = load_config(cfg_path)
    assert config.run.max_tokens == 123
    assert config.run.save_transcript is True  # untouched default


def test_models_are_deduplicated():
    config = SquadConfig(
        judge=AgentConfig(model="anthropic/x"),
        critic=AgentConfig(model="anthropic/x"),
        extractor=AgentConfig(model="anthropic/x"),
        clusterer=AgentConfig(model="anthropic/x"),
        workers=[AgentConfig(model="anthropic/x")],
    )
    assert config.models() == ["anthropic/x"]


def test_preflight_reports_missing_keys(monkeypatch):
    for var in ALL_KEYS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(MissingKeysError) as exc:
        preflight(default_config())
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    assert "OPENROUTER_API_KEY" in str(exc.value)  # the default roster is wide


def test_preflight_passes_when_keys_present(monkeypatch):
    for var in ALL_KEYS:
        monkeypatch.setenv(var, "x")
    preflight(default_config())


def test_preflight_skips_keys_no_configured_model_needs(monkeypatch):
    for var in ALL_KEYS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    anthropic_only = SquadConfig(
        judge=AgentConfig(model="anthropic/x"),
        critic=AgentConfig(model="anthropic/x"),
        extractor=AgentConfig(model="anthropic/x"),
        clusterer=AgentConfig(model="anthropic/x"),
        workers=[AgentConfig(model="anthropic/x")],
    )
    preflight(anthropic_only)  # openrouter etc. not referenced, so not required  # should not raise
