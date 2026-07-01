"""Config tests: authoritative defaults, override merge, commented starter."""

import pytest

from litesquad.config import (
    AgentConfig,
    SquadConfig,
    default_config,
    ensure_starter,
    load_config,
)
from litesquad.llm import MissingKeysError, preflight


def test_default_squad():
    config = default_config()
    assert config.judge.model == "anthropic/claude-opus-4-8"
    assert config.critic.model == "openai/gpt-5"
    assert [w.model for w in config.workers] == [
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-4.1-mini",
        "gemini/gemini-2.5-pro",
    ]
    # the openai worker ships tamed by default
    openai_worker = next(w for w in config.workers if w.model == "openai/gpt-4.1-mini")
    assert openai_worker.instructions and "prose" in openai_worker.instructions


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
        workers=[AgentConfig(model="anthropic/x")],
    )
    assert config.models() == ["anthropic/x"]


def test_preflight_reports_missing_keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(MissingKeysError) as exc:
        preflight(default_config())
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_preflight_passes_when_keys_present(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "x")
    preflight(default_config())  # should not raise
