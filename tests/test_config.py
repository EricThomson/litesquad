"""Config tests: authoritative defaults, override merge, commented starter."""

import pytest
from pydantic import ValidationError

from litesquad.config import (
    AgentConfig,
    RunConfig,
    SquadConfig,
    default_config,
    ensure_starter,
    load_config,
)
from litesquad.llm import OPENROUTER_KEY_VAR, MissingKeyError, preflight

# These tests pin the default roster's shape, never its model names, so updating a
# default model is a one-line config edit with no test churn.


def test_default_models_are_openrouter_ids():
    # OpenRouter IDs are exactly "<org>/<model>"; the old litellm-style
    # "openrouter/<org>/<model>" prefix would have two slashes
    for model in default_config().models():
        org, _, name = model.partition("/")
        assert org and name and "/" not in name, model


def test_default_workers_are_one_per_lab():
    # maximum variability: each worker from a different lab keeps the priors independent
    workers = default_config().workers
    labs = [w.model.split("/", 1)[0] for w in workers]
    assert len(set(labs)) == len(labs) > 1


def test_default_budgets():
    config = default_config()
    # clustering scales with roster width, so the clusterer ships with headroom
    assert config.clusterer.max_tokens == 24000
    assert all(w.max_tokens is None for w in config.workers)
    # workers keep hidden reasoning low; the strong seats use each model's own default
    assert all(w.reasoning_effort == "low" for w in config.workers)
    assert config.run.reasoning_effort is None
    assert config.judge.reasoning_effort is None


def test_run_class_defaults_match_toml_defaults():
    # RunConfig() is constructed directly in a few places (e.g. --check), so the
    # class defaults must not drift from the authoritative TOML values
    assert default_config().run.max_parallel == RunConfig().max_parallel == 4
    assert default_config().run.max_tokens == RunConfig().max_tokens == 16000


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


def test_reasoning_effort_override_parses(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[run]\nreasoning_effort = "medium"\n'
        '[agents.judge]\nmodel = "anthropic/x"\nreasoning_effort = "high"\n',
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    assert config.run.reasoning_effort == "medium"
    assert config.judge.reasoning_effort == "high"


def test_reasoning_effort_typo_fails_at_load(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[run]\nreasoning_effort = "lo"\n', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(cfg_path)


def test_preflight_reports_missing_key(monkeypatch):
    monkeypatch.delenv(OPENROUTER_KEY_VAR, raising=False)
    with pytest.raises(MissingKeyError) as exc:
        preflight()
    assert OPENROUTER_KEY_VAR in str(exc.value)


def test_preflight_passes_when_key_present(monkeypatch):
    monkeypatch.setenv(OPENROUTER_KEY_VAR, "x")
    preflight()
