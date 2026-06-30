"""Config load/default and preflight key-checking tests."""

import pytest

from litesquad.config import SquadConfig, ensure_config, load_config
from litesquad.llm import MissingKeysError, preflight


def test_default_config_roundtrips(tmp_path):
    cfg_path = tmp_path / "config.toml"
    assert not cfg_path.exists()
    ensure_config(cfg_path)
    assert cfg_path.exists()

    config = load_config(cfg_path)
    assert isinstance(config, SquadConfig)
    assert len(config.workers) == 2
    assert config.pm.model.startswith("anthropic/")


def test_models_are_deduplicated():
    from litesquad.config import AgentConfig, SquadConfig as SC

    config = SC(
        pm=AgentConfig(model="anthropic/x"),
        critic=AgentConfig(model="anthropic/x"),
        workers=[AgentConfig(model="anthropic/x")],
    )
    assert config.models() == ["anthropic/x"]


def test_preflight_reports_missing_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = load_config(ensure_config(tmp_path / "config.toml"))

    with pytest.raises(MissingKeysError) as exc:
        preflight(config)
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_preflight_passes_when_keys_present(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    config = load_config(ensure_config(tmp_path / "config.toml"))
    preflight(config)  # should not raise
