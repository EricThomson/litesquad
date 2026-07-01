"""Typed configuration loaded from a user-editable ``config.toml``.

On first run the default template below is written to
:func:`litesquad.paths.config_path` and the path is reported to the user.
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from . import paths

DEFAULT_CONFIG_TOML = """\
# litesquad configuration. Models are LiteLLM model strings
# (e.g. "anthropic/claude-sonnet-4-6", "openai/gpt-5", "gemini/gemini-2.5-pro").

[run]
# Caps output per stage. The reasoning models (GPT-5, Gemini 2.5 Pro) spend
# part of this on hidden reasoning before the visible answer, so keep it roomy.
max_tokens = 8000
save_transcript = true
# temperature is omitted by default: frontier models (Opus 4.7+, GPT-5, Fable 5)
# reject it with a 400. Uncomment only if every model in your squad supports it
# (e.g. Sonnet 4.6, Gemini, Opus 4.6 and earlier).
# temperature = 0.4

[agents.pm]
model = "anthropic/claude-opus-4-8"

[agents.critic]
model = "openai/gpt-5"

# One [[agents.workers]] block per worker. Add or remove blocks freely.
# Each worker proposes blind to the others, gets its own critique from the
# critic, and revises against it; the PM then synthesizes the revised set.
# Any agent (pm, critic, or a worker) may add an optional `instructions` string
# that is appended to its system prompt - tune one model without touching the
# others, e.g. instructions = "Write tight prose; avoid bullet spam."
[[agents.workers]]
model = "anthropic/claude-sonnet-4-6"

[[agents.workers]]
model = "gemini/gemini-2.5-pro"
"""


class AgentConfig(BaseModel):
    model: str
    instructions: str | None = None  # optional, appended to this agent's system prompt


class RunConfig(BaseModel):
    temperature: float | None = None
    max_tokens: int = 8000
    save_transcript: bool = True


class SquadConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    pm: AgentConfig
    critic: AgentConfig
    workers: list[AgentConfig] = Field(min_length=1)

    def models(self) -> list[str]:
        """Every distinct model string referenced by the squad."""
        seen: list[str] = []
        for model in [self.pm.model, self.critic.model, *(w.model for w in self.workers)]:
            if model not in seen:
                seen.append(model)
        return seen


def ensure_config(path: Path | None = None) -> Path:
    """Write the default config if none exists. Returns the config path."""
    cfg_path = path or paths.config_path()
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    return cfg_path


def load_config(path: Path | None = None) -> SquadConfig:
    """Load and validate config from TOML.

    The ``[agents]`` table is flattened into the top-level model: ``pm``,
    ``critic`` and ``workers`` are read from ``[agents.*]``.
    """
    cfg_path = path or paths.config_path()
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    agents = raw.get("agents", {})
    return SquadConfig(
        run=raw.get("run", {}),
        pm=agents.get("pm", {}),
        critic=agents.get("critic", {}),
        workers=agents.get("workers", []),
    )
