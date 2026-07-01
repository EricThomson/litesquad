"""Typed configuration: good defaults in code, user overrides from ~/.litesquad.

The defaults below are authoritative and versioned. On load, an optional user
config at :func:`litesquad.paths.config_path` is shallow-merged on top, so the
user's file only needs the handful of things they want to change.
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from . import paths

DEFAULT_CONFIG_TOML = """\
# litesquad default squad. Models are LiteLLM model strings.

[run]
# Caps output per stage. Reasoning models (GPT-5, Gemini 2.5 Pro) spend part of
# this on hidden reasoning before the visible answer, so keep it roomy.
max_tokens = 8000
save_transcript = true
# temperature is omitted: frontier models (Opus 4.7+, GPT-5) reject it with a 400.
# Set it only if every model in your squad supports it (Sonnet, Gemini, Opus 4.6-).
# temperature = 0.4

# The judge hears the workers, weighs them, and renders the final answer.
[agents.judge]
model = "anthropic/claude-opus-4-8"

[agents.critic]
model = "openai/gpt-5"

# Each worker responds independently (blind to the others), the critic gives each
# one feedback, the worker revises, and the judge renders the final answer. Any
# agent may add an `instructions` string that is appended to its system prompt.
[[agents.workers]]
model = "anthropic/claude-sonnet-4-6"

[[agents.workers]]
model = "openai/gpt-4.1-mini"
instructions = "Write in tight prose. Use at most a few bullets, only for genuinely list-like content, and never a step-by-step plan for a simple ask. Cut filler and preamble."

[[agents.workers]]
model = "gemini/gemini-2.5-pro"
"""

_STARTER_HEADER = """\
# litesquad overrides. This file mirrors the built-in defaults, fully commented.
# Uncomment and edit a line to override that default; anything left commented
# keeps following the library default, so you only carry the deltas you care about.

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
    judge: AgentConfig
    critic: AgentConfig
    workers: list[AgentConfig] = Field(min_length=1)

    def models(self) -> list[str]:
        """Every distinct model string referenced by the squad."""
        seen: list[str] = []
        for model in [self.judge.model, self.critic.model, *(w.model for w in self.workers)]:
            if model not in seen:
                seen.append(model)
        return seen


def _build(raw: dict) -> SquadConfig:
    """Build a SquadConfig from a raw TOML dict (flattening [agents.*])."""
    agents = raw.get("agents", {})
    return SquadConfig(
        run=raw.get("run", {}),
        judge=agents.get("judge", {}),
        critic=agents.get("critic", {}),
        workers=agents.get("workers", []),
    )


def default_config() -> SquadConfig:
    """The authoritative, versioned default squad."""
    return _build(tomllib.loads(DEFAULT_CONFIG_TOML))


def _merge(defaults: dict, overrides: dict) -> dict:
    """Shallow-merge user overrides onto the default raw config.

    ``run`` keys merge individually; a provided judge/critic replaces that agent; a
    provided (non-empty) workers list replaces the default workers. Nothing deeper.
    """
    d_agents = defaults.get("agents", {})
    o_agents = overrides.get("agents", {})
    return {
        "run": {**defaults.get("run", {}), **overrides.get("run", {})},
        "agents": {
            "judge": o_agents.get("judge") or d_agents.get("judge", {}),
            "critic": o_agents.get("critic") or d_agents.get("critic", {}),
            "workers": o_agents.get("workers") or d_agents.get("workers", []),
        },
    }


def load_config(path: Path | None = None) -> SquadConfig:
    """Load the default squad, with any user overrides merged on top."""
    defaults = tomllib.loads(DEFAULT_CONFIG_TOML)
    cfg_path = path or paths.config_path()
    if cfg_path.exists():
        overrides = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        return _build(_merge(defaults, overrides))
    return _build(defaults)


def _commented_starter() -> str:
    """The defaults rendered as an all-commented override file."""
    lines = []
    for line in DEFAULT_CONFIG_TOML.splitlines():
        stripped = line.lstrip()
        if stripped == "" or stripped.startswith("#"):
            lines.append(line)
        else:
            lines.append(f"# {line}")
    return _STARTER_HEADER + "\n".join(lines) + "\n"


def ensure_starter(path: Path | None = None) -> bool:
    """Write the commented starter override if none exists. True if it was written."""
    cfg_path = path or paths.config_path()
    if cfg_path.exists():
        return False
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(_commented_starter(), encoding="utf-8")
    return True
