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
# Randomize the order the judge sees the responses so no worker is permanently
# first (LLM judges have a primacy bias). Turn off for deterministic-order tests.
shuffle = true
# How many worker chains run at once (threads; model calls are I/O-bound).
# 1 runs everything serially, which is useful when debugging. Higher values
# speed up wide rosters but hit providers harder: the critic model receives
# roughly this many concurrent requests.
max_parallel = 4
# temperature is omitted: frontier models (Opus 4.7+, GPT-5) reject it with a 400.
# Set it only if every model in your squad supports it (Sonnet, Gemini, Opus 4.6-).
# temperature = 0.4

# The judge writes the final answer from the clustered content map (extract -> cluster -> judge).
[agents.judge]
model = "anthropic/claude-opus-4-8"

[agents.critic]
model = "openai/gpt-5"

# The extractor de-stylizes each revised response into content units (JSON). A mechanical step,
# so a cheap model is fine -- swap to openai/gpt-4.1-mini to cut cost.
[agents.extractor]
model = "openai/gpt-5"

# The clusterer groups equivalent units across responses and flags conflicts, building the
# content map the judge writes from. It makes no quality judgment (that is the judge's job).
[agents.clusterer]
model = "anthropic/claude-opus-4-8"

# Each worker responds independently (blind to the others), the critic gives each one feedback,
# and the worker revises. Then the revised answers are extracted into units, clustered into a
# content map, and the judge writes the final answer from it. Any agent may add an
# `instructions` string that is appended to its system prompt.
[[agents.workers]]
model = "anthropic/claude-sonnet-4-6"

[[agents.workers]]
model = "openai/gpt-4.1-mini"
instructions = "Write in tight prose. Use at most a few bullets, only for genuinely list-like content, and never a step-by-step plan for a simple ask. Cut filler and preamble."

[[agents.workers]]
model = "gemini/gemini-2.5-pro"

# openrouter/* workers: one OPENROUTER_API_KEY reaches every provider on
# openrouter.ai (deepseek, mistral, llama, grok, qwen, ...), which is how the
# roster grows wide without needing a key per provider. Prefer non-reasoning
# models as workers here: through OpenRouter a reasoning model can spend the
# whole max_tokens budget on hidden reasoning and return empty content. If you
# do add one, tell it to answer tersely -- instructions cap the visible answer
# reliably, the hidden reasoning only partly.
[[agents.workers]]
model = "openrouter/deepseek/deepseek-chat"

[[agents.workers]]
model = "openrouter/mistralai/mistral-large"

[[agents.workers]]
model = "openrouter/meta-llama/llama-3.3-70b-instruct"
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
    shuffle: bool = True  # randomize the order the judge sees responses (kills primacy bias)
    max_parallel: int = 4  # concurrent worker chains (1 = serial); keep equal to the TOML default


class SquadConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    judge: AgentConfig
    critic: AgentConfig
    extractor: AgentConfig
    clusterer: AgentConfig
    workers: list[AgentConfig] = Field(min_length=1)

    def models(self) -> list[str]:
        """Every distinct model string referenced by the squad."""
        seen: list[str] = []
        roles = [self.judge.model, self.critic.model, self.extractor.model, self.clusterer.model]
        for model in [*roles, *(w.model for w in self.workers)]:
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
        extractor=agents.get("extractor", {}),
        clusterer=agents.get("clusterer", {}),
        workers=agents.get("workers", []),
    )


def default_config() -> SquadConfig:
    """The authoritative, versioned default squad."""
    return _build(tomllib.loads(DEFAULT_CONFIG_TOML))


def _merge(defaults: dict, overrides: dict) -> dict:
    """Shallow-merge user overrides onto the default raw config.

    ``run`` keys merge individually; a provided single agent (judge/critic/extractor/clusterer)
    replaces that agent; a provided (non-empty) workers list replaces the default workers.
    Nothing deeper.
    """
    d_agents = defaults.get("agents", {})
    o_agents = overrides.get("agents", {})
    return {
        "run": {**defaults.get("run", {}), **overrides.get("run", {})},
        "agents": {
            "judge": o_agents.get("judge") or d_agents.get("judge", {}),
            "critic": o_agents.get("critic") or d_agents.get("critic", {}),
            "extractor": o_agents.get("extractor") or d_agents.get("extractor", {}),
            "clusterer": o_agents.get("clusterer") or d_agents.get("clusterer", {}),
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
