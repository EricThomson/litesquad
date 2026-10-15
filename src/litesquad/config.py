"""Typed configuration: good defaults in code, user overrides from ~/.litesquad.

The defaults below are authoritative and versioned. On load, an optional user
config at :func:`litesquad.paths.config_path` is shallow-merged on top, so the
user's file only needs the handful of things they want to change.
"""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import paths

DEFAULT_CONFIG_TOML = """\
# litesquad default squad. Every model is an OpenRouter model ID, copied exactly as
# https://openrouter.ai/models shows it (e.g. anthropic/claude-opus-5.5). After changing
# one, run `litesquad --check`: it flags unknown, retired, and soon-to-expire IDs for
# free, before a run fails partway through.

[run]
# Caps output per call, and the cap covers hidden reasoning AND the visible answer
# together. Long-form tasks get long drafts (4,000-7,000 tokens of visible text is
# normal), so at 8000 a model that reasons a little longer than usual runs out before
# answering. You pay only for tokens actually generated, so headroom is cheap.
max_tokens = 16000
save_transcript = true
# Randomize the order the judge sees the responses so no worker is permanently
# first (LLM judges have a primacy bias). Turn off for deterministic-order tests.
shuffle = true
# How many worker chains run at once (threads; model calls are I/O-bound).
# 1 runs everything serially, which is useful when debugging. Higher values
# speed up wide rosters but hit providers harder: the critic model receives
# roughly this many concurrent requests.
max_parallel = 4
# reasoning_effort sets how much hidden reasoning a model does before answering: one of
# max, xhigh, high, medium, low, minimal, none. OpenRouter translates it per provider (an
# effort level for OpenAI and Gemini, a thinking budget carved out of max_tokens for
# Anthropic). Left unset, each model uses its own default. Set here, it applies to every
# agent that does not set its own.
# reasoning_effort = "medium"
# temperature is omitted: many current models (Anthropic and OpenAI among them) do not
# accept it, and OpenRouter silently ignores it for those, so it only reaches the rest.
# temperature = 0.4

# The judge writes the final answer from the clustered content map (extract -> cluster -> judge).
[agents.judge]
model = "anthropic/claude-opus-5.5"

# The critic is the quality floor, so it gets a flagship model, and one that is not also a
# worker, so it never reviews its own drafts.
[agents.critic]
model = "openai/gpt-6-astra"

# The extractor and clusterer together are the distiller, and share one model that is not the
# judge's: deciding which ideas count as the same is itself a judgment, so it stays out of the
# judge's hands. The extractor de-stylizes each revised response into content units (JSON).
# It is the only step between the workers' prose and the judge, so it is not a place to save
# money: whatever it drops or blurs is lost for good.
[agents.extractor]
model = "openai/gpt-6.1-sol"

# The clusterer groups equivalent units across responses and flags conflicts, building the
# content map the judge writes from. It makes no quality judgment (that is the judge's job).
# Its load scales with roster width (every worker's units land in one pool), and a reasoning
# model in this seat spends hidden budget before emitting the JSON, so it gets extra headroom.
[agents.clusterer]
model = "openai/gpt-6.1-sol"
max_tokens = 24000

# Each worker responds independently (blind to the others), the critic gives each one feedback,
# and the worker revises. Then the revised answers are extracted into units, clustered into a
# content map, and the judge writes the final answer from it. Any agent may add an
# `instructions` string that is appended to its system prompt, a `max_tokens` that overrides
# [run] max_tokens for its own calls, and a `reasoning_effort` that overrides [run]
# reasoning_effort.
#
# The roster is built for maximum variability: one model per lab, so the workers' priors are
# as independent as possible (different pretraining data and post-training, from US and
# Chinese labs). Other labs worth a slot: z-ai (glm), minimax, cohere, nvidia (nemotron).
# Tried and cut: mistral (burned its whole budget on hidden reasoning even at "low") and
# xai's grok (defied plain instructions).
#
# Workers run at reasoning_effort = "low". Nearly every current model reasons, and unchecked
# hidden reasoning can spend the whole max_tokens budget and return no visible answer, which
# drops that worker from the turn. Low keeps that in check; the critic and revision supply
# the depth.
[[agents.workers]]
model = "anthropic/claude-sonnet-5.5"
reasoning_effort = "low"

[[agents.workers]]
model = "google/gemini-4-argon"
reasoning_effort = "low"

[[agents.workers]]
model = "openai/gpt-5.6-terra"
reasoning_effort = "low"

[[agents.workers]]
model = "deepseek/deepseek-v4-pro-0813"
reasoning_effort = "low"

[[agents.workers]]
model = "qwen/qwen3.8-max-0902"
reasoning_effort = "low"

[[agents.workers]]
model = "moonshotai/kimi-k3"
reasoning_effort = "low"
"""

_STARTER_HEADER = """\
# litesquad overrides. This file mirrors the built-in defaults, fully commented.
# Uncomment and edit a line to override that default; anything left commented
# keeps following the library default, so you only carry the deltas you care about.

"""

# OpenRouter's unified reasoning levels (https://openrouter.ai/docs/use-cases/reasoning-tokens).
# A Literal so a typo in config.toml fails at load time, not as a provider error mid-run.
ReasoningEffort = Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"]


class AgentConfig(BaseModel):
    model: str
    instructions: str | None = None  # optional, appended to this agent's system prompt
    max_tokens: int | None = None  # optional, overrides run.max_tokens for this agent's calls
    reasoning_effort: ReasoningEffort | None = None  # optional, overrides run.reasoning_effort


class RunConfig(BaseModel):
    temperature: float | None = None
    max_tokens: int = 16000  # keep equal to the TOML default
    reasoning_effort: ReasoningEffort | None = None  # None = each model's own default
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
