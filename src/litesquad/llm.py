"""Thin LiteLLM wrapper with explicit, clear failure modes."""

import os

import litellm
from dotenv import load_dotenv

from .config import RunConfig, SquadConfig

# Quiet litellm's "Give Feedback / Get Help" + "LiteLLM.Info" banners on errors;
# we surface a clean LLMError ourselves.
litellm.suppress_debug_info = True

# LiteLLM model strings are "<provider>/<model>"; map the provider to the env
# var that must hold its API key. Unknown providers are not blocked.
PROVIDER_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class LLMError(RuntimeError):
    """A model call failed. Carries the role and model for clear reporting."""

    def __init__(self, role: str, model: str, message: str) -> None:
        self.role = role
        self.model = model
        super().__init__(f"[{role}] {model}: {message}")


class MissingKeysError(RuntimeError):
    """Required API keys are not set in the environment."""


def load_env() -> None:
    """Load a ``.env`` from the working tree into the environment.

    ``override=True`` so the project's ``.env`` wins over any pre-existing
    (possibly stale) shell/Windows environment variable of the same name —
    otherwise an old exported key silently shadows the one you just edited.
    """
    load_dotenv(override=True)


def _provider(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else model


def preflight(config: SquadConfig) -> None:
    """Raise :class:`MissingKeysError` if any required provider key is unset."""
    missing: dict[str, str] = {}  # env var -> provider
    for model in config.models():
        provider = _provider(model)
        key = PROVIDER_KEYS.get(provider)
        if key and not os.environ.get(key):
            missing[key] = provider
    if missing:
        lines = "\n".join(f"  {key}  (for {provider} models)" for key, provider in missing.items())
        raise MissingKeysError(
            "Missing API key(s). Set them in your environment or a .env file:\n" + lines
        )


def mock_call_model(model: str, messages: list[dict], run_cfg: RunConfig, *, role: str = "") -> str:
    """Offline stand-in for :func:`call_model`: canned text, no API key needed.

    Same signature as :func:`call_model`, so it can be injected into the squad to
    exercise the CLI without any provider credentials.
    """
    return (
        f"_(mock {role or 'agent'} response from `{model}`)_\n\n"
        "- A concrete first point.\n"
        "- A second point with a small caveat.\n"
        "- A closing recommendation."
    )


def call_model(model: str, messages: list[dict], run_cfg: RunConfig, *, role: str = "") -> str:
    """Call ``model`` and return its text. Wraps failures in :class:`LLMError`."""
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": run_cfg.max_tokens,
        # Silently drop params a given model rejects (e.g. temperature on
        # Opus 4.7+ / GPT-5) instead of erroring.
        "drop_params": True,
    }
    if run_cfg.temperature is not None:
        params["temperature"] = run_cfg.temperature
    try:
        response = litellm.completion(**params)
    except Exception as exc:  # noqa: BLE001 - surface any provider error uniformly
        raise LLMError(role, model, str(exc)) from exc
    content = response.choices[0].message.content
    if not content:
        raise LLMError(role, model, "model returned an empty response")
    return content
