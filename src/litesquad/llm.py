"""Thin OpenRouter wrapper with explicit, clear failure modes.

Every model call goes through OpenRouter's OpenAI-compatible API using the
openai SDK, so one key reaches every lab and model strings are OpenRouter's
own model IDs, e.g. ``anthropic/claude-opus-5.5``.
"""

import functools
import json
import os

import openai
from dotenv import load_dotenv

from .config import RunConfig

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_VAR = "OPENROUTER_API_KEY"


# Map common non-ASCII typography to ASCII, then drop anything else, so model
# output in real runs is plain ASCII (no em-dashes, curly quotes, ellipses, ...).
_ASCII_MAP = str.maketrans({
    "—": "-",    # em dash
    "–": "-",    # en dash
    "‘": "'",    # left single quote
    "’": "'",    # right single quote / apostrophe
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "…": "...",  # horizontal ellipsis
    " ": " ",    # non-breaking space
    "•": "-",    # bullet
    "→": "->",   # rightwards arrow
})


def to_ascii(text: str) -> str:
    """Coerce text to plain ASCII: map common typography, drop the rest."""
    return text.translate(_ASCII_MAP).encode("ascii", "ignore").decode("ascii")


def parse_json(text: str):
    """Grab the outermost JSON value (object or array) from a model reply."""
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"no JSON found: {text[:200]!r}")
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    if end <= start:
        raise ValueError(f"unbalanced JSON: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def json_list(reply: str, key: str) -> list:
    """Parse a model reply and return the list under ``key``, tolerating a bare array."""
    parsed = parse_json(reply)
    return parsed if isinstance(parsed, list) else parsed.get(key, [])


class LLMError(RuntimeError):
    """A model call failed. Carries the role and model for clear reporting, and the
    OpenRouter host that served the call when it is known."""

    def __init__(self, role: str, model: str, message: str, served_by: str | None = None) -> None:
        self.role = role
        self.model = model
        self.served_by = served_by
        host = f" (served by {served_by})" if served_by else ""
        super().__init__(f"[{role}] {model}: {message}{host}")


class ModelReply(str):
    """A model's text, plus ``served_by``: the OpenRouter host that served the call.

    One model ID can be served by many hosts (OpenRouter routes each call to one of
    them), and hosts can behave differently, so the transcript records which one
    answered. A str subclass so the Caller seam stays plain text: callers, and every
    fake caller in the tests, keep working with strings, and squad._run_stage reads
    ``served_by`` when it is there.
    """

    served_by: str | None

    def __new__(cls, text: str, served_by: str | None) -> "ModelReply":
        reply = super().__new__(cls, text)
        reply.served_by = served_by
        return reply


class MissingKeyError(RuntimeError):
    """The OpenRouter API key is not set in the environment."""


def load_env() -> None:
    """Load a ``.env`` from the working tree into the environment.

    ``override=True`` so the project's ``.env`` wins over any pre-existing
    (possibly stale) shell/Windows environment variable of the same name;
    otherwise an old exported key silently shadows the one you just edited.
    """
    load_dotenv(override=True)


def preflight() -> None:
    """Raise :class:`MissingKeyError` if the OpenRouter API key is unset."""
    if not os.environ.get(OPENROUTER_KEY_VAR):
        raise MissingKeyError(
            f"Missing API key. Set {OPENROUTER_KEY_VAR} in your environment or a .env file\n"
            "(create one at https://openrouter.ai/keys)."
        )


@functools.cache
def openrouter_client() -> openai.OpenAI:
    """The one shared OpenRouter client, built on first use.

    Shared because the client is thread-safe and pools connections across the
    parallel worker chains. Built lazily so :func:`load_env` has already run,
    and so offline runs (the mock caller, tests) never construct it.
    """
    preflight()
    return openai.OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ[OPENROUTER_KEY_VAR],
        # Retry transient failures (429s, 5xx, dropped connections) with backoff.
        # Matters because parallel worker chains burst concurrent calls.
        max_retries=2,
    )


def mock_call_model(
    model: str, messages: list[dict], run_cfg: RunConfig, *, role: str = "", structured: bool = False
) -> str:
    """Offline stand-in for :func:`call_model`: canned output, no API key needed.

    Same signature as :func:`call_model`, so it can be injected into the squad to exercise the
    CLI without any provider credentials. When ``structured`` is set it returns valid JSON for
    the extract/cluster stages so the pipeline runs end to end offline.
    """
    if structured:
        if "extract" in role:
            return (
                '{"units": [{"kind": "claim", "text": "A concrete point from ' + model + '"}, '
                '{"kind": "claim", "text": "A second point with a small caveat"}]}'
            )
        return '{"clusters": []}'  # empty -> every unit becomes its own singleton downstream
    return (
        f"_(mock {role or 'agent'} response from `{model}`)_\n\n"
        "- A concrete first point.\n"
        "- A second point with a small caveat.\n"
        "- A closing recommendation."
    )


def call_model(
    model: str, messages: list[dict], run_cfg: RunConfig, *, role: str = "", structured: bool = False
) -> str:
    """Call ``model`` through OpenRouter and return its text. Wraps failures in :class:`LLMError`.

    With ``structured=True`` the model is asked for JSON (``response_format``) and the raw
    content is returned unchanged -- no ASCII coercion, which would corrupt JSON strings.
    OpenRouter routes such a request to a host that supports ``response_format`` when one
    exists and otherwise ignores the param, so callers must parse tolerantly either way.
    Params a model does not support (e.g. temperature on most Anthropic and OpenAI models)
    are likewise ignored by OpenRouter rather than rejected.
    """
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": run_cfg.max_tokens,
    }
    if structured:
        params["response_format"] = {"type": "json_object"}
    if run_cfg.temperature is not None:
        params["temperature"] = run_cfg.temperature
    # OpenRouter-only request fields go in extra_body, outside the openai SDK's typed params.
    openrouter_fields = {}
    if run_cfg.reasoning_effort is not None:
        # OpenRouter's documented, unified reasoning control, which it translates per provider.
        # The SDK's top-level reasoning_effort param is OpenAI's own API field, not this one.
        openrouter_fields["reasoning"] = {"effort": run_cfg.reasoning_effort}
    try:
        response = openrouter_client().chat.completions.create(
            **params, extra_body=openrouter_fields or None
        )
    except openai.APIError as exc:
        raise LLMError(role, model, str(exc)) from exc
    # OpenRouter names the host that served the call in a top-level "provider" field. It is
    # not in OpenRouter's documented response schema (observed in real responses), so read
    # it defensively: None if it is ever missing.
    served_by = (response.model_extra or {}).get("provider")
    if not response.choices:
        raise LLMError(role, model, "OpenRouter returned no choices", served_by)
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        if choice.finish_reason == "length":
            raise LLMError(
                role, model,
                f"spent all {run_cfg.max_tokens} max_tokens before any visible answer, most "
                "likely on hidden reasoning; raise this agent's max_tokens or lower its "
                "reasoning_effort",
                served_by,
            )
        raise LLMError(
            role, model, f"model returned an empty response (finish_reason={choice.finish_reason})",
            served_by,
        )
    return ModelReply(content if structured else to_ascii(content), served_by)


def key_info() -> dict:
    """OpenRouter's record for the configured key (label, usage, limit_remaining, ...).

    Free: it spends no tokens, so it is the cheap way to confirm the key is valid.
    """
    try:
        return openrouter_client().get("/key", cast_to=object)["data"]
    except openai.APIError as exc:
        raise LLMError("check", OPENROUTER_KEY_VAR, str(exc)) from exc


def model_catalog() -> dict[str, dict]:
    """OpenRouter's public model catalog, keyed by model ID. Free: it spends no tokens."""
    try:
        models = openrouter_client().get("/models", cast_to=object)["data"]
    except openai.APIError as exc:
        raise LLMError("check", "openrouter.ai/models", str(exc)) from exc
    return {entry["id"]: entry for entry in models}


def catalog_entry(model: str, catalog: dict[str, dict]) -> dict | None:
    """The catalog entry for a configured model ID, or None if OpenRouter does not list it.

    Routing shortcuts such as ``:nitro`` or ``:floor`` are valid suffixes that the catalog
    does not list separately, so a suffixed ID that is not listed falls back to its base model.
    """
    if model in catalog:
        return catalog[model]
    base_model = model.split(":", 1)[0]
    return catalog.get(base_model)
