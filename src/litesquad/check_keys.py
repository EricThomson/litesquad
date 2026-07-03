"""Dead-simple check that each provider API key is valid and active.

Run:  uv run litesquad-keys      (or: uv run python -m litesquad.check_keys)

Pings one cheap model per provider with a 1-token request and prints a
checkmark per key. Independent of the squad config, so it tests the key
itself, not whichever model your squad happens to use.
"""

import os

import litellm
from rich.console import Console

from .llm import load_env  # loads .env with override=True; importing also quiets litellm

console = Console()

# One cheap, broadly-available probe model per provider key.
PROBES = {
    "ANTHROPIC_API_KEY": "anthropic/claude-haiku-4-5",
    "OPENAI_API_KEY": "openai/gpt-4o-mini",
    "GEMINI_API_KEY": "gemini/gemini-2.5-flash",
    "OPENROUTER_API_KEY": "openrouter/openai/gpt-4o-mini",
}


def check(model: str) -> tuple[bool, str]:
    try:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            drop_params=True,
        )
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - any failure means the key isn't usable
        return False, str(exc).splitlines()[0].strip()


def main() -> None:
    load_env()
    for key, model in PROBES.items():
        # An unset key is a distinct state from an invalid one: say "not set"
        # instead of probing and printing a provider auth exception.
        if not os.environ.get(key):
            console.print(f"[red]✗[/] {key} — not set")
            continue
        ok, detail = check(model)
        if ok:
            console.print(f"[green]✓[/] {key} — active")
        else:
            console.print(f"[red]✗[/] {key} — invalid: {detail}")


if __name__ == "__main__":
    main()
