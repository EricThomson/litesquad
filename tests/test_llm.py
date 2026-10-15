"""LLM wrapper tests: ASCII sanitizer, OpenRouter request shape, catalog lookup. No network."""

from types import SimpleNamespace

import pytest

from litesquad import llm
from litesquad.config import RunConfig
from litesquad.llm import LLMError, call_model, catalog_entry, to_ascii


def test_to_ascii_maps_common_typography():
    src = "Plan — “phase one’s” goal… • item"
    assert to_ascii(src) == "Plan - \"phase one's\" goal... - item"


def test_to_ascii_output_is_ascii_even_for_exotic_input():
    assert to_ascii("café résumé \U0001F600").isascii()


class FakeClient:
    """Stands in for the openai client: records each request, returns one canned choice."""

    def __init__(self, content: str | None, finish_reason: str) -> None:
        self.requests: list[dict] = []
        choice = SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish_reason
        )
        self._response = SimpleNamespace(choices=[choice])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        return self._response


@pytest.fixture
def install_fake_client(monkeypatch):
    """Swap the shared OpenRouter client for a FakeClient with the given reply."""

    def install(content: str | None = "reply", finish_reason: str = "stop") -> FakeClient:
        client = FakeClient(content, finish_reason)
        monkeypatch.setattr(llm, "openrouter_client", lambda: client)
        return client

    return install


def test_reasoning_effort_is_sent_as_openrouter_reasoning_field(install_fake_client):
    client = install_fake_client()
    call_model("x-ai/grok", [{"role": "user", "content": "hi"}], RunConfig(reasoning_effort="low"))
    request = client.requests[0]
    assert request["model"] == "x-ai/grok"
    assert request["extra_body"] == {"reasoning": {"effort": "low"}}
    assert "temperature" not in request  # an unset run temperature is not sent at all


def test_unset_reasoning_effort_sends_no_extra_fields(install_fake_client):
    client = install_fake_client()
    call_model("org/model", [], RunConfig())
    assert client.requests[0]["extra_body"] is None


def test_structured_call_asks_for_json_and_skips_ascii_coercion(install_fake_client):
    client = install_fake_client(content='{"text": "café"}')
    reply = call_model("org/model", [], RunConfig(), structured=True)
    assert client.requests[0]["response_format"] == {"type": "json_object"}
    assert reply == '{"text": "café"}'  # coercion would corrupt JSON strings


def test_reasoning_that_exhausts_max_tokens_names_the_fix(install_fake_client):
    install_fake_client(content=None, finish_reason="length")
    with pytest.raises(LLMError) as exc:
        call_model("org/model", [], RunConfig(max_tokens=100), role="worker_1")
    assert exc.value.role == "worker_1"
    assert "reasoning_effort" in str(exc.value)


def test_catalog_entry_falls_back_from_unlisted_routing_suffix():
    catalog = {"org/model": {"id": "org/model"}, "org/model:free": {"id": "org/model:free"}}
    assert catalog_entry("org/model", catalog)["id"] == "org/model"
    assert catalog_entry("org/model:free", catalog)["id"] == "org/model:free"  # listed variant
    assert catalog_entry("org/model:nitro", catalog)["id"] == "org/model"  # unlisted shortcut
    assert catalog_entry("org/typo", catalog) is None
