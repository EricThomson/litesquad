"""ASCII sanitizer tests (the hard guarantee behind the prompt instruction)."""

from litesquad.llm import to_ascii


def test_to_ascii_maps_common_typography():
    src = "Plan — “phase one’s” goal… • item"
    assert to_ascii(src) == "Plan - \"phase one's\" goal... - item"


def test_to_ascii_output_is_ascii_even_for_exotic_input():
    assert to_ascii("café résumé \U0001F600").isascii()
