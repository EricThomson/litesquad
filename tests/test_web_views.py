"""View tests: content-map parsing and event grouping. Skipped without dash."""

import pytest

pytest.importorskip("dash")

from litesquad.models import TranscriptEvent  # noqa: E402
from litesquad.web import views  # noqa: E402


def test_parse_map_line_plain():
    line = "- [backed by 2 response(s)] Ship the boring version first"
    assert views.parse_map_line(line) == (2, None, "Ship the boring version first")


def test_parse_map_line_with_conflict():
    line = "- [backed by 3 response(s); CONFLICT: A says weekly, B says daily] Cadence"
    assert views.parse_map_line(line) == (3, "A says weekly, B says daily", "Cadence")


@pytest.mark.parametrize("line", [
    "Some free-form preamble the clusterer emitted",
    "- a bullet without the support tag",
    "- [something else] label",
    "- [backed by many response(s)] non-numeric count",
])
def test_parse_map_line_rejects_unexpected_shapes(line):
    assert views.parse_map_line(line) is None


def test_content_map_panel_renders_unparsed_lines_verbatim():
    event = _event(0, "cluster", "clusterer",
                   output="free-form line\n- [backed by 1 response(s)] a point")
    panel = views.content_map_panel(event)
    assert panel is not None  # smoke: mixed input must not raise


def _event(turn_index: int, stage: str, role: str, **kwargs) -> TranscriptEvent:
    return TranscriptEvent(
        turn_index=turn_index, stage=stage, role=role,
        model=f"provider/{role}-model", prompt="p", task=f"task {turn_index}",
        output=kwargs.pop("output", "some **markdown** output"), **kwargs,
    )


def _fake_turn_events(turn_index: int) -> list[TranscriptEvent]:
    return [
        _event(turn_index, "propose", "worker_1"),
        _event(turn_index, "critique", "critic->worker_1"),
        _event(turn_index, "revise", "worker_1"),
        _event(turn_index, "extract", "extractor->worker_1", output='{"units": []}'),
        _event(turn_index, "cluster", "clusterer",
               output="- [backed by 1 response(s)] a point"),
        _event(turn_index, "judge", "judge"),
    ]


def test_turn_sections_groups_by_turn():
    events = _fake_turn_events(0) + _fake_turn_events(1)
    sections = views.turn_sections(events)
    assert len(sections) == 2


def test_transcript_view_handles_error_events():
    events = [
        _event(0, "propose", "worker_1"),
        _event(0, "critique", "critic->worker_1", output="", error="rate limited"),
    ]
    sections = views.transcript_view(events)
    assert len(sections) == 1  # smoke: error rows must not raise
