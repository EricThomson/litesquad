"""View tests: content-map parsing and event grouping. Skipped without dash."""

import pytest

pytest.importorskip("dash")

from litesquad.config import AgentConfig, SquadConfig  # noqa: E402
from litesquad.models import TranscriptEvent  # noqa: E402
from litesquad.web import views  # noqa: E402
from litesquad.web.runner import RunState  # noqa: E402


def _state(**overrides) -> RunState:
    base = dict(running=False, stages_done=0, stages_expected=10, in_flight=(), error=None)
    return RunState(**{**base, **overrides})


def _config() -> SquadConfig:
    return SquadConfig(
        judge=AgentConfig(model="prov/judge"),
        critic=AgentConfig(model="prov/critic"),
        extractor=AgentConfig(model="prov/extractor"),
        clusterer=AgentConfig(model="prov/clusterer"),
        workers=[AgentConfig(model="prov/a"), AgentConfig(model="prov/b")],
    )


def _board_rows(panel) -> list[str]:
    """Flatten each status-board row to 'label: status text'."""
    table = panel.children[1]
    rows = []
    for tr in table.children.children:
        label = tr.children[0].children
        status = tr.children[2].children[1].children
        rows.append(f"{label}: {status}")
    return rows


def test_progress_panel_one_row_per_worker_plus_synthesis():
    state = _state(
        running=True, stages_done=1, stages_expected=10,
        in_flight=(("propose", "worker_2", "prov/b"),),
    )
    events = [
        _event(0, "propose", "worker_1"),
        _event(0, "critique", "critic->worker_1"),
        _event(0, "revise", "worker_1"),
    ]
    panel = views.progress_panel(state, _config(), events)
    assert "1 of 10 stages done" in panel.children[0].children
    rows = _board_rows(panel)
    assert rows == [
        "worker_1: revised, de-stylize queued",
        "worker_2: responding...",
        "clusterer: queued",
        "judge: queued",
    ]


def test_progress_panel_shows_chain_failure():
    state = _state(running=True, stages_done=1, stages_expected=10)
    events = [
        _event(0, "propose", "worker_1"),
        _event(0, "critique", "critic->worker_1", output="", error="rate limited"),
    ]
    rows = _board_rows(views.progress_panel(state, _config(), events))
    assert rows[0] == "worker_1: failed at critique"
    assert rows[1] == "worker_2: queued"


def test_progress_panel_quick_mode_is_a_single_judge_row():
    state = _state(running=True, stages_expected=1,
                   in_flight=(("reply", "judge", "prov/judge"),))
    rows = _board_rows(views.progress_panel(state, _config(), []))
    assert rows == ["judge: answering..."]


def test_progress_panel_error_and_done_states():
    assert "boom" in views.progress_panel(_state(error="boom"), _config(), []).children
    assert "10 stages" in views.progress_panel(_state(stages_done=10), _config(), []).children


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


def test_turn_sections_groups_by_turn_with_headings():
    events = _fake_turn_events(0) + _fake_turn_events(1)
    sections = views.turn_sections(events)
    assert len(sections) == 2
    first_heading = sections[0].children.children[0]
    assert first_heading.children.startswith("Turn 1:")


def test_turn_sections_renders_aborted_turns_with_their_error():
    events = [
        _event(0, "propose", "worker_1"),
        _event(0, "critique", "critic->worker_1", output="", error="rate limited"),
    ]
    sections = views.turn_sections(events)
    assert len(sections) == 1  # smoke: a dead turn still renders (error in the card)


def test_turn_heading_truncates_long_tasks():
    long_task = "Design a logo.\n\n" + "Lots of detailed requirements here. " * 30
    events = [_event(0, "propose", "worker_1")]
    events[0].task = long_task
    heading = views._turn_section(events).children[0]
    assert len(heading.children) < 120
    assert heading.children.endswith("...")
    assert "\n" not in heading.children


def test_worker_cards_sorted_by_number_not_completion_order():
    # worker_2's chain finished first, so its events precede worker_1's in the file
    events = [
        _event(0, "propose", "worker_2"),
        _event(0, "revise", "worker_2"),
        _event(0, "propose", "worker_1"),
    ]
    cards = views._worker_panels(events, show_prompts=False)
    summaries = [card.children[0].children for card in cards]
    assert summaries[0].startswith("worker_1") and summaries[1].startswith("worker_2")


def test_every_turn_section_child_is_keyed():
    # unkeyed siblings get remounted on live re-render, collapsing open cards
    children = views._turn_section(_fake_turn_events(0)).children
    assert all(getattr(child, "key", None) for child in children)


def test_transcript_view_handles_error_events():
    events = [
        _event(0, "propose", "worker_1"),
        _event(0, "critique", "critic->worker_1", output="", error="rate limited"),
    ]
    sections = views.transcript_view(events)
    assert len(sections) == 1  # smoke: error rows must not raise
