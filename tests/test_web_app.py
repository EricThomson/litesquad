"""App wiring tests: every callback id must exist in the layout.

A callback whose Output/Input/State names an id absent from the layout does not
error at import -- Dash only complains at request time -- so a stale id after a
rename silently makes the control "do nothing" in the browser. This walks the
layout and asserts every referenced id is present, catching that class of bug.
"""

import pytest

pytest.importorskip("dash")

import threading  # noqa: E402

from dash import no_update  # noqa: E402
from dash.development.base_component import Component  # noqa: E402

from litesquad.config import default_config  # noqa: E402
from litesquad.llm import mock_call_model  # noqa: E402
from litesquad.web import app as web_app  # noqa: E402
from litesquad.web.app import create_app  # noqa: E402
from litesquad.web.runner import TurnRunner  # noqa: E402


def _layout_ids(node, acc: set) -> set:
    if isinstance(node, Component):
        cid = getattr(node, "id", None)
        if isinstance(cid, str):
            acc.add(cid)
        _layout_ids(getattr(node, "children", None), acc)
    elif isinstance(node, (list, tuple)):
        for child in node:
            _layout_ids(child, acc)
    return acc


def _referenced_ids(callback: dict) -> set:
    """The plain string ids a callback names. Pattern-matching ids (serialized as JSON
    objects, e.g. the per-answer save buttons) are skipped: those components exist only
    once a turn has rendered, so test_saved_answer_controls_render_with_the_turn checks them."""
    ids = set()
    for dep in callback.get("inputs", []) + callback.get("state", []):
        if isinstance(dep.get("id"), str):
            ids.add(dep["id"])
    # output is a dotted string like "..a.prop...b.prop.." (string ids forbid ".")
    for segment in callback["output"].strip(".").split("..."):
        if segment:
            ids.add(segment.rsplit(".", 1)[0].strip("."))
    return {component_id for component_id in ids if not component_id.startswith("{")}


def test_every_callback_id_exists_in_layout(tmp_path):
    runner = TurnRunner(mock_call_model, transcript_path=tmp_path / "t.jsonl")
    app = create_app(default_config(), runner, mock=True)
    # the layout is a function (rebuilt on every page load), so build one page to inspect
    present = _layout_ids(app.layout(), set())
    referenced = set()
    for callback in app._callback_list:
        referenced |= _referenced_ids(callback)
    missing = referenced - present
    assert not missing, f"callbacks reference ids not in the layout: {sorted(missing)}"


# --- the live tab's conversation flow: Run locks the question, finishing unlocks a fresh box


def _find(node, component_id: str):
    """The component with ``component_id`` in a layout tree, or None."""
    if isinstance(node, Component):
        if getattr(node, "id", None) == component_id:
            return node
        return _find(getattr(node, "children", None), component_id)
    if isinstance(node, (list, tuple)):
        for child in node:
            found = _find(child, component_id)
            if found is not None:
                return found
    return None


def _runner(tmp_path, caller=mock_call_model) -> TurnRunner:
    return TurnRunner(caller, transcript_path=tmp_path / "t_web.jsonl")


def test_run_locks_the_question_in_its_box(tmp_path):
    runner = _runner(tmp_path)
    update = web_app._start_turn(runner, default_config(), "  First question  ", [])
    assert update["box_locked"] is True
    assert update["controls_locked"] is True
    assert update["poll_off"] is False
    assert update["box_text"] is no_update  # the text stays in the box, now read-only
    assert update["label"].startswith("Running (squad)")
    runner.wait(timeout=10)


def test_finished_turn_joins_the_thread_and_frees_a_fresh_box(tmp_path):
    runner = _runner(tmp_path)
    web_app._start_turn(runner, default_config(), "First question", ["quick"])
    runner.wait(timeout=10)

    update = web_app._refresh(runner, default_config(), rendered=0)

    assert update["turns_rendered"] == 1  # the turn (question + answer) was appended
    assert update["box_text"] == ""
    assert update["box_locked"] is False
    assert update["controls_locked"] is False
    assert update["quick_choice"] == []  # quick is chosen afresh for every turn
    assert update["poll_off"] is True
    assert update["label"].startswith("Follow-up")


def test_empty_run_is_refused_without_starting(tmp_path):
    runner = _runner(tmp_path)
    update = web_app._start_turn(runner, default_config(), "   ", [])
    assert update["box_locked"] is no_update
    assert runner.snapshot().running is False


def test_new_conversation_clears_the_thread_and_switches_file(tmp_path):
    runner = _runner(tmp_path)
    web_app._start_turn(runner, default_config(), "First question", ["quick"])
    runner.wait(timeout=10)
    first_file = runner.transcript_path

    update = web_app._start_new_conversation(runner)

    assert update["results"] == [] and update["turns_rendered"] == 0
    assert update["label"] == "Your question or task"
    assert runner.transcript_path != first_file and first_file.exists()


def test_reloaded_page_shows_the_conversation_as_it_stands(tmp_path):
    runner = _runner(tmp_path)
    app = create_app(default_config(), runner, mock=True)
    web_app._start_turn(runner, default_config(), "First question", ["quick"])
    runner.wait(timeout=10)

    page = app.layout()

    assert _find(page, "composer-label").children.startswith("Follow-up")
    assert len(_find(page, "results").children) == 1
    assert _find(page, "turns-rendered").data == 1
    assert _find(page, "task").readOnly is False


def test_reloaded_page_mid_run_keeps_the_question_locked(tmp_path):
    release = threading.Event()

    def held_call(model, messages, run_cfg, *, role="", structured=False):
        release.wait(timeout=10)
        return mock_call_model(model, messages, run_cfg, role=role, structured=structured)

    runner = _runner(tmp_path, held_call)
    app = create_app(default_config(), runner, mock=True)
    web_app._start_turn(runner, default_config(), "Still thinking about this", [])
    try:
        page = app.layout()
        box = _find(page, "task")
        assert box.value == "Still thinking about this" and box.readOnly is True
        assert _find(page, "composer-controls").disabled is True
        assert _find(page, "poll").disabled is False  # keeps polling the running turn
    finally:
        release.set()
        runner.wait(timeout=10)


# --- saving a final answer as markdown


def test_saved_answer_controls_render_with_the_turn(tmp_path):
    runner = _runner(tmp_path)
    app = create_app(default_config(), runner, mock=True)
    web_app._start_turn(runner, default_config(), "First question", ["quick"])
    runner.wait(timeout=10)

    page = app.layout()

    # the button, the export it saves, and the download that delivers it share one {view, turn}
    key = {"view": "live", "turn": 0}
    assert _find(page, {"type": "save-answer", **key}) is not None
    assert _find(page, {"type": "answer-download", **key}) is not None
    answer_export = _find(page, {"type": "answer-export", **key}).data
    assert answer_export["filename"] == "t_web_turn1.md"
    assert "> First question" in answer_export["text"]


@pytest.mark.parametrize("clicks", [None, 0])
def test_save_answer_ignores_anything_but_a_click(clicks):
    answer_export = {"filename": "t_turn1.md", "text": "answer"}
    assert web_app._answer_download(clicks, answer_export) is no_update


def test_save_answer_click_downloads_the_markdown():
    answer_export = {"filename": "t_turn1.md", "text": "**litesquad** | ...\n\nanswer\n"}
    download = web_app._answer_download(1, answer_export)
    assert download["filename"] == "t_turn1.md"
    assert download["content"] == answer_export["text"]
    assert download["type"] == "text/markdown"
