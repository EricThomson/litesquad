"""App wiring tests: every callback id must exist in the layout.

A callback whose Output/Input/State names an id absent from the layout does not
error at import -- Dash only complains at request time -- so a stale id after a
rename silently makes the control "do nothing" in the browser. This walks the
layout and asserts every referenced id is present, catching that class of bug.
"""

import pytest

pytest.importorskip("dash")

from dash.development.base_component import Component  # noqa: E402

from litesquad.config import default_config  # noqa: E402
from litesquad.llm import mock_call_model  # noqa: E402
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
    ids = set()
    for dep in callback.get("inputs", []) + callback.get("state", []):
        if isinstance(dep.get("id"), str):
            ids.add(dep["id"])
    # output is a dotted string like "..a.prop...b.prop.." (string ids forbid ".")
    for segment in callback["output"].strip(".").split("..."):
        if segment:
            ids.add(segment.rsplit(".", 1)[0].strip("."))
    return ids


def test_every_callback_id_exists_in_layout():
    app = create_app(default_config(), TurnRunner(mock_call_model), mock=True)
    present = _layout_ids(app.layout, set())
    referenced = set()
    for callback in app._callback_list:
        referenced |= _referenced_ids(callback)
    missing = referenced - present
    assert not missing, f"callbacks reference ids not in the layout: {sorted(missing)}"
