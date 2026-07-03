"""Rendering: transcript events and run state -> Dash components.

Pure functions only -- no callbacks, no server, no runner. The live tab and
the transcript browser both render through here, so a past JSONL file and a
running turn look the same on the page. The browser additionally shows the
prompt each stage received (show_prompts=True).
"""

from dash import dcc, html

from ..config import SquadConfig
from ..models import Stage, TranscriptEvent
from .runner import RunState

STAGE_LABEL: dict[Stage, str] = {
    "propose": "responding",
    "critique": "critiquing",
    "revise": "revising",
    "extract": "de-stylizing",
    "cluster": "clustering",
    "judge": "judging",
    "reply": "answering",
}

# Inline styles; deliberately no CSS framework, same spirit as the plain-rich CLI.
_PANEL = {"border": "1px solid #d0d0d0", "borderRadius": "6px", "padding": "0.6rem 0.9rem",
          "margin": "0.5rem 0", "background": "#fff"}
_FINAL = {**_PANEL, "border": "2px solid #2e7d32"}
_ERROR = {**_PANEL, "border": "2px solid #c62828", "color": "#c62828"}
_MUTED = {"color": "#666", "fontSize": "0.85rem"}
_BADGE = {"background": "#e3ecf7", "borderRadius": "10px", "padding": "0.05rem 0.5rem",
          "marginRight": "0.5rem", "fontSize": "0.8rem", "whiteSpace": "nowrap"}
_CONFLICT = {"color": "#c62828", "fontSize": "0.85rem", "marginLeft": "2.6rem"}
_PRE = {"whiteSpace": "pre-wrap", "fontSize": "0.8rem", "background": "#f6f6f6",
        "padding": "0.5rem", "borderRadius": "4px", "overflowX": "auto"}

# One light tint per worker slot (cycled), so everything worker N produced --
# proposal, its critique, revision -- reads as one card of one color.
_WORKER_TINTS = ["#eef4fb", "#eff8ef", "#fdf6ec", "#f5effa", "#eff7f7"]


def notice(text: str) -> html.Div:
    return html.Div(text, style=_MUTED)


def config_header(config: SquadConfig, mock: bool) -> html.Div:
    """The roster at a glance: which model holds each role, one line per worker."""
    roles = [
        ("judge", config.judge.model),
        ("critic", config.critic.model),
        ("extractor", config.extractor.model),
        ("clusterer", config.clusterer.model),
    ]
    workers = [(f"worker_{i + 1}", worker.model) for i, worker in enumerate(config.workers)]
    rows = [
        html.Div([html.Span(name + ": ", style={"fontWeight": "bold"}), html.Span(model)],
                 style={"marginRight": "1.5rem", "display": "inline-block"})
        for name, model in roles + workers
    ]
    if mock:
        rows.append(html.Span("MOCK MODE - canned responses, no API calls",
                              style={**_BADGE, "background": "#fff3cd"}))
    return html.Div(rows, style={**_MUTED, "margin": "0.3rem 0 0.8rem"})


def progress_line(state: RunState) -> html.Div:
    """One line of live status: stage counter plus everyone working right now
    (worker chains run in parallel, so several stages can be in flight at once)."""
    if state.running:
        parts = [f"{state.stages_done} of {state.stages_expected} stages done"]
        parts.extend(
            f"{role} ({model}) {STAGE_LABEL[stage]}..."
            for stage, role, model in state.in_flight
        )
        return html.Div(" -- ".join(parts), style={"fontWeight": "bold", "margin": "0.5rem 0"})
    if state.error:
        return html.Div(f"Turn aborted: {state.error}", style=_ERROR)
    if state.stages_done:
        return html.Div(f"Done: {state.stages_done} stages completed.",
                        style={**_MUTED, "margin": "0.5rem 0"})
    return html.Div()


def _group_by_turn(events: list[TranscriptEvent]) -> list[list[TranscriptEvent]]:
    """Split a flat event list into per-turn lists, in turn order."""
    turns: dict[int, list[TranscriptEvent]] = {}
    for event in events:
        turns.setdefault(event.turn_index, []).append(event)
    return [turns[i] for i in sorted(turns)]


def _worker_key(role: str) -> str:
    """propose/revise carry ``worker_N``; critique carries ``critic->worker_N``."""
    return role.split("->")[-1]


def _worker_tint(worker: str) -> str:
    suffix = worker.split("_")[-1]
    index = int(suffix) - 1 if suffix.isdigit() else 0
    return _WORKER_TINTS[index % len(_WORKER_TINTS)]


def parse_map_line(line: str) -> tuple[int, str | None, str] | None:
    """Parse one rendered content-map line into (support count, conflict, label).

    Lines look like ``- [backed by 2 response(s)] label`` or
    ``- [backed by 1 response(s); CONFLICT: why] label``. Anything else
    returns None and is rendered verbatim.
    """
    if not line.startswith("- [") or "] " not in line:
        return None
    tag, label = line[len("- ["):].split("] ", 1)
    conflict: str | None = None
    if "; CONFLICT: " in tag:
        tag, conflict = tag.split("; CONFLICT: ", 1)
    if not tag.startswith("backed by "):
        return None
    count = tag[len("backed by "):].split(" ", 1)[0]
    if not count.isdigit():
        return None
    return int(count), conflict, label


def _prompt_details(event: TranscriptEvent) -> html.Details:
    summary = html.Summary(
        "prompt", style={**_MUTED, "fontWeight": "normal", "cursor": "pointer"}
    )
    return html.Details([summary, html.Pre(event.prompt, style=_PRE)],
                        style={"margin": "0.3rem 0"})


def _stage_details(
    title: str, event: TranscriptEvent, *, show_prompt: bool = False, raw: bool = False,
    key: str | None = None,
) -> html.Details:
    """One expandable stage: optional prompt, then the output (or the error)."""
    body: list = []
    if show_prompt:
        body.append(_prompt_details(event))
    if event.error:
        body.append(html.Div(event.error, style=_ERROR))
    elif raw:
        body.append(html.Pre(event.output, style=_PRE))
    else:
        body.append(dcc.Markdown(event.output))
    return html.Details([html.Summary(title), *body], style=_PANEL, key=key)


def _worker_panels(events: list[TranscriptEvent], *, show_prompts: bool) -> list[html.Details]:
    """One card per worker, tinted, holding its proposal -> critique -> revision.

    ``key`` is set on each card so React updates a card in place across live
    re-renders (a remount would collapse it while the user is reading).
    """
    by_worker: dict[str, dict[str, TranscriptEvent]] = {}
    for event in events:
        if event.stage in ("propose", "critique", "revise"):
            by_worker.setdefault(_worker_key(event.role), {})[event.stage] = event
    turn = events[0].turn_index if events else 0
    panels = []
    for worker, stages in by_worker.items():
        model = stages.get("propose", next(iter(stages.values()))).model
        inner = []
        for stage, title in (("propose", "Proposal"), ("critique", "Critique"),
                             ("revise", "Revision")):
            if stage in stages:
                event = stages[stage]
                header = f"{title} ({event.model})" if stage == "critique" else title
                inner.append(_stage_details(header, event, show_prompt=show_prompts,
                                            key=f"t{turn}-{worker}-{stage}"))
        panels.append(html.Details(
            [html.Summary(f"{worker} - {model}"), *inner],
            style={**_PANEL, "background": _worker_tint(worker)},
            key=f"t{turn}-{worker}",
        ))
    return panels


def _extract_panel(
    events: list[TranscriptEvent], *, show_prompts: bool
) -> html.Details | None:
    extracts = [e for e in events if e.stage == "extract"]
    if not extracts:
        return None
    inner = [_stage_details(f"{e.role} ({e.model})", e, show_prompt=show_prompts, raw=True)
             for e in extracts]
    return html.Details(
        [html.Summary("De-stylized units (raw extractor output)"), *inner], style=_PANEL
    )


def content_map_panel(event: TranscriptEvent, *, show_prompt: bool = False) -> html.Details:
    """The clusterer's content map, collapsed: support counts as badges, conflicts flagged."""
    items = []
    points = 0
    conflicts = 0
    for line in event.output.splitlines():
        parsed = parse_map_line(line)
        if parsed is None:
            items.append(html.Div(line))
            continue
        points += 1
        count, conflict, label = parsed
        row = [html.Div([html.Span(f"x{count}", style=_BADGE), html.Span(label)])]
        if conflict:
            conflicts += 1
            row.append(html.Div(f"CONFLICT: {conflict}", style=_CONFLICT))
        items.append(html.Div(row, style={"margin": "0.25rem 0"}))
    summary = f"Clusters ({points} points, {conflicts} conflicts)"
    body: list = [
        html.Div("support = how many responses back each point (recurrence, not a vote)",
                 style=_MUTED),
    ]
    if show_prompt:
        body.append(_prompt_details(event))
    return html.Details(
        [html.Summary(summary), *body, *items],
        style=_PANEL, key=f"t{event.turn_index}-map",
    )


def _turn_section(
    events: list[TranscriptEvent], *, show_prompts: bool = False, heading: bool = True
) -> html.Div:
    """Everything one turn produced: worker cards, synthesis, final answer, any error."""
    children: list = []
    if heading:
        children.append(html.H3(f"Turn {events[0].turn_index + 1}: {events[0].task}"))
    children.extend(_worker_panels(events, show_prompts=show_prompts))
    extract_panel = _extract_panel(events, show_prompts=show_prompts)
    if extract_panel is not None:
        children.append(extract_panel)
    for event in events:
        if event.stage == "cluster" and not event.error:
            children.append(content_map_panel(event, show_prompt=show_prompts))
    for event in events:
        if event.stage in ("judge", "reply") and not event.error:
            body: list = [html.H4(f"Final answer ({event.model})",
                                  style={"margin": "0 0 0.4rem"})]
            if show_prompts:
                body.append(_prompt_details(event))
            body.append(dcc.Markdown(event.output))
            children.append(html.Div(body, style=_FINAL))
    for event in events:
        if event.error:
            children.append(html.Div(
                f"{event.role} ({event.model}) failed at {event.stage}: {event.error}",
                style=_ERROR,
            ))
    return html.Div(children)


def turn_sections(events: list[TranscriptEvent]) -> list[html.Div]:
    """The live tab body: one section per turn, newest last (like a chat log)."""
    return [
        html.Div(_turn_section(turn_events), key=f"turn-{turn_events[0].turn_index}")
        for turn_events in _group_by_turn(events)
    ]


def transcript_view(events: list[TranscriptEvent]) -> list:
    """The browser view: same layout as the live tab, plus every stage's prompt."""
    if not events:
        return [notice("Empty transcript.")]
    return [
        html.Details(
            [html.Summary(f"Turn {turn_events[0].turn_index + 1}: {turn_events[0].task}"),
             _turn_section(turn_events, show_prompts=True, heading=False)],
            open=True,
        )
        for turn_events in _group_by_turn(events)
    ]
