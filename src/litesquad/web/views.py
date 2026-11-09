"""Rendering: transcript events and run state -> Dash components.

Pure functions only -- no callbacks, no server, no runner. The live tab and
the transcript browser both render through here, so a past JSONL file and a
running turn look the same on the page. The browser additionally shows the
prompt each stage received (show_prompts=True).
"""

from pathlib import Path

from dash import dcc, html

from ..config import SquadConfig
from ..export import answer_filename, answer_markdown
from ..models import Stage, TranscriptEvent, final_event
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
# The user's own message, shown in full and read-only. Height-capped so a long prompt
# scrolls inside its box instead of pushing the answer far down the page.
_QUESTION = {**_PANEL, "background": "#f0f4fa", "whiteSpace": "pre-wrap",
             "maxHeight": "12rem", "overflowY": "auto"}
_SAVE_BUTTON = {"marginTop": "0.6rem", "padding": "0.2rem 0.8rem", "fontSize": "0.85rem",
                "cursor": "pointer"}

# One light tint per worker slot (cycled), so everything worker N produced --
# proposal, its critique, revision -- reads as one card of one color.
_WORKER_TINTS = ["#eef4fb", "#eff8ef", "#fdf6ec", "#f5effa", "#eff7f7"]


def notice(text: str) -> html.Div:
    return html.Div(text, style=_MUTED)


def composer_label(has_history: bool) -> str:
    """What the input box will do with its text: start fresh, or continue the thread."""
    if not has_history:
        return "Your question or task"
    return ("Follow-up: the squad will see the conversation above. For something unrelated, "
            "start a new conversation.")


def running_label(quick: bool) -> str:
    """Shown above the locked input box while its question runs."""
    mode = "quick: judge only" if quick else "squad"
    return (f"Running ({mode}). This question is now fixed; a fresh box for your follow-up "
            "appears when it finishes.")


def new_conversation_notice(previous_transcript: Path) -> html.Div:
    """Confirms the reset, and says where the old conversation is kept."""
    if not previous_transcript.exists():
        return notice("Started a new conversation.")
    return html.Div(["Started a new conversation. The previous one is saved in ",
                     html.Code(str(previous_transcript))], style=_MUTED)


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


def storage_note(transcript_path: Path, config_path: Path) -> html.Div:
    """Where this session lives on disk: the transcript being written, and the config read.

    The web UI always writes its transcript (it renders the live run from that file), so
    this holds regardless of run.save_transcript.
    """
    return html.Div([
        html.Div(["This session is saved to ", html.Code(str(transcript_path))]),
        html.Div(["Models and settings: ", html.Code(str(config_path)),
                  " (your overrides, merged over the built-in defaults)"]),
    ], style={**_MUTED, "margin": "-0.4rem 0 0.8rem"})


def transcripts_folder_note(folder: Path) -> html.Div:
    """Where the past transcripts listed in the Transcripts tab live."""
    return html.Div(["Every session's full record (each stage's prompt and output) is in ",
                     html.Code(str(folder))],
                    style={**_MUTED, "marginBottom": "0.5rem"})


# Live status board glyphs: one look per state, colored dot/check/cross.
_GLYPHS = {
    "waiting": ("○", "#9aa0a6"),
    "active": ("●", "#9a6b00"),
    "done": ("✓", "#2e7d32"),
    "failed": ("✗", "#c62828"),
}
_CELL = {"padding": "0.18rem 1rem 0.18rem 0", "textAlign": "left", "whiteSpace": "nowrap"}


def _status_row(label: str, model: str, kind: str, text: str, tint: str | None = None) -> html.Tr:
    glyph, color = _GLYPHS[kind]
    label_style = {**_CELL, "fontWeight": "bold"}
    if tint:  # same tint as this worker's card below, so row and card read as one
        label_style.update({"borderLeft": f"6px solid {tint}", "paddingLeft": "0.5rem"})
    return html.Tr([
        html.Td(label, style=label_style),
        html.Td(model, style={**_CELL, **_MUTED}),
        html.Td([html.Span(f"{glyph} ", style={"color": color}),
                 html.Span(text, style={"color": color if kind == "failed" else "#444"})],
                style=_CELL),
    ])


def _worker_row(index: int, model: str, turn_events: list[TranscriptEvent],
                in_flight: tuple) -> html.Tr:
    """One worker's chain, as a single status: the latest thing true of it."""
    role = f"worker_{index + 1}"
    chain_roles = (role, f"critic->{role}", f"extractor->{role}")
    failure = next((e for e in turn_events if e.error and e.role in chain_roles), None)
    active = {stage for stage, r, _ in in_flight if r in chain_roles}
    finished = {e.stage for e in turn_events if e.role in chain_roles and not e.error}
    if failure:
        kind, text = "failed", f"failed at {failure.stage}"
    elif "propose" in active:
        kind, text = "active", "responding..."
    elif "critique" in active:
        kind, text = "active", "critic reviewing..."
    elif "revise" in active:
        kind, text = "active", "revising..."
    elif "extract" in active:
        kind, text = "active", "de-stylizing..."
    elif "extract" in finished:
        kind, text = "done", "done"
    elif "revise" in finished:
        kind, text = "done", "revised, de-stylize queued"
    elif finished:
        kind, text = "active", "between stages..."
    else:
        kind, text = "waiting", "queued"
    return _status_row(role, model, kind, text, tint=_worker_tint(role))


def _synthesis_row(label: str, model: str, stages: tuple[str, ...], active_text: str,
                   turn_events: list[TranscriptEvent], in_flight: tuple) -> html.Tr:
    failure = next((e for e in turn_events if e.error and e.stage in stages), None)
    if failure:
        kind, text = "failed", f"failed at {failure.stage}"
    elif any(stage in stages for stage, _, _ in in_flight):
        kind, text = "active", active_text
    elif any(e.stage in stages and not e.error for e in turn_events):
        kind, text = "done", "done"
    else:
        kind, text = "waiting", "queued"
    return _status_row(label, model, kind, text)


def progress_panel(state: RunState, config: SquadConfig,
                   turn_events: list[TranscriptEvent]) -> html.Div:
    """The live status board: stage counter, then one row per worker chain plus
    the synthesis roles. Rebuilt wholesale every tick, which is safe -- unlike
    the results, it holds no state the user can lose."""
    if not state.running:
        if state.error:
            return html.Div(f"Turn aborted: {state.error}", style=_ERROR)
        if state.stages_done:
            return html.Div(f"Done: {state.stages_done} stages completed.",
                            style={**_MUTED, "margin": "0.5rem 0"})
        return html.Div()
    rows = []
    if state.stages_expected == 1:  # quick mode: just the judge answering
        rows.append(_synthesis_row("judge", config.judge.model, ("reply",),
                                   "answering...", turn_events, state.in_flight))
    else:
        for index, worker in enumerate(config.workers):
            rows.append(_worker_row(index, worker.model, turn_events, state.in_flight))
        rows.append(_synthesis_row("clusterer", config.clusterer.model, ("cluster",),
                                   "clustering...", turn_events, state.in_flight))
        rows.append(_synthesis_row("judge", config.judge.model, ("judge",),
                                   "writing final answer...", turn_events, state.in_flight))
    return html.Div([
        html.Div(f"{state.stages_done} of {state.stages_expected} stages done",
                 style={"fontWeight": "bold", "margin": "0.5rem 0 0.3rem"}),
        html.Table(html.Tbody(rows), style={"borderCollapse": "collapse"}),
    ], style={**_PANEL, "background": "#fcfcfc"})


def _task_snippet(task: str, limit: int = 90) -> str:
    """One tidy line of the task for headings: whitespace collapsed, long tails cut.
    A multi-paragraph task must never render as a wall of bold heading text."""
    snippet = " ".join(task.split())
    return snippet if len(snippet) <= limit else snippet[:limit].rstrip() + "..."


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


def _worker_number(worker: str) -> int:
    suffix = worker.split("_")[-1]
    return int(suffix) if suffix.isdigit() else 0


def _worker_panels(events: list[TranscriptEvent], *, show_prompts: bool) -> list[html.Details]:
    """One card per worker, tinted, holding its proposal -> critique -> revision.

    Cards are sorted by worker number (chains complete in any order, but the page
    should read worker_1..N and card positions must not shift mid-run). ``key`` is
    set on each card so React updates it in place across live re-renders (a remount
    would collapse it while the user is reading).
    """
    by_worker: dict[str, dict[str, TranscriptEvent]] = {}
    for event in events:
        if event.stage in ("propose", "critique", "revise"):
            by_worker.setdefault(_worker_key(event.role), {})[event.stage] = event
    turn = events[0].turn_index if events else 0
    panels = []
    for worker in sorted(by_worker, key=_worker_number):
        stages = by_worker[worker]
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
    turn = extracts[0].turn_index
    inner = [_stage_details(f"{e.role} ({e.model})", e, show_prompt=show_prompts, raw=True,
                            key=f"t{turn}-{e.role}")
             for e in sorted(extracts, key=lambda e: _worker_number(_worker_key(e.role)))]
    return html.Details(
        [html.Summary("De-stylized units (raw extractor output)"), *inner],
        style=_PANEL, key=f"t{turn}-extracts",
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


def _question_block(events: list[TranscriptEvent]) -> html.Div:
    """The user's message for one turn, in full and read-only, and how it was run."""
    turn = events[0].turn_index
    quick = any(event.stage == "reply" for event in events)
    mode = "quick (judge only)" if quick else "squad"
    return html.Div([
        html.Div(f"Turn {turn + 1} - {mode}", style={**_MUTED, "marginTop": "1rem"}),
        html.Div(events[0].task, style=_QUESTION),
    ], key=f"t{turn}-question")


def _save_answer_control(turn_events: list[TranscriptEvent], transcript_name: str,
                         view: str) -> html.Div:
    """The final answer's "Save as markdown" button, plus the two pieces it needs on the
    page: a Store holding the export (built here, at render time) and the Download that
    hands it to the browser. app.py's save callback pairs the three by matching ids.

    The ids carry the view ("live" or "browse") as well as the turn number, because both
    tabs can show the same turn at once and every id on the page must be unique.
    """
    turn = turn_events[0].turn_index
    return html.Div([
        html.Button("Save as markdown", n_clicks=0, style=_SAVE_BUTTON,
                    id={"type": "save-answer", "view": view, "turn": turn}),
        dcc.Store(id={"type": "answer-export", "view": view, "turn": turn},
                  data={"filename": answer_filename(transcript_name, turn),
                        "text": answer_markdown(turn_events)}),
        dcc.Download(id={"type": "answer-download", "view": view, "turn": turn}),
    ])


def _turn_section(events: list[TranscriptEvent], *, transcript_name: str, view: str,
                  show_prompts: bool = False) -> html.Div:
    """Everything one turn produced: the question, worker cards, synthesis, final answer
    (with its save button), any error. ``transcript_name`` names the saved answer's file;
    ``view`` ("live" or "browse") keeps the save button's ids unique across the two tabs.

    Every child carries a stable ``key``: the live tab re-renders this tree each time
    an event lands, and any unkeyed sibling would be remounted by React -- collapsing
    whatever cards the user had open mid-run.
    """
    turn = events[0].turn_index
    children: list = [_question_block(events)]
    children.extend(_worker_panels(events, show_prompts=show_prompts))
    extract_panel = _extract_panel(events, show_prompts=show_prompts)
    if extract_panel is not None:
        children.append(extract_panel)
    for event in events:
        if event.stage == "cluster" and not event.error:
            children.append(content_map_panel(event, show_prompt=show_prompts))
    final = final_event(events)
    if final is not None:
        body: list = [html.H4(f"Final answer ({final.model})", style={"margin": "0 0 0.4rem"})]
        if show_prompts:
            body.append(_prompt_details(final))
        body.append(dcc.Markdown(final.output))
        body.append(_save_answer_control(events, transcript_name, view))
        children.append(html.Div(body, style=_FINAL, key=f"t{turn}-final"))
    for index, event in enumerate(events):
        if event.error:
            children.append(html.Div(
                f"{event.role} ({event.model}) failed at {event.stage}: {event.error}",
                style=_ERROR, key=f"t{turn}-error-{index}",
            ))
    return html.Div(children)


def turn_sections(events: list[TranscriptEvent], transcript_name: str) -> list[html.Div]:
    """One grouped section per turn, in turn order, for the live tab. ``transcript_name``
    is the file the events come from (it names saved answers).

    The live tab renders a turn exactly once, after it finishes: while a run
    streams, only the progress line updates. Dash 4.2+ remounts the whole
    results subtree on any children write (plotly/dash#3846) -- rendering
    mid-run would collapse whatever cards the user has open, so nothing is put
    on the page that a later write could disturb.
    """
    return [
        html.Div(_turn_section(turn_events, transcript_name=transcript_name, view="live"),
                 key=f"turn-{turn_events[0].turn_index}")
        for turn_events in _group_by_turn(events)
    ]


def transcript_view(events: list[TranscriptEvent], transcript_name: str) -> list:
    """The browser view: same layout as the live tab, plus every stage's prompt.
    ``transcript_name`` is the file being browsed (it names saved answers)."""
    if not events:
        return [notice("Empty transcript.")]
    return [
        html.Details(
            [html.Summary(f"Turn {turn_events[0].turn_index + 1}: "
                          f"{_task_snippet(turn_events[0].task)}"),
             _turn_section(turn_events, transcript_name=transcript_name, view="browse",
                           show_prompts=True)],
            open=True,
        )
        for turn_events in _group_by_turn(events)
    ]
