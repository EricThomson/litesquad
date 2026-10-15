"""Dash application: layout and callbacks. All rendering lives in views.py.

The live tab reads as a conversation thread: each finished turn shows the user's
question (read-only), the run's details, and the final answer, and the input box
sits below the latest answer. Run locks the box with its text in place -- that
text is now the turn's fixed question -- and the status board appears beneath
it. When the turn finishes, the question and answer join the thread and the box
clears for the follow-up. "New conversation" clears the history and starts a
new transcript file, so continuing and starting over are both explicit choices.

One background thread (the TurnRunner) executes the turn; the page polls a
dcc.Interval once a second while a run is live. While a turn streams, only the
progress board updates; the turn's full section renders once, when it finishes.
The interval is disabled when idle, so an open tab costs nothing between runs.
"""

from dash import Dash, Input, Output, Patch, State, ctx, dcc, html, no_update

from .. import paths
from ..config import SquadConfig
from ..models import TranscriptEvent, load_events_lenient
from . import views
from .runner import RunState, TurnRunner

POLL_MS = 1000

_PAGE = {"maxWidth": "60rem", "margin": "0 auto", "padding": "1rem",
         "fontFamily": "system-ui, sans-serif", "background": "#fafafa"}
_INPUT = {"width": "100%", "height": "5rem", "fontFamily": "inherit", "padding": "0.5rem",
          "boxSizing": "border-box"}
# A locked box keeps its text readable but is shaded like the thread's question blocks.
_INPUT_LOCKED = {**_INPUT, "background": "#f0f4fa", "color": "#333"}
_BUTTON = {"padding": "0.4rem 1.5rem", "fontSize": "1rem", "marginTop": "0.4rem"}
_FIELDSET = {"border": "none", "padding": "0", "margin": "0", "minWidth": "0"}
_COMPOSER_LABEL = {"display": "block", "margin": "1rem 0 0.3rem", "fontWeight": "bold"}
_QUICK_LABEL = " quick: skip the squad, the judge answers alone"


def _finished_turn_count(events: list[TranscriptEvent], state: RunState) -> int:
    """How many turns are complete: while a run is live, its turn is still being written."""
    turn_count = len({event.turn_index for event in events})
    return turn_count - 1 if state.running else turn_count


# Everything the live tab's driver may change on the page. Each action starts from
# "leave it all as is" (_unchanged) and overrides only what it touches.
_DRIVE_OUTPUTS = {
    "poll_off": Output("poll", "disabled"),
    "box_text": Output("task", "value"),
    "box_locked": Output("task", "readOnly"),
    "box_style": Output("task", "style"),
    "controls_locked": Output("composer-controls", "disabled"),
    "quick_choice": Output("quick", "value"),
    "label": Output("composer-label", "children"),
    "progress": Output("progress", "children"),
    "results": Output("results", "children"),
    "turns_rendered": Output("turns-rendered", "data"),
    "storage_note": Output("storage-note", "children"),
}


def _unchanged() -> dict:
    return {key: no_update for key in _DRIVE_OUTPUTS}


def _start_turn(runner: TurnRunner, config: SquadConfig, typed: str | None,
                quick: list | None) -> dict:
    """Run: lock the box with its text in place (now the turn's fixed question) and
    show the status board beneath it."""
    task = (typed or "").strip()
    if not task:
        return {**_unchanged(), "progress": views.notice("Type a question or task first.")}
    try:
        runner.start(task, config, quick=bool(quick))
    except RuntimeError as exc:
        return {**_unchanged(), "progress": views.notice(str(exc))}
    return {
        **_unchanged(),
        "poll_off": False,
        "box_locked": True,
        "box_style": _INPUT_LOCKED,
        "controls_locked": True,
        "label": views.running_label(bool(quick)),
        # the new turn has no events yet: the board shows everyone queued
        "progress": views.progress_panel(runner.snapshot(), config, []),
    }


def _start_new_conversation(runner: TurnRunner) -> dict:
    """New conversation: clear the thread and history, and write to a new transcript
    file (the old conversation stays saved in its own file)."""
    try:
        previous_transcript = runner.reset()
    except RuntimeError as exc:
        return {**_unchanged(), "progress": views.notice(str(exc))}
    return {
        **_unchanged(),
        "results": [],
        "turns_rendered": 0,
        "box_text": "",
        "label": views.composer_label(has_history=False),
        "progress": views.new_conversation_notice(previous_transcript),
        "storage_note": views.storage_note(runner.transcript_path, paths.config_path()),
    }


def _refresh(runner: TurnRunner, config: SquadConfig, rendered: int | None) -> dict:
    """A poll tick: append newly finished turns (each exactly once, complete) and update
    the running turn's board. When the run is over, the question and answer are in the
    thread, so the box clears and unlocks for the follow-up. The page never rewrites
    what it already shows, so open cards stay as the user left them."""
    state = runner.snapshot()
    events = runner.events()
    rendered = rendered or 0
    finished = _finished_turn_count(events, state)
    update = _unchanged()
    if finished > rendered:
        patch = Patch()
        done = [e for e in events if rendered <= e.turn_index < finished]
        for section in views.turn_sections(done):
            patch.append(section)
        update["results"], update["turns_rendered"] = patch, finished
    if state.running:
        # the running turn's index: everything before it is finished or rendered
        current = max(finished, rendered)
        current_events = [e for e in events if e.turn_index == current]
        update["progress"] = views.progress_panel(state, config, current_events)
        return update
    return {
        **update,
        "poll_off": True,
        "box_text": "",
        "box_locked": False,
        "box_style": _INPUT,
        "controls_locked": False,
        "quick_choice": [],  # quick is a per-turn choice, so each turn starts unchecked
        "label": views.composer_label(runner.has_history()),
        # an aborted turn says so here; a finished one speaks for itself in the thread
        "progress": views.progress_panel(state, config, []) if state.error else None,
    }


def create_app(config: SquadConfig, runner: TurnRunner, *, mock: bool = False) -> Dash:
    app = Dash(__name__, title="litesquad")

    def live_tab() -> html.Div:
        """The thread as it stands right now: finished turns, then the input box (locked
        with its question if a turn is running), then that turn's progress board."""
        state = runner.snapshot()
        events = runner.events()
        finished = _finished_turn_count(events, state)
        finished_events = [e for e in events if e.turn_index < finished]
        running_events = [e for e in events if e.turn_index == finished]
        if state.running:
            label = views.running_label(state.quick)
            progress = views.progress_panel(state, config, running_events)
        else:
            label = views.composer_label(runner.has_history())
            progress = None
        return html.Div([
            html.Div(id="results", children=views.turn_sections(finished_events)),
            html.Label(label, id="composer-label", htmlFor="task", style=_COMPOSER_LABEL),
            dcc.Textarea(id="task", placeholder="Type here, then Run",
                         value=state.task if state.running else "",
                         readOnly=state.running,
                         style=_INPUT_LOCKED if state.running else _INPUT),
            # The controls sit in a fieldset so one switch locks them all during a run.
            html.Fieldset(id="composer-controls", disabled=state.running, style=_FIELDSET,
                          children=[
                html.Button("Run", id="run", n_clicks=0, style=_BUTTON),
                dcc.Checklist(id="quick", options=[{"label": _QUICK_LABEL, "value": "quick"}],
                              value=["quick"] if state.running and state.quick else [],
                              style={"display": "inline-block", "marginLeft": "1rem"}),
                html.Button("New conversation", id="new-conversation", n_clicks=0,
                            style={**_BUTTON, "float": "right"}),
            ]),
            html.Div(id="progress", children=progress),
            dcc.Interval(id="poll", interval=POLL_MS, disabled=not state.running),
            # How many finished turns are already on the page. A turn renders exactly
            # once, complete, after it ends; finished turns are Patch-appended so the
            # ones already rendered are never resent (a children write remounts the
            # whole subtree on Dash 4.2+, plotly/dash#3846 -- hence also the <4.2 pin).
            dcc.Store(id="turns-rendered", data=finished),
        ], style={"paddingTop": "0.8rem"})

    def page() -> html.Div:
        """The whole page, rebuilt on every load (Dash calls a layout function per page
        request). So a browser refresh shows the conversation as it stands -- its finished
        turns, the follow-up label, a running question still locked in its box, the current
        transcript file -- instead of a blank page whose next Run would silently be a
        follow-up."""
        browse_tab = html.Div([
            views.transcripts_folder_note(paths.transcripts_dir()),
            dcc.Dropdown(id="transcript-pick", placeholder="Pick a past transcript..."),
            html.Div(id="transcript-view"),
        ], style={"paddingTop": "0.8rem"})
        return html.Div([
            html.H1("litesquad", style={"marginBottom": "0.2rem"}),
            views.config_header(config, mock),
            html.Div(views.storage_note(runner.transcript_path, paths.config_path()),
                     id="storage-note"),
            dcc.Tabs(id="tabs", value="live", children=[
                dcc.Tab(label="Live run", value="live", children=live_tab()),
                dcc.Tab(label="Transcripts", value="browse", children=browse_tab),
            ]),
        ], style=_PAGE)

    app.layout = page

    @app.callback(
        output=_DRIVE_OUTPUTS,
        inputs={
            "run_clicks": Input("run", "n_clicks"),
            "poll_ticks": Input("poll", "n_intervals"),
            "new_conversation_clicks": Input("new-conversation", "n_clicks"),
        },
        state={
            "typed": State("task", "value"),
            "quick": State("quick", "value"),
            "rendered": State("turns-rendered", "data"),
        },
        prevent_initial_call=True,
    )
    def drive(run_clicks, poll_ticks, new_conversation_clicks, typed, quick, rendered):
        """Single driver for the live tab: a Run click, a poll tick, or New conversation."""
        if ctx.triggered_id == "run":
            return _start_turn(runner, config, typed, quick)
        if ctx.triggered_id == "new-conversation":
            return _start_new_conversation(runner)
        return _refresh(runner, config, rendered)

    @app.callback(
        Output("transcript-pick", "options"),
        Input("tabs", "value"),
    )
    def list_transcripts(tab):
        """Refresh the file list whenever the Transcripts tab is opened."""
        if tab != "browse":
            return no_update
        files = sorted(paths.transcripts_dir().glob("*.jsonl"), reverse=True)
        return [{"label": f.name, "value": f.name} for f in files]

    @app.callback(
        Output("transcript-view", "children"),
        Input("transcript-pick", "value"),
        prevent_initial_call=True,
    )
    def show_transcript(name):
        if not name:
            return views.notice("Pick a transcript.")
        # Dropdown values are bare file names, resolved only inside the
        # transcripts dir, so a crafted request can't read arbitrary paths.
        path = paths.transcripts_dir() / name
        if path.parent != paths.transcripts_dir() or not path.exists():
            return views.notice("Transcript not found.")
        try:
            events, skipped = load_events_lenient(path)
        except OSError as exc:
            return views.notice(f"Could not read transcript: {exc}")
        children: list = []
        if not events:
            children.append(views.notice(
                "No readable events; this file is not in the current transcript format."
            ))
        else:
            children.extend(views.transcript_view(events))
        if skipped:
            children.append(views.notice(f"({skipped} line(s) in an outdated format skipped.)"))
        return children

    return app
