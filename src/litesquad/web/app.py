"""Dash application: layout and callbacks. All rendering lives in views.py.

One background thread (the TurnRunner) executes the turn; the page polls a
dcc.Interval once a second while a run is live. While a turn streams, only the
progress line updates (who is working, stages done); the turn's full section
renders once, when it finishes. The interval is disabled when idle, so an open
tab costs nothing between runs.
"""

from dash import Dash, Input, Output, Patch, State, ctx, dcc, html, no_update

from .. import paths
from ..config import SquadConfig
from ..models import load_events_lenient
from . import views
from .runner import TurnRunner

POLL_MS = 1000

_PAGE = {"maxWidth": "60rem", "margin": "0 auto", "padding": "1rem",
         "fontFamily": "system-ui, sans-serif", "background": "#fafafa"}
_INPUT = {"width": "100%", "height": "5rem", "fontFamily": "inherit", "padding": "0.5rem"}
_BUTTON = {"padding": "0.4rem 1.5rem", "fontSize": "1rem", "marginTop": "0.4rem"}


def create_app(config: SquadConfig, runner: TurnRunner, *, mock: bool = False) -> Dash:
    app = Dash(__name__, title="litesquad")

    live_tab = html.Div([
        dcc.Textarea(id="task", placeholder="Your question or task for the ensemble "
                                            "(after a turn, this box is the follow-up)",
                     style=_INPUT),
        html.Div([
            html.Button("Run", id="run", n_clicks=0, style=_BUTTON),
            dcc.Checklist(id="quick", options=[{"label": " quick (judge only, no ensemble)",
                                                "value": "quick"}],
                          value=[], style={"display": "inline-block", "marginLeft": "1rem"}),
        ]),
        html.Div(id="progress"),
        html.Div(id="results", children=[]),
        dcc.Interval(id="poll", interval=POLL_MS, disabled=True),
        # How many finished turns are already on the page. A turn renders exactly
        # once, complete, after it ends; finished turns are Patch-appended so the
        # ones already rendered are never resent (a children write remounts the
        # whole subtree on Dash 4.2+, plotly/dash#3846 -- hence also the <4.2 pin).
        dcc.Store(id="turns-rendered", data=0),
    ], style={"paddingTop": "0.8rem"})

    browse_tab = html.Div([
        dcc.Dropdown(id="transcript-pick", placeholder="Pick a past transcript..."),
        html.Div(id="transcript-view"),
    ], style={"paddingTop": "0.8rem"})

    app.layout = html.Div([
        html.H1("litesquad", style={"marginBottom": "0.2rem"}),
        views.config_header(config, mock),
        dcc.Tabs(id="tabs", value="live", children=[
            dcc.Tab(label="Live run", value="live", children=live_tab),
            dcc.Tab(label="Transcripts", value="browse", children=browse_tab),
        ]),
    ], style=_PAGE)

    @app.callback(
        Output("poll", "disabled"),
        Output("run", "disabled"),
        Output("progress", "children"),
        Output("results", "children"),
        Output("turns-rendered", "data"),
        Input("run", "n_clicks"),
        Input("poll", "n_intervals"),
        State("task", "value"),
        State("quick", "value"),
        State("turns-rendered", "data"),
        prevent_initial_call=True,
    )
    def drive(n_clicks, n_intervals, task, quick, rendered):
        """Single driver: a Run click starts the turn; while it streams, only the
        progress line updates. A turn's full section is appended once, when it
        finishes -- the page never rewrites what it already shows, so open cards
        stay exactly as the user left them.
        """
        if ctx.triggered_id == "run":
            task = (task or "").strip()
            if not task:
                return (no_update, no_update, views.notice("Type a task first."),
                        no_update, no_update)
            try:
                runner.start(task, config, quick=bool(quick))
            except RuntimeError as exc:
                return no_update, no_update, views.notice(str(exc)), no_update, no_update
            # the new turn has no events yet: the board shows everyone queued
            return (False, True, views.progress_panel(runner.snapshot(), config, []),
                    no_update, no_update)
        state = runner.snapshot()
        events = runner.events()
        rendered = rendered or 0
        turn_count = len({e.turn_index for e in events})
        # the last turn is still being written while the runner is alive
        finished = turn_count - 1 if state.running else turn_count
        if finished > rendered:
            patch = Patch()
            done = [e for e in events if rendered <= e.turn_index < finished]
            for section in views.turn_sections(done):
                patch.append(section)
            results, new_rendered = patch, finished
        else:
            results, new_rendered = no_update, no_update
        # the running turn's index: everything before it is finished or rendered
        current = max(finished, rendered)
        current_events = [e for e in events if e.turn_index == current]
        return (not state.running, state.running,
                views.progress_panel(state, config, current_events),
                results, new_rendered)

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
