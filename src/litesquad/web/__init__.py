"""litesquad web UI, served by ``litesquad --web``.

Everything here is additive: the core pipeline never imports this package,
and the CLI imports it lazily, only when ``litesquad --web`` is used.
"""

from .. import paths
from ..config import ensure_starter, load_config
from ..llm import MissingKeyError, call_model, load_env, mock_call_model, preflight
from .app import create_app
from .runner import TurnRunner


def serve(port: int = 8050, mock: bool = False, debug: bool = False) -> None:
    """Start the web UI server on localhost; blocks until interrupted."""
    load_env()
    cfg_path = paths.config_path()
    if ensure_starter(cfg_path):
        print(f"Wrote a starter config (all defaults, commented) you can edit at {cfg_path}")
    config = load_config(cfg_path)

    caller = mock_call_model if mock else call_model
    if mock:
        print("Running in --mock mode: canned responses, no API calls.")
    else:
        try:
            preflight()
        except MissingKeyError as exc:
            raise SystemExit(str(exc))

    runner = TurnRunner(caller)
    print(f"Transcript for this session: {runner.transcript_path}")
    print(f"litesquad web UI: http://127.0.0.1:{port}")
    create_app(config, runner, mock=mock).run(port=port, debug=debug)
