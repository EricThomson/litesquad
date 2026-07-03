"""litesquad web UI, an optional extra: uv pip install "litesquad[web]".

Everything here is additive: the core pipeline never imports this package,
and the CLI imports it lazily, only when ``litesquad --web`` is used.
"""


def serve(port: int = 8050, mock: bool = False, debug: bool = False) -> None:
    """Start the web UI server on localhost; blocks until interrupted."""
    try:
        import dash  # noqa: F401 - presence check only
    except ModuleNotFoundError:
        raise SystemExit(
            'The web UI needs the "web" extra. Install it with:\n'
            '  uv pip install "litesquad[web]"'
        )

    # Imported here, after the dash check, so a missing extra gives the
    # friendly message above instead of a traceback.
    from .. import paths
    from ..config import ensure_starter, load_config
    from ..llm import MissingKeysError, call_model, load_env, mock_call_model, preflight
    from .app import create_app
    from .runner import TurnRunner

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
            preflight(config)
        except MissingKeysError as exc:
            raise SystemExit(str(exc))

    runner = TurnRunner(caller)
    print(f"Transcript for this session: {runner.transcript_path}")
    print(f"litesquad web UI: http://127.0.0.1:{port}")
    create_app(config, runner, mock=mock).run(port=port, debug=debug)
