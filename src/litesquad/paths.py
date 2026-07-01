"""Filesystem locations for litesquad, all under ~/.litesquad.

The only module that owns the app's on-disk layout, so it lives in one place.
Deliberately plain pathlib and a home-dir convention, no platformdirs.
"""

from pathlib import Path

APP_DIR = Path.home() / ".litesquad"


def config_path() -> Path:
    """Path to the user's override config (may not exist)."""
    return APP_DIR / "config.toml"


def transcripts_dir() -> Path:
    """Directory holding JSONL session transcripts, created if missing."""
    path = APP_DIR / "transcripts"
    path.mkdir(parents=True, exist_ok=True)
    return path
