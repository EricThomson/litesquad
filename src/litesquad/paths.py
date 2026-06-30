"""Filesystem locations for litesquad.

The only module that touches ``platformdirs`` directly. Everything else asks
here for a path so the app's layout lives in one place.
"""

from pathlib import Path

from platformdirs import PlatformDirs

_dirs = PlatformDirs(appname="litesquad", appauthor=False)


def config_path() -> Path:
    """Path to the user's ``config.toml`` (may not exist yet)."""
    return Path(_dirs.user_config_dir) / "config.toml"


def data_dir() -> Path:
    """Base data directory, created if missing."""
    path = Path(_dirs.user_data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def transcripts_dir() -> Path:
    """Directory holding JSONL session transcripts, created if missing."""
    path = data_dir() / "transcripts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_dir() -> Path:
    """Directory for logs, created if missing."""
    path = Path(_dirs.user_log_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
