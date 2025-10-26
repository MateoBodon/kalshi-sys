"""Environment loading utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@lru_cache(maxsize=1)
def load_env() -> None:
    """Load environment variables from .env files once."""
    for candidate in (Path(".env.local"), Path(".env")):
        if candidate.exists():
            load_dotenv(candidate, override=False)
