"""Secure secret loaders with macOS Keychain support."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from functools import lru_cache

from kalshi_alpha.utils.env import load_env


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _run_security(args: list[str]) -> subprocess.CompletedProcess[str]:
    command = shutil.which("security") or "security"
    return subprocess.run(  # noqa: S603,S607
        [command, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _keychain_lookup(label: str) -> str | None:
    if not _is_macos():
        return None
    if shutil.which("security") is None:
        return None

    attempts: list[list[str]] = [["find-generic-password", "-w", "-l", label]]
    if ":" in label:
        service, account = label.split(":", 1)
        if service and account:
            attempts.append(["find-generic-password", "-w", "-s", service, "-a", account])

    for attempt in attempts:
        result = _run_security(attempt)
        if result.returncode == 0:
            value = result.stdout.strip()
            if value:
                return value
    return None


@lru_cache(maxsize=4)
def load_secret(
    *,
    keychain_label: str | None,
    env_var: str,
    strip: bool = True,
) -> str | None:
    load_env()
    if keychain_label:
        secret = _keychain_lookup(keychain_label)
        if secret:
            return secret.strip() if strip else secret

    value = os.getenv(env_var)
    if not value:
        return None
    return value.strip() if strip else value


@lru_cache(maxsize=1)
def load_polygon_api_key() -> str | None:
    return load_secret(
        keychain_label="kalshi-sys:POLYGON_API_KEY",
        env_var="POLYGON_API_KEY",
    )


__all__ = ["load_polygon_api_key", "load_secret"]
