"""HTTP utilities with caching support."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

META_SUFFIX = ".meta.json"


class HTTPError(RuntimeError):
    """Raised when an HTTP request fails."""


def fetch_with_cache(  # noqa: PLR0913
    url: str,
    cache_path: Path,
    *,
    session: requests.Session | None = None,
    force_refresh: bool = False,
    timeout: float = 15.0,
    headers: dict[str, str] | None = None,
) -> bytes:
    """Fetch a URL with basic ETag/Last-Modified caching."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.with_suffix(cache_path.suffix + META_SUFFIX)
    cached_bytes = cache_path.read_bytes() if cache_path.exists() else None
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    request_headers: dict[str, str] = {}
    if headers:
        request_headers.update(headers)
    request_headers.setdefault(
        "User-Agent",
        "kalshi-alpha/0.1.0 (+https://kalshi.com) Python-requests",
    )

    if not force_refresh and meta:
        if etag := meta.get("etag"):
            request_headers["If-None-Match"] = etag
        if last_modified := meta.get("last_modified"):
            request_headers["If-Modified-Since"] = last_modified

    sess = session or requests.Session()
    response = sess.get(url, headers=request_headers, timeout=timeout)
    if response.status_code == 304 and cached_bytes is not None:
        return cached_bytes
    if not response.ok:
        if cached_bytes is not None:
            return cached_bytes
        raise HTTPError(f"Failed to fetch {url}: {response.status_code}")

    content = response.content
    cache_path.write_bytes(content)
    meta = {
        "url": url,
        "fetched_at": datetime.now(UTC).isoformat(),
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return content
