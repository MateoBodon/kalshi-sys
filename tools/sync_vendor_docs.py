# tools/sync_vendor_docs.py
from __future__ import annotations

import hashlib, json, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

@dataclass(frozen=True)
class Source:
    vendor: str
    url: str
    out: str
    type: str

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def main() -> None:
    manifest_path = Path("docs/vendor/sources.json")
    meta_path = Path("docs/vendor/_sync_meta.json")

    sources_raw = json.loads(manifest_path.read_text())
    sources = [Source(**s) for s in sources_raw]

    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    for s in sources:
        out_path = Path(s.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        headers = {}
        prev = meta.get(s.out, {})
        if "etag" in prev:
            headers["If-None-Match"] = prev["etag"]
        if "last_modified" in prev:
            headers["If-Modified-Since"] = prev["last_modified"]

        r = requests.get(s.url, headers=headers, timeout=60)
        if r.status_code == 304:
            continue
        r.raise_for_status()

        content = r.content
        out_path.write_bytes(content)

        meta[s.out] = {
            "vendor": s.vendor,
            "url": s.url,
            "type": s.type,
            "fetched_at_unix": int(time.time()),
            "sha256": sha256_bytes(content),
            "etag": r.headers.get("ETag"),
            "last_modified": r.headers.get("Last-Modified"),
        }

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

if __name__ == "__main__":
    main()
