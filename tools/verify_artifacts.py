#!/usr/bin/env python3
"""Verify exact file size and SHA256 for the frozen R383 artifact set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "manifest.json",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    failed = False
    for item in manifest["artifacts"]:
        path = args.root / item["name"]
        if not path.is_file():
            print(f"MISSING {path}")
            failed = True
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        ok = actual_size == item["bytes"] and actual_hash == item["sha256"]
        print(f"{'OK' if ok else 'FAIL'} {path.name}")
        if not ok:
            print(f"  size:   {actual_size} (expected {item['bytes']})")
            print(f"  sha256: {actual_hash} (expected {item['sha256']})")
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
