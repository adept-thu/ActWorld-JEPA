#!/usr/bin/env python3
"""Fail when a source release contains private paths, secrets, or large files."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cff", ".cfg", ".ini", ".json", ".md", ".py", ".sh", ".toml",
    ".txt", ".yaml", ".yml",
}
ALLOWED_BINARY = {Path("navsim/data/8192.npy")}
FORBIDDEN_SUFFIXES = {".ckpt", ".pkl", ".pickle", ".pt", ".pth"}
PATTERNS = {
    "private account": re.compile("ext_" + "songzhiying|ext_" + "zhangshengkai", re.I),
    "server address": re.compile("39" + r"\.105\.30\.146"),
    "private unix path": re.compile(r"/(?:home|media/disks|data/share)/[^\s`\"']+"),
    "private key": re.compile("BEGIN " + r"(?:RSA |OPENSSH )?PRIVATE KEY"),
}


def main() -> None:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact: {relative}")
        if path.stat().st_size > 50 * 1024 * 1024 and relative not in ALLOWED_BINARY:
            failures.append(f"file larger than 50 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # These are generic examples/defaults in the vendored V-JEPA 2 tree,
        # not paths copied from the ActWorld-JEPA development environment.
        text = text.replace("/home/username/", "").replace("/home/{USER}/", "")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {relative}")
    if failures:
        print("Release audit failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Release audit passed")


if __name__ == "__main__":
    main()
