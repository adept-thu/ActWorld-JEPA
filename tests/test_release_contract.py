from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_python_parses() -> None:
    for path in ROOT.rglob("*.py"):
        if ".git" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_r383_manifest_is_unique() -> None:
    manifest = json.loads(
        (ROOT / "artifacts" / "manifest.json").read_text(encoding="utf-8")
    )
    names = [item["name"] for item in manifest["artifacts"]]
    assert len(names) == len(set(names)) == 7


def test_only_final_actworld_selector_helper_is_released() -> None:
    package = ROOT / "navsim" / "navsim" / "agents" / "actworld_jepa"
    helper_names = {
        path.name
        for path in package.glob("*_model.py")
        if path.name not in {"actworld_model.py"}
    }
    assert helper_names == {"blended_gain_model.py"}
