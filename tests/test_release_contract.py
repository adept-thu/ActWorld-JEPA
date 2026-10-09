from __future__ import annotations

import ast
import json
from pathlib import Path

import setuptools


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
    assert len(names) == len(set(names)) == 5
    assert set(names) == {
        "actworld_jepa_navsim_v1_r383.ckpt",
        "actworld_jepa_navsim_v1_r383_selector.pkl",
        "ridge_strict_verifier.pkl",
        "multidomain_safe_oracle_set_head.pt",
        "aggressive_setwise_cascade_scorer.pkl",
    }


def test_only_final_actworld_selector_helper_is_released() -> None:
    package = ROOT / "navsim" / "navsim" / "agents" / "actworld_jepa"
    helper_names = {
        path.name
        for path in package.glob("*_model.py")
        if path.name not in {"actworld_model.py"}
    }
    assert helper_names == {"blended_gain_model.py"}


def test_r383_custom_pickle_modules_are_released() -> None:
    """Keep every repository-local class referenced by released artifacts importable."""

    package = ROOT / "navsim" / "navsim" / "agents" / "actworld_jepa"
    required_modules = {
        "blended_gain_model.py",
        "torch_setwise_estimator.py",
    }
    assert {path.name for path in package.glob("*.py")} >= required_modules


def test_runtime_packages_are_included_by_setup() -> None:
    """A regular install must contain ActWorld and its frozen dependencies."""

    packages = set(setuptools.find_packages(str(ROOT / "navsim")))
    assert {
        "navsim.agents.actworld_jepa",
        "navsim.agents.drive_jepa_perception_based",
        "navsim.agents.drive_jepa_perception_based.bevformer",
        "navsim.agents.drive_jepa_perception_based.score_module",
        "navsim.agents.transfuser",
        "navsim.planning.training.callbacks",
        "navsim.planning.utils.multithreading",
    } <= packages
