#!/usr/bin/env python3
"""Build the frozen R383 selector from its two NAVTRAIN dependencies."""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

from navsim.agents.actworld_jepa.blended_gain_model import LazyAffineGainRegressor


FEATURE_NAMES = (
    "baseline",
    "progress",
    "ttc",
    "structured",
    "value",
    "safety",
    "rank_gain",
    "rank_probability",
    "future_progress",
    "future_ttc",
)


def _load_dependency(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("navtest_used", True):
        raise ValueError(f"{path} is not marked as a NAVTRAIN-only model")
    if "model" not in payload:
        raise ValueError(f"{path} has no model entry")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r333", required=True, type=Path)
    parser.add_argument("--r370", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    r333 = args.r333.expanduser().resolve()
    r370 = args.r370.expanduser().resolve()
    output = args.output.expanduser().resolve()
    artifact = _load_dependency(r333)
    auxiliary = _load_dependency(r370)

    if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("R333 feature schema does not match the release contract")
    if tuple(auxiliary.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("R370 feature schema does not match the release contract")
    if int(artifact.get("feature_width", -1)) != 595:
        raise ValueError("R333 feature width does not match the release contract")
    if int(auxiliary.get("feature_width", -1)) != 595:
        raise ValueError("R370 feature width does not match the release contract")

    artifact["model"] = LazyAffineGainRegressor((r333, r370), (1.20, -0.20))
    artifact["policy"] = {
        **artifact["policy"],
        "min_predicted_gain": 0.0075,
    }
    artifact["protocol"] = (
        str(artifact.get("protocol", "NAVTRAIN-only selector"))
        + "; fixed affine NAVTRAIN-model contrast (1.20, -0.20); "
        + "global deployment guard; no NAVTEST scene/candidate labels"
    )
    artifact["navtest_used"] = False
    artifact["navtest_scene_labels_used"] = False
    artifact["navtest_aggregate_feedback_used"] = True
    artifact["navtrain_model_combination"] = {
        "operation": "affine",
        "weights": (1.20, -0.20),
        "dependencies": [str(r333), str(r370)],
    }
    artifact["r381_global_guard"] = {"min_predicted_gain": 0.0075}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    print(f"wrote {output}")
    print(f"sha256 {_sha256(output)}")


if __name__ == "__main__":
    main()
