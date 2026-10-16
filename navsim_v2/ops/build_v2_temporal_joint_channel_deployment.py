#!/usr/bin/env python3
"""Deploy a NAVTRAIN-approved channel-wise joint temporal projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from build_v2_temporal_joint_channel_variants import joint_channel_projection
from navsim.common.dataclasses import Trajectory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _poses(value: object) -> np.ndarray:
    if isinstance(value, dict):
        value = value["trajectory"]
    if not hasattr(value, "poses"):
        raise TypeError(type(value))
    return np.asarray(value.poses, dtype=np.float32)


def _atomic_pickle(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("xb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-trajectories", required=True, type=Path)
    parser.add_argument("--mapping-yaml", required=True, type=Path)
    parser.add_argument("--navtrain-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected-tokens", type=int, default=5912)
    args = parser.parse_args()

    name = "joint_x375_y125_h250"
    navtrain = json.loads(args.navtrain_report.read_text())
    selected = navtrain["variants"][name]
    folds = [float(value) for value in selected.get("fold_pair_utility_gains", [])]
    if float(selected["pair_utility_gain"]) <= 0 or len(folds) != 5 or min(folds) <= 0:
        raise RuntimeError("channel policy did not pass the positive-all-folds NAVTRAIN gate")
    if float(selected["safety_progress_valid_rate"]) < 0.98:
        raise RuntimeError("channel policy failed the NAVTRAIN safety/progress gate")

    with args.base_trajectories.open("rb") as stream:
        base: dict[str, Any] = pickle.load(stream)
    if len(base) != args.expected_tokens:
        raise RuntimeError(f"base coverage differs: {len(base)}")
    mapping = yaml.safe_load(args.mapping_yaml.read_text())
    pairs = [tuple(pair) for _, _, group_pairs in mapping["reactive_all_mapping"] for pair in group_pairs]
    tokens = [str(token) for pair in pairs for token in pair]
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("Stage-2 pairs are not disjoint")

    output = dict(base)
    for current_token, previous_token in pairs:
        current_token, previous_token = str(current_token), str(previous_token)
        previous_variant, current_variant = joint_channel_projection(
            _poses(base[previous_token]), _poses(base[current_token]), 0.375, 0.125, 0.25
        )
        output[previous_token] = {"trajectory": Trajectory(poses=previous_variant)}
        output[current_token] = {"trajectory": Trajectory(poses=current_variant)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_pickle(output, args.output)
    report = {
        "schema": "actworld-v2-navhard-temporal-joint-channel-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_split": "navtrain",
        "navhard_labels_used": False,
        "selection_rule": "highest NAVTRAIN pair utility among positive-all-fold temporal variants",
        "variant": name,
        "strengths": {"x": 0.375, "y": 0.125, "heading": 0.25},
        "navtrain_metrics": selected,
        "base_trajectories": str(args.base_trajectories),
        "base_sha256": _sha256(args.base_trajectories),
        "mapping_yaml": str(args.mapping_yaml),
        "pairs": len(pairs),
        "changed_tokens": len(tokens),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
