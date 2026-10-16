#!/usr/bin/env python3
"""Apply the NAVTRAIN-approved conservative arc-length projection to NAVHARD Stage-1."""

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
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from build_v2_temporal_joint_arclength_variants import joint_arclength_projection
from navsim.common.dataclasses import Trajectory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory(value: object) -> Trajectory:
    if isinstance(value, dict):
        value = value["trajectory"]
    if not isinstance(value, Trajectory) and not hasattr(value, "poses"):
        raise TypeError(f"unsupported trajectory payload {type(value)!r}")
    return value


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
    parser.add_argument("--strength", type=float, default=0.125)
    parser.add_argument("--variant", choices=("arc125", "arc250"), default="arc125")
    parser.add_argument("--expected-tokens", type=int, default=5912)
    args = parser.parse_args()

    navtrain = json.loads(args.navtrain_report.read_text())
    expected_strength = {"arc125": 0.125, "arc250": 0.25}[args.variant]
    if not np.isclose(args.strength, expected_strength):
        raise ValueError(
            f"strength {args.strength} does not match frozen variant {args.variant}"
        )
    selected = navtrain["variants"][args.variant]
    reference = navtrain["variants"]["selected"]
    if (
        selected["pair_utility"] <= reference["pair_utility"]
        or selected["ec_pass_rate"] <= reference["ec_pass_rate"]
        or selected["safety_progress_valid_rate"] < 0.99
    ):
        raise RuntimeError("Stage-1 arc125 policy did not satisfy the frozen NAVTRAIN gate")

    with args.base_trajectories.open("rb") as stream:
        base: dict[str, Any] = pickle.load(stream)
    if len(base) != args.expected_tokens:
        raise RuntimeError(f"base trajectory coverage differs: {len(base)}")

    mapping = yaml.safe_load(args.mapping_yaml.read_text())
    stage1_pairs = [
        (str(previous_token), str(current_token))
        for current_token, previous_token, _ in mapping["reactive_all_mapping"]
    ]
    stage2_tokens = {
        str(token)
        for _, _, group_pairs in mapping["reactive_all_mapping"]
        for pair in group_pairs
        for token in pair
    }
    stage1_tokens = {token for pair in stage1_pairs for token in pair}
    if len(stage1_tokens) != 2 * len(stage1_pairs):
        raise RuntimeError("Stage-1 temporal pairs are not disjoint")
    if stage1_tokens & stage2_tokens:
        raise RuntimeError("Stage-1 and Stage-2 token sets unexpectedly overlap")
    if not (stage1_tokens | stage2_tokens).issubset(base):
        raise RuntimeError("mapping contains tokens absent from the base deployment")

    sampling = TrajectorySampling(num_poses=8, time_horizon=4.0)
    output = dict(base)
    for previous_token, current_token in stage1_pairs:
        previous = np.asarray(_trajectory(base[previous_token]).poses, dtype=np.float32)
        current = np.asarray(_trajectory(base[current_token]).poses, dtype=np.float32)
        if previous.shape != (8, 3) or current.shape != (8, 3):
            raise ValueError(
                f"expected Stage-1 [8,3] trajectories, got {previous.shape} and {current.shape}"
            )
        previous_joint, current_joint = joint_arclength_projection(
            previous, current, args.strength
        )
        output[previous_token] = {
            "trajectory": Trajectory(
                poses=previous_joint,
                trajectory_sampling=sampling,
            )
        }
        output[current_token] = {
            "trajectory": Trajectory(
                poses=current_joint,
                trajectory_sampling=sampling,
            )
        }

    # The existing Stage-2 best must remain byte-equivalent at the pose level.
    for token in stage2_tokens:
        base_trajectory = _trajectory(base[token])
        output_trajectory = _trajectory(output[token])
        if not np.array_equal(base_trajectory.poses, output_trajectory.poses):
            raise RuntimeError(f"Stage-2 trajectory changed unexpectedly: {token}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_pickle(output, args.output)
    report = {
        "schema": "actworld-v2-navhard-stage1-arclength-projection-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_split": "navtrain",
        "navhard_labels_used": False,
        "selection_rule": f"reuse the frozen NAVTRAIN-approved {args.variant} policy",
        "policy": f"joint arc-length projection at strength {args.strength} on Stage-1; preserve Stage-2 best",
        "base_trajectories": str(args.base_trajectories),
        "base_sha256": _sha256(args.base_trajectories),
        "navtrain_report": str(args.navtrain_report),
        "navtrain_pair_utility": selected["pair_utility"],
        "navtrain_ec_pass_rate": selected["ec_pass_rate"],
        "navtrain_safety_progress_valid_rate": selected["safety_progress_valid_rate"],
        "stage1_pairs": len(stage1_pairs),
        "stage1_changed_tokens": len(stage1_tokens),
        "stage2_preserved_tokens": len(stage2_tokens),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
