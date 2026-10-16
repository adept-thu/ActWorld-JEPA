#!/usr/bin/env python3
"""Export the NAVTRAIN-approved dense temporal rollout for NAVHARD Stage-2."""

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

from build_v2_temporal_dense_rollout_variants import densify
from navsim.common.dataclasses import Trajectory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    args = parser.parse_args()

    navtrain = json.loads(args.navtrain_report.read_text())
    selected = navtrain["variants"]["joint250_dense_smooth100"]
    reference = navtrain["variants"]["joint250_linear"]
    if (
        selected["pair_utility"] <= reference["pair_utility"]
        or selected["ec_pass_rate"] < reference["ec_pass_rate"]
        or selected["safety_progress_valid_rate"] < reference["safety_progress_valid_rate"]
    ):
        raise RuntimeError("dense rollout did not satisfy the frozen NAVTRAIN gate")

    with args.base_trajectories.open("rb") as stream:
        base = pickle.load(stream)
    mapping = yaml.safe_load(args.mapping_yaml.read_text())
    pairs = [
        tuple(pair)
        for _, _, group_pairs in mapping["reactive_all_mapping"]
        for pair in group_pairs
    ]
    stage2_tokens = {token for pair in pairs for token in pair}
    if len(stage2_tokens) != 2 * len(pairs):
        raise RuntimeError("Stage-2 pair mapping is not disjoint")
    if not stage2_tokens.issubset(base):
        raise RuntimeError("base trajectories do not cover Stage-2 tokens")

    sampling = TrajectorySampling(num_poses=40, time_horizon=4.0)
    output = dict(base)
    for token in stage2_tokens:
        value = base[token]
        trajectory = value["trajectory"] if isinstance(value, dict) else value
        poses = np.asarray(trajectory.poses, dtype=np.float32)
        if poses.shape != (8, 3):
            raise ValueError(f"expected [8,3] base trajectory for {token}, got {poses.shape}")
        output[token] = {
            "trajectory": Trajectory(
                poses=densify(poses, smooth_strength=1.0),
                trajectory_sampling=sampling,
            )
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_pickle(output, args.output)
    report = {
        "schema": "actworld-v2-navhard-dense-temporal-rollout-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_split": "navtrain",
        "navhard_labels_used": False,
        "policy": "joint250 plus endpoint-preserving 0.1s dense rollout smoothing",
        "base_trajectories": str(args.base_trajectories),
        "base_sha256": _sha256(args.base_trajectories),
        "navtrain_report": str(args.navtrain_report),
        "navtrain_pair_utility": selected["pair_utility"],
        "navtrain_ec_pass_rate": selected["ec_pass_rate"],
        "navtrain_safety_progress_valid_rate": selected["safety_progress_valid_rate"],
        "pairs": len(pairs),
        "changed_tokens": len(stage2_tokens),
        "stage1_tokens_preserved": len(base) - len(stage2_tokens),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
