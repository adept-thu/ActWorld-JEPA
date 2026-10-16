#!/usr/bin/env python3
"""Apply a NAVTRAIN-fitted ActWorld temporal FCTS head to frozen candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from fit_actworld_temporal_fcts import SAFETY, coarse_motion, pair_feature
from navsim.common.dataclasses import Trajectory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def atomic_pickle(payload: dict[str, Any], path: Path) -> None:
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
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--mapping-yaml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected-tokens", type=int, default=5912)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Use a NAVTRAIN-audited policy threshold instead of the artifact winner.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Use a NAVTRAIN-audited coverage fraction instead of the artifact winner.",
    )
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("temporal FCTS output already exists")

    paths = sorted(args.diagnostics.glob("*.npz"))
    if len(paths) != args.expected_tokens:
        raise RuntimeError(f"diagnostic coverage differs: {len(paths)} != {args.expected_tokens}")
    records = {path.stem: load_npz(path) for path in paths}
    for token, record in records.items():
        if "world_features" not in record:
            raise KeyError(f"missing world_features for {token}")

    artifact = joblib.load(args.artifact)
    if artifact.get("format") != "actworld_v2_navtrain_temporal_fcts_pair_head_v1":
        raise ValueError("unsupported temporal FCTS artifact")
    if artifact.get("training_split") != "navtrain":
        raise ValueError("temporal FCTS head was not trained on NAVTRAIN")
    if artifact.get("navtest_used_for_training") is not False:
        raise ValueError("temporal FCTS artifact declares test-data use")
    threshold = float(
        artifact["policy"]["threshold"] if args.threshold is None else args.threshold
    )
    fraction = float(
        artifact["policy"]["fraction"] if args.fraction is None else args.fraction
    )
    if threshold < 0.0 or not 0.0 < fraction <= 1.0:
        raise ValueError("threshold must be nonnegative and fraction must be in (0, 1]")

    ordered = sorted(records)
    world = np.concatenate(
        [np.asarray(records[token]["world_features"], dtype=np.float32) for token in ordered],
        axis=0,
    )
    transformed = artifact["pca"].transform(world).reshape(
        len(ordered), 32, int(artifact["components"])
    )
    projected = {token: transformed[index] for index, token in enumerate(ordered)}
    motion = {
        token: np.stack([coarse_motion(plan) for plan in records[token]["refined_proposals"]])
        for token in ordered
    }

    mapping = yaml.safe_load(args.mapping_yaml.read_text())
    pairs = [
        tuple(pair)
        for _, _, group_pairs in mapping["reactive_all_mapping"]
        for pair in group_pairs
    ]
    degree: dict[str, int] = {}
    for pair in pairs:
        for token in pair:
            degree[token] = degree.get(token, 0) + 1
    if max(degree.values()) != 1:
        raise RuntimeError("temporal FCTS expects disjoint NAVHARD pairs")

    pending = []
    feature_blocks = []
    for number, (current_token, previous_token) in enumerate(pairs, start=1):
        current = records[current_token]
        previous = records[previous_token]
        current_baseline = int(np.argmax(current["pdm_score"]))
        previous_baseline = int(np.argmax(previous["pdm_score"]))
        current_error = float(
            np.mean(
                np.square(
                    current["refined_proposals"][current_baseline]
                    - current["trajectory"]
                )
            )
        )
        previous_error = float(
            np.mean(
                np.square(
                    previous["refined_proposals"][previous_baseline]
                    - previous["trajectory"]
                )
            )
        )
        # Preserve earlier frozen blend decisions: the pair head was fitted on
        # discrete candidates and must not extrapolate from an off-support plan.
        if current_error > 1.0e-10 or previous_error > 1.0e-10:
            continue
        current_pool = np.unique(
            np.r_[current_baseline, np.argsort(current["pdm_score"])[-4:]]
        ).astype(np.int64)
        previous_pool = np.unique(
            np.r_[previous_baseline, np.argsort(previous["pdm_score"])[-4:]]
        ).astype(np.int64)
        combinations = np.asarray(
            [(i, j) for i in current_pool for j in previous_pool], dtype=np.int64
        )
        features = np.stack(
            [
                pair_feature(
                    current,
                    previous,
                    int(i),
                    int(j),
                    projected,
                    motion,
                    current_token,
                    previous_token,
                )
                for i, j in combinations
            ]
        )
        incumbent_pos = int(
            np.flatnonzero(
                (combinations[:, 0] == current_baseline)
                & (combinations[:, 1] == previous_baseline)
            )[0]
        )
        incumbent_current = current["candidate_subscores"][current_baseline]
        incumbent_previous = previous["candidate_subscores"][previous_baseline]
        eligible = np.zeros(len(combinations), dtype=np.bool_)
        safety_ratio = np.zeros(len(combinations), dtype=np.float64)
        for row, (i, j) in enumerate(combinations):
            ratio_current = current["candidate_subscores"][i, SAFETY] / np.maximum(
                incumbent_current[SAFETY], 1.0e-6
            )
            ratio_previous = previous["candidate_subscores"][j, SAFETY] / np.maximum(
                incumbent_previous[SAFETY], 1.0e-6
            )
            safety_ratio[row] = min(float(ratio_current.min()), float(ratio_previous.min()))
            eligible[row] = (
                safety_ratio[row] >= 0.999
                and current["candidate_subscores"][i, 4] >= incumbent_current[4] - 0.001
                and previous["candidate_subscores"][j, 4] >= incumbent_previous[4] - 0.001
                and current["pdm_score"][i] >= current["pdm_score"][current_baseline] - 0.005
                and previous["pdm_score"][j] >= previous["pdm_score"][previous_baseline] - 0.005
            )
        feature_blocks.append(features)
        pending.append(
            {
                "current_token": current_token,
                "previous_token": previous_token,
                "current_baseline": current_baseline,
                "previous_baseline": previous_baseline,
                "combinations": combinations,
                "incumbent_pos": incumbent_pos,
                "eligible": eligible,
                "safety_ratio": safety_ratio,
            }
        )
        if number % 500 == 0:
            print(f"pairs={number}/{len(pairs)}", flush=True)

    all_features = np.concatenate(feature_blocks, axis=0)
    all_probability = np.mean(
        [model.predict_proba(all_features)[:, 1] for model in artifact["models"]],
        axis=0,
    )
    proposals = []
    offset = 0
    for item, features in zip(pending, feature_blocks):
        count = len(features)
        probability = all_probability[offset : offset + count]
        offset += count
        incumbent_pos = int(item["incumbent_pos"])
        incumbent_probability = float(probability[incumbent_pos])
        gain = probability - incumbent_probability
        objective = np.where(
            item["eligible"] & (gain >= threshold), probability, -np.inf
        )
        chosen = int(np.argmax(objective))
        if not np.isfinite(objective[chosen]) or chosen == incumbent_pos:
            continue
        current_token = str(item["current_token"])
        previous_token = str(item["previous_token"])
        current_index, previous_index = (
            int(value) for value in item["combinations"][chosen]
        )
        current = records[current_token]
        previous = records[previous_token]
        proposals.append(
            {
                "current_token": current_token,
                "previous_token": previous_token,
                "current_candidate": current_index,
                "previous_candidate": previous_index,
                "probability_gain": float(gain[chosen]),
                "predicted_probability": float(probability[chosen]),
                "incumbent_probability": incumbent_probability,
                "minimum_safety_ratio": float(item["safety_ratio"][chosen]),
                "current_pdm_delta": float(
                    current["pdm_score"][current_index]
                    - current["pdm_score"][item["current_baseline"]]
                ),
                "previous_pdm_delta": float(
                    previous["pdm_score"][previous_index]
                    - previous["pdm_score"][item["previous_baseline"]]
                ),
            }
        )

    proposals.sort(key=lambda item: -item["probability_gain"])
    limit = max(1, int(np.ceil(len(pairs) * fraction)))
    accepted = proposals[:limit]
    payload = {
        token: {"trajectory": Trajectory(poses=np.asarray(record["trajectory"], dtype=np.float32))}
        for token, record in records.items()
    }
    changed: set[str] = set()
    for item in accepted:
        for side in ("current", "previous"):
            token = str(item[f"{side}_token"])
            candidate = int(item[f"{side}_candidate"])
            baseline = int(np.argmax(records[token]["pdm_score"]))
            if candidate == baseline:
                continue
            payload[token] = {
                "trajectory": Trajectory(
                    poses=np.asarray(records[token]["refined_proposals"][candidate], dtype=np.float32)
                )
            }
            changed.add(token)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle(payload, args.output)
    report = {
        "schema": "actworld-navhard-temporal-fcts-pair-head-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_split": "navtrain",
        "navtest_used_for_training": False,
        "method": (
            "ActWorld Planning-JEPA latent agreement/discrepancy plus plan-pair "
            "kinematics and frozen FCTS safety/progress guards"
        ),
        "artifact": str(args.artifact),
        "artifact_sha256": sha256(args.artifact),
        "threshold": threshold,
        "fraction": fraction,
        "pairs": len(pairs),
        "eligible_pairs": len(proposals),
        "accepted_pairs": len(accepted),
        "changed_tokens": len(changed),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "accepted": accepted,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("pairs", "eligible_pairs", "accepted_pairs", "changed_tokens")}, indent=2))


if __name__ == "__main__":
    main()
