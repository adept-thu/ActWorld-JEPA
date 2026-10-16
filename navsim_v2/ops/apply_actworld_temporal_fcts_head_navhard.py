#!/usr/bin/env python3
"""Apply a NAVTRAIN-fitted temporal Planning-JEPA/FCTS head to NAVHARD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from build_actworld_temporal_fcts_dataset import _plan_tokens, _variant_strength
from build_v2_temporal_anchor_variants import blend, shift_previous
from build_v2_temporal_projection_variants import temporal_projection
from fit_actworld_temporal_fcts_head import Policy, TemporalFCTSHead, _choose
from navsim.common.dataclasses import Trajectory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _selected_index(payload: dict[str, np.ndarray]) -> int:
    if "candidate_selector_choice" in payload:
        return int(np.asarray(payload["candidate_selector_choice"]).item())
    return int(np.argmax(np.asarray(payload["pdm_score"])))


def _poses(value: object) -> np.ndarray:
    if isinstance(value, dict):
        value = value["trajectory"]
    if isinstance(value, Trajectory):
        return np.asarray(value.poses, dtype=np.float32)
    if hasattr(value, "poses"):
        return np.asarray(value.poses, dtype=np.float32)
    raise TypeError(f"unsupported trajectory payload {type(value)!r}")


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
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--mapping-yaml", required=True, type=Path)
    parser.add_argument("--base-trajectories", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected-tokens", type=int, default=5912)
    parser.add_argument("--max-variant", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--policy-margin", type=float)
    parser.add_argument("--policy-safety-threshold", type=float)
    parser.add_argument("--policy-progress-threshold", type=float)
    parser.add_argument("--policy-component-margin", type=float)
    parser.add_argument("--policy-hard-metric-threshold", type=float)
    parser.add_argument("--policy-quality-margin-threshold", type=float)
    parser.add_argument("--target-fraction", type=float)
    args = parser.parse_args()

    paths = sorted(args.diagnostics.glob("*.npz"))
    if len(paths) != args.expected_tokens:
        raise RuntimeError(f"diagnostic coverage differs: {len(paths)} != {args.expected_tokens}")
    records = {path.stem: _load_npz(path) for path in paths}
    with args.base_trajectories.open("rb") as stream:
        base_payload = pickle.load(stream)
    if set(base_payload) != set(records):
        raise RuntimeError("base trajectory and diagnostic token sets differ")

    artifact = torch.load(args.artifact, map_location="cpu")
    if artifact.get("format") not in {
        "actworld_v2_navtrain_temporal_fcts_head_v1",
        "actworld_v2_navtrain_temporal_fcts_head_v2",
        "actworld_v2_navtrain_temporal_fcts_head_v3",
    }:
        raise ValueError("unsupported temporal FCTS artifact")
    if artifact.get("split") != "navtrain" or artifact.get("navhard_used_for_training_or_selection") is not False:
        raise ValueError("temporal FCTS artifact is not NAVTRAIN-only")
    if artifact.get("policy") is None or artifact.get("target_fraction") is None:
        raise ValueError("temporal FCTS artifact has no accepted NAVTRAIN policy")
    state = artifact["model_state"]
    hard_metric_count = (
        int(state["hard_metric_head.weight"].shape[0])
        if "hard_metric_head.weight" in state
        else 0
    )
    quality_metric_count = (
        int(state["quality_margin_head.weight"].shape[0])
        if "quality_margin_head.weight" in state
        else 0
    )
    model = TemporalFCTSHead(
        state["plan_mean"].numpy(),
        state["plan_std"].numpy(),
        state["future_mean"].numpy(),
        state["future_std"].numpy(),
        hard_metric_count=hard_metric_count,
        quality_metric_count=quality_metric_count,
    )
    model.load_state_dict(state)
    model.eval()
    artifact_policy = Policy(**artifact["policy"])
    policy = Policy(
        margin=artifact_policy.margin if args.policy_margin is None else args.policy_margin,
        safety_threshold=(
            artifact_policy.safety_threshold
            if args.policy_safety_threshold is None
            else args.policy_safety_threshold
        ),
        progress_threshold=(
            artifact_policy.progress_threshold
            if args.policy_progress_threshold is None
            else args.policy_progress_threshold
        ),
        component_margin=(
            artifact_policy.component_margin
            if args.policy_component_margin is None
            else args.policy_component_margin
        ),
        hard_metric_threshold=(
            artifact_policy.hard_metric_threshold
            if args.policy_hard_metric_threshold is None
            else args.policy_hard_metric_threshold
        ),
        quality_margin_threshold=(
            artifact_policy.quality_margin_threshold
            if args.policy_quality_margin_threshold is None
            else args.policy_quality_margin_threshold
        ),
    )
    target_fraction = (
        float(artifact["target_fraction"])
        if args.target_fraction is None
        else args.target_fraction
    )
    names = [str(value) for value in artifact["variant_names"]]
    if not names or names[0] != "selected":
        raise ValueError(f"artifact variants must begin with selected, got {names}")
    weights = np.asarray([_variant_strength(name) for name in names], dtype=np.float32)
    families = {
        "anchor" if name.startswith("anchor") else "project"
        if name.startswith("project")
        else "selected"
        for name in names
    }
    families.discard("selected")
    if len(families) != 1:
        raise ValueError(f"artifact mixes unsupported temporal families: {names}")
    family = next(iter(families))
    if args.max_variant >= len(names):
        raise ValueError(
            f"max variant {args.max_variant} exceeds artifact variants {names}"
        )

    mapping = yaml.safe_load(args.mapping_yaml.read_text())
    pairs = [tuple(pair) for _, _, group_pairs in mapping["reactive_all_mapping"] for pair in group_pairs]
    degree: dict[str, int] = {}
    for pair in pairs:
        for token in pair:
            degree[token] = degree.get(token, 0) + 1
    if max(degree.values()) != 1:
        raise RuntimeError("NAVHARD temporal-FCTS deployment requires disjoint pairs")

    previous_plans = []
    current_plans = []
    previous_futures = []
    current_futures = []
    eligible_pairs = []
    skipped_off_support = 0
    for current_token, previous_token in pairs:
        current = records[current_token]
        previous = records[previous_token]
        current_base = _poses(base_payload[current_token])
        previous_base = _poses(base_payload[previous_token])
        if (
            np.mean(np.square(current_base - np.asarray(current["trajectory"], dtype=np.float32))) > 1e-10
            or np.mean(np.square(previous_base - np.asarray(previous["trajectory"], dtype=np.float32))) > 1e-10
        ):
            skipped_off_support += 1
            continue
        anchor = shift_previous(previous_base)
        target = (
            anchor
            if family == "anchor"
            else temporal_projection(current_base, anchor)
        )
        variants = np.stack(
            [
                current_base,
                *(blend(current_base, target, float(weight)) for weight in weights[1:]),
            ]
        )
        previous_index = _selected_index(previous)
        current_index = _selected_index(current)
        previous_world = np.asarray(previous["world_features"][previous_index], dtype=np.float32)
        current_world = np.asarray(current["world_features"][current_index], dtype=np.float32)
        futures = (1.0 - weights[:, None]) * current_world + weights[:, None] * previous_world
        previous_plans.append(_plan_tokens(previous_base))
        current_plans.append(_plan_tokens(variants))
        previous_futures.append(previous_world)
        current_futures.append(futures.astype(np.float32))
        eligible_pairs.append((current_token, previous_token, variants))

    with torch.no_grad():
        prediction_t = model(
            torch.as_tensor(np.stack(previous_plans), dtype=torch.float32),
            torch.as_tensor(np.stack(current_plans), dtype=torch.float32),
            torch.as_tensor(np.stack(previous_futures), dtype=torch.float32),
            torch.as_tensor(np.stack(current_futures), dtype=torch.float32),
        )
    prediction = {name: value.numpy() for name, value in prediction_t.items()}
    # Disable stronger anchors before selection. This makes it possible to
    # deploy a conservative policy selected entirely from NAVTRAIN OOF folds.
    prediction["refinement_gain"][:, args.max_variant + 1 :] = -np.inf
    selected = _choose(prediction, policy)
    rows = np.arange(len(selected))
    advantage = prediction["refinement_gain"][rows, selected] - prediction["refinement_gain"][:, 0]
    changed_rows = np.flatnonzero(selected != 0)
    limit = int(np.ceil(len(pairs) * target_fraction))
    if len(changed_rows) > limit:
        order = changed_rows[np.argsort(-advantage[changed_rows], kind="stable")]
        selected[order[limit:]] = 0

    output_payload = dict(base_payload)
    changes = []
    for row, choice in enumerate(selected):
        if choice == 0:
            continue
        current_token, previous_token, variants = eligible_pairs[row]
        output_payload[current_token] = {"trajectory": Trajectory(poses=np.asarray(variants[choice], dtype=np.float32))}
        changes.append(
            {
                "current_token": current_token,
                "previous_token": previous_token,
                "variant": names[int(choice)],
                "predicted_gain": float(advantage[row]),
                "predicted_component_ratio": prediction["component_ratio"][row, choice].tolist(),
                "predicted_safety": float(1.0 / (1.0 + np.exp(-prediction["safety_logit"][row, choice]))),
                "predicted_progress_ratio": float(prediction["progress_ratio"][row, choice]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_pickle(output_payload, args.output)
    report = {
        "schema": "actworld-navhard-temporal-planning-jepa-fcts-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_split": "navtrain",
        "navhard_used_for_training_or_selection": False,
        "artifact": str(args.artifact),
        "artifact_sha256": _sha256(args.artifact),
        "base_trajectories": str(args.base_trajectories),
        "base_sha256": _sha256(args.base_trajectories),
        "policy": asdict(policy),
        "artifact_default_policy": artifact["policy"],
        "target_fraction": target_fraction,
        "max_variant": args.max_variant,
        "pairs": len(pairs),
        "eligible_pairs": len(eligible_pairs),
        "skipped_off_support": skipped_off_support,
        "changed_tokens": len(changes),
        "variant_histogram": {name: sum(item["variant"] == name for item in changes) for name in names[1:]},
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "changes": changes,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("pairs", "eligible_pairs", "skipped_off_support", "changed_tokens", "variant_histogram")}, indent=2))


if __name__ == "__main__":
    main()
