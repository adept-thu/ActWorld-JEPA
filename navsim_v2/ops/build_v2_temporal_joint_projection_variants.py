#!/usr/bin/env python3
"""Build disjoint, symmetric temporal-projection variants on NAVTRAIN.

Unlike the current-only refiner, this construction distributes the temporal
correction over both members of a disjoint adjacent-frame pair.  A joint
strength of 0.25 produces the same relative overlap correction as a 0.50
current-only projection while moving each individual plan only half as far.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_v2_temporal_anchor_variants import angle_delta, blend, shift_previous
from build_v2_temporal_projection_variants import temporal_projection


def unshift_to_previous(anchor: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Put a next-ego-frame anchor back into the previous plan coordinates."""

    anchor64 = np.asarray(anchor, dtype=np.float64)
    previous64 = np.asarray(previous, dtype=np.float64)
    if anchor64.shape != (8, 3) or previous64.shape != (8, 3):
        raise ValueError("anchor and previous trajectory must both have shape [8,3]")
    origin_xy = previous64[0, :2]
    origin_heading = float(previous64[0, 2])
    cosine = np.cos(origin_heading)
    sine = np.sin(origin_heading)
    inverse_rotation = np.asarray(((cosine, sine), (-sine, cosine)))

    result = previous64.copy()
    result[1:, :2] = anchor64[:7, :2] @ inverse_rotation + origin_xy
    result[1:, 2] = origin_heading + anchor64[:7, 2]
    result[1:, 2] = np.arctan2(np.sin(result[1:, 2]), np.cos(result[1:, 2]))
    return result.astype(np.float32)


def joint_projection(
    previous: np.ndarray,
    current: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return symmetric previous/current plans at one joint strength."""

    anchor = shift_previous(previous)
    projected = temporal_projection(current, anchor)
    current_variant = blend(current, projected, strength)

    anchor_target = anchor.astype(np.float64, copy=True)
    anchor_target[:, :2] += np.asarray(current[:, :2], dtype=np.float64) - projected[:, :2]
    anchor_target[:, 2] += angle_delta(current[:, 2], projected[:, 2])
    anchor_target[:, 2] = np.arctan2(
        np.sin(anchor_target[:, 2]), np.cos(anchor_target[:, 2])
    )
    previous_anchor = blend(anchor, anchor_target, strength)
    previous_variant = unshift_to_previous(previous_anchor, previous)
    return previous_variant, current_variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--metadata-states", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs-output", required=True, type=Path)
    parser.add_argument("--strengths", nargs="+", type=float, default=(0.125, 0.25, 0.375, 0.50))
    args = parser.parse_args()
    if args.output.exists() or args.pairs_output.exists():
        raise FileExistsError(args.output if args.output.exists() else args.pairs_output)
    if any(not 0.0 < value <= 0.5 for value in args.strengths):
        raise ValueError("joint strengths must be in (0, 0.5]")

    paths = sorted(args.diagnostics.glob("*.npz"))
    trajectories: dict[str, np.ndarray] = {}
    metadata: list[tuple[str, str, float]] = []
    metadata_paths = {
        path.name: path
        for directory in args.metadata_states
        for path in directory.glob("*.npz")
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            trajectories[path.stem] = np.asarray(payload["trajectory"], dtype=np.float32)
        state_path = metadata_paths.get(path.name)
        if state_path is None:
            raise FileNotFoundError(f"missing metadata state for {path.name}")
        with np.load(state_path, allow_pickle=False) as payload:
            metadata.append(
                (path.stem, str(payload["log_name"]), float(payload["start_time"]))
            )

    by_log: dict[str, list[tuple[str, float]]] = {}
    for token, log_name, start_time in metadata:
        by_log.setdefault(log_name, []).append((token, start_time))
    pairs: list[tuple[str, str, str, float]] = []
    for log_name, values in by_log.items():
        ordered = sorted(values, key=lambda item: item[1])
        index = 0
        while index + 1 < len(ordered):
            previous, current = ordered[index], ordered[index + 1]
            interval = float(current[1] - previous[1])
            if 0.0 < interval < 0.55:
                pairs.append((previous[0], current[0], log_name, interval))
                index += 2
            else:
                index += 1

    names = np.asarray(
        ["selected", *(f"joint{int(round(value * 1000)):03d}" for value in args.strengths)]
    )
    variants = {
        token: [trajectory.copy() for _ in range(len(names))]
        for token, trajectory in trajectories.items()
    }
    for previous_token, current_token, _, _ in pairs:
        previous = trajectories[previous_token]
        current = trajectories[current_token]
        for variant_index, strength in enumerate(args.strengths, start=1):
            previous_variant, current_variant = joint_projection(previous, current, strength)
            variants[previous_token][variant_index] = previous_variant
            variants[current_token][variant_index] = current_variant

    args.output.mkdir(parents=True)
    for number, token in enumerate(sorted(trajectories), start=1):
        np.savez_compressed(
            args.output / f"{token}.npz",
            custom_names=names,
            custom_trajectories=np.stack(variants[token]),
        )
        if number % 400 == 0:
            print(f"written={number}/{len(trajectories)}", flush=True)
    np.savez_compressed(
        args.pairs_output,
        previous_tokens=np.asarray([pair[0] for pair in pairs]),
        current_tokens=np.asarray([pair[1] for pair in pairs]),
        log_names=np.asarray([pair[2] for pair in pairs]),
        intervals=np.asarray([pair[3] for pair in pairs], dtype=np.float32),
        variant_names=names,
        strengths=np.asarray(args.strengths, dtype=np.float32),
    )
    print(f"tokens={len(trajectories)} disjoint_pairs={len(pairs)} variants={names.tolist()}")


if __name__ == "__main__":
    main()
