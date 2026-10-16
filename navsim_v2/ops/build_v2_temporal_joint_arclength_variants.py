#!/usr/bin/env python3
"""Build symmetric temporal variants by aligning progress along each plan path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_v2_temporal_anchor_variants import angle_delta, shift_previous
from build_v2_temporal_joint_projection_variants import joint_projection, unshift_to_previous


STRENGTHS = (0.125, 0.25, 0.375, 0.50)


def _cumulative_distance(trajectory: np.ndarray) -> np.ndarray:
    xy = np.vstack((np.zeros((1, 2), dtype=np.float64), trajectory[:, :2]))
    return np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))


def _interp_extrapolate(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    result = np.interp(x, xp, fp)
    if len(xp) >= 2:
        left = x < xp[0]
        right = x > xp[-1]
        left_slope = (fp[1] - fp[0]) / max(xp[1] - xp[0], 1e-6)
        right_slope = (fp[-1] - fp[-2]) / max(xp[-1] - xp[-2], 1e-6)
        result[left] = fp[0] + (x[left] - xp[0]) * left_slope
        result[right] = fp[-1] + (x[right] - xp[-1]) * right_slope
    return result


def _resample_arclength(trajectory: np.ndarray, desired: np.ndarray) -> np.ndarray:
    value = np.asarray(trajectory, dtype=np.float64)
    original = _cumulative_distance(value)
    desired = np.maximum.accumulate(np.maximum(np.asarray(desired, dtype=np.float64), 1e-4))
    source_distance = np.concatenate(([0.0], original))
    source_xy = np.vstack((np.zeros((1, 2), dtype=np.float64), value[:, :2]))
    source_yaw = np.unwrap(np.concatenate(([0.0], value[:, 2])))
    result = value.copy()
    result[:, 0] = _interp_extrapolate(desired, source_distance, source_xy[:, 0])
    result[:, 1] = _interp_extrapolate(desired, source_distance, source_xy[:, 1])
    result[:, 2] = _interp_extrapolate(desired, source_distance, source_yaw)
    result[:, 2] = np.arctan2(np.sin(result[:, 2]), np.cos(result[:, 2]))
    return result.astype(np.float32)


def joint_arclength_projection(
    previous: np.ndarray,
    current: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = shift_previous(previous)
    previous_progress = _cumulative_distance(anchor)
    current_progress = _cumulative_distance(current)
    alpha = np.linspace(0.0, 1.0, len(current_progress), dtype=np.float64)
    residual_start = float(current_progress[0] - previous_progress[0])
    residual_end = float(current_progress[-1] - previous_progress[-1])
    affine_residual = (1.0 - alpha) * residual_start + alpha * residual_end
    projected_progress = previous_progress + affine_residual
    current_desired = current_progress + strength * (projected_progress - current_progress)
    previous_desired = previous_progress + strength * (current_progress - projected_progress)
    current_variant = _resample_arclength(current, current_desired)
    previous_anchor = _resample_arclength(anchor, previous_desired)
    return unshift_to_previous(previous_anchor, previous), current_variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--metadata-states", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs-output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.pairs_output.exists():
        raise FileExistsError(args.output if args.output.exists() else args.pairs_output)

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
        ["selected", "joint250", *(f"arc{int(round(value * 1000)):03d}" for value in STRENGTHS)]
    )
    variants = {
        token: [trajectory.copy() for _ in range(len(names))]
        for token, trajectory in trajectories.items()
    }
    for previous_token, current_token, _, _ in pairs:
        previous = trajectories[previous_token]
        current = trajectories[current_token]
        previous_joint, current_joint = joint_projection(previous, current, 0.25)
        variants[previous_token][1] = previous_joint
        variants[current_token][1] = current_joint
        for variant_index, strength in enumerate(STRENGTHS, start=2):
            previous_variant, current_variant = joint_arclength_projection(
                previous, current, strength
            )
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
    )
    print(f"tokens={len(trajectories)} disjoint_pairs={len(pairs)} variants={names.tolist()}")


if __name__ == "__main__":
    main()
