#!/usr/bin/env python3
"""Build dense 0.1 s temporal-rollout variants for NAVTRAIN.

ActWorld normally exports eight 0.5 s waypoints.  NAVSIM also accepts forty
0.1 s poses, which lets the planner make its intended velocity/acceleration
profile explicit instead of delegating it to evaluator interpolation.  The
dense transform is endpoint preserving and mirrors the local zero-phase
smoother used by the high-EC reference planner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_v2_temporal_joint_projection_variants import joint_projection


NAMES = np.asarray(
    (
        "selected_linear",
        "joint250_linear",
        "selected_smooth100",
        "joint250_smooth025",
        "joint250_dense_smooth050",
        "joint250_dense_smooth100",
        "joint250_dense_slow995",
        "joint250_dense_slow990",
    )
)


def _wrap(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def densify(poses: np.ndarray, slowdown: float = 1.0, smooth_strength: float = 0.0) -> np.ndarray:
    """Convert eight 0.5 s poses into an explicit forty-pose 0.1 s rollout."""

    value = np.asarray(poses, dtype=np.float64)
    if value.shape != (8, 3):
        raise ValueError(f"expected [8,3] poses, got {value.shape}")
    source_time = np.arange(9, dtype=np.float64) * 0.5
    target_time = np.arange(1, 41, dtype=np.float64) * 0.1 * slowdown
    extended = np.vstack((np.zeros((1, 3), dtype=np.float64), value))
    dense = np.empty((40, 3), dtype=np.float64)
    dense[:, 0] = np.interp(target_time, source_time, extended[:, 0])
    dense[:, 1] = np.interp(target_time, source_time, extended[:, 1])
    dense[:, 2] = np.interp(target_time, source_time, np.unwrap(extended[:, 2]))
    if smooth_strength:
        positions = np.vstack((np.zeros((1, 2)), dense[:, :2]))
        positions = np.vstack((positions, 2.0 * positions[-1] - positions[-2]))
        low_xy = 0.25 * positions[:-2] + 0.50 * positions[1:-1] + 0.25 * positions[2:]
        dense[:, :2] += smooth_strength * (low_xy - dense[:, :2])

        headings = np.unwrap(np.concatenate(([0.0], dense[:, 2])))
        headings = np.concatenate((headings, [2.0 * headings[-1] - headings[-2]]))
        low_yaw = 0.25 * headings[:-2] + 0.50 * headings[1:-1] + 0.25 * headings[2:]
        dense[:, 2] += smooth_strength * (low_yaw - dense[:, 2])
    dense[:, 2] = _wrap(dense[:, 2])
    return dense.astype(np.float32)


def _metadata(
    diagnostics: Path, metadata_states: list[Path]
) -> tuple[dict[str, np.ndarray], list[tuple[str, str, float]]]:
    state_paths = {
        path.name: path for directory in metadata_states for path in directory.glob("*.npz")
    }
    trajectories = {}
    metadata = []
    for path in sorted(diagnostics.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            trajectories[path.stem] = np.asarray(payload["trajectory"], dtype=np.float32)
        state_path = state_paths.get(path.name)
        if state_path is None:
            raise FileNotFoundError(f"missing metadata state for {path.name}")
        with np.load(state_path, allow_pickle=False) as payload:
            metadata.append((path.stem, str(payload["log_name"]), float(payload["start_time"])))
    return trajectories, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--metadata-states", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs-output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.pairs_output.exists():
        raise FileExistsError(args.output if args.output.exists() else args.pairs_output)

    trajectories, metadata = _metadata(args.diagnostics, args.metadata_states)
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

    variants: dict[str, list[np.ndarray]] = {}
    for token, trajectory in trajectories.items():
        variants[token] = [
            densify(trajectory),
            densify(trajectory),
            densify(trajectory, smooth_strength=1.00),
            densify(trajectory, smooth_strength=0.25),
            densify(trajectory, smooth_strength=0.50),
            densify(trajectory, smooth_strength=1.00),
            densify(trajectory, slowdown=0.995, smooth_strength=1.00),
            densify(trajectory, slowdown=0.990, smooth_strength=1.00),
        ]
    for previous_token, current_token, _, _ in pairs:
        previous_joint, current_joint = joint_projection(
            trajectories[previous_token], trajectories[current_token], 0.25
        )
        variants[previous_token][1] = densify(previous_joint)
        variants[current_token][1] = densify(current_joint)
        for token, joint in ((previous_token, previous_joint), (current_token, current_joint)):
            variants[token][3] = densify(joint, smooth_strength=0.25)
            variants[token][4] = densify(joint, smooth_strength=0.50)
            variants[token][5] = densify(joint, smooth_strength=1.00)
            variants[token][6] = densify(joint, slowdown=0.995, smooth_strength=1.00)
            variants[token][7] = densify(joint, slowdown=0.990, smooth_strength=1.00)

    args.output.mkdir(parents=True)
    for number, token in enumerate(sorted(trajectories), start=1):
        np.savez_compressed(
            args.output / f"{token}.npz",
            custom_names=NAMES,
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
        variant_names=NAMES,
    )
    print(f"tokens={len(trajectories)} disjoint_pairs={len(pairs)} variants={NAMES.tolist()}")


if __name__ == "__main__":
    main()
