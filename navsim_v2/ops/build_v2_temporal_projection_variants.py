#!/usr/bin/env python3
"""Build endpoint-preserving temporal projection variants on NAVTRAIN.

The projection expresses the current plan as a shifted previous plan plus an
affine residual.  An affine residual has zero second and third finite
difference, so it directly targets NAVSIM v2's two-frame acceleration/jerk
consistency while preserving the current plan's first and terminal poses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_v2_temporal_anchor_variants import angle_delta, blend, shift_previous


def temporal_projection(current: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Project ``current`` onto an affine residual around ``anchor``.

    Both plans are represented in the current ego frame.  The residual is
    linearly interpolated between the current plan's first and last residuals,
    which preserves both endpoints and removes high-frequency temporal drift.
    """

    current64 = np.asarray(current, dtype=np.float64)
    anchor64 = np.asarray(anchor, dtype=np.float64)
    if current64.shape != (8, 3) or anchor64.shape != (8, 3):
        raise ValueError(
            f"expected two [8,3] trajectories, got {current64.shape} and {anchor64.shape}"
        )

    alpha = np.linspace(0.0, 1.0, len(current64), dtype=np.float64)
    residual_start = current64[0, :2] - anchor64[0, :2]
    residual_end = current64[-1, :2] - anchor64[-1, :2]
    residual_xy = (
        (1.0 - alpha[:, None]) * residual_start[None]
        + alpha[:, None] * residual_end[None]
    )

    yaw_start = float(angle_delta(current64[0, 2], anchor64[0, 2]))
    yaw_end = float(angle_delta(current64[-1, 2], anchor64[-1, 2]))
    # Unwrap the terminal residual relative to the initial residual before
    # interpolation so the bridge never takes the long way around the circle.
    yaw_end = yaw_start + float(angle_delta(yaw_end, yaw_start))
    residual_yaw = (1.0 - alpha) * yaw_start + alpha * yaw_end

    result = np.empty_like(current64)
    result[:, :2] = anchor64[:, :2] + residual_xy
    result[:, 2] = anchor64[:, 2] + residual_yaw
    result[:, 2] = np.arctan2(np.sin(result[:, 2]), np.cos(result[:, 2]))

    # These invariants are the safety/progress motivation for this refiner.
    if not np.allclose(result[[0, -1], :2], current64[[0, -1], :2], atol=1e-6):
        raise RuntimeError("temporal projection failed to preserve XY endpoints")
    if not np.allclose(
        angle_delta(result[[0, -1], 2], current64[[0, -1], 2]),
        0.0,
        atol=1e-6,
    ):
        raise RuntimeError("temporal projection failed to preserve yaw endpoints")
    return result.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--metadata-states", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--strengths", nargs="+", type=float, default=(0.25, 0.50, 0.75, 1.0)
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(not 0.0 < value <= 1.0 for value in args.strengths):
        raise ValueError("projection strengths must be in (0, 1]")

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
        metadata_path = metadata_paths.get(path.name)
        if metadata_path is None:
            raise FileNotFoundError(f"missing metadata state for {path.name}")
        with np.load(metadata_path, allow_pickle=False) as payload:
            metadata.append(
                (path.stem, str(payload["log_name"]), float(payload["start_time"]))
            )

    previous: dict[str, str | None] = {token: None for token in trajectories}
    by_log: dict[str, list[tuple[str, float]]] = {}
    for token, log_name, start_time in metadata:
        by_log.setdefault(log_name, []).append((token, start_time))
    for values in by_log.values():
        ordered = sorted(values, key=lambda item: item[1])
        for earlier, current in zip(ordered, ordered[1:]):
            if 0.0 < current[1] - earlier[1] < 0.55:
                previous[current[0]] = earlier[0]

    names = np.asarray(
        ["selected", *(f"project{int(round(value * 100)):03d}" for value in args.strengths)]
    )
    args.output.mkdir(parents=True)
    paired = 0
    for number, token in enumerate(sorted(trajectories), start=1):
        current = trajectories[token]
        previous_token = previous[token]
        if previous_token is None:
            variants = [current.copy() for _ in range(len(names))]
        else:
            anchor = shift_previous(trajectories[previous_token])
            projected = temporal_projection(current, anchor)
            variants = [
                current,
                *(blend(current, projected, strength) for strength in args.strengths),
            ]
            paired += 1
        np.savez_compressed(
            args.output / f"{token}.npz",
            custom_names=names,
            custom_trajectories=np.stack(variants),
        )
        if number % 400 == 0:
            print(f"written={number}/{len(trajectories)}", flush=True)
    print(f"tokens={len(trajectories)} paired={paired} variants={names.tolist()}")


if __name__ == "__main__":
    main()
