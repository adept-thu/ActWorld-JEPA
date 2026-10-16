#!/usr/bin/env python3
"""Build causal previous-plan anchor variants for NAVTRAIN screening."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def angle_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(first - second), np.cos(first - second))


def shift_previous(previous: np.ndarray) -> np.ndarray:
    """Express the remaining previous plan in its predicted next ego frame."""

    value = np.asarray(previous, dtype=np.float64)
    if value.shape != (8, 3):
        raise ValueError(f"unexpected trajectory shape {value.shape}")
    origin_xy = value[0, :2]
    origin_heading = float(value[0, 2])
    cosine = np.cos(origin_heading)
    sine = np.sin(origin_heading)
    inverse_rotation = np.asarray(((cosine, sine), (-sine, cosine)))
    remaining = value[1:]
    result = np.zeros_like(value)
    result[:7, :2] = (remaining[:, :2] - origin_xy) @ inverse_rotation.T
    result[:7, 2] = angle_delta(remaining[:, 2], origin_heading)
    result[7, :2] = result[6, :2] + (result[6, :2] - result[5, :2])
    result[7, 2] = result[6, 2] + float(angle_delta(result[6, 2], result[5, 2]))
    result[:, 2] = np.arctan2(np.sin(result[:, 2]), np.cos(result[:, 2]))
    return result.astype(np.float32)


def blend(current: np.ndarray, anchor: np.ndarray, weight: float) -> np.ndarray:
    result = np.asarray(current, dtype=np.float64).copy()
    result[:, :2] = (1.0 - weight) * current[:, :2] + weight * anchor[:, :2]
    result[:, 2] = np.arctan2(
        (1.0 - weight) * np.sin(current[:, 2]) + weight * np.sin(anchor[:, 2]),
        (1.0 - weight) * np.cos(current[:, 2]) + weight * np.cos(anchor[:, 2]),
    )
    return result.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--metadata-states", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--weights", nargs="+", type=float, default=(0.05, 0.10, 0.20, 0.35, 0.50)
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(not 0.0 < value <= 1.0 for value in args.weights):
        raise ValueError("anchor weights must be in (0, 1]")

    paths = sorted(args.diagnostics.glob("*.npz"))
    trajectories = {}
    metadata = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            trajectories[path.stem] = np.asarray(payload["trajectory"], dtype=np.float32)
        with np.load(args.metadata_states / path.name, allow_pickle=False) as payload:
            metadata.append(
                (path.stem, str(payload["log_name"]), float(payload["start_time"]))
            )

    previous = {token: None for token in trajectories}
    by_log: dict[str, list[tuple[str, float]]] = {}
    for token, log_name, start_time in metadata:
        by_log.setdefault(log_name, []).append((token, start_time))
    for values in by_log.values():
        ordered = sorted(values, key=lambda item: item[1])
        for earlier, current in zip(ordered, ordered[1:]):
            if 0.0 < current[1] - earlier[1] < 0.55:
                previous[current[0]] = earlier[0]

    names = np.asarray(["selected", *(f"anchor{int(round(value * 100)):02d}" for value in args.weights)])
    args.output.mkdir(parents=True)
    paired = 0
    for number, token in enumerate(sorted(trajectories), start=1):
        current = trajectories[token]
        previous_token = previous[token]
        if previous_token is None:
            variants = [current.copy() for _ in range(len(names))]
        else:
            anchor = shift_previous(trajectories[previous_token])
            variants = [current, *(blend(current, anchor, weight) for weight in args.weights)]
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
