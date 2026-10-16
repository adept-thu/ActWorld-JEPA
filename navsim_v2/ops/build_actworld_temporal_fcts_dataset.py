#!/usr/bin/env python3
"""Build NAVTRAIN-only paired supervision for temporal Planning-JEPA/FCTS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    acceleration_threshold,
    calculate_rms,
    extract_features,
    jerk_threshold,
    yaw_accel_threshold,
    yaw_rate_threshold,
)


THRESHOLDS = np.asarray(
    (acceleration_threshold, jerk_threshold, yaw_rate_threshold, yaw_accel_threshold),
    dtype=np.float32,
)


def _variant_strength(name: str) -> float:
    """Return the causal temporal-refinement strength encoded in a name."""

    if name == "selected":
        return 0.0
    if name.startswith("anchor"):
        return float(name.removeprefix("anchor")) / 100.0
    if name.startswith("project"):
        return float(name.removeprefix("project")) / 100.0
    if name.startswith("power"):
        power = float(name.removeprefix("power")) / 100.0
        return 1.0 - 0.5**power
    if name == "overlap7":
        return 1.0
    if name.startswith("channel_x"):
        parts = name.removeprefix("channel_").split("_")
        values = [float(part[1:]) / 100.0 for part in parts]
        return float(np.mean(values))
    raise ValueError(f"unsupported temporal variant name {name!r}")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _selected_index(payload: dict[str, np.ndarray]) -> int:
    if "candidate_selector_choice" in payload:
        return int(np.asarray(payload["candidate_selector_choice"]).item())
    return int(np.argmax(np.asarray(payload["pdm_score"])))


def _parse_array(value: object) -> np.ndarray:
    return np.fromstring(str(value).strip().strip("[]"), sep=" ")


def _row_score(row: pd.Series, ec: float) -> float:
    soft = _parse_array(row["weighted_metrics"]).copy()
    weights = _parse_array(row["weighted_metrics_array"]).copy()
    soft[4] = ec
    return float(
        float(row["multiplicative_metrics_prod"])
        * np.dot(soft, weights)
        / weights.sum()
    )


def _angle_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(first - second), np.cos(first - second))


def _plan_tokens(poses: np.ndarray) -> np.ndarray:
    """Match the seven action channels used by ActWorldPlanner._action_features."""

    value = np.asarray(poses, dtype=np.float32)
    previous_xy = np.concatenate((np.zeros_like(value[..., :1, :2]), value[..., :-1, :2]), axis=-2)
    previous_yaw = np.concatenate((np.zeros_like(value[..., :1, 2]), value[..., :-1, 2]), axis=-1)
    delta_xy = value[..., :2] - previous_xy
    delta_yaw = _angle_delta(value[..., 2], previous_yaw)
    distance = np.linalg.norm(delta_xy, axis=-1)
    speed = distance / 0.5
    previous_speed = np.concatenate((np.zeros_like(speed[..., :1]), speed[..., :-1]), axis=-1)
    acceleration = (speed - previous_speed) / 0.5
    curvature = delta_yaw / np.maximum(distance, np.finfo(np.float32).eps)
    return np.stack(
        (
            np.tanh(delta_xy[..., 0] / 50.0),
            np.tanh(delta_xy[..., 1] / 50.0),
            np.sin(delta_yaw),
            np.cos(delta_yaw),
            np.tanh(speed / 20.0),
            np.tanh(acceleration / 10.0),
            np.tanh(curvature),
        ),
        axis=-1,
    ).astype(np.float32)


def _component_rms(
    previous_states: np.ndarray,
    current_states: np.ndarray,
    interval: float,
) -> np.ndarray:
    overlap = int(round(interval / 0.1))
    previous = np.repeat(previous_states[None, overlap:], len(current_states), axis=0)
    current = current_states[:, :-overlap]
    time_s = np.arange(previous.shape[1], dtype=np.float64) * 0.1
    first = extract_features(previous, time_s)
    second = extract_features(current, time_s)
    return np.stack(
        (
            calculate_rms(first["acceleration"] - second["acceleration"]),
            calculate_rms(first["jerk"] - second["jerk"]),
            calculate_rms(first["yaw_rate"] - second["yaw_rate"]),
            calculate_rms(first["yaw_accel"] - second["yaw_accel"]),
        ),
        axis=-1,
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--states", required=True, nargs="+", type=Path)
    parser.add_argument("--scores", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    raw = pd.concat([pd.read_csv(path) for path in args.scores], ignore_index=True)
    raw["token"] = raw["token"].astype(str)
    raw["variant"] = raw["variant"].astype(str)
    variants = raw["variant"].drop_duplicates().tolist()
    score_rows = raw.set_index(["token", "variant"], verify_integrity=True)

    states: dict[str, dict[str, np.ndarray]] = {}
    metadata = []
    for directory in args.states:
        for path in sorted(directory.glob("*.npz")):
            payload = _load(path)
            names = [str(value) for value in payload["variant_names"]]
            if names != variants:
                raise ValueError(f"variant mismatch for {path.stem}: {names} != {variants}")
            states[path.stem] = dict(zip(names, np.asarray(payload["variant_states"], dtype=np.float32)))
            metadata.append(
                (
                    path.stem,
                    str(np.asarray(payload["log_name"]).item()),
                    float(np.asarray(payload["start_time"]).item()),
                )
            )

    diagnostics = {path.stem: _load(path) for path in sorted(args.diagnostics.glob("*.npz"))}
    trajectory_variants = {path.stem: _load(path) for path in sorted(args.variants.glob("*.npz"))}
    tokens = set(states)
    if tokens != set(diagnostics) or tokens != set(trajectory_variants):
        raise ValueError("diagnostic, trajectory-variant, and state tokens differ")

    table = pd.DataFrame(metadata, columns=("token", "log_name", "start_time"))
    edges: list[tuple[str, str, str, float]] = []
    for log_name, group in table.groupby("log_name"):
        rows = list(group.sort_values("start_time").itertuples())
        for previous, current in zip(rows, rows[1:]):
            interval = float(current.start_time - previous.start_time)
            if 0.0 < interval < 0.55:
                edges.append((str(previous.token), str(current.token), str(log_name), interval))

    previous_plan = []
    current_plans = []
    previous_future = []
    current_futures = []
    component_rms = []
    component_ratio = []
    ec_pass = []
    safety_valid = []
    progress_ratio = []
    pair_utility = []
    current_utility = []
    log_names = []
    previous_tokens = []
    current_tokens = []

    hard_columns = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "traffic_light_compliance",
        "time_to_collision_within_bound",
    )
    for number, (previous_token, current_token, log_name, interval) in enumerate(edges, start=1):
        previous_diag = diagnostics[previous_token]
        current_diag = diagnostics[current_token]
        previous_index = _selected_index(previous_diag)
        current_index = _selected_index(current_diag)
        previous_world = np.asarray(previous_diag["world_features"][previous_index], dtype=np.float32)
        current_world = np.asarray(current_diag["world_features"][current_index], dtype=np.float32)
        plans = np.asarray(trajectory_variants[current_token]["custom_trajectories"], dtype=np.float32)
        names = [str(value) for value in trajectory_variants[current_token]["custom_names"]]
        if names != variants:
            raise ValueError(f"trajectory variant mismatch for {current_token}")

        # Candidate future latents interpolate from the current prediction toward
        # the previous selected future as temporal-anchor strength increases.
        weights = np.asarray([_variant_strength(name) for name in names], dtype=np.float32)
        futures = (1.0 - weights[:, None]) * current_world + weights[:, None] * previous_world
        rms = _component_rms(
            states[previous_token]["selected"],
            np.stack([states[current_token][name] for name in names]),
            interval,
        )

        selected_row = score_rows.loc[(current_token, "selected")]
        previous_row = score_rows.loc[(previous_token, "selected")]
        valid = []
        ratios = []
        for name in names:
            row = score_rows.loc[(current_token, name)]
            valid.append(
                all(float(row[column]) + 1e-8 >= float(selected_row[column]) for column in hard_columns)
                and float(row["ego_progress"]) + 1e-8 >= 0.98 * float(selected_row["ego_progress"])
            )
            denominator = max(float(selected_row["ego_progress"]), 1e-6)
            ratios.append(float(row["ego_progress"]) / denominator)

        previous_plan.append(_plan_tokens(np.asarray(previous_diag["trajectory"], dtype=np.float32)))
        current_plans.append(_plan_tokens(plans))
        previous_future.append(previous_world)
        current_futures.append(futures.astype(np.float32))
        component_rms.append(rms)
        ratio = rms / THRESHOLDS[None]
        passed = np.all(ratio <= 1.0, axis=-1)
        current_scores = np.asarray(
            [_row_score(score_rows.loc[(current_token, name)], float(ec)) for name, ec in zip(names, passed)],
            dtype=np.float32,
        )
        previous_scores = np.asarray(
            [_row_score(previous_row, float(ec)) for ec in passed], dtype=np.float32
        )
        component_ratio.append(ratio)
        ec_pass.append(passed)
        safety_valid.append(np.asarray(valid, dtype=np.bool_))
        progress_ratio.append(np.asarray(ratios, dtype=np.float32))
        current_utility.append(current_scores)
        pair_utility.append(0.5 * (current_scores + previous_scores))
        log_names.append(log_name)
        previous_tokens.append(previous_token)
        current_tokens.append(current_token)
        if number % 300 == 0:
            print(f"paired={number}/{len(edges)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        variant_names=np.asarray(variants),
        thresholds=THRESHOLDS,
        previous_tokens=np.asarray(previous_tokens),
        current_tokens=np.asarray(current_tokens),
        log_names=np.asarray(log_names),
        previous_plan_tokens=np.stack(previous_plan),
        current_plan_tokens=np.stack(current_plans),
        previous_future_latent=np.stack(previous_future),
        current_future_latent=np.stack(current_futures),
        component_rms=np.stack(component_rms),
        component_ratio=np.stack(component_ratio),
        ec_pass=np.stack(ec_pass),
        safety_progress_valid=np.stack(safety_valid),
        progress_ratio=np.stack(progress_ratio),
        current_utility=np.stack(current_utility),
        pair_utility=np.stack(pair_utility),
    )
    report = {
        "format": "actworld_v2_navtrain_temporal_fcts_pairs_v1",
        "split": "navtrain",
        "navhard_used": False,
        "pairs": len(edges),
        "variants": variants,
        "ec_pass_rate": {
            name: float(np.stack(ec_pass)[:, index].mean()) for index, name in enumerate(variants)
        },
        "safety_progress_valid_rate": {
            name: float(np.stack(safety_valid)[:, index].mean()) for index, name in enumerate(variants)
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
