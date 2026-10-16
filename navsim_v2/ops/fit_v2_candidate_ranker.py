#!/usr/bin/env python3
"""Fit a NAVTRAIN-only future-consistent ranker over all trajectory proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from navsim.agents.actworld_jepa.actworld_candidate_selector import (
    FEATURE_NAMES,
    candidate_features,
    candidate_shortlist,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_two_frame_extended_comfort,
)


def _parse_array(value: object) -> np.ndarray:
    return np.fromstring(str(value).strip().strip("[]"), sep=" ")


def _row_score(row: pd.Series, ec: float) -> float:
    soft = _parse_array(row["weighted_metrics"]).copy()
    weights = _parse_array(row["weighted_metrics_array"]).copy()
    if np.isnan(ec):
        soft[4] = 0.0
        weights[4] = 0.0
    else:
        soft[4] = ec
    return float(
        float(row["multiplicative_metrics_prod"])
        * np.dot(soft, weights)
        / weights.sum()
    )


def _comfort(first: np.ndarray, second: np.ndarray, interval: float) -> float:
    if abs(interval) > 0.55:
        return float("nan")
    overlap = int(interval / 0.1)
    time_s = np.arange(len(first) - overlap, dtype=np.float64) * 0.1
    return float(
        ego_is_two_frame_extended_comfort(
            first[overlap:][None], second[:-overlap][None], time_s
        )[0]
    )


def _candidate_index(name: str) -> int:
    if not name.startswith("candidate"):
        raise ValueError(f"unexpected candidate name {name!r}")
    return int(name.removeprefix("candidate"))


def _load_dataset(
    score_paths: list[Path], diagnostics: Path, states: Path
) -> dict[str, object]:
    raw = pd.concat([pd.read_csv(path) for path in score_paths], ignore_index=True)
    raw["candidate"] = raw["variant"].astype(str).map(_candidate_index)
    scores = raw.set_index(["token", "candidate"], verify_integrity=True)
    metadata = (
        raw[raw["candidate"] == 0]
        .set_index("token", verify_integrity=True)[["log_name", "start_time", "frame_type"]]
    )
    tokens = metadata.index.astype(str).tolist()

    payloads: dict[str, dict[str, np.ndarray]] = {}
    state_map: dict[str, np.ndarray] = {}
    baseline: dict[str, int] = {}
    for token in tokens:
        with np.load(diagnostics / f"{token}.npz", allow_pickle=False) as payload:
            item = {name: payload[name] for name in payload.files}
        payloads[token] = item
        baseline[token] = int(np.argmax(item["pdm_score"]))
        with np.load(states / f"{token}.npz", allow_pickle=False) as state_payload:
            names = [str(value) for value in state_payload["variant_names"]]
            order = [_candidate_index(name) for name in names]
            candidate_states = np.asarray(state_payload["variant_states"])
        if order != list(range(len(order))):
            raise ValueError(f"candidate-state order mismatch for {token}")
        state_map[token] = candidate_states

    previous: dict[str, str | None] = {token: None for token in tokens}
    following: dict[str, str | None] = {token: None for token in tokens}
    for _, group in metadata.reset_index().groupby("log_name"):
        group = group.sort_values("start_time")
        ordered = group["token"].astype(str).tolist()
        times = group["start_time"].astype(float).tolist()
        for index, token in enumerate(ordered):
            if index > 0 and abs(times[index] - times[index - 1]) <= 0.55:
                previous[token] = ordered[index - 1]
            if index + 1 < len(ordered) and abs(times[index + 1] - times[index]) <= 0.55:
                following[token] = ordered[index + 1]

    baseline_ec: dict[str, float] = {}
    baseline_score: dict[str, float] = {}
    for token in tokens:
        nxt = following[token]
        ec = float("nan")
        if nxt is not None:
            ec = _comfort(
                state_map[token][baseline[token]],
                state_map[nxt][baseline[nxt]],
                float(metadata.at[nxt, "start_time"] - metadata.at[token, "start_time"]),
            )
        baseline_ec[token] = ec
        baseline_score[token] = _row_score(scores.loc[(token, baseline[token])], ec)

    rows = []
    for token in tokens:
        payload = payloads[token]
        count = len(payload["proposals"])
        shortlists = {
            top_k: set(candidate_shortlist(payload, top_k).tolist())
            for top_k in (2, 4, 6, 8)
        }
        for candidate in range(count):
            feature, info = candidate_features(payload, candidate)
            nxt = following[token]
            ec = float("nan")
            if nxt is not None:
                ec = _comfort(
                    state_map[token][candidate],
                    state_map[nxt][baseline[nxt]],
                    float(metadata.at[nxt, "start_time"] - metadata.at[token, "start_time"]),
                )
            gain = _row_score(scores.loc[(token, candidate)], ec) - baseline_score[token]
            prev = previous[token]
            if prev is not None:
                previous_ec = _comfort(
                    state_map[prev][baseline[prev]],
                    state_map[token][candidate],
                    float(metadata.at[token, "start_time"] - metadata.at[prev, "start_time"]),
                )
                gain += (
                    _row_score(scores.loc[(prev, baseline[prev])], previous_ec)
                    - baseline_score[prev]
                )
            rows.append(
                (
                    token,
                    str(metadata.at[token, "log_name"]),
                    candidate,
                    feature,
                    float(gain),
                    float(info["base_gap"]),
                    float(info["hard_safety_delta"]),
                    tuple(top_k for top_k, values in shortlists.items() if candidate in values),
                )
            )

    return {
        "token": np.asarray([row[0] for row in rows]),
        "group": np.asarray([row[1] for row in rows]),
        "candidate": np.asarray([row[2] for row in rows], dtype=np.int64),
        "x": np.stack([row[3] for row in rows]),
        "gain": np.asarray([row[4] for row in rows]),
        "base_gap": np.asarray([row[5] for row in rows]),
        "hard_delta": np.asarray([row[6] for row in rows]),
        "shortlists": [row[7] for row in rows],
        "tokens": np.asarray(tokens),
        "token_groups": metadata.loc[tokens, "log_name"].astype(str).to_numpy(),
        "baseline": baseline,
    }


def _model(leaf: int, max_features: float, seed: int, final: bool = False):
    return ExtraTreesRegressor(
        n_estimators=600 if final else 250,
        min_samples_leaf=leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=seed,
    )


def _oof(
    data: dict[str, object], leaf: int, max_features: float
) -> tuple[np.ndarray, np.ndarray]:
    token_names = data["tokens"]
    token_groups = data["token_groups"]
    row_tokens = data["token"]
    prediction = np.full(len(row_tokens), -np.inf, dtype=np.float64)
    row_fold = np.full(len(row_tokens), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(token_groups))))
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(token_names, groups=token_groups)
    ):
        train_tokens = set(token_names[train_index])
        valid_tokens = set(token_names[valid_index])
        train = np.asarray([token in train_tokens for token in row_tokens])
        valid = np.asarray([token in valid_tokens for token in row_tokens])
        model = _model(leaf, max_features, 20261009 + fold).fit(
            data["x"][train], data["gain"][train]
        )
        prediction[valid] = model.predict(data["x"][valid])
        row_fold[valid] = fold
    return prediction, row_fold


def _policies(
    data: dict[str, object], prediction: np.ndarray, row_fold: np.ndarray
) -> list[dict[str, object]]:
    frame = pd.DataFrame(
        {
            "token": data["token"],
            "candidate": data["candidate"],
            "prediction": prediction,
            "actual_gain": data["gain"],
            "base_gap": data["base_gap"],
            "hard_delta": data["hard_delta"],
            "fold": row_fold,
        }
    )
    reports = []
    for top_k in (2, 4, 6, 8):
        eligible_top = np.asarray([top_k in values for values in data["shortlists"]])
        maxima = frame.loc[eligible_top].groupby("token")["prediction"].max()
        thresholds = sorted(
            set([0.0, 0.001, 0.002, 0.004] + np.quantile(maxima, [0.4, 0.6, 0.75, 0.85, 0.9]).tolist())
        )
        for threshold in thresholds:
            for base_gap in (0.002, 0.005, 0.01, 0.02):
                for hard_guard in (-0.05, -0.02, 0.0):
                    eligible = frame[
                        eligible_top
                        & (frame["prediction"] >= threshold)
                        & (frame["base_gap"] <= base_gap)
                        & (frame["hard_delta"] >= hard_guard)
                    ]
                    best = eligible.sort_values("prediction").groupby("token").tail(1)
                    # A no-op choice is not counted as an accepted rerank.
                    changed = best.loc[
                        np.asarray(
                            [
                            int(row.candidate) != int(data["baseline"][str(row.token)])
                            for row in best.itertuples()
                            ],
                            dtype=bool,
                        )
                    ]
                    fold_gains = []
                    for fold in sorted(frame["fold"].unique()):
                        fold_tokens = set(frame.loc[frame["fold"] == fold, "token"])
                        values = changed[changed["token"].isin(fold_tokens)]["actual_gain"]
                        fold_gains.append(float(values.sum() / len(fold_tokens)))
                    reports.append(
                        {
                            "top_k": top_k,
                            "min_predicted_gain": float(threshold),
                            "max_baseline_gap": float(base_gap),
                            "min_hard_safety_delta": float(hard_guard),
                            "accepted": int(len(changed)),
                            "mean_oof_gain": float(np.mean(fold_gains)),
                            "worst_fold_gain": float(min(fold_gains)),
                            "fold_gains": fold_gains,
                        }
                    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", nargs="+", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = _load_dataset(args.scores, args.diagnostics, args.states)
    reports = []
    prediction_cache = {}
    for leaf in (5, 10, 20, 40):
        for max_features in (0.5, 0.8, 1.0):
            prediction, row_fold = _oof(data, leaf, max_features)
            prediction_cache[(leaf, max_features)] = (prediction, row_fold)
            for report in _policies(data, prediction, row_fold):
                reports.append(
                    {"leaf": leaf, "max_features": max_features, **report}
                )
    reports.sort(
        key=lambda item: (
            item["accepted"] >= 20,
            item["worst_fold_gain"] > 0.0,
            item["mean_oof_gain"],
            item["worst_fold_gain"],
        ),
        reverse=True,
    )
    winner = reports[0]
    model = _model(
        int(winner["leaf"]), float(winner["max_features"]), 20261009, final=True
    ).fit(data["x"], data["gain"])
    artifact = {
        "format": "actworld_v2_navtrain_candidate_ranker_v1",
        "training_split": "navtrain",
        "navtest_used_for_training": False,
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "factorized_blend": 0.05,
        "model": model,
        "policy": {
            key: winner[key]
            for key in (
                "top_k",
                "min_predicted_gain",
                "max_baseline_gap",
                "min_hard_safety_delta",
            )
        },
        "audit": {
            "samples": int(len(data["tokens"])),
            "candidate_rows": int(len(data["token"])),
            "winner": winner,
            "top_reports": reports[:30],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps({key: value for key, value in artifact.items() if key != "model"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["audit"], indent=2))
    print(f"ARTIFACT={args.output}")


if __name__ == "__main__":
    main()
