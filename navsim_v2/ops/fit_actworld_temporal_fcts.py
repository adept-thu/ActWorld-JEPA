#!/usr/bin/env python3
"""Fit an ActWorld temporal plan--future compatibility head on NAVTRAIN only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

from fit_v2_candidate_ranker import _row_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_two_frame_extended_comfort,
)


SAFETY = np.asarray((0, 1, 2, 3, 5, 6), dtype=np.int64)


def candidate_index(value: object) -> int:
    return int(str(value).removeprefix("candidate"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def coarse_motion(poses: np.ndarray) -> np.ndarray:
    """Return frame-invariant coarse kinematics at the 0.5 s plan rate."""

    value = np.asarray(poses, dtype=np.float64)
    extended = np.concatenate((np.zeros((1, 3)), value), axis=0)
    delta_xy = np.diff(extended[:, :2], axis=0)
    speed = np.linalg.norm(delta_xy, axis=1) / 0.5
    accel = np.diff(np.r_[0.0, speed]) / 0.5
    heading = np.unwrap(extended[:, 2])
    yaw_rate = np.diff(heading) / 0.5
    yaw_accel = np.diff(np.r_[0.0, yaw_rate]) / 0.5
    return np.stack((speed, accel, yaw_rate, yaw_accel), axis=-1)


def relation_summary(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    first = np.asarray(current, dtype=np.float64)[:-1]
    second = np.asarray(previous, dtype=np.float64)[1:]
    difference = first - second
    product = first * second
    return np.concatenate(
        (
            np.mean(first, axis=0),
            np.std(first, axis=0),
            np.mean(second, axis=0),
            np.std(second, axis=0),
            np.mean(np.abs(difference), axis=0),
            np.sqrt(np.mean(np.square(difference), axis=0)),
            np.mean(product, axis=0),
        )
    )


def pair_feature(
    current: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    current_index: int,
    previous_index: int,
    projected: dict[str, np.ndarray],
    motion: dict[str, np.ndarray],
    current_token: str,
    previous_token: str,
) -> np.ndarray:
    future_current = projected[current_token][current_index]
    future_previous = projected[previous_token][previous_index]
    factor_current = np.asarray(current["candidate_subscores"][current_index], dtype=np.float64)
    factor_previous = np.asarray(previous["candidate_subscores"][previous_index], dtype=np.float64)
    latent_interaction = np.concatenate(
        (
            future_current,
            future_previous,
            future_current * future_previous,
            np.abs(future_current - future_previous),
        )
    )
    factor_interaction = np.concatenate(
        (
            factor_current,
            factor_previous,
            factor_current * factor_previous,
            np.abs(factor_current - factor_previous),
        )
    )
    score_context = np.asarray(
        (
            current["pdm_score"][current_index],
            previous["pdm_score"][previous_index],
            current["base_pdm_score"][current_index],
            previous["base_pdm_score"][previous_index],
            current["residual_logits"][current_index],
            previous["residual_logits"][previous_index],
        ),
        dtype=np.float64,
    )
    plan_interaction = relation_summary(
        motion[current_token][current_index], motion[previous_token][previous_index]
    )
    return np.concatenate(
        (latent_interaction, factor_interaction, plan_interaction, score_context)
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--scores", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    dataset = joblib.load(args.dataset)
    tokens = np.asarray(dataset["tokens"]).astype(str)
    baseline = {str(key): int(value) for key, value in dataset["baseline"].items()}
    groups_by_token = {
        str(token): str(group)
        for token, group in zip(tokens, np.asarray(dataset["token_groups"]).astype(str))
    }
    raw = pd.concat([pd.read_csv(path) for path in args.scores], ignore_index=True)
    raw["token"] = raw["token"].astype(str)
    raw["candidate"] = raw["variant"].map(candidate_index)
    scores = raw.set_index(["token", "candidate"], verify_integrity=True)
    metadata = (
        raw[raw["candidate"] == 0]
        .set_index("token", verify_integrity=True)[["log_name", "start_time"]]
        .loc[tokens]
    )

    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    states: dict[str, np.ndarray] = {}
    world_blocks = []
    for number, token in enumerate(tokens, start=1):
        item = load_npz(args.diagnostics / f"{token}.npz")
        if "world_features" not in item:
            raise KeyError(f"missing Planning-JEPA world_features for {token}")
        diagnostics[token] = item
        world_blocks.append(np.asarray(item["world_features"], dtype=np.float32))
        states[token] = np.asarray(
            load_npz(args.states / f"{token}.npz")["variant_states"],
            dtype=np.float32,
        )
        if number % 400 == 0:
            print(f"loaded={number}/{len(tokens)}", flush=True)

    world = np.concatenate(world_blocks, axis=0)
    pca = PCA(n_components=args.components, whiten=True, random_state=args.seed).fit(world)
    transformed = pca.transform(world).reshape(len(tokens), 32, args.components)
    projected = {token: transformed[index] for index, token in enumerate(tokens)}
    motion = {
        token: np.stack([coarse_motion(plan) for plan in diagnostics[token]["refined_proposals"]])
        for token in tokens
    }

    edges: list[tuple[str, str, float]] = []
    for _, group in metadata.reset_index().groupby("log_name"):
        ordered = group.sort_values("start_time")
        rows = list(ordered.itertuples())
        for previous, current in zip(rows, rows[1:]):
            interval = float(current.start_time - previous.start_time)
            if 0.0 < interval < 0.55:
                edges.append((str(current.token), str(previous.token), interval))

    edge_features: list[np.ndarray] = []
    edge_labels: list[np.ndarray] = []
    edge_candidates: list[np.ndarray] = []
    edge_groups: list[str] = []
    baseline_labels: list[float] = []
    for number, (current_token, previous_token, interval) in enumerate(edges, start=1):
        current = diagnostics[current_token]
        previous = diagnostics[previous_token]
        current_pool = np.unique(
            np.r_[baseline[current_token], np.argsort(current["pdm_score"])[-4:]]
        ).astype(np.int64)
        previous_pool = np.unique(
            np.r_[baseline[previous_token], np.argsort(previous["pdm_score"])[-4:]]
        ).astype(np.int64)
        combinations = np.asarray(
            [(i, j) for i in current_pool for j in previous_pool], dtype=np.int64
        )
        edge_candidates.append(combinations)
        edge_features.append(
            np.stack(
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
        )
        overlap = int(round(interval / 0.1))
        current_states = states[current_token][combinations[:, 0], :-overlap]
        previous_states = states[previous_token][combinations[:, 1], overlap:]
        time_s = np.arange(current_states.shape[1], dtype=np.float64) * 0.1
        labels = ego_is_two_frame_extended_comfort(
            current_states, previous_states, time_s
        ).astype(np.int8)
        edge_labels.append(labels)
        incumbent = np.flatnonzero(
            (combinations[:, 0] == baseline[current_token])
            & (combinations[:, 1] == baseline[previous_token])
        )
        if len(incumbent) != 1:
            raise RuntimeError("incumbent pair missing from shortlist")
        baseline_labels.append(float(labels[incumbent[0]]))
        edge_groups.append(str(metadata.at[current_token, "log_name"]))
        if number % 300 == 0:
            print(f"pairs={number}/{len(edges)}", flush=True)

    edge_index = np.arange(len(edges))
    oof = [np.full(len(block), np.nan, dtype=np.float64) for block in edge_features]
    models = []
    token_fold = np.full(len(edges), -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    for fold, (train_edges, valid_edges) in enumerate(
        splitter.split(edge_index, groups=np.asarray(edge_groups))
    ):
        x_train = np.concatenate([edge_features[i] for i in train_edges])
        y_train = np.concatenate([edge_labels[i] for i in train_edges])
        model = ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=30,
            max_features=0.45,
            class_weight="balanced",
            n_jobs=8,
            random_state=args.seed + fold,
        ).fit(x_train, y_train)
        models.append(model)
        for index in valid_edges:
            oof[index] = model.predict_proba(edge_features[index])[:, 1]
            token_fold[index] = fold
        print(f"fold={fold} train_rows={len(y_train)}", flush=True)

    baseline_labels_array = np.asarray(baseline_labels)
    policies = []
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.30):
        proposals = []
        for edge, combinations, prediction, labels in zip(
            edges, edge_candidates, oof, edge_labels
        ):
            current_token, previous_token, _ = edge
            current = diagnostics[current_token]
            previous = diagnostics[previous_token]
            bi = baseline[current_token]
            bj = baseline[previous_token]
            incumbent_pos = int(
                np.flatnonzero((combinations[:, 0] == bi) & (combinations[:, 1] == bj))[0]
            )
            baseline_probability = prediction[incumbent_pos]
            incumbent_current = current["candidate_subscores"][bi]
            incumbent_previous = previous["candidate_subscores"][bj]
            eligible = np.ones(len(combinations), dtype=np.bool_)
            for row, (i, j) in enumerate(combinations):
                current_ratio = current["candidate_subscores"][i, SAFETY] / np.maximum(
                    incumbent_current[SAFETY], 1e-6
                )
                previous_ratio = previous["candidate_subscores"][j, SAFETY] / np.maximum(
                    incumbent_previous[SAFETY], 1e-6
                )
                eligible[row] = (
                    min(float(current_ratio.min()), float(previous_ratio.min())) >= 0.999
                    and current["candidate_subscores"][i, 4] >= incumbent_current[4] - 0.001
                    and previous["candidate_subscores"][j, 4] >= incumbent_previous[4] - 0.001
                    and current["pdm_score"][i] >= current["pdm_score"][bi] - 0.005
                    and previous["pdm_score"][j] >= previous["pdm_score"][bj] - 0.005
                )
            gain = prediction - baseline_probability
            objective = np.where(eligible & (gain >= threshold), prediction, -np.inf)
            chosen = int(np.argmax(objective))
            if not np.isfinite(objective[chosen]):
                chosen = incumbent_pos
            current_index, previous_index = combinations[chosen]
            exact_quality_gain = 0.5 * (
                _row_score(scores.loc[(current_token, int(current_index))], float(labels[chosen]))
                - _row_score(scores.loc[(current_token, bi)], float(labels[incumbent_pos]))
                + _row_score(scores.loc[(previous_token, int(previous_index))], float(labels[chosen]))
                - _row_score(scores.loc[(previous_token, bj)], float(labels[incumbent_pos]))
            )
            proposals.append(
                {
                    "chosen": chosen,
                    "incumbent": incumbent_pos,
                    "probability_gain": float(gain[chosen]),
                    "ec_gain": float(labels[chosen] - labels[incumbent_pos]),
                    "score_gain": float(exact_quality_gain),
                }
            )
        order = np.argsort([-item["probability_gain"] for item in proposals])
        for fraction in (0.02, 0.05, 0.10):
            accepted_count = max(1, int(np.ceil(len(edges) * fraction)))
            accepted = set(order[:accepted_count].tolist())
            selected = [
                item if index in accepted else {
                    "chosen": item["incumbent"],
                    "incumbent": item["incumbent"],
                    "probability_gain": 0.0,
                    "ec_gain": 0.0,
                    "score_gain": 0.0,
                }
                for index, item in enumerate(proposals)
            ]
            fold_score = []
            fold_ec = []
            for fold in range(5):
                mask = token_fold == fold
                fold_score.append(float(np.mean([selected[i]["score_gain"] for i in edge_index[mask]])))
                fold_ec.append(float(np.mean([selected[i]["ec_gain"] for i in edge_index[mask]])))
            policies.append(
                {
                    "threshold": threshold,
                    "fraction": fraction,
                    "accepted": int(sum(item["chosen"] != item["incumbent"] for item in selected)),
                    "mean_score_gain": float(np.mean([item["score_gain"] for item in selected])),
                    "worst_fold_score_gain": float(min(fold_score)),
                    "mean_ec_gain": float(np.mean([item["ec_gain"] for item in selected])),
                    "worst_fold_ec_gain": float(min(fold_ec)),
                    "fold_score_gain": fold_score,
                    "fold_ec_gain": fold_ec,
                }
            )
    policies.sort(
        key=lambda item: (
            item["worst_fold_score_gain"] >= 0.0,
            item["worst_fold_ec_gain"] >= 0.0,
            item["mean_score_gain"],
            item["mean_ec_gain"],
        ),
        reverse=True,
    )
    winner = policies[0]
    artifact = {
        "format": "actworld_v2_navtrain_temporal_fcts_pair_head_v1",
        "training_split": "navtrain",
        "navtest_used_for_training": False,
        "components": args.components,
        "pca": pca,
        "models": tuple(models),
        "policy": winner,
        "safety_indices": tuple(int(v) for v in SAFETY),
        "feature_count": int(edge_features[0].shape[1]),
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    report = {
        "format": artifact["format"],
        "training_split": "navtrain",
        "navtest_used_for_training": False,
        "tokens": len(tokens),
        "pairs": len(edges),
        "training_rows": int(sum(len(item) for item in edge_labels)),
        "baseline_ec": float(np.mean(baseline_labels_array)),
        "feature_count": artifact["feature_count"],
        "winner": winner,
        "top_policies": policies[:10],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
