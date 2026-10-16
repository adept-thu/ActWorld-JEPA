"""NAVTRAIN-only strict variant-selector features and trajectory transforms."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


VARIANTS = (
    "base25",
    "base50",
    "factor25",
    "factor50",
    "ec25",
    "ep25",
    "second25",
    "progress102",
    "progress098",
)


def _geometry(proposal: np.ndarray) -> np.ndarray:
    xy = proposal[:, :2].astype(np.float64)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    heading = np.unwrap(proposal[:, 2].astype(np.float64))
    heading_steps = np.abs(np.diff(heading))
    return np.asarray(
        [
            xy[-1, 0],
            xy[-1, 1],
            heading[-1],
            steps.sum(),
            steps.mean() if len(steps) else 0.0,
            steps.max() if len(steps) else 0.0,
            heading_steps.sum(),
            heading_steps.max() if len(heading_steps) else 0.0,
            np.abs(xy[:, 1]).max(),
        ],
        dtype=np.float64,
    )


def _blend(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    result = first.astype(np.float64, copy=True)
    result[:, :2] = (1.0 - weight) * first[:, :2] + weight * second[:, :2]
    result[:, 2] = np.arctan2(
        (1.0 - weight) * np.sin(first[:, 2]) + weight * np.sin(second[:, 2]),
        (1.0 - weight) * np.cos(first[:, 2]) + weight * np.cos(second[:, 2]),
    )
    return result


def _factor_score(subscores: np.ndarray) -> np.ndarray:
    hard = np.prod(subscores[:, :4], axis=1)
    weights = np.asarray([5.0, 5.0, 2.0, 2.0, 2.0])
    return hard * (subscores[:, 4:] * weights).sum(axis=1) / weights.sum()


def variant_features(
    payload: Dict[str, np.ndarray], variant: str
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build the exact feature contract shared by fitting and deployment."""

    proposals = np.asarray(payload["proposals"], dtype=np.float64)
    deployed = np.asarray(payload["pdm_score"], dtype=np.float64)
    base = np.asarray(payload["base_pdm_score"], dtype=np.float64)
    factors = np.asarray(payload["candidate_subscores"], dtype=np.float64)
    residual = np.asarray(payload["residual_logits"], dtype=np.float64)
    selected = int(np.argmax(deployed))
    order = np.argsort(deployed)
    targets = {
        "base": int(np.argmax(base)),
        "factor": int(np.argmax(_factor_score(factors))),
        "ec": int(np.argmax(factors[:, 8])),
        "ep": int(np.argmax(factors[:, 4])),
        "second": int(order[-2]) if len(order) > 1 else selected,
    }
    if variant.startswith("base"):
        target, weight, scale = targets["base"], float(variant[-2:]) / 100.0, 1.0
    elif variant.startswith("factor"):
        target, weight, scale = targets["factor"], float(variant[-2:]) / 100.0, 1.0
    elif variant.startswith("ec"):
        target, weight, scale = targets["ec"], float(variant[-2:]) / 100.0, 1.0
    elif variant.startswith("ep"):
        target, weight, scale = targets["ep"], float(variant[-2:]) / 100.0, 1.0
    elif variant.startswith("second"):
        target, weight, scale = targets["second"], float(variant[-2:]) / 100.0, 1.0
    elif variant == "progress102":
        target, weight, scale = selected, 0.0, 1.02
    elif variant == "progress098":
        target, weight, scale = selected, 0.0, 0.98
    else:
        raise ValueError(f"unknown ActWorld trajectory variant: {variant}")

    selected_pose = proposals[selected]
    variant_pose = _blend(selected_pose, proposals[target], weight)
    variant_pose[:, :2] *= scale
    selected_factors = factors[selected]
    target_factors = factors[target]
    selected_geometry = _geometry(selected_pose)
    variant_geometry = _geometry(variant_pose)
    one_hot = np.zeros(len(VARIANTS), dtype=np.float64)
    one_hot[VARIANTS.index(variant)] = 1.0
    hard_delta = float(np.prod(target_factors[:4]) - np.prod(selected_factors[:4]))
    ec_factor_delta = float(target_factors[8] - selected_factors[8])
    feature = np.concatenate(
        [
            np.asarray(
                [
                    deployed[selected],
                    deployed[target],
                    deployed[target] - deployed[selected],
                    base[selected],
                    base[target],
                    base[target] - base[selected],
                    residual[selected],
                    residual[target],
                    residual[target] - residual[selected],
                    weight,
                    scale,
                    selected / max(len(proposals) - 1, 1),
                    target / max(len(proposals) - 1, 1),
                    abs(target - selected) / max(len(proposals) - 1, 1),
                ]
            ),
            selected_factors,
            target_factors,
            target_factors - selected_factors,
            selected_geometry,
            variant_geometry,
            variant_geometry - selected_geometry,
            one_hot,
        ]
    ).astype(np.float32)
    return feature, {
        "selected": float(selected),
        "target": float(target),
        "weight": weight,
        "scale": scale,
        "hard_delta": hard_delta,
        "ec_factor_delta": ec_factor_delta,
    }


def post_variant_features(
    payload: Dict[str, np.ndarray], variant: str
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build deployment-exact current/future interaction features.

    The post selector starts from the trajectory already selected by the
    frozen mother and moves it toward one actor-refined future proposal.
    """

    base_feature, metadata = variant_features(payload, variant)
    current = np.asarray(payload["trajectory"], dtype=np.float64)
    refined = np.asarray(payload["refined_proposals"], dtype=np.float64)
    future = refined[int(metadata["target"])]
    proposed = _blend(current, future, float(metadata["weight"]))
    proposed[:, :2] *= float(metadata["scale"])

    current_geometry = _geometry(current)
    future_geometry = _geometry(future)
    proposed_geometry = _geometry(proposed)
    interaction = np.concatenate(
        (
            current_geometry,
            future_geometry,
            proposed_geometry,
            future_geometry - current_geometry,
            proposed_geometry - current_geometry,
            current_geometry * future_geometry,
            np.abs(current_geometry - future_geometry),
        )
    ).astype(np.float32)
    return np.concatenate((base_feature, interaction)), metadata
