"""Frozen NAVTRAIN gain-selector features for ActWorld-JEPA F2 deployment."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _top_margin(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    top = np.partition(values, -2)[-2:]
    return float(top.max() - top.min())


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
            steps.mean() if steps.size else 0.0,
            steps.max() if steps.size else 0.0,
            heading_steps.sum(),
            heading_steps.max() if heading_steps.size else 0.0,
        ],
        dtype=np.float64,
    )


FEATURE_NAMES = (
    "f2_selected_score",
    "f2_base_score",
    "f2_advantage",
    "f2_top_margin",
    "base_best_score",
    "base_selected_score",
    "base_gap",
    "base_top_margin",
    "residual_selected",
    "residual_base",
    "residual_delta",
    *(f"selected_factor_{index}" for index in range(9)),
    *(f"base_factor_{index}" for index in range(9)),
    *(f"factor_delta_{index}" for index in range(9)),
    "selected_hard_safety",
    "base_hard_safety",
    "hard_safety_delta",
    "selected_soft_quality",
    "base_soft_quality",
    "soft_quality_delta",
    *(f"selected_geometry_{index}" for index in range(8)),
    *(f"base_geometry_{index}" for index in range(8)),
    *(f"geometry_delta_{index}" for index in range(8)),
    "selected_index_normalized",
    "base_index_normalized",
    "index_distance_normalized",
)


def selector_features(
    payload: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build the versioned feature vector used at fit and deployment time."""

    f2 = np.asarray(payload["pdm_score"], dtype=np.float64)
    base = np.asarray(payload["base_pdm_score"], dtype=np.float64)
    factors = np.asarray(payload["candidate_subscores"], dtype=np.float64)
    residual = np.asarray(payload["residual_logits"], dtype=np.float64)
    proposals = np.asarray(payload["proposals"], dtype=np.float64)
    selected = int(np.argmax(f2))
    mother = int(np.argmax(base))
    selected_factors = factors[selected]
    mother_factors = factors[mother]
    selected_safety = float(np.prod(selected_factors[:4]))
    mother_safety = float(np.prod(mother_factors[:4]))
    weights = np.asarray([5.0, 5.0, 2.0, 2.0, 2.0]) / 16.0
    selected_quality = float(selected_factors[4:] @ weights)
    mother_quality = float(mother_factors[4:] @ weights)
    selected_geometry = _geometry(proposals[selected])
    mother_geometry = _geometry(proposals[mother])
    denominator = max(len(f2) - 1, 1)
    feature = np.concatenate(
        (
            np.asarray(
                [
                    f2[selected],
                    f2[mother],
                    f2[selected] - f2[mother],
                    _top_margin(f2),
                    base[mother],
                    base[selected],
                    base[mother] - base[selected],
                    _top_margin(base),
                    residual[selected],
                    residual[mother],
                    residual[selected] - residual[mother],
                ]
            ),
            selected_factors,
            mother_factors,
            selected_factors - mother_factors,
            np.asarray(
                [
                    selected_safety,
                    mother_safety,
                    selected_safety - mother_safety,
                    selected_quality,
                    mother_quality,
                    selected_quality - mother_quality,
                ]
            ),
            selected_geometry,
            mother_geometry,
            selected_geometry - mother_geometry,
            np.asarray(
                [
                    selected / denominator,
                    mother / denominator,
                    abs(selected - mother) / denominator,
                ]
            ),
        )
    ).astype(np.float32)
    if feature.shape != (len(FEATURE_NAMES),):
        raise ValueError(
            f"gain-selector feature shape {feature.shape} does not match "
            f"contract {(len(FEATURE_NAMES),)}"
        )
    return feature, {
        "selected": float(selected),
        "mother": float(mother),
        "base_gap": float(base[mother] - base[selected]),
        "hard_safety_delta": selected_safety - mother_safety,
    }
