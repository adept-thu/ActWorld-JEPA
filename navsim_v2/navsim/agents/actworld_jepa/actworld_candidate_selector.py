"""Feature contract for a NAVTRAIN-only future-consistent candidate ranker."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _geometry(proposal: np.ndarray) -> np.ndarray:
    xy = np.asarray(proposal[:, :2], dtype=np.float64)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    heading = np.unwrap(np.asarray(proposal[:, 2], dtype=np.float64))
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


LEGACY_FACTOR_SEMANTICS = "legacy_recomposed_v1"
NAVSIM_V2_FACTOR_SEMANTICS = "navsim_v2_predicted_pdm_v2"


def _factor_scores(
    factors: np.ndarray,
    factor_semantics: str = LEGACY_FACTOR_SEMANTICS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the learned total score plus interpretable safety/quality terms.

    ``candidate_subscores`` follows Drive-JEPA's NAVSIM-v2 training target:
    ``[NC, DAC, TL, DDC, EP, TTC, LK, HC, PDM]``.  In particular, the final
    channel is already the candidate PDM target; it is not the cross-frame EC
    factor used only by the official two-stage evaluator.  The legacy feature
    contract accidentally treated that final PDM channel as a fifth soft
    factor.  Keep that behavior versioned for old frozen artifacts, while new
    NAVTRAIN-only artifacts consume the trained PDM channel directly.
    """

    hard = np.prod(factors[:, :4], axis=1)
    if factor_semantics == LEGACY_FACTOR_SEMANTICS:
        weights = np.asarray([5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float64)
        soft = (factors[:, 4:] * weights).sum(axis=1) / weights.sum()
        score = hard * soft
    elif factor_semantics == NAVSIM_V2_FACTOR_SEMANTICS:
        weights = np.asarray([5.0, 5.0, 2.0, 2.0], dtype=np.float64)
        soft = (factors[:, 4:8] * weights).sum(axis=1) / weights.sum()
        score = factors[:, 8]
    else:
        raise ValueError(f"unknown candidate factor semantics: {factor_semantics!r}")
    return score, hard, soft


def _rank(values: np.ndarray) -> np.ndarray:
    denominator = max(len(values) - 1, 1)
    return np.argsort(np.argsort(values, kind="stable"), kind="stable") / denominator


SCALAR_NAMES = (
    "deployed_score",
    "base_score",
    "residual_logit",
    "factor_score",
    "hard_safety",
    "soft_quality",
)
FEATURE_NAMES = (
    *(f"candidate_{name}" for name in SCALAR_NAMES),
    *(f"candidate_factor_{index}" for index in range(9)),
    *(f"candidate_geometry_{index}" for index in range(9)),
    *(f"selected_delta_{name}" for name in SCALAR_NAMES),
    *(f"selected_factor_delta_{index}" for index in range(9)),
    *(f"selected_geometry_delta_{index}" for index in range(9)),
    *(f"mother_delta_{name}" for name in SCALAR_NAMES),
    *(f"mother_factor_delta_{index}" for index in range(9)),
    *(f"mother_geometry_delta_{index}" for index in range(9)),
    "deployed_rank",
    "base_rank",
    "factor_rank",
    "candidate_index_normalized",
    "deployed_top_gap",
    "base_top_gap",
    "selected_index_distance",
    "mother_index_distance",
)

_DENSE_SAFETY_CHANNELS = (
    "collision",
    "ttc",
    "non_drivable",
    "oncoming",
)
_DENSE_SAFETY_STATS = (
    "mean",
    "max",
    "q75",
    "q90",
    "early_mean",
    "late_mean",
    "fraction_gt_050",
    "fraction_gt_080",
)
DENSE_SAFETY_FEATURE_NAMES = (
    *(
        f"candidate_dense_{channel}_{stat}"
        for channel in _DENSE_SAFETY_CHANNELS
        for stat in _DENSE_SAFETY_STATS
    ),
    *(
        f"selected_dense_delta_{channel}_{stat}"
        for channel in _DENSE_SAFETY_CHANNELS
        for stat in _DENSE_SAFETY_STATS
    ),
    *(
        f"mother_dense_delta_{channel}_{stat}"
        for channel in _DENSE_SAFETY_CHANNELS
        for stat in _DENSE_SAFETY_STATS
    ),
    *(f"dense_{channel}_mean_risk_rank" for channel in _DENSE_SAFETY_CHANNELS),
    *(f"dense_{channel}_max_risk_rank" for channel in _DENSE_SAFETY_CHANNELS),
)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def dense_safety_features(
    agent_logits: np.ndarray,
    area_logits: np.ndarray,
    candidate: int,
    selected: int,
    mother: int,
) -> np.ndarray:
    """Summarize Drive-JEPA's trained dense collision/area auxiliary heads."""

    agent = np.asarray(agent_logits, dtype=np.float64)
    area = np.asarray(area_logits, dtype=np.float64)
    count = agent.shape[0]
    if area.shape[0] != count:
        raise ValueError("dense safety candidate count mismatch")
    if agent.shape[-1] == 9:
        agent = agent[..., -1]
    agent = _sigmoid(agent.reshape(count, -1, 2))
    area = _sigmoid(area.reshape(count, -1, 2))
    channels = np.concatenate((agent, area), axis=-1)

    def summarize(values: np.ndarray) -> np.ndarray:
        horizon = values.shape[1]
        quarter = max(horizon // 4, 1)
        return np.stack(
            (
                values.mean(axis=1),
                values.max(axis=1),
                np.quantile(values, 0.75, axis=1),
                np.quantile(values, 0.90, axis=1),
                values[:, :quarter].mean(axis=1),
                values[:, -quarter:].mean(axis=1),
                (values > 0.50).mean(axis=1),
                (values > 0.80).mean(axis=1),
            ),
            axis=-1,
        ).reshape(count, -1)

    summary = summarize(channels)
    mean_ranks = np.stack([_rank(channels[..., index].mean(axis=1)) for index in range(4)])
    max_ranks = np.stack([_rank(channels[..., index].max(axis=1)) for index in range(4)])
    result = np.concatenate(
        (
            summary[candidate],
            summary[candidate] - summary[selected],
            summary[candidate] - summary[mother],
            mean_ranks[:, candidate],
            max_ranks[:, candidate],
        )
    ).astype(np.float32)
    if result.shape != (len(DENSE_SAFETY_FEATURE_NAMES),):
        raise AssertionError((result.shape, len(DENSE_SAFETY_FEATURE_NAMES)))
    return result


def candidate_features(
    payload: Dict[str, np.ndarray],
    candidate: int,
    factor_semantics: str = LEGACY_FACTOR_SEMANTICS,
) -> Tuple[np.ndarray, Dict[str, float]]:
    proposals = np.asarray(payload["proposals"], dtype=np.float64)
    deployed = np.asarray(payload["pdm_score"], dtype=np.float64)
    base = np.asarray(payload["base_pdm_score"], dtype=np.float64)
    factors = np.asarray(payload["candidate_subscores"], dtype=np.float64)
    residual = np.asarray(payload["residual_logits"], dtype=np.float64)
    if not 0 <= candidate < len(proposals):
        raise IndexError(f"candidate {candidate} outside {len(proposals)} proposals")
    factor_score, hard, soft = _factor_scores(factors, factor_semantics)
    selected = int(np.argmax(deployed))
    mother = int(np.argmax(base))
    geometry = np.stack([_geometry(proposal) for proposal in proposals])

    scalar = np.stack((deployed, base, residual, factor_score, hard, soft), axis=1)
    candidate_scalar = scalar[candidate]
    denominator = max(len(proposals) - 1, 1)
    feature = np.concatenate(
        (
            candidate_scalar,
            factors[candidate],
            geometry[candidate],
            candidate_scalar - scalar[selected],
            factors[candidate] - factors[selected],
            geometry[candidate] - geometry[selected],
            candidate_scalar - scalar[mother],
            factors[candidate] - factors[mother],
            geometry[candidate] - geometry[mother],
            np.asarray(
                [
                    _rank(deployed)[candidate],
                    _rank(base)[candidate],
                    _rank(factor_score)[candidate],
                    candidate / denominator,
                    deployed.max() - deployed[candidate],
                    base.max() - base[candidate],
                    abs(candidate - selected) / denominator,
                    abs(candidate - mother) / denominator,
                ],
                dtype=np.float64,
            ),
        )
    ).astype(np.float32)
    if feature.shape != (len(FEATURE_NAMES),):
        raise AssertionError((feature.shape, len(FEATURE_NAMES)))
    return feature, {
        "candidate": float(candidate),
        "selected": float(selected),
        "mother": float(mother),
        "base_gap": float(base.max() - base[candidate]),
        "hard_safety_delta": float(hard[candidate] - hard[mother]),
        "deployed_rank": float(_rank(deployed)[candidate]),
        "base_rank": float(_rank(base)[candidate]),
        "factor_rank": float(_rank(factor_score)[candidate]),
    }


def candidate_shortlist(
    payload: Dict[str, np.ndarray],
    top_k: int,
    factor_semantics: str = LEGACY_FACTOR_SEMANTICS,
) -> np.ndarray:
    deployed = np.asarray(payload["pdm_score"], dtype=np.float64)
    base = np.asarray(payload["base_pdm_score"], dtype=np.float64)
    factors = np.asarray(payload["candidate_subscores"], dtype=np.float64)
    factor_score, _, _ = _factor_scores(factors, factor_semantics)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    count = min(int(top_k), len(deployed))
    indices = np.concatenate(
        (
            np.argsort(deployed)[-count:],
            np.argsort(base)[-count:],
            np.argsort(factor_score)[-count:],
        )
    )
    return np.unique(indices)
