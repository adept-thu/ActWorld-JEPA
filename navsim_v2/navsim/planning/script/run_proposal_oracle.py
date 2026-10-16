"""Measure selected-versus-oracle scores for a fixed 32-proposal policy.

This diagnostic is deliberately separate from training.  It scores every
original and refined candidate with the NAVSIM simulator, but never feeds an
oracle index or score back into the model.
"""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.planning.script.proposal_oracle_guard import (
    apply_partition_allowlist,
    attach_partition_evidence,
    check_leakage_guard as _check_leakage_guard,
    identifier_evidence,
    metric_coverage,
    summarize,
)
from navsim.planning.training.dataset import Dataset

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

METRIC_NAMES = (
    "no_at_fault_collision",
    "drivable_area_compliance",
    "traffic_light_compliance",
    "driving_direction_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "epdms",
)


def _score_candidates(job: Tuple[str, np.ndarray]) -> np.ndarray:
    """Score one scene in an isolated process.

    The released training helper also loads anchor pseudo-label indices.  This
    diagnostic intentionally calls the lower-level scorer functions so that
    navtest requires only its official metric cache.
    """

    metric_cache_path, candidates = job
    from navsim.agents.drive_jepa_perception_based.score_module.compute_navsim_score import (
        after_score_proposals,
        before_score_proposals,
        load_metric_cache,
        scorer,
    )

    metric_cache = load_metric_cache(metric_cache_path)
    simulated_states = before_score_proposals(
        metric_cache, candidates, metric_cache.ego_state
    )
    final_scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        None,
        metric_cache.past_human_trajectory,
    )
    scores, _, _, _ = after_score_proposals(
        scorer, final_scores, metric_cache.ego_state, test=True
    )
    return scores


def _to_device(values: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in values.items()
    }


def _candidate_row(
    token: str,
    split: str,
    selected_index: int,
    original_scores: np.ndarray,
    refined_scores: np.ndarray,
) -> Dict[str, Any]:
    candidate_count = refined_scores.shape[0]
    if original_scores.shape != refined_scores.shape:
        raise ValueError("original and refined candidate score shapes differ")
    if original_scores.ndim != 2 or original_scores.shape[1] != len(METRIC_NAMES):
        raise ValueError(
            f"expected candidate scores [C, 9], got {original_scores.shape}"
        )
    if candidate_count != 32:
        raise ValueError(f"expected exactly 32 proposals, got {candidate_count}")
    if not 0 <= selected_index < candidate_count:
        raise ValueError(f"selected index {selected_index} is out of range")

    original_oracle_index = int(np.argmax(original_scores[:, -1]))
    refined_oracle_index = int(np.argmax(refined_scores[:, -1]))
    selected_metrics = refined_scores[selected_index]
    original_oracle_metrics = original_scores[original_oracle_index]
    refined_oracle_metrics = refined_scores[refined_oracle_index]

    row: Dict[str, Any] = {
        "token": token,
        "split": split,
        "candidate_count": candidate_count,
        "selected_index": selected_index,
        "original_oracle_index": original_oracle_index,
        "refined_oracle_index": refined_oracle_index,
        "selected_epdms": float(selected_metrics[-1]),
        "original_best_of_32_epdms": float(original_oracle_metrics[-1]),
        "refined_best_of_32_epdms": float(refined_oracle_metrics[-1]),
        "selected_to_refined_oracle_gap": float(
            refined_oracle_metrics[-1] - selected_metrics[-1]
        ),
        "refinement_oracle_delta": float(
            refined_oracle_metrics[-1] - original_oracle_metrics[-1]
        ),
    }
    for metric_index, metric_name in enumerate(METRIC_NAMES):
        row[f"selected_{metric_name}"] = float(selected_metrics[metric_index])
        row[f"original_oracle_{metric_name}"] = float(
            original_oracle_metrics[metric_index]
        )
        row[f"refined_oracle_{metric_name}"] = float(
            refined_oracle_metrics[metric_index]
        )
    return row


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    split = str(cfg.proposal_oracle.split)
    phase = str(cfg.proposal_oracle.phase)
    partition = str(cfg.proposal_oracle.partition)
    _check_leakage_guard(split, phase, partition)

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    metric_paths = metric_cache_loader.metric_cache_paths

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    agent.eval()

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    allowlist_logs = apply_partition_allowlist(
        scene_filter,
        partition=partition,
        val_logs=cfg.val_logs if partition == "validation" else None,
    )
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
    )
    valid_indices, _, _, token_evidence = metric_coverage(
        scene_loader.tokens, metric_paths
    )

    dataloader = DataLoader(
        Subset(dataset, valid_indices),
        batch_size=int(cfg.proposal_oracle.batch_size),
        num_workers=int(cfg.proposal_oracle.dataloader_workers),
        pin_memory=device.type == "cuda",
        shuffle=False,
    )

    rows: List[Dict[str, Any]] = []
    spawn_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(cfg.proposal_oracle.score_workers),
        mp_context=spawn_context,
    ) as score_pool:
        for features, targets in tqdm(dataloader, desc=f"{split} proposal oracle"):
            features = _to_device(features, device)
            with torch.inference_mode():
                predictions = agent.forward(features)

            original = predictions["proposals"].detach().float().cpu().numpy()
            refined = predictions["refined_proposals"].detach().float().cpu().numpy()
            selected = (
                predictions["selected_proposal"].detach().long().cpu().numpy()
            )
            tokens: Iterable[str] = targets["token"]
            jobs = []
            token_batch = list(tokens)
            for batch_index, token in enumerate(token_batch):
                all_candidates = np.concatenate(
                    [original[batch_index], refined[batch_index]], axis=0
                )
                jobs.append((str(metric_paths[token]), all_candidates))

            batch_scores = list(score_pool.map(_score_candidates, jobs))
            for batch_index, (token, scores) in enumerate(
                zip(token_batch, batch_scores)
            ):
                candidate_count = original.shape[1]
                rows.append(
                    _candidate_row(
                        token=token,
                        split=split,
                        selected_index=int(selected[batch_index]),
                        original_scores=scores[:candidate_count],
                        refined_scores=scores[candidate_count:],
                    )
                )

    evidence = {
        "partition": partition,
        **identifier_evidence(allowlist_logs, prefix="allowlist_log"),
        **token_evidence,
    }
    attach_partition_evidence(rows, evidence)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"proposal_oracle_{split}.csv"
    summary_path = output_dir / f"proposal_oracle_{split}_summary.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary = summarize(rows, split, phase, evidence)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Per-scene output: {csv_path}")
    print(f"Summary output: {summary_path}")


if __name__ == "__main__":
    main()
