"""Create a strict NAVSIM-v1 leaderboard submission with worker parallelism.

This follows the official v1.1 submission schema, but distributes physical
logs through NAVSIM's configured worker pool.  It fails closed: any duplicate,
missing, or failed token prevents submission.pkl from being written.
"""
from __future__ import annotations

from pathlib import Path
import logging
import multiprocessing
import os
import pickle
import uuid

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import torch.nn as nn

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.planning.script.builders.worker_pool_builder import build_worker


logger = logging.getLogger(__name__)
CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def move_all_modules_to_gpu(instance: object) -> None:
    for name in dir(instance):
        value = getattr(instance, name)
        if isinstance(value, nn.Module):
            setattr(instance, name, value.cuda())


def generate_trajectories(args: list[dict]) -> list[tuple[str, Trajectory]]:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    worker_devices = [
        value.strip()
        for value in os.environ.get("NAVSIM_WORKER_CUDA_DEVICES", "").split(",")
        if value.strip()
    ]
    if worker_devices:
        identity = multiprocessing.current_process()._identity
        worker_index = (identity[-1] - 1) if identity else os.getpid()
        device = int(worker_devices[worker_index % len(worker_devices)])
        torch.cuda.set_device(device)
        logger.info("Assigned worker %s to CUDA device %d", thread_id, device)

    cfg: DictConfig = args[0]["cfg"]
    log_names = [item["log_file"] for item in args]
    requested_tokens = [token for item in args for token in item["tokens"]]

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    move_all_modules_to_gpu(agent)

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = requested_tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    output: list[tuple[str, Trajectory]] = []
    for index, token in enumerate(scene_loader.tokens, start=1):
        logger.info(
            "Generating trajectory %d / %d in thread_id=%s node_id=%d",
            index,
            len(scene_loader.tokens),
            thread_id,
            node_id,
        )
        agent_input = scene_loader.get_agent_input_from_token(token)
        if agent.requires_scene:
            raise RuntimeError(
                "leaderboard generation forbids annotated Scene access"
            )
        trajectory = agent.compute_trajectory(agent_input)
        if not isinstance(trajectory, Trajectory):
            raise TypeError(f"invalid trajectory for token {token}: {type(trajectory)}")
        output.append((token, trajectory))
    return output


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    worker = build_worker(cfg)
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    expected_tokens = list(scene_loader.tokens)
    data_points = [
        {"cfg": cfg, "log_file": log_file, "tokens": token_list}
        for log_file, token_list in scene_loader.get_tokens_list_per_log().items()
    ]
    rows = worker_map(worker, generate_trajectories, data_points)
    predictions: dict[str, Trajectory] = {}
    duplicates: list[str] = []
    for token, trajectory in rows:
        if token in predictions:
            duplicates.append(token)
        predictions[token] = trajectory

    expected = set(expected_tokens)
    actual = set(predictions)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if duplicates or missing or unexpected or len(predictions) != len(expected_tokens):
        raise RuntimeError(
            "invalid generated token set: "
            f"expected={len(expected_tokens)} actual={len(predictions)} "
            f"duplicates={len(duplicates)} missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )

    submission = {
        "team_name": cfg.team_name,
        "authors": cfg.authors,
        "email": cfg.email,
        "institution": cfg.institution,
        "country / region": cfg.country,
        "predictions": [predictions],
    }
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / "submission.pkl"
    temporary = output_dir / "submission.pkl.tmp"
    with temporary.open("wb") as stream:
        pickle.dump(submission, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, filename)
    logger.info(
        "Saved strict leaderboard submission with %d trajectories to %s",
        len(predictions),
        filename,
    )


if __name__ == "__main__":
    main()
