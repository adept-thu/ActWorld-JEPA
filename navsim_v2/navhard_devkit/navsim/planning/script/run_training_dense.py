# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _resolve_trainer_params(
    params: DictConfig, strategy_override: Optional[Any] = None
) -> Dict[str, Any]:
    """Return one canonical Trainer kwargs mapping."""
    trainer_params = dict(params)
    if strategy_override is not None:
        trainer_params["strategy"] = strategy_override
    return trainer_params


def _is_single_device(devices: Any) -> bool:
    """Return whether Lightning's devices setting selects exactly one device."""
    if isinstance(devices, int):
        return devices == 1
    if isinstance(devices, str):
        return devices.strip() == "1"
    if isinstance(devices, (list, tuple)):
        return len(devices) == 1
    return False


def _filter_logs_by_sensor_availability(
    requested_logs: List[str], sensor_path: Path
) -> Tuple[List[str], List[str]]:
    """Split requested logs into available and missing sensor directories."""
    available_sensor_logs = {
        path.name for path in sensor_path.iterdir() if path.is_dir()
    }
    available = [name for name in requested_logs if name in available_sensor_logs]
    missing = [name for name in requested_logs if name not in available_sensor_logs]
    return available, missing


def _find_incomplete_sensor_logs(
    scene_frames_dicts: Dict[str, List[Dict[str, Any]]],
    sensor_path: Path,
    sensor_config: SensorConfig,
    existence_cache: Optional[Dict[Path, bool]] = None,
) -> Tuple[Dict[str, Path], Dict[Path, bool]]:
    """Find logs missing any sensor file required by the configured agent."""
    file_exists = existence_cache if existence_cache is not None else {}
    incomplete_logs: Dict[str, Path] = {}

    def exists(relative_path: Any) -> bool:
        if relative_path is None:
            return False
        path = sensor_path / Path(relative_path)
        if path not in file_exists:
            file_exists[path] = path.is_file()
        return file_exists[path]

    for scene_frames in scene_frames_dicts.values():
        log_name = scene_frames[0]["log_name"]
        if log_name in incomplete_logs:
            continue
        for frame_idx, frame in enumerate(scene_frames):
            required_sensors = set(sensor_config.get_sensors_at_iteration(frame_idx))
            for camera_name, camera in frame["cams"].items():
                if camera_name.lower() in required_sensors and not exists(
                    camera.get("data_path")
                ):
                    incomplete_logs[log_name] = Path(camera.get("data_path") or "<missing>")
                    break
            if log_name in incomplete_logs:
                break
            if "lidar_pc" in required_sensors and not exists(frame.get("lidar_path")):
                incomplete_logs[log_name] = Path(frame.get("lidar_path") or "<missing>")
                break

    return incomplete_logs, file_exists


def _drop_incomplete_sensor_logs(
    scene_loader: SceneLoader,
    sensor_path: Path,
    sensor_config: SensorConfig,
    existence_cache: Optional[Dict[Path, bool]] = None,
) -> Tuple[Dict[str, Path], Dict[Path, bool], int]:
    """Remove every scene belonging to an incomplete original-sensor log."""
    incomplete_logs, existence_cache = _find_incomplete_sensor_logs(
        scene_loader.scene_frames_dicts,
        sensor_path,
        sensor_config,
        existence_cache,
    )
    before = len(scene_loader.scene_frames_dicts)
    if incomplete_logs:
        scene_loader.scene_frames_dicts = {
            token: frames
            for token, frames in scene_loader.scene_frames_dicts.items()
            if frames[0]["log_name"] not in incomplete_logs
        }
    return incomplete_logs, existence_cache, before - len(scene_loader.scene_frames_dicts)


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_logs = [
            log_name
            for log_name in train_scene_filter.log_names
            if log_name in cfg.train_logs
        ]
    else:
        train_logs = list(cfg.train_logs)

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_logs = [
            log_name
            for log_name in val_scene_filter.log_names
            if log_name in cfg.val_logs
        ]
    else:
        val_logs = list(cfg.val_logs)

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)
    if cfg.get("skip_missing_sensor_logs", False):
        train_logs, missing_train_logs = _filter_logs_by_sensor_availability(
            train_logs, original_sensor_path
        )
        val_logs, missing_val_logs = _filter_logs_by_sensor_availability(
            val_logs, original_sensor_path
        )
        missing_logs = sorted(set(missing_train_logs + missing_val_logs))
        if missing_logs:
            logger.warning(
                "Skipping %d split logs without sensor directories under %s: %s",
                len(missing_logs),
                original_sensor_path,
                missing_logs,
            )

    train_scene_filter.log_names = train_logs
    val_scene_filter.log_names = val_logs
    sensor_config = agent.get_sensor_config()
    train_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=sensor_config,
    )

    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=sensor_config,
    )

    if cfg.get("skip_missing_sensor_logs", False):
        train_incomplete, existence_cache, train_scenes_dropped = (
            _drop_incomplete_sensor_logs(
                train_scene_loader,
                original_sensor_path,
                sensor_config,
            )
        )
        val_incomplete, _, val_scenes_dropped = _drop_incomplete_sensor_logs(
            val_scene_loader,
            original_sensor_path,
            sensor_config,
            existence_cache,
        )
        incomplete_logs = {**train_incomplete, **val_incomplete}
        if incomplete_logs:
            examples = [
                f"{log}: {path}"
                for log, path in list(sorted(incomplete_logs.items()))[:10]
            ]
            logger.warning(
                "Skipping %d incomplete sensor logs (%d train and %d validation "
                "scenes); first missing files: %s",
                len(incomplete_logs),
                train_scenes_dropped,
                val_scenes_dropped,
                examples,
            )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    memory_fraction = os.environ.get("GPU_MEMORY_FRACTION")
    if memory_fraction:
        fraction = float(memory_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("GPU_MEMORY_FRACTION must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(fraction)
        logger.info("Capped this process to %.1f%% of one GPU", 100.0 * fraction)

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    if agent._checkpoint_path is not None:
        agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
                cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    # Materialize the Hydra parameters before applying agent-specific overrides.
    # ``competition_training.yaml`` already defines ``strategy``, so passing a
    # second keyword directly to Trainer raises ``multiple values for strategy``.
    strategy_override = None
    if not isinstance(agent, TransfuserAgent):
        if _is_single_device(cfg.trainer.params.get("devices")):
            # Avoid DDP/NCCL synchronization on a one-GPU run. In particular,
            # its static-graph gradient synchronization can fail at the first
            # accumulated optimizer step even though all local backward passes
            # before that point succeeded.
            strategy_override = "auto"
        else:
            strategy_override = DDPStrategy(
                static_graph=True,
                timeout=datetime.timedelta(seconds=3600),
            )
    trainer_params = _resolve_trainer_params(
        cfg.trainer.params, strategy_override=strategy_override
    )
    trainer = pl.Trainer(
        **trainer_params,
        callbacks=agent.get_training_callbacks(),
    )

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.get('resume_ckpt_path', None)
    )


if __name__ == "__main__":
    main()
