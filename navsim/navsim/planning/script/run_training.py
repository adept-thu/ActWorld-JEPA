import os
import random
import re
from collections import Counter
from typing import Tuple
from pathlib import Path
import logging

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _physical_drive_log(log_name: str) -> str:
    """Collapse NAVSIM segment logs onto their originating physical drive."""
    return re.sub(r"_\d+_\d+$", "", log_name)


def _load_r94_anchor_artifact(r94_anchor_path: str):
    """Load a frozen NAVTRAIN-only r94 decision table.

    The official NAVTRAIN metric cache covers a strict subset of the much
    denser frame-level training split.  Keeping the dataset on exactly these
    tokens avoids inventing labels for neighbouring frames.
    """
    path = Path(r94_anchor_path)
    if "navtest" in str(path).lower():
        raise ValueError("NAVTEST artifacts are forbidden as r94 anchors")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    protocol = str(artifact.get("protocol", ""))
    if "r94" not in protocol or "NAVTRAIN" not in protocol or "no NAVTEST" not in protocol:
        raise ValueError("r94 anchor artifact is not NAVTRAIN-only")
    tokens = [str(token) for token in artifact["tokens"]]
    indices = artifact["indices"]
    if len(tokens) != len(indices) or len(tokens) != len(set(tokens)):
        raise ValueError("r94 anchor artifact has invalid or duplicate token rows")
    return artifact


def _build_physical_log_sampler(train_data, cfg: DictConfig):
    """Build a tempered inverse-frequency sampler that is safe under torchrun.

    Each rank draws an independent shard-sized stream.  Trainer-side sampler
    replacement is disabled below so these physical-drive weights are not lost.
    """
    if hasattr(train_data, "_scene_loader"):
        loader = train_data._scene_loader
        tokens = loader.tokens
        log_names = [
            loader.scene_frames_dicts[token][0]["log_name"] for token in tokens
        ]
    elif hasattr(train_data, "_valid_cache_paths"):
        tokens = train_data.tokens
        log_names = [
            train_data._valid_cache_paths[token].parent.name for token in tokens
        ]
    else:
        raise TypeError("physical-log balancing requires a NAVSIM Dataset")

    groups = [_physical_drive_log(str(name)) for name in log_names]
    counts = Counter(groups)
    alpha = float(cfg.get("physical_log_balance_alpha", 0.5))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("physical_log_balance_alpha must be in [0, 1]")
    weights = torch.tensor(
        [counts[group] ** (-alpha) for group in groups], dtype=torch.double
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    samples_per_rank = (len(train_data) + world_size - 1) // world_size
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed) + 1009 * rank)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples_per_rank,
        replacement=True,
        generator=generator,
    )
    logger.info(
        "Physical-log sampler: %d groups, alpha=%.3f, scenes/group min=%d max=%d, rank=%d/%d",
        len(counts), alpha, min(counts.values()), max(counts.values()), rank, world_size,
    )
    return sampler, rank, world_size


class _PhysicalLogTargetDataset(torch.utils.data.Dataset):
    """Attach robust-training metadata without changing cached assets."""

    def __init__(
        self,
        dataset,
        r94_anchor_path: str = "",
        include_physical_log_id: bool = True,
    ):
        self._dataset = dataset
        self._include_physical_log_id = include_physical_log_id
        if hasattr(dataset, "_scene_loader"):
            loader = dataset._scene_loader
            tokens = loader.tokens
            log_names = [
                loader.scene_frames_dicts[token][0]["log_name"] for token in tokens
            ]
        elif hasattr(dataset, "_valid_cache_paths"):
            tokens = dataset.tokens
            log_names = [
                dataset._valid_cache_paths[token].parent.name for token in tokens
            ]
        else:
            raise TypeError("physical-log targets require a NAVSIM Dataset")
        groups = [_physical_drive_log(str(name)) for name in log_names]
        group_index = {
            group: index for index, group in enumerate(sorted(set(groups)))
        }
        self.group_count = len(group_index)
        self._group_ids = [group_index[group] for group in groups]
        logger.info("Physical-log target ids: %d groups", self.group_count)
        self._r94_anchor_ids = None
        if r94_anchor_path:
            artifact = _load_r94_anchor_artifact(r94_anchor_path)
            anchor_by_token = {
                str(token): int(index)
                for token, index in zip(artifact["tokens"], artifact["indices"])
            }
            missing = [str(token) for token in tokens if str(token) not in anchor_by_token]
            if missing:
                raise ValueError(f"r94 anchor artifact misses {len(missing)} dataset tokens")
            self._r94_anchor_ids = [anchor_by_token[str(token)] for token in tokens]
            logger.info("Frozen r94 anchor ids: %d scenes", len(self._r94_anchor_ids))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        features, targets = self._dataset[index]
        targets = dict(targets)
        if self._include_physical_log_id:
            targets["actworld_physical_log_id"] = torch.tensor(
                self._group_ids[index], dtype=torch.long
            )
        if self._r94_anchor_ids is not None:
            targets["actworld_r94_anchor_index"] = torch.tensor(
                self._r94_anchor_ids[index], dtype=torch.long
            )
        return features, targets


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    r94_anchor_path = str(cfg.get("r94_anchor_path", ""))
    if r94_anchor_path:
        artifact = _load_r94_anchor_artifact(r94_anchor_path)
        anchor_tokens = [str(token) for token in artifact["tokens"]]
        train_scene_filter.tokens = anchor_tokens
        val_scene_filter.tokens = anchor_tokens
        logger.info(
            "Restricted SceneLoaders to %d frozen r94 NAVTRAIN supervision tokens",
            len(anchor_tokens),
        )

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
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

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

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
    physical_log_balance = bool(cfg.get("physical_log_balance", False))
    r94_anchor_path = str(cfg.get("r94_anchor_path", ""))
    if r94_anchor_path and len(train_data) == 0:
        if len(val_data) == 0:
            raise ValueError("frozen r94 anchor tokens matched neither NAVTRAIN partition")
        # The frozen decision table was produced from NAVTRAIN calibration
        # logs, which belong to the configured validation-log partition.  It
        # is nevertheless training supervision here; independent e--n
        # NAVTRAIN slices remain outside this dataset for the frozen audit.
        train_data = val_data
        logger.info(
            "Using %d frozen r94 NAVTRAIN calibration scenes for training",
            len(train_data),
        )
    if physical_log_balance:
        train_sampler, rank, world_size = _build_physical_log_sampler(train_data, cfg)
        if bool(cfg.get("physical_log_group_targets", False)) or r94_anchor_path:
            train_data = _PhysicalLogTargetDataset(train_data, r94_anchor_path)
            val_data = _PhysicalLogTargetDataset(
                val_data, r94_anchor_path, include_physical_log_id=False
            )
            logger.info("Enabled physical-log targets for robust group training")
        train_dataloader = DataLoader(
            train_data,
            **cfg.dataloader.params,
            sampler=train_sampler,
            shuffle=False,
            drop_last=True,
        )
        val_sampler = (
            DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)
            if world_size > 1 else None
        )
        val_dataloader = DataLoader(
            val_data, **cfg.dataloader.params, sampler=val_sampler, shuffle=False, drop_last=True
        )
    else:
        # Frozen r94 supervision is useful independently of physical-log
        # rebalancing.  Previously the wrapper that injects the anchor index
        # lived only in the balancing branch, which made an r94-anchored run
        # impossible on the intentionally restricted NAVTRAIN token set.
        if r94_anchor_path:
            include_group_id = bool(cfg.get("physical_log_group_targets", False))
            train_data = _PhysicalLogTargetDataset(
                train_data,
                r94_anchor_path,
                include_physical_log_id=include_group_id,
            )
            val_data = _PhysicalLogTargetDataset(
                val_data,
                r94_anchor_path,
                include_physical_log_id=False,
            )
            logger.info(
                "Enabled frozen r94 anchor targets without physical-log balancing"
            )
        train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False, drop_last=True)
    logger.info("Num training samples: %d", len(train_data))
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building wandb logger")
    log_config = {
        "agent": OmegaConf.to_container(cfg.agent, resolve=True),
        "trainer": OmegaConf.to_container(cfg.trainer, resolve=True),
    }
    wandb_dir = Path(cfg.output_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(
        name=cfg.experiment_name,
        save_dir=str(wandb_dir),
        project="navsim_v1",
    )
    wandb_logger.log_hyperparams(log_config)

    checkpoint_kwargs = {}
    if "checkpoint" in cfg.trainer and cfg.trainer.checkpoint is not None:
        checkpoint_kwargs = OmegaConf.to_container(cfg.trainer.checkpoint, resolve=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.output_dir + "/checkpoints/",
        **checkpoint_kwargs,
    )

    logger.info("Building Trainer")
    trainer_params = cfg.trainer.params
    if physical_log_balance:
        trainer_params = OmegaConf.to_container(trainer_params, resolve=True)
        trainer_params["use_distributed_sampler"] = False
    trainer = pl.Trainer(
        **trainer_params,
        logger=wandb_logger,
        callbacks=[checkpoint_cb] + agent.get_training_callbacks(),
    )

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.resume_checkpoint_path,
    )


if __name__ == "__main__":
    main()
