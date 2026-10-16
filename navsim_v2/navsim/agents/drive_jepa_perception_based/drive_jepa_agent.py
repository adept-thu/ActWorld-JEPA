from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Dict, List, Union

import logging
import multiprocessing
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
from navsim.agents.drive_jepa_perception_based.drive_jepa_features import DriveJEPAFeatureBuilder, DriveJEPATargetBuilder
from navsim.agents.drive_jepa_perception_based.drive_jepa_model import DriveJEPAModel
from navsim.agents.transfuser.transfuser_loss import _agent_loss
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.utils import mean_time_every_5_calls

from .score_module.compute_navsim_score import get_scores, get_sub_score


logger = logging.getLogger(__name__)


class DriveJEPAAgent(AbstractAgent):
    # Subclasses can replace only the model at the construction seam without
    # instantiating the 307M-parameter V-JEPA backbone twice.
    model_class = DriveJEPAModel

    def __init__(
        self,
        config: DriveJEPAConfig,
        lr: float,
        checkpoint_path: str = None,
    ):
        super().__init__(config.trajectory_sampling)
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path

        cache_data = False

        if not cache_data:
            self._pad_model = self.model_class(config)

        if not cache_data and self._checkpoint_path == "":  # only for training
            self.bce_logit_loss = nn.BCEWithLogitsLoss()

            score_workers = int(getattr(config, "actworld_score_workers", 16))
            if score_workers < 1:
                raise ValueError("actworld_score_workers must be positive")
            self._score_timeout_seconds = float(
                getattr(config, "actworld_score_timeout_seconds", 600.0)
            )
            self._score_slow_log_seconds = float(
                getattr(config, "actworld_score_slow_log_seconds", 60.0)
            )
            if self._score_timeout_seconds <= 0.0:
                raise ValueError("actworld_score_timeout_seconds must be positive")
            if self._score_slow_log_seconds < 0.0:
                raise ValueError("actworld_score_slow_log_seconds must be nonnegative")
            # The executor starts lazily during the first simulator-scoring
            # call, after CUDA/NCCL have normally been initialized.  Spawn
            # avoids inheriting native GPU and thread-pool locks via fork.
            self._score_workers = score_workers
            self._score_mp_context = multiprocessing.get_context("spawn")
            self.persistent_pool = self._create_score_pool()

            metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/Drive-JEPA-cache/train_metric_cache_v2_pruned"))
            self.train_metric_cache_paths = metric_cache.metric_cache_paths
            self.test_metric_cache_paths = metric_cache.metric_cache_paths
        
        poses = np.load("./data/8192.npy")
        self.anchors = poses[:, 4::5]

    def name(self) -> str:
        """Inherited, see superclass."""
        return 'drive_jepa_perception_based_agent' 

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"
                ]
            self.load_state_dict({k.replace("agent._pad_model", "_pad_model"): v for k, v in state_dict.items()})

    def get_sensor_config(self):
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[2, 3],
            cam_l0=[3],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[3],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[3],
            lidar_pc=[],
        )

    def get_target_builders(self):
        return [DriveJEPATargetBuilder(config=self._config)]

    def get_feature_builders(self):
        return [DriveJEPAFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._pad_model(features)

    def _create_score_pool(self):
        return ProcessPoolExecutor(
            max_workers=self._score_workers,
            mp_context=self._score_mp_context,
        )

    def _reset_score_pool(self):
        """Replace a broken scorer pool without changing the scored batch."""

        pool = self.persistent_pool
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # pragma: no cover - compatibility with older Python
            pool.shutdown(wait=False)
        self.persistent_pool = self._create_score_pool()

    def _worker_map_with_recovery(self, data_points, score_function, label):
        """Score a batch, rebuilding the process pool once if a worker dies."""

        chunks = [data_points[i : i + 2] for i in range(0, len(data_points), 2)]
        tokens = [Path(point["token"]).stem for point in data_points]
        started = time.monotonic()
        for attempt in range(2):
            try:
                futures = [
                    self.persistent_pool.submit(score_function, chunk)
                    for chunk in chunks
                ]
                all_res = []
                for chunk, future in zip(chunks, futures):
                    try:
                        all_res.extend(
                            future.result(timeout=self._score_timeout_seconds)
                        )
                    except FuturesTimeoutError as exc:
                        chunk_tokens = [Path(point["token"]).stem for point in chunk]
                        raise TimeoutError(
                            f"{label} exceeded "
                            f"{self._score_timeout_seconds:.1f}s for "
                            f"tokens={chunk_tokens}"
                        ) from exc
                break
            except BrokenProcessPool:
                if attempt == 1:
                    raise
                logger.exception(
                    "%s worker pool broke; rebuilding once and replaying the "
                    "same deterministic batch, tokens=%s",
                    label,
                    tokens,
                )
                self._reset_score_pool()

        elapsed = time.monotonic() - started
        if elapsed >= self._score_slow_log_seconds:
            logger.warning(
                "Slow %s: %.1fs for %d scenes, tokens=%s",
                label,
                elapsed,
                len(data_points),
                tokens,
            )
        return all_res

    def worker_map(self, data_points):
        return self._worker_map_with_recovery(
            data_points,
            get_scores,
            "NAVSIM proposal scoring",
        )

    def compute_score(self, targets, proposals, test=True, label_only=False):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        target_trajectory = targets["trajectory"]
        proposals = proposals.detach()

        data_points = [
            {"token": metric_cache_paths[token], "poses": poses, "test": test}
            for token, poses in zip(targets["token"], proposals.float().cpu().numpy())
        ]

        all_res = self.worker_map(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)
        final_scores = target_scores[:, :, -1]
        best_scores = torch.amax(final_scores, dim=-1)
        scores_index = [res[-1] for res in all_res]

        if label_only:
            return final_scores, best_scores, target_scores

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]
            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)
            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)
            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)
            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas, scores_index

    def score_loss(self, pred_logit, agents_state, pred_area_logits, target_scores, gt_states, gt_valid, gt_ego_areas):
        if agents_state is not None:
            pred_states = agents_state[..., :-1].reshape(gt_states.shape)
            pred_logits = agents_state[..., -1:].reshape(gt_valid.shape)

            pred_l1_loss = F.l1_loss(pred_states, gt_states, reduction="none")[gt_valid]

            if len(pred_l1_loss):
                pred_l1_loss = pred_l1_loss.mean()
            else:
                pred_l1_loss = pred_states.mean() * 0

            pred_ce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_valid.to(torch.float32), reduction="mean")

        else:
            pred_ce_loss = 0
            pred_l1_loss = 0

        if pred_area_logits is not None:
            pred_area_logits = pred_area_logits.reshape(gt_ego_areas.shape)

            pred_area_loss = F.binary_cross_entropy_with_logits(
                pred_area_logits, gt_ego_areas.to(torch.float32), reduction="mean"
            )
        else:
            pred_area_loss = 0

        sub_score_loss = self.bce_logit_loss(
            pred_logit, target_scores[..., -pred_logit.shape[-1] :]
        )  # .mean()[..., -6:]
        final_score_loss = self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])  # .mean()

        return sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss

    def diversity_loss(self, proposals):
        dist = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)

        dist = dist + (dist == 0)

        # dist[dist==0]=10000

        inter_loss = -dist.amin(1).amin(1).mean()

        return inter_loss
        
    def trajectory_loss_topk(self, proposals, proposal_list, target_trajectory, config, final_scores): 
        with torch.no_grad():
            # Select top-k proposals per batch element based on final_scores
            k = min(4, proposals.size(1))
            topk_scores, topk_indices = torch.topk(final_scores, k=k, dim=1)
            topk_proposals = proposals.gather(
                dim=1,
                index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, proposals.size(2), proposals.size(3)),
            )

        trajectory_loss = 0

        min_loss_list = []
        inter_loss_list = []
        for proposals_i in proposal_list:
            # original loss using ground-truth trajectory (kept but unused)
            min_loss = (
                torch.linalg.norm(proposals_i - target_trajectory[:, None], dim=-1, ord=1)
                .mean(-1)
                .amin(1)
                .mean()
            )

            # Compute loss for each of the top-k proposals used as target_trajectory,
            # weighted directly by their scores in [0, 1] (no normalization).
            diff = proposals_i[:, None] - topk_proposals[:, :, None]  # (B, k, P, T, D)
            per_k_min = (
                torch.linalg.norm(diff, dim=-1, ord=1)  # (B, k, P, T)
                .mean(-1)                               # (B, k, P)
                .amin(-1)                               # (B, k)
            )
            weighted_per_k = per_k_min * topk_scores    # (B, k)
            min_loss += weighted_per_k.sum(dim=1).mean()  # scalar
            min_loss = 0.5 * min_loss

            inter_loss = self.diversity_loss(proposals_i)
            trajectory_loss = config.prev_weight * trajectory_loss + min_loss + inter_loss * config.inter_weight

            min_loss_list.append(min_loss)
            inter_loss_list.append(inter_loss)

        return trajectory_loss, min_loss, inter_loss, min_loss_list, inter_loss_list

    def trajectory_loss_anchors(self, proposal_list, target_trajectory, config, scores_index): 
        trajectory_loss = 0

        min_loss_list = []
        inter_loss_list = []
        for proposals_i, idx_arr in zip(proposal_list, scores_index):
            min_loss = (
                torch.linalg.norm(proposals_i - target_trajectory[:, None], dim=-1, ord=1)
                .mean(-1)
                .amin(1)
                .mean()
            )

            pseudo_min_loss = 0 
            # Use scores_index to build pseudo targets from anchors
            for i, idx_arr in enumerate(scores_index):
                if idx_arr.shape[0] == 0:
                    continue

                if idx_arr.shape[0] > 4:
                    sampled_idx = np.random.choice(idx_arr, size=4, replace=False)
                else:
                    sampled_idx = idx_arr

                # self.anchors: numpy; convert to torch on same device/dtype as proposals_i
                anchors_np = self.anchors[sampled_idx]
                pseudo_targets = torch.from_numpy(anchors_np).to(
                    device=proposals_i.device, dtype=proposals_i.dtype
                )

                # proposals_i: (B, P, T, D)
                # pseudo_targets: (K, T, D)
                diff = proposals_i[i, None] - pseudo_targets[:, None]  # (B, K, P, T, D)
                pseudo_min_loss += (
                    torch.linalg.norm(diff, dim=-1, ord=1)   # (B, K, P, T)
                    .mean(-1)                                # (B, K, P)
                    .amin(-1)                                # (B, K)
                    .mean()                                  # scalar
                )
            
            min_loss += 0.5 * pseudo_min_loss / len(scores_index) 

            inter_loss = self.diversity_loss(proposals_i)
            trajectory_loss = config.prev_weight * trajectory_loss + min_loss + inter_loss * config.inter_weight

            min_loss_list.append(min_loss)
            inter_loss_list.append(inter_loss)

        return trajectory_loss, min_loss, inter_loss, min_loss_list, inter_loss_list
    
    def pad_loss(self, targets: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config: DriveJEPAConfig):
        proposals = pred["proposals"]
        proposal_list = pred["proposal_list"]
        target_trajectory = targets["trajectory"]

        final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas, scores_index = self.compute_score(
            targets, proposals.detach(), test=False
        )
        # Kept out of ``loss_dict`` because Lightning expects logged values to
        # be scalar.  Planning extensions can consume the exact simulator
        # supervision without running the expensive proposal simulation twice.
        pred["_target_scores"] = target_scores
        pred["_target_final_scores"] = final_scores
        
        proposals = proposals.detach()
        
        trajectory_loss, min_loss, inter_loss, min_loss_list, inter_loss_list = self.trajectory_loss_anchors(proposal_list, target_trajectory, config, scores_index)

        min_loss0 = min_loss_list[0]
        inter_loss0 = inter_loss_list[0]
        # min_loss1 = min_loss_list[1]
        # inter_loss1 = inter_loss_list[1]

        if "pred_logit" in pred.keys():
            sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = self.score_loss(
                pred["pred_logit"],
                pred["pred_agents_states"],
                pred["pred_area_logit"],
                target_scores,
                gt_states,
                gt_valid,
                gt_ego_areas,
            )
        else:
            sub_score_loss = final_score_loss = pred_ce_loss = pred_l1_loss = pred_area_loss = 0

        if pred["agent_states"] is not None:
            agent_class_loss, agent_box_loss = _agent_loss(targets, pred, config)
        else:
            agent_class_loss = 0
            agent_box_loss = 0

        if pred["bev_semantic_map"] is not None:
            bev_semantic_loss = F.cross_entropy(pred["bev_semantic_map"], targets["bev_semantic_map"].long())
        else:
            bev_semantic_loss = 0

        loss = (
            config.trajectory_weight * trajectory_loss
            + config.sub_score_weight * sub_score_loss
            + config.final_score_weight * final_score_loss
            + config.pred_ce_weight * pred_ce_loss
            + config.pred_l1_weight * pred_l1_loss
            + config.pred_area_weight * pred_area_loss
            + config.agent_class_weight * agent_class_loss
            + config.agent_box_weight * agent_box_loss
            + config.bev_semantic_weight * bev_semantic_loss
        )

        pdm_score = pred["pdm_score"].detach()
        top_proposals = torch.argmax(pdm_score, dim=1)
        score = final_scores[np.arange(len(final_scores)), top_proposals].mean()
        best_score = best_scores.mean()

        loss_dict = {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            "sub_score_loss": sub_score_loss,
            "final_score_loss": final_score_loss,
            "pred_ce_loss": pred_ce_loss,
            "pred_l1_loss": pred_l1_loss,
            "pred_area_loss": pred_area_loss,
            "inter_loss0": inter_loss0,
            "inter_loss": inter_loss,
            "min_loss0": min_loss0,
            "min_loss": min_loss,
            "score": score,
            "best_score": best_score,
        }

        return loss_dict

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
    ) -> Dict:
        return self.pad_loss(targets, pred, self._config)

    def get_optimizers(self):
        # Smaller lr for vjepa encoder
        return torch.optim.Adam(
            [
                {"params": self._pad_model._backbone.parameters(), "lr": 0.1 * self._lr},
                {"params": [p for n, p in self._pad_model.named_parameters() if "backbone" not in n], "lr": self._lr},
            ],
            lr=self._lr,
        )

    def get_training_callbacks(self):
        return []

    def __del__(self):
        """Cleanup persistent pool on deletion."""
        if hasattr(self, "persistent_pool"):
            self.persistent_pool.shutdown(wait=True)
