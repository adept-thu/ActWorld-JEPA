"""NAVSIM v1 integration and training objectives for ActWorld-JEPA."""

from pathlib import Path
import copy
import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
from navsim.common.dataclasses import SensorConfig

from .actworld_config import ActWorldJEPAConfig
from .actworld_features import ActWorldJEPAFeatureBuilder, ActWorldJEPATargetBuilder
from .actworld_model import ActWorldJEPAModel


class ActWorldJEPAAgent(DriveJEPAAgent):
    """Drive-JEPA v1 agent with a latent rollout/evaluate/refine selector."""

    model_class = ActWorldJEPAModel

    _PLANNING_HEAD_PREFIXES = (
        "_trajectory_head.",
        "hist_encoding.",
        "init_feature.",
    )

    @classmethod
    def _is_planning_head_parameter(cls, name: str) -> bool:
        return name.startswith(cls._PLANNING_HEAD_PREFIXES)

    @staticmethod
    def _safe_oracle_proposal_distillation_loss(
        proposals: torch.Tensor,
        baseline_scores: torch.Tensor,
        simulator_targets: torch.Tensor,
        margin: float,
        max_gain: float,
    ):
        """Distill top-1 toward a better candidate without safety regression."""
        if margin < 0 or max_gain <= 0:
            raise ValueError("safe-oracle margin must be nonnegative and max_gain positive")
        if proposals.ndim != 4 or baseline_scores.shape != proposals.shape[:2]:
            raise ValueError("invalid proposal or baseline-score shape")
        if simulator_targets.shape[:2] != proposals.shape[:2] or simulator_targets.shape[-1] < 6:
            raise ValueError("invalid simulator-target shape")

        rows = torch.arange(proposals.shape[0], device=proposals.device)
        baseline_index = baseline_scores.float().argmax(dim=1)
        channels = simulator_targets.float()
        baseline_channels = channels[rows, baseline_index]
        # NC, DAC, TTC and comfort must all be no worse than the current top-1.
        safety_channels = torch.tensor((0, 1, 3, 4), device=channels.device)
        safety_ok = (
            channels.index_select(-1, safety_channels)
            >= baseline_channels.index_select(-1, safety_channels)[:, None]
        ).all(dim=-1)
        score = channels[..., -1]
        baseline_score = score[rows, baseline_index]
        eligible = safety_ok & (score >= baseline_score[:, None] + margin)
        teacher_index = score.masked_fill(~eligible, float("-inf")).argmax(dim=1)
        valid = eligible.any(dim=1)
        if not bool(valid.any()):
            zero = proposals.sum() * 0.0
            return zero, valid.float().mean(), zero.detach()

        student = proposals[rows, baseline_index]
        teacher = proposals.detach()[rows, teacher_index]
        per_sample = F.smooth_l1_loss(student, teacher, reduction="none").mean(
            dim=(1, 2)
        )
        gain = (score[rows, teacher_index] - baseline_score).clamp(
            min=0.0, max=max_gain
        )
        weights = gain / max_gain
        loss = (per_sample[valid] * weights[valid]).mean()
        return loss, valid.float().mean(), gain[valid].mean()

    @staticmethod
    def _safe_oracle_refinement_distillation_loss(
        proposals: torch.Tensor,
        refined: torch.Tensor,
        baseline_scores: torch.Tensor,
        simulator_targets: torch.Tensor,
        top_k: int,
        margin: float,
        max_gain: float,
    ):
        """Teach each shortlisted plan a bounded, safety-preserving correction.

        Candidate ranking remains defined on the original proposals.  For each
        top-k candidate, the target is the highest-scoring original candidate
        whose NC/DAC/TTC/comfort factors are all no worse than that candidate.
        """
        if top_k < 2 or margin < 0 or max_gain <= 0:
            raise ValueError("invalid safe-oracle refinement configuration")
        if proposals.shape != refined.shape or proposals.ndim != 4:
            raise ValueError("invalid original/refined proposal shape")
        if baseline_scores.shape != proposals.shape[:2]:
            raise ValueError("invalid baseline-score shape")
        if simulator_targets.shape[:2] != proposals.shape[:2] or simulator_targets.shape[-1] < 6:
            raise ValueError("invalid simulator-target shape")

        top_k = min(int(top_k), proposals.shape[1])
        shortlist = baseline_scores.float().topk(top_k, dim=1).indices
        rows = torch.arange(proposals.shape[0], device=proposals.device)[:, None]
        channels = simulator_targets.float()[rows, shortlist]
        score = channels[..., -1]
        safety = channels[..., (0, 1, 3, 4)]
        # [B, source, teacher]: teacher must dominate the source on safety.
        safety_ok = (safety[:, None] >= safety[:, :, None]).all(dim=-1)
        gain = score[:, None] - score[:, :, None]
        eligible = safety_ok & (gain >= margin)
        teacher_position = score[:, None].expand(-1, top_k, -1).masked_fill(
            ~eligible, float("-inf")
        ).argmax(dim=-1)
        valid = eligible.any(dim=-1)
        if not bool(valid.any()):
            zero = refined.sum() * 0.0
            return zero, valid.float().mean(), zero.detach()

        source_index = shortlist
        teacher_index = shortlist.gather(1, teacher_position)
        student = refined[rows, source_index]
        teacher = proposals.detach()[rows, teacher_index]
        per_item = F.smooth_l1_loss(student, teacher, reduction="none").mean(
            dim=(2, 3)
        )
        realized_gain = score.gather(1, teacher_position) - score
        weights = realized_gain.clamp(min=0.0, max=max_gain) / max_gain
        loss = (per_item[valid] * weights[valid]).mean()
        return loss, valid.float().mean(), realized_gain[valid].mean()

    def __init__(
        self,
        config: ActWorldJEPAConfig,
        lr: float,
        checkpoint_path: str = "",
    ) -> None:
        checkpoint_path = checkpoint_path or ""
        super().__init__(config=config, lr=lr, checkpoint_path=checkpoint_path)
        self._config = config
        self._teacher_current_future_interaction_v2 = None
        if checkpoint_path == "" and config.baseline_checkpoint_path:
            self._load_state(config.baseline_checkpoint_path, baseline_only=True)
        if checkpoint_path == "" and config.actworld_freeze_drive_jepa:
            if config.actworld_planning_adaptation_only and not (
                config.actworld_freeze_planner_during_proposal_adaptation
                and config.actworld_unfreeze_planning_heads
            ):
                raise ValueError(
                    "planning-adaptation-only requires a frozen ActWorld planner "
                    "and unfrozen Drive-JEPA planning heads"
                )
            if (
                config.actworld_freeze_planner_during_proposal_adaptation
                and not config.actworld_unfreeze_planning_heads
            ):
                raise ValueError(
                    "freezing the planner during proposal adaptation requires "
                    "actworld_unfreeze_planning_heads"
                )
            for name, parameter in self._pad_model.named_parameters():
                current_future_parameter = name.startswith(
                    (
                        "_actworld_planner.current_projection.",
                        "_actworld_planner.current_future_interaction.",
                    )
                )
                current_future_v2_parameter = name.startswith(
                    (
                        "_actworld_planner.current_future_interaction_v2.",
                        "_actworld_planner.future_compatibility_head.",
                        "_actworld_planner.current_future_rank_head.",
                        "_actworld_planner.current_future_factorized_rank_head.",
                        "_actworld_planner.current_future_progress_head.",
                    )
                )
                current_future_rank_parameter = name.startswith(
                    "_actworld_planner.current_future_rank_head."
                )
                current_future_factorized_rank_parameter = name.startswith(
                    "_actworld_planner.current_future_factorized_rank_head."
                )
                current_future_progress_parameter = name.startswith(
                    "_actworld_planner.current_future_progress_head."
                )
                current_future_rank_representation_parameter = name.startswith(
                    "_actworld_planner.current_future_interaction_v2."
                )
                risk_aware_v3_parameter = name.startswith(
                    (
                        "_actworld_planner.current_future_interaction_v2.",
                        "_actworld_planner.risk_aware_compatibility_head.",
                    )
                )
                frozen_risk_veto_parameter = name.startswith(
                    "_actworld_planner.risk_aware_compatibility_head."
                )
                refinement_parameter = name.startswith(
                    (
                        "_actworld_planner.refinement_pool.",
                        "_actworld_planner.refinement_head.",
                        "_actworld_planner.refinement_gate_logits",
                    )
                )
                planner_parameter = (
                    name.startswith("_actworld_planner.")
                    and not config.actworld_freeze_planner_during_proposal_adaptation
                )
                preference_parameter = name.startswith(
                    "_actworld_planner.preference_head."
                ) and not config.actworld_freeze_planner_during_proposal_adaptation
                planning_head_parameter = (
                    config.actworld_unfreeze_planning_heads
                    and self._is_planning_head_parameter(name)
                )
                if config.actworld_planning_last_layer_only:
                    planning_head_parameter = (
                        config.actworld_unfreeze_planning_heads
                        and name.startswith("_trajectory_head.")
                        and ".traj_decoder.mlp.6." in name
                    )
                if config.actworld_refinement_adaptation_only:
                    parameter.requires_grad_(refinement_parameter)
                elif config.actworld_frozen_risk_veto_only:
                    parameter.requires_grad_(frozen_risk_veto_parameter)
                elif config.actworld_current_future_progress_only:
                    parameter.requires_grad_(current_future_progress_parameter)
                elif config.actworld_current_future_factorized_rank_only:
                    parameter.requires_grad_(current_future_factorized_rank_parameter)
                elif config.actworld_current_future_rank_joint:
                    parameter.requires_grad_(
                        current_future_rank_parameter
                        or current_future_progress_parameter
                        or current_future_rank_representation_parameter
                    )
                elif config.actworld_current_future_rank_only:
                    parameter.requires_grad_(
                        current_future_rank_parameter or current_future_progress_parameter
                    )
                elif config.actworld_risk_aware_v3_only:
                    parameter.requires_grad_(risk_aware_v3_parameter)
                elif config.actworld_current_future_v2_only:
                    parameter.requires_grad_(current_future_v2_parameter)
                elif config.actworld_planning_adaptation_only:
                    parameter.requires_grad_(planning_head_parameter)
                elif config.actworld_current_future_fusion_only:
                    parameter.requires_grad_(current_future_parameter)
                else:
                    parameter.requires_grad_(
                        planning_head_parameter
                        or (
                            preference_parameter
                            if config.actworld_preference_head_only
                            else planner_parameter
                        )
                    )
            if config.actworld_current_future_teacher_anchor_loss_weight > 0:
                teacher = copy.deepcopy(
                    self._pad_model._actworld_planner.current_future_interaction_v2
                ).eval()
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)
                self._teacher_current_future_interaction_v2 = teacher
            self._pad_model.train(self.training)

    def name(self) -> str:
        return "actworld_jepa_v1_agent"

    @staticmethod
    def _checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
        checkpoint: Dict[str, Any] = torch.load(path, map_location=torch.device("cpu"))
        if "state_dict" not in checkpoint or not isinstance(checkpoint["state_dict"], dict):
            raise ValueError(f"Checkpoint {path!r} does not contain a Lightning state_dict")
        return {
            key.replace("agent._pad_model", "_pad_model"): value
            for key, value in checkpoint["state_dict"].items()
            if "agent._teacher_current_future_interaction_v2." not in key
        }

    def _load_state(self, path: str, *, baseline_only: bool) -> None:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        incompatible = self.load_state_dict(self._checkpoint_state(path), strict=False)
        preference_prefix = "_pad_model._actworld_planner.preference_head."
        outcome_prefix = "_pad_model._actworld_planner.outcome_predictor."
        outcome_factor_prefix = "_pad_model._actworld_planner.outcome_factor_head."
        outcome_delta_prefix = "_pad_model._actworld_planner.outcome_delta_head."
        outcome_action_delta_prefix = (
            "_pad_model._actworld_planner.outcome_action_delta_head."
        )
        temporal_action_delta_prefix = (
            "_pad_model._actworld_planner.temporal_action_jepa_head."
        )
        current_projection_prefix = (
            "_pad_model._actworld_planner.current_projection."
        )
        current_future_interaction_prefix = (
            "_pad_model._actworld_planner.current_future_interaction."
        )
        current_future_v2_prefixes = (
            "_pad_model._actworld_planner.current_future_interaction_v2.",
            "_pad_model._actworld_planner.future_compatibility_head.",
            "_pad_model._actworld_planner.current_future_rank_head.",
            "_pad_model._actworld_planner.current_future_factorized_rank_head.",
            "_pad_model._actworld_planner.current_future_progress_head.",
        )
        risk_aware_v3_prefix = (
            "_pad_model._actworld_planner.risk_aware_compatibility_head."
        )
        if baseline_only:
            unexpected = list(incompatible.unexpected_keys)
            invalid_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith("_pad_model._actworld_planner.")
            ]
            if unexpected or invalid_missing:
                raise RuntimeError(
                    "Drive-JEPA v1 warm start is incompatible: "
                    f"unexpected={unexpected}, missing={invalid_missing}"
                )
        else:
            allow_untrained_preference = (
                float(self._config.actworld_preference_residual_scale) == 0.0
                and float(self._config.actworld_preference_loss_weight) == 0.0
            )
            invalid_missing = [
                key
                for key in incompatible.missing_keys
                if not (
                    key.startswith(outcome_prefix)
                    or key.startswith(outcome_factor_prefix)
                    or key.startswith(outcome_delta_prefix)
                    or key.startswith(outcome_action_delta_prefix)
                    or key.startswith(temporal_action_delta_prefix)
                    or key.startswith(current_projection_prefix)
                    or key.startswith(current_future_interaction_prefix)
                    or key.startswith(current_future_v2_prefixes)
                    or key.startswith(risk_aware_v3_prefix)
                    or (
                        allow_untrained_preference
                        and key.startswith(preference_prefix)
                    )
                )
            ]
            unexpected = list(incompatible.unexpected_keys)
            if unexpected or invalid_missing:
                raise RuntimeError(
                    "ActWorld-JEPA v1 checkpoint is incompatible: "
                    f"unexpected={unexpected}, missing={invalid_missing}"
                )

    def initialize(self) -> None:
        if self._checkpoint_path:
            self._load_state(self._checkpoint_path, baseline_only=False)

    def get_sensor_config(self) -> SensorConfig:
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
        return [ActWorldJEPATargetBuilder(config=self._config)]

    def get_feature_builders(self):
        return [ActWorldJEPAFeatureBuilder(config=self._config)]

    @staticmethod
    def _soft_rank_loss(
        logits: torch.Tensor, target_scores: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("actworld_rank_temperature must be positive")
        targets = torch.softmax(target_scores.float() / temperature, dim=1)
        return -(targets * torch.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()

    @staticmethod
    def _coarse_to_fine_rank_loss(
        predicted_scores: torch.Tensor,
        target_scores: torch.Tensor,
        baseline_scores: torch.Tensor,
        top_k: int,
        temperature: float,
    ) -> torch.Tensor:
        """Rank simulator-best candidates against hard negatives in a coarse shortlist."""
        if top_k < 2:
            raise ValueError("actworld_hard_negative_top_k must be at least two")
        if temperature <= 0:
            raise ValueError("actworld_hard_negative_temperature must be positive")
        top_k = min(top_k, predicted_scores.shape[1])
        shortlist = baseline_scores.float().topk(k=top_k, dim=1).indices
        predicted = predicted_scores.float().gather(1, shortlist)
        targets = target_scores.float().gather(1, shortlist)
        teacher_best = targets.argmax(dim=1)
        rows = torch.arange(predicted.shape[0], device=predicted.device)
        best_prediction = predicted[rows, teacher_best]
        best_target = targets[rows, teacher_best]
        target_gap = (best_target[:, None] - targets).clamp_min(0.0)
        valid = target_gap > 1e-4
        prediction_gap = best_prediction[:, None] - predicted
        pair_loss = F.softplus(-prediction_gap / temperature)
        # Simulator gaps focus learning on meaningful mistakes while retaining
        # all hard negatives in the Drive-JEPA coarse shortlist.
        weights = target_gap.clamp_max(1.0)
        denominator = (weights * valid).sum().clamp_min(1.0)
        return (pair_loss * weights * valid).sum() / denominator

    @staticmethod
    def _mean_tail_risk(
        per_scene: torch.Tensor,
        fraction: float,
        weight: float,
    ) -> torch.Tensor:
        """Blend mean loss with batch-CVaR over the hardest scenes."""
        mean = per_scene.mean()
        if weight == 0.0:
            return mean
        count = max(1, int(math.ceil(per_scene.numel() * fraction)))
        tail = per_scene.flatten().topk(count, largest=True).values.mean()
        return (1.0 - weight) * mean + weight * tail

    def _group_dro_scene_weights(
        self,
        per_scene: torch.Tensor,
        group_ids: torch.Tensor,
        num_groups: int,
        eta: float,
    ) -> torch.Tensor:
        """Update adversarial physical-log weights and return local weights."""
        group_ids = group_ids.to(device=per_scene.device, dtype=torch.long).view(-1)
        if group_ids.shape[0] != per_scene.shape[0]:
            raise ValueError("physical-log ids must match the scene batch")
        if torch.any(group_ids < 0) or torch.any(group_ids >= num_groups):
            raise ValueError("physical-log id is outside configured group range")

        detached = per_scene.detach().float()
        group_sum = detached.new_zeros(num_groups)
        group_count = detached.new_zeros(num_groups)
        group_sum.scatter_add_(0, group_ids, detached)
        group_count.scatter_add_(0, group_ids, torch.ones_like(detached))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(group_sum)
            torch.distributed.all_reduce(group_count)

        if (
            not hasattr(self, "_actworld_group_dro_logits")
            or self._actworld_group_dro_logits.shape[0] != num_groups
            or self._actworld_group_dro_logits.device != per_scene.device
        ):
            self._actworld_group_dro_logits = per_scene.new_zeros(num_groups)
        seen = group_count > 0
        with torch.no_grad():
            group_mean = group_sum[seen] / group_count[seen].clamp_min(1.0)
            scale = group_mean.mean().clamp_min(1e-6)
            self._actworld_group_dro_logits[seen] += eta * group_mean / scale
            self._actworld_group_dro_logits -= self._actworld_group_dro_logits.max()
            self._actworld_group_dro_logits.clamp_(min=-20.0, max=0.0)
            probabilities = torch.softmax(self._actworld_group_dro_logits, dim=0)
        return probabilities[group_ids]

    @staticmethod
    def _baseline_pairwise_loss(
        predicted_scores: torch.Tensor,
        target_scores: torch.Tensor,
        baseline_scores: torch.Tensor,
        top_k: int,
        temperature: float,
    ) -> torch.Tensor:
        """Classify whether each coarse candidate beats the baseline winner."""
        if top_k < 2:
            raise ValueError("actworld_pairwise_loss_top_k must be at least two")
        if temperature <= 0:
            raise ValueError("actworld_pairwise_loss_temperature must be positive")
        top_k = min(top_k, predicted_scores.shape[1])
        shortlist = baseline_scores.float().topk(k=top_k, dim=1).indices
        baseline_index = baseline_scores.float().argmax(dim=1)
        rows = torch.arange(predicted_scores.shape[0], device=predicted_scores.device)
        predicted_delta = (
            predicted_scores.float()
            - predicted_scores.float()[rows, baseline_index, None]
        ).gather(1, shortlist)
        target_delta = (
            target_scores.float() - target_scores.float()[rows, baseline_index, None]
        ).gather(1, shortlist)
        valid = target_delta.abs() > 1e-4
        labels = (target_delta > 0).to(predicted_delta.dtype)
        weights = target_delta.abs().clamp_min(0.005) * valid
        loss = F.binary_cross_entropy_with_logits(
            predicted_delta / temperature, labels, reduction="none"
        )
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)

    @staticmethod
    def _all_pairs_rank_loss(
        predicted_scores: torch.Tensor,
        target_scores: torch.Tensor,
        baseline_scores: torch.Tensor,
        top_k: int,
        temperature: float,
    ) -> torch.Tensor:
        """Rank every meaningful pair in the Drive-JEPA coarse shortlist.

        Unlike the baseline-relative preference term, this supervision retains
        ordering information between non-baseline candidates.  Simulator PDMS
        gaps weight the RankNet comparisons so near-ties do not dominate.
        """
        if top_k < 2:
            raise ValueError("actworld_all_pairs_loss_top_k must be at least two")
        if temperature <= 0:
            raise ValueError("actworld_all_pairs_loss_temperature must be positive")
        top_k = min(top_k, predicted_scores.shape[1])
        shortlist = baseline_scores.float().topk(k=top_k, dim=1).indices
        predicted = predicted_scores.float().gather(1, shortlist)
        targets = target_scores.float().gather(1, shortlist)
        predicted_delta = predicted[:, :, None] - predicted[:, None, :]
        target_delta = targets[:, :, None] - targets[:, None, :]
        upper = torch.triu(
            torch.ones(
                top_k, top_k, dtype=torch.bool, device=predicted_scores.device
            ),
            diagonal=1,
        )[None]
        valid = upper & (target_delta.abs() > 1e-4)
        labels = (target_delta > 0).to(predicted_delta.dtype)
        weights = target_delta.abs().clamp_min(0.005) * valid
        loss = F.binary_cross_entropy_with_logits(
            predicted_delta / temperature, labels, reduction="none"
        )
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)

    @staticmethod
    def _progress_preserving_pairwise_loss(
        predicted_scores: torch.Tensor,
        simulator_targets: torch.Tensor,
        baseline_scores: torch.Tensor,
        top_k: int,
        temperature: float,
        progress_margin: float,
        critical_margin: float,
    ) -> torch.Tensor:
        """Preserve progress ordering among candidates that do not reduce safety.

        The final PDMS target already trades progress against safety.  This
        auxiliary term only breaks ties inside the safety-preserving subset,
        so it cannot reward extra progress obtained by sacrificing NC, DAC,
        or TTC.
        """
        if top_k < 2:
            raise ValueError("actworld_progress_pairwise_top_k must be at least two")
        if temperature <= 0:
            raise ValueError("actworld_progress_pairwise_temperature must be positive")
        if progress_margin < 0:
            raise ValueError("actworld_progress_pairwise_margin must be nonnegative")

        top_k = min(top_k, predicted_scores.shape[1])
        shortlist = baseline_scores.float().topk(k=top_k, dim=1).indices
        baseline_index = baseline_scores.float().argmax(dim=1)
        rows = torch.arange(predicted_scores.shape[0], device=predicted_scores.device)
        predicted_delta = (
            predicted_scores.float()
            - predicted_scores.float()[rows, baseline_index, None]
        ).gather(1, shortlist)

        targets = simulator_targets.float()
        baseline_targets = targets[rows, baseline_index]
        shortlisted_targets = targets.gather(
            1, shortlist[:, :, None].expand(-1, -1, targets.shape[-1])
        )
        channel_delta = shortlisted_targets - baseline_targets[:, None]
        progress_delta = channel_delta[..., 2]
        critical_delta = channel_delta[..., [0, 1, 3]].amin(dim=-1)
        valid = (
            (progress_delta.abs() > progress_margin)
            & (critical_delta >= critical_margin)
        )
        labels = (progress_delta > 0).to(predicted_delta.dtype)
        weights = progress_delta.abs().clamp_min(0.005) * valid
        loss = F.binary_cross_entropy_with_logits(
            predicted_delta / temperature, labels, reduction="none"
        )
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)

    @staticmethod
    def _risk_averse_pairwise_loss(
        predicted_scores: torch.Tensor,
        target_scores: torch.Tensor,
        baseline_scores: torch.Tensor,
        top_k: int,
        temperature: float,
        positive_margin: float,
        false_positive_weight: float,
        anchor_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prefer high-precision improvements over frequent risky switches.

        Candidates must beat the simulator score of the Drive-JEPA winner by a
        non-trivial margin to receive a positive label.  False-positive switches
        receive extra weight because one unsafe rerank can erase many small
        progress gains in multiplicative PDMS.
        """
        if top_k < 2:
            raise ValueError("actworld_risk_averse_top_k must be at least two")
        if temperature <= 0:
            raise ValueError("actworld_risk_averse_temperature must be positive")
        if positive_margin < 0:
            raise ValueError("actworld_risk_averse_positive_margin must be nonnegative")
        if false_positive_weight < 1:
            raise ValueError(
                "actworld_risk_averse_false_positive_weight must be at least one"
            )
        top_k = min(top_k, predicted_scores.shape[1])
        shortlist = baseline_scores.float().topk(k=top_k, dim=1).indices
        baseline_index = (
            baseline_scores.float().argmax(dim=1)
            if anchor_index is None
            else anchor_index.detach().long()
        )
        rows = torch.arange(predicted_scores.shape[0], device=predicted_scores.device)
        predicted_delta = (
            predicted_scores.float()
            - predicted_scores.float()[rows, baseline_index, None]
        ).gather(1, shortlist)
        target_delta = (
            target_scores.float() - target_scores.float()[rows, baseline_index, None]
        ).gather(1, shortlist)
        is_baseline = shortlist == baseline_index[:, None]
        labels = (target_delta >= positive_margin).to(predicted_delta.dtype)
        valid = ~is_baseline
        class_cost = torch.where(
            labels.bool(),
            torch.ones_like(target_delta),
            torch.full_like(target_delta, false_positive_weight),
        )
        utility_weight = target_delta.abs().clamp_min(max(positive_margin, 0.005))
        weights = class_cost * utility_weight * valid
        loss = F.binary_cross_entropy_with_logits(
            predicted_delta / temperature, labels, reduction="none"
        )
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)

    @staticmethod
    def _boundary_rank_loss(
        predicted_scores: torch.Tensor,
        simulator_targets: torch.Tensor,
        baseline_scores: torch.Tensor,
        top_k: int,
        temperature: float,
        boundary_band: float,
        positive_margin: float,
        false_positive_weight: float,
        require_progress: bool = False,
        anchor_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Focus ranking supervision on high-leverage mother decisions.

        Easy, large-margin candidates dominate a conventional pairwise loss,
        while deployment changes only when a proposal is close to the
        Drive-JEPA mother. This NAVTRAIN-only objective keeps near-boundary
        candidates, accepts a positive label only for a score gain with no
        NC/DAC/TTC/comfort regression, and heavily penalizes unsafe switches.
        """
        if top_k < 2:
            raise ValueError("boundary rank top_k must be at least two")
        if temperature <= 0 or boundary_band <= 0 or positive_margin < 0:
            raise ValueError("invalid boundary rank configuration")
        if false_positive_weight < 1:
            raise ValueError("boundary false-positive weight must be at least one")
        top_k = min(int(top_k), predicted_scores.shape[1])
        rows = torch.arange(predicted_scores.shape[0], device=predicted_scores.device)
        shortlist = baseline_scores.float().topk(top_k, dim=1).indices
        anchor = (
            baseline_scores.float().argmax(dim=1)
            if anchor_index is None
            else anchor_index.detach().long()
        )
        target_delta = (
            simulator_targets[..., -1].float()
            - simulator_targets[rows, anchor, -1].float()[:, None]
        ).gather(1, shortlist)
        predicted_delta = (
            predicted_scores.float()
            - predicted_scores.float()[rows, anchor, None]
        ).gather(1, shortlist)
        channel_delta = (
            simulator_targets.float()[..., [0, 1, 3, 4]]
            - simulator_targets.float()[rows, anchor][:, None, [0, 1, 3, 4]]
        ).gather(1, shortlist[..., None].expand(-1, -1, 4))
        safety_ok = (channel_delta >= 0.0).all(dim=-1)
        progress_delta = (
            simulator_targets.float()[..., 2]
            - simulator_targets.float()[rows, anchor, 2][:, None]
        ).gather(1, shortlist)
        progress_ok = progress_delta >= 0.0
        is_anchor = shortlist == anchor[:, None]
        valid = (~is_anchor) & (target_delta.abs() <= float(boundary_band))
        labels = (
            (target_delta >= float(positive_margin))
            & safety_ok
            & (progress_ok if require_progress else torch.ones_like(progress_ok, dtype=torch.bool))
        ).to(
            predicted_delta.dtype
        )
        proximity = (1.0 - target_delta.abs() / float(boundary_band)).clamp_min(0.0)
        class_weight = torch.where(
            labels > 0.5,
            torch.ones_like(labels),
            torch.full_like(labels, float(false_positive_weight)),
        )
        weights = (0.25 + 0.75 * proximity) * class_weight * valid
        loss = F.binary_cross_entropy_with_logits(
            predicted_delta / float(temperature), labels, reduction="none"
        )
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)

    @staticmethod
    def _shortlist_distillation_loss(
        predicted_scores: torch.Tensor,
        target_scores: torch.Tensor,
        baseline_scores: torch.Tensor,
        top_k: int,
        student_temperature: float,
        teacher_temperature: float,
    ) -> torch.Tensor:
        """Distill simulator utility over the deployment-relevant coarse shortlist.

        Centering both distributions at the Drive-JEPA winner makes the objective
        invariant to scene-level score offsets.  Scenes where the baseline is
        already best remain in the loss, explicitly teaching the fallback action.
        """
        if top_k < 2:
            raise ValueError("actworld_shortlist_distillation_top_k must be at least two")
        if student_temperature <= 0 or teacher_temperature <= 0:
            raise ValueError("shortlist distillation temperatures must be positive")
        top_k = min(top_k, predicted_scores.shape[1])
        shortlist = baseline_scores.float().topk(k=top_k, dim=1).indices
        predicted = predicted_scores.float().gather(1, shortlist)
        targets = target_scores.float().gather(1, shortlist)
        predicted = predicted - predicted[:, :1]
        targets = targets - targets[:, :1]
        teacher = torch.softmax(targets / teacher_temperature, dim=1)
        student_log = torch.log_softmax(predicted / student_temperature, dim=1)
        return F.kl_div(student_log, teacher, reduction="batchmean")

    @staticmethod
    def _factorwise_pairwise_loss(
        predicted_logits: torch.Tensor,
        simulator_targets: torch.Tensor,
        baseline_scores: torch.Tensor,
        top_k: int,
        temperature: float,
        safety_false_positive_weight: float,
        progress_weight: float,
        nc_weight: float,
        dac_weight: float,
        ttc_weight: float,
        comfort_weight: float,
        focal_gamma: float = 0.0,
    ) -> torch.Tensor:
        """Rank explicit PDMS factors relative to Drive-JEPA's winner."""
        if top_k < 2:
            raise ValueError("actworld_factorwise_pairwise_top_k must be at least two")
        if temperature <= 0:
            raise ValueError("actworld_factorwise_pairwise_temperature must be positive")
        if safety_false_positive_weight < 1:
            raise ValueError(
                "actworld_factorwise_safety_false_positive_weight must be at least one"
            )
        if progress_weight < 0:
            raise ValueError("actworld_factorwise_progress_weight must be nonnegative")
        if focal_gamma < 0:
            raise ValueError("actworld_factorwise_focal_gamma must be nonnegative")
        factor_weights = torch.tensor(
            [nc_weight, dac_weight, ttc_weight, comfort_weight],
            dtype=simulator_targets.dtype,
            device=simulator_targets.device,
        )
        if not bool(torch.isfinite(factor_weights).all()) or bool((factor_weights < 0).any()):
            raise ValueError("factorwise channel weights must be finite and nonnegative")
        top_k = min(top_k, predicted_logits.shape[1])
        shortlist = baseline_scores.float().topk(k=top_k, dim=1).indices
        baseline_index = baseline_scores.float().argmax(dim=1)
        rows = torch.arange(predicted_logits.shape[0], device=predicted_logits.device)
        predicted_delta = (
            predicted_logits.float()
            - predicted_logits.float()[rows, baseline_index, None]
        ).gather(1, shortlist[:, :, None].expand(-1, -1, predicted_logits.shape[-1]))
        targets = simulator_targets.float()
        target_delta = targets.gather(
            1, shortlist[:, :, None].expand(-1, -1, targets.shape[-1])
        ) - targets[rows, baseline_index, None]
        is_baseline = shortlist == baseline_index[:, None]

        # A critical false positive means predicting a safety improvement where
        # the simulator factor is actually worse. Weight that asymmetrically.
        critical = [0, 1, 3, 4]
        critical_delta = target_delta[..., critical]
        critical_prediction = predicted_delta[..., critical]
        critical_valid = (critical_delta.abs() > 1e-4) & ~is_baseline[..., None]
        critical_labels = (critical_delta > 0).to(critical_prediction.dtype)
        critical_cost = torch.where(
            critical_labels.bool(),
            torch.ones_like(critical_delta),
            torch.full_like(critical_delta, safety_false_positive_weight),
        )
        critical_weights = (
            critical_delta.abs()
            * critical_cost
            * factor_weights[None, None, :]
            * critical_valid
        )
        critical_logits = critical_prediction / temperature
        critical_loss = F.binary_cross_entropy_with_logits(
            critical_logits,
            critical_labels,
            reduction="none",
        )
        if focal_gamma:
            probability = torch.sigmoid(critical_logits)
            target_probability = torch.where(
                critical_labels.bool(), probability, 1.0 - probability
            )
            critical_loss = critical_loss * (1.0 - target_probability).pow(
                focal_gamma
            )
        safety_loss = (
            (critical_loss * critical_weights).sum()
            / critical_weights.sum().clamp_min(1e-6)
        )

        progress_delta = target_delta[..., 2]
        progress_prediction = predicted_delta[..., 2]
        progress_valid = (progress_delta.abs() > 0.005) & ~is_baseline
        progress_labels = (progress_delta > 0).to(progress_prediction.dtype)
        progress_weights = progress_delta.abs() * progress_valid
        progress_loss = F.binary_cross_entropy_with_logits(
            progress_prediction / temperature,
            progress_labels,
            reduction="none",
        )
        progress_loss = (
            (progress_loss * progress_weights).sum()
            / progress_weights.sum().clamp_min(1e-6)
        )
        return safety_loss + progress_weight * progress_loss

    @staticmethod
    def _counterfactual_outcome_jepa_loss(
        predicted: torch.Tensor,
        target: torch.Tensor,
        simulator_targets: torch.Tensor,
        baseline_scores: torch.Tensor,
        top_k: int,
        relative_weight: float,
    ) -> torch.Tensor:
        """Predict factorized outcomes in joint-embedding space.

        Absolute alignment teaches what each proposal causes; relative
        alignment teaches the deployment question: how it changes the future
        compared with Drive-JEPA's current winner.
        """
        if predicted.shape != target.shape or predicted.ndim != 3:
            raise ValueError("outcome JEPA prediction and target shapes must match")
        if simulator_targets.shape[:2] != predicted.shape[:2]:
            raise ValueError("outcome JEPA simulator-target shape is incompatible")
        if top_k < 2:
            raise ValueError("actworld_outcome_jepa_top_k must be at least two")
        if not 0.0 <= relative_weight <= 1.0:
            raise ValueError("actworld_outcome_jepa_relative_weight must be in [0, 1]")
        top_k = min(top_k, predicted.shape[1])
        shortlist = baseline_scores.float().topk(k=top_k, dim=1).indices
        rows = torch.arange(predicted.shape[0], device=predicted.device)
        baseline_index = baseline_scores.float().argmax(dim=1)
        gather = shortlist[:, :, None].expand(-1, -1, predicted.shape[-1])

        predicted_float = predicted.float()
        target_float = target.detach().float()
        predicted_absolute = F.normalize(predicted_float, dim=-1)
        target_absolute = F.normalize(target_float, dim=-1)
        absolute_distance = 1.0 - (predicted_absolute * target_absolute).sum(dim=-1)

        predicted_relative = (
            predicted_float - predicted_float[rows, baseline_index, None]
        )
        target_relative = target_float - target_float[rows, baseline_index, None]
        predicted_relative = F.normalize(predicted_relative, dim=-1)
        target_relative = F.normalize(target_relative, dim=-1)
        relative_distance = 1.0 - (predicted_relative * target_relative).sum(dim=-1)

        scores = simulator_targets[..., -1].float()
        score_gap = (
            scores - scores[rows, baseline_index, None]
        ).abs()
        weights = 1.0 + 10.0 * score_gap
        absolute = absolute_distance.gather(1, shortlist)
        relative = relative_distance.gather(1, shortlist)
        selected_weights = weights.gather(1, shortlist)
        non_baseline = shortlist != baseline_index[:, None]
        absolute_loss = (absolute * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)
        relative_weights = selected_weights * non_baseline
        relative_loss = (relative * relative_weights).sum() / relative_weights.sum().clamp_min(1e-6)
        return (1.0 - relative_weight) * absolute_loss + relative_weight * relative_loss

    @staticmethod
    def _factorized_outcome_prediction_loss(
        logits: torch.Tensor,
        simulator_targets: torch.Tensor,
        baseline_scores: torch.Tensor,
        top_k: int,
        safety_negative_weight: float,
        false_positive_weight: float,
        temperature: float,
    ):
        """Decode safety/progress factors and rank candidates through them.

        The target has six NAVSIM channels: NC, DAC, progress, TTC, comfort,
        and the factorized PDM score. The score is not decoded independently;
        it is reconstructed from the first five physical factors, keeping the
        auxiliary task interpretable and consistent with the official metric.
        """
        if logits.ndim != 3 or logits.shape[-1] != 6:
            raise ValueError("outcome factor logits must have shape [B,C,6]")
        if simulator_targets.shape != logits.shape:
            raise ValueError("outcome factor target shape is incompatible")
        if baseline_scores.shape != logits.shape[:2]:
            raise ValueError("outcome factor baseline-score shape is incompatible")
        if top_k < 2:
            raise ValueError("actworld_outcome_factor_top_k must be at least two")
        if safety_negative_weight < 0:
            raise ValueError("outcome factor safety-negative weight must be nonnegative")
        if false_positive_weight < 1:
            raise ValueError("outcome factor false-positive weight must be at least one")
        if temperature <= 0:
            raise ValueError("outcome factor temperature must be positive")

        target = simulator_targets.detach().float()
        logits_float = logits.float()
        top_k = min(top_k, logits.shape[1])
        shortlist = baseline_scores.float().topk(k=top_k, dim=1).indices
        gather = shortlist[:, :, None].expand(-1, -1, logits.shape[-1])
        selected_logits = logits_float.gather(1, gather)
        selected_target = target.gather(1, gather)

        # A failed critical factor is rare but dominates PDMS. Weight its
        # negative labels so high progress cannot hide an unsafe prediction.
        weights = torch.ones_like(selected_target)
        critical = (0, 1, 3)
        weights[..., list(critical)] += safety_negative_weight * (
            1.0 - selected_target[..., list(critical)]
        )
        channel_weights = logits_float.new_tensor((2.0, 2.0, 1.0, 4.0, 1.0, 1.0))
        weights = weights * channel_weights
        metric = F.binary_cross_entropy_with_logits(
            selected_logits, selected_target, reduction="none"
        )
        metric_loss = (metric * weights).sum() / weights.sum().clamp_min(1e-6)

        rows = torch.arange(logits.shape[0], device=logits.device)
        baseline_index = baseline_scores.float().argmax(dim=1)
        probabilities = torch.sigmoid(logits_float)
        probability_delta = probabilities - probabilities[rows, baseline_index, None]
        target_delta = target - target[rows, baseline_index, None]
        selected_probability_delta = probability_delta.gather(1, gather)
        selected_target_delta = target_delta.gather(1, gather)
        non_baseline = shortlist != baseline_index[:, None]
        relative_weights = (
            1.0 + 10.0 * selected_target_delta[..., -1].abs()
        ) * non_baseline
        relative_per_candidate = F.smooth_l1_loss(
            selected_probability_delta,
            selected_target_delta,
            reduction="none",
        )
        relative_per_candidate = (
            relative_per_candidate * channel_weights
        ).sum(dim=-1) / channel_weights.sum()
        relative_loss = (
            relative_per_candidate * relative_weights
        ).sum() / relative_weights.sum().clamp_min(1e-6)

        factorized_score = (
            probabilities[..., 0]
            * probabilities[..., 1]
            * (
                5.0 * probabilities[..., 2]
                + 5.0 * probabilities[..., 3]
                + 2.0 * probabilities[..., 4]
            )
            / 12.0
        )
        rank_loss = ActWorldJEPAAgent._risk_averse_pairwise_loss(
            factorized_score,
            target[..., -1],
            baseline_scores,
            top_k,
            temperature,
            0.0025,
            false_positive_weight,
        )
        return metric_loss, relative_loss, rank_loss

    def _counterfactual_outcome_delta_loss(
        self,
        predicted_delta: torch.Tensor,
        simulator_targets: torch.Tensor,
        baseline_scores: torch.Tensor,
        top_k: int,
        temperature: float,
        false_positive_weight: float,
        focal_gamma: float = 0.0,
    ):
        """Regress and rank factor changes relative to the baseline winner."""
        if predicted_delta.shape != simulator_targets.shape:
            raise ValueError("outcome delta prediction and target shapes must match")
        top_k = min(top_k, predicted_delta.shape[1])
        if top_k < 2:
            raise ValueError("actworld_outcome_delta_top_k must be at least two")
        rows = torch.arange(predicted_delta.shape[0], device=predicted_delta.device)
        baseline_index = baseline_scores.float().argmax(dim=1)
        target = simulator_targets.detach().float()
        target_delta = target - target[rows, baseline_index, None]
        shortlist = baseline_scores.float().topk(k=top_k, dim=1).indices
        gather = shortlist[:, :, None].expand(-1, -1, predicted_delta.shape[-1])
        predicted_selected = predicted_delta.float().gather(1, gather)
        target_selected = target_delta.gather(1, gather)
        non_baseline = shortlist != baseline_index[:, None]
        channel_weights = predicted_delta.new_tensor(
            (2.0, 2.0, 1.0, 4.0, 1.0, 2.0)
        ).float()
        utility_weights = (
            1.0 + 10.0 * target_selected[..., -1].abs()
        ) * non_baseline
        regression = F.smooth_l1_loss(
            10.0 * predicted_selected,
            10.0 * target_selected,
            reduction="none",
        )
        regression = (regression * channel_weights).sum(dim=-1) / channel_weights.sum()
        regression_loss = (
            regression * utility_weights
        ).sum() / utility_weights.sum().clamp_min(1e-6)
        rank_loss = self._risk_averse_pairwise_loss(
            predicted_delta[..., -1],
            target[..., -1],
            baseline_scores,
            top_k,
            temperature,
            0.0025,
            false_positive_weight,
        )
        factorwise_loss = self._factorwise_pairwise_loss(
            predicted_delta,
            target,
            baseline_scores,
            top_k,
            temperature,
            false_positive_weight,
            0.5,
            2.0,
            2.0,
            4.0,
            1.0,
            focal_gamma,
        )
        return regression_loss, rank_loss, factorwise_loss

    def _counterfactual_sign_calibration_loss(
        self,
        predicted_delta: torch.Tensor,
        simulator_targets: torch.Tensor,
        baseline_scores: torch.Tensor,
        top_k: int,
        margin: float = 0.0025,
        temperature: float = 0.05,
    ) -> torch.Tensor:
        """Class-balance improvement/regression signs on the baseline shortlist."""
        if predicted_delta.shape != simulator_targets.shape:
            raise ValueError("sign prediction and target shapes must match")
        top_k = min(top_k, predicted_delta.shape[1])
        rows = torch.arange(predicted_delta.shape[0], device=predicted_delta.device)
        baseline_index = baseline_scores.float().argmax(dim=1)
        target = simulator_targets.detach().float()
        target_delta = target - target[rows, baseline_index, None]
        shortlist = baseline_scores.float().topk(k=top_k, dim=1).indices
        gather = shortlist[:, :, None].expand(-1, -1, predicted_delta.shape[-1])
        prediction = predicted_delta.float().gather(1, gather)
        target_selected = target_delta.gather(1, gather)
        non_baseline = shortlist != baseline_index[:, None]
        positive = target_selected > margin
        negative = target_selected < -margin
        valid = non_baseline[..., None] & (positive | negative)
        positive_count = (positive & valid).sum(dim=(0, 1)).float()
        negative_count = (negative & valid).sum(dim=(0, 1)).float()
        positive_weight = (negative_count / positive_count.clamp_min(1.0)).clamp(1.0, 8.0)
        loss = F.binary_cross_entropy_with_logits(
            prediction / temperature,
            positive.float(),
            pos_weight=positive_weight,
            reduction="none",
        )
        channel_weights = prediction.new_tensor((2.0, 2.0, 1.0, 2.0, 1.0, 2.0))
        weights = valid.to(loss.dtype) * channel_weights
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)

    def _world_supervision(
        self,
        world_logits: torch.Tensor,
        value_logits: torch.Tensor,
        collision_logits: torch.Tensor,
        fusion_scores: torch.Tensor,
        rank_scores: torch.Tensor,
        preference_logits: torch.Tensor,
        baseline_scores: torch.Tensor,
        simulator_targets: torch.Tensor,
        temperature: float,
    ) -> Dict[str, torch.Tensor]:
        simulator_targets = simulator_targets.float()
        metric_losses = F.binary_cross_entropy_with_logits(
            world_logits.float(), simulator_targets, reduction="none"
        )
        safety_negative_weight = float(self._config.actworld_safety_negative_weight)
        if safety_negative_weight < 0:
            raise ValueError("actworld_safety_negative_weight must be nonnegative")
        metric_weights = torch.ones_like(metric_losses)
        if safety_negative_weight:
            critical_targets = simulator_targets[..., [0, 1, 3]]
            metric_weights[..., [0, 1, 3]] += (
                safety_negative_weight * (1.0 - critical_targets)
            )
        collision_target = 1.0 - simulator_targets[..., 0]
        collision_weights = 1.0 + safety_negative_weight * collision_target
        return {
            "metric": (metric_losses * metric_weights).sum() / metric_weights.sum(),
            "value": F.binary_cross_entropy_with_logits(
                value_logits.float(), simulator_targets[..., -1]
            ),
            "collision": (
                F.binary_cross_entropy_with_logits(
                    collision_logits.float(), collision_target, reduction="none"
                )
                * collision_weights
            ).sum()
            / collision_weights.sum(),
            # v1 Drive-JEPA fuses calibrated probabilities rather than logits.
            "fusion": F.smooth_l1_loss(
                fusion_scores.float(), simulator_targets[..., -1]
            ),
            "rank": self._soft_rank_loss(
                fusion_scores, simulator_targets[..., -1], temperature
            ),
            "hard_negative_rank": self._coarse_to_fine_rank_loss(
                rank_scores,
                simulator_targets[..., -1],
                baseline_scores,
                int(self._config.actworld_hard_negative_top_k),
                float(self._config.actworld_hard_negative_temperature),
            ),
            "pairwise_preference": self._baseline_pairwise_loss(
                rank_scores,
                simulator_targets[..., -1],
                baseline_scores,
                int(self._config.actworld_pairwise_loss_top_k),
                float(self._config.actworld_pairwise_loss_temperature),
            ),
            "all_pairs_rank": self._all_pairs_rank_loss(
                rank_scores,
                simulator_targets[..., -1],
                baseline_scores,
                int(self._config.actworld_all_pairs_loss_top_k),
                float(self._config.actworld_all_pairs_loss_temperature),
            ),
            "progress_pairwise": self._progress_preserving_pairwise_loss(
                rank_scores,
                simulator_targets,
                baseline_scores,
                int(self._config.actworld_progress_pairwise_top_k),
                float(self._config.actworld_progress_pairwise_temperature),
                float(self._config.actworld_progress_pairwise_margin),
                float(self._config.actworld_progress_pairwise_critical_margin),
            ),
            "risk_averse_pairwise": self._risk_averse_pairwise_loss(
                rank_scores,
                simulator_targets[..., -1],
                baseline_scores,
                int(self._config.actworld_risk_averse_top_k),
                float(self._config.actworld_risk_averse_temperature),
                float(self._config.actworld_risk_averse_positive_margin),
                float(self._config.actworld_risk_averse_false_positive_weight),
            ),
            "shortlist_distillation": self._shortlist_distillation_loss(
                rank_scores,
                simulator_targets[..., -1],
                baseline_scores,
                int(self._config.actworld_shortlist_distillation_top_k),
                float(self._config.actworld_shortlist_student_temperature),
                float(self._config.actworld_shortlist_teacher_temperature),
            ),
            "direct_preference": self._risk_averse_pairwise_loss(
                preference_logits,
                simulator_targets[..., -1],
                baseline_scores,
                int(self._config.actworld_preference_top_k),
                float(self._config.actworld_preference_temperature),
                float(self._config.actworld_preference_positive_margin),
                float(self._config.actworld_preference_false_positive_weight),
            ),
            "factorwise_pairwise": self._factorwise_pairwise_loss(
                world_logits,
                simulator_targets,
                baseline_scores,
                int(self._config.actworld_factorwise_pairwise_top_k),
                float(self._config.actworld_factorwise_pairwise_temperature),
                float(self._config.actworld_factorwise_safety_false_positive_weight),
                float(self._config.actworld_factorwise_progress_weight),
                float(self._config.actworld_factorwise_nc_weight),
                float(self._config.actworld_factorwise_dac_weight),
                float(self._config.actworld_factorwise_ttc_weight),
                float(self._config.actworld_factorwise_comfort_weight),
                float(self._config.actworld_factorwise_focal_gamma),
            ),
        }

    def pad_loss(
        self,
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
        config: ActWorldJEPAConfig,
    ) -> Dict[str, torch.Tensor]:
        sparse_simulator_supervision = (
            config.actworld_current_future_v2_only
            or config.actworld_current_future_rank_only
            or config.actworld_current_future_factorized_rank_only
            or config.actworld_current_future_rank_joint
            or config.actworld_current_future_progress_only
            or config.actworld_planning_adaptation_only
            or config.actworld_refinement_adaptation_only
        )
        if sparse_simulator_supervision:
            # The current/future objective below is defined exclusively on the
            # baseline top-k shortlist.  Scoring every proposal with the NAVSIM
            # simulator is therefore redundant when all proposal-generating and
            # legacy Drive-JEPA heads are frozen.  Score exactly the candidates
            # consumed by the active objective, then scatter them back so the
            # existing loss implementation remains unchanged.
            baseline_scores = pred["baseline_pdm_score"].float()
            top_k = min(int(config.actworld_outcome_delta_top_k), baseline_scores.shape[1])
            shortlist_index = baseline_scores.topk(top_k, dim=1).indices
            rows = torch.arange(
                baseline_scores.shape[0], device=baseline_scores.device
            )[:, None]
            shortlist_proposals = pred["proposals"][rows, shortlist_index]
            _, _, shortlist_targets = self.compute_score(
                targets, shortlist_proposals, test=True, label_only=True
            )
            scatter_index = shortlist_index.unsqueeze(-1).expand(
                -1, -1, shortlist_targets.shape[-1]
            )
            simulator_targets = shortlist_targets.new_zeros(
                (*baseline_scores.shape, shortlist_targets.shape[-1])
            ).scatter(1, scatter_index, shortlist_targets)
            pred["_target_scores"] = simulator_targets
            pred["_target_final_scores"] = simulator_targets[..., -1]

            # Frozen legacy losses cannot update this head.  Keeping zero-valued
            # scalar entries preserves Lightning's logging contract without
            # executing their full-proposal simulator supervision.
            zero_source = (
                pred["proposals"]
                if (
                    config.actworld_planning_adaptation_only
                    or config.actworld_refinement_adaptation_only
                )
                else pred["actworld_initial_current_future_factor_deltas"]
            )
            zero = zero_source.sum() * 0.0
            loss_dict = {
                name: zero
                for name in (
                    "loss",
                    "trajectory_loss",
                    "sub_score_loss",
                    "final_score_loss",
                    "pred_ce_loss",
                    "pred_l1_loss",
                    "pred_area_loss",
                    "inter_loss0",
                    "inter_loss",
                    "min_loss0",
                    "min_loss",
                    "score",
                    "best_score",
                )
            }
        else:
            loss_dict = super().pad_loss(targets, pred, config)
            simulator_targets = pred["_target_scores"].float()

        initial_terms = self._world_supervision(
            pred["actworld_initial_world_logits"],
            pred["actworld_initial_value_logits"],
            pred["actworld_initial_collision_logits"],
            pred["actworld_initial_world_scores"],
            pred["actworld_initial_conservative_scores"],
            pred["actworld_initial_preference_logits"],
            pred["baseline_pdm_score"],
            simulator_targets,
            config.actworld_rank_temperature,
        )

        refined = pred["refined_proposals"].float()
        refined_targets = None
        refined_weight = float(config.actworld_refined_score_loss_weight)
        if refined_weight < 0:
            raise ValueError("actworld_refined_score_loss_weight must be nonnegative")
        if config.actworld_rescore_refined_proposals:
            _, _, refined_targets = self.compute_score(
                targets, refined.detach(), test=True, label_only=True
            )
            refined_terms = self._world_supervision(
                pred["actworld_world_logits"],
                pred["actworld_value_logits"],
                pred["actworld_collision_logits"],
                pred["actworld_world_scores"],
                pred["actworld_conservative_scores"],
                pred["actworld_preference_logits"],
                pred["baseline_pdm_score"],
                refined_targets,
                config.actworld_rank_temperature,
            )
            denominator = 1.0 + refined_weight
            supervision = {
                name: (initial_terms[name] + refined_weight * refined_terms[name])
                / denominator
                for name in initial_terms
            }
        else:
            supervision = initial_terms

        metric_loss = supervision["metric"]
        value_loss = supervision["value"]
        collision_loss = supervision["collision"]
        fusion_loss = supervision["fusion"]
        rank_loss = supervision["rank"]
        hard_negative_rank_loss = supervision["hard_negative_rank"]
        pairwise_preference_loss = supervision["pairwise_preference"]
        all_pairs_rank_loss = supervision["all_pairs_rank"]
        progress_pairwise_loss = supervision["progress_pairwise"]
        risk_averse_pairwise_loss = supervision["risk_averse_pairwise"]
        shortlist_distillation_loss = supervision["shortlist_distillation"]
        direct_preference_loss = supervision["direct_preference"]
        factorwise_pairwise_loss = supervision["factorwise_pairwise"]
        target_trajectory = targets["trajectory"].float()
        planning_jepa_weight = float(config.actworld_planning_jepa_loss_weight)
        planning_jepa_temperature = float(config.actworld_planning_jepa_temperature)
        if planning_jepa_weight < 0 or planning_jepa_temperature <= 0:
            raise ValueError(
                "planning-JEPA weight must be nonnegative and temperature positive"
            )
        if planning_jepa_weight:
            planner = self._pad_model._actworld_planner
            target_world = planner.target_world_features(
                pred["actworld_scene_tokens"], target_trajectory
            ).detach()
            candidate_world = pred["actworld_initial_world_features"].float()
            target_world = F.normalize(target_world.float(), dim=-1)
            candidate_world = F.normalize(candidate_world, dim=-1)
            planning_logits = torch.einsum(
                "bcd,bd->bc", candidate_world, target_world
            )
            teacher = torch.softmax(
                simulator_targets[..., -1].float() / planning_jepa_temperature,
                dim=1,
            )
            planning_jepa_loss = -(
                teacher
                * torch.log_softmax(
                    planning_logits / planning_jepa_temperature, dim=1
                )
            ).sum(dim=1).mean()
        else:
            planning_jepa_loss = loss_dict["loss"].new_zeros(())
        outcome_jepa_weight = float(config.actworld_outcome_jepa_loss_weight)
        if outcome_jepa_weight < 0:
            raise ValueError("actworld_outcome_jepa_loss_weight must be nonnegative")
        if outcome_jepa_weight:
            baseline_index = pred["baseline_pdm_score"].float().argmax(dim=1)
            outcome_target = self._pad_model._actworld_planner.factor_target_features(
                simulator_targets, baseline_index
            )
            outcome_jepa_loss = self._counterfactual_outcome_jepa_loss(
                pred["actworld_initial_outcome_features"],
                outcome_target,
                simulator_targets,
                pred["baseline_pdm_score"],
                int(config.actworld_outcome_jepa_top_k),
                float(config.actworld_outcome_jepa_relative_weight),
            )
        else:
            outcome_jepa_loss = loss_dict["loss"].new_zeros(())
        outcome_factor_weight = float(config.actworld_outcome_factor_loss_weight)
        if outcome_factor_weight < 0:
            raise ValueError("actworld_outcome_factor_loss_weight must be nonnegative")
        if outcome_factor_weight:
            (
                outcome_factor_metric_loss,
                outcome_factor_relative_loss,
                outcome_factor_rank_loss,
            ) = self._factorized_outcome_prediction_loss(
                pred["actworld_initial_outcome_factor_logits"],
                simulator_targets,
                pred["baseline_pdm_score"],
                int(config.actworld_outcome_factor_top_k),
                float(config.actworld_outcome_factor_safety_negative_weight),
                float(config.actworld_outcome_factor_false_positive_weight),
                float(config.actworld_outcome_factor_temperature),
            )
            outcome_factor_loss = (
                outcome_factor_metric_loss
                + float(config.actworld_outcome_factor_relative_weight)
                * outcome_factor_relative_loss
                + float(config.actworld_outcome_factor_rank_weight)
                * outcome_factor_rank_loss
            )
        else:
            outcome_factor_metric_loss = loss_dict["loss"].new_zeros(())
            outcome_factor_relative_loss = loss_dict["loss"].new_zeros(())
            outcome_factor_rank_loss = loss_dict["loss"].new_zeros(())
            outcome_factor_loss = loss_dict["loss"].new_zeros(())
        outcome_delta_weight = float(config.actworld_outcome_delta_loss_weight)
        if outcome_delta_weight < 0:
            raise ValueError("actworld_outcome_delta_loss_weight must be nonnegative")
        if outcome_delta_weight:
            (
                outcome_delta_regression_loss,
                outcome_delta_rank_loss,
                outcome_delta_factorwise_loss,
            ) = self._counterfactual_outcome_delta_loss(
                pred["actworld_initial_outcome_factor_deltas"],
                simulator_targets,
                pred["baseline_pdm_score"],
                int(config.actworld_outcome_delta_top_k),
                float(config.actworld_outcome_delta_temperature),
                float(config.actworld_outcome_delta_false_positive_weight),
                float(config.actworld_factorwise_focal_gamma),
            )
            outcome_delta_loss = (
                outcome_delta_regression_loss
                + outcome_delta_rank_loss
                + float(config.actworld_outcome_delta_factorwise_weight)
                * outcome_delta_factorwise_loss
            )
        else:
            outcome_delta_regression_loss = loss_dict["loss"].new_zeros(())
            outcome_delta_rank_loss = loss_dict["loss"].new_zeros(())
            outcome_delta_factorwise_loss = loss_dict["loss"].new_zeros(())
            outcome_delta_loss = loss_dict["loss"].new_zeros(())
        outcome_action_delta_weight = float(
            config.actworld_outcome_action_delta_loss_weight
        )
        if outcome_action_delta_weight < 0:
            raise ValueError(
                "actworld_outcome_action_delta_loss_weight must be nonnegative"
            )
        if outcome_action_delta_weight:
            (
                outcome_action_delta_regression_loss,
                outcome_action_delta_rank_loss,
                outcome_action_delta_factorwise_loss,
            ) = self._counterfactual_outcome_delta_loss(
                pred["actworld_initial_outcome_action_factor_deltas"],
                simulator_targets,
                pred["baseline_pdm_score"],
                int(config.actworld_outcome_delta_top_k),
                float(config.actworld_outcome_delta_temperature),
                float(config.actworld_outcome_delta_false_positive_weight),
                float(config.actworld_factorwise_focal_gamma),
            )
            outcome_action_delta_loss = (
                outcome_action_delta_regression_loss
                + outcome_action_delta_rank_loss
                + float(config.actworld_outcome_delta_factorwise_weight)
                * outcome_action_delta_factorwise_loss
            )
        else:
            outcome_action_delta_regression_loss = loss_dict["loss"].new_zeros(())
            outcome_action_delta_rank_loss = loss_dict["loss"].new_zeros(())
            outcome_action_delta_factorwise_loss = loss_dict["loss"].new_zeros(())
            outcome_action_delta_loss = loss_dict["loss"].new_zeros(())
        temporal_action_delta_weight = float(
            config.actworld_temporal_action_delta_loss_weight
        )
        if temporal_action_delta_weight < 0:
            raise ValueError(
                "actworld_temporal_action_delta_loss_weight must be nonnegative"
            )
        if temporal_action_delta_weight:
            (
                temporal_action_delta_regression_loss,
                temporal_action_delta_rank_loss,
                temporal_action_delta_factorwise_loss,
            ) = self._counterfactual_outcome_delta_loss(
                pred["actworld_initial_temporal_action_factor_deltas"],
                simulator_targets,
                pred["baseline_pdm_score"],
                int(config.actworld_outcome_delta_top_k),
                float(config.actworld_outcome_delta_temperature),
                float(config.actworld_outcome_delta_false_positive_weight),
                float(config.actworld_factorwise_focal_gamma),
            )
            temporal_action_delta_loss = (
                temporal_action_delta_regression_loss
                + temporal_action_delta_rank_loss
                + float(config.actworld_outcome_delta_factorwise_weight)
                * temporal_action_delta_factorwise_loss
            )
        else:
            temporal_action_delta_regression_loss = loss_dict["loss"].new_zeros(())
            temporal_action_delta_rank_loss = loss_dict["loss"].new_zeros(())
            temporal_action_delta_factorwise_loss = loss_dict["loss"].new_zeros(())
            temporal_action_delta_loss = loss_dict["loss"].new_zeros(())
        current_future_v2_weight = float(
            config.actworld_current_future_v2_loss_weight
        )
        if current_future_v2_weight < 0:
            raise ValueError(
                "actworld_current_future_v2_loss_weight must be nonnegative"
            )
        if current_future_v2_weight:
            (
                current_future_v2_regression_loss,
                current_future_v2_rank_loss,
                current_future_v2_factorwise_loss,
            ) = self._counterfactual_outcome_delta_loss(
                pred["actworld_initial_current_future_factor_deltas"],
                simulator_targets,
                pred["baseline_pdm_score"],
                int(config.actworld_outcome_delta_top_k),
                float(config.actworld_outcome_delta_temperature),
                float(config.actworld_outcome_delta_false_positive_weight),
                float(config.actworld_factorwise_focal_gamma),
            )
            current_future_v2_loss = (
                current_future_v2_regression_loss
                + current_future_v2_rank_loss
                + float(config.actworld_outcome_delta_factorwise_weight)
                * current_future_v2_factorwise_loss
            )
            current_future_sign_weight = float(
                config.actworld_current_future_sign_loss_weight
            )
            if current_future_sign_weight < 0:
                raise ValueError(
                    "actworld_current_future_sign_loss_weight must be nonnegative"
                )
            current_future_sign_loss = self._counterfactual_sign_calibration_loss(
                pred["actworld_initial_current_future_factor_deltas"],
                simulator_targets,
                pred["baseline_pdm_score"],
                int(config.actworld_outcome_delta_top_k),
            )
            current_future_v2_loss = (
                current_future_v2_loss
                + current_future_sign_weight * current_future_sign_loss
            )
        else:
            current_future_v2_regression_loss = loss_dict["loss"].new_zeros(())
            current_future_v2_rank_loss = loss_dict["loss"].new_zeros(())
            current_future_v2_factorwise_loss = loss_dict["loss"].new_zeros(())
            current_future_sign_loss = loss_dict["loss"].new_zeros(())
            current_future_v2_loss = loss_dict["loss"].new_zeros(())
        current_future_jepa_weight = float(
            config.actworld_current_future_jepa_loss_weight
        )
        current_future_jepa_temperature = float(
            config.actworld_current_future_jepa_temperature
        )
        if current_future_jepa_weight < 0 or current_future_jepa_temperature <= 0:
            raise ValueError(
                "current-future JEPA weight must be nonnegative and temperature positive"
            )
        if current_future_jepa_weight:
            # The target encoder is stopped, following JEPA's asymmetric
            # predictor/target design. In decoupled mode this target uses the
            # frozen world path, so only the Current-Future interaction learns.
            planner = self._pad_model._actworld_planner
            target_future = planner.target_world_features(
                pred["actworld_scene_tokens"], target_trajectory
            ).detach()
            candidate_future = pred[
                "actworld_initial_current_future_world_features"
            ].float()
            target_future = F.normalize(target_future.float(), dim=-1)
            candidate_future = F.normalize(candidate_future, dim=-1)
            compatibility = torch.einsum(
                "bcd,bd->bc", candidate_future, target_future
            )
            score_teacher = torch.softmax(
                simulator_targets[..., -1].float()
                / current_future_jepa_temperature,
                dim=1,
            )
            trajectory_teacher_weight = float(
                config.actworld_current_future_trajectory_teacher_weight
            )
            trajectory_teacher_temperature = float(
                config.actworld_current_future_trajectory_teacher_temperature
            )
            if not 0.0 <= trajectory_teacher_weight <= 1.0:
                raise ValueError("trajectory teacher weight must be in [0, 1]")
            if trajectory_teacher_temperature <= 0.0:
                raise ValueError("trajectory teacher temperature must be positive")
            if trajectory_teacher_weight:
                candidate_trajectory = pred["proposals"].float()
                trajectory_delta = candidate_trajectory - target_trajectory[:, None]
                xy_error = torch.linalg.vector_norm(
                    trajectory_delta[..., :2], dim=-1
                )
                yaw_error = torch.atan2(
                    torch.sin(trajectory_delta[..., 2]),
                    torch.cos(trajectory_delta[..., 2]),
                ).abs()
                trajectory_cost = (xy_error + 0.5 * yaw_error).mean(dim=-1)
                trajectory_teacher = torch.softmax(
                    -trajectory_cost / trajectory_teacher_temperature, dim=1
                )
                teacher = (
                    (1.0 - trajectory_teacher_weight) * score_teacher
                    + trajectory_teacher_weight * trajectory_teacher
                )
            else:
                teacher = score_teacher
            current_future_jepa_loss = -(
                teacher
                * torch.log_softmax(
                    compatibility / current_future_jepa_temperature, dim=1
                )
            ).sum(dim=1).mean()
        else:
            current_future_jepa_loss = loss_dict["loss"].new_zeros(())
        current_future_cross_scene_weight = float(
            config.actworld_current_future_cross_scene_loss_weight
        )
        if current_future_cross_scene_weight < 0:
            raise ValueError(
                "current-future cross-scene loss weight must be nonnegative"
            )
        if current_future_cross_scene_weight:
            if not current_future_jepa_weight:
                raise ValueError(
                    "cross-scene future matching requires Current-Future JEPA"
                )
            # The NAVTRAIN-score-weighted candidate future represents the
            # scene's desired plan-conditioned rollout.  Matching it to its
            # own stopped target future while rejecting other scenes in the
            # minibatch prevents a constant/collapsed latent solution.
            scene_future = F.normalize(
                torch.einsum("bc,bcd->bd", teacher.detach(), candidate_future),
                dim=-1,
            )
            scene_logits = torch.matmul(scene_future, target_future.T)
            scene_logits = scene_logits / current_future_jepa_temperature
            scene_labels = torch.arange(
                scene_logits.shape[0], device=scene_logits.device
            )
            current_future_cross_scene_loss = 0.5 * (
                F.cross_entropy(scene_logits, scene_labels)
                + F.cross_entropy(scene_logits.T, scene_labels)
            )
        else:
            current_future_cross_scene_loss = loss_dict["loss"].new_zeros(())
        teacher_anchor_weight = float(
            config.actworld_current_future_teacher_anchor_loss_weight
        )
        if teacher_anchor_weight < 0:
            raise ValueError(
                "current-future teacher anchor weight must be nonnegative"
            )
        if teacher_anchor_weight:
            if self._teacher_current_future_interaction_v2 is None:
                raise RuntimeError(
                    "teacher anchor is enabled but no frozen interaction teacher exists"
                )
            with torch.no_grad():
                teacher_world = self._teacher_current_future_interaction_v2(
                    pred["actworld_initial_world_features"].detach(),
                    pred["actworld_scene_tokens"].detach(),
                )
            student_world = F.normalize(
                pred["actworld_initial_current_future_world_features"].float(),
                dim=-1,
            )
            teacher_world = F.normalize(teacher_world.float(), dim=-1)
            current_future_teacher_anchor_loss = (
                1.0
                - (student_world * teacher_world).sum(dim=-1)
            ).mean()
        else:
            current_future_teacher_anchor_loss = loss_dict["loss"].new_zeros(())
        current_future_rank_weight = float(
            config.actworld_current_future_rank_loss_weight
        )
        current_future_rank_temperature = float(
            config.actworld_current_future_rank_temperature
        )
        current_future_rank_false_positive_weight = float(
            config.actworld_current_future_rank_false_positive_weight
        )
        current_future_rank_factor_weights = torch.as_tensor(
            (
                float(config.actworld_current_future_rank_nc_weight),
                float(config.actworld_current_future_rank_dac_weight),
                float(config.actworld_current_future_rank_ttc_weight),
                float(config.actworld_current_future_rank_comfort_weight),
            ),
            device=loss_dict["loss"].device,
            dtype=loss_dict["loss"].dtype,
        )
        current_future_rank_component_weights = (
            float(config.actworld_current_future_rank_safety_weight),
            float(config.actworld_current_future_rank_gain_weight),
            float(config.actworld_current_future_rank_improvement_weight),
            float(config.actworld_current_future_rank_pairwise_weight),
            float(config.actworld_current_future_rank_listwise_weight),
        )
        current_future_rank_cvar_fraction = float(
            config.actworld_current_future_rank_cvar_fraction
        )
        current_future_rank_cvar_weight = float(
            config.actworld_current_future_rank_cvar_weight
        )
        current_future_group_dro_eta = float(
            config.actworld_current_future_group_dro_eta
        )
        current_future_group_dro_num_groups = int(
            config.actworld_current_future_group_dro_num_groups
        )
        current_future_rank_boundary_weight = float(
            config.actworld_current_future_rank_boundary_weight
        )
        current_future_rank_boundary_band = float(
            config.actworld_current_future_rank_boundary_band
        )
        current_future_rank_boundary_margin = float(
            config.actworld_current_future_rank_boundary_margin
        )
        current_future_rank_boundary_false_positive_weight = float(
            config.actworld_current_future_rank_boundary_false_positive_weight
        )
        current_future_rank_boundary_require_progress = bool(
            config.actworld_current_future_rank_boundary_require_progress
        )
        if (
            current_future_rank_weight < 0
            or current_future_rank_temperature <= 0
            or current_future_rank_false_positive_weight < 1
            or min(current_future_rank_component_weights) < 0
            or torch.any(current_future_rank_factor_weights <= 0)
            or current_future_rank_boundary_weight < 0
            or current_future_rank_boundary_band <= 0
            or current_future_rank_boundary_margin < 0
            or current_future_rank_boundary_false_positive_weight < 1
            or not 0.0 <= current_future_rank_cvar_weight <= 1.0
            or (
                current_future_rank_cvar_weight > 0.0
                and not 0.0 < current_future_rank_cvar_fraction <= 1.0
            )
            or current_future_group_dro_eta < 0.0
            or (
                current_future_group_dro_eta > 0.0
                and current_future_group_dro_num_groups <= 0
            )
        ):
            raise ValueError("invalid Current-Future direct-rank configuration")
        if current_future_rank_weight:
            rank_prediction = pred[
                "actworld_initial_current_future_rank_prediction"
            ].float()
            baseline = pred["baseline_pdm_score"].float()
            rows = torch.arange(baseline.shape[0], device=baseline.device)
            rank_anchor_mode = getattr(
                config, "actworld_current_future_rank_anchor", "baseline"
            )
            if rank_anchor_mode == "baseline":
                anchor = baseline.argmax(dim=1)
            elif rank_anchor_mode == "deployment":
                anchor = pred.get("actworld_conservative_selected")
                if anchor is None:
                    raise ValueError(
                        "deployment rank anchor requires "
                        "actworld_conservative_selected"
                    )
            elif rank_anchor_mode == "r94_frozen":
                anchor = targets.get("actworld_r94_anchor_index")
                if anchor is None:
                    raise ValueError(
                        "r94_frozen rank anchor requires "
                        "actworld_r94_anchor_index"
                    )
            else:
                raise ValueError(
                    "actworld_current_future_rank_anchor must be "
                    "'baseline', 'deployment', or 'r94_frozen'"
                )
            anchor = anchor.detach().long()
            # The rank head is initialized around the raw Drive-JEPA winner,
            # while deployment starts from the complete frozen mother policy.
            # Re-centering every output channel makes the learned deltas use
            # the same reference proposal as the deployment decision.
            rank_prediction = (
                rank_prediction
                - rank_prediction[rows, anchor, None]
            )
            actual_delta = (
                simulator_targets
                - simulator_targets[rows, anchor, None]
            )
            top_k = min(
                int(config.actworld_outcome_delta_top_k), baseline.shape[1]
            )
            mask = torch.zeros_like(baseline, dtype=torch.bool)
            mask.scatter_(1, baseline.topk(top_k, dim=1).indices, True)
            mask[rows, anchor] = False
            safety_target = (
                actual_delta[..., [0, 1, 3, 4]] >= 0.0
            ).float()
            safety_weight = torch.where(
                safety_target > 0.5,
                torch.ones_like(safety_target),
                torch.full_like(
                    safety_target,
                    current_future_rank_false_positive_weight,
                ),
            )
            rank_safety_per_item = F.binary_cross_entropy_with_logits(
                rank_prediction[..., :4],
                safety_target,
                weight=safety_weight,
                reduction="none",
            )
            # Keep this loss on the deployment rank head, but do not let the
            # nearly-saturated comfort channel dilute DAC/TTC supervision.
            # Normalizing by the sum preserves the scale of the historical
            # unweighted mean when all four weights are one.
            rank_safety_per_item = (
                rank_safety_per_item * current_future_rank_factor_weights
            ).sum(dim=-1) / current_future_rank_factor_weights.sum()
            rank_gain_target = actual_delta[..., 5].clamp(-1.0, 1.0)
            rank_gain_per_item = F.smooth_l1_loss(
                10.0 * rank_prediction[..., 4],
                10.0 * rank_gain_target,
                reduction="none",
            )
            rank_improvement_target = (
                (actual_delta[..., 5] >= 0.0025)
                & (safety_target > 0.5).all(dim=-1)
            ).float()
            rank_improvement_weight = torch.where(
                rank_improvement_target > 0.5,
                torch.ones_like(rank_improvement_target),
                torch.full_like(
                    rank_improvement_target,
                    current_future_rank_false_positive_weight,
                ),
            )
            rank_improvement_per_item = F.binary_cross_entropy_with_logits(
                rank_prediction[..., 5],
                rank_improvement_target,
                weight=rank_improvement_weight,
                reduction="none",
            )
            # Aggregate candidates within each scene before CVaR so each
            # physical scene contributes one robust-risk observation.
            mask_float = mask.to(rank_safety_per_item.dtype)
            candidates_per_scene = mask_float.sum(dim=1).clamp_min(1.0)
            safety_per_scene = (
                rank_safety_per_item * mask_float
            ).sum(dim=1) / candidates_per_scene
            gain_per_scene = (
                rank_gain_per_item * mask_float
            ).sum(dim=1) / candidates_per_scene
            improvement_per_scene = (
                rank_improvement_per_item * mask_float
            ).sum(dim=1) / candidates_per_scene
            if current_future_group_dro_eta > 0.0:
                group_ids = targets.get("actworld_physical_log_id")
                if group_ids is None:
                    raise RuntimeError(
                        "group-DRO requires physical_log_group_targets=true"
                    )
                priority_per_scene = (
                    current_future_rank_component_weights[0] * safety_per_scene
                    + current_future_rank_component_weights[1] * gain_per_scene
                    + current_future_rank_component_weights[2]
                    * improvement_per_scene
                )
                group_weights = self._group_dro_scene_weights(
                    priority_per_scene,
                    group_ids,
                    current_future_group_dro_num_groups,
                    current_future_group_dro_eta,
                )
                group_denominator = group_weights.sum().clamp_min(1e-8)
                current_future_rank_safety_loss = (
                    safety_per_scene * group_weights
                ).sum() / group_denominator
                current_future_rank_gain_loss = (
                    gain_per_scene * group_weights
                ).sum() / group_denominator
                current_future_rank_improvement_loss = (
                    improvement_per_scene * group_weights
                ).sum() / group_denominator
            else:
                current_future_rank_safety_loss = self._mean_tail_risk(
                    safety_per_scene,
                    current_future_rank_cvar_fraction,
                    current_future_rank_cvar_weight,
                )
                current_future_rank_gain_loss = self._mean_tail_risk(
                    gain_per_scene,
                    current_future_rank_cvar_fraction,
                    current_future_rank_cvar_weight,
                )
                current_future_rank_improvement_loss = self._mean_tail_risk(
                    improvement_per_scene,
                    current_future_rank_cvar_fraction,
                    current_future_rank_cvar_weight,
                )
            current_future_rank_pairwise_loss = self._risk_averse_pairwise_loss(
                rank_prediction[..., 4],
                simulator_targets[..., 5],
                baseline,
                top_k,
                current_future_rank_temperature,
                0.0025,
                current_future_rank_false_positive_weight,
                anchor_index=anchor,
            )
            current_future_rank_listwise_loss = self._shortlist_distillation_loss(
                rank_prediction[..., 4],
                simulator_targets[..., 5],
                baseline,
                top_k,
                current_future_rank_temperature,
                0.02,
            )
            current_future_rank_boundary_loss = self._boundary_rank_loss(
                rank_prediction[..., 4],
                simulator_targets,
                baseline,
                top_k,
                current_future_rank_temperature,
                current_future_rank_boundary_band,
                current_future_rank_boundary_margin,
                current_future_rank_boundary_false_positive_weight,
                current_future_rank_boundary_require_progress,
                anchor_index=anchor,
            )
            current_future_rank_loss = (
                current_future_rank_component_weights[0]
                * current_future_rank_safety_loss
                + current_future_rank_component_weights[1]
                * current_future_rank_gain_loss
                + current_future_rank_component_weights[2]
                * current_future_rank_improvement_loss
                + current_future_rank_component_weights[3]
                * current_future_rank_pairwise_loss
                + current_future_rank_component_weights[4]
                * current_future_rank_listwise_loss
                + current_future_rank_boundary_weight
                * current_future_rank_boundary_loss
            )
        else:
            current_future_rank_safety_loss = loss_dict["loss"].new_zeros(())
            current_future_rank_gain_loss = loss_dict["loss"].new_zeros(())
            current_future_rank_improvement_loss = loss_dict["loss"].new_zeros(())
            current_future_rank_pairwise_loss = loss_dict["loss"].new_zeros(())
            current_future_rank_listwise_loss = loss_dict["loss"].new_zeros(())
            current_future_rank_boundary_loss = loss_dict["loss"].new_zeros(())
            current_future_rank_loss = loss_dict["loss"].new_zeros(())
        current_future_progress_head_weight = float(
            config.actworld_current_future_progress_head_loss_weight
        )
        if current_future_progress_head_weight < 0:
            raise ValueError(
                "actworld_current_future_progress_head_loss_weight must be nonnegative"
            )
        if current_future_progress_head_weight:
            progress_prediction = pred[
                "actworld_initial_current_future_progress_prediction"
            ].float().squeeze(-1)
            baseline = pred["baseline_pdm_score"].float()
            rows = torch.arange(baseline.shape[0], device=baseline.device)
            # Align the auxiliary target with the proposal used by the
            # conservative deployment path.  baseline.argmax() is only the
            # raw Drive-JEPA winner and creates a train/deploy anchor drift.
            anchor_mode = getattr(
                config, "actworld_current_future_progress_anchor", "conservative"
            )
            if anchor_mode == "baseline":
                anchor = baseline.argmax(dim=1)
            elif anchor_mode == "conservative":
                anchor = pred.get("actworld_conservative_selected")
                if anchor is None:
                    anchor = baseline.argmax(dim=1)
            elif anchor_mode == "r94_frozen":
                anchor = targets.get("actworld_r94_anchor_index")
                if anchor is None:
                    raise ValueError(
                        "r94_frozen progress anchor requires "
                        "actworld_r94_anchor_index"
                    )
            else:
                raise ValueError(
                    "actworld_current_future_progress_anchor must be "
                    "'conservative', 'baseline', or 'r94_frozen'"
                )
            anchor = anchor.detach().long()
            # The planner consumes progress as a delta relative to its current
            # mother proposal.  The predictor is initially centred on the raw
            # baseline winner, so re-centre it on the same anchor used by the
            # supervision target before computing the auxiliary loss.
            progress_prediction = (
                progress_prediction
                - progress_prediction[rows, anchor][:, None]
            )
            actual_progress_delta = (
                simulator_targets[..., 2]
                - simulator_targets[rows, anchor, None, 2]
            ).clamp(-1.0, 1.0)
            top_k = min(
                int(config.actworld_outcome_delta_top_k), baseline.shape[1]
            )
            progress_mask = torch.zeros_like(baseline, dtype=torch.bool)
            progress_mask.scatter_(1, baseline.topk(top_k, dim=1).indices, True)
            progress_mask[rows, anchor] = False
            progress_per_item = F.smooth_l1_loss(
                10.0 * progress_prediction,
                10.0 * actual_progress_delta,
                reduction="none",
            )
            current_future_progress_head_loss = progress_per_item[progress_mask].mean()
        else:
            current_future_progress_head_loss = loss_dict["loss"].new_zeros(())
        risk_aware_v3_weight = float(config.actworld_risk_aware_v3_loss_weight)
        if risk_aware_v3_weight < 0:
            raise ValueError("actworld_risk_aware_v3_loss_weight must be nonnegative")
        if risk_aware_v3_weight:
            prediction = pred["actworld_initial_risk_aware_prediction"].float()
            baseline = pred["baseline_pdm_score"].float()
            rows = torch.arange(baseline.shape[0], device=baseline.device)
            anchor = baseline.argmax(dim=1)
            actual_delta = simulator_targets - simulator_targets[rows, anchor, None]
            top_k = min(int(config.actworld_outcome_delta_top_k), baseline.shape[1])
            mask = torch.zeros_like(baseline, dtype=torch.bool)
            mask.scatter_(1, baseline.topk(top_k, dim=1).indices, True)
            mask[rows, anchor] = False
            safety_target = (actual_delta[..., [0, 1, 3, 4]] >= 0.0).float()
            false_positive_weight = float(
                config.actworld_risk_aware_v3_false_positive_weight
            )
            safety_weight = torch.where(
                safety_target > 0.5,
                torch.ones_like(safety_target),
                torch.full_like(safety_target, false_positive_weight),
            )
            safety_per_item = F.binary_cross_entropy_with_logits(
                prediction[..., :4], safety_target, weight=safety_weight, reduction="none"
            ).mean(dim=-1)
            gain_target = actual_delta[..., 5].clamp(-1.0, 1.0)
            gain_per_item = F.smooth_l1_loss(
                prediction[..., 4], gain_target, reduction="none"
            )
            improvement_target = (
                (actual_delta[..., 5] > 0.0)
                & (safety_target > 0.5).all(dim=-1)
            ).float()
            improvement_weight = torch.where(
                improvement_target > 0.5,
                torch.ones_like(improvement_target),
                torch.full_like(improvement_target, false_positive_weight),
            )
            improvement_per_item = F.binary_cross_entropy_with_logits(
                prediction[..., 5], improvement_target,
                weight=improvement_weight, reduction="none"
            )
            risk_aware_v3_safety_loss = safety_per_item[mask].mean()
            risk_aware_v3_gain_loss = gain_per_item[mask].mean()
            risk_aware_v3_improvement_loss = improvement_per_item[mask].mean()
            risk_aware_v3_loss = (
                risk_aware_v3_safety_loss
                + risk_aware_v3_gain_loss
                + risk_aware_v3_improvement_loss
            )
        else:
            risk_aware_v3_safety_loss = loss_dict["loss"].new_zeros(())
            risk_aware_v3_gain_loss = loss_dict["loss"].new_zeros(())
            risk_aware_v3_improvement_loss = loss_dict["loss"].new_zeros(())
            risk_aware_v3_loss = loss_dict["loss"].new_zeros(())
        oracle_weight = float(config.actworld_safe_oracle_distill_loss_weight)
        if oracle_weight < 0:
            raise ValueError("actworld_safe_oracle_distill_loss_weight must be nonnegative")
        if oracle_weight:
            (
                safe_oracle_distill_loss,
                safe_oracle_adoption_rate,
                safe_oracle_mean_gain,
            ) = self._safe_oracle_proposal_distillation_loss(
                pred["proposals"],
                pred["baseline_pdm_score"],
                simulator_targets,
                float(config.actworld_safe_oracle_distill_margin),
                float(config.actworld_safe_oracle_distill_max_gain),
            )
        else:
            safe_oracle_distill_loss = loss_dict["loss"].new_zeros(())
            safe_oracle_adoption_rate = loss_dict["loss"].new_zeros(())
            safe_oracle_mean_gain = loss_dict["loss"].new_zeros(())
        refinement_oracle_weight = float(
            config.actworld_safe_oracle_refinement_loss_weight
        )
        if refinement_oracle_weight < 0:
            raise ValueError(
                "actworld_safe_oracle_refinement_loss_weight must be nonnegative"
            )
        if refinement_oracle_weight:
            (
                safe_oracle_refinement_loss,
                safe_oracle_refinement_adoption_rate,
                safe_oracle_refinement_mean_gain,
            ) = self._safe_oracle_refinement_distillation_loss(
                pred["proposals"],
                pred["refined_proposals"],
                pred["baseline_pdm_score"],
                simulator_targets,
                int(config.actworld_outcome_delta_top_k),
                float(config.actworld_safe_oracle_distill_margin),
                float(config.actworld_safe_oracle_distill_max_gain),
            )
        else:
            safe_oracle_refinement_loss = loss_dict["loss"].new_zeros(())
            safe_oracle_refinement_adoption_rate = loss_dict["loss"].new_zeros(())
            safe_oracle_refinement_mean_gain = loss_dict["loss"].new_zeros(())
        world_loss = (
            metric_loss
            + value_loss
            + config.actworld_collision_loss_weight * collision_loss
            + config.actworld_fusion_loss_weight * fusion_loss
        )

        refine_loss = (
            torch.linalg.vector_norm(
                refined - target_trajectory[:, None], ord=1, dim=-1
            )
            .mean(dim=-1)
            .amin(dim=1)
            .mean()
        )
        residual = pred["actworld_trajectory_residual"].float()
        trust_region_loss = residual.square().mean()
        smoothness_loss = pred["actworld_refinement_smoothness_loss"].float()
        expert_refinement_weight = float(
            config.actworld_expert_refinement_loss_weight
        )
        if expert_refinement_weight < 0:
            raise ValueError(
                "actworld_expert_refinement_loss_weight must be nonnegative"
            )
        refinement_rows = torch.arange(
            pred["proposals"].shape[0], device=pred["proposals"].device
        )
        baseline_index = pred["baseline_pdm_score"].float().argmax(dim=1)
        selected_refined = pred["refined_proposals"][
            refinement_rows, baseline_index
        ].float()
        expert_refinement_loss = F.smooth_l1_loss(
            selected_refined, target_trajectory, reduction="none"
        ).mean()

        if config.actworld_refinement_adaptation_only:
            if refinement_oracle_weight <= 0.0:
                raise ValueError(
                    "refinement-adaptation-only requires positive safe-oracle "
                    "refinement supervision"
                )
            # Optional simulator-aware refinement calibration.  Earlier r104
            # optimized only trajectory imitation, so the refined path could
            # move toward the expert while worsening NAVSIM factors.  This
            # term is disabled by default and is used only in the short r108
            # NAVTRAIN screening run.
            refined_score_term = loss_dict["loss"].new_zeros(())
            if refined_weight > 0.0 and refined_targets is not None:
                refined_score_term = (
                    refined_terms["metric"]
                    + config.actworld_collision_loss_weight * refined_terms["collision"]
                    + 0.25 * refined_terms["value"]
                )
            loss_dict["loss"] = (
                refinement_oracle_weight * safe_oracle_refinement_loss
                + expert_refinement_weight * expert_refinement_loss
                + refined_weight * refined_score_term
                + 0.05 * trust_region_loss
                + float(config.actworld_smoothness_loss_weight) * smoothness_loss
            )
        elif config.actworld_planning_adaptation_only:
            if planning_jepa_weight <= 0.0 and oracle_weight <= 0.0:
                raise ValueError(
                    "planning-adaptation-only requires a positive Planning-JEPA "
                    "or safe-oracle distillation weight"
                )
            loss_dict["loss"] = (
                planning_jepa_weight * planning_jepa_loss
                + oracle_weight * safe_oracle_distill_loss
            )
        else:
            loss_dict["loss"] = (
                loss_dict["loss"]
                + config.actworld_value_loss_weight * world_loss
                + config.actworld_rank_loss_weight * rank_loss
                + config.actworld_hard_negative_loss_weight * hard_negative_rank_loss
                + config.actworld_pairwise_loss_weight * pairwise_preference_loss
                + config.actworld_all_pairs_loss_weight * all_pairs_rank_loss
                + config.actworld_progress_pairwise_loss_weight
                * progress_pairwise_loss
                + config.actworld_risk_averse_loss_weight * risk_averse_pairwise_loss
                + config.actworld_shortlist_distillation_loss_weight
                * shortlist_distillation_loss
                + config.actworld_preference_loss_weight * direct_preference_loss
                + config.actworld_factorwise_pairwise_loss_weight
                * factorwise_pairwise_loss
                + planning_jepa_weight * planning_jepa_loss
                + outcome_jepa_weight * outcome_jepa_loss
                + outcome_factor_weight * outcome_factor_loss
                + outcome_delta_weight * outcome_delta_loss
                + outcome_action_delta_weight * outcome_action_delta_loss
                + temporal_action_delta_weight * temporal_action_delta_loss
                + current_future_v2_weight * current_future_v2_loss
                + current_future_jepa_weight * current_future_jepa_loss
                + current_future_cross_scene_weight
                * current_future_cross_scene_loss
                + teacher_anchor_weight * current_future_teacher_anchor_loss
                + current_future_rank_weight * current_future_rank_loss
                + current_future_progress_head_weight
                * current_future_progress_head_loss
                + risk_aware_v3_weight * risk_aware_v3_loss
                + oracle_weight * safe_oracle_distill_loss
                + refinement_oracle_weight * safe_oracle_refinement_loss
                + expert_refinement_weight * expert_refinement_loss
                + config.actworld_refine_loss_weight
                * (refine_loss + 0.1 * trust_region_loss)
                + config.actworld_smoothness_loss_weight * smoothness_loss
            )
        loss_dict.update(
            {
                "actworld_metric_loss": metric_loss,
                "actworld_value_loss": value_loss,
                "actworld_collision_loss": collision_loss,
                "actworld_fusion_loss": fusion_loss,
                "actworld_rank_loss": rank_loss,
                "actworld_hard_negative_rank_loss": hard_negative_rank_loss,
                "actworld_pairwise_preference_loss": pairwise_preference_loss,
                "actworld_all_pairs_rank_loss": all_pairs_rank_loss,
                "actworld_progress_pairwise_loss": progress_pairwise_loss,
                "actworld_risk_averse_pairwise_loss": risk_averse_pairwise_loss,
                "actworld_shortlist_distillation_loss": shortlist_distillation_loss,
                "actworld_direct_preference_loss": direct_preference_loss,
                "actworld_factorwise_pairwise_loss": factorwise_pairwise_loss,
                "actworld_planning_jepa_loss": planning_jepa_loss,
                "actworld_outcome_jepa_loss": outcome_jepa_loss,
                "actworld_outcome_factor_metric_loss": outcome_factor_metric_loss,
                "actworld_outcome_factor_relative_loss": outcome_factor_relative_loss,
                "actworld_outcome_factor_rank_loss": outcome_factor_rank_loss,
                "actworld_outcome_factor_loss": outcome_factor_loss,
                "actworld_outcome_delta_regression_loss": outcome_delta_regression_loss,
                "actworld_outcome_delta_rank_loss": outcome_delta_rank_loss,
                "actworld_outcome_delta_factorwise_loss": outcome_delta_factorwise_loss,
                "actworld_outcome_delta_loss": outcome_delta_loss,
                "actworld_outcome_action_delta_regression_loss": outcome_action_delta_regression_loss,
                "actworld_outcome_action_delta_rank_loss": outcome_action_delta_rank_loss,
                "actworld_outcome_action_delta_factorwise_loss": outcome_action_delta_factorwise_loss,
                "actworld_outcome_action_delta_loss": outcome_action_delta_loss,
                "actworld_temporal_action_delta_regression_loss": temporal_action_delta_regression_loss,
                "actworld_temporal_action_delta_rank_loss": temporal_action_delta_rank_loss,
                "actworld_temporal_action_delta_factorwise_loss": temporal_action_delta_factorwise_loss,
                "actworld_temporal_action_delta_loss": temporal_action_delta_loss,
                "actworld_current_future_v2_regression_loss": current_future_v2_regression_loss,
                "actworld_current_future_v2_rank_loss": current_future_v2_rank_loss,
                "actworld_current_future_v2_factorwise_loss": current_future_v2_factorwise_loss,
                "actworld_current_future_sign_loss": current_future_sign_loss,
                "actworld_current_future_v2_loss": current_future_v2_loss,
                "actworld_current_future_jepa_loss": current_future_jepa_loss,
                "actworld_current_future_cross_scene_loss": current_future_cross_scene_loss,
                "actworld_current_future_teacher_anchor_loss": current_future_teacher_anchor_loss,
                "actworld_current_future_rank_safety_loss": current_future_rank_safety_loss,
                "actworld_current_future_rank_gain_loss": current_future_rank_gain_loss,
                "actworld_current_future_rank_improvement_loss": current_future_rank_improvement_loss,
                "actworld_current_future_rank_pairwise_loss": current_future_rank_pairwise_loss,
                "actworld_current_future_rank_listwise_loss": current_future_rank_listwise_loss,
                "actworld_current_future_rank_boundary_loss": current_future_rank_boundary_loss,
                "actworld_current_future_rank_loss": current_future_rank_loss,
                "actworld_current_future_progress_head_loss": current_future_progress_head_loss,
                "actworld_risk_aware_v3_safety_loss": risk_aware_v3_safety_loss,
                "actworld_risk_aware_v3_gain_loss": risk_aware_v3_gain_loss,
                "actworld_risk_aware_v3_improvement_loss": risk_aware_v3_improvement_loss,
                "actworld_risk_aware_v3_loss": risk_aware_v3_loss,
                "actworld_safe_oracle_distill_loss": safe_oracle_distill_loss,
                "actworld_safe_oracle_adoption_rate": safe_oracle_adoption_rate,
                "actworld_safe_oracle_mean_gain": safe_oracle_mean_gain,
                "actworld_safe_oracle_refinement_loss": safe_oracle_refinement_loss,
                "actworld_safe_oracle_refinement_adoption_rate": safe_oracle_refinement_adoption_rate,
                "actworld_safe_oracle_refinement_mean_gain": safe_oracle_refinement_mean_gain,
                "actworld_expert_refinement_loss": expert_refinement_loss,
                "actworld_refine_loss": refine_loss,
                "actworld_trust_region_loss": trust_region_loss,
                "actworld_smoothness_loss": smoothness_loss,
                "actworld_refinement_gate": pred["actworld_refinement_gate"].mean(),
            }
        )
        return loss_dict

    def get_optimizers(self):
        if self._config.actworld_planning_head_lr_scale <= 0:
            raise ValueError("actworld_planning_head_lr_scale must be positive")
        if self._config.actworld_outcome_jepa_lr_scale <= 0:
            raise ValueError("actworld_outcome_jepa_lr_scale must be positive")
        if self._config.actworld_current_future_rank_representation_lr_scale <= 0:
            raise ValueError(
                "actworld_current_future_rank_representation_lr_scale must be positive"
            )
        outcome_parameters = [
            parameter
            for name, parameter in self._pad_model._actworld_planner.named_parameters()
            if (
                name.startswith("outcome_predictor.")
                or name.startswith("outcome_factor_head.")
                or name.startswith("outcome_delta_head.")
                or name.startswith("outcome_action_delta_head.")
                or name.startswith("temporal_action_jepa_head.")
            )
            and parameter.requires_grad
        ]
        joint_rank_head_parameters = [
            parameter
            for name, parameter in self._pad_model._actworld_planner.named_parameters()
            if self._config.actworld_current_future_rank_joint
            and (
                name.startswith("current_future_rank_head.")
                or name.startswith("current_future_progress_head.")
            )
            and parameter.requires_grad
        ]
        joint_rank_representation_parameters = [
            parameter
            for name, parameter in self._pad_model._actworld_planner.named_parameters()
            if self._config.actworld_current_future_rank_joint
            and name.startswith("current_future_interaction_v2.")
            and parameter.requires_grad
        ]
        planner_parameters = [
            parameter
            for name, parameter in self._pad_model._actworld_planner.named_parameters()
            if not (
                name.startswith("outcome_predictor.")
                or name.startswith("outcome_factor_head.")
                or name.startswith("outcome_delta_head.")
                or name.startswith("outcome_action_delta_head.")
                or name.startswith("temporal_action_jepa_head.")
                or (
                    self._config.actworld_current_future_rank_joint
                    and (
                        name.startswith("current_future_rank_head.")
                        or name.startswith("current_future_progress_head.")
                        or name.startswith("current_future_interaction_v2.")
                    )
                )
            )
            and parameter.requires_grad
        ]
        groups = []
        if joint_rank_head_parameters:
            groups.append({"params": joint_rank_head_parameters, "lr": self._lr})
        if joint_rank_representation_parameters:
            groups.append(
                {
                    "params": joint_rank_representation_parameters,
                    "lr": (
                        self._config.actworld_current_future_rank_representation_lr_scale
                        * self._lr
                    ),
                }
            )
        if planner_parameters:
            groups.append({"params": planner_parameters, "lr": self._lr})
        if outcome_parameters and (
            float(self._config.actworld_outcome_jepa_loss_weight) > 0
            or float(self._config.actworld_outcome_factor_loss_weight) > 0
            or float(self._config.actworld_outcome_delta_loss_weight) > 0
            or float(self._config.actworld_outcome_action_delta_loss_weight) > 0
            or float(self._config.actworld_temporal_action_delta_loss_weight) > 0
        ):
            groups.append(
                {
                    "params": outcome_parameters,
                    "lr": self._config.actworld_outcome_jepa_lr_scale * self._lr,
                }
            )
        if (
            self._config.actworld_freeze_drive_jepa
            and self._config.actworld_unfreeze_planning_heads
        ):
            planning_head_parameters = [
                parameter
                for name, parameter in self._pad_model.named_parameters()
                if parameter.requires_grad and self._is_planning_head_parameter(name)
            ]
            if not planning_head_parameters:
                raise RuntimeError("No trainable Drive-JEPA planning-head parameters found")
            groups.append(
                {
                    "params": planning_head_parameters,
                    "lr": self._config.actworld_planning_head_lr_scale * self._lr,
                }
            )
        if not self._config.actworld_freeze_drive_jepa:
            backbone_parameters = [
                parameter
                for parameter in self._pad_model._backbone.parameters()
                if parameter.requires_grad
            ]
            base_parameters = [
                parameter
                for name, parameter in self._pad_model.named_parameters()
                if parameter.requires_grad
                and not name.startswith("_actworld_planner.")
                and not name.startswith("_backbone.")
            ]
            groups.extend(
                [
                    {
                        "params": backbone_parameters,
                        "lr": self._config.actworld_backbone_lr_scale * self._lr,
                    },
                    {
                        "params": base_parameters,
                        "lr": self._config.actworld_base_lr_scale * self._lr,
                    },
                ]
            )
        if not any(group["params"] for group in groups):
            raise RuntimeError("ActWorld-JEPA optimizer has no trainable parameters")
        return torch.optim.AdamW(groups, lr=self._lr, weight_decay=1e-4)


__all__ = ["ActWorldJEPAAgent"]
