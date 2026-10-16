"""Drive-JEPA model with action-conditioned latent planning."""

from contextlib import contextmanager
import math
import os
from typing import Dict, Iterator, Tuple

import torch

from navsim.agents.drive_jepa_perception_based.drive_jepa_model import (
    DriveJEPAModel,
)

from .actworld_config import ActWorldJEPAConfig
from .actworld_planner import ActWorldPlanner


class ActWorldJEPAModel(DriveJEPAModel):
    """Retains Drive-JEPA representation/proposals and replaces direct argmax.

    Checkpoint-compatible Drive-JEPA modules keep their original names.  The
    only new state lives under ``_actworld_planner`` so an official checkpoint
    can be loaded with ``strict=False`` as a deterministic warm start.
    """

    def __init__(self, config: ActWorldJEPAConfig):
        super().__init__(config)
        self._actworld_planner = ActWorldPlanner(config)
        self.configure_drive_jepa_trainability()

    def _selective_stage2_modules(self) -> Tuple[torch.nn.Module, ...]:
        """Resolve the decoder and top V-JEPA modules used by Stage 2."""

        image_backbone = getattr(self._backbone, "img_backbone", None)
        blocks = getattr(image_backbone, "blocks", None)
        top_blocks = self._config.actworld_backbone_top_blocks
        if (
            isinstance(top_blocks, bool)
            or not isinstance(top_blocks, int)
            or top_blocks <= 0
        ):
            raise ValueError("actworld_backbone_top_blocks must be a positive integer")
        if not isinstance(blocks, torch.nn.ModuleList) or top_blocks > len(blocks):
            count = len(blocks) if isinstance(blocks, torch.nn.ModuleList) else None
            raise ValueError(
                "selective Stage 2 requires a V-JEPA ModuleList at "
                f"_backbone.img_backbone.blocks; found {count!r} blocks"
            )
        modules = [self._trajectory_head, self._backbone.projector]
        modules.extend(blocks[-top_blocks:])
        norm = getattr(image_backbone, "norm", None)
        if isinstance(norm, torch.nn.Module):
            modules.append(norm)
        return tuple(modules)

    def configure_drive_jepa_trainability(self) -> None:
        """Apply frozen, selective Stage-2, or legacy full-finetune policy."""

        selective = bool(
            getattr(
                self._config,
                "actworld_selective_unfreeze_planning_top",
                False,
            )
        )
        for name, parameter in self.named_parameters():
            if not name.startswith("_actworld_planner."):
                parameter.requires_grad_(
                    not self._config.actworld_freeze_drive_jepa and not selective
                )
        if selective:
            for module in self._selective_stage2_modules():
                module.requires_grad_(True)
        self._apply_drive_jepa_module_modes(self.training)

    def _apply_drive_jepa_module_modes(self, mode: bool) -> None:
        """Keep frozen base modules deterministic while selected layers train."""

        if not mode:
            return
        selective = bool(
            getattr(
                self._config,
                "actworld_selective_unfreeze_planning_top",
                False,
            )
        )
        if self._config.actworld_freeze_drive_jepa or selective:
            for name, module in self.named_children():
                if name != "_actworld_planner":
                    module.eval()
        if selective:
            for module in self._selective_stage2_modules():
                module.train(True)

    def train(self, mode: bool = True) -> "ActWorldJEPAModel":
        super().train(mode)
        if mode:
            self._apply_drive_jepa_module_modes(True)
            self._actworld_planner.train(True)
        return self

    def _f2_frozen_drive_jepa_enabled(self) -> bool:
        """Whether this forward must reproduce the official F32 teacher."""

        config = getattr(self, "_config", None)
        planner = getattr(self, "_actworld_planner", None)
        planner_config = getattr(planner, "config", None)
        return bool(
            getattr(config, "actworld_freeze_drive_jepa", False)
            and getattr(planner_config, "critic_variant", None)
            == "f2_base_residual"
            and getattr(planner, "training_stage", None) == "critic"
        )

    @contextmanager
    def _f2_frozen_drive_jepa_context(
        self, reference: torch.Tensor
    ) -> Iterator[None]:
        """Disable gradients/autocast only around the frozen F2 teacher."""

        if not self._f2_frozen_drive_jepa_enabled():
            yield
            return
        with torch.no_grad(), torch.autocast(
            device_type=reference.device.type, enabled=False
        ):
            yield

    def _f2_float32_features(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not self._f2_frozen_drive_jepa_enabled():
            return features
        return {
            name: value.float()
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
            else value
            for name, value in features.items()
        }

    @staticmethod
    def _scene_tokens(image_feature: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Convert Drive-JEPA's `[camera,tokens,batch,dim]` memory to BND."""

        memory = image_feature[0]
        if memory.ndim != 4 or memory.shape[0] != 1:
            raise ValueError(
                "ActWorld-JEPA expects one-camera V-JEPA memory shaped "
                f"[1,N,B,D], got {tuple(memory.shape)}"
            )
        return memory[0].permute(1, 0, 2).contiguous()

    @staticmethod
    def _observed_ego_action(ego_status: torch.Tensor) -> torch.Tensor:
        if ego_status.ndim != 3 or ego_status.shape[1] < 2 or ego_status.shape[2] < 3:
            raise ValueError("ego_status must contain at least two [x,y,yaw] history states")
        action = ego_status[:, -1, :3] - ego_status[:, -2, :3]
        wrapped_yaw = torch.atan2(torch.sin(action[:, 2]), torch.cos(action[:, 2]))
        return torch.cat((action[:, :2], wrapped_yaw[:, None]), dim=-1)

    def _encode_current_clip(
        self, features: Dict[str, torch.Tensor]
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        newest = self.transform(features["camera_feature_1"])
        previous = self.transform(features["camera_feature_2"])
        clip = torch.cat((previous[:, None], newest[:, None]), dim=1)
        image_feature = self._backbone(clip, img_metas=features)
        return image_feature, self._scene_tokens(image_feature)

    def _historical_latent_loss(
        self,
        features: Dict[str, torch.Tensor],
        current_tokens: torch.Tensor,
        ego_status: torch.Tensor,
    ) -> torch.Tensor:
        teacher_features = self._f2_float32_features(features)
        teacher_ego_status = (
            ego_status.float()
            if self._f2_frozen_drive_jepa_enabled()
            else ego_status
        )
        with self._f2_frozen_drive_jepa_context(teacher_ego_status):
            older = self.transform(teacher_features["camera_feature_3"])
            previous = self.transform(teacher_features["camera_feature_2"])
            previous_clip = torch.cat((older[:, None], previous[:, None]), dim=1)
            previous_feature = self._backbone(
                previous_clip, img_metas=teacher_features
            )
            previous_tokens = self._scene_tokens(previous_feature)
            action = self._observed_ego_action(teacher_ego_status)
        previous_tokens = previous_tokens.to(current_tokens.dtype)
        action = action.to(current_tokens.dtype)
        return self._actworld_planner.latent_consistency_loss(
            previous_tokens, current_tokens, action
        )

    @torch.no_grad()
    def _encode_future_clip(self, future_clip: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(future_clip, torch.Tensor)
            or future_clip.ndim != 5
            or tuple(future_clip.shape[1:3]) != (2, 3)
        ):
            shape = (
                tuple(future_clip.shape)
                if isinstance(future_clip, torch.Tensor)
                else type(future_clip)
            )
            raise ValueError(
                "future camera clip must have shape [B, 2, 3, H, W], "
                f"got {shape}"
            )
        teacher_clip = (
            future_clip.float()
            if self._f2_frozen_drive_jepa_enabled()
            else future_clip
        )
        with self._f2_frozen_drive_jepa_context(teacher_clip):
            older = self.transform(teacher_clip[:, 0])
            terminal = self.transform(teacher_clip[:, 1])
            encoded = self._backbone(
                torch.stack((older, terminal), dim=1), img_metas={}
            )
        return self._scene_tokens(encoded)

    def future_latent_grounding_loss(
        self,
        current_scene_latent: torch.Tensor,
        expert_trajectory: torch.Tensor,
        future_clip: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Build the stop-gradient future target and supervise expert rollout."""

        future_tokens = self._encode_future_clip(future_clip).to(
            device=current_scene_latent.device,
            dtype=current_scene_latent.dtype,
        )
        return self._actworld_planner.terminal_latent_consistency_loss(
            current_scene_latent,
            expert_trajectory.to(dtype=current_scene_latent.dtype),
            future_tokens,
            valid,
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        *,
        f2_parallel_loss_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run normal inference or the explicit F2 parallel-loss path.

        ``f2_parallel_loss_diagnostics`` is reserved for Lightning validation
        of a training-time F2 critic.  It exposes the same auxiliary tensors
        and historical latent loss as a training forward, but deliberately
        leaves the module in evaluation mode and never updates the EMA target.
        Ordinary causal checkpoint inference does not set this flag.
        """

        if not isinstance(f2_parallel_loss_diagnostics, bool):
            raise TypeError("f2_parallel_loss_diagnostics must be a bool")
        if f2_parallel_loss_diagnostics:
            if self.training:
                raise RuntimeError(
                    "F2 parallel loss diagnostics require model.eval()"
                )
            if (
                self._actworld_planner.training_stage != "critic"
                or self._actworld_planner.config.critic_variant
                != "f2_base_residual"
            ):
                raise RuntimeError(
                    "F2 parallel loss diagnostics require an F2 Stage-A critic"
                )

        critic_training = (
            self.training
            and self._actworld_planner.training_stage == "critic"
        )
        if critic_training:
            # Once per optimizer step in the usual Lightning loop: on the next
            # batch this incorporates the preceding online-planner update.
            self._actworld_planner.update_target_encoder()
        features["lidar2img"] = features["lidar2img"][:, 1:2]
        teacher_features = self._f2_float32_features(features)
        ego_status: torch.Tensor = teacher_features["ego_status"][:, -1]

        # The official baseline artifact is generated in full float32.  Keep
        # exactly the frozen encoder/proposal/scorer path outside mixed
        # precision, then leave this context before running the trainable
        # world critic so Lightning's outer AMP policy remains effective.
        with self._f2_frozen_drive_jepa_context(ego_status):
            image_feature, current_tokens = self._encode_current_clip(
                teacher_features
            )
            batch_size = ego_status.shape[0]
            ego_feature = self.hist_encoding(ego_status)[:, None]
            proposal_feature = ego_feature + self.init_feature.weight[None]

            proposal_list = []
            for refine in self._trajectory_head:
                proposal_feature, proposals = refine(
                    proposal_feature, image_feature
                )
                proposal_list.append(proposals)
            proposals = proposal_list[-1]

            (
                pred_logit,
                pred_agents_states,
                pred_area_logit,
                bev_semantic_map,
                agent_states,
                agent_labels,
            ) = self.scorer(proposals, proposal_feature)

        # Keep planner/candidate dtypes aligned under mixed precision.  Legacy
        # training retains its original input graph; the F2 teacher tensors
        # are intentionally detached while the world critic stays trainable.
        current_tokens = current_tokens.to(dtype=proposals.dtype)
        world_output = self._actworld_planner(
            current_tokens,
            proposals,
            pred_logit,
            return_auxiliary=(self.training or f2_parallel_loss_diagnostics),
        )
        planner_refined_proposals = world_output["refined_candidates"]
        f2_original_proposal_deployment = (
            self._actworld_planner.config.critic_variant == "f2_base_residual"
            and self._actworld_planner.training_stage == "critic"
        )
        # F2 is a Stage-A critic over the original Drive-JEPA proposal set.
        # Keep the planner's refiner tensors under the actworld_* diagnostic
        # namespace, but make every public deployment path use the originals
        # even if an invalid checkpoint were to contain a nonzero refiner.
        refined_proposals = (
            proposals
            if f2_original_proposal_deployment
            else planner_refined_proposals
        )
        pdm_score = world_output["world_scores"]
        base_pdm_score = torch.sigmoid(pred_logit[..., -1])

        # Planning-JEPA predicts the nine official metric factors for every
        # candidate.  A small NAVTRAIN-selected blend can expose that
        # decomposed future signal to ranking while the separately anchored
        # base score remains the exact released aggregate-score mother for
        # safe fallback.
        factorized_blend = float(
            os.environ.get("ACTWORLD_BASE_FACTORIZED_BLEND", "0.0")
        )
        if (
            not math.isfinite(factorized_blend)
            or not 0.0 <= factorized_blend <= 1.0
        ):
            raise ValueError(
                "ACTWORLD_BASE_FACTORIZED_BLEND must be finite and in [0, 1], "
                f"got {factorized_blend!r}"
            )
        if factorized_blend > 0.0:
            base_factorized_score = self._factorized_candidate_score(
                world_output["candidate_subscores"]
            )
            pdm_score = (
                (1.0 - factorized_blend) * pdm_score
                + factorized_blend * base_factorized_score
            )
        else:
            # Do not impose the factor-head shape contract on the exact
            # backward-compatible path (including compact test scorers).
            base_factorized_score = torch.zeros_like(pdm_score)

        # The released scorer exposes a separately supervised ego-progress
        # channel at index 4.  A small NAVTRAIN-selected residual lets the
        # candidate ranker recover progress that the aggregate score can
        # underweight, while preserving the released behavior at zero.
        progress_selection_weight = float(
            os.environ.get("ACTWORLD_PROGRESS_SELECTION_WEIGHT", "0.0")
        )
        if (
            not math.isfinite(progress_selection_weight)
            or progress_selection_weight < 0.0
        ):
            raise ValueError(
                "ACTWORLD_PROGRESS_SELECTION_WEIGHT must be finite and "
                f"non-negative, got {progress_selection_weight!r}"
            )
        if progress_selection_weight > 0.0:
            progress_score = torch.sigmoid(pred_logit[..., 4])
            denominator = 1.0 + progress_selection_weight
            pdm_score = (
                pdm_score + progress_selection_weight * progress_score
            ) / denominator
            base_pdm_score = (
                base_pdm_score + progress_selection_weight * progress_score
            ) / denominator

        if (
            "past_ego_simulated_states" in features
            and features["past_ego_simulated_states"] is not None
        ):
            pdm_score = self.calibrate_score(
                features, refined_proposals, pdm_score.clone()
            )
            base_ec_weight = os.environ.get(
                "ACTWORLD_BASE_EC_SELECTION_WEIGHT"
            )
            if base_ec_weight is None:
                base_pdm_score = self.calibrate_score(
                    features, proposals, base_pdm_score.clone()
                )
            else:
                base_pdm_score = self.calibrate_score(
                    features,
                    proposals,
                    base_pdm_score.clone(),
                    ec_selection_weight=float(base_ec_weight),
                )

        safety_veto = torch.zeros(
            batch_size, dtype=torch.bool, device=pdm_score.device
        )
        safety_veto_margin = os.environ.get("ACTWORLD_F2_SAFETY_VETO_MARGIN")
        if safety_veto_margin is not None:
            pdm_score, safety_veto = self._apply_f2_safety_veto(
                pdm_score,
                base_pdm_score,
                world_output["candidate_subscores"],
                float(safety_veto_margin),
            )

        selected = torch.argmax(pdm_score, dim=1)
        rows = torch.arange(batch_size, device=selected.device)
        trajectory = refined_proposals[rows, selected]

        output: Dict[str, torch.Tensor] = {
            # Original proposals remain the supervision contract for the
            # released Drive-JEPA scorer and pseudo-trajectory losses.
            "proposals": proposals,
            "proposal_list": proposal_list,
            "refined_proposals": refined_proposals,
            "trajectory": trajectory,
            "selected_proposal": selected,
            "pdm_score": pdm_score,
            "base_pdm_score": base_pdm_score,
            "actworld_base_factorized_score": base_factorized_score,
            "actworld_safety_veto": safety_veto,
            "pred_logit": pred_logit,
            "pred_agents_states": pred_agents_states,
            "pred_area_logit": pred_area_logit,
            "bev_semantic_map": bev_semantic_map,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            # A compact, causal summary of the current Drive-JEPA scene
            # tokens.  It is diagnostic-only unless a frozen NAVTRAIN ranker
            # explicitly consumes it; exposing it does not alter inference.
            "actworld_current_context": torch.cat(
                (
                    current_tokens.mean(dim=1),
                    current_tokens.std(dim=1, unbiased=False),
                    current_tokens.amax(dim=1),
                ),
                dim=-1,
            ),
        }
        output.update({f"actworld_{key}": value for key, value in world_output.items()})

        if critic_training or f2_parallel_loss_diagnostics:
            if "camera_feature_3" not in features:
                raise ValueError(
                    "ActWorld critic loss diagnostics require the third causal "
                    "history frame"
                )
            output["actworld_latent_consistency_loss"] = self._historical_latent_loss(
                teacher_features,
                current_tokens,
                teacher_features["ego_status"],
            )
        return output

    @staticmethod
    def _factorized_candidate_score(
        candidate_subscores: torch.Tensor,
    ) -> torch.Tensor:
        """Recompose Planning-JEPA's nine official factor predictions."""

        if candidate_subscores.ndim != 3 or candidate_subscores.shape[-1] != 9:
            raise ValueError(
                "Planning-JEPA candidate subscores must have shape [B, C, 9]"
            )
        factors = candidate_subscores
        hard = factors[..., :4].prod(dim=-1)
        soft = (
            5.0 * factors[..., 4]
            + 5.0 * factors[..., 5]
            + 2.0 * factors[..., 6]
            + 2.0 * factors[..., 7]
            + 2.0 * factors[..., 8]
        ) / 16.0
        return hard * soft

    @staticmethod
    def _apply_f2_safety_veto(
        pdm_score: torch.Tensor,
        base_pdm_score: torch.Tensor,
        candidate_subscores: torch.Tensor,
        margin: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fall back when the JEPA choice has lower predicted hard safety."""

        if tuple(pdm_score.shape) != tuple(base_pdm_score.shape):
            raise ValueError("F2 and base scores must have identical shape")
        if (
            candidate_subscores.ndim != 3
            or tuple(candidate_subscores.shape[:2]) != tuple(pdm_score.shape)
            or candidate_subscores.shape[-1] != 9
        ):
            raise ValueError("F2 candidate subscores must have shape [B, C, 9]")
        if not math.isfinite(margin) or not 0.0 <= margin <= 1.0:
            raise ValueError("F2 safety veto margin must be finite and in [0, 1]")

        rows = torch.arange(pdm_score.shape[0], device=pdm_score.device)
        selected = torch.argmax(pdm_score, dim=1)
        base_selected = torch.argmax(base_pdm_score, dim=1)
        hard_safety = candidate_subscores[..., :4].prod(dim=-1)
        selected_safety = hard_safety[rows, selected]
        base_safety = hard_safety[rows, base_selected]
        veto = (selected != base_selected) & (
            selected_safety + margin < base_safety
        )

        adjusted = pdm_score.clone()
        if bool(veto.any().item()):
            winning_score = adjusted.amax(dim=1)
            epsilon = torch.finfo(adjusted.dtype).eps
            adjusted[rows[veto], base_selected[veto]] = (
                winning_score[veto] + epsilon
            )
        return adjusted, veto
