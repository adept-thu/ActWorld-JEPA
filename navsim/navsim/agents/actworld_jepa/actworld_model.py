"""Drive-JEPA v1 with candidate-conditioned latent world planning."""

from typing import Dict, Tuple

import torch

from navsim.agents.drive_jepa_perception_based.drive_jepa_model import DriveJEPAModel

from .actworld_config import ActWorldJEPAConfig
from .actworld_planner import ActWorldPlanner


class ActWorldJEPAModel(DriveJEPAModel):
    """Preserve the v1 encoder/proposals/scorer and replace direct selection."""

    def __init__(self, config: ActWorldJEPAConfig):
        super().__init__(config)
        self._actworld_planner = ActWorldPlanner(config)
        self._keep_frozen_drive_jepa_in_eval()

    def _keep_frozen_drive_jepa_in_eval(self) -> None:
        if not self._config.actworld_freeze_drive_jepa:
            return
        for name, module in self.named_children():
            if name != "_actworld_planner":
                module.eval()
        if self.training and self._config.actworld_unfreeze_planning_heads:
            self._trajectory_head.train(True)
            self.hist_encoding.train(True)
            self.init_feature.train(True)
        if self._config.actworld_freeze_planner_during_proposal_adaptation:
            self._actworld_planner.eval()

    def train(self, mode: bool = True) -> "ActWorldJEPAModel":
        super().train(mode)
        if mode:
            self._keep_frozen_drive_jepa_in_eval()
            self._actworld_planner.train(
                not self._config.actworld_freeze_planner_during_proposal_adaptation
            )
        return self

    @staticmethod
    def _scene_tokens(image_feature: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        memory = image_feature[0]
        if memory.ndim != 4 or memory.shape[0] != 1:
            raise ValueError(
                "ActWorld-JEPA v1 expects V-JEPA memory [1,N,B,D], got "
                f"{tuple(memory.shape)}"
            )
        return memory[0].permute(1, 0, 2).contiguous()

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features["lidar2img"] = features["lidar2img"][:, 1:2]
        ego_status: torch.Tensor = features["ego_status"][:, -1]
        if self.b2d:
            ego_status = ego_status.clone()
            ego_status[:, 1:3] = 0

        previous = self.transform(features["camera_feature_2"])
        current = self.transform(features["camera_feature_1"])
        camera_clip = torch.cat((previous[:, None], current[:, None]), dim=1)
        image_feature = self._backbone(camera_clip, img_metas=features)
        scene_tokens = self._scene_tokens(image_feature)

        proposal_feature = self.hist_encoding(ego_status)[:, None]
        proposal_feature = proposal_feature + self.init_feature.weight[None]
        proposal_list = []
        for refiner in self._trajectory_head:
            proposal_feature, proposal_list = refiner(
                proposal_feature, proposal_list, image_feature
            )
        proposals = proposal_list[-1]

        (
            pred_logit,
            pred_logit2,
            pred_agents_states,
            pred_area_logit,
            bev_semantic_map,
            agent_states,
            agent_labels,
        ) = self.scorer(proposals, proposal_feature)
        proposal_plan_features = proposal_feature.reshape(
            proposals.shape[0], proposals.shape[1], proposals.shape[2], -1
        ).amax(dim=2)
        drive_jepa_subscores = torch.sigmoid(pred_logit)
        if pred_logit2 is not None:
            drive_jepa_subscores = 0.5 * (
                drive_jepa_subscores + torch.sigmoid(pred_logit2)
            )
        baseline_score = torch.sigmoid(pred_logit)[..., -1]
        if pred_logit2 is not None:
            baseline_score = (
                baseline_score + torch.sigmoid(pred_logit2)[..., -1]
            ) / 2

        world_output = self._actworld_planner(
            scene_tokens.to(dtype=proposals.dtype),
            proposals,
            baseline_score,
            return_auxiliary=self.training,
        )
        refined_proposals = world_output["refined_candidates"]
        if self._config.actworld_selection_mode == "conservative":
            # Selection is always computed from untouched Drive-JEPA
            # proposals/features.  An explicitly trained refinement branch may
            # correct only the already selected candidate, so scorer decisions
            # remain bit-exact while planning quality can improve.
            deployment_proposals = (
                refined_proposals
                if self._config.actworld_deploy_refined_selected
                else proposals
            )
            pdm_score = world_output["conservative_gated_scores"]
            selected = world_output["conservative_selected"]
        else:
            deployment_proposals = refined_proposals
            pdm_score = world_output["world_scores"]
            selected = torch.argmax(pdm_score, dim=1)
        rows = torch.arange(proposals.shape[0], device=selected.device)
        selected_trajectory_scale = float(
            getattr(self._config, "actworld_selected_trajectory_scale", 1.0)
        )
        if selected_trajectory_scale <= 0.0:
            raise ValueError("actworld_selected_trajectory_scale must be positive")
        if (
            self._config.actworld_selection_mode == "conservative"
            and selected_trajectory_scale != 1.0
        ):
            # Preserve every frozen selector decision and adjust only the
            # trajectory it executes.  This separates proposal selection from
            # a small plan-space continuation step and makes the latter easy to
            # audit without retraining or changing any candidate score.
            anchor = world_output["conservative_baseline_selected"]
            anchor_trajectory = proposals[rows, anchor]
            selected_trajectory = proposals[rows, selected]
            scaled_trajectory = anchor_trajectory + selected_trajectory_scale * (
                selected_trajectory - anchor_trajectory
            )
            deployment_proposals = proposals.clone()
            deployment_proposals[rows, selected] = scaled_trajectory
            refined_proposals = deployment_proposals

        output: Dict[str, torch.Tensor] = {
            # Activation used only by the NAVTRAIN planning-JEPA auxiliary
            # objective; it introduces no new checkpoint parameter.
            "actworld_scene_tokens": scene_tokens,
            # Original candidates remain the contract for released v1 losses.
            "proposals": proposals,
            "proposal_list": proposal_list,
            "refined_proposals": refined_proposals,
            "trajectory": deployment_proposals[rows, selected],
            "selected_proposal": selected,
            "pdm_score": pdm_score,
            "baseline_pdm_score": baseline_score,
            # Preserve the candidate-specific representation and all six
            # structured Drive-JEPA predictions for NAVTRAIN-only audits.
            # These are read-only activations; exposing them does not alter
            # the historical r94 inference path or checkpoint parameters.
            "drive_jepa_proposal_features": proposal_plan_features,
            "drive_jepa_candidate_subscores": drive_jepa_subscores,
            "pred_logit": pred_logit,
            "pred_logit2": pred_logit2,
            "pred_agents_states": pred_agents_states,
            "pred_area_logit": pred_area_logit,
            "bev_semantic_map": bev_semantic_map,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
        }
        output.update({f"actworld_{key}": value for key, value in world_output.items()})
        return output


__all__ = ["ActWorldJEPAModel"]
