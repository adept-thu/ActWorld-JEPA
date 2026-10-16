import pytorch_lightning as pl

import torch
from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self.agent.forward(features)
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        if isinstance(loss_dict, dict):
            synchronized_keys = {"loss", "score", "best_score"}
            progress_keys = {
                "loss",
                "score",
                "best_score",
                "actworld_future_latent_loss",
                "actworld_refined_oracle_gap",
                "actworld_actor_improvement_loss",
                "actworld_actor_fixed_delta",
                "actworld_e2e_delta",
            }
            for key,value in loss_dict.items():
                self.log(
                    f"{logging_prefix}/" + key,
                    value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=key in progress_keys,
                    sync_dist=key in synchronized_keys,
                )
            return loss_dict["loss"]
        else:
            return loss_dict

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def _f2_parallel_validation_step(
        self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]]
    ) -> Tensor:
        """Log only explicitly non-authoritative F2 loss diagnostics."""

        features, targets = batch
        prediction = self.agent.forward_f2_parallel_loss_diagnostics(features)
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        if not isinstance(loss_dict, dict) or "loss" not in loss_dict:
            raise TypeError("F2 parallel validation requires a loss dictionary")

        diagnostic_losses = {
            key: value
            for key, value in loss_dict.items()
            if key == "loss" or key.endswith("_loss")
        }
        for key, value in diagnostic_losses.items():
            self.log(
                f"val/f2_parallel_non_authoritative_{key}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict["loss"]

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if (
            getattr(self.agent, "_training_stage", None) == "critic"
            and getattr(self.agent, "_critic_variant", "legacy")
            == "f2_base_residual"
        ):
            # The parallel Lightning loop is loss diagnostics only for F2.
            # Checkpoint ranking and acceptance belong to the separate causal
            # navtrain validator, which owns per-log comfort state.
            return self._f2_parallel_validation_step(batch)
        if 'Pad' in self.agent.name() or 'perception_based' in self.agent.name():
            features, targets = batch
            predictions = self.agent.forward(features)
            actworld_stage = getattr(self.agent, "_training_stage", None)
            actor_validation = (
                actworld_stage == "actor"
                and "base_pdm_score" in predictions
                and "refined_proposals" in predictions
            )
            critic_validation = (
                actworld_stage == "critic"
                and "base_pdm_score" in predictions
                and "actworld_utility_logits" in predictions
            )
            actor_fixed_delta = None
            e2e_delta = None
            rerank_delta = None
            actor_hard_deltas = None
            e2e_hard_deltas = None
            critic_delta = None
            critic_regret = None
            critic_pairwise_accuracy = None
            critic_selected_score = None
            critic_score_error = None
            critic_hard_deltas = None
            if actor_validation:
                rows = torch.arange(
                    predictions["proposals"].shape[0],
                    device=predictions["proposals"].device,
                )
                base_selected = torch.argmax(
                    predictions["base_pdm_score"], dim=1
                )
                final_selected = torch.argmax(predictions["pdm_score"], dim=1)
                original = predictions["proposals"]
                refined = predictions["refined_proposals"]
                diagnostic_trajectories = torch.stack(
                    (
                        original[rows, base_selected],
                        refined[rows, base_selected],
                        original[rows, final_selected],
                        refined[rows, final_selected],
                    ),
                    dim=1,
                )
                proposal_scores, best_scores, diagnostic_scores = self.agent.compute_score(
                    targets,
                    diagnostic_trajectories,
                    test=True,
                    label_only=True,
                )
                base_true = proposal_scores[:, 0]
                final_true = proposal_scores[:, 3]
                final_score = final_true.mean()
                best_score = best_scores.mean()
                mean_score = proposal_scores.mean()
                actor_fixed_delta = (proposal_scores[:, 1] - base_true).mean()
                rerank_delta = (proposal_scores[:, 2] - base_true).mean()
                e2e_delta = (final_true - base_true).mean()
                actor_hard_deltas = (
                    diagnostic_scores[:, 1, :4]
                    - diagnostic_scores[:, 0, :4]
                ).mean(dim=0)
                e2e_hard_deltas = (
                    diagnostic_scores[:, 3, :4]
                    - diagnostic_scores[:, 0, :4]
                ).mean(dim=0)
                l2 = torch.linalg.vector_norm(
                    diagnostic_trajectories[:, 3] - targets["trajectory"],
                    dim=-1,
                )[:, :4].mean()
                selected_probability = predictions["pdm_score"][rows, final_selected]
                score_error = torch.abs(selected_probability - final_true).mean()
                trajectoy_scores = diagnostic_scores[:, 3]
            elif critic_validation:
                original = predictions["proposals"]
                rows = torch.arange(original.shape[0], device=original.device)
                proposal_scores, best_scores, diagnostic_scores = self.agent.compute_score(
                    targets,
                    original,
                    test=True,
                    label_only=True,
                )
                utility_logits = predictions["actworld_utility_logits"]
                critic_selected = torch.argmax(utility_logits, dim=1)
                base_selected = torch.argmax(predictions["base_pdm_score"], dim=1)
                critic_true = proposal_scores[rows, critic_selected]
                base_true = proposal_scores[rows, base_selected]
                # Keep val/score as the score of the actually deployed
                # zero-gated planner.  Stage-A checkpointing monitors the
                # separate critic-selected score below.
                final_score = base_true.mean()
                best_score = best_scores.mean()
                mean_score = proposal_scores.mean()
                critic_selected_score = critic_true.mean()
                critic_delta = (critic_true - base_true).mean()
                critic_regret = (best_scores - critic_true).mean()
                critic_selected_probability = torch.sigmoid(
                    utility_logits[rows, critic_selected]
                )
                critic_score_error = torch.abs(
                    critic_selected_probability - critic_true
                ).mean()
                critic_hard_deltas = (
                    diagnostic_scores[rows, critic_selected, :4]
                    - diagnostic_scores[rows, base_selected, :4]
                ).mean(dim=0)
                selected_probability = predictions["pdm_score"][rows, base_selected]
                score_error = torch.abs(selected_probability - base_true).mean()
                selected_trajectories = original[rows, base_selected]
                l2 = torch.linalg.vector_norm(
                    selected_trajectories - targets["trajectory"], dim=-1
                )[:, :4].mean()
                trajectoy_scores = diagnostic_scores[rows, base_selected]

                utility_difference = (
                    utility_logits[:, :, None] - utility_logits[:, None, :]
                )
                target_difference = (
                    proposal_scores[:, :, None] - proposal_scores[:, None, :]
                )
                comparable = target_difference.ne(0)
                comparable_count = comparable.sum()
                correct_pairs = (
                    utility_difference * target_difference > 0
                ) & comparable
                critic_pairwise_accuracy = (
                    correct_pairs.sum().float()
                    / comparable_count.clamp_min(1).float()
                )
            else:
                all_res=predictions["trajectory"][:,None]
                final_score,best_score,proposal_scores,l2,trajectoy_scores=self.agent.compute_score(targets,all_res)
                mean_score=proposal_scores.mean()
                pdm_score=predictions["pdm_score"]
                pdm_score=pdm_score[torch.arange(len(pdm_score)), torch.argmax(pdm_score, dim=1)]
                score_error=torch.abs(pdm_score - proposal_scores).mean()
            logging_prefix="val"
            self.log(f"{logging_prefix}/score", final_score, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/best_score", best_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/mean_score", mean_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/score_error", score_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/l2", l2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            if actor_validation:
                self.log(
                    f"{logging_prefix}/actor_fixed_delta",
                    actor_fixed_delta,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"{logging_prefix}/rerank_original_delta",
                    rerank_delta,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    f"{logging_prefix}/e2e_delta",
                    e2e_delta,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
            if critic_validation:
                self.log(
                    f"{logging_prefix}/critic_selected_score",
                    critic_selected_score,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"{logging_prefix}/critic_delta",
                    critic_delta,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"{logging_prefix}/critic_oracle_regret",
                    critic_regret,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"{logging_prefix}/critic_pairwise_accuracy",
                    critic_pairwise_accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    f"{logging_prefix}/critic_score_error",
                    critic_score_error,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
                for index, name in enumerate(("nc", "dac", "tlc", "ddc")):
                    self.log(
                        f"{logging_prefix}/critic_{name}_delta",
                        critic_hard_deltas[index],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
            
            collision=trajectoy_scores[:,0].mean()
            drivable_area_compliance=trajectoy_scores[:,1].mean()
            tlc = trajectoy_scores[:, 2].mean()
            ddc = trajectoy_scores[:, 3].mean()
            ego_progress=trajectoy_scores[:, 4].mean()
            time_to_collision_within_bound=trajectoy_scores[:, 5].mean()
            lk = trajectoy_scores[:, 6].mean()
            comfort=trajectoy_scores[:, 7].mean()
            
            self.log(f"{logging_prefix}/collision", collision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/dac", drivable_area_compliance, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/tlc", tlc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/ddc", ddc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/progress", ego_progress, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/ttc", time_to_collision_within_bound, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/lk", lk, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/comfort", comfort, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            if actor_validation:
                for index, name in enumerate(("nc", "dac", "tlc", "ddc")):
                    self.log(
                        f"{logging_prefix}/actor_fixed_{name}_delta",
                        actor_hard_deltas[index],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
                    self.log(
                        f"{logging_prefix}/e2e_{name}_delta",
                        e2e_hard_deltas[index],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )

            return final_score
        else:
            return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        """Persist ActWorld stage lineage for safe optimizer-state resumes."""

        stage = getattr(self.agent, "_training_stage", None)
        if stage in {"critic", "actor"}:
            checkpoint["actworld_training_stage"] = stage
            checkpoint["actworld_critic_variant"] = getattr(
                self.agent, "_critic_variant", "legacy"
            )

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        """Reject cross-stage full resumes; Stage-A transfer is selective."""

        current_stage = getattr(self.agent, "_training_stage", None)
        saved_stage = checkpoint.get("actworld_training_stage")
        current_variant = getattr(self.agent, "_critic_variant", "legacy")
        saved_variant = checkpoint.get("actworld_critic_variant", "legacy")
        critic_stage_is_compatible = (
            saved_stage == "critic"
            if current_variant == "f2_base_residual"
            else saved_stage in {None, "critic"}
        )
        incompatible = (
            current_stage == "actor" and saved_stage != "actor"
        ) or (
            current_stage == "critic"
            and not critic_stage_is_compatible
        )
        if incompatible:
            raise RuntimeError(
                "ActWorld full-checkpoint stage mismatch: "
                f"checkpoint={saved_stage!r}, configured={current_stage!r}. "
                "Stage-A weights must be loaded through the selective parent "
                "checkpoint path, not as a full actor resume."
            )
        if saved_variant != current_variant:
            raise RuntimeError(
                "ActWorld full-checkpoint critic variant mismatch: "
                f"checkpoint={saved_variant!r}, configured={current_variant!r}."
            )
