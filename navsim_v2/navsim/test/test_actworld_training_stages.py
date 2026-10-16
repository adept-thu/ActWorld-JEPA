"""Regression tests for ActWorld stage lineage and validation semantics."""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from navsim.agents.actworld_jepa.actworld_agent import ActWorldJEPAAgent
from navsim.agents.actworld_jepa.actworld_config import ActWorldJEPAConfig
from navsim.agents.actworld_jepa.actworld_model import ActWorldJEPAModel
from navsim.agents.actworld_jepa.actworld_planner import (
    ActWorldPlanner,
    ActWorldPlannerConfig,
)
from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
from navsim.agents.drive_jepa_perception_based.bevformer.multi_scale_deformable_attn_function import (
    _resolve_compatible_im2col_step,
)
from navsim.planning.script.run_training import (
    F2_NONFORMAL_DIRECT_TRAIN_ENV,
    F2_NONFORMAL_RESUME_BATCH96_ENV,
    F2_NONFORMAL_RESUME_BATCH288_ENV,
    build_checkpoint_callback,
    resolve_multiprocessing_start_method,
    validate_f2_run_protocol,
    validate_training_lineage,
)
from navsim.planning.training.agent_lightning_module import AgentLightningModule


def _cpu_autocast_enabled() -> bool:
    try:
        return bool(torch.is_autocast_enabled("cpu"))
    except TypeError:  # torch 2.1
        return bool(torch.is_autocast_cpu_enabled())


class _ValidationAgent(torch.nn.Module):
    def __init__(self, stage: str) -> None:
        super().__init__()
        self._training_stage = stage
        self.last_scored = None

    def name(self) -> str:
        return "actworld_jepa_perception_based_agent"

    def forward(self, features):
        proposals = torch.zeros(1, 3, 4, 3)
        refined = proposals.clone()
        refined[:, :, :, 0] = torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1)
        base_score = torch.tensor([[0.9, 0.1, 0.2]])
        final_score = torch.tensor([[0.1, 0.2, 0.9]])
        return {
            "proposals": proposals,
            "refined_proposals": refined,
            "trajectory": proposals[:, 0],
            "base_pdm_score": base_score,
            "pdm_score": final_score if self._training_stage == "actor" else base_score,
            "actworld_utility_logits": torch.tensor([[0.0, 2.0, 1.0]]),
        }

    def compute_score(self, targets, proposals, test=True, label_only=False):
        self.last_scored = proposals.detach().clone()
        if not label_only:
            raise AssertionError("ActWorld staged validation must use label_only scoring")
        if self._training_stage == "critic":
            final = torch.tensor([[0.3, 0.8, 0.4]])
        else:
            final = torch.tensor([[0.4, 0.5, 0.45, 0.7]])
        factors = final[..., None].expand(*final.shape, 9).clone()
        return final, final.amax(dim=1), factors


class _RecordingStageAPlanner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loaded_state = None

    def load_stage_a_critic_state(self, state) -> None:
        self.loaded_state = dict(state)


class _StageALoaderPadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._actworld_planner = _RecordingStageAPlanner()


class _F2FuturePadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.future_loss = torch.nn.Parameter(torch.tensor(4.0))

    def future_latent_grounding_loss(self, *args) -> torch.Tensor:
        return self.future_loss


class _TinyTrajectoryHead(torch.nn.Module):
    def __init__(self, candidate_count: int, horizon: int) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        self.horizon = horizon
        self.observations = []

    def forward(self, proposal_feature, image_feature):
        del image_feature
        self.observations.append(
            {
                "autocast": _cpu_autocast_enabled(),
                "grad": torch.is_grad_enabled(),
                "dtype": proposal_feature.dtype,
            }
        )
        proposals = proposal_feature.new_zeros(
            proposal_feature.shape[0], self.candidate_count, self.horizon, 3
        )
        return proposal_feature, proposals


class _TinyScorer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observations = []

    def forward(self, proposals, proposal_feature):
        del proposal_feature
        self.observations.append(
            {
                "autocast": _cpu_autocast_enabled(),
                "grad": torch.is_grad_enabled(),
                "dtype": proposals.dtype,
            }
        )
        batch, candidate_count = proposals.shape[:2]
        logits = proposals.new_zeros(batch, candidate_count, 6)
        logits[..., -1] = torch.linspace(
            -1.0,
            1.0,
            candidate_count,
            device=proposals.device,
            dtype=proposals.dtype,
        )
        return logits, None, None, None, None, None


class _TinyF2Model(ActWorldJEPAModel):
    """Lightweight model using the real ActWorld forward/planner contract."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        planner_config = ActWorldPlannerConfig(
            token_dim=4,
            num_candidates=32,
            horizon=2,
            latent_tokens=2,
            num_heads=2,
            num_layers=1,
            hidden_dim=8,
            dropout=0.0,
            finite_checks=True,
            training_stage="critic",
            critic_variant="f2_base_residual",
        )
        self._actworld_planner = ActWorldPlanner(planner_config)
        self._config = SimpleNamespace(actworld_freeze_drive_jepa=True)
        self.hist_encoding = torch.nn.Linear(3, 4, bias=False)
        self.init_feature = torch.nn.Embedding(32, 4)
        self._trajectory_head = torch.nn.ModuleList(
            [_TinyTrajectoryHead(candidate_count=32, horizon=2)]
        )
        self.scorer = _TinyScorer()
        self.historical_loss_calls = 0
        self.future_loss_calls = 0
        self.current_encode_observations = []

    def _encode_current_clip(self, features):
        self.current_encode_observations.append(
            {
                "autocast": _cpu_autocast_enabled(),
                "grad": torch.is_grad_enabled(),
                "ego_dtype": features["ego_status"].dtype,
            }
        )
        batch = features["ego_status"].shape[0]
        tokens = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        tokens = tokens.expand(batch, -1, -1).contiguous()
        return (tokens,), tokens

    def _historical_latent_loss(self, features, current_tokens, ego_status):
        del features, ego_status
        self.historical_loss_calls += 1
        return current_tokens.sum() * 0.0 + current_tokens.new_tensor(0.75)

    def future_latent_grounding_loss(
        self,
        current_scene_latent,
        expert_trajectory,
        future_clip,
        valid,
    ):
        del expert_trajectory, future_clip, valid
        self.future_loss_calls += 1
        return (
            current_scene_latent.sum() * 0.0
            + current_scene_latent.new_tensor(0.5)
        )


class _TinyF2DiagnosticAgent(ActWorldJEPAAgent):
    """Avoids the heavy Drive-JEPA backbone while retaining real F2 loss code."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self._config = ActWorldJEPAConfig(
            actworld_critic_variant="f2_base_residual",
            actworld_future_latent_grounding=True,
        )
        self._training_stage = "critic"
        self._critic_variant = "f2_base_residual"
        self._future_grounding_enabled = True
        self._pad_model = _TinyF2Model()
        self.score_call = None


    def compute_score(self, targets, proposals, test=True, label_only=False):
        del targets
        self.score_call = {
            "test": test,
            "label_only": label_only,
            "proposal_shape": tuple(proposals.shape),
        }
        if not test or not label_only:
            raise AssertionError("F2 label-only scoring must use test=True")
        labels = proposals.new_zeros(proposals.shape[0], 32, 9)
        labels[..., :4] = 1.0
        labels[..., 8] = torch.linspace(
            0.1,
            0.9,
            32,
            device=proposals.device,
            dtype=proposals.dtype,
        )
        final = labels[..., 8]
        return final, final.amax(dim=1), labels


class _SelectiveStage2Model(ActWorldJEPAModel):
    """Small module graph for exact Stage-2 trainability assertions."""

    def __init__(self, *, selective: bool, frozen: bool, top_blocks: int = 2):
        torch.nn.Module.__init__(self)
        self._config = SimpleNamespace(
            actworld_freeze_drive_jepa=frozen,
            actworld_selective_unfreeze_planning_top=selective,
            actworld_backbone_top_blocks=top_blocks,
        )
        image_backbone = torch.nn.Module()
        image_backbone.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2) for _ in range(4)]
        )
        image_backbone.norm = torch.nn.LayerNorm(2)
        self._backbone = torch.nn.Module()
        self._backbone.img_backbone = image_backbone
        self._backbone.projector = torch.nn.Linear(2, 2)
        self._backbone.stem = torch.nn.Linear(2, 2)
        self.hist_encoding = torch.nn.Linear(2, 2)
        self.init_feature = torch.nn.Embedding(2, 2)
        self._trajectory_head = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.scorer = torch.nn.Linear(2, 2)
        self._actworld_planner = torch.nn.Linear(2, 2)
        self.configure_drive_jepa_trainability()


class _BaselinePlanner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(1, 2),
            torch.nn.GELU(),
            torch.nn.Linear(2, 1),
        )
        torch.nn.init.zeros_(self.value_head[-1].weight)
        torch.nn.init.zeros_(self.value_head[-1].bias)


class _BaselinePadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(1, 1)
        self._actworld_planner = _BaselinePlanner()


class _RecordingBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(4))
        self.observations = []

    def forward(self, clip, img_metas):
        del img_metas
        self.observations.append(
            {
                "autocast": _cpu_autocast_enabled(),
                "grad": torch.is_grad_enabled(),
                "dtype": clip.dtype,
            }
        )
        memory = self.anchor.reshape(1, 1, 1, 4).expand(
            1, 3, clip.shape[0], 4
        )
        return (memory,)


class ActWorldTrainingStageTest(unittest.TestCase):
    def test_legacy_critic_retains_drive_jepa_planning_loss_gradient(self) -> None:
        config = ActWorldJEPAConfig(actworld_future_latent_grounding=False)
        agent = object.__new__(ActWorldJEPAAgent)
        torch.nn.Module.__init__(agent)
        agent._training_stage = "critic"
        agent._critic_variant = "legacy"
        agent._future_grounding_enabled = False
        agent._pad_model = torch.nn.Module()

        base_loss = torch.tensor(3.0, requires_grad=True)
        metric = torch.tensor(1.0, requires_grad=True)
        value = torch.tensor(2.0, requires_grad=True)
        collision = torch.tensor(4.0, requires_grad=True)
        rank = torch.tensor(5.0, requires_grad=True)

        def supervision(*args, **kwargs):
            del args, kwargs
            return {
                "metric": metric,
                "value": value,
                "collision": collision,
                "fusion": metric.new_zeros(()),
                "rank": rank,
            }

        agent._world_supervision = supervision
        pred = {
            "_target_scores": torch.zeros(1, 2, 9),
            "actworld_value_logits": torch.zeros(1, 2),
            "actworld_initial_world_logits": torch.zeros(1, 2, 9),
            "actworld_initial_value_logits": torch.zeros(1, 2),
            "actworld_initial_collision_logits": torch.zeros(1, 2),
            "actworld_initial_composite_logits": torch.zeros(1, 2),
            "refined_proposals": torch.zeros(1, 2, 4, 3),
            "actworld_latent_consistency_loss": torch.tensor(0.5, requires_grad=True),
            "actworld_trajectory_residual": torch.zeros(1, 2, 4, 3),
            "actworld_refinement_smoothness_loss": torch.tensor(0.0),
            "actworld_refinement_gate": torch.zeros(3),
            "actworld_score_delta_gate": torch.tensor(0.0),
        }
        targets = {"trajectory": torch.zeros(1, 4, 3)}

        with patch(
            "navsim.agents.actworld_jepa.actworld_agent.DriveJEPAAgent.pad_loss",
            return_value={"loss": base_loss},
        ):
            result = agent.pad_loss(targets, pred, config)

        expected = (
            base_loss
            + metric
            + value
            + collision
            + config.actworld_rank_loss_weight * rank
            + config.actworld_latent_loss_weight
            * pred["actworld_latent_consistency_loss"]
        )
        torch.testing.assert_close(result["loss"], expected)
        torch.testing.assert_close(result["drive_jepa_planning_loss"], base_loss)
        result["loss"].backward()
        torch.testing.assert_close(base_loss.grad, torch.tensor(1.0))

    def test_selective_stage2_unfreezes_only_decoder_and_top_backbone(self) -> None:
        model = _SelectiveStage2Model(selective=True, frozen=False, top_blocks=2)
        blocks = model._backbone.img_backbone.blocks

        self.assertTrue(
            all(parameter.requires_grad for parameter in model._actworld_planner.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model._trajectory_head.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model._backbone.projector.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in blocks[-1].parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in blocks[-2].parameters())
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in blocks[0].parameters())
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in blocks[1].parameters())
        )
        for module in (model._backbone.stem, model.hist_encoding, model.scorer):
            self.assertTrue(
                all(not parameter.requires_grad for parameter in module.parameters())
            )

        model.train(True)
        self.assertFalse(model._backbone.training)
        self.assertFalse(blocks[0].training)
        self.assertTrue(blocks[-1].training)
        self.assertTrue(model._backbone.projector.training)
        self.assertTrue(model._trajectory_head.training)
        self.assertTrue(model._actworld_planner.training)

    def test_selective_stage2_requires_model_only_stage1_parent(self) -> None:
        config = ActWorldJEPAConfig(
            actworld_freeze_drive_jepa=False,
            actworld_selective_unfreeze_planning_top=True,
            actworld_backbone_top_blocks=2,
        )
        with self.assertRaisesRegex(ValueError, "stage1_checkpoint_path"):
            ActWorldJEPAAgent._validate_initial_configuration(
                config, checkpoint_path="", lr=1e-5
            )
        with tempfile.NamedTemporaryFile() as stage1:
            config.actworld_stage1_checkpoint_path = stage1.name
            self.assertEqual(
                ActWorldJEPAAgent._validate_initial_configuration(
                    config, checkpoint_path="", lr=1e-5
                ),
                ("critic", "legacy"),
            )
            config.baseline_checkpoint_path = "official.ckpt"
            with self.assertRaisesRegex(ValueError, "instead of baseline"):
                ActWorldJEPAAgent._validate_initial_configuration(
                    config, checkpoint_path="", lr=1e-5
                )

    def test_selective_stage2_optimizer_contains_only_selected_base_groups(self) -> None:
        model = _SelectiveStage2Model(selective=True, frozen=False, top_blocks=2)
        agent = SimpleNamespace(
            _pad_model=model,
            _lr=1e-5,
            _config=SimpleNamespace(
                actworld_freeze_drive_jepa=False,
                actworld_backbone_lr_scale=1.0,
                actworld_base_lr_scale=1.0,
            ),
        )
        optimizer = ActWorldJEPAAgent.get_optimizers(agent)
        grouped_ids = [
            {id(parameter) for parameter in group["params"]}
            for group in optimizer.param_groups
        ]
        self.assertEqual(len(grouped_ids), 3)
        self.assertEqual(
            grouped_ids[0],
            {
                id(parameter)
                for parameter in model._actworld_planner.parameters()
                if parameter.requires_grad
            },
        )
        self.assertEqual(
            grouped_ids[1],
            {
                id(parameter)
                for parameter in model._backbone.parameters()
                if parameter.requires_grad
            },
        )
        self.assertEqual(
            grouped_ids[2],
            {
                id(parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and not name.startswith("_actworld_planner.")
                and not name.startswith("_backbone.")
            },
        )

    def test_dataloader_start_method_is_explicit_and_validated(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_multiprocessing_start_method(), "spawn")
        with patch.dict(os.environ, {"ACTWORLD_MP_START_METHOD": "fork"}):
            self.assertEqual(resolve_multiprocessing_start_method(), "fork")
        with patch.dict(os.environ, {"ACTWORLD_MP_START_METHOD": "invalid"}):
            with self.assertRaisesRegex(ValueError, "must be spawn"):
                resolve_multiprocessing_start_method()

    def test_im2col_step_is_compatible_for_every_batch(self) -> None:
        self.assertEqual(_resolve_compatible_im2col_step(96, 64), 32)
        self.assertEqual(_resolve_compatible_im2col_step(128, 64), 64)
        self.assertEqual(_resolve_compatible_im2col_step(288, 64), 32)
        self.assertEqual(_resolve_compatible_im2col_step(8, 64), 8)

        with self.assertRaisesRegex(ValueError, "must be positive"):
            _resolve_compatible_im2col_step(0, 64)

    @staticmethod
    def _logged_values(log_mock):
        return {call.args[0]: call.args[1] for call in log_mock.call_args_list}

    def test_critic_validation_separates_deployed_and_critic_scores(self) -> None:
        agent = _ValidationAgent("critic")
        module = AgentLightningModule(agent)
        targets = {"trajectory": torch.zeros(1, 4, 3)}

        with patch.object(module, "log") as log_mock:
            deployed_score = module.validation_step(({}, targets), 0)
        logged = self._logged_values(log_mock)

        self.assertEqual(agent.last_scored.shape, (1, 3, 4, 3))
        torch.testing.assert_close(deployed_score, torch.tensor(0.3))
        torch.testing.assert_close(
            logged["val/critic_selected_score"], torch.tensor(0.8)
        )
        torch.testing.assert_close(logged["val/critic_delta"], torch.tensor(0.5))
        torch.testing.assert_close(
            logged["val/critic_oracle_regret"], torch.tensor(0.0)
        )
        for name in ("nc", "dac", "tlc", "ddc"):
            torch.testing.assert_close(
                logged[f"val/critic_{name}_delta"], torch.tensor(0.5)
            )
        torch.testing.assert_close(logged["val/score"], torch.tensor(0.3))

    def test_f2_parallel_validation_is_loss_diagnostics_only(self) -> None:
        agent = _TinyF2DiagnosticAgent()
        module = AgentLightningModule(agent)
        module.eval()

        def features():
            return {
                "lidar2img": torch.zeros(1, 2, 1),
                "ego_status": torch.zeros(1, 2, 3),
                "camera_feature_3": torch.zeros(1, 3, 2, 2),
            }

        targets = {
            "trajectory": torch.zeros(1, 2, 3),
            "actworld_future_camera_clip": torch.zeros(1, 2, 3, 2, 2),
            "actworld_future_camera_valid": torch.ones(1, dtype=torch.bool),
        }

        # Ordinary F2 inference remains compact and does not compute either
        # latent diagnostic.  Only the explicit Lightning path opts in.
        with torch.no_grad():
            normal_prediction = agent.forward(features())
        self.assertIn("actworld_residual_logits", normal_prediction)
        self.assertIn("actworld_utility_logits", normal_prediction)
        self.assertNotIn("actworld_initial_world_logits", normal_prediction)
        self.assertNotIn("actworld_latent_consistency_loss", normal_prediction)
        self.assertEqual(agent._pad_model.historical_loss_calls, 0)

        with patch.object(module, "log") as log_mock, patch.object(
            agent._pad_model._actworld_planner,
            "update_target_encoder",
            side_effect=AssertionError("eval diagnostics must not update EMA"),
        ):
            with torch.no_grad():
                result = module.validation_step((features(), targets), 0)

        self.assertTrue(torch.isfinite(result))
        self.assertEqual(agent._pad_model.historical_loss_calls, 1)
        self.assertEqual(agent._pad_model.future_loss_calls, 1)
        self.assertEqual(
            agent.score_call,
            {
                "test": True,
                "label_only": True,
                "proposal_shape": (1, 32, 2, 3),
            },
        )

        logged_names = {call.args[0] for call in log_mock.call_args_list}
        self.assertIn("val/f2_parallel_non_authoritative_loss", logged_names)
        self.assertIn(
            "val/f2_parallel_non_authoritative_actworld_f2_aggregate_bce_loss",
            logged_names,
        )
        self.assertTrue(
            all(
                name.startswith("val/f2_parallel_non_authoritative_")
                for name in logged_names
            )
        )
        self.assertFalse(
            any(
                "best_score" in name
                or name.endswith("/score")
                or "actworld_gate_" in name
                for name in logged_names
            )
        )

        checkpoint = {}
        module.on_save_checkpoint(checkpoint)
        self.assertEqual(checkpoint["actworld_training_stage"], "critic")
        self.assertEqual(
            checkpoint["actworld_critic_variant"], "f2_base_residual"
        )
        module.on_load_checkpoint(checkpoint)
        with self.assertRaisesRegex(RuntimeError, "stage mismatch"):
            module.on_load_checkpoint(
                {"actworld_critic_variant": "f2_base_residual"}
            )

    def test_actor_validation_decomposes_fixed_rerank_and_e2e_gain(self) -> None:
        agent = _ValidationAgent("actor")
        module = AgentLightningModule(agent)
        targets = {"trajectory": torch.zeros(1, 4, 3)}

        with patch.object(module, "log") as log_mock:
            final_score = module.validation_step(({}, targets), 0)
        logged = self._logged_values(log_mock)

        self.assertEqual(agent.last_scored.shape, (1, 4, 4, 3))
        torch.testing.assert_close(final_score, torch.tensor(0.7))
        torch.testing.assert_close(
            logged["val/actor_fixed_delta"], torch.tensor(0.1)
        )
        torch.testing.assert_close(
            logged["val/rerank_original_delta"], torch.tensor(0.05)
        )
        torch.testing.assert_close(logged["val/e2e_delta"], torch.tensor(0.3))

    def test_actor_full_resume_requires_matching_lineage_tag(self) -> None:
        agent = _ValidationAgent("actor")
        module = AgentLightningModule(agent)
        checkpoint = {}
        module.on_save_checkpoint(checkpoint)
        self.assertEqual(checkpoint["actworld_training_stage"], "actor")
        self.assertEqual(checkpoint["actworld_critic_variant"], "legacy")
        module.on_load_checkpoint(checkpoint)

        with self.assertRaisesRegex(RuntimeError, "stage mismatch"):
            module.on_load_checkpoint({"actworld_training_stage": "critic"})
        with self.assertRaisesRegex(RuntimeError, "stage mismatch"):
            module.on_load_checkpoint({})
        with self.assertRaisesRegex(RuntimeError, "variant mismatch"):
            module.on_load_checkpoint(
                {
                    "actworld_training_stage": "actor",
                    "actworld_critic_variant": "f2_base_residual",
                }
            )

    def test_f2_training_preflight_requires_future_only_for_critic(self) -> None:
        missing_future = ActWorldJEPAConfig(
            actworld_critic_variant="f2_base_residual",
            actworld_future_latent_grounding=False,
        )
        with self.assertRaisesRegex(ValueError, "future_latent_grounding=true"):
            ActWorldJEPAAgent._validate_initial_configuration(
                missing_future, checkpoint_path="", lr=5e-5
            )

        # A full-checkpoint inference instance never consumes privileged future
        # cameras and may therefore keep the public inference default disabled.
        self.assertEqual(
            ActWorldJEPAAgent._validate_initial_configuration(
                missing_future, checkpoint_path="f2.ckpt", lr=1e-4
            ),
            ("critic", "f2_base_residual"),
        )

        actor = ActWorldJEPAConfig(
            actworld_critic_variant="f2_base_residual",
            actworld_training_stage="actor",
            actworld_future_latent_grounding=False,
        )
        self.assertEqual(
            ActWorldJEPAAgent._validate_initial_configuration(
                actor, checkpoint_path="", lr=1e-4
            ),
            ("actor", "f2_base_residual"),
        )

    def test_f2_fresh_preflight_requires_baseline_file_and_exact_lr(self) -> None:
        config = ActWorldJEPAConfig(
            actworld_critic_variant="f2_base_residual",
            actworld_future_latent_grounding=True,
        )
        with self.assertRaisesRegex(TypeError, "checkpoint_path must be a string"):
            ActWorldJEPAAgent._validate_initial_configuration(
                config, checkpoint_path=None, lr=5e-5
            )
        with self.assertRaisesRegex(ValueError, "baseline_checkpoint_path"):
            ActWorldJEPAAgent._validate_initial_configuration(
                config, checkpoint_path="", lr=5e-5
            )

        config.baseline_checkpoint_path = "missing-f2-baseline.ckpt"
        with self.assertRaises(FileNotFoundError):
            ActWorldJEPAAgent._validate_initial_configuration(
                config, checkpoint_path="", lr=5e-5
            )

        with tempfile.NamedTemporaryFile() as baseline:
            config.baseline_checkpoint_path = baseline.name
            with self.assertRaisesRegex(ValueError, "lr exactly"):
                ActWorldJEPAAgent._validate_initial_configuration(
                    config, checkpoint_path="", lr=1e-4
                )
            with self.assertRaisesRegex(ValueError, "lr exactly"):
                ActWorldJEPAAgent._validate_initial_configuration(
                    config, checkpoint_path="", lr="5e-5"
                )
            self.assertEqual(
                ActWorldJEPAAgent._validate_initial_configuration(
                    config, checkpoint_path="", lr=5e-5
                ),
                ("critic", "f2_base_residual"),
            )

        legacy = ActWorldJEPAConfig()
        self.assertEqual(
            ActWorldJEPAAgent._validate_initial_configuration(
                legacy, checkpoint_path="", lr=1e-4
            ),
            ("critic", "legacy"),
        )

    def test_f2_model_deployment_forces_original_proposals(self) -> None:
        model = _TinyF2Model().eval()
        planner = model._actworld_planner
        with torch.no_grad():
            planner.refinement_head[-1].bias.fill_(1.0)
            planner.refinement_gate_logits.fill_(1.0)

        calibrated_candidates = []

        def calibrate_score(features, candidates, scores):
            del features
            calibrated_candidates.append(candidates.detach().clone())
            return scores

        features = {
            "lidar2img": torch.zeros(1, 2, 1),
            "ego_status": torch.zeros(1, 2, 3),
            "past_ego_simulated_states": torch.ones(1),
        }
        with patch.object(model, "calibrate_score", side_effect=calibrate_score):
            with torch.no_grad():
                output = model(features)

        self.assertFalse(
            torch.equal(output["actworld_refined_candidates"], output["proposals"])
        )
        self.assertTrue(torch.equal(output["refined_proposals"], output["proposals"]))
        selected = output["selected_proposal"]
        rows = torch.arange(selected.shape[0])
        self.assertTrue(
            torch.equal(output["trajectory"], output["proposals"][rows, selected])
        )
        self.assertEqual(len(calibrated_candidates), 2)
        for candidates in calibrated_candidates:
            self.assertTrue(torch.equal(candidates, output["proposals"]))

    def test_f2_safety_veto_restores_base_for_lower_hard_safety(self) -> None:
        pdm_score = torch.tensor([[0.2, 0.8, 0.3], [0.7, 0.6, 0.1]])
        base_score = torch.tensor([[0.9, 0.1, 0.2], [0.8, 0.1, 0.2]])
        factors = torch.ones((2, 3, 9))
        factors[0, 1, 1] = 0.2
        adjusted, veto = ActWorldJEPAModel._apply_f2_safety_veto(
            pdm_score, base_score, factors, margin=0.1
        )

        self.assertTrue(torch.equal(veto, torch.tensor([True, False])))
        self.assertEqual(torch.argmax(adjusted[0]).item(), 0)
        self.assertEqual(torch.argmax(adjusted[1]).item(), 0)

    def test_f2_safety_veto_keeps_safe_jepa_choice_and_validates_inputs(self) -> None:
        pdm_score = torch.tensor([[0.2, 0.8]])
        base_score = torch.tensor([[0.9, 0.1]])
        factors = torch.ones((1, 2, 9))
        adjusted, veto = ActWorldJEPAModel._apply_f2_safety_veto(
            pdm_score, base_score, factors, margin=0.0
        )
        self.assertFalse(veto.item())
        self.assertTrue(torch.equal(adjusted, pdm_score))

        with self.assertRaisesRegex(ValueError, "shape"):
            ActWorldJEPAModel._apply_f2_safety_veto(
                pdm_score, base_score, torch.ones((1, 2, 8)), margin=0.0
            )
        with self.assertRaisesRegex(ValueError, "margin"):
            ActWorldJEPAModel._apply_f2_safety_veto(
                pdm_score, base_score, factors, margin=-0.1
            )

    def test_factorized_candidate_score_matches_official_composition(self) -> None:
        factors = torch.full((1, 2, 9), 0.5)
        score = ActWorldJEPAModel._factorized_candidate_score(factors)
        expected = torch.full((1, 2), 0.5**4 * 0.5)
        self.assertTrue(torch.allclose(score, expected))

        with self.assertRaisesRegex(ValueError, "shape"):
            ActWorldJEPAModel._factorized_candidate_score(torch.zeros((1, 2, 8)))

    def test_f2_gain_selector_accepts_only_navtrain_predicted_gain(self) -> None:
        class _Predictor:
            @staticmethod
            def predict(features):
                self.assertEqual(features.shape[0], 2)
                return np.asarray([0.1, -0.1])

        artifact = {
            "model": _Predictor(),
            "policy": {
                "min_predicted_gain": 0.0,
                "max_baseline_gap": 1.0,
                "min_hard_safety_delta": -1.0,
            },
        }
        dummy = SimpleNamespace(_load_frozen_gain_selector=lambda: artifact)
        proposals = torch.zeros((2, 2, 4, 3))
        proposals[:, 1, :, 0] = 1.0
        prediction = {
            "proposals": proposals,
            "refined_proposals": proposals,
            "pdm_score": torch.tensor([[0.2, 0.8], [0.2, 0.8]]),
            "base_pdm_score": torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
            "actworld_candidate_subscores": torch.ones((2, 2, 9)),
            "actworld_residual_logits": torch.zeros((2, 2)),
        }
        output = ActWorldJEPAAgent._apply_frozen_gain_selector(dummy, prediction)
        self.assertTrue(
            torch.equal(output["selected_proposal"], torch.tensor([1, 0]))
        )
        self.assertTrue(
            torch.equal(
                output["actworld_gain_selector_accepted"],
                torch.tensor([True, False]),
            )
        )
        self.assertTrue(torch.equal(output["trajectory"][0], proposals[0, 1]))
        self.assertTrue(torch.equal(output["trajectory"][1], proposals[1, 0]))

    def test_strict_variant_selector_applies_navtrain_selected_transform(self) -> None:
        class _Predictor:
            @staticmethod
            def predict(features):
                self.assertEqual(features.shape, (9, 77))
                values = np.zeros(9, dtype=np.float64)
                values[7] = 0.2  # progress102
                return values

        artifact = {
            "model": _Predictor(),
            "feature_count": 77,
            "policy": {
                "allowed_variants": ("progress102",),
                "threshold": 0.1,
                "hard_guard": -1.0,
            },
        }
        dummy = SimpleNamespace(
            _load_frozen_variant_selector=lambda: artifact,
            _variant_trajectory=ActWorldJEPAAgent._variant_trajectory,
        )
        proposals = torch.zeros((1, 2, 4, 3))
        proposals[:, 1, :, 0] = 1.0
        prediction = {
            "proposals": proposals,
            "refined_proposals": proposals,
            "trajectory": proposals[:, 1].clone(),
            "pdm_score": torch.tensor([[0.2, 0.8]]),
            "base_pdm_score": torch.tensor([[0.9, 0.1]]),
            "actworld_candidate_subscores": torch.ones((1, 2, 9)),
            "actworld_residual_logits": torch.zeros((1, 2)),
        }
        output = ActWorldJEPAAgent._apply_frozen_variant_selector(dummy, prediction)
        self.assertEqual(output["actworld_strict_variant"].item(), 7)
        self.assertTrue(
            torch.allclose(
                output["trajectory"][0, :, 0], torch.full((4,), 1.02)
            )
        )

    def test_f2_actor_scores_refined_proposals_with_navtrain_cache(self) -> None:
        agent = object.__new__(ActWorldJEPAAgent)
        agent._critic_variant = ActWorldJEPAAgent.F2_CRITIC_VARIANT
        agent._training_stage = "actor"
        data_points = [object()]
        with (
            patch.object(
                DriveJEPAAgent,
                "worker_map",
                return_value="navtrain-score",
            ) as navtrain_worker,
            patch.object(
                ActWorldJEPAAgent,
                "_worker_map_with_recovery",
                return_value="sealed-score",
            ) as sealed_worker,
        ):
            result = ActWorldJEPAAgent.worker_map(agent, data_points)
        self.assertEqual(result, "navtrain-score")
        navtrain_worker.assert_called_once_with(data_points)
        sealed_worker.assert_not_called()

    def test_f2_frozen_teacher_is_f32_no_grad_but_critic_keeps_amp(self) -> None:
        model = _TinyF2Model().eval()
        planner_observations = []
        history_observations = []

        def planner_hook(module, args):
            del module
            planner_observations.append(
                {
                    "autocast": _cpu_autocast_enabled(),
                    "grad": torch.is_grad_enabled(),
                    "dtypes": tuple(value.dtype for value in args[:3]),
                }
            )

        def history_hook(module, args):
            del module
            history_observations.append(
                {
                    "autocast": _cpu_autocast_enabled(),
                    "grad": torch.is_grad_enabled(),
                    "dtype": args[0].dtype,
                }
            )

        planner_handle = model._actworld_planner.register_forward_pre_hook(
            planner_hook
        )
        history_handle = model.hist_encoding.register_forward_pre_hook(
            history_hook
        )
        try:
            features = {
                "lidar2img": torch.zeros(1, 2, 1, dtype=torch.bfloat16),
                "ego_status": torch.zeros(1, 2, 3, dtype=torch.bfloat16),
            }
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = model(features)
                output["actworld_utility_logits"].sum().backward()
        finally:
            planner_handle.remove()
            history_handle.remove()

        for observation in (
            model.current_encode_observations[-1],
            history_observations[-1],
            model._trajectory_head[0].observations[-1],
            model.scorer.observations[-1],
        ):
            self.assertFalse(observation["autocast"])
            self.assertFalse(observation["grad"])
            dtype = observation.get("dtype", observation.get("ego_dtype"))
            self.assertEqual(dtype, torch.float32)
        self.assertTrue(planner_observations[-1]["autocast"])
        self.assertTrue(planner_observations[-1]["grad"])
        self.assertEqual(
            planner_observations[-1]["dtypes"],
            (torch.float32, torch.float32, torch.float32),
        )
        self.assertEqual(output["proposals"].dtype, torch.float32)
        self.assertEqual(output["pred_logit"].dtype, torch.float32)
        self.assertEqual(
            output["actworld_base_candidate_logits"].dtype, torch.float32
        )
        self.assertIsNone(model.hist_encoding.weight.grad)
        self.assertIsNone(model.init_feature.weight.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model._actworld_planner.parameters()
                if parameter.requires_grad
            )
        )

    def test_f2_history_and_future_encoders_are_f32_no_grad(self) -> None:
        model = _TinyF2Model().eval()
        backbone = _RecordingBackbone()
        model._backbone = backbone
        model.transform = lambda tensor: tensor
        features = {
            "camera_feature_2": torch.zeros(
                1, 3, 2, 2, dtype=torch.bfloat16
            ),
            "camera_feature_3": torch.ones(
                1, 3, 2, 2, dtype=torch.bfloat16
            ),
        }
        current_tokens = torch.ones(1, 3, 4)
        ego_status = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            historical_loss = ActWorldJEPAModel._historical_latent_loss(
                model, features, current_tokens, ego_status
            )
            future_tokens = ActWorldJEPAModel._encode_future_clip(
                model,
                torch.zeros(1, 2, 3, 2, 2, dtype=torch.bfloat16),
            )

        self.assertTrue(historical_loss.requires_grad)
        self.assertEqual(future_tokens.dtype, torch.float32)
        self.assertFalse(future_tokens.requires_grad)
        self.assertEqual(len(backbone.observations), 2)
        for observation in backbone.observations:
            self.assertFalse(observation["autocast"])
            self.assertFalse(observation["grad"])
            self.assertEqual(observation["dtype"], torch.float32)
        self.assertIsNone(backbone.anchor.grad)

    def test_actor_periodic_checkpoints_are_unmonitored_and_named_by_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"ACTWORLD_ACTOR_CHECKPOINT_STEPS": "500"}):
                callback = build_checkpoint_callback(temp_dir, "actor", "f2_base_residual")
                self.assertIsNone(callback.monitor)
                self.assertEqual(callback.save_top_k, -1)
                self.assertEqual(callback._every_n_train_steps, 500)
                self.assertTrue(callback.format_checkpoint_name({"step": 500}).endswith("step_500.ckpt"))
            with patch.dict(os.environ, {"ACTWORLD_ACTOR_CHECKPOINT_STEPS": "-1"}):
                with self.assertRaises(ValueError):
                    build_checkpoint_callback(temp_dir, "actor", "f2_base_residual")

    def test_f2_checkpoint_policy_saves_every_epoch_without_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = build_checkpoint_callback(
                temp_dir, "critic", "f2_base_residual"
            )
            self.assertIsNone(callback.monitor)
            self.assertEqual(callback.save_top_k, -1)
            self.assertTrue(callback.save_last)
            self.assertEqual(callback.filename, "epoch_{epoch}")
            self.assertFalse(callback.auto_insert_metric_name)
            self.assertEqual(callback._every_n_epochs, 1)
            self.assertTrue(
                callback.format_checkpoint_name({"epoch": 2}).endswith(
                    "epoch_2.ckpt"
                )
            )

            legacy = build_checkpoint_callback(temp_dir, "critic", "legacy")
            self.assertEqual(
                legacy.monitor, "val/critic_selected_score_epoch"
            )
            self.assertEqual(legacy.save_top_k, 1)
            self.assertEqual(legacy.filename, "best_{epoch}")

    def test_f2_run_protocol_requires_frozen_four_epoch_fresh_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = Path(temp_dir) / "drive_jepa_stage_a.ckpt"
            baseline.touch()
            base = {
                "seed": 0,
                "use_cache_without_dataset": False,
                "resume_checkpoint_path": None,
                "agent": {
                    "checkpoint_path": "",
                    "lr": 5e-5,
                    "config": {
                        "actworld_critic_variant": "f2_base_residual",
                        "actworld_training_stage": "critic",
                        "actworld_future_latent_grounding": True,
                        "actworld_freeze_drive_jepa": True,
                        "baseline_checkpoint_path": str(baseline),
                    },
                },
                "trainer": {"params": {"max_epochs": 4}},
            }

            def config_with(path, value):
                cfg = OmegaConf.create(base)
                OmegaConf.update(cfg, path, value, merge=False)
                return cfg

            def validate(config):
                # Qualification artifact parsing has dedicated synthetic tests;
                # this test isolates the remaining run-protocol predicates.
                with patch(
                    "navsim.planning.script.run_training."
                    "load_configured_f2_qualification_manifest"
                ), patch(
                    "navsim.planning.script.run_training."
                    "authenticate_configured_f2_cache_metadata"
                ), patch(
                    "navsim.planning.script.run_training."
                    "authenticate_configured_f2_raw_inputs"
                ):
                    return validate_f2_run_protocol(
                        config, runtime_admission_verifier=lambda _: None
                    )

            validate(OmegaConf.create(base))
            invalid_cases = (
                ("seed", 1, "seed=0"),
                ("seed", False, "seed=0"),
                ("agent.lr", 1e-4, "agent.lr=5e-5"),
                ("agent.lr", "5e-5", "agent.lr=5e-5"),
                ("trainer.params.max_epochs", 5, "max_epochs=4"),
                (
                    "agent.config.actworld_future_latent_grounding",
                    1,
                    "future latent grounding",
                ),
                (
                    "agent.config.actworld_freeze_drive_jepa",
                    1,
                    "frozen Drive-JEPA",
                ),
                (
                    "use_cache_without_dataset",
                    True,
                    "forbids use_cache_without_dataset",
                ),
                (
                    "use_cache_without_dataset",
                    0,
                    "must be a bool",
                ),
                (
                    "agent.checkpoint_path",
                    "f2-full.ckpt",
                    "empty agent.checkpoint_path",
                ),
                (
                    "agent.checkpoint_path",
                    None,
                    "empty agent.checkpoint_path",
                ),
                (
                    "resume_checkpoint_path",
                    "epoch_1.ckpt",
                    "fresh, non-resumed",
                ),
            )
            for path, value, pattern in invalid_cases:
                with self.subTest(path=path, value=value):
                    with self.assertRaisesRegex(ValueError, pattern):
                        validate(config_with(path, value))

            missing_baseline = config_with(
                "agent.config.baseline_checkpoint_path",
                str(Path(temp_dir) / "missing.ckpt"),
            )
            with self.assertRaises(FileNotFoundError):
                validate(missing_baseline)

            # The protocol guard is an F2-only seam; legacy defaults retain
            # their existing seed/lr/epoch/cache behavior.
            legacy = config_with(
                "agent.config.actworld_critic_variant", "legacy"
            )
            legacy.seed = False
            legacy.agent.lr = "arbitrary-legacy-value"
            legacy.trainer.params.max_epochs = 100
            legacy.use_cache_without_dataset = True
            validate(legacy)

    def test_f2_nonformal_direct_mode_bypasses_only_runtime_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = Path(temp_dir) / "drive_jepa.ckpt"
            baseline.touch()
            cfg = OmegaConf.create(
                {
                    "seed": 0,
                    "use_cache_without_dataset": False,
                    "resume_checkpoint_path": None,
                    "agent": {
                        "checkpoint_path": "",
                        "lr": 5e-5,
                        "config": {
                            "actworld_critic_variant": "f2_base_residual",
                            "actworld_training_stage": "critic",
                            "actworld_future_latent_grounding": True,
                            "actworld_freeze_drive_jepa": True,
                            "baseline_checkpoint_path": str(baseline),
                        },
                    },
                    "trainer": {"params": {"max_epochs": 4}},
                }
            )
            with patch.dict(
                os.environ, {F2_NONFORMAL_DIRECT_TRAIN_ENV: "1"}
            ), patch(
                "navsim.planning.script.run_training."
                "verify_configured_f2_runtime_admission"
            ) as runtime_admission, patch(
                "navsim.planning.script.run_training."
                "_validate_frozen_f2_hydra_protocol"
            ), patch(
                "navsim.planning.script.run_training."
                "load_configured_f2_qualification_manifest"
            ), patch(
                "navsim.planning.script.run_training."
                "authenticate_configured_f2_cache_metadata"
            ), patch(
                "navsim.planning.script.run_training."
                "authenticate_configured_f2_raw_inputs"
            ):
                validate_f2_run_protocol(cfg)
            runtime_admission.assert_not_called()

    def test_f2_nonformal_batch96_mode_allows_only_resume_and_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = Path(temp_dir) / "drive_jepa.ckpt"
            baseline.touch()
            resume = Path(temp_dir) / "last.ckpt"
            resume.touch()
            cfg = OmegaConf.create(
                {
                    "seed": 0,
                    "use_cache_without_dataset": False,
                    "resume_checkpoint_path": str(resume),
                    "dataloader": {
                        "params": {
                            "batch_size": 96,
                            "num_workers": 0,
                            "prefetch_factor": None,
                        }
                    },
                    "agent": {
                        "checkpoint_path": "",
                        "lr": 5e-5,
                        "config": {
                            "actworld_critic_variant": "f2_base_residual",
                            "actworld_training_stage": "critic",
                            "actworld_future_latent_grounding": True,
                            "actworld_freeze_drive_jepa": True,
                            "baseline_checkpoint_path": str(baseline),
                        },
                    },
                    "trainer": {"params": {"max_epochs": 4}},
                }
            )
            environment = {
                F2_NONFORMAL_DIRECT_TRAIN_ENV: "1",
                F2_NONFORMAL_RESUME_BATCH96_ENV: "1",
            }
            with patch.dict(os.environ, environment), patch(
                "navsim.planning.script.run_training."
                "verify_configured_f2_runtime_admission"
            ) as runtime_admission, patch(
                "navsim.planning.script.run_training."
                "_validate_frozen_f2_hydra_protocol"
            ) as frozen_protocol, patch(
                "navsim.planning.script.run_training."
                "load_configured_f2_qualification_manifest"
            ), patch(
                "navsim.planning.script.run_training."
                "authenticate_configured_f2_cache_metadata"
            ), patch(
                "navsim.planning.script.run_training."
                "authenticate_configured_f2_raw_inputs"
            ):
                validate_f2_run_protocol(cfg)
            runtime_admission.assert_not_called()
            frozen_protocol.assert_called_once_with(
                cfg, allow_nonformal_resume_batch96=True
            )

            cfg.dataloader.params.batch_size = 288
            batch288_environment = {
                F2_NONFORMAL_DIRECT_TRAIN_ENV: "1",
                F2_NONFORMAL_RESUME_BATCH288_ENV: "1",
            }
            with patch.dict(os.environ, batch288_environment), patch(
                "navsim.planning.script.run_training."
                "verify_configured_f2_runtime_admission"
            ) as runtime_admission, patch(
                "navsim.planning.script.run_training."
                "_validate_frozen_f2_hydra_protocol"
            ) as frozen_protocol, patch(
                "navsim.planning.script.run_training."
                "load_configured_f2_qualification_manifest"
            ), patch(
                "navsim.planning.script.run_training."
                "authenticate_configured_f2_cache_metadata"
            ), patch(
                "navsim.planning.script.run_training."
                "authenticate_configured_f2_raw_inputs"
            ):
                validate_f2_run_protocol(cfg)
            runtime_admission.assert_not_called()
            frozen_protocol.assert_called_once_with(
                cfg, allow_nonformal_resume_batch96=True
            )

            cfg.dataloader.params.batch_size = 95
            with patch.dict(os.environ, environment), patch(
                "navsim.planning.script.run_training."
                "_validate_frozen_f2_hydra_protocol"
            ), patch(
                "navsim.planning.script.run_training."
                "load_configured_f2_qualification_manifest"
            ), patch(
                "navsim.planning.script.run_training."
                "authenticate_configured_f2_cache_metadata"
            ), patch(
                "navsim.planning.script.run_training."
                "authenticate_configured_f2_raw_inputs"
            ), self.assertRaisesRegex(ValueError, "batch_size=96"):
                validate_f2_run_protocol(cfg)

            cfg.dataloader.params.batch_size = 96
            cfg.dataloader.params.num_workers = 1
            with patch.dict(os.environ, environment), patch(
                "navsim.planning.script.run_training."
                "_validate_frozen_f2_hydra_protocol"
            ), patch(
                "navsim.planning.script.run_training."
                "load_configured_f2_qualification_manifest"
            ), patch(
                "navsim.planning.script.run_training."
                "authenticate_configured_f2_cache_metadata"
            ), patch(
                "navsim.planning.script.run_training."
                "authenticate_configured_f2_raw_inputs"
            ), self.assertRaisesRegex(ValueError, "num_workers=0"):
                validate_f2_run_protocol(cfg)

    def test_baseline_warm_start_filters_planner_pollution(self) -> None:
        agent = object.__new__(ActWorldJEPAAgent)
        torch.nn.Module.__init__(agent)
        agent._critic_variant = "f2_base_residual"
        agent._pad_model = _BaselinePadModel()

        planner = agent._pad_model._actworld_planner
        self.assertEqual(planner.value_head[-1].weight.count_nonzero().item(), 0)
        self.assertEqual(planner.value_head[-1].bias.count_nonzero().item(), 0)

        state_dict = {
            "agent._pad_model.base.weight": torch.full((1, 1), 3.0),
            "agent._pad_model.base.bias": torch.full((1,), 4.0),
            "agent._pad_model._actworld_planner.value_head.2.weight": torch.ones(
                1, 2
            ),
            "agent._pad_model._actworld_planner.value_head.2.bias": torch.ones(1),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "polluted_actworld_baseline.ckpt"
            torch.save({"state_dict": state_dict}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(ActWorldJEPAAgent, "F2_BASELINE_SHA256", digest):
                agent._load_state(str(path), baseline_only=True)

        torch.testing.assert_close(
            agent._pad_model.base.weight, torch.full((1, 1), 3.0)
        )
        torch.testing.assert_close(
            agent._pad_model.base.bias, torch.full((1,), 4.0)
        )
        self.assertEqual(planner.value_head[-1].weight.count_nonzero().item(), 0)
        self.assertEqual(planner.value_head[-1].bias.count_nonzero().item(), 0)
        agent._assert_f2_zero_initialized_residual()

    def test_f2_baseline_rejects_hash_mismatch_and_inflight_tamper(self) -> None:
        payload = {"state_dict": {"agent._pad_model.base.weight": torch.ones(1)}}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "baseline.ckpt"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                ActWorldJEPAAgent._checkpoint_payload(
                    str(path), f2_baseline=True
                )

            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            def tampering_load(handle, *, map_location):
                del handle, map_location
                current = path.stat()
                os.utime(
                    path,
                    ns=(current.st_atime_ns, current.st_mtime_ns + 10_000_000_000),
                )
                return payload

            with (
                patch.object(ActWorldJEPAAgent, "F2_BASELINE_SHA256", digest),
                patch("torch.load", side_effect=tampering_load),
                self.assertRaisesRegex(ValueError, "changed while being loaded"),
            ):
                ActWorldJEPAAgent._checkpoint_payload(
                    str(path), f2_baseline=True
                )

            link = Path(temp_dir) / "baseline-link.ckpt"
            try:
                link.symlink_to(path)
            except OSError:
                pass
            else:
                with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                    ActWorldJEPAAgent._checkpoint_payload(
                        str(link), f2_baseline=True
                    )

    def test_f2_baseline_validates_keys_shapes_and_dtypes_before_load(self) -> None:
        agent = object.__new__(ActWorldJEPAAgent)
        torch.nn.Module.__init__(agent)
        agent._critic_variant = "f2_base_residual"
        agent._pad_model = _BaselinePadModel()
        valid = {
            "agent._pad_model.base.weight": torch.ones(1, 1),
            "agent._pad_model.base.bias": torch.ones(1),
        }

        cases = {
            "missing": {
                "agent._pad_model.base.weight": torch.ones(1, 1),
            },
            "bad_shape": {
                **valid,
                "agent._pad_model.base.weight": torch.ones(2, 1),
            },
            "bad_dtype": {
                **valid,
                "agent._pad_model.base.weight": torch.ones(
                    1, 1, dtype=torch.float64
                ),
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            for expected, state_dict in cases.items():
                with self.subTest(expected=expected):
                    path = Path(temp_dir) / f"{expected}.ckpt"
                    torch.save({"state_dict": state_dict}, path)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    with (
                        patch.object(
                            ActWorldJEPAAgent, "F2_BASELINE_SHA256", digest
                        ),
                        self.assertRaisesRegex(RuntimeError, expected),
                    ):
                        agent._load_state(str(path), baseline_only=True)

    def test_f2_full_checkpoint_requires_critic_lineage_and_zero_actor(self) -> None:
        agent = _TinyF2DiagnosticAgent()
        planner = agent._pad_model._actworld_planner
        # A selected F2 checkpoint is expected to have a trained, nonzero
        # residual value terminal.  This must not be confused with actor drift.
        with torch.no_grad():
            planner.value_head[-1].weight.fill_(0.25)
            planner.value_head[-1].bias.fill_(-0.5)

        state = {
            "agent." + key: value.detach().clone()
            for key, value in agent.state_dict().items()
        }
        checkpoint = {
            "actworld_training_stage": "critic",
            "actworld_critic_variant": "f2_base_residual",
            "state_dict": state,
        }
        agent.validate_full_checkpoint_contract(checkpoint, "valid-f2.ckpt")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "valid-f2.ckpt"
            torch.save(checkpoint, path)
            agent._load_state(str(path), baseline_only=False)
        self.assertGreater(planner.value_head[-1].weight.count_nonzero().item(), 0)

        for saved_stage in (None, "actor"):
            invalid_lineage = dict(checkpoint)
            if saved_stage is None:
                invalid_lineage.pop("actworld_training_stage")
            else:
                invalid_lineage["actworld_training_stage"] = saved_stage
            with self.subTest(saved_stage=saved_stage):
                with self.assertRaisesRegex(ValueError, "must have top-level"):
                    agent.validate_full_checkpoint_contract(
                        invalid_lineage, "bad-lineage.ckpt"
                    )

        actor_suffixes = (
            "refinement_head.2.weight",
            "refinement_head.2.bias",
            "refinement_gate_logits",
            "score_delta_gate_raw",
        )
        for suffix in actor_suffixes:
            bad_state = {key: value.clone() for key, value in state.items()}
            key = next(key for key in bad_state if key.endswith(suffix))
            bad_state[key].reshape(-1)[0] = 1.0
            invalid_actor = {**checkpoint, "state_dict": bad_state}
            with self.subTest(nonzero=suffix):
                with self.assertRaisesRegex(RuntimeError, "frozen actor invariant"):
                    agent.validate_full_checkpoint_contract(
                        invalid_actor, "bad-actor.ckpt"
                    )

        missing_state = {key: value.clone() for key, value in state.items()}
        missing_key = next(
            key for key in missing_state if key.endswith("refinement_gate_logits")
        )
        missing_state.pop(missing_key)
        with self.assertRaisesRegex(RuntimeError, "missing="):
            agent.validate_full_checkpoint_contract(
                {**checkpoint, "state_dict": missing_state}, "missing-actor.ckpt"
            )

    def test_f2_recursive_state_restore_rejects_actor_drift(self) -> None:
        module = AgentLightningModule(_TinyF2DiagnosticAgent())
        state = {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }
        actor_key = next(
            key for key in state if key.endswith("refinement_head.2.bias")
        )
        state[actor_key].fill_(1.0)
        with self.assertRaisesRegex(RuntimeError, "frozen actor invariant"):
            module.load_state_dict(state, strict=True)

    def test_f2_margin_contract_covers_ties_empty_rows_and_soft_hard_labels(self) -> None:
        utility = torch.tensor(
            [
                [0.4, 0.4, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        labels = torch.zeros(3, 4, 9)

        # Row 0: aggregate-oracle and utility ties select indices 0 and 2.
        labels[0, 0, :4] = 1.0
        labels[0, 1, :4] = 1.0
        labels[0, :, 8] = torch.tensor([1.0, 1.0, 0.2, 0.1])
        # Row 1: all candidates are tied oracles and safe, so both margin
        # scenes are ineligible and contribute literal zero to the batch mean.
        labels[1, :, :4] = 1.0
        labels[1, :, 8] = 0.7
        # Row 2: 0.999 is a valid soft label but is not exactly safe.
        labels[2, 0, :4] = torch.tensor([0.999, 1.0, 1.0, 1.0])
        labels[2, :, 8] = torch.tensor([0.9, 0.2, 0.1, 0.0])

        terms = ActWorldJEPAAgent._f2_margin_terms(utility, labels)
        self.assertEqual(terms["top_positive_index"].tolist(), [0, 0, 0])
        self.assertEqual(terms["top_negative_index"].tolist(), [0, 0, 1])
        self.assertEqual(terms["safe_positive_index"].tolist(), [0, 0, 0])
        self.assertEqual(terms["unsafe_negative_index"].tolist(), [2, 0, 0])
        self.assertEqual(terms["top_eligible"].tolist(), [False, False, True])
        self.assertEqual(terms["safe_eligible"].tolist(), [True, False, False])
        torch.testing.assert_close(
            terms["top_regret_margin"], torch.tensor(0.2 / 3.0)
        )
        torch.testing.assert_close(
            terms["safe_vs_unsafe_margin"], torch.tensor(0.6 / 3.0)
        )

    def test_f2_conflicting_unsafe_oracle_has_safe_priority_gradient(self) -> None:
        utility = torch.tensor([[0.0, 0.0, 0.4]], requires_grad=True)
        labels = torch.zeros(1, 3, 9)
        labels[0, 1, :4] = 1.0
        labels[0, :, 8] = torch.tensor([1.0, 0.8, 0.1])

        terms = ActWorldJEPAAgent._f2_margin_terms(utility, labels)
        torch.testing.assert_close(terms["top_regret_margin"], torch.tensor(0.0))
        torch.testing.assert_close(
            terms["safe_vs_unsafe_margin"], torch.tensor(0.9)
        )
        (
            terms["top_regret_margin"] + terms["safe_vs_unsafe_margin"]
        ).backward()
        torch.testing.assert_close(
            utility.grad, torch.tensor([[0.0, -1.0, 1.0]])
        )

    def test_f2_loss_algebra_uses_fixed_weights_and_never_calls_legacy_loss(self) -> None:
        config = ActWorldJEPAConfig(
            actworld_critic_variant="f2_base_residual",
            actworld_future_latent_grounding=True,
        )
        agent = object.__new__(ActWorldJEPAAgent)
        torch.nn.Module.__init__(agent)
        agent._training_stage = "critic"
        agent._critic_variant = "f2_base_residual"
        agent._future_grounding_enabled = True
        agent._pad_model = _F2FuturePadModel()

        labels = torch.zeros(1, 32, 9)
        labels[0, 0] = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.9]
        )
        labels[0, 1] = torch.tensor(
            [0.0, 1.0, 1.0, 1.0, 0.2, 0.3, 0.4, 0.5, 0.1]
        )
        final_scores = labels[..., 8]

        def compute_score(*args, **kwargs):
            del args
            self.assertTrue(kwargs["test"])
            self.assertTrue(kwargs["label_only"])
            return final_scores, final_scores.amax(dim=1), labels

        agent.compute_score = compute_score
        utility_data = torch.full((1, 32), -2.0)
        utility_data[0, :2] = torch.tensor([0.2, -0.1])
        utility = utility_data.requires_grad_(True)
        world = torch.zeros(1, 32, 9, requires_grad=True)
        collision = torch.zeros(1, 32, requires_grad=True)
        latent = torch.tensor(2.0, requires_grad=True)
        pdm_score = torch.zeros(1, 32)
        pdm_score[0, 0] = 1.0
        pred = {
            "proposals": torch.zeros(1, 32, 4, 3),
            "pred_logit": torch.zeros(1, 32, 6),
            "pdm_score": pdm_score,
            "actworld_base_candidate_logits": torch.zeros(1, 32),
            "actworld_initial_world_logits": world,
            "actworld_initial_utility_logits": utility,
            "actworld_initial_collision_logits": collision,
            "actworld_latent_consistency_loss": latent,
            "actworld_scene_latent": torch.zeros(1, 2, 3),
            "actworld_initial_residual_logits": utility_data,
            "actworld_initial_delta_logits": torch.tanh(utility_data),
        }

        with patch(
            "navsim.agents.actworld_jepa.actworld_agent.DriveJEPAAgent.pad_loss",
            side_effect=AssertionError("legacy loss must not run"),
        ):
            result = agent.pad_loss(
                {
                    "trajectory": torch.zeros(1, 4, 3),
                    "actworld_future_camera_clip": torch.zeros(1, 2, 3, 2, 2),
                    "actworld_future_camera_valid": torch.ones(1, dtype=torch.bool),
                },
                pred,
                config,
            )

        terms = ActWorldJEPAAgent._f2_world_supervision(
            world,
            utility,
            collision,
            labels,
            top_margin=0.2,
            safe_unsafe_margin=0.5,
        )
        expected = (
            sum(terms.values())
            + 0.25 * latent
            + 0.25 * agent._pad_model.future_loss
        )
        torch.testing.assert_close(result["loss"], expected)
        self.assertNotIn("actworld_rank_loss", result)
        self.assertNotIn("actworld_fusion_loss", result)
        result["loss"].backward()
        self.assertIsNotNone(utility.grad)
        self.assertIsNotNone(world.grad)
        self.assertIsNotNone(collision.grad)
        self.assertIsNotNone(latent.grad)
        torch.testing.assert_close(
            agent._pad_model.future_loss.grad, torch.tensor(0.25)
        )

        bad_dtype = dict(pred)
        bad_dtype["proposals"] = pred["proposals"].bfloat16()
        with self.assertRaisesRegex(ValueError, "must be present in float32"):
            agent.pad_loss(
                {
                    "trajectory": torch.zeros(1, 4, 3),
                    "actworld_future_camera_clip": torch.zeros(1, 2, 3, 2, 2),
                    "actworld_future_camera_valid": torch.ones(
                        1, dtype=torch.bool
                    ),
                },
                bad_dtype,
                config,
            )

        config.actworld_f2_top_regret_margin_weight = 0.5
        with self.assertRaisesRegex(ValueError, "fixed"):
            ActWorldJEPAAgent._validate_f2_fixed_config(config)
        config.actworld_f2_top_regret_margin_weight = "1.0"
        with self.assertRaisesRegex(ValueError, "fixed"):
            ActWorldJEPAAgent._validate_f2_fixed_config(config)

    def test_training_lineage_preflight_rejects_ambiguous_actor_runs(self) -> None:
        base = {
            "agent": {
                "config": {
                    "actworld_training_stage": "actor",
                    "actworld_stage_a_checkpoint_path": "",
                }
            },
            "resume_checkpoint_path": None,
        }
        with self.assertRaisesRegex(ValueError, "requires either"):
            validate_training_lineage(OmegaConf.create(base))

        with tempfile.NamedTemporaryFile() as resume:
            resumed = OmegaConf.create(base)
            resumed.resume_checkpoint_path = str(Path(resume.name))
            validate_training_lineage(resumed)

            ambiguous = OmegaConf.create(base)
            ambiguous.resume_checkpoint_path = str(Path(resume.name))
            ambiguous.agent.config.actworld_stage_a_checkpoint_path = "stage_a.ckpt"
            with self.assertRaisesRegex(ValueError, "either"):
                validate_training_lineage(ambiguous)

        stage2 = OmegaConf.create(
            {
                "agent": {
                    "config": {
                        "actworld_training_stage": "critic",
                        "actworld_stage_a_checkpoint_path": "",
                        "actworld_stage1_checkpoint_path": "stage1.ckpt",
                    }
                },
                "resume_checkpoint_path": "last.ckpt",
            }
        )
        with self.assertRaisesRegex(ValueError, "model-only parent"):
            validate_training_lineage(stage2)

    @staticmethod
    def _stage_a_loader_agent() -> ActWorldJEPAAgent:
        agent = object.__new__(ActWorldJEPAAgent)
        torch.nn.Module.__init__(agent)
        agent._pad_model = _StageALoaderPadModel()
        return agent

    @staticmethod
    def _save_checkpoint(path: Path, stage, state_dict=None) -> None:
        checkpoint = {"actworld_training_stage": stage}
        if state_dict is not None:
            checkpoint["state_dict"] = state_dict
        torch.save(checkpoint, path)

    def test_selective_stage_a_load_requires_critic_lineage(self) -> None:
        agent = self._stage_a_loader_agent()
        planner_key = (
            "agent._pad_model._actworld_planner.world_feature.0.weight"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for stage, filename in (("actor", "actor.ckpt"), (None, "legacy.ckpt")):
                path = root / filename
                checkpoint = {"state_dict": {planner_key: torch.ones(1)}}
                if stage is not None:
                    checkpoint["actworld_training_stage"] = stage
                torch.save(checkpoint, path)
                with self.subTest(stage=stage):
                    with self.assertRaisesRegex(
                        ValueError, "actworld_training_stage='critic'"
                    ):
                        agent._load_stage_a_critic(str(path))

    def test_selective_stage_a_load_rejects_missing_or_incompatible_state(self) -> None:
        agent = self._stage_a_loader_agent()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError):
                agent._load_stage_a_critic(str(root / "missing.ckpt"))

            missing_state = root / "missing_state.ckpt"
            self._save_checkpoint(missing_state, "critic")
            with self.assertRaisesRegex(ValueError, "Lightning state_dict"):
                agent._load_stage_a_critic(str(missing_state))

            wrong_namespace = root / "wrong_namespace.ckpt"
            self._save_checkpoint(
                wrong_namespace,
                "critic",
                {"agent._pad_model.base.weight": torch.ones(1)},
            )
            with self.assertRaisesRegex(RuntimeError, "no ActWorld planner state"):
                agent._load_stage_a_critic(str(wrong_namespace))

    def test_selective_stage_a_load_accepts_tagged_critic(self) -> None:
        agent = self._stage_a_loader_agent()
        tensor = torch.tensor([1.25])
        planner_key = (
            "agent._pad_model._actworld_planner.world_feature.0.weight"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "critic.ckpt"
            self._save_checkpoint(path, "critic", {planner_key: tensor})
            agent._load_stage_a_critic(str(path))

        loaded = agent._pad_model._actworld_planner.loaded_state
        self.assertEqual(set(loaded), {"world_feature.0.weight"})
        torch.testing.assert_close(loaded["world_feature.0.weight"], tensor)

    def test_f2_checkpoint_variant_guard_rejects_legacy_state(self) -> None:
        agent = object.__new__(ActWorldJEPAAgent)
        torch.nn.Module.__init__(agent)
        agent._critic_variant = "f2_base_residual"
        with self.assertRaisesRegex(ValueError, "variant mismatch"):
            agent._validate_checkpoint_variant({}, "legacy.ckpt")
        agent._validate_checkpoint_variant(
            {"actworld_critic_variant": "f2_base_residual"}, "f2.ckpt"
        )


if __name__ == "__main__":
    unittest.main()
