"""Pure PyTorch tests for the candidate-conditioned ActWorld planner."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from navsim.agents.actworld_jepa.actworld_planner import (
    ActWorldPlanner,
    ActWorldPlannerConfig,
)


class _DriveLikeConfig:
    tf_d_model = 12
    proposal_num = 3
    num_poses = 4
    tf_d_ffn = 24
    tf_dropout = 0.0
    actworld_latent_tokens = 2
    actworld_num_heads = 3
    actworld_num_layers = 1
    actworld_position_scale = 40.0
    actworld_refine_max_xy = 0.7
    actworld_refine_max_heading = 0.12
    actworld_refine_gate_init = -3.5
    actworld_target_ema_decay = 0.9


class ActWorldPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = ActWorldPlannerConfig(
            token_dim=16,
            num_candidates=4,
            horizon=5,
            latent_tokens=3,
            num_heads=4,
            num_layers=1,
            hidden_dim=32,
            max_refinement_xy=0.8,
            max_refinement_yaw=0.1,
        )
        self.planner = ActWorldPlanner(self.config)
        self.batch = 2
        self.tokens = torch.randn(self.batch, 9, self.config.token_dim)
        self.candidates = torch.randn(
            self.batch,
            self.config.num_candidates,
            self.config.horizon,
            3,
        )
        self.base_logits = torch.randn(
            self.batch, self.config.num_candidates, self.config.num_subscores
        )

    @staticmethod
    def _gradient_l1(parameters) -> float:
        return float(
            sum(
                parameter.grad.detach().abs().sum().item()
                for parameter in parameters
                if parameter.grad is not None
            )
        )

    @staticmethod
    def _assert_no_gradient(test_case, parameters) -> None:
        for parameter in parameters:
            if parameter.grad is not None:
                test_case.assertEqual(parameter.grad.detach().abs().max().item(), 0.0)

    def test_actor_scale_is_shared_by_train_eval_and_preserves_gradient(self):
        self.planner.configure_training_stage("actor")
        with torch.no_grad():
            for parameter in self.planner.refinement_head.parameters():
                parameter.fill_(0.01)
        rollout = torch.randn(self.batch, self.config.num_candidates,
                              self.config.horizon, self.config.latent_tokens,
                              self.config.token_dim)
        with patch.dict("os.environ", {"ACTWORLD_ACTOR_RESIDUAL_SCALE": "1.0"}):
            full, _, _ = self.planner._refine(rollout)
        self.assertGreater(full.abs().max().item(), 0)
        for scale in (0.05, 0.2):
            with patch.dict("os.environ", {"ACTWORLD_ACTOR_RESIDUAL_SCALE": str(scale)}):
                self.planner.train()
                train, _, _ = self.planner._refine(rollout)
                self.planner.eval()
                evaluation, _, _ = self.planner._refine(rollout)
                torch.testing.assert_close(train, full * scale)
                torch.testing.assert_close(train, evaluation)
                self.planner.zero_grad()
                train.sum().backward()
                self.assertGreater(self._gradient_l1(self.planner.refinement_head.parameters()), 0)
                self.assertLessEqual(train[..., :2].abs().max().item(),
                                     self.config.max_refinement_xy * scale)
        with patch.dict("os.environ", {"ACTWORLD_ACTOR_RESIDUAL_SCALE": "1.1"}):
            with self.assertRaises(ValueError):
                self.planner._refine(rollout)

    def test_two_pass_outputs_are_shaped_finite_and_bounded(self) -> None:
        output = self.planner(
            self.tokens, self.candidates, self.base_logits, return_auxiliary=True
        )
        batch = self.batch
        candidates = self.config.num_candidates
        horizon = self.config.horizon
        dim = self.config.token_dim
        latent_tokens = self.config.latent_tokens

        self.assertEqual(output["refined_candidates"].shape, (batch, candidates, horizon, 3))
        self.assertEqual(output["world_logits"].shape, (batch, candidates, 9))
        self.assertEqual(output["candidate_subscores"].shape, (batch, candidates, 9))
        self.assertEqual(output["candidate_values"].shape, (batch, candidates))
        self.assertEqual(output["utility_logits"].shape, (batch, candidates))
        self.assertEqual(output["delta_logits"].shape, (batch, candidates))
        self.assertEqual(output["score_delta_gate"].shape, ())
        self.assertEqual(output["world_scores"].shape, (batch, candidates))
        self.assertEqual(output["collision_logits"].shape, (batch, candidates))
        self.assertEqual(output["composite_logits"].shape, (batch, candidates))
        self.assertEqual(output["initial_composite_logits"].shape, (batch, candidates))
        self.assertEqual(output["initial_utility_logits"].shape, (batch, candidates))
        self.assertEqual(output["initial_delta_logits"].shape, (batch, candidates))
        self.assertEqual(
            output["rollout_latents"].shape,
            (batch, candidates, horizon, latent_tokens, dim),
        )
        self.assertEqual(
            output["initial_rollout_latents"].shape,
            (batch, candidates, horizon, latent_tokens, dim),
        )
        self.assertEqual(output["scene_attention"].shape, (batch, latent_tokens, 9))
        self.assertEqual(
            output["evaluation_spatial_attention"].shape,
            (batch, candidates, horizon, latent_tokens),
        )
        self.assertIs(output["original_candidates"], self.candidates)
        self.assertTrue(torch.equal(output["original_candidates"], self.candidates))

        for name, value in output.items():
            if torch.is_floating_point(value):
                self.assertTrue(torch.isfinite(value).all(), name)
        self.assertTrue(((0.0 <= output["world_scores"]) & (output["world_scores"] <= 1.0)).all())
        self.assertTrue(
            ((0.0 <= output["collision_probabilities"]) & (output["collision_probabilities"] <= 1.0)).all()
        )

        residual = output["trajectory_residual"]
        gate = output["refinement_gate"]
        xy_bound = self.config.max_refinement_xy * gate[:2]
        yaw_bound = self.config.max_refinement_yaw * gate[2]
        self.assertTrue((residual[..., :2].abs() <= xy_bound.view(1, 1, 1, 2) + 1e-7).all())
        self.assertTrue((residual[..., 2].abs() <= yaw_bound + 1e-7).all())
        # Warm start is an exact Drive-JEPA fallback, not merely close to one.
        self.assertEqual(residual.abs().max().item(), 0.0)
        self.assertTrue(torch.equal(output["refined_candidates"], self.candidates + residual))
        self.assertIs(output["value_logits"], output["utility_logits"])
        torch.testing.assert_close(
            output["delta_logits"],
            output["utility_logits"] - output["base_candidate_logits"],
        )
        self.assertEqual(output["refinement_smoothness_loss"].shape, ())

    def test_second_pass_is_differentiable_and_compact_mode_keeps_core_api(self) -> None:
        tokens = self.tokens.clone().requires_grad_(True)
        candidates = self.candidates.clone().requires_grad_(True)
        output = self.planner(tokens, candidates, self.base_logits, return_auxiliary=False)
        self.assertNotIn("initial_rollout_latents", output)
        required = {
            "refined_candidates",
            "world_logits",
            "world_scores",
            "collision_logits",
            "rollout_latents",
        }
        self.assertTrue(required.issubset(output))

        loss = (
            output["world_scores"].mean()
            + output["world_logits"].square().mean()
            + output["refined_candidates"].square().mean()
        )
        loss.backward()
        self.assertIsNotNone(tokens.grad)
        self.assertIsNotNone(candidates.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue(torch.isfinite(candidates.grad).all())

    def test_base_logits_are_preserved_as_monotonic_score_evidence(self) -> None:
        self.planner.eval()
        zeros = torch.zeros(self.batch, self.config.num_candidates)
        with torch.no_grad():
            score_0 = self.planner(self.tokens, self.candidates, zeros)["world_scores"]
            score_1 = self.planner(self.tokens, self.candidates, zeros + 1.0)["world_scores"]
            score_6d = self.planner(
                self.tokens,
                self.candidates,
                torch.stack((zeros, zeros, zeros, zeros, zeros, zeros + 1.0), dim=-1),
            )["world_scores"]
        self.assertTrue((score_1 > score_0).all())
        torch.testing.assert_close(score_1, score_6d)
        torch.testing.assert_close(score_0, torch.full_like(score_0, 0.5))
        torch.testing.assert_close(score_1, torch.sigmoid(torch.ones_like(score_1)))

    def test_zero_delta_gate_is_bit_exact_drive_jepa_fallback(self) -> None:
        self.planner.eval()
        with torch.no_grad():
            output = self.planner(
                self.tokens, self.candidates, self.base_logits, return_auxiliary=True
            )

        base = self.base_logits[..., -1]
        self.assertEqual(self.planner.score_delta_gate_raw.item(), 0.0)
        self.assertEqual(output["score_delta_gate"].item(), 0.0)
        self.assertTrue(torch.equal(output["trajectory_residual"], torch.zeros_like(output["trajectory_residual"])))
        self.assertTrue(torch.equal(output["refined_candidates"], self.candidates))
        self.assertTrue(torch.equal(output["initial_composite_logits"], base))
        self.assertTrue(torch.equal(output["composite_logits"], base))
        self.assertTrue(torch.equal(output["initial_world_scores"], torch.sigmoid(base)))
        self.assertTrue(torch.equal(output["world_scores"], torch.sigmoid(base)))
        self.assertTrue(
            torch.equal(
                torch.argmax(output["world_scores"], dim=1),
                torch.argmax(base, dim=1),
            )
        )

        # Critic heads retain ordinary initialization; selection parity comes
        # from the separate zero delta gate rather than zeroing the critic.
        for terminal_head in (
            self.planner.subscore_head[-1],
            self.planner.value_head[-1],
            self.planner.collision_head[-1],
        ):
            self.assertGreater(terminal_head.weight.detach().abs().sum().item(), 0.0)

    def test_f2_zero_init_is_bit_exact_base_residual_parity(self) -> None:
        config = ActWorldPlannerConfig(
            **{
                **self.config.__dict__,
                "critic_variant": "f2_base_residual",
            }
        )
        planner = ActWorldPlanner(config).eval()
        with patch.object(
            planner,
            "_fuse_scores",
            side_effect=AssertionError("F2 must not use legacy fusion"),
        ):
            with torch.no_grad():
                output = planner(
                    self.tokens,
                    self.candidates,
                    self.base_logits,
                    return_auxiliary=True,
                )

        base = self.base_logits[..., -1]
        self.assertEqual(planner.value_head[-1].weight.abs().max().item(), 0.0)
        self.assertEqual(planner.value_head[-1].bias.abs().max().item(), 0.0)
        self.assertTrue(torch.equal(output["residual_logits"], torch.zeros_like(base)))
        self.assertTrue(
            torch.equal(output["initial_residual_logits"], torch.zeros_like(base))
        )
        self.assertTrue(torch.equal(output["delta_logits"], torch.zeros_like(base)))
        self.assertTrue(torch.equal(output["utility_logits"], base))
        self.assertTrue(
            torch.equal(
                output["utility_logits"], output["initial_utility_logits"]
            )
        )
        self.assertTrue(
            torch.equal(
                output["rollout_latents"], output["initial_rollout_latents"]
            )
        )
        self.assertTrue(torch.equal(output["composite_logits"], base))
        self.assertTrue(torch.equal(output["world_scores"], torch.sigmoid(base)))
        self.assertEqual(output["score_delta_gate"].item(), 0.0)

    def test_default_variant_is_bit_exact_explicit_legacy(self) -> None:
        torch.manual_seed(31)
        default = ActWorldPlanner(self.config).eval()
        torch.manual_seed(31)
        explicit = ActWorldPlanner(
            ActWorldPlannerConfig(
                **{
                    **self.config.__dict__,
                    "critic_variant": "legacy",
                }
            )
        ).eval()

        for key, value in default.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]), key)
        with torch.no_grad():
            default_output = default(self.tokens, self.candidates, self.base_logits)
            explicit_output = explicit(self.tokens, self.candidates, self.base_logits)
        self.assertEqual(set(default_output), set(explicit_output))
        for key in default_output:
            self.assertTrue(
                torch.equal(default_output[key], explicit_output[key]), key
            )

    def test_f2_delta_is_bounded_and_base_is_stop_gradient(self) -> None:
        base = torch.tensor([[0.25, -0.5, 1.0]], requires_grad=True)
        residual = torch.tensor([[0.0, 100.0, -100.0]], requires_grad=True)
        scores = ActWorldPlanner._f2_base_residual_scores(base, residual)

        self.assertTrue(torch.equal(scores["residual_logits"], residual))
        self.assertTrue((scores["delta_logits"].abs() <= 1.0).all())
        torch.testing.assert_close(
            scores["delta_logits"], torch.tensor([[0.0, 1.0, -1.0]])
        )
        torch.testing.assert_close(
            scores["utility_logits"], base.detach() + scores["delta_logits"]
        )

        scores["utility_logits"].sum().backward()
        self.assertIsNone(base.grad)
        self.assertIsNotNone(residual.grad)
        self.assertEqual(residual.grad[0, 0].item(), 1.0)

    def test_f2_relative_delta_uses_absolute_critic_as_base_advantage(self) -> None:
        base = torch.tensor([[0.25, -0.5, 1.0]], requires_grad=True)
        absolute = torch.tensor([[0.5, -0.25, 0.0]], requires_grad=True)
        with patch.dict("os.environ", {"ACTWORLD_F2_DELTA_MODE": "relative"}):
            scores = ActWorldPlanner._f2_base_residual_scores(base, absolute)

        expected = torch.tanh(absolute - base.detach())
        torch.testing.assert_close(scores["delta_logits"], expected)
        torch.testing.assert_close(scores["utility_logits"], base.detach() + expected)
        scores["utility_logits"].sum().backward()
        self.assertIsNone(base.grad)
        self.assertIsNotNone(absolute.grad)

    def test_f2_delta_mode_rejects_unknown_value(self) -> None:
        base = torch.zeros((1, 2))
        residual = torch.zeros((1, 2))
        with patch.dict("os.environ", {"ACTWORLD_F2_DELTA_MODE": "unknown"}):
            with self.assertRaisesRegex(ValueError, "ACTWORLD_F2_DELTA_MODE"):
                ActWorldPlanner._f2_base_residual_scores(base, residual)

    def test_f2_factorized_utility_matches_official_epdms_composition(self) -> None:
        factors = torch.tensor(
            [[[0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]]
        )
        logits = torch.logit(factors)
        utility = ActWorldPlanner._f2_factorized_utility_logits(logits)
        expected_score = (
            factors[..., :4].prod(dim=-1)
            * (
                5.0 * factors[..., 4]
                + 5.0 * factors[..., 5]
                + 2.0 * factors[..., 6]
                + 2.0 * factors[..., 7]
                + 2.0 * factors[..., 8]
            )
            / 16.0
        )
        torch.testing.assert_close(torch.sigmoid(utility), expected_score)

    def test_f2_factorized_utility_rejects_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            ActWorldPlanner._f2_factorized_utility_logits(torch.zeros((1, 2, 8)))

    def test_f2_integrated_loss_updates_heads_without_base_gradient(self) -> None:
        planner = ActWorldPlanner(
            ActWorldPlannerConfig(
                **{
                    **self.config.__dict__,
                    "critic_variant": "f2_base_residual",
                }
            )
        )
        planner.train()
        base = self.base_logits.clone().requires_grad_(True)
        output = planner(
            self.tokens,
            self.candidates,
            base,
            return_auxiliary=True,
        )
        loss = (
            F.binary_cross_entropy_with_logits(
                output["utility_logits"],
                torch.zeros_like(output["utility_logits"]),
            )
            + F.binary_cross_entropy_with_logits(
                output["world_logits"],
                torch.zeros_like(output["world_logits"]),
            )
            + F.binary_cross_entropy_with_logits(
                output["collision_logits"],
                torch.ones_like(output["collision_logits"]),
            )
        )
        loss.backward()

        self.assertIsNone(base.grad)
        for terminal in (
            planner.value_head[-1],
            planner.subscore_head[-1],
            planner.collision_head[-1],
        ):
            self.assertIsNotNone(terminal.weight.grad)
            self.assertGreater(terminal.weight.grad.abs().sum().item(), 0.0)
        self._assert_no_gradient(self, planner.actor_parameters())

    def test_f2_zeroing_does_not_change_factor_or_collision_initialization(self) -> None:
        torch.manual_seed(73)
        legacy = ActWorldPlanner(
            ActWorldPlannerConfig(
                **{
                    **self.config.__dict__,
                    "critic_variant": "legacy",
                }
            )
        )
        torch.manual_seed(73)
        f2 = ActWorldPlanner(
            ActWorldPlannerConfig(
                **{
                    **self.config.__dict__,
                    "critic_variant": "f2_base_residual",
                }
            )
        )

        for prefix in ("subscore_head.", "collision_head."):
            for key, value in legacy.state_dict().items():
                if key.startswith(prefix):
                    self.assertTrue(torch.equal(value, f2.state_dict()[key]), key)
        self.assertGreater(legacy.value_head[-1].weight.abs().sum().item(), 0.0)
        self.assertEqual(f2.value_head[-1].weight.abs().sum().item(), 0.0)

    def test_delta_fusion_is_a_single_detached_base_residual(self) -> None:
        self.planner.configure_training_stage("actor")
        base = torch.tensor([[0.2, -0.3]], requires_grad=True)
        utility = torch.tensor([[1.2, 0.7]], requires_grad=True)

        fused = self.planner._fuse_scores(base, utility)
        self.assertTrue(torch.equal(fused["composite_logits"], base))
        self.assertTrue(torch.equal(fused["world_scores"], torch.sigmoid(base)))
        self.assertTrue(torch.equal(fused["delta_logits"], utility - base.detach()))
        self.assertEqual(fused["score_delta_gate"].item(), 0.0)

        # At exact parity tanh'(0)=1, so calibration can move the score gate on
        # its very first update while the detached utility remains untouched.
        calibration_loss = F.binary_cross_entropy_with_logits(
            fused["composite_logits"], torch.zeros_like(base)
        )
        calibration_loss.backward()
        self.assertIsNotNone(self.planner.score_delta_gate_raw.grad)
        self.assertGreater(self.planner.score_delta_gate_raw.grad.abs().item(), 0.0)
        torch.testing.assert_close(utility.grad, torch.zeros_like(utility))
        torch.testing.assert_close(base.grad, torch.sigmoid(base) / base.numel())

        self.planner.zero_grad(set_to_none=True)
        base.grad = None
        utility.grad = None
        with torch.no_grad():
            self.planner.score_delta_gate_raw.copy_(
                torch.atanh(torch.tensor(0.25))
            )
        fused = self.planner._fuse_scores(base, utility)
        torch.testing.assert_close(fused["score_delta_gate"], torch.tensor(0.25))
        torch.testing.assert_close(
            fused["composite_logits"],
            base + 0.25 * (utility - base.detach()),
        )

    def test_configure_training_stage_partitions_parameters_and_modes(self) -> None:
        critic_parameters = self.planner.critic_parameters()
        actor_parameters = self.planner.actor_parameters()
        self.assertTrue(set(map(id, critic_parameters)).isdisjoint(map(id, actor_parameters)))

        self.planner.train(True)
        self.planner.configure_training_stage("critic")
        self.assertEqual(self.planner.training_stage, "critic")
        self.assertTrue(all(parameter.requires_grad for parameter in critic_parameters))
        self.assertTrue(all(not parameter.requires_grad for parameter in actor_parameters))
        self.assertTrue(self.planner.world_feature.training)
        self.assertFalse(self.planner.refinement_head.training)

        self.planner.configure_training_stage("actor")
        self.assertEqual(self.planner.training_stage, "actor")
        self.assertTrue(all(not parameter.requires_grad for parameter in critic_parameters))
        self.assertTrue(all(parameter.requires_grad for parameter in actor_parameters))
        self.assertFalse(self.planner.world_feature.training)
        self.assertTrue(self.planner.refinement_head.training)
        self.assertFalse(self.planner.target_scene_projection.training)

        with self.assertRaisesRegex(ValueError, "critic.*actor"):
            self.planner.configure_training_stage("joint")

    def test_f2_planner_supports_frozen_critic_actor_stage(self) -> None:
        planner = ActWorldPlanner(
            ActWorldPlannerConfig(
                training_stage="actor",
                critic_variant="f2_base_residual",
            )
        )
        planner.train(True)
        self.assertEqual(planner.training_stage, "actor")
        self.assertTrue(
            all(not parameter.requires_grad for parameter in planner.critic_parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in planner.actor_parameters())
        )

        planner.configure_training_stage("critic")
        self.assertEqual(planner.training_stage, "critic")
        self.assertTrue(
            all(parameter.requires_grad for parameter in planner.critic_parameters())
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in planner.actor_parameters())
        )

    def test_critic_stage_detaches_refined_evaluation_from_candidates(self) -> None:
        self.planner.configure_training_stage("critic")
        self.planner.train(True)
        candidates = self.candidates.clone().requires_grad_(True)
        output = self.planner(
            self.tokens, candidates, self.base_logits, return_auxiliary=True
        )

        candidate_gradient = torch.autograd.grad(
            output["utility_logits"].sum(),
            candidates,
            retain_graph=True,
            allow_unused=True,
        )[0]
        self.assertIsNone(candidate_gradient)

        factor_targets = torch.sigmoid(
            torch.linspace(
                -2.0,
                2.0,
                output["world_logits"].numel(),
                dtype=output["world_logits"].dtype,
            ).reshape_as(output["world_logits"])
        )
        value_targets = torch.sigmoid(
            torch.linspace(
                -1.0,
                1.0,
                output["utility_logits"].numel(),
                dtype=output["utility_logits"].dtype,
            ).reshape_as(output["utility_logits"])
        )
        critic_loss = (
            F.binary_cross_entropy_with_logits(output["world_logits"], factor_targets)
            + F.binary_cross_entropy_with_logits(output["utility_logits"], value_targets)
            + F.binary_cross_entropy_with_logits(
                output["collision_logits"], 1.0 - value_targets
            )
        )
        critic_loss.backward()
        self.assertGreater(self._gradient_l1(self.planner.critic_parameters()), 0.0)
        self._assert_no_gradient(self, self.planner.actor_parameters())

    def test_actor_stage_first_and_second_step_gradient_flow(self) -> None:
        self.planner.configure_training_stage("actor")
        self.planner.train(True)
        critic_parameters = self.planner.critic_parameters()
        optimizer = torch.optim.SGD(self.planner.actor_parameters(), lr=0.1)

        output = self.planner(
            self.tokens, self.candidates, self.base_logits, return_auxiliary=True
        )
        actor_loss = F.softplus(
            output["initial_utility_logits"].detach() - output["utility_logits"]
        ).mean()
        actor_loss = (
            actor_loss
            + 0.01 * output["trajectory_residual"].square().mean()
            + 0.01 * output["refinement_smoothness_loss"]
        )
        actor_loss.backward()

        terminal_parameters = tuple(self.planner.refinement_head[-1].parameters())
        upstream_parameters = tuple(self.planner.refinement_head[0].parameters()) + tuple(
            self.planner.refinement_spatial_attention.parameters()
        )
        self.assertGreater(self._gradient_l1(terminal_parameters), 0.0)
        self.assertEqual(self._gradient_l1(upstream_parameters), 0.0)
        self._assert_no_gradient(self, critic_parameters)
        self.assertIsNone(self.planner.score_delta_gate_raw.grad)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output = self.planner(
            self.tokens, self.candidates, self.base_logits, return_auxiliary=True
        )
        second_actor_loss = F.softplus(
            output["initial_utility_logits"].detach() - output["utility_logits"]
        ).mean()
        second_actor_loss.backward()
        self.assertGreater(self._gradient_l1(terminal_parameters), 0.0)
        self.assertGreater(self._gradient_l1(upstream_parameters), 0.0)
        self._assert_no_gradient(self, critic_parameters)

    def test_stage_a_critic_loader_is_selective_and_atomic(self) -> None:
        torch.manual_seed(17)
        source = ActWorldPlanner(self.config)
        source_state = {
            key: value.detach().clone() for key, value in source.state_dict().items()
        }

        torch.manual_seed(23)
        target = ActWorldPlanner(self.config)
        target.configure_training_stage("actor")
        with torch.no_grad():
            target.refinement_head[-1].weight.fill_(0.5)
            target.refinement_head[-1].bias.fill_(0.25)
            target.refinement_gate_logits.fill_(1.0)
            target.score_delta_gate_raw.fill_(0.75)
        target.load_stage_a_critic_state(source_state)

        loaded_state = target.state_dict()
        for key, value in loaded_state.items():
            if target._is_stage_a_critic_key(key):
                self.assertTrue(torch.equal(value, source_state[key]), key)
        self.assertEqual(target.refinement_head[-1].weight.abs().max().item(), 0.0)
        self.assertEqual(target.refinement_head[-1].bias.abs().max().item(), 0.0)
        self.assertEqual(target.refinement_gate_logits.abs().max().item(), 0.0)
        self.assertEqual(target.score_delta_gate_raw.item(), 0.0)

        before_failure = {
            key: value.detach().clone() for key, value in target.state_dict().items()
        }
        missing = dict(source_state)
        del missing["transition.weight_hh"]
        with self.assertRaisesRegex(RuntimeError, "missing"):
            target.load_stage_a_critic_state(missing)
        for key, value in target.state_dict().items():
            self.assertTrue(torch.equal(value, before_failure[key]), key)

        mismatched = dict(source_state)
        mismatched["world_feature.0.weight"] = source_state[
            "world_feature.0.weight"
        ][:-1]
        with self.assertRaisesRegex(RuntimeError, "shape_mismatch"):
            target.load_stage_a_critic_state(mismatched)
        for key, value in target.state_dict().items():
            self.assertTrue(torch.equal(value, before_failure[key]), key)

    def test_historical_one_step_consistency_is_scalar_and_differentiable(self) -> None:
        previous = torch.randn(self.batch, 7, self.config.token_dim, requires_grad=True)
        current = torch.randn(self.batch, 11, self.config.token_dim, requires_grad=True)
        action = torch.randn(self.batch, 3, requires_grad=True)
        predicted = self.planner.predict_next_latent(previous, action)
        self.assertEqual(
            predicted.shape,
            (self.batch, self.config.latent_tokens, self.config.token_dim),
        )

        loss = self.planner.latent_consistency_loss(previous, current, action)
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(previous.grad)
        self.assertIsNotNone(action.grad)
        self.assertTrue(torch.isfinite(previous.grad).all())
        self.assertTrue(torch.isfinite(action.grad).all())
        # The current-token target is deliberately stop-gradient for frozen V-JEPA.
        self.assertIsNone(current.grad)

    def test_terminal_expert_rollout_uses_one_action_pass_and_ema_target(self) -> None:
        current_tokens = torch.randn(
            self.batch, 9, self.config.token_dim, requires_grad=True
        )
        future_tokens = torch.randn(
            self.batch, 13, self.config.token_dim, requires_grad=True
        )
        expert = torch.randn(
            self.batch, self.config.horizon, 3, requires_grad=True
        )
        scene_latent, _ = self.planner._encode_scene(current_tokens)
        valid = torch.tensor([True, False])

        with patch.object(
            self.planner,
            "_action_features",
            wraps=self.planner._action_features,
        ) as action_features:
            loss = self.planner.terminal_latent_consistency_loss(
                scene_latent, expert, future_tokens, valid
            )
        self.assertEqual(action_features.call_count, 1)
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(current_tokens.grad)
        self.assertIsNotNone(expert.grad)
        self.assertIsNone(future_tokens.grad)
        self.assertTrue(torch.isfinite(current_tokens.grad).all())
        self.assertTrue(torch.isfinite(expert.grad).all())

        no_valid_loss = self.planner.terminal_latent_consistency_loss(
            scene_latent.detach(), expert.detach(), future_tokens, torch.zeros(2, dtype=torch.bool)
        )
        self.assertEqual(no_valid_loss.item(), 0.0)

    def test_ema_target_moves_without_grad_and_stays_in_eval(self) -> None:
        online_parameter = next(self.planner.scene_projection.parameters())
        target_parameter = next(self.planner.target_scene_projection.parameters())
        original_target = target_parameter.detach().clone()
        torch.testing.assert_close(online_parameter, original_target)

        with torch.no_grad():
            online_parameter.add_(2.0)
        self.planner.update_target_encoder(decay=0.75)
        torch.testing.assert_close(target_parameter, original_target + 0.5)

        self.planner.train(True)
        self.assertFalse(self.planner.target_token_norm.training)
        self.assertFalse(self.planner.target_scene_cross_attention.training)
        self.assertFalse(self.planner.target_scene_projection.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in self.planner._target_scene_modules()
                for parameter in module.parameters()
            )
        )
        self.assertFalse(self.planner.target_scene_queries.requires_grad)

    def test_runtime_config_aliases_and_explicit_override(self) -> None:
        planner = ActWorldPlanner(_DriveLikeConfig(), num_candidates=2)
        self.assertEqual(planner.config.token_dim, 12)
        self.assertEqual(planner.config.num_candidates, 2)
        self.assertEqual(planner.config.horizon, 4)
        self.assertEqual(planner.config.hidden_dim, 24)
        self.assertEqual(planner.config.latent_tokens, 2)
        self.assertEqual(planner.config.num_heads, 3)
        self.assertEqual(planner.config.position_scale, 40.0)
        self.assertEqual(planner.config.max_refinement_xy, 0.7)
        self.assertEqual(planner.config.max_refinement_yaw, 0.12)
        self.assertEqual(planner.config.target_ema_decay, 0.9)

    def test_invalid_config_shapes_and_values_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "token_dim"):
            ActWorldPlannerConfig(token_dim=0).validate()
        with self.assertRaisesRegex(ValueError, "num_subscores"):
            ActWorldPlanner(num_subscores=5)
        with self.assertRaisesRegex(ValueError, "divisible"):
            ActWorldPlanner(token_dim=10, num_heads=4)
        with self.assertRaisesRegex(ValueError, "target_ema_decay"):
            ActWorldPlanner(target_ema_decay=1.0)
        with self.assertRaisesRegex(ValueError, "critic_variant"):
            ActWorldPlanner(critic_variant="unregistered")
        with self.assertRaisesRegex(ValueError, "candidates must have shape"):
            self.planner(
                self.tokens,
                self.candidates[:, :, :-1],
                self.base_logits,
            )
        with self.assertRaisesRegex(ValueError, "base_logits must have shape"):
            self.planner(
                self.tokens,
                self.candidates,
                torch.randn(self.batch, self.config.num_candidates + 1),
            )
        bad_tokens = self.tokens.clone()
        bad_tokens[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            self.planner(bad_tokens, self.candidates, self.base_logits)
        bad_action = torch.randn(self.batch, 4)
        with self.assertRaisesRegex(ValueError, "observed_action must have shape"):
            self.planner.latent_consistency_loss(self.tokens, self.tokens, bad_action)


if __name__ == "__main__":
    unittest.main()
