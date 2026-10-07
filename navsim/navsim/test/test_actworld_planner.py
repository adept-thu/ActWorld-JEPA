"""Pure-PyTorch checks for the NAVSIM v1 ActWorld planner."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from navsim.agents.actworld_jepa.actworld_planner import (
    ActWorldPlanner,
    ActWorldPlannerConfig,
)


class _RuntimeConfig:
    tf_d_model = 12
    proposal_num = 3
    num_poses = 4
    actworld_latent_tokens = 2
    actworld_num_heads = 3
    actworld_num_layers = 1
    actworld_hidden_dim = 24
    actworld_dropout = 0.0
    actworld_finite_checks = True


class _ProbabilityModel:
    def predict_proba(self, features):
        positive = np.linspace(0.1, 0.9, len(features), dtype=np.float32)
        return np.stack((1.0 - positive, positive), axis=1)


class _ConstantProbabilityModel:
    def __init__(self, positive):
        self.positive = float(positive)

    def predict_proba(self, features):
        positive = np.full(len(features), self.positive, dtype=np.float32)
        return np.stack((1.0 - positive, positive), axis=1)


class _IncreasingRegressionModel:
    def predict(self, features):
        return np.arange(len(features), dtype=np.float32)


class _ConstantFactorModel:
    def __init__(self, progress: float, critical: float):
        self.progress = float(progress)
        self.critical = float(critical)

    def predict(self, features):
        result = np.zeros((len(features), 5), dtype=np.float32)
        result[:, [0, 1, 3]] = self.critical
        result[:, 2] = self.progress
        return result


class ActWorldPlannerV1Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.config = ActWorldPlannerConfig(
            token_dim=16,
            num_candidates=4,
            horizon=5,
            latent_tokens=3,
            num_heads=4,
            num_layers=1,
            hidden_dim=32,
            dropout=0.0,
            finite_checks=True,
        )
        self.planner = ActWorldPlanner(self.config)
        self.tokens = torch.randn(2, 9, 16)
        self.candidates = torch.randn(2, 4, 5, 3)
        self.base_scores = torch.rand(2, 4)

    def test_warm_start_is_exact_drive_jepa_fallback(self) -> None:
        self.planner.eval()
        with torch.no_grad():
            output = self.planner(
                self.tokens,
                self.candidates,
                self.base_scores,
                return_auxiliary=True,
            )
        self.assertTrue(torch.equal(output["refined_candidates"], self.candidates))
        self.assertTrue(torch.equal(output["world_scores"], self.base_scores))
        self.assertTrue(torch.equal(output["initial_world_scores"], self.base_scores))
        self.assertEqual(output["preference_logits"].abs().max().item(), 0.0)
        self.assertEqual(output["initial_preference_logits"].abs().max().item(), 0.0)
        self.assertEqual(output["trajectory_residual"].abs().max().item(), 0.0)
        self.assertTrue(
            torch.equal(
                output["world_scores"].argmax(dim=1), self.base_scores.argmax(dim=1)
            )
        )

    def test_shapes_bounds_and_gradients(self) -> None:
        tokens = self.tokens.clone().requires_grad_(True)
        candidates = self.candidates.clone().requires_grad_(True)
        output = self.planner(tokens, candidates, self.base_scores, return_auxiliary=True)
        self.assertEqual(output["rollout_latents"].shape, (2, 4, 5, 3, 16))
        self.assertEqual(output["world_logits"].shape, (2, 4, 6))
        self.assertEqual(output["value_logits"].shape, (2, 4))
        self.assertEqual(output["collision_logits"].shape, (2, 4))
        self.assertEqual(output["preference_logits"].shape, (2, 4))
        rows = torch.arange(self.base_scores.shape[0])
        baseline_top = self.base_scores.argmax(dim=1)
        self.assertTrue(
            torch.equal(
                output["preference_logits"][rows, baseline_top],
                torch.zeros_like(output["preference_logits"][rows, baseline_top]),
            )
        )
        limits = self.planner.refinement_limits
        gate = output["refinement_gate"]
        self.assertTrue(
            (output["trajectory_residual"].abs() <= limits * gate + 1e-7).all()
        )
        loss = (
            output["world_logits"].square().mean()
            + output["refined_candidates"].square().mean()
            + output["rollout_latents"].square().mean()
        )
        loss.backward()
        self.assertIsNotNone(tokens.grad)
        self.assertIsNotNone(candidates.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue(torch.isfinite(candidates.grad).all())

    def test_factor_target_features_are_deterministic_and_baseline_relative(self) -> None:
        factors = torch.rand(2, 4, 6)
        baseline = torch.tensor([0, 2])
        first = self.planner.factor_target_features(factors, baseline)
        second = self.planner.factor_target_features(factors, baseline)
        self.assertEqual(tuple(first.shape), (2, 4, self.config.hidden_dim))
        self.assertTrue(torch.equal(first, second))
        changed = factors.clone()
        changed[0, 1, 2] += 0.1
        changed_features = self.planner.factor_target_features(changed, baseline)
        self.assertFalse(torch.equal(first[0, 1], changed_features[0, 1]))

    def test_geometry_progress_gate_rejects_predicted_progress_drop(self) -> None:
        planner = ActWorldPlanner(self.config)
        planner.geometry_calibrator = object()
        planner.geometry_policy = {
            "top_k": 4.0,
            "min_advantage": 0.0,
            "max_baseline_margin": 1.0,
            "min_critical_delta": -1.0,
            "min_safety_delta": -1.0,
            "min_progress_delta": 0.0,
        }
        planner._geometry_preferences = lambda *args: torch.tensor(  # type: ignore[method-assign]
            [[0.0, 1.0, 0.5, 0.25]]
        )
        baseline = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
        subscores = torch.full((1, 4, 6), 0.8)
        subscores[0, 1, 2] = 0.7
        evaluation = {
            "conservative_scores": baseline,
            "critic_confidence": torch.ones(1, 4),
            "predicted_safety": torch.ones(1, 4),
            "candidate_subscores": subscores,
        }
        selected = planner._conservative_select(
            evaluation, baseline, torch.zeros(1, 4, 5, 3)
        )
        self.assertFalse(selected["selection_accepted"].item())
        self.assertEqual(selected["selected"].item(), 0)
        self.assertLess(selected["selection_progress_delta"].item(), 0.0)

    def test_geometry_classifier_uses_positive_probability(self) -> None:
        planner = ActWorldPlanner(self.config)
        planner.geometry_calibrator = _ProbabilityModel()
        planner.geometry_prediction_mode = "positive_probability"
        evaluation = {
            "candidate_subscores": torch.full((1, 4, 6), 0.5),
            "critic_value_probabilities": torch.full((1, 4), 0.5),
            "structured_scores": torch.full((1, 4), 0.125),
            "predicted_safety": torch.full((1, 4), 0.5),
        }
        prediction = planner._geometry_preferences(
            evaluation,
            torch.tensor([[0.9, 0.8, 0.7, 0.6]]),
            torch.zeros(1, 4, 5, 3),
            torch.tensor([0]),
            torch.tensor([0.1]),
        )
        self.assertEqual(tuple(prediction.shape), (1, 4))
        self.assertAlmostEqual(prediction[0, 0].item(), 0.1, places=5)
        self.assertAlmostEqual(prediction[0, -1].item(), 0.9, places=5)

    def test_geometry_candidate_probability_gate_rejects_low_confidence(self) -> None:
        planner = ActWorldPlanner(self.config)
        planner.geometry_calibrator = object()
        planner.geometry_policy = {
            "top_k": 4.0,
            "min_advantage": 0.0,
            "max_baseline_margin": 1.0,
            "min_critical_delta": -1.0,
            "min_safety_delta": -1.0,
            "min_progress_delta": -1.0,
            "min_candidate_probability": 0.9,
        }
        planner._geometry_preferences = lambda *args: torch.tensor(  # type: ignore[method-assign]
            [[0.1, 0.8, 0.5, 0.25]]
        )
        baseline = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
        evaluation = {
            "conservative_scores": baseline,
            "critic_confidence": torch.ones(1, 4),
            "predicted_safety": torch.ones(1, 4),
            "candidate_subscores": torch.full((1, 4, 6), 0.8),
        }
        selected = planner._conservative_select(
            evaluation, baseline, torch.zeros(1, 4, 5, 3)
        )
        self.assertFalse(selected["selection_accepted"].item())
        self.assertEqual(selected["selected"].item(), 0)

    def test_planning_switch_selects_ranked_candidate(self) -> None:
        config = ActWorldPlannerConfig(
            token_dim=64,
            num_candidates=4,
            horizon=8,
            latent_tokens=2,
            num_heads=8,
            num_layers=1,
            hidden_dim=64,
            dropout=0.0,
            finite_checks=True,
            selection_mode="conservative",
        )
        planner = ActWorldPlanner(config)
        planner.planning_switch_projection = torch.eye(64)
        planner.planning_switch = {
            "top2_model": _ConstantProbabilityModel(0.2),
            "topk_model": [
                _ConstantProbabilityModel(0.2),
                _ConstantProbabilityModel(0.8),
                _ConstantProbabilityModel(0.4),
            ],
            "top2_config": {"variant": "context"},
            "topk_config": {"variant": "context"},
            "policy": {
                "top_k": 3,
                "min_probability": 0.7,
                "max_candidate_gap": 0.1,
                "min_critical_delta": -0.02,
                "min_safety_delta": -0.05,
                "min_progress_delta": 0.0,
                "top2_weight": 0.5,
                "runner_bonus": 0.05,
            },
        }
        with torch.no_grad():
            output = planner(
                torch.randn(1, 5, 64),
                torch.randn(1, 4, 8, 3),
                torch.tensor([[0.9, 0.89, 0.88, 0.87]]),
            )
        self.assertTrue(output["conservative_selection_accepted"].item())
        self.assertEqual(output["conservative_selected"].item(), 2)

    def test_latent_verifier_selects_candidate_online(self) -> None:
        planner = ActWorldPlanner(
            ActWorldPlannerConfig(
                token_dim=16,
                num_candidates=4,
                horizon=5,
                latent_tokens=3,
                num_heads=4,
                num_layers=1,
                hidden_dim=32,
                dropout=0.0,
                selection_mode="conservative",
                finite_checks=True,
            )
        )
        planner.latent_verifier = _IncreasingRegressionModel()
        planner.latent_verifier_config = {"kind": "scalar"}
        planner.latent_verifier_policy = {
            "top_k": 4.0,
            "min_advantage": 0.0,
            "max_baseline_margin": 1.0,
            "min_critical_delta": -1.0,
            "min_safety_delta": -1.0,
            "min_progress_delta": -1.0,
        }
        planner.latent_verifier_projection = torch.eye(32)
        with torch.no_grad():
            output = planner(
                self.tokens[:1],
                self.candidates[:1],
                torch.tensor([[0.9, 0.89, 0.88, 0.87]]),
            )
        self.assertTrue(output["conservative_selection_accepted"].item())
        self.assertEqual(output["conservative_selected"].item(), 3)

    def test_dualhead_latent_verifier_vetoes_unsafe_factor_prediction(self) -> None:
        planner = ActWorldPlanner(
            ActWorldPlannerConfig(
                token_dim=16,
                num_candidates=4,
                horizon=5,
                latent_tokens=3,
                num_heads=4,
                num_layers=1,
                hidden_dim=32,
                dropout=0.0,
                selection_mode="conservative",
                finite_checks=True,
            )
        )
        planner.latent_verifier = _IncreasingRegressionModel()
        planner.latent_verifier_config = {"kind": "scalar"}
        planner.latent_verifier_projection = torch.eye(32)
        planner.latent_verifier_factor = _ConstantFactorModel(
            progress=0.1, critical=-0.1
        )
        planner.latent_verifier_factor_config = {"kind": "factor_relative"}
        planner.latent_verifier_factor_projection = torch.eye(32)
        planner.latent_verifier_policy = {
            "top_k": 4.0,
            "min_advantage": 0.0,
            "max_baseline_margin": 1.0,
            "min_critical_delta": -1.0,
            "min_safety_delta": -1.0,
            "min_progress_delta": -1.0,
            "min_factor_critical_delta": 0.0,
            "min_factor_progress_delta": 0.0,
        }
        with torch.no_grad():
            output = planner(
                self.tokens[:1],
                self.candidates[:1],
                torch.tensor([[0.9, 0.89, 0.88, 0.87]]),
            )
        self.assertFalse(output["conservative_selection_accepted"].item())
        self.assertEqual(output["conservative_conservative_selected"].item(), 3)
        self.assertEqual(output["conservative_selected"].item(), 0)

    def test_runtime_aliases_and_validation(self) -> None:
        planner = ActWorldPlanner(_RuntimeConfig())
        self.assertEqual(planner.config.token_dim, 12)
        self.assertEqual(planner.config.num_candidates, 3)
        self.assertEqual(planner.config.horizon, 4)
        self.assertEqual(planner.config.latent_tokens, 2)
        with self.assertRaisesRegex(ValueError, "candidates must have shape"):
            self.planner(self.tokens, self.candidates[:, :, :-1], self.base_scores)
        bad_tokens = self.tokens.clone()
        bad_tokens[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            self.planner(bad_tokens, self.candidates, self.base_scores)

    def test_conservative_policy_has_exact_neutral_fallback(self) -> None:
        planner = ActWorldPlanner(
            ActWorldPlannerConfig(
                token_dim=16,
                num_candidates=4,
                horizon=5,
                latent_tokens=3,
                num_heads=4,
                num_layers=1,
                hidden_dim=32,
                dropout=0.0,
                selection_mode="conservative",
                finite_checks=True,
            )
        )
        planner.eval()
        with torch.no_grad():
            output = planner(self.tokens, self.candidates, self.base_scores)
        self.assertTrue(
            torch.equal(output["initial_conservative_scores"], self.base_scores)
        )
        self.assertTrue(
            torch.equal(
                output["conservative_selected"], self.base_scores.argmax(dim=1)
            )
        )
        self.assertFalse(output["conservative_selection_accepted"].logical_not().all())

    def test_conservative_policy_rejects_invalid_mixture(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to at most one"):
            ActWorldPlannerConfig(
                conservative_value_weight=0.8,
                conservative_structured_weight=0.3,
            ).validate()

    def test_preference_residual_scale_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "preference_residual_scale"):
            ActWorldPlannerConfig(preference_residual_scale=1.1).validate()

    def test_action_conditioned_outcome_delta_is_baseline_relative(self) -> None:
        planner = ActWorldPlanner(
            ActWorldPlannerConfig(
                token_dim=16,
                num_candidates=4,
                horizon=5,
                latent_tokens=3,
                num_heads=4,
                num_layers=1,
                hidden_dim=32,
                dropout=0.0,
                selection_mode="conservative",
            )
        ).eval()
        with torch.no_grad():
            output = planner(self.tokens, self.candidates, self.base_scores)
        delta = output["initial_outcome_action_factor_deltas"]
        rows = torch.arange(self.base_scores.shape[0])
        baseline = self.base_scores.argmax(dim=1)
        self.assertEqual(delta.shape, (2, 4, 6))
        self.assertTrue(torch.equal(delta[rows, baseline], torch.zeros(2, 6)))

    def test_temporal_action_jepa_is_baseline_relative(self) -> None:
        planner = ActWorldPlanner(
            ActWorldPlannerConfig(
                token_dim=16,
                num_candidates=4,
                horizon=5,
                latent_tokens=3,
                num_heads=4,
                num_layers=1,
                hidden_dim=32,
                dropout=0.0,
                selection_mode="conservative",
            )
        ).eval()
        with torch.no_grad():
            output = planner(self.tokens, self.candidates, self.base_scores)
        delta = output["initial_temporal_action_factor_deltas"]
        rows = torch.arange(self.base_scores.shape[0])
        baseline = self.base_scores.argmax(dim=1)
        self.assertEqual(delta.shape, (2, 4, 6))
        self.assertTrue(torch.equal(delta[rows, baseline], torch.zeros(2, 6)))

    def test_only_one_outcome_delta_selector_can_be_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one Outcome-JEPA"):
            ActWorldPlannerConfig(
                outcome_delta_selection=True,
                outcome_action_delta_selection=True,
            ).validate()
        with self.assertRaisesRegex(ValueError, "only one Outcome-JEPA"):
            ActWorldPlannerConfig(
                outcome_action_delta_selection=True,
                temporal_action_delta_selection=True,
            ).validate()

    def test_conservative_inference_skips_unused_refinement(self) -> None:
        planner = ActWorldPlanner(
            ActWorldPlannerConfig(
                token_dim=16,
                num_candidates=4,
                horizon=5,
                latent_tokens=3,
                num_heads=4,
                num_layers=1,
                hidden_dim=32,
                dropout=0.0,
                selection_mode="conservative",
                finite_checks=True,
            )
        ).eval()
        with torch.no_grad():
            inference = planner(self.tokens, self.candidates, self.base_scores)
            auxiliary = planner(
                self.tokens,
                self.candidates,
                self.base_scores,
                return_auxiliary=True,
            )
        self.assertTrue(torch.equal(inference["refined_candidates"], self.candidates))
        self.assertEqual(inference["trajectory_residual"].abs().max().item(), 0.0)
        self.assertTrue(
            torch.equal(
                inference["conservative_selected"],
                auxiliary["conservative_selected"],
            )
        )
        self.assertTrue(
            torch.equal(
                inference["conservative_gated_scores"],
                auxiliary["conservative_gated_scores"],
            )
        )

    def test_conservative_top_k_and_critical_channel_gate(self) -> None:
        config = ActWorldPlannerConfig(
            token_dim=16,
            num_candidates=4,
            horizon=5,
            latent_tokens=3,
            num_heads=4,
            num_layers=1,
            hidden_dim=32,
            conservative_top_k=2,
            conservative_min_critical_delta=0.0,
        )
        planner = ActWorldPlanner(config)
        baseline = torch.tensor([[0.9, 0.8, 0.1, 0.0]])
        subscores = torch.full((1, 4, 6), 0.8)
        # Candidate 1 is shortlisted but predicts worse TTC than the baseline.
        subscores[0, 1, 3] = 0.7
        evaluation = {
            "conservative_scores": torch.tensor([[0.9, 1.0, 1.5, 2.0]]),
            "critic_confidence": torch.ones(1, 4),
            "predicted_safety": torch.ones(1, 4),
            "candidate_subscores": subscores,
        }
        selected = planner._conservative_select(evaluation, baseline)
        # Candidate 3 has the largest critic score but is outside baseline top-2.
        self.assertEqual(selected["conservative_selected"].item(), 1)
        # The per-channel safety constraint then falls back to candidate 0.
        self.assertFalse(selected["selection_accepted"].item())
        self.assertEqual(selected["selected"].item(), 0)

    def test_pairwise_verifier_can_veto_and_rescue(self) -> None:
        weights = [0.0] * 95
        # Raw feature 7 is the candidate-relative value probability.
        weights[7] = 1.0
        artifact = {
            "robust": {
                "all_development_fit": {
                    "scale": [1.0] * 95,
                    "weights": weights,
                    "intercept": 0.0,
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "navtrain_pairwise.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            planner = ActWorldPlanner(
                ActWorldPlannerConfig(
                    token_dim=16,
                    num_candidates=4,
                    horizon=5,
                    latent_tokens=3,
                    num_heads=4,
                    num_layers=1,
                    hidden_dim=32,
                    pairwise_calibrator_path=str(path),
                    pairwise_top_k=4,
                    pairwise_veto_threshold=-0.1,
                    pairwise_rescue_threshold=0.1,
                    pairwise_rescue_margin=1.0,
                )
            )
        baseline = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
        subscores = torch.full((1, 4, 6), 0.8)
        value = torch.tensor([[0.5, 0.2, 0.9, 0.4]])
        evaluation = {
            "conservative_scores": torch.tensor([[0.9, 1.0, 0.7, 0.6]]),
            "critic_confidence": torch.ones(1, 4),
            "predicted_safety": torch.ones(1, 4),
            "candidate_subscores": subscores,
            "critic_value_probabilities": value,
            "structured_scores": torch.full((1, 4), 0.5),
        }
        selected = planner._conservative_select(evaluation, baseline)
        self.assertTrue(selected["pairwise_vetoed"].item())
        self.assertTrue(selected["pairwise_rescued"].item())
        self.assertEqual(selected["pairwise_candidate"].item(), 2)
        self.assertEqual(selected["selected"].item(), 2)
        self.assertEqual(selected["gated_scores"].argmax(dim=1).item(), 2)


if __name__ == "__main__":
    unittest.main()
