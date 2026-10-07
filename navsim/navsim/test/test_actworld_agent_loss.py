"""Focused tests for ActWorld v1 verifier ranking losses."""

import unittest

import torch

from navsim.agents.actworld_jepa.actworld_agent import ActWorldJEPAAgent


class ActWorldAgentLossTest(unittest.TestCase):
    def test_planning_head_parameter_filter_is_narrow(self) -> None:
        selected = {
            name
            for name in (
                "_trajectory_head.0.decoder.weight",
                "hist_encoding.0.weight",
                "init_feature.weight",
                "_backbone.encoder.weight",
                "scorer.mlp.weight",
                "_actworld_planner.world_tokens",
            )
            if ActWorldJEPAAgent._is_planning_head_parameter(name)
        }
        self.assertEqual(
            selected,
            {
                "_trajectory_head.0.decoder.weight",
                "hist_encoding.0.weight",
                "init_feature.weight",
            },
        )

    def test_safe_oracle_distillation_uses_safe_better_teacher(self) -> None:
        proposals = torch.zeros(1, 3, 2, 3, requires_grad=True)
        proposals.data[:, 1] = 1.0
        baseline = torch.tensor([[0.9, 0.8, 0.7]])
        targets = torch.tensor(
            [[[1.0, 1.0, 0.5, 1.0, 1.0, 0.6],
              [1.0, 1.0, 0.7, 1.0, 1.0, 0.8],
              [1.0, 1.0, 0.4, 1.0, 1.0, 0.5]]]
        )
        loss, adoption, gain = (
            ActWorldJEPAAgent._safe_oracle_proposal_distillation_loss(
                proposals, baseline, targets, margin=0.01, max_gain=0.5
            )
        )
        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(adoption.item(), 1.0)
        self.assertAlmostEqual(gain.item(), 0.2, places=6)
        loss.backward()
        self.assertGreater(proposals.grad[:, 0].abs().sum().item(), 0.0)
        self.assertEqual(proposals.grad[:, 1:].abs().sum().item(), 0.0)

    def test_safe_oracle_distillation_rejects_unsafe_gain(self) -> None:
        proposals = torch.zeros(1, 3, 2, 3, requires_grad=True)
        baseline = torch.tensor([[0.9, 0.8, 0.7]])
        targets = torch.tensor(
            [[[1.0, 1.0, 0.5, 1.0, 1.0, 0.6],
              [0.0, 1.0, 0.9, 1.0, 1.0, 0.9],
              [1.0, 1.0, 0.4, 1.0, 1.0, 0.5]]]
        )
        loss, adoption, gain = (
            ActWorldJEPAAgent._safe_oracle_proposal_distillation_loss(
                proposals, baseline, targets, margin=0.01, max_gain=0.5
            )
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(adoption.item(), 0.0)
        self.assertEqual(gain.item(), 0.0)

    def test_safe_oracle_distillation_validates_thresholds(self) -> None:
        proposals = torch.zeros(1, 2, 2, 3)
        baseline = torch.zeros(1, 2)
        targets = torch.zeros(1, 2, 6)
        with self.assertRaisesRegex(ValueError, "margin"):
            ActWorldJEPAAgent._safe_oracle_proposal_distillation_loss(
                proposals, baseline, targets, margin=-0.1, max_gain=0.5
            )

    def test_hard_negative_loss_rewards_correct_order(self) -> None:
        targets = torch.tensor([[0.9, 0.2, 0.1]])
        baseline = torch.tensor([[0.8, 0.7, 0.6]])
        correct = torch.tensor([[1.0, 0.0, -0.5]])
        wrong = torch.tensor([[0.0, 1.0, -0.5]])
        correct_loss = ActWorldJEPAAgent._coarse_to_fine_rank_loss(
            correct, targets, baseline, top_k=3, temperature=0.1
        )
        wrong_loss = ActWorldJEPAAgent._coarse_to_fine_rank_loss(
            wrong, targets, baseline, top_k=3, temperature=0.1
        )
        self.assertLess(correct_loss.item(), wrong_loss.item())

    def test_hard_negative_loss_uses_only_coarse_shortlist(self) -> None:
        targets = torch.tensor([[0.8, 0.2, 1.0]])
        baseline = torch.tensor([[0.9, 0.8, 0.1]])
        first = torch.tensor([[0.6, 0.4, -100.0]])
        second = torch.tensor([[0.6, 0.4, 100.0]])
        first_loss = ActWorldJEPAAgent._coarse_to_fine_rank_loss(
            first, targets, baseline, top_k=2, temperature=0.1
        )
        second_loss = ActWorldJEPAAgent._coarse_to_fine_rank_loss(
            second, targets, baseline, top_k=2, temperature=0.1
        )
        self.assertEqual(first_loss.item(), second_loss.item())

    def test_hard_negative_loss_validates_configuration(self) -> None:
        values = torch.zeros(1, 3)
        with self.assertRaisesRegex(ValueError, "at least two"):
            ActWorldJEPAAgent._coarse_to_fine_rank_loss(
                values, values, values, top_k=1, temperature=0.1
            )

    def test_pairwise_loss_rewards_baseline_relative_preferences(self) -> None:
        targets = torch.tensor([[0.5, 0.9, 0.1]])
        baseline = torch.tensor([[0.9, 0.8, 0.7]])
        correct = torch.tensor([[0.0, 1.0, -1.0]])
        wrong = torch.tensor([[0.0, -1.0, 1.0]])
        correct_loss = ActWorldJEPAAgent._baseline_pairwise_loss(
            correct, targets, baseline, top_k=3, temperature=0.1
        )
        wrong_loss = ActWorldJEPAAgent._baseline_pairwise_loss(
            wrong, targets, baseline, top_k=3, temperature=0.1
        )
        self.assertLess(correct_loss.item(), wrong_loss.item())

    def test_pairwise_loss_uses_only_coarse_shortlist(self) -> None:
        targets = torch.tensor([[0.5, 0.9, 1.0]])
        baseline = torch.tensor([[0.9, 0.8, 0.1]])
        first = torch.tensor([[0.0, 1.0, -100.0]])
        second = torch.tensor([[0.0, 1.0, 100.0]])
        first_loss = ActWorldJEPAAgent._baseline_pairwise_loss(
            first, targets, baseline, top_k=2, temperature=0.1
        )
        second_loss = ActWorldJEPAAgent._baseline_pairwise_loss(
            second, targets, baseline, top_k=2, temperature=0.1
        )
        self.assertEqual(first_loss.item(), second_loss.item())

    def test_all_pairs_loss_rewards_complete_order(self) -> None:
        targets = torch.tensor([[0.9, 0.6, 0.2]])
        baseline = torch.tensor([[0.9, 0.8, 0.7]])
        correct = torch.tensor([[1.0, 0.0, -1.0]])
        wrong = torch.tensor([[1.0, -1.0, 0.0]])
        correct_loss = ActWorldJEPAAgent._all_pairs_rank_loss(
            correct, targets, baseline, top_k=3, temperature=0.1
        )
        wrong_loss = ActWorldJEPAAgent._all_pairs_rank_loss(
            wrong, targets, baseline, top_k=3, temperature=0.1
        )
        self.assertLess(correct_loss.item(), wrong_loss.item())

    def test_all_pairs_loss_uses_only_coarse_shortlist(self) -> None:
        targets = torch.tensor([[0.9, 0.6, 1.0]])
        baseline = torch.tensor([[0.9, 0.8, 0.1]])
        first = torch.tensor([[1.0, 0.0, -100.0]])
        second = torch.tensor([[1.0, 0.0, 100.0]])
        first_loss = ActWorldJEPAAgent._all_pairs_rank_loss(
            first, targets, baseline, top_k=2, temperature=0.1
        )
        second_loss = ActWorldJEPAAgent._all_pairs_rank_loss(
            second, targets, baseline, top_k=2, temperature=0.1
        )
        self.assertEqual(first_loss.item(), second_loss.item())

    def test_all_pairs_loss_validates_configuration(self) -> None:
        values = torch.zeros(1, 3)
        with self.assertRaisesRegex(ValueError, "at least two"):
            ActWorldJEPAAgent._all_pairs_rank_loss(
                values, values, values, top_k=1, temperature=0.1
            )

    def test_progress_pairwise_rewards_safe_progress_order(self) -> None:
        baseline = torch.tensor([[0.9, 0.8, 0.7]])
        targets = torch.tensor(
            [[
                [1.0, 1.0, 0.5, 1.0, 1.0, 0.8],
                [1.0, 1.0, 0.8, 1.0, 1.0, 0.9],
                [1.0, 1.0, 0.2, 1.0, 1.0, 0.7],
            ]]
        )
        correct = torch.tensor([[0.0, 1.0, -1.0]])
        wrong = torch.tensor([[0.0, -1.0, 1.0]])
        kwargs = dict(
            simulator_targets=targets,
            baseline_scores=baseline,
            top_k=3,
            temperature=0.1,
            progress_margin=0.005,
            critical_margin=0.0,
        )
        correct_loss = ActWorldJEPAAgent._progress_preserving_pairwise_loss(
            correct, **kwargs
        )
        wrong_loss = ActWorldJEPAAgent._progress_preserving_pairwise_loss(
            wrong, **kwargs
        )
        self.assertLess(correct_loss.item(), wrong_loss.item())

    def test_progress_pairwise_ignores_unsafe_progress_gain(self) -> None:
        baseline = torch.tensor([[0.9, 0.8, 0.7]])
        targets = torch.tensor(
            [[
                [1.0, 1.0, 0.5, 1.0, 1.0, 0.8],
                [0.0, 1.0, 0.9, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.2, 1.0, 1.0, 0.7],
            ]]
        )
        first = torch.tensor([[0.0, -100.0, -1.0]])
        second = torch.tensor([[0.0, 100.0, -1.0]])
        kwargs = dict(
            simulator_targets=targets,
            baseline_scores=baseline,
            top_k=3,
            temperature=0.1,
            progress_margin=0.005,
            critical_margin=0.0,
        )
        first_loss = ActWorldJEPAAgent._progress_preserving_pairwise_loss(
            first, **kwargs
        )
        second_loss = ActWorldJEPAAgent._progress_preserving_pairwise_loss(
            second, **kwargs
        )
        self.assertEqual(first_loss.item(), second_loss.item())

    def test_risk_averse_loss_penalizes_false_positive_switch(self) -> None:
        targets = torch.tensor([[0.8, 0.7, 0.9]])
        baseline = torch.tensor([[0.9, 0.8, 0.7]])
        safe = torch.tensor([[0.0, -1.0, 1.0]])
        risky = torch.tensor([[0.0, 1.0, -1.0]])
        kwargs = dict(
            target_scores=targets,
            baseline_scores=baseline,
            top_k=3,
            temperature=0.1,
            positive_margin=0.005,
            false_positive_weight=4.0,
        )
        safe_loss = ActWorldJEPAAgent._risk_averse_pairwise_loss(safe, **kwargs)
        risky_loss = ActWorldJEPAAgent._risk_averse_pairwise_loss(risky, **kwargs)
        self.assertLess(safe_loss.item(), risky_loss.item())

    def test_risk_averse_loss_weights_false_positives(self) -> None:
        targets = torch.tensor([[0.8, 0.7, 0.9]])
        baseline = torch.tensor([[0.9, 0.8, 0.7]])
        prediction = torch.tensor([[0.0, 0.5, 0.5]])
        kwargs = dict(
            predicted_scores=prediction,
            target_scores=targets,
            baseline_scores=baseline,
            top_k=3,
            temperature=0.1,
            positive_margin=0.005,
        )
        balanced = ActWorldJEPAAgent._risk_averse_pairwise_loss(
            false_positive_weight=1.0, **kwargs
        )
        risk_averse = ActWorldJEPAAgent._risk_averse_pairwise_loss(
            false_positive_weight=8.0, **kwargs
        )
        self.assertGreater(risk_averse.item(), balanced.item())

    def test_factorwise_loss_penalizes_hidden_safety_regression(self) -> None:
        targets = torch.full((1, 3, 6), 0.8)
        targets[0, 0, 1] = 1.0
        targets[0, 1, 1] = 0.0
        baseline = torch.tensor([[0.9, 0.8, 0.7]])
        correct = torch.zeros(1, 3, 6)
        wrong = torch.zeros(1, 3, 6)
        correct[0, 1, 1] = -1.0
        wrong[0, 1, 1] = 1.0
        kwargs = dict(
            simulator_targets=targets,
            baseline_scores=baseline,
            top_k=3,
            temperature=0.1,
            safety_false_positive_weight=8.0,
            progress_weight=0.5,
            nc_weight=1.0,
            dac_weight=1.0,
            ttc_weight=1.0,
            comfort_weight=1.0,
        )
        correct_loss = ActWorldJEPAAgent._factorwise_pairwise_loss(
            correct, **kwargs
        )
        wrong_loss = ActWorldJEPAAgent._factorwise_pairwise_loss(wrong, **kwargs)
        self.assertLess(correct_loss.item(), wrong_loss.item())

    def test_counterfactual_outcome_jepa_rewards_factor_alignment(self) -> None:
        torch.manual_seed(17)
        target = torch.randn(2, 4, 16)
        simulator = torch.rand(2, 4, 6)
        baseline = torch.tensor(
            [[0.9, 0.8, 0.7, 0.6], [0.85, 0.82, 0.8, 0.7]]
        )
        aligned = ActWorldJEPAAgent._counterfactual_outcome_jepa_loss(
            target.clone(), target, simulator, baseline, top_k=4, relative_weight=0.5
        )
        permuted = ActWorldJEPAAgent._counterfactual_outcome_jepa_loss(
            target.flip(1), target, simulator, baseline, top_k=4, relative_weight=0.5
        )
        self.assertLess(aligned.item(), permuted.item())

    def test_counterfactual_outcome_jepa_validates_relative_weight(self) -> None:
        values = torch.ones(1, 2, 8)
        with self.assertRaisesRegex(ValueError, "relative_weight"):
            ActWorldJEPAAgent._counterfactual_outcome_jepa_loss(
                values,
                values,
                torch.ones(1, 2, 6),
                torch.ones(1, 2),
                top_k=2,
                relative_weight=1.1,
            )

    def test_factorized_outcome_prediction_rewards_correct_factors(self) -> None:
        targets = torch.tensor(
            [[
                [1.0, 1.0, 0.80, 1.0, 1.0, 0.92],
                [1.0, 1.0, 0.95, 1.0, 1.0, 0.98],
                [0.0, 1.0, 1.00, 0.0, 1.0, 0.00],
            ]]
        )
        baseline = torch.tensor([[0.9, 0.8, 0.7]])
        correct = torch.logit(targets.clamp(0.01, 0.99)).requires_grad_()
        wrong = (-correct.detach()).requires_grad_()
        kwargs = dict(
            simulator_targets=targets,
            baseline_scores=baseline,
            top_k=3,
            safety_negative_weight=16.0,
            false_positive_weight=8.0,
            temperature=0.1,
        )
        correct_terms = ActWorldJEPAAgent._factorized_outcome_prediction_loss(
            correct, **kwargs
        )
        wrong_terms = ActWorldJEPAAgent._factorized_outcome_prediction_loss(
            wrong, **kwargs
        )
        self.assertLess(sum(correct_terms).item(), sum(wrong_terms).item())
        sum(correct_terms).backward()
        self.assertTrue(torch.isfinite(correct.grad).all())

    def test_factorized_outcome_prediction_validates_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            ActWorldJEPAAgent._factorized_outcome_prediction_loss(
                torch.zeros(1, 2, 5),
                torch.zeros(1, 2, 6),
                torch.zeros(1, 2),
                top_k=2,
                safety_negative_weight=1.0,
                false_positive_weight=1.0,
                temperature=0.1,
            )

    def test_counterfactual_outcome_delta_rewards_correct_direction(self) -> None:
        targets = torch.tensor(
            [[
                [1.0, 1.0, 0.80, 1.0, 1.0, 0.90],
                [1.0, 1.0, 0.95, 1.0, 1.0, 0.97],
                [0.0, 1.0, 1.00, 0.0, 1.0, 0.00],
            ]]
        )
        baseline = torch.tensor([[0.9, 0.8, 0.7]])
        target_delta = targets - targets[:, :1]
        correct = target_delta.clone().requires_grad_()
        wrong = (-target_delta).requires_grad_()
        agent = object.__new__(ActWorldJEPAAgent)
        kwargs = dict(
            simulator_targets=targets,
            baseline_scores=baseline,
            top_k=3,
            temperature=0.1,
            false_positive_weight=8.0,
        )
        correct_terms = agent._counterfactual_outcome_delta_loss(correct, **kwargs)
        wrong_terms = agent._counterfactual_outcome_delta_loss(wrong, **kwargs)
        self.assertLess(sum(correct_terms).item(), sum(wrong_terms).item())
        sum(correct_terms).backward()
        self.assertTrue(torch.isfinite(correct.grad).all())

    def test_shortlist_distillation_rewards_teacher_order(self) -> None:
        targets = torch.tensor([[0.8, 0.9, 0.4]])
        baseline = torch.tensor([[0.9, 0.8, 0.7]])
        correct = torch.tensor([[0.0, 1.0, -1.0]])
        wrong = torch.tensor([[1.0, 0.0, -1.0]])
        kwargs = dict(
            target_scores=targets,
            baseline_scores=baseline,
            top_k=3,
            student_temperature=0.1,
            teacher_temperature=0.1,
        )
        correct_loss = ActWorldJEPAAgent._shortlist_distillation_loss(correct, **kwargs)
        wrong_loss = ActWorldJEPAAgent._shortlist_distillation_loss(wrong, **kwargs)
        self.assertLess(correct_loss.item(), wrong_loss.item())

    def test_shortlist_distillation_ignores_candidates_outside_top_k(self) -> None:
        targets = torch.tensor([[0.8, 0.9, 1.0]])
        baseline = torch.tensor([[0.9, 0.8, 0.1]])
        first = torch.tensor([[0.0, 1.0, -100.0]])
        second = torch.tensor([[0.0, 1.0, 100.0]])
        kwargs = dict(
            target_scores=targets,
            baseline_scores=baseline,
            top_k=2,
            student_temperature=0.1,
            teacher_temperature=0.1,
        )
        first_loss = ActWorldJEPAAgent._shortlist_distillation_loss(first, **kwargs)
        second_loss = ActWorldJEPAAgent._shortlist_distillation_loss(second, **kwargs)
        self.assertEqual(first_loss.item(), second_loss.item())


if __name__ == "__main__":
    unittest.main()
