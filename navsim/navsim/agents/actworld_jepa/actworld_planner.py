"""Pure-PyTorch candidate-conditioned world planning for Drive-JEPA v1.

The module retains Drive-JEPA's 32 proposals.  It rolls a compact scene latent
forward under each proposal, evaluates the imagined futures, predicts a small
bounded residual, then rolls out and evaluates the corrected proposals again.
No NAVSIM or nuPlan import is required by this file.
"""

from __future__ import annotations

import math
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ActWorldPlannerConfig:
    token_dim: int = 256
    num_candidates: int = 32
    horizon: int = 8
    latent_tokens: int = 8
    num_heads: int = 8
    num_layers: int = 1
    hidden_dim: int = 512
    dropout: float = 0.0
    time_delta: float = 0.5
    position_scale: float = 50.0
    speed_scale: float = 20.0
    acceleration_scale: float = 10.0
    curvature_scale: float = 1.0
    max_refinement_xy: float = 1.0
    max_refinement_yaw: float = 0.20
    refinement_gate_init: float = -4.0
    world_value_weight: float = 0.35
    subscore_weight: float = 0.25
    collision_penalty: float = 0.25
    finite_checks: bool = False
    selection_mode: str = "legacy"
    deploy_refined_selected: bool = False
    conservative_value_weight: float = 0.5
    conservative_structured_weight: float = 0.25
    conservative_residual_scale: float = 0.1
    conservative_consistency_temperature: float = 1.0
    conservative_min_confidence: float = 0.0
    conservative_min_advantage: float = 0.0
    conservative_max_baseline_margin: float = 1.0
    conservative_safety_margin: float = -1.0
    conservative_min_critical_delta: float = -1.0
    conservative_top_k: int = 32
    preference_residual_scale: float = 0.0
    pairwise_calibrator_path: str = ""
    pairwise_top_k: int = 16
    pairwise_veto_threshold: float = -0.2
    pairwise_rescue_threshold: float = 0.05
    pairwise_rescue_margin: float = 0.005
    pairwise_rescue_critical_delta: float = 0.0
    pairwise_rescue_safety_delta: float = -0.05
    geometry_calibrator_path: str = ""
    planning_jepa_verifier_path: str = ""
    planning_jepa_verifier_reader_checkpoint_path: str = ""
    planning_switch_path: str = ""
    latent_verifier_path: str = ""
    latent_residual_veto_path: str = ""
    candidate_set_scorer_path: str = ""
    tree_meta_scorer_path: str = ""
    post_r94_selector_path: str = ""
    post_r94_selector_consensus_path: str = ""
    post_r94_selector_allow_navtest_tuning: bool = False
    candidate_set_max_proxy_acceleration: float = 6.1125
    candidate_set_max_proxy_jerk: float = 16.74
    outcome_delta_selection: bool = False
    outcome_delta_selection_top_k: int = 2
    outcome_delta_min_score_delta: float = 0.0
    outcome_delta_min_critical_delta: float = 0.0
    outcome_delta_min_comfort_delta: float = 0.0
    outcome_delta_max_candidate_gap: float = 0.005
    outcome_action_delta_selection: bool = False
    temporal_action_delta_selection: bool = False
    decoupled_current_future_branch: bool = False
    token_attention_gate_init: float = 0.0
    current_future_v2_selection: bool = False
    current_future_v2_top_k: int = 4
    current_future_v2_min_score_delta: float = 0.0
    current_future_v2_min_critical_delta: float = 0.0
    current_future_v2_min_progress_delta: float = 0.0
    current_future_v2_min_comfort_delta: float = 0.0
    current_future_v2_max_candidate_gap: float = 0.005
    current_future_v2_cascade_on_candidate_set: bool = False
    current_future_v2_safety_weight: float = 0.0
    current_future_v2_progress_weight: float = 0.0
    current_future_v2_comfort_weight: float = 0.0
    current_future_v2_min_frozen_critical: float = -1.0
    current_future_v2_min_frozen_progress: float = -1.0
    current_future_v2_min_frozen_safety: float = -1.0
    current_future_rank_selection: bool = False
    current_future_rank_cascade_on_candidate_set: bool = False
    current_future_rank_top_k: int = 8
    current_future_rank_probability_weight: float = 0.0
    current_future_rank_min_gain: float = 0.0
    current_future_rank_ensemble_checkpoint_paths: tuple = ()
    current_future_rank_consensus_fallback: bool = False
    current_future_rank_consensus_main_min_probability: float = 0.5
    current_future_rank_consensus_main_min_gain: float = 0.0
    current_future_rank_consensus_main_min_safety_probability: float = 0.2
    current_future_rank_residual_fallback: bool = False
    current_future_rank_residual_main_is_mother: bool = False
    current_future_rank_residual_consensus: bool = False
    current_future_rank_max_initial_lateral_acceleration_increase: float = 1.0e9
    current_future_rank_min_positive_factor_count: int = 0
    current_future_rank_positive_factor_floor: float = -1.0e9
    current_future_rank_normalized_ensemble: bool = False
    current_future_rank_normalized_baseline_weight: float = 0.5
    current_future_rank_normalized_subscore_weight: float = 0.0
    current_future_rank_normalized_structured_weight: float = 0.25
    current_future_rank_normalized_value_weight: float = 0.0
    current_future_rank_normalized_safety_weight: float = 0.0
    current_future_rank_normalized_gain_weight: float = 0.2
    current_future_rank_normalized_probability_weight: float = 0.1
    current_future_rank_normalized_relative_safety: bool = True
    current_future_rank_min_probability: float = 0.5
    current_future_rank_min_safety_probability: float = 0.2
    current_future_rank_max_candidate_gap: float = 0.01
    current_future_rank_min_frozen_critical: float = -0.02
    current_future_rank_min_frozen_progress: float = 0.0
    current_future_rank_min_frozen_safety: float = -0.05
    current_future_rank_min_frozen_ttc: float = 0.0
    candidate_set_min_progress_delta: float = 0.0
    current_future_progress_selection: bool = False
    current_future_progress_min_delta: float = 0.0
    current_future_progress_weight: float = 0.0
    post_r94_progress_expansion: bool = False
    post_r94_progress_top_k: int = 8
    post_r94_progress_min_delta: float = 0.005
    post_r94_progress_min_rank_probability: float = 0.2
    post_r94_progress_min_rank_safety_probability: float = 0.2
    post_r94_progress_max_baseline_gap: float = 0.01
    post_r94_progress_min_frozen_critical: float = -0.02
    post_r94_progress_min_frozen_safety: float = 0.0
    post_r94_progress_min_frozen_ttc: float = -1.0
    risk_aware_v3_selection: bool = False
    latent_risk_veto: bool = False
    risk_aware_v3_top_k: int = 4
    risk_aware_v3_min_probability: float = 0.8
    risk_aware_v3_min_safety_probability: float = 0.8
    risk_aware_v3_min_gain: float = 0.0
    risk_aware_v3_max_candidate_gap: float = 0.005

    def validate(self) -> None:
        integer_fields = (
            "token_dim", "num_candidates", "horizon", "latent_tokens",
            "num_heads", "num_layers", "hidden_dim",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.horizon < 2:
            raise ValueError("horizon must be at least two")
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        for name in (
            "time_delta", "position_scale", "speed_scale",
            "acceleration_scale", "curvature_scale", "max_refinement_xy",
            "max_refinement_yaw", "candidate_set_max_proxy_acceleration",
            "candidate_set_max_proxy_jerk",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.selection_mode not in {"legacy", "conservative"}:
            raise ValueError("selection_mode must be 'legacy' or 'conservative'")
        mixture_sum = (
            float(self.conservative_value_weight)
            + float(self.conservative_structured_weight)
        )
        if not 0.0 <= float(self.conservative_value_weight) <= 1.0:
            raise ValueError("conservative_value_weight must be in [0, 1]")
        if not 0.0 <= float(self.conservative_structured_weight) <= 1.0:
            raise ValueError("conservative_structured_weight must be in [0, 1]")
        if mixture_sum > 1.0:
            raise ValueError("conservative critic weights must sum to at most one")
        if not 0.0 <= float(self.conservative_residual_scale) <= 1.0:
            raise ValueError("conservative_residual_scale must be in [0, 1]")
        if not 0.0 <= float(self.preference_residual_scale) <= 1.0:
            raise ValueError("preference_residual_scale must be in [0, 1]")
        if float(self.conservative_consistency_temperature) <= 0.0:
            raise ValueError("conservative_consistency_temperature must be positive")
        if not 0.0 <= float(self.conservative_min_confidence) <= 1.0:
            raise ValueError("conservative_min_confidence must be in [0, 1]")
        if float(self.conservative_min_advantage) < 0.0:
            raise ValueError("conservative_min_advantage must be nonnegative")
        if float(self.conservative_max_baseline_margin) < 0.0:
            raise ValueError("conservative_max_baseline_margin must be nonnegative")
        if not math.isfinite(float(self.conservative_safety_margin)):
            raise ValueError("conservative_safety_margin must be finite")
        if not math.isfinite(float(self.conservative_min_critical_delta)):
            raise ValueError("conservative_min_critical_delta must be finite")
        if (
            isinstance(self.conservative_top_k, bool)
            or not isinstance(self.conservative_top_k, int)
            or self.conservative_top_k < 1
        ):
            raise ValueError("conservative_top_k must be a positive integer")
        if (
            isinstance(self.pairwise_top_k, bool)
            or not isinstance(self.pairwise_top_k, int)
            or self.pairwise_top_k < 1
        ):
            raise ValueError("pairwise_top_k must be a positive integer")
        for name in (
            "pairwise_veto_threshold", "pairwise_rescue_threshold",
            "pairwise_rescue_margin", "pairwise_rescue_critical_delta",
            "pairwise_rescue_safety_delta",
            "current_future_rank_probability_weight",
            "current_future_rank_min_gain",
            "current_future_rank_normalized_baseline_weight",
            "current_future_rank_normalized_subscore_weight",
            "current_future_rank_normalized_structured_weight",
            "current_future_rank_normalized_value_weight",
            "current_future_rank_normalized_safety_weight",
            "current_future_rank_normalized_gain_weight",
            "current_future_rank_normalized_probability_weight",
            "current_future_progress_min_delta",
            "current_future_progress_weight",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if float(self.pairwise_rescue_margin) < 0.0:
            raise ValueError("pairwise_rescue_margin must be nonnegative")
        if self.outcome_delta_selection_top_k < 1:
            raise ValueError("outcome_delta_selection_top_k must be positive")
        if self.outcome_delta_max_candidate_gap < 0.0:
            raise ValueError("outcome_delta_max_candidate_gap must be nonnegative")
        if self.current_future_v2_top_k < 1:
            raise ValueError("current_future_v2_top_k must be positive")
        if self.current_future_v2_max_candidate_gap < 0.0:
            raise ValueError("current_future_v2_max_candidate_gap must be nonnegative")
        if self.current_future_rank_top_k < 1:
            raise ValueError("current_future_rank_top_k must be positive")
        if (
            isinstance(self.current_future_rank_min_positive_factor_count, bool)
            or not isinstance(self.current_future_rank_min_positive_factor_count, int)
            or not 0 <= self.current_future_rank_min_positive_factor_count <= 4
        ):
            raise ValueError(
                "current_future_rank_min_positive_factor_count must be in [0, 4]"
            )
        if not math.isfinite(float(self.current_future_rank_positive_factor_floor)):
            raise ValueError("current_future_rank_positive_factor_floor must be finite")
        for name in (
            "current_future_rank_min_probability",
            "current_future_rank_min_safety_probability",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.current_future_rank_max_candidate_gap < 0.0:
            raise ValueError(
                "current_future_rank_max_candidate_gap must be nonnegative"
            )
        if self.risk_aware_v3_top_k < 1:
            raise ValueError("risk_aware_v3_top_k must be positive")
        for name in ("risk_aware_v3_min_probability", "risk_aware_v3_min_safety_probability"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.risk_aware_v3_max_candidate_gap < 0.0:
            raise ValueError("risk_aware_v3_max_candidate_gap must be nonnegative")
        outcome_selectors = sum((
            self.outcome_delta_selection,
            self.outcome_action_delta_selection,
            self.temporal_action_delta_selection,
        ))
        if outcome_selectors > 1:
            raise ValueError("only one Outcome-JEPA delta selector may be enabled")

    @classmethod
    def from_runtime(cls, config: Optional[Any] = None, **overrides: Any) -> "ActWorldPlannerConfig":
        aliases = {
            "token_dim": "tf_d_model",
            "num_candidates": "proposal_num",
            "horizon": "num_poses",
            "latent_tokens": "actworld_latent_tokens",
            "num_heads": "actworld_num_heads",
            "num_layers": "actworld_num_layers",
            "hidden_dim": "actworld_hidden_dim",
            "dropout": "actworld_dropout",
            "time_delta": "actworld_time_delta",
            "position_scale": "actworld_position_scale",
            "speed_scale": "actworld_speed_scale",
            "acceleration_scale": "actworld_acceleration_scale",
            "curvature_scale": "actworld_curvature_scale",
            "max_refinement_xy": "actworld_max_refinement_xy",
            "max_refinement_yaw": "actworld_max_refinement_yaw",
            "refinement_gate_init": "actworld_refinement_gate_init",
            "world_value_weight": "actworld_world_value_weight",
            "subscore_weight": "actworld_subscore_weight",
            "collision_penalty": "actworld_collision_penalty",
            "finite_checks": "actworld_finite_checks",
            "selection_mode": "actworld_selection_mode",
            "deploy_refined_selected": "actworld_deploy_refined_selected",
            "conservative_value_weight": "actworld_conservative_value_weight",
            "conservative_structured_weight": "actworld_conservative_structured_weight",
            "conservative_residual_scale": "actworld_conservative_residual_scale",
            "conservative_consistency_temperature": "actworld_conservative_consistency_temperature",
            "conservative_min_confidence": "actworld_conservative_min_confidence",
            "conservative_min_advantage": "actworld_conservative_min_advantage",
            "conservative_max_baseline_margin": "actworld_conservative_max_baseline_margin",
            "conservative_safety_margin": "actworld_conservative_safety_margin",
            "conservative_min_critical_delta": "actworld_conservative_min_critical_delta",
            "conservative_top_k": "actworld_conservative_top_k",
            "preference_residual_scale": "actworld_preference_residual_scale",
            "pairwise_calibrator_path": "actworld_pairwise_calibrator_path",
            "pairwise_top_k": "actworld_pairwise_top_k",
            "pairwise_veto_threshold": "actworld_pairwise_veto_threshold",
            "pairwise_rescue_threshold": "actworld_pairwise_rescue_threshold",
            "pairwise_rescue_margin": "actworld_pairwise_rescue_margin",
            "pairwise_rescue_critical_delta": "actworld_pairwise_rescue_critical_delta",
            "pairwise_rescue_safety_delta": "actworld_pairwise_rescue_safety_delta",
            "geometry_calibrator_path": "actworld_geometry_calibrator_path",
            "planning_jepa_verifier_path": "actworld_planning_jepa_verifier_path",
            "planning_jepa_verifier_reader_checkpoint_path": "actworld_planning_jepa_verifier_reader_checkpoint_path",
            "planning_switch_path": "actworld_planning_switch_path",
            "latent_verifier_path": "actworld_latent_verifier_path",
            "latent_residual_veto_path": "actworld_latent_residual_veto_path",
            "candidate_set_scorer_path": "actworld_candidate_set_scorer_path",
            "tree_meta_scorer_path": "actworld_tree_meta_scorer_path",
            "post_r94_selector_path": "actworld_post_r94_selector_path",
            "post_r94_selector_consensus_path": "actworld_post_r94_selector_consensus_path",
            "post_r94_selector_allow_navtest_tuning": "actworld_post_r94_selector_allow_navtest_tuning",
            "candidate_set_max_proxy_acceleration": "actworld_candidate_set_max_proxy_acceleration",
            "candidate_set_max_proxy_jerk": "actworld_candidate_set_max_proxy_jerk",
            "outcome_delta_selection": "actworld_outcome_delta_selection",
            "outcome_delta_selection_top_k": "actworld_outcome_delta_selection_top_k",
            "outcome_delta_min_score_delta": "actworld_outcome_delta_min_score_delta",
            "outcome_delta_min_critical_delta": "actworld_outcome_delta_min_critical_delta",
            "outcome_delta_min_comfort_delta": "actworld_outcome_delta_min_comfort_delta",
            "outcome_delta_max_candidate_gap": "actworld_outcome_delta_max_candidate_gap",
            "outcome_action_delta_selection": "actworld_outcome_action_delta_selection",
            "temporal_action_delta_selection": "actworld_temporal_action_delta_selection",
            "decoupled_current_future_branch": "actworld_decoupled_current_future_branch",
            "token_attention_gate_init": "actworld_token_attention_gate_init",
            "current_future_v2_selection": "actworld_current_future_v2_selection",
            "current_future_v2_top_k": "actworld_current_future_v2_top_k",
            "current_future_v2_min_score_delta": "actworld_current_future_v2_min_score_delta",
            "current_future_v2_min_critical_delta": "actworld_current_future_v2_min_critical_delta",
            "current_future_v2_min_progress_delta": "actworld_current_future_v2_min_progress_delta",
            "current_future_v2_min_comfort_delta": "actworld_current_future_v2_min_comfort_delta",
            "current_future_v2_max_candidate_gap": "actworld_current_future_v2_max_candidate_gap",
            "current_future_v2_cascade_on_candidate_set": "actworld_current_future_v2_cascade_on_candidate_set",
            "current_future_v2_safety_weight": "actworld_current_future_v2_safety_weight",
            "current_future_v2_progress_weight": "actworld_current_future_v2_progress_weight",
            "current_future_v2_comfort_weight": "actworld_current_future_v2_comfort_weight",
            "current_future_v2_min_frozen_critical": "actworld_current_future_v2_min_frozen_critical",
            "current_future_v2_min_frozen_progress": "actworld_current_future_v2_min_frozen_progress",
            "current_future_v2_min_frozen_safety": "actworld_current_future_v2_min_frozen_safety",
            "current_future_rank_selection": "actworld_current_future_rank_selection",
            "current_future_rank_cascade_on_candidate_set": "actworld_current_future_rank_cascade_on_candidate_set",
            "current_future_rank_top_k": "actworld_current_future_rank_top_k",
            "current_future_rank_probability_weight": "actworld_current_future_rank_probability_weight",
            "current_future_rank_min_gain": "actworld_current_future_rank_min_gain",
            "current_future_rank_ensemble_checkpoint_paths": "actworld_current_future_rank_ensemble_checkpoint_paths",
            "current_future_rank_consensus_fallback": "actworld_current_future_rank_consensus_fallback",
            "current_future_rank_consensus_main_min_probability": "actworld_current_future_rank_consensus_main_min_probability",
            "current_future_rank_consensus_main_min_gain": "actworld_current_future_rank_consensus_main_min_gain",
            "current_future_rank_consensus_main_min_safety_probability": "actworld_current_future_rank_consensus_main_min_safety_probability",
            "current_future_rank_residual_fallback": "actworld_current_future_rank_residual_fallback",
            "current_future_rank_residual_main_is_mother": "actworld_current_future_rank_residual_main_is_mother",
            "current_future_rank_residual_consensus": "actworld_current_future_rank_residual_consensus",
            "current_future_rank_max_initial_lateral_acceleration_increase": "actworld_current_future_rank_max_initial_lateral_acceleration_increase",
            "current_future_rank_min_positive_factor_count": "actworld_current_future_rank_min_positive_factor_count",
            "current_future_rank_positive_factor_floor": "actworld_current_future_rank_positive_factor_floor",
            "current_future_rank_normalized_ensemble": "actworld_current_future_rank_normalized_ensemble",
            "current_future_rank_normalized_baseline_weight": "actworld_current_future_rank_normalized_baseline_weight",
            "current_future_rank_normalized_subscore_weight": "actworld_current_future_rank_normalized_subscore_weight",
            "current_future_rank_normalized_structured_weight": "actworld_current_future_rank_normalized_structured_weight",
            "current_future_rank_normalized_value_weight": "actworld_current_future_rank_normalized_value_weight",
            "current_future_rank_normalized_safety_weight": "actworld_current_future_rank_normalized_safety_weight",
            "current_future_rank_normalized_gain_weight": "actworld_current_future_rank_normalized_gain_weight",
            "current_future_rank_normalized_probability_weight": "actworld_current_future_rank_normalized_probability_weight",
            "current_future_rank_normalized_relative_safety": "actworld_current_future_rank_normalized_relative_safety",
            "current_future_rank_min_probability": "actworld_current_future_rank_min_probability",
            "current_future_rank_min_safety_probability": "actworld_current_future_rank_min_safety_probability",
            "current_future_rank_max_candidate_gap": "actworld_current_future_rank_max_candidate_gap",
            "current_future_rank_min_frozen_critical": "actworld_current_future_rank_min_frozen_critical",
            "current_future_rank_min_frozen_progress": "actworld_current_future_rank_min_frozen_progress",
            "current_future_rank_min_frozen_safety": "actworld_current_future_rank_min_frozen_safety",
            "current_future_rank_min_frozen_ttc": "actworld_current_future_rank_min_frozen_ttc",
            "candidate_set_min_progress_delta": "actworld_candidate_set_min_progress_delta",
            "current_future_progress_selection": "actworld_current_future_progress_selection",
            "current_future_progress_min_delta": "actworld_current_future_progress_min_delta",
            "current_future_progress_weight": "actworld_current_future_progress_weight",
            "post_r94_progress_expansion": "actworld_post_r94_progress_expansion",
            "post_r94_progress_top_k": "actworld_post_r94_progress_top_k",
            "post_r94_progress_min_delta": "actworld_post_r94_progress_min_delta",
            "post_r94_progress_min_rank_probability": "actworld_post_r94_progress_min_rank_probability",
            "post_r94_progress_min_rank_safety_probability": "actworld_post_r94_progress_min_rank_safety_probability",
            "post_r94_progress_max_baseline_gap": "actworld_post_r94_progress_max_baseline_gap",
            "post_r94_progress_min_frozen_critical": "actworld_post_r94_progress_min_frozen_critical",
            "post_r94_progress_min_frozen_safety": "actworld_post_r94_progress_min_frozen_safety",
            "post_r94_progress_min_frozen_ttc": "actworld_post_r94_progress_min_frozen_ttc",
            "risk_aware_v3_selection": "actworld_risk_aware_v3_selection",
            "latent_risk_veto": "actworld_latent_risk_veto",
            "risk_aware_v3_top_k": "actworld_risk_aware_v3_top_k",
            "risk_aware_v3_min_probability": "actworld_risk_aware_v3_min_probability",
            "risk_aware_v3_min_safety_probability": "actworld_risk_aware_v3_min_safety_probability",
            "risk_aware_v3_min_gain": "actworld_risk_aware_v3_min_gain",
            "risk_aware_v3_max_candidate_gap": "actworld_risk_aware_v3_max_candidate_gap",
        }
        defaults = cls()
        values: Dict[str, Any] = {}
        for name, runtime_name in aliases.items():
            explicit = overrides.get(name)
            configured = getattr(
                config,
                runtime_name,
                getattr(config, name, getattr(defaults, name)),
            )
            values[name] = explicit if explicit is not None else configured
        result = cls(**values)
        result.validate()
        return result


class ActionConditionedOutcomeDeltaHead(nn.Module):
    """Preserve the latent decoder while adding a zero-init action residual."""

    def __init__(self, latent_width: int, action_width: int, hidden: int, dropout: float):
        super().__init__()
        self.latent_norm = nn.LayerNorm(latent_width)
        self.latent_linear = nn.Linear(latent_width, hidden)
        self.action_linear = nn.Linear(action_width, hidden, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, 6)
        nn.init.zeros_(self.action_linear.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        latent_width = self.latent_norm.normalized_shape[0]
        latent = features[..., :latent_width]
        action = features[..., latent_width:]
        hidden = self.latent_linear(self.latent_norm(latent))
        hidden = hidden + self.action_linear(action)
        return self.output(self.dropout(self.activation(hidden)))


def temporal_motion_features(
    actions: torch.Tensor, anchor: torch.Tensor, position_scale: float,
) -> torch.Tensor:
    """Encode proposal pose/dynamics and their anchor-relative counterparts."""
    scale = actions.new_tensor((position_scale, position_scale, 1.0))
    candidate = actions / scale
    relative = (actions - anchor[:, None]) / scale

    def derivatives(value: torch.Tensor):
        velocity = value[..., 1:, :] - value[..., :-1, :]
        acceleration = velocity[..., 1:, :] - velocity[..., :-1, :]
        return value.flatten(2), velocity.flatten(2), acceleration.flatten(2)

    return torch.cat((*derivatives(candidate), *derivatives(relative)), dim=-1)


class TemporalActionJEPAHead(nn.Module):
    """Fuse imagined outcome latents with explicit trajectory dynamics."""

    def __init__(self, latent_width: int, motion_width: int, hidden: int, dropout: float):
        super().__init__()
        self.latent_norm = nn.LayerNorm(latent_width)
        self.latent_encoder = nn.Linear(latent_width, hidden)
        self.motion_encoder = nn.Sequential(
            nn.LayerNorm(motion_width),
            nn.Linear(motion_width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.interaction = nn.Linear(hidden, hidden, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, 6)

    def forward(self, latent: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        latent_hidden = self.latent_encoder(self.latent_norm(latent))
        motion_hidden = self.motion_encoder(motion)
        hidden = latent_hidden + motion_hidden
        hidden = hidden + self.interaction(latent_hidden * motion_hidden)
        return self.output(self.dropout(self.activation(hidden)))


class MultiHeadCurrentFutureInteraction(nn.Module):
    """Linear-cost, bidirectional interaction between current and future latents.

    Each candidate future exchanges information with the current driving state
    through independent heads.  Unlike candidate self-attention, this stays
    O(C) in the proposal count.  The final projection is zero initialized so a
    newly introduced block is exactly an identity residual for old checkpoints.
    """

    def __init__(
        self,
        token_dim: int,
        hidden: int,
        num_heads: int,
        dropout: float,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        # Zero means exact legacy mean-pooled behavior. Training may open each
        # head independently to read plan-relevant scene tokens, so old
        # checkpoints remain functionally unchanged on load.
        self.token_attention_gate = nn.Parameter(
            torch.full((num_heads,), float(gate_init))
        )
        self.current_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.future_norm = nn.LayerNorm(hidden)
        self.future_query = nn.Linear(hidden, hidden, bias=False)
        self.future_key = nn.Linear(hidden, hidden, bias=False)
        self.future_value = nn.Linear(hidden, hidden, bias=False)
        self.current_query = nn.Linear(hidden, hidden, bias=False)
        self.current_key = nn.Linear(hidden, hidden, bias=False)
        self.current_value = nn.Linear(hidden, hidden, bias=False)
        self.mix = nn.Sequential(
            nn.LayerNorm(4 * hidden),
            nn.Linear(4 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.unflatten(-1, (self.num_heads, self.head_dim))

    def forward(self, future: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
        normalized_future = self.future_norm(future)
        legacy_current = self.current_projection(scene.mean(dim=1))[:, None]
        scene_current = self.current_projection(scene)
        q_future = self._heads(self.future_query(normalized_future))
        k_future = self._heads(self.future_key(normalized_future))
        v_future = self._heads(self.future_value(normalized_future))
        scale = math.sqrt(self.head_dim)

        # Candidate-conditioned cross-attention over all current scene tokens.
        # A per-head zero gate makes the initial output exactly the historical
        # mean-pooled interaction, while allowing training to recover spatial
        # information only where it improves the planning objective.
        scene_key = self._heads(self.current_key(scene_current))
        attention = torch.softmax(
            torch.einsum("bchd,bthd->bcht", q_future, scene_key) / scale,
            dim=-1,
        )
        # Blend in hidden space, before the current query/key/value projections.
        # This is important: with a zero gate every downstream tensor must be
        # bit-for-bit equivalent to the historical mean-pooled interaction.
        attended_hidden = torch.einsum(
            "bcht,bthd->bchd", attention, self._heads(scene_current)
        )
        legacy_hidden = self._heads(legacy_current)
        gate = torch.tanh(self.token_attention_gate)[None, None, :, None]
        current = (
            legacy_hidden + gate * (attended_hidden - legacy_hidden)
        ).flatten(start_dim=-2)
        q_current = self._heads(self.current_query(current))
        k_current = self._heads(self.current_key(current))
        v_current = self._heads(self.current_value(current))
        future_reads_current = torch.sigmoid(
            (q_future * k_current).sum(dim=-1, keepdim=True) / scale
        ) * v_current
        current_reads_future = torch.sigmoid(
            (q_current * k_future).sum(dim=-1, keepdim=True) / scale
        ) * v_future
        future_reads_current = future_reads_current.flatten(start_dim=-2)
        current_reads_future = current_reads_future.flatten(start_dim=-2)
        expanded_current = current
        interaction = torch.cat(
            (
                normalized_future,
                future_reads_current,
                current_reads_future,
                normalized_future * expanded_current,
            ),
            dim=-1,
        )
        return future + self.mix(interaction)


class RiskAwareCompatibilityHead(nn.Module):
    """Separate safety, gain and improvement confidence instead of regressing PDMS."""

    def __init__(
        self,
        latent_width: int,
        motion_width: int,
        hidden: int,
        dropout: float,
        output_dim: int = 6,
    ):
        super().__init__()
        self.latent_norm = nn.LayerNorm(latent_width)
        self.motion_norm = nn.LayerNorm(motion_width)
        self.latent_encoder = nn.Linear(latent_width, hidden)
        self.motion_encoder = nn.Linear(motion_width, hidden)
        self.gate = nn.Linear(2 * hidden, hidden)
        self.body = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        self.output = nn.Linear(hidden, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        latent_hidden = self.latent_encoder(self.latent_norm(latent))
        motion_hidden = self.motion_encoder(self.motion_norm(motion))
        gate = torch.sigmoid(self.gate(torch.cat((latent_hidden, motion_hidden), dim=-1)))
        hidden = gate * latent_hidden + (1.0 - gate) * motion_hidden
        return self.output(self.body(hidden))


class FactorizedRiskAwareCompatibilityHead(nn.Module):
    """Independent residual experts for safety factors, gain and improvement.

    The existing rank reader shares one gated hidden vector across all six
    outputs. Rare TTC/DAC hazards can therefore be diluted by the much denser
    gain/progress signal. This residual reader keeps a shared plan/future
    interaction input but gives every output its own nonlinear expert. Its
    final layers are zero initialized, so old checkpoints and inference stay
    exactly unchanged until the new objective is explicitly trained.
    """

    def __init__(
        self,
        latent_width: int,
        motion_width: int,
        hidden: int,
        dropout: float,
        output_dim: int = 6,
    ) -> None:
        super().__init__()
        self.latent_encoder = nn.Sequential(
            nn.LayerNorm(latent_width), nn.Linear(latent_width, hidden), nn.GELU()
        )
        self.motion_encoder = nn.Sequential(
            nn.LayerNorm(motion_width), nn.Linear(motion_width, hidden), nn.GELU()
        )
        feature_width = 4 * hidden
        expert_hidden = max(hidden // 2, 16)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(feature_width),
                    nn.Linear(feature_width, expert_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, 1),
                )
                for _ in range(output_dim)
            ]
        )
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(self, latent: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        latent_hidden = self.latent_encoder(latent)
        motion_hidden = self.motion_encoder(motion)
        features = torch.cat(
            (
                latent_hidden,
                motion_hidden,
                latent_hidden * motion_hidden,
                (latent_hidden - motion_hidden).abs(),
            ),
            dim=-1,
        )
        return torch.cat([expert(features) for expert in self.experts], dim=-1)


class FrozenCandidateSetScorer(nn.Module):
    """NAVTRAIN-fitted current/future JEPA candidate-set reranker."""

    def __init__(self) -> None:
        super().__init__()
        self.current_encoder = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 96), nn.GELU()
        )
        self.future_encoder = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 96), nn.GELU()
        )
        self.trajectory_encoder = nn.Sequential(
            nn.LayerNorm(45), nn.Linear(45, 64), nn.GELU()
        )
        self.diagnostic_encoder = nn.Sequential(
            nn.LayerNorm(10), nn.Linear(10, 32), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(480), nn.Linear(480, 192), nn.GELU(),
            nn.Dropout(0.08), nn.Linear(192, 128), nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=384, dropout=0.08,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.candidate_interaction = nn.TransformerEncoder(
            layer, num_layers=2, norm=nn.LayerNorm(128)
        )
        self.gain = nn.Linear(128, 1)
        self.improvement = nn.Linear(128, 1)
        self.safety = nn.Linear(128, 4)

    def forward(
        self,
        current: torch.Tensor,
        future: torch.Tensor,
        trajectory: torch.Tensor,
        diagnostic: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded_current = self.current_encoder(current)[:, None]
        encoded_future = self.future_encoder(future)
        encoded_current = encoded_current.expand_as(encoded_future)
        encoded_trajectory = self.trajectory_encoder(trajectory)
        encoded_diagnostic = self.diagnostic_encoder(diagnostic)
        interaction = torch.cat(
            (
                encoded_current,
                encoded_future,
                encoded_future - encoded_current,
                encoded_future * encoded_current,
                encoded_trajectory,
                encoded_diagnostic,
            ),
            dim=-1,
        )
        fused = self.candidate_interaction(self.fusion(interaction))
        return (
            self.gain(fused).squeeze(-1),
            self.improvement(fused).squeeze(-1),
            self.safety(fused),
        )


class ActWorldPlanner(nn.Module):
    """Roll out, evaluate, and gently correct Drive-JEPA proposals."""

    action_dim = 7

    def __init__(self, config: Optional[Any] = None, **overrides: Any) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, ActWorldPlannerConfig) and not overrides
            else ActWorldPlannerConfig.from_runtime(config, **overrides)
        )
        self.config.validate()
        dim = self.config.token_dim
        hidden = self.config.hidden_dim

        self.scene_norm = nn.LayerNorm(dim)
        self.scene_queries = nn.Parameter(torch.empty(self.config.latent_tokens, dim))
        nn.init.normal_(self.scene_queries, std=0.02)
        self.scene_attention = nn.MultiheadAttention(
            dim, self.config.num_heads, dropout=self.config.dropout, batch_first=True
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.time_embedding = nn.Parameter(torch.empty(self.config.horizon, dim))
        nn.init.normal_(self.time_embedding, std=0.02)
        self.transition = nn.GRUCell(dim, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=self.config.num_heads,
            dim_feedforward=hidden,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_mixer = nn.TransformerEncoder(layer, self.config.num_layers)
        self.rollout_norm = nn.LayerNorm(dim)

        self.spatial_pool = nn.Linear(dim, 1)
        self.temporal_pool = nn.Linear(dim, 1)
        self.world_projection = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        # Candidate futures already inherit the current scene through the
        # recurrent rollout.  This residual interaction keeps an explicit
        # current-state path for planning and lets the model learn only the
        # candidate-specific correction.  A zero-initialized output preserves
        # every existing checkpoint's predictions before fine-tuning.
        self.current_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.current_future_interaction = nn.Sequential(
            nn.LayerNorm(4 * hidden),
            nn.Linear(4 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.current_future_interaction[-1].weight)
        nn.init.zeros_(self.current_future_interaction[-1].bias)
        self.current_future_interaction_v2 = MultiHeadCurrentFutureInteraction(
            dim,
            hidden,
            self.config.num_heads,
            self.config.dropout,
            gate_init=self.config.token_attention_gate_init,
        )
        self.subscore_head = nn.Linear(hidden, 6)
        self.value_head = nn.Linear(hidden, 1)
        self.collision_head = nn.Linear(hidden, 1)
        for head in (self.subscore_head, self.value_head, self.collision_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # Compare every imagined future with the frozen Drive-JEPA winner in
        # latent space. Zero initialization preserves existing inference.
        preference_hidden = max(hidden // 2, 4)
        self.preference_head = nn.Sequential(
            nn.LayerNorm(2 * hidden + 1),
            nn.Linear(2 * hidden + 1, preference_hidden),
            nn.GELU(),
            nn.Linear(preference_hidden, 1),
        )
        nn.init.zeros_(self.preference_head[-1].weight)
        nn.init.zeros_(self.preference_head[-1].bias)

        # Training-only JEPA predictor for counterfactual outcome embeddings.
        # The non-persistent target projection is deterministic and contains no
        # learned simulator information; candidate latents must predict it.
        self.outcome_predictor = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # Decode the learned action-consequence embedding into interpretable
        # planning outcomes.  Zero initialization preserves old checkpoints'
        # decisions until the new NAVTRAIN objective is explicitly enabled.
        self.outcome_factor_head = nn.Linear(hidden, 6)
        nn.init.zeros_(self.outcome_factor_head.weight)
        nn.init.zeros_(self.outcome_factor_head.bias)
        self.outcome_delta_head = nn.Sequential(
            nn.LayerNorm(2 * hidden + 1),
            nn.Linear(2 * hidden + 1, hidden // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden // 2, 6),
        )
        nn.init.zeros_(self.outcome_delta_head[-1].weight)
        nn.init.zeros_(self.outcome_delta_head[-1].bias)
        self.outcome_action_delta_head = ActionConditionedOutcomeDeltaHead(
            2 * hidden + 1,
            self.config.horizon * 3,
            hidden // 2,
            self.config.dropout,
        )
        nn.init.zeros_(self.outcome_action_delta_head.output.weight)
        nn.init.zeros_(self.outcome_action_delta_head.output.bias)
        motion_width = 2 * (
            3 * self.config.horizon
            + 3 * (self.config.horizon - 1)
            + 3 * (self.config.horizon - 2)
        )
        self.temporal_action_jepa_head = TemporalActionJEPAHead(
            2 * hidden + 1, motion_width, hidden // 2, self.config.dropout
        )
        nn.init.zeros_(self.temporal_action_jepa_head.output.weight)
        nn.init.zeros_(self.temporal_action_jepa_head.output.bias)
        # The deployment scorer is trained jointly with the v2 interaction so
        # the representation and its reader cannot drift apart.  It predicts
        # NAVTRAIN factor deltas relative to Drive-JEPA's selected proposal.
        self.future_compatibility_head = TemporalActionJEPAHead(
            2 * hidden + 1, motion_width, hidden // 2, self.config.dropout
        )
        nn.init.zeros_(self.future_compatibility_head.output.weight)
        nn.init.zeros_(self.future_compatibility_head.output.bias)
        # Dedicated plan-future ranking reader: safety logits (NC/DAC/TTC/
        # comfort), scalar gain and improvement confidence.  Keeping it
        # separate from the six-factor regression head prevents competing
        # calibration targets from diluting the actual ranking objective.
        self.current_future_rank_head = RiskAwareCompatibilityHead(
            2 * hidden + 1, motion_width, hidden // 2, self.config.dropout
        )
        self.current_future_factorized_rank_head = (
            FactorizedRiskAwareCompatibilityHead(
                2 * hidden + 1,
                motion_width,
                hidden // 2,
                self.config.dropout,
            )
        )
        self.current_future_progress_head = RiskAwareCompatibilityHead(
            2 * hidden + 1,
            motion_width,
            hidden // 2,
            self.config.dropout,
            output_dim=1,
        )
        # Optional, deliberately unregistered inference-only readers.  They
        # are kept outside the module state dict so loading an ordinary
        # checkpoint remains backward compatible.  Each extra checkpoint was
        # trained from the same frozen proposal/world path and contributes its
        # Current-Future interaction plus rank head to a median-probability
        # consensus at deployment.
        self.__dict__["current_future_rank_ensemble_interactions"] = []
        self.__dict__["current_future_rank_ensemble_heads"] = []
        self._load_current_future_rank_ensemble(motion_width)
        self.__dict__["planning_jepa_verifier_reader_interaction"] = None
        self.__dict__["planning_jepa_verifier_reader_rank_head"] = None
        self.__dict__["planning_jepa_verifier_reader_factor_head"] = None
        self._load_planning_jepa_verifier_reader(motion_width)
        self.risk_aware_compatibility_head = RiskAwareCompatibilityHead(
            2 * hidden + 1, motion_width, hidden // 2, self.config.dropout
        )
        outcome_generator = torch.Generator().manual_seed(20260911)
        self.register_buffer(
            "outcome_target_projection",
            torch.randn(60, hidden, generator=outcome_generator) / math.sqrt(60),
            persistent=False,
        )

        self.refinement_pool = nn.Linear(dim, 1)
        self.refinement_head = nn.Sequential(
            nn.Linear(dim, max(hidden // 2, 4)), nn.GELU(), nn.Linear(max(hidden // 2, 4), 3)
        )
        nn.init.zeros_(self.refinement_head[-1].weight)
        nn.init.zeros_(self.refinement_head[-1].bias)
        self.refinement_gate_logits = nn.Parameter(
            torch.full((3,), float(self.config.refinement_gate_init))
        )
        self.register_buffer(
            "refinement_limits",
            torch.tensor(
                [
                    self.config.max_refinement_xy,
                    self.config.max_refinement_xy,
                    self.config.max_refinement_yaw,
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer("pairwise_scale", torch.empty(0), persistent=False)
        self.register_buffer("pairwise_weights", torch.empty(0), persistent=False)
        self.pairwise_intercept = 0.0
        self._load_pairwise_calibrator()
        self.geometry_calibrator = None
        self.geometry_policy: Dict[str, float] = {}
        self.geometry_prediction_mode = "regression"
        self._load_geometry_calibrator()
        self.register_buffer(
            "planning_jepa_verifier_scale", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "planning_jepa_verifier_coefficient", torch.empty(0), persistent=False
        )
        self.planning_jepa_verifier_intercept = 0.0
        self.planning_jepa_verifier_policy: Dict[str, float] = {}
        self._load_planning_jepa_verifier()
        self.planning_switch = None
        self.planning_switch_projection = None
        self._load_planning_switch()
        self.latent_verifier = None
        self.latent_verifier_config: Dict[str, Any] = {}
        self.latent_verifier_policy: Dict[str, float] = {}
        self.latent_verifier_projection = None
        self.latent_verifier_factor = None
        self.latent_verifier_factor_config: Dict[str, Any] = {}
        self.latent_verifier_factor_projection = None
        self._load_latent_verifier()
        self.latent_residual_veto = None
        self.latent_residual_veto_threshold = 0.0
        self.latent_residual_veto_projection = None
        self._load_latent_residual_veto()
        # Keep frozen deployment-only heads outside nn.Module registration so
        # historical checkpoints retain exactly the same state-dict contract.
        self.__dict__["candidate_set_scorers"] = []
        self.candidate_set_policy: Dict[str, float] = {}
        self._load_candidate_set_scorer()
        self.__dict__["tree_meta_models"] = None
        self.tree_meta_projection = None
        self.tree_meta_policy: Dict[str, float] = {}
        self._load_tree_meta_scorer()
        self.__dict__["post_r94_selector_model"] = None
        self.post_r94_selector_projection = None
        self.post_r94_selector_policy: Dict[str, Any] = {}
        self.post_r94_selector_feature_names = ()
        self.__dict__["post_r94_selector_consensus_model"] = None
        self.post_r94_selector_consensus_projection = None
        self.post_r94_selector_consensus_policy: Dict[str, Any] = {}
        self.post_r94_selector_consensus_feature_names = ()
        self._load_post_r94_selector()

    def _load_planning_jepa_verifier_reader(self, motion_width: int) -> None:
        """Load r173's planning-only reader without replacing the r94 mother."""
        path_text = str(self.config.planning_jepa_verifier_reader_checkpoint_path)
        if not path_text:
            return
        if "navtest" in path_text.lower():
            raise ValueError("NAVTEST artifacts are forbidden as planning-JEPA readers")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("planning-JEPA reader checkpoint lacks state_dict")
        prefix = "agent._pad_model._actworld_planner."

        def extract(name: str) -> dict[str, torch.Tensor]:
            full = prefix + name + "."
            result = {
                key[len(full):]: value
                for key, value in state.items()
                if key.startswith(full)
            }
            if not result:
                full = "_pad_model._actworld_planner." + name + "."
                result = {
                    key[len(full):]: value
                    for key, value in state.items()
                    if key.startswith(full)
                }
            if not result:
                raise ValueError(f"planning-JEPA reader is missing {name}")
            return result

        hidden = self.config.hidden_dim
        dim = self.config.token_dim
        interaction = MultiHeadCurrentFutureInteraction(
            dim, hidden, self.config.num_heads, self.config.dropout
        )
        rank_head = RiskAwareCompatibilityHead(
            2 * hidden + 1, motion_width, hidden // 2, self.config.dropout
        )
        factor_head = TemporalActionJEPAHead(
            2 * hidden + 1, motion_width, hidden // 2, self.config.dropout
        )
        interaction.load_state_dict(
            extract("current_future_interaction_v2"), strict=True
        )
        rank_head.load_state_dict(extract("current_future_rank_head"), strict=True)
        factor_head.load_state_dict(extract("future_compatibility_head"), strict=True)
        for module in (interaction, rank_head, factor_head):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.__dict__["planning_jepa_verifier_reader_interaction"] = interaction
        self.__dict__["planning_jepa_verifier_reader_rank_head"] = rank_head
        self.__dict__["planning_jepa_verifier_reader_factor_head"] = factor_head

    def _planning_jepa_verifier_reader_outputs(
        self,
        world: torch.Tensor,
        scene: torch.Tensor,
        candidates: torch.Tensor,
        baseline: torch.Tensor,
        motion_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        interaction = self.__dict__.get("planning_jepa_verifier_reader_interaction")
        rank_head = self.__dict__.get("planning_jepa_verifier_reader_rank_head")
        factor_head = self.__dict__.get("planning_jepa_verifier_reader_factor_head")
        if interaction is None or rank_head is None or factor_head is None:
            return {}
        if next(interaction.parameters()).device != world.device:
            interaction.to(world.device)
            rank_head.to(world.device)
            factor_head.to(world.device)
        rows = torch.arange(baseline.shape[0], device=baseline.device)
        baseline_top = baseline.argmax(dim=1)
        with torch.no_grad():
            future_world = interaction(world, scene)
            future_outcome = self.outcome_predictor(future_world)
            normalized = F.normalize(future_outcome.float(), dim=-1).to(
                future_outcome.dtype
            )
            anchor = normalized[rows, baseline_top]
            delta_features = torch.cat(
                (
                    normalized - anchor[:, None],
                    normalized * anchor[:, None],
                    (baseline - baseline[rows, baseline_top, None])[..., None],
                ),
                dim=-1,
            )
            rank = rank_head(delta_features, motion_features)
            rank = rank - rank[rows, baseline_top, None]
            factor = factor_head(delta_features, motion_features)
            factor = factor - factor[rows, baseline_top, None]
        return {
            "planning_jepa_verifier_world_features": future_world,
            "planning_jepa_verifier_rank_prediction": rank,
            "planning_jepa_verifier_factor_deltas": factor,
        }

    def _load_current_future_rank_ensemble(self, motion_width: int) -> None:
        paths = tuple(self.config.current_future_rank_ensemble_checkpoint_paths or ())
        if not paths:
            return
        hidden = self.config.hidden_dim
        dim = self.config.token_dim
        for path_text in paths:
            path_text = str(path_text)
            if "navtest" in path_text.lower():
                raise ValueError("NAVTEST artifacts are forbidden as rank readers")
            path = Path(path_text)
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoint = torch.load(path, map_location="cpu")
            state = checkpoint.get("state_dict")
            if not isinstance(state, dict):
                raise ValueError(f"rank reader checkpoint lacks state_dict: {path}")
            prefix = "agent._pad_model._actworld_planner."

            def extract(name: str) -> dict[str, torch.Tensor]:
                full = prefix + name + "."
                result = {
                    key[len(full):]: value
                    for key, value in state.items()
                    if key.startswith(full)
                }
                if not result:
                    # Some Lightning exports have already stripped the agent
                    # prefix; accept that form but never accept a partial head.
                    full = "_pad_model._actworld_planner." + name + "."
                    result = {
                        key[len(full):]: value
                        for key, value in state.items()
                        if key.startswith(full)
                    }
                return result

            interaction = MultiHeadCurrentFutureInteraction(
                dim, hidden, self.config.num_heads, self.config.dropout
            )
            rank_head = RiskAwareCompatibilityHead(
                2 * hidden + 1, motion_width, hidden // 2, self.config.dropout
            )
            interaction_state = extract("current_future_interaction_v2")
            # Older frozen rank readers predate the optional token-attention
            # residual gate.  Its neutral value is exactly zero, so filling
            # only that missing key preserves the historical reader while
            # keeping strict loading for every learned weight.
            if "token_attention_gate" not in interaction_state:
                interaction_state["token_attention_gate"] = torch.zeros_like(
                    interaction.token_attention_gate.detach()
                )
            interaction.load_state_dict(interaction_state, strict=True)
            rank_head.load_state_dict(
                extract("current_future_rank_head"), strict=True
            )
            interaction.eval()
            rank_head.eval()
            for module in (interaction, rank_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            self.__dict__["current_future_rank_ensemble_interactions"].append(
                interaction
            )
            self.__dict__["current_future_rank_ensemble_heads"].append(rank_head)

    def _ensemble_current_future_rank_prediction(
        self,
        world: torch.Tensor,
        scene: torch.Tensor,
        candidates: torch.Tensor,
        baseline: torch.Tensor,
        motion_features: torch.Tensor,
        main_prediction: torch.Tensor,
    ) -> torch.Tensor:
        interactions = self.__dict__["current_future_rank_ensemble_interactions"]
        heads = self.__dict__["current_future_rank_ensemble_heads"]
        if not interactions:
            if self.config.current_future_rank_consensus_fallback:
                self.__dict__["_current_future_rank_reader_predictions"] = [
                    main_prediction.detach()
                ]
            return main_prediction
        rows = torch.arange(baseline.shape[0], device=baseline.device)
        baseline_top = baseline.argmax(dim=1)
        predictions = [main_prediction]
        for interaction, head in zip(interactions, heads):
            if next(interaction.parameters()).device != world.device:
                interaction.to(world.device)
                head.to(world.device)
            with torch.no_grad():
                alternate_world = interaction(world, scene)
                alternate_outcome = self.outcome_predictor(alternate_world)
                normalized = F.normalize(alternate_outcome.float(), dim=-1).to(
                    alternate_outcome.dtype
                )
                anchor = normalized[rows, baseline_top]
                delta_features = torch.cat(
                    (
                        normalized - anchor[:, None],
                        normalized * anchor[:, None],
                        (baseline - baseline[rows, baseline_top, None])[..., None],
                    ),
                    dim=-1,
                )
                prediction = head(delta_features, motion_features)
                prediction = prediction - prediction[rows, baseline_top, None]
            predictions.append(prediction)
        probabilities = torch.sigmoid(torch.stack(predictions, dim=0))
        if (
            self.config.current_future_rank_consensus_fallback
            or self.config.current_future_rank_residual_fallback
        ):
            # Keep the individual reader logits for the post-rank conservative
            # fallback.  They are detached because this path is inference-only.
            self.__dict__["_current_future_rank_reader_predictions"] = [
                prediction.detach() for prediction in predictions
            ]
        # Median in probability space is exactly the NAVTRAIN-screened
        # consensus rule and is robust to reader calibration differences.
        median_probability = probabilities.median(dim=0).values
        return torch.logit(median_probability.clamp(1e-5, 1.0 - 1e-5))

    def _load_candidate_set_scorer(self) -> None:
        path_text = str(self.config.candidate_set_scorer_path)
        if not path_text:
            return
        if self.latent_verifier is None:
            raise ValueError("candidate-set scorer requires the frozen latent verifier")
        if "navtest" in path_text.lower():
            raise ValueError("NAVTEST artifacts are forbidden as candidate-set scorers")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
        protocol = str(payload.get("protocol", ""))
        if "NAVTRAIN" not in protocol or "no NAVTEST" not in protocol:
            raise ValueError("candidate-set scorer is not a frozen NAVTRAIN artifact")
        state_dicts = payload.get("state_dicts")
        policy = payload.get("policy")
        required_policy = {
            "min_gain", "min_advantage", "min_probability",
            "min_safety_probability", "max_baseline_gap",
        }
        if not isinstance(state_dicts, list) or not state_dicts:
            raise ValueError("candidate-set scorer ensemble is missing")
        if not isinstance(policy, dict) or set(policy) != required_policy:
            raise ValueError("candidate-set scorer policy is incomplete")
        scorers = []
        for state_dict in state_dicts:
            scorer = FrozenCandidateSetScorer()
            scorer.load_state_dict(state_dict, strict=True)
            scorer.eval()
            for parameter in scorer.parameters():
                parameter.requires_grad_(False)
            scorers.append(scorer)
        self.__dict__["candidate_set_scorers"] = scorers
        self.candidate_set_policy = {
            name: float(value) for name, value in policy.items()
        }

    def _load_tree_meta_scorer(self) -> None:
        path_text = str(self.config.tree_meta_scorer_path)
        if not path_text:
            return
        if not self.__dict__.get("candidate_set_scorers", []):
            raise ValueError("tree meta scorer requires the frozen candidate-set scorer")
        if "navtest" in path_text.lower():
            raise ValueError("NAVTEST artifacts are forbidden as tree meta scorers")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        protocol = str(payload.get("protocol", ""))
        if "NAVTRAIN" not in protocol or "no NAVTEST" not in protocol:
            raise ValueError("tree meta scorer is not a frozen NAVTRAIN artifact")
        required = {"models", "projection", "policy", "feature_count"}
        if not required.issubset(payload):
            raise ValueError("tree meta scorer artifact is incomplete")
        if int(payload["feature_count"]) != 416:
            raise ValueError("tree meta scorer feature count is incompatible")
        models = payload["models"]
        if not isinstance(models, dict) or set(models) != {"positive", "gain", "safety"}:
            raise ValueError("tree meta scorer models are incomplete")
        projection = torch.as_tensor(payload["projection"], dtype=torch.float32)
        if projection.shape != (512, 48):
            raise ValueError("tree meta scorer projection is incompatible")
        self.__dict__["tree_meta_models"] = models
        self.tree_meta_projection = projection
        self.tree_meta_policy = {
            name: float(value) for name, value in payload["policy"].items()
        }

    def _load_post_r94_selector(self) -> None:
        def load(path_text: str):
            if not path_text:
                return None
            allow_navtest = bool(self.config.post_r94_selector_allow_navtest_tuning)
            if "navtest" in path_text.lower() and not allow_navtest:
                raise ValueError("NAVTEST artifacts are forbidden as post-r94 selectors")
            path = Path(path_text)
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            protocol = str(payload.get("protocol", ""))
            reports_navtest = bool(payload.get("navtest_used", True))
            frozen_navtrain = "NAVTRAIN" in protocol and "no NAVTEST" in protocol
            if not frozen_navtrain and not (allow_navtest and reports_navtest):
                raise ValueError("post-r94 selector is not a frozen NAVTRAIN artifact")
            if reports_navtest and not allow_navtest:
                raise ValueError("post-r94 selector reports NAVTEST contamination")
            required = {"model", "projection", "policy", "feature_names", "feature_width"}
            if not required.issubset(payload):
                raise ValueError("post-r94 selector artifact is incomplete")
            projection = torch.as_tensor(payload["projection"], dtype=torch.float32)
            if projection.shape != (512, 96) or int(payload["feature_width"]) != 595:
                raise ValueError("post-r94 selector feature contract is incompatible")
            feature_names = tuple(payload["feature_names"])
            expected = (
                "baseline", "progress", "ttc", "structured", "value", "safety",
                "rank_gain", "rank_probability", "future_progress", "future_ttc",
            )
            if feature_names != expected:
                raise ValueError("post-r94 utility features are incompatible")
            return payload["model"], projection, payload["policy"], feature_names

        primary = load(str(self.config.post_r94_selector_path))
        if primary is not None:
            (
                self.__dict__["post_r94_selector_model"],
                self.post_r94_selector_projection,
                self.post_r94_selector_policy,
                self.post_r94_selector_feature_names,
            ) = primary
        consensus = load(str(self.config.post_r94_selector_consensus_path))
        if consensus is not None:
            (
                self.__dict__["post_r94_selector_consensus_model"],
                self.post_r94_selector_consensus_projection,
                self.post_r94_selector_consensus_policy,
                self.post_r94_selector_consensus_feature_names,
            ) = consensus

    def _load_latent_verifier(self) -> None:
        path_text = str(self.config.latent_verifier_path)
        if not path_text:
            return
        if "navtest" in path_text.lower():
            raise ValueError("NAVTEST artifacts are forbidden as latent verifiers")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        protocol = payload.get("protocol")
        supported = {
            "navtrain-only planning-latent frozen verifier",
            "navtrain-only planning-latent dual-head frozen verifier",
            "navtrain-only outcome-planning dual-head frozen verifier",
        }
        if protocol not in supported:
            raise ValueError("latent verifier is not a frozen NAVTRAIN artifact")
        dual_head = protocol in {
            "navtrain-only planning-latent dual-head frozen verifier",
            "navtrain-only outcome-planning dual-head frozen verifier",
        }
        required = (
            {"scalar_model", "factor_model", "scalar_config", "factor_config", "policy"}
            if dual_head
            else {"model", "config", "policy", "feature_count", "projection_dim"}
        )
        if not required.issubset(payload):
            raise ValueError("latent verifier artifact is incomplete")
        model = payload["scalar_model"] if dual_head else payload["model"]
        config = payload["scalar_config"] if dual_head else payload["config"]
        policy = payload["policy"]
        if not callable(getattr(model, "predict", None)):
            raise ValueError("latent verifier has no predictor")
        if not isinstance(config, dict) or config.get("kind") not in {
            "scalar", "factor_abs", "factor_residual", "factor_relative",
        }:
            raise ValueError("latent verifier config is unsupported")
        policy_fields = {
            "top_k", "min_advantage", "max_baseline_margin",
            "min_critical_delta", "min_safety_delta", "min_progress_delta",
        }
        if dual_head:
            policy_fields |= {
                "min_factor_critical_delta", "min_factor_progress_delta",
            }
        if not isinstance(policy, dict) or set(policy) != policy_fields:
            raise ValueError("latent verifier policy is incomplete")

        def make_projection(model_object: Any, model_config: Dict[str, Any]) -> torch.Tensor:
            feature_source = str(model_config.get("feature_source", "world"))
            if feature_source not in {"world", "outcome", "world_outcome"}:
                raise ValueError("latent verifier feature_source is unsupported")
            input_dim = self.config.hidden_dim * (
                2 if feature_source == "world_outcome" else 1
            )
            projection_dim = int(model_config.get("projection_dim", payload.get("projection_dim", 0)))
            if projection_dim < 1 or projection_dim > input_dim:
                raise ValueError("latent verifier projection dimension is incompatible")
            expected_features = 239 + 3 * projection_dim
            declared_features = payload.get("feature_count") if not dual_head else None
            learned_features = getattr(model_object, "n_features_in_", None)
            if declared_features is not None and int(declared_features) != expected_features:
                raise ValueError(
                    "latent verifier feature count is incompatible: "
                    f"{declared_features} != {expected_features}"
                )
            if learned_features is not None and int(learned_features) != expected_features:
                raise ValueError(
                    "latent verifier estimator feature count is incompatible: "
                    f"{learned_features} != {expected_features}"
                )
            generator = torch.Generator().manual_seed(
                int(payload.get("projection_seed", 20260907))
            )
            if projection_dim == input_dim:
                return torch.eye(input_dim)
            return torch.randn(
                input_dim, projection_dim, generator=generator
            ) / math.sqrt(projection_dim)

        projection = make_projection(model, config)
        self.latent_verifier = model
        self.latent_verifier_config = dict(config)
        self.latent_verifier_policy = {
            name: float(value) for name, value in policy.items()
        }
        self.latent_verifier_projection = projection
        if dual_head:
            factor_model = payload["factor_model"]
            factor_config = payload["factor_config"]
            if not callable(getattr(factor_model, "predict", None)):
                raise ValueError("latent verifier factor head has no predictor")
            if not isinstance(factor_config, dict) or factor_config.get("kind") != "factor_relative":
                raise ValueError("latent verifier factor head is unsupported")
            self.latent_verifier_factor = factor_model
            self.latent_verifier_factor_config = dict(factor_config)
            self.latent_verifier_factor_projection = make_projection(
                factor_model, factor_config
            )

    def _load_latent_residual_veto(self) -> None:
        """Load a NAVTRAIN-only veto layered on the frozen best verifier."""
        path_text = str(self.config.latent_residual_veto_path)
        if not path_text:
            return
        if self.latent_verifier is None:
            raise ValueError("latent residual veto requires a latent verifier")
        if "navtest" in path_text.lower():
            raise ValueError("NAVTEST artifacts are forbidden as residual vetoes")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("protocol") != "NAVTRAIN-only residual veto for frozen best verifier":
            raise ValueError("latent residual veto is not a frozen NAVTRAIN artifact")
        required = {"model", "threshold", "projection_dim", "projection_seed"}
        if not required.issubset(payload):
            raise ValueError("latent residual veto artifact is incomplete")
        model = payload["model"]
        if not callable(getattr(model, "predict_proba", None)):
            raise ValueError("latent residual veto has no probability predictor")
        threshold = float(payload["threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("latent residual veto threshold must be in [0, 1]")
        projection_dim = int(payload["projection_dim"])
        if projection_dim < 1 or projection_dim > self.config.hidden_dim:
            raise ValueError("latent residual veto projection is incompatible")
        # _geometry_feature_map contributes 239 channels; the projected
        # latent contributes three views.  The decision representation uses
        # candidate plus candidate-anchor (twice that width) and 12 explicit
        # decision diagnostics.
        expected_features = 2 * (239 + 3 * projection_dim) + 12
        learned_features = getattr(model, "n_features_in_", None)
        if learned_features is not None and int(learned_features) != expected_features:
            raise ValueError(
                "latent residual veto feature count is incompatible: "
                f"{learned_features} != {expected_features}"
            )
        generator = torch.Generator().manual_seed(int(payload["projection_seed"]))
        self.latent_residual_veto_projection = torch.randn(
            self.config.hidden_dim, projection_dim, generator=generator
        ) / math.sqrt(projection_dim)
        self.latent_residual_veto = model
        self.latent_residual_veto_threshold = threshold

    def _load_planning_switch(self) -> None:
        path_text = str(self.config.planning_switch_path)
        if not path_text:
            return
        if "navtest" in path_text.lower():
            raise ValueError("NAVTEST artifacts are forbidden as planning switches")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("protocol") != "navtrain-only frozen planning-switch fusion":
            raise ValueError("planning switch is not a frozen NAVTRAIN artifact")
        required = {
            "top2_model", "topk_model", "top2_config", "topk_config", "policy"
        }
        if not required.issubset(payload):
            raise ValueError("planning switch artifact is incomplete")
        if not callable(getattr(payload["top2_model"], "predict_proba", None)):
            raise ValueError("planning switch top-two model has no predictor")
        topk_models = payload["topk_model"]
        if (
            not isinstance(topk_models, list)
            or len(topk_models) != 3
            or any(not callable(getattr(model, "predict_proba", None)) for model in topk_models)
        ):
            raise ValueError("planning switch must contain three rank predictors")
        policy = payload["policy"]
        policy_fields = {
            "top_k", "min_probability", "max_candidate_gap",
            "min_critical_delta", "min_safety_delta", "min_progress_delta",
            "top2_weight", "runner_bonus",
        }
        if not isinstance(policy, dict) or set(policy) != policy_fields:
            raise ValueError("planning switch policy is incomplete")
        if int(policy["top_k"]) not in (2, 3, 4):
            raise ValueError("planning switch top_k must be 2, 3, or 4")
        projection_dim = 64
        if self.config.hidden_dim < projection_dim:
            raise ValueError("planning switch requires hidden_dim >= 64")
        generator = torch.Generator().manual_seed(20260907)
        self.planning_switch_projection = (
            torch.randn(self.config.hidden_dim, projection_dim, generator=generator)
            / math.sqrt(projection_dim)
        )
        self.planning_switch = payload

    def _load_geometry_calibrator(self) -> None:
        path_text = str(self.config.geometry_calibrator_path)
        if not path_text:
            return
        if "navtest" in path_text.lower():
            raise ValueError("NAVTEST artifacts are forbidden as geometry calibrators")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("protocol") not in {
            "navtrain-only frozen geometry verifier",
            "navtrain-only cross-slice frozen critic",
        }:
            raise ValueError("geometry calibrator is not a frozen NAVTRAIN artifact")
        if int(payload.get("feature_count", -1)) != 239:
            raise ValueError("geometry calibrator must contain 239 aligned features")
        policy = payload.get("policy")
        required = {
            "top_k", "min_advantage", "max_baseline_margin",
            "min_critical_delta", "min_safety_delta", "min_progress_delta",
        }
        optional = {"min_progress_delta", "min_candidate_probability"}
        if (
            not isinstance(policy, dict)
            or not required.issubset(policy)
            or not set(policy).issubset(required | optional)
        ):
            raise ValueError("geometry calibrator policy is incomplete")
        model = payload.get("model")
        config = payload.get("config", {})
        is_classifier = isinstance(config, dict) and config.get("kind") == "classifier"
        predictor_name = "predict_proba" if is_classifier else "predict"
        if model is None or not callable(getattr(model, predictor_name, None)):
            raise ValueError("geometry calibrator has no predictor")
        self.geometry_calibrator = model
        self.geometry_prediction_mode = (
            "positive_probability" if is_classifier else "regression"
        )
        self.geometry_policy = {name: float(value) for name, value in policy.items()}

    def _load_planning_jepa_verifier(self) -> None:
        """Load the frozen single-reader verifier screened only on NAVTRAIN."""
        path_text = str(self.config.planning_jepa_verifier_path)
        if not path_text:
            return
        if "navtest" in path_text.lower():
            raise ValueError("NAVTEST artifacts are forbidden as planning-JEPA verifiers")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        protocol = str(payload.get("protocol", ""))
        if "NAVTRAIN" not in protocol or "no NAVTEST" not in protocol:
            raise ValueError("planning-JEPA verifier is not a NAVTRAIN-only artifact")
        if payload.get("feature_version") != "planning_jepa_single_v1":
            raise ValueError("unsupported planning-JEPA verifier feature version")
        if payload.get("mother_key") != "r94_selected":
            raise ValueError("planning-JEPA verifier is not anchored to r94")
        if int(payload.get("feature_count", -1)) != 186:
            raise ValueError("planning-JEPA verifier must contain 186 aligned features")
        scale = torch.as_tensor(payload.get("scale"), dtype=torch.float32)
        coefficient = torch.as_tensor(payload.get("coefficient"), dtype=torch.float32)
        if scale.shape != (186,) or coefficient.shape != (186,):
            raise ValueError("planning-JEPA verifier tensors have invalid shapes")
        if not torch.isfinite(scale).all() or not torch.isfinite(coefficient).all():
            raise ValueError("planning-JEPA verifier contains non-finite tensors")
        policy = payload.get("policy")
        required = {
            "top_k", "min_advantage", "max_baseline_margin",
            "min_critical_delta", "min_safety_delta", "min_progress_delta",
        }
        if not isinstance(policy, dict) or set(policy) != required:
            raise ValueError("planning-JEPA verifier policy is incomplete")
        self.planning_jepa_verifier_scale = scale
        self.planning_jepa_verifier_coefficient = coefficient
        self.planning_jepa_verifier_intercept = float(payload.get("intercept", 0.0))
        self.planning_jepa_verifier_policy = {
            name: float(value) for name, value in policy.items()
        }

    @staticmethod
    def _planning_jepa_latent_summary(
        latent: torch.Tensor, baseline_top: torch.Tensor
    ) -> torch.Tensor:
        latent = latent.float()
        rows = torch.arange(latent.shape[0], device=latent.device)
        anchor = latent[rows, baseline_top]
        delta = latent - anchor[:, None]
        delta_norm = delta.square().mean(dim=-1, keepdim=True).sqrt()
        cosine = F.cosine_similarity(
            latent, anchor[:, None], dim=-1, eps=1e-6
        ).unsqueeze(-1)
        return torch.cat(
            (
                latent.mean(dim=-1, keepdim=True),
                latent.std(dim=-1, unbiased=False, keepdim=True),
                latent.square().mean(dim=-1, keepdim=True).sqrt(),
                delta.mean(dim=-1, keepdim=True),
                delta.std(dim=-1, unbiased=False, keepdim=True),
                delta_norm,
                cosine,
            ),
            dim=-1,
        ).nan_to_num()

    def _planning_jepa_feature_map(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        candidates: torch.Tensor,
        baseline_top: torch.Tensor,
        baseline_margin: torch.Tensor,
    ) -> torch.Tensor:
        """Exactly reproduce planning_jepa_single_v1's 186-D feature map."""
        geometry_map = self._geometry_feature_map(
            evaluation, baseline, candidates, baseline_top, baseline_margin
        )
        # _geometry_feature_map layout is combined(79), relative(79),
        # centered(79), rank(1), margin(1); raw occupies the first 10 values.
        raw_relative = geometry_map[..., 79:89]
        raw_centered = geometry_map[..., 158:168]

        rank_prediction = evaluation.get(
            "planning_jepa_verifier_rank_prediction",
            evaluation["current_future_rank_prediction"],
        )
        factor_deltas = evaluation.get(
            "planning_jepa_verifier_factor_deltas",
            evaluation["current_future_factor_deltas"],
        )
        planning = torch.cat(
            (
                rank_prediction.float(),
                factor_deltas.float(),
            ),
            dim=-1,
        ).nan_to_num()
        rows = torch.arange(planning.shape[0], device=planning.device)
        planning_relative = planning - planning[rows, baseline_top, None]
        planning_centered = planning - planning.mean(dim=1, keepdim=True)

        latent = self._planning_jepa_latent_summary(
            evaluation.get(
                "planning_jepa_verifier_world_features",
                evaluation["current_future_world_features"],
            ),
            baseline_top,
        )
        latent_relative = latent - latent[rows, baseline_top, None]
        latent_centered = latent - latent.mean(dim=1, keepdim=True)

        # A one-reader ensemble has mean==reader and exactly zero uncertainty.
        planning_zero = torch.zeros_like(planning_relative)
        latent_zero = torch.zeros_like(latent_relative)
        return torch.cat(
            (
                raw_relative,
                raw_centered,
                planning_relative,
                planning_centered,
                latent_relative,
                latent_centered,
                planning_relative,
                planning_zero,
                planning_zero,
                latent_relative,
                latent_zero,
                latent_zero,
                geometry_map[..., 89:158],
                geometry_map[..., -2:],
            ),
            dim=-1,
        ).nan_to_num()

    def _planning_jepa_preferences(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        candidates: torch.Tensor,
        baseline_top: torch.Tensor,
        baseline_margin: torch.Tensor,
    ) -> torch.Tensor:
        features = self._planning_jepa_feature_map(
            evaluation, baseline, candidates, baseline_top, baseline_margin
        )
        scale = self.planning_jepa_verifier_scale.to(
            device=features.device, dtype=features.dtype
        )
        coefficient = self.planning_jepa_verifier_coefficient.to(
            device=features.device, dtype=features.dtype
        )
        return (
            (features / scale) @ coefficient
            + float(self.planning_jepa_verifier_intercept)
        )

    def _planning_jepa_residual_selection(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        candidates: torch.Tensor,
        mother: torch.Tensor,
        baseline_top: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply the verifier strictly after the frozen r94 mother decision."""
        rows = torch.arange(baseline.shape[0], device=baseline.device)
        top_values = baseline.topk(k=min(2, baseline.shape[1]), dim=1).values
        baseline_margin = (
            top_values[:, 0] - top_values[:, 1]
            if top_values.shape[1] == 2
            else torch.full_like(top_values[:, 0], float("inf"))
        )
        absolute_preference = self._planning_jepa_preferences(
            evaluation, baseline, candidates, mother, baseline_margin
        )
        preference = absolute_preference - absolute_preference[
            rows, mother, None
        ]
        policy = self.planning_jepa_verifier_policy
        top_k = min(int(policy["top_k"]), baseline.shape[1])
        shortlist = torch.zeros_like(baseline, dtype=torch.bool)
        shortlist.scatter_(1, baseline.topk(k=top_k, dim=1).indices, True)
        candidate = preference.masked_fill(
            ~shortlist, float("-inf")
        ).argmax(dim=1)
        advantage = preference[rows, candidate]
        critical = evaluation["candidate_subscores"][..., [0, 1, 3]]
        critical_delta = (
            critical[rows, candidate] - critical[rows, mother]
        ).amin(dim=1)
        safety_delta = (
            evaluation["predicted_safety"][rows, candidate]
            - evaluation["predicted_safety"][rows, mother]
        )
        progress_delta = (
            evaluation["candidate_subscores"][rows, candidate, 2]
            - evaluation["candidate_subscores"][rows, mother, 2]
        )
        accepted = (
            (candidate != mother)
            & (advantage >= policy["min_advantage"])
            & (baseline_margin <= policy["max_baseline_margin"])
            & (critical_delta >= policy["min_critical_delta"])
            & (safety_delta >= policy["min_safety_delta"])
            & (progress_delta >= policy["min_progress_delta"])
        )
        selected = torch.where(accepted, candidate, mother)
        return selected, accepted, {
            "planning_jepa_verifier_candidate": candidate,
            "planning_jepa_verifier_accepted": accepted,
            "planning_jepa_verifier_advantage": advantage,
            "planning_jepa_verifier_critical_delta": critical_delta,
            "planning_jepa_verifier_safety_delta": safety_delta,
            "planning_jepa_verifier_progress_delta": progress_delta,
        }

    def _geometry_feature_map(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        candidates: torch.Tensor,
        baseline_top: torch.Tensor,
        baseline_margin: torch.Tensor,
    ) -> torch.Tensor:
        """Reproduce the frozen NAVTRAIN geometry feature map online."""
        xy = candidates[..., :2].float()
        yaw = candidates[..., 2].float()
        delta_xy = torch.diff(xy, dim=-2, prepend=torch.zeros_like(xy[..., :1, :]))
        step = torch.linalg.vector_norm(delta_xy, dim=-1)
        speed = step / 0.5
        acceleration = torch.diff(
            speed, dim=-1, prepend=torch.zeros_like(speed[..., :1])
        ) / 0.5
        delta_yaw = torch.atan2(
            torch.sin(torch.diff(yaw, dim=-1, prepend=torch.zeros_like(yaw[..., :1]))),
            torch.cos(torch.diff(yaw, dim=-1, prepend=torch.zeros_like(yaw[..., :1]))),
        )
        curvature = delta_yaw / step.clamp_min(0.1)
        geometry_summary = torch.stack(
            (
                xy[..., -1, 0] / 50.0,
                xy[..., -1, 1] / 50.0,
                torch.sin(yaw[..., -1]),
                torch.cos(yaw[..., -1]),
                torch.linalg.vector_norm(xy[..., -1, :], dim=-1) / 50.0,
                step.sum(dim=-1) / 50.0,
                xy[..., 1].abs().amax(dim=-1) / 20.0,
                speed.mean(dim=-1) / 20.0,
                speed.amax(dim=-1) / 20.0,
                acceleration.abs().mean(dim=-1) / 10.0,
                acceleration.abs().amax(dim=-1) / 10.0,
                curvature.abs().mean(dim=-1),
                curvature.abs().amax(dim=-1),
            ),
            dim=-1,
        )
        temporal = torch.cat(
            (
                xy / 50.0,
                torch.sin(yaw).unsqueeze(-1),
                torch.cos(yaw).unsqueeze(-1),
                (speed / 20.0).unsqueeze(-1),
                (acceleration / 10.0).unsqueeze(-1),
                curvature.unsqueeze(-1),
            ),
            dim=-1,
        ).flatten(start_dim=-2)
        geometry = torch.cat((geometry_summary, temporal), dim=-1).nan_to_num()
        raw = torch.cat(
            (
                baseline.unsqueeze(-1),
                evaluation["candidate_subscores"],
                evaluation["critic_value_probabilities"].unsqueeze(-1),
                evaluation["structured_scores"].unsqueeze(-1),
                evaluation["predicted_safety"].unsqueeze(-1),
            ),
            dim=-1,
        ).float()
        combined = torch.cat((raw, geometry), dim=-1)
        rows = torch.arange(combined.shape[0], device=combined.device)
        anchor = combined[rows, baseline_top]
        relative = combined - anchor[:, None]
        centered = combined - combined.mean(dim=1, keepdim=True)
        baseline_rank = torch.argsort(
            torch.argsort(baseline, dim=1, descending=True), dim=1
        ).float().unsqueeze(-1) / max(combined.shape[1] - 1, 1)
        margin = baseline_margin[:, None, None].expand(
            combined.shape[0], combined.shape[1], 1
        )
        features = torch.cat(
            (combined, relative, centered, baseline_rank, margin), dim=-1
        ).nan_to_num()
        return features

    def _geometry_preferences(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        candidates: torch.Tensor,
        baseline_top: torch.Tensor,
        baseline_margin: torch.Tensor,
    ) -> torch.Tensor:
        features = self._geometry_feature_map(
            evaluation, baseline, candidates, baseline_top, baseline_margin
        )
        shape = features.shape[:2]
        flat_features = features.detach().cpu().reshape(-1, features.shape[-1]).numpy()
        if self.geometry_prediction_mode == "positive_probability":
            prediction = self.geometry_calibrator.predict_proba(flat_features)[:, 1]
        else:
            prediction = self.geometry_calibrator.predict(flat_features)
        return torch.as_tensor(
            prediction, device=baseline.device, dtype=baseline.dtype
        ).reshape(shape)

    def _planning_switch_probabilities(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        candidates: torch.Tensor,
        baseline_top: torch.Tensor,
        baseline_margin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fused improvement probabilities and ranks two through four."""
        artifact = self.planning_switch
        if artifact is None or self.planning_switch_projection is None:
            raise RuntimeError("planning switch is not loaded")
        base = self._geometry_feature_map(
            evaluation, baseline, candidates, baseline_top, baseline_margin
        )
        projection = self.planning_switch_projection.to(
            device=baseline.device, dtype=evaluation["world_features"].dtype
        )
        projected = evaluation["world_features"] @ projection
        rows = torch.arange(projected.shape[0], device=projected.device)
        relative = projected - projected[rows, baseline_top, None]
        centered = projected - projected.mean(dim=1, keepdim=True)
        features = torch.cat((base, projected, relative, centered), dim=-1).nan_to_num()
        ranked = baseline.topk(k=4, dim=1).indices[:, 1:]
        anchor = features[rows, baseline_top]
        ranked_features = features[rows[:, None], ranked]
        ranked_relative = ranked_features - anchor[:, None]
        margin = baseline_margin[:, None, None].expand(-1, 3, -1)

        top2_config = artifact["top2_config"]
        runner = ranked_features[:, 0]
        if top2_config["variant"] == "relative":
            top2_features = torch.cat((ranked_relative[:, 0], baseline_margin[:, None]), dim=-1)
        else:
            top2_features = torch.cat(
                (ranked_relative[:, 0], anchor, runner, baseline_margin[:, None]), dim=-1
            )
        top2_probability = artifact["top2_model"].predict_proba(
            top2_features.detach().cpu().numpy()
        )[:, 1]

        baseline_score = baseline[rows, baseline_top]
        ranked_score = baseline[rows[:, None], ranked]
        score_gap = (baseline_score[:, None] - ranked_score).unsqueeze(-1)
        ranks = torch.eye(3, device=baseline.device, dtype=features.dtype)[None].expand(
            len(features), -1, -1
        )
        topk_config = artifact["topk_config"]
        if topk_config["variant"] == "relative":
            topk_features = torch.cat(
                (ranked_relative, score_gap, margin, ranks), dim=-1
            ).nan_to_num()
        else:
            expanded_anchor = anchor[:, None].expand_as(ranked_features)
            topk_features = torch.cat(
                (
                    ranked_relative, expanded_anchor, ranked_features,
                    score_gap, margin, ranks,
                ),
                dim=-1,
            ).nan_to_num()
        topk_probability = torch.stack(
            tuple(
                torch.as_tensor(
                    model.predict_proba(
                        topk_features[:, rank].detach().cpu().numpy()
                    )[:, 1],
                    device=baseline.device,
                    dtype=baseline.dtype,
                )
                for rank, model in enumerate(artifact["topk_model"])
            ),
            dim=1,
        )
        top2_probability = torch.as_tensor(
            top2_probability, device=baseline.device, dtype=baseline.dtype
        )
        policy = artifact["policy"]
        fused = topk_probability.clone()
        weight = float(policy["top2_weight"])
        fused[:, 0] = (
            weight * top2_probability
            + (1.0 - weight) * topk_probability[:, 0]
            + float(policy["runner_bonus"])
        )
        return fused, ranked

    def _latent_verifier_preferences(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        candidates: torch.Tensor,
        baseline_top: torch.Tensor,
        baseline_margin: torch.Tensor,
    ) -> torch.Tensor:
        """Reproduce the frozen NAVTRAIN planning-latent verifier online."""
        if self.latent_verifier is None or self.latent_verifier_projection is None:
            raise RuntimeError("latent verifier is not loaded")
        base = self._geometry_feature_map(
            evaluation, baseline, candidates, baseline_top, baseline_margin
        )
        latent = self._verifier_latent(
            evaluation, self.latent_verifier_config
        )
        projection = self.latent_verifier_projection.to(
            device=baseline.device, dtype=latent.dtype
        )
        projected = latent @ projection
        rows = torch.arange(projected.shape[0], device=projected.device)
        relative = projected - projected[rows, baseline_top, None]
        centered = projected - projected.mean(dim=1, keepdim=True)
        features = torch.cat((base, projected, relative, centered), dim=-1).nan_to_num()
        shape = features.shape[:2]
        raw = torch.as_tensor(
            self.latent_verifier.predict(
                features.detach().cpu().reshape(-1, features.shape[-1]).numpy()
            ),
            device=baseline.device,
            dtype=baseline.dtype,
        )
        kind = self.latent_verifier_config["kind"]
        if kind == "scalar":
            prediction = raw.reshape(shape)
        else:
            raw = raw.reshape(shape[0], shape[1], 5)
            subscores = evaluation["candidate_subscores"][..., :5]
            if kind == "factor_abs":
                channels = raw
            elif kind == "factor_residual":
                channels = subscores + raw
            else:
                anchor = subscores[rows, baseline_top]
                channels = anchor[:, None] + raw
            channels = channels.clamp(0.0, 1.0)
            prediction = channels[..., 0] * channels[..., 1] * (
                5.0 * channels[..., 2]
                + 5.0 * channels[..., 3]
                + 2.0 * channels[..., 4]
            ) / 12.0
        return prediction - prediction[rows, baseline_top, None]

    def _latent_residual_veto_probabilities(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        candidates: torch.Tensor,
        baseline_top: torch.Tensor,
        baseline_margin: torch.Tensor,
        verifier_candidate: torch.Tensor,
        verifier_preference: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate whether the best verifier's proposed switch should survive.

        The feature definition is intentionally identical to the NAVTRAIN
        cross-fit study.  Only one candidate per scene is evaluated, and a low
        probability can only restore the original Drive-JEPA winner.
        """
        if (
            self.latent_residual_veto is None
            or self.latent_residual_veto_projection is None
        ):
            return torch.ones_like(baseline_margin)
        base = self._geometry_feature_map(
            evaluation, baseline, candidates, baseline_top, baseline_margin
        )
        latent = evaluation["world_features"]
        projection = self.latent_residual_veto_projection.to(
            device=latent.device, dtype=latent.dtype
        )
        projected = latent @ projection
        rows = torch.arange(projected.shape[0], device=projected.device)
        relative = projected - projected[rows, baseline_top, None]
        centered = projected - projected.mean(dim=1, keepdim=True)
        decision_features = torch.cat(
            (base, projected, relative, centered), dim=-1
        ).float().nan_to_num()
        chosen = decision_features[rows, verifier_candidate]
        anchor = decision_features[rows, baseline_top]
        subscore_delta = (
            evaluation["candidate_subscores"][rows, verifier_candidate]
            - evaluation["candidate_subscores"][rows, baseline_top]
        )
        diagnostics = torch.cat(
            (
                verifier_preference[rows, verifier_candidate, None],
                (
                    baseline[rows, baseline_top]
                    - baseline[rows, verifier_candidate]
                )[:, None],
                baseline_margin[:, None],
                subscore_delta,
                (
                    evaluation["critic_value_probabilities"][rows, verifier_candidate]
                    - evaluation["critic_value_probabilities"][rows, baseline_top]
                )[:, None],
                (
                    evaluation["structured_scores"][rows, verifier_candidate]
                    - evaluation["structured_scores"][rows, baseline_top]
                )[:, None],
                (
                    evaluation["predicted_safety"][rows, verifier_candidate]
                    - evaluation["predicted_safety"][rows, baseline_top]
                )[:, None],
            ),
            dim=1,
        )
        features = torch.cat((chosen, chosen - anchor, diagnostics), dim=1).nan_to_num()
        probability = self.latent_residual_veto.predict_proba(
            features.detach().cpu().numpy()
        )[:, 1]
        return torch.as_tensor(
            probability, device=baseline.device, dtype=baseline.dtype
        )

    def _latent_verifier_factor_deltas(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        candidates: torch.Tensor,
        baseline_top: torch.Tensor,
        baseline_margin: torch.Tensor,
    ) -> torch.Tensor:
        """Predict true factor changes for the optional dual-head safety veto."""
        if (
            self.latent_verifier_factor is None
            or self.latent_verifier_factor_projection is None
        ):
            raise RuntimeError("latent verifier factor head is not loaded")
        base = self._geometry_feature_map(
            evaluation, baseline, candidates, baseline_top, baseline_margin
        )
        latent = self._verifier_latent(
            evaluation, self.latent_verifier_factor_config
        )
        projection = self.latent_verifier_factor_projection.to(
            device=baseline.device, dtype=latent.dtype
        )
        projected = latent @ projection
        rows = torch.arange(projected.shape[0], device=projected.device)
        relative = projected - projected[rows, baseline_top, None]
        centered = projected - projected.mean(dim=1, keepdim=True)
        features = torch.cat((base, projected, relative, centered), dim=-1).nan_to_num()
        shape = features.shape[:2]
        return torch.as_tensor(
            self.latent_verifier_factor.predict(
                features.detach().cpu().reshape(-1, features.shape[-1]).numpy()
            ),
            device=baseline.device,
            dtype=baseline.dtype,
        ).reshape(shape[0], shape[1], 5)

    def _verifier_latent(
        self,
        evaluation: Dict[str, torch.Tensor],
        verifier_config: Dict[str, Any],
    ) -> torch.Tensor:
        """Select the planning representation declared by a NAVTRAIN artifact."""
        world = evaluation["world_features"]
        source = str(verifier_config.get("feature_source", "world"))
        if source == "world":
            return world
        outcome = evaluation.get("outcome_features")
        if outcome is None:
            outcome = self.outcome_predictor(world)
        outcome = F.normalize(outcome.float(), dim=-1).to(world.dtype)
        if source == "outcome":
            return outcome
        if source == "world_outcome":
            return torch.cat((world, outcome), dim=-1)
        raise ValueError(f"unsupported latent verifier feature_source: {source}")

    def _load_pairwise_calibrator(self) -> None:
        path_text = str(self.config.pairwise_calibrator_path)
        if not path_text:
            return
        if "navtest" in path_text.lower():
            raise ValueError("NAVTEST artifacts are forbidden as pairwise calibrators")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        fit = payload.get("robust", payload).get("all_development_fit")
        if not isinstance(fit, dict):
            raise ValueError("pairwise calibrator has no robust all_development_fit")
        scale = torch.tensor(fit["scale"], dtype=torch.float32)
        weights = torch.tensor(fit["weights"], dtype=torch.float32)
        if scale.ndim != 1 or weights.shape != scale.shape or scale.numel() != 95:
            raise ValueError("pairwise calibrator must contain 95 aligned features")
        if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(weights).all()):
            raise ValueError("pairwise calibrator contains non-finite coefficients")
        self.pairwise_scale = scale.clamp_min(1e-5)
        self.pairwise_weights = weights
        self.pairwise_intercept = float(fit.get("intercept", 0.0))

    def _check(self, tensor: torch.Tensor, name: str) -> None:
        if self.config.finite_checks and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains NaN or Inf")

    def _validate(self, tokens: torch.Tensor, candidates: torch.Tensor, base_scores: torch.Tensor) -> None:
        expected_candidates = (
            tokens.shape[0], self.config.num_candidates, self.config.horizon, 3
        )
        if tokens.ndim != 3 or tokens.shape[-1] != self.config.token_dim:
            raise ValueError("scene_tokens must have shape [B, N, token_dim]")
        if tuple(candidates.shape) != expected_candidates:
            raise ValueError(f"candidates must have shape {expected_candidates}")
        valid_score_shapes = {
            (tokens.shape[0], self.config.num_candidates),
            (tokens.shape[0], self.config.num_candidates, 6),
        }
        if tuple(base_scores.shape) not in valid_score_shapes:
            raise ValueError("base_scores must have shape [B, C] or [B, C, 6]")
        if candidates.device != tokens.device or base_scores.device != tokens.device:
            raise ValueError("all planner inputs must be on the same device")
        self._check(tokens, "scene_tokens")
        self._check(candidates, "candidates")
        self._check(base_scores, "base_scores")

    def _actions(self, candidates: torch.Tensor) -> torch.Tensor:
        delta = torch.cat((candidates[..., :1, :], candidates[..., 1:, :] - candidates[..., :-1, :]), dim=-2)
        yaw_delta = torch.atan2(torch.sin(delta[..., 2]), torch.cos(delta[..., 2]))
        distance = torch.linalg.vector_norm(delta[..., :2], dim=-1)
        speed = distance / self.config.time_delta
        acceleration = torch.cat((speed[..., :1], speed[..., 1:] - speed[..., :-1]), dim=-1)
        acceleration = acceleration / self.config.time_delta
        curvature = yaw_delta / distance.clamp_min(1e-3)
        return torch.stack(
            (
                delta[..., 0] / self.config.position_scale,
                delta[..., 1] / self.config.position_scale,
                torch.sin(yaw_delta),
                torch.cos(yaw_delta),
                speed / self.config.speed_scale,
                acceleration / self.config.acceleration_scale,
                curvature / self.config.curvature_scale,
            ),
            dim=-1,
        )

    def _encode_scene(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.scene_norm(tokens)
        queries = self.scene_queries.to(tokens.dtype)[None].expand(tokens.shape[0], -1, -1)
        latent, _ = self.scene_attention(queries, tokens, tokens, need_weights=False)
        return latent

    def _rollout(self, scene: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        actions = self._actions(candidates)
        batch, candidates_count, horizon, _ = actions.shape
        latent_count, dim = scene.shape[1:]
        state = scene[:, None].expand(-1, candidates_count, -1, -1).contiguous()
        sequence = []
        for step in range(horizon):
            encoded = self.action_encoder(actions[:, :, step]) + self.time_embedding[step]
            encoded = encoded[:, :, None].expand(-1, -1, latent_count, -1)
            state = self.transition(
                encoded.reshape(-1, dim), state.reshape(-1, dim)
            ).reshape(batch, candidates_count, latent_count, dim)
            state = self.spatial_mixer(state.reshape(-1, latent_count, dim)).reshape_as(state)
            state = self.rollout_norm(state)
            sequence.append(state)
        return torch.stack(sequence, dim=2)

    def _world_features_from_rollout(
        self, rollout: torch.Tensor, scene: torch.Tensor
    ) -> torch.Tensor:
        """Pool an imagined rollout into the planner's world representation."""
        spatial_weights = torch.softmax(self.spatial_pool(rollout).squeeze(-1), dim=-1)
        temporal_tokens = (spatial_weights[..., None] * rollout).sum(dim=-2)
        temporal_weights = torch.softmax(self.temporal_pool(temporal_tokens).squeeze(-1), dim=-1)
        summary = (temporal_weights[..., None] * temporal_tokens).sum(dim=-2)
        future = self.world_projection(summary)
        current = self.current_projection(scene.mean(dim=1))[:, None]
        current = current.expand(-1, future.shape[1], -1)
        interaction = torch.cat(
            (future, current, future - current, future * current), dim=-1
        )
        future = future + self.current_future_interaction(interaction)
        if self.config.decoupled_current_future_branch:
            return future
        return self.current_future_interaction_v2(future, scene)

    def target_world_features(
        self, scene_tokens: torch.Tensor, trajectory: torch.Tensor
    ) -> torch.Tensor:
        """Encode a training trajectory as an action-conditioned JEPA target."""
        if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
            raise ValueError("trajectory must have shape [B, horizon, 3]")
        scene = self._encode_scene(scene_tokens)
        rollout = self._rollout(scene, trajectory[:, None])
        return self._world_features_from_rollout(rollout, scene)[:, 0]

    def factor_target_features(
        self, factors: torch.Tensor, baseline_index: torch.Tensor
    ) -> torch.Tensor:
        """Encode absolute and baseline-relative counterfactual outcomes.

        The fixed Fourier/JL target prevents a learned target branch from
        collapsing while retaining distances between the six NAVTRAIN
        simulator factors.  It is used only as a detached training target.
        """
        if factors.ndim != 3 or factors.shape[-1] != 6:
            raise ValueError("factors must have shape [B, C, 6]")
        if tuple(baseline_index.shape) != (factors.shape[0],):
            raise ValueError("baseline_index must have shape [B]")
        rows = torch.arange(factors.shape[0], device=factors.device)
        values = factors.float().clamp(0.0, 1.0)
        relative = values - values[rows, baseline_index.long(), None]
        inputs = torch.cat((values, relative), dim=-1)
        encoded = torch.cat(
            (
                inputs,
                torch.sin(math.pi * inputs),
                torch.cos(math.pi * inputs),
                torch.sin(2.0 * math.pi * inputs),
                torch.cos(2.0 * math.pi * inputs),
            ),
            dim=-1,
        )
        projection = self.outcome_target_projection.to(
            device=encoded.device, dtype=encoded.dtype
        )
        return encoded @ projection

    def _evaluate(
        self,
        rollout: torch.Tensor,
        base_scores: torch.Tensor,
        scene: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        world = self._world_features_from_rollout(rollout, scene)
        current_future_world = (
            self.current_future_interaction_v2(world, scene)
            if self.config.decoupled_current_future_branch
            else world
        )
        world_logits = self.subscore_head(world)
        value_logits = self.value_head(world).squeeze(-1)
        collision_logits = self.collision_head(world).squeeze(-1)
        baseline = base_scores[..., -1] if base_scores.ndim == 3 else base_scores
        rows = torch.arange(world.shape[0], device=world.device)
        baseline_top = baseline.argmax(dim=1)
        anchor_world = world[rows, baseline_top]
        relative_world = world - anchor_world[:, None]
        baseline_delta = baseline - baseline[rows, baseline_top, None]
        preference_features = torch.cat(
            (
                relative_world,
                world * anchor_world[:, None],
                baseline_delta[..., None],
            ),
            dim=-1,
        )
        raw_preference_logits = self.preference_head(preference_features).squeeze(-1)
        preference_logits = (
            raw_preference_logits
            - raw_preference_logits[rows, baseline_top, None]
        )
        value_logits = (
            value_logits
            + self.config.preference_residual_scale * preference_logits
        )
        # All residual terms are exactly zero at initialization.  Therefore an
        # official v1 checkpoint preserves both baseline scores and argmax.
        world_scores = (
            baseline
            + self.config.world_value_weight * value_logits
            + self.config.subscore_weight * world_logits.mean(dim=-1)
            - self.config.collision_penalty * collision_logits
        )
        metric_probabilities = torch.sigmoid(world_logits)
        value_probabilities = torch.sigmoid(value_logits)
        collision_probabilities = torch.sigmoid(collision_logits)

        # NAVSIM v1's PDMS is NC * DAC times the weighted progress/TTC/comfort
        # sum.  Keeping this factorization explicit makes the critic's safety
        # evidence inspectable instead of hiding it in an unconstrained mean.
        structured_scores = (
            metric_probabilities[..., 0]
            * metric_probabilities[..., 1]
            * (
                5.0 * metric_probabilities[..., 2]
                + 5.0 * metric_probabilities[..., 3]
                + 2.0 * metric_probabilities[..., 4]
            )
            / 12.0
        )
        final_head_probabilities = metric_probabilities[..., 5]
        value_weight = self.config.conservative_value_weight
        structured_weight = self.config.conservative_structured_weight
        final_weight = 1.0 - value_weight - structured_weight

        # Every learned head is initialized at logit zero.  Centering each
        # signal at its exact neutral value therefore guarantees bit-exact
        # Drive-JEPA selection before training, while keeping the residual
        # bounded after training.
        conservative_residual = (
            value_weight * (value_probabilities - 0.5)
            + structured_weight * (structured_scores - 0.125)
            + final_weight * (final_head_probabilities - 0.5)
        )
        evidence = torch.stack(
            (
                value_probabilities - 0.5,
                structured_scores - 0.125,
                final_head_probabilities - 0.5,
            ),
            dim=-1,
        )
        disagreement = evidence.amax(dim=-1) - evidence.amin(dim=-1)
        confidence = torch.exp(
            -disagreement / self.config.conservative_consistency_temperature
        )
        conservative_scores = (
            baseline
            + self.config.conservative_residual_scale
            * confidence
            * torch.tanh(conservative_residual)
        )
        predicted_safety = (
            metric_probabilities[..., 1]
            * 0.5
            * (
                metric_probabilities[..., 0]
                + (1.0 - collision_probabilities)
            )
        )
        return {
            # Expose the candidate-conditioned planning latent for optional
            # NAVTRAIN-only frozen verifier studies.  This is an activation,
            # not a new parameter, so old checkpoints remain load-compatible.
            "world_features": world,
            "current_future_world_features": current_future_world,
            "world_logits": world_logits,
            "candidate_subscores": metric_probabilities,
            "value_logits": value_logits,
            "preference_logits": preference_logits,
            "collision_logits": collision_logits,
            "collision_probabilities": collision_probabilities,
            "world_scores": world_scores,
            "structured_scores": structured_scores,
            "critic_value_probabilities": value_probabilities,
            "critic_final_probabilities": final_head_probabilities,
            "critic_residual": conservative_residual,
            "critic_confidence": confidence,
            "predicted_safety": predicted_safety,
            "conservative_scores": conservative_scores,
        }

    def _pairwise_preferences(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        baseline_top: torch.Tensor,
        baseline_margin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the frozen NAVTRAIN pairwise verifier as a Torch feature map."""
        raw = torch.cat(
            (
                baseline.unsqueeze(-1),
                evaluation["candidate_subscores"],
                evaluation["critic_value_probabilities"].unsqueeze(-1),
                evaluation["structured_scores"].unsqueeze(-1),
                evaluation["predicted_safety"].unsqueeze(-1),
            ),
            dim=-1,
        )
        rows = torch.arange(raw.shape[0], device=raw.device)
        anchor = raw[rows, baseline_top]
        relative = raw - anchor[:, None]
        upper = torch.triu_indices(
            relative.shape[-1], relative.shape[-1], device=relative.device
        )
        pairwise = relative[..., upper[0]] * relative[..., upper[1]]
        centered = raw - raw.mean(dim=1, keepdim=True)
        contextual = relative * anchor[:, None]
        margin_context = relative * baseline_margin[:, None, None]
        features = torch.cat(
            (relative, pairwise, centered, contextual, margin_context), dim=-1
        )
        scale = self.pairwise_scale.to(device=raw.device, dtype=raw.dtype)
        weights = self.pairwise_weights.to(device=raw.device, dtype=raw.dtype)
        return torch.einsum("bcf,f->bc", features / scale, weights) + float(
            self.pairwise_intercept
        )

    def _post_r94_progress_expansion(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        candidates: torch.Tensor,
        mother: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add only NAVTRAIN-screened, safety-guarded progress improvements."""
        progress = evaluation.get("current_future_progress_prediction")
        if progress is None:
            raise KeyError(
                "current_future_progress_prediction is required by "
                "post-r94 progress expansion"
            )
        batch = baseline.shape[0]
        rows = torch.arange(batch, device=baseline.device)
        top_k = min(self.config.post_r94_progress_top_k, baseline.shape[1])
        top = baseline.topk(k=top_k, dim=1).indices
        mother_in_top = (top == mother[:, None]).any(dim=1)
        progress = progress.squeeze(-1)
        progress_delta = (
            progress.gather(1, top) - progress[rows, mother, None]
        )
        position = progress_delta.argmax(dim=1)
        proposed = top[rows, position]
        proposed_progress = progress_delta[rows, position]

        rank_prediction = evaluation["current_future_rank_prediction"]
        rank_probability = rank_prediction[rows, proposed, 5].sigmoid()
        rank_safety_probability = rank_prediction[
            rows, proposed, :4
        ].sigmoid().amin(dim=1)
        gap = baseline[rows, mother] - baseline[rows, proposed]
        frozen_delta = (
            evaluation["candidate_subscores"][rows, proposed]
            - evaluation["candidate_subscores"][rows, mother]
        )
        frozen_safety = (
            evaluation["predicted_safety"][rows, proposed]
            - evaluation["predicted_safety"][rows, mother]
        )

        plan = candidates.float()[rows, proposed]
        velocity = torch.diff(plan[..., :2], dim=1) / float(self.config.time_delta)
        acceleration = torch.diff(velocity, dim=1) / float(self.config.time_delta)
        jerk = torch.diff(acceleration, dim=1) / float(self.config.time_delta)
        comfort_safe = (
            acceleration.norm(dim=-1).amax(dim=1)
            <= self.config.candidate_set_max_proxy_acceleration
        ) & (
            jerk.norm(dim=-1).amax(dim=1)
            <= self.config.candidate_set_max_proxy_jerk
        )
        accepted = (
            mother_in_top
            & (proposed != mother)
            & (proposed_progress >= self.config.post_r94_progress_min_delta)
            & (
                rank_probability
                >= self.config.post_r94_progress_min_rank_probability
            )
            & (
                rank_safety_probability
                >= self.config.post_r94_progress_min_rank_safety_probability
            )
            & (gap <= self.config.post_r94_progress_max_baseline_gap)
            & (
                frozen_delta[:, 0]
                >= self.config.post_r94_progress_min_frozen_critical
            )
            & (
                frozen_safety
                >= self.config.post_r94_progress_min_frozen_safety
            )
            & (
                frozen_delta[:, 3]
                >= self.config.post_r94_progress_min_frozen_ttc
            )
            & comfort_safe
        )
        return (
            torch.where(accepted, proposed, mother),
            accepted,
            proposed_progress,
        )

    def _post_r94_selection(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        candidates: torch.Tensor,
        mother: torch.Tensor,
        model: Any = None,
        projection: Optional[torch.Tensor] = None,
        policy: Optional[Dict[str, Any]] = None,
        feature_names: Optional[tuple[str, ...]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a frozen NAVTRAIN post-r94 future-utility selector."""
        if model is None:
            model = self.__dict__.get("post_r94_selector_model")
            projection = self.post_r94_selector_projection
            policy = self.post_r94_selector_policy
            feature_names = self.post_r94_selector_feature_names
        if model is None:
            return mother, torch.zeros_like(mother, dtype=torch.bool), baseline.new_zeros(
                len(mother)
            )
        if projection is None or policy is None or feature_names is None:
            raise RuntimeError("post-r94 selector payload is incomplete")
        batch = baseline.shape[0]
        rows = torch.arange(batch, device=baseline.device)
        top_k = min(8, baseline.shape[1])
        top = baseline.topk(k=top_k, dim=1).indices

        def gather(values: torch.Tensor) -> torch.Tensor:
            extra = (None,) * (values.ndim - 2)
            index = top[(...,) + extra].expand(*top.shape, *values.shape[2:])
            return values.gather(1, index)

        utility_values = torch.stack(
            (
                baseline.gather(1, top),
                gather(evaluation["candidate_subscores"])[..., 2],
                gather(evaluation["candidate_subscores"])[..., 3],
                evaluation["structured_scores"].gather(1, top),
                evaluation["critic_value_probabilities"].gather(1, top),
                evaluation["predicted_safety"].gather(1, top),
                gather(evaluation["current_future_rank_prediction"])[..., 4],
                gather(evaluation["current_future_rank_prediction"])[..., 5],
                gather(evaluation["current_future_factor_deltas"])[..., 2],
                gather(evaluation["current_future_factor_deltas"])[..., 3],
            ),
            dim=-1,
        ).float()
        utility_values = (
            utility_values - utility_values.mean(dim=1, keepdim=True)
        ) / utility_values.std(dim=1, keepdim=True).clamp_min(1e-5)
        weights = utility_values.new_tensor(
            [policy["weights"][name] for name in feature_names]
        )
        utility = torch.einsum("bcf,f->bc", utility_values, weights)
        position = utility.argmax(dim=1)

        # The sklearn verifier is deliberately evaluated on CPU, matching its
        # frozen NAVTRAIN feature construction and avoiding split-boundary
        # differences from GPU matrix multiplication.
        top_cpu = top.detach().cpu()
        mother_cpu = mother.detach().cpu()
        rows_cpu = torch.arange(batch)

        def gather_cpu(values: torch.Tensor) -> torch.Tensor:
            values = values.detach().float().cpu()
            extra = (None,) * (values.ndim - 2)
            index = top_cpu[(...,) + extra].expand(
                *top_cpu.shape, *values.shape[2:]
            )
            return values.gather(1, index)

        feature_blocks = []
        scalar_values = (
            baseline.detach().float().cpu().unsqueeze(-1),
            evaluation["candidate_subscores"].detach().float().cpu(),
            evaluation["structured_scores"].detach().float().cpu().unsqueeze(-1),
            evaluation["critic_value_probabilities"].detach().float().cpu().unsqueeze(-1),
            evaluation["predicted_safety"].detach().float().cpu().unsqueeze(-1),
        )
        for value in scalar_values:
            candidate_value = gather_cpu(value)
            anchor_value = value[rows_cpu, mother_cpu][:, None].expand_as(
                candidate_value
            )
            feature_blocks.extend(
                (
                    candidate_value,
                    anchor_value,
                    candidate_value - anchor_value,
                    candidate_value * anchor_value,
                )
            )
        world = F.normalize(
            evaluation["world_features"].detach().float().cpu(), dim=-1
        ) @ projection.float().cpu()
        candidate_world = gather_cpu(world)
        anchor_world = world[rows_cpu, mother_cpu][:, None].expand_as(
            candidate_world
        )
        scene_world = world.mean(dim=1, keepdim=True).expand_as(candidate_world)
        feature_blocks.extend(
            (
                candidate_world,
                anchor_world,
                candidate_world - anchor_world,
                candidate_world * anchor_world,
                candidate_world - scene_world,
            )
        )
        proposal = candidates.detach().float().cpu()
        candidate_plan = gather_cpu(proposal)
        anchor_plan = proposal[rows_cpu, mother_cpu][:, None].expand_as(
            candidate_plan
        )
        plan_scale = proposal.new_tensor((10.0, 10.0, math.pi))
        relative_plan = (candidate_plan - anchor_plan) / plan_scale
        relative_step = (
            torch.diff(candidate_plan, dim=2)
            - torch.diff(anchor_plan, dim=2)
        ) / plan_scale

        def trajectory_summary(trajectory: torch.Tensor) -> torch.Tensor:
            xy = trajectory[..., :2]
            yaw = trajectory[..., 2]
            velocity = torch.diff(xy, dim=-2) / 0.5
            speed = velocity.norm(dim=-1)
            acceleration = torch.diff(velocity, dim=-2) / 0.5
            acceleration_norm = acceleration.norm(dim=-1)
            jerk = torch.diff(acceleration, dim=-2) / 0.5
            jerk_norm = jerk.norm(dim=-1)
            yaw_step = torch.diff(yaw, dim=-1).abs()
            return torch.stack(
                (
                    xy[..., -1, 0],
                    xy[..., -1, 1],
                    yaw[..., -1],
                    torch.diff(xy, dim=-2).norm(dim=-1).sum(dim=-1),
                    xy[..., -1, :].norm(dim=-1),
                    speed.mean(dim=-1),
                    speed.amax(dim=-1),
                    speed[..., -1],
                    acceleration_norm.mean(dim=-1),
                    acceleration_norm.amax(dim=-1),
                    jerk_norm.mean(dim=-1),
                    jerk_norm.amax(dim=-1),
                    yaw_step.mean(dim=-1),
                    yaw_step.amax(dim=-1),
                ),
                dim=-1,
            )

        candidate_geometry = trajectory_summary(candidate_plan)
        anchor_geometry = trajectory_summary(anchor_plan)
        top_position = torch.arange(top_k, dtype=torch.float32)[None, :, None]
        top_position = top_position.expand(batch, -1, -1) / max(top_k - 1, 1)
        is_anchor = (top_cpu == mother_cpu[:, None]).float()[..., None]
        feature_blocks.extend(
            (
                relative_plan.flatten(2),
                relative_step.flatten(2),
                candidate_geometry,
                candidate_geometry - anchor_geometry,
                top_position,
                is_anchor,
            )
        )
        features = torch.cat(feature_blocks, dim=-1)
        if features.shape[-1] != 595:
            raise RuntimeError(
                f"post-r94 feature width mismatch: {features.shape[-1]}"
            )
        selection_mode = policy.get("selection_mode")
        flat_features = features.reshape(-1, features.shape[-1]).numpy()
        if selection_mode == "gain_set":
            gain_prediction = torch.from_numpy(
                model.predict(flat_features).reshape(batch, top_k)
            ).to(device=baseline.device, dtype=baseline.dtype)
            gap_set = baseline[rows, mother, None] - baseline.gather(1, top)
            candidate_subscores = gather(evaluation["candidate_subscores"])
            anchor_subscores = evaluation["candidate_subscores"][rows, mother]
            frozen_delta_set = candidate_subscores - anchor_subscores[:, None]
            candidate_safety = evaluation["predicted_safety"].gather(1, top)
            anchor_safety = evaluation["predicted_safety"][rows, mother]
            feasible = (
                (gap_set <= float(policy.get("max_baseline_gap", 1.0)))
                & (
                    frozen_delta_set[..., 0]
                    >= float(policy.get("min_frozen_critical", -1.0))
                )
                & (
                    frozen_delta_set[..., 2]
                    >= float(policy.get("min_frozen_progress", -1.0))
                )
                & (
                    frozen_delta_set[..., 3]
                    >= float(policy.get("min_frozen_ttc", -1.0))
                )
                & (
                    candidate_safety - anchor_safety[:, None]
                    >= float(policy.get("min_frozen_safety", -1.0))
                )
            )
            # Optional safety-aware expansion: preserve the complete inner
            # policy, but apply stricter factor consistency only to candidates
            # admitted by a wider baseline-gap band.  All signals are frozen
            # model predictions; no evaluation labels are consumed here.
            expansion_start = policy.get("expanded_gap_start")
            if expansion_start is not None:
                expanded = gap_set > float(expansion_start)
                expansion_feasible = (
                    (
                        frozen_delta_set[..., 0]
                        >= float(policy.get("expanded_min_frozen_critical", -1.0))
                    )
                    & (
                        frozen_delta_set[..., 2]
                        >= float(policy.get("expanded_min_frozen_progress", -1.0))
                    )
                    & (
                        frozen_delta_set[..., 3]
                        >= float(policy.get("expanded_min_frozen_ttc", -1.0))
                    )
                    & (
                        candidate_safety - anchor_safety[:, None]
                        >= float(policy.get("expanded_min_frozen_safety", -1.0))
                    )
                )
                feasible &= (~expanded) | expansion_feasible
            feasible |= top == mother[:, None]
            position = gain_prediction.masked_fill(~feasible, -torch.inf).argmax(dim=1)
            proposed = top[rows, position]
            proposed_gain = gain_prediction[rows, position]
            anchor_mask = top == mother[:, None]
            anchor_found = anchor_mask.any(dim=1)
            anchor_gain = gain_prediction.masked_fill(~anchor_mask, -torch.inf).amax(dim=1)
            anchor_gain = torch.where(anchor_found, anchor_gain, torch.zeros_like(anchor_gain))
            proposed_margin = proposed_gain - anchor_gain
            gap = baseline[rows, mother] - baseline[rows, proposed]
            accepted = (
                (proposed != mother)
                & anchor_found
                & (proposed_gain >= float(policy.get("min_predicted_gain", 0.0)))
                & (proposed_margin >= float(policy.get("min_predicted_margin", 0.0)))
                & (gap <= float(policy.get("max_baseline_gap", 1.0)))
            )
            return torch.where(accepted, proposed, mother), accepted, proposed_gain

        probabilities = model.predict_proba(flat_features)
        positive_column = int(list(model.classes_).index(1))
        safety_probability = torch.from_numpy(
            probabilities[:, positive_column].reshape(batch, top_k)
        ).to(device=baseline.device, dtype=baseline.dtype)
        if policy.get("selection_mode") in {"feasible_set", "positive_set"}:
            candidate_subscores = gather(evaluation["candidate_subscores"])
            anchor_subscores = evaluation["candidate_subscores"][rows, mother]
            frozen_delta_set = candidate_subscores - anchor_subscores[:, None]
            candidate_safety = evaluation["predicted_safety"].gather(1, top)
            anchor_safety = evaluation["predicted_safety"][rows, mother]
            feasible = (
                (safety_probability >= float(policy["safety_probability"]))
                & (
                    baseline[rows, mother, None] - baseline.gather(1, top)
                    <= float(policy.get("max_baseline_gap", 0.01))
                )
                & (frozen_delta_set[..., 0] >= 0.0)
                & (frozen_delta_set[..., 2] >= 0.0)
                & (frozen_delta_set[..., 3] >= 0.0)
                & (candidate_safety - anchor_safety[:, None] >= 0.0)
            )
            # The deployed mother is always a valid fallback, exactly matching
            # the frozen NAVTRAIN studies.
            feasible |= top == mother[:, None]
            ranking = (
                safety_probability
                if policy.get("selection_mode") == "positive_set"
                else utility
            )
            position = ranking.masked_fill(~feasible, -torch.inf).argmax(dim=1)
        proposed = top[rows, position]
        proposed_safety = safety_probability[rows, position]
        gap = baseline[rows, mother] - baseline[rows, proposed]
        frozen_delta = (
            evaluation["candidate_subscores"][rows, proposed]
            - evaluation["candidate_subscores"][rows, mother]
        )
        safety_delta = (
            evaluation["predicted_safety"][rows, proposed]
            - evaluation["predicted_safety"][rows, mother]
        )
        accepted = (
            (proposed != mother)
            & (proposed_safety >= float(policy["safety_probability"]))
            & (gap <= 0.01)
            & (frozen_delta[:, 0] >= 0.0)
            & (frozen_delta[:, 2] >= 0.0)
            & (frozen_delta[:, 3] >= 0.0)
            & (safety_delta >= 0.0)
        )
        return torch.where(accepted, proposed, mother), accepted, proposed_safety

    def _candidate_set_selection(
        self,
        evaluation: Dict[str, torch.Tensor],
        baseline: torch.Tensor,
        candidates: torch.Tensor,
        mother: torch.Tensor,
        baseline_top: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply the frozen r53 scorer and an official-metric comfort proxy."""
        scorers = self.__dict__.get("candidate_set_scorers", [])
        if not scorers:
            unchanged = torch.zeros_like(mother, dtype=torch.bool)
            return mother, unchanged, {}
        batch = baseline.shape[0]
        rows = torch.arange(batch, device=baseline.device)
        top = baseline.topk(k=min(4, baseline.shape[1]), dim=1).indices
        mother_in_top = (top == mother[:, None]).any(dim=1)

        def gather(values: torch.Tensor) -> torch.Tensor:
            extra = (None,) * (values.ndim - 2)
            index = top[(...,) + extra].expand(
                *top.shape, *values.shape[2:]
            )
            return values.gather(1, index)

        world = F.normalize(evaluation["world_features"].float(), dim=-1)
        current = world[rows, mother]
        future = gather(world)
        candidate_trajectory = gather(candidates.float())
        mother_trajectory = candidates.float()[rows, mother]
        scale = candidates.new_tensor((10.0, 10.0, math.pi)).float()
        relative = (candidate_trajectory - mother_trajectory[:, None]) / scale
        candidate_steps = candidate_trajectory[:, :, 1:] - candidate_trajectory[:, :, :-1]
        mother_steps = mother_trajectory[:, 1:] - mother_trajectory[:, :-1]
        relative_steps = (candidate_steps - mother_steps[:, None]) / scale
        trajectory = torch.cat(
            (relative.flatten(2), relative_steps.flatten(2)), dim=-1
        )

        raw = torch.cat(
            (
                baseline.unsqueeze(-1).float(),
                evaluation["candidate_subscores"].float(),
                evaluation["critic_value_probabilities"].unsqueeze(-1).float(),
                evaluation["structured_scores"].unsqueeze(-1).float(),
                evaluation["predicted_safety"].unsqueeze(-1).float(),
            ),
            dim=-1,
        )
        selected_raw = gather(raw)
        diagnostic = selected_raw - raw[rows, mother][:, None]

        gains, probabilities, safeties = [], [], []
        with torch.no_grad():
            for scorer in scorers:
                parameter = next(scorer.parameters())
                if parameter.device != baseline.device:
                    scorer.to(baseline.device)
                gain, improvement, safety = scorer(
                    current, future, trajectory, diagnostic
                )
                mother_position = top == mother[:, None]
                gain = gain - gain.masked_fill(
                    ~mother_position, 0.0
                ).sum(dim=1, keepdim=True)
                gains.append(gain)
                probabilities.append(improvement.sigmoid())
                safeties.append(safety.sigmoid())
        gain = torch.stack(gains).mean(0)
        probability = torch.stack(probabilities).mean(0)
        safety = torch.stack(safeties).mean(0)
        mother_position = (top == mother[:, None]).float().argmax(dim=1)
        tree_models = self.__dict__.get("tree_meta_models")
        if tree_models is not None:
            # r53/r58 use the frozen r32 selection as their residual anchor.
            tree_meta_gain = gain
            # Build the tree features on CPU, exactly as in the frozen r57/r58
            # study.  Histogram-tree split decisions are sensitive to tiny
            # CPU/GPU matmul rounding differences near a learned threshold.
            top_cpu = top.detach().cpu()
            mother_cpu = mother.detach().cpu()
            rows_cpu = torch.arange(batch)

            def gather_cpu(values: torch.Tensor) -> torch.Tensor:
                extra = (None,) * (values.ndim - 2)
                index = top_cpu[(...,) + extra].expand(
                    *top_cpu.shape, *values.shape[2:]
                )
                return values.gather(1, index)

            world_cpu = F.normalize(
                evaluation["world_features"].detach().float().cpu(), dim=-1
            )
            projection = self.tree_meta_projection.float().cpu()
            projected = world_cpu @ projection
            candidate_latent = gather_cpu(projected)
            mother_latent = projected[rows_cpu, mother_cpu][:, None]
            latent = torch.cat(
                (
                    candidate_latent,
                    mother_latent.expand_as(candidate_latent),
                    candidate_latent - mother_latent,
                    candidate_latent * mother_latent,
                ),
                dim=-1,
            )
            raw_cpu = torch.cat(
                (
                    baseline.detach().float().cpu().unsqueeze(-1),
                    evaluation["candidate_subscores"].detach().float().cpu(),
                    evaluation["critic_value_probabilities"].detach().float().cpu().unsqueeze(-1),
                    evaluation["structured_scores"].detach().float().cpu().unsqueeze(-1),
                    evaluation["predicted_safety"].detach().float().cpu().unsqueeze(-1),
                ),
                dim=-1,
            )
            selected_raw_cpu = gather_cpu(raw_cpu)
            mother_raw = raw_cpu[rows_cpu, mother_cpu][:, None]
            tree_diagnostic = torch.cat(
                (selected_raw_cpu, selected_raw_cpu - mother_raw), dim=-1
            )
            candidates_cpu = candidates.detach().float().cpu()
            candidate_trajectory_cpu = gather_cpu(candidates_cpu)
            mother_trajectory_cpu = candidates_cpu[
                rows_cpu, mother_cpu
            ][:, None]
            scale_cpu = candidates_cpu.new_tensor((10.0, 10.0, math.pi))
            candidate_scaled = candidate_trajectory_cpu / scale_cpu
            mother_scaled = mother_trajectory_cpu / scale_cpu
            candidate_scaled_steps = torch.diff(candidate_scaled, dim=-2)
            mother_scaled_steps = torch.diff(mother_scaled, dim=-2)
            tree_trajectory = torch.cat(
                (
                    candidate_scaled.flatten(2),
                    (candidate_scaled - mother_scaled).flatten(2),
                    candidate_scaled_steps.flatten(2),
                    (candidate_scaled_steps - mother_scaled_steps).flatten(2),
                ),
                dim=-1,
            )

            def dynamics(plans: torch.Tensor) -> torch.Tensor:
                normalized = plans / scale_cpu
                velocity = torch.diff(normalized, dim=-2) / 0.5
                acceleration = torch.diff(velocity, dim=-2) / 0.5
                jerk = torch.diff(acceleration, dim=-2) / 0.5
                return torch.cat(
                    (
                        velocity.flatten(-2),
                        acceleration.flatten(-2),
                        jerk.flatten(-2),
                    ),
                    dim=-1,
                )

            candidate_dynamics = dynamics(candidate_trajectory_cpu)
            mother_dynamics = dynamics(mother_trajectory_cpu)
            motion = torch.cat(
                (candidate_dynamics, candidate_dynamics - mother_dynamics),
                dim=-1,
            )
            meta_features = torch.cat(
                (
                    tree_meta_gain.detach().cpu()[..., None],
                    probability.detach().cpu()[..., None],
                    safety.detach().cpu(),
                ),
                dim=-1,
            )
            tree_features = torch.cat(
                (
                    latent,
                    tree_diagnostic,
                    tree_trajectory,
                    motion,
                    meta_features,
                ),
                dim=-1,
            )
            if tree_features.shape[-1] != 416:
                raise RuntimeError("tree meta scorer feature width drifted")
            flat = tree_features.numpy().reshape(-1, 416)
            tree_probability = sum(
                model.predict_proba(flat)[:, 1]
                for model in tree_models["positive"]
            ) / len(tree_models["positive"])
            tree_gain = sum(
                model.predict(flat) for model in tree_models["gain"]
            ) / len(tree_models["gain"])
            tree_safety = []
            for channel_models in tree_models["safety"]:
                channel = sum(
                    model.predict_proba(flat)[:, 1]
                    for model in channel_models
                ) / len(channel_models)
                tree_safety.append(channel)
            tree_probability = torch.as_tensor(
                tree_probability.reshape(batch, top.shape[1]),
                device=baseline.device,
                dtype=baseline.dtype,
            )
            tree_gain = torch.as_tensor(
                tree_gain.reshape(batch, top.shape[1]),
                device=baseline.device,
                dtype=baseline.dtype,
            )
            tree_safety = torch.stack(
                [
                    torch.as_tensor(
                        channel.reshape(batch, top.shape[1]),
                        device=baseline.device,
                        dtype=baseline.dtype,
                    )
                    for channel in tree_safety
                ],
                dim=-1,
            )
            policy = self.tree_meta_policy
            utility = tree_gain + policy["probability_weight"] * tree_probability
            candidate_position = utility.argmax(dim=1)
            proposed_gain = tree_gain[rows, candidate_position]
            proposed_probability = tree_probability[rows, candidate_position]
            proposed_safety = tree_safety[
                rows, candidate_position
            ].amin(dim=1)
            mother_gain = torch.zeros_like(proposed_gain)
        else:
            policy = self.candidate_set_policy
            candidate_position = gain.argmax(dim=1)
            proposed_gain = gain[rows, candidate_position]
            mother_gain = gain[rows, mother_position]
            proposed_probability = probability[rows, candidate_position]
            proposed_safety = safety[rows, candidate_position].amin(dim=1)
        proposed = top[rows, candidate_position]

        baseline_gap = baseline[rows, baseline_top] - baseline[rows, proposed]
        subscores = evaluation["candidate_subscores"]
        critical_delta = (
            subscores[rows, proposed, 0] - subscores[rows, baseline_top, 0]
        )
        predicted_safety_delta = (
            evaluation["predicted_safety"][rows, proposed]
            - evaluation["predicted_safety"][rows, baseline_top]
        )
        progress_delta = (
            subscores[rows, proposed, 2] - subscores[rows, baseline_top, 2]
        )

        # Raw proposals are sampled at 0.5 s.  The frozen bounds include a
        # fixed margin for the official scorer's Savitzky-Golay smoothing.
        proposed_plan = candidates.float()[rows, proposed]
        velocity = torch.diff(proposed_plan[..., :2], dim=1) / float(
            self.config.time_delta
        )
        acceleration = torch.diff(velocity, dim=1) / float(
            self.config.time_delta
        )
        jerk = torch.diff(acceleration, dim=1) / float(self.config.time_delta)
        max_acceleration = acceleration.norm(dim=-1).amax(dim=1)
        max_jerk = jerk.norm(dim=-1).amax(dim=1)
        comfort_safe = (
            (max_acceleration <= self.config.candidate_set_max_proxy_acceleration)
            & (max_jerk <= self.config.candidate_set_max_proxy_jerk)
        )
        accepted = (
            mother_in_top
            & (proposed != mother)
            & (proposed_gain >= policy["min_gain"])
            & (proposed_probability >= policy["min_probability"])
            & (proposed_safety >= policy["min_safety_probability"])
            & (baseline_gap <= policy["max_baseline_gap"])
            & (critical_delta >= policy.get("min_critical_delta", -0.02))
            & (
                predicted_safety_delta
                >= policy.get("min_predicted_safety_delta", -0.05)
            )
            & (progress_delta >= max(
                policy.get("min_progress_delta", 0.0),
                self.config.candidate_set_min_progress_delta,
            ))
            & comfort_safe
        )
        if tree_models is None:
            accepted = accepted & (
                proposed_gain - mother_gain >= policy["min_advantage"]
            )
        selected = torch.where(accepted, proposed, mother)
        factor_cascade_accepted = torch.zeros_like(accepted)
        factor_cascade_candidate = selected
        if (
            self.config.current_future_v2_selection
            and self.config.current_future_v2_cascade_on_candidate_set
        ):
            # r62 remains the mother decision.  Planning-JEPA may override it
            # only inside the frozen Drive-JEPA shortlist and only when both
            # learned factor deltas and frozen safety evidence pass the policy
            # selected before the sealed audit is opened.
            factor_top_k = min(
                self.config.current_future_v2_top_k, baseline.shape[1]
            )
            factor_top = baseline.topk(k=factor_top_k, dim=1).indices
            extra = (None,) * (
                evaluation["current_future_factor_deltas"].ndim - 2
            )
            factor_index = factor_top[(...,) + extra].expand(
                *factor_top.shape,
                *evaluation["current_future_factor_deltas"].shape[2:],
            )
            factor_prediction = evaluation[
                "current_future_factor_deltas"
            ].gather(1, factor_index)
            mother_prediction = evaluation["current_future_factor_deltas"][
                rows, selected
            ][:, None]
            factor_relative = factor_prediction - mother_prediction
            factor_critical = factor_relative[..., [0, 1, 3, 4]].amin(dim=-1)
            factor_utility = (
                factor_relative[..., 5]
                + self.config.current_future_v2_safety_weight * factor_critical
                + self.config.current_future_v2_progress_weight
                * factor_relative[..., 2]
                + self.config.current_future_v2_comfort_weight
                * factor_relative[..., 4]
            )
            factor_position = factor_utility.argmax(dim=1)
            factor_cascade_candidate = factor_top[rows, factor_position]
            chosen_factor = factor_relative[rows, factor_position]
            chosen_critical = factor_critical[rows, factor_position]
            factor_gap = (
                baseline[rows, selected]
                - baseline[rows, factor_cascade_candidate]
            )
            frozen_critical = (
                evaluation["candidate_subscores"][
                    rows, factor_cascade_candidate, 0
                ]
                - evaluation["candidate_subscores"][rows, selected, 0]
            )
            frozen_progress = (
                evaluation["candidate_subscores"][
                    rows, factor_cascade_candidate, 2
                ]
                - evaluation["candidate_subscores"][rows, selected, 2]
            )
            frozen_safety = (
                evaluation["predicted_safety"][rows, factor_cascade_candidate]
                - evaluation["predicted_safety"][rows, selected]
            )
            mother_in_factor_top = (factor_top == selected[:, None]).any(dim=1)
            factor_cascade_accepted = (
                mother_in_factor_top
                & (factor_cascade_candidate != selected)
                & (
                    chosen_factor[:, 5]
                    >= self.config.current_future_v2_min_score_delta
                )
                & (
                    chosen_critical
                    >= self.config.current_future_v2_min_critical_delta
                )
                & (
                    chosen_factor[:, 2]
                    >= self.config.current_future_v2_min_progress_delta
                )
                & (
                    chosen_factor[:, 4]
                    >= self.config.current_future_v2_min_comfort_delta
                )
                & (factor_gap <= self.config.current_future_v2_max_candidate_gap)
                & (
                    frozen_critical
                    >= self.config.current_future_v2_min_frozen_critical
                )
                & (
                    frozen_progress
                    >= self.config.current_future_v2_min_frozen_progress
                )
                & (
                    frozen_safety
                    >= self.config.current_future_v2_min_frozen_safety
                )
            )
            selected = torch.where(
                factor_cascade_accepted, factor_cascade_candidate, selected
            )
            accepted = selected != mother

        # r93: a deliberately small second-stage Planning-JEPA correction.
        # The rank head is trained only on NAVTRAIN.  It may replace the
        # frozen candidate-set winner only with a Drive-JEPA top-k proposal
        # that passes the same frozen-score and physical guards used by the
        # pre-registered offline study.  In particular, the improvement
        # probability chooses the proposal; no NAVTEST signal is consumed.
        rank_cascade_accepted = torch.zeros_like(accepted)
        rank_cascade_candidate = selected
        rank_mother = selected
        rank_probability = torch.zeros_like(baseline[rows, selected])
        rank_safety_probability = torch.zeros_like(rank_probability)
        chosen_progress_prediction = torch.zeros_like(rank_probability)
        rank_max_acceleration = torch.zeros_like(rank_probability)
        rank_max_jerk = torch.zeros_like(rank_probability)
        if (
            self.config.current_future_rank_selection
            and self.config.current_future_rank_cascade_on_candidate_set
        ):
            if "current_future_rank_prediction" not in evaluation:
                raise KeyError(
                    "current_future_rank_prediction is required by rank selection"
                )
            rank_mother = selected
            rank_top_k = min(
                self.config.current_future_rank_top_k, baseline.shape[1]
            )
            rank_top = baseline.topk(k=rank_top_k, dim=1).indices
            rank_prediction_all = evaluation["current_future_rank_prediction"]
            if self.config.current_future_rank_residual_fallback:
                reader_predictions = self.__dict__.get(
                    "_current_future_rank_reader_predictions", []
                )
                if len(reader_predictions) < 2:
                    raise RuntimeError(
                        "residual rank fallback requires one main reader and "
                        "one auxiliary reader"
                    )
                if self.config.current_future_rank_residual_main_is_mother:
                    # Run the complete frozen r94 checkpoint as the main model
                    # and load the newly trained residual as the first reader.
                    # This preserves every mother-side representation and
                    # avoids reconstructing r94 from a partial reader state.
                    fallback_prediction_all = reader_predictions[0]
                    residual_prediction_all = reader_predictions[1]
                else:
                    fallback_prediction_all = reader_predictions[1]
                    residual_prediction_all = reader_predictions[0]
                # Reproduce the frozen r94 rank stage exactly.  Its accepted
                # choice becomes the fallback for the new residual reader,
                # rather than falling all the way back to the pre-r94 mother.
                fallback_index = rank_top[..., None].expand(
                    *rank_top.shape, fallback_prediction_all.shape[-1]
                )
                fallback_prediction = fallback_prediction_all.gather(
                    1, fallback_index
                )
                fallback_utility = (
                    fallback_prediction[..., 5]
                    + self.config.current_future_rank_probability_weight
                    * fallback_prediction[..., 4]
                )
                fallback_position = fallback_utility.argmax(dim=1)
                fallback_candidate = rank_top[rows, fallback_position]
                fallback_chosen = fallback_prediction[rows, fallback_position]
                fallback_probability = fallback_chosen[:, 5].sigmoid()
                fallback_safety_probability = (
                    fallback_chosen[:, :4].sigmoid().amin(dim=1)
                )
                fallback_gap = (
                    baseline[rows, rank_mother]
                    - baseline[rows, fallback_candidate]
                )
                fallback_critical = (
                    evaluation["candidate_subscores"][
                        rows, fallback_candidate, 0
                    ]
                    - evaluation["candidate_subscores"][rows, rank_mother, 0]
                )
                fallback_progress = (
                    evaluation["candidate_subscores"][
                        rows, fallback_candidate, 2
                    ]
                    - evaluation["candidate_subscores"][rows, rank_mother, 2]
                )
                fallback_safety = (
                    evaluation["predicted_safety"][rows, fallback_candidate]
                    - evaluation["predicted_safety"][rows, rank_mother]
                )
                fallback_ttc = (
                    evaluation["candidate_subscores"][
                        rows, fallback_candidate, 3
                    ]
                    - evaluation["candidate_subscores"][rows, rank_mother, 3]
                )
                fallback_plan = candidates.float()[rows, fallback_candidate]
                fallback_velocity = torch.diff(
                    fallback_plan[..., :2], dim=1
                ) / float(self.config.time_delta)
                fallback_acceleration = torch.diff(
                    fallback_velocity, dim=1
                ) / float(self.config.time_delta)
                fallback_jerk = torch.diff(
                    fallback_acceleration, dim=1
                ) / float(self.config.time_delta)
                fallback_comfort_safe = (
                    fallback_acceleration.norm(dim=-1).amax(dim=1)
                    <= self.config.candidate_set_max_proxy_acceleration
                ) & (
                    fallback_jerk.norm(dim=-1).amax(dim=1)
                    <= self.config.candidate_set_max_proxy_jerk
                )
                fallback_accepted = (
                    (fallback_candidate != rank_mother)
                    & (fallback_probability >= 0.5)
                    & (fallback_chosen[:, 4] >= 0.0)
                    & (fallback_safety_probability >= 0.2)
                    & (
                        fallback_gap
                        <= self.config.current_future_rank_max_candidate_gap
                    )
                    & (
                        fallback_critical
                        >= self.config.current_future_rank_min_frozen_critical
                    )
                    & (
                        fallback_progress
                        >= self.config.current_future_rank_min_frozen_progress
                    )
                    & (
                        fallback_safety
                        >= self.config.current_future_rank_min_frozen_safety
                    )
                    & (
                        fallback_ttc
                        >= self.config.current_future_rank_min_frozen_ttc
                    )
                    & fallback_comfort_safe
                )
                rank_mother = torch.where(
                    fallback_accepted, fallback_candidate, rank_mother
                )
                # Match the r94-anchored training loss: every residual output
                # is interpreted relative to the frozen r94 choice.
                rank_prediction_all = residual_prediction_all
                rank_prediction_all = (
                    rank_prediction_all
                    - rank_prediction_all[rows, rank_mother, None]
                )
            rank_index = rank_top[..., None].expand(
                *rank_top.shape, rank_prediction_all.shape[-1]
            )
            rank_prediction = rank_prediction_all.gather(1, rank_index)
            progress_prediction_all = evaluation.get(
                "current_future_progress_prediction"
            )
            if self.config.current_future_progress_selection:
                if progress_prediction_all is None:
                    raise KeyError(
                        "current_future_progress_prediction is required by progress selection"
                    )
                progress_prediction = progress_prediction_all.gather(
                    1, rank_top[..., None]
                ).squeeze(-1)
                # The auxiliary head is trained on candidate progress relative
                # to the current rank mother.  Center the deployment utility
                # by that same mother prediction; using the raw absolute head
                # output creates a train/deploy offset and makes the weight
                # scene-dependent.
                mother_progress_prediction = progress_prediction_all.gather(
                    1, rank_mother[:, None, None]
                ).squeeze(-1)
                progress_prediction = (
                    progress_prediction - mother_progress_prediction
                )
            else:
                progress_prediction = None
            if self.config.current_future_rank_normalized_ensemble:
                # Normalize each frozen online head within the current scene.
                # This makes the auxiliary utility invariant to domain-level
                # score offsets/scales while keeping the learned rank head
                # and all proposal generation unchanged.
                rank_features = torch.stack(
                    (
                        baseline.gather(1, rank_top),
                        evaluation["candidate_subscores"][..., 5].gather(1, rank_top),
                        evaluation["structured_scores"].gather(1, rank_top),
                        evaluation["critic_value_probabilities"].gather(1, rank_top),
                        evaluation["predicted_safety"].gather(1, rank_top),
                        rank_prediction[..., 4],
                        rank_prediction[..., 5],
                    ),
                    dim=-1,
                )
                rank_features = (
                    rank_features - rank_features.mean(dim=1, keepdim=True)
                ) / rank_features.std(dim=1, keepdim=True).clamp_min(1e-5)
                rank_weights = rank_features.new_tensor(
                    (
                        self.config.current_future_rank_normalized_baseline_weight,
                        self.config.current_future_rank_normalized_subscore_weight,
                        self.config.current_future_rank_normalized_structured_weight,
                        self.config.current_future_rank_normalized_value_weight,
                        self.config.current_future_rank_normalized_safety_weight,
                        self.config.current_future_rank_normalized_gain_weight,
                        self.config.current_future_rank_normalized_probability_weight,
                    )
                )
                rank_utility = (rank_features * rank_weights).sum(dim=-1)
            else:
                rank_utility = (
                    rank_prediction[..., 5]
                    + self.config.current_future_rank_probability_weight
                    * rank_prediction[..., 4]
                )
            if progress_prediction is not None:
                rank_utility = rank_utility + (
                    self.config.current_future_progress_weight
                    * progress_prediction
                )
            rank_candidate_position = rank_utility.argmax(dim=1)
            rank_cascade_candidate = rank_top[rows, rank_candidate_position]
            chosen_rank_prediction = rank_prediction[
                rows, rank_candidate_position
            ]
            chosen_progress_prediction = (
                progress_prediction[rows, rank_candidate_position]
                if progress_prediction is not None
                else torch.zeros_like(chosen_rank_prediction[:, 0])
            )
            rank_probability = chosen_rank_prediction[:, 5].sigmoid()
            rank_safety_probability = chosen_rank_prediction[:, :4].sigmoid().amin(
                dim=1
            )
            rank_relative_safety = torch.ones_like(rank_safety_probability, dtype=torch.bool)
            if self.config.current_future_rank_normalized_ensemble and self.config.current_future_rank_normalized_relative_safety:
                mother_rank_prediction = rank_prediction_all[
                    rows, rank_mother, :4
                ]
                mother_rank_safety_probability = mother_rank_prediction.sigmoid().amin(dim=1)
                rank_relative_safety = rank_safety_probability >= mother_rank_safety_probability
            rank_gap = (
                baseline[rows, rank_mother]
                - baseline[rows, rank_cascade_candidate]
            )
            rank_frozen_critical = (
                evaluation["candidate_subscores"][
                    rows, rank_cascade_candidate, 0
                ]
                - evaluation["candidate_subscores"][rows, rank_mother, 0]
            )
            rank_frozen_progress = (
                evaluation["candidate_subscores"][
                    rows, rank_cascade_candidate, 2
                ]
                - evaluation["candidate_subscores"][rows, rank_mother, 2]
            )
            rank_frozen_safety = (
                evaluation["predicted_safety"][rows, rank_cascade_candidate]
                - evaluation["predicted_safety"][rows, rank_mother]
            )
            rank_frozen_ttc = (
                evaluation["candidate_subscores"][rows, rank_cascade_candidate, 3]
                - evaluation["candidate_subscores"][rows, rank_mother, 3]
            )
            rank_plan = candidates.float()[rows, rank_cascade_candidate]
            rank_velocity = torch.diff(rank_plan[..., :2], dim=1) / float(
                self.config.time_delta
            )
            rank_acceleration = torch.diff(rank_velocity, dim=1) / float(
                self.config.time_delta
            )
            rank_jerk = torch.diff(rank_acceleration, dim=1) / float(
                self.config.time_delta
            )
            rank_max_acceleration = rank_acceleration.norm(dim=-1).amax(dim=1)
            rank_max_jerk = rank_jerk.norm(dim=-1).amax(dim=1)
            rank_comfort_safe = (
                rank_max_acceleration
                <= self.config.candidate_set_max_proxy_acceleration
            ) & (rank_max_jerk <= self.config.candidate_set_max_proxy_jerk)
            # The historical proxy above sees only differences between future
            # waypoints, so it misses the transition from the current ego pose
            # to the first predicted pose.  Bound the residual plan's increase
            # in first-step lateral acceleration relative to the frozen r94
            # plan.  This uses trajectory geometry only (no metric labels).
            rank_mother_plan = candidates.float()[rows, rank_mother]
            delta_t = float(self.config.time_delta)

            def initial_lateral_acceleration(plan: torch.Tensor) -> torch.Tensor:
                first_speed = plan[:, 0, :2].norm(dim=-1) / delta_t
                first_heading = plan[:, 0, 2]
                first_yaw_rate = torch.atan2(
                    torch.sin(first_heading), torch.cos(first_heading)
                ).abs() / delta_t
                return first_speed * first_yaw_rate

            rank_initial_lateral_acceleration_increase = (
                initial_lateral_acceleration(rank_plan)
                - initial_lateral_acceleration(rank_mother_plan)
            )
            rank_transition_comfort_safe = (
                rank_initial_lateral_acceleration_increase
                <= self.config.current_future_rank_max_initial_lateral_acceleration_increase
            )
            # Sparse factor-consistency gate for residual deployment.  A new
            # plan is adopted only when a configurable number of the learned
            # NC/DAC/progress/TTC deltas agree that it is non-regressive.
            # Defaults disable this guard and preserve historical behavior.
            rank_positive_factor_count = (
                chosen_rank_prediction[:, :4]
                >= self.config.current_future_rank_positive_factor_floor
            ).sum(dim=1)
            rank_factor_consistent = (
                rank_positive_factor_count
                >= self.config.current_future_rank_min_positive_factor_count
            )
            rank_mother_in_top = (rank_top == rank_mother[:, None]).any(dim=1)
            rank_cascade_accepted = (
                rank_mother_in_top
                & (rank_cascade_candidate != rank_mother)
                & (
                    rank_probability
                    >= self.config.current_future_rank_min_probability
                )
                & (
                    chosen_rank_prediction[:, 4]
                    >= self.config.current_future_rank_min_gain
                )
                & (
                    rank_safety_probability
                    >= self.config.current_future_rank_min_safety_probability
                )
                & rank_relative_safety
                & (rank_gap <= self.config.current_future_rank_max_candidate_gap)
                & (
                    rank_frozen_critical
                    >= self.config.current_future_rank_min_frozen_critical
                )
                & (
                    rank_frozen_progress
                    >= self.config.current_future_rank_min_frozen_progress
                )
                & (
                    rank_frozen_safety
                    >= self.config.current_future_rank_min_frozen_safety
                )
                & (
                    rank_frozen_ttc
                    >= self.config.current_future_rank_min_frozen_ttc
                )
                & (
                    (chosen_progress_prediction >= self.config.current_future_progress_min_delta)
                    if progress_prediction is not None
                    else torch.ones_like(rank_cascade_accepted)
                )
                & rank_comfort_safe
                & rank_transition_comfort_safe
                & rank_factor_consistent
            )
            selected = torch.where(
                rank_cascade_accepted, rank_cascade_candidate, rank_mother
            )
            # Cross-fold residual consensus: the primary residual may change
            # the exact r94 mother only when every additional residual reader
            # independently accepts the very same candidate under identical
            # frozen-score, factor, and trajectory guards.  Readers are
            # trained on disjoint physical-log holdouts; no metric is queried
            # at inference time.  Disabled by default for historical parity.
            reader_predictions = self.__dict__.get(
                "_current_future_rank_reader_predictions", []
            )
            if (
                self.config.current_future_rank_residual_consensus
                and self.config.current_future_rank_residual_fallback
            ):
                if len(reader_predictions) < 3:
                    raise RuntimeError(
                        "residual consensus requires the exact mother and "
                        "at least two residual readers"
                    )
                residual_consensus = rank_cascade_accepted.clone()
                for auxiliary_prediction_all in reader_predictions[2:]:
                    auxiliary_prediction_all = (
                        auxiliary_prediction_all
                        - auxiliary_prediction_all[rows, rank_mother, None]
                    )
                    auxiliary_rank = auxiliary_prediction_all.gather(
                        1, rank_index
                    )
                    auxiliary_utility = (
                        auxiliary_rank[..., 5]
                        + self.config.current_future_rank_probability_weight
                        * auxiliary_rank[..., 4]
                    )
                    auxiliary_position = auxiliary_utility.argmax(dim=1)
                    auxiliary_candidate = rank_top[rows, auxiliary_position]
                    auxiliary_chosen = auxiliary_rank[rows, auxiliary_position]
                    auxiliary_probability = auxiliary_chosen[:, 5].sigmoid()
                    auxiliary_safety_probability = (
                        auxiliary_chosen[:, :4].sigmoid().amin(dim=1)
                    )
                    auxiliary_gap = (
                        baseline[rows, rank_mother]
                        - baseline[rows, auxiliary_candidate]
                    )
                    auxiliary_critical = (
                        evaluation["candidate_subscores"][
                            rows, auxiliary_candidate, 0
                        ]
                        - evaluation["candidate_subscores"][
                            rows, rank_mother, 0
                        ]
                    )
                    auxiliary_progress = (
                        evaluation["candidate_subscores"][
                            rows, auxiliary_candidate, 2
                        ]
                        - evaluation["candidate_subscores"][
                            rows, rank_mother, 2
                        ]
                    )
                    auxiliary_safety = (
                        evaluation["predicted_safety"][
                            rows, auxiliary_candidate
                        ]
                        - evaluation["predicted_safety"][rows, rank_mother]
                    )
                    auxiliary_ttc = (
                        evaluation["candidate_subscores"][
                            rows, auxiliary_candidate, 3
                        ]
                        - evaluation["candidate_subscores"][
                            rows, rank_mother, 3
                        ]
                    )
                    auxiliary_plan = candidates.float()[
                        rows, auxiliary_candidate
                    ]
                    auxiliary_velocity = torch.diff(
                        auxiliary_plan[..., :2], dim=1
                    ) / float(self.config.time_delta)
                    auxiliary_acceleration = torch.diff(
                        auxiliary_velocity, dim=1
                    ) / float(self.config.time_delta)
                    auxiliary_jerk = torch.diff(
                        auxiliary_acceleration, dim=1
                    ) / float(self.config.time_delta)
                    auxiliary_comfort_safe = (
                        auxiliary_acceleration.norm(dim=-1).amax(dim=1)
                        <= self.config.candidate_set_max_proxy_acceleration
                    ) & (
                        auxiliary_jerk.norm(dim=-1).amax(dim=1)
                        <= self.config.candidate_set_max_proxy_jerk
                    )
                    auxiliary_transition_safe = (
                        initial_lateral_acceleration(auxiliary_plan)
                        - initial_lateral_acceleration(rank_mother_plan)
                        <= self.config.current_future_rank_max_initial_lateral_acceleration_increase
                    )
                    auxiliary_factor_consistent = (
                        (
                            auxiliary_chosen[:, :4]
                            >= self.config.current_future_rank_positive_factor_floor
                        ).sum(dim=1)
                        >= self.config.current_future_rank_min_positive_factor_count
                    )
                    auxiliary_accepted = (
                        rank_mother_in_top
                        & (auxiliary_candidate != rank_mother)
                        & (
                            auxiliary_probability
                            >= self.config.current_future_rank_min_probability
                        )
                        & (
                            auxiliary_chosen[:, 4]
                            >= self.config.current_future_rank_min_gain
                        )
                        & (
                            auxiliary_safety_probability
                            >= self.config.current_future_rank_min_safety_probability
                        )
                        & (
                            auxiliary_gap
                            <= self.config.current_future_rank_max_candidate_gap
                        )
                        & (
                            auxiliary_critical
                            >= self.config.current_future_rank_min_frozen_critical
                        )
                        & (
                            auxiliary_progress
                            >= self.config.current_future_rank_min_frozen_progress
                        )
                        & (
                            auxiliary_safety
                            >= self.config.current_future_rank_min_frozen_safety
                        )
                        & (
                            auxiliary_ttc
                            >= self.config.current_future_rank_min_frozen_ttc
                        )
                        & auxiliary_comfort_safe
                        & auxiliary_transition_safe
                        & auxiliary_factor_consistent
                    )
                    residual_consensus = (
                        residual_consensus
                        & auxiliary_accepted
                        & (auxiliary_candidate == rank_cascade_candidate)
                    )
                rank_cascade_accepted = residual_consensus
                selected = torch.where(
                    rank_cascade_accepted, rank_cascade_candidate, rank_mother
                )
            # r123: keep the validated single-reader decision whenever it
            # agrees with at least one auxiliary reader.  The median-reader
            # proposal is used only when all readers disagree.  This is a
            # label-free inference gate; the auxiliary readers are loaded from
            # NAVTRAIN-screened checkpoints and no simulator score is used.
            if (
                self.config.current_future_rank_consensus_fallback
                and not self.config.current_future_rank_residual_fallback
                and len(reader_predictions) >= 2
                and not self.config.current_future_rank_normalized_ensemble
            ):
                def reader_choice(
                    reader_prediction: torch.Tensor,
                    min_probability: float,
                    min_gain: float,
                    min_safety_probability: float,
                ) -> torch.Tensor:
                    reader_rank = reader_prediction.gather(1, rank_index)
                    reader_utility = (
                        reader_rank[..., 5]
                        + self.config.current_future_rank_probability_weight
                        * reader_rank[..., 4]
                    )
                    reader_position = reader_utility.argmax(dim=1)
                    reader_candidate = rank_top[rows, reader_position]
                    reader_chosen = reader_rank[rows, reader_position]
                    reader_probability = reader_chosen[:, 5].sigmoid()
                    reader_safety = reader_chosen[:, :4].sigmoid().amin(dim=1)
                    reader_gap = (
                        baseline[rows, rank_mother]
                        - baseline[rows, reader_candidate]
                    )
                    reader_critical = (
                        evaluation["candidate_subscores"][
                            rows, reader_candidate, 0
                        ]
                        - evaluation["candidate_subscores"][rows, rank_mother, 0]
                    )
                    reader_progress = (
                        evaluation["candidate_subscores"][
                            rows, reader_candidate, 2
                        ]
                        - evaluation["candidate_subscores"][rows, rank_mother, 2]
                    )
                    reader_frozen_safety = (
                        evaluation["predicted_safety"][rows, reader_candidate]
                        - evaluation["predicted_safety"][rows, rank_mother]
                    )
                    reader_plan = candidates.float()[rows, reader_candidate]
                    reader_velocity = torch.diff(
                        reader_plan[..., :2], dim=1
                    ) / float(self.config.time_delta)
                    reader_acceleration = torch.diff(
                        reader_velocity, dim=1
                    ) / float(self.config.time_delta)
                    reader_jerk = torch.diff(
                        reader_acceleration, dim=1
                    ) / float(self.config.time_delta)
                    reader_comfort_safe = (
                        reader_acceleration.norm(dim=-1).amax(dim=1)
                        <= self.config.candidate_set_max_proxy_acceleration
                    ) & (
                        reader_jerk.norm(dim=-1).amax(dim=1)
                        <= self.config.candidate_set_max_proxy_jerk
                    )
                    reader_accepted = (
                        rank_mother_in_top
                        & (reader_candidate != rank_mother)
                        & (reader_probability >= min_probability)
                        & (reader_chosen[:, 4] >= min_gain)
                        & (reader_safety >= min_safety_probability)
                        & (reader_gap <= self.config.current_future_rank_max_candidate_gap)
                        & (
                            reader_critical
                            >= self.config.current_future_rank_min_frozen_critical
                        )
                        & (
                            reader_progress
                            >= self.config.current_future_rank_min_frozen_progress
                        )
                        & (
                            reader_frozen_safety
                            >= self.config.current_future_rank_min_frozen_safety
                        )
                        & reader_comfort_safe
                    )
                    return torch.where(
                        reader_accepted, reader_candidate, rank_mother
                    )

                main_choice = reader_choice(
                    reader_predictions[0],
                    self.config.current_future_rank_consensus_main_min_probability,
                    self.config.current_future_rank_consensus_main_min_gain,
                    self.config.current_future_rank_consensus_main_min_safety_probability,
                )
                auxiliary_choices = [
                    reader_choice(
                        prediction,
                        self.config.current_future_rank_min_probability,
                        self.config.current_future_rank_min_gain,
                        self.config.current_future_rank_min_safety_probability,
                    )
                    for prediction in reader_predictions[1:]
                ]
                agrees_with_auxiliary = torch.stack(
                    [main_choice == choice for choice in auxiliary_choices], dim=0
                ).any(dim=0)
                selected = torch.where(
                    agrees_with_auxiliary, main_choice, selected
                )
                rank_cascade_candidate = selected
                rank_cascade_accepted = selected != rank_mother
            accepted = selected != mother
        post_r94_progress_accepted = torch.zeros_like(mother, dtype=torch.bool)
        post_r94_progress_prediction = torch.zeros_like(baseline[rows, selected])
        if self.config.post_r94_progress_expansion:
            (
                selected,
                post_r94_progress_accepted,
                post_r94_progress_prediction,
            ) = self._post_r94_progress_expansion(
                evaluation, baseline, candidates, selected
            )
            accepted = selected != mother
        post_r94_accepted = torch.zeros_like(mother, dtype=torch.bool)
        post_r94_safety_probability = torch.zeros_like(baseline[rows, selected])
        if self.__dict__.get("post_r94_selector_model") is not None:
            selector_mother = selected
            primary_selected, primary_accepted, primary_safety = (
                self._post_r94_selection(
                    evaluation, baseline, candidates, selector_mother
                )
            )
            consensus_model = self.__dict__.get("post_r94_selector_consensus_model")
            if consensus_model is None:
                selected = primary_selected
                post_r94_accepted = primary_accepted
                post_r94_safety_probability = primary_safety
            else:
                secondary_selected, secondary_accepted, secondary_safety = (
                    self._post_r94_selection(
                        evaluation,
                        baseline,
                        candidates,
                        selector_mother,
                        model=consensus_model,
                        projection=self.post_r94_selector_consensus_projection,
                        policy=self.post_r94_selector_consensus_policy,
                        feature_names=self.post_r94_selector_consensus_feature_names,
                    )
                )
                post_r94_accepted = (
                    primary_accepted
                    & secondary_accepted
                    & (primary_selected == secondary_selected)
                )
                selected = torch.where(
                    post_r94_accepted, primary_selected, selector_mother
                )
                post_r94_safety_probability = torch.minimum(
                    primary_safety, secondary_safety
                )
            accepted = selected != mother
        planning_jepa_diagnostics: Dict[str, torch.Tensor] = {}
        if self.planning_jepa_verifier_scale.numel() > 0:
            selected, _, planning_jepa_diagnostics = (
                self._planning_jepa_residual_selection(
                    evaluation, baseline, candidates, selected, baseline_top
                )
            )
            accepted = selected != mother
        return selected, accepted, {
            "candidate_set_gain": proposed_gain,
            "candidate_set_probability": proposed_probability,
            "candidate_set_safety": proposed_safety,
            "candidate_set_tree_meta": torch.full_like(
                accepted, tree_models is not None
            ),
            "candidate_set_max_acceleration": max_acceleration,
            "candidate_set_max_jerk": max_jerk,
            "current_future_cascade_accepted": factor_cascade_accepted,
            "current_future_cascade_candidate": factor_cascade_candidate,
            "current_future_rank_cascade_accepted": rank_cascade_accepted,
            "current_future_rank_cascade_candidate": rank_cascade_candidate,
            "current_future_rank_mother": rank_mother,
            "current_future_rank_probability": rank_probability,
            "current_future_rank_safety_probability": rank_safety_probability,
            "current_future_progress_prediction": chosen_progress_prediction,
            "current_future_rank_max_acceleration": rank_max_acceleration,
            "current_future_rank_max_jerk": rank_max_jerk,
            "post_r94_progress_accepted": post_r94_progress_accepted,
            "post_r94_progress_prediction": post_r94_progress_prediction,
            "post_r94_accepted": post_r94_accepted,
            "post_r94_safety_probability": post_r94_safety_probability,
            **planning_jepa_diagnostics,
        }

    def _conservative_select(
        self,
        evaluation: Dict[str, torch.Tensor],
        base_scores: torch.Tensor,
        candidates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        baseline = base_scores[..., -1] if base_scores.ndim == 3 else base_scores
        batch = baseline.shape[0]
        rows = torch.arange(batch, device=baseline.device)
        baseline_top = baseline.argmax(dim=1)
        top_k = min(self.config.conservative_top_k, baseline.shape[1])
        if top_k < baseline.shape[1]:
            shortlist = torch.zeros_like(baseline, dtype=torch.bool)
            shortlist.scatter_(1, baseline.topk(k=top_k, dim=1).indices, True)
            conservative_top = evaluation["conservative_scores"].masked_fill(
                ~shortlist, float("-inf")
            ).argmax(dim=1)
        else:
            conservative_top = evaluation["conservative_scores"].argmax(dim=1)
        top_values = baseline.topk(k=min(2, baseline.shape[1]), dim=1).values
        baseline_margin = (
            top_values[:, 0] - top_values[:, 1]
            if top_values.shape[1] == 2
            else torch.full_like(top_values[:, 0], float("inf"))
        )
        advantage = (
            evaluation["conservative_scores"][rows, conservative_top]
            - evaluation["conservative_scores"][rows, baseline_top]
        )
        confidence = evaluation["critic_confidence"][rows, conservative_top]
        safety_delta = (
            evaluation["predicted_safety"][rows, conservative_top]
            - evaluation["predicted_safety"][rows, baseline_top]
        )
        critical_probabilities = evaluation["candidate_subscores"][..., [0, 1, 3]]
        critical_delta = (
            critical_probabilities[rows, conservative_top]
            - critical_probabilities[rows, baseline_top]
        ).amin(dim=1)
        if self.config.risk_aware_v3_selection:
            prediction = evaluation["risk_aware_prediction"]
            safety_probability = torch.sigmoid(prediction[..., :4]).amin(dim=-1)
            gain = prediction[..., 4]
            improvement_probability = torch.sigmoid(prediction[..., 5])
            compatibility = improvement_probability * safety_probability
            top_k = min(self.config.risk_aware_v3_top_k, baseline.shape[1])
            shortlist = torch.zeros_like(baseline, dtype=torch.bool)
            shortlist.scatter_(1, baseline.topk(k=top_k, dim=1).indices, True)
            candidate = compatibility.masked_fill(~shortlist, float("-inf")).argmax(dim=1)
            candidate_gap = baseline[rows, baseline_top] - baseline[rows, candidate]
            chosen_safety = safety_probability[rows, candidate]
            chosen_gain = gain[rows, candidate]
            chosen_probability = improvement_probability[rows, candidate]
            accepted = (
                (candidate != baseline_top)
                & (chosen_probability >= self.config.risk_aware_v3_min_probability)
                & (chosen_safety >= self.config.risk_aware_v3_min_safety_probability)
                & (chosen_gain >= self.config.risk_aware_v3_min_gain)
                & (candidate_gap <= self.config.risk_aware_v3_max_candidate_gap)
            )
            selected = torch.where(accepted, candidate, baseline_top)
            gated_scores = baseline.clone()
            gated_scores.scatter_(
                1,
                selected[:, None],
                (baseline[rows, baseline_top] + accepted.to(baseline.dtype) * 1e-4)[:, None],
            )
            zeros = torch.zeros_like(accepted)
            return {
                "baseline_selected": baseline_top,
                "conservative_selected": candidate,
                "selected": selected,
                "selection_accepted": accepted,
                "selection_advantage": chosen_probability,
                "baseline_margin": baseline_margin,
                "selection_confidence": chosen_probability,
                "selection_safety_delta": chosen_safety,
                "selection_critical_delta": chosen_safety,
                "selection_progress_delta": chosen_gain,
                "selection_factor_critical_delta": chosen_safety,
                "selection_factor_progress_delta": chosen_gain,
                "pairwise_enabled": zeros,
                "pairwise_candidate": baseline_top,
                "pairwise_vetoed": zeros,
                "pairwise_rescued": zeros,
                "pairwise_preference": compatibility,
                "gated_scores": gated_scores,
            }
        if (
            self.config.current_future_v2_selection
            and not self.config.current_future_v2_cascade_on_candidate_set
        ):
            predicted_delta = evaluation["current_future_factor_deltas"]
            # Rank by the predicted PDMS change while requiring independent
            # factor evidence.  The small auxiliary term resolves near ties
            # without allowing progress to hide a safety regression.
            factor_utility = (
                2.0 * predicted_delta[..., 0]
                + 2.0 * predicted_delta[..., 1]
                + predicted_delta[..., 2]
                + 2.0 * predicted_delta[..., 3]
                + predicted_delta[..., 4]
            ) / 8.0
            compatibility = predicted_delta[..., 5] + 0.1 * factor_utility
            interaction_top_k = min(
                self.config.current_future_v2_top_k, baseline.shape[1]
            )
            shortlist = torch.zeros_like(baseline, dtype=torch.bool)
            shortlist.scatter_(
                1, baseline.topk(k=interaction_top_k, dim=1).indices, True
            )
            interaction_candidate = compatibility.masked_fill(
                ~shortlist, float("-inf")
            ).argmax(dim=1)
            chosen_delta = predicted_delta[rows, interaction_candidate]
            interaction_critical = chosen_delta[:, [0, 1, 3]].amin(dim=1)
            candidate_gap = (
                baseline[rows, baseline_top]
                - baseline[rows, interaction_candidate]
            )
            accepted = (
                (interaction_candidate != baseline_top)
                & (
                    chosen_delta[:, 5]
                    >= self.config.current_future_v2_min_score_delta
                )
                & (
                    interaction_critical
                    >= self.config.current_future_v2_min_critical_delta
                )
                & (
                    chosen_delta[:, 2]
                    >= self.config.current_future_v2_min_progress_delta
                )
                & (
                    chosen_delta[:, 4]
                    >= self.config.current_future_v2_min_comfort_delta
                )
                & (
                    candidate_gap
                    <= self.config.current_future_v2_max_candidate_gap
                )
            )
            selected = torch.where(accepted, interaction_candidate, baseline_top)
            gated_scores = baseline.clone()
            selected_score = baseline[rows, baseline_top] + accepted.to(
                baseline.dtype
            ) * 1e-4
            gated_scores.scatter_(1, selected[:, None], selected_score[:, None])
            zeros = torch.zeros_like(accepted)
            return {
                "baseline_selected": baseline_top,
                "conservative_selected": interaction_candidate,
                "selected": selected,
                "selection_accepted": accepted,
                "selection_advantage": chosen_delta[:, 5],
                "baseline_margin": baseline_margin,
                "selection_confidence": torch.ones_like(chosen_delta[:, 5]),
                "selection_safety_delta": interaction_critical,
                "selection_critical_delta": interaction_critical,
                "selection_progress_delta": chosen_delta[:, 2],
                "selection_factor_critical_delta": interaction_critical,
                "selection_factor_progress_delta": chosen_delta[:, 2],
                "pairwise_enabled": zeros,
                "pairwise_candidate": baseline_top,
                "pairwise_vetoed": zeros,
                "pairwise_rescued": zeros,
                "pairwise_preference": compatibility,
                "gated_scores": gated_scores,
            }
        if self.latent_verifier is not None:
            if candidates is None:
                raise ValueError("latent-verifier selection requires candidate trajectories")
            preference = self._latent_verifier_preferences(
                evaluation, baseline, candidates, baseline_top, baseline_margin
            )
            policy = self.latent_verifier_policy
            verifier_top_k = min(int(policy["top_k"]), baseline.shape[1])
            shortlist = torch.zeros_like(baseline, dtype=torch.bool)
            shortlist.scatter_(
                1, baseline.topk(k=verifier_top_k, dim=1).indices, True
            )
            verifier_candidate = preference.masked_fill(
                ~shortlist, float("-inf")
            ).argmax(dim=1)
            verifier_advantage = preference[rows, verifier_candidate]
            verifier_critical_delta = (
                critical_probabilities[rows, verifier_candidate]
                - critical_probabilities[rows, baseline_top]
            ).amin(dim=1)
            verifier_safety_delta = (
                evaluation["predicted_safety"][rows, verifier_candidate]
                - evaluation["predicted_safety"][rows, baseline_top]
            )
            verifier_progress_delta = (
                evaluation["candidate_subscores"][rows, verifier_candidate, 2]
                - evaluation["candidate_subscores"][rows, baseline_top, 2]
            )
            # Finite neutral diagnostics preserve compatibility with the
            # single-head artifact while satisfying optional finite checks.
            factor_critical_delta = torch.zeros_like(verifier_advantage)
            factor_progress_delta = torch.zeros_like(verifier_advantage)
            if self.latent_verifier_factor is not None:
                factor_deltas = self._latent_verifier_factor_deltas(
                    evaluation, baseline, candidates, baseline_top, baseline_margin
                )
                chosen_factor_deltas = factor_deltas[rows, verifier_candidate]
                factor_critical_delta = chosen_factor_deltas[:, [0, 1, 3]].amin(dim=1)
                factor_progress_delta = chosen_factor_deltas[:, 2]
            accepted = (
                (verifier_advantage >= policy["min_advantage"])
                & (baseline_margin <= policy["max_baseline_margin"])
                & (verifier_critical_delta >= policy["min_critical_delta"])
                & (verifier_safety_delta >= policy["min_safety_delta"])
                & (verifier_progress_delta >= policy["min_progress_delta"])
            )
            if self.latent_verifier_factor is not None:
                accepted = (
                    accepted
                    & (
                        factor_critical_delta
                        >= policy["min_factor_critical_delta"]
                    )
                    & (
                        factor_progress_delta
                        >= policy["min_factor_progress_delta"]
                    )
                )
            residual_veto_probability = self._latent_residual_veto_probabilities(
                evaluation,
                baseline,
                candidates,
                baseline_top,
                baseline_margin,
                verifier_candidate,
                preference,
            )
            if self.latent_residual_veto is not None:
                accepted = accepted & (
                    residual_veto_probability
                    >= self.latent_residual_veto_threshold
                )
            if self.config.latent_risk_veto:
                risk_prediction = evaluation["risk_aware_prediction"]
                risk_safety = torch.sigmoid(
                    risk_prediction[rows, verifier_candidate, :4]
                ).amin(dim=-1)
                risk_gain = risk_prediction[rows, verifier_candidate, 4]
                risk_probability = torch.sigmoid(
                    risk_prediction[rows, verifier_candidate, 5]
                )
                accepted = (
                    accepted
                    & (
                        risk_probability
                        >= self.config.risk_aware_v3_min_probability
                    )
                    & (
                        risk_safety
                        >= self.config.risk_aware_v3_min_safety_probability
                    )
                    & (risk_gain >= self.config.risk_aware_v3_min_gain)
                )
            selected = torch.where(accepted, verifier_candidate, baseline_top)
            candidate_set_diagnostics: Dict[str, torch.Tensor] = {}
            if self.__dict__.get("candidate_set_scorers", []):
                selected, _, candidate_set_diagnostics = self._candidate_set_selection(
                    evaluation, baseline, candidates, selected, baseline_top
                )
                accepted = selected != baseline_top
            gated_scores = baseline.clone()
            selected_score = baseline[rows, baseline_top] + accepted.to(
                baseline.dtype
            ) * 1e-4
            gated_scores.scatter_(1, selected[:, None], selected_score[:, None])
            zeros = torch.zeros_like(accepted)
            return {
                "baseline_selected": baseline_top,
                "conservative_selected": verifier_candidate,
                "selected": selected,
                "selection_accepted": accepted,
                "selection_advantage": verifier_advantage,
                "baseline_margin": baseline_margin,
                "selection_confidence": torch.minimum(
                    evaluation["critic_confidence"][rows, verifier_candidate],
                    residual_veto_probability,
                ),
                "selection_safety_delta": verifier_safety_delta,
                "selection_critical_delta": verifier_critical_delta,
                "selection_progress_delta": verifier_progress_delta,
                "selection_factor_critical_delta": factor_critical_delta,
                "selection_factor_progress_delta": factor_progress_delta,
                "pairwise_enabled": zeros,
                "pairwise_candidate": baseline_top,
                "pairwise_vetoed": zeros,
                "pairwise_rescued": zeros,
                "pairwise_preference": preference,
                "gated_scores": gated_scores,
                **candidate_set_diagnostics,
            }
        if (
            self.config.outcome_delta_selection
            or self.config.outcome_action_delta_selection
            or self.config.temporal_action_delta_selection
        ):
            if self.config.temporal_action_delta_selection:
                predicted_delta = evaluation["temporal_action_factor_deltas"]
            elif self.config.outcome_action_delta_selection:
                predicted_delta = evaluation["outcome_action_factor_deltas"]
            else:
                predicted_delta = evaluation["outcome_factor_deltas"]
            delta_top_k = min(
                self.config.outcome_delta_selection_top_k, baseline.shape[1]
            )
            shortlist = torch.zeros_like(baseline, dtype=torch.bool)
            shortlist.scatter_(
                1, baseline.topk(k=delta_top_k, dim=1).indices, True
            )
            delta_candidate = predicted_delta[..., -1].masked_fill(
                ~shortlist, float("-inf")
            ).argmax(dim=1)
            candidate_gap = (
                baseline[rows, baseline_top] - baseline[rows, delta_candidate]
            )
            chosen_delta = predicted_delta[rows, delta_candidate]
            delta_critical = chosen_delta[:, [0, 1, 3]].amin(dim=1)
            accepted = (
                (delta_candidate != baseline_top)
                & (
                    chosen_delta[:, -1]
                    >= self.config.outcome_delta_min_score_delta
                )
                & (
                    delta_critical
                    >= self.config.outcome_delta_min_critical_delta
                )
                & (
                    chosen_delta[:, 4]
                    >= self.config.outcome_delta_min_comfort_delta
                )
                & (
                    candidate_gap
                    <= self.config.outcome_delta_max_candidate_gap
                )
            )
            selected = torch.where(accepted, delta_candidate, baseline_top)
            gated_scores = baseline.clone()
            selected_score = baseline[rows, baseline_top] + accepted.to(
                baseline.dtype
            ) * 1e-4
            gated_scores.scatter_(1, selected[:, None], selected_score[:, None])
            zeros = torch.zeros_like(accepted)
            return {
                "baseline_selected": baseline_top,
                "conservative_selected": delta_candidate,
                "selected": selected,
                "selection_accepted": accepted,
                "selection_advantage": chosen_delta[:, -1],
                "baseline_margin": baseline_margin,
                "selection_confidence": torch.ones_like(chosen_delta[:, -1]),
                "selection_safety_delta": delta_critical,
                "selection_critical_delta": delta_critical,
                "selection_progress_delta": chosen_delta[:, 2],
                "pairwise_enabled": zeros,
                "pairwise_candidate": baseline_top,
                "pairwise_vetoed": zeros,
                "pairwise_rescued": zeros,
                "pairwise_preference": predicted_delta[..., -1],
                "gated_scores": gated_scores,
            }
        if self.planning_switch is not None:
            if candidates is None:
                raise ValueError("planning-switch selection requires candidate trajectories")
            probability, ranked = self._planning_switch_probabilities(
                evaluation, baseline, candidates, baseline_top, baseline_margin
            )
            policy = self.planning_switch["policy"]
            usable = int(policy["top_k"]) - 1
            chosen_rank = probability[:, :usable].argmax(dim=1)
            switch_candidate = ranked[rows, chosen_rank]
            switch_confidence = probability[rows, chosen_rank]
            candidate_gap = (
                baseline[rows, baseline_top]
                - baseline[rows, switch_candidate]
            )
            switch_critical_delta = (
                critical_probabilities[rows, switch_candidate]
                - critical_probabilities[rows, baseline_top]
            ).amin(dim=1)
            switch_safety_delta = (
                evaluation["predicted_safety"][rows, switch_candidate]
                - evaluation["predicted_safety"][rows, baseline_top]
            )
            switch_progress_delta = (
                evaluation["candidate_subscores"][rows, switch_candidate, 2]
                - evaluation["candidate_subscores"][rows, baseline_top, 2]
            )
            accepted = (
                (switch_confidence >= float(policy["min_probability"]))
                & (candidate_gap <= float(policy["max_candidate_gap"]))
                & (switch_critical_delta >= float(policy["min_critical_delta"]))
                & (switch_safety_delta >= float(policy["min_safety_delta"]))
                & (switch_progress_delta >= float(policy["min_progress_delta"]))
            )
            selected = torch.where(accepted, switch_candidate, baseline_top)
            gated_scores = baseline.clone()
            selected_score = baseline[rows, baseline_top] + accepted.to(
                baseline.dtype
            ) * 1e-4
            gated_scores.scatter_(1, selected[:, None], selected_score[:, None])
            zeros = torch.zeros_like(accepted)
            return {
                "baseline_selected": baseline_top,
                "conservative_selected": switch_candidate,
                "selected": selected,
                "selection_accepted": accepted,
                "selection_advantage": switch_confidence,
                "baseline_margin": baseline_margin,
                "selection_confidence": switch_confidence,
                "selection_safety_delta": switch_safety_delta,
                "selection_critical_delta": switch_critical_delta,
                "selection_progress_delta": switch_progress_delta,
                "pairwise_enabled": zeros,
                "pairwise_candidate": baseline_top,
                "pairwise_vetoed": zeros,
                "pairwise_rescued": zeros,
                "pairwise_preference": probability,
                "gated_scores": gated_scores,
            }
        if self.planning_jepa_verifier_scale.numel() > 0:
            if candidates is None:
                raise ValueError(
                    "planning-JEPA verifier requires candidate trajectories"
                )
            absolute_preference = self._planning_jepa_preferences(
                evaluation, baseline, candidates, baseline_top, baseline_margin
            )
            preference = (
                absolute_preference
                - absolute_preference[rows, baseline_top, None]
            )
            policy = self.planning_jepa_verifier_policy
            top_k = min(int(policy["top_k"]), baseline.shape[1])
            shortlist = torch.zeros_like(baseline, dtype=torch.bool)
            shortlist.scatter_(
                1, baseline.topk(k=top_k, dim=1).indices, True
            )
            verifier_candidate = preference.masked_fill(
                ~shortlist, float("-inf")
            ).argmax(dim=1)
            verifier_advantage = preference[rows, verifier_candidate]
            verifier_critical_delta = (
                critical_probabilities[rows, verifier_candidate]
                - critical_probabilities[rows, baseline_top]
            ).amin(dim=1)
            verifier_safety_delta = (
                evaluation["predicted_safety"][rows, verifier_candidate]
                - evaluation["predicted_safety"][rows, baseline_top]
            )
            verifier_progress_delta = (
                evaluation["candidate_subscores"][rows, verifier_candidate, 2]
                - evaluation["candidate_subscores"][rows, baseline_top, 2]
            )
            accepted = (
                (verifier_advantage >= policy["min_advantage"])
                & (baseline_margin <= policy["max_baseline_margin"])
                & (verifier_critical_delta >= policy["min_critical_delta"])
                & (verifier_safety_delta >= policy["min_safety_delta"])
                & (verifier_progress_delta >= policy["min_progress_delta"])
            )
            selected = torch.where(accepted, verifier_candidate, baseline_top)
            gated_scores = baseline.clone()
            selected_score = baseline[rows, baseline_top] + accepted.to(
                baseline.dtype
            ) * 1e-4
            gated_scores.scatter_(1, selected[:, None], selected_score[:, None])
            zeros = torch.zeros_like(accepted)
            return {
                "baseline_selected": baseline_top,
                "conservative_selected": verifier_candidate,
                "selected": selected,
                "selection_accepted": accepted,
                "selection_advantage": verifier_advantage,
                "baseline_margin": baseline_margin,
                "selection_confidence": evaluation["critic_confidence"][
                    rows, verifier_candidate
                ],
                "selection_safety_delta": verifier_safety_delta,
                "selection_critical_delta": verifier_critical_delta,
                "selection_progress_delta": verifier_progress_delta,
                "pairwise_enabled": zeros,
                "pairwise_candidate": baseline_top,
                "pairwise_vetoed": zeros,
                "pairwise_rescued": zeros,
                "pairwise_preference": preference,
                "gated_scores": gated_scores,
            }
        if self.geometry_calibrator is not None:
            if candidates is None:
                raise ValueError("geometry selection requires candidate trajectories")
            absolute_preference = self._geometry_preferences(
                evaluation, baseline, candidates, baseline_top, baseline_margin
            )
            preference = (
                absolute_preference
                - absolute_preference[rows, baseline_top, None]
            )
            policy = self.geometry_policy
            geometry_top_k = min(int(policy["top_k"]), baseline.shape[1])
            shortlist = torch.zeros_like(baseline, dtype=torch.bool)
            shortlist.scatter_(
                1, baseline.topk(k=geometry_top_k, dim=1).indices, True
            )
            geometry_candidate = preference.masked_fill(
                ~shortlist, float("-inf")
            ).argmax(dim=1)
            geometry_advantage = preference[rows, geometry_candidate]
            geometry_critical_delta = (
                critical_probabilities[rows, geometry_candidate]
                - critical_probabilities[rows, baseline_top]
            ).amin(dim=1)
            geometry_safety_delta = (
                evaluation["predicted_safety"][rows, geometry_candidate]
                - evaluation["predicted_safety"][rows, baseline_top]
            )
            geometry_progress_delta = (
                evaluation["candidate_subscores"][rows, geometry_candidate, 2]
                - evaluation["candidate_subscores"][rows, baseline_top, 2]
            )
            accepted = (
                (geometry_advantage >= policy["min_advantage"])
                & (
                    absolute_preference[rows, geometry_candidate]
                    >= policy.get("min_candidate_probability", float("-inf"))
                )
                & (baseline_margin <= policy["max_baseline_margin"])
                & (geometry_critical_delta >= policy["min_critical_delta"])
                & (geometry_safety_delta >= policy["min_safety_delta"])
                & (
                    geometry_progress_delta
                    >= policy.get("min_progress_delta", float("-inf"))
                )
            )
            selected = torch.where(accepted, geometry_candidate, baseline_top)
            gated_scores = baseline.clone()
            selected_score = baseline[rows, baseline_top] + accepted.to(
                baseline.dtype
            ) * 1e-4
            gated_scores.scatter_(1, selected[:, None], selected_score[:, None])
            zeros = torch.zeros_like(accepted)
            return {
                "baseline_selected": baseline_top,
                "conservative_selected": geometry_candidate,
                "selected": selected,
                "selection_accepted": accepted,
                "selection_advantage": geometry_advantage,
                "baseline_margin": baseline_margin,
                "selection_confidence": evaluation["critic_confidence"][rows, geometry_candidate],
                "selection_safety_delta": geometry_safety_delta,
                "selection_critical_delta": geometry_critical_delta,
                "selection_progress_delta": geometry_progress_delta,
                "pairwise_enabled": zeros,
                "pairwise_candidate": baseline_top,
                "pairwise_vetoed": zeros,
                "pairwise_rescued": zeros,
                "pairwise_preference": preference,
                "gated_scores": gated_scores,
            }
        accepted = (
            (advantage >= self.config.conservative_min_advantage)
            & (confidence >= self.config.conservative_min_confidence)
            & (baseline_margin <= self.config.conservative_max_baseline_margin)
            & (safety_delta >= self.config.conservative_safety_margin)
            & (critical_delta >= self.config.conservative_min_critical_delta)
        )
        selected = torch.where(accepted, conservative_top, baseline_top)
        gated_scores = torch.where(
            accepted[:, None], evaluation["conservative_scores"], baseline
        )
        pairwise_enabled = self.pairwise_weights.numel() > 0
        pairwise_candidate = baseline_top
        pairwise_vetoed = torch.zeros_like(accepted)
        pairwise_rescued = torch.zeros_like(accepted)
        pairwise_preference = torch.zeros_like(baseline)
        if pairwise_enabled:
            pairwise_preference = self._pairwise_preferences(
                evaluation, baseline, baseline_top, baseline_margin
            )
            baseline_preference = pairwise_preference[rows, baseline_top]
            selected_preference = pairwise_preference[rows, selected]
            pairwise_vetoed = (
                (selected != baseline_top)
                & (
                    selected_preference - baseline_preference
                    < self.config.pairwise_veto_threshold
                )
            )
            verified = torch.where(pairwise_vetoed, baseline_top, selected)
            pairwise_top_k = min(self.config.pairwise_top_k, baseline.shape[1])
            pairwise_shortlist = torch.zeros_like(baseline, dtype=torch.bool)
            pairwise_shortlist.scatter_(
                1, baseline.topk(k=pairwise_top_k, dim=1).indices, True
            )
            pairwise_candidate = pairwise_preference.masked_fill(
                ~pairwise_shortlist, float("-inf")
            ).argmax(dim=1)
            pairwise_advantage = (
                pairwise_preference[rows, pairwise_candidate] - baseline_preference
            )
            pairwise_critical_delta = (
                critical_probabilities[rows, pairwise_candidate]
                - critical_probabilities[rows, baseline_top]
            ).amin(dim=1)
            pairwise_safety_delta = (
                evaluation["predicted_safety"][rows, pairwise_candidate]
                - evaluation["predicted_safety"][rows, baseline_top]
            )
            pairwise_rescued = (
                (verified == baseline_top)
                & (pairwise_advantage >= self.config.pairwise_rescue_threshold)
                & (baseline_margin <= self.config.pairwise_rescue_margin)
                & (
                    pairwise_critical_delta
                    >= self.config.pairwise_rescue_critical_delta
                )
                & (
                    pairwise_safety_delta
                    >= self.config.pairwise_rescue_safety_delta
                )
            )
            selected = torch.where(pairwise_rescued, pairwise_candidate, verified)
            accepted = selected != baseline_top
            # Keep the public score tensor consistent with the explicit final
            # selection even when the verifier rescues a different candidate.
            gated_scores = baseline.clone()
            selected_score = baseline[rows, baseline_top] + accepted.to(
                baseline.dtype
            ) * 1e-4
            gated_scores.scatter_(1, selected[:, None], selected_score[:, None])
        return {
            "baseline_selected": baseline_top,
            "conservative_selected": conservative_top,
            "selected": selected,
            "selection_accepted": accepted,
            "selection_advantage": advantage,
            "baseline_margin": baseline_margin,
            "selection_confidence": confidence,
            "selection_safety_delta": safety_delta,
            "selection_critical_delta": critical_delta,
            "pairwise_enabled": torch.full_like(accepted, pairwise_enabled),
            "pairwise_candidate": pairwise_candidate,
            "pairwise_vetoed": pairwise_vetoed,
            "pairwise_rescued": pairwise_rescued,
            "pairwise_preference": pairwise_preference,
            "gated_scores": gated_scores,
        }

    def _refine(self, rollout: torch.Tensor) -> Dict[str, torch.Tensor]:
        weights = torch.softmax(self.refinement_pool(rollout).squeeze(-1), dim=-1)
        step_features = (weights[..., None] * rollout).sum(dim=-2)
        gate = torch.sigmoid(self.refinement_gate_logits).to(step_features.dtype)
        limits = self.refinement_limits.to(step_features.dtype)
        residual = torch.tanh(self.refinement_head(step_features)) * gate * limits
        smoothness = (residual[..., 1:, :] - residual[..., :-1, :]).square().mean()
        return {"trajectory_residual": residual, "refinement_gate": gate, "smoothness": smoothness}

    def forward(
        self,
        scene_tokens: torch.Tensor,
        candidates: torch.Tensor,
        base_scores: torch.Tensor,
        *,
        return_auxiliary: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self._validate(scene_tokens, candidates, base_scores)
        scene = self._encode_scene(scene_tokens)
        initial_rollout = self._rollout(scene, candidates)
        initial_evaluation = self._evaluate(initial_rollout, base_scores, scene)
        # Outcome-JEPA is useful only if deployment can consume the learned
        # action-consequence representation.  Keep it alongside the ordinary
        # world latent so NAVTRAIN-fitted verifiers may select either or both.
        initial_evaluation["outcome_features"] = self.outcome_predictor(
            initial_evaluation["world_features"]
        )
        initial_evaluation["current_future_outcome_features"] = self.outcome_predictor(
            initial_evaluation["current_future_world_features"]
        )
        initial_evaluation["outcome_factor_logits"] = self.outcome_factor_head(
            initial_evaluation["outcome_features"]
        )
        normalized_outcome = F.normalize(
            initial_evaluation["outcome_features"].float(), dim=-1
        ).to(initial_evaluation["outcome_features"].dtype)
        rows = torch.arange(base_scores.shape[0], device=base_scores.device)
        baseline = base_scores[..., -1] if base_scores.ndim == 3 else base_scores
        baseline_top = baseline.argmax(dim=1)
        anchor_outcome = normalized_outcome[rows, baseline_top]
        outcome_delta_features = torch.cat(
            (
                normalized_outcome - anchor_outcome[:, None],
                normalized_outcome * anchor_outcome[:, None],
                (baseline - baseline[rows, baseline_top, None])[..., None],
            ),
            dim=-1,
        )
        normalized_current_future = F.normalize(
            initial_evaluation["current_future_outcome_features"].float(), dim=-1
        ).to(initial_evaluation["current_future_outcome_features"].dtype)
        anchor_current_future = normalized_current_future[rows, baseline_top]
        current_future_delta_features = torch.cat(
            (
                normalized_current_future - anchor_current_future[:, None],
                normalized_current_future * anchor_current_future[:, None],
                (baseline - baseline[rows, baseline_top, None])[..., None],
            ),
            dim=-1,
        )
        initial_evaluation["outcome_factor_deltas"] = self.outcome_delta_head(
            outcome_delta_features
        )
        initial_evaluation["outcome_factor_deltas"] = (
            initial_evaluation["outcome_factor_deltas"]
            - initial_evaluation["outcome_factor_deltas"][rows, baseline_top, None]
        )
        action_scale = candidates.new_tensor(
            (self.config.position_scale, self.config.position_scale, 1.0)
        )
        anchor_action = candidates[rows, baseline_top]
        relative_action = (
            (candidates - anchor_action[:, None]) / action_scale
        ).flatten(start_dim=2)
        outcome_action_delta_features = torch.cat(
            (outcome_delta_features, relative_action), dim=-1
        )
        initial_evaluation["outcome_action_factor_deltas"] = (
            self.outcome_action_delta_head(outcome_action_delta_features)
        )
        motion_features = temporal_motion_features(
            candidates, anchor_action, self.config.position_scale
        )
        initial_evaluation["risk_aware_prediction"] = (
            self.risk_aware_compatibility_head(outcome_delta_features, motion_features)
        )
        initial_evaluation["risk_aware_prediction"] = (
            initial_evaluation["risk_aware_prediction"]
            - initial_evaluation["risk_aware_prediction"][rows, baseline_top, None]
        )
        initial_evaluation["current_future_factor_deltas"] = (
            self.future_compatibility_head(
                current_future_delta_features, motion_features
            )
        )
        initial_evaluation["current_future_factor_deltas"] = (
            initial_evaluation["current_future_factor_deltas"]
            - initial_evaluation["current_future_factor_deltas"][
                rows, baseline_top, None
            ]
        )
        initial_evaluation["current_future_rank_prediction"] = (
            self.current_future_rank_head(
                current_future_delta_features, motion_features
            )
            + self.current_future_factorized_rank_head(
                current_future_delta_features, motion_features
            )
        )
        initial_evaluation["current_future_rank_prediction"] = (
            initial_evaluation["current_future_rank_prediction"]
            - initial_evaluation["current_future_rank_prediction"][
                rows, baseline_top, None
            ]
        )
        initial_evaluation["current_future_rank_prediction"] = (
            self._ensemble_current_future_rank_prediction(
                initial_evaluation["world_features"],
                scene,
                candidates,
                baseline,
                motion_features,
                initial_evaluation["current_future_rank_prediction"],
            )
        )
        initial_evaluation["current_future_progress_prediction"] = (
            self.current_future_progress_head(
                current_future_delta_features, motion_features
            )
        )
        initial_evaluation["current_future_progress_prediction"] = (
            initial_evaluation["current_future_progress_prediction"]
            - initial_evaluation["current_future_progress_prediction"][
                rows, baseline_top, None
            ]
        )
        initial_evaluation.update(
            self._planning_jepa_verifier_reader_outputs(
                initial_evaluation["world_features"],
                scene,
                candidates,
                baseline,
                motion_features,
            )
        )
        initial_evaluation["temporal_action_factor_deltas"] = (
            self.temporal_action_jepa_head(outcome_delta_features, motion_features)
        )
        initial_evaluation["temporal_action_factor_deltas"] = (
            initial_evaluation["temporal_action_factor_deltas"]
            - initial_evaluation["temporal_action_factor_deltas"][
                rows, baseline_top, None
            ]
        )
        initial_evaluation["outcome_action_factor_deltas"] = (
            initial_evaluation["outcome_action_factor_deltas"]
            - initial_evaluation["outcome_action_factor_deltas"][
                rows, baseline_top, None
            ]
        )
        outcome_probabilities = torch.sigmoid(
            initial_evaluation["outcome_factor_logits"]
        )
        initial_evaluation["outcome_candidate_subscores"] = outcome_probabilities
        initial_evaluation["outcome_structured_scores"] = (
            outcome_probabilities[..., 0]
            * outcome_probabilities[..., 1]
            * (
                5.0 * outcome_probabilities[..., 2]
                + 5.0 * outcome_probabilities[..., 3]
                + 2.0 * outcome_probabilities[..., 4]
            )
            / 12.0
        )
        conservative_selection = self._conservative_select(
            initial_evaluation, base_scores, candidates
        )
        skip_unused_refinement = (
            self.config.selection_mode == "conservative"
            and not self.config.deploy_refined_selected
            and not return_auxiliary
        )
        if skip_unused_refinement:
            # Conservative deployment selects an untouched original proposal.
            # Avoid a refinement head and a second recurrent world rollout whose
            # outputs the agent deliberately discards.
            zero_residual = torch.zeros_like(candidates)
            refinement = {
                "trajectory_residual": zero_residual,
                "refinement_gate": torch.sigmoid(self.refinement_gate_logits).to(
                    candidates.dtype
                ),
                "smoothness": candidates.new_zeros(()),
            }
            refined = candidates
            rollout = initial_rollout
            evaluation = initial_evaluation
        else:
            refinement = self._refine(initial_rollout)
            refined = candidates + refinement["trajectory_residual"]
            rollout = self._rollout(scene, refined)
            evaluation = self._evaluate(rollout, base_scores, scene)
        output = {
            "original_candidates": candidates,
            "refined_candidates": refined,
            "trajectory_residual": refinement["trajectory_residual"],
            "refinement_gate": refinement["refinement_gate"],
            "refinement_smoothness_loss": refinement["smoothness"],
            "rollout_latents": rollout,
            "initial_structured_scores": initial_evaluation["structured_scores"],
            "initial_candidate_subscores": initial_evaluation[
                "candidate_subscores"
            ],
            "initial_collision_probabilities": initial_evaluation[
                "collision_probabilities"
            ],
            "initial_critic_value_probabilities": initial_evaluation[
                "critic_value_probabilities"
            ],
            "initial_preference_logits": initial_evaluation["preference_logits"],
            "initial_critic_final_probabilities": initial_evaluation[
                "critic_final_probabilities"
            ],
            "initial_critic_residual": initial_evaluation["critic_residual"],
            "initial_critic_confidence": initial_evaluation["critic_confidence"],
            "initial_predicted_safety": initial_evaluation["predicted_safety"],
            "initial_conservative_scores": initial_evaluation[
                "conservative_scores"
            ],
            "initial_world_features": initial_evaluation["world_features"],
            "initial_current_future_world_features": initial_evaluation[
                "current_future_world_features"
            ],
            "initial_outcome_features": initial_evaluation["outcome_features"],
            "initial_outcome_factor_logits": initial_evaluation[
                "outcome_factor_logits"
            ],
            "initial_outcome_candidate_subscores": initial_evaluation[
                "outcome_candidate_subscores"
            ],
            "initial_outcome_structured_scores": initial_evaluation[
                "outcome_structured_scores"
            ],
            "initial_outcome_factor_deltas": initial_evaluation[
                "outcome_factor_deltas"
            ],
            "initial_outcome_action_factor_deltas": initial_evaluation[
                "outcome_action_factor_deltas"
            ],
            "initial_temporal_action_factor_deltas": initial_evaluation[
                "temporal_action_factor_deltas"
            ],
            "initial_current_future_factor_deltas": initial_evaluation[
                "current_future_factor_deltas"
            ],
            "initial_current_future_rank_prediction": initial_evaluation[
                "current_future_rank_prediction"
            ],
            "initial_current_future_progress_prediction": initial_evaluation[
                "current_future_progress_prediction"
            ],
            "initial_risk_aware_prediction": initial_evaluation[
                "risk_aware_prediction"
            ],
            **{f"conservative_{key}": value for key, value in conservative_selection.items()},
            **evaluation,
        }
        if return_auxiliary:
            output.update(
                {
                    "initial_rollout_latents": initial_rollout,
                    "initial_world_logits": initial_evaluation["world_logits"],
                    "initial_value_logits": initial_evaluation["value_logits"],
                    "initial_collision_logits": initial_evaluation["collision_logits"],
                    "initial_world_scores": initial_evaluation["world_scores"],
                }
            )
        for name, value in output.items():
            if torch.is_floating_point(value):
                self._check(value, name)
        return output


__all__ = ["ActWorldPlanner", "ActWorldPlannerConfig"]
