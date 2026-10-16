"""Configuration for the ActWorld-JEPA planner.

The perception backbone and proposal generator are inherited unchanged from
Drive-JEPA.  The fields below only control the candidate-conditioned latent
rollout, evaluation, and bounded proposal correction layers.
"""

from dataclasses import dataclass

from navsim.agents.drive_jepa_perception_based.drive_jepa_config import (
    DriveJEPAConfig,
)


@dataclass
class ActWorldJEPAConfig(DriveJEPAConfig):
    """Drive-JEPA configuration extended with latent-world planning options."""

    actworld_latent_tokens: int = 8
    actworld_num_heads: int = 8
    actworld_num_layers: int = 2
    actworld_hidden_dim: int = 512
    actworld_num_subscores: int = 9
    actworld_dropout: float = 0.05

    # Candidate rollouts use trajectory increments as actions.  Exact warm
    # start comes from the zero terminal refiner layer, while a raw gate of
    # zero keeps a usable sigmoid derivative for the first actor update.
    actworld_time_delta: float = 0.5
    actworld_position_scale: float = 50.0
    actworld_speed_scale: float = 20.0
    actworld_acceleration_scale: float = 10.0
    actworld_curvature_scale: float = 1.0
    actworld_max_refinement_xy: float = 1.0
    actworld_max_refinement_yaw: float = 0.20
    actworld_refinement_gate_init: float = 0.0

    actworld_consistency_cosine_weight: float = 0.5
    # Full finite-value scans synchronize the GPU.  Keep them available for
    # smoke/debug runs, but disable them for normal multi-GPU training.
    actworld_finite_checks: bool = False
    actworld_rank_temperature: float = 0.1

    # Critic objective/parameterization.  ``legacy`` is the released Stage-A
    # route and remains the default for checkpoint and numerical parity.  The
    # F2 route is an explicitly selected, frozen-protocol fallback.
    actworld_critic_variant: str = "legacy"
    actworld_f2_aggregate_bce_weight: float = 1.0
    actworld_f2_top_regret_margin_weight: float = 1.0
    actworld_f2_top_regret_margin: float = 0.2
    actworld_f2_safe_unsafe_margin_weight: float = 1.0
    actworld_f2_safe_unsafe_margin: float = 0.5
    actworld_f2_nine_channel_bce_weight: float = 1.0
    actworld_f2_collision_bce_weight: float = 1.0
    actworld_f2_historical_latent_weight: float = 0.25
    actworld_f2_future_latent_weight: float = 0.25

    # Stage-A objectives plus legacy fields retained for reproducibility.  The
    # latent term is an observed historical transition; future frames, when
    # enabled, are training-only targets and are never consumed online.
    actworld_value_loss_weight: float = 1.0
    # Keep the released Drive-JEPA planning objective in both incremental
    # phases.  It is gradient-constant while Stage 1 freezes the base, then
    # supervises the selectively unfrozen decoder/backbone layers in Stage 2.
    actworld_drive_jepa_planning_loss_weight: float = 1.0
    actworld_rank_loss_weight: float = 0.5
    actworld_latent_loss_weight: float = 0.25
    actworld_future_latent_loss_weight: float = 0.25
    actworld_rescore_refined_proposals: bool = True

    # Explicit two-stage optimization.  The critic stage trains the latent
    # dynamics/evaluator with actor inputs detached.  The actor stage loads a
    # frozen Stage-A critic and improves its utility prediction through the
    # trajectory input only.
    actworld_training_stage: str = "critic"
    actworld_stage_a_checkpoint_path: str = ""
    actworld_actor_improvement_weight: float = 1.0
    actworld_actor_trust_weight: float = 0.1
    actworld_actor_smoothness_weight: float = 0.05
    actworld_gate_loss_weight: float = 1.0
    actworld_gate_rank_weight: float = 0.5
    actworld_actor_rank_temperature: float = 1.0
    # Optional Stage-B anchor for the candidates that Drive-JEPA itself
    # considers most plausible.  It is disabled by default to preserve all
    # released checkpoints and is selected using NAVTRAIN only.
    actworld_actor_selected_imitation_weight: float = 0.0

    # Optional training-only target branch.  The target builder streams the
    # terminal h=8 camera clip (scene frames 10/11); inference keeps this off.
    actworld_future_latent_grounding: bool = False
    actworld_target_ema_decay: float = 0.996

    # Both stages retain the reproduced Drive-JEPA path as a frozen anchor.
    # Actor training explicitly rejects an unfrozen baseline.
    actworld_freeze_drive_jepa: bool = True
    # Stage-2 model-only warm start.  Unlike Lightning resume, this restores
    # the complete Stage-1 model while intentionally constructing a new
    # low-learning-rate optimizer with the selective parameter groups below.
    actworld_stage1_checkpoint_path: str = ""
    actworld_selective_unfreeze_planning_top: bool = False
    actworld_backbone_top_blocks: int = 2
    actworld_backbone_lr_scale: float = 0.01
    actworld_base_lr_scale: float = 0.1
    actworld_score_workers: int = 2
    actworld_score_timeout_seconds: float = 600.0
    actworld_score_slow_log_seconds: float = 60.0

    # A separate baseline checkpoint is used only to warm-start training.
    # ``checkpoint_path`` keeps its original meaning: a complete evaluation
    # checkpoint for the current agent.
    baseline_checkpoint_path: str = ""
