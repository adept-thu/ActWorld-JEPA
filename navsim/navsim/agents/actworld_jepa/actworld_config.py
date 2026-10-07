"""Configuration for the NAVSIM v1 ActWorld-JEPA agent."""

from dataclasses import dataclass

from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig


@dataclass
class ActWorldJEPAConfig(DriveJEPAConfig):
    """Drive-JEPA v1 plus a proposal-conditioned latent world planner."""

    actworld_latent_tokens: int = 8
    actworld_num_heads: int = 8
    actworld_num_layers: int = 1
    actworld_hidden_dim: int = 512
    actworld_dropout: float = 0.05

    actworld_time_delta: float = 0.5
    actworld_position_scale: float = 50.0
    actworld_speed_scale: float = 20.0
    actworld_acceleration_scale: float = 10.0
    actworld_curvature_scale: float = 1.0
    actworld_max_refinement_xy: float = 1.0
    actworld_max_refinement_yaw: float = 0.20
    actworld_refinement_gate_init: float = -4.0

    actworld_world_value_weight: float = 0.35
    actworld_subscore_weight: float = 0.25
    actworld_collision_penalty: float = 0.25
    actworld_finite_checks: bool = False
    actworld_rank_temperature: float = 0.1

    actworld_value_loss_weight: float = 1.0
    actworld_fusion_loss_weight: float = 1.0
    actworld_rank_loss_weight: float = 0.5
    actworld_hard_negative_loss_weight: float = 0.0
    actworld_hard_negative_top_k: int = 8
    actworld_hard_negative_temperature: float = 0.05
    actworld_pairwise_loss_weight: float = 0.0
    actworld_pairwise_loss_top_k: int = 16
    actworld_pairwise_loss_temperature: float = 0.05
    actworld_all_pairs_loss_weight: float = 0.0
    actworld_all_pairs_loss_top_k: int = 16
    actworld_all_pairs_loss_temperature: float = 0.05
    actworld_progress_pairwise_loss_weight: float = 0.0
    actworld_progress_pairwise_top_k: int = 16
    actworld_progress_pairwise_temperature: float = 0.05
    actworld_progress_pairwise_margin: float = 0.005
    actworld_progress_pairwise_critical_margin: float = 0.0
    actworld_risk_averse_loss_weight: float = 0.0
    actworld_risk_averse_top_k: int = 16
    actworld_risk_averse_temperature: float = 0.05
    actworld_risk_averse_positive_margin: float = 0.005
    actworld_risk_averse_false_positive_weight: float = 4.0
    actworld_shortlist_distillation_loss_weight: float = 0.0
    actworld_shortlist_distillation_top_k: int = 4
    actworld_shortlist_student_temperature: float = 0.05
    actworld_shortlist_teacher_temperature: float = 0.02
    # A baseline-conditioned preference head learns the deployment decision
    # directly: whether a shortlisted proposal improves on Drive-JEPA's
    # winner. Both switches default to zero for old-checkpoint compatibility.
    actworld_preference_residual_scale: float = 0.0
    actworld_preference_loss_weight: float = 0.0
    actworld_preference_top_k: int = 4
    actworld_preference_temperature: float = 0.05
    actworld_preference_positive_margin: float = 0.0025
    actworld_preference_false_positive_weight: float = 2.0
    actworld_preference_head_only: bool = False
    actworld_factorwise_pairwise_loss_weight: float = 0.0
    actworld_factorwise_pairwise_top_k: int = 4
    actworld_factorwise_pairwise_temperature: float = 0.05
    actworld_factorwise_safety_false_positive_weight: float = 8.0
    actworld_factorwise_progress_weight: float = 0.5
    # Factor-specific weights keep multiplicative safety terms from being
    # washed out by the easier progress channel. Defaults preserve the old
    # factorwise objective; guarded runs can emphasize TTC/NC/DAC explicitly.
    actworld_factorwise_nc_weight: float = 1.0
    actworld_factorwise_dac_weight: float = 1.0
    actworld_factorwise_ttc_weight: float = 1.0
    actworld_factorwise_comfort_weight: float = 1.0
    # Focus factor supervision on hard, wrongly-safe candidates. Zero exactly
    # preserves all previous checkpoints and loss values.
    actworld_factorwise_focal_gamma: float = 0.0
    # NAVTRAIN-only action-conditioned latent alignment. The target rollout is
    # detached and is never used during deployment or official evaluation.
    actworld_planning_jepa_loss_weight: float = 0.0
    actworld_planning_jepa_temperature: float = 0.05
    # Counterfactual Outcome JEPA aligns every shortlisted action-conditioned
    # world latent with a factorized NAVTRAIN outcome embedding.  Defaults are
    # disabled for full checkpoint and inference compatibility.
    actworld_outcome_jepa_loss_weight: float = 0.0
    actworld_outcome_jepa_top_k: int = 8
    actworld_outcome_jepa_relative_weight: float = 0.5
    actworld_outcome_jepa_lr_scale: float = 20.0
    # Decode interpretable physical outcomes from the Outcome-JEPA latent.
    # This makes the planning representation itself decision-aware instead of
    # relying on a post-hoc verifier. All labels come from NAVTRAIN simulation.
    actworld_outcome_factor_loss_weight: float = 0.0
    actworld_outcome_factor_top_k: int = 8
    actworld_outcome_factor_relative_weight: float = 1.0
    actworld_outcome_factor_rank_weight: float = 1.0
    actworld_outcome_factor_safety_negative_weight: float = 16.0
    actworld_outcome_factor_false_positive_weight: float = 16.0
    actworld_outcome_factor_temperature: float = 0.05
    # Anchor-conditioned counterfactual decoder. Unlike absolute factor
    # prediction, this head learns the decision-relevant change relative to
    # Drive-JEPA's winner and can be deployed without a post-hoc estimator.
    actworld_outcome_delta_loss_weight: float = 0.0
    actworld_outcome_delta_top_k: int = 8
    actworld_outcome_delta_factorwise_weight: float = 1.0
    actworld_outcome_delta_temperature: float = 0.05
    actworld_outcome_delta_false_positive_weight: float = 16.0
    actworld_outcome_delta_selection: bool = False
    actworld_outcome_delta_selection_top_k: int = 2
    actworld_outcome_delta_min_score_delta: float = 0.0
    actworld_outcome_delta_min_critical_delta: float = 0.0
    actworld_outcome_delta_min_comfort_delta: float = 0.0
    actworld_outcome_delta_max_candidate_gap: float = 0.005
    # Action-conditioned counterfactual JEPA: make the candidate trajectory
    # displacement explicit instead of asking the latent decoder to infer it.
    actworld_outcome_action_delta_loss_weight: float = 0.0
    actworld_outcome_action_delta_selection: bool = False
    # Temporal action-JEPA additionally represents proposal velocity and
    # acceleration before fusing action and imagined-outcome latents.
    actworld_temporal_action_delta_loss_weight: float = 0.0
    actworld_temporal_action_delta_selection: bool = False
    # v2 Current-Future interaction and its jointly adapted factorized scorer.
    # The scorer predicts NAVTRAIN factor deltas relative to Drive-JEPA top-1
    # and falls back to that proposal unless every configured guard passes.
    actworld_current_future_v2_loss_weight: float = 0.0
    # Direct planning-JEPA supervision for the decoupled Current-Future
    # representation. Candidate futures are ranked by similarity to a detached
    # expert-trajectory future latent, with NAVTRAIN simulator scores providing
    # the soft teacher distribution. This complements factor regression without
    # modifying the frozen deployment/world latent used by older scorers.
    actworld_current_future_jepa_loss_weight: float = 0.0
    actworld_current_future_jepa_temperature: float = 0.07
    # Mix the simulator-score teacher with a domain-stable expert-trajectory
    # teacher. Zero preserves every historical experiment.
    actworld_current_future_trajectory_teacher_weight: float = 0.0
    actworld_current_future_trajectory_teacher_temperature: float = 1.0
    # Cross-scene future-identity matching keeps the auxiliary planning-JEPA
    # representation from collapsing to a scene-agnostic shortcut.  The
    # detached target branch remains unchanged; the default preserves all
    # historical checkpoints and training recipes.
    actworld_current_future_cross_scene_loss_weight: float = 0.0
    # Direct future-consistent trajectory ranking.  Unlike the factor decoder,
    # this head is optimized for the final candidate ordering while retaining
    # four explicit non-regression channels for safety gating.
    actworld_current_future_rank_loss_weight: float = 0.0
    # Target deltas can be anchored either at raw Drive-JEPA top-1 (historical
    # behavior) or at the complete frozen deployment mother.  Deployment
    # alignment removes train/inference drift without changing old recipes.
    actworld_current_future_rank_anchor: str = "baseline"
    actworld_current_future_rank_temperature: float = 0.05
    actworld_current_future_rank_false_positive_weight: float = 24.0
    # Per-factor weights for the active planning-rank safety channels
    # (NC/DAC/TTC/comfort).  A non-uniform setting lets NAVTRAIN emphasize
    # the factors that are actually limiting deployment score while retaining
    # the same candidate set and frozen official protocol.
    actworld_current_future_rank_nc_weight: float = 1.0
    actworld_current_future_rank_dac_weight: float = 1.0
    actworld_current_future_rank_ttc_weight: float = 1.0
    actworld_current_future_rank_comfort_weight: float = 1.0
    actworld_current_future_rank_safety_weight: float = 1.0
    actworld_current_future_rank_gain_weight: float = 1.0
    actworld_current_future_rank_improvement_weight: float = 1.0
    actworld_current_future_rank_pairwise_weight: float = 1.0
    actworld_current_future_rank_listwise_weight: float = 0.5
    # Optional batch-CVaR reweighting for the direct Current-Future rank
    # objective. Defaults preserve every historical checkpoint/run. When
    # enabled, the loss interpolates between the batch mean and the worst
    # ``fraction`` of scenes so rare unsafe mistakes are not averaged away.
    actworld_current_future_rank_cvar_fraction: float = 0.0
    actworld_current_future_rank_cvar_weight: float = 0.0
    # Adaptive worst-log optimization. The dataset supplies a stable physical
    # drive id only when explicitly requested by the training recipe.
    actworld_current_future_group_dro_eta: float = 0.0
    actworld_current_future_group_dro_num_groups: int = 0
    # Boundary-aware ranking focuses the deployable reader on candidates whose
    # simulator outcome is close to the Drive-JEPA mother. These are the
    # cases where a small calibration error changes the selected trajectory.
    actworld_current_future_rank_boundary_weight: float = 0.0
    actworld_current_future_rank_boundary_band: float = 0.01
    actworld_current_future_rank_boundary_margin: float = 0.0025
    actworld_current_future_rank_boundary_false_positive_weight: float = 12.0
    # Optional NAVTRAIN guard: a boundary-positive switch cannot reduce raw
    # progress. Disabled by default for old-checkpoint compatibility.
    actworld_current_future_rank_boundary_require_progress: bool = False
    # NAVTRAIN-only teacher anchor for the auxiliary Current-Future latent.
    # The teacher is copied from the initialization checkpoint and is absent
    # from deployment; zero preserves every historical recipe.
    actworld_current_future_teacher_anchor_loss_weight: float = 0.0
    # Explicit candidate progress-delta regression for the deployable
    # Future-Consistent ranking gate.  Zero keeps historical recipes intact.
    actworld_current_future_progress_head_loss_weight: float = 0.0
    # Anchor for the auxiliary progress-delta target.  "conservative" matches
    # deployment's rank mother; "baseline" is retained for ablations.
    actworld_current_future_progress_anchor: str = "conservative"
    actworld_current_future_progress_only: bool = False
    # Optional inference-time ensemble of independently trained Planning-JEPA
    # readers.  The main checkpoint remains the first reader; extra paths are
    # NAVTRAIN-screened and are disabled by default for old checkpoints.
    actworld_current_future_rank_ensemble_checkpoint_paths: tuple = ()
    # Conservative deployment fallback for the reader ensemble. When enabled,
    # the validated single-reader decision is retained whenever it agrees with
    # at least one auxiliary reader; the median reader is used only when all
    # readers disagree. This is inference-only and disabled by default.
    actworld_current_future_rank_consensus_fallback: bool = False
    actworld_current_future_rank_consensus_main_min_probability: float = 0.5
    actworld_current_future_rank_consensus_main_min_gain: float = 0.0
    actworld_current_future_rank_consensus_main_min_safety_probability: float = 0.2
    # Residual deployment mode: replay the frozen r94 reader first, then let
    # the newly trained reader act only as a guarded residual on that choice.
    # By default the first auxiliary checkpoint is r94; the exact-mother mode
    # below reverses the roles and runs r94 as the complete main checkpoint.
    actworld_current_future_rank_residual_fallback: bool = False
    # Exact-mother variant: the main checkpoint is the frozen r94 model and
    # the first auxiliary reader is the new residual.  This avoids rebuilding
    # the mother from only a subset of its checkpoint at deployment time.
    actworld_current_future_rank_residual_main_is_mother: bool = False
    # Require independently trained residual readers to accept the same
    # candidate before changing the exact r94 mother.
    actworld_current_future_rank_residual_consensus: bool = False
    # Label-free transition-comfort guard for residual choices.  It bounds the
    # increase in first-step lateral acceleration relative to the r94 plan.
    # A very large default keeps historical behavior unchanged.
    actworld_current_future_rank_max_initial_lateral_acceleration_increase: float = 1.0e9
    # Optional inference-only consistency gate over predicted relative
    # NC/DAC/progress/TTC factors.  Count zero preserves historical behavior.
    actworld_current_future_rank_min_positive_factor_count: int = 0
    actworld_current_future_rank_positive_factor_floor: float = -1.0e9
    # Calibrate only the direct rank reader on a frozen Current-Future
    # representation.  This permits the newly zero-initialized head to use an
    # appropriate learning rate without disturbing the validated JEPA latent.
    actworld_current_future_rank_only: bool = False
    # Train only the zero-initialized factor-specific residual rank experts.
    # This preserves the validated r94 reader while separating rare
    # NC/DAC/TTC/comfort hazards from dense gain/progress supervision.
    actworld_current_future_factorized_rank_only: bool = False
    # Jointly adapt the auxiliary Current-Future interaction and its ranking
    # reader while keeping the public Drive-JEPA proposal/world path frozen.
    # The interaction receives the scaled learning rate below; the reader uses
    # the full agent learning rate.
    actworld_current_future_rank_joint: bool = False
    actworld_current_future_rank_representation_lr_scale: float = 0.1
    # Non-zero values explicitly activate candidate-conditioned reads over raw
    # scene tokens for new experiments.  Zero preserves historical checkpoints.
    actworld_token_attention_gate_init: float = 0.0
    # Deploy the learned improvement-confidence reader only as a guarded
    # second stage on top of the frozen candidate-set (r80) decision.  The
    # defaults are disabled so historical checkpoints and runs are unchanged.
    actworld_current_future_rank_selection: bool = False
    actworld_current_future_rank_cascade_on_candidate_set: bool = False
    actworld_current_future_rank_top_k: int = 8
    # Optional deployment utility: combine the predicted improvement logit
    # with the predicted gain logit.  Zero preserves the r94 rule exactly.
    actworld_current_future_rank_probability_weight: float = 0.0
    actworld_current_future_rank_min_gain: float = 0.0
    # Optional scene-wise normalized frozen-head utility.  Disabled by
    # default; the NAVTRAIN-only candidate uses weights selected by the
    # rank-normalized audit and a relative predicted-safety gate.
    actworld_current_future_rank_normalized_ensemble: bool = False
    actworld_current_future_rank_normalized_baseline_weight: float = 0.5
    actworld_current_future_rank_normalized_subscore_weight: float = 0.0
    actworld_current_future_rank_normalized_structured_weight: float = 0.25
    actworld_current_future_rank_normalized_value_weight: float = 0.0
    actworld_current_future_rank_normalized_safety_weight: float = 0.0
    actworld_current_future_rank_normalized_gain_weight: float = 0.2
    actworld_current_future_rank_normalized_probability_weight: float = 0.1
    actworld_current_future_rank_normalized_relative_safety: bool = True
    actworld_current_future_rank_min_probability: float = 0.5
    actworld_current_future_rank_min_safety_probability: float = 0.2
    actworld_current_future_rank_max_candidate_gap: float = 0.01
    actworld_current_future_rank_min_frozen_critical: float = -0.02
    actworld_current_future_rank_min_frozen_progress: float = 0.0
    actworld_current_future_rank_min_frozen_safety: float = -0.05
    # Optional TTC-preservation guard for the rank cascade.  Zero keeps the
    # historical behavior; positive values require the candidate's frozen TTC
    # factor not to regress relative to the rank mother.
    actworld_current_future_rank_min_frozen_ttc: float = 0.0
    actworld_candidate_set_min_progress_delta: float = 0.0
    # Explicit progress-delta reader deployment gate.  Disabled by default so
    # older checkpoints/configurations remain bit-compatible.
    actworld_current_future_progress_selection: bool = False
    actworld_current_future_progress_min_delta: float = 0.0
    actworld_current_future_progress_weight: float = 0.0
    # Optional post-r94 Planning-JEPA expansion.  It keeps the validated r94
    # choice unless the learned future-progress reader proposes a guarded,
    # safety-preserving top-k alternative.  Disabled for old-run parity.
    actworld_post_r94_progress_expansion: bool = False
    actworld_post_r94_progress_top_k: int = 8
    actworld_post_r94_progress_min_delta: float = 0.005
    actworld_post_r94_progress_min_rank_probability: float = 0.2
    actworld_post_r94_progress_min_rank_safety_probability: float = 0.2
    actworld_post_r94_progress_max_baseline_gap: float = 0.01
    actworld_post_r94_progress_min_frozen_critical: float = -0.02
    actworld_post_r94_progress_min_frozen_safety: float = 0.0
    actworld_post_r94_progress_min_frozen_ttc: float = -1.0
    # Keep the frozen deployment latent and every scorer that reads it
    # unchanged.  The v2 interaction becomes an auxiliary planning-JEPA
    # branch consumed only by the future-compatibility head.
    actworld_decoupled_current_future_branch: bool = False
    # Class-balanced sign supervision prevents the delta reader from solving
    # an imbalanced NAVTRAIN objective by predicting every alternative as a
    # regression.  It is applied only to non-baseline shortlist proposals.
    actworld_current_future_sign_loss_weight: float = 0.0
    actworld_current_future_v2_only: bool = False
    actworld_planning_adaptation_only: bool = False
    actworld_planning_last_layer_only: bool = False
    # Adapt only the conditional Planning-JEPA trajectory-refinement branch.
    # Original Drive-JEPA proposals and every deployment selector remain frozen.
    actworld_refinement_adaptation_only: bool = False
    actworld_safe_oracle_refinement_loss_weight: float = 0.0
    actworld_expert_refinement_loss_weight: float = 0.0
    actworld_current_future_v2_selection: bool = False
    actworld_current_future_v2_top_k: int = 4
    actworld_current_future_v2_min_score_delta: float = 0.0
    actworld_current_future_v2_min_critical_delta: float = 0.0
    actworld_current_future_v2_min_progress_delta: float = 0.0
    actworld_current_future_v2_min_comfort_delta: float = 0.0
    actworld_current_future_v2_max_candidate_gap: float = 0.005
    # Optional second-stage deployment on top of the frozen r62 candidate-set
    # winner.  These values are selected on NAVTRAIN A/B and opened once on
    # sealed NAVTRAIN C/D; defaults preserve every previous runtime exactly.
    actworld_current_future_v2_cascade_on_candidate_set: bool = False
    actworld_current_future_v2_safety_weight: float = 0.0
    actworld_current_future_v2_progress_weight: float = 0.0
    actworld_current_future_v2_comfort_weight: float = 0.0
    actworld_current_future_v2_min_frozen_critical: float = -1.0
    actworld_current_future_v2_min_frozen_progress: float = -1.0
    actworld_current_future_v2_min_frozen_safety: float = -1.0
    actworld_risk_aware_v3_loss_weight: float = 0.0
    actworld_risk_aware_v3_only: bool = False
    # Preserve the frozen best JEPA latent distribution: train only the small
    # risk reader, never the current/future interaction that feeds the verifier.
    actworld_frozen_risk_veto_only: bool = False
    actworld_risk_aware_v3_selection: bool = False
    actworld_latent_risk_veto: bool = False
    actworld_risk_aware_v3_top_k: int = 4
    actworld_risk_aware_v3_min_probability: float = 0.8
    actworld_risk_aware_v3_min_safety_probability: float = 0.8
    actworld_risk_aware_v3_min_gain: float = 0.0
    actworld_risk_aware_v3_max_candidate_gap: float = 0.005
    actworld_risk_aware_v3_false_positive_weight: float = 20.0
    actworld_safety_negative_weight: float = 0.0
    actworld_collision_loss_weight: float = 1.0
    actworld_refine_loss_weight: float = 0.25
    actworld_smoothness_loss_weight: float = 0.05
    actworld_rescore_refined_proposals: bool = True
    actworld_refined_score_loss_weight: float = 1.0
    # Select with original proposals/features, but execute the corresponding
    # refined trajectory.  This decouples proposal quality from rank stability.
    actworld_deploy_refined_selected: bool = False
    # Optional deterministic post-selection continuation along the displacement
    # from the frozen Drive-JEPA proposal to the future-consistent proposal.
    # 1.0 is bit-exact legacy behavior; values above one provide a small,
    # scorer-independent progress refinement that can be audited on NAVTRAIN.
    actworld_selected_trajectory_scale: float = 1.0

    # Deployment policy.  ``legacy`` exactly preserves the original v1
    # rollout/evaluate/refine behavior and checkpoint semantics.  The
    # conservative policy keeps Drive-JEPA's proposals unchanged and treats
    # the learned world heads as a bounded residual reranker with an explicit
    # fallback to the frozen baseline.
    actworld_selection_mode: str = "legacy"
    actworld_conservative_value_weight: float = 0.5
    actworld_conservative_structured_weight: float = 0.25
    actworld_conservative_residual_scale: float = 0.1
    actworld_conservative_consistency_temperature: float = 1.0
    actworld_conservative_min_confidence: float = 0.0
    actworld_conservative_min_advantage: float = 0.0
    actworld_conservative_max_baseline_margin: float = 1.0
    actworld_conservative_safety_margin: float = -1.0
    actworld_conservative_min_critical_delta: float = -1.0
    actworld_conservative_top_k: int = 32

    # Optional post-hoc pairwise verifier fitted only on NAVTRAIN log folds.
    # Empty keeps legacy checkpoints and conservative inference unchanged.
    actworld_pairwise_calibrator_path: str = ""
    actworld_pairwise_top_k: int = 16
    actworld_pairwise_veto_threshold: float = -0.2
    actworld_pairwise_rescue_threshold: float = 0.05
    actworld_pairwise_rescue_margin: float = 0.005
    actworld_pairwise_rescue_critical_delta: float = 0.0
    actworld_pairwise_rescue_safety_delta: float = -0.05

    # Frozen geometry-aware verifier fitted exclusively on NAVTRAIN.  The
    # artifact contains the estimator and its already selected safety policy;
    # an empty path preserves all existing inference behavior.
    actworld_geometry_calibrator_path: str = ""

    # Frozen NAVTRAIN-only planning-JEPA verifier.  It consumes the trained
    # Current-Future rank/factor outputs and latent interaction summaries;
    # an empty path preserves all historical checkpoint behavior.
    actworld_planning_jepa_verifier_path: str = ""
    # Optional planning-only reader checkpoint.  This lets the official r90/r94
    # mother remain untouched while r173 supplies the residual future features.
    actworld_planning_jepa_verifier_reader_checkpoint_path: str = ""

    # Frozen NAVTRAIN-only switch that fuses a top-two and rank-aware
    # planning-latent verifier. Empty preserves all existing checkpoints and
    # selection behavior.
    actworld_planning_switch_path: str = ""

    # Frozen NAVTRAIN-only candidate-conditioned latent verifier. This is the
    # deployable counterpart of the strict offline planning-latent study.
    # Empty preserves all existing checkpoint and inference behavior.
    actworld_latent_verifier_path: str = ""
    # Optional NAVTRAIN-only residual verifier.  It never proposes a new
    # trajectory: it may only veto the frozen latent verifier back to the
    # original Drive-JEPA winner.  This preserves the validated best policy as
    # the parent decision rule and makes the added scorer strictly residual.
    actworld_latent_residual_veto_path: str = ""
    # NAVTRAIN-only candidate-set scorer layered on the frozen best verifier.
    # It may replace the verifier choice only inside Drive-JEPA's top-4 and is
    # protected by deterministic raw-plan acceleration/jerk comfort guards.
    actworld_candidate_set_scorer_path: str = ""
    actworld_tree_meta_scorer_path: str = ""
    actworld_post_r94_selector_path: str = ""
    # Optional second frozen selector.  When present, deployment leaves the
    # r94 mother only if both selectors independently choose the same plan.
    actworld_post_r94_selector_consensus_path: str = ""
    # Explicit opt-in for diagnostic selectors fitted with NAVTEST feedback.
    # False keeps the paper/default path protected from test-set tuning.
    actworld_post_r94_selector_allow_navtest_tuning: bool = False
    actworld_candidate_set_max_proxy_acceleration: float = 6.1125
    actworld_candidate_set_max_proxy_jerk: float = 16.74

    actworld_freeze_drive_jepa: bool = True
    # Selectively adapt only Drive-JEPA's trajectory proposal layers while
    # keeping the visual backbone and released scorer frozen.  This lets the
    # planning-JEPA objective improve the candidate trajectories themselves
    # without turning the experiment into full-model fine-tuning.
    actworld_unfreeze_planning_heads: bool = False
    actworld_planning_head_lr_scale: float = 0.1
    # When adapting proposals, optionally keep the learned world model fully
    # frozen so it acts as a stable differentiable planning critic.
    actworld_freeze_planner_during_proposal_adaptation: bool = False
    # Train only the zero-init current/future residual introduced for the 2.2
    # ablation.  This leaves the released proposal generator and every
    # previously validated ActWorld head bit-exact at initialization.
    actworld_current_future_fusion_only: bool = False
    # NAVTRAIN-only safe-oracle distillation: move the released scorer's
    # current top-1 proposal toward a strictly better, non-regressing proposal
    # already present in its candidate set.  The teacher trajectory is
    # detached; all defaults preserve prior runs and checkpoints.
    actworld_safe_oracle_distill_loss_weight: float = 0.0
    actworld_safe_oracle_distill_margin: float = 0.0025
    actworld_safe_oracle_distill_max_gain: float = 0.05
    actworld_backbone_lr_scale: float = 0.01
    actworld_base_lr_scale: float = 0.1
    actworld_score_workers: int = 2
    baseline_checkpoint_path: str = ""
