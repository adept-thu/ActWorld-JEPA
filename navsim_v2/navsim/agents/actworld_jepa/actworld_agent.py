"""NAVSIM agent integration and objectives for ActWorld-JEPA."""

import hashlib
import os
import stat
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import (
    DriveJEPAAgent,
)
from navsim.common.dataclasses import SensorConfig
from navsim.planning.validation.f2_metric_cache_manifest import (
    score_authenticated_metric_caches,
)

from .actworld_config import ActWorldJEPAConfig
from .actworld_candidate_selector import (
    DENSE_SAFETY_FEATURE_NAMES,
    FEATURE_NAMES as CANDIDATE_FEATURE_NAMES,
    LEGACY_FACTOR_SEMANTICS,
    NAVSIM_V2_FACTOR_SEMANTICS,
    candidate_features,
    candidate_shortlist,
    dense_safety_features,
)
from .actworld_features import ActWorldJEPAFeatureBuilder, ActWorldJEPATargetBuilder
from .actworld_gain_selector import FEATURE_NAMES, selector_features
from .actworld_variant_selector import (
    VARIANTS,
    post_variant_features,
    variant_features,
)
from .actworld_model import ActWorldJEPAModel

class ActWorldJEPAAgent(DriveJEPAAgent):
    """Drive-JEPA agent with latent rollout/evaluate/refine objectives."""

    model_class = ActWorldJEPAModel

    F2_CRITIC_VARIANT = "f2_base_residual"
    F2_AGENT_LR = 5e-5
    F2_BASELINE_SHA256 = (
        "740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05"
    )
    _F2_FIXED_CONFIG = {
        "actworld_f2_aggregate_bce_weight": 1.0,
        "actworld_f2_top_regret_margin_weight": 1.0,
        "actworld_f2_top_regret_margin": 0.2,
        "actworld_f2_safe_unsafe_margin_weight": 1.0,
        "actworld_f2_safe_unsafe_margin": 0.5,
        "actworld_f2_nine_channel_bce_weight": 1.0,
        "actworld_f2_collision_bce_weight": 1.0,
        "actworld_f2_historical_latent_weight": 0.25,
        "actworld_f2_future_latent_weight": 0.25,
    }
    _F2_PARALLEL_REQUIRED_PREDICTIONS = frozenset(
        {
            "proposals",
            "pred_logit",
            "pdm_score",
            "actworld_base_candidate_logits",
            "actworld_initial_world_logits",
            "actworld_initial_utility_logits",
            "actworld_initial_collision_logits",
            "actworld_latent_consistency_loss",
            "actworld_scene_latent",
            "actworld_initial_residual_logits",
            "actworld_initial_delta_logits",
        }
    )

    def __init__(
        self,
        config: ActWorldJEPAConfig,
        lr: float,
        checkpoint_path: str = "",
    ) -> None:
        training_stage, critic_variant = self._validate_initial_configuration(
            config, checkpoint_path, lr
        )
        super().__init__(config=config, lr=lr, checkpoint_path=checkpoint_path)
        self._config = config
        self._training_stage = training_stage
        self._critic_variant = critic_variant
        # A full-checkpoint agent is an online inference/evaluation instance.
        # Future cameras are privileged training targets only.
        self._future_grounding_enabled = bool(
            config.actworld_future_latent_grounding
            and checkpoint_path == ""
            and self._training_stage == "critic"
        )

        stage1_path = str(config.actworld_stage1_checkpoint_path)
        if checkpoint_path == "" and stage1_path:
            self._load_state(stage1_path, baseline_only=False)
        elif checkpoint_path == "" and config.baseline_checkpoint_path:
            self._load_state(config.baseline_checkpoint_path, baseline_only=True)
        if (
            checkpoint_path == ""
            and self._critic_variant == self.F2_CRITIC_VARIANT
        ):
            self._assert_f2_zero_initialized_residual()
            self._validate_f2_actor_zero_state(
                self.state_dict(), "<fresh F2 initialization>"
            )

        stage_a_path = str(config.actworld_stage_a_checkpoint_path)
        if checkpoint_path == "" and self._training_stage == "actor":
            # A fresh actor run loads Stage-A critic weights explicitly.  A
            # full Lightning resume may leave this empty because strict model
            # state restoration occurs after agent construction.
            if stage_a_path:
                self._load_stage_a_critic(stage_a_path)
        elif checkpoint_path == "" and stage_a_path:
            raise ValueError(
                "actworld_stage_a_checkpoint_path is only valid for actor training"
            )

        self._pad_model._actworld_planner.configure_training_stage(
            self._training_stage
        )

        if checkpoint_path == "":
            self._pad_model.configure_drive_jepa_trainability()

    def name(self) -> str:
        return "actworld_jepa_perception_based_agent"

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run ActWorld and optionally persist selector-training diagnostics.

        Diagnostics are disabled unless ``ACTWORLD_SELECTOR_DIAGNOSTICS_DIR``
        is set.  The scoring entrypoint supplies the scenario token, so the
        frozen NAVTRAIN outcomes can be joined without relying on worker order.
        """

        prediction = self._pad_model(features)
        prediction = self._apply_frozen_gain_selector(prediction)
        prediction = self._apply_frozen_candidate_selector(prediction)
        prediction = self._apply_frozen_variant_selector(prediction)
        prediction = self._apply_frozen_candidate_selector(
            prediction, dense_post=True
        )
        prediction = self._apply_frozen_variant_selector(prediction, post=True)
        output_dir = os.environ.get("ACTWORLD_SELECTOR_DIAGNOSTICS_DIR")
        token = getattr(self, "_actworld_selector_token", None)
        if output_dir and token:
            self._write_selector_diagnostics(Path(output_dir), str(token), prediction)
        return prediction

    def _load_frozen_candidate_selector(self, *, dense_post: bool = False) -> Any:
        """Load a proposal ranker fitted exclusively on NAVTRAIN."""

        path_environment = (
            "ACTWORLD_DENSE_SAFETY_SELECTOR_PATH"
            if dense_post
            else "ACTWORLD_CANDIDATE_SELECTOR_PATH"
        )
        cache_prefix = "_dense_safety_selector" if dense_post else "_candidate_selector"
        path = os.environ.get(path_environment)
        cached_path = self.__dict__.get(f"{cache_prefix}_path")
        if not path:
            self.__dict__[f"{cache_prefix}_path"] = None
            self.__dict__[f"{cache_prefix}_artifact"] = None
            return None
        if cached_path == path:
            return self.__dict__.get(f"{cache_prefix}_artifact")

        import joblib

        artifact = joblib.load(path)
        artifact_format = artifact.get("format")
        if artifact_format not in {
            "actworld_v2_navtrain_candidate_ranker_v1",
            "actworld_v2_navtrain_candidate_ranker_v2",
            "actworld_v2_navtrain_candidate_ranker_context_gate_v2",
            "actworld_v2_navtrain_candidate_ranker_context_gate_v3",
            "actworld_v2_navtrain_candidate_ranker_multicomponent_gate_v4",
            "actworld_v2_navtrain_candidate_lambdaranker_v3",
            "actworld_v2_navtrain_candidate_lgbm_regressor_v3",
            "actworld_v2_navtrain_candidate_dense_safety_v4",
        }:
            raise ValueError("unsupported ActWorld candidate-selector format")
        if dense_post and artifact_format != "actworld_v2_navtrain_candidate_dense_safety_v4":
            raise ValueError(
                "ActWorld dense post-selector requires the dense-safety v4 format"
            )
        if artifact.get("training_split") != "navtrain":
            raise ValueError("ActWorld candidate selector must use NAVTRAIN")
        if artifact.get("navtest_used_for_training") is not False:
            raise ValueError("ActWorld candidate selector reports NAVTEST training use")
        if artifact_format == "actworld_v2_navtrain_candidate_dense_safety_v4":
            expected_features = CANDIDATE_FEATURE_NAMES + DENSE_SAFETY_FEATURE_NAMES
        else:
            expected_features = CANDIDATE_FEATURE_NAMES
        if tuple(artifact.get("feature_names", ())) != expected_features:
            raise ValueError("ActWorld candidate-selector feature contract mismatch")
        if int(artifact.get("feature_count", -1)) != len(expected_features):
            raise ValueError("ActWorld candidate-selector feature-count mismatch")
        factor_semantics = artifact.get(
            "factor_semantics", LEGACY_FACTOR_SEMANTICS
        )
        if factor_semantics not in {
            LEGACY_FACTOR_SEMANTICS,
            NAVSIM_V2_FACTOR_SEMANTICS,
        }:
            raise ValueError("ActWorld candidate selector has invalid factor semantics")
        if (
            artifact_format == "actworld_v2_navtrain_candidate_ranker_v2"
            and factor_semantics != NAVSIM_V2_FACTOR_SEMANTICS
        ):
            raise ValueError(
                "ActWorld candidate-ranker v2 requires NAVSIM-v2 factor semantics"
            )
        if artifact_format in {
            "actworld_v2_navtrain_candidate_ranker_context_gate_v2",
            "actworld_v2_navtrain_candidate_ranker_context_gate_v3",
            "actworld_v2_navtrain_candidate_ranker_multicomponent_gate_v4",
        }:
            if int(artifact.get("candidate_count", -1)) != 32:
                raise ValueError("ActWorld context selector requires 32 candidates")
            if int(artifact.get("context_feature_count", -1)) <= 0:
                raise ValueError("ActWorld context selector has invalid context size")
            if (
                artifact_format == "actworld_v2_navtrain_candidate_ranker_context_gate_v3"
                and artifact.get("gate_kind") not in {"classifier", "regressor"}
            ):
                raise ValueError("ActWorld context gate v3 has invalid gate kind")
            if artifact_format == "actworld_v2_navtrain_candidate_ranker_multicomponent_gate_v4":
                expected_outputs = (
                    "epdms_delta",
                    "no_at_fault_collisions_delta",
                    "drivable_area_compliance_delta",
                    "ego_progress_delta",
                    "time_to_collision_within_bound_delta",
                )
                if tuple(artifact.get("gate_output_names", ())) != expected_outputs:
                    raise ValueError(
                        "ActWorld multi-component gate has invalid output contract"
                    )
                if not tuple(artifact.get("gate_models", ())):
                    raise ValueError("ActWorld multi-component gate has no models")
                required_gate_policy = {
                    "epdms_delta",
                    "no_at_fault_collisions_delta",
                    "drivable_area_compliance_delta",
                    "ego_progress_delta",
                    "time_to_collision_within_bound_delta",
                }
                if set(artifact.get("gate_policy", ())) != required_gate_policy:
                    raise ValueError(
                        "ActWorld multi-component gate has invalid policy contract"
                    )
        configured_blend = float(
            os.environ.get("ACTWORLD_BASE_FACTORIZED_BLEND", "0.0")
        )
        artifact_blend = float(artifact.get("factorized_blend", 0.05))
        if abs(artifact_blend - configured_blend) > 1e-12:
            raise ValueError(
                "ActWorld candidate-selector factorized blend does not match: "
                f"artifact={artifact_blend}, configured={configured_blend}"
            )
        self.__dict__[f"{cache_prefix}_path"] = path
        self.__dict__[f"{cache_prefix}_artifact"] = artifact
        return artifact

    def _apply_frozen_candidate_selector(
        self,
        prediction: Dict[str, torch.Tensor],
        *,
        dense_post: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Apply future-consistent proposal ranking learned on NAVTRAIN."""

        artifact = self._load_frozen_candidate_selector(dense_post=dense_post)
        if artifact is None:
            return prediction
        required = {
            "proposals",
            "refined_proposals",
            "pdm_score",
            "base_pdm_score",
            "actworld_candidate_subscores",
            "actworld_residual_logits",
        }
        if (
            artifact.get("format")
            in {
                "actworld_v2_navtrain_candidate_ranker_context_gate_v2",
                "actworld_v2_navtrain_candidate_ranker_context_gate_v3",
                "actworld_v2_navtrain_candidate_ranker_multicomponent_gate_v4",
            }
        ):
            required.add("actworld_current_context")
        dense_safety = (
            artifact.get("format")
            == "actworld_v2_navtrain_candidate_dense_safety_v4"
        )
        if dense_safety:
            required.update({"pred_agents_states", "pred_area_logit"})
        missing = sorted(required.difference(prediction))
        if missing:
            raise KeyError(f"candidate selector missing predictions: {missing}")

        def batch_array(name: str) -> np.ndarray:
            value = prediction[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"candidate-selector input {name} must be a tensor")
            return value.detach().float().cpu().numpy()

        arrays = {
            name: batch_array(name)
            for name in required
            if name != "refined_proposals"
        }
        dense_area_logits = None
        if dense_safety:
            batch_size, candidate_count = arrays["pdm_score"].shape[:2]
            dense_area_logits = arrays["pred_area_logit"].reshape(
                batch_size, candidate_count, -1, 2
            )
        if (
            artifact.get("format")
            in {
                "actworld_v2_navtrain_candidate_ranker_context_gate_v2",
                "actworld_v2_navtrain_candidate_ranker_context_gate_v3",
                "actworld_v2_navtrain_candidate_ranker_multicomponent_gate_v4",
            }
        ):
            return self._apply_context_candidate_selector(
                prediction, artifact, arrays
            )
        policy = artifact["policy"]
        factor_semantics = artifact.get(
            "factor_semantics", LEGACY_FACTOR_SEMANTICS
        )
        top_k = int(policy["top_k"])
        listwise = (
            artifact.get("format")
            == "actworld_v2_navtrain_candidate_lambdaranker_v3"
        )
        threshold = float(
            policy["min_score_margin"]
            if listwise
            else policy["min_predicted_gain"]
        )
        max_gap = float(policy["max_baseline_gap"])
        hard_guard = float(policy["min_hard_safety_delta"])
        chosen_indices = []
        chosen_gains = []
        for batch in range(arrays["pdm_score"].shape[0]):
            payload = {
                "proposals": arrays["proposals"][batch],
                "pdm_score": arrays["pdm_score"][batch],
                "base_pdm_score": arrays["base_pdm_score"][batch],
                "candidate_subscores": arrays["actworld_candidate_subscores"][batch],
                "residual_logits": arrays["actworld_residual_logits"][batch],
            }
            fallback = int(np.argmax(payload["pdm_score"]))
            shortlist = candidate_shortlist(
                payload, top_k, factor_semantics=factor_semantics
            )
            features = []
            metadata = []
            for candidate in shortlist:
                feature, item = candidate_features(
                    payload,
                    int(candidate),
                    factor_semantics=factor_semantics,
                )
                if dense_safety:
                    feature = np.concatenate(
                        (
                            feature,
                            dense_safety_features(
                                arrays["pred_agents_states"][batch],
                                dense_area_logits[batch],
                                int(candidate),
                                int(item["selected"]),
                                int(item["mother"]),
                            ),
                        )
                    ).astype(np.float32)
                features.append(feature)
                metadata.append(item)
            predicted = np.asarray(artifact["model"].predict(np.stack(features)))
            if listwise:
                fallback_positions = [
                    index
                    for index, item in enumerate(metadata)
                    if int(item["candidate"]) == fallback
                ]
                if len(fallback_positions) != 1:
                    raise ValueError(
                        "ActWorld LambdaRank shortlist must contain the fallback"
                    )
                fallback_score = float(predicted[fallback_positions[0]])
                decision_score = predicted - fallback_score
            else:
                decision_score = predicted
            eligible = [
                index
                for index, item in enumerate(metadata)
                if decision_score[index] >= threshold
                and float(item["base_gap"]) <= max_gap
                and float(item["hard_safety_delta"]) >= hard_guard
            ]
            if eligible:
                best = max(eligible, key=lambda index: predicted[index])
                chosen = int(metadata[best]["candidate"])
                gain = float(decision_score[best])
            else:
                chosen = fallback
                gain = float("-inf")
            chosen_indices.append(chosen)
            chosen_gains.append(gain)

        pdm_score = prediction["pdm_score"].clone()
        device = pdm_score.device
        chosen = torch.as_tensor(chosen_indices, device=device, dtype=torch.long)
        rows = torch.arange(pdm_score.shape[0], device=device)
        winning = pdm_score.amax(dim=1)
        epsilon = torch.finfo(pdm_score.dtype).eps
        pdm_score[rows, chosen] = winning + epsilon
        result = dict(prediction)
        result["pdm_score"] = pdm_score
        result["selected_proposal"] = chosen
        result["trajectory"] = prediction["refined_proposals"][rows, chosen]
        result["actworld_candidate_selector_gain"] = torch.as_tensor(
            chosen_gains, device=device, dtype=pdm_score.dtype
        )
        return result

    @staticmethod
    def _context_selector_stats(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        sorted_value = np.sort(value)
        return np.concatenate(
            (
                np.quantile(value, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]),
                np.asarray(
                    [
                        value.mean(),
                        value.std(),
                        sorted_value[-1] - sorted_value[-2],
                    ],
                    dtype=np.float64,
                ),
            )
        )

    @staticmethod
    def _context_selector_choice(
        prediction: np.ndarray,
        payload: Dict[str, np.ndarray],
        metadata: list[Dict[str, float]],
        policy: Dict[str, float],
    ) -> int:
        fallback = int(np.argmax(payload["pdm_score"]))
        shortlist = candidate_shortlist(payload, 4)
        eligible = [
            int(candidate)
            for candidate in shortlist
            if float(prediction[int(candidate)]) >= float(policy["threshold"])
            and float(metadata[int(candidate)]["base_gap"]) <= float(policy["gap"])
            and float(metadata[int(candidate)]["hard_safety_delta"])
            >= float(policy["hard"])
        ]
        if not eligible:
            return fallback
        return max(eligible, key=lambda candidate: float(prediction[candidate]))

    def _apply_context_candidate_selector(
        self,
        prediction: Dict[str, torch.Tensor],
        artifact: Dict[str, Any],
        arrays: Dict[str, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """Route between local and latent-neighbour NAVTRAIN rankers."""

        chosen_indices = []
        chosen_gains = []
        for batch in range(arrays["pdm_score"].shape[0]):
            payload = {
                "proposals": arrays["proposals"][batch],
                "pdm_score": arrays["pdm_score"][batch],
                "base_pdm_score": arrays["base_pdm_score"][batch],
                "candidate_subscores": arrays["actworld_candidate_subscores"][batch],
                "residual_logits": arrays["actworld_residual_logits"][batch],
            }
            if len(payload["proposals"]) != int(artifact["candidate_count"]):
                raise ValueError("ActWorld context selector candidate-count mismatch")
            feature_rows = []
            metadata = []
            for candidate in range(len(payload["proposals"])):
                feature, item = candidate_features(payload, candidate)
                feature_rows.append(feature)
                metadata.append(item)
            feature_array = np.stack(feature_rows)
            base_prediction = np.asarray(
                artifact["base_model"].predict(feature_array), dtype=np.float64
            )

            context = np.asarray(
                arrays["actworld_current_context"][batch], dtype=np.float64
            ).reshape(1, -1)
            if context.shape[1] != int(artifact["context_feature_count"]):
                raise ValueError("ActWorld context selector context-size mismatch")
            reduced = artifact["context_pca"].transform(
                artifact["context_scaler"].transform(context)
            )[0]
            normalized = reduced / max(float(np.linalg.norm(reduced)), 1e-8)
            similarity = np.asarray(artifact["neighbor_context"]) @ normalized
            count = int(artifact["neighbor_count"])
            nearest = np.argsort(similarity)[-count:]
            knn_prediction = np.asarray(artifact["neighbor_gain"])[nearest].mean(axis=0)

            base_choice = self._context_selector_choice(
                base_prediction, payload, metadata, artifact["base_policy"]
            )
            knn_choice = self._context_selector_choice(
                knn_prediction, payload, metadata, artifact["knn_policy"]
            )
            fallback = int(np.argmax(payload["pdm_score"]))
            scalar = np.asarray(
                [
                    base_prediction[base_choice],
                    base_prediction[knn_choice],
                    knn_prediction[base_choice],
                    knn_prediction[knn_choice],
                    base_choice / 31.0,
                    knn_choice / 31.0,
                    abs(base_choice - knn_choice) / 31.0,
                    float(base_choice == fallback),
                    float(knn_choice == fallback),
                    float(base_choice == knn_choice),
                    metadata[base_choice]["base_gap"],
                    metadata[knn_choice]["base_gap"],
                    metadata[base_choice]["hard_safety_delta"],
                    metadata[knn_choice]["hard_safety_delta"],
                ],
                dtype=np.float64,
            )
            gate_feature = np.concatenate(
                (
                    reduced,
                    feature_array[base_choice],
                    feature_array[knn_choice],
                    feature_array[knn_choice] - feature_array[base_choice],
                    self._context_selector_stats(base_prediction),
                    self._context_selector_stats(knn_prediction),
                    scalar,
                )
            )[None].astype(np.float32)
            if (
                artifact.get("format")
                == "actworld_v2_navtrain_candidate_ranker_multicomponent_gate_v4"
            ):
                gate_outputs = np.mean(
                    [
                        np.asarray(model.predict(gate_feature), dtype=np.float64)[0]
                        for model in artifact["gate_models"]
                    ],
                    axis=0,
                )
                output_names = tuple(artifact["gate_output_names"])
                if gate_outputs.shape != (len(output_names),):
                    raise ValueError(
                        "ActWorld multi-component gate prediction shape mismatch"
                    )
                gate_values = dict(zip(output_names, gate_outputs.tolist()))
                gate_policy = artifact["gate_policy"]
                use_knn = all(
                    gate_values[name] >= float(gate_policy[name])
                    for name in output_names
                )
                gate_score = float(gate_values["epdms_delta"])
            else:
                gate_kind = str(artifact.get("gate_kind", "classifier"))
                if gate_kind == "classifier":
                    probabilities = artifact["gate_model"].predict_proba(gate_feature)[0]
                    classes = np.asarray(artifact["gate_model"].classes_)
                    positive = np.flatnonzero(classes == 1)
                    if len(positive) != 1:
                        raise ValueError("ActWorld context gate has invalid class contract")
                    gate_score = float(probabilities[int(positive[0])])
                elif gate_kind == "regressor":
                    gate_score = float(artifact["gate_model"].predict(gate_feature)[0])
                else:
                    raise ValueError(
                        f"unsupported ActWorld context gate kind: {gate_kind!r}"
                    )
                use_knn = gate_score >= float(artifact["gate_threshold"])
            chosen = knn_choice if use_knn else base_choice
            chosen_indices.append(chosen)
            chosen_gains.append(
                float(knn_prediction[chosen])
                if use_knn
                else float(base_prediction[chosen])
            )

        pdm_score = prediction["pdm_score"].clone()
        device = pdm_score.device
        chosen = torch.as_tensor(chosen_indices, device=device, dtype=torch.long)
        rows = torch.arange(pdm_score.shape[0], device=device)
        pdm_score[rows, chosen] = pdm_score.amax(dim=1) + torch.finfo(
            pdm_score.dtype
        ).eps
        result = dict(prediction)
        result["pdm_score"] = pdm_score
        result["selected_proposal"] = chosen
        result["trajectory"] = prediction["refined_proposals"][rows, chosen]
        result["actworld_candidate_selector_gain"] = torch.as_tensor(
            chosen_gains, device=device, dtype=pdm_score.dtype
        )
        return result

    def _load_frozen_variant_selector(self) -> Any:
        """Load a sequence-exact selector trained exclusively on NAVTRAIN."""

        return self._load_variant_selector_artifact(
            "ACTWORLD_STRICT_VARIANT_SELECTOR_PATH", "_strict_variant_selector"
        )

    def _load_frozen_post_variant_selector(self) -> Any:
        """Load an optional second-stage selector fitted after the frozen mother."""

        return self._load_variant_selector_artifact(
            "ACTWORLD_POST_VARIANT_SELECTOR_PATH", "_post_variant_selector"
        )

    def _load_variant_selector_artifact(
        self, path_environment: str, cache_prefix: str
    ) -> Any:
        """Load and validate one NAVTRAIN-only trajectory selector artifact."""

        path = os.environ.get(path_environment)
        path_key = f"{cache_prefix}_path"
        artifact_key = f"{cache_prefix}_artifact"
        cached_path = self.__dict__.get(path_key)
        if not path:
            self.__dict__[path_key] = None
            self.__dict__[artifact_key] = None
            return None
        if cached_path == path:
            return self.__dict__.get(artifact_key)

        import joblib

        artifact = joblib.load(path)
        if artifact.get("format") not in {
            "actworld_v2_navtrain_strict_variant_selector_v1",
            "actworld_v2_navtrain_strict_variant_selector_ec_v2",
            "actworld_v2_navtrain_crossfold_variant_selector_v2",
            "actworld_v2_navtrain_post_variant_selector_v1",
        }:
            raise ValueError("unsupported ActWorld strict variant-selector format")
        if artifact.get("training_split") != "navtrain":
            raise ValueError("ActWorld strict variant selector must use NAVTRAIN")
        if artifact.get("navtest_used_for_training") is not False:
            raise ValueError("ActWorld strict variant selector reports NAVTEST training use")
        if tuple(artifact.get("variants", ())) != VARIANTS:
            raise ValueError("ActWorld strict variant-selector contract mismatch")
        policy = artifact.get("policy", {})
        allowed = tuple(policy.get("allowed_variants", ()))
        if not allowed or not set(allowed).issubset(VARIANTS):
            raise ValueError("ActWorld strict variant selector has invalid policy variants")
        self.__dict__[path_key] = path
        self.__dict__[artifact_key] = artifact
        return artifact

    @staticmethod
    def _variant_trajectory(
        current: torch.Tensor,
        target: torch.Tensor,
        weight: float,
        scale: float,
    ) -> torch.Tensor:
        result = current.clone()
        if weight != 0.0:
            result[..., :2] = (1.0 - weight) * current[..., :2] + weight * target[..., :2]
            result[..., 2] = torch.atan2(
                (1.0 - weight) * torch.sin(current[..., 2])
                + weight * torch.sin(target[..., 2]),
                (1.0 - weight) * torch.cos(current[..., 2])
                + weight * torch.cos(target[..., 2]),
            )
        if scale != 1.0:
            result[..., :2] = result[..., :2] * scale
        return result

    def _apply_frozen_variant_selector(
        self, prediction: Dict[str, torch.Tensor], *, post: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Apply future-factor trajectory variants selected by NAVTRAIN OOF."""

        artifact = (
            self._load_frozen_post_variant_selector()
            if post
            else self._load_frozen_variant_selector()
        )
        if artifact is None:
            return prediction
        required = {
            "proposals",
            "refined_proposals",
            "pdm_score",
            "base_pdm_score",
            "actworld_candidate_subscores",
            "actworld_residual_logits",
            "trajectory",
        }
        missing = sorted(required.difference(prediction))
        if missing:
            raise KeyError(f"strict variant selector missing predictions: {missing}")

        def batch_array(name: str) -> np.ndarray:
            value = prediction[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"strict variant-selector input {name} must be a tensor")
            return value.detach().float().cpu().numpy()

        array_names = required.difference({"refined_proposals", "trajectory"})
        arrays = {name: batch_array(name) for name in array_names}
        refined_array = batch_array("refined_proposals")
        trajectory_array = batch_array("trajectory")
        batch_size = arrays["pdm_score"].shape[0]
        all_features = []
        all_metadata = []
        for batch in range(batch_size):
            payload = {
                "proposals": arrays["proposals"][batch],
                "pdm_score": arrays["pdm_score"][batch],
                "base_pdm_score": arrays["base_pdm_score"][batch],
                "candidate_subscores": arrays["actworld_candidate_subscores"][batch],
                "residual_logits": arrays["actworld_residual_logits"][batch],
            }
            use_post_features = (
                post
                and artifact.get("format")
                == "actworld_v2_navtrain_post_variant_selector_v1"
            )
            if use_post_features:
                payload["refined_proposals"] = refined_array[batch]
                payload["trajectory"] = trajectory_array[batch]
            row_features = []
            row_metadata = []
            for variant in VARIANTS:
                feature, metadata = (
                    post_variant_features(payload, variant)
                    if use_post_features
                    else variant_features(payload, variant)
                )
                row_features.append(feature)
                row_metadata.append(metadata)
            all_features.extend(row_features)
            all_metadata.append(row_metadata)

        feature_array = np.stack(all_features)
        if feature_array.shape[1] != int(artifact["feature_count"]):
            raise ValueError("ActWorld strict variant-selector feature-count mismatch")
        if artifact.get("format") == "actworld_v2_navtrain_crossfold_variant_selector_v2":
            models = tuple(artifact.get("models", ()))
            if not models:
                raise ValueError("ActWorld cross-fold variant selector has no models")
            predicted = np.mean(
                [np.asarray(model.predict(feature_array)) for model in models],
                axis=0,
            ).reshape(batch_size, len(VARIANTS))
        else:
            predicted = np.asarray(artifact["model"].predict(feature_array)).reshape(
                batch_size, len(VARIANTS)
            )
        predicted_ec = None
        if artifact.get("format") == "actworld_v2_navtrain_strict_variant_selector_ec_v2":
            predicted_ec = np.asarray(
                artifact["ec_model"].predict(feature_array)
            ).reshape(batch_size, len(VARIANTS))
        policy = artifact["policy"]
        allowed = set(policy["allowed_variants"])
        threshold = float(policy["threshold"])
        hard_guard = float(policy["hard_guard"])
        ec_threshold = float(policy.get("ec_threshold", float("-inf")))
        ec_factor_guard = float(policy.get("ec_factor_guard", float("-inf")))
        chosen_variant = []
        chosen_gain = []
        output_trajectory = prediction["trajectory"].clone()
        refined = prediction["refined_proposals"]
        proposals = prediction["proposals"]
        # The gain selector preserves its chosen index as the argmax of the
        # (possibly fallback-adjusted) deployed score tensor.
        selected_proposal = torch.argmax(prediction["pdm_score"], dim=1)
        # Preserve the ranking decision while allowing a NAVTRAIN-selected
        # attenuation of actor geometry.  This decouples "which candidate"
        # from "how much learned refinement" and avoids changing scorer
        # calibration when only the final physical correction is adjusted.
        base_trajectory = proposals[
            torch.arange(batch_size, device=proposals.device), selected_proposal
        ].clone()
        for batch in range(batch_size):
            eligible = [
                index
                for index, variant in enumerate(VARIANTS)
                if variant in allowed
                and predicted[batch, index] >= threshold
                and float(all_metadata[batch][index]["hard_delta"]) >= hard_guard
                and (
                    predicted_ec is None
                    or predicted_ec[batch, index] >= ec_threshold
                )
                and float(all_metadata[batch][index]["ec_factor_delta"])
                >= ec_factor_guard
            ]
            if not eligible:
                chosen_variant.append(-1)
                chosen_gain.append(float("-inf"))
                continue
            best = max(eligible, key=lambda index: predicted[batch, index])
            metadata = all_metadata[batch][best]
            target = int(metadata["target"])
            output_trajectory[batch] = self._variant_trajectory(
                output_trajectory[batch],
                refined[batch, target],
                float(metadata["weight"]),
                float(metadata["scale"]),
            )
            base_trajectory[batch] = self._variant_trajectory(
                base_trajectory[batch],
                proposals[batch, target],
                float(metadata["weight"]),
                float(metadata["scale"]),
            )
            chosen_variant.append(best)
            chosen_gain.append(float(predicted[batch, best]))

        final_blend = float(
            os.environ.get("ACTWORLD_FINAL_REFINEMENT_BLEND", "1.0")
        )
        if not np.isfinite(final_blend) or not 0.0 <= final_blend <= 1.0:
            raise ValueError(
                "ACTWORLD_FINAL_REFINEMENT_BLEND must be finite and in [0, 1], "
                f"got {final_blend!r}"
            )
        if final_blend != 1.0:
            blended = base_trajectory.clone()
            blended[..., :2] = base_trajectory[..., :2] + final_blend * (
                output_trajectory[..., :2] - base_trajectory[..., :2]
            )
            blended[..., 2] = torch.atan2(
                (1.0 - final_blend) * torch.sin(base_trajectory[..., 2])
                + final_blend * torch.sin(output_trajectory[..., 2]),
                (1.0 - final_blend) * torch.cos(base_trajectory[..., 2])
                + final_blend * torch.cos(output_trajectory[..., 2]),
            )
            output_trajectory = blended

        result = dict(prediction)
        result["trajectory"] = output_trajectory
        result["actworld_strict_variant"] = torch.as_tensor(
            chosen_variant, device=output_trajectory.device, dtype=torch.long
        )
        result["actworld_strict_variant_predicted_gain"] = torch.as_tensor(
            chosen_gain, device=output_trajectory.device, dtype=output_trajectory.dtype
        )
        result["actworld_final_refinement_blend"] = output_trajectory.new_full(
            (batch_size,), final_blend
        )
        return result

    def _load_frozen_gain_selector(self) -> Any:
        path = os.environ.get("ACTWORLD_F2_GAIN_SELECTOR_PATH")
        cached_path = self.__dict__.get("_gain_selector_path")
        if not path:
            self.__dict__["_gain_selector_path"] = None
            self.__dict__["_gain_selector_artifact"] = None
            return None
        if cached_path == path:
            return self.__dict__.get("_gain_selector_artifact")

        import joblib

        artifact = joblib.load(path)
        if artifact.get("format") != "actworld_v2_navtrain_gain_selector_v1":
            raise ValueError("unsupported ActWorld gain-selector artifact format")
        if artifact.get("training_split") != "navtrain":
            raise ValueError("ActWorld gain selector must be trained on NAVTRAIN")
        if artifact.get("navtest_used_for_training") is not False:
            raise ValueError("ActWorld gain selector reports NAVTEST training use")
        if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("ActWorld gain-selector feature contract mismatch")
        configured_scale = float(os.environ.get("ACTWORLD_F2_RESIDUAL_SCALE", "1.0"))
        if abs(float(artifact["scale"]) - configured_scale) > 1e-12:
            raise ValueError(
                "ActWorld gain-selector scale does not match deployment: "
                f"artifact={artifact['scale']}, configured={configured_scale}"
            )
        configured_ec_weight = float(
            os.environ.get("ACTWORLD_EC_SELECTION_WEIGHT", "0.0")
        )
        artifact_ec_weight = float(artifact.get("ec_weight", 2.0))
        if abs(artifact_ec_weight - configured_ec_weight) > 1e-12:
            raise ValueError(
                "ActWorld gain-selector EC weight does not match deployment: "
                f"artifact={artifact_ec_weight}, configured={configured_ec_weight}"
            )
        configured_base_ec_weight = float(
            os.environ.get(
                "ACTWORLD_BASE_EC_SELECTION_WEIGHT", str(configured_ec_weight)
            )
        )
        artifact_base_ec_weight = float(
            artifact.get("base_ec_weight", artifact_ec_weight)
        )
        if abs(artifact_base_ec_weight - configured_base_ec_weight) > 1e-12:
            raise ValueError(
                "ActWorld gain-selector base EC weight does not match deployment: "
                f"artifact={artifact_base_ec_weight}, "
                f"configured={configured_base_ec_weight}"
            )
        configured_factorized_blend = float(
            os.environ.get("ACTWORLD_BASE_FACTORIZED_BLEND", "0.0")
        )
        artifact_factorized_blend = float(
            artifact.get("factorized_blend", 0.0)
        )
        if abs(artifact_factorized_blend - configured_factorized_blend) > 1e-12:
            raise ValueError(
                "ActWorld gain-selector factorized blend does not match deployment: "
                f"artifact={artifact_factorized_blend}, "
                f"configured={configured_factorized_blend}"
            )
        self.__dict__["_gain_selector_path"] = path
        self.__dict__["_gain_selector_artifact"] = artifact
        return artifact

    def _apply_frozen_gain_selector(
        self, prediction: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Accept an F2 rerank only when its frozen NAVTRAIN gain passes."""

        artifact = self._load_frozen_gain_selector()
        if artifact is None:
            return prediction
        required = {
            "proposals",
            "refined_proposals",
            "pdm_score",
            "base_pdm_score",
            "actworld_candidate_subscores",
            "actworld_residual_logits",
        }
        missing = sorted(required.difference(prediction))
        if missing:
            raise KeyError(f"gain selector missing predictions: {missing}")

        def batch_array(name: str) -> np.ndarray:
            value = prediction[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"gain-selector input {name} must be a tensor")
            return value.detach().float().cpu().numpy()

        arrays = {name: batch_array(name) for name in required if name != "refined_proposals"}
        features = []
        metadata = []
        for index in range(arrays["pdm_score"].shape[0]):
            feature, item = selector_features(
                {
                    "proposals": arrays["proposals"][index],
                    "pdm_score": arrays["pdm_score"][index],
                    "base_pdm_score": arrays["base_pdm_score"][index],
                    "candidate_subscores": arrays["actworld_candidate_subscores"][index],
                    "residual_logits": arrays["actworld_residual_logits"][index],
                }
            )
            features.append(feature)
            metadata.append(item)

        predicted_gain = np.asarray(artifact["model"].predict(np.stack(features)))
        policy = artifact["policy"]
        accepted = np.asarray(
            [
                item["selected"] != item["mother"]
                and gain >= float(policy["min_predicted_gain"])
                and item["base_gap"] <= float(policy["max_baseline_gap"])
                and item["hard_safety_delta"]
                >= float(policy["min_hard_safety_delta"])
                for gain, item in zip(predicted_gain, metadata)
            ],
            dtype=bool,
        )

        pdm_score = prediction["pdm_score"].clone()
        device = pdm_score.device
        selected = torch.as_tensor(
            [int(item["selected"]) for item in metadata], device=device
        )
        mother = torch.as_tensor(
            [int(item["mother"]) for item in metadata], device=device
        )
        accepted_tensor = torch.as_tensor(accepted, device=device)
        chosen = torch.where(accepted_tensor, selected, mother)
        rows = torch.arange(pdm_score.shape[0], device=device)
        rejected = (~accepted_tensor) & (selected != mother)
        if bool(rejected.any().item()):
            epsilon = torch.finfo(pdm_score.dtype).eps
            pdm_score[rows[rejected], mother[rejected]] = (
                pdm_score.amax(dim=1)[rejected] + epsilon
            )
        prediction = dict(prediction)
        prediction["pdm_score"] = pdm_score
        prediction["selected_proposal"] = chosen
        prediction["trajectory"] = prediction["refined_proposals"][rows, chosen]
        prediction["actworld_gain_selector_accepted"] = accepted_tensor
        prediction["actworld_gain_selector_predicted_gain"] = torch.as_tensor(
            predicted_gain, dtype=pdm_score.dtype, device=device
        )
        return prediction

    @staticmethod
    def _write_selector_diagnostics(
        output_dir: Path,
        token: str,
        prediction: Dict[str, torch.Tensor],
    ) -> None:
        """Atomically save the frozen signals used by the gain selector."""

        required = {
            "proposals",
            "refined_proposals",
            "trajectory",
            "pdm_score",
            "base_pdm_score",
            "actworld_candidate_subscores",
            "actworld_residual_logits",
        }
        missing = sorted(required.difference(prediction))
        if missing:
            raise KeyError(f"selector diagnostics missing predictions: {missing}")

        def array(name: str) -> np.ndarray:
            value = prediction[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"selector diagnostic {name} must be a tensor")
            result = value.detach().float().cpu().numpy()
            if result.shape[0] != 1:
                raise ValueError(
                    f"selector diagnostics require batch size one, got {result.shape}"
                )
            return result[0]

        output_dir.mkdir(parents=True, exist_ok=True)
        safe_token = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in token
        )
        destination = output_dir / f"{safe_token}.npz"
        temporary = output_dir / f".{safe_token}.{os.getpid()}.npz"
        payload = dict(
            token=np.asarray(token),
            proposals=array("proposals"),
            refined_proposals=array("refined_proposals"),
            trajectory=array("trajectory"),
            pdm_score=array("pdm_score"),
            base_pdm_score=array("base_pdm_score"),
            candidate_subscores=array("actworld_candidate_subscores"),
            residual_logits=array("actworld_residual_logits"),
        )
        if "actworld_current_context" in prediction:
            payload["current_context"] = array("actworld_current_context")
        if "actworld_world_features" in prediction:
            payload["world_features"] = array("actworld_world_features")
        if prediction.get("pred_agents_states") is not None:
            payload["pred_agent_logits"] = array("pred_agents_states")[..., -1]
        if prediction.get("pred_area_logit") is not None:
            value = prediction["pred_area_logit"]
            if not isinstance(value, torch.Tensor):
                raise TypeError("selector diagnostic pred_area_logit must be a tensor")
            batch_size, candidate_count = prediction["pdm_score"].shape[:2]
            dense_area = value.detach().float().cpu().numpy().reshape(
                batch_size, candidate_count, -1, 2
            )
            if batch_size != 1:
                raise ValueError(
                    "selector diagnostics require batch size one, "
                    f"got dense area shape {dense_area.shape}"
                )
            payload["pred_area_logits"] = dense_area[0]
        np.savez_compressed(temporary, **payload)
        temporary.replace(destination)

    def worker_map(self, data_points: Any) -> Any:
        """Use sealed caches for Stage-A F2 and NAVTRAIN caches for Stage-B."""

        if (
            self._critic_variant != self.F2_CRITIC_VARIANT
            or self._training_stage == "actor"
        ):
            return super().worker_map(data_points)
        return self._worker_map_with_recovery(
            data_points,
            score_authenticated_metric_caches,
            "authenticated F2 NAVSIM proposal scoring",
        )

    @classmethod
    def _validate_initial_configuration(
        cls,
        config: ActWorldJEPAConfig,
        checkpoint_path: str,
        lr: float,
    ) -> Tuple[str, str]:
        """Fail fast on stage/variant contracts before model construction."""

        training_stage = str(config.actworld_training_stage)
        critic_variant = str(config.actworld_critic_variant)
        if training_stage not in {"critic", "actor"}:
            raise ValueError("actworld_training_stage must be 'critic' or 'actor'")
        if critic_variant not in {"legacy", cls.F2_CRITIC_VARIANT}:
            raise ValueError(
                "actworld_critic_variant must be 'legacy' or "
                f"'{cls.F2_CRITIC_VARIANT}'"
            )
        if critic_variant == cls.F2_CRITIC_VARIANT:
            cls._validate_f2_fixed_config(config)
            if not isinstance(checkpoint_path, str):
                raise TypeError("F2 checkpoint_path must be a string")
            if config.actworld_freeze_drive_jepa is not True:
                raise ValueError("F2 requires actworld_freeze_drive_jepa=true")
            if checkpoint_path == "" and training_stage == "critic":
                if config.actworld_future_latent_grounding is not True:
                    raise ValueError(
                        "F2 critic training requires "
                        "actworld_future_latent_grounding=true"
                    )
                baseline_path = str(config.baseline_checkpoint_path)
                if not baseline_path:
                    raise ValueError(
                        "Fresh F2 critic training requires "
                        "baseline_checkpoint_path"
                    )
                if not Path(baseline_path).is_file():
                    raise FileNotFoundError(
                        "Fresh F2 baseline checkpoint is not a file: "
                        f"{baseline_path}"
                    )
                if (
                    isinstance(lr, bool)
                    or not isinstance(lr, (int, float))
                    or float(lr) != cls.F2_AGENT_LR
                ):
                    raise ValueError(
                        "Fresh F2 critic training requires agent lr exactly "
                        f"{cls.F2_AGENT_LR!r}; found {lr!r}"
                    )
        if training_stage == "actor" and not config.actworld_freeze_drive_jepa:
            raise ValueError("actor training requires actworld_freeze_drive_jepa=true")
        selective = config.actworld_selective_unfreeze_planning_top
        if not isinstance(selective, bool):
            raise TypeError(
                "actworld_selective_unfreeze_planning_top must be a bool"
            )
        stage1_path = str(config.actworld_stage1_checkpoint_path)
        if selective:
            if config.actworld_freeze_drive_jepa:
                raise ValueError(
                    "selective Stage 2 requires actworld_freeze_drive_jepa=false"
                )
            if training_stage != "critic":
                raise ValueError("selective Stage 2 requires critic training")
            top_blocks = config.actworld_backbone_top_blocks
            if (
                isinstance(top_blocks, bool)
                or not isinstance(top_blocks, int)
                or top_blocks <= 0
            ):
                raise ValueError(
                    "actworld_backbone_top_blocks must be a positive integer"
                )
            if checkpoint_path == "" and not stage1_path:
                raise ValueError(
                    "fresh selective Stage 2 requires "
                    "actworld_stage1_checkpoint_path"
                )
        elif stage1_path:
            raise ValueError(
                "actworld_stage1_checkpoint_path is only valid for selective Stage 2"
            )
        if stage1_path and config.baseline_checkpoint_path:
            raise ValueError(
                "Stage 2 must use actworld_stage1_checkpoint_path instead of "
                "baseline_checkpoint_path"
            )
        if stage1_path and checkpoint_path:
            raise ValueError(
                "Use either actworld_stage1_checkpoint_path for a model-only "
                "Stage-2 warm start or checkpoint_path for evaluation"
            )
        if stage1_path and not Path(stage1_path).is_file():
            raise FileNotFoundError(stage1_path)
        return training_stage, critic_variant

    def _assert_f2_zero_initialized_residual(self) -> None:
        """Verify warm-start loading did not alter F2's residual terminal."""

        terminal = self._pad_model._actworld_planner.value_head[-1]
        tensors = (terminal.weight, terminal.bias)
        if any(
            tensor is None or bool(torch.count_nonzero(tensor.detach()).item())
            for tensor in tensors
        ):
            raise RuntimeError(
                "F2 residual value-head terminal must remain exactly zero "
                "after the Drive-JEPA warm start"
            )

    @classmethod
    def _checkpoint_payload(
        cls, path: str, *, f2_baseline: bool = False
    ) -> Dict[str, Any]:
        checkpoint_path = Path(path)
        if not f2_baseline:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            checkpoint = torch.load(path, map_location=torch.device("cpu"))
        else:
            # A fresh F2 run is tied to one immutable official Drive-JEPA
            # checkpoint.  Resolve, hash, and deserialize through the same
            # descriptor so swapping the pathname between those operations
            # cannot change the loaded payload.
            try:
                path_stat = os.lstat(checkpoint_path)
            except FileNotFoundError:
                raise FileNotFoundError(checkpoint_path) from None
            if stat.S_ISLNK(path_stat.st_mode):
                raise ValueError("F2 baseline checkpoint must not be a symlink")
            if not stat.S_ISREG(path_stat.st_mode):
                raise ValueError("F2 baseline checkpoint must be a regular file")

            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(checkpoint_path, flags)
            try:
                with os.fdopen(descriptor, "rb", closefd=True) as handle:
                    descriptor = -1
                    opened_stat = os.fstat(handle.fileno())
                    current_stat = os.lstat(checkpoint_path)
                    if (
                        not stat.S_ISREG(opened_stat.st_mode)
                        or stat.S_ISLNK(current_stat.st_mode)
                        or (opened_stat.st_dev, opened_stat.st_ino)
                        != (current_stat.st_dev, current_stat.st_ino)
                    ):
                        raise ValueError(
                            "F2 baseline checkpoint pathname changed during open"
                        )

                    digest = hashlib.sha256()
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                    observed_sha256 = digest.hexdigest()
                    if observed_sha256 != cls.F2_BASELINE_SHA256:
                        raise ValueError(
                            "F2 baseline checkpoint SHA-256 mismatch: "
                            f"expected={cls.F2_BASELINE_SHA256}, "
                            f"observed={observed_sha256}"
                        )

                    handle.seek(0)
                    checkpoint = torch.load(
                        handle, map_location=torch.device("cpu")
                    )

                    # Detect an in-place writer that changed the already-open
                    # file while torch.load was consuming it.
                    handle.seek(0)
                    verify_digest = hashlib.sha256()
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        verify_digest.update(chunk)
                    final_stat = os.fstat(handle.fileno())
                    if (
                        verify_digest.hexdigest() != observed_sha256
                        or final_stat.st_size != opened_stat.st_size
                        or final_stat.st_mtime_ns != opened_stat.st_mtime_ns
                    ):
                        raise ValueError(
                            "F2 baseline checkpoint changed while being loaded"
                        )
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint {path!r} is not a mapping")
        return checkpoint

    @staticmethod
    def _state_from_checkpoint(
        checkpoint: Dict[str, Any], path: str
    ) -> Dict[str, torch.Tensor]:
        if "state_dict" not in checkpoint or not isinstance(checkpoint["state_dict"], dict):
            raise ValueError(f"Checkpoint {path!r} does not contain a Lightning state_dict")
        if not all(isinstance(key, str) for key in checkpoint["state_dict"]):
            raise ValueError(f"Checkpoint {path!r} contains non-string state_dict keys")
        return {
            key.replace("agent._pad_model", "_pad_model"): value
            for key, value in checkpoint["state_dict"].items()
        }

    @classmethod
    def _checkpoint_state(cls, path: str) -> Dict[str, torch.Tensor]:
        return cls._state_from_checkpoint(cls._checkpoint_payload(path), path)

    def _validate_checkpoint_variant(
        self, checkpoint: Dict[str, Any], path: str
    ) -> None:
        configured = getattr(self, "_critic_variant", "legacy")
        # Checkpoints predating explicit variants are legacy by definition.
        saved = checkpoint.get("actworld_critic_variant", "legacy")
        if saved != configured:
            raise ValueError(
                f"ActWorld checkpoint {path!r} critic variant mismatch: "
                f"checkpoint={saved!r}, configured={configured!r}"
            )

    def _f2_actor_zero_state_keys(self) -> Tuple[str, ...]:
        """Return state keys whose zeros make the frozen F2 actor inert."""

        planner = self._pad_model._actworld_planner
        terminal = planner.refinement_head[-1]
        parameter_names = {
            id(parameter): name for name, parameter in planner.named_parameters()
        }
        parameters = (
            terminal.weight,
            terminal.bias,
            planner.refinement_gate_logits,
            planner.score_delta_gate_raw,
        )
        try:
            names = tuple(parameter_names[id(parameter)] for parameter in parameters)
        except KeyError as exc:
            raise RuntimeError(
                "F2 planner does not expose the required functional actor state"
            ) from exc
        prefix = "_pad_model._actworld_planner."
        return tuple(prefix + name for name in names)

    def _validate_f2_actor_zero_state(
        self, state: Dict[str, torch.Tensor], path: str
    ) -> None:
        """Reject checkpoints whose frozen F2 refiner can alter proposals."""

        if (
            getattr(self, "_critic_variant", "legacy") != self.F2_CRITIC_VARIANT
            or getattr(self, "_training_stage", "critic") == "actor"
        ):
            return
        required = self._f2_actor_zero_state_keys()
        missing = [key for key in required if key not in state]
        non_tensor = [
            key for key in required if key in state and not isinstance(state[key], torch.Tensor)
        ]
        nonzero = [
            key
            for key in required
            if key in state
            and isinstance(state[key], torch.Tensor)
            and bool(torch.count_nonzero(state[key].detach()).item())
        ]
        if missing or non_tensor or nonzero:
            raise RuntimeError(
                f"F2 checkpoint {path!r} violates the frozen actor invariant: "
                f"missing={missing}, non_tensor={non_tensor}, nonzero={nonzero}"
            )

    def validate_full_checkpoint_contract(
        self, checkpoint: Dict[str, Any], path: str
    ) -> None:
        """Validate lineage and functional state before a full restore.

        This method is also the single validation seam for Lightning resume.
        A trained F2 residual value head is intentionally not constrained.
        """

        self._validate_checkpoint_variant(checkpoint, path)
        if self._critic_variant != self.F2_CRITIC_VARIANT:
            return
        saved_stage = checkpoint.get("actworld_training_stage")
        expected_stage = getattr(self, "_training_stage", "critic")
        if saved_stage != expected_stage:
            raise ValueError(
                f"F2 checkpoint {path!r} must have top-level "
                f"actworld_training_stage={expected_stage!r}; "
                f"found {saved_stage!r}"
            )
        state = self._state_from_checkpoint(checkpoint, path)
        if expected_stage == "critic":
            self._validate_f2_actor_zero_state(state, path)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Enforce the F2 actor invariant during recursive Lightning restores."""

        planner_prefix = prefix + "_pad_model._actworld_planner."
        if (
            getattr(self, "_critic_variant", "legacy") == self.F2_CRITIC_VARIANT
            and any(key.startswith(planner_prefix) for key in state_dict)
        ):
            local_state = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            # Validate before delegating so a rejected recursive restore cannot
            # partially copy the nonzero actor state into this agent.
            self._validate_f2_actor_zero_state(local_state, "<state_dict>")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _load_state(self, path: str, *, baseline_only: bool) -> None:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        f2_baseline = bool(
            baseline_only
            and getattr(self, "_critic_variant", "legacy")
            == self.F2_CRITIC_VARIANT
        )
        checkpoint = self._checkpoint_payload(path, f2_baseline=f2_baseline)
        if not baseline_only:
            self.validate_full_checkpoint_contract(checkpoint, path)
        state = self._state_from_checkpoint(checkpoint, path)
        if baseline_only:
            planner_prefix = "_pad_model._actworld_planner."
            state = {
                key: value
                for key, value in state.items()
                if not key.startswith(planner_prefix)
            }
            if f2_baseline:
                expected = {
                    key: value
                    for key, value in self.state_dict().items()
                    if not key.startswith(planner_prefix)
                }
                unexpected = sorted(set(state).difference(expected))
                missing = sorted(set(expected).difference(state))
                non_tensor = sorted(
                    key for key, value in state.items()
                    if not isinstance(value, torch.Tensor)
                )
                bad_shape = sorted(
                    key for key in set(state).intersection(expected)
                    if isinstance(state[key], torch.Tensor)
                    and tuple(state[key].shape) != tuple(expected[key].shape)
                )
                bad_dtype = sorted(
                    key for key in set(state).intersection(expected)
                    if isinstance(state[key], torch.Tensor)
                    and state[key].dtype != expected[key].dtype
                )
                if unexpected or missing or non_tensor or bad_shape or bad_dtype:
                    raise RuntimeError(
                        "F2 Drive-JEPA baseline state is structurally incompatible: "
                        f"unexpected={unexpected}, missing={missing}, "
                        f"non_tensor={non_tensor}, bad_shape={bad_shape}, "
                        f"bad_dtype={bad_dtype}"
                    )
        incompatible = self.load_state_dict(
            state, strict=not baseline_only
        )
        if baseline_only:
            unexpected = list(incompatible.unexpected_keys)
            invalid_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith("_pad_model._actworld_planner.")
            ]
            if unexpected or invalid_missing:
                raise RuntimeError(
                    "Drive-JEPA warm start is structurally incompatible: "
                    f"unexpected={unexpected}, missing={invalid_missing}"
                )

    def _load_stage_a_critic(self, path: str) -> None:
        checkpoint = self._checkpoint_payload(path)
        saved_stage = checkpoint.get("actworld_training_stage")
        if saved_stage != "critic":
            raise ValueError(
                f"Stage-A checkpoint {path!r} must have top-level "
                "actworld_training_stage='critic'; "
                f"found {saved_stage!r}"
            )
        self._validate_checkpoint_variant(checkpoint, path)
        state = self._state_from_checkpoint(checkpoint, path)
        prefix = "_pad_model._actworld_planner."
        critic_source = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not critic_source:
            raise RuntimeError(
                f"Stage-A checkpoint {path!r} contains no ActWorld planner state"
            )
        self._pad_model._actworld_planner.load_stage_a_critic_state(
            critic_source
        )

    def initialize(self) -> None:
        if self._checkpoint_path:
            self._load_state(self._checkpoint_path, baseline_only=False)

    def get_sensor_config(self) -> SensorConfig:
        front_indices = [1, 2, 3]
        if self._future_grounding_enabled:
            front_indices.extend([10, 11])
        return SensorConfig(
            cam_f0=front_indices,
            cam_l0=[3],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[3],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[3],
            lidar_pc=[],
        )

    def get_target_builders(self):
        return [
            ActWorldJEPATargetBuilder(
                config=self._config,
                enable_future=self._future_grounding_enabled,
            )
        ]

    def get_feature_builders(self):
        return [ActWorldJEPAFeatureBuilder(config=self._config)]

    def forward_f2_parallel_loss_diagnostics(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Explicit non-causal validation path for F2 loss diagnostics only."""

        if self.training:
            raise RuntimeError(
                "F2 parallel loss diagnostics require agent.eval()"
            )
        if (
            self._training_stage != "critic"
            or self._critic_variant != self.F2_CRITIC_VARIANT
        ):
            raise RuntimeError(
                "F2 parallel loss diagnostics require an F2 Stage-A critic"
            )
        prediction = self._pad_model(
            features, f2_parallel_loss_diagnostics=True
        )
        missing = sorted(
            self._F2_PARALLEL_REQUIRED_PREDICTIONS.difference(prediction)
        )
        if missing:
            raise RuntimeError(
                "F2 parallel loss diagnostic forward is missing predictions: "
                f"{missing}"
            )
        return prediction

    @classmethod
    def _validate_f2_fixed_config(cls, config: ActWorldJEPAConfig) -> None:
        """Reject any drift from the preregistered F2 coefficients."""

        mismatched = []
        for name, expected in cls._F2_FIXED_CONFIG.items():
            value = getattr(config, name, None)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) != expected
            ):
                mismatched.append(f"{name}={value!r} (required {expected!r})")
        if mismatched:
            raise ValueError(
                "F2 weights and margins are fixed by the preregistered protocol: "
                + ", ".join(mismatched)
            )

    @staticmethod
    def _f2_validate_labels(
        utility_logits: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate and normalize the fixed 32-candidate/9-channel contract."""

        if not isinstance(utility_logits, torch.Tensor) or utility_logits.ndim != 2:
            raise ValueError("F2 utility_logits must have shape [B, C]")
        if (
            not isinstance(target_scores, torch.Tensor)
            or target_scores.ndim != 3
            or target_scores.shape[-1] != 9
            or tuple(target_scores.shape[:2]) != tuple(utility_logits.shape)
        ):
            raise ValueError("F2 target_scores must have shape [B, C, 9]")
        if utility_logits.shape[0] < 1 or utility_logits.shape[1] < 1:
            raise ValueError("F2 requires at least one scene and one candidate")
        utility = utility_logits.float()
        labels = target_scores.float()
        if not bool(torch.isfinite(utility).all().item()):
            raise ValueError("F2 utility logits must be finite")
        if not bool(torch.isfinite(labels).all().item()):
            raise ValueError("F2 target scores must be finite")
        if bool(((labels < 0.0) | (labels > 1.0)).any().item()):
            raise ValueError("F2 target scores must be in [0, 1]")
        return utility, labels

    @staticmethod
    def _f2_masked_argmax(
        utility_logits: torch.Tensor,
        eligible: torch.Tensor,
    ) -> torch.Tensor:
        """Highest detached utility with lowest-index tie breaking."""

        if tuple(eligible.shape) != tuple(utility_logits.shape):
            raise ValueError("F2 argmax mask and utilities must have the same shape")
        if eligible.dtype != torch.bool:
            raise TypeError("F2 argmax mask must be boolean")
        masked = utility_logits.detach().masked_fill(~eligible, -torch.inf)
        # torch.argmax returns the first maximum, i.e. the lowest original
        # candidate index.  Empty rows intentionally return zero; callers mask
        # their per-scene hinge to zero before the batch mean.
        return torch.argmax(masked, dim=1)

    @classmethod
    def _f2_margin_terms(
        cls,
        utility_logits: torch.Tensor,
        target_scores: torch.Tensor,
        *,
        top_margin: float = 0.2,
        safe_unsafe_margin: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Compute the two deterministic hardest-negative F2 margins."""

        utility, labels = cls._f2_validate_labels(
            utility_logits, target_scores
        )
        aggregate = labels[..., 8]
        rows = torch.arange(utility.shape[0], device=utility.device)

        # Safety has priority over aggregate regret.  If any exactly-safe
        # candidate exists, both sides of the top-margin comparison are drawn
        # from that same safety stratum; otherwise the full candidate set is
        # used.  This prevents an unsafe aggregate oracle from producing the
        # exact opposite gradient of the safe-vs-unsafe margin.
        safe = (labels[..., :4] == 1.0).all(dim=-1)
        has_safe = safe.any(dim=1, keepdim=True)
        top_pool = torch.where(has_safe, safe, torch.ones_like(safe))
        pooled_aggregate = aggregate.masked_fill(~top_pool, -torch.inf)
        oracle_value = pooled_aggregate.amax(dim=1, keepdim=True)
        oracle = top_pool & (aggregate == oracle_value)
        non_oracle = top_pool & (aggregate < oracle_value)
        top_positive_index = cls._f2_masked_argmax(utility, oracle)
        top_negative_index = cls._f2_masked_argmax(utility, non_oracle)
        top_hinge = F.relu(
            utility.new_tensor(float(top_margin))
            - utility[rows, top_positive_index]
            + utility[rows, top_negative_index]
        )
        top_eligible = non_oracle.any(dim=1)
        top_per_scene = torch.where(
            top_eligible, top_hinge, torch.zeros_like(top_hinge)
        )

        # A candidate is safe only when all four hard channels equal one
        # exactly; fractional simulator targets remain valid soft BCE labels
        # but do not enter S.
        unsafe = ~safe
        safe_aggregate = aggregate.masked_fill(~safe, -torch.inf)
        best_safe_value = safe_aggregate.amax(dim=1, keepdim=True)
        best_safe = safe & (aggregate == best_safe_value)
        safe_positive_index = cls._f2_masked_argmax(utility, best_safe)
        unsafe_negative_index = cls._f2_masked_argmax(utility, unsafe)
        safe_hinge = F.relu(
            utility.new_tensor(float(safe_unsafe_margin))
            - utility[rows, safe_positive_index]
            + utility[rows, unsafe_negative_index]
        )
        safe_eligible = safe.any(dim=1) & unsafe.any(dim=1)
        safe_per_scene = torch.where(
            safe_eligible, safe_hinge, torch.zeros_like(safe_hinge)
        )

        return {
            "top_regret_margin": top_per_scene.mean(),
            "safe_vs_unsafe_margin": safe_per_scene.mean(),
            "top_positive_index": top_positive_index,
            "top_negative_index": top_negative_index,
            "safe_positive_index": safe_positive_index,
            "unsafe_negative_index": unsafe_negative_index,
            "top_eligible": top_eligible,
            "safe_eligible": safe_eligible,
        }

    @classmethod
    def _f2_world_supervision(
        cls,
        world_logits: torch.Tensor,
        utility_logits: torch.Tensor,
        collision_logits: torch.Tensor,
        target_scores: torch.Tensor,
        *,
        top_margin: float,
        safe_unsafe_margin: float,
    ) -> Dict[str, torch.Tensor]:
        """Absolute and hard-negative terms in the fixed F2 objective."""

        utility, labels = cls._f2_validate_labels(
            utility_logits, target_scores
        )
        if tuple(world_logits.shape) != tuple(labels.shape):
            raise ValueError("F2 world logits and target scores must have the same shape")
        if tuple(collision_logits.shape) != tuple(utility.shape):
            raise ValueError("F2 collision logits must have shape [B, C]")
        margins = cls._f2_margin_terms(
            utility,
            labels,
            top_margin=top_margin,
            safe_unsafe_margin=safe_unsafe_margin,
        )
        return {
            "aggregate_bce": F.binary_cross_entropy_with_logits(
                utility, labels[..., 8]
            ),
            "top_regret_margin": margins["top_regret_margin"],
            "safe_vs_unsafe_margin": margins["safe_vs_unsafe_margin"],
            "nine_channel_bce": F.binary_cross_entropy_with_logits(
                world_logits.float(), labels
            ),
            "collision_bce": F.binary_cross_entropy_with_logits(
                collision_logits.float(), 1.0 - labels[..., 0]
            ),
        }

    @staticmethod
    def _soft_rank_loss(
        logits: torch.Tensor,
        target_scores: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError("actworld_rank_temperature must be positive")
        target_distribution = torch.softmax(
            target_scores.float() / temperature, dim=1
        )
        return -(target_distribution * torch.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()

    def _world_supervision(
        self,
        world_logits: torch.Tensor,
        value_logits: torch.Tensor,
        collision_logits: torch.Tensor,
        composite_logits: torch.Tensor,
        target_scores: torch.Tensor,
        temperature: float,
    ) -> Dict[str, torch.Tensor]:
        """Train the absolute critic utility and its auxiliary factor heads."""

        target_scores = target_scores.float()
        return {
            "metric": F.binary_cross_entropy_with_logits(
                world_logits.float(), target_scores
            ),
            "value": F.binary_cross_entropy_with_logits(
                value_logits.float(), target_scores[..., -1]
            ),
            "collision": F.binary_cross_entropy_with_logits(
                collision_logits.float(), 1.0 - target_scores[..., 0]
            ),
            "fusion": F.binary_cross_entropy_with_logits(
                composite_logits.float(), target_scores[..., -1]
            ),
            "rank": self._soft_rank_loss(
                value_logits, target_scores[..., -1], temperature
            ),
        }

    def _f2_pad_loss(
        self,
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
        config: ActWorldJEPAConfig,
    ) -> Dict[str, torch.Tensor]:
        """Independent F2 branch; no legacy rank, fusion, or base loss."""

        self._validate_f2_fixed_config(config)
        required_f32 = {
            "proposals": pred.get("proposals"),
            "pred_logit": pred.get("pred_logit"),
            "actworld_base_candidate_logits": pred.get(
                "actworld_base_candidate_logits"
            ),
        }
        missing_f32 = [
            name for name, value in required_f32.items()
            if not isinstance(value, torch.Tensor)
        ]
        wrong_dtype = [
            name for name, value in required_f32.items()
            if isinstance(value, torch.Tensor) and value.dtype != torch.float32
        ]
        if missing_f32 or wrong_dtype:
            raise ValueError(
                "F2 frozen Drive-JEPA outputs must be present in float32: "
                f"missing={missing_f32}, wrong_dtype={wrong_dtype}"
            )
        pred_logit = required_f32["pred_logit"]
        base_logits = required_f32["actworld_base_candidate_logits"]
        assert isinstance(pred_logit, torch.Tensor)
        assert isinstance(base_logits, torch.Tensor)
        if pred_logit.ndim != 3 or pred_logit.shape[-1] < 1:
            raise ValueError("F2 pred_logit must have shape [B, C, K>=1]")
        if (
            tuple(base_logits.shape) != tuple(pred_logit.shape[:2])
            or not torch.equal(base_logits, pred_logit[..., -1])
        ):
            raise ValueError(
                "F2 base logits must exactly equal pred_logit[..., -1]"
            )
        final_scores, best_scores, target_scores = self.compute_score(
            targets,
            pred["proposals"].detach(),
            test=True,
            label_only=True,
        )
        if target_scores.shape[1] != 32 or pred["proposals"].shape[1] != 32:
            raise ValueError(
                "F2 requires the original 32 Drive-JEPA candidates in index order"
            )
        pred["_target_scores"] = target_scores
        pred["_target_final_scores"] = final_scores

        terms = self._f2_world_supervision(
            pred["actworld_initial_world_logits"],
            pred["actworld_initial_utility_logits"],
            pred["actworld_initial_collision_logits"],
            target_scores,
            top_margin=config.actworld_f2_top_regret_margin,
            safe_unsafe_margin=config.actworld_f2_safe_unsafe_margin,
        )
        latent_loss = pred["actworld_latent_consistency_loss"].float()
        zero = latent_loss.new_zeros(())
        future_latent_loss = zero
        future_valid_fraction = zero
        if self._future_grounding_enabled:
            required_targets = {
                "actworld_future_camera_clip",
                "actworld_future_camera_valid",
            }
            missing_targets = sorted(required_targets.difference(targets))
            if missing_targets:
                raise KeyError(
                    "Future latent grounding is enabled but target tensors are "
                    f"missing: {missing_targets}"
                )
            future_valid = targets["actworld_future_camera_valid"]
            future_latent_loss = self._pad_model.future_latent_grounding_loss(
                pred["actworld_scene_latent"],
                targets["trajectory"],
                targets["actworld_future_camera_clip"],
                future_valid,
            ).float()
            future_valid_fraction = future_valid.float().mean()

        total = (
            config.actworld_f2_aggregate_bce_weight * terms["aggregate_bce"]
            + config.actworld_f2_top_regret_margin_weight
            * terms["top_regret_margin"]
            + config.actworld_f2_safe_unsafe_margin_weight
            * terms["safe_vs_unsafe_margin"]
            + config.actworld_f2_nine_channel_bce_weight
            * terms["nine_channel_bce"]
            + config.actworld_f2_collision_bce_weight * terms["collision_bce"]
            + config.actworld_f2_historical_latent_weight * latent_loss
            + config.actworld_f2_future_latent_weight * future_latent_loss
        )

        selected = torch.argmax(pred["pdm_score"].detach(), dim=1)
        rows = torch.arange(selected.shape[0], device=selected.device)
        score = final_scores[rows, selected].mean()
        return {
            "loss": total,
            "score": score,
            "best_score": best_scores.mean(),
            "actworld_f2_aggregate_bce_loss": terms["aggregate_bce"],
            "actworld_f2_top_regret_margin_loss": terms[
                "top_regret_margin"
            ],
            "actworld_f2_safe_vs_unsafe_margin_loss": terms[
                "safe_vs_unsafe_margin"
            ],
            "actworld_f2_nine_channel_bce_loss": terms[
                "nine_channel_bce"
            ],
            "actworld_f2_collision_bce_loss": terms["collision_bce"],
            "actworld_latent_loss": latent_loss,
            "actworld_future_latent_loss": future_latent_loss,
            "actworld_future_camera_valid": future_valid_fraction,
            "actworld_f2_residual_abs_mean": pred[
                "actworld_initial_residual_logits"
            ].float().abs().mean(),
            "actworld_f2_delta_abs_mean": pred[
                "actworld_initial_delta_logits"
            ].float().abs().mean(),
        }

    def pad_loss(
        self,
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
        config: ActWorldJEPAConfig,
    ) -> Dict[str, torch.Tensor]:
        if (
            self._training_stage == "critic"
            and self._critic_variant == self.F2_CRITIC_VARIANT
        ):
            return self._f2_pad_loss(targets, pred, config)
        loss_dict = super().pad_loss(targets, pred, config)
        drive_jepa_planning_loss = loss_dict["loss"].float()
        target_scores = pred["_target_scores"].float()
        reference = pred["actworld_value_logits"].float().mean()
        zero = reference.new_zeros(())

        initial_terms = self._world_supervision(
            pred["actworld_initial_world_logits"],
            pred["actworld_initial_value_logits"],
            pred["actworld_initial_collision_logits"],
            pred["actworld_initial_composite_logits"],
            target_scores,
            config.actworld_rank_temperature,
        )

        refined = pred["refined_proposals"].float()
        refined_target_scores = None
        refined_terms = None
        if self._training_stage == "actor":
            if not config.actworld_rescore_refined_proposals:
                raise ValueError(
                    "actor training requires actworld_rescore_refined_proposals"
                )
            _, _, refined_target_scores = self.compute_score(
                targets,
                refined.detach(),
                test=True,
                label_only=True,
            )
            refined_terms = self._world_supervision(
                pred["actworld_world_logits"],
                pred["actworld_value_logits"],
                pred["actworld_collision_logits"],
                pred["actworld_composite_logits"],
                refined_target_scores,
                config.actworld_rank_temperature,
            )

        diagnostic_terms = refined_terms or initial_terms
        metric_loss = diagnostic_terms["metric"]
        value_loss = diagnostic_terms["value"]
        collision_loss = diagnostic_terms["collision"]
        fusion_loss = diagnostic_terms["fusion"]
        rank_loss = diagnostic_terms["rank"]

        latent_loss = zero
        future_latent_loss = zero
        future_valid_fraction = zero
        if self._training_stage == "critic":
            latent_loss = pred["actworld_latent_consistency_loss"].float()
            if self._future_grounding_enabled:
                required_targets = {
                    "actworld_future_camera_clip",
                    "actworld_future_camera_valid",
                }
                missing_targets = sorted(required_targets.difference(targets))
                if missing_targets:
                    raise KeyError(
                        "Future latent grounding is enabled but target tensors are "
                        f"missing: {missing_targets}"
                    )
                future_valid = targets["actworld_future_camera_valid"]
                future_latent_loss = self._pad_model.future_latent_grounding_loss(
                    pred["actworld_scene_latent"],
                    targets["trajectory"],
                    targets["actworld_future_camera_clip"],
                    future_valid,
                ).float()
                future_valid_fraction = future_valid.float().mean()

        target_trajectory = targets["trajectory"].float()
        refinement_imitation = (
            torch.linalg.vector_norm(
                refined - target_trajectory[:, None], ord=1, dim=-1
            )
            .mean(dim=-1)
            .amin(dim=1)
            .mean()
        )
        residual = pred["actworld_trajectory_residual"].float()
        trust_region_loss = residual.square().mean()
        smoothness_loss = pred["actworld_refinement_smoothness_loss"].float()

        actor_improvement_loss = zero
        actor_predicted_improvement = zero
        actor_factor_loss = zero
        actor_selected_imitation_loss = zero
        gate_loss = zero
        gate_rank_loss = zero
        if self._training_stage == "actor":
            temperature = float(config.actworld_actor_rank_temperature)
            if temperature <= 0.0:
                raise ValueError("actworld_actor_rank_temperature must be positive")
            base_logits = pred["actworld_base_candidate_logits"].float()
            actor_weights = torch.softmax(base_logits.detach() / temperature, dim=1)
            candidate_imitation = torch.linalg.vector_norm(
                refined - target_trajectory[:, None], ord=1, dim=-1
            ).mean(dim=-1)
            actor_selected_imitation_loss = (
                actor_weights * candidate_imitation
            ).sum(dim=1).mean()
            original_utility = pred["actworld_initial_utility_logits"].float().detach()
            refined_utility = pred["actworld_utility_logits"].float()
            utility_improvement = refined_utility - original_utility
            actor_improvement_loss = (
                actor_weights * F.softplus(-utility_improvement)
            ).sum(dim=1).mean()
            actor_predicted_improvement = (
                actor_weights * utility_improvement.detach()
            ).sum(dim=1).mean()

            # Stage-B already obtains exact NAVTRAIN outcomes for every
            # refined candidate.  Use all nine factors to train the actual
            # trajectory correction instead of retaining them as diagnostics
            # only.  The critic is frozen, so gradients flow through its
            # trajectory inputs only into the actor/refiner.
            actor_factor_loss = (
                metric_loss
                + value_loss
                + collision_loss
                + config.actworld_rank_loss_weight * rank_loss
            )

            if self._critic_variant == self.F2_CRITIC_VARIANT:
                # F2 deploys composite_logits directly and never reads the
                # legacy learned gate.  Supervise the deployed score path.
                # Keep a zero-valued dependency so DDP does not mistake the
                # retained compatibility scalar for an unused parameter.
                gate_logits = pred["actworld_composite_logits"].float()
                gate_logits = (
                    gate_logits
                    + 0.0
                    * self._pad_model._actworld_planner.score_delta_gate_raw
                )
            else:
                gate_output = self._pad_model._actworld_planner._fuse_scores(
                    base_logits.detach(), refined_utility.detach()
                )
                gate_logits = gate_output["composite_logits"]
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_logits.float(), refined_target_scores[..., -1].float()
            )
            gate_rank_loss = self._soft_rank_loss(
                gate_logits,
                refined_target_scores[..., -1],
                config.actworld_rank_temperature,
            )

        refined_selected_score = zero
        refined_oracle_score = zero
        refined_oracle_gap = zero
        refined_mean_delta = zero
        refined_labels_enabled = zero
        actor_fixed_delta = zero
        e2e_delta = zero
        if refined_target_scores is not None:
            rows = torch.arange(refined.shape[0], device=refined.device)
            selected = torch.argmax(pred["pdm_score"].detach(), dim=1)
            base_selected = torch.argmax(pred["base_pdm_score"].detach(), dim=1)
            refined_selected_score = refined_target_scores[rows, selected, -1].mean()
            refined_oracle_score = refined_target_scores[..., -1].amax(dim=1).mean()
            refined_oracle_gap = refined_oracle_score - refined_selected_score
            refined_mean_delta = (
                refined_target_scores[..., -1] - target_scores[..., -1]
            ).mean()
            original_base_score = target_scores[rows, base_selected, -1]
            actor_fixed_delta = (
                refined_target_scores[rows, base_selected, -1]
                - original_base_score
            ).mean()
            e2e_delta = refined_selected_score - original_base_score.mean()
            refined_labels_enabled = reference.new_ones(())

        if self._training_stage == "critic":
            critic_world_loss = metric_loss + value_loss + collision_loss
            total = (
                config.actworld_drive_jepa_planning_loss_weight
                * drive_jepa_planning_loss
                + config.actworld_value_loss_weight * critic_world_loss
                + config.actworld_rank_loss_weight * rank_loss
                + config.actworld_latent_loss_weight * latent_loss
                + config.actworld_future_latent_loss_weight * future_latent_loss
            )
        else:
            total = (
                config.actworld_drive_jepa_planning_loss_weight
                * drive_jepa_planning_loss
                + config.actworld_actor_improvement_weight * actor_improvement_loss
                + config.actworld_value_loss_weight * actor_factor_loss
                + config.actworld_actor_trust_weight * trust_region_loss
                + config.actworld_actor_smoothness_weight * smoothness_loss
                + config.actworld_gate_loss_weight * gate_loss
                + config.actworld_gate_rank_weight * gate_rank_loss
                + config.actworld_actor_selected_imitation_weight
                * actor_selected_imitation_loss
            )

        loss_dict.update(
            {
                "loss": total,
                "drive_jepa_planning_loss": drive_jepa_planning_loss,
                "actworld_metric_loss": metric_loss,
                "actworld_value_loss": value_loss,
                "actworld_collision_loss": collision_loss,
                "actworld_fusion_loss": fusion_loss,
                "actworld_rank_loss": rank_loss,
                "actworld_latent_loss": latent_loss,
                "actworld_future_latent_loss": future_latent_loss,
                "actworld_future_camera_valid": future_valid_fraction,
                "actworld_actor_improvement_loss": actor_improvement_loss,
                "actworld_actor_predicted_improvement": actor_predicted_improvement,
                "actworld_actor_factor_loss": actor_factor_loss,
                "actworld_actor_selected_imitation_loss": actor_selected_imitation_loss,
                "actworld_gate_loss": gate_loss,
                "actworld_gate_rank_loss": gate_rank_loss,
                "actworld_refinement_loss": refinement_imitation,
                "actworld_trust_region_loss": trust_region_loss,
                "actworld_smoothness_loss": smoothness_loss,
                "actworld_refinement_gate": pred["actworld_refinement_gate"].mean(),
                "actworld_score_delta_gate": pred["actworld_score_delta_gate"],
                "actworld_refined_labels_enabled": refined_labels_enabled,
                "actworld_refined_selected_score": refined_selected_score,
                "actworld_refined_oracle_score": refined_oracle_score,
                "actworld_refined_oracle_gap": refined_oracle_gap,
                "actworld_refined_mean_delta": refined_mean_delta,
                "actworld_actor_fixed_delta": actor_fixed_delta,
                "actworld_e2e_delta": e2e_delta,
            }
        )
        return loss_dict

    def get_optimizers(self):
        planner_parameters = [
            parameter
            for parameter in self._pad_model._actworld_planner.parameters()
            if parameter.requires_grad
        ]
        groups = [{"params": planner_parameters, "lr": self._lr}]

        if not self._config.actworld_freeze_drive_jepa:
            backbone_parameters = [
                parameter
                for parameter in self._pad_model._backbone.parameters()
                if parameter.requires_grad
            ]
            base_parameters = [
                parameter
                for name, parameter in self._pad_model.named_parameters()
                if parameter.requires_grad
                and not name.startswith("_actworld_planner.")
                and not name.startswith("_backbone.")
            ]
            groups.extend(
                [
                    {
                        "params": backbone_parameters,
                        "lr": self._config.actworld_backbone_lr_scale * self._lr,
                    },
                    {
                        "params": base_parameters,
                        "lr": self._config.actworld_base_lr_scale * self._lr,
                    },
                ]
            )
        return torch.optim.AdamW(groups, lr=self._lr, weight_decay=1e-4)
