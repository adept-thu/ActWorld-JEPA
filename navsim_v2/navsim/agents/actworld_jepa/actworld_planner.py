"""Candidate-conditioned latent world planning for Drive-JEPA.

This module deliberately starts from an existing proposal tensor.  It does not
contain a trajectory generator and therefore cannot replace Drive-JEPA's
multi-modal proposal head.  Instead, it performs the following differentiable
pipeline:

1. turn each proposed waypoint sequence into action tokens;
2. roll the current V-JEPA scene latent forward under those actions;
3. predict all nine NAVSIM score components, a value and collision risk;
4. make a small, smooth, trust-region correction to each proposal; and
5. roll out and evaluate the corrected proposals a second time.

Only PyTorch is required.  In particular, importing this file does not import
NAVSIM, nuPlan, mmdet, or the Drive-JEPA runtime configuration module.
"""

from __future__ import annotations

import copy
import math
import operator
import os
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ActWorldPlannerConfig:
    """Configuration for :class:`ActWorldPlanner`.

    The defaults match the public Drive-JEPA v2 planning seam: 256-dimensional
    V-JEPA tokens, 32 proposals and eight future poses.  ``position_scale`` and
    the dynamic scales only normalize action features; they do not clamp or
    modify the proposals supplied by Drive-JEPA.
    """

    token_dim: int = 256
    num_candidates: int = 32
    horizon: int = 8
    latent_tokens: int = 8
    num_heads: int = 8
    num_layers: int = 1
    hidden_dim: int = 512
    num_subscores: int = 9
    dropout: float = 0.0
    time_delta: float = 0.5
    position_scale: float = 50.0
    speed_scale: float = 20.0
    acceleration_scale: float = 10.0
    curvature_scale: float = 1.0
    max_refinement_xy: float = 1.0
    max_refinement_yaw: float = 0.15
    refinement_gate_init: float = 0.0
    consistency_cosine_weight: float = 0.5
    target_ema_decay: float = 0.996
    finite_checks: bool = True
    training_stage: str = "critic"
    critic_variant: str = "legacy"

    def validate(self) -> None:
        integer_values = {
            "token_dim": self.token_dim,
            "num_candidates": self.num_candidates,
            "horizon": self.horizon,
            "latent_tokens": self.latent_tokens,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "num_subscores": self.num_subscores,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2 for dynamic action features")
        if self.hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        if self.token_dim % self.num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if self.num_subscores != 9:
            raise ValueError("num_subscores must be 9 to match the full NAVSIM supervision")

        positive_values = {
            "time_delta": self.time_delta,
            "position_scale": self.position_scale,
            "speed_scale": self.speed_scale,
            "acceleration_scale": self.acceleration_scale,
            "curvature_scale": self.curvature_scale,
            "max_refinement_xy": self.max_refinement_xy,
            "max_refinement_yaw": self.max_refinement_yaw,
        }
        for name, value in positive_values.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")

        nonnegative_values = {
            "consistency_cosine_weight": self.consistency_cosine_weight,
        }
        for name, value in nonnegative_values.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout!r}")
        if not math.isfinite(float(self.refinement_gate_init)):
            raise ValueError("refinement_gate_init must be finite")
        if (
            not math.isfinite(float(self.target_ema_decay))
            or not 0.0 <= float(self.target_ema_decay) < 1.0
        ):
            raise ValueError(
                "target_ema_decay must be finite and in [0, 1), "
                f"got {self.target_ema_decay!r}"
            )
        if not isinstance(self.finite_checks, bool):
            raise ValueError("finite_checks must be a bool")
        if self.training_stage not in {"critic", "actor"}:
            raise ValueError("training_stage must be 'critic' or 'actor'")
        if self.critic_variant not in {"legacy", "f2_base_residual"}:
            raise ValueError(
                "critic_variant must be 'legacy' or 'f2_base_residual'"
            )
        # The F2 critic can be frozen as the world model for Stage-B actor
        # training.  Stage-A still keeps the actor exactly inert; Stage-B
        # starts from that zero state and optimizes only the refiner/gates.


_CONFIG_ALIASES: Dict[str, Tuple[str, ...]] = {
    "token_dim": ("token_dim", "tf_d_model"),
    "num_candidates": ("num_candidates", "proposal_num"),
    "horizon": ("horizon", "num_poses"),
    "latent_tokens": ("actworld_latent_tokens", "latent_tokens"),
    "num_heads": ("actworld_num_heads", "num_heads", "tf_num_head"),
    "num_layers": ("actworld_num_layers", "num_layers"),
    "hidden_dim": ("actworld_hidden_dim", "hidden_dim", "tf_d_ffn"),
    "num_subscores": ("actworld_num_subscores", "num_subscores", "score_num"),
    "dropout": ("actworld_dropout", "dropout", "tf_dropout"),
    "time_delta": ("actworld_time_delta", "time_delta"),
    "position_scale": ("actworld_position_scale", "world_position_scale", "position_scale"),
    "speed_scale": ("actworld_speed_scale", "speed_scale"),
    "acceleration_scale": ("actworld_acceleration_scale", "acceleration_scale"),
    "curvature_scale": ("actworld_curvature_scale", "curvature_scale"),
    "max_refinement_xy": (
        "actworld_refine_max_xy",
        "actworld_max_refinement_xy",
        "max_refinement_xy",
    ),
    "max_refinement_yaw": (
        "actworld_refine_max_heading",
        "actworld_max_refinement_yaw",
        "max_refinement_yaw",
    ),
    "refinement_gate_init": (
        "actworld_refine_gate_init",
        "actworld_refinement_gate_init",
        "refinement_gate_init",
    ),
    "consistency_cosine_weight": (
        "actworld_consistency_cosine_weight",
        "consistency_cosine_weight",
    ),
    "target_ema_decay": ("actworld_target_ema_decay", "target_ema_decay"),
    "finite_checks": ("actworld_finite_checks", "finite_checks"),
    "training_stage": ("actworld_training_stage", "training_stage"),
    "critic_variant": ("actworld_critic_variant", "critic_variant"),
}

_MISSING = object()


def _source_value(source: Any, names: Sequence[str]) -> Any:
    """Read the first available name from an object or mapping."""

    for name in names:
        if isinstance(source, Mapping):
            value = source.get(name, _MISSING)
        else:
            value = getattr(source, name, _MISSING)
        if value is not _MISSING:
            return value
    return _MISSING


def _resolve_config(source: Optional[Any], overrides: Mapping[str, Any]) -> ActWorldPlannerConfig:
    if source is not None and not isinstance(source, (ActWorldPlannerConfig, Mapping)):
        # A normal runtime dataclass/object is supported through attribute aliases.
        if isinstance(source, (str, bytes, int, float, bool)):
            raise TypeError("config must be a config object, mapping, or None")

    defaults = ActWorldPlannerConfig()
    values: Dict[str, Any] = {}
    for item in fields(ActWorldPlannerConfig):
        explicit = overrides.get(item.name, None)
        if explicit is not None:
            values[item.name] = explicit
            continue
        configured = _MISSING
        if source is not None:
            configured = _source_value(source, _CONFIG_ALIASES[item.name])
        values[item.name] = getattr(defaults, item.name) if configured is _MISSING else configured

    # Runtime configs occasionally use numpy integer scalars.  Normalize only
    # architecture dimensions, leaving invalid floating values for validate().
    for name in (
        "token_dim",
        "num_candidates",
        "horizon",
        "latent_tokens",
        "num_heads",
        "num_layers",
        "hidden_dim",
        "num_subscores",
    ):
        value = values[name]
        if not isinstance(value, bool):
            try:
                values[name] = operator.index(value)
            except TypeError:
                pass

    result = ActWorldPlannerConfig(**values)
    result.validate()
    return result


class _SpatialLatentMixer(nn.Module):
    """Small pre-norm Transformer operating only on the latent scene tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attention_norm": nn.LayerNorm(dim),
                        "attention": nn.MultiheadAttention(
                            dim,
                            num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "attention_dropout": nn.Dropout(dropout),
                        "ffn_norm": nn.LayerNorm(dim),
                        "ffn": nn.Sequential(
                            nn.Linear(dim, hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim, dim),
                            nn.Dropout(dropout),
                        ),
                    }
                )
            )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        state = latent_tokens
        for layer in self.layers:
            normalized = layer["attention_norm"](state)
            attended, _ = layer["attention"](
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )
            state = state + layer["attention_dropout"](attended)
            state = state + layer["ffn"](layer["ffn_norm"](state))
        return self.final_norm(state)


class ActWorldPlanner(nn.Module):
    """Evaluate and gently refine Drive-JEPA trajectory proposals.

    Args:
        config: An :class:`ActWorldPlannerConfig`, a mapping, or a Drive-JEPA-like
            object.  For the latter, ``tf_d_model``, ``proposal_num``,
            ``num_poses`` and ``tf_d_ffn`` are recognized without importing the
            Drive-JEPA config (and therefore without importing nuPlan).
        **explicit options: Optional overrides for the corresponding config
            fields.  This permits ``ActWorldPlanner(token_dim=..., ...)``.

    ``forward`` consumes the original proposals; it never creates or replaces
    the proposal distribution.  The output retains ``original_candidates`` so
    callers can train, compare, or fall back to Drive-JEPA's direct selection.
    """

    ACTION_FEATURE_DIM = 7

    def __init__(
        self,
        config: Optional[Any] = None,
        *,
        token_dim: Optional[int] = None,
        num_candidates: Optional[int] = None,
        horizon: Optional[int] = None,
        latent_tokens: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_subscores: Optional[int] = None,
        dropout: Optional[float] = None,
        time_delta: Optional[float] = None,
        position_scale: Optional[float] = None,
        speed_scale: Optional[float] = None,
        acceleration_scale: Optional[float] = None,
        curvature_scale: Optional[float] = None,
        max_refinement_xy: Optional[float] = None,
        max_refinement_yaw: Optional[float] = None,
        refinement_gate_init: Optional[float] = None,
        consistency_cosine_weight: Optional[float] = None,
        target_ema_decay: Optional[float] = None,
        finite_checks: Optional[bool] = None,
        training_stage: Optional[str] = None,
        critic_variant: Optional[str] = None,
    ) -> None:
        super().__init__()
        overrides = {
            "token_dim": token_dim,
            "num_candidates": num_candidates,
            "horizon": horizon,
            "latent_tokens": latent_tokens,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "num_subscores": num_subscores,
            "dropout": dropout,
            "time_delta": time_delta,
            "position_scale": position_scale,
            "speed_scale": speed_scale,
            "acceleration_scale": acceleration_scale,
            "curvature_scale": curvature_scale,
            "max_refinement_xy": max_refinement_xy,
            "max_refinement_yaw": max_refinement_yaw,
            "refinement_gate_init": refinement_gate_init,
            "consistency_cosine_weight": consistency_cosine_weight,
            "target_ema_decay": target_ema_decay,
            "finite_checks": finite_checks,
            "training_stage": training_stage,
            "critic_variant": critic_variant,
        }
        self.config = _resolve_config(config, overrides)
        dim = self.config.token_dim
        hidden = self.config.hidden_dim

        self.token_norm = nn.LayerNorm(dim)
        self.scene_queries = nn.Parameter(
            torch.empty(self.config.latent_tokens, dim)
        )
        nn.init.normal_(self.scene_queries, mean=0.0, std=0.02)
        self.scene_cross_attention = nn.MultiheadAttention(
            dim,
            self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.scene_projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        # BYOL-style target scene reduction.  The V-JEPA image encoder is
        # frozen, while these EMA copies prevent the online reducer/projector
        # from satisfying latent consistency by moving both sides together.
        self.target_token_norm = copy.deepcopy(self.token_norm)
        self.target_scene_queries = nn.Parameter(
            self.scene_queries.detach().clone(), requires_grad=False
        )
        self.target_scene_cross_attention = copy.deepcopy(self.scene_cross_attention)
        self.target_scene_projection = copy.deepcopy(self.scene_projection)
        for module in self._target_scene_modules():
            module.requires_grad_(False)
            module.eval()

        self.action_encoder = nn.Sequential(
            nn.Linear(self.ACTION_FEATURE_DIM, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.time_embedding = nn.Parameter(torch.empty(self.config.horizon, dim))
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
        self.transition = nn.GRUCell(dim, dim)
        self.spatial_mixer = _SpatialLatentMixer(
            dim,
            self.config.num_heads,
            hidden,
            self.config.dropout,
            self.config.num_layers,
        )
        self.rollout_norm = nn.LayerNorm(dim)

        feature_dim = dim * 3
        self.world_feature = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.evaluation_spatial_attention = nn.Linear(dim, 1)
        self.temporal_attention = nn.Linear(dim, 1)
        self.subscore_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, self.config.num_subscores),
        )
        self.value_head = nn.Sequential(
            nn.Linear(dim, max(hidden // 2, 2)),
            nn.GELU(),
            nn.Linear(max(hidden // 2, 2), 1),
        )
        if self.config.critic_variant == "f2_base_residual":
            # In F2 this existing scalar head is reinterpreted as r_i.  Keep
            # the legacy module graph and RNG sequence untouched by adding no
            # new parameters; only the explicitly selected F2 initialization
            # is changed.
            nn.init.zeros_(self.value_head[-1].weight)
            nn.init.zeros_(self.value_head[-1].bias)
        self.collision_head = nn.Sequential(
            nn.Linear(dim, max(hidden // 2, 2)),
            nn.GELU(),
            nn.Linear(max(hidden // 2, 2), 1),
        )

        # Auxiliary critic heads keep their ordinary initialization so the
        # complete rollout trunk receives gradients on the first critic step.
        # Exact Drive-JEPA parity is enforced by a separate zero score gate.
        self.score_delta_gate_raw = nn.Parameter(torch.zeros(()))

        refinement_hidden = max(hidden // 2, 4)
        self.refinement_head = nn.Sequential(
            nn.Linear(dim, refinement_hidden),
            nn.GELU(),
            nn.Linear(refinement_hidden, 3),
        )
        # Zero residual preserves the original proposals exactly on warm start;
        # the terminal layer receives gradients on the first update.
        nn.init.zeros_(self.refinement_head[-1].weight)
        nn.init.zeros_(self.refinement_head[-1].bias)
        self.refinement_spatial_attention = nn.Linear(dim, 1)
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
            persistent=True,
        )
        self._training_stage = self.config.training_stage
        self.configure_training_stage(self._training_stage)

    def _target_scene_modules(self) -> Tuple[nn.Module, ...]:
        return (
            self.target_token_norm,
            self.target_scene_cross_attention,
            self.target_scene_projection,
        )

    def _critic_modules(self) -> Tuple[nn.Module, ...]:
        return (
            self.token_norm,
            self.scene_cross_attention,
            self.scene_projection,
            self.action_encoder,
            self.transition,
            self.spatial_mixer,
            self.rollout_norm,
            self.world_feature,
            self.evaluation_spatial_attention,
            self.temporal_attention,
            self.subscore_head,
            self.value_head,
            self.collision_head,
        )

    def _actor_modules(self) -> Tuple[nn.Module, ...]:
        return (
            self.refinement_spatial_attention,
            self.refinement_head,
        )

    @staticmethod
    def _unique_parameters(
        modules: Sequence[nn.Module], extra: Sequence[nn.Parameter]
    ) -> Tuple[nn.Parameter, ...]:
        parameters = []
        seen = set()
        for parameter in tuple(extra) + tuple(
            parameter for module in modules for parameter in module.parameters()
        ):
            identifier = id(parameter)
            if identifier not in seen:
                seen.add(identifier)
                parameters.append(parameter)
        return tuple(parameters)

    def critic_parameters(self) -> Tuple[nn.Parameter, ...]:
        return self._unique_parameters(
            self._critic_modules(),
            (self.scene_queries, self.time_embedding),
        )

    def actor_parameters(self) -> Tuple[nn.Parameter, ...]:
        return self._unique_parameters(
            self._actor_modules(),
            (self.refinement_gate_logits, self.score_delta_gate_raw),
        )

    @property
    def training_stage(self) -> str:
        return self._training_stage

    def configure_training_stage(self, stage: str) -> None:
        if stage not in {"critic", "actor"}:
            raise ValueError("stage must be 'critic' or 'actor'")
        self._training_stage = stage
        critic_trainable = stage == "critic"
        for parameter in self.critic_parameters():
            parameter.requires_grad_(critic_trainable)
        for parameter in self.actor_parameters():
            parameter.requires_grad_(not critic_trainable)
        for module in self._target_scene_modules():
            module.requires_grad_(False)
            module.eval()
        self.target_scene_queries.requires_grad_(False)
        self._apply_stage_module_modes(self.training)

    def _apply_stage_module_modes(self, mode: bool) -> None:
        if not mode:
            return
        train_critic = self._training_stage == "critic"
        for module in self._critic_modules():
            module.train(train_critic)
        for module in self._actor_modules():
            module.train(not train_critic)
        for module in self._target_scene_modules():
            module.eval()

    @torch.no_grad()
    def reset_stage_b_actor(self) -> None:
        """Restore exact baseline parity before actor optimization."""

        nn.init.zeros_(self.refinement_head[-1].weight)
        nn.init.zeros_(self.refinement_head[-1].bias)
        self.refinement_gate_logits.zero_()
        self.score_delta_gate_raw.zero_()

    @staticmethod
    def _is_stage_a_critic_key(key: str) -> bool:
        prefixes = (
            "token_norm.",
            "scene_queries",
            "scene_cross_attention.",
            "scene_projection.",
            "target_token_norm.",
            "target_scene_queries",
            "target_scene_cross_attention.",
            "target_scene_projection.",
            "action_encoder.",
            "time_embedding",
            "transition.",
            "spatial_mixer.",
            "rollout_norm.",
            "world_feature.",
            "evaluation_spatial_attention.",
            "temporal_attention.",
            "subscore_head.",
            "value_head.",
            "collision_head.",
        )
        return key.startswith(prefixes)

    @torch.no_grad()
    def load_stage_a_critic_state(self, source: Mapping[str, torch.Tensor]) -> None:
        """Atomically copy a complete, shape-compatible Stage-A critic."""

        current = self.state_dict()
        required = [key for key in current if self._is_stage_a_critic_key(key)]
        missing = [key for key in required if key not in source]
        mismatched = [
            key
            for key in required
            if key in source and tuple(source[key].shape) != tuple(current[key].shape)
        ]
        if missing or mismatched:
            raise RuntimeError(
                "Stage-A critic checkpoint is incompatible: "
                f"missing={missing}, shape_mismatch={mismatched}"
            )
        replacement = dict(current)
        for key in required:
            replacement[key] = source[key].detach().to(
                device=current[key].device, dtype=current[key].dtype
            )
        self.load_state_dict(replacement, strict=True)
        self.reset_stage_b_actor()

    def train(self, mode: bool = True) -> "ActWorldPlanner":
        super().train(mode)
        self._apply_stage_module_modes(mode)
        return self

    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None) -> None:
        """Move the target scene reducer/projector toward the online branch."""

        momentum = self.config.target_ema_decay if decay is None else float(decay)
        if not math.isfinite(momentum) or not 0.0 <= momentum < 1.0:
            raise ValueError("EMA decay must be finite and in [0, 1)")

        self.target_scene_queries.mul_(momentum).add_(
            self.scene_queries.detach(), alpha=1.0 - momentum
        )
        online_modules = (
            self.token_norm,
            self.scene_cross_attention,
            self.scene_projection,
        )
        for target_module, online_module in zip(
            self._target_scene_modules(), online_modules
        ):
            for target_parameter, online_parameter in zip(
                target_module.parameters(), online_module.parameters()
            ):
                target_parameter.mul_(momentum).add_(
                    online_parameter.detach(), alpha=1.0 - momentum
                )
            for target_buffer, online_buffer in zip(
                target_module.buffers(), online_module.buffers()
            ):
                if torch.is_floating_point(target_buffer):
                    target_buffer.mul_(momentum).add_(
                        online_buffer.detach(), alpha=1.0 - momentum
                    )
                else:
                    target_buffer.copy_(online_buffer)

    def _check_finite(self, tensor: torch.Tensor, name: str) -> None:
        if self.config.finite_checks and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains NaN or Inf values")

    def _validate_scene_tokens(self, tokens: torch.Tensor, name: str) -> None:
        expected_dim = self.config.token_dim
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tokens.ndim != 3:
            raise ValueError(
                f"{name} must have shape [B, N, {expected_dim}], got {tuple(tokens.shape)}"
            )
        if tokens.shape[0] < 1 or tokens.shape[1] < 1 or tokens.shape[2] != expected_dim:
            raise ValueError(
                f"{name} must have shape [B>0, N>0, {expected_dim}], got {tuple(tokens.shape)}"
            )
        if not torch.is_floating_point(tokens):
            raise TypeError(f"{name} must have a floating-point dtype")
        self._check_finite(tokens, name)

    def _validate_candidates(self, candidates: torch.Tensor, batch_size: int) -> None:
        expected = (
            batch_size,
            self.config.num_candidates,
            self.config.horizon,
            3,
        )
        if not isinstance(candidates, torch.Tensor):
            raise TypeError("candidates must be a torch.Tensor")
        if tuple(candidates.shape) != expected:
            raise ValueError(
                "candidates must have shape "
                f"[B, C, H, 3]={expected}, got {tuple(candidates.shape)}"
            )
        if not torch.is_floating_point(candidates):
            raise TypeError("candidates must have a floating-point dtype")
        self._check_finite(candidates, "candidates")

    def _validate_base_logits(
        self,
        base_logits: Optional[torch.Tensor],
        batch_size: int,
        candidates: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits is None:
            return reference.new_zeros((batch_size, candidates))
        if not isinstance(base_logits, torch.Tensor):
            raise TypeError("base_logits must be a torch.Tensor or None")
        if base_logits.device != reference.device:
            raise ValueError("base_logits and current_tokens must be on the same device")
        if not torch.is_floating_point(base_logits):
            raise TypeError("base_logits must have a floating-point dtype")
        if base_logits.ndim == 2 and tuple(base_logits.shape) == (batch_size, candidates):
            candidate_logits = base_logits
        elif (
            base_logits.ndim == 3
            and tuple(base_logits.shape[:2]) == (batch_size, candidates)
            and base_logits.shape[2] >= 1
        ):
            # Drive-JEPA provides [B, C, 6], with the final channel used for
            # direct proposal selection.  [B, C, 1] is also convenient.
            candidate_logits = base_logits[..., -1]
        else:
            raise ValueError(
                "base_logits must have shape [B, C] or [B, C, K>=1], "
                f"got {tuple(base_logits.shape)}"
            )
        self._check_finite(candidate_logits, "base_logits")
        return candidate_logits.to(dtype=reference.dtype)

    def _validate_module_device(self, tensor: torch.Tensor) -> None:
        module_device = self.time_embedding.device
        if tensor.device != module_device:
            raise ValueError(
                f"inputs are on {tensor.device}, but ActWorldPlanner is on {module_device}"
            )

    def _encode_scene(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized = self.token_norm(tokens)
        queries = self.scene_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, weights = self.scene_cross_attention(
            queries,
            normalized,
            normalized,
            need_weights=True,
        )
        # Keep K spatial slots rather than collapsing all V-JEPA patches into a
        # single vector.  Cross-attention weights are returned for inspection.
        return self.scene_projection(attended + queries), weights

    @torch.no_grad()
    def _encode_target_scene(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reduce frozen V-JEPA tokens with the stop-gradient EMA branch."""

        self._validate_scene_tokens(tokens, "target_tokens")
        self._validate_module_device(tokens)
        normalized = self.target_token_norm(tokens.detach())
        queries = self.target_scene_queries.unsqueeze(0).expand(
            tokens.shape[0], -1, -1
        )
        attended, _ = self.target_scene_cross_attention(
            queries,
            normalized,
            normalized,
            need_weights=False,
        )
        return self.target_scene_projection(attended + queries).detach()

    def _action_features(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Derive stable seven-dimensional action tokens from waypoint poses."""

        xy = trajectories[..., :2]
        yaw = trajectories[..., 2]
        previous_xy = torch.cat((torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]), dim=-2)
        previous_yaw = torch.cat((torch.zeros_like(yaw[..., :1]), yaw[..., :-1]), dim=-1)
        delta_xy = xy - previous_xy
        delta_yaw = torch.atan2(torch.sin(yaw - previous_yaw), torch.cos(yaw - previous_yaw))

        distance = torch.linalg.vector_norm(delta_xy, dim=-1)
        speed = distance / self.config.time_delta
        previous_speed = torch.cat(
            (torch.zeros_like(speed[..., :1]), speed[..., :-1]), dim=-1
        )
        acceleration = (speed - previous_speed) / self.config.time_delta
        distance_floor = distance.clamp_min(torch.finfo(distance.dtype).eps)
        curvature = delta_yaw / distance_floor

        return torch.stack(
            (
                torch.tanh(delta_xy[..., 0] / self.config.position_scale),
                torch.tanh(delta_xy[..., 1] / self.config.position_scale),
                torch.sin(delta_yaw),
                torch.cos(delta_yaw),
                torch.tanh(speed / self.config.speed_scale),
                torch.tanh(acceleration / self.config.acceleration_scale),
                torch.tanh(curvature / self.config.curvature_scale),
            ),
            dim=-1,
        )

    def _rollout(
        self,
        scene_latent: torch.Tensor,
        trajectories: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, candidates, horizon, _ = trajectories.shape
        latent_tokens = self.config.latent_tokens
        actions = self._action_features(trajectories)
        encoded_actions = self.action_encoder(actions)
        encoded_actions = encoded_actions + self.time_embedding.view(1, 1, horizon, -1)

        state = scene_latent[:, None, :, :].expand(-1, candidates, -1, -1).reshape(
            batch * candidates, latent_tokens, -1
        )
        rollout = []
        for step in range(horizon):
            action = encoded_actions[:, :, step].reshape(batch * candidates, -1)
            action = action[:, None, :].expand(-1, latent_tokens, -1)
            state = self.transition(
                action.reshape(batch * candidates * latent_tokens, -1),
                state.reshape(batch * candidates * latent_tokens, -1),
            ).reshape(batch * candidates, latent_tokens, -1)
            state = self.spatial_mixer(state)
            state = self.rollout_norm(state)
            rollout.append(state.reshape(batch, candidates, latent_tokens, -1))
        return torch.stack(rollout, dim=2), actions

    def _evaluate_world(
        self,
        rollout_latents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        spatial_weights = torch.softmax(
            self.evaluation_spatial_attention(rollout_latents).squeeze(-1), dim=3
        )
        spatial_latents = torch.sum(
            rollout_latents * spatial_weights.unsqueeze(-1), dim=3
        )
        temporal_weights = torch.softmax(
            self.temporal_attention(spatial_latents).squeeze(-1), dim=2
        )
        attended = torch.sum(spatial_latents * temporal_weights.unsqueeze(-1), dim=2)
        mean = spatial_latents.mean(dim=2)
        last = spatial_latents[:, :, -1]
        features = self.world_feature(torch.cat((attended, mean, last), dim=-1))

        world_logits = self.subscore_head(features)
        value_logits = self.value_head(features).squeeze(-1)
        collision_logits = self.collision_head(features).squeeze(-1)
        collision_probabilities = torch.sigmoid(collision_logits)
        return {
            "world_logits": world_logits,
            "candidate_subscores": torch.sigmoid(world_logits),
            "value_logits": value_logits,
            "utility_logits": value_logits,
            "candidate_values": torch.sigmoid(value_logits),
            "collision_logits": collision_logits,
            "collision_probabilities": collision_probabilities,
            "spatial_attention": spatial_weights,
            "temporal_attention": temporal_weights,
            "world_features": features,
        }

    def _fuse_scores(
        self,
        base_candidate_logits: torch.Tensor,
        utility_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Interpolate from the exact Drive-JEPA prior to critic utility."""

        if tuple(base_candidate_logits.shape) != tuple(utility_logits.shape):
            raise ValueError("base and utility logits must have the same shape")
        gate_override = os.environ.get("ACTWORLD_SCORE_DELTA_GATE")
        if gate_override is None:
            gate = torch.tanh(self.score_delta_gate_raw).to(utility_logits.dtype)
        else:
            gate_value = float(gate_override)
            if not math.isfinite(gate_value) or not -1.0 <= gate_value <= 1.0:
                raise ValueError(
                    "ACTWORLD_SCORE_DELTA_GATE must be finite and in [-1, 1], "
                    f"got {gate_value!r}"
                )
            gate = utility_logits.new_tensor(gate_value)
        delta_logits = utility_logits - base_candidate_logits.detach()
        composite_logits = base_candidate_logits + gate * delta_logits
        return {
            "base_candidate_logits": base_candidate_logits,
            "delta_logits": delta_logits,
            "score_delta_gate": gate,
            "composite_logits": composite_logits,
            # Drive-JEPA applies sigmoid directly to its final score logit.
            "world_scores": torch.sigmoid(composite_logits),
        }

    @staticmethod
    def _f2_base_residual_scores(
        base_candidate_logits: torch.Tensor,
        residual_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compose the preregistered F2 utility without a fusion gate.

        The Drive-JEPA aggregate logit is a frozen prior.  The learned scalar
        head can move that prior by at most one logit in either direction.
        Detaching the base here is part of the F2 contract, independent of the
        broader model-freezing policy enforced by the agent.
        """

        if tuple(base_candidate_logits.shape) != tuple(residual_logits.shape):
            raise ValueError("base and F2 residual logits must have the same shape")
        residual_scale = float(
            os.environ.get("ACTWORLD_F2_RESIDUAL_SCALE", "1.0")
        )
        if not math.isfinite(residual_scale) or not 0.0 <= residual_scale <= 1.0:
            raise ValueError(
                "ACTWORLD_F2_RESIDUAL_SCALE must be finite and in [0, 1], "
                f"got {residual_scale!r}"
            )
        delta_mode = os.environ.get("ACTWORLD_F2_DELTA_MODE", "absolute")
        if delta_mode == "absolute":
            # Backward-compatible F2 deployment: interpret the learned head
            # directly as a bounded correction.
            raw_delta = residual_logits
        elif delta_mode == "relative":
            # The F2 value head is supervised against the absolute aggregate
            # PDM target.  Convert that absolute prediction into an advantage
            # over the frozen Drive-JEPA prior before residual deployment;
            # otherwise the prior is counted twice.
            raw_delta = residual_logits - base_candidate_logits.detach()
        else:
            raise ValueError(
                "ACTWORLD_F2_DELTA_MODE must be 'absolute' or 'relative', "
                f"got {delta_mode!r}"
            )
        bounded_delta = residual_scale * torch.tanh(raw_delta)
        utility_logits = base_candidate_logits.detach() + bounded_delta
        return {
            "base_candidate_logits": base_candidate_logits,
            "residual_logits": residual_logits,
            "delta_logits": bounded_delta,
            # Retained as a zero diagnostic for the stable planner API.  F2
            # never reads score_delta_gate_raw and has no fusion-loss branch.
            "score_delta_gate": utility_logits.new_zeros(()),
            "utility_logits": utility_logits,
            "composite_logits": utility_logits,
            "world_scores": torch.sigmoid(utility_logits),
        }

    @staticmethod
    def _f2_factorized_utility_logits(world_logits: torch.Tensor) -> torch.Tensor:
        """Recompose the nine trained factor heads into the official EPDMS form."""

        if world_logits.ndim != 3 or world_logits.shape[-1] != 9:
            raise ValueError("F2 world logits must have shape [B, C, 9]")
        factors = torch.sigmoid(world_logits)
        safety_multiplier = factors[..., :4].prod(dim=-1)
        weighted_quality = (
            5.0 * factors[..., 4]
            + 5.0 * factors[..., 5]
            + 2.0 * factors[..., 6]
            + 2.0 * factors[..., 7]
            + 2.0 * factors[..., 8]
        ) / 16.0
        score = safety_multiplier * weighted_quality
        epsilon = torch.finfo(score.dtype).eps
        return torch.logit(score.clamp(min=epsilon, max=1.0 - epsilon))

    def _score_evaluation(
        self,
        base_candidate_logits: torch.Tensor,
        evaluation: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Apply the selected critic parameterization to one rollout."""

        if self.config.critic_variant == "legacy":
            return evaluation, self._fuse_scores(
                base_candidate_logits, evaluation["utility_logits"]
            )

        utility_mode = os.environ.get("ACTWORLD_F2_UTILITY_MODE", "aggregate")
        if utility_mode == "aggregate":
            residual_logits = evaluation["utility_logits"]
        elif utility_mode == "factorized":
            residual_logits = self._f2_factorized_utility_logits(
                evaluation["world_logits"]
            )
        else:
            raise ValueError(
                "ACTWORLD_F2_UTILITY_MODE must be 'aggregate' or 'factorized', "
                f"got {utility_mode!r}"
            )
        scores = self._f2_base_residual_scores(
            base_candidate_logits, residual_logits
        )
        evaluation = dict(evaluation)
        evaluation.update(
            {
                "residual_logits": residual_logits,
                "value_logits": scores["utility_logits"],
                "utility_logits": scores["utility_logits"],
                "candidate_values": scores["world_scores"],
            }
        )
        return evaluation, scores

    def actor_value_logits(
        self,
        scene_latent: torch.Tensor,
        refined_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate actor candidates with a frozen critic and input gradients."""

        rollout, _ = self._rollout(scene_latent, refined_candidates)
        return self._evaluate_world(rollout)["utility_logits"]

    @staticmethod
    def _low_pass(sequence: torch.Tensor) -> torch.Tensor:
        previous = torch.cat((sequence[:, :, :1], sequence[:, :, :-1]), dim=2)
        following = torch.cat((sequence[:, :, 1:], sequence[:, :, -1:]), dim=2)
        return (previous + 2.0 * sequence + following) * 0.25

    def _refine(
        self, rollout_latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Two low-pass applications suppress the second temporal difference.
        spatial_weights = torch.softmax(
            self.refinement_spatial_attention(rollout_latents).squeeze(-1), dim=3
        )
        spatial_latents = torch.sum(
            rollout_latents * spatial_weights.unsqueeze(-1), dim=3
        )
        raw = torch.tanh(self.refinement_head(spatial_latents))
        smooth = self._low_pass(self._low_pass(raw))
        ramp = torch.linspace(
            1.0 / self.config.horizon,
            1.0,
            self.config.horizon,
            device=smooth.device,
            dtype=smooth.dtype,
        ).view(1, 1, -1, 1)
        gate = torch.sigmoid(self.refinement_gate_logits).to(dtype=smooth.dtype)
        limits = self.refinement_limits.to(dtype=smooth.dtype)
        deployment_scale = float(
            os.environ.get("ACTWORLD_ACTOR_RESIDUAL_SCALE", "1.0")
        )
        if (
            not math.isfinite(deployment_scale)
            or not 0.0 <= deployment_scale <= 1.0
        ):
            raise ValueError(
                "ACTWORLD_ACTOR_RESIDUAL_SCALE must be finite and in [0, 1], "
                f"got {deployment_scale!r}"
            )
        residual = (
            smooth
            * ramp
            * gate.view(1, 1, 1, 3)
            * limits.view(1, 1, 1, 3)
            * deployment_scale
        )
        return residual, gate, spatial_weights

    @staticmethod
    def residual_smoothness_loss(residual: torch.Tensor) -> torch.Tensor:
        """Squared second-difference regularizer for optional training use."""

        if not isinstance(residual, torch.Tensor) or residual.ndim != 4 or residual.shape[-1] != 3:
            shape = tuple(residual.shape) if isinstance(residual, torch.Tensor) else type(residual)
            raise ValueError(f"residual must have shape [B, C, H, 3], got {shape}")
        if residual.shape[2] < 3:
            return residual.sum() * 0.0
        second_difference = residual[:, :, 2:] - 2.0 * residual[:, :, 1:-1] + residual[:, :, :-2]
        return second_difference.square().mean()

    def forward(
        self,
        current_tokens: torch.Tensor,
        candidates: torch.Tensor,
        base_logits: Optional[torch.Tensor] = None,
        *,
        return_auxiliary: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Roll out, refine, and re-evaluate Drive-JEPA candidates.

        Args:
            current_tokens: Current V-JEPA tokens with shape ``[B, N, D]``.
            candidates: Existing Drive-JEPA proposals with shape ``[B, C, H, 3]``.
            base_logits: Drive-JEPA scorer logits as ``[B, C]`` or ``[B, C, K]``.
                For the public six-channel scorer, the final channel is used.
            return_auxiliary: Include first-pass rollout/evaluation and diagnostics.

        Returns:
            A dictionary whose primary results are all from the second pass.
            ``world_logits`` has shape ``[B, C, 9]``; ``world_scores`` and
            ``collision_logits`` have shape ``[B, C]``; ``rollout_latents`` has
            shape ``[B, C, H, K, D]``, preserving ``K`` spatial world tokens.
        """

        if not isinstance(return_auxiliary, bool):
            raise TypeError("return_auxiliary must be a bool")
        self._validate_scene_tokens(current_tokens, "current_tokens")
        self._validate_module_device(current_tokens)
        self._validate_candidates(candidates, current_tokens.shape[0])
        if candidates.device != current_tokens.device:
            raise ValueError("candidates and current_tokens must be on the same device")
        if candidates.dtype != current_tokens.dtype:
            raise ValueError("candidates and current_tokens must have the same dtype")

        batch, candidate_count = candidates.shape[:2]
        base_candidate_logits = self._validate_base_logits(
            base_logits, batch, candidate_count, current_tokens
        )
        actor_training = self.training and self._training_stage == "actor"
        critic_training = self.training and self._training_stage == "critic"
        if actor_training:
            # The critic is frozen for Stage-B.  The initial path is a
            # comparison target only, so retaining its graph wastes memory.
            with torch.no_grad():
                scene_latent, scene_attention = self._encode_scene(current_tokens)
                initial_rollout, initial_actions = self._rollout(
                    scene_latent, candidates
                )
                initial_evaluation = self._evaluate_world(initial_rollout)
        else:
            scene_latent, scene_attention = self._encode_scene(current_tokens)
            critic_candidates = candidates.detach() if critic_training else candidates
            initial_rollout, initial_actions = self._rollout(
                scene_latent, critic_candidates
            )
            initial_evaluation = self._evaluate_world(initial_rollout)
        initial_evaluation, initial_fusion = self._score_evaluation(
            base_candidate_logits, initial_evaluation
        )
        trajectory_residual, refinement_gate, refinement_spatial_attention = self._refine(
            initial_rollout.detach()
        )
        refined_candidates = candidates + trajectory_residual
        self._check_finite(refined_candidates, "refined_candidates")

        # Stage-A critic targets must not train the refiner.  Stage-B instead
        # freezes critic parameters once, retains the trajectory-input graph,
        # and optimizes the actor through the absolute utility prediction.
        f2_original_proposal_route = (
            self.config.critic_variant == "f2_base_residual"
            and self._training_stage == "critic"
        )
        if critic_training or f2_original_proposal_route:
            # The Stage-A actor is frozen at exact zero, and only original
            # proposal labels enter its objective.  F2 also keeps this route
            # in eval mode so its compact utility output stays aligned with
            # the original 32-candidate validator manifest.
            rollout_latents = initial_rollout
            refined_actions = initial_actions
            final_evaluation = initial_evaluation
            if f2_original_proposal_route:
                final_fusion = initial_fusion
            else:
                # Preserve the legacy graph exactly: it historically built a
                # second score-fusion node even though the rollout was reused.
                final_evaluation, final_fusion = self._score_evaluation(
                    base_candidate_logits, final_evaluation
                )
        else:
            critic_candidates = (
                refined_candidates if actor_training else refined_candidates.detach()
            )
            critic_scene = scene_latent.detach() if actor_training else scene_latent
            rollout_latents, refined_actions = self._rollout(
                critic_scene, critic_candidates
            )
            final_evaluation = self._evaluate_world(rollout_latents)
            final_evaluation, final_fusion = self._score_evaluation(
                base_candidate_logits, final_evaluation
            )

        output: Dict[str, torch.Tensor] = {
            "original_candidates": candidates,
            "refined_candidates": refined_candidates,
            "trajectory_residual": trajectory_residual,
            "world_logits": final_evaluation["world_logits"],
            "candidate_subscores": final_evaluation["candidate_subscores"],
            "value_logits": final_evaluation["value_logits"],
            "utility_logits": final_evaluation["utility_logits"],
            "candidate_values": final_evaluation["candidate_values"],
            "base_candidate_logits": final_fusion["base_candidate_logits"],
            "delta_logits": final_fusion["delta_logits"],
            "score_delta_gate": final_fusion["score_delta_gate"],
            "composite_logits": final_fusion["composite_logits"],
            "world_scores": final_fusion["world_scores"],
            "collision_logits": final_evaluation["collision_logits"],
            "collision_probabilities": final_evaluation["collision_probabilities"],
            "world_features": final_evaluation["world_features"],
            "rollout_latents": rollout_latents,
        }
        if "residual_logits" in final_evaluation:
            output["residual_logits"] = final_evaluation["residual_logits"]
        if return_auxiliary:
            output.update(
                {
                    "scene_latent": scene_latent,
                    "scene_attention": scene_attention,
                    "action_features": refined_actions,
                    "initial_action_features": initial_actions,
                    "initial_rollout_latents": initial_rollout,
                    "initial_world_logits": initial_evaluation["world_logits"],
                    "initial_value_logits": initial_evaluation["value_logits"],
                    "initial_utility_logits": initial_evaluation["utility_logits"],
                    "initial_delta_logits": initial_fusion["delta_logits"],
                    "initial_composite_logits": initial_fusion["composite_logits"],
                    "initial_world_scores": initial_fusion["world_scores"],
                    "initial_collision_logits": initial_evaluation["collision_logits"],
                    "refinement_gate": refinement_gate,
                    "evaluation_spatial_attention": final_evaluation["spatial_attention"],
                    "refinement_spatial_attention": refinement_spatial_attention,
                    "refinement_smoothness_loss": self.residual_smoothness_loss(
                        trajectory_residual
                    ),
                }
            )
            if "residual_logits" in initial_evaluation:
                output["initial_residual_logits"] = initial_evaluation[
                    "residual_logits"
                ]

        if self.config.finite_checks:
            for name, value in output.items():
                if torch.is_floating_point(value) and not bool(torch.isfinite(value).all().item()):
                    raise RuntimeError(f"ActWorldPlanner produced non-finite output {name!r}")
        return output

    def predict_next_latent(
        self,
        previous_tokens: torch.Tensor,
        observed_action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one scene latent from history and an observed ``[dx, dy, yaw]`` action."""

        self._validate_scene_tokens(previous_tokens, "previous_tokens")
        self._validate_module_device(previous_tokens)
        if not isinstance(observed_action, torch.Tensor):
            raise TypeError("observed_action must be a torch.Tensor")
        if observed_action.ndim == 3 and observed_action.shape[1:] == (1, 3):
            observed_action = observed_action[:, 0]
        expected = (previous_tokens.shape[0], 3)
        if tuple(observed_action.shape) != expected:
            raise ValueError(f"observed_action must have shape {expected}, got {tuple(observed_action.shape)}")
        if not torch.is_floating_point(observed_action):
            raise TypeError("observed_action must have a floating-point dtype")
        if observed_action.device != previous_tokens.device:
            raise ValueError("observed_action and previous_tokens must be on the same device")
        if observed_action.dtype != previous_tokens.dtype:
            raise ValueError("observed_action and previous_tokens must have the same dtype")
        self._check_finite(observed_action, "observed_action")

        previous_latent, _ = self._encode_scene(previous_tokens)
        one_step_trajectory = observed_action[:, None, None, :]
        features = self._action_features(one_step_trajectory)[:, 0, 0]
        encoded = self.action_encoder(features) + self.time_embedding[0]
        latent_tokens = self.config.latent_tokens
        encoded = encoded[:, None, :].expand(-1, latent_tokens, -1)
        batch = previous_tokens.shape[0]
        predicted = self.transition(
            encoded.reshape(batch * latent_tokens, -1),
            previous_latent.reshape(batch * latent_tokens, -1),
        ).reshape(batch, latent_tokens, -1)
        return self.rollout_norm(self.spatial_mixer(predicted))

    def predict_terminal_latent(
        self,
        current_scene_latent: torch.Tensor,
        expert_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Roll one complete expert trajectory to its terminal spatial latent.

        Action features are derived once for the complete eight-step path, then
        consumed sequentially by the same transition used for proposals.
        """

        expected_latent_tail = (self.config.latent_tokens, self.config.token_dim)
        if (
            not isinstance(current_scene_latent, torch.Tensor)
            or current_scene_latent.ndim != 3
            or tuple(current_scene_latent.shape[1:]) != expected_latent_tail
        ):
            shape = (
                tuple(current_scene_latent.shape)
                if isinstance(current_scene_latent, torch.Tensor)
                else type(current_scene_latent)
            )
            raise ValueError(
                "current_scene_latent must have shape "
                f"[B, {self.config.latent_tokens}, {self.config.token_dim}], got {shape}"
            )
        expected_trajectory = (
            current_scene_latent.shape[0],
            self.config.horizon,
            3,
        )
        if (
            not isinstance(expert_trajectory, torch.Tensor)
            or tuple(expert_trajectory.shape) != expected_trajectory
        ):
            shape = (
                tuple(expert_trajectory.shape)
                if isinstance(expert_trajectory, torch.Tensor)
                else type(expert_trajectory)
            )
            raise ValueError(
                f"expert_trajectory must have shape {expected_trajectory}, got {shape}"
            )
        if not torch.is_floating_point(current_scene_latent) or not torch.is_floating_point(
            expert_trajectory
        ):
            raise TypeError("scene latent and expert trajectory must be floating point")
        if expert_trajectory.device != current_scene_latent.device:
            raise ValueError("expert trajectory and scene latent must be on the same device")
        if expert_trajectory.dtype != current_scene_latent.dtype:
            raise ValueError("expert trajectory and scene latent must have the same dtype")
        self._validate_module_device(current_scene_latent)
        self._check_finite(current_scene_latent, "current_scene_latent")
        self._check_finite(expert_trajectory, "expert_trajectory")

        rollout, _ = self._rollout(
            current_scene_latent, expert_trajectory[:, None]
        )
        return rollout[:, 0, -1]

    def terminal_latent_consistency_loss(
        self,
        current_scene_latent: torch.Tensor,
        expert_trajectory: torch.Tensor,
        future_tokens: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Align expert-action terminal prediction with an EMA future target."""

        batch = current_scene_latent.shape[0]
        self._validate_scene_tokens(future_tokens, "future_tokens")
        if future_tokens.shape[0] != batch:
            raise ValueError("future_tokens and current_scene_latent batch sizes differ")
        if future_tokens.device != current_scene_latent.device:
            raise ValueError("future tokens and scene latent must be on the same device")
        if future_tokens.dtype != current_scene_latent.dtype:
            raise ValueError("future tokens and scene latent must have the same dtype")
        if not isinstance(valid, torch.Tensor):
            raise TypeError("valid must be a torch.Tensor")
        if valid.ndim == 2 and valid.shape[1] == 1:
            valid = valid[:, 0]
        if tuple(valid.shape) != (batch,):
            raise ValueError(f"valid must have shape [{batch}], got {tuple(valid.shape)}")
        if valid.device != current_scene_latent.device:
            raise ValueError("valid and scene latent must be on the same device")

        predicted = self.predict_terminal_latent(
            current_scene_latent, expert_trajectory
        )
        target = self._encode_target_scene(future_tokens)
        predicted_normalized = F.normalize(predicted, dim=-1)
        target_normalized = F.normalize(target, dim=-1)
        regression = F.smooth_l1_loss(
            predicted_normalized, target_normalized, reduction="none"
        ).mean(dim=(1, 2))
        cosine = 1.0 - (
            predicted_normalized * target_normalized
        ).sum(dim=-1).mean(dim=1)
        per_sample = regression + self.config.consistency_cosine_weight * cosine
        valid_weight = valid.to(dtype=per_sample.dtype)
        loss = (per_sample * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)
        if self.config.finite_checks and not bool(torch.isfinite(loss).item()):
            raise RuntimeError("terminal_latent_consistency_loss produced NaN or Inf")
        return loss

    def latent_consistency_loss(
        self,
        previous_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        observed_action: torch.Tensor,
    ) -> torch.Tensor:
        """Historical one-step latent consistency objective.

        The target branch is detached, so this auxiliary loss trains the
        action-conditioned transition without pushing a frozen V-JEPA encoder.
        It combines normalized Smooth-L1 distance and cosine distance.
        """

        self._validate_scene_tokens(current_tokens, "current_tokens")
        if current_tokens.shape[0] != previous_tokens.shape[0]:
            raise ValueError("previous_tokens and current_tokens must have the same batch size")
        if current_tokens.device != previous_tokens.device:
            raise ValueError("previous_tokens and current_tokens must be on the same device")
        if current_tokens.dtype != previous_tokens.dtype:
            raise ValueError("previous_tokens and current_tokens must have the same dtype")

        predicted = self.predict_next_latent(previous_tokens, observed_action)
        target = self._encode_target_scene(current_tokens)
        predicted_normalized = F.normalize(predicted, dim=-1)
        target_normalized = F.normalize(target, dim=-1)
        regression = F.smooth_l1_loss(predicted_normalized, target_normalized)
        cosine = (1.0 - (predicted_normalized * target_normalized).sum(dim=-1)).mean()
        loss = regression + self.config.consistency_cosine_weight * cosine
        if self.config.finite_checks and not bool(torch.isfinite(loss).item()):
            raise RuntimeError("latent_consistency_loss produced NaN or Inf")
        return loss


__all__ = ["ActWorldPlanner", "ActWorldPlannerConfig"]
