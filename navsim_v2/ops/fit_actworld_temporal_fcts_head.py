#!/usr/bin/env python3
"""Train a NAVTRAIN-only temporal Planning-JEPA/FCTS head with group CV."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Policy:
    margin: float
    safety_threshold: float
    progress_threshold: float
    component_margin: float = 0.0
    hard_metric_threshold: float = 0.0
    quality_margin_threshold: float = -1000.0


class TemporalFCTSHead(nn.Module):
    """Explicit current/future and cross-cycle plan/future interaction head."""

    def __init__(
        self,
        plan_mean: np.ndarray,
        plan_std: np.ndarray,
        future_mean: np.ndarray,
        future_std: np.ndarray,
        latent_dim: int = 64,
        hard_metric_count: int = 0,
        quality_metric_count: int = 0,
    ) -> None:
        super().__init__()
        self.register_buffer("plan_mean", torch.as_tensor(plan_mean, dtype=torch.float32))
        self.register_buffer("plan_std", torch.as_tensor(plan_std, dtype=torch.float32))
        self.register_buffer("future_mean", torch.as_tensor(future_mean, dtype=torch.float32))
        self.register_buffer("future_std", torch.as_tensor(future_std, dtype=torch.float32))
        self.plan_encoder = nn.Sequential(
            nn.Linear(56, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, latent_dim)
        )
        self.future_encoder = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, latent_dim)
        )
        # 4 blocks for plan--future compatibility, plus 4 each for temporal
        # plan and temporal future consistency.
        self.interaction = nn.Sequential(
            nn.Linear(latent_dim * 12, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.component_head = nn.Linear(128, 4)
        self.safety_head = nn.Linear(128, 1)
        self.hard_metric_head = (
            nn.Linear(128, hard_metric_count) if hard_metric_count else None
        )
        self.quality_margin_head = (
            nn.Linear(128, quality_metric_count) if quality_metric_count else None
        )
        self.progress_head = nn.Linear(128, 1)
        self.refinement_head = nn.Linear(128, 1)

    @staticmethod
    def _relation(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.cat((first, second, first * second, torch.abs(first - second)), dim=-1)

    def forward(
        self,
        previous_plan: torch.Tensor,
        current_plan: torch.Tensor,
        previous_future: torch.Tensor,
        current_future: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, variants = current_plan.shape[:2]
        previous_plan = previous_plan.reshape(batch, 1, -1).expand(-1, variants, -1)
        current_plan = current_plan.reshape(batch, variants, -1)
        previous_future = previous_future[:, None].expand(-1, variants, -1)
        previous_plan = (previous_plan - self.plan_mean) / self.plan_std
        current_plan = (current_plan - self.plan_mean) / self.plan_std
        previous_future = (previous_future - self.future_mean) / self.future_std
        current_future = (current_future - self.future_mean) / self.future_std
        previous_plan_latent = self.plan_encoder(previous_plan)
        current_plan_latent = self.plan_encoder(current_plan)
        previous_future_latent = self.future_encoder(previous_future)
        current_future_latent = self.future_encoder(current_future)
        fused = self.interaction(
            torch.cat(
                (
                    self._relation(current_plan_latent, current_future_latent),
                    self._relation(current_plan_latent, previous_plan_latent),
                    self._relation(current_future_latent, previous_future_latent),
                ),
                dim=-1,
            )
        )
        result = {
            "component_ratio": F.softplus(self.component_head(fused)),
            "safety_logit": self.safety_head(fused).squeeze(-1),
            "progress_ratio": F.softplus(self.progress_head(fused).squeeze(-1)),
            "refinement_gain": self.refinement_head(fused).squeeze(-1),
        }
        if self.hard_metric_head is not None:
            result["hard_metric_logits"] = self.hard_metric_head(fused)
        if self.quality_margin_head is not None:
            result["quality_metric_margin"] = self.quality_margin_head(fused)
        return result


def _normalization(data: dict[str, np.ndarray], indices: np.ndarray) -> tuple[np.ndarray, ...]:
    previous_plan = data["previous_plan_tokens"][indices].reshape(-1, 56)
    current_plan = data["current_plan_tokens"][indices].reshape(-1, 56)
    plans = np.concatenate((previous_plan, current_plan), axis=0)
    previous_future = data["previous_future_latent"][indices].reshape(-1, 256)
    current_future = data["current_future_latent"][indices].reshape(-1, 256)
    futures = np.concatenate((previous_future, current_future), axis=0)
    return (
        plans.mean(axis=0).astype(np.float32),
        np.maximum(plans.std(axis=0), 1e-4).astype(np.float32),
        futures.mean(axis=0).astype(np.float32),
        np.maximum(futures.std(axis=0), 1e-4).astype(np.float32),
    )


def _tensor(data: dict[str, np.ndarray], name: str, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(data[name][indices], dtype=torch.float32, device=device)


def _fit(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> TemporalFCTSHead:
    torch.manual_seed(seed)
    hard_metric_count = int(data["hard_metric_valid"].shape[-1]) if "hard_metric_valid" in data else 0
    quality_metric_count = (
        int(data["quality_metric_margin"].shape[-1])
        if "quality_metric_margin" in data
        else 0
    )
    model = TemporalFCTSHead(
        *_normalization(data, indices),
        hard_metric_count=hard_metric_count,
        quality_metric_count=quality_metric_count,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    previous_plan = _tensor(data, "previous_plan_tokens", indices, device)
    current_plan = _tensor(data, "current_plan_tokens", indices, device)
    previous_future = _tensor(data, "previous_future_latent", indices, device)
    current_future = _tensor(data, "current_future_latent", indices, device)
    component = _tensor(data, "component_ratio", indices, device).clamp(max=20.0)
    safety = _tensor(data, "safety_progress_valid", indices, device)
    hard_metric = (
        _tensor(data, "hard_metric_valid", indices, device)
        if "hard_metric_valid" in data
        else None
    )
    quality_margin = (
        _tensor(data, "quality_metric_margin", indices, device).clamp(-1.0, 1.0)
        if "quality_metric_margin" in data
        else None
    )
    progress = _tensor(data, "progress_ratio", indices, device).clamp(max=2.0)
    pair = _tensor(data, "pair_utility", indices, device)
    gain = (pair - pair[:, :1]) * 100.0
    target_rank = gain.masked_fill(safety < 0.5, -100.0).argmax(dim=1)

    model.train()
    for epoch in range(epochs):
        prediction = model(previous_plan, current_plan, previous_future, current_future)
        component_loss = F.smooth_l1_loss(
            torch.log1p(prediction["component_ratio"]), torch.log1p(component)
        )
        safety_loss = F.binary_cross_entropy_with_logits(prediction["safety_logit"], safety)
        hard_metric_loss = (
            (
                F.binary_cross_entropy_with_logits(
                    prediction["hard_metric_logits"], hard_metric, reduction="none"
                )
                * torch.where(
                    hard_metric > 0.5,
                    torch.ones_like(hard_metric),
                    (
                        hard_metric.sum(dim=(0, 1), keepdim=True)
                        / (1.0 - hard_metric).sum(dim=(0, 1), keepdim=True).clamp_min(1.0)
                    ).clamp(max=200.0),
                )
            ).mean()
            if hard_metric is not None
            else torch.zeros((), device=device)
        )
        quality_margin_loss = (
            F.smooth_l1_loss(
                prediction["quality_metric_margin"], quality_margin * 20.0
            )
            if quality_margin is not None
            else torch.zeros((), device=device)
        )
        progress_loss = F.smooth_l1_loss(prediction["progress_ratio"], progress)
        gain_loss = F.smooth_l1_loss(prediction["refinement_gain"], gain)
        rank_loss = F.cross_entropy(prediction["refinement_gain"], target_rank)
        loss = (
            component_loss
            + 0.25 * safety_loss
            + 0.75 * hard_metric_loss
            + 0.5 * quality_margin_loss
            + 0.25 * progress_loss
            + gain_loss
            + 0.5 * rank_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            print(
                f"epoch={epoch + 1}/{epochs} loss={loss.item():.6f} "
                f"component={component_loss.item():.6f} rank={rank_loss.item():.6f}",
                flush=True,
            )
    return model.eval()


@torch.no_grad()
def _predict(model: TemporalFCTSHead, data: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    result = model(
        _tensor(data, "previous_plan_tokens", indices, device),
        _tensor(data, "current_plan_tokens", indices, device),
        _tensor(data, "previous_future_latent", indices, device),
        _tensor(data, "current_future_latent", indices, device),
    )
    return {name: value.detach().cpu().numpy() for name, value in result.items()}


def _choose(prediction: dict[str, np.ndarray], policy: Policy) -> np.ndarray:
    gain = prediction["refinement_gain"].copy()
    safety = 1.0 / (1.0 + np.exp(-prediction["safety_logit"]))
    allowed = (safety >= policy.safety_threshold) & (
        prediction["progress_ratio"] >= policy.progress_threshold
    )
    if "hard_metric_logits" in prediction and policy.hard_metric_threshold > 0.0:
        hard_metric_probability = 1.0 / (
            1.0 + np.exp(-prediction["hard_metric_logits"])
        )
        allowed &= np.all(
            hard_metric_probability >= policy.hard_metric_threshold, axis=-1
        )
    if "quality_metric_margin" in prediction:
        allowed &= np.all(
            prediction["quality_metric_margin"] >= policy.quality_margin_threshold,
            axis=-1,
        )
    component_quality = -np.log1p(prediction["component_ratio"]).mean(axis=-1)
    allowed &= component_quality >= (
        component_quality[:, :1] + policy.component_margin
    )
    allowed[:, 0] = True
    gain[~allowed] = -np.inf
    selected = np.argmax(gain, axis=1)
    best = gain[np.arange(len(gain)), selected]
    selected[(selected != 0) & (best - gain[:, 0] < policy.margin)] = 0
    return selected


def _evaluate(
    data: dict[str, np.ndarray], prediction: dict[str, np.ndarray], fold_ids: np.ndarray, policy: Policy
) -> dict[str, object]:
    selected = _choose(prediction, policy)
    rows = np.arange(len(selected))
    baseline_pair = data["pair_utility"][:, 0]
    chosen_pair = data["pair_utility"][rows, selected]
    baseline_ec = data["ec_pass"][:, 0].astype(np.float32)
    chosen_ec = data["ec_pass"][rows, selected].astype(np.float32)
    chosen_safe = data["safety_progress_valid"][rows, selected].astype(np.float32)
    fold_gains = [float((chosen_pair[fold_ids == fold] - baseline_pair[fold_ids == fold]).mean()) for fold in range(5)]
    return {
        "policy": asdict(policy),
        "changed": int(np.sum(selected != 0)),
        "changed_fraction": float(np.mean(selected != 0)),
        "pair_utility": float(chosen_pair.mean()),
        "pair_utility_gain": float((chosen_pair - baseline_pair).mean()),
        "ec_pass_rate": float(chosen_ec.mean()),
        "ec_gain": float((chosen_ec - baseline_ec).mean()),
        "safety_progress_valid_rate": float(chosen_safe.mean()),
        "fold_pair_utility_gains": fold_gains,
        "minimum_fold_gain": float(min(fold_gains)),
        "selected": selected.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as payload:
        data = {name: payload[name] for name in payload.files}
    groups = data["log_names"].astype(str)
    count = len(groups)
    variants = [str(value) for value in data["variant_names"]]
    device = torch.device(args.device)
    oof = {
        "component_ratio": np.zeros_like(data["component_ratio"], dtype=np.float32),
        "safety_logit": np.zeros_like(data["safety_progress_valid"], dtype=np.float32),
        "progress_ratio": np.zeros_like(data["progress_ratio"], dtype=np.float32),
        "refinement_gain": np.zeros_like(data["progress_ratio"], dtype=np.float32),
    }
    if "hard_metric_valid" in data:
        oof["hard_metric_logits"] = np.zeros_like(
            data["hard_metric_valid"], dtype=np.float32
        )
    if "quality_metric_margin" in data:
        oof["quality_metric_margin"] = np.zeros_like(
            data["quality_metric_margin"], dtype=np.float32
        )
    fold_ids = np.full(count, -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    for fold, (train, valid) in enumerate(splitter.split(np.arange(count), groups=groups)):
        print(f"fold={fold} train={len(train)} valid={len(valid)}", flush=True)
        model = _fit(data, train, epochs=args.epochs, seed=args.seed + fold, device=device)
        prediction = _predict(model, data, valid, device)
        for name in oof:
            oof[name][valid] = prediction[name]
        fold_ids[valid] = fold

    policies = []
    hard_metric_thresholds = (
        (0.5, 0.7, 0.85, 0.95, 0.98)
        if "hard_metric_valid" in data
        else (0.0,)
    )
    quality_margin_thresholds = (
        (-0.05, 0.0, 0.02, 0.05)
        if "quality_metric_margin" in data
        else (-1000.0,)
    )
    for margin in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50):
        for safety_threshold in (0.5, 0.7, 0.85, 0.95, 0.98):
            for progress_threshold in (0.97, 0.98, 0.99, 1.0):
                for component_margin in (0.0, 0.01, 0.02, 0.05, 0.10):
                    for hard_metric_threshold in hard_metric_thresholds:
                        for quality_margin_threshold in quality_margin_thresholds:
                            policies.append(
                                _evaluate(
                                    data,
                                    oof,
                                    fold_ids,
                                    Policy(
                                        margin,
                                        safety_threshold,
                                        progress_threshold,
                                        component_margin,
                                        hard_metric_threshold,
                                        quality_margin_threshold,
                                    ),
                                )
                            )
    acceptable = [
        value
        for value in policies
        if value["changed"] > 0
        and value["pair_utility_gain"] > 0.0
        and value["ec_gain"] > 0.0
        and value["safety_progress_valid_rate"] >= 0.999
        and value["minimum_fold_gain"] >= -1e-5
    ]
    if not acceptable:
        acceptable = [
            value
            for value in policies
            if value["changed"] > 0
            and value["pair_utility_gain"] > 0.0
            and value["ec_gain"] > 0.0
            and value["safety_progress_valid_rate"] >= 0.997
            and value["minimum_fold_gain"] >= -5e-4
        ]
    best = max(
        acceptable,
        key=lambda value: (
            value["pair_utility_gain"],
            value["ec_gain"],
            value["safety_progress_valid_rate"],
        ),
        default=None,
    )

    full = np.arange(count)
    model = _fit(data, full, epochs=args.epochs, seed=args.seed + 99, device=device)
    artifact_format = (
        "actworld_v2_navtrain_temporal_fcts_head_v3"
        if "quality_metric_margin" in data
        else (
            "actworld_v2_navtrain_temporal_fcts_head_v2"
            if "hard_metric_valid" in data
            else "actworld_v2_navtrain_temporal_fcts_head_v1"
        )
    )
    artifact = {
        "format": artifact_format,
        "split": "navtrain",
        "navhard_used_for_training_or_selection": False,
        "variant_names": variants,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "policy": None if best is None else best["policy"],
        "target_fraction": None if best is None else best["changed_fraction"],
        "seed": args.seed,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.artifact)
    np.savez_compressed(args.oof, fold_ids=fold_ids, **oof)
    report = {
        "format": artifact["format"],
        "split": "navtrain",
        "navhard_used": False,
        "pairs": count,
        "best_policy": None if best is None else {key: value for key, value in best.items() if key != "selected"},
        "acceptable_policies": len(acceptable),
        "baseline_pair_utility": float(data["pair_utility"][:, 0].mean()),
        "baseline_ec_pass_rate": float(data["ec_pass"][:, 0].mean()),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
