"""Pickle-safe setwise candidate ranker used by the frozen NAVTRAIN scorer."""

from __future__ import annotations

import weakref

import numpy as np
import torch
import torch.nn as nn


class FutureConsistentSetRanker(nn.Module):
    """Rank all four planning futures jointly instead of independently.

    The input is the exact frozen r62 feature tensor.  Candidate-local JEPA
    evidence is fused with mean/max set context and a learned baseline-rank
    embedding.  This leaves proposals and the released checkpoint untouched.
    """

    def __init__(self, feature_width: int = 416, hidden: int = 128, dropout: float = 0.06) -> None:
        super().__init__()
        self.feature_width = int(feature_width)
        self.local = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.rank_embedding = nn.Parameter(torch.zeros(4, hidden))
        nn.init.normal_(self.rank_embedding, std=0.02)
        self.interaction = nn.Sequential(
            nn.LayerNorm(5 * hidden),
            nn.Linear(5 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
        )
        self.positive = nn.Linear(hidden, 1)
        self.gain = nn.Linear(hidden, 1)
        self.safety = nn.Linear(hidden, 4)

    def forward(self, features: torch.Tensor):
        if features.ndim != 3 or features.shape[-1] != self.feature_width:
            raise ValueError("features must have shape [batch, candidates, feature_width]")
        if features.shape[1] != 4:
            raise ValueError("the frozen candidate scorer requires exactly four candidates")
        local = self.local(features) + self.rank_embedding[None].to(features.dtype)
        mean = local.mean(dim=1, keepdim=True).expand_as(local)
        maximum = local.amax(dim=1, keepdim=True).expand_as(local)
        hidden = self.interaction(
            torch.cat((local, mean, maximum, local - mean, local * mean), dim=-1)
        )
        return (
            self.positive(hidden).squeeze(-1),
            self.gain(hidden).squeeze(-1),
            self.safety(hidden),
        )


class SharedSetwisePredictor:
    """Expose one set model through the sklearn-like scorer interface."""

    def __init__(self, network, mean, scale, candidates: int = 4, gain_scale: float = 10.0) -> None:
        self.network = network.cpu().eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.candidates = int(candidates)
        self.gain_scale = float(gain_scale)
        self._last_ref = None
        self._last_outputs = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_last_ref"] = None
        state["_last_outputs"] = None
        return state

    def predict_all(self, values):
        array = np.asarray(values, dtype=np.float32)
        if self._last_ref is not None and self._last_ref() is array:
            return self._last_outputs
        if array.ndim != 2 or len(array) % self.candidates:
            raise ValueError("flattened candidate features have incompatible shape")
        shaped = array.reshape(-1, self.candidates, array.shape[-1])
        normalized = (shaped - self.mean) / self.scale
        with torch.no_grad():
            positive, gain, safety = self.network(torch.from_numpy(normalized))
        outputs = (
            positive.sigmoid().numpy().reshape(-1),
            (gain / self.gain_scale).numpy().reshape(-1),
            safety.sigmoid().numpy().reshape(-1, 4),
        )
        self._last_ref = weakref.ref(array)
        self._last_outputs = outputs
        return outputs


class SetwiseEstimator:
    def __init__(self, predictor, head: str, channel: int = -1) -> None:
        self.predictor = predictor
        self.head = head
        self.channel = int(channel)

    def predict(self, values):
        if self.head != "gain":
            raise AttributeError("only the gain adapter supports predict")
        return self.predictor.predict_all(values)[1]

    def predict_proba(self, values):
        positive, _, safety = self.predictor.predict_all(values)
        probability = positive if self.head == "positive" else safety[:, self.channel]
        return np.stack((1.0 - probability, probability), axis=1)


def setwise_estimator_ensemble(networks, means, scales):
    result = {"positive": [], "gain": [], "safety": [[] for _ in range(4)]}
    for network, mean, scale in zip(networks, means, scales):
        predictor = SharedSetwisePredictor(network, mean, scale)
        result["positive"].append(SetwiseEstimator(predictor, "positive"))
        result["gain"].append(SetwiseEstimator(predictor, "gain"))
        for channel in range(4):
            result["safety"][channel].append(SetwiseEstimator(predictor, "safety", channel))
    return result


class CascadeSetwisePredictor:
    """Keep r62 as fallback and allow only tree-confirmed setwise overrides."""

    def __init__(self, tree_models, networks, means, scales, policy) -> None:
        self.tree_models = tree_models
        self.set_predictors = [
            SharedSetwisePredictor(network, mean, scale)
            for network, mean, scale in zip(networks, means, scales)
        ]
        self.policy = dict(policy)
        self._last_ref = None
        self._last_outputs = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_last_ref"] = None
        state["_last_outputs"] = None
        return state

    @staticmethod
    def _tree_predictions(models, flat):
        probability = np.mean(
            [model.predict_proba(flat)[:, 1] for model in models["positive"]], axis=0
        )
        gain = np.mean([model.predict(flat) for model in models["gain"]], axis=0)
        safety = np.stack([
            np.mean([model.predict_proba(flat)[:, 1] for model in channel], axis=0)
            for channel in models["safety"]
        ], axis=-1)
        return probability, gain, safety

    def predict_all(self, values):
        flat = np.asarray(values, dtype=np.float32)
        if self._last_ref is not None and self._last_ref() is flat:
            return self._last_outputs
        if flat.ndim != 2 or len(flat) % 4:
            raise ValueError("cascade scorer expects flattened top-4 features")
        batch = len(flat) // 4
        shaped = flat.reshape(batch, 4, flat.shape[-1])
        tree_probability, tree_gain, tree_safety = self._tree_predictions(
            self.tree_models, flat
        )
        tree_probability = tree_probability.reshape(batch, 4)
        tree_gain = tree_gain.reshape(batch, 4)
        tree_safety = tree_safety.reshape(batch, 4, 4)
        set_outputs = [predictor.predict_all(flat) for predictor in self.set_predictors]
        set_probability = np.mean(
            [item[0].reshape(batch, 4) for item in set_outputs], axis=0
        )
        set_gain = np.mean(
            [item[1].reshape(batch, 4) for item in set_outputs], axis=0
        )
        set_safety = np.mean(
            [item[2].reshape(batch, 4, 4) for item in set_outputs], axis=0
        )

        rows = np.arange(batch)
        mother = np.abs(shaped[..., 96:144]).sum(axis=-1).argmin(axis=1)
        tree_position = tree_gain.argmax(axis=1)
        base_ok = (
            (tree_probability[rows, tree_position] >= 0.90)
            & (tree_gain[rows, tree_position] >= 0.03)
            & (tree_safety[rows, tree_position].min(axis=1) >= 0.90)
        )
        baseline_gap = shaped[:, :1, 192] - shaped[:, :, 192]
        critical_delta = shaped[:, :, 193] - shaped[:, :1, 193]
        progress_delta = shaped[:, :, 195] - shaped[:, :1, 195]
        predicted_safety_delta = shaped[:, :, 201] - shaped[:, :1, 201]
        base_ok &= (
            (baseline_gap[rows, tree_position] <= 0.005)
            & (critical_delta[rows, tree_position] >= -0.02)
            & (progress_delta[rows, tree_position] >= 0.0)
            & (predicted_safety_delta[rows, tree_position] >= -0.05)
        )
        base = np.where(base_ok, tree_position, mother)

        utility = set_gain + float(self.policy["probability_weight"]) * set_probability
        candidate = utility.argmax(axis=1)
        accepted = (
            (candidate != base)
            & (set_probability[rows, candidate] >= self.policy["min_probability"])
            & (set_gain[rows, candidate] >= self.policy["min_gain"])
            & (set_safety[rows, candidate].min(axis=1) >= self.policy["min_safety_probability"])
            & (tree_probability[rows, candidate] >= self.policy["tree_min_probability"])
            & (tree_gain[rows, candidate] >= self.policy["tree_min_gain"])
            & (tree_safety[rows, candidate].min(axis=1) >= self.policy["tree_min_safety_probability"])
            & (
                set_gain[rows, candidate] - set_gain[rows, base]
                >= self.policy["min_set_advantage"]
            )
            & (
                tree_gain[rows, candidate] - tree_gain[rows, base]
                >= self.policy["min_tree_advantage"]
            )
            & (baseline_gap[rows, candidate] <= self.policy["max_baseline_gap"])
            & (critical_delta[rows, candidate] >= self.policy["min_critical_delta"])
            & (progress_delta[rows, candidate] >= 0.0)
            & (
                predicted_safety_delta[rows, candidate]
                >= self.policy["min_predicted_safety_delta"]
            )
        )
        if self.policy.get("require_tree_winner", False):
            accepted &= candidate == tree_position
        selected = np.where(accepted, candidate, base)

        # The existing runtime guard remains the final authority.  Synthetic
        # heads merely communicate the already-frozen cascade choice through
        # its sklearn-compatible interface.
        gain = np.zeros((batch, 4), dtype=np.float32)
        probability = np.zeros((batch, 4), dtype=np.float32)
        safety = np.zeros((batch, 4, 4), dtype=np.float32)
        gain[rows, selected] = 1.0
        probability[rows, selected] = 1.0
        safety[rows, selected] = 1.0
        outputs = (probability.reshape(-1), gain.reshape(-1), safety.reshape(-1, 4))
        self._last_ref = weakref.ref(flat)
        self._last_outputs = outputs
        return outputs


def cascade_estimator_bundle(tree_models, networks, means, scales, policy):
    predictor = CascadeSetwisePredictor(tree_models, networks, means, scales, policy)
    result = {"positive": [SetwiseEstimator(predictor, "positive")],
              "gain": [SetwiseEstimator(predictor, "gain")],
              "safety": [[] for _ in range(4)]}
    for channel in range(4):
        result["safety"][channel].append(SetwiseEstimator(predictor, "safety", channel))
    return result
