"""Lazy prediction ensemble for frozen NAVTRAIN post-r94 gain models."""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np


class LazyArtifactGainRegressor:
    """Load one immutable NAVTRAIN model lazily from its artifact."""

    def __init__(self, artifact_path):
        self.artifact_path = str(Path(artifact_path))
        self._model = None

    def _load(self):
        if self._model is None:
            with open(self.artifact_path, "rb") as stream:
                payload = pickle.load(stream)
            if payload.get("navtest_used", True):
                raise ValueError("scorer dependency is not NAVTRAIN-only")
            self._model = payload["model"]
        return self._model

    def predict(self, values):
        return self._load().predict(values)


class LazyBlendedGainRegressor:
    """Blend predictions from immutable scorer artifacts without duplicating them."""

    def __init__(self, artifact_paths, weights):
        self.artifact_paths = tuple(str(Path(path)) for path in artifact_paths)
        values = np.asarray(weights, dtype=np.float64)
        if len(values) != len(self.artifact_paths) or np.any(values < 0):
            raise ValueError("invalid blend weights")
        if not np.isfinite(values).all() or values.sum() <= 0:
            raise ValueError("blend weights must be finite and positive")
        self.weights = tuple((values / values.sum()).tolist())
        self._models = None

    def _load(self):
        if self._models is None:
            models = []
            for path in self.artifact_paths:
                with open(path, "rb") as stream:
                    payload = pickle.load(stream)
                if payload.get("navtest_used", True):
                    raise ValueError("blended scorer dependency is not NAVTRAIN-only")
                models.append(payload["model"])
            self._models = tuple(models)
        return self._models

    def predict(self, values):
        outputs = [
            weight * np.asarray(model.predict(values), dtype=np.float64)
            for weight, model in zip(self.weights, self._load())
        ]
        return np.sum(outputs, axis=0)


class LazyConsensusGainRegressor(LazyBlendedGainRegressor):
    """Use the lower scorer prediction as a NAVTRAIN-only agreement gate."""

    def __init__(self, artifact_paths):
        super().__init__(artifact_paths, np.ones(len(artifact_paths)))

    def predict(self, values):
        outputs = np.stack(
            [
                np.asarray(model.predict(values), dtype=np.float64)
                for model in self._load()
            ],
            axis=0,
        )
        return np.min(outputs, axis=0)


class LazyAffineGainRegressor(LazyBlendedGainRegressor):
    """Apply a fixed affine combination, including NAVTRAIN-model contrast."""

    def __init__(self, artifact_paths, weights):
        self.artifact_paths = tuple(str(Path(path)) for path in artifact_paths)
        values = np.asarray(weights, dtype=np.float64)
        if len(values) != len(self.artifact_paths):
            raise ValueError("invalid affine weights")
        if not np.isfinite(values).all() or not np.isclose(values.sum(), 1.0):
            raise ValueError("affine weights must be finite and sum to one")
        self.weights = tuple(values.tolist())
        self._models = None


class LazyHingeContrastGainRegressor(LazyBlendedGainRegressor):
    """Apply asymmetric correction only on positive/negative model disagreement."""

    def __init__(self, artifact_paths, boost_weight, penalty_weight):
        if len(artifact_paths) != 2:
            raise ValueError("hinge contrast requires exactly two artifacts")
        self.artifact_paths = tuple(str(Path(path)) for path in artifact_paths)
        self.boost_weight = float(boost_weight)
        self.penalty_weight = float(penalty_weight)
        if (
            not np.isfinite(self.boost_weight)
            or not np.isfinite(self.penalty_weight)
            or self.boost_weight < 0
            or self.penalty_weight < 0
        ):
            raise ValueError("hinge weights must be finite and non-negative")
        self.weights = (1.0, 0.0)
        self._models = None

    def predict(self, values):
        primary, auxiliary = (
            np.asarray(model.predict(values), dtype=np.float64)
            for model in self._load()
        )
        disagreement = primary - auxiliary
        return (
            primary
            + self.boost_weight * np.maximum(disagreement, 0.0)
            + self.penalty_weight * np.minimum(disagreement, 0.0)
        )
