"""NAVSIM-backed builder foundation for the immutable F2 token artifact.

This module closes the storage/runtime gap left intentionally open by
``f2_frozen_artifact_io``.  It authenticates and cross-checks the held-out
qualification and causal manifests, runs the original Drive-JEPA proposal
head in float32 with autocast disabled, reproduces the training-time
``simulation_v2`` + training ``PDMScorer`` target path, computes a separate
standard-simulator carry, and streams records into the fail-closed artifact
writer.

The real feature/model construction remains an outer-adapter responsibility.
In particular, this module does not guess how a Drive-JEPA checkpoint maps to
an instantiated model or how a NAVSIM ``SceneLoader`` becomes model features.
Those objects are injected and their authenticated identities are bound by the
caller.  The default NAVSIM scoring adapter is imported lazily so synthetic
tests can audit the complete control flow without a NAVSIM installation.

No token is skippable.  A failure is wrapped with its exact
``(log_name, time_us, token)`` identity and aborts sealing; the binary writer
then leaves only its diagnostic ``.partial`` file and never publishes the
target path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from navsim.planning.training.f2_qualification_manifest import (
    QualificationManifest,
    canonical_manifest_bytes as canonical_qualification_manifest_bytes,
    load_qualification_manifest,
)
from navsim.planning.validation.f2_causal_model_adapter import (
    ORIGINAL_PROPOSAL_HASH_SCHEMA,
    canonical_original_proposal_sha256,
)
from navsim.planning.validation.f2_causal_validator import (
    F2_MODEL_NUMERIC_MODE,
    CanonicalManifest,
    ManifestRow,
    canonical_manifest_bytes as canonical_causal_manifest_bytes,
    manifest_from_bytes as causal_manifest_from_bytes,
    rows_by_log,
)
from navsim.planning.validation.f2_frozen_artifact_io import (
    CANDIDATE_COUNT,
    RECORD_SIZE,
    STATE_DIMENSION,
    STATE_TIME_STEPS,
    TARGET_CHANNEL_ORDER,
    ArtifactBindings,
    ArtifactDescriptor,
    CoverageEntry,
    FrozenArtifactChunk,
    FrozenTokenRecord,
    build_artifact_coverage,
    seal_frozen_artifact,
)
from navsim.planning.validation.f2_live_provenance import (
    F2LiveProvenanceError,
    VerifiedF2Provenance,
    verify_live_f2_provenance,
)
from navsim.planning.validation.f2_metric_cache_manifest import (
    VerifiedMetricCacheManifest,
)


DRIVE_JEPA_PROPOSAL_SHAPE = (CANDIDATE_COUNT, 8, 3)
F2_PRUNED_CACHE_FAMILY = "train_metric_cache_v2_pruned"
F2_DATASET = "navtrain"
F2_HELD_OUT_PARTITIONS = ("held_out", "validation")
CALIBRATION_SIMULATOR_NAME = "navsim-train-simulation-v2"
CARRY_SIMULATOR_NAME = "navsim-standard-simulation"
TRAIN_SCORER_NAME = "drive-jepa-train-pdm-scorer-nine-channel"
_DRY_RUN_BASELINE_SEAL = object()


class F2FrozenRecordBuildError(RuntimeError):
    """Raised whenever a frozen NAVSIM record cannot be built exactly."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validated_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise F2FrozenRecordBuildError(f"{name} must be a 64-character SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise F2FrozenRecordBuildError(
            f"{name} must use lowercase hexadecimal"
        )
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise F2FrozenRecordBuildError(f"{name} must be a non-empty string")
    return value


def _nonnegative_time(value: Any, name: str = "time_us") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise F2FrozenRecordBuildError(f"{name} must be a nonnegative integer")
    return value


def _unique_tokens(values: Iterable[Any], name: str) -> Tuple[str, ...]:
    tokens = tuple(_nonempty_string(value, name) for value in values)
    duplicates = sorted(
        token for token, count in Counter(tokens).items() if count > 1
    )
    if duplicates:
        raise F2FrozenRecordBuildError(
            f"{name} contains duplicate tokens: {duplicates[:8]}"
        )
    return tokens


def _manifest_row_identity(row: Any) -> Tuple[str, int, str, bool]:
    try:
        identity = (
            row.log_name,
            row.time_us,
            row.token,
            row.future_camera_valid,
        )
    except AttributeError as error:
        raise F2FrozenRecordBuildError("manifest row is missing an identity field") from error
    if not isinstance(identity[3], bool):
        raise F2FrozenRecordBuildError("future_camera_valid must be a bool")
    return (
        _nonempty_string(identity[0], "row.log_name"),
        _nonnegative_time(identity[1], "row.time_us"),
        _nonempty_string(identity[2], "row.token"),
        identity[3],
    )


@dataclass(frozen=True)
class AuthenticatedHeldOutManifests:
    """Authenticated qualification + causal views of the same held-out rows."""

    qualification: QualificationManifest
    qualification_sha256: str
    causal: CanonicalManifest
    causal_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.qualification, QualificationManifest):
            raise F2FrozenRecordBuildError(
                "qualification must be an authenticated QualificationManifest"
            )
        if not isinstance(self.causal, CanonicalManifest):
            raise F2FrozenRecordBuildError(
                "causal must be an authenticated CanonicalManifest"
            )
        qualification_sha256 = _validated_sha256(
            self.qualification_sha256, "qualification_sha256"
        )
        causal_sha256 = _validated_sha256(self.causal_sha256, "causal_sha256")
        if _sha256_bytes(
            canonical_qualification_manifest_bytes(self.qualification)
        ) != qualification_sha256:
            raise F2FrozenRecordBuildError(
                "authenticated qualification manifest digest drifted in memory"
            )
        if _sha256_bytes(canonical_causal_manifest_bytes(self.causal)) != causal_sha256:
            raise F2FrozenRecordBuildError(
                "authenticated causal manifest digest drifted in memory"
            )
        object.__setattr__(self, "qualification_sha256", qualification_sha256)
        object.__setattr__(self, "causal_sha256", causal_sha256)

        qualified_rows = tuple(
            _manifest_row_identity(row) for row in self.qualification.held_out.rows
        )
        causal_rows = tuple(_manifest_row_identity(row) for row in self.causal.rows)
        if causal_rows != qualified_rows:
            raise F2FrozenRecordBuildError(
                "held-out qualification and causal manifest rows differ in "
                "canonical (log_name, time_us, token, future_camera_valid) coverage"
            )
        if self.qualification.sensor_contract != self.causal.sensor_contract:
            raise F2FrozenRecordBuildError(
                "held-out qualification and causal sensor contracts differ"
            )
        if self.qualification.cache_sha256 != self.causal.cache_metadata_sha256:
            raise F2FrozenRecordBuildError(
                "held-out qualification and causal cache digests differ"
            )

    def require_bindings(self, bindings: ArtifactBindings) -> None:
        """Bind both authenticated manifests to the artifact header identities."""

        if not isinstance(bindings, ArtifactBindings):
            raise F2FrozenRecordBuildError("bindings must be ArtifactBindings")
        mismatches = []
        comparisons = (
            (
                "qualification manifest",
                self.qualification_sha256,
                bindings.qualification_manifest_sha256,
            ),
            (
                "causal manifest",
                self.causal_sha256,
                bindings.causal_manifest_sha256,
            ),
            (
                "qualification config",
                self.qualification.config_sha256,
                bindings.qualification_config_sha256,
            ),
            (
                "qualification cache",
                self.qualification.cache_sha256,
                bindings.cache_metadata_sha256,
            ),
            (
                "qualification source",
                self.qualification.source_sha256,
                bindings.qualification_source_manifest_sha256,
            ),
            (
                "causal source revision",
                self.causal.source_revision,
                bindings.source_revision,
            ),
        )
        for name, actual, expected in comparisons:
            if actual != expected:
                mismatches.append(name)
        if mismatches:
            raise F2FrozenRecordBuildError(
                "artifact bindings differ from authenticated held-out inputs: "
                + ", ".join(mismatches)
            )


def load_authenticated_held_out_manifests(
    *,
    qualification_path: Path,
    expected_qualification_sha256: str,
    expected_config_sha256: str,
    expected_cache_sha256: str,
    expected_source_sha256: str,
    causal_path: Path,
    expected_causal_sha256: str,
) -> AuthenticatedHeldOutManifests:
    """Authenticate both files and require exact held-out row parity."""

    qualification = load_qualification_manifest(
        Path(qualification_path),
        expected_manifest_sha256=expected_qualification_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_cache_sha256=expected_cache_sha256,
        expected_source_sha256=expected_source_sha256,
    )
    causal_file = Path(causal_path)
    if not causal_file.is_file():
        raise FileNotFoundError(causal_file)
    try:
        causal_bytes = causal_file.read_bytes()
    except OSError as error:
        raise F2FrozenRecordBuildError(
            f"cannot read F2 causal manifest {causal_file}"
        ) from error
    causal = causal_manifest_from_bytes(
        causal_bytes,
        expected_sha256=expected_causal_sha256,
        expected_tokens=qualification.held_out.tokens,
    )
    return AuthenticatedHeldOutManifests(
        qualification=qualification,
        qualification_sha256=expected_qualification_sha256,
        causal=causal,
        causal_sha256=expected_causal_sha256,
    )


@dataclass(frozen=True)
class TokenFeatures:
    """One feature payload with an independently checkable row identity."""

    token: str
    log_name: str
    time_us: int
    features: Mapping[str, Any]

    def __post_init__(self) -> None:
        _nonempty_string(self.token, "feature token")
        _nonempty_string(self.log_name, "feature log_name")
        _nonnegative_time(self.time_us, "feature time_us")
        if not isinstance(self.features, Mapping):
            raise F2FrozenRecordBuildError("features must be a mapping")


@dataclass(frozen=True)
class PrunedMetricCacheItem:
    """One pruned-cache object with its authoritative NAVSIM identity."""

    token: str
    log_name: str
    time_us: int
    metric_cache: Any
    path: Optional[Path] = None

    def __post_init__(self) -> None:
        _nonempty_string(self.token, "metric-cache token")
        _nonempty_string(self.log_name, "metric-cache log_name")
        _nonnegative_time(self.time_us, "metric-cache time_us")
        if self.metric_cache is None:
            raise F2FrozenRecordBuildError("metric_cache cannot be None")
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class BaselineTokenOutput:
    """Original Drive-JEPA candidates and their raw final-channel logits."""

    token: str
    proposals: torch.Tensor
    base_logits: torch.Tensor
    numeric_mode: str = F2_MODEL_NUMERIC_MODE


def _prepare_features(value: Any, device: torch.device) -> Any:
    """Copy a nested feature tree to one device without mutating the source."""

    if isinstance(value, torch.Tensor):
        if torch.is_floating_point(value):
            return value.detach().to(device=device, dtype=torch.float32)
        return value.detach().to(device=device)
    if isinstance(value, Mapping):
        return {key: _prepare_features(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_prepare_features(item, device) for item in value)
    if isinstance(value, list):
        return [_prepare_features(item, device) for item in value]
    return value


def _require_device_float32_tensor(
    value: Any,
    *,
    name: str,
    shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise F2FrozenRecordBuildError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise F2FrozenRecordBuildError(f"{name} must be float32")
    if value.device != device:
        raise F2FrozenRecordBuildError(
            f"{name} must remain on {device}, got {value.device}"
        )
    if tuple(value.shape) != shape:
        raise F2FrozenRecordBuildError(
            f"{name} shape must be exactly {shape}, got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise F2FrozenRecordBuildError(f"{name} contains NaN or infinity")
    return value


class DriveJepaBaselineForward:
    """Strict float32/no-autocast forward seam for an authenticated baseline."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        authenticated_baseline_sha256: str,
        device: str | torch.device = "cpu",
        forward_fn: Optional[
            Callable[[torch.nn.Module, Mapping[str, Any]], Mapping[str, Any]]
        ] = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise F2FrozenRecordBuildError("baseline model must be a torch.nn.Module")
        resolved_device = torch.device(device)
        if resolved_device.type not in {"cpu", "cuda"}:
            raise F2FrozenRecordBuildError(
                "authoritative baseline inference supports only CPU or CUDA"
            )
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise F2FrozenRecordBuildError("requested CUDA device is unavailable")
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        self._baseline_sha256 = _validated_sha256(
            authenticated_baseline_sha256, "authenticated_baseline_sha256"
        )
        self._device = resolved_device
        self._model = model.to(device=resolved_device, dtype=torch.float32)
        self._model.eval()
        for name, tensor in tuple(self._model.named_parameters()) + tuple(
            self._model.named_buffers()
        ):
            if torch.is_floating_point(tensor) and tensor.dtype != torch.float32:
                raise F2FrozenRecordBuildError(
                    f"baseline floating tensor {name!r} is not float32"
                )
        self._forward_fn = forward_fn or (
            lambda restored_model, features: restored_model(features)
        )
        if not callable(self._forward_fn):
            raise F2FrozenRecordBuildError("forward_fn must be callable")

    @property
    def authenticated_baseline_sha256(self) -> str:
        return self._baseline_sha256

    @property
    def model(self) -> torch.nn.Module:
        """Expose the injected model for read-only audit/introspection."""

        return self._model

    def infer(self, token: str, features: Mapping[str, Any]) -> BaselineTokenOutput:
        token = _nonempty_string(token, "token")
        if not isinstance(features, Mapping):
            raise F2FrozenRecordBuildError("baseline features must be a mapping")
        self._model.eval()
        prepared = _prepare_features(features, self._device)
        with torch.inference_mode():
            with torch.autocast(device_type=self._device.type, enabled=False):
                output = self._forward_fn(self._model, prepared)
        if not isinstance(output, Mapping):
            raise F2FrozenRecordBuildError(
                "Drive-JEPA baseline forward must return a mapping"
            )
        missing = [name for name in ("proposals", "pred_logit") if name not in output]
        if missing:
            raise F2FrozenRecordBuildError(
                f"Drive-JEPA baseline output is missing required values: {missing}"
            )
        proposals = _require_device_float32_tensor(
            output["proposals"],
            name="Drive-JEPA proposals",
            shape=(1,) + DRIVE_JEPA_PROPOSAL_SHAPE,
            device=self._device,
        )
        pred_logit = output["pred_logit"]
        if not isinstance(pred_logit, torch.Tensor):
            raise F2FrozenRecordBuildError("Drive-JEPA pred_logit must be a tensor")
        if (
            pred_logit.ndim != 3
            or tuple(pred_logit.shape[:2]) != (1, CANDIDATE_COUNT)
            or pred_logit.shape[-1] < 1
        ):
            raise F2FrozenRecordBuildError(
                "Drive-JEPA pred_logit must have shape [1, 32, channels>=1]"
            )
        if pred_logit.dtype != torch.float32 or pred_logit.device != self._device:
            raise F2FrozenRecordBuildError(
                "Drive-JEPA pred_logit must be float32 on the inference device"
            )
        if not bool(torch.isfinite(pred_logit).all().item()):
            raise F2FrozenRecordBuildError(
                "Drive-JEPA pred_logit contains NaN or infinity"
            )
        # This exact slice is the frozen Drive-JEPA baseline definition.  No
        # sigmoid, calibration, sorting, or candidate reindexing is allowed.
        base_logits = pred_logit[..., -1]
        return BaselineTokenOutput(
            token=token,
            proposals=proposals[0].detach().clone(),
            base_logits=base_logits[0].detach().clone(),
        )


@dataclass(frozen=True)
class NavsimTokenSimulation:
    """Nine-channel targets plus separate calibration and causal state streams."""

    targets: np.ndarray
    calibration_states: np.ndarray
    carry_states: np.ndarray


def _exact_array(
    value: Any,
    *,
    name: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise F2FrozenRecordBuildError(f"{name} must be a numpy.ndarray")
    expected = np.dtype(dtype)
    if value.dtype != expected or value.dtype.str != expected.str:
        raise F2FrozenRecordBuildError(
            f"{name} dtype must be exactly {expected.str}, got {value.dtype.str}"
        )
    if value.shape != shape:
        raise F2FrozenRecordBuildError(
            f"{name} shape must be exactly {shape}, got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise F2FrozenRecordBuildError(f"{name} must be C-contiguous")
    if not np.isfinite(value).all():
        raise F2FrozenRecordBuildError(f"{name} contains NaN or infinity")
    return value


class DriveJepaNavsimRecordSemantics:
    """Exact training scorer and dual-simulator semantics for one token.

    The first desired state is the cached PDM trajectory.  It participates in
    training ``PDMScorer`` progress normalization but is removed before the
    32 candidate targets/calibration states are returned.  Standard-simulator
    carry never includes this reference trajectory.
    """

    def __init__(
        self,
        *,
        proposal_sampling: Any,
        trajectory_factory: Callable[[np.ndarray], Any],
        reference_transform: Callable[[Any, Any, Any], np.ndarray],
        candidate_transform: Callable[[Any, Any, Any], np.ndarray],
        calibration_simulator: Any,
        carry_simulator: Any,
        scorer: Any,
        multi_metric_indices: Sequence[int],
        weighted_metric_indices: Sequence[int],
    ) -> None:
        if len(tuple(multi_metric_indices)) != 4:
            raise F2FrozenRecordBuildError("four multiplicative indices are required")
        if len(tuple(weighted_metric_indices)) != 4:
            raise F2FrozenRecordBuildError("four weighted indices are required")
        self._sampling = proposal_sampling
        self._trajectory_factory = trajectory_factory
        self._reference_transform = reference_transform
        self._candidate_transform = candidate_transform
        self._calibration_simulator = calibration_simulator
        self._carry_simulator = carry_simulator
        self._scorer = scorer
        self._multi_indices = tuple(int(index) for index in multi_metric_indices)
        self._weighted_indices = tuple(int(index) for index in weighted_metric_indices)

    @classmethod
    def official(cls) -> "DriveJepaNavsimRecordSemantics":
        """Construct the exact Drive-JEPA navtrain scoring stack lazily."""

        from nuplan.planning.simulation.trajectory.trajectory_sampling import (
            TrajectorySampling,
        )

        from navsim.agents.drive_jepa_perception_based.score_module.train_pdm_scorer import (
            PDMScorer,
            PDMScorerConfig,
        )
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import (
            fast_transform_trajectory,
            get_trajectory_as_array,
        )
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
            MultiMetricIndex,
            WeightedMetricIndex,
        )
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
            PDMSimulator as StandardPDMSimulator,
        )
        from navsim.planning.simulation.planner.pdm_planner.simulation_v2.pdm_simulator import (
            PDMSimulator as V2PDMSimulator,
        )

        sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        return cls(
            proposal_sampling=sampling,
            trajectory_factory=Trajectory,
            reference_transform=get_trajectory_as_array,
            candidate_transform=fast_transform_trajectory,
            calibration_simulator=V2PDMSimulator(sampling),
            carry_simulator=StandardPDMSimulator(sampling),
            scorer=PDMScorer(sampling, PDMScorerConfig()),
            multi_metric_indices=(
                MultiMetricIndex.NO_COLLISION,
                MultiMetricIndex.DRIVABLE_AREA,
                MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE,
                MultiMetricIndex.DRIVING_DIRECTION,
            ),
            weighted_metric_indices=(
                WeightedMetricIndex.PROGRESS,
                WeightedMetricIndex.TTC,
                WeightedMetricIndex.LANE_KEEPING,
                WeightedMetricIndex.HISTORY_COMFORT,
            ),
        )

    def run(self, metric_cache: Any, proposals: np.ndarray) -> NavsimTokenSimulation:
        proposals = _exact_array(
            proposals,
            name="original Drive-JEPA proposals",
            shape=DRIVE_JEPA_PROPOSAL_SHAPE,
            dtype=np.dtype("<f4"),
        )
        required_cache_fields = (
            "ego_state",
            "trajectory",
            "observation",
            "centerline",
            "route_lane_ids",
            "drivable_area_map",
            "map_parameters",
            "past_human_trajectory",
        )
        missing = [name for name in required_cache_fields if not hasattr(metric_cache, name)]
        if missing:
            raise F2FrozenRecordBuildError(
                f"pruned metric cache is missing required fields: {missing}"
            )
        initial_ego_state = metric_cache.ego_state
        reference = self._reference_transform(
            metric_cache.trajectory,
            self._sampling,
            initial_ego_state.time_point,
        )
        reference = np.asarray(reference)
        if reference.ndim != 2 or reference.shape != (STATE_TIME_STEPS, STATE_DIMENSION):
            # The official cache path may return an 11-D state directly.  The
            # Drive-JEPA scorer uses only SE(2) as the desired tracker input.
            if reference.ndim != 2 or reference.shape != (STATE_TIME_STEPS, 3):
                raise F2FrozenRecordBuildError(
                    "cached PDM trajectory must transform to [41, 3] or [41, 11]"
                )
        reference_se2 = reference[:, :3]

        candidate_desired = []
        for candidate_index in range(CANDIDATE_COUNT):
            trajectory = self._trajectory_factory(proposals[candidate_index])
            transformed = np.asarray(
                self._candidate_transform(
                    trajectory,
                    self._sampling,
                    initial_ego_state,
                )
            )
            if transformed.shape != (STATE_TIME_STEPS, 3):
                raise F2FrozenRecordBuildError(
                    "candidate transform must return [41, 3]; "
                    f"candidate={candidate_index}, shape={transformed.shape}"
                )
            if not np.isfinite(transformed).all():
                raise F2FrozenRecordBuildError(
                    f"candidate transform is non-finite at index {candidate_index}"
                )
            candidate_desired.append(transformed)
        desired_se2 = np.stack([reference_se2, *candidate_desired], axis=0)
        if not np.isfinite(desired_se2).all():
            raise F2FrozenRecordBuildError("desired simulator states are non-finite")
        # Match Drive-JEPA exactly: allocate the 11-D tracker input with the
        # transform's dtype and fill only x/y/heading.
        desired_extended = np.zeros(
            desired_se2.shape[:-1] + (STATE_DIMENSION,), dtype=desired_se2.dtype
        )
        desired_extended[..., :3] = desired_se2

        calibration_all = self._calibration_simulator.simulate_proposals(
            desired_extended, initial_ego_state
        )
        calibration_all = np.asarray(calibration_all)
        if calibration_all.shape != (
            CANDIDATE_COUNT + 1,
            STATE_TIME_STEPS,
            STATE_DIMENSION,
        ):
            raise F2FrozenRecordBuildError(
                "simulation_v2 must return [33, 41, 11] including the PDM reference"
            )
        if calibration_all.dtype != np.float64:
            raise F2FrozenRecordBuildError("simulation_v2 states must be float64")

        # Pruned navtrain semantics are intentionally explicit: future agent
        # detections were removed from the cache and are never reconstructed.
        aggregate = self._scorer.score_proposals(
            calibration_all,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            simulated_agent_detections_tracks=None,
            human_past_trajectory=metric_cache.past_human_trajectory,
        )
        aggregate = np.asarray(aggregate)
        if aggregate.shape != (CANDIDATE_COUNT + 1,):
            raise F2FrozenRecordBuildError(
                "training PDMScorer aggregate must contain 33 scores"
            )
        multi = np.asarray(getattr(self._scorer, "_multi_metrics", None))
        weighted = np.asarray(getattr(self._scorer, "_weighted_metrics", None))
        try:
            channels = [multi[index] for index in self._multi_indices]
            channels.extend(weighted[index] for index in self._weighted_indices)
        except (IndexError, TypeError) as error:
            raise F2FrozenRecordBuildError(
                "training PDMScorer metric arrays do not expose the frozen channels"
            ) from error
        channels.append(aggregate)
        targets_all = np.stack(channels, axis=-1)
        if targets_all.shape != (CANDIDATE_COUNT + 1, len(TARGET_CHANNEL_ORDER)):
            raise F2FrozenRecordBuildError(
                "training PDMScorer channels must have shape [33, 9]"
            )

        # The standard simulator receives only the 32 original candidates.
        carry = self._carry_simulator.simulate_proposals(
            desired_extended[1:], initial_ego_state
        )
        carry = np.asarray(carry)
        if carry.shape != (
            CANDIDATE_COUNT,
            STATE_TIME_STEPS,
            STATE_DIMENSION,
        ):
            raise F2FrozenRecordBuildError(
                "standard simulator must return [32, 41, 11]"
            )
        if carry.dtype != np.float64:
            raise F2FrozenRecordBuildError("standard-simulator carry must be float64")

        targets = np.ascontiguousarray(targets_all[1:], dtype=np.dtype("<f4"))
        calibration = np.ascontiguousarray(
            calibration_all[1:], dtype=np.dtype("<f8")
        )
        carry = np.ascontiguousarray(carry, dtype=np.dtype("<f8"))
        _exact_array(
            targets,
            name="target scores",
            shape=(CANDIDATE_COUNT, len(TARGET_CHANNEL_ORDER)),
            dtype=np.dtype("<f4"),
        )
        if np.any(targets < np.float32(0.0)) or np.any(targets > np.float32(1.0)):
            raise F2FrozenRecordBuildError("target scores must lie in [0, 1]")
        _exact_array(
            calibration,
            name="calibration states",
            shape=(CANDIDATE_COUNT, STATE_TIME_STEPS, STATE_DIMENSION),
            dtype=np.dtype("<f8"),
        )
        _exact_array(
            carry,
            name="carry states",
            shape=(CANDIDATE_COUNT, STATE_TIME_STEPS, STATE_DIMENSION),
            dtype=np.dtype("<f8"),
        )
        return NavsimTokenSimulation(
            targets=targets,
            calibration_states=calibration,
            carry_states=carry,
        )


def _metric_cache_time_us(metric_cache: Any) -> int:
    timepoint = getattr(metric_cache, "timepoint", None)
    value = getattr(timepoint, "time_us", None)
    return _nonnegative_time(value, "metric_cache.timepoint.time_us")


class PrunedMetricCacheSource:
    """Exact-token loader for the official pruned navtrain metric cache."""

    cache_family = F2_PRUNED_CACHE_FAMILY

    def __init__(
        self,
        root: Path,
        *,
        metadata_path: Path,
        expected_metadata_sha256: str,
        bindings: ArtifactBindings,
        content_manifest: VerifiedMetricCacheManifest,
        loader: Any = None,
    ) -> None:
        root = Path(root)
        if root.name != F2_PRUNED_CACHE_FAMILY:
            raise F2FrozenRecordBuildError(
                f"metric-cache root must be {F2_PRUNED_CACHE_FAMILY!r}"
            )
        if type(bindings) is not ArtifactBindings:
            raise F2FrozenRecordBuildError(
                "metric-cache source requires concrete artifact bindings"
            )
        expected_digest = _validated_sha256(
            expected_metadata_sha256, "expected_metadata_sha256"
        )
        if expected_digest != bindings.cache_metadata_sha256:
            raise F2FrozenRecordBuildError(
                "metric-cache metadata digest differs from artifact bindings"
            )
        if type(content_manifest) is not VerifiedMetricCacheManifest:
            raise F2FrozenRecordBuildError(
                "metric-cache source requires a live-verified content manifest"
            )
        if content_manifest.manifest_sha256 != bindings.metric_cache_manifest_sha256:
            raise F2FrozenRecordBuildError(
                "metric-cache content manifest differs from artifact bindings"
            )
        from navsim.common.dataloader import MetricCacheLoader

        authenticated_loader = MetricCacheLoader(
            root,
            expected_metadata_path=Path(metadata_path),
            expected_metadata_sha256=expected_digest,
            authenticated_content_manifest=content_manifest,
        )
        if loader is not None:
            raise F2FrozenRecordBuildError(
                "production metric-cache source forbids an injected loader"
            )
        loader = authenticated_loader
        paths = getattr(loader, "metric_cache_paths", None)
        if not isinstance(paths, Mapping):
            raise F2FrozenRecordBuildError(
                "MetricCacheLoader must expose metric_cache_paths"
            )
        self._root = root
        self.metadata_path = authenticated_loader.metadata_path
        self.metadata_sha256 = authenticated_loader.metadata_sha256
        self._loader = loader
        self._paths = dict(paths)
        self.tokens = tuple(sorted(_unique_tokens(self._paths, "metric-cache tokens")))

    def load(self, row: ManifestRow) -> PrunedMetricCacheItem:
        if row.token not in self._paths:
            raise F2FrozenRecordBuildError(
                f"pruned metric cache is missing token {row.token!r}"
            )
        path = Path(self._paths[row.token])
        try:
            path.resolve().relative_to(self._root.resolve())
        except ValueError as error:
            raise F2FrozenRecordBuildError(
                f"metric-cache token {row.token!r} resolves outside the pruned root"
            ) from error
        if not path.is_file():
            raise F2FrozenRecordBuildError(
                f"metric-cache file does not exist for token {row.token!r}: {path}"
            )
        metric_cache = self._loader.get_from_token(row.token)
        return PrunedMetricCacheItem(
            token=row.token,
            log_name=_nonempty_string(
                getattr(metric_cache, "log_name", None), "metric_cache.log_name"
            ),
            time_us=_metric_cache_time_us(metric_cache),
            metric_cache=metric_cache,
            path=path,
        )


@dataclass(frozen=True)
class FrozenRecordBuildPlan:
    destination: Path
    token_count: int
    log_count: int
    record_payload_bytes: int
    log_names: Tuple[str, ...]
    proposal_shape: Tuple[int, int, int] = DRIVE_JEPA_PROPOSAL_SHAPE
    target_shape: Tuple[int, int] = (CANDIDATE_COUNT, len(TARGET_CHANNEL_ORDER))
    state_shape: Tuple[int, int, int] = (
        CANDIDATE_COUNT,
        STATE_TIME_STEPS,
        STATE_DIMENSION,
    )

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "destination": str(self.destination),
            "log_count": self.log_count,
            "log_names": list(self.log_names),
            "proposal_hash_schema": ORIGINAL_PROPOSAL_HASH_SCHEMA,
            "proposal_shape": list(self.proposal_shape),
            "record_payload_bytes": self.record_payload_bytes,
            "state_shape": list(self.state_shape),
            "target_channel_order": list(TARGET_CHANNEL_ORDER),
            "target_shape": list(self.target_shape),
            "token_count": self.token_count,
        }


def _require_row_identity(
    *, row: ManifestRow, token: str, log_name: str, time_us: int, source: str
) -> None:
    observed = (log_name, time_us, token)
    expected = (row.log_name, row.time_us, row.token)
    if observed != expected:
        raise F2FrozenRecordBuildError(
            f"{source} row identity drift: expected={expected!r}, observed={observed!r}"
        )


class FrozenNavsimRecordBuilder:
    """Stream exact held-out NAVSIM records into ``f2_frozen_artifact_io``."""

    def __init__(
        self,
        *,
        manifests: AuthenticatedHeldOutManifests,
        bindings: ArtifactBindings,
        live_provenance: VerifiedF2Provenance,
        feature_source: Any,
        metric_cache_source: Any,
        restored_baseline: Any,
        navsim_semantics: Any,
        _dry_run_baseline_seal: object | None = None,
    ) -> None:
        manifests.require_bindings(bindings)
        if type(live_provenance) is not VerifiedF2Provenance:
            raise F2FrozenRecordBuildError(
                "record building requires byte-verified live F2 provenance"
            )
        try:
            live_provenance.require_bindings(bindings)
        except F2LiveProvenanceError as error:
            raise F2FrozenRecordBuildError(str(error)) from error
        if (
            live_provenance.qualification != manifests.qualification
            or live_provenance.causal != manifests.causal
        ):
            raise F2FrozenRecordBuildError(
                "record-builder manifests differ from live verified provenance"
            )
        self._manifests = manifests
        self._bindings = bindings
        self._live_provenance = live_provenance
        self._planning_only = _dry_run_baseline_seal is _DRY_RUN_BASELINE_SEAL
        if _dry_run_baseline_seal is not _DRY_RUN_BASELINE_SEAL:
            from navsim.planning.validation.f2_navsim_runtime_adapters import (
                QualifiedHeldOutNavtrainFeatureSource,
            )

            if type(feature_source) is not QualifiedHeldOutNavtrainFeatureSource:
                raise F2FrozenRecordBuildError(
                    "feature source must be the concrete qualified NAVSIM adapter"
                )
            if feature_source.live_provenance is not live_provenance:
                raise F2FrozenRecordBuildError(
                    "feature source and record builder do not share verified provenance"
                )
            if type(metric_cache_source) is not PrunedMetricCacheSource:
                raise F2FrozenRecordBuildError(
                    "metric cache must be the concrete pruned NAVSIM cache adapter"
                )
            if type(navsim_semantics) is not DriveJepaNavsimRecordSemantics:
                raise F2FrozenRecordBuildError(
                    "NAVSIM semantics must be the concrete dual-simulator/scorer adapter"
                )
        self._features = feature_source
        self._caches = metric_cache_source
        if _dry_run_baseline_seal is _DRY_RUN_BASELINE_SEAL:
            self._baseline = restored_baseline.forward
        else:
            # Local import avoids the intentional adapter -> record-builder
            # dependency becoming an import cycle at module initialization.
            from navsim.planning.validation.f2_navsim_runtime_adapters import (
                OFFICIAL_DRIVE_JEPA_FACTORY_REVISION,
                DriveJepaFactorySourceIdentity,
                RestoredDriveJepaBaseline,
            )

            if type(restored_baseline) is not RestoredDriveJepaBaseline:
                raise F2FrozenRecordBuildError(
                    "baseline must be the concrete strictly restored Drive-JEPA adapter"
                )
            if not restored_baseline.is_official_factory:
                raise F2FrozenRecordBuildError(
                    "baseline was not constructed by the sealed official factory"
                )
            factory_source = restored_baseline.factory_source
            if (
                type(factory_source) is not DriveJepaFactorySourceIdentity
                or factory_source.revision != OFFICIAL_DRIVE_JEPA_FACTORY_REVISION
            ):
                raise F2FrozenRecordBuildError(
                    "baseline lacks the authenticated Drive-JEPA factory source"
                )
            if restored_baseline.live_provenance is not live_provenance:
                raise F2FrozenRecordBuildError(
                    "baseline and record builder do not share verified provenance"
                )
            if (
                restored_baseline.checkpoint_sha256
                != bindings.baseline_checkpoint_sha256
            ):
                raise F2FrozenRecordBuildError(
                    "restored baseline checkpoint differs from artifact bindings"
                )
            self._baseline = restored_baseline.forward
        self._semantics = navsim_semantics

    def plan(self, destination: Path) -> FrozenRecordBuildPlan:
        destination = Path(destination)
        rows = self._manifests.causal.rows
        expected_tokens = tuple(row.token for row in rows)

        dataset = getattr(self._features, "dataset", None)
        partition = getattr(self._features, "partition", None)
        if dataset != F2_DATASET or partition not in F2_HELD_OUT_PARTITIONS:
            raise F2FrozenRecordBuildError(
                "feature source must be authenticated navtrain held_out/validation"
            )
        feature_tokens = _unique_tokens(
            getattr(self._features, "tokens", ()), "feature-source tokens"
        )
        if feature_tokens != expected_tokens:
            missing = sorted(set(expected_tokens) - set(feature_tokens))
            extra = sorted(set(feature_tokens) - set(expected_tokens))
            raise F2FrozenRecordBuildError(
                "feature source must exactly equal canonical manifest coverage: "
                f"missing={missing[:8]}, extra={extra[:8]}, order_drift="
                f"{not missing and not extra}"
            )

        if getattr(self._caches, "cache_family", None) != F2_PRUNED_CACHE_FAMILY:
            raise F2FrozenRecordBuildError(
                "metric-cache source is not train_metric_cache_v2_pruned"
            )
        cache_tokens = set(
            _unique_tokens(getattr(self._caches, "tokens", ()), "metric-cache tokens")
        )
        missing_cache = sorted(set(expected_tokens) - cache_tokens)
        if missing_cache:
            raise F2FrozenRecordBuildError(
                f"pruned metric cache does not cover the manifest: {missing_cache[:8]}"
            )

        baseline_sha256 = _validated_sha256(
            getattr(self._baseline, "authenticated_baseline_sha256", None),
            "baseline_forward.authenticated_baseline_sha256",
        )
        if baseline_sha256 != self._bindings.baseline_checkpoint_sha256:
            raise F2FrozenRecordBuildError(
                "baseline forward identity differs from artifact bindings"
            )
        for name in ("load",):
            if not callable(getattr(self._features, name, None)):
                raise F2FrozenRecordBuildError(f"feature source must implement {name}()")
            if not callable(getattr(self._caches, name, None)):
                raise F2FrozenRecordBuildError(
                    f"metric-cache source must implement {name}()"
                )
        if not callable(getattr(self._baseline, "infer", None)):
            raise F2FrozenRecordBuildError("baseline forward must implement infer()")
        if not callable(getattr(self._semantics, "run", None)):
            raise F2FrozenRecordBuildError("NAVSIM semantics must implement run()")

        if not destination.parent.is_dir():
            raise F2FrozenRecordBuildError(
                "artifact destination parent directory does not exist"
            )
        partial = destination.with_name(destination.name + ".partial")
        if destination.exists() or partial.exists():
            raise F2FrozenRecordBuildError(
                "artifact destination or diagnostic partial already exists"
            )
        log_names = tuple(sorted({row.log_name for row in rows}))
        return FrozenRecordBuildPlan(
            destination=destination,
            token_count=len(rows),
            log_count=len(log_names),
            record_payload_bytes=len(rows) * RECORD_SIZE,
            log_names=log_names,
        )

    def _build_record(self, row: ManifestRow) -> FrozenTokenRecord:
        identity = f"log={row.log_name!r}, time_us={row.time_us}, token={row.token!r}"
        try:
            feature_item = self._features.load(row)
            if not isinstance(feature_item, TokenFeatures):
                raise F2FrozenRecordBuildError(
                    "feature source must return TokenFeatures"
                )
            _require_row_identity(
                row=row,
                token=feature_item.token,
                log_name=feature_item.log_name,
                time_us=feature_item.time_us,
                source="feature",
            )

            cache_item = self._caches.load(row)
            if not isinstance(cache_item, PrunedMetricCacheItem):
                raise F2FrozenRecordBuildError(
                    "metric-cache source must return PrunedMetricCacheItem"
                )
            _require_row_identity(
                row=row,
                token=cache_item.token,
                log_name=cache_item.log_name,
                time_us=cache_item.time_us,
                source="metric cache",
            )

            baseline = self._baseline.infer(row.token, feature_item.features)
            if not isinstance(baseline, BaselineTokenOutput):
                raise F2FrozenRecordBuildError(
                    "baseline forward must return BaselineTokenOutput"
                )
            if baseline.token != row.token:
                raise F2FrozenRecordBuildError("baseline output token drifted")
            if baseline.numeric_mode != F2_MODEL_NUMERIC_MODE:
                raise F2FrozenRecordBuildError(
                    f"baseline numeric mode must be {F2_MODEL_NUMERIC_MODE!r}"
                )
            proposals = baseline.proposals
            base_logits = baseline.base_logits
            if not isinstance(proposals, torch.Tensor) or proposals.dtype != torch.float32:
                raise F2FrozenRecordBuildError(
                    "baseline proposals must be a float32 torch.Tensor"
                )
            if tuple(proposals.shape) != DRIVE_JEPA_PROPOSAL_SHAPE:
                raise F2FrozenRecordBuildError(
                    "baseline proposals must preserve original [32, 8, 3] ordering"
                )
            if not bool(torch.isfinite(proposals).all().item()):
                raise F2FrozenRecordBuildError("baseline proposals are non-finite")
            if not isinstance(base_logits, torch.Tensor) or base_logits.dtype != torch.float32:
                raise F2FrozenRecordBuildError(
                    "baseline pred_logit[..., -1] must be a float32 tensor"
                )
            if tuple(base_logits.shape) != (CANDIDATE_COUNT,):
                raise F2FrozenRecordBuildError(
                    "baseline pred_logit[..., -1] must have shape [32]"
                )
            if not bool(torch.isfinite(base_logits).all().item()):
                raise F2FrozenRecordBuildError("baseline logits are non-finite")

            # The shared F2 adapter owns this order-sensitive hash definition.
            proposal_sha256 = canonical_original_proposal_sha256(proposals)
            proposals_numpy = np.ascontiguousarray(
                proposals.detach().cpu().numpy(), dtype=np.dtype("<f4")
            )
            base_numpy = np.ascontiguousarray(
                base_logits.detach().cpu().numpy(), dtype=np.dtype("<f4")
            )
            simulation = self._semantics.run(
                cache_item.metric_cache, proposals_numpy
            )
            if not isinstance(simulation, NavsimTokenSimulation):
                raise F2FrozenRecordBuildError(
                    "NAVSIM semantics must return NavsimTokenSimulation"
                )
            _exact_array(
                base_numpy,
                name="base logits",
                shape=(CANDIDATE_COUNT,),
                dtype=np.dtype("<f4"),
            )
            _exact_array(
                simulation.targets,
                name="targets",
                shape=(CANDIDATE_COUNT, len(TARGET_CHANNEL_ORDER)),
                dtype=np.dtype("<f4"),
            )
            _exact_array(
                simulation.calibration_states,
                name="calibration states",
                shape=(CANDIDATE_COUNT, STATE_TIME_STEPS, STATE_DIMENSION),
                dtype=np.dtype("<f8"),
            )
            _exact_array(
                simulation.carry_states,
                name="carry states",
                shape=(CANDIDATE_COUNT, STATE_TIME_STEPS, STATE_DIMENSION),
                dtype=np.dtype("<f8"),
            )
            return FrozenTokenRecord(
                token=row.token,
                time_us=row.time_us,
                proposal_sha256=proposal_sha256,
                base=base_numpy,
                targets=simulation.targets,
                calibration=simulation.calibration_states,
                carry=simulation.carry_states,
            )
        except F2FrozenRecordBuildError as error:
            raise F2FrozenRecordBuildError(
                f"fatal frozen-record failure ({identity}): {error}"
            ) from error
        except Exception as error:
            raise F2FrozenRecordBuildError(
                f"fatal frozen-record failure ({identity}): "
                f"{type(error).__name__}: {error}"
            ) from error

    def build(self, destination: Path) -> ArtifactDescriptor:
        """Build every canonical row once and publish only after full verification."""

        if self._planning_only:
            raise F2FrozenRecordBuildError(
                "dry-run planning adapters are forbidden from sealing artifacts"
            )
        plan = self.plan(destination)
        coverage = build_artifact_coverage(
            CoverageEntry(row.log_name, row.time_us, row.token)
            for row in self._manifests.causal.rows
        )
        grouped = rows_by_log(self._manifests.causal.rows)
        chunks = (
            FrozenArtifactChunk(
                log_name=log_name,
                records=(self._build_record(row) for row in grouped[log_name]),
            )
            for log_name in plan.log_names
        )
        return seal_frozen_artifact(
            plan.destination,
            bindings=self._bindings,
            coverage=coverage,
            chunks=chunks,
        )


def _canonical_json_file(path: Path, *, expected_sha256: str, name: str) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    if _sha256_bytes(payload) != _validated_sha256(
        expected_sha256, f"expected {name} SHA-256"
    ):
        raise F2FrozenRecordBuildError(f"{name} SHA-256 mismatch")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F2FrozenRecordBuildError(f"{name} is not valid JSON") from error
    canonical = (
        json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if canonical != payload:
        raise F2FrozenRecordBuildError(f"{name} is not canonical JSON")
    return decoded


def _bindings_from_payload(payload: Any) -> ArtifactBindings:
    if not isinstance(payload, dict):
        raise F2FrozenRecordBuildError("bindings JSON must be an object")
    required = {
        "qualification_manifest_sha256",
        "causal_manifest_sha256",
        "baseline_checkpoint_sha256",
        "vjepa_checkpoint_sha256",
        "cache_metadata_sha256",
        "metric_cache_manifest_sha256",
        "raw_input_manifest_sha256",
        "qualification_config_sha256",
        "runtime_config_sha256",
        "qualification_source_manifest_sha256",
        "runtime_source_manifest_sha256",
        "source_revision",
        "proposal_schema_name",
        "proposal_schema_sha256",
        "simulator_name",
        "simulator_sha256",
        "scorer_name",
        "scorer_sha256",
        "channel_order",
    }
    if set(payload) != required:
        raise F2FrozenRecordBuildError("bindings JSON keys differ from the schema")
    converted = dict(payload)
    for field_name in (
        "qualification_manifest_sha256",
        "causal_manifest_sha256",
        "baseline_checkpoint_sha256",
        "vjepa_checkpoint_sha256",
        "cache_metadata_sha256",
        "metric_cache_manifest_sha256",
        "raw_input_manifest_sha256",
        "qualification_config_sha256",
        "runtime_config_sha256",
        "qualification_source_manifest_sha256",
        "runtime_source_manifest_sha256",
        "proposal_schema_sha256",
        "simulator_sha256",
        "scorer_sha256",
    ):
        converted[field_name] = _validated_sha256(
            converted[field_name], f"bindings.{field_name}"
        )
    if not isinstance(converted["channel_order"], list):
        raise F2FrozenRecordBuildError("bindings.channel_order must be a JSON array")
    converted["channel_order"] = tuple(converted["channel_order"])
    bindings = ArtifactBindings(**converted)
    reconstructed = {
        field_name: (
            list(bindings.channel_order)
            if field_name == "channel_order"
            else getattr(bindings, field_name)
        )
        for field_name in required
    }
    if reconstructed != payload:
        raise F2FrozenRecordBuildError(
            "bindings JSON is not idempotently canonical"
        )
    return bindings


def _token_file(path: Path, name: str) -> Tuple[str, ...]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise F2FrozenRecordBuildError(f"cannot read {name} token file") from error
    if any(not line for line in lines):
        raise F2FrozenRecordBuildError(f"{name} token file contains an empty line")
    return _unique_tokens(lines, f"{name} tokens")


def dry_run_plan(
    *,
    manifests: AuthenticatedHeldOutManifests,
    bindings: ArtifactBindings,
    live_provenance: VerifiedF2Provenance,
    destination: Path,
    feature_tokens: Sequence[str],
    cache_tokens: Sequence[str],
    baseline_sha256: str,
) -> FrozenRecordBuildPlan:
    """CLI-safe preflight that authenticates coverage without loading a model."""

    class _DryFeatureSource:
        dataset = F2_DATASET
        partition = "held_out"

        def __init__(self) -> None:
            self.tokens = tuple(feature_tokens)

        def load(self, row: ManifestRow) -> TokenFeatures:  # pragma: no cover
            raise AssertionError(f"dry-run attempted to load {row.token}")

    class _DryCacheSource:
        cache_family = F2_PRUNED_CACHE_FAMILY

        def __init__(self) -> None:
            self.tokens = tuple(cache_tokens)

        def load(self, row: ManifestRow) -> PrunedMetricCacheItem:  # pragma: no cover
            raise AssertionError(f"dry-run attempted to load {row.token}")

    class _DryBaseline:
        def __init__(self) -> None:
            self.authenticated_baseline_sha256 = baseline_sha256

        def infer(self, token: str, features: Mapping[str, Any]) -> Any:  # pragma: no cover
            raise AssertionError(f"dry-run attempted baseline inference for {token}")

    class _DryRestoredBaseline:
        def __init__(self) -> None:
            self.forward = _DryBaseline()

    class _DrySemantics:
        def run(self, metric_cache: Any, proposals: np.ndarray) -> Any:  # pragma: no cover
            raise AssertionError("dry-run attempted NAVSIM simulation")

    builder = FrozenNavsimRecordBuilder(
        manifests=manifests,
        bindings=bindings,
        live_provenance=live_provenance,
        feature_source=_DryFeatureSource(),
        metric_cache_source=_DryCacheSource(),
        restored_baseline=_DryRestoredBaseline(),
        navsim_semantics=_DrySemantics(),
        _dry_run_baseline_seal=_DRY_RUN_BASELINE_SEAL,
    )
    return builder.plan(destination)


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    """Authenticate a real build plan; intentionally never constructs records."""

    parser = argparse.ArgumentParser(
        description="Authenticate and dry-run an F2 frozen NAVSIM record build"
    )
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--causal-manifest", type=Path, required=True)
    parser.add_argument("--qualification-config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--cache-metadata", type=Path, required=True)
    parser.add_argument("--metric-cache-manifest", type=Path, required=True)
    parser.add_argument("--metric-cache-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--qualification-source-manifest", type=Path, required=True)
    parser.add_argument("--qualification-source-root", type=Path, required=True)
    parser.add_argument("--runtime-source-manifest", type=Path, required=True)
    parser.add_argument("--runtime-source-root", type=Path, required=True)
    parser.add_argument("--bindings-json", type=Path, required=True)
    parser.add_argument("--bindings-sha256", required=True)
    parser.add_argument("--feature-tokens", type=Path, required=True)
    parser.add_argument("--cache-tokens", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args(argv)

    bindings_payload = _canonical_json_file(
        args.bindings_json,
        expected_sha256=args.bindings_sha256,
        name="artifact bindings",
    )
    bindings = _bindings_from_payload(bindings_payload)
    live_provenance = verify_live_f2_provenance(
        bindings=bindings,
        qualification_manifest_path=args.qualification_manifest,
        causal_manifest_path=args.causal_manifest,
        qualification_config_path=args.qualification_config,
        runtime_config_path=args.runtime_config,
        cache_metadata_path=args.cache_metadata,
        metric_cache_manifest_path=args.metric_cache_manifest,
        metric_cache_root=args.metric_cache_root,
        baseline_checkpoint_path=args.baseline_checkpoint,
        vjepa_checkpoint_path=args.vjepa_checkpoint,
        qualification_source_manifest_path=args.qualification_source_manifest,
        qualification_source_root=args.qualification_source_root,
        runtime_source_manifest_path=args.runtime_source_manifest,
        runtime_source_root=args.runtime_source_root,
    )
    manifests = AuthenticatedHeldOutManifests(
        qualification=live_provenance.qualification,
        qualification_sha256=bindings.qualification_manifest_sha256,
        causal=live_provenance.causal,
        causal_sha256=bindings.causal_manifest_sha256,
    )
    plan = dry_run_plan(
        manifests=manifests,
        bindings=bindings,
        live_provenance=live_provenance,
        destination=args.destination,
        feature_tokens=_token_file(args.feature_tokens, "feature"),
        cache_tokens=_token_file(args.cache_tokens, "cache"),
        baseline_sha256=bindings.baseline_checkpoint_sha256,
    )
    payload = dict(plan.to_payload())
    payload.update(
        {
            "artifact_written": False,
            "dry_run": True,
            "outer_adapter_blocker": (
                "authenticated Drive-JEPA checkpoint-to-model and "
                "NAVSIM scene-to-feature construction remain required"
            ),
            "qualification_sha256": manifests.qualification_sha256,
            "causal_sha256": manifests.causal_sha256,
        }
    )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())


__all__ = (
    "AuthenticatedHeldOutManifests",
    "BaselineTokenOutput",
    "CALIBRATION_SIMULATOR_NAME",
    "CARRY_SIMULATOR_NAME",
    "DRIVE_JEPA_PROPOSAL_SHAPE",
    "DriveJepaBaselineForward",
    "DriveJepaNavsimRecordSemantics",
    "F2FrozenRecordBuildError",
    "F2_PRUNED_CACHE_FAMILY",
    "FrozenNavsimRecordBuilder",
    "FrozenRecordBuildPlan",
    "NavsimTokenSimulation",
    "PrunedMetricCacheItem",
    "PrunedMetricCacheSource",
    "TRAIN_SCORER_NAME",
    "TokenFeatures",
    "cli_main",
    "dry_run_plan",
    "load_authenticated_held_out_manifests",
)
