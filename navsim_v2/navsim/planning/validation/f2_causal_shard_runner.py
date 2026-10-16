"""Fail-closed 1/2/8-shard orchestration for F2 causal validation.

The heavy model and SceneLoader construction stay in explicit runtime
adapters.  This module owns everything after those authenticated adapters are
available: whole-log assignment, sealed-artifact reading, one-token inference,
independent critic/base state evolution, exact shard merge, aggregation, and
cross-world-size invariance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from navsim.planning.validation.f2_causal_model_adapter import (
    F2_CHECKPOINT_STAGE,
    F2_CHECKPOINT_VARIANT,
    F2CheckpointIdentity,
    F2CausalModelAdapter,
)
from navsim.planning.validation.f2_live_provenance import (
    F2LiveProvenanceError,
    VerifiedF2Provenance,
)
from navsim.planning.validation.f2_navsim_comfort import DriveJepaExtendedComfort
from navsim.planning.validation.f2_causal_validator import (
    F2_CHECKPOINT_EPOCHS,
    F2_MODEL_NUMERIC_MODE,
    AggregateMetrics,
    CanonicalManifest,
    CheckpointEvaluation,
    CausalValidationError,
    DualBranchCausalStateMachine,
    FrozenTokenArtifact,
    ManifestRow,
    ModelTokenOutput,
    TokenCausalResult,
    aggregate_results,
    canonical_evaluation_bytes,
    canonical_manifest_bytes,
    canonical_metrics_bytes,
    deterministic_log_assignments,
    merge_shard_results,
    rows_by_log,
)
from navsim.planning.validation.f2_frozen_artifact_io import (
    ArtifactBindings,
    FrozenArtifactCoverage,
    FrozenArtifactReader,
    ReadTokenRecord,
)
from navsim.planning.validation.f2_navsim_record_builder import TokenFeatures
from navsim.planning.validation.f2_navsim_runtime_adapters import (
    OFFICIAL_DRIVE_JEPA_FACTORY_REVISION,
    DriveJepaFactorySourceIdentity,
    QualifiedHeldOutNavtrainFeatureSource,
)


SUPPORTED_WORLD_SIZES = (1, 2, 8)


class F2CausalShardRunnerError(RuntimeError):
    """Raised when a causal shard cannot produce authoritative output."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise F2CausalShardRunnerError(f"{name} must be a 64-character SHA-256")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise F2CausalShardRunnerError(f"{name} must be hexadecimal")
    return normalized


def _rank_world(rank: Any, world_size: Any) -> Tuple[int, int]:
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise F2CausalShardRunnerError("world_size must be an integer")
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise F2CausalShardRunnerError(
            f"world_size must be one of {SUPPORTED_WORLD_SIZES!r}"
        )
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise F2CausalShardRunnerError("rank must be an integer")
    if not 0 <= rank < world_size:
        raise F2CausalShardRunnerError("rank is outside world_size")
    return rank, world_size


def _feature_identity(item: TokenFeatures) -> Tuple[str, int, str]:
    return item.log_name, item.time_us, item.token


def _row_identity(row: ManifestRow) -> Tuple[str, int, str]:
    return row.log_name, row.time_us, row.token


def _artifact_from_record(record: ReadTokenRecord) -> FrozenTokenArtifact:
    """Detach one mmap/stream record before its reader batch is released."""

    calibration = np.ascontiguousarray(record.calibration, dtype="<f8")
    carry = np.ascontiguousarray(record.carry, dtype="<f8")
    return FrozenTokenArtifact(
        token=record.token,
        proposal_sha256=record.proposal_sha256,
        base_logits=tuple(float(value) for value in record.base.tolist()),
        target_scores=tuple(
            tuple(float(value) for value in row) for row in record.targets.tolist()
        ),
        calibration_states=tuple(calibration[index].copy() for index in range(32)),
        carry_states=tuple(carry[index].copy() for index in range(32)),
    )


@dataclass(frozen=True)
class CausalShardEvaluation:
    """One rank's complete, in-memory whole-log evaluation."""

    rank: int
    world_size: int
    assigned_logs: Tuple[str, ...]
    checkpoint_sha256: str
    checkpoint_epoch: int
    manifest_sha256: str
    artifact_sha256: str
    config_sha256: str
    source_sha256: str
    results: Tuple[TokenCausalResult, ...]
    fatal_count: int = 0
    numeric_mode: str = F2_MODEL_NUMERIC_MODE

    def __post_init__(self) -> None:
        _rank_world(self.rank, self.world_size)
        object.__setattr__(
            self, "checkpoint_sha256", _sha256(self.checkpoint_sha256, "checkpoint")
        )
        object.__setattr__(
            self, "manifest_sha256", _sha256(self.manifest_sha256, "manifest")
        )
        object.__setattr__(
            self, "artifact_sha256", _sha256(self.artifact_sha256, "artifact")
        )
        object.__setattr__(self, "config_sha256", _sha256(self.config_sha256, "config"))
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "source"))
        if self.checkpoint_epoch not in F2_CHECKPOINT_EPOCHS:
            raise F2CausalShardRunnerError("checkpoint epoch must be 0, 1, 2, or 3")
        if self.numeric_mode != F2_MODEL_NUMERIC_MODE:
            raise F2CausalShardRunnerError("causal shard numeric mode drifted")
        if self.fatal_count != 0:
            raise F2CausalShardRunnerError(
                "authoritative shard output cannot contain a fatal token"
            )
        logs = tuple(self.assigned_logs)
        if logs != tuple(sorted(set(logs))):
            raise F2CausalShardRunnerError("assigned logs are not unique and sorted")
        results = tuple(
            sorted(self.results, key=lambda row: (row.log_name, row.time_us, row.token))
        )
        if len({row.token for row in results}) != len(results):
            raise F2CausalShardRunnerError("causal shard contains duplicate tokens")
        if {row.log_name for row in results} - set(logs):
            raise F2CausalShardRunnerError("causal shard contains an unassigned log")
        object.__setattr__(self, "assigned_logs", logs)
        object.__setattr__(self, "results", results)

    @property
    def evaluation_sha256(self) -> str:
        return _sha256_bytes(canonical_evaluation_bytes(self.results))


@dataclass(frozen=True)
class CausalWorldEvaluation:
    """Merged authoritative output for one supported world size."""

    world_size: int
    checkpoint_sha256: str
    checkpoint_epoch: int
    manifest_sha256: str
    artifact_sha256: str
    config_sha256: str
    source_sha256: str
    results: Tuple[TokenCausalResult, ...]
    metrics: AggregateMetrics
    numeric_mode: str = F2_MODEL_NUMERIC_MODE

    def __post_init__(self) -> None:
        _rank_world(0, self.world_size)
        for field in (
            "checkpoint_sha256",
            "manifest_sha256",
            "artifact_sha256",
            "config_sha256",
            "source_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        if self.checkpoint_epoch not in F2_CHECKPOINT_EPOCHS:
            raise F2CausalShardRunnerError("checkpoint epoch must be 0, 1, 2, or 3")
        if self.numeric_mode != F2_MODEL_NUMERIC_MODE:
            raise F2CausalShardRunnerError("causal world numeric mode drifted")
        if not isinstance(self.metrics, AggregateMetrics):
            raise F2CausalShardRunnerError("world metrics must be AggregateMetrics")
        results = tuple(
            sorted(self.results, key=lambda row: (row.log_name, row.time_us, row.token))
        )
        if len({row.token for row in results}) != len(results):
            raise F2CausalShardRunnerError("causal world contains duplicate tokens")
        if self.metrics.observed_token_count != len(results):
            raise F2CausalShardRunnerError(
                "causal world metrics do not match its token results"
            )
        object.__setattr__(self, "results", results)

    @property
    def evaluation_sha256(self) -> str:
        return _sha256_bytes(canonical_evaluation_bytes(self.results))

    @property
    def metrics_sha256(self) -> str:
        return _sha256_bytes(canonical_metrics_bytes(self.metrics))

    @property
    def checkpoint_evaluation(self) -> CheckpointEvaluation:
        return CheckpointEvaluation(
            epoch=self.checkpoint_epoch,
            checkpoint_sha256=self.checkpoint_sha256,
            manifest_sha256=self.manifest_sha256,
            artifact_sha256=self.artifact_sha256,
            numeric_mode=self.numeric_mode,
            metrics=self.metrics,
        )


def evaluate_causal_shard(
    *,
    manifest: CanonicalManifest,
    artifact_path: Any,
    artifact_bindings: ArtifactBindings,
    artifact_coverage: FrozenArtifactCoverage,
    expected_artifact_sha256: str,
    live_provenance: VerifiedF2Provenance,
    model_adapter: Any,
    feature_source: Any,
    comfort_function: Any,
    rank: int,
    world_size: int,
    reader_mode: str = "stream",
) -> CausalShardEvaluation:
    """Evaluate exactly the whole logs assigned to one rank.

    Any feature, model, artifact, or comfort failure aborts the rank.  Tokens
    are never skipped and an exception can never be converted into a partial
    authoritative shard.
    """

    rank, world_size = _rank_world(rank, world_size)
    if type(live_provenance) is not VerifiedF2Provenance:
        raise F2CausalShardRunnerError(
            "causal evaluation requires byte-verified live F2 provenance"
        )
    try:
        live_provenance.require_bindings(artifact_bindings)
    except F2LiveProvenanceError as error:
        raise F2CausalShardRunnerError(str(error)) from error
    if live_provenance.causal != manifest:
        raise F2CausalShardRunnerError(
            "runtime causal manifest differs from live verified provenance"
        )
    manifest_sha256 = artifact_bindings.causal_manifest_sha256
    if _sha256_bytes(canonical_manifest_bytes(manifest)) != manifest_sha256:
        raise F2CausalShardRunnerError("in-memory causal manifest SHA-256 drifted")
    artifact_sha256 = _sha256(expected_artifact_sha256, "artifact")
    config_sha256 = artifact_bindings.runtime_config_sha256
    source_sha256 = artifact_bindings.runtime_source_manifest_sha256
    expected_coverage = tuple(
        (row.log_name, row.time_us, row.token) for row in manifest.rows
    )
    actual_coverage = tuple(
        (entry.log_name, entry.time_us, entry.token)
        for entry in artifact_coverage.entries
    )
    if actual_coverage != expected_coverage:
        raise F2CausalShardRunnerError(
            "artifact coverage is not the exact causal manifest row sequence"
        )

    if type(model_adapter) is not F2CausalModelAdapter:
        raise F2CausalShardRunnerError(
            "model adapter must be an authenticated F2CausalModelAdapter"
        )
    if not model_adapter.is_official_factory:
        raise F2CausalShardRunnerError(
            "model adapter was not constructed by the sealed official F2 factory"
        )
    if model_adapter.live_provenance is not live_provenance:
        raise F2CausalShardRunnerError(
            "model adapter and causal runner do not share verified provenance"
        )
    factory_source = model_adapter.factory_source
    if (
        type(factory_source) is not DriveJepaFactorySourceIdentity
        or factory_source.revision != OFFICIAL_DRIVE_JEPA_FACTORY_REVISION
    ):
        raise F2CausalShardRunnerError(
            "model adapter lacks the authenticated Drive-JEPA factory source"
        )
    identity = model_adapter.checkpoint_identity
    if not isinstance(identity, F2CheckpointIdentity):
        raise F2CausalShardRunnerError(
            "model adapter lacks an authenticated F2CheckpointIdentity"
        )
    if identity.epoch not in F2_CHECKPOINT_EPOCHS:
        raise F2CausalShardRunnerError("model checkpoint epoch is outside F2")
    if identity.stage != F2_CHECKPOINT_STAGE or identity.variant != F2_CHECKPOINT_VARIANT:
        raise F2CausalShardRunnerError(
            "model checkpoint is not the frozen F2 critic/base-residual variant"
        )
    if not callable(getattr(model_adapter, "infer_token", None)):
        raise F2CausalShardRunnerError("model adapter must implement infer_token()")
    if (
        type(comfort_function) is not DriveJepaExtendedComfort
        or not comfort_function.is_official
    ):
        raise F2CausalShardRunnerError(
            "comfort function must be the official authenticated Drive-JEPA adapter"
        )

    assignments = deterministic_log_assignments(manifest.rows, world_size)
    assigned_logs = assignments[rank]
    grouped = rows_by_log(manifest.rows)
    assigned_rows = tuple(row for log in assigned_logs for row in grouped[log])
    assigned_tokens = tuple(row.token for row in assigned_rows)
    if type(feature_source) is not QualifiedHeldOutNavtrainFeatureSource:
        raise F2CausalShardRunnerError(
            "feature source must be the concrete qualified NAVSIM runtime adapter"
        )
    if feature_source.live_provenance is not live_provenance:
        raise F2CausalShardRunnerError(
            "feature source and causal runner do not share the same verified provenance"
        )
    if (
        getattr(feature_source, "rank", None) != rank
        or getattr(feature_source, "world_size", None) != world_size
    ):
        raise F2CausalShardRunnerError(
            "feature source rank/world_size differs from the causal assignment"
        )
    if getattr(feature_source, "dataset", None) != "navtrain" or getattr(
        feature_source, "partition", None
    ) not in ("held_out", "validation"):
        raise F2CausalShardRunnerError(
            "feature source must be authenticated held-out navtrain"
        )
    source_tokens = tuple(getattr(feature_source, "tokens", ()))
    if source_tokens != assigned_tokens:
        raise F2CausalShardRunnerError(
            "feature source tokens must exactly equal this rank's canonical assignment"
        )
    if not callable(getattr(feature_source, "load", None)):
        raise F2CausalShardRunnerError("feature source must implement load()")

    results = []
    try:
        with FrozenArtifactReader.open(
            artifact_path,
            expected_bindings=artifact_bindings,
            expected_coverage=artifact_coverage,
            mode=reader_mode,
        ) as reader:
            if reader.descriptor.root_sha256 != artifact_sha256:
                raise F2CausalShardRunnerError("sealed artifact root SHA-256 drifted")
            machine = DualBranchCausalStateMachine(comfort_function)
            for log_name in assigned_logs:
                rows = grouped[log_name]
                records = tuple(reader.iter_records(log_name=log_name, batch_records=4))
                if tuple(record.token for record in records) != tuple(
                    row.token for row in rows
                ):
                    raise F2CausalShardRunnerError(
                        f"artifact log {log_name!r} differs from manifest order"
                    )
                artifacts = {
                    record.token: _artifact_from_record(record) for record in records
                }
                # In mmap mode the typed records own exported buffer views.
                # Every artifact field above is detached, so release those
                # views before the reader attempts to close its mapping.
                del records
                outputs = {}
                for row in rows:
                    feature_item = feature_source.load(row)
                    if not isinstance(feature_item, TokenFeatures):
                        raise F2CausalShardRunnerError(
                            "feature source must return TokenFeatures"
                        )
                    if _feature_identity(feature_item) != _row_identity(row):
                        raise F2CausalShardRunnerError(
                            f"feature row identity drifted for token {row.token!r}"
                        )
                    output = model_adapter.infer_token(row.token, feature_item.features)
                    if not isinstance(output, ModelTokenOutput):
                        raise F2CausalShardRunnerError(
                            "model adapter must return ModelTokenOutput"
                        )
                    outputs[row.token] = output
                results.extend(machine.evaluate_log(rows, artifacts, outputs))
    except (F2CausalShardRunnerError, CausalValidationError):
        raise
    except Exception as error:
        raise F2CausalShardRunnerError(
            f"fatal causal shard failure: {type(error).__name__}: {error}"
        ) from error

    return CausalShardEvaluation(
        rank=rank,
        world_size=world_size,
        assigned_logs=assigned_logs,
        checkpoint_sha256=identity.sha256,
        checkpoint_epoch=identity.epoch,
        manifest_sha256=manifest_sha256,
        artifact_sha256=artifact_sha256,
        config_sha256=config_sha256,
        source_sha256=source_sha256,
        results=tuple(results),
    )


def merge_causal_world(
    manifest: CanonicalManifest,
    shards: Sequence[CausalShardEvaluation],
) -> CausalWorldEvaluation:
    """Require every rank once, exact assignments, and exact token coverage."""

    if not shards:
        raise F2CausalShardRunnerError("cannot merge zero causal shards")
    world_sizes = {shard.world_size for shard in shards}
    if len(world_sizes) != 1:
        raise F2CausalShardRunnerError("causal shards use different world sizes")
    world_size = next(iter(world_sizes))
    _rank_world(0, world_size)
    by_rank = {shard.rank: shard for shard in shards}
    if len(by_rank) != len(shards) or set(by_rank) != set(range(world_size)):
        raise F2CausalShardRunnerError("causal shard rank coverage is not exact")
    assignments = deterministic_log_assignments(manifest.rows, world_size)
    reference = by_rank[0]
    identity_fields = (
        "checkpoint_sha256",
        "checkpoint_epoch",
        "manifest_sha256",
        "artifact_sha256",
        "config_sha256",
        "source_sha256",
        "numeric_mode",
    )
    for rank in range(world_size):
        shard = by_rank[rank]
        if shard.assigned_logs != assignments[rank]:
            raise F2CausalShardRunnerError(
                f"rank {rank} whole-log assignment differs from the manifest"
            )
        for field in identity_fields:
            if getattr(shard, field) != getattr(reference, field):
                raise F2CausalShardRunnerError(
                    f"causal shard {field} differs across ranks"
                )
    merged = merge_shard_results(
        (by_rank[rank].results for rank in range(world_size)), manifest.tokens
    )
    metrics = aggregate_results(merged, expected_tokens=manifest.tokens, fatal_count=0)
    return CausalWorldEvaluation(
        world_size=world_size,
        checkpoint_sha256=reference.checkpoint_sha256,
        checkpoint_epoch=reference.checkpoint_epoch,
        manifest_sha256=reference.manifest_sha256,
        artifact_sha256=reference.artifact_sha256,
        config_sha256=reference.config_sha256,
        source_sha256=reference.source_sha256,
        results=merged,
        metrics=metrics,
    )


def prove_1_2_8_invariance(
    evaluations: Sequence[CausalWorldEvaluation],
) -> CausalWorldEvaluation:
    """Require byte-identical results and metrics for worlds 1, 2, and 8."""

    by_world = {evaluation.world_size: evaluation for evaluation in evaluations}
    if len(by_world) != len(evaluations) or set(by_world) != set(SUPPORTED_WORLD_SIZES):
        raise F2CausalShardRunnerError(
            "invariance proof requires exactly one 1-, 2-, and 8-rank evaluation"
        )
    reference = by_world[1]
    identity_fields = (
        "checkpoint_sha256",
        "checkpoint_epoch",
        "manifest_sha256",
        "artifact_sha256",
        "config_sha256",
        "source_sha256",
        "numeric_mode",
    )
    reference_results = canonical_evaluation_bytes(reference.results)
    reference_metrics = canonical_metrics_bytes(reference.metrics)
    for world_size in (2, 8):
        current = by_world[world_size]
        for field in identity_fields:
            if getattr(current, field) != getattr(reference, field):
                raise F2CausalShardRunnerError(
                    f"world-size invariance identity {field} drifted"
                )
        if canonical_evaluation_bytes(current.results) != reference_results:
            raise F2CausalShardRunnerError(
                f"{world_size}-rank causal token results differ from 1 rank"
            )
        if canonical_metrics_bytes(current.metrics) != reference_metrics:
            raise F2CausalShardRunnerError(
                f"{world_size}-rank causal metrics differ from 1 rank"
            )
    return reference


__all__ = (
    "CausalShardEvaluation",
    "CausalWorldEvaluation",
    "F2CausalShardRunnerError",
    "SUPPORTED_WORLD_SIZES",
    "evaluate_causal_shard",
    "merge_causal_world",
    "prove_1_2_8_invariance",
)
