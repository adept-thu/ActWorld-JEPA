"""Whole-log sharded production builder for the sealed F2 frozen artifact.

The record semantics remain owned by :mod:`f2_navsim_record_builder` and the
binary format remains owned by :mod:`f2_frozen_artifact_io`.  This module only
adds production orchestration around those frozen components:

* deterministic whole-log rank plans for worlds 1, 2, and 8;
* rank-local binary artifacts built by the authenticated official adapters;
* strict canonical rank certificates with SHA-256 sidecars;
* exact, non-overlapping all-rank validation; and
* bounded-memory re-sealing of a single full-coverage artifact.

The only accepted data view is the authenticated held-out portion of
``navtrain``.  Split selection is intentionally absent from both API and CLI.
"""

from __future__ import annotations

import argparse
import hashlib
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from navsim.planning.validation.f2_causal_validator import (
    CanonicalManifest,
    ManifestRow,
    canonical_manifest_bytes,
    deterministic_log_assignments,
    rows_by_log,
)
from navsim.planning.validation.f2_frozen_artifact_io import (
    ArtifactBindings,
    ArtifactDescriptor,
    CoverageEntry,
    FrozenArtifactChunk,
    FrozenArtifactCoverage,
    FrozenArtifactReader,
    FrozenTokenRecord,
    ReadTokenRecord,
    build_artifact_coverage,
    seal_frozen_artifact,
    verify_frozen_artifact,
)
from navsim.planning.validation.f2_live_provenance import (
    VerifiedF2Provenance,
    verify_live_f2_provenance,
)
from navsim.planning.validation.f2_navsim_record_builder import (
    AuthenticatedHeldOutManifests,
    DriveJepaNavsimRecordSemantics,
    F2FrozenRecordBuildError,
    FrozenNavsimRecordBuilder,
    PrunedMetricCacheSource,
)
from navsim.planning.validation.f2_navsim_runtime_adapters import (
    build_official_drive_jepa_baseline,
    build_qualified_held_out_navtrain_feature_source,
)
from navsim.planning.validation.f2_navtrain_input_manifest import (
    VerifiedNavtrainRawInputs,
    load_and_verify_raw_input_manifest,
)
from navsim.planning.validation.f2_production_runner import (
    DATASET,
    PARTITION,
    F2ProductionRunnerError,
    ProductionRuntimeConfig,
    _parse_runtime_config,
    artifact_bindings_bytes,
    canonical_json_bytes,
    initialize_deterministic_runtime,
    parse_canonical_json_bytes,
    read_artifact_bindings,
    read_bytes_with_sidecar,
    sha256_sidecar_path,
    write_bytes_no_overwrite,
)


SUPPORTED_WORLD_SIZES = (1, 2, 8)
RANK_CERTIFICATE_SCHEMA_VERSION = "actworld-f2-frozen-rank-artifact-v1"
MERGED_CERTIFICATE_SCHEMA_VERSION = "actworld-f2-frozen-merged-artifact-v1"

_ROW_KEYS = {"future_camera_valid", "log_name", "time_us", "token"}
_BINDING_IDENTITY_KEYS = {
    "baseline_checkpoint_sha256",
    "bindings_sha256",
    "causal_manifest_sha256",
    "metric_cache_manifest_sha256",
    "qualification_manifest_sha256",
    "raw_input_manifest_sha256",
    "runtime_config_sha256",
    "runtime_source_manifest_sha256",
    "source_revision",
    "vjepa_checkpoint_sha256",
}
_ARTIFACT_KEYS = {
    "coverage_sha256",
    "file_name",
    "file_size",
    "record_count",
    "root_sha256",
}


class F2ArtifactProductionError(RuntimeError):
    """Raised if a shard or merged frozen artifact is not authoritative."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise F2ArtifactProductionError(f"{name} must be a 64-character SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise F2ArtifactProductionError(
            f"{name} must use lowercase hexadecimal"
        )
    return value


def _exact_keys(payload: Any, expected: set, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise F2ArtifactProductionError(f"{name} must be a JSON object")
    actual = set(payload)
    if actual != expected:
        raise F2ArtifactProductionError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return payload


def _rank_world(rank: Any, world_size: Any) -> Tuple[int, int]:
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size not in SUPPORTED_WORLD_SIZES
        or not 0 <= rank < world_size
    ):
        raise F2ArtifactProductionError(
            "rank/world_size must describe a supported 1-, 2-, or 8-rank world"
        )
    return rank, world_size


def _row_identity(row: Any) -> Tuple[str, int, str, bool]:
    try:
        identity = (
            row.log_name,
            row.time_us,
            row.token,
            row.future_camera_valid,
        )
    except AttributeError as error:
        raise F2ArtifactProductionError("manifest row lacks canonical identity") from error
    if (
        not isinstance(identity[0], str)
        or not identity[0]
        or isinstance(identity[1], bool)
        or not isinstance(identity[1], int)
        or identity[1] < 0
        or not isinstance(identity[2], str)
        or not identity[2]
        or not isinstance(identity[3], bool)
    ):
        raise F2ArtifactProductionError("manifest row identity is malformed")
    return identity


def _manifest_sha256(manifest: CanonicalManifest) -> str:
    if not isinstance(manifest, CanonicalManifest):
        raise F2ArtifactProductionError("causal manifest must be CanonicalManifest")
    return _sha256_bytes(canonical_manifest_bytes(manifest))


def _bindings_sha256(bindings: ArtifactBindings) -> str:
    if type(bindings) is not ArtifactBindings:
        raise F2ArtifactProductionError("bindings must be concrete ArtifactBindings")
    return _sha256_bytes(artifact_bindings_bytes(bindings))


def _binding_identity(bindings: ArtifactBindings) -> Dict[str, Any]:
    return {
        "baseline_checkpoint_sha256": bindings.baseline_checkpoint_sha256,
        "bindings_sha256": _bindings_sha256(bindings),
        "causal_manifest_sha256": bindings.causal_manifest_sha256,
        "metric_cache_manifest_sha256": bindings.metric_cache_manifest_sha256,
        "qualification_manifest_sha256": bindings.qualification_manifest_sha256,
        "raw_input_manifest_sha256": bindings.raw_input_manifest_sha256,
        "runtime_config_sha256": bindings.runtime_config_sha256,
        "runtime_source_manifest_sha256": bindings.runtime_source_manifest_sha256,
        "source_revision": bindings.source_revision,
        "vjepa_checkpoint_sha256": bindings.vjepa_checkpoint_sha256,
    }


def _validate_binding_identity(
    payload: Any, bindings: ArtifactBindings
) -> Mapping[str, Any]:
    identity = _exact_keys(payload, _BINDING_IDENTITY_KEYS, "binding identity")
    expected = _binding_identity(bindings)
    if dict(identity) != expected:
        raise F2ArtifactProductionError(
            "artifact certificate binding identity differs from live bindings"
        )
    return identity


def _direct_sibling_name(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise F2ArtifactProductionError(f"{name} must be a direct sibling file name")
    return value


def _same_parent(left: Path, right: Path, name: str) -> None:
    if left.parent.resolve() != right.parent.resolve():
        raise F2ArtifactProductionError(f"{name} must be a direct sibling")


def _require_output_absent(path: Path, name: str) -> None:
    candidates = (
        path,
        path.with_name(path.name + ".partial"),
        sha256_sidecar_path(path),
        sha256_sidecar_path(path).with_name(
            sha256_sidecar_path(path).name + ".partial"
        ),
    )
    occupied = [candidate for candidate in candidates if candidate.exists()]
    if occupied:
        raise F2ArtifactProductionError(
            f"{name} output/sidecar/partial already exists; overwrite is forbidden"
        )


@dataclass(frozen=True)
class RankArtifactPlan:
    rank: int
    world_size: int
    assigned_logs: Tuple[str, ...]
    rows: Tuple[ManifestRow, ...]
    coverage: FrozenArtifactCoverage

    def __post_init__(self) -> None:
        _rank_world(self.rank, self.world_size)
        logs = tuple(sorted(set(self.assigned_logs)))
        if logs != self.assigned_logs or not logs:
            raise F2ArtifactProductionError(
                "rank artifact logs must be non-empty, unique, and sorted"
            )
        rows = tuple(self.rows)
        canonical = tuple(
            sorted(rows, key=lambda row: (row.log_name, row.time_us, row.token))
        )
        if rows != canonical or not rows:
            raise F2ArtifactProductionError(
                "rank artifact rows must be non-empty and canonically sorted"
            )
        if {row.log_name for row in rows} != set(logs):
            raise F2ArtifactProductionError("rank rows and assigned logs differ")
        if not isinstance(self.coverage, FrozenArtifactCoverage):
            raise F2ArtifactProductionError("rank coverage is malformed")
        expected = tuple((row.log_name, row.time_us, row.token) for row in rows)
        observed = tuple(
            (row.log_name, row.time_us, row.token) for row in self.coverage.entries
        )
        if observed != expected:
            raise F2ArtifactProductionError("rank coverage differs from assigned rows")


def build_rank_artifact_plan(
    *,
    causal_manifest: CanonicalManifest,
    qualification_held_out_rows: Sequence[Any],
    rank: int,
    world_size: int,
) -> RankArtifactPlan:
    """Derive one rank only from qualification held-out whole-log assignment."""

    rank, world_size = _rank_world(rank, world_size)
    if not isinstance(causal_manifest, CanonicalManifest):
        raise F2ArtifactProductionError("causal manifest must be CanonicalManifest")
    qualification_rows = tuple(qualification_held_out_rows)
    qualification_identities = tuple(_row_identity(row) for row in qualification_rows)
    causal_identities = tuple(_row_identity(row) for row in causal_manifest.rows)
    if qualification_identities != causal_identities:
        raise F2ArtifactProductionError(
            "qualification held-out and causal row coverage differ"
        )
    assignments = deterministic_log_assignments(qualification_rows, world_size)
    assigned_logs = assignments[rank]
    if not assigned_logs:
        raise F2ArtifactProductionError(
            "rank received zero whole logs; empty binary artifacts are forbidden"
        )
    grouped = rows_by_log(causal_manifest.rows)
    rows = tuple(row for log_name in assigned_logs for row in grouped[log_name])
    rows = tuple(sorted(rows, key=lambda row: (row.log_name, row.time_us, row.token)))
    coverage = build_artifact_coverage(
        CoverageEntry(row.log_name, row.time_us, row.token) for row in rows
    )
    return RankArtifactPlan(
        rank=rank,
        world_size=world_size,
        assigned_logs=assigned_logs,
        rows=rows,
        coverage=coverage,
    )


@dataclass(frozen=True)
class RankArtifactCertificate:
    rank: int
    world_size: int
    assigned_logs: Tuple[str, ...]
    rows: Tuple[ManifestRow, ...]
    artifact_path: Path
    descriptor: ArtifactDescriptor
    bindings_sha256: str
    _seal: object = field(repr=False, compare=False)


_RANK_CERTIFICATE_SEAL = object()


def _artifact_payload(path: Path, descriptor: ArtifactDescriptor) -> Dict[str, Any]:
    return {
        "coverage_sha256": descriptor.coverage_sha256,
        "file_name": path.name,
        "file_size": descriptor.file_size,
        "record_count": descriptor.record_count,
        "root_sha256": descriptor.root_sha256,
    }


def _rank_certificate_payload(
    *,
    plan: RankArtifactPlan,
    bindings: ArtifactBindings,
    artifact_path: Path,
    descriptor: ArtifactDescriptor,
) -> Dict[str, Any]:
    return {
        "artifact": _artifact_payload(artifact_path, descriptor),
        "assigned_logs": list(plan.assigned_logs),
        "dataset": DATASET,
        "fatal_count": 0,
        "identity": _binding_identity(bindings),
        "kind": "frozen_rank_artifact",
        "partition": PARTITION,
        "rank": plan.rank,
        "rows": [row.to_payload() for row in plan.rows],
        "schema_version": RANK_CERTIFICATE_SCHEMA_VERSION,
        "world_size": plan.world_size,
    }


RecordFactory = Callable[[ManifestRow], FrozenTokenRecord]


def seal_rank_artifact(
    *,
    causal_manifest: CanonicalManifest,
    qualification_held_out_rows: Sequence[Any],
    bindings: ArtifactBindings,
    rank: int,
    world_size: int,
    artifact_path: Union[Path, str],
    certificate_path: Union[Path, str],
    record_factory: RecordFactory,
) -> RankArtifactCertificate:
    """Build and certify exactly one deterministic whole-log rank."""

    if not callable(record_factory):
        raise F2ArtifactProductionError("record_factory must be callable")
    if _manifest_sha256(causal_manifest) != bindings.causal_manifest_sha256:
        raise F2ArtifactProductionError("causal manifest differs from bindings")
    artifact = Path(artifact_path)
    certificate = Path(certificate_path)
    if not artifact.parent.is_dir() or not certificate.parent.is_dir():
        raise F2ArtifactProductionError("rank output parent does not exist")
    _same_parent(artifact, certificate, "rank artifact and certificate")
    if artifact == certificate:
        raise F2ArtifactProductionError("artifact and certificate paths must differ")
    _require_output_absent(artifact, "rank artifact")
    _require_output_absent(certificate, "rank certificate")
    plan = build_rank_artifact_plan(
        causal_manifest=causal_manifest,
        qualification_held_out_rows=qualification_held_out_rows,
        rank=rank,
        world_size=world_size,
    )
    grouped = rows_by_log(plan.rows)

    def checked_records(log_name: str) -> Iterable[FrozenTokenRecord]:
        for row in grouped[log_name]:
            try:
                record = record_factory(row)
            except F2ArtifactProductionError:
                raise
            except Exception as error:
                raise F2ArtifactProductionError(
                    "fatal rank record build for "
                    f"{(row.log_name, row.time_us, row.token)!r}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            if not isinstance(record, FrozenTokenRecord):
                raise F2ArtifactProductionError(
                    "record_factory must return FrozenTokenRecord"
                )
            if (record.time_us, record.token) != (row.time_us, row.token):
                raise F2ArtifactProductionError(
                    "record_factory returned a different canonical row"
                )
            yield record

    chunks = (
        FrozenArtifactChunk(
            log_name=log_name,
            records=checked_records(log_name),
        )
        for log_name in plan.assigned_logs
    )
    try:
        descriptor = seal_frozen_artifact(
            artifact,
            bindings=bindings,
            coverage=plan.coverage,
            chunks=chunks,
        )
    except Exception as error:
        raise F2ArtifactProductionError(
            f"rank binary artifact sealing failed: {type(error).__name__}: {error}"
        ) from error
    payload = _rank_certificate_payload(
        plan=plan,
        bindings=bindings,
        artifact_path=artifact,
        descriptor=descriptor,
    )
    try:
        write_bytes_no_overwrite(certificate, canonical_json_bytes(payload))
    except F2ProductionRunnerError as error:
        raise F2ArtifactProductionError(
            "rank certificate publication failed; binary artifact remains uncertified"
        ) from error
    return RankArtifactCertificate(
        rank=plan.rank,
        world_size=plan.world_size,
        assigned_logs=plan.assigned_logs,
        rows=plan.rows,
        artifact_path=artifact,
        descriptor=descriptor,
        bindings_sha256=_bindings_sha256(bindings),
        _seal=_RANK_CERTIFICATE_SEAL,
    )


def _parse_manifest_row(payload: Any) -> ManifestRow:
    row = _exact_keys(payload, _ROW_KEYS, "certificate row")
    try:
        return ManifestRow(
            log_name=row["log_name"],
            time_us=row["time_us"],
            token=row["token"],
            future_camera_valid=row["future_camera_valid"],
        )
    except Exception as error:
        raise F2ArtifactProductionError("certificate row is malformed") from error


def _validate_artifact_document(
    payload: Any,
    *,
    certificate_parent: Path,
    bindings: ArtifactBindings,
    coverage: FrozenArtifactCoverage,
) -> Tuple[Path, ArtifactDescriptor]:
    artifact = _exact_keys(payload, _ARTIFACT_KEYS, "artifact identity")
    file_name = _direct_sibling_name(artifact["file_name"], "artifact file")
    artifact_path = certificate_parent / file_name
    try:
        descriptor = verify_frozen_artifact(
            artifact_path,
            expected_bindings=bindings,
            expected_coverage=coverage,
        )
    except Exception as error:
        raise F2ArtifactProductionError(
            f"certified binary artifact verification failed: {type(error).__name__}: {error}"
        ) from error
    expected = _artifact_payload(artifact_path, descriptor)
    if dict(artifact) != expected:
        raise F2ArtifactProductionError(
            "certificate artifact identity differs from verified binary bytes"
        )
    return artifact_path, descriptor


def read_rank_artifact_certificate(
    path: Union[Path, str],
    *,
    bindings: ArtifactBindings,
    causal_manifest: CanonicalManifest,
    qualification_held_out_rows: Sequence[Any],
) -> RankArtifactCertificate:
    """Read a rank certificate and fully re-verify its binary artifact."""

    certificate = Path(path).resolve(strict=True)
    try:
        encoded, _ = read_bytes_with_sidecar(certificate)
        decoded = parse_canonical_json_bytes(
            encoded, name="frozen rank artifact certificate"
        )
    except F2ProductionRunnerError as error:
        raise F2ArtifactProductionError("rank certificate authentication failed") from error
    document = _exact_keys(
        decoded,
        {
            "artifact",
            "assigned_logs",
            "dataset",
            "fatal_count",
            "identity",
            "kind",
            "partition",
            "rank",
            "rows",
            "schema_version",
            "world_size",
        },
        "rank artifact certificate",
    )
    if (
        document["schema_version"] != RANK_CERTIFICATE_SCHEMA_VERSION
        or document["kind"] != "frozen_rank_artifact"
        or document["dataset"] != DATASET
        or document["partition"] != PARTITION
    ):
        raise F2ArtifactProductionError("rank certificate contract drifted")
    if document["fatal_count"] != 0 or isinstance(document["fatal_count"], bool):
        raise F2ArtifactProductionError("authoritative rank artifact contains a fatal")
    _validate_binding_identity(document["identity"], bindings)
    if _manifest_sha256(causal_manifest) != bindings.causal_manifest_sha256:
        raise F2ArtifactProductionError("causal manifest differs from bindings")
    if not isinstance(document["rows"], list) or not isinstance(
        document["assigned_logs"], list
    ):
        raise F2ArtifactProductionError("rank certificate rows/logs must be arrays")
    rows = tuple(_parse_manifest_row(item) for item in document["rows"])
    plan = build_rank_artifact_plan(
        causal_manifest=causal_manifest,
        qualification_held_out_rows=qualification_held_out_rows,
        rank=document["rank"],
        world_size=document["world_size"],
    )
    if rows != plan.rows or tuple(document["assigned_logs"]) != plan.assigned_logs:
        raise F2ArtifactProductionError(
            "rank certificate differs from deterministic whole-log assignment"
        )
    artifact_path, descriptor = _validate_artifact_document(
        document["artifact"],
        certificate_parent=certificate.parent,
        bindings=bindings,
        coverage=plan.coverage,
    )
    return RankArtifactCertificate(
        rank=plan.rank,
        world_size=plan.world_size,
        assigned_logs=plan.assigned_logs,
        rows=plan.rows,
        artifact_path=artifact_path,
        descriptor=descriptor,
        bindings_sha256=_bindings_sha256(bindings),
        _seal=_RANK_CERTIFICATE_SEAL,
    )


@dataclass(frozen=True)
class RankCertificateReference:
    rank: int
    file_name: str
    file_sha256: str
    artifact_root_sha256: str
    coverage_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise F2ArtifactProductionError("rank reference rank is invalid")
        object.__setattr__(
            self, "file_name", _direct_sibling_name(self.file_name, "rank certificate")
        )
        for field_name in (
            "file_sha256",
            "artifact_root_sha256",
            "coverage_sha256",
        ):
            object.__setattr__(
                self, field_name, _sha256(getattr(self, field_name), field_name)
            )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "artifact_root_sha256": self.artifact_root_sha256,
            "coverage_sha256": self.coverage_sha256,
            "file_name": self.file_name,
            "file_sha256": self.file_sha256,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class MergedArtifactCertificate:
    world_size: int
    artifact_path: Path
    descriptor: ArtifactDescriptor
    shards: Tuple[RankCertificateReference, ...]
    bindings_sha256: str
    _seal: object = field(repr=False, compare=False)


_MERGED_CERTIFICATE_SEAL = object()


def _copy_record(record: ReadTokenRecord) -> FrozenTokenRecord:
    return FrozenTokenRecord(
        token=record.token,
        time_us=record.time_us,
        proposal_sha256=record.proposal_sha256,
        base=np.ascontiguousarray(record.base, dtype=np.dtype("<f4")).copy(),
        targets=np.ascontiguousarray(record.targets, dtype=np.dtype("<f4")).copy(),
        calibration=np.ascontiguousarray(
            record.calibration, dtype=np.dtype("<f8")
        ).copy(),
        carry=np.ascontiguousarray(record.carry, dtype=np.dtype("<f8")).copy(),
    )


def _validate_rank_set(
    *,
    certificates: Sequence[RankArtifactCertificate],
    causal_manifest: CanonicalManifest,
    world_size: int,
    bindings_sha256: str,
) -> Dict[str, RankArtifactCertificate]:
    if len(certificates) != world_size:
        raise F2ArtifactProductionError("merge requires exactly one certificate per rank")
    by_rank = {item.rank: item for item in certificates}
    if len(by_rank) != len(certificates) or set(by_rank) != set(range(world_size)):
        raise F2ArtifactProductionError("rank coverage is incomplete or duplicated")
    if any(item.world_size != world_size for item in certificates):
        raise F2ArtifactProductionError("rank certificate world_size drifted")
    if any(item.bindings_sha256 != bindings_sha256 for item in certificates):
        raise F2ArtifactProductionError("rank certificate bindings differ")
    observed_rows = [
        _row_identity(row)
        for rank in range(world_size)
        for row in by_rank[rank].rows
    ]
    if len(observed_rows) != len(set(observed_rows)):
        raise F2ArtifactProductionError("rank artifact row coverage overlaps")
    expected_rows = tuple(_row_identity(row) for row in causal_manifest.rows)
    if set(observed_rows) != set(expected_rows) or len(observed_rows) != len(expected_rows):
        raise F2ArtifactProductionError(
            "rank artifact union does not equal full causal coverage"
        )
    observed_logs = [
        log_name
        for rank in range(world_size)
        for log_name in by_rank[rank].assigned_logs
    ]
    if len(observed_logs) != len(set(observed_logs)):
        raise F2ArtifactProductionError("whole-log rank assignments overlap")
    if set(observed_logs) != set(causal_manifest.logs):
        raise F2ArtifactProductionError("rank log union differs from causal manifest")
    return {log_name: item for item in certificates for log_name in item.assigned_logs}


def _merged_certificate_payload(
    *,
    world_size: int,
    bindings: ArtifactBindings,
    artifact_path: Path,
    descriptor: ArtifactDescriptor,
    shards: Sequence[RankCertificateReference],
) -> Dict[str, Any]:
    return {
        "artifact": _artifact_payload(artifact_path, descriptor),
        "dataset": DATASET,
        "fatal_count": 0,
        "identity": _binding_identity(bindings),
        "kind": "frozen_merged_artifact",
        "partition": PARTITION,
        "schema_version": MERGED_CERTIFICATE_SCHEMA_VERSION,
        "shards": [item.to_payload() for item in shards],
        "world_size": world_size,
    }


def merge_rank_artifacts(
    *,
    causal_manifest: CanonicalManifest,
    qualification_held_out_rows: Sequence[Any],
    bindings: ArtifactBindings,
    world_size: int,
    rank_certificate_paths: Sequence[Union[Path, str]],
    artifact_path: Union[Path, str],
    certificate_path: Union[Path, str],
) -> MergedArtifactCertificate:
    """Re-verify every rank and stream-reseal one full artifact."""

    _, world_size = _rank_world(0, world_size)
    if _manifest_sha256(causal_manifest) != bindings.causal_manifest_sha256:
        raise F2ArtifactProductionError("causal manifest differs from bindings")
    output_artifact = Path(artifact_path)
    output_certificate = Path(certificate_path)
    if not output_artifact.parent.is_dir() or not output_certificate.parent.is_dir():
        raise F2ArtifactProductionError("merged output parent does not exist")
    _same_parent(output_artifact, output_certificate, "merged artifact and certificate")
    _require_output_absent(output_artifact, "merged artifact")
    _require_output_absent(output_certificate, "merged certificate")
    if len(rank_certificate_paths) != world_size:
        raise F2ArtifactProductionError("merge requires exactly one certificate per rank")

    certificates = []
    references = []
    for raw_path in rank_certificate_paths:
        rank_path = Path(raw_path).resolve(strict=True)
        if rank_path.parent != output_certificate.parent.resolve():
            raise F2ArtifactProductionError(
                "rank certificates must be direct siblings of merged output"
            )
        try:
            _, certificate_sha = read_bytes_with_sidecar(rank_path)
        except F2ProductionRunnerError as error:
            raise F2ArtifactProductionError("rank certificate sidecar failed") from error
        item = read_rank_artifact_certificate(
            rank_path,
            bindings=bindings,
            causal_manifest=causal_manifest,
            qualification_held_out_rows=qualification_held_out_rows,
        )
        certificates.append(item)
        references.append(
            RankCertificateReference(
                rank=item.rank,
                file_name=rank_path.name,
                file_sha256=certificate_sha,
                artifact_root_sha256=item.descriptor.root_sha256,
                coverage_sha256=item.descriptor.coverage_sha256,
            )
        )
    owner_by_log = _validate_rank_set(
        certificates=certificates,
        causal_manifest=causal_manifest,
        world_size=world_size,
        bindings_sha256=_bindings_sha256(bindings),
    )
    full_coverage = build_artifact_coverage(
        CoverageEntry(row.log_name, row.time_us, row.token)
        for row in causal_manifest.rows
    )
    grouped = rows_by_log(causal_manifest.rows)
    try:
        with ExitStack() as stack:
            readers = {
                item.rank: stack.enter_context(
                    FrozenArtifactReader.open(
                        item.artifact_path,
                        expected_bindings=bindings,
                        expected_coverage=build_artifact_coverage(
                            CoverageEntry(row.log_name, row.time_us, row.token)
                            for row in item.rows
                        ),
                        mode="stream",
                    )
                )
                for item in certificates
            }

            def copied_records(log_name: str) -> Iterable[FrozenTokenRecord]:
                owner = owner_by_log[log_name]
                expected = tuple(row.token for row in grouped[log_name])
                observed = []
                for record in readers[owner.rank].iter_records(
                    log_name=log_name, batch_records=4
                ):
                    observed.append(record.token)
                    yield _copy_record(record)
                if tuple(observed) != expected:
                    raise F2ArtifactProductionError(
                        f"rank binary log {log_name!r} differs from causal order"
                    )

            chunks = (
                FrozenArtifactChunk(
                    log_name=log_name,
                    records=copied_records(log_name),
                )
                for log_name in causal_manifest.logs
            )
            descriptor = seal_frozen_artifact(
                output_artifact,
                bindings=bindings,
                coverage=full_coverage,
                chunks=chunks,
            )
    except F2ArtifactProductionError:
        raise
    except Exception as error:
        raise F2ArtifactProductionError(
            f"streaming full-artifact reseal failed: {type(error).__name__}: {error}"
        ) from error
    ordered_references = tuple(sorted(references, key=lambda item: item.rank))
    payload = _merged_certificate_payload(
        world_size=world_size,
        bindings=bindings,
        artifact_path=output_artifact,
        descriptor=descriptor,
        shards=ordered_references,
    )
    try:
        write_bytes_no_overwrite(output_certificate, canonical_json_bytes(payload))
    except F2ProductionRunnerError as error:
        raise F2ArtifactProductionError(
            "merged certificate publication failed; full artifact remains uncertified"
        ) from error
    return MergedArtifactCertificate(
        world_size=world_size,
        artifact_path=output_artifact,
        descriptor=descriptor,
        shards=ordered_references,
        bindings_sha256=_bindings_sha256(bindings),
        _seal=_MERGED_CERTIFICATE_SEAL,
    )


def _parse_rank_reference(payload: Any) -> RankCertificateReference:
    value = _exact_keys(
        payload,
        {
            "artifact_root_sha256",
            "coverage_sha256",
            "file_name",
            "file_sha256",
            "rank",
        },
        "rank certificate reference",
    )
    return RankCertificateReference(
        rank=value["rank"],
        file_name=value["file_name"],
        file_sha256=value["file_sha256"],
        artifact_root_sha256=value["artifact_root_sha256"],
        coverage_sha256=value["coverage_sha256"],
    )


def read_merged_artifact_certificate(
    path: Union[Path, str],
    *,
    bindings: ArtifactBindings,
    causal_manifest: CanonicalManifest,
    qualification_held_out_rows: Sequence[Any],
) -> MergedArtifactCertificate:
    """Replay all rank certificates and verify the full binary artifact."""

    certificate = Path(path).resolve(strict=True)
    try:
        encoded, _ = read_bytes_with_sidecar(certificate)
        decoded = parse_canonical_json_bytes(
            encoded, name="frozen merged artifact certificate"
        )
    except F2ProductionRunnerError as error:
        raise F2ArtifactProductionError("merged certificate authentication failed") from error
    document = _exact_keys(
        decoded,
        {
            "artifact",
            "dataset",
            "fatal_count",
            "identity",
            "kind",
            "partition",
            "schema_version",
            "shards",
            "world_size",
        },
        "merged artifact certificate",
    )
    if (
        document["schema_version"] != MERGED_CERTIFICATE_SCHEMA_VERSION
        or document["kind"] != "frozen_merged_artifact"
        or document["dataset"] != DATASET
        or document["partition"] != PARTITION
    ):
        raise F2ArtifactProductionError("merged certificate contract drifted")
    if document["fatal_count"] != 0 or isinstance(document["fatal_count"], bool):
        raise F2ArtifactProductionError("authoritative merged artifact contains a fatal")
    _validate_binding_identity(document["identity"], bindings)
    _, world_size = _rank_world(0, document["world_size"])
    if not isinstance(document["shards"], list):
        raise F2ArtifactProductionError("merged shard references must be an array")
    references = tuple(_parse_rank_reference(item) for item in document["shards"])
    if tuple(item.rank for item in references) != tuple(range(world_size)):
        raise F2ArtifactProductionError("merged rank references are incomplete or unsorted")
    certificates = []
    for reference in references:
        rank_path = certificate.parent / reference.file_name
        try:
            _, digest = read_bytes_with_sidecar(rank_path)
        except F2ProductionRunnerError as error:
            raise F2ArtifactProductionError("referenced rank sidecar failed") from error
        if digest != reference.file_sha256:
            raise F2ArtifactProductionError("referenced rank certificate SHA-256 drifted")
        item = read_rank_artifact_certificate(
            rank_path,
            bindings=bindings,
            causal_manifest=causal_manifest,
            qualification_held_out_rows=qualification_held_out_rows,
        )
        if (
            item.rank != reference.rank
            or item.descriptor.root_sha256 != reference.artifact_root_sha256
            or item.descriptor.coverage_sha256 != reference.coverage_sha256
        ):
            raise F2ArtifactProductionError("referenced rank artifact identity drifted")
        certificates.append(item)
    _validate_rank_set(
        certificates=certificates,
        causal_manifest=causal_manifest,
        world_size=world_size,
        bindings_sha256=_bindings_sha256(bindings),
    )
    full_coverage = build_artifact_coverage(
        CoverageEntry(row.log_name, row.time_us, row.token)
        for row in causal_manifest.rows
    )
    artifact_path, descriptor = _validate_artifact_document(
        document["artifact"],
        certificate_parent=certificate.parent,
        bindings=bindings,
        coverage=full_coverage,
    )
    return MergedArtifactCertificate(
        world_size=world_size,
        artifact_path=artifact_path,
        descriptor=descriptor,
        shards=references,
        bindings_sha256=_bindings_sha256(bindings),
        _seal=_MERGED_CERTIFICATE_SEAL,
    )


def require_metric_cache_inventory_join(
    bindings: ArtifactBindings,
    cache_source: PrunedMetricCacheSource,
    verified_manifest: Any,
) -> None:
    """Close the explicit bindings -> live inventory -> cache-source join."""

    if type(cache_source) is not PrunedMetricCacheSource:
        raise F2ArtifactProductionError(
            "metric-cache inventory join requires the concrete cache source"
        )
    actual = getattr(verified_manifest, "manifest_sha256", None)
    if actual != bindings.metric_cache_manifest_sha256:
        raise F2ArtifactProductionError(
            "live per-file metric-cache manifest differs from artifact bindings"
        )


def _verify_live_from_runtime(
    *,
    bindings: ArtifactBindings,
    runtime_config_path: Path,
    config: ProductionRuntimeConfig,
) -> VerifiedF2Provenance:
    paths = config.path_map
    try:
        live = verify_live_f2_provenance(
            bindings=bindings,
            qualification_manifest_path=paths["qualification_manifest"],
            causal_manifest_path=paths["causal_manifest"],
            qualification_config_path=paths["qualification_config"],
            runtime_config_path=runtime_config_path,
            cache_metadata_path=paths["cache_metadata"],
            metric_cache_manifest_path=paths["metric_cache_manifest"],
            metric_cache_root=paths["metric_cache_root"],
            baseline_checkpoint_path=paths["baseline_checkpoint"],
            vjepa_checkpoint_path=paths["vjepa_checkpoint"],
            qualification_source_manifest_path=paths[
                "qualification_source_manifest"
            ],
            qualification_source_root=paths["qualification_source_root"],
            runtime_source_manifest_path=paths["runtime_source_manifest"],
            runtime_source_root=paths["runtime_source_root"],
        )
    except Exception as error:
        raise F2ArtifactProductionError(
            f"live provenance verification failed: {type(error).__name__}: {error}"
        ) from error
    if live.causal.split != DATASET or live.qualification.dataset != DATASET:
        raise F2ArtifactProductionError("verified provenance is not held-out navtrain")
    return live


def _verify_raw_inputs_from_runtime(
    *,
    live: VerifiedF2Provenance,
    bindings: ArtifactBindings,
    config: ProductionRuntimeConfig,
) -> VerifiedNavtrainRawInputs:
    paths = config.path_map
    try:
        raw_inputs = load_and_verify_raw_input_manifest(
            paths["raw_input_manifest"],
            expected_sha256=bindings.raw_input_manifest_sha256,
            log_root=paths["navsim_log"],
            sensor_root=paths["sensor_blobs"],
            qualification=live.qualification,
            expected_qualification_manifest_sha256=(
                bindings.qualification_manifest_sha256
            ),
        )
        raw_inputs.verify_all(workers=8)
    except Exception as error:
        raise F2ArtifactProductionError(
            f"raw-input verification failed: {type(error).__name__}: {error}"
        ) from error
    if raw_inputs.manifest_sha256 != bindings.raw_input_manifest_sha256:
        raise F2ArtifactProductionError("raw-input manifest identity drifted")
    return raw_inputs


def _authenticated_manifests(
    live: VerifiedF2Provenance, bindings: ArtifactBindings
) -> AuthenticatedHeldOutManifests:
    try:
        manifests = AuthenticatedHeldOutManifests(
            qualification=live.qualification,
            qualification_sha256=bindings.qualification_manifest_sha256,
            causal=live.causal,
            causal_sha256=bindings.causal_manifest_sha256,
        )
        manifests.require_bindings(bindings)
        return manifests
    except F2FrozenRecordBuildError as error:
        raise F2ArtifactProductionError("authenticated manifest join failed") from error


def build_production_rank_artifact(
    *,
    bindings_path: Union[Path, str],
    runtime_config_path: Union[Path, str],
    rank: int,
    world_size: int,
    artifact_path: Union[Path, str],
    certificate_path: Union[Path, str],
    device: Optional[str] = None,
) -> RankArtifactCertificate:
    """Build one shard exclusively through official production adapters."""

    rank, world_size = _rank_world(rank, world_size)
    try:
        runtime_state = initialize_deterministic_runtime(
            logical_rank=rank,
            world_size=world_size,
            explicit_device=device,
        )
    except F2ProductionRunnerError as error:
        raise F2ArtifactProductionError(
            "deterministic CUDA runtime initialization failed"
        ) from error
    try:
        bindings = read_artifact_bindings(bindings_path)
        runtime_path = Path(runtime_config_path).resolve(strict=True)
        config = _parse_runtime_config(
            runtime_path, expected_sha256=bindings.runtime_config_sha256
        )
    except (F2ProductionRunnerError, OSError) as error:
        raise F2ArtifactProductionError("bound runtime configuration failed") from error
    live = _verify_live_from_runtime(
        bindings=bindings, runtime_config_path=runtime_path, config=config
    )
    raw_inputs = _verify_raw_inputs_from_runtime(
        live=live, bindings=bindings, config=config
    )
    manifests = _authenticated_manifests(live, bindings)
    paths = config.path_map
    try:
        feature_build = build_qualified_held_out_navtrain_feature_source(
            live.qualification,
            live_provenance=live,
            navsim_log_path=paths["navsim_log"],
            sensor_blobs_path=paths["sensor_blobs"],
            authenticated_raw_inputs=raw_inputs,
            rank=rank,
            world_size=world_size,
        )
        cache_source = PrunedMetricCacheSource(
            paths["metric_cache_root"],
            metadata_path=paths["cache_metadata"],
            expected_metadata_sha256=bindings.cache_metadata_sha256,
            bindings=bindings,
            content_manifest=live.metric_cache,
        )
        require_metric_cache_inventory_join(
            bindings, cache_source, live.metric_cache
        )
        baseline = build_official_drive_jepa_baseline(
            paths["baseline_checkpoint"],
            expected_checkpoint_sha256=bindings.baseline_checkpoint_sha256,
            repository_root=paths["repository_root"],
            experiment_root=paths["experiment_root"],
            live_provenance=live,
            device=runtime_state.device,
        )
        semantics = DriveJepaNavsimRecordSemantics.official()
        builder = FrozenNavsimRecordBuilder(
            manifests=manifests,
            bindings=bindings,
            live_provenance=live,
            feature_source=feature_build.source,
            metric_cache_source=cache_source,
            restored_baseline=baseline,
            navsim_semantics=semantics,
        )
        plan = build_rank_artifact_plan(
            causal_manifest=live.causal,
            qualification_held_out_rows=live.qualification.held_out.rows,
            rank=rank,
            world_size=world_size,
        )
        if tuple(feature_build.source.tokens) != tuple(row.token for row in plan.rows):
            raise F2ArtifactProductionError(
                "official feature source differs from rank artifact plan"
            )
        missing_cache = sorted(
            set(row.token for row in plan.rows) - set(cache_source.tokens)
        )
        if missing_cache:
            raise F2ArtifactProductionError(
                f"official metric cache misses rank tokens: {missing_cache[:8]}"
            )
        record_factory = getattr(builder, "_build_record", None)
        if not callable(record_factory):
            raise F2ArtifactProductionError(
                "frozen record builder no longer exposes its audited record seam"
            )
        return seal_rank_artifact(
            causal_manifest=live.causal,
            qualification_held_out_rows=live.qualification.held_out.rows,
            bindings=bindings,
            rank=rank,
            world_size=world_size,
            artifact_path=artifact_path,
            certificate_path=certificate_path,
            record_factory=record_factory,
        )
    except F2ArtifactProductionError:
        raise
    except Exception as error:
        raise F2ArtifactProductionError(
            f"fatal production rank build: {type(error).__name__}: {error}"
        ) from error


def merge_production_rank_artifacts(
    *,
    bindings_path: Union[Path, str],
    runtime_config_path: Union[Path, str],
    world_size: int,
    rank_certificate_paths: Sequence[Union[Path, str]],
    artifact_path: Union[Path, str],
    certificate_path: Union[Path, str],
) -> MergedArtifactCertificate:
    """Authenticate live manifests and re-seal the configured full artifact."""

    _, world_size = _rank_world(0, world_size)
    try:
        bindings = read_artifact_bindings(bindings_path)
        runtime_path = Path(runtime_config_path).resolve(strict=True)
        config = _parse_runtime_config(
            runtime_path, expected_sha256=bindings.runtime_config_sha256
        )
    except (F2ProductionRunnerError, OSError) as error:
        raise F2ArtifactProductionError("bound runtime configuration failed") from error
    live = _verify_live_from_runtime(
        bindings=bindings, runtime_config_path=runtime_path, config=config
    )
    _verify_raw_inputs_from_runtime(
        live=live, bindings=bindings, config=config
    )
    configured_artifact = config.path_map["artifact"].resolve()
    requested_artifact = Path(artifact_path).resolve()
    if requested_artifact != configured_artifact:
        raise F2ArtifactProductionError(
            "merged artifact destination differs from bound runtime config"
        )
    return merge_rank_artifacts(
        causal_manifest=live.causal,
        qualification_held_out_rows=live.qualification.held_out.rows,
        bindings=bindings,
        world_size=world_size,
        rank_certificate_paths=rank_certificate_paths,
        artifact_path=artifact_path,
        certificate_path=certificate_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and merge held-out navtrain F2 frozen artifacts"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-shard")
    build.add_argument("--bindings", type=Path, required=True)
    build.add_argument("--runtime-config", type=Path, required=True)
    build.add_argument("--rank", type=int, required=True)
    build.add_argument("--world-size", type=int, required=True)
    build.add_argument(
        "--device",
        help="physical CUDA device (cuda:<index>); otherwise LOCAL_RANK is required",
    )
    build.add_argument("--artifact-output", type=Path, required=True)
    build.add_argument("--certificate-output", type=Path, required=True)

    merge = commands.add_parser("merge")
    merge.add_argument("--bindings", type=Path, required=True)
    merge.add_argument("--runtime-config", type=Path, required=True)
    merge.add_argument("--world-size", type=int, required=True)
    merge.add_argument("--rank-certificate", type=Path, action="append", required=True)
    merge.add_argument("--artifact-output", type=Path, required=True)
    merge.add_argument("--certificate-output", type=Path, required=True)
    return parser


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-shard":
            build_production_rank_artifact(
                bindings_path=args.bindings,
                runtime_config_path=args.runtime_config,
                rank=args.rank,
                world_size=args.world_size,
                artifact_path=args.artifact_output,
                certificate_path=args.certificate_output,
                device=args.device,
            )
        elif args.command == "merge":
            merge_production_rank_artifacts(
                bindings_path=args.bindings,
                runtime_config_path=args.runtime_config,
                world_size=args.world_size,
                rank_certificate_paths=args.rank_certificate,
                artifact_path=args.artifact_output,
                certificate_path=args.certificate_output,
            )
        else:  # pragma: no cover
            raise F2ArtifactProductionError("unknown artifact production command")
    except F2ArtifactProductionError as error:
        parser.exit(2, f"F2 artifact production failed: {error}\n")
    return 0


__all__ = (
    "F2ArtifactProductionError",
    "MERGED_CERTIFICATE_SCHEMA_VERSION",
    "MergedArtifactCertificate",
    "RANK_CERTIFICATE_SCHEMA_VERSION",
    "RankArtifactCertificate",
    "RankArtifactPlan",
    "RankCertificateReference",
    "SUPPORTED_WORLD_SIZES",
    "build_production_rank_artifact",
    "build_rank_artifact_plan",
    "cli_main",
    "merge_production_rank_artifacts",
    "merge_rank_artifacts",
    "read_merged_artifact_certificate",
    "read_rank_artifact_certificate",
    "require_metric_cache_inventory_join",
    "seal_rank_artifact",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
