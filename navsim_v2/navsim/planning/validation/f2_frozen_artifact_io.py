"""Fail-closed binary I/O for the preregistered F2 frozen artifact.

The real F2 artifact is intentionally a plain, fixed-layout binary file.  It
does not use pickle, NPZ, object arrays, or implicit dtype conversion.  Each
record contains exactly one token's immutable simulator/scorer inputs:

* token SHA-256, timestamp, and proposal SHA-256;
* 32 float32 Drive-JEPA base logits;
* 32 x 9 float32 target scores;
* 32 x 41 x 11 float64 calibration states; and
* 32 x 41 x 11 float64 causal carry states.

Records are grouped into lexicographically ordered per-log chunks.  Within a
chunk they must arrive in strict ``(time_us, token)`` order.  The canonical
header binds the held-out manifest, baseline checkpoint, cache metadata, run
configuration, source tree, proposal schema, simulator, scorer, channel order,
coverage, and every chunk digest.  A root digest binds that header and the
wire-format preamble.

Writers create ``<destination>.partial``, fsync it, verify every byte and every
coverage row through the reader-side verifier, and only then publish the final
name with an atomic hard-link operation.  A failed build deliberately leaves
the unsealed ``.partial`` file for diagnosis and never publishes the target.

This module is only the storage foundation.  It does not build records from
NAVSIM, run a simulator/scorer/model, or assert that a real artifact exists.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np


ARTIFACT_SCHEMA_VERSION = "actworld-f2-frozen-artifact-v3"
COVERAGE_SCHEMA_VERSION = "actworld-f2-frozen-coverage-v1"
F2_VJEPA_CHECKPOINT_SHA256 = (
    "4649182770ef68f84a001780c6579435345948fd80f186fa3616ab078ced668f"
)
CANDIDATE_COUNT = 32
TARGET_CHANNEL_ORDER = (
    "NC",
    "DAC",
    "TLC",
    "DDC",
    "ego_progress",
    "TTC",
    "lane_keeping",
    "history_comfort",
    "aggregate",
)
TARGET_CHANNEL_COUNT = len(TARGET_CHANNEL_ORDER)
STATE_TIME_STEPS = 41
STATE_DIMENSION = 11

_PREAMBLE_MAGIC = b"AWF2BIN\0"
_HEADER_MAGIC = b"AWF2HDR\0"
_TRAILER_MAGIC = b"AWF2END\0"
_FORMAT_VERSION = 3
_ROOT_DOMAIN = b"ActWorld-JEPA/F2/frozen-artifact/root/v3\0"
_PREAMBLE = struct.Struct("<8sII")
_TRAILER = struct.Struct("<8sQQQ32s8s")
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_VERIFY_BATCH_RECORDS = 8


# Explicit offsets are part of the format contract.  ``align=False`` prevents
# NumPy from inserting platform-dependent padding.
RECORD_DTYPE = np.dtype(
    [
        ("token_sha256", "u1", (32,)),
        ("time_us", "<u8"),
        ("proposal_sha256", "u1", (32,)),
        ("base", "<f4", (CANDIDATE_COUNT,)),
        (
            "targets",
            "<f4",
            (CANDIDATE_COUNT, TARGET_CHANNEL_COUNT),
        ),
        (
            "calibration",
            "<f8",
            (CANDIDATE_COUNT, STATE_TIME_STEPS, STATE_DIMENSION),
        ),
        (
            "carry",
            "<f8",
            (CANDIDATE_COUNT, STATE_TIME_STEPS, STATE_DIMENSION),
        ),
    ],
    align=False,
)
RECORD_SIZE = RECORD_DTYPE.itemsize

_RECORD_LAYOUT_PAYLOAD = {
    "byte_order": "little",
    "fields": [
        {"dtype": "u1", "name": "token_sha256", "shape": [32]},
        {"dtype": "u8", "name": "time_us", "shape": []},
        {"dtype": "u1", "name": "proposal_sha256", "shape": [32]},
        {"dtype": "f4", "name": "base", "shape": [32]},
        {"dtype": "f4", "name": "targets", "shape": [32, 9]},
        {
            "dtype": "f8",
            "name": "calibration",
            "shape": [32, 41, 11],
        },
        {"dtype": "f8", "name": "carry", "shape": [32, 41, 11]},
    ],
    "itemsize": RECORD_SIZE,
}


class FrozenArtifactIOError(ValueError):
    """Raised whenever the immutable F2 artifact contract is violated."""


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrozenArtifactIOError(f"{name} must be a non-empty string")
    return value


def _sha256_hex(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FrozenArtifactIOError(f"{name} must be a 64-character SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise FrozenArtifactIOError(
            f"{name} must use lowercase hexadecimal"
        )
    return value


def _timestamp(value: Any, name: str = "time_us") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenArtifactIOError(f"{name} must be an integer")
    if value < 0 or value > np.iinfo(np.uint64).max:
        raise FrozenArtifactIOError(f"{name} is outside the uint64 range")
    return value


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identifier_digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _reject_json_constant(value: str) -> None:
    raise FrozenArtifactIOError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FrozenArtifactIOError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_canonical_json(payload: bytes) -> Mapping[str, Any]:
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except FrozenArtifactIOError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrozenArtifactIOError("artifact header is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise FrozenArtifactIOError("artifact header must be a JSON object")
    if _canonical_json_bytes(decoded) != payload:
        raise FrozenArtifactIOError("artifact header is not canonical JSON")
    return decoded


def _require_exact_keys(payload: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    actual = set(payload)
    expected = set(keys)
    if actual != expected:
        raise FrozenArtifactIOError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class ArtifactBindings:
    """External identities that the artifact root must bind."""

    qualification_manifest_sha256: str
    causal_manifest_sha256: str
    baseline_checkpoint_sha256: str
    vjepa_checkpoint_sha256: str
    cache_metadata_sha256: str
    metric_cache_manifest_sha256: str
    raw_input_manifest_sha256: str
    qualification_config_sha256: str
    runtime_config_sha256: str
    qualification_source_manifest_sha256: str
    runtime_source_manifest_sha256: str
    source_revision: str
    proposal_schema_name: str
    proposal_schema_sha256: str
    simulator_name: str
    simulator_sha256: str
    scorer_name: str
    scorer_sha256: str
    channel_order: Tuple[str, ...] = TARGET_CHANNEL_ORDER

    def __post_init__(self) -> None:
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
            object.__setattr__(
                self,
                field_name,
                _sha256_hex(getattr(self, field_name), field_name),
            )
        for field_name in (
            "source_revision",
            "proposal_schema_name",
            "simulator_name",
            "scorer_name",
        ):
            _nonempty(getattr(self, field_name), field_name)
        normalized_channels = tuple(
            _nonempty(name, "channel name") for name in self.channel_order
        )
        if normalized_channels != TARGET_CHANNEL_ORDER:
            raise FrozenArtifactIOError(
                "channel_order must exactly match the preregistered F2 target order"
            )
        object.__setattr__(self, "channel_order", normalized_channels)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "baseline_checkpoint_sha256": self.baseline_checkpoint_sha256,
            "vjepa_checkpoint_sha256": self.vjepa_checkpoint_sha256,
            "cache_metadata_sha256": self.cache_metadata_sha256,
            "metric_cache_manifest_sha256": self.metric_cache_manifest_sha256,
            "raw_input_manifest_sha256": self.raw_input_manifest_sha256,
            "channel_order": list(self.channel_order),
            "manifests": {
                "causal_sha256": self.causal_manifest_sha256,
                "qualification_sha256": self.qualification_manifest_sha256,
            },
            "proposal_schema": {
                "candidate_count": CANDIDATE_COUNT,
                "name": self.proposal_schema_name,
                "sha256": self.proposal_schema_sha256,
            },
            "scorer": {
                "name": self.scorer_name,
                "sha256": self.scorer_sha256,
            },
            "simulator": {
                "name": self.simulator_name,
                "sha256": self.simulator_sha256,
            },
            "source": {
                "qualification_config_sha256": self.qualification_config_sha256,
                "qualification_source_manifest_sha256": (
                    self.qualification_source_manifest_sha256
                ),
                "revision": self.source_revision,
                "runtime_config_sha256": self.runtime_config_sha256,
                "runtime_manifest_sha256": self.runtime_source_manifest_sha256,
            },
        }


@dataclass(frozen=True)
class CoverageEntry:
    """One expected manifest row used for exact artifact coverage checks."""

    log_name: str
    time_us: int
    token: str

    def __post_init__(self) -> None:
        _nonempty(self.log_name, "log_name")
        _timestamp(self.time_us)
        _nonempty(self.token, "token")

    @property
    def token_sha256(self) -> str:
        return _identifier_digest(self.token).hex()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "log_name": self.log_name,
            "time_us": self.time_us,
            "token": self.token,
        }


@dataclass(frozen=True)
class FrozenArtifactCoverage:
    """Canonical, unique token coverage derived from the bound manifest."""

    entries: Tuple[CoverageEntry, ...]

    def __post_init__(self) -> None:
        converted = tuple(self.entries)
        if not converted:
            raise FrozenArtifactIOError("coverage must contain at least one token")
        for entry in converted:
            if not isinstance(entry, CoverageEntry):
                raise FrozenArtifactIOError("coverage entries must be CoverageEntry values")
        canonical = tuple(
            sorted(converted, key=lambda row: (row.log_name, row.time_us, row.token))
        )
        tokens = [row.token for row in canonical]
        if len(tokens) != len(set(tokens)):
            raise FrozenArtifactIOError("coverage contains duplicate tokens")
        digests = [row.token_sha256 for row in canonical]
        if len(digests) != len(set(digests)):
            raise FrozenArtifactIOError("coverage contains a token SHA-256 collision")
        object.__setattr__(self, "entries", canonical)

    @property
    def sha256(self) -> str:
        return _digest(canonical_coverage_bytes(self))

    @property
    def log_names(self) -> Tuple[str, ...]:
        return tuple(sorted({entry.log_name for entry in self.entries}))

    def by_log(self) -> Dict[str, Tuple[CoverageEntry, ...]]:
        rows: Dict[str, list[CoverageEntry]] = {name: [] for name in self.log_names}
        for entry in self.entries:
            rows[entry.log_name].append(entry)
        return {name: tuple(values) for name, values in rows.items()}


def canonical_coverage_bytes(coverage: FrozenArtifactCoverage) -> bytes:
    if not isinstance(coverage, FrozenArtifactCoverage):
        raise FrozenArtifactIOError("coverage must be FrozenArtifactCoverage")
    return _canonical_json_bytes(
        {
            "rows": [entry.to_payload() for entry in coverage.entries],
            "schema_version": COVERAGE_SCHEMA_VERSION,
        }
    )


def build_artifact_coverage(entries: Iterable[CoverageEntry]) -> FrozenArtifactCoverage:
    return FrozenArtifactCoverage(entries=tuple(entries))


@dataclass(frozen=True)
class FrozenTokenRecord:
    """One in-memory fixed-shape record before canonical binary encoding."""

    token: str
    time_us: int
    proposal_sha256: str
    base: np.ndarray
    targets: np.ndarray
    calibration: np.ndarray
    carry: np.ndarray


@dataclass(frozen=True)
class FrozenArtifactChunk:
    """A caller-supplied, already sorted stream of records for one log."""

    log_name: str
    records: Iterable[FrozenTokenRecord]


@dataclass(frozen=True)
class ArtifactChunkDescriptor:
    log_name: str
    offset: int
    byte_size: int
    record_count: int
    sha256: str


@dataclass(frozen=True)
class ArtifactDescriptor:
    path: Path
    root_sha256: str
    file_size: int
    header_offset: int
    header_size: int
    coverage_sha256: str
    record_count: int
    chunks: Tuple[ArtifactChunkDescriptor, ...]


@dataclass(frozen=True)
class ArtifactBatch:
    """A read-only structured-array batch from one log chunk."""

    log_name: str
    records: np.ndarray


@dataclass(frozen=True)
class ReadTokenRecord:
    """A zero-copy record view returned by :class:`FrozenArtifactReader`."""

    log_name: str
    token: str
    token_sha256: str
    time_us: int
    proposal_sha256: str
    base: np.ndarray
    targets: np.ndarray
    calibration: np.ndarray
    carry: np.ndarray


def _require_exact_array(
    value: Any,
    *,
    dtype: np.dtype,
    shape: Tuple[int, ...],
    name: str,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise FrozenArtifactIOError(f"{name} must be a numpy.ndarray")
    if value.dtype != dtype or value.dtype.str != dtype.str:
        raise FrozenArtifactIOError(
            f"{name} dtype must be exactly {dtype.str}, got {value.dtype.str}"
        )
    if value.shape != shape:
        raise FrozenArtifactIOError(
            f"{name} shape must be exactly {shape}, got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise FrozenArtifactIOError(f"{name} must be C-contiguous")
    if not np.isfinite(value).all():
        raise FrozenArtifactIOError(f"{name} contains NaN or infinity")
    return value


def _encode_record(record: FrozenTokenRecord) -> Tuple[bytes, Tuple[int, str]]:
    if not isinstance(record, FrozenTokenRecord):
        raise FrozenArtifactIOError("chunk records must be FrozenTokenRecord values")
    token = _nonempty(record.token, "token")
    time_us = _timestamp(record.time_us)
    proposal_sha = bytes.fromhex(_sha256_hex(record.proposal_sha256, "proposal_sha256"))
    base = _require_exact_array(
        record.base,
        dtype=np.dtype("<f4"),
        shape=(CANDIDATE_COUNT,),
        name="base",
    )
    targets = _require_exact_array(
        record.targets,
        dtype=np.dtype("<f4"),
        shape=(CANDIDATE_COUNT, TARGET_CHANNEL_COUNT),
        name="targets",
    )
    if np.any(targets < np.float32(0.0)) or np.any(targets > np.float32(1.0)):
        raise FrozenArtifactIOError("targets must lie in the closed interval [0, 1]")
    calibration = _require_exact_array(
        record.calibration,
        dtype=np.dtype("<f8"),
        shape=(CANDIDATE_COUNT, STATE_TIME_STEPS, STATE_DIMENSION),
        name="calibration",
    )
    carry = _require_exact_array(
        record.carry,
        dtype=np.dtype("<f8"),
        shape=(CANDIDATE_COUNT, STATE_TIME_STEPS, STATE_DIMENSION),
        name="carry",
    )

    encoded = np.empty((1,), dtype=RECORD_DTYPE)
    encoded["token_sha256"][0] = np.frombuffer(_identifier_digest(token), dtype=np.uint8)
    encoded["time_us"][0] = time_us
    encoded["proposal_sha256"][0] = np.frombuffer(proposal_sha, dtype=np.uint8)
    encoded["base"][0] = base
    encoded["targets"][0] = targets
    encoded["calibration"][0] = calibration
    encoded["carry"][0] = carry
    payload = encoded.tobytes(order="C")
    if len(payload) != RECORD_SIZE:
        raise FrozenArtifactIOError("internal record encoder produced the wrong byte size")
    return payload, (time_us, token)


def _write_all(stream: BinaryIO, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise FrozenArtifactIOError("short write while building artifact")
        view = view[written:]


def _read_exact(stream: BinaryIO, size: int, name: str) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise FrozenArtifactIOError(
            f"short read for {name}: expected {size} bytes, got {len(payload)}"
        )
    return payload


def _root_digest(header_bytes: bytes) -> bytes:
    return hashlib.sha256(
        _ROOT_DOMAIN
        + _PREAMBLE.pack(_PREAMBLE_MAGIC, _FORMAT_VERSION, RECORD_SIZE)
        + header_bytes
    ).digest()


def _chunk_payload(descriptor: ArtifactChunkDescriptor) -> Dict[str, Any]:
    return {
        "byte_size": descriptor.byte_size,
        "log_name": descriptor.log_name,
        "offset": descriptor.offset,
        "record_count": descriptor.record_count,
        "sha256": descriptor.sha256,
    }


def _header_payload(
    bindings: ArtifactBindings,
    coverage: FrozenArtifactCoverage,
    chunks: Sequence[ArtifactChunkDescriptor],
) -> Dict[str, Any]:
    return {
        "bindings": bindings.to_payload(),
        "chunks": [_chunk_payload(chunk) for chunk in chunks],
        "coverage": {
            "log_count": len(coverage.log_names),
            "sha256": coverage.sha256,
            "token_count": len(coverage.entries),
        },
        "record_layout": _RECORD_LAYOUT_PAYLOAD,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
    }


def _fsync_parent(path: Path) -> None:
    """Best-effort directory fsync; unsupported platforms remain file-safe."""

    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(str(path.parent), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def seal_frozen_artifact(
    destination: os.PathLike[str] | str,
    *,
    bindings: ArtifactBindings,
    coverage: FrozenArtifactCoverage,
    chunks: Iterable[FrozenArtifactChunk],
) -> ArtifactDescriptor:
    """Write, fully verify, and atomically publish one frozen artifact.

    The caller must supply chunks in lexicographic log order and records in
    strict ``(time_us, token)`` order.  Both sequences must exactly equal the
    supplied manifest-derived coverage.  Existing targets and partials are
    never overwritten.
    """

    if not isinstance(bindings, ArtifactBindings):
        raise FrozenArtifactIOError("bindings must be ArtifactBindings")
    if not isinstance(coverage, FrozenArtifactCoverage):
        raise FrozenArtifactIOError("coverage must be FrozenArtifactCoverage")
    target = Path(destination)
    partial = target.with_name(target.name + ".partial")
    if not target.parent.is_dir():
        raise FrozenArtifactIOError("destination parent directory does not exist")
    if target.exists():
        raise FrozenArtifactIOError("destination already exists; overwrite is forbidden")
    if partial.exists():
        raise FrozenArtifactIOError("partial artifact already exists; overwrite is forbidden")

    expected_by_log = coverage.by_log()
    expected_logs = coverage.log_names
    descriptors: list[ArtifactChunkDescriptor] = []

    # Exclusive creation is important: a stale or concurrently produced
    # partial can never be mistaken for this build.
    with partial.open("xb") as stream:
        _write_all(
            stream,
            _PREAMBLE.pack(_PREAMBLE_MAGIC, _FORMAT_VERSION, RECORD_SIZE),
        )
        observed_log_count = 0
        for log_index, chunk in enumerate(chunks):
            if not isinstance(chunk, FrozenArtifactChunk):
                raise FrozenArtifactIOError("chunks must be FrozenArtifactChunk values")
            if log_index >= len(expected_logs):
                raise FrozenArtifactIOError("artifact contains an unexpected extra log chunk")
            log_name = _nonempty(chunk.log_name, "chunk log_name")
            expected_log = expected_logs[log_index]
            if log_name != expected_log:
                raise FrozenArtifactIOError(
                    f"chunk log order/coverage mismatch: expected {expected_log!r}, "
                    f"got {log_name!r}"
                )
            observed_log_count += 1
            expected_rows = expected_by_log[log_name]
            offset = stream.tell()
            chunk_hasher = hashlib.sha256()
            record_count = 0
            previous_key: Optional[Tuple[int, str]] = None
            for record in chunk.records:
                if record_count >= len(expected_rows):
                    raise FrozenArtifactIOError(
                        f"chunk {log_name!r} contains an unexpected extra record"
                    )
                payload, sort_key = _encode_record(record)
                if previous_key is not None and sort_key <= previous_key:
                    raise FrozenArtifactIOError(
                        f"chunk {log_name!r} is not in strict (time_us, token) order"
                    )
                expected = expected_rows[record_count]
                if sort_key != (expected.time_us, expected.token):
                    raise FrozenArtifactIOError(
                        f"chunk {log_name!r} coverage mismatch at record {record_count}: "
                        f"expected {(expected.time_us, expected.token)!r}, got {sort_key!r}"
                    )
                _write_all(stream, payload)
                chunk_hasher.update(payload)
                previous_key = sort_key
                record_count += 1
            if record_count != len(expected_rows):
                raise FrozenArtifactIOError(
                    f"chunk {log_name!r} is incomplete: expected {len(expected_rows)} "
                    f"records, got {record_count}"
                )
            byte_size = record_count * RECORD_SIZE
            if stream.tell() != offset + byte_size:
                raise FrozenArtifactIOError("internal chunk byte-size accounting failed")
            descriptors.append(
                ArtifactChunkDescriptor(
                    log_name=log_name,
                    offset=offset,
                    byte_size=byte_size,
                    record_count=record_count,
                    sha256=chunk_hasher.hexdigest(),
                )
            )
        if observed_log_count != len(expected_logs):
            raise FrozenArtifactIOError(
                f"artifact is missing log chunks: expected {len(expected_logs)}, "
                f"got {observed_log_count}"
            )

        header_offset = stream.tell()
        header_bytes = _canonical_json_bytes(
            _header_payload(bindings, coverage, descriptors)
        )
        if len(header_bytes) > _MAX_HEADER_BYTES:
            raise FrozenArtifactIOError("artifact header exceeds the safety limit")
        _write_all(stream, header_bytes)
        root = _root_digest(header_bytes)
        file_size = stream.tell() + _TRAILER.size
        _write_all(
            stream,
            _TRAILER.pack(
                _HEADER_MAGIC,
                header_offset,
                len(header_bytes),
                file_size,
                root,
                _TRAILER_MAGIC,
            ),
        )
        stream.flush()
        os.fsync(stream.fileno())

    # Sealing is deliberately a complete second pass: this detects short
    # writes, dtype/layout drift, non-finite data, coverage drift, and digest
    # errors before the final path can become visible.
    descriptor = verify_frozen_artifact(
        partial,
        expected_bindings=bindings,
        expected_coverage=coverage,
    )

    try:
        # The final name appears atomically only after verification.  Unlike
        # os.replace, hard-link publication also refuses a raced destination.
        os.link(partial, target)
    except FileExistsError as error:
        raise FrozenArtifactIOError(
            "destination appeared during seal; overwrite is forbidden"
        ) from error
    except OSError as error:
        raise FrozenArtifactIOError(
            "atomic hard-link publication failed; partial remains unsealed"
        ) from error
    _fsync_parent(target)
    partial.unlink()
    _fsync_parent(target)

    return ArtifactDescriptor(
        path=target,
        root_sha256=descriptor.root_sha256,
        file_size=descriptor.file_size,
        header_offset=descriptor.header_offset,
        header_size=descriptor.header_size,
        coverage_sha256=descriptor.coverage_sha256,
        record_count=descriptor.record_count,
        chunks=descriptor.chunks,
    )


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenArtifactIOError(f"{name} must be an integer")
    if value < 0:
        raise FrozenArtifactIOError(f"{name} must be nonnegative")
    return value


def _parse_chunk_descriptors(
    raw_chunks: Any,
    *,
    expected_coverage: FrozenArtifactCoverage,
    header_offset: int,
) -> Tuple[ArtifactChunkDescriptor, ...]:
    if not isinstance(raw_chunks, list):
        raise FrozenArtifactIOError("header chunks must be a list")
    expected_logs = expected_coverage.log_names
    expected_by_log = expected_coverage.by_log()
    if len(raw_chunks) != len(expected_logs):
        raise FrozenArtifactIOError("header chunk count differs from coverage")
    descriptors = []
    next_offset = _PREAMBLE.size
    for index, raw in enumerate(raw_chunks):
        if not isinstance(raw, dict):
            raise FrozenArtifactIOError("each header chunk must be an object")
        _require_exact_keys(
            raw,
            ("byte_size", "log_name", "offset", "record_count", "sha256"),
            "chunk",
        )
        log_name = _nonempty(raw["log_name"], "chunk log_name")
        if log_name != expected_logs[index]:
            raise FrozenArtifactIOError("header chunks are not in canonical log order")
        offset = _integer(raw["offset"], "chunk offset")
        byte_size = _integer(raw["byte_size"], "chunk byte_size")
        record_count = _integer(raw["record_count"], "chunk record_count")
        sha256 = _sha256_hex(raw["sha256"], "chunk sha256")
        if offset != next_offset:
            raise FrozenArtifactIOError("chunk offsets contain a gap or overlap")
        if record_count != len(expected_by_log[log_name]):
            raise FrozenArtifactIOError("chunk record count differs from coverage")
        if byte_size != record_count * RECORD_SIZE:
            raise FrozenArtifactIOError("chunk byte size differs from fixed record layout")
        descriptor = ArtifactChunkDescriptor(
            log_name=log_name,
            offset=offset,
            byte_size=byte_size,
            record_count=record_count,
            sha256=sha256,
        )
        descriptors.append(descriptor)
        next_offset += byte_size
    if next_offset != header_offset:
        raise FrozenArtifactIOError("chunk region does not end at the header offset")
    return tuple(descriptors)


def _validate_wire_batch(array: np.ndarray, name: str) -> None:
    if array.dtype != RECORD_DTYPE:
        raise FrozenArtifactIOError(f"{name} decoded with an unexpected dtype")
    expected_shapes = {
        "token_sha256": (len(array), 32),
        "time_us": (len(array),),
        "proposal_sha256": (len(array), 32),
        "base": (len(array), 32),
        "targets": (len(array), 32, 9),
        "calibration": (len(array), 32, 41, 11),
        "carry": (len(array), 32, 41, 11),
    }
    for field, expected_shape in expected_shapes.items():
        if array[field].shape != expected_shape:
            raise FrozenArtifactIOError(
                f"{name} field {field!r} has shape {array[field].shape}, "
                f"expected {expected_shape}"
            )
    for field in ("base", "targets", "calibration", "carry"):
        if not np.isfinite(array[field]).all():
            raise FrozenArtifactIOError(f"{name} field {field!r} is non-finite")
    if np.any(array["targets"] < np.float32(0.0)) or np.any(
        array["targets"] > np.float32(1.0)
    ):
        raise FrozenArtifactIOError(f"{name} targets are outside [0, 1]")


def _verify_open_file(
    stream: BinaryIO,
    path: Path,
    *,
    expected_bindings: ArtifactBindings,
    expected_coverage: FrozenArtifactCoverage,
) -> ArtifactDescriptor:
    if not isinstance(expected_bindings, ArtifactBindings):
        raise FrozenArtifactIOError("expected_bindings must be ArtifactBindings")
    if not isinstance(expected_coverage, FrozenArtifactCoverage):
        raise FrozenArtifactIOError("expected_coverage must be FrozenArtifactCoverage")

    stream.seek(0, os.SEEK_END)
    actual_size = stream.tell()
    minimum_size = _PREAMBLE.size + 2 + _TRAILER.size
    if actual_size < minimum_size:
        raise FrozenArtifactIOError("artifact is too small to contain its framing")
    stream.seek(actual_size - _TRAILER.size)
    trailer = _read_exact(stream, _TRAILER.size, "trailer")
    (
        header_magic,
        header_offset,
        header_size,
        declared_size,
        stored_root,
        trailer_magic,
    ) = _TRAILER.unpack(trailer)
    if header_magic != _HEADER_MAGIC or trailer_magic != _TRAILER_MAGIC:
        raise FrozenArtifactIOError("artifact trailer magic is invalid")
    if declared_size != actual_size:
        raise FrozenArtifactIOError(
            f"artifact exact byte size mismatch: declared {declared_size}, "
            f"actual {actual_size}"
        )
    if header_size <= 0 or header_size > _MAX_HEADER_BYTES:
        raise FrozenArtifactIOError("artifact header size is invalid")
    if header_offset < _PREAMBLE.size:
        raise FrozenArtifactIOError("artifact header overlaps the preamble")
    if header_offset + header_size + _TRAILER.size != actual_size:
        raise FrozenArtifactIOError("header/trailer offsets do not cover the exact file")

    stream.seek(0)
    preamble = _read_exact(stream, _PREAMBLE.size, "preamble")
    magic, version, itemsize = _PREAMBLE.unpack(preamble)
    if magic != _PREAMBLE_MAGIC:
        raise FrozenArtifactIOError("artifact preamble magic is invalid")
    if version != _FORMAT_VERSION:
        raise FrozenArtifactIOError("artifact format version is unsupported")
    if itemsize != RECORD_SIZE:
        raise FrozenArtifactIOError("artifact record itemsize differs from the fixed dtype")

    stream.seek(header_offset)
    header_bytes = _read_exact(stream, header_size, "header")
    calculated_root = _root_digest(header_bytes)
    if calculated_root != stored_root:
        raise FrozenArtifactIOError("artifact root SHA-256 mismatch")
    header = _parse_canonical_json(header_bytes)
    _require_exact_keys(
        header,
        ("bindings", "chunks", "coverage", "record_layout", "schema_version"),
        "header",
    )
    if header["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise FrozenArtifactIOError("artifact schema version is unsupported")
    if header["record_layout"] != _RECORD_LAYOUT_PAYLOAD:
        raise FrozenArtifactIOError("artifact record layout is not the frozen layout")
    if header["bindings"] != expected_bindings.to_payload():
        raise FrozenArtifactIOError("artifact external bindings do not match expectations")

    raw_coverage = header["coverage"]
    if not isinstance(raw_coverage, dict):
        raise FrozenArtifactIOError("header coverage must be an object")
    _require_exact_keys(
        raw_coverage,
        ("log_count", "sha256", "token_count"),
        "coverage",
    )
    if _sha256_hex(raw_coverage["sha256"], "coverage sha256") != expected_coverage.sha256:
        raise FrozenArtifactIOError("artifact coverage SHA-256 differs from expectations")
    if _integer(raw_coverage["token_count"], "coverage token_count") != len(
        expected_coverage.entries
    ):
        raise FrozenArtifactIOError("artifact token count differs from coverage")
    if _integer(raw_coverage["log_count"], "coverage log_count") != len(
        expected_coverage.log_names
    ):
        raise FrozenArtifactIOError("artifact log count differs from coverage")

    descriptors = _parse_chunk_descriptors(
        header["chunks"],
        expected_coverage=expected_coverage,
        header_offset=header_offset,
    )
    expected_by_log = expected_coverage.by_log()
    observed_total = 0
    for chunk in descriptors:
        expected_rows = expected_by_log[chunk.log_name]
        chunk_hasher = hashlib.sha256()
        stream.seek(chunk.offset)
        consumed = 0
        while consumed < chunk.record_count:
            batch_count = min(_VERIFY_BATCH_RECORDS, chunk.record_count - consumed)
            raw = _read_exact(
                stream,
                batch_count * RECORD_SIZE,
                f"chunk {chunk.log_name!r} batch",
            )
            chunk_hasher.update(raw)
            batch = np.frombuffer(raw, dtype=RECORD_DTYPE, count=batch_count)
            _validate_wire_batch(batch, f"chunk {chunk.log_name!r}")
            for local_index in range(batch_count):
                expected = expected_rows[consumed + local_index]
                actual_time = int(batch["time_us"][local_index])
                actual_token_sha = bytes(batch["token_sha256"][local_index]).hex()
                if actual_time != expected.time_us:
                    raise FrozenArtifactIOError(
                        f"chunk {chunk.log_name!r} timestamp differs from coverage"
                    )
                if actual_token_sha != expected.token_sha256:
                    raise FrozenArtifactIOError(
                        f"chunk {chunk.log_name!r} token SHA-256 differs from coverage"
                    )
            consumed += batch_count
        if consumed * RECORD_SIZE != chunk.byte_size:
            raise FrozenArtifactIOError("verified chunk byte size is inconsistent")
        if chunk_hasher.hexdigest() != chunk.sha256:
            raise FrozenArtifactIOError(
                f"chunk {chunk.log_name!r} SHA-256 mismatch"
            )
        observed_total += consumed
    if observed_total != len(expected_coverage.entries):
        raise FrozenArtifactIOError("verified token coverage is incomplete")

    return ArtifactDescriptor(
        path=path,
        root_sha256=stored_root.hex(),
        file_size=actual_size,
        header_offset=header_offset,
        header_size=header_size,
        coverage_sha256=expected_coverage.sha256,
        record_count=observed_total,
        chunks=descriptors,
    )


def verify_frozen_artifact(
    path: os.PathLike[str] | str,
    *,
    expected_bindings: ArtifactBindings,
    expected_coverage: FrozenArtifactCoverage,
) -> ArtifactDescriptor:
    """Fully verify framing, identities, bytes, numerics, and exact coverage."""

    artifact_path = Path(path)
    try:
        with artifact_path.open("rb") as stream:
            return _verify_open_file(
                stream,
                artifact_path,
                expected_bindings=expected_bindings,
                expected_coverage=expected_coverage,
            )
    except FrozenArtifactIOError:
        raise
    except OSError as error:
        raise FrozenArtifactIOError(f"cannot read artifact {artifact_path}") from error


class FrozenArtifactReader:
    """Verified mmap or bounded-memory streaming reader for a sealed artifact."""

    def __init__(
        self,
        path: Path,
        stream: BinaryIO,
        descriptor: ArtifactDescriptor,
        coverage: FrozenArtifactCoverage,
        mode: str,
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self._stream = stream
        self._mode = mode
        self._closed = False
        self._mapping: Optional[mmap.mmap] = None
        self._token_by_sha = {
            entry.token_sha256: entry.token for entry in coverage.entries
        }
        if mode == "mmap":
            self._mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)

    @classmethod
    def open(
        cls,
        path: os.PathLike[str] | str,
        *,
        expected_bindings: ArtifactBindings,
        expected_coverage: FrozenArtifactCoverage,
        mode: str = "mmap",
    ) -> "FrozenArtifactReader":
        if mode not in ("mmap", "stream"):
            raise FrozenArtifactIOError("reader mode must be 'mmap' or 'stream'")
        artifact_path = Path(path)
        try:
            stream = artifact_path.open("rb")
        except OSError as error:
            raise FrozenArtifactIOError(f"cannot open artifact {artifact_path}") from error
        try:
            descriptor = _verify_open_file(
                stream,
                artifact_path,
                expected_bindings=expected_bindings,
                expected_coverage=expected_coverage,
            )
            return cls(artifact_path, stream, descriptor, expected_coverage, mode)
        except BaseException:
            stream.close()
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise FrozenArtifactIOError("artifact reader is closed")

    def iter_batches(
        self,
        *,
        log_name: Optional[str] = None,
        batch_records: int = 8,
    ) -> Iterator[ArtifactBatch]:
        """Yield read-only structured batches without whole-artifact loading."""

        self._require_open()
        if isinstance(batch_records, bool) or not isinstance(batch_records, int):
            raise FrozenArtifactIOError("batch_records must be an integer")
        if batch_records <= 0:
            raise FrozenArtifactIOError("batch_records must be positive")
        if log_name is not None:
            _nonempty(log_name, "log_name")
            if log_name not in {chunk.log_name for chunk in self.descriptor.chunks}:
                raise FrozenArtifactIOError(f"unknown log chunk {log_name!r}")

        for chunk in self.descriptor.chunks:
            if log_name is not None and chunk.log_name != log_name:
                continue
            consumed = 0
            while consumed < chunk.record_count:
                count = min(batch_records, chunk.record_count - consumed)
                offset = chunk.offset + consumed * RECORD_SIZE
                if self._mode == "mmap":
                    assert self._mapping is not None
                    array = np.frombuffer(
                        self._mapping,
                        dtype=RECORD_DTYPE,
                        count=count,
                        offset=offset,
                    )
                else:
                    self._stream.seek(offset)
                    raw = _read_exact(
                        self._stream,
                        count * RECORD_SIZE,
                        f"stream batch for {chunk.log_name!r}",
                    )
                    array = np.frombuffer(raw, dtype=RECORD_DTYPE, count=count)
                array.flags.writeable = False
                yield ArtifactBatch(log_name=chunk.log_name, records=array)
                consumed += count

    def iter_records(
        self,
        *,
        log_name: Optional[str] = None,
        batch_records: int = 8,
    ) -> Iterator[ReadTokenRecord]:
        """Yield typed zero-copy views and recover token text from coverage."""

        for batch in self.iter_batches(
            log_name=log_name,
            batch_records=batch_records,
        ):
            for index in range(len(batch.records)):
                row = batch.records[index]
                token_sha = bytes(row["token_sha256"]).hex()
                token = self._token_by_sha.get(token_sha)
                if token is None:
                    raise FrozenArtifactIOError(
                        "record token SHA-256 is absent from verified coverage"
                    )
                yield ReadTokenRecord(
                    log_name=batch.log_name,
                    token=token,
                    token_sha256=token_sha,
                    time_us=int(row["time_us"]),
                    proposal_sha256=bytes(row["proposal_sha256"]).hex(),
                    base=row["base"],
                    targets=row["targets"],
                    calibration=row["calibration"],
                    carry=row["carry"],
                )

    def close(self) -> None:
        if self._closed:
            return
        if self._mapping is not None:
            try:
                self._mapping.close()
            except BufferError as error:
                raise FrozenArtifactIOError(
                    "cannot close mmap reader while record views are still alive"
                ) from error
            self._mapping = None
        self._stream.close()
        self._closed = True

    def __enter__(self) -> "FrozenArtifactReader":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactBatch",
    "ArtifactBindings",
    "ArtifactChunkDescriptor",
    "ArtifactDescriptor",
    "CANDIDATE_COUNT",
    "CoverageEntry",
    "FrozenArtifactChunk",
    "FrozenArtifactCoverage",
    "FrozenArtifactIOError",
    "FrozenArtifactReader",
    "FrozenTokenRecord",
    "F2_VJEPA_CHECKPOINT_SHA256",
    "RECORD_DTYPE",
    "RECORD_SIZE",
    "ReadTokenRecord",
    "STATE_DIMENSION",
    "STATE_TIME_STEPS",
    "TARGET_CHANNEL_ORDER",
    "build_artifact_coverage",
    "canonical_coverage_bytes",
    "seal_frozen_artifact",
    "verify_frozen_artifact",
]
