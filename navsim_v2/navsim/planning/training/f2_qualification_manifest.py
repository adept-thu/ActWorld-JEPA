"""Frozen, sensor-qualified navtrain split manifests for F2 training.

This module is intentionally independent from the F2 causal validator.  The
causal manifest describes only held-out validation execution, while this
manifest authenticates the *training input split* before either SceneLoader is
built.  Explicit token rows are authoritative: log names are identities and
I/O hints, never a substitute for token-level partitioning (a log can contain
both retained and rejected training scenes).

The real-data qualification job is responsible for producing the rows, an
independent canonical removed-token/reason report for each partition, and the
provenance digests.  Runtime code only accepts canonical JSON whose whole-file
SHA-256 and embedded config/cache/source SHA-256 values were provided
independently in the resolved F2 configuration.  It does *not* hash the live
source tree or cache contents itself; that external provenance preflight is a
required real-run step, not a property established by this loader.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

from navsim.planning.validation.f2_causal_validator import F2_SENSOR_CONTRACT


QUALIFICATION_SCHEMA_VERSION = "actworld-f2-navtrain-qualification-v1"
QUALIFICATION_DATASET = "navtrain"
QUALIFICATION_PARTITIONS = ("train", "held_out")


class F2QualificationError(ValueError):
    """Raised when an F2 qualification artifact or runtime input drifts."""


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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise F2QualificationError(f"{name} must be a 64-character SHA-256")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise F2QualificationError(f"{name} is not hexadecimal")
    if len(set(normalized)) == 1:
        raise F2QualificationError(f"{name} cannot be a placeholder digest")
    return normalized


def _validate_report_sha256(value: Any, name: str) -> str:
    """Validate an independently produced report digest, rejecting sentinels."""

    return _validate_sha256(value, name)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise F2QualificationError(f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise F2QualificationError(f"{name} must be a nonnegative integer")
    return value


def _unique_strings(values: Iterable[Any], name: str) -> Tuple[str, ...]:
    converted = tuple(_nonempty_string(value, name) for value in values)
    duplicates = sorted(
        value for value, count in Counter(converted).items() if count > 1
    )
    if duplicates:
        raise F2QualificationError(
            f"{name} contains duplicate identifiers: {duplicates[:8]}"
        )
    return converted


def identifier_sha256(values: Iterable[Any]) -> str:
    """Hash sorted UTF-8 identifiers, one per line, with a final newline."""

    identifiers = sorted(str(value) for value in values)
    payload = (
        ("\n".join(identifiers) + "\n").encode("utf-8")
        if identifiers
        else b""
    )
    return _sha256(payload)


def _normalize_sensor_contract(
    contract: Mapping[str, Sequence[int]]
    | Sequence[Tuple[str, Sequence[int]]],
) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    items = contract.items() if isinstance(contract, Mapping) else contract
    normalized: Dict[str, Tuple[int, ...]] = {}
    for raw_name, raw_indices in items:
        name = _nonempty_string(raw_name, "sensor name")
        if name in normalized:
            raise F2QualificationError(f"duplicate sensor field {name!r}")
        if isinstance(raw_indices, (str, bytes)):
            raise F2QualificationError(
                f"sensor {name!r} indices must be an integer sequence"
            )
        indices = []
        for index in raw_indices:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise F2QualificationError(
                    f"sensor {name!r} indices must be nonnegative integers"
                )
            indices.append(index)
        if len(indices) != len(set(indices)):
            raise F2QualificationError(f"sensor {name!r} contains duplicate indices")
        normalized[name] = tuple(sorted(indices))
    return tuple(sorted(normalized.items()))


def sensor_config_contract(sensor_config: Any) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    """Canonicalize a NAVSIM SensorConfig without accepting implicit ``True``."""

    if is_dataclass(sensor_config) and not isinstance(sensor_config, type):
        values = asdict(sensor_config)
    elif isinstance(sensor_config, Mapping):
        values = dict(sensor_config)
    else:
        raise F2QualificationError("F2 sensor config must be a dataclass or mapping")

    normalized: Dict[str, Tuple[int, ...]] = {}
    for raw_name, raw_value in values.items():
        name = _nonempty_string(raw_name, "sensor config field")
        if raw_value is False:
            indices: Tuple[int, ...] = ()
        elif raw_value is True:
            # ``True`` means every loaded frame, which cannot prove equality to
            # the preregistered finite-index sensor contract.
            raise F2QualificationError(
                f"F2 sensor {name!r} must use explicit frame indices, not True"
            )
        elif isinstance(raw_value, list):
            indices = _normalize_sensor_contract(((name, raw_value),))[0][1]
        else:
            raise F2QualificationError(
                f"F2 sensor {name!r} must be False or a list of frame indices"
            )
        normalized[name] = indices
    return tuple(sorted(normalized.items()))


def require_f2_sensor_contract(sensor_config: Any) -> None:
    """Fail unless the runtime agent requests exactly the frozen F2 sensors."""

    if sensor_config_contract(sensor_config) != F2_SENSOR_CONTRACT:
        raise F2QualificationError("F2 runtime sensor contract drifted")


@dataclass(frozen=True)
class QualificationRow:
    """One retained scene identity from the sensor qualification audit."""

    log_name: str
    time_us: int
    token: str
    future_camera_valid: bool

    def __post_init__(self) -> None:
        _nonempty_string(self.log_name, "log_name")
        _nonnegative_int(self.time_us, "time_us")
        _nonempty_string(self.token, "token")
        if not isinstance(self.future_camera_valid, bool):
            raise F2QualificationError("future_camera_valid must be a bool")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "future_camera_valid": self.future_camera_valid,
            "log_name": self.log_name,
            "time_us": self.time_us,
            "token": self.token,
        }


def row_identity_sha256(rows: Iterable[QualificationRow]) -> str:
    """Hash canonical ``log<TAB>time<TAB>token<TAB>future_bit`` lines."""

    identities = sorted(
        (
            row.log_name,
            row.time_us,
            row.token,
            "1" if row.future_camera_valid else "0",
        )
        for row in rows
    )
    payload = "".join(
        f"{log_name}\t{time_us}\t{token}\t{future_bit}\n"
        for log_name, time_us, token, future_bit in identities
    ).encode("utf-8")
    return _sha256(payload)


@dataclass(frozen=True)
class QualificationPartition:
    """Canonical retained rows plus auditable raw/removal cardinalities."""

    name: str
    rows: Tuple[QualificationRow, ...]
    raw_token_count: int
    raw_log_count: int
    removed_token_count: int
    removed_log_count: int
    mixed_log_count: int
    exclusion_report_sha256: str

    def __post_init__(self) -> None:
        if self.name not in QUALIFICATION_PARTITIONS:
            raise F2QualificationError(f"unsupported qualification partition {self.name!r}")
        rows = tuple(self.rows)
        if not rows:
            raise F2QualificationError(f"{self.name} qualification rows cannot be empty")
        canonical_rows = tuple(
            sorted(rows, key=lambda row: (row.log_name, row.time_us, row.token))
        )
        if rows != canonical_rows:
            raise F2QualificationError(
                f"{self.name} qualification rows are not canonically sorted"
            )
        _unique_strings((row.token for row in rows), f"{self.name} tokens")
        identities = [(row.log_name, row.time_us, row.token) for row in rows]
        if len(identities) != len(set(identities)):
            raise F2QualificationError(f"{self.name} contains duplicate row identities")

        for field_name in (
            "raw_token_count",
            "raw_log_count",
            "removed_token_count",
            "removed_log_count",
            "mixed_log_count",
        ):
            _nonnegative_int(getattr(self, field_name), f"{self.name}.{field_name}")

        object.__setattr__(
            self,
            "exclusion_report_sha256",
            _validate_report_sha256(
                self.exclusion_report_sha256,
                f"{self.name}.exclusion_report_sha256",
            ),
        )

        if self.raw_token_count != self.retained_token_count + self.removed_token_count:
            raise F2QualificationError(
                f"{self.name} raw token count is inconsistent with retained + removed"
            )
        if self.raw_log_count != (
            self.retained_log_count + self.removed_log_count - self.mixed_log_count
        ):
            raise F2QualificationError(
                f"{self.name} raw log count is inconsistent with retained/removed overlap"
            )
        if self.mixed_log_count > min(self.retained_log_count, self.removed_log_count):
            raise F2QualificationError(f"{self.name} mixed log count is impossible")

    @property
    def tokens(self) -> Tuple[str, ...]:
        return tuple(row.token for row in self.rows)

    @property
    def logs(self) -> Tuple[str, ...]:
        return tuple(sorted({row.log_name for row in self.rows}))

    @property
    def retained_token_count(self) -> int:
        return len(self.rows)

    @property
    def retained_log_count(self) -> int:
        return len(self.logs)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "evidence": {
                "retained_log_count": self.retained_log_count,
                "retained_log_sha256": identifier_sha256(self.logs),
                "retained_token_count": self.retained_token_count,
                "retained_token_sha256": identifier_sha256(self.tokens),
                "row_identity_sha256": row_identity_sha256(self.rows),
            },
            "qualification": {
                "exclusion_report_sha256": self.exclusion_report_sha256,
                "mixed_log_count": self.mixed_log_count,
                "raw_log_count": self.raw_log_count,
                "raw_token_count": self.raw_token_count,
                "removed_log_count": self.removed_log_count,
                "removed_token_count": self.removed_token_count,
            },
            "rows": [row.to_payload() for row in self.rows],
        }


def build_qualification_partition(
    name: str,
    rows: Iterable[QualificationRow],
    *,
    raw_token_count: int,
    raw_log_count: int,
    removed_token_count: int,
    removed_log_count: int,
    mixed_log_count: int,
    exclusion_report_sha256: str,
) -> QualificationPartition:
    """Sort retained rows and bind them to one qualification audit summary."""

    canonical_rows = tuple(
        sorted(tuple(rows), key=lambda row: (row.log_name, row.time_us, row.token))
    )
    return QualificationPartition(
        name=name,
        rows=canonical_rows,
        raw_token_count=raw_token_count,
        raw_log_count=raw_log_count,
        removed_token_count=removed_token_count,
        removed_log_count=removed_log_count,
        mixed_log_count=mixed_log_count,
        exclusion_report_sha256=exclusion_report_sha256,
    )


@dataclass(frozen=True)
class QualificationManifest:
    """Canonical train/held-out token split accepted by F2 dataset loading."""

    train: QualificationPartition
    held_out: QualificationPartition
    config_sha256: str
    cache_sha256: str
    source_sha256: str
    sensor_contract: Tuple[Tuple[str, Tuple[int, ...]], ...] = F2_SENSOR_CONTRACT
    schema_version: str = QUALIFICATION_SCHEMA_VERSION
    dataset: str = QUALIFICATION_DATASET

    def __post_init__(self) -> None:
        if self.schema_version != QUALIFICATION_SCHEMA_VERSION:
            raise F2QualificationError("unsupported F2 qualification schema")
        if self.dataset != QUALIFICATION_DATASET:
            raise F2QualificationError("F2 qualification is restricted to navtrain")
        if self.train.name != "train" or self.held_out.name != "held_out":
            raise F2QualificationError("qualification partition names drifted")
        object.__setattr__(
            self, "config_sha256", _validate_sha256(self.config_sha256, "config_sha256")
        )
        object.__setattr__(
            self, "cache_sha256", _validate_sha256(self.cache_sha256, "cache_sha256")
        )
        object.__setattr__(
            self, "source_sha256", _validate_sha256(self.source_sha256, "source_sha256")
        )
        normalized_contract = _normalize_sensor_contract(self.sensor_contract)
        if normalized_contract != F2_SENSOR_CONTRACT:
            raise F2QualificationError("F2 qualification sensor contract drifted")
        object.__setattr__(self, "sensor_contract", normalized_contract)

        overlap = sorted(set(self.train.tokens) & set(self.held_out.tokens))
        if overlap:
            raise F2QualificationError(
                f"train and held_out token partitions overlap: {overlap[:8]}"
            )

    @property
    def all_tokens(self) -> Tuple[str, ...]:
        return self.train.tokens + self.held_out.tokens

    def partition(self, name: str) -> QualificationPartition:
        if name == "train":
            return self.train
        if name in ("held_out", "val"):
            return self.held_out
        raise F2QualificationError(f"unknown qualification partition {name!r}")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "digests": {
                "cache_sha256": self.cache_sha256,
                "config_sha256": self.config_sha256,
                "source_sha256": self.source_sha256,
            },
            "evidence": {
                "all_token_count": len(self.all_tokens),
                "all_token_sha256": identifier_sha256(self.all_tokens),
                "partitions_disjoint": True,
            },
            "partitions": {
                "held_out": self.held_out.to_payload(),
                "train": self.train.to_payload(),
            },
            "schema_version": self.schema_version,
            "sensor_contract": {
                name: list(indices) for name, indices in self.sensor_contract
            },
        }


def build_qualification_manifest(
    train: QualificationPartition,
    held_out: QualificationPartition,
    *,
    config_sha256: str,
    cache_sha256: str,
    source_sha256: str,
    sensor_contract: Mapping[str, Sequence[int]]
    | Sequence[Tuple[str, Sequence[int]]] = F2_SENSOR_CONTRACT,
) -> QualificationManifest:
    """Build the deterministic two-partition manifest without writing it."""

    return QualificationManifest(
        train=train,
        held_out=held_out,
        config_sha256=config_sha256,
        cache_sha256=cache_sha256,
        source_sha256=source_sha256,
        sensor_contract=_normalize_sensor_contract(sensor_contract),
    )


def canonical_manifest_bytes(manifest: QualificationManifest) -> bytes:
    return _canonical_json_bytes(manifest.to_payload())


def qualification_manifest_sha256(manifest: QualificationManifest) -> str:
    return _sha256(canonical_manifest_bytes(manifest))


def _partition_from_payload(name: str, payload: Any) -> QualificationPartition:
    required_keys = {"evidence", "qualification", "rows"}
    if not isinstance(payload, dict) or set(payload) != required_keys:
        raise F2QualificationError(f"{name} partition keys do not match the schema")
    qualification = payload["qualification"]
    qualification_keys = {
        "exclusion_report_sha256",
        "mixed_log_count",
        "raw_log_count",
        "raw_token_count",
        "removed_log_count",
        "removed_token_count",
    }
    if not isinstance(qualification, dict) or set(qualification) != qualification_keys:
        raise F2QualificationError(f"{name} qualification keys do not match the schema")
    try:
        rows = tuple(
            QualificationRow(
                log_name=row["log_name"],
                time_us=row["time_us"],
                token=row["token"],
                future_camera_valid=row["future_camera_valid"],
            )
            for row in payload["rows"]
        )
        if any(
            not isinstance(row, dict)
            or set(row)
            != {"future_camera_valid", "log_name", "time_us", "token"}
            for row in payload["rows"]
        ):
            raise F2QualificationError(f"{name} rows do not match the schema")
        partition = QualificationPartition(
            name=name,
            rows=rows,
            raw_token_count=qualification["raw_token_count"],
            raw_log_count=qualification["raw_log_count"],
            removed_token_count=qualification["removed_token_count"],
            removed_log_count=qualification["removed_log_count"],
            mixed_log_count=qualification["mixed_log_count"],
            exclusion_report_sha256=qualification["exclusion_report_sha256"],
        )
    except (KeyError, TypeError) as error:
        raise F2QualificationError(f"{name} partition is malformed") from error
    if payload["evidence"] != partition.to_payload()["evidence"]:
        raise F2QualificationError(f"{name} evidence does not match its rows")
    return partition


def manifest_from_bytes(
    payload: bytes,
    *,
    expected_manifest_sha256: str,
    expected_config_sha256: str,
    expected_cache_sha256: str,
    expected_source_sha256: str,
) -> QualificationManifest:
    """Authenticate and compare embedded provenance to independent expectations.

    This comparison prevents manifest substitution, but deliberately does not
    recompute config/cache/source hashes from the live runtime environment.
    """

    expected_manifest = _validate_sha256(
        expected_manifest_sha256, "expected_manifest_sha256"
    )
    if _sha256(payload) != expected_manifest:
        raise F2QualificationError("F2 qualification manifest SHA-256 mismatch")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F2QualificationError(
            "F2 qualification manifest is not valid canonical JSON"
        ) from error
    required_keys = {
        "dataset",
        "digests",
        "evidence",
        "partitions",
        "schema_version",
        "sensor_contract",
    }
    if not isinstance(decoded, dict) or set(decoded) != required_keys:
        raise F2QualificationError("qualification manifest keys do not match the schema")
    if not isinstance(decoded["digests"], dict) or set(decoded["digests"]) != {
        "cache_sha256",
        "config_sha256",
        "source_sha256",
    }:
        raise F2QualificationError("qualification digest keys do not match the schema")
    if not isinstance(decoded["partitions"], dict) or set(decoded["partitions"]) != {
        "held_out",
        "train",
    }:
        raise F2QualificationError("qualification partitions do not match the schema")

    train = _partition_from_payload("train", decoded["partitions"]["train"])
    held_out = _partition_from_payload(
        "held_out", decoded["partitions"]["held_out"]
    )
    try:
        manifest = QualificationManifest(
            train=train,
            held_out=held_out,
            config_sha256=decoded["digests"]["config_sha256"],
            cache_sha256=decoded["digests"]["cache_sha256"],
            source_sha256=decoded["digests"]["source_sha256"],
            sensor_contract=_normalize_sensor_contract(decoded["sensor_contract"]),
            schema_version=decoded["schema_version"],
            dataset=decoded["dataset"],
        )
    except (KeyError, TypeError) as error:
        raise F2QualificationError("qualification manifest is malformed") from error

    expected_digests = {
        "config_sha256": _validate_sha256(
            expected_config_sha256, "expected_config_sha256"
        ),
        "cache_sha256": _validate_sha256(
            expected_cache_sha256, "expected_cache_sha256"
        ),
        "source_sha256": _validate_sha256(
            expected_source_sha256, "expected_source_sha256"
        ),
    }
    actual_digests = {
        "config_sha256": manifest.config_sha256,
        "cache_sha256": manifest.cache_sha256,
        "source_sha256": manifest.source_sha256,
    }
    if actual_digests != expected_digests:
        raise F2QualificationError(
            "F2 qualification config/cache/source digest mismatch"
        )
    if decoded["evidence"] != manifest.to_payload()["evidence"]:
        raise F2QualificationError("qualification manifest evidence drifted")
    if canonical_manifest_bytes(manifest) != payload:
        raise F2QualificationError("qualification manifest bytes are not canonical")
    return manifest


def load_qualification_manifest(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_config_sha256: str,
    expected_cache_sha256: str,
    expected_source_sha256: str,
) -> QualificationManifest:
    """Read and authenticate a frozen manifest from one explicit path."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        payload = manifest_path.read_bytes()
    except OSError as error:
        raise F2QualificationError(
            f"cannot read F2 qualification manifest {manifest_path}"
        ) from error
    return manifest_from_bytes(
        payload,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_cache_sha256=expected_cache_sha256,
        expected_source_sha256=expected_source_sha256,
    )


def constrain_scene_filter(
    scene_filter: Any, partition: QualificationPartition
) -> None:
    """Replace broad NAVSIM filtering with an authenticated explicit token set."""

    if getattr(scene_filter, "max_scenes", None) is not None:
        raise F2QualificationError("F2 qualification forbids SceneFilter.max_scenes")
    if getattr(scene_filter, "include_synthetic_scenes", False):
        raise F2QualificationError("F2 qualification forbids synthetic scenes")
    synthetic_tokens = getattr(scene_filter, "synthetic_scene_tokens", None)
    if synthetic_tokens:
        raise F2QualificationError("F2 qualification forbids synthetic scene tokens")

    configured_tokens = getattr(scene_filter, "tokens", None)
    if configured_tokens is not None:
        configured = set(_unique_strings(configured_tokens, "configured navtrain tokens"))
        outside = sorted(set(partition.tokens) - configured)
        if outside:
            raise F2QualificationError(
                "F2 manifest contains tokens outside the configured navtrain allowlist: "
                f"{outside[:8]}"
            )
    configured_logs = getattr(scene_filter, "log_names", None)
    if configured_logs is not None:
        allowed_logs = set(_unique_strings(configured_logs, "configured navtrain logs"))
        outside_logs = sorted(set(partition.logs) - allowed_logs)
        if outside_logs:
            raise F2QualificationError(
                "F2 manifest contains logs outside the configured navtrain source: "
                f"{outside_logs[:8]}"
            )

    # Explicit tokens are authoritative.  The derived logs only reduce I/O;
    # they do not define the split and may overlap across qualifications.
    scene_filter.tokens = list(partition.tokens)
    scene_filter.log_names = list(partition.logs)


def _assert_exact_tokens(
    actual_tokens: Iterable[Any], partition: QualificationPartition, phase: str
) -> None:
    actual = _unique_strings(actual_tokens, f"{partition.name} {phase} tokens")
    missing = sorted(set(partition.tokens) - set(actual))
    extra = sorted(set(actual) - set(partition.tokens))
    if missing or extra or len(actual) != len(partition.tokens):
        raise F2QualificationError(
            f"F2 {partition.name} {phase} token set differs from the manifest: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )


def _verify_regular_scene_identities(
    scene_loader: Any, partition: QualificationPartition
) -> None:
    if getattr(scene_loader, "synthetic_scenes", None):
        raise F2QualificationError("F2 qualification loaded unexpected synthetic scenes")
    frames_by_token = getattr(scene_loader, "scene_frames_dicts", None)
    scene_filter = getattr(scene_loader, "_scene_filter", None)
    if not isinstance(frames_by_token, Mapping) or scene_filter is None:
        raise F2QualificationError("F2 SceneLoader lacks auditable regular-scene metadata")
    history_frames = getattr(scene_filter, "num_history_frames", None)
    if isinstance(history_frames, bool) or not isinstance(history_frames, int):
        raise F2QualificationError("F2 SceneFilter history length is malformed")
    current_index = history_frames - 1
    expected_by_token = {row.token: row for row in partition.rows}
    for token in partition.tokens:
        frames = frames_by_token.get(token)
        if not isinstance(frames, list) or not (0 <= current_index < len(frames)):
            raise F2QualificationError(f"F2 token {token!r} has malformed scene frames")
        current = frames[current_index]
        if not isinstance(current, Mapping):
            raise F2QualificationError(f"F2 token {token!r} current frame is malformed")
        expected = expected_by_token[token]
        actual_token = current.get("token")
        actual_log = current.get("log_name")
        actual_time = current.get("timestamp")
        if (actual_log, actual_time, actual_token) != (
            expected.log_name,
            expected.time_us,
            expected.token,
        ):
            raise F2QualificationError(
                "F2 SceneLoader row identity differs from the manifest for token "
                f"{token!r}"
            )


def qualify_scene_loader(
    scene_loader: Any,
    partition: QualificationPartition,
    *,
    preflight: Callable[[], Any] | None = None,
) -> Any:
    """Audit one loaded split, require zero removals, then reindex canonically."""

    _assert_exact_tokens(scene_loader.tokens, partition, "preflight")
    _verify_regular_scene_identities(scene_loader, partition)
    availability = (
        preflight()
        if preflight is not None
        else scene_loader.filter_scenes_by_sensor_availability()
    )
    removed = getattr(availability, "removed_scenes", None)
    if isinstance(removed, bool) or not isinstance(removed, int):
        raise F2QualificationError("F2 sensor preflight returned malformed counts")
    if removed != 0:
        raise F2QualificationError(
            "F2 sensor preflight cannot preserve the frozen manifest: "
            f"split={partition.name}, removed={removed}"
        )
    _assert_exact_tokens(scene_loader.tokens, partition, "postflight")

    frames_by_token = scene_loader.scene_frames_dicts
    scene_loader.scene_frames_dicts = {
        token: frames_by_token[token] for token in partition.tokens
    }
    _assert_exact_tokens(scene_loader.tokens, partition, "canonical reindex")
    if tuple(scene_loader.tokens) != partition.tokens:
        raise F2QualificationError(
            f"F2 {partition.name} SceneLoader order is not canonical"
        )
    return availability
