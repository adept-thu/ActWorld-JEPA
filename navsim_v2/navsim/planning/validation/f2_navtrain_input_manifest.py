"""Authenticate every raw NAVSIM byte consumed by qualified F2 navtrain.

The qualification manifest freezes scene identities but not the underlying
annotation pickles or sensor blobs.  This module closes that gap without ever
enumerating an evaluation split.  It binds exactly the qualified ``trainval``
log pickles and the unique sensor blobs implied by those pickles, the frozen
four-history/ten-future window, and :data:`F2_SENSOR_CONTRACT`.

All root-relative reads use an ``openat`` chain with ``O_NOFOLLOW``.  The file
is hashed and consumed from the same descriptor, and every directory link plus
the final pathname is revalidated before returning.  A caller therefore
receives authenticated bytes, never a pathname that is subsequently reopened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import stat
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

from navsim.planning.training.f2_qualification_manifest import (
    QUALIFICATION_DATASET,
    QUALIFICATION_SCHEMA_VERSION,
    QualificationManifest,
    canonical_manifest_bytes as canonical_qualification_manifest_bytes,
    identifier_sha256,
    load_qualification_manifest,
    row_identity_sha256,
)
from navsim.planning.validation.f2_causal_validator import F2_SENSOR_CONTRACT


RAW_INPUT_MANIFEST_SCHEMA_VERSION = "actworld-f2-navtrain-raw-input-content-v1"
F2_NAVTRAIN_STORAGE_SPLIT = "trainval"
F2_NUM_HISTORY_FRAMES = 4
F2_NUM_FUTURE_FRAMES = 10
F2_NUM_SCENE_FRAMES = F2_NUM_HISTORY_FRAMES + F2_NUM_FUTURE_FRAMES
F2_FRAME_INTERVAL = 1
_READ_CHUNK_BYTES = 4 * 1024 * 1024
_VERIFIED_SEAL = object()


class F2NavtrainInputManifestError(ValueError):
    """Raised when raw navtrain content or its authenticated join drifts."""


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise F2NavtrainInputManifestError(f"{name} must be a 64-character SHA-256")
    if value != value.lower():
        raise F2NavtrainInputManifestError(f"{name} must use lowercase hexadecimal")
    if any(character not in "0123456789abcdef" for character in value):
        raise F2NavtrainInputManifestError(f"{name} must be hexadecimal")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise F2NavtrainInputManifestError(f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise F2NavtrainInputManifestError(f"{name} must be a nonnegative integer")
    return value


def _canonical_relative_path(value: Any, name: str) -> str:
    raw = _nonempty(value, name)
    if "\\" in raw:
        raise F2NavtrainInputManifestError(f"{name} must use POSIX separators")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise F2NavtrainInputManifestError(f"{name} must be a canonical relative path")
    canonical = pure.as_posix()
    if canonical != raw:
        raise F2NavtrainInputManifestError(f"{name} is not canonical POSIX form")
    return canonical


def _case_unique(values: Iterable[str], name: str) -> None:
    exact: set[str] = set()
    folded: Dict[str, str] = {}
    for value in values:
        if value in exact:
            raise F2NavtrainInputManifestError(f"duplicate {name}: {value!r}")
        exact.add(value)
        collision = folded.get(value.casefold())
        if collision is not None:
            raise F2NavtrainInputManifestError(
                f"case-colliding {name}: {collision!r} and {value!r}"
            )
        folded[value.casefold()] = value


def _stat_identity(value: os.stat_result) -> Tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_root(value: Path, name: str) -> Path:
    requested = Path(value).expanduser()
    try:
        lexical = Path(os.path.abspath(os.fspath(requested)))
        resolved = requested.resolve(strict=True)
        live = requested.lstat()
    except OSError as error:
        raise F2NavtrainInputManifestError(f"cannot resolve {name}: {requested}") from error
    if stat.S_ISLNK(live.st_mode) or not stat.S_ISDIR(live.st_mode):
        raise F2NavtrainInputManifestError(f"{name} must be a non-symlink directory")
    if os.path.normcase(os.fspath(lexical)) != os.path.normcase(os.fspath(resolved)):
        raise F2NavtrainInputManifestError(f"{name} must not traverse a symlink")
    if resolved.name != F2_NAVTRAIN_STORAGE_SPLIT:
        raise F2NavtrainInputManifestError(
            f"{name} must be the navtrain storage split {F2_NAVTRAIN_STORAGE_SPLIT!r}"
        )
    return resolved


@contextmanager
def _open_relative_no_symlinks(
    root: Path,
    relative_path: str,
    name: str,
) -> Iterator[Tuple[int, Tuple[int, int, int, int, int, int], Any]]:
    """Securely open one file and retain every parent descriptor."""

    relative = _canonical_relative_path(relative_path, f"{name} relative path")
    parts = PurePosixPath(relative).parts
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    opened_directories: list[int] = []
    directory_links: list[Tuple[int, str, int, Tuple[int, int, int, int, int, int]]] = []
    file_descriptor = -1
    try:
        root_before = root.lstat()
        if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
            raise F2NavtrainInputManifestError(f"{name} root changed or became a symlink")
        root_descriptor = os.open(root, directory_flags)
        opened_directories.append(root_descriptor)
        root_opened = os.fstat(root_descriptor)
        if _stat_identity(root_opened) != _stat_identity(root_before):
            raise F2NavtrainInputManifestError(f"{name} root changed before open")

        parent_descriptor = root_descriptor
        for part in parts[:-1]:
            before = os.stat(part, dir_fd=parent_descriptor, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise F2NavtrainInputManifestError(
                    f"{name} traverses a symlink or non-directory component"
                )
            child_descriptor = os.open(part, directory_flags, dir_fd=parent_descriptor)
            opened_directories.append(child_descriptor)
            opened = os.fstat(child_descriptor)
            identity = _stat_identity(opened)
            if identity != _stat_identity(before):
                raise F2NavtrainInputManifestError(f"{name} directory changed before open")
            directory_links.append((parent_descriptor, part, child_descriptor, identity))
            parent_descriptor = child_descriptor

        file_name = parts[-1]
        file_before = os.stat(file_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(file_before.st_mode) or not stat.S_ISREG(file_before.st_mode):
            raise F2NavtrainInputManifestError(
                f"{name} must be a regular non-symlink file"
            )
        file_descriptor = os.open(file_name, file_flags, dir_fd=parent_descriptor)
        file_opened = os.fstat(file_descriptor)
        file_identity = _stat_identity(file_opened)
        if file_identity != _stat_identity(file_before):
            raise F2NavtrainInputManifestError(f"{name} changed before open")

        def revalidate() -> None:
            if _stat_identity(os.fstat(root_descriptor)) != _stat_identity(root_opened):
                raise F2NavtrainInputManifestError(f"{name} root changed while open")
            if _stat_identity(root.lstat()) != _stat_identity(root_opened):
                raise F2NavtrainInputManifestError(f"{name} root path changed while open")
            for parent_fd, part, child_fd, identity in directory_links:
                linked = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    _stat_identity(os.fstat(child_fd)) != identity
                    or _stat_identity(linked) != identity
                ):
                    raise F2NavtrainInputManifestError(
                        f"{name} directory path changed while open"
                    )
            linked_file = os.stat(
                file_name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            if (
                _stat_identity(os.fstat(file_descriptor)) != file_identity
                or _stat_identity(linked_file) != file_identity
            ):
                raise F2NavtrainInputManifestError(f"{name} changed while open")

        yield file_descriptor, file_identity, revalidate
    except F2NavtrainInputManifestError:
        raise
    except OSError as error:
        raise F2NavtrainInputManifestError(f"cannot securely open {name}") from error
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        for descriptor in reversed(opened_directories):
            os.close(descriptor)


def _authenticated_payload(
    root: Path,
    entry: "RawInputContentEntry",
    name: str,
) -> bytes:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with _open_relative_no_symlinks(root, entry.relative_path, name) as (
        descriptor,
        identity,
        revalidate,
    ):
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        revalidate()
    payload = b"".join(chunks)
    if identity[3] != entry.byte_size or len(payload) != entry.byte_size:
        raise F2NavtrainInputManifestError(f"{name} byte size mismatch")
    if digest.hexdigest() != entry.sha256:
        raise F2NavtrainInputManifestError(f"{name} SHA-256 mismatch")
    return payload


def _hash_relative(root: Path, relative_path: str, name: str) -> "RawInputContentEntry":
    digest = hashlib.sha256()
    with _open_relative_no_symlinks(root, relative_path, name) as (
        descriptor,
        identity,
        revalidate,
    ):
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        revalidate()
    return RawInputContentEntry(
        relative_path=relative_path,
        byte_size=identity[3],
        sha256=digest.hexdigest(),
    )


def _stable_file_payload(path: Path, name: str) -> bytes:
    requested = Path(path).expanduser()
    try:
        lexical = Path(os.path.abspath(os.fspath(requested)))
        resolved = requested.resolve(strict=True)
        before = requested.lstat()
    except OSError as error:
        raise F2NavtrainInputManifestError(f"cannot resolve {name}: {requested}") from error
    if os.path.normcase(os.fspath(lexical)) != os.path.normcase(os.fspath(resolved)):
        raise F2NavtrainInputManifestError(f"{name} must not traverse a symlink")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise F2NavtrainInputManifestError(f"{name} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
        try:
            opened = os.fstat(descriptor)
            if _stat_identity(opened) != _stat_identity(before):
                raise F2NavtrainInputManifestError(f"{name} changed before open")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        linked = requested.lstat()
    except F2NavtrainInputManifestError:
        raise
    except OSError as error:
        raise F2NavtrainInputManifestError(f"cannot read stable {name}") from error
    if (
        _stat_identity(opened) != _stat_identity(after)
        or _stat_identity(opened) != _stat_identity(linked)
    ):
        raise F2NavtrainInputManifestError(f"{name} changed during authentication")
    payload = b"".join(chunks)
    if len(payload) != opened.st_size:
        raise F2NavtrainInputManifestError(f"{name} produced a short read")
    return payload


def _reject_constant(value: str) -> None:
    raise F2NavtrainInputManifestError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise F2NavtrainInputManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_canonical_json(payload: bytes, name: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except F2NavtrainInputManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F2NavtrainInputManifestError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict) or _canonical_json_bytes(decoded) != payload:
        raise F2NavtrainInputManifestError(f"{name} is not canonical JSON")
    return decoded


def _exact_keys(payload: Any, keys: Iterable[str], name: str) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        raise F2NavtrainInputManifestError(f"{name} must be a JSON object")
    expected = set(keys)
    actual = set(payload)
    if actual != expected:
        raise F2NavtrainInputManifestError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return payload


def _qualification_identity(
    qualification: QualificationManifest,
    manifest_sha256: str,
) -> Dict[str, Any]:
    return {
        "all_token_count": len(qualification.all_tokens),
        "all_token_sha256": identifier_sha256(qualification.all_tokens),
        "cache_sha256": qualification.cache_sha256,
        "config_sha256": qualification.config_sha256,
        "held_out_row_identity_sha256": row_identity_sha256(
            qualification.held_out.rows
        ),
        "manifest_sha256": manifest_sha256,
        "schema_version": qualification.schema_version,
        "sensor_contract": {
            name: list(indices) for name, indices in qualification.sensor_contract
        },
        "source_sha256": qualification.source_sha256,
        "train_row_identity_sha256": row_identity_sha256(qualification.train.rows),
    }


@dataclass(frozen=True)
class RawInputContentEntry:
    relative_path: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _canonical_relative_path(self.relative_path, "entry relative_path"),
        )
        _nonnegative_int(self.byte_size, "entry byte_size")
        object.__setattr__(self, "sha256", _sha256(self.sha256, "entry sha256"))

    def to_payload(self) -> Dict[str, Any]:
        return {
            "byte_size": self.byte_size,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class NavtrainLogContentEntry:
    log_name: str
    content: RawInputContentEntry

    def __post_init__(self) -> None:
        log_name = _nonempty(self.log_name, "log_name")
        if "/" in log_name or "\\" in log_name or log_name in (".", ".."):
            raise F2NavtrainInputManifestError("log_name must be one file-stem component")
        if type(self.content) is not RawInputContentEntry:
            raise F2NavtrainInputManifestError("log content entry is malformed")
        expected_path = f"{log_name}.pkl"
        if self.content.relative_path != expected_path:
            raise F2NavtrainInputManifestError(
                "qualified log relative path must be exactly '<log_name>.pkl'"
            )

    def to_payload(self) -> Dict[str, Any]:
        result = self.content.to_payload()
        result["log_name"] = self.log_name
        return result


@dataclass(frozen=True)
class NavtrainRawInputManifest:
    log_root: str
    sensor_root: str
    qualification_identity: Mapping[str, Any]
    logs: Tuple[NavtrainLogContentEntry, ...]
    sensor_blobs: Tuple[RawInputContentEntry, ...]
    schema_version: str = RAW_INPUT_MANIFEST_SCHEMA_VERSION
    dataset: str = QUALIFICATION_DATASET
    storage_split: str = F2_NAVTRAIN_STORAGE_SPLIT

    def __post_init__(self) -> None:
        if self.schema_version != RAW_INPUT_MANIFEST_SCHEMA_VERSION:
            raise F2NavtrainInputManifestError("unsupported raw-input manifest schema")
        if self.dataset != QUALIFICATION_DATASET:
            raise F2NavtrainInputManifestError("raw-input manifest is not navtrain")
        if self.storage_split != F2_NAVTRAIN_STORAGE_SPLIT:
            raise F2NavtrainInputManifestError("raw-input storage split drifted")
        for field_name in ("log_root", "sensor_root"):
            root = _nonempty(getattr(self, field_name), field_name)
            if not Path(root).is_absolute() or Path(root).as_posix() != root:
                raise F2NavtrainInputManifestError(
                    f"{field_name} must be an absolute canonical POSIX path"
                )
            if Path(root).name != F2_NAVTRAIN_STORAGE_SPLIT:
                raise F2NavtrainInputManifestError(f"{field_name} is not trainval")
        identity = _validate_qualification_identity(self.qualification_identity)
        object.__setattr__(self, "qualification_identity", identity)
        logs = tuple(self.logs)
        sensors = tuple(self.sensor_blobs)
        if not logs or not sensors:
            raise F2NavtrainInputManifestError("raw-input entries cannot be empty")
        if any(type(entry) is not NavtrainLogContentEntry for entry in logs):
            raise F2NavtrainInputManifestError("log entries are malformed")
        if any(type(entry) is not RawInputContentEntry for entry in sensors):
            raise F2NavtrainInputManifestError("sensor entries are malformed")
        if logs != tuple(sorted(logs, key=lambda entry: entry.log_name)):
            raise F2NavtrainInputManifestError("log entries are not canonically sorted")
        if sensors != tuple(sorted(sensors, key=lambda entry: entry.relative_path)):
            raise F2NavtrainInputManifestError("sensor entries are not canonically sorted")
        _case_unique((entry.log_name for entry in logs), "log name")
        _case_unique((entry.content.relative_path for entry in logs), "log path")
        _case_unique((entry.relative_path for entry in sensors), "sensor path")
        object.__setattr__(self, "logs", logs)
        object.__setattr__(self, "sensor_blobs", sensors)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "qualification": dict(self.qualification_identity),
            "roots": {
                "logs": {
                    "entries": [entry.to_payload() for entry in self.logs],
                    "path": self.log_root,
                },
                "sensor_blobs": {
                    "entries": [entry.to_payload() for entry in self.sensor_blobs],
                    "path": self.sensor_root,
                },
            },
            "schema_version": self.schema_version,
            "storage_split": self.storage_split,
        }


def _validate_qualification_identity(value: Any) -> Dict[str, Any]:
    identity = _exact_keys(
        value,
        (
            "all_token_count",
            "all_token_sha256",
            "cache_sha256",
            "config_sha256",
            "held_out_row_identity_sha256",
            "manifest_sha256",
            "schema_version",
            "sensor_contract",
            "source_sha256",
            "train_row_identity_sha256",
        ),
        "qualification identity",
    )
    result = dict(identity)
    _nonnegative_int(result["all_token_count"], "qualification all_token_count")
    for field_name in (
        "all_token_sha256",
        "cache_sha256",
        "config_sha256",
        "held_out_row_identity_sha256",
        "manifest_sha256",
        "source_sha256",
        "train_row_identity_sha256",
    ):
        result[field_name] = _sha256(result[field_name], f"qualification {field_name}")
    if result["schema_version"] != QUALIFICATION_SCHEMA_VERSION:
        raise F2NavtrainInputManifestError("qualification schema version drifted")
    expected_contract = {name: list(indices) for name, indices in F2_SENSOR_CONTRACT}
    if result["sensor_contract"] != expected_contract:
        raise F2NavtrainInputManifestError("qualification sensor contract drifted")
    return result


def canonical_raw_input_manifest_bytes(manifest: NavtrainRawInputManifest) -> bytes:
    if type(manifest) is not NavtrainRawInputManifest:
        raise F2NavtrainInputManifestError("manifest must be NavtrainRawInputManifest")
    return _canonical_json_bytes(manifest.to_payload())


def raw_input_manifest_sha256(manifest: NavtrainRawInputManifest) -> str:
    return _sha256_bytes(canonical_raw_input_manifest_bytes(manifest))


def raw_input_manifest_from_bytes(
    payload: bytes,
    *,
    expected_sha256: str,
) -> NavtrainRawInputManifest:
    expected = _sha256(expected_sha256, "expected raw-input manifest SHA-256")
    if _sha256_bytes(payload) != expected:
        raise F2NavtrainInputManifestError("raw-input manifest SHA-256 mismatch")
    root = _exact_keys(
        _parse_canonical_json(payload, "raw-input manifest"),
        ("dataset", "qualification", "roots", "schema_version", "storage_split"),
        "raw-input manifest",
    )
    roots = _exact_keys(root["roots"], ("logs", "sensor_blobs"), "raw-input roots")
    log_root = _exact_keys(roots["logs"], ("entries", "path"), "log root")
    sensor_root = _exact_keys(
        roots["sensor_blobs"], ("entries", "path"), "sensor root"
    )
    if not isinstance(log_root["entries"], list) or not isinstance(
        sensor_root["entries"], list
    ):
        raise F2NavtrainInputManifestError("raw-input entries must be lists")
    logs = []
    for raw in log_root["entries"]:
        value = _exact_keys(
            raw, ("byte_size", "log_name", "relative_path", "sha256"), "log entry"
        )
        logs.append(
            NavtrainLogContentEntry(
                log_name=value["log_name"],
                content=RawInputContentEntry(
                    relative_path=value["relative_path"],
                    byte_size=value["byte_size"],
                    sha256=value["sha256"],
                ),
            )
        )
    sensors = []
    for raw in sensor_root["entries"]:
        value = _exact_keys(
            raw, ("byte_size", "relative_path", "sha256"), "sensor entry"
        )
        sensors.append(
            RawInputContentEntry(
                relative_path=value["relative_path"],
                byte_size=value["byte_size"],
                sha256=value["sha256"],
            )
        )
    manifest = NavtrainRawInputManifest(
        log_root=log_root["path"],
        sensor_root=sensor_root["path"],
        qualification_identity=root["qualification"],
        logs=tuple(logs),
        sensor_blobs=tuple(sensors),
        schema_version=root["schema_version"],
        dataset=root["dataset"],
        storage_split=root["storage_split"],
    )
    if canonical_raw_input_manifest_bytes(manifest) != payload:
        raise F2NavtrainInputManifestError("raw-input manifest is not idempotently canonical")
    return manifest


def _unpickle_log(payload: bytes, log_name: str) -> list[Any]:
    try:
        value = pickle.loads(payload)
    except Exception as error:
        raise F2NavtrainInputManifestError(
            f"qualified log {log_name!r} is not a readable pickle"
        ) from error
    if not isinstance(value, list):
        raise F2NavtrainInputManifestError(
            f"qualified log {log_name!r} must contain a frame list"
        )
    return value


def _qualified_scenes(
    qualification: QualificationManifest,
    logs_by_name: Mapping[str, list[Any]],
) -> Dict[str, list[Mapping[str, Any]]]:
    expected_logs = {
        row.log_name
        for partition in (qualification.train, qualification.held_out)
        for row in partition.rows
    }
    if set(logs_by_name) != expected_logs:
        raise F2NavtrainInputManifestError("qualified log set differs from qualification")
    scenes: Dict[str, list[Mapping[str, Any]]] = {}
    for log_name in sorted(logs_by_name):
        scenes.update(
            _qualified_scenes_for_log(
                qualification, log_name, logs_by_name[log_name]
            )
        )
    return scenes


def _qualified_scenes_for_log(
    qualification: QualificationManifest,
    log_name: str,
    frames: list[Any],
) -> Dict[str, list[Mapping[str, Any]]]:
    rows_by_token = {
        row.token: row
        for partition in (qualification.train, qualification.held_out)
        for row in partition.rows
        if row.log_name == log_name
    }
    if not rows_by_token:
        raise F2NavtrainInputManifestError(
            f"raw-input loader received unqualified log {log_name!r}"
        )
    scenes: Dict[str, list[Mapping[str, Any]]] = {}
    for start in range(0, len(frames), F2_FRAME_INTERVAL):
        scene = frames[start : start + F2_NUM_SCENE_FRAMES]
        if len(scene) != F2_NUM_SCENE_FRAMES:
            continue
        current = scene[F2_NUM_HISTORY_FRAMES - 1]
        if not isinstance(current, Mapping):
            continue
        token = current.get("token")
        row = rows_by_token.get(token)
        if row is None:
            continue
        if token in scenes:
            raise F2NavtrainInputManifestError(
                f"qualified token {token!r} occurs in multiple raw windows"
            )
        if (
            current.get("log_name"),
            current.get("timestamp"),
            current.get("token"),
        ) != (row.log_name, row.time_us, row.token):
            raise F2NavtrainInputManifestError(
                f"qualified raw row identity drifted for token {token!r}"
            )
        route = current.get("roadblock_ids")
        if not isinstance(route, (list, tuple)) or not route:
            raise F2NavtrainInputManifestError(
                f"qualified token {token!r} no longer has a route"
            )
        if any(not isinstance(frame, Mapping) for frame in scene):
            raise F2NavtrainInputManifestError(
                f"qualified token {token!r} contains a malformed frame"
            )
        scenes[token] = scene  # type: ignore[assignment]
    missing = sorted(set(rows_by_token) - set(scenes))
    extra = sorted(set(scenes) - set(rows_by_token))
    if missing or extra:
        raise F2NavtrainInputManifestError(
            "raw logs do not reconstruct exact qualification tokens: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    return scenes


def _camera_entry(cameras: Any, sensor_name: str, token: str, index: int) -> Mapping[str, Any]:
    if not isinstance(cameras, Mapping):
        raise F2NavtrainInputManifestError(
            f"qualified token {token!r} frame {index} lacks camera metadata"
        )
    matches = [
        value for key, value in cameras.items() if str(key).lower() == sensor_name
    ]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise F2NavtrainInputManifestError(
            f"qualified token {token!r} frame {index} has ambiguous/missing {sensor_name}"
        )
    return matches[0]


def _consumed_sensor_paths(
    qualification: QualificationManifest,
    scenes: Mapping[str, list[Mapping[str, Any]]],
) -> Tuple[str, ...]:
    if qualification.sensor_contract != F2_SENSOR_CONTRACT:
        raise F2NavtrainInputManifestError("qualification sensor contract drifted")
    paths: set[str] = set()
    for token in sorted(scenes):
        scene = scenes[token]
        if len(scene) != F2_NUM_SCENE_FRAMES:
            raise F2NavtrainInputManifestError("qualified scene frame count drifted")
        for sensor_name, indices in F2_SENSOR_CONTRACT:
            for index in indices:
                if not 0 <= index < len(scene):
                    raise F2NavtrainInputManifestError("sensor contract index is out of range")
                frame = scene[index]
                if sensor_name == "lidar_pc":
                    raw_path = frame.get("lidar_path")
                else:
                    camera = _camera_entry(frame.get("cams"), sensor_name, token, index)
                    raw_path = camera.get("data_path")
                paths.add(
                    _canonical_relative_path(
                        raw_path,
                        f"qualified token {token!r} {sensor_name} frame {index} path",
                    )
                )
    result = tuple(sorted(paths))
    _case_unique(result, "consumed sensor path")
    return result


def _expected_log_relative_paths(
    qualification: QualificationManifest,
) -> Tuple[Tuple[str, str], ...]:
    logs = tuple(
        sorted(set(qualification.train.logs) | set(qualification.held_out.logs))
    )
    _case_unique(logs, "qualification log")
    result = []
    for log_name in logs:
        if "/" in log_name or "\\" in log_name or log_name in (".", ".."):
            raise F2NavtrainInputManifestError(
                "qualification log names must be one file-stem component"
            )
        result.append((log_name, _canonical_relative_path(f"{log_name}.pkl", "log path")))
    return tuple(result)


def build_raw_input_manifest(
    *,
    log_root: Path,
    sensor_root: Path,
    qualification: QualificationManifest,
    expected_qualification_manifest_sha256: str,
    workers: int = 8,
) -> NavtrainRawInputManifest:
    """Build a content manifest for exactly the qualified navtrain inputs."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise F2NavtrainInputManifestError("workers must be a positive integer")
    if type(qualification) is not QualificationManifest:
        raise F2NavtrainInputManifestError("qualification must be QualificationManifest")
    if qualification.dataset != QUALIFICATION_DATASET:
        raise F2NavtrainInputManifestError("raw-input builder accepts navtrain only")
    expected_qualification = _sha256(
        expected_qualification_manifest_sha256,
        "expected qualification manifest SHA-256",
    )
    if _sha256_bytes(canonical_qualification_manifest_bytes(qualification)) != expected_qualification:
        raise F2NavtrainInputManifestError("qualification manifest SHA-256 mismatch")
    logs_root = _canonical_root(log_root, "navtrain log root")
    blobs_root = _canonical_root(sensor_root, "navtrain sensor root")

    log_entries: list[NavtrainLogContentEntry] = []
    sensor_path_set: set[str] = set()
    for log_name, relative_path in _expected_log_relative_paths(qualification):
        content = _hash_relative(logs_root, relative_path, f"qualified log {log_name!r}")
        payload = _authenticated_payload(
            logs_root, content, f"qualified log {log_name!r}"
        )
        frames = _unpickle_log(payload, log_name)
        scenes = _qualified_scenes_for_log(qualification, log_name, frames)
        sensor_path_set.update(_consumed_sensor_paths(qualification, scenes))
        log_entries.append(NavtrainLogContentEntry(log_name=log_name, content=content))
    sensor_paths = tuple(sorted(sensor_path_set))
    _case_unique(sensor_paths, "consumed sensor path")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        sensor_entries = tuple(
            executor.map(
                lambda path: _hash_relative(
                    blobs_root, path, f"qualified sensor blob {path!r}"
                ),
                sensor_paths,
            )
        )
    return NavtrainRawInputManifest(
        log_root=logs_root.as_posix(),
        sensor_root=blobs_root.as_posix(),
        qualification_identity=_qualification_identity(
            qualification, expected_qualification
        ),
        logs=tuple(log_entries),
        sensor_blobs=sensor_entries,
    )


def raw_input_manifest_sidecar_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".sha256")


def _sidecar_bytes(path: Path, digest: str) -> bytes:
    return f"{digest}  {path.name}\n".encode("ascii")


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_partial(parent: Path, prefix: str, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".partial", dir=parent)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def write_raw_input_manifest(path: Path, manifest: NavtrainRawInputManifest) -> str:
    """Atomically write canonical JSON and its exact adjacent sidecar."""

    destination = Path(path).expanduser()
    if not destination.is_absolute():
        raise F2NavtrainInputManifestError("manifest destination must be absolute")
    try:
        lexical_parent = Path(os.path.abspath(os.fspath(destination.parent)))
        parent = destination.parent.resolve(strict=True)
        parent_stat = destination.parent.lstat()
    except OSError as error:
        raise F2NavtrainInputManifestError("cannot resolve manifest parent") from error
    if (
        stat.S_ISLNK(parent_stat.st_mode)
        or not stat.S_ISDIR(parent_stat.st_mode)
        or os.path.normcase(os.fspath(lexical_parent))
        != os.path.normcase(os.fspath(parent))
    ):
        raise F2NavtrainInputManifestError(
            "manifest parent must be a non-symlink canonical directory"
        )
    destination = parent / destination.name
    sidecar = raw_input_manifest_sidecar_path(destination)
    if destination.exists() or sidecar.exists():
        raise F2NavtrainInputManifestError("raw-input manifest destination already exists")
    payload = canonical_raw_input_manifest_bytes(manifest)
    digest = _sha256_bytes(payload)
    manifest_partial = _write_partial(parent, f".{destination.name}.", payload)
    sidecar_partial = _write_partial(
        parent, f".{sidecar.name}.", _sidecar_bytes(destination, digest)
    )
    expected_sidecar = _sidecar_bytes(destination, digest)
    if _stable_file_payload(manifest_partial, "manifest partial") != payload:
        raise F2NavtrainInputManifestError("manifest partial failed byte verification")
    if _stable_file_payload(sidecar_partial, "sidecar partial") != expected_sidecar:
        raise F2NavtrainInputManifestError("sidecar partial failed byte verification")
    sidecar_published = False
    manifest_published = False
    try:
        os.link(sidecar_partial, sidecar)
        sidecar_published = True
        _fsync_parent(sidecar)
        os.link(manifest_partial, destination)
        manifest_published = True
        _fsync_parent(destination)
    except BaseException:
        if sidecar_published and not manifest_published:
            try:
                if sidecar.stat().st_ino == sidecar_partial.stat().st_ino:
                    sidecar.unlink()
            except OSError:
                pass
        raise
    finally:
        manifest_partial.unlink(missing_ok=True)
        sidecar_partial.unlink(missing_ok=True)
    return digest


@dataclass(frozen=True)
class VerifiedNavtrainRawInputs:
    """Sealed manifest whose every consumer read is same-descriptor verified."""

    manifest: NavtrainRawInputManifest
    manifest_sha256: str
    log_root: Path
    sensor_root: Path
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _VERIFIED_SEAL:
            raise F2NavtrainInputManifestError(
                "VerifiedNavtrainRawInputs must come from load_and_verify_raw_input_manifest()"
            )

    def _log_entry(self, log_name: str) -> Optional[NavtrainLogContentEntry]:
        entries = self.manifest.logs
        lower = 0
        upper = len(entries)
        while lower < upper:
            middle = (lower + upper) // 2
            value = entries[middle].log_name
            if value < log_name:
                lower = middle + 1
            else:
                upper = middle
        if lower < len(entries) and entries[lower].log_name == log_name:
            return entries[lower]
        return None

    def _sensor_entry(self, relative_path: str) -> Optional[RawInputContentEntry]:
        entries = self.manifest.sensor_blobs
        lower = 0
        upper = len(entries)
        while lower < upper:
            middle = (lower + upper) // 2
            value = entries[middle].relative_path
            if value < relative_path:
                lower = middle + 1
            else:
                upper = middle
        if lower < len(entries) and entries[lower].relative_path == relative_path:
            return entries[lower]
        return None

    def require_roots(self, log_root: Path, sensor_root: Path) -> None:
        if _canonical_root(log_root, "navtrain log root") != self.log_root:
            raise F2NavtrainInputManifestError("SceneLoader log root differs from manifest")
        if _canonical_root(sensor_root, "navtrain sensor root") != self.sensor_root:
            raise F2NavtrainInputManifestError("SceneLoader sensor root differs from manifest")

    def log_paths(self, log_names: Iterable[str]) -> Tuple[Path, ...]:
        requested = tuple(_nonempty(name, "requested log name") for name in log_names)
        _case_unique(requested, "requested log name")
        resolved = tuple((name, self._log_entry(name)) for name in requested)
        missing = sorted(name for name, entry in resolved if entry is None)
        if missing:
            raise F2NavtrainInputManifestError(
                f"SceneLoader requested logs outside raw-input manifest: {missing[:8]}"
            )
        return tuple(
            self.log_root / entry.content.relative_path
            for _, entry in resolved
            if entry is not None
        )

    def load_log_pickle(self, log_path: Path) -> list[Any]:
        requested = Path(log_path).expanduser()
        lexical = Path(os.path.abspath(os.fspath(requested)))
        try:
            relative = lexical.relative_to(self.log_root).as_posix()
        except ValueError as error:
            raise F2NavtrainInputManifestError("requested log path escapes bound root") from error
        relative = _canonical_relative_path(relative, "requested log relative path")
        if not relative.endswith(".pkl") or "/" in relative:
            raise F2NavtrainInputManifestError("requested log is outside raw-input manifest")
        entry = self._log_entry(relative[:-4])
        if entry is None or entry.content.relative_path != relative:
            raise F2NavtrainInputManifestError("requested log is outside raw-input manifest")
        payload = _authenticated_payload(
            self.log_root, entry.content, f"qualified log {entry.log_name!r}"
        )
        return _unpickle_log(payload, entry.log_name)

    def has_sensor_blob(self, relative_path: Any) -> bool:
        try:
            relative = _canonical_relative_path(relative_path, "sensor blob path")
        except F2NavtrainInputManifestError:
            return False
        return self._sensor_entry(relative) is not None

    def read_sensor_blob(self, relative_path: Any) -> bytes:
        relative = _canonical_relative_path(relative_path, "sensor blob path")
        entry = self._sensor_entry(relative)
        if entry is None:
            raise F2NavtrainInputManifestError(
                f"sensor blob is outside raw-input manifest: {relative!r}"
            )
        return _authenticated_payload(
            self.sensor_root, entry, f"qualified sensor blob {relative!r}"
        )

    def verify_all(self, workers: int = 8) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise F2NavtrainInputManifestError("workers must be a positive integer")

        def verify_log(entry: NavtrainLogContentEntry) -> None:
            payload = _authenticated_payload(
                self.log_root, entry.content, f"qualified log {entry.log_name!r}"
            )
            _unpickle_log(payload, entry.log_name)

        def verify_sensor(entry: RawInputContentEntry) -> None:
            _authenticated_payload(
                self.sensor_root,
                entry,
                f"qualified sensor blob {entry.relative_path!r}",
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            tuple(executor.map(verify_log, self.manifest.logs))
            tuple(executor.map(verify_sensor, self.manifest.sensor_blobs))


def load_and_verify_raw_input_manifest(
    path: Path,
    *,
    expected_sha256: str,
    log_root: Path,
    sensor_root: Path,
    qualification: QualificationManifest,
    expected_qualification_manifest_sha256: str,
) -> VerifiedNavtrainRawInputs:
    """Authenticate canonical metadata and independently derive exact coverage."""

    expected = _sha256(expected_sha256, "expected raw-input manifest SHA-256")
    destination = Path(path).expanduser()
    payload = _stable_file_payload(destination, "raw-input manifest")
    sidecar_payload = _stable_file_payload(
        raw_input_manifest_sidecar_path(destination), "raw-input manifest sidecar"
    )
    if sidecar_payload != _sidecar_bytes(destination, expected):
        raise F2NavtrainInputManifestError("raw-input manifest sidecar mismatch")
    manifest = raw_input_manifest_from_bytes(payload, expected_sha256=expected)
    logs_root = _canonical_root(log_root, "navtrain log root")
    blobs_root = _canonical_root(sensor_root, "navtrain sensor root")
    if manifest.log_root != logs_root.as_posix():
        raise F2NavtrainInputManifestError("live navtrain log root differs from manifest")
    if manifest.sensor_root != blobs_root.as_posix():
        raise F2NavtrainInputManifestError("live navtrain sensor root differs from manifest")
    expected_qualification = _sha256(
        expected_qualification_manifest_sha256,
        "expected qualification manifest SHA-256",
    )
    if _sha256_bytes(canonical_qualification_manifest_bytes(qualification)) != expected_qualification:
        raise F2NavtrainInputManifestError("qualification manifest SHA-256 mismatch")
    if manifest.qualification_identity != _qualification_identity(
        qualification, expected_qualification
    ):
        raise F2NavtrainInputManifestError("raw-input qualification identity mismatch")
    expected_log_pairs = _expected_log_relative_paths(qualification)
    actual_log_pairs = tuple(
        (entry.log_name, entry.content.relative_path) for entry in manifest.logs
    )
    if actual_log_pairs != expected_log_pairs:
        raise F2NavtrainInputManifestError(
            "raw-input manifest has missing or extra qualified log entries"
        )
    expected_sensor_path_set: set[str] = set()
    for entry in manifest.logs:
        frames = _unpickle_log(
            _authenticated_payload(
                logs_root, entry.content, f"qualified log {entry.log_name!r}"
            ),
            entry.log_name,
        )
        scenes = _qualified_scenes_for_log(
            qualification, entry.log_name, frames
        )
        expected_sensor_path_set.update(
            _consumed_sensor_paths(qualification, scenes)
        )
    expected_sensor_paths = tuple(sorted(expected_sensor_path_set))
    _case_unique(expected_sensor_paths, "consumed sensor path")
    actual_sensor_paths = tuple(
        entry.relative_path for entry in manifest.sensor_blobs
    )
    if actual_sensor_paths != expected_sensor_paths:
        raise F2NavtrainInputManifestError(
            "raw-input manifest has missing or extra consumed sensor entries"
        )
    return VerifiedNavtrainRawInputs(
        manifest=manifest,
        manifest_sha256=expected,
        log_root=logs_root,
        sensor_root=blobs_root,
        _seal=_VERIFIED_SEAL,
    )


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or fully verify qualified F2 navtrain raw inputs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    verify = subparsers.add_parser("verify")
    for command in (build, verify):
        command.add_argument("--qualification-manifest", required=True, type=Path)
        command.add_argument("--qualification-sha256", required=True)
        command.add_argument("--qualification-config-sha256", required=True)
        command.add_argument("--qualification-cache-sha256", required=True)
        command.add_argument("--qualification-source-sha256", required=True)
        command.add_argument("--log-root", required=True, type=Path)
        command.add_argument("--sensor-root", required=True, type=Path)
        command.add_argument("--workers", type=int, default=8)
    build.add_argument("--output", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    qualification = load_qualification_manifest(
        args.qualification_manifest,
        expected_manifest_sha256=args.qualification_sha256,
        expected_config_sha256=args.qualification_config_sha256,
        expected_cache_sha256=args.qualification_cache_sha256,
        expected_source_sha256=args.qualification_source_sha256,
    )
    if args.command == "build":
        manifest = build_raw_input_manifest(
            log_root=args.log_root,
            sensor_root=args.sensor_root,
            qualification=qualification,
            expected_qualification_manifest_sha256=args.qualification_sha256,
            workers=args.workers,
        )
        print(write_raw_input_manifest(args.output, manifest))
        return 0
    verified = load_and_verify_raw_input_manifest(
        args.manifest,
        expected_sha256=args.manifest_sha256,
        log_root=args.log_root,
        sensor_root=args.sensor_root,
        qualification=qualification,
        expected_qualification_manifest_sha256=args.qualification_sha256,
    )
    verified.verify_all(workers=args.workers)
    print(verified.manifest_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())


__all__ = [
    "F2NavtrainInputManifestError",
    "NavtrainLogContentEntry",
    "NavtrainRawInputManifest",
    "RawInputContentEntry",
    "VerifiedNavtrainRawInputs",
    "build_raw_input_manifest",
    "canonical_raw_input_manifest_bytes",
    "cli_main",
    "load_and_verify_raw_input_manifest",
    "raw_input_manifest_sha256",
    "raw_input_manifest_sidecar_path",
    "raw_input_manifest_from_bytes",
    "write_raw_input_manifest",
]
