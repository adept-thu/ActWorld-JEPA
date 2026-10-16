"""Content authentication for the qualified F2 navtrain metric cache.

The upstream metric-cache metadata CSV is an index, not a content manifest.
This module closes that gap without touching any NAVSIM evaluation split: an
offline builder selects exactly the union of the frozen qualification
``train`` and ``held_out`` tokens, hashes every selected compressed pickle,
and publishes canonical JSON plus an adjacent SHA-256 sidecar.

At runtime a manifest is joined back to the same qualification manifest,
metadata CSV, and cache root.  Every pickle is opened without following a
symlink and is authenticated through the *same descriptor* that is rewound
and passed to ``lzma``/``pickle``.  A process-local identity cache avoids
rehashing an unchanged inode; it is guarded by a lock and is deliberately not
shared between spawned scoring workers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import os
import pickle
import stat
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

from navsim.planning.training.f2_qualification_manifest import (
    QUALIFICATION_DATASET,
    QUALIFICATION_SCHEMA_VERSION,
    QualificationManifest,
    canonical_manifest_bytes as canonical_qualification_manifest_bytes,
    load_qualification_manifest,
)


METRIC_CACHE_MANIFEST_SCHEMA_VERSION = "actworld-f2-navtrain-metric-cache-content-v1"
F2_PRUNED_CACHE_FAMILY = "train_metric_cache_v2_pruned"
METRIC_CACHE_FILE_NAME = "metric_cache.pkl"
_VERIFIED_SEAL = object()
_READ_CHUNK_BYTES = 4 * 1024 * 1024


class F2MetricCacheManifestError(ValueError):
    """Raised when any cache-content identity or join is not exact."""


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
        raise F2MetricCacheManifestError(f"{name} must be a 64-character SHA-256")
    if value != value.lower():
        raise F2MetricCacheManifestError(f"{name} must use lowercase hexadecimal")
    if any(character not in "0123456789abcdef" for character in value):
        raise F2MetricCacheManifestError(f"{name} must be hexadecimal")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise F2MetricCacheManifestError(f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise F2MetricCacheManifestError(f"{name} must be a nonnegative integer")
    return value


def _canonical_relative_path(value: Any, name: str) -> str:
    raw = _nonempty(value, name)
    if "\\" in raw:
        raise F2MetricCacheManifestError(f"{name} must use POSIX separators")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise F2MetricCacheManifestError(f"{name} must be a canonical relative path")
    canonical = pure.as_posix()
    if canonical != raw:
        raise F2MetricCacheManifestError(f"{name} is not canonical POSIX form")
    return canonical


def _canonical_absolute_root(value: Path) -> Path:
    path = Path(value).expanduser()
    try:
        lexical = Path(os.path.abspath(str(path)))
        resolved = path.resolve(strict=True)
        root_stat = path.lstat()
    except OSError as error:
        raise F2MetricCacheManifestError(f"cannot resolve metric-cache root: {path}") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise F2MetricCacheManifestError("metric-cache root must be a non-symlink directory")
    if os.path.normcase(str(lexical)) != os.path.normcase(str(resolved)):
        raise F2MetricCacheManifestError(
            "metric-cache root must not traverse a symbolic link"
        )
    if resolved.name != F2_PRUNED_CACHE_FAMILY:
        raise F2MetricCacheManifestError(
            f"metric-cache root must be named {F2_PRUNED_CACHE_FAMILY!r}"
        )
    return resolved


def _stat_identity(value: os.stat_result) -> Tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file_payload(path: Path, name: str) -> bytes:
    """Read regular-file bytes while binding pathname and descriptor identity."""

    path = Path(path)
    try:
        path_before = path.lstat()
    except OSError as error:
        raise F2MetricCacheManifestError(f"cannot stat {name}: {path}") from error
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise F2MetricCacheManifestError(f"{name} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if _stat_identity(opened) != _stat_identity(path_before):
                raise F2MetricCacheManifestError(f"{name} changed before authentication")
            chunks = []
            while True:
                chunk = os.read(descriptor, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = path.lstat()
    except F2MetricCacheManifestError:
        raise
    except OSError as error:
        raise F2MetricCacheManifestError(f"cannot read stable {name}: {path}") from error
    identity = _stat_identity(opened)
    if identity != _stat_identity(after) or identity != _stat_identity(path_after):
        raise F2MetricCacheManifestError(f"{name} changed during authentication")
    payload = b"".join(chunks)
    if len(payload) != opened.st_size:
        raise F2MetricCacheManifestError(f"{name} produced a short read")
    return payload


def _reject_constant(value: str) -> None:
    raise F2MetricCacheManifestError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise F2MetricCacheManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_canonical_json(payload: bytes, name: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except F2MetricCacheManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F2MetricCacheManifestError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict) or _canonical_json_bytes(decoded) != payload:
        raise F2MetricCacheManifestError(f"{name} is not canonical JSON")
    return decoded


def _exact_keys(payload: Any, keys: Iterable[str], name: str) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        raise F2MetricCacheManifestError(f"{name} must be a JSON object")
    expected = set(keys)
    actual = set(payload)
    if actual != expected:
        raise F2MetricCacheManifestError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return payload


def _case_unique(values: Iterable[str], name: str) -> None:
    exact: Dict[str, str] = {}
    folded: Dict[str, str] = {}
    for value in values:
        if value in exact:
            raise F2MetricCacheManifestError(f"duplicate {name}: {value!r}")
        exact[value] = value
        collision = folded.get(value.casefold())
        if collision is not None:
            raise F2MetricCacheManifestError(
                f"case-colliding {name}: {collision!r} and {value!r}"
            )
        folded[value.casefold()] = value


def _qualification_partition_by_token(
    qualification: QualificationManifest,
) -> Dict[str, str]:
    if type(qualification) is not QualificationManifest:
        raise F2MetricCacheManifestError(
            "qualification must be a concrete QualificationManifest"
        )
    result: Dict[str, str] = {}
    for partition_name in ("train", "held_out"):
        for token in qualification.partition(partition_name).tokens:
            _nonempty(token, f"{partition_name} token")
            if "/" in token or "\\" in token:
                raise F2MetricCacheManifestError("qualification tokens cannot contain separators")
            if token in result:
                raise F2MetricCacheManifestError(
                    f"qualification token occurs in both partitions: {token!r}"
                )
            result[token] = partition_name
    _case_unique(result, "qualification token")
    return result


def _join_relative(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(_canonical_relative_path(relative_path, "relative_path"))
    return root.joinpath(*pure.parts)


def _relative_to_root(root: Path, raw_path: str) -> str:
    if not raw_path or "\x00" in raw_path:
        raise F2MetricCacheManifestError("cache metadata contains an empty path")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    lexical = Path(os.path.abspath(str(candidate)))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise F2MetricCacheManifestError(
            f"cache metadata path escapes the bound root: {raw_path!r}"
        ) from error
    return _canonical_relative_path(relative.as_posix(), "cache metadata relative path")


def _lexical_file_under_root(
    root: Path,
    requested_path: Path,
    name: str,
) -> Tuple[Path, str, os.stat_result]:
    """Bind an un-resolved caller path and reject symlinks before normalization."""

    requested = Path(requested_path).expanduser()
    lexical = Path(os.path.abspath(str(requested)))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise F2MetricCacheManifestError(f"{name} is outside metric-cache root") from error
    relative_posix = _canonical_relative_path(relative.as_posix(), f"{name} relative path")
    live_path, live_stat = _path_without_symlinks(root, relative_posix)
    return live_path, relative_posix, live_stat


def _path_without_symlinks(root: Path, relative_path: str) -> Tuple[Path, os.stat_result]:
    """Resolve a lexical child while rejecting a symlink in every component."""

    path = root
    parts = PurePosixPath(relative_path).parts
    for index, part in enumerate(parts):
        path = path / part
        try:
            value = path.lstat()
        except OSError as error:
            raise F2MetricCacheManifestError(f"cache path cannot be stated: {path}") from error
        if stat.S_ISLNK(value.st_mode):
            raise F2MetricCacheManifestError(f"cache path traverses a symlink: {path}")
        if index + 1 < len(parts):
            if not stat.S_ISDIR(value.st_mode):
                raise F2MetricCacheManifestError(f"cache parent is not a directory: {path}")
        elif not stat.S_ISREG(value.st_mode):
            raise F2MetricCacheManifestError(f"cache entry is not a regular file: {path}")
    return path, value


@contextmanager
def _open_relative_no_symlinks(
    root: Path,
    relative_path: str,
    name: str,
) -> Iterator[Tuple[int, Tuple[int, int, int, int, int, int], Any]]:
    """Open one root-relative file with an lstat/openat/fstat chain.

    Keeping every parent directory descriptor open prevents a directory swap
    from redirecting the final open outside ``root``.  The returned validator
    rechecks every directory entry and the final file entry through those same
    descriptors after hashing/deserialization.
    """

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
    opened_directories = []
    directory_links = []
    file_descriptor = -1
    try:
        root_before = root.lstat()
        if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
            raise F2MetricCacheManifestError("metric-cache root changed or became a symlink")
        root_descriptor = os.open(root, directory_flags)
        opened_directories.append(root_descriptor)
        root_opened = os.fstat(root_descriptor)
        if _stat_identity(root_opened) != _stat_identity(root_before):
            raise F2MetricCacheManifestError("metric-cache root changed before open")

        parent_descriptor = root_descriptor
        for part in parts[:-1]:
            before = os.stat(
                part,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise F2MetricCacheManifestError(
                    f"{name} traverses a symlink or non-directory component"
                )
            child_descriptor = os.open(part, directory_flags, dir_fd=parent_descriptor)
            opened_directories.append(child_descriptor)
            opened = os.fstat(child_descriptor)
            identity = _stat_identity(opened)
            if identity != _stat_identity(before):
                raise F2MetricCacheManifestError(
                    f"{name} directory changed before open"
                )
            directory_links.append(
                (parent_descriptor, part, child_descriptor, identity)
            )
            parent_descriptor = child_descriptor

        file_name = parts[-1]
        file_before = os.stat(
            file_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(file_before.st_mode) or not stat.S_ISREG(file_before.st_mode):
            raise F2MetricCacheManifestError(
                f"{name} must be a regular non-symlink file"
            )
        file_descriptor = os.open(file_name, file_flags, dir_fd=parent_descriptor)
        file_opened = os.fstat(file_descriptor)
        file_identity = _stat_identity(file_opened)
        if file_identity != _stat_identity(file_before):
            raise F2MetricCacheManifestError(f"{name} changed before open")

        def revalidate() -> None:
            if _stat_identity(os.fstat(root_descriptor)) != _stat_identity(root_opened):
                raise F2MetricCacheManifestError("metric-cache root changed while open")
            if _stat_identity(root.lstat()) != _stat_identity(root_opened):
                raise F2MetricCacheManifestError("metric-cache root path changed while open")
            for parent_fd, part, child_fd, identity in directory_links:
                linked = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    _stat_identity(os.fstat(child_fd)) != identity
                    or _stat_identity(linked) != identity
                ):
                    raise F2MetricCacheManifestError(
                        f"{name} directory path changed while open"
                    )
            linked_file = os.stat(
                file_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                _stat_identity(os.fstat(file_descriptor)) != file_identity
                or _stat_identity(linked_file) != file_identity
            ):
                raise F2MetricCacheManifestError(f"{name} changed while open")

        yield file_descriptor, file_identity, revalidate
    except F2MetricCacheManifestError:
        raise
    except OSError as error:
        raise F2MetricCacheManifestError(f"cannot securely open {name}") from error
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        for descriptor in reversed(opened_directories):
            os.close(descriptor)


def _stable_relative_payload(root: Path, relative_path: str, name: str) -> bytes:
    chunks = []
    with _open_relative_no_symlinks(root, relative_path, name) as (
        descriptor,
        identity,
        revalidate,
    ):
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
        revalidate()
    payload = b"".join(chunks)
    if len(payload) != identity[3]:
        raise F2MetricCacheManifestError(f"{name} produced a short read")
    return payload


def _stable_relative_digest(
    root: Path, relative_path: str, name: str
) -> Tuple[int, str, Tuple[int, int]]:
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
    return identity[3], digest.hexdigest(), (identity[0], identity[1])


def _metadata_mapping(
    root: Path,
    metadata_path: Path,
    *,
    expected_sha256: str,
) -> Tuple[Dict[str, str], int]:
    metadata_relative = _relative_to_root(root, os.fspath(metadata_path))
    payload = _stable_relative_payload(
        root, metadata_relative, "metric-cache metadata"
    )
    expected = _sha256(expected_sha256, "expected metadata SHA-256")
    if _sha256_bytes(payload) != expected:
        raise F2MetricCacheManifestError("metric-cache metadata SHA-256 mismatch")
    try:
        rows = list(csv.reader(payload.decode("utf-8").splitlines()))
    except (UnicodeDecodeError, csv.Error) as error:
        raise F2MetricCacheManifestError("metric-cache metadata is not valid UTF-8 CSV") from error
    if len(rows) < 2 or len(rows[0]) != 1 or not rows[0][0]:
        raise F2MetricCacheManifestError("metric-cache metadata must have one named column")
    mapping: Dict[str, str] = {}
    paths = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != 1 or not row[0]:
            raise F2MetricCacheManifestError(
                f"metric-cache metadata row {row_number} must contain one path"
            )
        relative = _relative_to_root(root, row[0])
        pure = PurePosixPath(relative)
        if pure.name != METRIC_CACHE_FILE_NAME or len(pure.parts) < 2:
            raise F2MetricCacheManifestError(
                f"metric-cache metadata row {row_number} is not a {METRIC_CACHE_FILE_NAME!r}"
            )
        token = _nonempty(pure.parent.name, "metric-cache token")
        if token in mapping:
            raise F2MetricCacheManifestError(
                f"metric-cache metadata has duplicate token {token!r}"
            )
        mapping[token] = relative
        paths.append(relative)
    _case_unique(mapping, "metadata token")
    _case_unique(paths, "metadata relative path")
    return mapping, len(payload)


@dataclass(frozen=True)
class MetricCacheContentEntry:
    token: str
    partition: str
    relative_path: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        token = _nonempty(self.token, "entry token")
        if "/" in token or "\\" in token:
            raise F2MetricCacheManifestError("entry token cannot contain separators")
        if self.partition not in ("train", "held_out"):
            raise F2MetricCacheManifestError("entry partition must be train or held_out")
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
            "partition": self.partition,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "token": self.token,
        }


@dataclass(frozen=True)
class MetricCacheContentManifest:
    cache_root: str
    metadata_relative_path: str
    metadata_byte_size: int
    metadata_sha256: str
    qualification_manifest_sha256: str
    entries: Tuple[MetricCacheContentEntry, ...]
    schema_version: str = METRIC_CACHE_MANIFEST_SCHEMA_VERSION
    qualification_schema_version: str = QUALIFICATION_SCHEMA_VERSION
    dataset: str = QUALIFICATION_DATASET
    cache_family: str = F2_PRUNED_CACHE_FAMILY

    def __post_init__(self) -> None:
        if self.schema_version != METRIC_CACHE_MANIFEST_SCHEMA_VERSION:
            raise F2MetricCacheManifestError("unsupported metric-cache manifest schema")
        if self.qualification_schema_version != QUALIFICATION_SCHEMA_VERSION:
            raise F2MetricCacheManifestError("qualification schema binding drifted")
        if self.dataset != QUALIFICATION_DATASET:
            raise F2MetricCacheManifestError("metric-cache manifest is not navtrain")
        if self.cache_family != F2_PRUNED_CACHE_FAMILY:
            raise F2MetricCacheManifestError("metric-cache family drifted")
        root = _nonempty(self.cache_root, "cache_root")
        if not Path(root).is_absolute() or Path(root).as_posix() != root:
            raise F2MetricCacheManifestError("cache_root must be an absolute canonical POSIX path")
        object.__setattr__(
            self,
            "metadata_relative_path",
            _canonical_relative_path(
                self.metadata_relative_path, "metadata_relative_path"
            ),
        )
        _nonnegative_int(self.metadata_byte_size, "metadata_byte_size")
        object.__setattr__(
            self, "metadata_sha256", _sha256(self.metadata_sha256, "metadata_sha256")
        )
        object.__setattr__(
            self,
            "qualification_manifest_sha256",
            _sha256(
                self.qualification_manifest_sha256,
                "qualification_manifest_sha256",
            ),
        )
        entries = tuple(self.entries)
        if not entries or any(type(entry) is not MetricCacheContentEntry for entry in entries):
            raise F2MetricCacheManifestError("metric-cache entries must be non-empty concrete entries")
        if entries != tuple(sorted(entries, key=lambda entry: entry.token)):
            raise F2MetricCacheManifestError("metric-cache entries are not sorted by token")
        _case_unique((entry.token for entry in entries), "entry token")
        _case_unique((entry.relative_path for entry in entries), "entry relative path")
        object.__setattr__(self, "entries", entries)

    @property
    def by_token(self) -> Dict[str, MetricCacheContentEntry]:
        return {entry.token: entry for entry in self.entries}

    def to_payload(self) -> Dict[str, Any]:
        return {
            "cache": {
                "family": self.cache_family,
                "metadata_byte_size": self.metadata_byte_size,
                "metadata_relative_path": self.metadata_relative_path,
                "metadata_sha256": self.metadata_sha256,
                "root": self.cache_root,
            },
            "dataset": self.dataset,
            "entries": [entry.to_payload() for entry in self.entries],
            "qualification": {
                "manifest_sha256": self.qualification_manifest_sha256,
                "schema_version": self.qualification_schema_version,
            },
            "schema_version": self.schema_version,
        }


def canonical_metric_cache_manifest_bytes(manifest: MetricCacheContentManifest) -> bytes:
    if type(manifest) is not MetricCacheContentManifest:
        raise F2MetricCacheManifestError("manifest must be MetricCacheContentManifest")
    return _canonical_json_bytes(manifest.to_payload())


def metric_cache_manifest_sha256(manifest: MetricCacheContentManifest) -> str:
    return _sha256_bytes(canonical_metric_cache_manifest_bytes(manifest))


def metric_cache_manifest_from_bytes(
    payload: bytes,
    *,
    expected_sha256: str,
) -> MetricCacheContentManifest:
    expected = _sha256(expected_sha256, "expected metric-cache manifest SHA-256")
    if _sha256_bytes(payload) != expected:
        raise F2MetricCacheManifestError("metric-cache manifest SHA-256 mismatch")
    root = _exact_keys(
        _parse_canonical_json(payload, "metric-cache manifest"),
        ("cache", "dataset", "entries", "qualification", "schema_version"),
        "metric-cache manifest",
    )
    cache = _exact_keys(
        root["cache"],
        ("family", "metadata_byte_size", "metadata_relative_path", "metadata_sha256", "root"),
        "metric-cache identity",
    )
    qualification = _exact_keys(
        root["qualification"],
        ("manifest_sha256", "schema_version"),
        "qualification identity",
    )
    if not isinstance(root["entries"], list):
        raise F2MetricCacheManifestError("metric-cache entries must be a list")
    entries = []
    for raw in root["entries"]:
        value = _exact_keys(
            raw,
            ("byte_size", "partition", "relative_path", "sha256", "token"),
            "metric-cache entry",
        )
        entries.append(
            MetricCacheContentEntry(
                token=value["token"],
                partition=value["partition"],
                relative_path=value["relative_path"],
                byte_size=value["byte_size"],
                sha256=value["sha256"],
            )
        )
    manifest = MetricCacheContentManifest(
        cache_root=cache["root"],
        metadata_relative_path=cache["metadata_relative_path"],
        metadata_byte_size=cache["metadata_byte_size"],
        metadata_sha256=cache["metadata_sha256"],
        qualification_manifest_sha256=qualification["manifest_sha256"],
        qualification_schema_version=qualification["schema_version"],
        entries=tuple(entries),
        schema_version=root["schema_version"],
        dataset=root["dataset"],
        cache_family=cache["family"],
    )
    if canonical_metric_cache_manifest_bytes(manifest) != payload:
        raise F2MetricCacheManifestError(
            "metric-cache manifest is not idempotently canonical"
        )
    return manifest


def _hash_path(root: Path, token: str, partition: str, relative_path: str) -> Tuple[MetricCacheContentEntry, Tuple[int, int]]:
    byte_size, sha256, inode = _stable_relative_digest(
        root, relative_path, f"metric-cache pickle {token!r}"
    )
    return (
        MetricCacheContentEntry(
            token=token,
            partition=partition,
            relative_path=relative_path,
            byte_size=byte_size,
            sha256=sha256,
        ),
        inode,
    )


def build_metric_cache_manifest(
    *,
    cache_root: Path,
    metadata_path: Path,
    expected_metadata_sha256: str,
    qualification: QualificationManifest,
    expected_qualification_manifest_sha256: str,
    workers: int = 8,
) -> MetricCacheContentManifest:
    """Hash exactly the qualified navtrain cache entries, in parallel."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise F2MetricCacheManifestError("workers must be a positive integer")
    root = _canonical_absolute_root(cache_root)
    metadata, metadata_relative, _ = _lexical_file_under_root(
        root, Path(metadata_path), "metadata path"
    )
    mapping, metadata_byte_size = _metadata_mapping(
        root, metadata, expected_sha256=expected_metadata_sha256
    )
    partition_by_token = _qualification_partition_by_token(qualification)
    expected_qualification = _sha256(
        expected_qualification_manifest_sha256,
        "expected qualification manifest SHA-256",
    )
    if _sha256_bytes(canonical_qualification_manifest_bytes(qualification)) != expected_qualification:
        raise F2MetricCacheManifestError("qualification manifest SHA-256 mismatch")
    missing = sorted(set(partition_by_token) - set(mapping))
    if missing:
        raise F2MetricCacheManifestError(
            f"metric-cache metadata lacks qualified tokens: {missing[:8]}"
        )
    selected = tuple(
        (token, partition_by_token[token], mapping[token])
        for token in sorted(partition_by_token)
    )
    _case_unique((relative for _, _, relative in selected), "qualified cache path")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = tuple(executor.map(lambda value: _hash_path(root, *value), selected))
    inodes = [identity for _, identity in results]
    if len(inodes) != len(set(inodes)):
        raise F2MetricCacheManifestError(
            "qualified metric-cache paths contain duplicate hard-linked files"
        )
    return MetricCacheContentManifest(
        cache_root=root.as_posix(),
        metadata_relative_path=metadata_relative,
        metadata_byte_size=metadata_byte_size,
        metadata_sha256=_sha256(expected_metadata_sha256, "metadata SHA-256"),
        qualification_manifest_sha256=expected_qualification,
        entries=tuple(entry for entry, _ in results),
    )


def metric_cache_manifest_sidecar_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".sha256")


def _sidecar_bytes(path: Path, digest: str) -> bytes:
    return f"{_sha256(digest, 'manifest digest')}  {path.name}\n".encode("ascii")


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_partial(parent: Path, prefix: str, payload: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, suffix=".partial", dir=str(parent))
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path


def write_metric_cache_manifest(
    path: Path,
    manifest: MetricCacheContentManifest,
) -> str:
    """Publish canonical bytes and sibling digest without overwriting either."""

    target = Path(path)
    if not target.is_absolute():
        raise F2MetricCacheManifestError("manifest destination must be absolute")
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = metric_cache_manifest_sidecar_path(target)
    if target.exists() or sidecar.exists():
        raise FileExistsError("metric-cache manifest or sidecar already exists")
    payload = canonical_metric_cache_manifest_bytes(manifest)
    digest = _sha256_bytes(payload)
    manifest_partial = _write_partial(target.parent, target.name + ".", payload)
    sidecar_partial = _write_partial(
        target.parent, sidecar.name + ".", _sidecar_bytes(target, digest)
    )
    if _stable_file_payload(manifest_partial, "manifest partial") != payload:
        raise F2MetricCacheManifestError("manifest partial failed byte verification")
    expected_sidecar = _sidecar_bytes(target, digest)
    if _stable_file_payload(sidecar_partial, "sidecar partial") != expected_sidecar:
        raise F2MetricCacheManifestError("sidecar partial failed byte verification")
    sidecar_published = False
    target_published = False
    try:
        os.link(sidecar_partial, sidecar)
        sidecar_published = True
        _fsync_parent(sidecar)
        os.link(manifest_partial, target)
        target_published = True
        _fsync_parent(target)
    except BaseException:
        if sidecar_published and not target_published:
            try:
                if sidecar.stat().st_ino == sidecar_partial.stat().st_ino:
                    sidecar.unlink()
            except OSError:
                pass
        raise
    finally:
        for partial in (manifest_partial, sidecar_partial):
            try:
                partial.unlink()
            except OSError:
                pass
    return digest


_IDENTITY_CACHE: Dict[Tuple[str, str, str], Tuple[int, int, int, int, int, int]] = {}
_IDENTITY_CACHE_LOCK = threading.RLock()
_INODE_OWNERS: Dict[Tuple[str, int, int], str] = {}


@contextmanager
def _open_authenticated_entry(
    root: Path,
    manifest_sha256: str,
    entry: MetricCacheContentEntry,
    *,
    force_rehash: bool = False,
) -> Iterator[BinaryIO]:
    stream: Optional[BinaryIO] = None
    name = f"metric-cache pickle {entry.token!r}"
    with _open_relative_no_symlinks(root, entry.relative_path, name) as (
        descriptor,
        identity,
        revalidate,
    ):
        inode_key = (manifest_sha256, identity[0], identity[1])
        with _IDENTITY_CACHE_LOCK:
            inode_owner = _INODE_OWNERS.get(inode_key)
            if inode_owner is not None and inode_owner != entry.token:
                raise F2MetricCacheManifestError(
                    "qualified metric-cache tokens alias the same live inode"
                )
            _INODE_OWNERS[inode_key] = entry.token
        if identity[3] != entry.byte_size:
            raise F2MetricCacheManifestError(
                f"metric-cache pickle {entry.token!r} byte size differs"
            )
        cache_key = (manifest_sha256, entry.relative_path, entry.sha256)
        with _IDENTITY_CACHE_LOCK:
            cached = _IDENTITY_CACHE.get(cache_key)
        if force_rehash or cached != identity:
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
            revalidate()
            if digest.hexdigest() != entry.sha256:
                raise F2MetricCacheManifestError(
                    f"metric-cache pickle {entry.token!r} SHA-256 mismatch"
                )
            with _IDENTITY_CACHE_LOCK:
                _IDENTITY_CACHE[cache_key] = identity
        os.lseek(descriptor, 0, os.SEEK_SET)
        stream = os.fdopen(descriptor, "rb", closefd=False)
        try:
            yield stream
            if stream.closed:
                raise F2MetricCacheManifestError(
                    "authenticated metric-cache stream was closed by its consumer"
                )
            revalidate()
        finally:
            stream.close()


@dataclass(frozen=True)
class AuthenticatedMetricCacheReference(os.PathLike):
    """Pickle-safe reference consumed by spawned F2 scoring workers."""

    cache_root: str
    manifest_sha256: str
    entry: MetricCacheContentEntry

    def __post_init__(self) -> None:
        if not Path(self.cache_root).is_absolute():
            raise F2MetricCacheManifestError("authenticated cache root must be absolute")
        object.__setattr__(
            self,
            "manifest_sha256",
            _sha256(self.manifest_sha256, "metric-cache manifest SHA-256"),
        )
        if type(self.entry) is not MetricCacheContentEntry:
            raise F2MetricCacheManifestError("authenticated cache entry is malformed")

    @property
    def path(self) -> Path:
        return _join_relative(Path(self.cache_root), self.entry.relative_path)

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    @contextmanager
    def open_authenticated(self, *, force_rehash: bool = False) -> Iterator[BinaryIO]:
        with _open_authenticated_entry(
            Path(self.cache_root),
            self.manifest_sha256,
            self.entry,
            force_rehash=force_rehash,
        ) as stream:
            yield stream


@dataclass(frozen=True)
class VerifiedMetricCacheManifest:
    """Opaque structural verification result used by loaders and workers."""

    manifest: MetricCacheContentManifest
    manifest_sha256: str
    cache_root: Path
    metadata_path: Path
    _expected_identities: Tuple[Tuple[str, str, int], ...]
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _VERIFIED_SEAL:
            raise F2MetricCacheManifestError(
                "VerifiedMetricCacheManifest must come from load_and_verify_metric_cache_manifest()"
            )
        identities = tuple(self._expected_identities)
        if identities != tuple(sorted(identities, key=lambda value: value[0])):
            raise F2MetricCacheManifestError(
                "verified metric-cache identities are not canonically sorted"
            )
        expected_tokens = tuple(entry.token for entry in self.manifest.entries)
        identity_tokens = tuple(identity[0] for identity in identities)
        if identity_tokens != expected_tokens:
            raise F2MetricCacheManifestError(
                "verified metric-cache identities differ from manifest coverage"
            )
        for token, log_name, time_us in identities:
            _nonempty(token, "verified identity token")
            _nonempty(log_name, f"verified identity log for {token!r}")
            _nonnegative_int(time_us, f"verified identity time for {token!r}")

    def authenticated_references(self) -> Dict[str, AuthenticatedMetricCacheReference]:
        return {
            entry.token: AuthenticatedMetricCacheReference(
                cache_root=self.cache_root.as_posix(),
                manifest_sha256=self.manifest_sha256,
                entry=entry,
            )
            for entry in self.manifest.entries
        }

    def require_loader_mapping(
        self,
        mapping: Mapping[Any, Any],
        *,
        cache_root: Path,
        metadata_path: Optional[Path],
        metadata_sha256: Optional[str],
    ) -> Dict[str, AuthenticatedMetricCacheReference]:
        if _canonical_absolute_root(cache_root) != self.cache_root:
            raise F2MetricCacheManifestError("loader cache root differs from content manifest")
        if metadata_path is None:
            raise F2MetricCacheManifestError("loader metadata path differs from content manifest")
        loader_metadata, _, _ = _lexical_file_under_root(
            self.cache_root, Path(metadata_path), "loader metadata path"
        )
        if loader_metadata != self.metadata_path:
            raise F2MetricCacheManifestError("loader metadata path differs from content manifest")
        if metadata_sha256 != self.manifest.metadata_sha256:
            raise F2MetricCacheManifestError("loader metadata SHA-256 differs from content manifest")
        normalized = {
            str(token): _relative_to_root(self.cache_root, os.fspath(path))
            for token, path in mapping.items()
        }
        expected = {
            entry.token: entry.relative_path for entry in self.manifest.entries
        }
        missing = sorted(set(expected) - set(normalized))
        mismatched = sorted(
            token
            for token in expected.keys() & normalized.keys()
            if expected[token] != normalized[token]
        )
        if missing or mismatched:
            raise F2MetricCacheManifestError(
                "loader metadata does not cover the qualified cache manifest: "
                f"missing={missing[:8]}, mismatched={mismatched[:8]}"
            )
        return self.authenticated_references()

    def verify_all(self, workers: int = 8) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise F2MetricCacheManifestError("workers must be a positive integer")
        references_by_token = self.authenticated_references()
        expected = {
            token: (log_name, time_us)
            for token, log_name, time_us in self._expected_identities
        }
        if set(references_by_token) != set(expected):
            raise F2MetricCacheManifestError(
                "authenticated references differ from sealed qualification coverage"
            )
        references = tuple(
            references_by_token[token] for token in sorted(references_by_token)
        )

        def verify(reference: AuthenticatedMetricCacheReference) -> str:
            value = _load_authenticated_metric_cache(
                reference,
                force_rehash=True,
            )
            actual_identity = _metric_cache_identity(value, reference.entry.token)
            if actual_identity != expected[reference.entry.token]:
                raise F2MetricCacheManifestError(
                    "metric-cache pickle identity differs from qualification for "
                    f"token {reference.entry.token!r}: "
                    f"expected={expected[reference.entry.token]!r}, "
                    f"actual={actual_identity!r}"
                )
            return reference.entry.token

        with ThreadPoolExecutor(max_workers=workers) as executor:
            verified_tokens = tuple(executor.map(verify, references))
        if verified_tokens != tuple(sorted(expected)):
            raise F2MetricCacheManifestError(
                "metric-cache verify_all did not consume exact qualification coverage"
            )


def load_and_verify_metric_cache_manifest(
    path: Path,
    *,
    expected_sha256: str,
    cache_root: Path,
    metadata_path: Path,
    expected_metadata_sha256: str,
    qualification: QualificationManifest,
    expected_qualification_manifest_sha256: str,
) -> VerifiedMetricCacheManifest:
    """Close manifest, sidecar, qualification, CSV mapping, and live paths."""

    manifest_path = Path(path)
    payload = _stable_file_payload(manifest_path, "metric-cache content manifest")
    expected = _sha256(expected_sha256, "expected metric-cache manifest SHA-256")
    if _sha256_bytes(payload) != expected:
        raise F2MetricCacheManifestError("metric-cache content manifest SHA-256 mismatch")
    sidecar_payload = _stable_file_payload(
        metric_cache_manifest_sidecar_path(manifest_path),
        "metric-cache content manifest sidecar",
    )
    if sidecar_payload != _sidecar_bytes(manifest_path, expected):
        raise F2MetricCacheManifestError("metric-cache manifest sidecar differs")
    manifest = metric_cache_manifest_from_bytes(payload, expected_sha256=expected)
    root = _canonical_absolute_root(cache_root)
    if manifest.cache_root != root.as_posix():
        raise F2MetricCacheManifestError("live cache root differs from manifest")
    qualification_sha = _sha256(
        expected_qualification_manifest_sha256,
        "expected qualification manifest SHA-256",
    )
    if manifest.qualification_manifest_sha256 != qualification_sha:
        raise F2MetricCacheManifestError("qualification binding differs from cache manifest")
    if _sha256_bytes(canonical_qualification_manifest_bytes(qualification)) != qualification_sha:
        raise F2MetricCacheManifestError("live qualification bytes differ from cache manifest")
    partition_by_token = _qualification_partition_by_token(qualification)
    entries = manifest.by_token
    missing = sorted(set(partition_by_token) - set(entries))
    extra = sorted(set(entries) - set(partition_by_token))
    wrong_partition = sorted(
        token
        for token in entries.keys() & partition_by_token.keys()
        if entries[token].partition != partition_by_token[token]
    )
    if missing or extra or wrong_partition:
        raise F2MetricCacheManifestError(
            "metric-cache manifest qualification coverage differs: "
            f"missing={missing[:8]}, extra={extra[:8]}, partition={wrong_partition[:8]}"
        )
    live_metadata, live_metadata_relative, _ = _lexical_file_under_root(
        root, Path(metadata_path), "live metadata path"
    )
    embedded_metadata = _join_relative(root, manifest.metadata_relative_path)
    if (
        live_metadata != embedded_metadata
        or live_metadata_relative != manifest.metadata_relative_path
    ):
        raise F2MetricCacheManifestError("live metadata path differs from cache manifest")
    mapping, metadata_size = _metadata_mapping(
        root,
        live_metadata,
        expected_sha256=expected_metadata_sha256,
    )
    if (
        manifest.metadata_sha256
        != _sha256(expected_metadata_sha256, "expected metadata SHA-256")
        or manifest.metadata_byte_size != metadata_size
    ):
        raise F2MetricCacheManifestError("metadata identity differs from cache manifest")
    for token, entry in entries.items():
        if mapping.get(token) != entry.relative_path:
            raise F2MetricCacheManifestError(
                f"metadata mapping differs for qualified token {token!r}"
            )
    return VerifiedMetricCacheManifest(
        manifest=manifest,
        manifest_sha256=expected,
        cache_root=root,
        metadata_path=live_metadata,
        _expected_identities=tuple(
            sorted(
                (
                    row.token,
                    row.log_name,
                    row.time_us,
                )
                for partition_name in ("train", "held_out")
                for row in qualification.partition(partition_name).rows
            )
        ),
        _seal=_VERIFIED_SEAL,
    )


def _metric_cache_identity(value: Any, token: str) -> Tuple[str, int]:
    log_name = getattr(value, "log_name", None)
    timepoint = getattr(value, "timepoint", None)
    time_us = getattr(timepoint, "time_us", None)
    if (
        not isinstance(log_name, str)
        or not log_name
        or isinstance(time_us, bool)
        or not isinstance(time_us, int)
        or time_us < 0
    ):
        raise F2MetricCacheManifestError(
            f"metric-cache pickle for token {token!r} lacks a canonical log/time identity"
        )
    return log_name, time_us


def _load_authenticated_metric_cache(
    reference: AuthenticatedMetricCacheReference,
    *,
    force_rehash: bool,
) -> Any:
    if type(reference) is not AuthenticatedMetricCacheReference:
        raise F2MetricCacheManifestError(
            "F2 scoring requires an AuthenticatedMetricCacheReference"
        )
    try:
        with reference.open_authenticated(force_rehash=force_rehash) as raw:
            with lzma.LZMAFile(raw, mode="rb") as compressed:
                value = pickle.load(compressed)
                if compressed.read(1) != b"":
                    raise F2MetricCacheManifestError(
                        f"metric-cache pickle for token {reference.entry.token!r} "
                        "contains trailing decompressed data"
                    )
                return value
    except F2MetricCacheManifestError:
        raise
    except Exception as error:
        raise F2MetricCacheManifestError(
            f"metric-cache pickle for token {reference.entry.token!r} "
            "failed complete LZMA/pickle decoding"
        ) from error


def load_authenticated_metric_cache(
    reference: AuthenticatedMetricCacheReference,
) -> Any:
    """Authenticate and deserialize one complete LZMA pickle through one descriptor."""

    return _load_authenticated_metric_cache(reference, force_rehash=False)


def score_authenticated_metric_caches(arguments: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Spawn-safe adapter around the unchanged official Drive-JEPA scorer."""

    from navsim.agents.drive_jepa_perception_based.score_module import (
        compute_navsim_score,
    )

    if not arguments:
        return []
    for argument in arguments:
        if not isinstance(argument, Mapping) or type(argument.get("token")) is not AuthenticatedMetricCacheReference:
            raise F2MetricCacheManifestError(
                "F2 scorer received a non-authenticated metric-cache argument"
            )
    original_loader = compute_navsim_score.load_metric_cache

    def authenticated_loader(value: Any) -> Any:
        return load_authenticated_metric_cache(value)

    compute_navsim_score.load_metric_cache = authenticated_loader
    try:
        return compute_navsim_score.get_scores(arguments)
    finally:
        compute_navsim_score.load_metric_cache = original_loader


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an immutable content manifest for qualified navtrain caches"
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--qualification-sha256", required=True)
    parser.add_argument("--qualification-config-sha256", required=True)
    parser.add_argument("--qualification-source-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    args = parser.parse_args(argv)

    qualification = load_qualification_manifest(
        args.qualification_manifest,
        expected_manifest_sha256=args.qualification_sha256,
        expected_config_sha256=args.qualification_config_sha256,
        expected_cache_sha256=args.metadata_sha256,
        expected_source_sha256=args.qualification_source_sha256,
    )
    manifest = build_metric_cache_manifest(
        cache_root=args.cache_root,
        metadata_path=args.metadata,
        expected_metadata_sha256=args.metadata_sha256,
        qualification=qualification,
        expected_qualification_manifest_sha256=args.qualification_sha256,
        workers=args.workers,
    )
    digest = write_metric_cache_manifest(args.destination.resolve(), manifest)
    print(
        json.dumps(
            {
                "byte_size": len(canonical_metric_cache_manifest_bytes(manifest)),
                "entry_count": len(manifest.entries),
                "manifest": str(args.destination.resolve()),
                "sha256": digest,
                "sidecar": str(metric_cache_manifest_sidecar_path(args.destination.resolve())),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())


__all__ = (
    "AuthenticatedMetricCacheReference",
    "F2MetricCacheManifestError",
    "F2_PRUNED_CACHE_FAMILY",
    "METRIC_CACHE_MANIFEST_SCHEMA_VERSION",
    "MetricCacheContentEntry",
    "MetricCacheContentManifest",
    "VerifiedMetricCacheManifest",
    "build_metric_cache_manifest",
    "canonical_metric_cache_manifest_bytes",
    "cli_main",
    "load_and_verify_metric_cache_manifest",
    "load_authenticated_metric_cache",
    "metric_cache_manifest_from_bytes",
    "metric_cache_manifest_sha256",
    "metric_cache_manifest_sidecar_path",
    "score_authenticated_metric_caches",
    "write_metric_cache_manifest",
)
