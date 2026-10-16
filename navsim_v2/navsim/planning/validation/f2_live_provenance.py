"""Byte-authenticated provenance closure for F2 artifact build and validation.

F2 identities are useful only when they are recomputed from the files that the
current process will actually consume.  This module therefore has no API that
turns caller-supplied digest strings directly into an authenticated identity.
It hashes the qualification/causal manifests, configs, cache metadata, and
checkpoints; verifies every file named by the archived qualification source
manifest; and verifies an exact-scope runtime source manifest against the live
source tree.

The qualification source digest and runtime source digest intentionally have
different meanings.  The former authenticates the code snapshot that produced
the already-frozen sensor qualification.  The latter authenticates the code
that builds the binary artifact and executes causal evaluation, including the
record builder, comfort adapter, and shard runner added after qualification.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from navsim.planning.training.f2_qualification_manifest import (
    QualificationManifest,
    manifest_from_bytes as qualification_manifest_from_bytes,
)
from navsim.planning.validation.f2_causal_validator import (
    CanonicalManifest,
    manifest_from_bytes as causal_manifest_from_bytes,
)
from navsim.planning.validation.f2_frozen_artifact_io import ArtifactBindings
from navsim.planning.validation.f2_metric_cache_manifest import (
    VerifiedMetricCacheManifest,
    load_and_verify_metric_cache_manifest,
)


RUNTIME_SOURCE_SCHEMA_VERSION = "actworld-f2-runtime-source-files-v1"
QUALIFICATION_SOURCE_SCHEMA_VERSION = "actworld-f2-qualification-source-files-v1"
DEFAULT_RUNTIME_SOURCE_SCOPES = (
    ("navsim_v2/data", (".npy",)),
    ("navsim_v2/navsim", (".py", ".yaml", ".yml")),
    ("navsim_v2/scripts/training", (".sh",)),
    ("navsim_v2/vjepa2/configs/eval/vitl", (".yaml", ".yml")),
    (
        "navsim_v2/vjepa2/evals/image_classification_frozen/modelcustom",
        (".py",),
    ),
    ("navsim_v2/vjepa2/src/masks", (".py",)),
    ("navsim_v2/vjepa2/src/models", (".py",)),
    ("navsim_v2/vjepa2/src/utils", (".py",)),
)

VJEPA_RUNTIME_SOURCE_PATHS = (
    "navsim_v2/vjepa2/configs/eval/vitl/in1k.yaml",
    "navsim_v2/vjepa2/evals/image_classification_frozen/modelcustom/vit_encoder.py",
    "navsim_v2/vjepa2/src/masks/utils.py",
    "navsim_v2/vjepa2/src/models/utils/modules.py",
    "navsim_v2/vjepa2/src/models/utils/patch_embed.py",
    "navsim_v2/vjepa2/src/models/utils/pos_embs.py",
    "navsim_v2/vjepa2/src/models/vision_transformer.py",
    "navsim_v2/vjepa2/src/utils/tensors.py",
)

PROPOSAL_COMPONENT = "proposal_schema"
SIMULATOR_COMPONENT = "dual_simulator"
SCORER_COMPONENT = "training_scorer_and_comfort"
EXPECTED_PROPOSAL_SCHEMA_NAME = "actworld-f2-original-proposals-f32le-v1"
EXPECTED_SIMULATOR_NAME = (
    "navsim-train-simulation-v2+navsim-standard-simulation"
)
EXPECTED_SCORER_NAME = (
    "drive-jepa-train-pdm-scorer-nine-channel+navsim-two-frame-extended-comfort"
)

PROPOSAL_COMPONENT_PATHS = (
    "navsim_v2/navsim/planning/validation/f2_causal_model_adapter.py",
)
SIMULATOR_COMPONENT_PATHS = tuple(
    f"navsim_v2/navsim/planning/simulation/planner/pdm_planner/{family}/{name}"
    for family in ("simulation", "simulation_v2")
    for name in (
        "__init__.py",
        "batch_kinematic_bicycle.py",
        "batch_lqr.py",
        "batch_lqr_utils.py",
        "pdm_simulator.py",
    )
)
SCORER_COMPONENT_PATHS = (
    "navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/train_pdm_scorer.py",
    "navsim_v2/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py",
    "navsim_v2/navsim/planning/validation/f2_navsim_comfort.py",
)
EXPECTED_COMPONENT_PATHS = {
    PROPOSAL_COMPONENT: PROPOSAL_COMPONENT_PATHS,
    SIMULATOR_COMPONENT: SIMULATOR_COMPONENT_PATHS,
    SCORER_COMPONENT: SCORER_COMPONENT_PATHS,
}

REQUIRED_RUNTIME_SOURCE_PATHS = tuple(
    sorted(
        {
            "navsim_v2/navsim/planning/training/f2_qualification_manifest.py",
            "navsim_v2/navsim/planning/script/f2_launch_guard.py",
            "navsim_v2/navsim/planning/validation/f2_causal_validator.py",
            "navsim_v2/navsim/planning/validation/f2_causal_model_adapter.py",
            "navsim_v2/navsim/planning/validation/f2_frozen_artifact_io.py",
            "navsim_v2/navsim/planning/validation/f2_artifact_production.py",
            "navsim_v2/navsim/planning/validation/f2_live_provenance.py",
            "navsim_v2/navsim/planning/validation/f2_metric_cache_manifest.py",
            "navsim_v2/navsim/planning/validation/f2_navtrain_input_manifest.py",
            "navsim_v2/navsim/planning/validation/f2_navsim_record_builder.py",
            "navsim_v2/navsim/planning/validation/f2_navsim_runtime_adapters.py",
            "navsim_v2/navsim/planning/validation/f2_navsim_comfort.py",
            "navsim_v2/navsim/planning/validation/f2_causal_shard_runner.py",
            "navsim_v2/navsim/planning/validation/f2_production_runner.py",
            "navsim_v2/navsim/common/dataclasses.py",
            "navsim_v2/navsim/common/dataloader.py",
            "navsim_v2/navsim/agents/actworld_jepa/actworld_agent.py",
            "navsim_v2/navsim/agents/actworld_jepa/actworld_config.py",
            "navsim_v2/navsim/agents/actworld_jepa/actworld_features.py",
            "navsim_v2/navsim/agents/actworld_jepa/actworld_model.py",
            "navsim_v2/navsim/agents/actworld_jepa/actworld_planner.py",
            "navsim_v2/data/8192.npy",
            "navsim_v2/scripts/training/train_actworld_jepa_f2.sh",
            *VJEPA_RUNTIME_SOURCE_PATHS,
            *PROPOSAL_COMPONENT_PATHS,
            *SIMULATOR_COMPONENT_PATHS,
            *SCORER_COMPONENT_PATHS,
        }
    )
)

_COMPONENT_DOMAIN = b"ActWorld-JEPA/F2/source-component/v1\0"
_VERIFIED_SEAL = object()
_IMPORT_SHADOW_SUFFIXES = tuple(
    sorted(
        {
            ".pyc",
            ".pyd",
            ".so",
            *(suffix.lower() for suffix in importlib.machinery.EXTENSION_SUFFIXES),
        }
    )
)


class F2LiveProvenanceError(RuntimeError):
    """Raised when a live byte or provenance relationship is not exact."""


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
        raise F2LiveProvenanceError(f"{name} must be a 64-character SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise F2LiveProvenanceError(
            f"{name} must use lowercase hexadecimal"
        )
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise F2LiveProvenanceError(f"{name} must be a non-empty string")
    return value


def _canonical_relative_path(value: Any, name: str) -> str:
    raw = _nonempty(value, name)
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise F2LiveProvenanceError(f"{name} must be a canonical relative path")
    canonical = pure.as_posix()
    if canonical != raw or canonical.startswith("/"):
        raise F2LiveProvenanceError(f"{name} is not canonical POSIX form")
    return canonical


def _reject_constant(value: str) -> None:
    raise F2LiveProvenanceError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise F2LiveProvenanceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_json_file(
    path: Path, expected_sha256: str, name: str
) -> Tuple[Mapping[str, Any], bytes]:
    path = Path(path)
    payload = _stable_file_payload(path, name)
    if _sha256_bytes(payload) != _sha256(expected_sha256, f"expected {name}"):
        raise F2LiveProvenanceError(f"{name} SHA-256 mismatch")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except F2LiveProvenanceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F2LiveProvenanceError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict) or _canonical_json_bytes(decoded) != payload:
        raise F2LiveProvenanceError(f"{name} is not canonical JSON")
    return decoded, payload


def _stat_identity(value: os.stat_result) -> Tuple[int, int, int, int, int, int]:
    """Filesystem identity used to bind a pathname to one open descriptor."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file_identity(path: Path, name: str) -> Tuple[int, str]:
    """Hash a regular non-symlink file while detecting replacement mid-read."""

    path = Path(path)
    try:
        before_lstat = path.lstat()
    except OSError as error:
        raise F2LiveProvenanceError(f"cannot stat {name}: {path}") from error
    if stat.S_ISLNK(before_lstat.st_mode) or not stat.S_ISREG(before_lstat.st_mode):
        raise F2LiveProvenanceError(f"{name} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if _stat_identity(before) != _stat_identity(before_lstat):
                raise F2LiveProvenanceError(
                    f"{name} changed before it was authenticated"
                )
            for payload in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
                digest.update(payload)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        final_lstat = path.lstat()
    except F2LiveProvenanceError:
        raise
    except OSError as error:
        raise F2LiveProvenanceError(f"cannot read stable bytes for {name}: {path}") from error
    identity_before = _stat_identity(before)
    if (
        identity_before != _stat_identity(before_lstat)
        or identity_before != _stat_identity(after)
        or identity_before != _stat_identity(final_lstat)
    ):
        raise F2LiveProvenanceError(f"{name} changed while it was authenticated")
    return before.st_size, digest.hexdigest()


def _stable_file_payload(
    path: Path, name: str, *, maximum_bytes: int = 128 * 1024 * 1024
) -> bytes:
    """Read a small provenance file once from a stable, non-symlink descriptor."""

    path = Path(path)
    try:
        before_lstat = path.lstat()
    except OSError as error:
        raise F2LiveProvenanceError(f"cannot stat {name}: {path}") from error
    if stat.S_ISLNK(before_lstat.st_mode) or not stat.S_ISREG(before_lstat.st_mode):
        raise F2LiveProvenanceError(f"{name} must be a regular non-symlink file")
    if before_lstat.st_size > maximum_bytes:
        raise F2LiveProvenanceError(f"{name} exceeds the provenance size limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    chunks = []
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if _stat_identity(before) != _stat_identity(before_lstat):
                raise F2LiveProvenanceError(f"{name} changed before it was read")
            for payload in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
                chunks.append(payload)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        final_lstat = path.lstat()
    except F2LiveProvenanceError:
        raise
    except OSError as error:
        raise F2LiveProvenanceError(f"cannot read stable bytes for {name}: {path}") from error
    before_identity = _stat_identity(before)
    result = b"".join(chunks)
    if (
        before_identity != _stat_identity(before_lstat)
        or before_identity != _stat_identity(after)
        or before_identity != _stat_identity(final_lstat)
        or len(result) != before.st_size
    ):
        raise F2LiveProvenanceError(f"{name} changed while it was read")
    return result


def _path_under_root(root: Path, relative: str, name: str) -> Path:
    root = Path(root).resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise F2LiveProvenanceError(f"{name} resolves outside its source root") from error
    if resolved != candidate.absolute():
        raise F2LiveProvenanceError(f"{name} traverses a symlink")
    return resolved


@dataclass(frozen=True)
class SourceFileIdentity:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _canonical_relative_path(self.path, "source path"))
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise F2LiveProvenanceError("source file bytes must be nonnegative")
        object.__setattr__(self, "sha256", _sha256(self.sha256, "source file SHA-256"))

    def to_payload(self) -> Dict[str, Any]:
        return {"bytes": self.bytes, "path": self.path, "sha256": self.sha256}


def _files_from_payload(payload: Any, name: str) -> Tuple[SourceFileIdentity, ...]:
    if not isinstance(payload, list):
        raise F2LiveProvenanceError(f"{name} files must be a list")
    files = []
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"bytes", "path", "sha256"}:
            raise F2LiveProvenanceError(f"{name} file row keys differ")
        files.append(SourceFileIdentity(row["path"], row["bytes"], row["sha256"]))
    paths = tuple(item.path for item in files)
    if len(paths) != len(set(paths)):
        raise F2LiveProvenanceError(f"{name} contains duplicate source paths")
    return tuple(files)


def _verify_files(root: Path, files: Sequence[SourceFileIdentity], name: str) -> None:
    for item in files:
        path = _path_under_root(root, item.path, f"{name} file {item.path!r}")
        byte_count, digest = _stable_file_identity(path, f"{name} file {item.path!r}")
        if byte_count != item.bytes or digest != item.sha256:
            raise F2LiveProvenanceError(
                f"live {name} file differs from its manifest: {item.path!r}"
            )


def verify_qualification_source_manifest(
    path: Path, *, expected_sha256: str, source_root: Path
) -> Tuple[SourceFileIdentity, ...]:
    """Verify the archived, listed-file qualification source manifest."""

    payload, raw = _canonical_json_file(
        path, expected_sha256, "qualification source manifest"
    )
    if set(payload) != {"files", "foundation_commit", "schema_version"}:
        raise F2LiveProvenanceError("qualification source manifest keys differ")
    if payload["schema_version"] != QUALIFICATION_SOURCE_SCHEMA_VERSION:
        raise F2LiveProvenanceError("qualification source manifest schema drifted")
    _nonempty(payload["foundation_commit"], "qualification foundation_commit")
    files = _files_from_payload(payload["files"], "qualification source manifest")
    if not files:
        raise F2LiveProvenanceError("qualification source manifest is empty")
    reconstructed = {
        "files": [item.to_payload() for item in files],
        "foundation_commit": payload["foundation_commit"],
        "schema_version": payload["schema_version"],
    }
    if _canonical_json_bytes(reconstructed) != raw:
        raise F2LiveProvenanceError(
            "qualification source manifest is not idempotently canonical"
        )
    _verify_files(source_root, files, "qualification source")
    return files


@dataclass(frozen=True)
class RuntimeSourceManifest:
    source_revision: str
    scopes: Tuple[Tuple[str, Tuple[str, ...]], ...]
    files: Tuple[SourceFileIdentity, ...]
    components: Tuple[Tuple[str, Tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        _nonempty(self.source_revision, "runtime source revision")
        scopes = tuple(self.scopes)
        if scopes != tuple(sorted(scopes)) or not scopes:
            raise F2LiveProvenanceError("runtime source scopes must be nonempty and sorted")
        normalized_scopes = []
        for root, suffixes in scopes:
            canonical_root = _canonical_relative_path(root, "runtime source scope")
            normalized_suffixes = tuple(suffixes)
            if normalized_suffixes != tuple(sorted(set(normalized_suffixes))):
                raise F2LiveProvenanceError("runtime scope suffixes must be unique and sorted")
            if not normalized_suffixes or any(
                not isinstance(suffix, str) or not suffix.startswith(".")
                for suffix in normalized_suffixes
            ):
                raise F2LiveProvenanceError("runtime scope suffixes are malformed")
            normalized_scopes.append((canonical_root, normalized_suffixes))
        if tuple(normalized_scopes) != DEFAULT_RUNTIME_SOURCE_SCOPES:
            raise F2LiveProvenanceError("runtime source scopes differ from the frozen scope")
        files = tuple(self.files)
        if files != tuple(sorted(files, key=lambda item: item.path)) or not files:
            raise F2LiveProvenanceError("runtime source files must be nonempty and sorted")
        paths = tuple(item.path for item in files)
        if len(paths) != len(set(paths)):
            raise F2LiveProvenanceError("runtime source manifest contains duplicate paths")
        missing = sorted(set(REQUIRED_RUNTIME_SOURCE_PATHS) - set(paths))
        if missing:
            raise F2LiveProvenanceError(
                f"runtime source manifest omits required F2 files: {missing[:8]}"
            )
        components = tuple(self.components)
        if tuple(name for name, _ in components) != tuple(sorted(EXPECTED_COMPONENT_PATHS)):
            raise F2LiveProvenanceError("runtime source components differ from the schema")
        for name, component_paths in components:
            normalized = tuple(component_paths)
            expected = tuple(sorted(EXPECTED_COMPONENT_PATHS[name]))
            if normalized != expected:
                raise F2LiveProvenanceError(f"runtime component {name!r} paths drifted")
            if set(normalized) - set(paths):
                raise F2LiveProvenanceError(f"runtime component {name!r} is not covered")
        object.__setattr__(self, "scopes", tuple(normalized_scopes))
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "components", components)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "components": {name: list(paths) for name, paths in self.components},
            "files": [item.to_payload() for item in self.files],
            "schema_version": RUNTIME_SOURCE_SCHEMA_VERSION,
            "scopes": [
                {"path": root, "suffixes": list(suffixes)}
                for root, suffixes in self.scopes
            ],
            "source_revision": self.source_revision,
        }

    def component_sha256(self, name: str) -> str:
        path_map = {item.path: item for item in self.files}
        component_map = dict(self.components)
        if name not in component_map:
            raise F2LiveProvenanceError(f"unknown runtime source component {name!r}")
        payload = _canonical_json_bytes(
            {
                "component": name,
                "files": [path_map[path].to_payload() for path in component_map[name]],
            }
        )
        return hashlib.sha256(_COMPONENT_DOMAIN + payload).hexdigest()


def _discover_scope_files(
    source_root: Path,
    scopes: Sequence[Tuple[str, Sequence[str]]],
) -> Tuple[str, ...]:
    discovered = set()
    for relative_root, suffixes in scopes:
        scope_root = _path_under_root(source_root, relative_root, "runtime source scope")
        for path in scope_root.rglob("*"):
            if path.is_symlink():
                raise F2LiveProvenanceError("runtime source scope contains a symlink")
            if not path.is_file():
                continue
            if path.name.lower().endswith(_IMPORT_SHADOW_SUFFIXES):
                raise F2LiveProvenanceError(
                    "runtime source scope contains an import-shadow artifact: "
                    f"{path.relative_to(Path(source_root).resolve()).as_posix()!r}"
                )
            if "__pycache__" in path.parts:
                continue
            if path.suffix not in suffixes:
                continue
            discovered.add(path.relative_to(Path(source_root).resolve()).as_posix())
    return tuple(sorted(discovered))


def build_runtime_source_manifest(
    source_root: Path,
    *,
    source_revision: str,
) -> RuntimeSourceManifest:
    """Build, but do not write, an exact-scope manifest from live source bytes."""

    source_root = Path(source_root).resolve(strict=True)
    paths = _discover_scope_files(source_root, DEFAULT_RUNTIME_SOURCE_SCOPES)
    files = []
    for relative in paths:
        byte_count, digest = _stable_file_identity(
            _path_under_root(source_root, relative, "runtime source file"),
            f"runtime source file {relative!r}",
        )
        files.append(SourceFileIdentity(relative, byte_count, digest))
    return RuntimeSourceManifest(
        source_revision=source_revision,
        scopes=DEFAULT_RUNTIME_SOURCE_SCOPES,
        files=tuple(files),
        components=tuple(
            (name, tuple(sorted(paths)))
            for name, paths in sorted(EXPECTED_COMPONENT_PATHS.items())
        ),
    )


def canonical_runtime_source_manifest_bytes(manifest: RuntimeSourceManifest) -> bytes:
    if not isinstance(manifest, RuntimeSourceManifest):
        raise F2LiveProvenanceError("manifest must be RuntimeSourceManifest")
    return _canonical_json_bytes(manifest.to_payload())


def load_and_verify_runtime_source_manifest(
    path: Path, *, expected_sha256: str, source_root: Path
) -> RuntimeSourceManifest:
    payload, raw = _canonical_json_file(
        path, expected_sha256, "runtime source manifest"
    )
    required = {"components", "files", "schema_version", "scopes", "source_revision"}
    if set(payload) != required or payload["schema_version"] != RUNTIME_SOURCE_SCHEMA_VERSION:
        raise F2LiveProvenanceError("runtime source manifest schema drifted")
    if not isinstance(payload["scopes"], list):
        raise F2LiveProvenanceError("runtime source scopes must be a list")
    scopes = []
    for row in payload["scopes"]:
        if not isinstance(row, dict) or set(row) != {"path", "suffixes"}:
            raise F2LiveProvenanceError("runtime source scope keys differ")
        if not isinstance(row["suffixes"], list):
            raise F2LiveProvenanceError("runtime source suffixes must be a list")
        scopes.append((row["path"], tuple(row["suffixes"])))
    if not isinstance(payload["components"], dict):
        raise F2LiveProvenanceError("runtime source components must be an object")
    components = tuple(
        (name, tuple(paths)) for name, paths in sorted(payload["components"].items())
    )
    manifest = RuntimeSourceManifest(
        source_revision=payload["source_revision"],
        scopes=tuple(scopes),
        files=_files_from_payload(payload["files"], "runtime source manifest"),
        components=components,
    )
    if canonical_runtime_source_manifest_bytes(manifest) != raw:
        raise F2LiveProvenanceError(
            "runtime source manifest is not idempotently canonical"
        )
    discovered = _discover_scope_files(Path(source_root).resolve(strict=True), manifest.scopes)
    declared = tuple(item.path for item in manifest.files)
    if discovered != declared:
        missing = sorted(set(discovered) - set(declared))
        extra = sorted(set(declared) - set(discovered))
        raise F2LiveProvenanceError(
            "runtime source exact-scope coverage drifted: "
            f"undeclared={missing[:8]}, absent={extra[:8]}"
        )
    _verify_files(source_root, manifest.files, "runtime source")
    return manifest


@dataclass(frozen=True)
class VerifiedF2Provenance:
    """Opaque result of complete live verification; direct construction fails."""

    bindings: ArtifactBindings
    qualification: QualificationManifest
    causal: CanonicalManifest
    metric_cache: VerifiedMetricCacheManifest
    runtime_source: RuntimeSourceManifest
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _VERIFIED_SEAL:
            raise F2LiveProvenanceError(
                "VerifiedF2Provenance must come from verify_live_f2_provenance()"
            )
        if type(self.bindings) is not ArtifactBindings:
            raise F2LiveProvenanceError("verified provenance bindings are malformed")
        if type(self.metric_cache) is not VerifiedMetricCacheManifest:
            raise F2LiveProvenanceError("verified metric-cache provenance is malformed")

    def require_bindings(self, bindings: ArtifactBindings) -> None:
        if bindings != self.bindings:
            raise F2LiveProvenanceError("runtime bindings differ from live provenance")


def _require_direct_file(path: Path, expected_sha256: str, name: str) -> None:
    _, actual = _stable_file_identity(Path(path), name)
    if actual != _sha256(expected_sha256, f"expected {name}"):
        raise F2LiveProvenanceError(f"live {name} SHA-256 mismatch")


def verify_live_f2_provenance(
    *,
    bindings: ArtifactBindings,
    qualification_manifest_path: Path,
    causal_manifest_path: Path,
    qualification_config_path: Path,
    runtime_config_path: Path,
    cache_metadata_path: Path,
    metric_cache_manifest_path: Path,
    metric_cache_root: Path,
    baseline_checkpoint_path: Path,
    vjepa_checkpoint_path: Path,
    qualification_source_manifest_path: Path,
    qualification_source_root: Path,
    runtime_source_manifest_path: Path,
    runtime_source_root: Path,
) -> VerifiedF2Provenance:
    """Recompute every bound identity from live bytes and close all joins."""

    if type(bindings) is not ArtifactBindings:
        raise F2LiveProvenanceError("bindings must be ArtifactBindings")
    _require_direct_file(
        qualification_config_path,
        bindings.qualification_config_sha256,
        "qualification config",
    )
    _require_direct_file(runtime_config_path, bindings.runtime_config_sha256, "runtime config")
    _require_direct_file(cache_metadata_path, bindings.cache_metadata_sha256, "cache metadata")
    _require_direct_file(
        baseline_checkpoint_path,
        bindings.baseline_checkpoint_sha256,
        "baseline checkpoint",
    )
    _require_direct_file(
        vjepa_checkpoint_path,
        bindings.vjepa_checkpoint_sha256,
        "V-JEPA checkpoint",
    )
    verify_qualification_source_manifest(
        qualification_source_manifest_path,
        expected_sha256=bindings.qualification_source_manifest_sha256,
        source_root=qualification_source_root,
    )
    runtime_source = load_and_verify_runtime_source_manifest(
        runtime_source_manifest_path,
        expected_sha256=bindings.runtime_source_manifest_sha256,
        source_root=runtime_source_root,
    )
    if runtime_source.source_revision != bindings.source_revision:
        raise F2LiveProvenanceError("live runtime source revision differs from bindings")
    expected_names = {
        "proposal schema": (
            bindings.proposal_schema_name,
            EXPECTED_PROPOSAL_SCHEMA_NAME,
        ),
        "simulator": (bindings.simulator_name, EXPECTED_SIMULATOR_NAME),
        "scorer": (bindings.scorer_name, EXPECTED_SCORER_NAME),
    }
    for name, (actual, expected) in expected_names.items():
        if actual != expected:
            raise F2LiveProvenanceError(
                f"{name} identity name differs from the frozen runtime contract"
            )
    component_expectations = {
        PROPOSAL_COMPONENT: bindings.proposal_schema_sha256,
        SIMULATOR_COMPONENT: bindings.simulator_sha256,
        SCORER_COMPONENT: bindings.scorer_sha256,
    }
    for name, expected in component_expectations.items():
        if runtime_source.component_sha256(name) != expected:
            raise F2LiveProvenanceError(f"live runtime component {name!r} differs")

    qualification_bytes = _stable_file_payload(
        Path(qualification_manifest_path), "qualification manifest"
    )
    qualification = qualification_manifest_from_bytes(
        qualification_bytes,
        expected_manifest_sha256=bindings.qualification_manifest_sha256,
        expected_config_sha256=bindings.qualification_config_sha256,
        expected_cache_sha256=bindings.cache_metadata_sha256,
        expected_source_sha256=bindings.qualification_source_manifest_sha256,
    )
    metric_cache = load_and_verify_metric_cache_manifest(
        metric_cache_manifest_path,
        expected_sha256=bindings.metric_cache_manifest_sha256,
        cache_root=metric_cache_root,
        metadata_path=cache_metadata_path,
        expected_metadata_sha256=bindings.cache_metadata_sha256,
        qualification=qualification,
        expected_qualification_manifest_sha256=(
            bindings.qualification_manifest_sha256
        ),
    )
    causal_bytes = _stable_file_payload(Path(causal_manifest_path), "causal manifest")
    if _sha256_bytes(causal_bytes) != bindings.causal_manifest_sha256:
        raise F2LiveProvenanceError("live causal manifest SHA-256 mismatch")
    causal = causal_manifest_from_bytes(
        causal_bytes,
        expected_sha256=bindings.causal_manifest_sha256,
        expected_tokens=qualification.held_out.tokens,
    )
    qualification_rows = tuple(
        (row.log_name, row.time_us, row.token, row.future_camera_valid)
        for row in qualification.held_out.rows
    )
    causal_rows = tuple(
        (row.log_name, row.time_us, row.token, row.future_camera_valid)
        for row in causal.rows
    )
    if qualification_rows != causal_rows:
        raise F2LiveProvenanceError("qualification and causal manifest rows differ")
    if qualification.sensor_contract != causal.sensor_contract:
        raise F2LiveProvenanceError("qualification and causal sensor contracts differ")
    if causal.cache_metadata_sha256 != bindings.cache_metadata_sha256:
        raise F2LiveProvenanceError("causal manifest cache metadata differs")
    if causal.source_revision != bindings.source_revision:
        raise F2LiveProvenanceError("causal manifest runtime source revision differs")
    return VerifiedF2Provenance(
        bindings=bindings,
        qualification=qualification,
        causal=causal,
        metric_cache=metric_cache,
        runtime_source=runtime_source,
        _seal=_VERIFIED_SEAL,
    )


__all__ = (
    "DEFAULT_RUNTIME_SOURCE_SCOPES",
    "EXPECTED_PROPOSAL_SCHEMA_NAME",
    "EXPECTED_SCORER_NAME",
    "EXPECTED_SIMULATOR_NAME",
    "EXPECTED_COMPONENT_PATHS",
    "F2LiveProvenanceError",
    "QUALIFICATION_SOURCE_SCHEMA_VERSION",
    "REQUIRED_RUNTIME_SOURCE_PATHS",
    "RuntimeSourceManifest",
    "SourceFileIdentity",
    "VerifiedF2Provenance",
    "VJEPA_RUNTIME_SOURCE_PATHS",
    "build_runtime_source_manifest",
    "canonical_runtime_source_manifest_bytes",
    "load_and_verify_runtime_source_manifest",
    "verify_live_f2_provenance",
    "verify_qualification_source_manifest",
)
