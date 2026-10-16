"""Production construction of the train-only F2 residual-feasibility audit.

This module deliberately runs before any F2 checkpoint, causal manifest, or
evaluation runtime exists.  Its only claim is an analytical capacity bound for
the frozen residual selector.  The production path authenticates the qualified
``navtrain`` inputs, the per-pickle metric-cache inventory, the official
Drive-JEPA/V-JEPA assets, and both factory and orchestration source before it
computes the deterministic 2048-row audit.

The CLI intentionally has no data-view selector.  A canonical pretraining
configuration, itself protected by a SHA-256 sidecar, is the only source of
runtime paths and independently verified identities.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from navsim.planning.training.f2_qualification_manifest import (
    F2_SENSOR_CONTRACT,
    QualificationManifest,
    constrain_scene_filter,
    manifest_from_bytes,
    qualify_scene_loader,
)
from navsim.planning.validation.f2_live_provenance import (
    verify_qualification_source_manifest,
)
from navsim.planning.validation.f2_causal_validator import (
    HARD_FACTOR_NAMES,
    causal_true_epdms,
)
from navsim.planning.validation.f2_metric_cache_manifest import (
    F2_PRUNED_CACHE_FAMILY,
    AuthenticatedMetricCacheReference,
    VerifiedMetricCacheManifest,
    load_and_verify_metric_cache_manifest,
    load_authenticated_metric_cache,
)
from navsim.planning.validation.f2_navtrain_input_manifest import (
    VerifiedNavtrainRawInputs,
    load_and_verify_raw_input_manifest,
)
from navsim.planning.validation.f2_navsim_record_builder import (
    DriveJepaNavsimRecordSemantics,
)
from navsim.planning.validation.f2_navsim_runtime_adapters import (
    OFFICIAL_DRIVE_JEPA_CHECKPOINT_NAME,
    OFFICIAL_DRIVE_JEPA_FACTORY_REVISION,
    OFFICIAL_DRIVE_JEPA_VJEPA_NAME,
    OFFICIAL_DRIVE_JEPA_VJEPA_SHA256,
    DriveJepaFactorySourceIdentity,
    RestoredDriveJepaBaseline,
    _official_drive_jepa_model_factory,
    _validate_unbatched_drive_jepa_features,
    authenticate_drive_jepa_factory_source,
    f2_sensor_config,
    restore_authenticated_drive_jepa_baseline,
)
from navsim.planning.validation.f2_production_runner import (
    F2ProductionRunnerError,
    _read_stable_regular_file,
    _validate_deterministic_runtime_payload,
    canonical_json_bytes,
    initialize_deterministic_runtime,
    parse_canonical_json_bytes,
    read_bytes_with_sidecar,
    write_bytes_no_overwrite,
)
from navsim.planning.validation.f2_residual_feasibility import (
    FEASIBILITY_DATASET,
    FEASIBILITY_PARTITION,
    FEASIBILITY_SAMPLE_SIZE,
    FEASIBILITY_SCHEMA_VERSION,
    F2ResidualFeasibilityAudit,
    FrozenStaticTokenInput,
    TrainSampleRow,
    analyze_tanh_residual_reachability,
    build_train_only_residual_feasibility_audit,
    feasibility_audit_sha256,
    select_deterministic_navtrain_train_sample,
)


PRETRAINING_CONFIG_SCHEMA_VERSION = (
    "actworld-f2-feasibility-pretraining-config-v2"
)
PRODUCTION_AUDIT_SCHEMA_VERSION = "actworld-f2-feasibility-production-audit-v3"
PRODUCTION_AUDIT_KIND = "f2_train_residual_analytical_capacity_audit"
PRODUCTION_CLAIM_SCOPE = (
    "analytical_train_only_capacity_not_checkpoint_or_evaluation_score"
)
PRETRAINING_PURPOSE = "f2_residual_feasibility_prelaunch"
PRODUCTION_DEVICE = "cuda:0"
CAPACITY_GATE_METRIC = "aggregate.reachable_delta_upper"
CAPACITY_GATE_OPERATOR = ">"
CAPACITY_GATE_THRESHOLD = 0.01
CAPACITY_GATE_UNIT = "raw_score_point"

# Fixed paper-baseline asset.  It is repeated here rather than imported from
# the ActWorld agent so this pretraining audit never constructs or depends on
# an F2 checkpoint class.
OFFICIAL_DRIVE_JEPA_BASELINE_SHA256 = (
    "740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05"
)

# Files/directories whose live bytes define this production audit.  The
# official Drive-JEPA factory is authenticated separately at its own frozen
# source revision.
FEASIBILITY_CODE_PATHS = (
    "navsim_v2/navsim/agents/actworld_jepa/actworld_agent.py",
    "navsim_v2/navsim/agents/actworld_jepa/actworld_model.py",
    "navsim_v2/navsim/agents/actworld_jepa/actworld_planner.py",
    "navsim_v2/navsim/common/dataclasses.py",
    "navsim_v2/navsim/common/dataloader.py",
    "navsim_v2/navsim/planning/training/f2_qualification_manifest.py",
    "navsim_v2/navsim/planning/validation/f2_causal_validator.py",
    "navsim_v2/navsim/planning/validation/f2_feasibility_production.py",
    "navsim_v2/navsim/planning/validation/f2_live_provenance.py",
    "navsim_v2/navsim/planning/validation/f2_metric_cache_manifest.py",
    "navsim_v2/navsim/planning/validation/f2_navtrain_input_manifest.py",
    "navsim_v2/navsim/planning/validation/f2_navsim_record_builder.py",
    "navsim_v2/navsim/planning/validation/f2_navsim_runtime_adapters.py",
    "navsim_v2/navsim/planning/validation/f2_production_runner.py",
    "navsim_v2/navsim/planning/validation/f2_residual_feasibility.py",
)

_CONFIG_PATH_KEYS = {
    "baseline_checkpoint",
    "experiment_root",
    "metric_cache_manifest",
    "metric_cache_metadata",
    "metric_cache_root",
    "navsim_log",
    "qualification_config",
    "qualification_manifest",
    "qualification_source_manifest",
    "qualification_source_root",
    "raw_input_manifest",
    "repository_root",
    "sensor_blobs",
    "vjepa_checkpoint",
}
_CONFIG_DIGEST_KEYS = {
    "baseline_checkpoint_sha256",
    "metric_cache_manifest_sha256",
    "metric_cache_metadata_sha256",
    "qualification_config_sha256",
    "qualification_manifest_sha256",
    "qualification_source_manifest_sha256",
    "raw_input_manifest_sha256",
    "vjepa_checkpoint_sha256",
}
_CONFIG_SOURCE_KEYS = {
    "drive_jepa_factory_revision",
    "production_code_revision",
}
_TOP_LEVEL_CERTIFICATE_KEYS = {
    "audit",
    "audit_sha256",
    "capacity_gate",
    "claim_scope",
    "dataset",
    "kind",
    "pretraining_config_sha256",
    "provenance",
    "schema_version",
}
_GATE_KEYS = {"metric", "operator", "passed", "threshold", "unit", "value"}
_PROVENANCE_KEYS = {
    "baseline",
    "code_source",
    "deterministic_runtime",
    "feature_sample",
    "metric_cache",
    "qualification",
    "raw_input",
}
_BASELINE_PROVENANCE_KEYS = {
    "checkpoint_sha256",
    "factory_path_tree_sha256",
    "factory_revision",
    "state_tensor_count",
    "vjepa_checkpoint_sha256",
}
_CODE_SOURCE_PROVENANCE_KEYS = {"path_tree_sha256", "paths", "revision"}
_FEATURE_SAMPLE_PROVENANCE_KEYS = {
    "removed_scenes",
    "row_identity_sha256",
    "token_count",
}
_METRIC_CACHE_PROVENANCE_KEYS = {
    "content_entry_count",
    "content_entries_sha256",
    "content_manifest_sha256",
    "metadata_sha256",
}
_QUALIFICATION_PROVENANCE_KEYS = {
    "config_sha256",
    "manifest_sha256",
    "source_file_count",
    "source_files_sha256",
    "source_manifest_sha256",
}
_RAW_INPUT_PROVENANCE_KEYS = {
    "content_manifest_sha256",
    "log_entry_count",
    "sensor_entry_count",
}


class F2FeasibilityProductionError(RuntimeError):
    """Raised when the prelaunch analytical audit cannot be trusted."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise F2FeasibilityProductionError(
            f"{name} must be exactly 64 hexadecimal SHA-256 characters"
        )
    if any(character not in "0123456789abcdef" for character in value):
        raise F2FeasibilityProductionError(
            f"{name} must be lowercase hexadecimal"
        )
    return value


def _revision(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise F2FeasibilityProductionError(
            f"{name} must be one full 40-character Git revision"
        )
    if any(character not in "0123456789abcdef" for character in value):
        raise F2FeasibilityProductionError(
            f"{name} must be lowercase hexadecimal"
        )
    return value


def _exact_keys(payload: Any, expected: set, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise F2FeasibilityProductionError(f"{name} must be a JSON object")
    actual = set(payload)
    if actual != expected:
        raise F2FeasibilityProductionError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return payload


def _absolute_path(value: Any, name: str) -> Path:
    if isinstance(value, Path):
        path = value.expanduser()
    elif isinstance(value, str) and value:
        path = Path(value).expanduser()
    else:
        raise F2FeasibilityProductionError(f"{name} must be a non-empty path")
    if not path.is_absolute():
        raise F2FeasibilityProductionError(f"{name} must be absolute")
    _reject_symlink_components(path, name=name, require_all=False)
    return path


def _reject_symlink_components(
    path: Path, *, name: str, require_all: bool
) -> None:
    """Reject lexical ``..`` and every existing symlink before resolution."""

    if not path.is_absolute():
        raise F2FeasibilityProductionError(f"{name} must be absolute")
    if any(part in (".", "..") for part in path.parts):
        raise F2FeasibilityProductionError(
            f"{name} cannot contain lexical dot components"
        )
    anchor = Path(path.anchor)
    remaining = path.parts[1:] if path.anchor else path.parts
    candidate = anchor
    for part in remaining:
        candidate = candidate / part
        try:
            identity = candidate.lstat()
        except FileNotFoundError as error:
            if require_all:
                raise F2FeasibilityProductionError(
                    f"{name} path component does not exist: {candidate}"
                ) from error
            break
        except OSError as error:
            raise F2FeasibilityProductionError(
                f"cannot inspect {name} path component: {candidate}"
            ) from error
        if stat.S_ISLNK(identity.st_mode) or candidate.is_symlink():
            raise F2FeasibilityProductionError(
                f"{name} path components must not be symlinks: {candidate}"
            )


def _resolved_directory(value: Path, name: str) -> Path:
    lexical = Path(value).expanduser()
    _reject_symlink_components(lexical, name=name, require_all=True)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise F2FeasibilityProductionError(f"cannot resolve {name}: {value}") from error
    if not resolved.is_dir():
        raise F2FeasibilityProductionError(f"{name} must be a directory: {resolved}")
    return resolved


def _resolved_file(value: Path, name: str) -> Path:
    lexical = Path(value).expanduser()
    _reject_symlink_components(lexical, name=name, require_all=True)
    try:
        resolved = lexical.resolve(strict=True)
        identity = resolved.lstat()
    except OSError as error:
        raise F2FeasibilityProductionError(f"cannot resolve {name}: {value}") from error
    if stat.S_ISLNK(identity.st_mode) or not stat.S_ISREG(identity.st_mode):
        raise F2FeasibilityProductionError(
            f"{name} must be a regular non-symlink file: {resolved}"
        )
    return resolved


def _hash_stable_file(path: Path, expected_sha256: str, name: str) -> bytes:
    payload = _read_stable_regular_file(path, name=name)
    expected = _sha256(expected_sha256, f"{name} SHA-256")
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise F2FeasibilityProductionError(
            f"{name} SHA-256 mismatch: expected={expected}, actual={actual}"
        )
    return payload


@dataclass(frozen=True)
class FeasibilityPretrainingConfig:
    """Closed, pre-F2 runtime configuration for one production audit."""

    qualification_manifest: Path
    qualification_config: Path
    qualification_source_manifest: Path
    qualification_source_root: Path
    raw_input_manifest: Path
    metric_cache_metadata: Path
    metric_cache_manifest: Path
    metric_cache_root: Path
    baseline_checkpoint: Path
    vjepa_checkpoint: Path
    repository_root: Path
    experiment_root: Path
    navsim_log: Path
    sensor_blobs: Path
    qualification_manifest_sha256: str
    qualification_config_sha256: str
    qualification_source_manifest_sha256: str
    raw_input_manifest_sha256: str
    metric_cache_metadata_sha256: str
    metric_cache_manifest_sha256: str
    baseline_checkpoint_sha256: str
    vjepa_checkpoint_sha256: str
    production_code_revision: str
    drive_jepa_factory_revision: str = OFFICIAL_DRIVE_JEPA_FACTORY_REVISION
    device: str = PRODUCTION_DEVICE
    sample_size: int = FEASIBILITY_SAMPLE_SIZE
    schema_version: str = PRETRAINING_CONFIG_SCHEMA_VERSION
    dataset: str = FEASIBILITY_DATASET
    purpose: str = PRETRAINING_PURPOSE

    def __post_init__(self) -> None:
        if self.schema_version != PRETRAINING_CONFIG_SCHEMA_VERSION:
            raise F2FeasibilityProductionError(
                "unsupported feasibility pretraining config schema"
            )
        if self.dataset != FEASIBILITY_DATASET or self.purpose != PRETRAINING_PURPOSE:
            raise F2FeasibilityProductionError(
                "feasibility pretraining config scope drifted"
            )
        if self.device != PRODUCTION_DEVICE:
            raise F2FeasibilityProductionError(
                f"production feasibility device must be {PRODUCTION_DEVICE!r}"
            )
        if isinstance(self.sample_size, bool) or self.sample_size != FEASIBILITY_SAMPLE_SIZE:
            raise F2FeasibilityProductionError(
                f"production feasibility sample_size must be {FEASIBILITY_SAMPLE_SIZE}"
            )

        for field_name in _CONFIG_PATH_KEYS:
            value = getattr(self, field_name)
            path = _absolute_path(value, field_name)
            object.__setattr__(self, field_name, path)
        for field_name in _CONFIG_DIGEST_KEYS:
            object.__setattr__(self, field_name, _sha256(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "production_code_revision",
            _revision(self.production_code_revision, "production_code_revision"),
        )
        factory_revision = _revision(
            self.drive_jepa_factory_revision, "drive_jepa_factory_revision"
        )
        if factory_revision != OFFICIAL_DRIVE_JEPA_FACTORY_REVISION:
            raise F2FeasibilityProductionError(
                "Drive-JEPA factory revision differs from the frozen official revision"
            )
        object.__setattr__(self, "drive_jepa_factory_revision", factory_revision)
        if self.baseline_checkpoint_sha256 != OFFICIAL_DRIVE_JEPA_BASELINE_SHA256:
            raise F2FeasibilityProductionError(
                "baseline digest differs from the frozen official Drive-JEPA checkpoint"
            )
        if self.vjepa_checkpoint_sha256 != OFFICIAL_DRIVE_JEPA_VJEPA_SHA256:
            raise F2FeasibilityProductionError(
                "V-JEPA digest differs from the frozen official checkpoint"
            )
        if self.baseline_checkpoint.name != OFFICIAL_DRIVE_JEPA_CHECKPOINT_NAME:
            raise F2FeasibilityProductionError(
                "baseline checkpoint file name differs from the official asset"
            )
        expected_vjepa = (
            self.experiment_root / "Drive-JEPA-cache" / OFFICIAL_DRIVE_JEPA_VJEPA_NAME
        )
        if self.vjepa_checkpoint.resolve() != expected_vjepa.resolve():
            raise F2FeasibilityProductionError(
                "V-JEPA checkpoint is not at the official experiment-root path"
            )
        if self.metric_cache_root.name != F2_PRUNED_CACHE_FAMILY:
            raise F2FeasibilityProductionError(
                "metric-cache root differs from the frozen pruned cache family"
            )


def pretraining_config_payload(config: FeasibilityPretrainingConfig) -> Dict[str, Any]:
    if type(config) is not FeasibilityPretrainingConfig:
        raise F2FeasibilityProductionError(
            "config must be concrete FeasibilityPretrainingConfig"
        )
    return {
        "dataset": config.dataset,
        "device": config.device,
        "digests": {
            name: getattr(config, name) for name in sorted(_CONFIG_DIGEST_KEYS)
        },
        "paths": {
            name: getattr(config, name).as_posix()
            for name in sorted(_CONFIG_PATH_KEYS)
        },
        "purpose": config.purpose,
        "sample_size": config.sample_size,
        "schema_version": config.schema_version,
        "source": {
            "drive_jepa_factory_revision": config.drive_jepa_factory_revision,
            "production_code_revision": config.production_code_revision,
        },
    }


def pretraining_config_bytes(config: FeasibilityPretrainingConfig) -> bytes:
    return canonical_json_bytes(pretraining_config_payload(config))


def parse_pretraining_config_bytes(payload: bytes) -> FeasibilityPretrainingConfig:
    try:
        decoded = parse_canonical_json_bytes(
            payload, name="F2 feasibility pretraining config"
        )
    except F2ProductionRunnerError as error:
        raise F2FeasibilityProductionError(str(error)) from error
    root = _exact_keys(
        decoded,
        {"dataset", "device", "digests", "paths", "purpose", "sample_size", "schema_version", "source"},
        "pretraining config",
    )
    paths = _exact_keys(root["paths"], _CONFIG_PATH_KEYS, "pretraining paths")
    digests = _exact_keys(
        root["digests"], _CONFIG_DIGEST_KEYS, "pretraining digests"
    )
    source = _exact_keys(root["source"], _CONFIG_SOURCE_KEYS, "pretraining source")
    try:
        config = FeasibilityPretrainingConfig(
            qualification_manifest=_absolute_path(paths["qualification_manifest"], "qualification_manifest"),
            qualification_config=_absolute_path(paths["qualification_config"], "qualification_config"),
            qualification_source_manifest=_absolute_path(paths["qualification_source_manifest"], "qualification_source_manifest"),
            qualification_source_root=_absolute_path(paths["qualification_source_root"], "qualification_source_root"),
            raw_input_manifest=_absolute_path(paths["raw_input_manifest"], "raw_input_manifest"),
            metric_cache_metadata=_absolute_path(paths["metric_cache_metadata"], "metric_cache_metadata"),
            metric_cache_manifest=_absolute_path(paths["metric_cache_manifest"], "metric_cache_manifest"),
            metric_cache_root=_absolute_path(paths["metric_cache_root"], "metric_cache_root"),
            baseline_checkpoint=_absolute_path(paths["baseline_checkpoint"], "baseline_checkpoint"),
            vjepa_checkpoint=_absolute_path(paths["vjepa_checkpoint"], "vjepa_checkpoint"),
            repository_root=_absolute_path(paths["repository_root"], "repository_root"),
            experiment_root=_absolute_path(paths["experiment_root"], "experiment_root"),
            navsim_log=_absolute_path(paths["navsim_log"], "navsim_log"),
            sensor_blobs=_absolute_path(paths["sensor_blobs"], "sensor_blobs"),
            qualification_manifest_sha256=digests["qualification_manifest_sha256"],
            qualification_config_sha256=digests["qualification_config_sha256"],
            qualification_source_manifest_sha256=digests["qualification_source_manifest_sha256"],
            raw_input_manifest_sha256=digests["raw_input_manifest_sha256"],
            metric_cache_metadata_sha256=digests["metric_cache_metadata_sha256"],
            metric_cache_manifest_sha256=digests["metric_cache_manifest_sha256"],
            baseline_checkpoint_sha256=digests["baseline_checkpoint_sha256"],
            vjepa_checkpoint_sha256=digests["vjepa_checkpoint_sha256"],
            production_code_revision=source["production_code_revision"],
            drive_jepa_factory_revision=source["drive_jepa_factory_revision"],
            device=root["device"],
            sample_size=root["sample_size"],
            schema_version=root["schema_version"],
            dataset=root["dataset"],
            purpose=root["purpose"],
        )
    except (KeyError, TypeError) as error:
        raise F2FeasibilityProductionError(
            "feasibility pretraining config is malformed"
        ) from error
    if pretraining_config_bytes(config) != payload:
        raise F2FeasibilityProductionError(
            "feasibility pretraining config did not round-trip canonically"
        )
    return config


@dataclass(frozen=True)
class LoadedPretrainingConfig:
    config: FeasibilityPretrainingConfig
    sha256: str


def read_pretraining_config(path: Union[str, Path]) -> LoadedPretrainingConfig:
    try:
        payload, digest = read_bytes_with_sidecar(path)
    except F2ProductionRunnerError as error:
        raise F2FeasibilityProductionError(str(error)) from error
    return LoadedPretrainingConfig(
        config=parse_pretraining_config_bytes(payload), sha256=digest
    )


def write_pretraining_config(
    path: Union[str, Path], config: FeasibilityPretrainingConfig
) -> str:
    try:
        return write_bytes_no_overwrite(path, pretraining_config_bytes(config))
    except F2ProductionRunnerError as error:
        raise F2FeasibilityProductionError(str(error)) from error


def _run_git(
    repository_root: Path, arguments: Sequence[str], *, text: bool = True
) -> Union[str, bytes]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise F2FeasibilityProductionError(
            "cannot authenticate feasibility production source with Git: "
            f"arguments={list(arguments)!r}, stderr={str(stderr).strip()!r}"
        ) from error
    return completed.stdout


@dataclass(frozen=True)
class FeasibilityCodeSourceIdentity:
    revision: str
    path_tree_sha256: str
    paths: Tuple[str, ...]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "path_tree_sha256": self.path_tree_sha256,
            "paths": list(self.paths),
            "revision": self.revision,
        }


def authenticate_feasibility_code_source(
    repository_root: Union[str, Path],
    *,
    expected_revision: str,
    paths: Sequence[str] = FEASIBILITY_CODE_PATHS,
) -> FeasibilityCodeSourceIdentity:
    """Authenticate every live orchestration file against one named commit."""

    root = _resolved_directory(Path(repository_root), "repository_root")
    revision = _revision(expected_revision, "production_code_revision")
    top_level = Path(str(_run_git(root, ["rev-parse", "--show-toplevel"])).strip()).resolve()
    if top_level != root:
        raise F2FeasibilityProductionError(
            "repository_root is not the authenticated Git worktree root"
        )
    resolved = str(
        _run_git(root, ["rev-parse", "--verify", f"{revision}^{{commit}}"])
    ).strip().lower()
    if resolved != revision:
        raise F2FeasibilityProductionError(
            "Git resolved the production code revision ambiguously"
        )
    live_head = str(_run_git(root, ["rev-parse", "HEAD"])).strip().lower()
    if live_head != revision:
        raise F2FeasibilityProductionError(
            "live repository HEAD differs from the production code revision"
        )
    normalized = tuple(str(path).replace("\\", "/") for path in paths)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or any(not path or path.startswith("/") or ".." in Path(path).parts for path in normalized)
    ):
        raise F2FeasibilityProductionError(
            "production source paths must be unique repository-relative paths"
        )
    # The revision is the trust root for the complete imported runtime, not
    # merely for the small list summarized in the certificate.  This closes
    # transitive imports such as the official EPDMS reconstruction, selector,
    # Dataset/SceneLoader, and simulator helpers.
    changed = str(
        _run_git(root, ["diff", "--name-only", revision, "--", "."])
    ).splitlines()
    untracked = str(
        _run_git(
            root,
            ["ls-files", "--others", "--exclude-standard"],
        )
    ).splitlines()
    ignored_runtime = str(
        _run_git(
            root,
            [
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--",
                "navsim_v2/navsim",
                "navsim_v2/vjepa2",
            ],
        )
    ).splitlines()
    import_shadows = tuple(
        path
        for path in ignored_runtime
        if Path(path).suffix.lower() in {".py", ".pyc", ".pyd", ".so"}
    )
    if changed or untracked or import_shadows:
        raise F2FeasibilityProductionError(
            "live feasibility production source differs from its revision: "
            f"changed={changed[:8]}, untracked={untracked[:8]}, "
            f"import_shadows={list(import_shadows[:8])}"
        )

    digest = hashlib.sha256()
    for relative_path in normalized:
        tracked = str(
            _run_git(root, ["ls-tree", "-r", "--name-only", revision, "--", relative_path])
        ).splitlines()
        if not tracked:
            raise F2FeasibilityProductionError(
                f"production revision lacks source path {relative_path!r}"
            )
        for file_name in tracked:
            expected = _run_git(root, ["show", f"{revision}:{file_name}"], text=False)
            assert isinstance(expected, bytes)
            live_path = root / file_name
            live = _read_stable_regular_file(
                live_path, name=f"production source {file_name!r}"
            )
            if live != expected:
                raise F2FeasibilityProductionError(
                    f"live production source bytes drifted: {file_name!r}"
                )
            encoded = file_name.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
            digest.update(len(live).to_bytes(8, "little"))
            digest.update(live)
    return FeasibilityCodeSourceIdentity(
        revision=revision,
        path_tree_sha256=digest.hexdigest(),
        paths=normalized,
    )


@dataclass(frozen=True)
class _TrainSamplePartitionView:
    rows: Tuple[TrainSampleRow, ...]
    name: str = FEASIBILITY_PARTITION

    @property
    def tokens(self) -> Tuple[str, ...]:
        return tuple(row.token for row in self.rows)

    @property
    def logs(self) -> Tuple[str, ...]:
        return tuple(sorted({row.log_name for row in self.rows}))


class QualifiedTrainFeatureSource:
    """Exact 2048-row official Drive-JEPA feature Dataset for the audit."""

    def __init__(
        self,
        *,
        qualification: QualificationManifest,
        sample_rows: Sequence[TrainSampleRow],
        scene_loader: Any,
        feature_dataset: Any,
    ) -> None:
        from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
            DriveJEPAFeatureBuilder,
        )

        if type(qualification) is not QualificationManifest:
            raise F2FeasibilityProductionError(
                "train feature source requires an authenticated qualification"
            )
        rows = tuple(sample_rows)
        if len(rows) != FEASIBILITY_SAMPLE_SIZE:
            raise F2FeasibilityProductionError(
                f"train feature source requires exactly {FEASIBILITY_SAMPLE_SIZE} rows"
            )
        if len({row.token for row in rows}) != len(rows):
            raise F2FeasibilityProductionError("train sample contains duplicate tokens")
        if not set(row.token for row in rows).issubset(set(qualification.train.tokens)):
            raise F2FeasibilityProductionError(
                "train feature sample is outside the qualified train partition"
            )
        self._rows = rows
        self._scene_loader = scene_loader
        self._feature_dataset = feature_dataset
        self.tokens = tuple(scene_loader.tokens)
        expected_tokens = tuple(row.token for row in rows)
        if self.tokens != expected_tokens:
            raise F2FeasibilityProductionError(
                "train SceneLoader order differs from deterministic sample order"
            )
        if getattr(feature_dataset, "_scene_loader", None) is not scene_loader:
            raise F2FeasibilityProductionError(
                "train feature Dataset is not backed by the qualified SceneLoader"
            )
        builders = tuple(getattr(feature_dataset, "_feature_builders", ()))
        if len(builders) != 1 or type(builders[0]) is not DriveJEPAFeatureBuilder:
            raise F2FeasibilityProductionError(
                "train feature Dataset must use the official DriveJEPAFeatureBuilder"
            )
        if builders[0].get_unique_name() != "pad_feature":
            raise F2FeasibilityProductionError(
                "official Drive-JEPA feature builder identity drifted"
            )
        if tuple(getattr(feature_dataset, "_target_builders", ())) != ():
            raise F2FeasibilityProductionError(
                "train feature Dataset must not expose target builders"
            )
        if getattr(feature_dataset, "_cache_path", None) is not None:
            raise F2FeasibilityProductionError(
                "train Drive-JEPA features must be computed from sensor blobs"
            )
        if len(feature_dataset) != len(rows):
            raise F2FeasibilityProductionError(
                "train feature Dataset length differs from deterministic sample"
            )
        self._index = {row.token: index for index, row in enumerate(rows)}

    def load(self, row: TrainSampleRow) -> Mapping[str, Any]:
        index = self._index.get(row.token)
        if index is None or self._rows[index] != row or self.tokens[index] != row.token:
            raise F2FeasibilityProductionError(
                "requested train feature row differs from deterministic sample"
            )
        item = self._feature_dataset[index]
        if not isinstance(item, tuple) or len(item) != 2:
            raise F2FeasibilityProductionError(
                "train NAVSIM Dataset item must be (features, targets)"
            )
        features, targets = item
        if not isinstance(targets, Mapping) or targets:
            raise F2FeasibilityProductionError(
                "train feature Dataset unexpectedly exposed target values"
            )
        return _validate_unbatched_drive_jepa_features(features)


@dataclass(frozen=True)
class QualifiedTrainFeatureBuild:
    source: QualifiedTrainFeatureSource
    removed_scenes: int


def build_qualified_train_feature_source(
    qualification: QualificationManifest,
    sample_rows: Sequence[TrainSampleRow],
    *,
    navsim_log_path: Union[str, Path],
    sensor_blobs_path: Union[str, Path],
    authenticated_raw_inputs: VerifiedNavtrainRawInputs,
) -> QualifiedTrainFeatureBuild:
    """Construct the official feature Dataset for the deterministic sample."""

    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import (
        DriveJEPAConfig,
    )
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
        DriveJEPAFeatureBuilder,
    )
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.training.dataset import Dataset

    if type(qualification) is not QualificationManifest:
        raise F2FeasibilityProductionError(
            "feature construction requires authenticated QualificationManifest"
        )
    if qualification.dataset != FEASIBILITY_DATASET:
        raise F2FeasibilityProductionError("feature construction accepts navtrain only")
    if qualification.sensor_contract != F2_SENSOR_CONTRACT:
        raise F2FeasibilityProductionError("qualification sensor contract drifted")
    if type(authenticated_raw_inputs) is not VerifiedNavtrainRawInputs:
        raise F2FeasibilityProductionError(
            "train feature construction requires sealed navtrain raw inputs"
        )
    authenticated_raw_inputs.require_roots(
        Path(navsim_log_path), Path(sensor_blobs_path)
    )
    rows = tuple(sample_rows)
    if rows != select_deterministic_navtrain_train_sample(qualification):
        raise F2FeasibilityProductionError(
            "feature rows differ from deterministic train sampling"
        )
    if len(rows) != FEASIBILITY_SAMPLE_SIZE:
        raise F2FeasibilityProductionError(
            f"qualified train must supply exactly {FEASIBILITY_SAMPLE_SIZE} audit rows"
        )
    partition = _TrainSamplePartitionView(rows)
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        max_scenes=None,
        log_names=list(qualification.train.logs),
        tokens=list(qualification.train.tokens),
        include_synthetic_scenes=False,
        synthetic_scene_tokens=None,
    )
    constrain_scene_filter(scene_filter, partition)  # type: ignore[arg-type]
    loader = SceneLoader(
        data_path=_resolved_directory(Path(navsim_log_path), "navsim_log"),
        sensor_blobs_path=_resolved_directory(Path(sensor_blobs_path), "sensor_blobs"),
        synthetic_scenes_path=None,
        scene_filter=scene_filter,
        sensor_config=f2_sensor_config(),
        authenticated_raw_inputs=authenticated_raw_inputs,
    )
    availability = qualify_scene_loader(loader, partition)  # type: ignore[arg-type]
    removed = getattr(availability, "removed_scenes", None)
    if isinstance(removed, bool) or removed != 0:
        raise F2FeasibilityProductionError(
            "deterministic train feature sample lost sensor rows"
        )
    dataset = Dataset(
        scene_loader=loader,
        feature_builders=[DriveJEPAFeatureBuilder(config=DriveJEPAConfig())],
        target_builders=[],
        cache_path=None,
        force_cache_computation=False,
    )
    return QualifiedTrainFeatureBuild(
        source=QualifiedTrainFeatureSource(
            qualification=qualification,
            sample_rows=rows,
            scene_loader=loader,
            feature_dataset=dataset,
        ),
        removed_scenes=removed,
    )


def _metric_cache_identity(metric_cache: Any) -> Tuple[str, int]:
    log_name = getattr(metric_cache, "log_name", None)
    timepoint = getattr(metric_cache, "timepoint", None)
    time_us = getattr(timepoint, "time_us", None)
    if (
        not isinstance(log_name, str)
        or not log_name
        or isinstance(time_us, bool)
        or not isinstance(time_us, int)
        or time_us < 0
    ):
        raise F2FeasibilityProductionError(
            "authenticated metric cache lacks a canonical log/time identity"
        )
    return log_name, time_us


def build_frozen_static_inputs(
    sample_rows: Sequence[TrainSampleRow],
    *,
    feature_source: Any,
    references: Mapping[str, AuthenticatedMetricCacheReference],
    baseline: RestoredDriveJepaBaseline,
    semantics: DriveJepaNavsimRecordSemantics,
    cache_loader: Callable[[AuthenticatedMetricCacheReference], Any] = load_authenticated_metric_cache,
) -> Tuple[FrozenStaticTokenInput, ...]:
    """Run the authenticated baseline and official train semantics tokenwise."""

    rows = tuple(sample_rows)
    if len(rows) != FEASIBILITY_SAMPLE_SIZE:
        raise F2FeasibilityProductionError(
            f"static-input production requires exactly {FEASIBILITY_SAMPLE_SIZE} rows"
        )
    if len({row.token for row in rows}) != len(rows):
        raise F2FeasibilityProductionError("static-input rows contain duplicate tokens")
    if type(baseline) is not RestoredDriveJepaBaseline:
        raise F2FeasibilityProductionError(
            "baseline must be the authenticated restored Drive-JEPA baseline"
        )
    if baseline.checkpoint_sha256 != OFFICIAL_DRIVE_JEPA_BASELINE_SHA256:
        raise F2FeasibilityProductionError("restored baseline identity drifted")
    if type(semantics) is not DriveJepaNavsimRecordSemantics:
        raise F2FeasibilityProductionError(
            "semantics must be official DriveJepaNavsimRecordSemantics"
        )
    expected_tokens = {row.token for row in rows}
    missing = sorted(expected_tokens - set(references))
    if missing:
        raise F2FeasibilityProductionError(
            f"metric-cache content manifest misses sampled tokens: {missing[:8]}"
        )

    frozen = []
    for row in rows:
        reference = references[row.token]
        if type(reference) is not AuthenticatedMetricCacheReference:
            raise F2FeasibilityProductionError(
                f"metric-cache reference is not authenticated for token {row.token!r}"
            )
        if reference.entry.token != row.token or reference.entry.partition != FEASIBILITY_PARTITION:
            raise F2FeasibilityProductionError(
                f"metric-cache entry scope drifted for sampled token {row.token!r}"
            )
        features = feature_source.load(row)
        output = baseline.forward.infer(row.token, features)
        if output.token != row.token:
            raise F2FeasibilityProductionError(
                "Drive-JEPA baseline changed the sampled token identity"
            )
        proposals = output.proposals.detach().cpu().numpy()
        base_logits = output.base_logits.detach().cpu().numpy()
        proposals = np.ascontiguousarray(proposals, dtype=np.dtype("<f4"))
        base_logits = np.ascontiguousarray(base_logits, dtype=np.dtype("<f4"))
        if proposals.shape != (32, 8, 3) or base_logits.shape != (32,):
            raise F2FeasibilityProductionError(
                "Drive-JEPA baseline output shape drifted during feasibility production"
            )
        if not np.isfinite(proposals).all() or not np.isfinite(base_logits).all():
            raise F2FeasibilityProductionError(
                "Drive-JEPA baseline output contains NaN or infinity"
            )
        metric_cache = cache_loader(reference)
        if _metric_cache_identity(metric_cache) != (row.log_name, row.time_us):
            raise F2FeasibilityProductionError(
                f"metric-cache identity differs for sampled token {row.token!r}"
            )
        simulation = semantics.run(metric_cache, proposals)
        targets = np.asarray(simulation.targets)
        if (
            targets.dtype != np.dtype("<f4")
            or targets.shape != (32, 9)
            or not targets.flags.c_contiguous
            or not np.isfinite(targets).all()
        ):
            raise F2FeasibilityProductionError(
                "official navtrain static targets must be contiguous float32 [32, 9]"
            )
        frozen.append(
            FrozenStaticTokenInput(
                log_name=row.log_name,
                time_us=row.time_us,
                token=row.token,
                base_logits=tuple(float(value) for value in base_logits.tolist()),
                target_scores=tuple(
                    tuple(float(value) for value in candidate.tolist())
                    for candidate in targets
                ),
            )
        )
    return tuple(frozen)


def capacity_gate_passes(value: Any) -> bool:
    if isinstance(value, bool):
        raise F2FeasibilityProductionError("capacity-gate value cannot be bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise F2FeasibilityProductionError(
            "capacity-gate value must be numeric"
        ) from error
    if not math.isfinite(numeric):
        raise F2FeasibilityProductionError("capacity-gate value must be finite")
    return numeric > CAPACITY_GATE_THRESHOLD


def _sample_identity_sha256(rows: Sequence[TrainSampleRow]) -> str:
    return _sha256_bytes(
        canonical_json_bytes([row.to_payload() for row in rows])
    )


def _source_files_sha256(files: Sequence[Any]) -> str:
    return _sha256_bytes(
        canonical_json_bytes([item.to_payload() for item in files])
    )


def _cache_entries_sha256(cache: VerifiedMetricCacheManifest) -> str:
    return _sha256_bytes(
        canonical_json_bytes(
            [entry.to_payload() for entry in cache.manifest.entries]
        )
    )


def production_certificate_payload(
    audit: F2ResidualFeasibilityAudit,
    *,
    pretraining_config_sha256: str,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Wrap one computed audit in the production claim/provenance envelope."""

    if type(audit) is not F2ResidualFeasibilityAudit:
        raise F2FeasibilityProductionError(
            "production certificate requires concrete F2ResidualFeasibilityAudit"
        )
    audit_payload = audit.to_payload()
    _validate_embedded_audit(audit_payload)
    qualification_provenance = (
        provenance.get("qualification") if isinstance(provenance, Mapping) else None
    )
    if (
        not isinstance(qualification_provenance, Mapping)
        or audit.qualification_manifest_sha256
        != qualification_provenance.get("manifest_sha256")
    ):
        raise F2FeasibilityProductionError(
            "audit qualification differs from production provenance"
        )
    value = audit.aggregate.reachable_delta_upper
    certificate = {
        "audit": audit_payload,
        "audit_sha256": feasibility_audit_sha256(audit),
        "capacity_gate": {
            "metric": CAPACITY_GATE_METRIC,
            "operator": CAPACITY_GATE_OPERATOR,
            "passed": capacity_gate_passes(value),
            "threshold": CAPACITY_GATE_THRESHOLD,
            "unit": CAPACITY_GATE_UNIT,
            "value": value,
        },
        "claim_scope": PRODUCTION_CLAIM_SCOPE,
        "dataset": FEASIBILITY_DATASET,
        "kind": PRODUCTION_AUDIT_KIND,
        "pretraining_config_sha256": _sha256(
            pretraining_config_sha256, "pretraining_config_sha256"
        ),
        "provenance": dict(provenance),
        "schema_version": PRODUCTION_AUDIT_SCHEMA_VERSION,
    }
    return dict(validate_production_certificate_payload(certificate))


def _finite_unit_sequence(value: Any, *, size: int, name: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise F2FeasibilityProductionError(
            f"{name} must contain exactly {size} values"
        )
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise F2FeasibilityProductionError(f"{name}[{index}] must be numeric")
        numeric = float(item)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise F2FeasibilityProductionError(
                f"{name}[{index}] must be finite and lie in [0, 1]"
            )
        result.append(numeric)
    return tuple(result)


def _best_truth_index(indices: Sequence[int], truths: Sequence[float]) -> int:
    if not indices:
        raise F2FeasibilityProductionError(
            "embedded reachability contains no selectable candidate"
        )
    best = indices[0]
    for index in indices[1:]:
        if truths[index] > truths[best]:
            best = index
    return best


def _recompute_gate_delta_from_tokens(
    rows: Sequence[Any], tokens: Sequence[Any]
) -> float:
    """Replay every gate-dependent token value from base logits and truths."""

    token_keys = {
        "base_selected",
        "future_camera_valid",
        "frozen_static_input",
        "frozen_static_input_sha256",
        "hard_factor_reachable_marginal_upper",
        "log_name",
        "reachability",
        "reachable_delta_upper",
        "reachable_headroom_closed_upper",
        "reachable_oracle",
        "safety_candidate_coverage",
        "sampling_sha256",
        "selection_stage",
        "static_true_epdms",
        "time_us",
        "token",
        "unbounded_headroom",
        "unbounded_oracle",
    }
    selected_keys = {"candidate_index", "static_true_epdms"}
    deltas = []
    # Identical proposal logits are common in synthetic audits and can also
    # occur after a deterministic model collapse.  Reachability is a pure
    # function of the binary32 logits, so memoizing within this verification
    # pass preserves the exact replay while avoiding thousands of duplicate
    # endpoint enumerations.
    replay_cache: Dict[Tuple[float, ...], Any] = {}
    for offset, (raw_row, raw_result) in enumerate(zip(rows, tokens)):
        row = _exact_keys(
            raw_row,
            {
                "future_camera_valid",
                "log_name",
                "sampling_sha256",
                "selection_stage",
                "time_us",
                "token",
            },
            f"embedded sample row {offset}",
        )
        result = _exact_keys(
            raw_result, token_keys, f"embedded token result {offset}"
        )
        for identity_key in (
            "future_camera_valid",
            "log_name",
            "sampling_sha256",
            "selection_stage",
            "time_us",
            "token",
        ):
            if result[identity_key] != row[identity_key]:
                raise F2FeasibilityProductionError(
                    f"embedded token result {offset} identity differs from its sample row"
                )
        _sha256(
            result["frozen_static_input_sha256"],
            f"embedded token result {offset} frozen static input",
        )
        proof = _exact_keys(
            result["frozen_static_input"],
            {"base_logits", "target_scores"},
            f"embedded token result {offset} frozen static input",
        )
        try:
            frozen = FrozenStaticTokenInput(
                log_name=row["log_name"],
                time_us=row["time_us"],
                token=row["token"],
                base_logits=proof["base_logits"],
                target_scores=proof["target_scores"],
            )
        except Exception as error:
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} frozen static input is invalid"
            ) from error
        normalized_proof = {
            "base_logits": list(frozen.base_logits),
            "target_scores": [list(candidate) for candidate in frozen.target_scores],
        }
        if normalized_proof != proof:
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} frozen static input is not exact float32"
            )
        if result["frozen_static_input_sha256"] != _sha256_bytes(
            canonical_json_bytes(normalized_proof)
        ):
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} frozen static input digest differs"
            )
        reachability = result["reachability"]
        if not isinstance(reachability, Mapping):
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} reachability is malformed"
            )
        base_logits = reachability.get("base_logits")
        if not isinstance(base_logits, list):
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} base logits are malformed"
            )
        try:
            replay_key = tuple(float(value) for value in base_logits)
            replayed = replay_cache.get(replay_key)
            if replayed is None:
                replayed = analyze_tanh_residual_reachability(base_logits)
                replay_cache[replay_key] = replayed
        except Exception as error:
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} reachability cannot be replayed"
            ) from error
        if replayed.to_payload() != reachability:
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} reachability differs from replay"
            )
        if replayed.base_logits != frozen.base_logits:
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} reachability differs from frozen logits"
            )
        truths = _finite_unit_sequence(
            result["static_true_epdms"],
            size=32,
            name=f"embedded token result {offset} static truths",
        )
        recomputed_truths = tuple(
            causal_true_epdms(candidate, two_frame_comfort=None)
            for candidate in frozen.target_scores
        )
        if truths != recomputed_truths:
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} static truths differ from frozen targets"
            )
        base_index = replayed.base_selected_index
        reachable_index = _best_truth_index(
            replayed.reachable_candidate_indices, truths
        )
        unbounded_index = _best_truth_index(tuple(range(32)), truths)
        base_score = truths[base_index]
        reachable_score = truths[reachable_index]
        unbounded_score = truths[unbounded_index]
        expected_sections = (
            ("base_selected", base_index, base_score),
            ("reachable_oracle", reachable_index, reachable_score),
            ("unbounded_oracle", unbounded_index, unbounded_score),
        )
        for section_name, expected_index, expected_score in expected_sections:
            section = _exact_keys(
                result[section_name],
                selected_keys,
                f"embedded token result {offset} {section_name}",
            )
            if (
                section["candidate_index"] != expected_index
                or section["static_true_epdms"] != expected_score
            ):
                raise F2FeasibilityProductionError(
                    f"embedded token result {offset} {section_name} differs from replay"
                )
        delta = max(0.0, reachable_score - base_score)
        if result["reachable_delta_upper"] != delta:
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} reachable delta differs from replay"
            )
        unbounded_headroom = max(0.0, unbounded_score - base_score)
        expected_headroom_closed = (
            1.0 if unbounded_headroom == 0.0 else delta / unbounded_headroom
        )
        if (
            result["unbounded_headroom"] != unbounded_headroom
            or result["reachable_headroom_closed_upper"]
            != expected_headroom_closed
        ):
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} headroom differs from frozen-target replay"
            )
        hard_factors = _exact_keys(
            result["hard_factor_reachable_marginal_upper"],
            set(HARD_FACTOR_NAMES),
            f"embedded token result {offset} hard-factor bounds",
        )
        expected_hard_factors = {
            name: max(
                frozen.target_scores[index][channel]
                for index in replayed.reachable_candidate_indices
            )
            for channel, name in enumerate(HARD_FACTOR_NAMES)
        }
        if hard_factors != expected_hard_factors:
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} hard factors differ from frozen targets"
            )
        safety = _exact_keys(
            result["safety_candidate_coverage"],
            {
                "has_reachable_safe_candidate",
                "has_safe_candidate",
                "reachable_safe_candidate_count",
                "safe_candidate_count",
                "safe_candidate_reachability_fraction",
                "safe_definition",
            },
            f"embedded token result {offset} safety coverage",
        )
        safe_indices = tuple(
            index
            for index, candidate in enumerate(frozen.target_scores)
            if all(candidate[channel] == 1.0 for channel in range(4))
        )
        reachable_set = set(replayed.reachable_candidate_indices)
        reachable_safe = tuple(
            index for index in safe_indices if index in reachable_set
        )
        expected_safety = {
            "has_reachable_safe_candidate": bool(reachable_safe),
            "has_safe_candidate": bool(safe_indices),
            "reachable_safe_candidate_count": len(reachable_safe),
            "safe_candidate_count": len(safe_indices),
            "safe_candidate_reachability_fraction": (
                len(reachable_safe) / len(safe_indices) if safe_indices else None
            ),
            "safe_definition": "all_four_hard_channels_exactly_one",
        }
        if safety != expected_safety:
            raise F2FeasibilityProductionError(
                f"embedded token result {offset} safety coverage differs from frozen targets"
            )
        deltas.append(delta)
    return math.fsum(deltas) / len(deltas)


def _validate_embedded_audit(audit: Any) -> Mapping[str, Any]:
    root = _exact_keys(
        audit,
        {
            "aggregate",
            "bootstrap",
            "claim_scope",
            "dataset_scope",
            "qualification_manifest_sha256",
            "residual_contract",
            "sample",
            "schema_version",
            "tokens",
        },
        "embedded feasibility audit",
    )
    if root["schema_version"] != FEASIBILITY_SCHEMA_VERSION:
        raise F2FeasibilityProductionError("embedded feasibility schema drifted")
    if root["claim_scope"] != "analytical_train_only_upper_bound_not_checkpoint_or_test_score":
        raise F2FeasibilityProductionError("embedded feasibility claim scope drifted")
    scope = root["dataset_scope"]
    if not isinstance(scope, Mapping) or scope.get("dataset") != FEASIBILITY_DATASET or scope.get("partition") != FEASIBILITY_PARTITION:
        raise F2FeasibilityProductionError("embedded feasibility dataset scope drifted")
    sample = root["sample"]
    if not isinstance(sample, Mapping):
        raise F2FeasibilityProductionError("embedded feasibility sample is malformed")
    rows = sample.get("rows")
    tokens = root["tokens"]
    if (
        sample.get("target_size") != FEASIBILITY_SAMPLE_SIZE
        or sample.get("token_count") != FEASIBILITY_SAMPLE_SIZE
        or not isinstance(rows, list)
        or len(rows) != FEASIBILITY_SAMPLE_SIZE
        or not isinstance(tokens, list)
        or len(tokens) != FEASIBILITY_SAMPLE_SIZE
    ):
        raise F2FeasibilityProductionError(
            f"embedded audit must cover exactly {FEASIBILITY_SAMPLE_SIZE} rows"
        )
    sample_tokens = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("token"), str):
            raise F2FeasibilityProductionError("embedded sample row is malformed")
        sample_tokens.append(row["token"])
    result_tokens = []
    for result in tokens:
        if not isinstance(result, Mapping) or not isinstance(result.get("token"), str):
            raise F2FeasibilityProductionError("embedded token result is malformed")
        result_tokens.append(result["token"])
    if len(set(sample_tokens)) != FEASIBILITY_SAMPLE_SIZE or result_tokens != sample_tokens:
        raise F2FeasibilityProductionError(
            "embedded sample/result token coverage differs or contains duplicates"
        )
    aggregate = root["aggregate"]
    if not isinstance(aggregate, Mapping) or aggregate.get("token_count") != FEASIBILITY_SAMPLE_SIZE:
        raise F2FeasibilityProductionError("embedded aggregate token count drifted")
    replayed_delta = _recompute_gate_delta_from_tokens(rows, tokens)
    if aggregate.get("reachable_delta_upper") != replayed_delta:
        raise F2FeasibilityProductionError(
            "embedded aggregate reachable delta differs from token replay"
        )
    _sha256(root["qualification_manifest_sha256"], "audit qualification SHA-256")
    return root


def validate_production_certificate_payload(payload: Any) -> Mapping[str, Any]:
    root = _exact_keys(
        payload, _TOP_LEVEL_CERTIFICATE_KEYS, "production feasibility certificate"
    )
    if (
        root["schema_version"] != PRODUCTION_AUDIT_SCHEMA_VERSION
        or root["kind"] != PRODUCTION_AUDIT_KIND
        or root["dataset"] != FEASIBILITY_DATASET
        or root["claim_scope"] != PRODUCTION_CLAIM_SCOPE
    ):
        raise F2FeasibilityProductionError(
            "production feasibility certificate identity drifted"
        )
    _sha256(root["pretraining_config_sha256"], "pretraining config SHA-256")
    audit = _validate_embedded_audit(root["audit"])
    expected_audit_sha = _sha256_bytes(canonical_json_bytes(root["audit"]))
    if _sha256(root["audit_sha256"], "audit SHA-256") != expected_audit_sha:
        raise F2FeasibilityProductionError("embedded audit SHA-256 differs")
    provenance = _exact_keys(
        root["provenance"], _PROVENANCE_KEYS, "audit provenance"
    )
    baseline = _exact_keys(
        provenance["baseline"],
        _BASELINE_PROVENANCE_KEYS,
        "baseline provenance",
    )
    if (
        _sha256(baseline["checkpoint_sha256"], "baseline checkpoint SHA-256")
        != OFFICIAL_DRIVE_JEPA_BASELINE_SHA256
        or _sha256(baseline["vjepa_checkpoint_sha256"], "V-JEPA checkpoint SHA-256")
        != OFFICIAL_DRIVE_JEPA_VJEPA_SHA256
        or _revision(baseline["factory_revision"], "factory revision")
        != OFFICIAL_DRIVE_JEPA_FACTORY_REVISION
    ):
        raise F2FeasibilityProductionError(
            "certificate baseline identity differs from frozen official assets"
        )
    _sha256(baseline["factory_path_tree_sha256"], "factory path-tree SHA-256")
    if (
        isinstance(baseline["state_tensor_count"], bool)
        or not isinstance(baseline["state_tensor_count"], int)
        or baseline["state_tensor_count"] <= 0
    ):
        raise F2FeasibilityProductionError(
            "baseline state-tensor count must be positive"
        )

    code_source = _exact_keys(
        provenance["code_source"],
        _CODE_SOURCE_PROVENANCE_KEYS,
        "code-source provenance",
    )
    _revision(code_source["revision"], "production code revision")
    _sha256(code_source["path_tree_sha256"], "production path-tree SHA-256")
    if (
        not isinstance(code_source["paths"], list)
        or tuple(code_source["paths"]) != FEASIBILITY_CODE_PATHS
    ):
        raise F2FeasibilityProductionError(
            "production code-source paths differ from the frozen scope"
        )

    try:
        deterministic_runtime = _validate_deterministic_runtime_payload(
            provenance["deterministic_runtime"]
        )
    except F2ProductionRunnerError as error:
        raise F2FeasibilityProductionError(
            "certificate deterministic runtime is invalid"
        ) from error
    if (
        deterministic_runtime["logical_rank"] != 0
        or deterministic_runtime["world_size"] != 1
        or deterministic_runtime["device"] != PRODUCTION_DEVICE
    ):
        raise F2FeasibilityProductionError(
            "certificate deterministic runtime is not the fixed feasibility runtime"
        )

    feature_sample = _exact_keys(
        provenance["feature_sample"],
        _FEATURE_SAMPLE_PROVENANCE_KEYS,
        "feature-sample provenance",
    )
    if (
        feature_sample["removed_scenes"] != 0
        or feature_sample["token_count"] != FEASIBILITY_SAMPLE_SIZE
    ):
        raise F2FeasibilityProductionError(
            "feature sample did not preserve exact train coverage"
        )
    sample_row_sha256 = _sha256(
        feature_sample["row_identity_sha256"], "sample row SHA-256"
    )
    if sample_row_sha256 != _sha256_bytes(
        canonical_json_bytes(audit["sample"]["rows"])
    ):
        raise F2FeasibilityProductionError(
            "feature-sample row identity differs from the embedded audit"
        )

    metric_cache = _exact_keys(
        provenance["metric_cache"],
        _METRIC_CACHE_PROVENANCE_KEYS,
        "metric-cache provenance",
    )
    for key in (
        "content_entries_sha256",
        "content_manifest_sha256",
        "metadata_sha256",
    ):
        _sha256(metric_cache[key], f"metric-cache {key}")
    if (
        isinstance(metric_cache["content_entry_count"], bool)
        or not isinstance(metric_cache["content_entry_count"], int)
        or metric_cache["content_entry_count"] < FEASIBILITY_SAMPLE_SIZE
    ):
        raise F2FeasibilityProductionError(
            "metric-cache content inventory is smaller than the audit sample"
        )

    qualification = _exact_keys(
        provenance["qualification"],
        _QUALIFICATION_PROVENANCE_KEYS,
        "qualification provenance",
    )
    for key in (
        "config_sha256",
        "manifest_sha256",
        "source_files_sha256",
        "source_manifest_sha256",
    ):
        _sha256(qualification[key], f"qualification {key}")
    if (
        isinstance(qualification["source_file_count"], bool)
        or not isinstance(qualification["source_file_count"], int)
        or qualification["source_file_count"] <= 0
    ):
        raise F2FeasibilityProductionError(
            "qualification source inventory must be non-empty"
        )
    if qualification["manifest_sha256"] != audit["qualification_manifest_sha256"]:
        raise F2FeasibilityProductionError(
            "certificate qualification provenance differs from embedded audit"
        )
    raw_input = _exact_keys(
        provenance["raw_input"],
        _RAW_INPUT_PROVENANCE_KEYS,
        "raw-input provenance",
    )
    _sha256(
        raw_input["content_manifest_sha256"],
        "raw-input content manifest SHA-256",
    )
    for key in ("log_entry_count", "sensor_entry_count"):
        value = raw_input[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise F2FeasibilityProductionError(
                f"raw-input {key} must be a positive integer"
            )
    gate = _exact_keys(root["capacity_gate"], _GATE_KEYS, "capacity gate")
    if (
        gate["metric"] != CAPACITY_GATE_METRIC
        or gate["operator"] != CAPACITY_GATE_OPERATOR
        or gate["threshold"] != CAPACITY_GATE_THRESHOLD
        or gate["unit"] != CAPACITY_GATE_UNIT
        or not isinstance(gate["passed"], bool)
    ):
        raise F2FeasibilityProductionError("capacity-gate contract drifted")
    aggregate = audit["aggregate"]
    value = aggregate.get("reachable_delta_upper")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise F2FeasibilityProductionError("audit reachable delta is malformed")
    if gate["value"] != value or gate["passed"] is not capacity_gate_passes(value):
        raise F2FeasibilityProductionError(
            "capacity gate is not the strict raw-point audit result"
        )
    return root


@dataclass(frozen=True)
class LoadedProductionFeasibilityAudit:
    payload: Mapping[str, Any]
    sha256: str


def _cross_bind_certificate_to_pretraining_config(
    certificate: Mapping[str, Any],
    config: FeasibilityPretrainingConfig,
    *,
    config_sha256: str,
) -> None:
    """Bind every config-defined certificate identity to external authority.

    The certificate cannot choose any raw/cache/qualification/model/code
    identity merely by repeating it in its own provenance.  Those identities
    must agree field-for-field with the independently loaded, externally
    digest-pinned pretraining configuration.
    """

    if certificate["pretraining_config_sha256"] != config_sha256:
        raise F2FeasibilityProductionError(
            "production feasibility audit differs from its external pretraining config"
        )
    provenance = certificate["provenance"]
    baseline = provenance["baseline"]
    code_source = provenance["code_source"]
    deterministic_runtime = provenance["deterministic_runtime"]
    feature_sample = provenance["feature_sample"]
    metric_cache = provenance["metric_cache"]
    qualification = provenance["qualification"]
    raw_input = provenance["raw_input"]
    audit = certificate["audit"]
    bindings = (
        (
            baseline["checkpoint_sha256"],
            config.baseline_checkpoint_sha256,
            "baseline checkpoint",
        ),
        (
            baseline["vjepa_checkpoint_sha256"],
            config.vjepa_checkpoint_sha256,
            "V-JEPA checkpoint",
        ),
        (
            baseline["factory_revision"],
            config.drive_jepa_factory_revision,
            "Drive-JEPA factory revision",
        ),
        (
            code_source["revision"],
            config.production_code_revision,
            "production code revision",
        ),
        (
            deterministic_runtime["device"],
            config.device,
            "deterministic runtime device",
        ),
        (
            metric_cache["content_manifest_sha256"],
            config.metric_cache_manifest_sha256,
            "metric-cache content manifest",
        ),
        (
            metric_cache["metadata_sha256"],
            config.metric_cache_metadata_sha256,
            "metric-cache metadata",
        ),
        (
            qualification["config_sha256"],
            config.qualification_config_sha256,
            "qualification config",
        ),
        (
            qualification["manifest_sha256"],
            config.qualification_manifest_sha256,
            "qualification manifest",
        ),
        (
            qualification["source_manifest_sha256"],
            config.qualification_source_manifest_sha256,
            "qualification source manifest",
        ),
        (
            raw_input["content_manifest_sha256"],
            config.raw_input_manifest_sha256,
            "raw-input content manifest",
        ),
        (
            audit["qualification_manifest_sha256"],
            config.qualification_manifest_sha256,
            "embedded-audit qualification manifest",
        ),
        (
            audit["sample"]["target_size"],
            config.sample_size,
            "embedded-audit sample size",
        ),
        (
            feature_sample["token_count"],
            config.sample_size,
            "feature-sample token count",
        ),
    )
    for actual, expected, name in bindings:
        if actual != expected:
            raise F2FeasibilityProductionError(
                f"certificate {name} differs from external pretraining config"
            )


def read_production_feasibility_audit(
    path: Union[str, Path],
    *,
    expected_sha256: str,
    expected_pretraining_config_path: Union[str, Path],
    expected_pretraining_config_sha256: str,
) -> LoadedProductionFeasibilityAudit:
    expected = _sha256(expected_sha256, "expected production audit SHA-256")
    expected_config = _sha256(
        expected_pretraining_config_sha256,
        "expected pretraining config SHA-256",
    )
    loaded_config = read_pretraining_config(expected_pretraining_config_path)
    if loaded_config.sha256 != expected_config:
        raise F2FeasibilityProductionError(
            "external pretraining config differs from its expected SHA-256"
        )
    try:
        raw, digest = read_bytes_with_sidecar(path)
        decoded = parse_canonical_json_bytes(
            raw, name="production feasibility audit"
        )
    except F2ProductionRunnerError as error:
        raise F2FeasibilityProductionError(str(error)) from error
    if digest != expected:
        raise F2FeasibilityProductionError(
            "production feasibility audit differs from its external expected SHA-256"
        )
    validated = validate_production_certificate_payload(decoded)
    _cross_bind_certificate_to_pretraining_config(
        validated,
        loaded_config.config,
        config_sha256=expected_config,
    )
    return LoadedProductionFeasibilityAudit(
        payload=validated,
        sha256=digest,
    )


def write_production_feasibility_audit(
    path: Union[str, Path], payload: Mapping[str, Any]
) -> str:
    validated = validate_production_certificate_payload(payload)
    try:
        return write_bytes_no_overwrite(path, canonical_json_bytes(validated))
    except F2ProductionRunnerError as error:
        raise F2FeasibilityProductionError(str(error)) from error


def _load_authenticated_qualification(
    config: FeasibilityPretrainingConfig,
) -> Tuple[QualificationManifest, Tuple[Any, ...]]:
    _hash_stable_file(
        _resolved_file(config.qualification_config, "qualification_config"),
        config.qualification_config_sha256,
        "qualification config",
    )
    manifest_payload = _hash_stable_file(
        _resolved_file(config.qualification_manifest, "qualification_manifest"),
        config.qualification_manifest_sha256,
        "qualification manifest",
    )
    try:
        qualification = manifest_from_bytes(
            manifest_payload,
            expected_manifest_sha256=config.qualification_manifest_sha256,
            expected_config_sha256=config.qualification_config_sha256,
            expected_cache_sha256=config.metric_cache_metadata_sha256,
            expected_source_sha256=config.qualification_source_manifest_sha256,
        )
        source_files = verify_qualification_source_manifest(
            _resolved_file(
                config.qualification_source_manifest,
                "qualification_source_manifest",
            ),
            expected_sha256=config.qualification_source_manifest_sha256,
            source_root=_resolved_directory(
                config.qualification_source_root, "qualification_source_root"
            ),
        )
    except Exception as error:
        raise F2FeasibilityProductionError(
            "qualification train inputs failed authentication"
        ) from error
    if qualification.dataset != FEASIBILITY_DATASET or qualification.sensor_contract != F2_SENSOR_CONTRACT:
        raise F2FeasibilityProductionError("qualification train scope drifted")
    return qualification, tuple(source_files)


def _load_authenticated_metric_cache_manifest(
    config: FeasibilityPretrainingConfig,
    qualification: QualificationManifest,
) -> VerifiedMetricCacheManifest:
    try:
        verified = load_and_verify_metric_cache_manifest(
            _resolved_file(config.metric_cache_manifest, "metric_cache_manifest"),
            expected_sha256=config.metric_cache_manifest_sha256,
            cache_root=_resolved_directory(config.metric_cache_root, "metric_cache_root"),
            metadata_path=_resolved_file(
                config.metric_cache_metadata, "metric_cache_metadata"
            ),
            expected_metadata_sha256=config.metric_cache_metadata_sha256,
            qualification=qualification,
            expected_qualification_manifest_sha256=config.qualification_manifest_sha256,
        )
    except Exception as error:
        raise F2FeasibilityProductionError(
            "metric-cache content manifest failed authentication"
        ) from error
    if verified.manifest.dataset != FEASIBILITY_DATASET:
        raise F2FeasibilityProductionError("metric-cache dataset scope drifted")
    return verified


def _load_authenticated_raw_inputs(
    config: FeasibilityPretrainingConfig,
    qualification: QualificationManifest,
) -> VerifiedNavtrainRawInputs:
    try:
        verified = load_and_verify_raw_input_manifest(
            _resolved_file(config.raw_input_manifest, "raw_input_manifest"),
            expected_sha256=config.raw_input_manifest_sha256,
            log_root=_resolved_directory(config.navsim_log, "navsim_log"),
            sensor_root=_resolved_directory(config.sensor_blobs, "sensor_blobs"),
            qualification=qualification,
            expected_qualification_manifest_sha256=(
                config.qualification_manifest_sha256
            ),
        )
    except Exception as error:
        raise F2FeasibilityProductionError(
            "navtrain raw-input content manifest failed authentication"
        ) from error
    return verified


def _restore_official_baseline(
    config: FeasibilityPretrainingConfig,
    factory_source: DriveJepaFactorySourceIdentity,
) -> RestoredDriveJepaBaseline:
    repository_root = _resolved_directory(config.repository_root, "repository_root")
    experiment_root = _resolved_directory(config.experiment_root, "experiment_root")
    checkpoint = _resolved_file(config.baseline_checkpoint, "baseline_checkpoint")
    vjepa = _resolved_file(config.vjepa_checkpoint, "vjepa_checkpoint")
    expected_vjepa = (
        experiment_root / "Drive-JEPA-cache" / OFFICIAL_DRIVE_JEPA_VJEPA_NAME
    ).resolve()
    if vjepa != expected_vjepa:
        raise F2FeasibilityProductionError(
            "live V-JEPA path differs from the official experiment-root asset"
        )
    try:
        restored = restore_authenticated_drive_jepa_baseline(
            checkpoint,
            expected_checkpoint_sha256=OFFICIAL_DRIVE_JEPA_BASELINE_SHA256,
            model_factory=lambda: _official_drive_jepa_model_factory(
                repository_root,
                experiment_root,
                OFFICIAL_DRIVE_JEPA_VJEPA_SHA256,
            ),
            device=config.device,
            factory_source=factory_source,
        )
    except Exception as error:
        raise F2FeasibilityProductionError(
            "official Drive-JEPA/V-JEPA baseline restore failed"
        ) from error
    if restored.factory_source != factory_source:
        raise F2FeasibilityProductionError(
            "restored Drive-JEPA baseline lost factory-source authentication"
        )
    return restored


def preflight_production_runtime(
    *,
    config_path: Union[str, Path],
    expected_config_sha256: str,
) -> Mapping[str, Any]:
    """Restore the signed F2 baseline before the one-shot launch claim.

    The feasibility certificate proves that this path ran when the analytical
    audit was produced.  This preflight deliberately runs it again immediately
    before launch so a broken Python/CUDA environment cannot consume the
    one-shot marker.  The expected config digest comes from the independently
    signed launch authorization; the config and its adjacent sidecar are not
    allowed to nominate their own trust root.
    """

    expected = _sha256(
        expected_config_sha256,
        "external expected feasibility pretraining config SHA-256",
    )
    loaded_config = read_pretraining_config(config_path)
    if loaded_config.sha256 != expected:
        raise F2FeasibilityProductionError(
            "feasibility pretraining config differs from its external expected SHA-256"
        )
    config = loaded_config.config
    code_source = authenticate_feasibility_code_source(
        config.repository_root,
        expected_revision=config.production_code_revision,
    )
    try:
        factory_source = authenticate_drive_jepa_factory_source(
            config.repository_root,
            expected_revision=config.drive_jepa_factory_revision,
        )
    except Exception as error:
        raise F2FeasibilityProductionError(
            "official Drive-JEPA factory source failed prelaunch authentication"
        ) from error
    qualification, qualification_source_files = _load_authenticated_qualification(
        config
    )
    cache = _load_authenticated_metric_cache_manifest(config, qualification)
    raw_inputs = _load_authenticated_raw_inputs(config, qualification)
    baseline = _restore_official_baseline(config, factory_source)
    if (
        baseline.factory_source != factory_source
        or baseline.checkpoint_path != _resolved_file(
            config.baseline_checkpoint, "baseline_checkpoint"
        )
        or baseline.checkpoint_sha256 != config.baseline_checkpoint_sha256
        or isinstance(baseline.state_tensor_count, bool)
        or baseline.state_tensor_count <= 0
    ):
        raise F2FeasibilityProductionError(
            "restored prelaunch baseline identity differs from the signed config"
        )
    return {
        "baseline_checkpoint_sha256": baseline.checkpoint_sha256,
        "baseline_state_tensor_count": baseline.state_tensor_count,
        "code_path_tree_sha256": code_source.path_tree_sha256,
        "code_revision": code_source.revision,
        "config_sha256": loaded_config.sha256,
        "factory_path_tree_sha256": factory_source.path_tree_sha256,
        "factory_revision": factory_source.revision,
        "metric_cache_entry_count": len(cache.manifest.entries),
        "qualification_source_file_count": len(qualification_source_files),
        "raw_input_log_count": len(raw_inputs.manifest.logs),
        "raw_input_sensor_count": len(raw_inputs.manifest.sensor_blobs),
        "vjepa_checkpoint_sha256": config.vjepa_checkpoint_sha256,
    }


def _compute_production_feasibility_certificate(
    loaded_config: LoadedPretrainingConfig,
) -> Mapping[str, Any]:
    """Rebuild one certificate from authenticated raw/cache/model inputs."""

    if type(loaded_config) is not LoadedPretrainingConfig:
        raise F2FeasibilityProductionError(
            "production certificate replay requires a loaded pretraining config"
        )
    config = loaded_config.config
    try:
        deterministic_runtime = initialize_deterministic_runtime(
            logical_rank=0,
            world_size=1,
            explicit_device=config.device,
        )
    except F2ProductionRunnerError as error:
        raise F2FeasibilityProductionError(
            "deterministic feasibility runtime initialization failed"
        ) from error
    code_source = authenticate_feasibility_code_source(
        config.repository_root,
        expected_revision=config.production_code_revision,
    )
    try:
        factory_source = authenticate_drive_jepa_factory_source(
            config.repository_root,
            expected_revision=config.drive_jepa_factory_revision,
        )
    except Exception as error:
        raise F2FeasibilityProductionError(
            "official Drive-JEPA factory source failed authentication"
        ) from error
    qualification, qualification_source_files = _load_authenticated_qualification(
        config
    )
    cache = _load_authenticated_metric_cache_manifest(config, qualification)
    raw_inputs = _load_authenticated_raw_inputs(config, qualification)
    sample_rows = select_deterministic_navtrain_train_sample(qualification)
    if len(sample_rows) != FEASIBILITY_SAMPLE_SIZE:
        raise F2FeasibilityProductionError(
            f"qualification train must provide exactly {FEASIBILITY_SAMPLE_SIZE} sampled rows"
        )
    references = cache.authenticated_references()
    sampled_entries = tuple(references[row.token].entry for row in sample_rows)
    if any(entry.partition != FEASIBILITY_PARTITION for entry in sampled_entries):
        raise F2FeasibilityProductionError(
            "sampled metric-cache entries are outside the train partition"
        )
    feature_build = build_qualified_train_feature_source(
        qualification,
        sample_rows,
        navsim_log_path=config.navsim_log,
        sensor_blobs_path=config.sensor_blobs,
        authenticated_raw_inputs=raw_inputs,
    )
    baseline = _restore_official_baseline(config, factory_source)
    try:
        semantics = DriveJepaNavsimRecordSemantics.official()
        frozen = build_frozen_static_inputs(
            sample_rows,
            feature_source=feature_build.source,
            references=references,
            baseline=baseline,
            semantics=semantics,
        )
        audit = build_train_only_residual_feasibility_audit(
            qualification, frozen
        )
    except Exception as error:
        raise F2FeasibilityProductionError(
            "real train-only residual feasibility computation failed"
        ) from error
    if len(audit.sample_rows) != FEASIBILITY_SAMPLE_SIZE:
        raise F2FeasibilityProductionError(
            "computed feasibility audit lost exact 2048-row coverage"
        )
    provenance = {
        "baseline": {
            "checkpoint_sha256": baseline.checkpoint_sha256,
            "factory_path_tree_sha256": factory_source.path_tree_sha256,
            "factory_revision": factory_source.revision,
            "state_tensor_count": baseline.state_tensor_count,
            "vjepa_checkpoint_sha256": config.vjepa_checkpoint_sha256,
        },
        "code_source": code_source.to_payload(),
        "deterministic_runtime": deterministic_runtime.to_payload(),
        "feature_sample": {
            "removed_scenes": feature_build.removed_scenes,
            "row_identity_sha256": _sample_identity_sha256(sample_rows),
            "token_count": len(sample_rows),
        },
        "metric_cache": {
            "content_entry_count": len(cache.manifest.entries),
            "content_entries_sha256": _cache_entries_sha256(cache),
            "content_manifest_sha256": cache.manifest_sha256,
            "metadata_sha256": cache.manifest.metadata_sha256,
        },
        "qualification": {
            "config_sha256": config.qualification_config_sha256,
            "manifest_sha256": config.qualification_manifest_sha256,
            "source_file_count": len(qualification_source_files),
            "source_files_sha256": _source_files_sha256(
                qualification_source_files
            ),
            "source_manifest_sha256": config.qualification_source_manifest_sha256,
        },
        "raw_input": {
            "content_manifest_sha256": raw_inputs.manifest_sha256,
            "log_entry_count": len(raw_inputs.manifest.logs),
            "sensor_entry_count": len(raw_inputs.manifest.sensor_blobs),
        },
    }
    certificate = production_certificate_payload(
        audit,
        pretraining_config_sha256=loaded_config.sha256,
        provenance=provenance,
    )
    return certificate


def build_production_feasibility_audit(
    *,
    config_path: Union[str, Path],
    output_path: Union[str, Path],
) -> LoadedProductionFeasibilityAudit:
    """Compute and atomically publish the real prelaunch train-only audit."""

    loaded_config = read_pretraining_config(config_path)
    certificate = _compute_production_feasibility_certificate(loaded_config)
    published_sha256 = write_production_feasibility_audit(
        output_path, certificate
    )
    return read_production_feasibility_audit(
        output_path,
        expected_sha256=published_sha256,
        expected_pretraining_config_path=config_path,
        expected_pretraining_config_sha256=loaded_config.sha256,
    )


def replay_production_feasibility_audit(
    *,
    config_path: Union[str, Path],
    expected_config_sha256: str,
    audit_path: Union[str, Path],
    expected_audit_sha256: str,
) -> LoadedProductionFeasibilityAudit:
    """Admission replay from signed inputs, before any one-shot launch marker.

    This is intentionally expensive: it authenticates the external config,
    raw-input and metric-cache manifests, source revisions, official baseline
    and V-JEPA assets; rebuilds the deterministic 2,048 frozen inputs; then
    requires byte-identical canonical certificate content.  A certificate's
    own provenance or frozen-target claims therefore cannot serve as the
    source of truth for admission.
    """

    expected_config = _sha256(
        expected_config_sha256,
        "external expected feasibility pretraining config SHA-256",
    )
    expected_audit = _sha256(
        expected_audit_sha256,
        "external expected production feasibility audit SHA-256",
    )
    loaded_config = read_pretraining_config(config_path)
    if loaded_config.sha256 != expected_config:
        raise F2FeasibilityProductionError(
            "feasibility replay config differs from its external expected SHA-256"
        )
    signed = read_production_feasibility_audit(
        audit_path,
        expected_sha256=expected_audit,
        expected_pretraining_config_path=config_path,
        expected_pretraining_config_sha256=expected_config,
    )
    replayed = _compute_production_feasibility_certificate(loaded_config)
    signed_bytes = canonical_json_bytes(signed.payload)
    replayed_bytes = canonical_json_bytes(replayed)
    if replayed_bytes != signed_bytes:
        raise F2FeasibilityProductionError(
            "replayed feasibility certificate differs from signed audit bytes"
        )
    if _sha256_bytes(replayed_bytes) != expected_audit:
        raise F2FeasibilityProductionError(
            "replayed feasibility certificate differs from external audit SHA-256"
        )
    return LoadedProductionFeasibilityAudit(
        payload=replayed,
        sha256=expected_audit,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the authenticated train-only F2 capacity audit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build-audit",
        help="compute and seal the fixed 2048-row prelaunch audit",
    )
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    replay = subparsers.add_parser(
        "replay-audit",
        help="rebuild the 2048-row certificate and compare signed bytes",
    )
    replay.add_argument("--config", required=True, type=Path)
    replay.add_argument("--expected-config-sha256", required=True)
    replay.add_argument("--audit", required=True, type=Path)
    replay.add_argument("--expected-audit-sha256", required=True)
    preflight = subparsers.add_parser(
        "preflight-runtime",
        help="authenticate and restore the signed baseline before launch",
    )
    preflight.add_argument("--config", required=True, type=Path)
    preflight.add_argument("--expected-config-sha256", required=True)
    return parser


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "build-audit":
        result = build_production_feasibility_audit(
            config_path=arguments.config,
            output_path=arguments.output,
        )
        print(result.sha256)
        return 0
    if arguments.command == "preflight-runtime":
        receipt = preflight_production_runtime(
            config_path=arguments.config,
            expected_config_sha256=arguments.expected_config_sha256,
        )
        print(canonical_json_bytes(receipt).decode("utf-8"), end="")
        return 0
    if arguments.command == "replay-audit":
        replayed = replay_production_feasibility_audit(
            config_path=arguments.config,
            expected_config_sha256=arguments.expected_config_sha256,
            audit_path=arguments.audit,
            expected_audit_sha256=arguments.expected_audit_sha256,
        )
        print(replayed.sha256)
        return 0
    raise F2FeasibilityProductionError("unknown feasibility command")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())


__all__ = [
    "CAPACITY_GATE_METRIC",
    "CAPACITY_GATE_OPERATOR",
    "CAPACITY_GATE_THRESHOLD",
    "CAPACITY_GATE_UNIT",
    "FEASIBILITY_CODE_PATHS",
    "OFFICIAL_DRIVE_JEPA_BASELINE_SHA256",
    "PRETRAINING_CONFIG_SCHEMA_VERSION",
    "PRODUCTION_AUDIT_SCHEMA_VERSION",
    "PRODUCTION_CLAIM_SCOPE",
    "F2FeasibilityProductionError",
    "FeasibilityCodeSourceIdentity",
    "FeasibilityPretrainingConfig",
    "LoadedPretrainingConfig",
    "LoadedProductionFeasibilityAudit",
    "QualifiedTrainFeatureBuild",
    "QualifiedTrainFeatureSource",
    "authenticate_feasibility_code_source",
    "build_frozen_static_inputs",
    "build_production_feasibility_audit",
    "build_qualified_train_feature_source",
    "capacity_gate_passes",
    "cli_main",
    "parse_pretraining_config_bytes",
    "preflight_production_runtime",
    "pretraining_config_bytes",
    "pretraining_config_payload",
    "production_certificate_payload",
    "read_pretraining_config",
    "read_production_feasibility_audit",
    "replay_production_feasibility_audit",
    "validate_production_certificate_payload",
    "write_pretraining_config",
    "write_production_feasibility_audit",
]
