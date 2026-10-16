"""Authenticated Drive-JEPA and qualified-navtrain adapters for F2 records.

The frozen-record builder deliberately accepts injected model and feature
sources.  This module supplies the production NAVSIM implementations without
weakening those injection seams:

* the original official Drive-JEPA Lightning checkpoint is hashed before it
  is deserialized, its uniform ``agent._pad_model.`` namespace is stripped,
  and every model key and tensor shape is checked before a strict restore;
* the exact Drive-JEPA model factory is constructed from an authenticated
  source revision with the same configuration and V-JEPA initialization path
  used by the official Hydra agent; and
* a sensor-qualified held-out navtrain ``SceneLoader``/``Dataset`` is exposed
  in canonical manifest order as batched ``TokenFeatures``.

The runtime source check is intentionally independent from Git ``HEAD``.  A
caller supplies one full expected commit identity, and every tracked source
path that controls the Drive-JEPA factory/forward must match that commit in
the live worktree.  Local edits or untracked files under those paths are a
hard failure.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import os
import re
import stat
import subprocess
import sys
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch

from navsim.planning.training.f2_qualification_manifest import (
    F2_SENSOR_CONTRACT,
    QualificationManifest,
    QualificationPartition,
    constrain_scene_filter,
    qualify_scene_loader,
    require_f2_sensor_contract,
)
from navsim.planning.validation.f2_causal_model_adapter import (
    F2CausalModelAdapter,
    checkpoint_file_sha256,
    strip_state_dict_prefix,
)
from navsim.planning.validation.f2_causal_validator import (
    deterministic_log_assignments,
)
from navsim.planning.validation.f2_frozen_artifact_io import (
    F2_VJEPA_CHECKPOINT_SHA256,
)
from navsim.planning.validation.f2_navsim_record_builder import (
    DriveJepaBaselineForward,
    F2FrozenRecordBuildError,
    TokenFeatures,
)
from navsim.planning.validation.f2_live_provenance import VerifiedF2Provenance
from navsim.planning.validation.f2_navtrain_input_manifest import (
    VerifiedNavtrainRawInputs,
)


OFFICIAL_DRIVE_JEPA_CHECKPOINT_PREFIX = "agent._pad_model."
OFFICIAL_DRIVE_JEPA_CHECKPOINT_NAME = (
    "drive_jepa_perception_based_agent_vitl_v2.ckpt"
)
OFFICIAL_DRIVE_JEPA_VJEPA_NAME = "vitl_merge_3dataset_e50.pt"
OFFICIAL_DRIVE_JEPA_VJEPA_SHA256 = F2_VJEPA_CHECKPOINT_SHA256
OFFICIAL_DRIVE_JEPA_FACTORY_REVISION = (
    "f58f6b6ff850b5c797830dacb524a5bc533e753e"
)
OFFICIAL_DRIVE_JEPA_FACTORY_PATHS = (
    "navsim_v2/navsim/agents/drive_jepa_perception_based",
    "navsim_v2/vjepa2/configs/eval/vitl/in1k.yaml",
    "navsim_v2/vjepa2/evals/image_classification_frozen/modelcustom/vit_encoder.py",
    "navsim_v2/vjepa2/src/masks",
    "navsim_v2/vjepa2/src/models",
    "navsim_v2/vjepa2/src/utils",
)
_IGNORED_RUNTIME_SHADOW_SUFFIXES = (
    ".py",
    ".pyc",
    ".pyd",
    ".pyi",
    ".so",
    ".yaml",
    ".yml",
)
_VJEPA_IMPORTED_MODULE_PATHS = {
    "vjepa2.evals.image_classification_frozen.modelcustom.vit_encoder": (
        "navsim_v2/vjepa2/evals/image_classification_frozen/modelcustom/vit_encoder.py"
    ),
    "vjepa2.src.masks.utils": "navsim_v2/vjepa2/src/masks/utils.py",
    "vjepa2.src.models.utils.modules": (
        "navsim_v2/vjepa2/src/models/utils/modules.py"
    ),
    "vjepa2.src.models.utils.patch_embed": (
        "navsim_v2/vjepa2/src/models/utils/patch_embed.py"
    ),
    "vjepa2.src.models.utils.pos_embs": (
        "navsim_v2/vjepa2/src/models/utils/pos_embs.py"
    ),
    "vjepa2.src.models.vision_transformer": (
        "navsim_v2/vjepa2/src/models/vision_transformer.py"
    ),
    "vjepa2.src.utils.tensors": "navsim_v2/vjepa2/src/utils/tensors.py",
}
DRIVE_JEPA_FEATURE_SHAPES = {
    "camera_feature": (4, 3, 448, 768),
    "img_shape": (4, 3),
    "lidar2img": (4, 4, 4),
    "camera_feature_1": (3, 256, 512),
    "camera_feature_2": (3, 256, 512),
    "ego_status": (4, 11),
}
SUPPORTED_F2_WORLD_SIZES = (1, 2, 8)

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_FACTORY_CONTEXT_LOCK = threading.RLock()
_OFFICIAL_BASELINE_FACTORY_SEAL = object()


class F2NavsimRuntimeAdapterError(F2FrozenRecordBuildError):
    """Raised when a live Drive-JEPA/NAVSIM runtime cannot be authenticated."""


def _validated_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise F2NavsimRuntimeAdapterError(
            f"{name} must be exactly 64 hexadecimal SHA-256 characters"
        )
    return value.lower()


def _validated_commit(value: Any, name: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise F2NavsimRuntimeAdapterError(
            f"{name} must be one full 40-character Git commit identity"
        )
    return value.lower()


def _resolved_file(value: Union[str, Path], name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    return path


def _resolved_directory(value: Union[str, Path], name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{name}: {path}")
    return path


def _run_git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    text: bool = True,
) -> Union[str, bytes]:
    command = ["git", "-C", str(repository_root), *arguments]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise F2NavsimRuntimeAdapterError(
            "cannot authenticate Drive-JEPA runtime source with Git: "
            f"arguments={list(arguments)!r}, stderr={str(stderr).strip()!r}"
        ) from error
    return completed.stdout


@dataclass(frozen=True)
class DriveJepaFactorySourceIdentity:
    """Live source evidence for the exact official Drive-JEPA factory paths."""

    repository_root: Path
    revision: str
    path_tree_sha256: str
    paths: Tuple[str, ...]


def authenticate_drive_jepa_factory_source(
    repository_root: Union[str, Path],
    *,
    expected_revision: str = OFFICIAL_DRIVE_JEPA_FACTORY_REVISION,
    paths: Sequence[str] = OFFICIAL_DRIVE_JEPA_FACTORY_PATHS,
) -> DriveJepaFactorySourceIdentity:
    """Require live factory/forward files to equal one independently named commit."""

    root = _resolved_directory(repository_root, "repository_root")
    revision = _validated_commit(expected_revision, "expected_revision")
    normalized_paths = tuple(str(path).replace("\\", "/") for path in paths)
    if not normalized_paths or any(
        not path or path.startswith("/") or ".." in Path(path).parts
        for path in normalized_paths
    ):
        raise F2NavsimRuntimeAdapterError(
            "Drive-JEPA factory paths must be non-empty repository-relative paths"
        )
    if len(set(normalized_paths)) != len(normalized_paths):
        raise F2NavsimRuntimeAdapterError(
            "Drive-JEPA factory source paths contain duplicates"
        )

    resolved = str(
        _run_git(root, ["rev-parse", "--verify", f"{revision}^{{commit}}"])
    ).strip()
    if resolved.lower() != revision:
        raise F2NavsimRuntimeAdapterError(
            "Git resolved the expected Drive-JEPA source revision ambiguously"
        )

    changed = str(
        _run_git(
            root,
            ["diff", "--name-only", revision, "--", *normalized_paths],
        )
    ).splitlines()
    if changed:
        raise F2NavsimRuntimeAdapterError(
            "live Drive-JEPA factory source differs from the authenticated "
            f"revision: {changed[:8]}"
        )
    untracked = str(
        _run_git(
            root,
            [
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                *normalized_paths,
            ],
        )
    ).splitlines()
    if untracked:
        raise F2NavsimRuntimeAdapterError(
            "untracked files can shadow the authenticated Drive-JEPA runtime: "
            f"{untracked[:8]}"
        )
    ignored = str(
        _run_git(
            root,
            [
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--",
                *normalized_paths,
            ],
        )
    ).splitlines()
    ignored_shadows = [
        path
        for path in ignored
        if path.lower().endswith(_IGNORED_RUNTIME_SHADOW_SUFFIXES)
    ]
    if ignored_shadows:
        raise F2NavsimRuntimeAdapterError(
            "ignored files can shadow the authenticated Drive-JEPA runtime: "
            f"{ignored_shadows[:8]}"
        )

    digest = hashlib.sha256()
    for relative_path in normalized_paths:
        tracked = str(
            _run_git(root, ["ls-tree", "-r", "--name-only", revision, "--", relative_path])
        ).splitlines()
        if not tracked:
            raise F2NavsimRuntimeAdapterError(
                f"authenticated revision lacks Drive-JEPA source path {relative_path!r}"
            )
        for file_path in tracked:
            payload = _run_git(root, ["show", f"{revision}:{file_path}"], text=False)
            assert isinstance(payload, bytes)
            live_path = root / file_path
            if not live_path.is_file():
                raise F2NavsimRuntimeAdapterError(
                    f"live Drive-JEPA source file is missing: {file_path!r}"
                )
            live_payload = live_path.read_bytes()
            if live_payload != payload:
                raise F2NavsimRuntimeAdapterError(
                    f"live Drive-JEPA source bytes drifted: {file_path!r}"
                )
            encoded_name = file_path.encode("utf-8")
            digest.update(len(encoded_name).to_bytes(4, "little"))
            digest.update(encoded_name)
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return DriveJepaFactorySourceIdentity(
        repository_root=root,
        revision=revision,
        path_tree_sha256=digest.hexdigest(),
        paths=normalized_paths,
    )


class _AuthenticatedVJepaTorchProxy:
    """Override only vit_encoder's module-local ``torch.load`` lookup."""

    def __init__(self, delegate: Any, load: Callable[..., Any]) -> None:
        self._delegate = delegate
        self.load = load

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def _file_identity(value: os.stat_result) -> Tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@contextmanager
def _official_factory_context(
    runtime_root: Path,
    experiment_root: Path,
    *,
    expected_vjepa_sha256: str,
) -> Iterable[None]:
    """Serialize the legacy factory's process-wide cwd/env requirements."""

    navsim_root = runtime_root / "navsim_v2"
    config_path = navsim_root / "vjepa2" / "configs" / "eval" / "vitl" / "in1k.yaml"
    pretrained_path = (
        experiment_root / "Drive-JEPA-cache" / OFFICIAL_DRIVE_JEPA_VJEPA_NAME
    )
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not pretrained_path.is_file():
        raise FileNotFoundError(pretrained_path)
    expected_digest = _validated_sha256(
        expected_vjepa_sha256, "expected_vjepa_sha256"
    )
    if expected_digest != OFFICIAL_DRIVE_JEPA_VJEPA_SHA256:
        raise F2NavsimRuntimeAdapterError(
            "official V-JEPA checkpoint differs from the frozen F2 asset"
        )

    with _FACTORY_CONTEXT_LOCK:
        previous_directory = Path.cwd()
        previous_root = os.environ.get("NAVSIM_EXP_ROOT")
        checkpoint_file = None
        vit_encoder_module = None
        original_vit_torch = None
        loaded = 0
        try:
            path_before = pretrained_path.lstat()
            if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(
                path_before.st_mode
            ):
                raise F2NavsimRuntimeAdapterError(
                    "official V-JEPA checkpoint must be a regular non-symlink file"
                )
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(pretrained_path, flags)
            checkpoint_file = os.fdopen(descriptor, "rb", closefd=True)
            before = os.fstat(checkpoint_file.fileno())
            if _file_identity(before) != _file_identity(path_before):
                raise F2NavsimRuntimeAdapterError(
                    "official V-JEPA checkpoint changed before authentication"
                )
            digest = hashlib.sha256()
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
            actual_digest = digest.hexdigest()
            if actual_digest != expected_digest:
                raise F2NavsimRuntimeAdapterError(
                    "official V-JEPA checkpoint SHA-256 mismatch: "
                    f"expected={expected_digest}, actual={actual_digest}"
                )
            checkpoint_file.seek(0)

            os.environ["NAVSIM_EXP_ROOT"] = str(experiment_root)
            os.chdir(navsim_root)
            vit_encoder_module = importlib.import_module(
                "vjepa2.evals.image_classification_frozen.modelcustom.vit_encoder"
            )
            for module_name, relative_path in _VJEPA_IMPORTED_MODULE_PATHS.items():
                module = sys.modules.get(module_name)
                module_file = getattr(module, "__file__", None)
                expected_path = (runtime_root / relative_path).resolve()
                if module_file is None or Path(module_file).resolve() != expected_path:
                    raise F2NavsimRuntimeAdapterError(
                        "loaded V-JEPA module is outside the authenticated runtime: "
                        f"{module_name}"
                    )
            original_vit_torch = vit_encoder_module.torch

            def authenticated_load(source: Any, *args: Any, **kwargs: Any) -> Any:
                nonlocal loaded
                if isinstance(source, (str, os.PathLike)):
                    source_path = Path(source).expanduser().resolve()
                    if source_path == pretrained_path:
                        loaded += 1
                        if loaded != 1:
                            raise F2NavsimRuntimeAdapterError(
                                "official V-JEPA checkpoint was loaded more than once"
                            )
                        assert checkpoint_file is not None
                        checkpoint_file.seek(0)
                        return original_vit_torch.load(
                            checkpoint_file, *args, **kwargs
                        )
                return original_vit_torch.load(source, *args, **kwargs)

            vit_encoder_module.torch = _AuthenticatedVJepaTorchProxy(
                original_vit_torch, authenticated_load
            )
            yield
            if loaded != 1:
                raise F2NavsimRuntimeAdapterError(
                    "official factory did not load the authenticated V-JEPA checkpoint"
                )
            after = os.fstat(checkpoint_file.fileno())
            path_after = pretrained_path.lstat()
            if (
                _file_identity(before) != _file_identity(after)
                or _file_identity(before) != _file_identity(path_after)
            ):
                raise F2NavsimRuntimeAdapterError(
                    "official V-JEPA checkpoint changed while loaded"
                )
        finally:
            if vit_encoder_module is not None and original_vit_torch is not None:
                vit_encoder_module.torch = original_vit_torch
            if checkpoint_file is not None:
                checkpoint_file.close()
            os.chdir(previous_directory)
            if previous_root is None:
                os.environ.pop("NAVSIM_EXP_ROOT", None)
            else:
                os.environ["NAVSIM_EXP_ROOT"] = previous_root


@contextmanager
def authenticated_f2_vjepa_load_context(
    repository_root: Union[str, Path],
    experiment_root: Union[str, Path],
    *,
    expected_vjepa_sha256: str = OFFICIAL_DRIVE_JEPA_VJEPA_SHA256,
) -> Iterable[None]:
    """Authenticate and deserialize the F2 V-JEPA asset from one descriptor.

    The regular official Drive-JEPA path remains untouched.  F2 training and
    causal factories explicitly opt into this context at their model
    construction boundary.
    """

    runtime_root = _resolved_directory(repository_root, "repository_root")
    experiment = _resolved_directory(experiment_root, "experiment_root")
    with _official_factory_context(
        runtime_root,
        experiment,
        expected_vjepa_sha256=expected_vjepa_sha256,
    ):
        yield


def _official_drive_jepa_model_factory(
    runtime_root: Path,
    experiment_root: Path,
    expected_vjepa_sha256: str,
) -> torch.nn.Module:
    """Instantiate the same model/configuration selected by official Hydra YAML."""

    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import (
        DriveJEPAConfig,
    )
    from navsim.agents.drive_jepa_perception_based.drive_jepa_model import (
        DriveJEPAModel,
    )

    with _official_factory_context(
        runtime_root,
        experiment_root,
        expected_vjepa_sha256=expected_vjepa_sha256,
    ):
        config = DriveJEPAConfig()
        if (
            config.proposal_num != 32
            or config.num_poses != 8
            or config.ref_num != 4
            or config.latent is not False
            or config.trajectory_sampling.time_horizon != 4
            or config.trajectory_sampling.interval_length != 0.5
        ):
            raise F2NavsimRuntimeAdapterError(
                "official Drive-JEPA default model configuration drifted"
            )
        return DriveJEPAModel(config)


CheckpointLoader = Callable[[Path], Any]
ModelFactory = Callable[[], torch.nn.Module]


def _default_checkpoint_loader(path: Path) -> Any:
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))


def _stable_authenticated_checkpoint_load(path: Path, expected_sha256: str) -> Any:
    """Hash and load one unchanged regular-file descriptor."""

    try:
        path_before = path.lstat()
    except OSError as error:
        raise F2NavsimRuntimeAdapterError(
            f"cannot stat official Drive-JEPA checkpoint {path}"
        ) from error
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise F2NavsimRuntimeAdapterError(
            "official Drive-JEPA checkpoint must be a regular non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as checkpoint_file:
            before = os.fstat(checkpoint_file.fileno())
            path_identity = (
                path_before.st_dev,
                path_before.st_ino,
                path_before.st_size,
                path_before.st_mtime_ns,
            )
            descriptor_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            if descriptor_identity != path_identity:
                raise F2NavsimRuntimeAdapterError(
                    "official Drive-JEPA checkpoint changed before authentication"
                )
            digest = hashlib.sha256()
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected_sha256:
                raise F2NavsimRuntimeAdapterError(
                    "official Drive-JEPA checkpoint SHA-256 mismatch: "
                    f"expected={expected_sha256}, actual={actual}"
                )
            checkpoint_file.seek(0)
            try:
                payload = torch.load(
                    checkpoint_file,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
            except TypeError:
                checkpoint_file.seek(0)
                payload = torch.load(
                    checkpoint_file, map_location=torch.device("cpu")
                )
            after = os.fstat(checkpoint_file.fileno())
        path_after = path.lstat()
    except F2NavsimRuntimeAdapterError:
        raise
    except OSError as error:
        raise F2NavsimRuntimeAdapterError(
            f"cannot read stable official Drive-JEPA checkpoint {path}"
        ) from error
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    final_path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
    )
    if descriptor_identity != after_identity or descriptor_identity != final_path_identity:
        raise F2NavsimRuntimeAdapterError(
            "official Drive-JEPA checkpoint changed while authenticated and loaded"
        )
    return payload


@dataclass(frozen=True)
class RestoredDriveJepaBaseline:
    """Strict official-baseline forward plus immutable authentication evidence."""

    forward: DriveJepaBaselineForward
    checkpoint_path: Path
    checkpoint_sha256: str
    state_tensor_count: int
    factory_source: Optional[DriveJepaFactorySourceIdentity]
    live_provenance: Optional[VerifiedF2Provenance]
    _official_factory_seal: object = field(
        default=None, repr=False, compare=False
    )

    @property
    def is_official_factory(self) -> bool:
        """True only for the closed official model-factory construction path."""

        return self._official_factory_seal is _OFFICIAL_BASELINE_FACTORY_SEAL


def restore_authenticated_drive_jepa_baseline(
    checkpoint_path: Union[str, Path],
    *,
    expected_checkpoint_sha256: str,
    model_factory: ModelFactory,
    device: Union[str, torch.device] = "cpu",
    checkpoint_loader: Optional[CheckpointLoader] = None,
    factory_source: Optional[DriveJepaFactorySourceIdentity] = None,
    live_provenance: Optional[VerifiedF2Provenance] = None,
) -> RestoredDriveJepaBaseline:
    """Low-level injected restore; never accepted as an official baseline."""

    return _restore_authenticated_drive_jepa_baseline(
        checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        model_factory=model_factory,
        device=device,
        checkpoint_loader=checkpoint_loader,
        factory_source=factory_source,
        live_provenance=live_provenance,
    )


def _restore_authenticated_drive_jepa_baseline(
    checkpoint_path: Union[str, Path],
    *,
    expected_checkpoint_sha256: str,
    model_factory: ModelFactory,
    device: Union[str, torch.device] = "cpu",
    checkpoint_loader: Optional[CheckpointLoader] = None,
    factory_source: Optional[DriveJepaFactorySourceIdentity] = None,
    live_provenance: Optional[VerifiedF2Provenance] = None,
    _official_factory_seal: object | None = None,
) -> RestoredDriveJepaBaseline:
    """Internal strict restore core shared by generic and fixed factories."""

    path = _resolved_file(checkpoint_path, "checkpoint_path")
    expected_digest = _validated_sha256(
        expected_checkpoint_sha256, "expected_checkpoint_sha256"
    )
    if live_provenance is not None:
        if type(live_provenance) is not VerifiedF2Provenance:
            raise F2NavsimRuntimeAdapterError(
                "live_provenance must be the concrete sealed VerifiedF2Provenance"
            )
        if live_provenance.bindings.baseline_checkpoint_sha256 != expected_digest:
            raise F2NavsimRuntimeAdapterError(
                "official baseline checkpoint differs from live provenance bindings"
            )
    if not callable(model_factory):
        raise F2NavsimRuntimeAdapterError("model_factory must be callable")

    if checkpoint_loader is None:
        checkpoint = _stable_authenticated_checkpoint_load(path, expected_digest)
        actual_digest = expected_digest
    else:
        actual_digest = checkpoint_file_sha256(path)
        if actual_digest != expected_digest:
            raise F2NavsimRuntimeAdapterError(
                "official Drive-JEPA checkpoint SHA-256 mismatch: "
                f"expected={expected_digest}, actual={actual_digest}"
            )
        checkpoint = checkpoint_loader(path)
    if not isinstance(checkpoint, Mapping):
        raise F2NavsimRuntimeAdapterError(
            "official Drive-JEPA checkpoint payload must be a mapping"
        )
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise F2NavsimRuntimeAdapterError(
            "official Drive-JEPA checkpoint state_dict must be a non-empty mapping"
        )
    validated_state: Dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str) or not key:
            raise F2NavsimRuntimeAdapterError(
                "official Drive-JEPA state_dict contains a non-string/empty key"
            )
        if not isinstance(value, torch.Tensor):
            raise F2NavsimRuntimeAdapterError(
                f"official Drive-JEPA state tensor {key!r} is not a tensor"
            )
        if torch.is_complex(value):
            raise F2NavsimRuntimeAdapterError(
                f"official Drive-JEPA state tensor {key!r} is complex"
            )
        if torch.is_floating_point(value) and value.dtype != torch.float32:
            raise F2NavsimRuntimeAdapterError(
                f"official Drive-JEPA state tensor {key!r} is not float32"
            )
        validated_state[key] = value

    try:
        transformed = strip_state_dict_prefix(
            OFFICIAL_DRIVE_JEPA_CHECKPOINT_PREFIX
        )(validated_state)
    except Exception as error:
        raise F2NavsimRuntimeAdapterError(
            "official Drive-JEPA checkpoint namespace is not uniformly "
            f"{OFFICIAL_DRIVE_JEPA_CHECKPOINT_PREFIX!r}"
        ) from error

    model = model_factory()
    if not isinstance(model, torch.nn.Module):
        raise F2NavsimRuntimeAdapterError(
            "model_factory must return a torch.nn.Module"
        )
    model_state = model.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(transformed)
    if model_keys != checkpoint_keys:
        raise F2NavsimRuntimeAdapterError(
            "strict official Drive-JEPA state key mismatch before restore: "
            f"missing={sorted(model_keys - checkpoint_keys)[:8]}, "
            f"unexpected={sorted(checkpoint_keys - model_keys)[:8]}"
        )
    shape_drift = sorted(
        (
            key,
            tuple(transformed[key].shape),
            tuple(model_state[key].shape),
        )
        for key in model_keys
        if tuple(transformed[key].shape) != tuple(model_state[key].shape)
    )
    if shape_drift:
        raise F2NavsimRuntimeAdapterError(
            "strict official Drive-JEPA state tensor shape mismatch: "
            f"{shape_drift[:8]}"
        )
    dtype_drift = sorted(
        (key, str(transformed[key].dtype), str(model_state[key].dtype))
        for key in model_keys
        if transformed[key].dtype != model_state[key].dtype
    )
    if dtype_drift:
        raise F2NavsimRuntimeAdapterError(
            "strict official Drive-JEPA state tensor dtype mismatch: "
            f"{dtype_drift[:8]}"
        )
    try:
        incompatible = model.load_state_dict(transformed, strict=True)
    except Exception as error:
        raise F2NavsimRuntimeAdapterError(
            "strict official Drive-JEPA checkpoint restore failed"
        ) from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise F2NavsimRuntimeAdapterError(
            "strict official Drive-JEPA restore returned incompatible keys"
        )

    state_tensor_count = len(transformed)
    del transformed
    del validated_state
    del raw_state
    del checkpoint
    gc.collect()
    forward = DriveJepaBaselineForward(
        model=model,
        authenticated_baseline_sha256=actual_digest,
        device=device,
    )
    return RestoredDriveJepaBaseline(
        forward=forward,
        checkpoint_path=path,
        checkpoint_sha256=actual_digest,
        state_tensor_count=state_tensor_count,
        factory_source=factory_source,
        live_provenance=live_provenance,
        _official_factory_seal=_official_factory_seal,
    )


def build_official_drive_jepa_baseline(
    checkpoint_path: Union[str, Path],
    *,
    expected_checkpoint_sha256: str,
    repository_root: Union[str, Path],
    experiment_root: Union[str, Path],
    live_provenance: VerifiedF2Provenance,
    expected_source_revision: str = OFFICIAL_DRIVE_JEPA_FACTORY_REVISION,
    device: Union[str, torch.device] = "cpu",
) -> RestoredDriveJepaBaseline:
    """Authenticate source + checkpoint and construct the production baseline."""

    runtime_root = _resolved_directory(repository_root, "repository_root")
    exp_root = _resolved_directory(experiment_root, "experiment_root")
    if type(live_provenance) is not VerifiedF2Provenance:
        raise F2NavsimRuntimeAdapterError(
            "live_provenance must be the concrete sealed VerifiedF2Provenance"
        )
    if (
        live_provenance.bindings.baseline_checkpoint_sha256
        != _validated_sha256(
            expected_checkpoint_sha256, "expected_checkpoint_sha256"
        )
    ):
        raise F2NavsimRuntimeAdapterError(
            "official baseline checkpoint differs from live provenance bindings"
        )
    source = authenticate_drive_jepa_factory_source(
        runtime_root, expected_revision=expected_source_revision
    )
    return _restore_authenticated_drive_jepa_baseline(
        checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        model_factory=lambda: _official_drive_jepa_model_factory(
            runtime_root,
            exp_root,
            live_provenance.bindings.vjepa_checkpoint_sha256,
        ),
        device=device,
        factory_source=source,
        live_provenance=live_provenance,
        _official_factory_seal=_OFFICIAL_BASELINE_FACTORY_SEAL,
    )


def _official_f2_lightning_model_factory(
    runtime_root: Path,
    experiment_root: Path,
    checkpoint_path: Path,
    expected_vjepa_sha256: str,
) -> torch.nn.Module:
    """Build the exact Lightning module whose state namespace F2 saved."""

    from navsim.agents.actworld_jepa.actworld_agent import ActWorldJEPAAgent
    from navsim.agents.actworld_jepa.actworld_config import ActWorldJEPAConfig
    from navsim.planning.training.agent_lightning_module import AgentLightningModule

    with _official_factory_context(
        runtime_root,
        experiment_root,
        expected_vjepa_sha256=expected_vjepa_sha256,
    ):
        config = ActWorldJEPAConfig(
            actworld_training_stage="critic",
            actworld_critic_variant="f2_base_residual",
            actworld_freeze_drive_jepa=True,
            actworld_future_latent_grounding=False,
            baseline_checkpoint_path="",
        )
        ActWorldJEPAAgent._validate_f2_fixed_config(config)
        agent = ActWorldJEPAAgent(
            config=config,
            lr=ActWorldJEPAAgent.F2_AGENT_LR,
            # A non-empty path selects inference construction without loading
            # it here.  F2CausalModelAdapter owns the single strict restore.
            checkpoint_path=str(checkpoint_path),
        )
        if type(agent) is not ActWorldJEPAAgent:
            raise F2NavsimRuntimeAdapterError(
                "official F2 factory returned a substituted agent type"
            )
        lightning = AgentLightningModule(agent=agent)
        if type(lightning) is not AgentLightningModule:
            raise F2NavsimRuntimeAdapterError(
                "official F2 factory returned a substituted Lightning wrapper"
            )
        return lightning


def _official_f2_forward(
    restored_model: torch.nn.Module, features: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Fixed causal inference call for an official F2 Lightning checkpoint."""

    from navsim.agents.actworld_jepa.actworld_agent import ActWorldJEPAAgent
    from navsim.planning.training.agent_lightning_module import AgentLightningModule

    if type(restored_model) is not AgentLightningModule:
        raise F2NavsimRuntimeAdapterError(
            "restored official F2 model is not AgentLightningModule"
        )
    agent = restored_model.agent
    if type(agent) is not ActWorldJEPAAgent:
        raise F2NavsimRuntimeAdapterError(
            "restored official F2 Lightning agent type drifted"
        )
    return agent.forward(features)


def build_official_f2_causal_model_adapter(
    checkpoint_path: Union[str, Path],
    *,
    expected_checkpoint_sha256: str,
    expected_epoch: int,
    repository_root: Union[str, Path],
    experiment_root: Union[str, Path],
    live_provenance: VerifiedF2Provenance,
    expected_source_revision: str = OFFICIAL_DRIVE_JEPA_FACTORY_REVISION,
    device: Union[str, torch.device] = "cpu",
) -> F2CausalModelAdapter:
    """Restore F2 only through the authenticated official model/forward path."""

    if type(live_provenance) is not VerifiedF2Provenance:
        raise F2NavsimRuntimeAdapterError(
            "live_provenance must be the concrete sealed VerifiedF2Provenance"
        )
    runtime_root = _resolved_directory(repository_root, "repository_root")
    experiment = _resolved_directory(experiment_root, "experiment_root")
    checkpoint = _resolved_file(checkpoint_path, "checkpoint_path")
    source = authenticate_drive_jepa_factory_source(
        runtime_root, expected_revision=expected_source_revision
    )
    adapter = F2CausalModelAdapter._from_official_checkpoint(
        checkpoint,
        expected_sha256=expected_checkpoint_sha256,
        expected_epoch=expected_epoch,
        model_factory=lambda: _official_f2_lightning_model_factory(
            runtime_root,
            experiment,
            checkpoint,
            live_provenance.bindings.vjepa_checkpoint_sha256,
        ),
        forward_fn=_official_f2_forward,
        live_provenance=live_provenance,
        factory_source=source,
        device=device,
    )
    if not adapter.is_official_factory or adapter.live_provenance is not live_provenance:
        raise F2NavsimRuntimeAdapterError(
            "official F2 adapter lost its factory/provenance seal"
        )
    return adapter


def f2_sensor_config() -> Any:
    """Build the exact explicit finite-index SensorConfig frozen by F2."""

    from navsim.common.dataclasses import SensorConfig

    contract = {name: list(indices) for name, indices in F2_SENSOR_CONTRACT}
    sensor_config = SensorConfig(**contract)
    require_f2_sensor_contract(sensor_config)
    return sensor_config


def _row_identity(row: Any) -> Tuple[str, int, str, bool]:
    try:
        identity = (
            row.log_name,
            row.time_us,
            row.token,
            row.future_camera_valid,
        )
    except AttributeError as error:
        raise F2NavsimRuntimeAdapterError(
            "runtime row lacks a canonical F2 identity"
        ) from error
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
        raise F2NavsimRuntimeAdapterError("runtime row identity is malformed")
    return identity


def _rank_world(rank: Any, world_size: Any) -> Tuple[int, int]:
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size not in SUPPORTED_F2_WORLD_SIZES
        or not (0 <= rank < world_size)
    ):
        raise F2NavsimRuntimeAdapterError(
            "F2 feature-source rank/world_size must describe a supported 1/2/8 world"
        )
    return rank, world_size


def _assigned_held_out_rows(
    qualification: QualificationManifest,
    rank: int,
    world_size: int,
) -> Tuple[Tuple[Any, ...], Tuple[str, ...]]:
    """Derive one whole-log shard only from the authenticated qualification."""

    assignments = deterministic_log_assignments(
        qualification.held_out.rows, world_size
    )
    logs = assignments[rank]
    grouped = {
        log_name: tuple(
            row
            for row in qualification.held_out.rows
            if row.log_name == log_name
        )
        for log_name in qualification.held_out.logs
    }
    rows = tuple(row for log_name in logs for row in grouped[log_name])
    return rows, logs


@dataclass(frozen=True)
class _AssignedQualificationPartitionView:
    """Minimal view accepted by the qualification preflight helpers."""

    rows: Tuple[Any, ...]
    name: str = "held_out"

    @property
    def tokens(self) -> Tuple[str, ...]:
        return tuple(row.token for row in self.rows)

    @property
    def logs(self) -> Tuple[str, ...]:
        return tuple(sorted({row.log_name for row in self.rows}))


def _validate_unbatched_drive_jepa_features(
    features: Any,
) -> Dict[str, torch.Tensor]:
    if not isinstance(features, Mapping):
        raise F2NavsimRuntimeAdapterError(
            "Drive-JEPA feature Dataset must return a mapping"
        )
    actual_keys = set(features)
    expected_keys = set(DRIVE_JEPA_FEATURE_SHAPES)
    if actual_keys != expected_keys:
        raise F2NavsimRuntimeAdapterError(
            "Drive-JEPA feature key contract drifted: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    validated: Dict[str, torch.Tensor] = {}
    for name, shape in DRIVE_JEPA_FEATURE_SHAPES.items():
        value = features[name]
        if not isinstance(value, torch.Tensor):
            raise F2NavsimRuntimeAdapterError(
                f"Drive-JEPA feature {name!r} must be a tensor"
            )
        if value.dtype != torch.float32:
            raise F2NavsimRuntimeAdapterError(
                f"Drive-JEPA feature {name!r} must be float32"
            )
        if tuple(value.shape) != shape:
            raise F2NavsimRuntimeAdapterError(
                f"Drive-JEPA feature {name!r} shape must be {shape}, "
                f"got {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise F2NavsimRuntimeAdapterError(
                f"Drive-JEPA feature {name!r} contains NaN or infinity"
            )
        validated[name] = value.detach().clone().contiguous().unsqueeze(0)
    return validated


class QualifiedHeldOutNavtrainFeatureSource:
    """Canonical one-token feature source backed by real NAVSIM Dataset semantics."""

    dataset = "navtrain"
    partition = "held_out"

    def __init__(
        self,
        *,
        qualification: QualificationManifest,
        live_provenance: VerifiedF2Provenance,
        scene_loader: Any,
        feature_dataset: Any,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
            DriveJEPAFeatureBuilder,
        )

        if not isinstance(qualification, QualificationManifest):
            raise F2NavsimRuntimeAdapterError(
                "qualification must be an authenticated QualificationManifest"
            )
        if type(live_provenance) is not VerifiedF2Provenance:
            raise F2NavsimRuntimeAdapterError(
                "live_provenance must be the concrete sealed VerifiedF2Provenance"
            )
        if live_provenance.qualification != qualification:
            raise F2NavsimRuntimeAdapterError(
                "live provenance qualification differs from the feature manifest"
            )
        self._qualification = qualification
        self._live_provenance = live_provenance
        self._partition = qualification.held_out
        self._rank, self._world_size = _rank_world(rank, world_size)
        self._rows, self._logs = _assigned_held_out_rows(
            qualification, self._rank, self._world_size
        )
        self._scene_loader = scene_loader
        self._feature_dataset = feature_dataset
        require_f2_sensor_contract(getattr(scene_loader, "_sensor_config", None))
        if qualification.sensor_contract != F2_SENSOR_CONTRACT:
            raise F2NavsimRuntimeAdapterError(
                "qualification sensor contract differs from frozen F2 sensors"
            )
        self.tokens = tuple(scene_loader.tokens)
        expected_tokens = tuple(row.token for row in self._rows)
        if self.tokens != expected_tokens:
            raise F2NavsimRuntimeAdapterError(
                "held-out SceneLoader order differs from the canonical rank assignment"
            )
        if getattr(feature_dataset, "_scene_loader", None) is not scene_loader:
            raise F2NavsimRuntimeAdapterError(
                "feature Dataset is not backed by the qualified SceneLoader"
            )
        builders = tuple(getattr(feature_dataset, "_feature_builders", ()))
        if len(builders) != 1 or type(builders[0]) is not DriveJEPAFeatureBuilder:
            raise F2NavsimRuntimeAdapterError(
                "feature Dataset must use exactly the official DriveJEPAFeatureBuilder"
            )
        if builders[0].get_unique_name() != "pad_feature":
            raise F2NavsimRuntimeAdapterError(
                "official Drive-JEPA feature-builder identity drifted"
            )
        if tuple(getattr(feature_dataset, "_target_builders", ())) != ():
            raise F2NavsimRuntimeAdapterError(
                "held-out feature Dataset must not load privileged target builders"
            )
        if getattr(feature_dataset, "_cache_path", None) is not None:
            raise F2NavsimRuntimeAdapterError(
                "F2 runtime features must be computed from qualified sensor blobs"
            )
        if len(feature_dataset) != len(self.tokens):
            raise F2NavsimRuntimeAdapterError(
                "held-out feature Dataset length differs from canonical coverage"
            )
        self._index_by_token = {row.token: index for index, row in enumerate(self._rows)}
        frames_by_token = getattr(scene_loader, "scene_frames_dicts", None)
        scene_filter = getattr(scene_loader, "_scene_filter", None)
        history = getattr(scene_filter, "num_history_frames", None)
        if (
            not isinstance(frames_by_token, Mapping)
            or isinstance(history, bool)
            or not isinstance(history, int)
        ):
            raise F2NavsimRuntimeAdapterError(
                "qualified SceneLoader lacks auditable regular-scene metadata"
            )
        current_index = history - 1
        for row in self._rows:
            frames = frames_by_token.get(row.token)
            if not isinstance(frames, list) or not (0 <= current_index < len(frames)):
                raise F2NavsimRuntimeAdapterError(
                    f"qualified row {row.token!r} has malformed raw frames"
                )
            current = frames[current_index]
            if not isinstance(current, Mapping) or (
                current.get("log_name"),
                current.get("timestamp"),
                current.get("token"),
            ) != (row.log_name, row.time_us, row.token):
                raise F2NavsimRuntimeAdapterError(
                    f"qualified raw row identity drifted for token {row.token!r}"
                )

    @property
    def live_provenance(self) -> VerifiedF2Provenance:
        """Return the exact sealed provenance instance bound at construction."""

        return self._live_provenance

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def logs(self) -> Tuple[str, ...]:
        return self._logs

    def load(self, row: Any) -> TokenFeatures:
        identity = _row_identity(row)
        index = self._index_by_token.get(identity[2])
        if index is None or _row_identity(self._rows[index]) != identity:
            raise F2NavsimRuntimeAdapterError(
                "requested feature row differs from canonical held-out qualification"
            )
        if self.tokens[index] != identity[2]:
            raise F2NavsimRuntimeAdapterError(
                "held-out feature Dataset token order drifted at access time"
            )
        item = self._feature_dataset[index]
        if not isinstance(item, tuple) or len(item) != 2:
            raise F2NavsimRuntimeAdapterError(
                "NAVSIM feature Dataset item must be a (features, targets) tuple"
            )
        features, targets = item
        if not isinstance(targets, Mapping) or targets:
            raise F2NavsimRuntimeAdapterError(
                "held-out feature Dataset unexpectedly exposed target values"
            )
        batched = _validate_unbatched_drive_jepa_features(features)
        return TokenFeatures(
            token=identity[2],
            log_name=identity[0],
            time_us=identity[1],
            features=batched,
        )


@dataclass(frozen=True)
class QualifiedFeatureSourceBuild:
    """Production source together with the zero-removal availability witness."""

    source: QualifiedHeldOutNavtrainFeatureSource
    availability: Any


def build_qualified_held_out_navtrain_feature_source(
    qualification: QualificationManifest,
    *,
    live_provenance: VerifiedF2Provenance,
    navsim_log_path: Union[str, Path],
    sensor_blobs_path: Union[str, Path],
    authenticated_raw_inputs: VerifiedNavtrainRawInputs,
    rank: int = 0,
    world_size: int = 1,
) -> QualifiedFeatureSourceBuild:
    """Build, sensor-preflight, canonicalize, and wrap held-out navtrain."""

    if not isinstance(qualification, QualificationManifest):
        raise F2NavsimRuntimeAdapterError(
            "qualification must be an authenticated QualificationManifest"
        )
    if qualification.dataset != "navtrain":
        raise F2NavsimRuntimeAdapterError("F2 runtime accepts navtrain only")
    if type(live_provenance) is not VerifiedF2Provenance:
        raise F2NavsimRuntimeAdapterError(
            "live_provenance must be the concrete sealed VerifiedF2Provenance"
        )
    if live_provenance.qualification != qualification:
        raise F2NavsimRuntimeAdapterError(
            "live provenance qualification differs from the feature manifest"
        )
    if type(authenticated_raw_inputs) is not VerifiedNavtrainRawInputs:
        raise F2NavsimRuntimeAdapterError(
            "production feature source requires sealed navtrain raw inputs"
        )
    authenticated_raw_inputs.require_roots(
        Path(navsim_log_path), Path(sensor_blobs_path)
    )

    from navsim.agents.drive_jepa_perception_based.drive_jepa_config import (
        DriveJEPAConfig,
    )
    from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
        DriveJEPAFeatureBuilder,
    )
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.training.dataset import Dataset

    rank, world_size = _rank_world(rank, world_size)
    assigned_rows, _ = _assigned_held_out_rows(qualification, rank, world_size)
    partition = _AssignedQualificationPartitionView(assigned_rows)
    sensor_config = f2_sensor_config()
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        max_scenes=None,
        # Start from the authenticated complete allowlist; constrain_scene_filter
        # then narrows it to the deterministic whole-log rank assignment.
        log_names=list(qualification.held_out.logs),
        tokens=list(qualification.held_out.tokens),
        include_synthetic_scenes=False,
        synthetic_scene_tokens=None,
    )
    constrain_scene_filter(scene_filter, partition)
    loader = SceneLoader(
        data_path=_resolved_directory(navsim_log_path, "navsim_log_path"),
        sensor_blobs_path=_resolved_directory(
            sensor_blobs_path, "sensor_blobs_path"
        ),
        synthetic_scenes_path=None,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
        authenticated_raw_inputs=authenticated_raw_inputs,
    )
    availability = qualify_scene_loader(loader, partition)
    if availability.removed_scenes != 0:
        raise F2NavsimRuntimeAdapterError(
            "qualified held-out navtrain unexpectedly removed sensor rows"
        )
    feature_builder = DriveJEPAFeatureBuilder(config=DriveJEPAConfig())
    dataset = Dataset(
        scene_loader=loader,
        feature_builders=[feature_builder],
        target_builders=[],
        cache_path=None,
        force_cache_computation=False,
    )
    source = QualifiedHeldOutNavtrainFeatureSource(
        qualification=qualification,
        live_provenance=live_provenance,
        scene_loader=loader,
        feature_dataset=dataset,
        rank=rank,
        world_size=world_size,
    )
    return QualifiedFeatureSourceBuild(source=source, availability=availability)
