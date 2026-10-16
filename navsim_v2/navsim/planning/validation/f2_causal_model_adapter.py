"""Strict F2 checkpoint-to-``ModelTokenOutput`` inference adapter.

This module owns only the model boundary of the preregistered F2 validator:

* authenticate and structurally validate one F2 Stage-A checkpoint;
* restore its state into an injected model with ``strict=True``;
* run one-token ActWorld inference in float32 with autocast disabled; and
* expose the original proposals and raw F2 logits as ``ModelTokenOutput``.

It deliberately does not build a held-out manifest, read a metric cache, run
either NAVSIM simulator, calibrate comfort, or aggregate benchmark metrics.
Those operations remain outside this adapter so CPU-only mock tests can audit
the checkpoint and numeric boundary without importing the NAVSIM runtime.

The checkpoint-file digest authenticated here is not a run-configuration
digest.  Binding the frozen Hydra/configuration SHA to a manifest is an
explicit blocker for the external runner; this module does not claim that
binding has happened.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from navsim.planning.validation.f2_causal_validator import (
    CANDIDATE_COUNT,
    F2_CHECKPOINT_EPOCHS,
    F2_MODEL_NUMERIC_MODE,
    ModelTokenOutput,
)


F2_CHECKPOINT_STAGE = "critic"
F2_CHECKPOINT_VARIANT = "f2_base_residual"
_AUTHENTICATED_ADAPTER_SEAL = object()
_OFFICIAL_F2_FACTORY_SEAL = object()
ORIGINAL_PROPOSAL_HASH_SCHEMA = "actworld-f2-original-proposals-f32le-v1"
CONFIGURATION_DIGEST_BINDING = "external_runner_required"
F2_FROZEN_ACTOR_TENSOR_NAMES = (
    "refinement_terminal_weight",
    "refinement_terminal_bias",
    "refinement_gate_logits",
    "score_delta_gate_raw",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_MODEL_OUTPUT_KEYS = (
    "proposals",
    "actworld_original_candidates",
    "pred_logit",
    "actworld_base_candidate_logits",
    "actworld_residual_logits",
    "actworld_delta_logits",
    "actworld_utility_logits",
)


class F2CausalModelAdapterError(RuntimeError):
    """Raised when the checkpoint or model boundary violates the F2 contract."""


@dataclass(frozen=True)
class F2CheckpointIdentity:
    """Authenticated identity retained after a strict checkpoint restore."""

    path: Path
    sha256: str
    epoch: int
    stage: str = F2_CHECKPOINT_STAGE
    variant: str = F2_CHECKPOINT_VARIANT


CheckpointLoader = Callable[[Path], Any]
ModelFactory = Callable[[], torch.nn.Module]
ForwardFunction = Callable[[torch.nn.Module, Mapping[str, Any]], Mapping[str, Any]]
StateDictTransform = Callable[
    [Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]
]
FrozenActorTensorGetter = Callable[
    [torch.nn.Module], Mapping[str, torch.Tensor]
]


def _validated_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise F2CausalModelAdapterError(
            f"{name} must be exactly 64 hexadecimal SHA-256 characters"
        )
    return value.lower()


def checkpoint_file_sha256(path: Union[str, Path]) -> str:
    """Hash checkpoint bytes without deserializing them."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_original_proposal_sha256(proposals: torch.Tensor) -> str:
    """Hash one token's original proposals in a stable float32 wire format.

    Both the batched ``[1, 32, poses, 3]`` ActWorld output and its batchless
    ``[32, poses, 3]`` form are accepted.  The batch dimension is not part of
    the identity.  Values are canonicalized to finite CPU float32, signed zero
    is collapsed, and row-major values are encoded as little-endian binary32.
    The artifact builder must call this same function for digest parity.
    """

    if not isinstance(proposals, torch.Tensor):
        raise F2CausalModelAdapterError("original proposals must be a torch.Tensor")
    if proposals.ndim == 4:
        if proposals.shape[0] != 1:
            raise F2CausalModelAdapterError(
                "original proposal hashing accepts exactly one batch item"
            )
        canonical = proposals[0]
    elif proposals.ndim == 3:
        canonical = proposals
    else:
        raise F2CausalModelAdapterError(
            "original proposals must have shape [1, 32, poses, 3] or "
            f"[32, poses, 3], got {tuple(proposals.shape)}"
        )
    if (
        canonical.shape[0] != CANDIDATE_COUNT
        or canonical.shape[1] < 1
        or canonical.shape[2] != 3
    ):
        raise F2CausalModelAdapterError(
            "original proposals must have shape [32, poses>=1, 3], "
            f"got {tuple(canonical.shape)}"
        )
    if not torch.is_floating_point(canonical):
        raise F2CausalModelAdapterError("original proposals must be floating point")
    canonical = canonical.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(canonical).all().item()):
        raise F2CausalModelAdapterError("original proposals contain non-finite values")
    canonical = torch.where(canonical == 0, torch.zeros_like(canonical), canonical)

    header = json.dumps(
        {
            "dtype": "float32-le",
            "schema": ORIGINAL_PROPOSAL_HASH_SCHEMA,
            "shape": list(canonical.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    flat_values = canonical.reshape(-1).tolist()
    payload = struct.pack(f"<{len(flat_values)}f", *flat_values)
    return hashlib.sha256(header + payload).hexdigest()


def strip_state_dict_prefix(prefix: str) -> StateDictTransform:
    """Return a strict transformer for a uniformly prefixed state dictionary.

    A direct ``ActWorldJEPAModel`` restore can use
    ``strip_state_dict_prefix('agent._pad_model.')``.  Restoring the original
    Lightning wrapper needs no transform because its keys already match.
    """

    if not isinstance(prefix, str) or not prefix:
        raise F2CausalModelAdapterError("state-dict prefix must be non-empty")

    def transform(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        invalid = sorted(key for key in state_dict if not key.startswith(prefix))
        if invalid:
            raise F2CausalModelAdapterError(
                "checkpoint state contains keys outside the required prefix: "
                f"{invalid[:8]}"
            )
        stripped: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            transformed_key = key[len(prefix) :]
            if not transformed_key:
                raise F2CausalModelAdapterError(
                    "checkpoint state contains an empty key after prefix stripping"
                )
            if transformed_key in stripped:
                raise F2CausalModelAdapterError(
                    f"duplicate state key after prefix stripping: {transformed_key!r}"
                )
            stripped[transformed_key] = value
        return stripped

    return transform


def _default_checkpoint_loader(path: Path) -> Any:
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        # PyTorch 2.1 does not expose the later ``weights_only`` keyword.
        return torch.load(path, map_location=torch.device("cpu"))


def _stable_authenticated_checkpoint_load(path: Path, expected_sha256: str) -> Any:
    """Hash and deserialize the same regular-file descriptor.

    The production path must not hash one pathname instance and then reopen a
    substituted checkpoint for deserialization.  Keeping one descriptor open
    also avoids materializing a multi-gigabyte checkpoint as an extra bytes
    object.
    """

    try:
        path_before = path.lstat()
    except OSError as error:
        raise F2CausalModelAdapterError(
            f"cannot stat F2 checkpoint {path}"
        ) from error
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise F2CausalModelAdapterError(
            "F2 checkpoint must be a regular non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as checkpoint_file:
            before = os.fstat(checkpoint_file.fileno())
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                path_before.st_dev,
                path_before.st_ino,
                path_before.st_size,
                path_before.st_mtime_ns,
            ):
                raise F2CausalModelAdapterError(
                    "F2 checkpoint changed before authentication"
                )
            digest = hashlib.sha256()
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise F2CausalModelAdapterError(
                    "F2 checkpoint SHA-256 mismatch: "
                    f"expected={expected_sha256}, actual={actual_sha256}"
                )
            checkpoint_file.seek(0)
            try:
                checkpoint = torch.load(
                    checkpoint_file,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
            except TypeError:
                checkpoint_file.seek(0)
                checkpoint = torch.load(
                    checkpoint_file, map_location=torch.device("cpu")
                )
            after = os.fstat(checkpoint_file.fileno())
        path_after = path.lstat()
    except F2CausalModelAdapterError:
        raise
    except OSError as error:
        raise F2CausalModelAdapterError(
            f"cannot read stable F2 checkpoint {path}"
        ) from error
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(path_after):
        raise F2CausalModelAdapterError(
            "F2 checkpoint changed while it was authenticated and loaded"
        )
    return checkpoint


def _validated_expected_epoch(epoch: int) -> int:
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise F2CausalModelAdapterError("expected_epoch must be an integer")
    if epoch not in F2_CHECKPOINT_EPOCHS:
        raise F2CausalModelAdapterError(
            "expected_epoch must be one of the frozen F2 checkpoint epochs "
            f"{F2_CHECKPOINT_EPOCHS}, found {epoch!r}"
        )
    return epoch


def _validated_state_dict(
    value: Any, name: str = "checkpoint state_dict"
) -> Dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise F2CausalModelAdapterError(f"{name} must be a non-empty mapping")
    state_dict: Dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not key:
            raise F2CausalModelAdapterError(f"{name} contains a non-string/empty key")
        if not isinstance(tensor, torch.Tensor):
            raise F2CausalModelAdapterError(
                f"{name}[{key!r}] must be a torch.Tensor"
            )
        if torch.is_complex(tensor):
            raise F2CausalModelAdapterError(
                f"{name}[{key!r}] must not have a complex dtype"
            )
        if torch.is_floating_point(tensor) and tensor.dtype != torch.float32:
            raise F2CausalModelAdapterError(
                f"{name}[{key!r}] must be float32, found {tensor.dtype}"
            )
        state_dict[key] = tensor
    return state_dict


def _load_authenticated_checkpoint(
    checkpoint_path: Union[str, Path],
    expected_sha256: str,
    expected_epoch: int,
    checkpoint_loader: Optional[CheckpointLoader],
) -> Tuple[F2CheckpointIdentity, Dict[str, torch.Tensor]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_digest = _validated_sha256(expected_sha256, "expected_sha256")
    epoch = _validated_expected_epoch(expected_epoch)
    if checkpoint_loader is None:
        checkpoint = _stable_authenticated_checkpoint_load(path, expected_digest)
        actual_digest = expected_digest
    else:
        # Dependency injection remains available for low-level tests.  It is
        # never used by the fixed production factory.
        actual_digest = checkpoint_file_sha256(path)
        if actual_digest != expected_digest:
            raise F2CausalModelAdapterError(
                "F2 checkpoint SHA-256 mismatch: "
                f"expected={expected_digest}, actual={actual_digest}"
            )
        checkpoint = checkpoint_loader(path)
    if not isinstance(checkpoint, Mapping):
        raise F2CausalModelAdapterError("F2 checkpoint payload must be a mapping")

    stage = checkpoint.get("actworld_training_stage")
    if stage != F2_CHECKPOINT_STAGE:
        raise F2CausalModelAdapterError(
            "F2 checkpoint must have actworld_training_stage='critic'; "
            f"found {stage!r}"
        )
    variant = checkpoint.get("actworld_critic_variant")
    if variant != F2_CHECKPOINT_VARIANT:
        raise F2CausalModelAdapterError(
            "F2 checkpoint must have actworld_critic_variant="
            f"'{F2_CHECKPOINT_VARIANT}'; found {variant!r}"
        )
    saved_epoch = checkpoint.get("epoch")
    if isinstance(saved_epoch, bool) or not isinstance(saved_epoch, int):
        raise F2CausalModelAdapterError(
            f"F2 checkpoint epoch must be an integer; found {saved_epoch!r}"
        )
    if saved_epoch != epoch:
        raise F2CausalModelAdapterError(
            f"F2 checkpoint epoch mismatch: expected={epoch}, found={saved_epoch}"
        )
    state_dict = _validated_state_dict(checkpoint.get("state_dict"))
    identity = F2CheckpointIdentity(
        path=path.resolve(), sha256=actual_digest, epoch=epoch
    )
    return identity, state_dict


def _prepare_features(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        if torch.is_complex(value):
            raise F2CausalModelAdapterError("model features must not be complex")
        dtype = torch.float32 if torch.is_floating_point(value) else value.dtype
        return value.detach().to(device=device, dtype=dtype)
    if isinstance(value, Mapping):
        return {key: _prepare_features(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_prepare_features(item, device) for item in value)
    if isinstance(value, list):
        return [_prepare_features(item, device) for item in value]
    return value


def _require_float32_output(
    output: Mapping[str, Any],
    key: str,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_device: Optional[torch.device] = None,
) -> torch.Tensor:
    value = output.get(key)
    if not isinstance(value, torch.Tensor):
        raise F2CausalModelAdapterError(
            f"ActWorld inference output {key!r} must be a torch.Tensor"
        )
    if value.dtype != torch.float32:
        raise F2CausalModelAdapterError(
            f"ActWorld inference output {key!r} must be float32, found {value.dtype}"
        )
    if expected_device is not None and value.device != expected_device:
        raise F2CausalModelAdapterError(
            f"ActWorld inference output {key!r} must remain on device "
            f"{expected_device}, found {value.device}"
        )
    if expected_shape is not None and tuple(value.shape) != expected_shape:
        raise F2CausalModelAdapterError(
            f"ActWorld inference output {key!r} must have shape "
            f"{expected_shape}, got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise F2CausalModelAdapterError(
            f"ActWorld inference output {key!r} contains non-finite values"
        )
    if value.requires_grad:
        raise F2CausalModelAdapterError(
            f"ActWorld inference output {key!r} unexpectedly requires gradients"
        )
    return value


def _default_frozen_actor_tensor_getter(
    model: torch.nn.Module,
) -> Mapping[str, torch.Tensor]:
    """Locate the four functionally active F2 actor tensors."""

    attribute_paths = (
        ("agent", "_pad_model", "_actworld_planner"),
        ("_pad_model", "_actworld_planner"),
        ("_actworld_planner",),
    )
    planner: Any = None
    for path in attribute_paths:
        candidate: Any = model
        for attribute in path:
            if not hasattr(candidate, attribute):
                candidate = None
                break
            candidate = getattr(candidate, attribute)
        if candidate is not None:
            planner = candidate
            break
    if planner is None:
        raise F2CausalModelAdapterError(
            "cannot locate the ActWorld planner for mandatory frozen-actor "
            "functional-zero audit; inject frozen_actor_tensor_getter for a "
            "custom model wrapper"
        )
    try:
        terminal = planner.refinement_head[-1]
        tensors = {
            "refinement_terminal_weight": terminal.weight,
            "refinement_terminal_bias": terminal.bias,
            "refinement_gate_logits": planner.refinement_gate_logits,
            "score_delta_gate_raw": planner.score_delta_gate_raw,
        }
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise F2CausalModelAdapterError(
            "ActWorld planner does not expose the four mandatory frozen-actor "
            "functional tensors"
        ) from error
    return tensors


def _validate_frozen_actor_zero(
    model: torch.nn.Module,
    device: torch.device,
    getter: Optional[FrozenActorTensorGetter],
) -> None:
    resolver = getter or _default_frozen_actor_tensor_getter
    if not callable(resolver):
        raise F2CausalModelAdapterError(
            "frozen_actor_tensor_getter must be callable"
        )
    tensors = resolver(model)
    if not isinstance(tensors, Mapping):
        raise F2CausalModelAdapterError(
            "frozen actor tensor getter must return a mapping"
        )
    expected_names = set(F2_FROZEN_ACTOR_TENSOR_NAMES)
    actual_names = set(tensors.keys())
    if actual_names != expected_names:
        raise F2CausalModelAdapterError(
            "frozen actor tensor audit requires exactly "
            f"{F2_FROZEN_ACTOR_TENSOR_NAMES}; missing="
            f"{sorted(expected_names - actual_names)}, unexpected="
            f"{sorted(actual_names - expected_names)}"
        )
    nonzero = []
    for name in F2_FROZEN_ACTOR_TENSOR_NAMES:
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise F2CausalModelAdapterError(
                f"frozen actor tensor {name!r} must be a torch.Tensor"
            )
        if tensor.dtype != torch.float32:
            raise F2CausalModelAdapterError(
                f"frozen actor tensor {name!r} must be float32, found {tensor.dtype}"
            )
        if tensor.device != device:
            raise F2CausalModelAdapterError(
                f"frozen actor tensor {name!r} must be on {device}, "
                f"found {tensor.device}"
            )
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise F2CausalModelAdapterError(
                f"frozen actor tensor {name!r} contains non-finite values"
            )
        if bool(torch.count_nonzero(tensor.detach()).item()):
            nonzero.append(name)
    if nonzero:
        raise F2CausalModelAdapterError(
            "F2 checkpoint violates the frozen actor functional-zero "
            f"invariant: nonzero={nonzero}"
        )


class F2CausalModelAdapter:
    """Authenticated F2 critic restored behind a one-token inference seam."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        identity: F2CheckpointIdentity,
        device: torch.device,
        forward_fn: ForwardFunction,
        _authenticated_seal: object | None = None,
        _official_factory_seal: object | None = None,
        _live_provenance: Any = None,
        _factory_source: Any = None,
    ) -> None:
        if _authenticated_seal is not _AUTHENTICATED_ADAPTER_SEAL:
            raise F2CausalModelAdapterError(
                "F2CausalModelAdapter must be constructed by from_checkpoint()"
            )
        self._model = model
        self._identity = identity
        self._device = device
        self._forward_fn = forward_fn
        self._official_factory = _official_factory_seal is _OFFICIAL_F2_FACTORY_SEAL
        self._live_provenance = _live_provenance if self._official_factory else None
        self._factory_source = _factory_source if self._official_factory else None

        if self._official_factory:
            # Import lazily so the low-level, dependency-injected adapter stays
            # usable in CPU-only contract tests without creating an import
            # cycle through the provenance verifier.
            from navsim.planning.validation.f2_live_provenance import (
                VerifiedF2Provenance,
            )

            if type(_live_provenance) is not VerifiedF2Provenance:
                raise F2CausalModelAdapterError(
                    "the official F2 factory requires sealed live provenance"
                )

    @property
    def checkpoint_identity(self) -> F2CheckpointIdentity:
        return self._identity

    @property
    def model(self) -> torch.nn.Module:
        """Expose the restored model for read-only audit/introspection."""

        return self._model

    @property
    def is_official_factory(self) -> bool:
        """Whether construction used the fixed production ActWorld factory."""

        return self._official_factory

    @property
    def live_provenance(self) -> Any:
        """The exact verified provenance instance used by production restore."""

        return self._live_provenance

    @property
    def factory_source(self) -> Any:
        """Authenticated source identity for the fixed production factory."""

        return self._factory_source

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        expected_sha256: str,
        expected_epoch: int,
        model_factory: ModelFactory,
        device: Union[str, torch.device] = "cpu",
        forward_fn: Optional[ForwardFunction] = None,
        state_dict_transform: Optional[StateDictTransform] = None,
        checkpoint_loader: Optional[CheckpointLoader] = None,
        frozen_actor_tensor_getter: Optional[FrozenActorTensorGetter] = None,
    ) -> "F2CausalModelAdapter":
        """Low-level injected adapter; never authoritative in the F2 runner."""

        return cls._restore_checkpoint(
            checkpoint_path,
            expected_sha256=expected_sha256,
            expected_epoch=expected_epoch,
            model_factory=model_factory,
            device=device,
            forward_fn=forward_fn,
            state_dict_transform=state_dict_transform,
            checkpoint_loader=checkpoint_loader,
            frozen_actor_tensor_getter=frozen_actor_tensor_getter,
        )

    @classmethod
    def _restore_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        expected_sha256: str,
        expected_epoch: int,
        model_factory: ModelFactory,
        device: Union[str, torch.device] = "cpu",
        forward_fn: Optional[ForwardFunction] = None,
        state_dict_transform: Optional[StateDictTransform] = None,
        checkpoint_loader: Optional[CheckpointLoader] = None,
        frozen_actor_tensor_getter: Optional[FrozenActorTensorGetter] = None,
        _official_factory_seal: object | None = None,
        _live_provenance: Any = None,
        _factory_source: Any = None,
    ) -> "F2CausalModelAdapter":
        """Internal restore core shared by generic tests and the fixed factory."""

        if not callable(model_factory):
            raise F2CausalModelAdapterError("model_factory must be callable")
        resolved_device = torch.device(device)
        if resolved_device.type not in {"cpu", "cuda"}:
            raise F2CausalModelAdapterError(
                "F2 authoritative inference supports only CPU or CUDA devices"
            )
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise F2CausalModelAdapterError("requested CUDA device is unavailable")
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())

        identity, checkpoint_state = _load_authenticated_checkpoint(
            checkpoint_path,
            expected_sha256,
            expected_epoch,
            checkpoint_loader,
        )
        transformed = (
            state_dict_transform(checkpoint_state)
            if state_dict_transform is not None
            else checkpoint_state
        )
        transformed_state = _validated_state_dict(
            transformed, "transformed checkpoint state_dict"
        )

        model = model_factory()
        if not isinstance(model, torch.nn.Module):
            raise F2CausalModelAdapterError(
                "model_factory must return a torch.nn.Module"
            )
        model_state = model.state_dict()
        model_keys = set(model_state.keys())
        checkpoint_keys = set(transformed_state.keys())
        if model_keys != checkpoint_keys:
            missing = sorted(model_keys - checkpoint_keys)
            unexpected = sorted(checkpoint_keys - model_keys)
            raise F2CausalModelAdapterError(
                "strict F2 state key mismatch before restore: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        shape_mismatches = sorted(
            (
                key,
                tuple(transformed_state[key].shape),
                tuple(model_state[key].shape),
            )
            for key in model_keys
            if tuple(transformed_state[key].shape) != tuple(model_state[key].shape)
        )
        if shape_mismatches:
            raise F2CausalModelAdapterError(
                "strict F2 state tensor shape mismatch before restore: "
                f"{shape_mismatches[:8]}"
            )
        dtype_mismatches = sorted(
            (
                key,
                str(transformed_state[key].dtype),
                str(model_state[key].dtype),
            )
            for key in model_keys
            if transformed_state[key].dtype != model_state[key].dtype
        )
        if dtype_mismatches:
            raise F2CausalModelAdapterError(
                "strict F2 state tensor dtype mismatch before restore: "
                f"{dtype_mismatches[:8]}"
            )
        try:
            incompatible = model.load_state_dict(transformed_state, strict=True)
        except Exception as error:
            raise F2CausalModelAdapterError(
                "strict F2 checkpoint state restore failed"
            ) from error
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise F2CausalModelAdapterError(
                "strict F2 restore returned incompatible keys: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        model.to(device=resolved_device, dtype=torch.float32)
        model.eval()
        _validate_frozen_actor_zero(
            model, resolved_device, frozen_actor_tensor_getter
        )
        inference = forward_fn or (lambda restored_model, features: restored_model(features))
        if not callable(inference):
            raise F2CausalModelAdapterError("forward_fn must be callable")
        return cls(
            model=model,
            identity=identity,
            device=resolved_device,
            forward_fn=inference,
            _authenticated_seal=_AUTHENTICATED_ADAPTER_SEAL,
            _official_factory_seal=_official_factory_seal,
            _live_provenance=_live_provenance,
            _factory_source=_factory_source,
        )

    @classmethod
    def _from_official_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        expected_sha256: str,
        expected_epoch: int,
        model_factory: ModelFactory,
        forward_fn: ForwardFunction,
        live_provenance: Any,
        factory_source: Any,
        device: Union[str, torch.device] = "cpu",
        checkpoint_loader: Optional[CheckpointLoader] = None,
        frozen_actor_tensor_getter: Optional[FrozenActorTensorGetter] = None,
    ) -> "F2CausalModelAdapter":
        """Internal production seam used by the fixed NAVSIM runtime factory."""

        return cls._restore_checkpoint(
            checkpoint_path,
            expected_sha256=expected_sha256,
            expected_epoch=expected_epoch,
            model_factory=model_factory,
            device=device,
            forward_fn=forward_fn,
            checkpoint_loader=checkpoint_loader,
            frozen_actor_tensor_getter=frozen_actor_tensor_getter,
            _official_factory_seal=_OFFICIAL_F2_FACTORY_SEAL,
            _live_provenance=live_provenance,
            _factory_source=factory_source,
        )

    def infer_token(
        self, token: str, features: Mapping[str, Any]
    ) -> ModelTokenOutput:
        """Run one causal token and return raw logits for the frozen proposals."""

        if not isinstance(token, str) or not token:
            raise F2CausalModelAdapterError("token must be a non-empty string")
        if not isinstance(features, Mapping):
            raise F2CausalModelAdapterError("features must be a mapping")
        self._model.eval()
        prepared_features = _prepare_features(features, self._device)
        with torch.inference_mode():
            with torch.autocast(device_type=self._device.type, enabled=False):
                output = self._forward_fn(self._model, prepared_features)
        if not isinstance(output, Mapping):
            raise F2CausalModelAdapterError(
                "ActWorld inference must return a mapping"
            )
        missing = [key for key in _MODEL_OUTPUT_KEYS if key not in output]
        if missing:
            raise F2CausalModelAdapterError(
                f"ActWorld inference is missing required F2 outputs: {missing}"
            )

        proposals = _require_float32_output(
            output, "proposals", expected_device=self._device
        )
        if (
            proposals.ndim != 4
            or proposals.shape[0] != 1
            or proposals.shape[1] != CANDIDATE_COUNT
            or proposals.shape[2] < 1
            or proposals.shape[3] != 3
        ):
            raise F2CausalModelAdapterError(
                "ActWorld original proposals must have shape [1, 32, poses>=1, 3], "
                f"got {tuple(proposals.shape)}"
            )
        planner_original = _require_float32_output(
            output,
            "actworld_original_candidates",
            tuple(proposals.shape),
            self._device,
        )
        if not torch.equal(proposals, planner_original):
            raise F2CausalModelAdapterError(
                "ActWorld planner original candidates differ from Drive-JEPA proposals"
            )

        expected_logit_shape = (1, CANDIDATE_COUNT)
        base_logits = _require_float32_output(
            output,
            "actworld_base_candidate_logits",
            expected_logit_shape,
            self._device,
        )
        residual_logits = _require_float32_output(
            output,
            "actworld_residual_logits",
            expected_logit_shape,
            self._device,
        )
        delta_logits = _require_float32_output(
            output,
            "actworld_delta_logits",
            expected_logit_shape,
            self._device,
        )
        utility_logits = _require_float32_output(
            output,
            "actworld_utility_logits",
            expected_logit_shape,
            self._device,
        )
        pred_logit = _require_float32_output(
            output, "pred_logit", expected_device=self._device
        )
        if (
            pred_logit.ndim != 3
            or tuple(pred_logit.shape[:2]) != expected_logit_shape
            or pred_logit.shape[2] < 1
        ):
            raise F2CausalModelAdapterError(
                "ActWorld pred_logit must have shape [1, 32, channels>=1], "
                f"got {tuple(pred_logit.shape)}"
            )
        if not torch.equal(base_logits, pred_logit[..., -1]):
            raise F2CausalModelAdapterError(
                "ActWorld base_candidate_logits differ from pred_logit[..., -1]"
            )
        expected_delta = torch.tanh(residual_logits)
        if not torch.equal(delta_logits, expected_delta):
            mismatch_indices = torch.nonzero(
                delta_logits != expected_delta, as_tuple=False
            )[:, 1].detach().cpu().tolist()
            raise F2CausalModelAdapterError(
                "ActWorld delta_logits do not exactly equal device float32 "
                "torch.tanh(residual_logits); candidate_indices="
                f"{mismatch_indices[:8]}"
            )
        expected_utility = base_logits.detach() + delta_logits
        if not torch.equal(utility_logits, expected_utility):
            mismatch_indices = torch.nonzero(
                utility_logits != expected_utility, as_tuple=False
            )[:, 1].detach().cpu().tolist()
            raise F2CausalModelAdapterError(
                "ActWorld utility_logits do not exactly equal device float32 "
                "base_logits.detach() + delta_logits; candidate_indices="
                f"{mismatch_indices[:8]}"
            )

        proposal_sha256 = canonical_original_proposal_sha256(proposals)
        return ModelTokenOutput(
            token=token,
            proposal_sha256=proposal_sha256,
            base_logits=tuple(float(value) for value in base_logits[0].cpu().tolist()),
            residual_logits=tuple(
                float(value) for value in residual_logits[0].cpu().tolist()
            ),
            delta_logits=tuple(
                float(value) for value in delta_logits[0].cpu().tolist()
            ),
            utility_logits=tuple(
                float(value) for value in utility_logits[0].cpu().tolist()
            ),
            numeric_mode=F2_MODEL_NUMERIC_MODE,
        )


__all__ = (
    "F2CausalModelAdapter",
    "F2CausalModelAdapterError",
    "F2CheckpointIdentity",
    "CONFIGURATION_DIGEST_BINDING",
    "F2_FROZEN_ACTOR_TENSOR_NAMES",
    "canonical_original_proposal_sha256",
    "checkpoint_file_sha256",
    "strip_state_dict_prefix",
)
