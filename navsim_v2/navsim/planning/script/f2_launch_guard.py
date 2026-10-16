"""Fail-closed authorization guard for the single preregistered F2 run.

The production v2 trust root is canonical JSON at one repository-external
fixed path plus an armored detached OpenSSH signature.  The verifier trusts
only the public key embedded in reviewed source; it accepts no caller-supplied
revision or artifact digest.  It re-hashes all named inputs, authenticates the
exact clean Git revision, replays the referenced F1 YAML evidence, derives the
fixed-gate conjunction from raw metrics, and checks one-shot filesystem state.
The same module provides explicit offline issuance and verified O_EXCL install
commands, neither of which is reachable from the no-argument launcher.

``F1_epoch_4`` is a legacy natural-language label.  The executable F1 schedule
has ``max_epochs=4`` and therefore contains Lightning indices 0, 1, 2, and 3.
This module never accepts or requests an index-4/fifth epoch.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableSet, Optional, Sequence, Tuple

try:  # Offline issuance intentionally needs only the Python standard library.
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised on the offline signer host
    yaml = None  # type: ignore[assignment]


ATTESTATION_SCHEMA_VERSION = "actworld-f2-trigger-attestation-v1"
AUTHORIZATION_SCHEMA_VERSION = "actworld-f2-launch-authorization-v2"
AUTHORIZATION_PURPOSE = "single-preregistered-navtrain-f2"
AUTHORIZATION_SSH_NAMESPACE = "actworld-f2-launch-authorization-v2"
AUTHORIZATION_SSH_PRINCIPAL = "actworld-f2-offline-authority"
CLAIM_RECEIPT_SCHEMA_VERSION = "actworld-f2-runtime-claim-v1"

# These are deliberately source constants, not environment-controlled paths.
# A production rank must reopen the same authority bytes used by the dedicated
# launcher; an arbitrary Hydra/torchrun caller cannot redirect verification to
# another, self-signed contract.
PRODUCTION_F2_AUTHORIZATION_PATH = Path(
    "/opt/actworld/"
    "ActWorld-JEPA-f2-launch-authority/f2_launch_authorization_v2.json"
)
PRODUCTION_F2_AUTHORIZATION_SIGNATURE_PATH = Path(
    "/opt/actworld/"
    "ActWorld-JEPA-f2-launch-authority/f2_launch_authorization_v2.json.sig"
)

# This is the dedicated ActWorld-JEPA offline signing identity.  Only the
# public half is embedded.  The private key is never read during verification
# and is never stored in the repository or on the training server.  A source
# revision containing a different key cannot satisfy an authorization that
# names the original revision.
TRUSTED_AUTHORITY_SSH_PUBLIC_KEY = (
    "ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIOvDNQsyZ2xrRJhHIouym3l7apgQG6C7nZZWi71T1QiT"
)

F2_EXPERIMENT_NAME = "train_actworld_jepa_v2_f2"
F2_EXPERIMENT_UID = "f2-seed0-preregistered"

AUTHORIZED_ARTIFACT_KEYS: Tuple[str, ...] = (
    "baseline_checkpoint",
    "vjepa_checkpoint",
    "cache_metadata",
    "cache_content_manifest",
    "raw_input_manifest",
    "qualification_manifest",
    "qualification_config",
    "qualification_source_manifest",
    "feasibility_config",
    "feasibility_audit",
)

VERIFIED_CONTRACT_KEYS: Tuple[str, ...] = (
    "AUTHORIZATION_SHA256",
    "SIGNATURE_SHA256",
    "SOURCE_REVISION",
    "REPOSITORY_ROOT",
    "NAVSIM_EXP_ROOT",
    "OPENSCENE_DATA_ROOT",
    "NUPLAN_MAPS_ROOT",
    "NAVSIM_LOG_ROOT",
    "SENSOR_BLOBS_ROOT",
    "QUALIFICATION_SOURCE_ROOT",
    "EXPERIMENT_NAME",
    "EXPERIMENT_UID",
    "EXPERIMENT_ROOT",
    "OUTPUT_DIR",
    "LAUNCH_MARKER",
) + tuple(
    field
    for artifact in AUTHORIZED_ARTIFACT_KEYS
    for field in (f"{artifact.upper()}_PATH", f"{artifact.upper()}_SHA256")
)
F1_RECORD_REPOSITORY_PATH = (
    "docs/experiments/stage_a_f1_f58f6b6_20260824.yaml"
)
SCHEDULED_EPOCH_COUNT = 4
FINAL_SCHEDULED_EPOCH_INDEX = 3
LEGACY_LABEL_INTERPRETATION = (
    "F1_epoch_4 is the fourth completed epoch (zero-based index 3), "
    "not index 4"
)

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
RUN_UID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)

GATE_ORDER: Tuple[str, ...] = (
    "critic_selected_score",
    "critic_delta",
    "critic_pairwise_accuracy",
    "proposal_headroom_closed",
    "critic_score_error_excess",
    "nc_delta",
    "dac_delta",
    "tlc_delta",
    "ddc_delta",
    "future_camera_valid",
    "baseline_and_oracle_drift",
    "fatal_or_nonfinite",
)

FIXED_GATE_VALUES: Mapping[str, Decimal] = {
    "baseline_score_reference": Decimal("0.8717616200"),
    "best_of_32_score_reference": Decimal("0.9576369524"),
    "critic_delta_min": Decimal("0.0214688331"),
    "critic_selected_score_min": Decimal("0.8932304531"),
    "critic_pairwise_accuracy_min": Decimal("0.70"),
    "proposal_headroom_closed_min": Decimal("0.25"),
    "critic_score_error_excess_max": Decimal("0.02"),
    "hard_factor_delta_min": Decimal("-0.001"),
    "future_camera_valid_min": Decimal("0.99"),
    "baseline_and_oracle_drift_max": Decimal("0.001"),
    "fatal_or_nonfinite_matches_max": Decimal("0"),
}

_F1_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "run",
        "lineage",
        "data",
        "optimization",
        "frozen_hypothesis",
        "selection",
        "epoch_indexing_clarification",
        "startup_evidence",
        "epoch_records",
        "checkpoints",
        "results",
    }
)

_RUN_ALLOWED_KEYS = frozenset(
    {
        "project",
        "stage",
        "fallback",
        "status",
        "started_at",
        "completed_at",
        "finished_at",
        "stopped_at",
        "uid",
        "code_revision",
        "repository_head_at_launch",
        "repository_head_note",
        "torchrun_pid_at_launch",
        "output_dir",
        "log",
        "log_bytes",
        "log_sha256",
        "terminal_log_marker",
        "wandb_offline_record",
        "wandb_offline_record_bytes",
        "wandb_offline_record_sha256",
        "resolved_config_path",
        "resolved_config_bytes",
        "resolved_config_sha256",
        "hydra_overrides_path",
        "hydra_overrides_bytes",
        "hydra_overrides_sha256",
    }
)

_RAW_METRIC_ALLOWED_KEYS = frozenset(
    {
        "baseline_score",
        "best_of_32_score",
        "mean_candidate_score",
        "critic_selected_score",
        "critic_delta",
        "critic_oracle_regret",
        "critic_pairwise_accuracy",
        "critic_score_error",
        "baseline_score_error",
        "critic_score_error_excess",
        "proposal_headroom",
        "proposal_headroom_closed_fraction",
        "critic_nc_delta",
        "critic_dac_delta",
        "critic_tlc_delta",
        "critic_ddc_delta",
        "future_camera_valid_min",
        "future_camera_valid_mean",
        "future_camera_valid_max",
        "future_camera_valid_samples",
        "baseline_reference_drift",
        "best_of_32_reference_drift",
    }
)


class F2LaunchGuardError(ValueError):
    """Raised whenever the F2 one-shot launch authorization is not proven."""


@dataclass(frozen=True)
class VerifiedF2TriggerAttestation:
    """Result of complete byte, record, gate, and prelaunch verification."""

    attestation_sha256: str
    f1_record_sha256: str
    f1_run_uid: str
    failed_gates: Tuple[str, ...]
    experiment_uid: str
    experiment_root: Path
    output_dir: Path
    launch_marker: Path


@dataclass(frozen=True)
class AuthorizedArtifact:
    """One absolute, byte-authenticated input named by the offline authority."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class VerifiedF2LaunchAuthorization:
    """Authenticated v2 launch contract returned to the dedicated launcher."""

    authorization_sha256: str
    signature_sha256: str
    source_revision: str
    repository_root: Path
    navsim_exp_root: Path
    openscene_data_root: Path
    nuplan_maps_root: Path
    navsim_log_root: Path
    sensor_blobs_root: Path
    qualification_source_root: Path
    artifacts: Mapping[str, AuthorizedArtifact]
    f1_record_sha256: str
    f1_run_uid: str
    failed_gates: Tuple[str, ...]
    experiment_name: str
    experiment_uid: str
    experiment_root: Path
    output_dir: Path
    launch_marker: Path

    def contract_lines(self) -> Tuple[str, ...]:
        """Return a fixed-key, tab-delimited projection safe for Bash parsing."""

        values: Dict[str, str] = {
            "AUTHORIZATION_SHA256": self.authorization_sha256,
            "SIGNATURE_SHA256": self.signature_sha256,
            "SOURCE_REVISION": self.source_revision,
            "REPOSITORY_ROOT": str(self.repository_root),
            "NAVSIM_EXP_ROOT": str(self.navsim_exp_root),
            "OPENSCENE_DATA_ROOT": str(self.openscene_data_root),
            "NUPLAN_MAPS_ROOT": str(self.nuplan_maps_root),
            "NAVSIM_LOG_ROOT": str(self.navsim_log_root),
            "SENSOR_BLOBS_ROOT": str(self.sensor_blobs_root),
            "QUALIFICATION_SOURCE_ROOT": str(self.qualification_source_root),
            "EXPERIMENT_NAME": self.experiment_name,
            "EXPERIMENT_UID": self.experiment_uid,
            "EXPERIMENT_ROOT": str(self.experiment_root),
            "OUTPUT_DIR": str(self.output_dir),
            "LAUNCH_MARKER": str(self.launch_marker),
        }
        for name in AUTHORIZED_ARTIFACT_KEYS:
            artifact = self.artifacts[name]
            values[f"{name.upper()}_PATH"] = str(artifact.path)
            values[f"{name.upper()}_SHA256"] = artifact.sha256
        if set(values) != set(VERIFIED_CONTRACT_KEYS):
            raise F2LaunchGuardError("internal verified-contract field drift")
        lines = tuple(f"{key}\t{values[key]}" for key in VERIFIED_CONTRACT_KEYS)
        for line in lines:
            forbidden = (
                ord(character) < 32 or ord(character) == 127
                for character in line
                if character != "\t"
            )
            if any(forbidden):
                raise F2LaunchGuardError("verified contract contains a control character")
        return lines


@dataclass(frozen=True)
class VerifiedF2SignedContract:
    """Lightweight SSHSIG-authenticated projection used by every F2 rank.

    Unlike :func:`verify_f2_launch_authorization`, this verifier never hashes
    multi-gigabyte checkpoints or replays the F1/feasibility evidence.  It
    authenticates the fixed authority bytes and their exact shape, checks the
    live source revision, and exposes only signed paths/digests.  The existing
    model/cache/raw-input loaders remain responsible for content verification.
    """

    authorization_sha256: str
    signature_sha256: str
    source_revision: str
    repository_root: Path
    runtime: Mapping[str, Path]
    artifacts: Mapping[str, AuthorizedArtifact]
    experiment_name: str
    experiment_uid: str
    experiment_root: Path
    output_dir: Path
    launch_marker: Path


@dataclass(frozen=True)
class VerifiedF2RuntimeClaim:
    """Canonical one-shot claim bound to a live torchrun parent identity."""

    path: Path
    sha256: str
    authorization_sha256: str
    signature_sha256: str
    source_revision: str
    repository_root: Path
    experiment_name: str
    experiment_uid: str
    experiment_root: Path
    output_dir: Path
    launch_marker: Path
    boot_id: str
    launcher_pid: int
    launcher_start_time_ticks: int


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise F2LaunchGuardError(f"{context} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise F2LaunchGuardError(f"{context} keys must all be strings")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise F2LaunchGuardError(f"{context} must be a list")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], context: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        raise F2LaunchGuardError(
            f"{context} fields drifted: missing={sorted(expected_set - actual_set)}, "
            f"unexpected={sorted(actual_set - expected_set)}"
        )


def _allowed_keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    allowed: Iterable[str],
    context: str,
) -> None:
    required_set = set(required)
    allowed_set = set(allowed)
    actual_set = set(value)
    missing = required_set - actual_set
    unexpected = actual_set - allowed_set
    if missing or unexpected:
        raise F2LaunchGuardError(
            f"{context} fields drifted: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def _strict_bool(value: Any, expected: bool, context: str) -> None:
    if type(value) is not bool or value is not expected:
        raise F2LaunchGuardError(f"{context} must be exactly {expected!r}")


def _strict_int(value: Any, expected: int, context: str) -> None:
    if type(value) is not int or value != expected:
        raise F2LaunchGuardError(f"{context} must be exactly {expected}")


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise F2LaunchGuardError(f"{context} must be a non-empty trimmed string")
    return value


def _plain_string(value: Any, context: str) -> str:
    """Accept a trimmed string that is safe as one contract-line value."""

    value = _nonempty_string(value, context)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise F2LaunchGuardError(f"{context} contains a control character")
    return value


def _git_revision(value: Any, context: str) -> str:
    if not isinstance(value, str) or GIT_REVISION_RE.fullmatch(value) is None:
        raise F2LaunchGuardError(
            f"{context} must be exactly 40 lowercase hexadecimal characters"
        )
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise F2LaunchGuardError(
            f"{context} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F2LaunchGuardError(f"{context} must be a finite JSON/YAML number")
    number = float(value)
    if not math.isfinite(number):
        raise F2LaunchGuardError(f"{context} must be finite")
    return number


def _positive_int(value: Any, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise F2LaunchGuardError(f"{context} must be a positive integer")
    return value


def _file_identity(value: os.stat_result) -> Tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_stable_regular_file(path: Path, context: str):
    """Open a non-symlink regular file and return its initial identity."""

    try:
        path_before = path.lstat()
    except OSError as error:
        raise F2LaunchGuardError(f"cannot stat {context}: {path}") from error
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise F2LaunchGuardError(
            f"{context} must be a regular non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        handle = os.fdopen(descriptor, "rb", closefd=True)
    except OSError as error:
        raise F2LaunchGuardError(f"cannot open {context}: {path}") from error
    opened = os.fstat(handle.fileno())
    if _file_identity(opened) != _file_identity(path_before):
        handle.close()
        raise F2LaunchGuardError(f"{context} changed before open")
    return handle, opened


def _finish_stable_file(
    path: Path, handle: Any, opened: os.stat_result, context: str
) -> None:
    after = os.fstat(handle.fileno())
    try:
        path_after = path.lstat()
    except OSError as error:
        raise F2LaunchGuardError(
            f"cannot restat {context}: {path}"
        ) from error
    identity = _file_identity(opened)
    if identity != _file_identity(after) or identity != _file_identity(path_after):
        raise F2LaunchGuardError(f"{context} changed while being read")


def _read_stable_bytes(path: Path, context: str) -> bytes:
    handle, opened = _open_stable_regular_file(path, context)
    try:
        payload = handle.read()
        _finish_stable_file(path, handle, opened, context)
        return payload
    except F2LaunchGuardError:
        raise
    except OSError as error:
        raise F2LaunchGuardError(f"cannot read {context}: {path}") from error
    finally:
        handle.close()


def _verify_stable_file_digest(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    context: str,
    required_substring: Optional[bytes] = None,
) -> None:
    expected_sha256 = _sha256(expected_sha256, f"{context} SHA-256")
    expected_bytes = _positive_int(expected_bytes, f"{context} bytes")
    handle, opened = _open_stable_regular_file(path, context)
    digest = hashlib.sha256()
    total = 0
    found = required_substring is None
    tail = b""
    try:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
            if not found and required_substring is not None:
                searchable = tail + chunk
                found = required_substring in searchable
                keep = max(0, len(required_substring) - 1)
                tail = searchable[-keep:] if keep else b""
        _finish_stable_file(path, handle, opened, context)
    except F2LaunchGuardError:
        raise
    except OSError as error:
        raise F2LaunchGuardError(f"cannot hash {context}: {path}") from error
    finally:
        handle.close()
    if total != expected_bytes:
        raise F2LaunchGuardError(
            f"{context} byte size mismatch: expected {expected_bytes}, got {total}"
        )
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise F2LaunchGuardError(
            f"{context} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    if not found:
        raise F2LaunchGuardError(
            f"{context} lacks the recorded terminal marker"
        )


def _decimal(value: Any, context: str) -> Decimal:
    _finite_number(value, context)
    return Decimal(str(value))


def _require_fixed_value(value: Any, expected: Decimal, context: str) -> None:
    if _decimal(value, context) != expected:
        raise F2LaunchGuardError(
            f"{context} drifted from frozen value {expected}"
        )


def canonical_attestation_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize the attestation in the only accepted canonical representation."""

    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise F2LaunchGuardError(f"attestation contains forbidden constant {value}")


def _strict_json_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise F2LaunchGuardError(
                f"attestation contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _decode_strict_json(raw: bytes, context: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise F2LaunchGuardError(f"{context} must be strict UTF-8") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise F2LaunchGuardError(f"{context} is not valid strict JSON") from error
    return _mapping(payload, context)


def _parse_canonical_json_bytes(raw: bytes, context: str) -> Mapping[str, Any]:
    payload = _decode_strict_json(raw, context)
    try:
        canonical = canonical_attestation_bytes(payload)
    except (TypeError, ValueError) as error:
        raise F2LaunchGuardError(
            f"{context} cannot be represented as canonical JSON"
        ) from error
    if raw != canonical:
        raise F2LaunchGuardError(
            f"{context} bytes are not canonical sorted compact UTF-8 JSON "
            "with one trailing LF"
        )
    return payload


def _load_canonical_json(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    expected_sha256 = _sha256(
        expected_sha256, "expected attestation SHA-256"
    )
    raw = _read_stable_bytes(path, "trigger attestation")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise F2LaunchGuardError(
            "trigger attestation SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return _parse_canonical_json_bytes(raw, "attestation")


def read_expected_attestation_sha256(path: Path) -> str:
    """Read the exact one-line detached digest used by the dedicated launcher."""

    raw = _read_stable_bytes(path, "detached attestation digest")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise F2LaunchGuardError(
            "detached attestation digest must be ASCII"
        ) from error
    if re.fullmatch(r"[0-9a-f]{64}\n", text) is None:
        raise F2LaunchGuardError(
            "detached attestation digest must be one lowercase SHA-256 plus LF"
        )
    return text[:-1]


def _normalized_ssh_public_key(value: Any, context: str) -> str:
    value = _plain_string(value, context)
    parts = value.split(" ")
    if len(parts) != 2 or parts[0] not in {"ssh-ed25519", "ssh-rsa"}:
        raise F2LaunchGuardError(
            f"{context} must be one comment-free ssh-ed25519 or ssh-rsa public key"
        )
    try:
        decoded = base64.b64decode(parts[1], validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise F2LaunchGuardError(f"{context} contains malformed base64") from error
    if not decoded:
        raise F2LaunchGuardError(f"{context} is empty")
    return value


def _ssh_keygen_executable() -> str:
    # The production host has the system binary at /usr/bin.  The fallback is
    # solely for the Windows/unit-test and explicit offline-issuance paths.
    system = Path("/usr/bin/ssh-keygen")
    if system.is_file():
        return str(system)
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise F2LaunchGuardError("ssh-keygen with SSHSIG support is required")
    return executable


def verify_detached_ssh_signature(
    message: bytes,
    signature: bytes,
    *,
    trusted_public_key: str = TRUSTED_AUTHORITY_SSH_PUBLIC_KEY,
) -> None:
    """Verify an SSHSIG detached signature against the embedded authority."""

    public_key = _normalized_ssh_public_key(
        trusted_public_key, "trusted F2 authority public key"
    )
    if not signature.startswith(b"-----BEGIN SSH SIGNATURE-----\n") or not signature.endswith(
        b"-----END SSH SIGNATURE-----\n"
    ):
        raise F2LaunchGuardError(
            "detached F2 authorization signature must be canonical armored SSHSIG"
        )
    with tempfile.TemporaryDirectory(prefix="actworld-f2-verify-") as temporary:
        root = Path(temporary)
        allowed_signers = root / "allowed_signers"
        signature_path = root / "authorization.sig"
        allowed_signers.write_bytes(
            f"{AUTHORIZATION_SSH_PRINCIPAL} {public_key}\n".encode("ascii")
        )
        signature_path.write_bytes(signature)
        completed = subprocess.run(
            [
                _ssh_keygen_executable(),
                "-Y",
                "verify",
                "-f",
                str(allowed_signers),
                "-I",
                AUTHORIZATION_SSH_PRINCIPAL,
                "-n",
                AUTHORIZATION_SSH_NAMESPACE,
                "-s",
                str(signature_path),
            ],
            input=message,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise F2LaunchGuardError(
            "detached F2 authorization signature is invalid"
            + (f": {detail}" if detail else "")
        )


def _strict_nonsymlink_path(path: Path, context: str, *, directory: bool) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise F2LaunchGuardError(f"{context} must be absolute")
    try:
        lexical = Path(os.path.abspath(str(path)))
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as error:
        raise F2LaunchGuardError(f"cannot resolve {context}: {path}") from error
    if os.path.normcase(str(lexical)) != os.path.normcase(str(resolved)):
        raise F2LaunchGuardError(f"{context} must not traverse a symbolic link")
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode) or not expected:
        kind = "directory" if directory else "file"
        raise F2LaunchGuardError(f"{context} must be a non-symlink {kind}")
    return resolved


def _hash_authorized_file(path: Path, expected_sha256: str, context: str) -> Path:
    resolved = _strict_nonsymlink_path(path, context, directory=False)
    expected = _sha256(expected_sha256, f"{context} SHA-256")
    handle, opened = _open_stable_regular_file(resolved, context)
    digest = hashlib.sha256()
    try:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
        _finish_stable_file(resolved, handle, opened, context)
    except F2LaunchGuardError:
        raise
    except OSError as error:
        raise F2LaunchGuardError(f"cannot hash {context}: {resolved}") from error
    finally:
        handle.close()
    actual = digest.hexdigest()
    if actual != expected:
        raise F2LaunchGuardError(
            f"{context} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return resolved


def _write_bytes_no_overwrite(path: Path, payload: bytes, context: str) -> None:
    """Create one regular file with O_EXCL; never truncate or replace."""

    path = Path(path)
    if not path.is_absolute():
        raise F2LaunchGuardError(f"{context} output path must be absolute")
    parent = _strict_nonsymlink_path(path.parent, f"{context} parent", directory=True)
    destination = parent / path.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileExistsError as error:
        raise F2LaunchGuardError(
            f"{context} already exists; overwrite is forbidden: {destination}"
        ) from error
    except OSError as error:
        raise F2LaunchGuardError(f"cannot create {context}: {destination}") from error


def _sign_with_ssh_private_key(
    message: bytes,
    *,
    private_key_path: Path,
    trusted_public_key: str,
) -> bytes:
    private_key = _strict_nonsymlink_path(
        private_key_path, "offline signing private key", directory=False
    )
    if os.name == "posix" and stat.S_IMODE(private_key.stat().st_mode) & 0o077:
        raise F2LaunchGuardError(
            "offline signing private key permissions must deny group/other access"
        )
    public = subprocess.run(
        [_ssh_keygen_executable(), "-y", "-f", str(private_key)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if public.returncode != 0:
        raise F2LaunchGuardError("could not derive the offline signing public key")
    try:
        derived_parts = public.stdout.decode("ascii").split()
    except UnicodeDecodeError as error:
        raise F2LaunchGuardError("offline signing public key is not ASCII") from error
    expected = _normalized_ssh_public_key(
        trusted_public_key, "trusted F2 authority public key"
    )
    derived = " ".join(derived_parts[:2])
    if _normalized_ssh_public_key(derived, "derived signing public key") != expected:
        raise F2LaunchGuardError(
            "offline private key does not match the embedded F2 authority"
        )
    with tempfile.TemporaryDirectory(prefix="actworld-f2-sign-") as temporary:
        authorization = Path(temporary) / "authorization.json"
        authorization.write_bytes(message)
        completed = subprocess.run(
            [
                _ssh_keygen_executable(),
                "-Y",
                "sign",
                "-f",
                str(private_key),
                "-n",
                AUTHORIZATION_SSH_NAMESPACE,
                str(authorization),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        signature_path = Path(str(authorization) + ".sig")
        if completed.returncode != 0 or not signature_path.is_file():
            raise F2LaunchGuardError("offline SSHSIG signing failed")
        signature = signature_path.read_bytes()
    verify_detached_ssh_signature(
        message, signature, trusted_public_key=trusted_public_key
    )
    return signature


def _reject_yaml_aliases_and_tags(text: str) -> None:
    try:
        tokens = yaml.scan(text)
        for token in tokens:
            if isinstance(
                token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken, yaml.tokens.TagToken)
            ):
                raise F2LaunchGuardError(
                    "F1 YAML anchors, aliases, and explicit tags are forbidden"
                )
    except yaml.YAMLError as error:
        raise F2LaunchGuardError("F1 record is not valid YAML") from error


def _reject_duplicate_yaml_keys(node: yaml.Node, seen: MutableSet[int]) -> None:
    identity = id(node)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(node, yaml.MappingNode):
        keys = set()
        for key_node, value_node in node.value:
            if not isinstance(key_node, yaml.ScalarNode):
                raise F2LaunchGuardError(
                    "F1 YAML mapping keys must be scalar strings"
                )
            key = key_node.value
            if key == "<<":
                raise F2LaunchGuardError("F1 YAML merge keys are forbidden")
            if key in keys:
                raise F2LaunchGuardError(
                    f"F1 YAML contains duplicate key {key!r}"
                )
            keys.add(key)
            _reject_duplicate_yaml_keys(value_node, seen)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            _reject_duplicate_yaml_keys(item, seen)


def _load_strict_f1_yaml(raw: bytes) -> Mapping[str, Any]:
    if yaml is None:
        raise F2LaunchGuardError(
            "PyYAML is required for launch verification but not offline issuance"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise F2LaunchGuardError("F1 YAML must be strict UTF-8") from error
    _reject_yaml_aliases_and_tags(text)
    try:
        syntax_tree = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as error:
        raise F2LaunchGuardError("F1 record is not valid safe YAML") from error
    if syntax_tree is None:
        raise F2LaunchGuardError("F1 YAML is empty")
    _reject_duplicate_yaml_keys(syntax_tree, set())
    try:
        # Deliberately use safe_load: no Python/object constructors are allowed.
        record = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise F2LaunchGuardError("F1 record is not valid safe YAML") from error
    _validate_safe_yaml_tree(record, "F1")
    return _mapping(record, "F1 record")


def _validate_safe_yaml_tree(value: Any, context: str) -> None:
    """Reject implicit exotic YAML types and non-finite values everywhere."""

    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise F2LaunchGuardError(f"{context} contains a non-string key")
            _validate_safe_yaml_tree(child, f"{context}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_safe_yaml_tree(child, f"{context}[{index}]")
        return
    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is float and math.isfinite(value):
        return
    raise F2LaunchGuardError(
        f"{context} contains a forbidden or non-finite YAML scalar"
    )


def _resolve_repository_record(repository_root: Path, relative_path: str) -> Path:
    if relative_path != F1_RECORD_REPOSITORY_PATH:
        raise F2LaunchGuardError(
            "attestation references a non-canonical F1 record path"
        )
    root = repository_root.resolve(strict=True)
    candidate = (root / Path(relative_path)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise F2LaunchGuardError("F1 record escapes repository root") from error
    if not candidate.is_file():
        raise F2LaunchGuardError("F1 record path is not a regular file")
    return candidate


def _validate_attestation_shape(payload: Mapping[str, Any]) -> None:
    _exact_keys(
        payload,
        {"schema_version", "f1", "decision", "f2_prelaunch"},
        "attestation",
    )
    if payload["schema_version"] != ATTESTATION_SCHEMA_VERSION:
        raise F2LaunchGuardError("attestation schema_version is unsupported")

    f1 = _mapping(payload["f1"], "attestation.f1")
    _exact_keys(
        f1,
        {
            "record_path",
            "record_sha256",
            "run_uid",
            "scheduled_epoch_count",
            "completed_epoch_count",
            "final_scheduled_epoch_index",
            "legacy_label_interpretation",
        },
        "attestation.f1",
    )
    if f1["record_path"] != F1_RECORD_REPOSITORY_PATH:
        raise F2LaunchGuardError("attestation.f1.record_path drifted")
    _sha256(f1["record_sha256"], "attestation.f1.record_sha256")
    run_uid = _nonempty_string(f1["run_uid"], "attestation.f1.run_uid")
    if RUN_UID_RE.fullmatch(run_uid) is None:
        raise F2LaunchGuardError("attestation.f1.run_uid is malformed")
    _strict_int(
        f1["scheduled_epoch_count"],
        SCHEDULED_EPOCH_COUNT,
        "attestation.f1.scheduled_epoch_count",
    )
    _strict_int(
        f1["completed_epoch_count"],
        SCHEDULED_EPOCH_COUNT,
        "attestation.f1.completed_epoch_count",
    )
    _strict_int(
        f1["final_scheduled_epoch_index"],
        FINAL_SCHEDULED_EPOCH_INDEX,
        "attestation.f1.final_scheduled_epoch_index",
    )
    if f1["legacy_label_interpretation"] != LEGACY_LABEL_INTERPRETATION:
        raise F2LaunchGuardError(
            "attestation legacy F1_epoch_4 interpretation drifted"
        )

    decision = _mapping(payload["decision"], "attestation.decision")
    _exact_keys(
        decision,
        {
            "fixed_gate_conjunction_passed",
            "failed_gates",
            "authorized_action",
            "test_splits_consulted",
        },
        "attestation.decision",
    )
    _strict_bool(
        decision["fixed_gate_conjunction_passed"],
        False,
        "attestation.decision.fixed_gate_conjunction_passed",
    )
    failed = _sequence(
        decision["failed_gates"], "attestation.decision.failed_gates"
    )
    if not failed or any(type(name) is not str for name in failed):
        raise F2LaunchGuardError(
            "attestation.decision.failed_gates must be a non-empty string list"
        )
    if len(set(failed)) != len(failed) or any(name not in GATE_ORDER for name in failed):
        raise F2LaunchGuardError(
            "attestation.decision.failed_gates contains duplicate/unknown gates"
        )
    if decision["authorized_action"] != "execute_F2_only":
        raise F2LaunchGuardError(
            "attestation does not authorize the single F2-only action"
        )
    _strict_bool(
        decision["test_splits_consulted"],
        False,
        "attestation.decision.test_splits_consulted",
    )

    prelaunch = _mapping(payload["f2_prelaunch"], "attestation.f2_prelaunch")
    _exact_keys(
        prelaunch,
        {
            "started",
            "experiment_uid",
            "experiment_root",
            "output_dir",
            "launch_marker",
        },
        "attestation.f2_prelaunch",
    )
    _strict_bool(
        prelaunch["started"], False, "attestation.f2_prelaunch.started"
    )
    uid = _nonempty_string(
        prelaunch["experiment_uid"], "attestation.f2_prelaunch.experiment_uid"
    )
    if RUN_UID_RE.fullmatch(uid) is None:
        raise F2LaunchGuardError(
            "attestation.f2_prelaunch.experiment_uid is malformed"
        )
    for field in ("experiment_root", "output_dir", "launch_marker"):
        value = _nonempty_string(
            prelaunch[field], f"attestation.f2_prelaunch.{field}"
        )
        if not Path(value).is_absolute():
            raise F2LaunchGuardError(
                f"attestation.f2_prelaunch.{field} must be absolute"
            )


def _validate_frozen_f1_contract(record: Mapping[str, Any], run_uid: str) -> None:
    _exact_keys(record, _F1_TOP_LEVEL_KEYS, "F1 record")
    _strict_int(record["schema_version"], 1, "F1 schema_version")

    run = _mapping(record["run"], "F1.run")
    _allowed_keys(
        run,
        required={
            "project",
            "stage",
            "fallback",
            "status",
            "completed_at",
            "uid",
            "code_revision",
            "output_dir",
            "log",
            "log_bytes",
            "log_sha256",
            "terminal_log_marker",
            "wandb_offline_record",
            "wandb_offline_record_bytes",
            "wandb_offline_record_sha256",
            "resolved_config_path",
            "resolved_config_bytes",
            "resolved_config_sha256",
            "hydra_overrides_path",
            "hydra_overrides_bytes",
            "hydra_overrides_sha256",
        },
        allowed=_RUN_ALLOWED_KEYS,
        context="F1.run",
    )
    if run["project"] != "ActWorld-JEPA" or run["stage"] != "critic":
        raise F2LaunchGuardError("F1 run project/stage identity drifted")
    if run["fallback"] != "F1":
        raise F2LaunchGuardError("F1 run is not the F1 fallback")
    if run["uid"] != run_uid:
        raise F2LaunchGuardError("attested run UID differs from the F1 record")
    status = _nonempty_string(run["status"], "F1.run.status").lower()
    if status in {"running", "pending", "tbd", "in_progress"} or not any(
        marker in status for marker in ("complete", "failed", "stopped_after")
    ):
        raise F2LaunchGuardError("F1 run status does not prove terminal completion")
    _nonempty_string(run["completed_at"], "F1.run.completed_at")
    _validate_git_revision(run["code_revision"], "F1.run.code_revision")
    for field in (
        "output_dir",
        "log",
        "wandb_offline_record",
        "resolved_config_path",
        "hydra_overrides_path",
    ):
        value = _nonempty_string(run[field], f"F1.run.{field}")
        if not Path(value).is_absolute():
            raise F2LaunchGuardError(f"F1.run.{field} must be absolute")
    for field in (
        "log_bytes",
        "wandb_offline_record_bytes",
        "resolved_config_bytes",
        "hydra_overrides_bytes",
    ):
        _positive_int(run[field], f"F1.run.{field}")
    _sha256(run["log_sha256"], "F1.run.log_sha256")
    _sha256(
        run["wandb_offline_record_sha256"],
        "F1.run.wandb_offline_record_sha256",
    )
    marker = _nonempty_string(
        run["terminal_log_marker"], "F1.run.terminal_log_marker"
    )
    if not all(fragment in marker for fragment in ("Trainer.fit", "max_epochs=4", "reached")):
        raise F2LaunchGuardError(
            "F1 terminal log marker does not prove max_epochs=4 completion"
        )
    _sha256(
        run["resolved_config_sha256"], "F1.run.resolved_config_sha256"
    )
    _sha256(
        run["hydra_overrides_sha256"], "F1.run.hydra_overrides_sha256"
    )

    lineage = _mapping(record["lineage"], "F1.lineage")
    _exact_keys(
        lineage,
        {
            "initialization",
            "trigger",
            "primary_parent_uid",
            "resume_checkpoint",
            "optimizer_inherited",
            "agent_checkpoint_inherited",
            "drive_jepa_checkpoint",
            "drive_jepa_checkpoint_sha256",
        },
        "F1.lineage",
    )
    if lineage["initialization"] != "fresh_stage_a":
        raise F2LaunchGuardError("F1 was not a fresh Stage-A initialization")
    if lineage["trigger"] != "primary_epoch_4_failed_any_gate":
        raise F2LaunchGuardError("F1 trigger identity drifted")
    if lineage["resume_checkpoint"] is not None:
        raise F2LaunchGuardError("F1 unexpectedly resumed")
    _strict_bool(
        lineage["optimizer_inherited"], False, "F1.lineage.optimizer_inherited"
    )
    _strict_bool(
        lineage["agent_checkpoint_inherited"],
        False,
        "F1.lineage.agent_checkpoint_inherited",
    )
    _sha256(
        lineage["drive_jepa_checkpoint_sha256"],
        "F1.lineage.drive_jepa_checkpoint_sha256",
    )

    data = _mapping(record["data"], "F1.data")
    _exact_keys(
        data,
        {
            "split",
            "test_splits_consulted",
            "metric_scale",
            "one_reported_point_raw",
            "train_samples",
            "validation_samples",
            "streaming",
            "future_latent_grounding",
            "metric_cache",
            "metric_cache_metadata_sha256",
        },
        "F1.data",
    )
    if data["split"] != "navtrain":
        raise F2LaunchGuardError("F1 record is not navtrain-only")
    _strict_bool(data["test_splits_consulted"], False, "F1.data.test_splits_consulted")
    _strict_bool(data["future_latent_grounding"], True, "F1.data.future_latent_grounding")
    _sha256(
        data["metric_cache_metadata_sha256"],
        "F1.data.metric_cache_metadata_sha256",
    )

    optimization = _mapping(record["optimization"], "F1.optimization")
    _exact_keys(
        optimization,
        {
            "seed",
            "accelerator",
            "strategy",
            "precision",
            "per_rank_batch_size",
            "global_batch_size",
            "dataloader_workers_per_rank",
            "score_workers_per_rank",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "max_epochs",
            "train_steps_per_epoch",
            "validation_batches_per_epoch",
        },
        "F1.optimization",
    )
    _strict_int(optimization["seed"], 0, "F1.optimization.seed")
    if optimization["strategy"] != "ddp":
        raise F2LaunchGuardError("F1 optimization.strategy drifted")
    _strict_int(
        optimization["per_rank_batch_size"],
        8,
        "F1.optimization.per_rank_batch_size",
    )
    _strict_int(
        optimization["global_batch_size"],
        64,
        "F1.optimization.global_batch_size",
    )
    _require_fixed_value(
        optimization["learning_rate"], Decimal("0.00005"), "F1.optimization.learning_rate"
    )
    _strict_int(
        optimization["max_epochs"],
        SCHEDULED_EPOCH_COUNT,
        "F1.optimization.max_epochs",
    )
    for field in ("train_steps_per_epoch", "validation_batches_per_epoch"):
        value = optimization[field]
        if type(value) is not int or value <= 0:
            raise F2LaunchGuardError(f"F1.optimization.{field} must be positive")

    frozen = _mapping(record["frozen_hypothesis"], "F1.frozen_hypothesis")
    _exact_keys(
        frozen,
        {
            "purpose",
            "code_change_from_primary",
            "actworld_rank_loss_weight",
            "actworld_rank_temperature",
            "unchanged",
            "hyperparameter_search",
        },
        "F1.frozen_hypothesis",
    )
    _strict_bool(
        frozen["code_change_from_primary"],
        False,
        "F1.frozen_hypothesis.code_change_from_primary",
    )
    if frozen["hyperparameter_search"] != "none":
        raise F2LaunchGuardError("F1 performed a hyperparameter search")

    selection = _mapping(record["selection"], "F1.selection")
    _exact_keys(
        selection,
        {"development_split_only", "checkpoint_monitor", "mode", "effective_gates", "decision"},
        "F1.selection",
    )
    if selection["development_split_only"] != "navtrain":
        raise F2LaunchGuardError("F1 selection was not navtrain-only")
    gates = _mapping(selection["effective_gates"], "F1.selection.effective_gates")
    _exact_keys(gates, FIXED_GATE_VALUES, "F1.selection.effective_gates")
    for field, expected in FIXED_GATE_VALUES.items():
        _require_fixed_value(gates[field], expected, f"F1.selection.effective_gates.{field}")

    decision = _mapping(selection["decision"], "F1.selection.decision")
    _exact_keys(
        decision,
        {
            "scheduled_epoch_count",
            "final_scheduled_epoch_index",
            "require_final_scheduled_epoch_gate_conjunction",
            "legacy_ambiguous_label",
            "legacy_label_interpretation",
            "actor_allowed_before_gate_pass",
            "on_final_scheduled_epoch_failure",
            "test_evaluation_before_selection",
        },
        "F1.selection.decision",
    )
    _strict_int(decision["scheduled_epoch_count"], 4, "F1.selection.decision.scheduled_epoch_count")
    _strict_int(
        decision["final_scheduled_epoch_index"],
        3,
        "F1.selection.decision.final_scheduled_epoch_index",
    )
    _strict_bool(
        decision["require_final_scheduled_epoch_gate_conjunction"],
        True,
        "F1.selection.decision.require_final_scheduled_epoch_gate_conjunction",
    )
    if decision["legacy_ambiguous_label"] != "require_epoch_4_gate_conjunction":
        raise F2LaunchGuardError("F1 legacy decision label drifted")
    if decision["legacy_label_interpretation"] != (
        "fourth completed epoch, zero-based index 3; no index-4 epoch is authorized"
    ):
        raise F2LaunchGuardError("F1 legacy decision interpretation drifted")
    _strict_bool(
        decision["actor_allowed_before_gate_pass"],
        False,
        "F1.selection.decision.actor_allowed_before_gate_pass",
    )
    if decision["on_final_scheduled_epoch_failure"] != "execute_F2_only":
        raise F2LaunchGuardError("F1 decision does not authorize F2-only fallback")
    _strict_bool(
        decision["test_evaluation_before_selection"],
        False,
        "F1.selection.decision.test_evaluation_before_selection",
    )

    clarification = _mapping(
        record["epoch_indexing_clarification"], "F1.epoch_indexing_clarification"
    )
    _exact_keys(
        clarification,
        {
            "recorded_at",
            "evidence_cutoff",
            "executable_authority",
            "lightning_epoch_indices",
            "final_gate_epoch_index",
            "schedule_changed",
            "fifth_epoch_authorized",
            "note",
        },
        "F1.epoch_indexing_clarification",
    )
    if clarification["evidence_cutoff"] != (
        "F1 epoch 0 complete; epoch 1 training in progress; epoch 1 validation unavailable"
    ):
        raise F2LaunchGuardError(
            "F1 indexing clarification was not frozen before epoch-1 validation"
        )
    if clarification["executable_authority"] != (
        "trainer.params.max_epochs=4 in the frozen resolved config"
    ):
        raise F2LaunchGuardError("F1 executable epoch authority drifted")
    if clarification["lightning_epoch_indices"] != [0, 1, 2, 3]:
        raise F2LaunchGuardError("F1 Lightning epoch indices drifted")
    _strict_int(
        clarification["final_gate_epoch_index"],
        3,
        "F1.epoch_indexing_clarification.final_gate_epoch_index",
    )
    _strict_bool(
        clarification["schedule_changed"],
        False,
        "F1.epoch_indexing_clarification.schedule_changed",
    )
    _strict_bool(
        clarification["fifth_epoch_authorized"],
        False,
        "F1.epoch_indexing_clarification.fifth_epoch_authorized",
    )


def _validate_git_revision(value: Any, context: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{7,40}", value) is None:
        raise F2LaunchGuardError(f"{context} must be a lowercase Git revision")


def _derive_gate_passes(
    raw: Mapping[str, Any], fatal_error_matches: Any
) -> Mapping[str, bool]:
    required = {
        "baseline_score",
        "best_of_32_score",
        "critic_selected_score",
        "critic_delta",
        "critic_pairwise_accuracy",
        "critic_score_error_excess",
        "proposal_headroom_closed_fraction",
        "critic_nc_delta",
        "critic_dac_delta",
        "critic_tlc_delta",
        "critic_ddc_delta",
        "future_camera_valid_min",
    }
    _allowed_keys(
        raw,
        required=required,
        allowed=_RAW_METRIC_ALLOWED_KEYS,
        context="F1 epoch raw_metrics",
    )
    values = {
        key: _finite_number(value, f"F1 epoch raw_metrics.{key}")
        for key, value in raw.items()
    }
    if type(fatal_error_matches) is not int or fatal_error_matches < 0:
        raise F2LaunchGuardError("F1 epoch fatal_error_matches must be a nonnegative int")

    if not math.isclose(
        values["critic_delta"],
        values["critic_selected_score"] - values["baseline_score"],
        rel_tol=0.0,
        abs_tol=2e-7,
    ):
        raise F2LaunchGuardError("F1 critic_delta is inconsistent with raw scores")
    if "critic_score_error" in values and "baseline_score_error" in values:
        if not math.isclose(
            values["critic_score_error_excess"],
            values["critic_score_error"] - values["baseline_score_error"],
            rel_tol=0.0,
            abs_tol=2e-7,
        ):
            raise F2LaunchGuardError(
                "F1 critic_score_error_excess is inconsistent with raw errors"
            )

    gates = {key: float(value) for key, value in FIXED_GATE_VALUES.items()}
    baseline_drift = abs(values["baseline_score"] - gates["baseline_score_reference"])
    oracle_drift = abs(
        values["best_of_32_score"] - gates["best_of_32_score_reference"]
    )
    return {
        "critic_selected_score": values["critic_selected_score"]
        >= gates["critic_selected_score_min"],
        "critic_delta": values["critic_delta"] >= gates["critic_delta_min"],
        "critic_pairwise_accuracy": values["critic_pairwise_accuracy"]
        >= gates["critic_pairwise_accuracy_min"],
        "proposal_headroom_closed": values["proposal_headroom_closed_fraction"]
        >= gates["proposal_headroom_closed_min"],
        "critic_score_error_excess": values["critic_score_error_excess"]
        <= gates["critic_score_error_excess_max"],
        "nc_delta": values["critic_nc_delta"] >= gates["hard_factor_delta_min"],
        "dac_delta": values["critic_dac_delta"] >= gates["hard_factor_delta_min"],
        "tlc_delta": values["critic_tlc_delta"] >= gates["hard_factor_delta_min"],
        "ddc_delta": values["critic_ddc_delta"] >= gates["hard_factor_delta_min"],
        "future_camera_valid": values["future_camera_valid_min"]
        >= gates["future_camera_valid_min"],
        "baseline_and_oracle_drift": max(baseline_drift, oracle_drift)
        <= gates["baseline_and_oracle_drift_max"],
        "fatal_or_nonfinite": fatal_error_matches
        <= gates["fatal_or_nonfinite_matches_max"],
    }


def _validate_complete_epochs(record: Mapping[str, Any]) -> Tuple[str, ...]:
    optimization = _mapping(record["optimization"], "F1.optimization")
    expected_steps = optimization["train_steps_per_epoch"]
    expected_batches = optimization["validation_batches_per_epoch"]
    records = _sequence(record["epoch_records"], "F1.epoch_records")
    if len(records) != SCHEDULED_EPOCH_COUNT:
        raise F2LaunchGuardError(
            "F1 must contain exactly four complete scheduled epoch records"
        )

    derived_by_epoch: Dict[int, Mapping[str, bool]] = {}
    for expected_epoch, item in enumerate(records):
        epoch = _mapping(item, f"F1.epoch_records[{expected_epoch}]")
        _allowed_keys(
            epoch,
            required={
                "epoch",
                "status",
                "train_steps",
                "validation_batches",
                "raw_metrics",
                "gate_pass",
                "fatal_error_matches",
            },
            allowed={
                "epoch",
                "status",
                "train_steps",
                "validation_batches",
                "wandb_epoch",
                "wandb_trainer_global_step",
                "raw_metrics",
                "gate_pass",
                "epoch_recovery_corridor",
                "fatal_error_matches",
            },
            context=f"F1.epoch_records[{expected_epoch}]",
        )
        _strict_int(epoch["epoch"], expected_epoch, f"F1 epoch index {expected_epoch}")
        status = _nonempty_string(epoch["status"], f"F1 epoch {expected_epoch} status").lower()
        if any(marker in status for marker in ("running", "partial", "incomplete", "pending")):
            raise F2LaunchGuardError(f"F1 epoch {expected_epoch} is incomplete")
        _strict_int(epoch["train_steps"], expected_steps, f"F1 epoch {expected_epoch} train_steps")
        _strict_int(
            epoch["validation_batches"],
            expected_batches,
            f"F1 epoch {expected_epoch} validation_batches",
        )
        if "wandb_epoch" in epoch:
            _strict_int(
                epoch["wandb_epoch"],
                expected_epoch,
                f"F1 epoch {expected_epoch} wandb_epoch",
            )
        if "wandb_trainer_global_step" in epoch:
            _strict_int(
                epoch["wandb_trainer_global_step"],
                (expected_epoch + 1) * expected_steps - 1,
                f"F1 epoch {expected_epoch} wandb_trainer_global_step",
            )
        raw = _mapping(epoch["raw_metrics"], f"F1 epoch {expected_epoch} raw_metrics")
        derived = _derive_gate_passes(raw, epoch["fatal_error_matches"])
        recorded = _mapping(epoch["gate_pass"], f"F1 epoch {expected_epoch} gate_pass")
        _exact_keys(recorded, GATE_ORDER, f"F1 epoch {expected_epoch} gate_pass")
        for gate_name in GATE_ORDER:
            _strict_bool(
                recorded[gate_name],
                derived[gate_name],
                f"F1 epoch {expected_epoch} gate_pass.{gate_name}",
            )
        derived_by_epoch[expected_epoch] = derived

    final = derived_by_epoch[FINAL_SCHEDULED_EPOCH_INDEX]
    failed = tuple(name for name in GATE_ORDER if not final[name])
    if not failed:
        raise F2LaunchGuardError(
            "F1 final scheduled epoch passed the fixed-gate conjunction; F2 is forbidden"
        )
    return failed


def _validate_final_checkpoint_and_result(record: Mapping[str, Any]) -> None:
    checkpoints = _mapping(record["checkpoints"], "F1.checkpoints")
    _exact_keys(checkpoints, {"best", "last", "selected"}, "F1.checkpoints")
    best = _mapping(checkpoints["best"], "F1.checkpoints.best")
    _allowed_keys(
        best,
        required={"epoch", "path", "bytes", "sha256"},
        allowed={"epoch", "global_step", "path", "bytes", "sha256"},
        context="F1.checkpoints.best",
    )
    best_epoch = best["epoch"]
    if (
        type(best_epoch) is not int
        or best_epoch < 0
        or best_epoch > FINAL_SCHEDULED_EPOCH_INDEX
    ):
        raise F2LaunchGuardError("F1.checkpoints.best.epoch is outside 0..3")
    _nonempty_string(best["path"], "F1.checkpoints.best.path")
    _positive_int(best["bytes"], "F1.checkpoints.best.bytes")
    _sha256(best["sha256"], "F1.checkpoints.best.sha256")
    last = _mapping(checkpoints["last"], "F1.checkpoints.last")
    _allowed_keys(
        last,
        required={"epoch", "global_step", "path", "bytes", "sha256", "training_stage"},
        allowed={
            "epoch",
            "global_step",
            "path",
            "bytes",
            "sha256",
            "training_stage",
            "state_dict_keys",
            "state_dict_numel",
        },
        context="F1.checkpoints.last",
    )
    _strict_int(last["epoch"], FINAL_SCHEDULED_EPOCH_INDEX, "F1.checkpoints.last.epoch")
    optimization = _mapping(record["optimization"], "F1.optimization")
    _strict_int(
        last["global_step"],
        SCHEDULED_EPOCH_COUNT * optimization["train_steps_per_epoch"],
        "F1.checkpoints.last.global_step",
    )
    _nonempty_string(last["path"], "F1.checkpoints.last.path")
    if type(last["bytes"]) is not int or last["bytes"] <= 0:
        raise F2LaunchGuardError("F1.checkpoints.last.bytes must be positive")
    _sha256(last["sha256"], "F1.checkpoints.last.sha256")
    if last["training_stage"] != "critic":
        raise F2LaunchGuardError("F1 final checkpoint is not a critic checkpoint")
    if checkpoints["selected"] is not None:
        raise F2LaunchGuardError("failed F1 must not have a selected checkpoint")

    results = _mapping(record["results"], "F1.results")
    _allowed_keys(
        results,
        required={"status", "note"},
        allowed={"status", "note"},
        context="F1.results",
    )
    status = _nonempty_string(results["status"], "F1.results.status").lower()
    if "fail" not in status or status in {"tbd", "pending"}:
        raise F2LaunchGuardError("F1 results do not record a terminal gate failure")
    _nonempty_string(results["note"], "F1.results.note")


def _resolved_recorded_file(value: Any, context: str) -> Path:
    raw = _nonempty_string(value, context)
    path = Path(raw)
    if not path.is_absolute():
        raise F2LaunchGuardError(f"{context} must be absolute")
    normalized = Path(os.path.abspath(str(path)))
    try:
        normalized.lstat()
    except OSError as error:
        raise F2LaunchGuardError(f"{context} does not exist: {path}") from error
    return normalized


def _require_within(path: Path, parent: Path, context: str) -> None:
    try:
        path.resolve(strict=True).relative_to(parent.resolve(strict=True))
    except ValueError as error:
        raise F2LaunchGuardError(
            f"{context} is outside the recorded F1 output directory"
        ) from error


def _validate_recorded_f1_files(record: Mapping[str, Any]) -> None:
    """Re-hash terminal F1 evidence and checkpoints immediately pre-launch."""

    run = _mapping(record["run"], "F1.run")
    output_dir = Path(os.path.abspath(str(Path(run["output_dir"]))))
    try:
        output_stat = output_dir.lstat()
    except OSError as error:
        raise F2LaunchGuardError("F1 output_dir does not exist") from error
    if stat.S_ISLNK(output_stat.st_mode) or not stat.S_ISDIR(output_stat.st_mode):
        raise F2LaunchGuardError(
            "F1 output_dir must be a non-symlink directory"
        )

    log_path = _resolved_recorded_file(run["log"], "F1.run.log")
    marker = _nonempty_string(
        run["terminal_log_marker"], "F1.run.terminal_log_marker"
    ).encode("utf-8")
    _verify_stable_file_digest(
        log_path,
        expected_sha256=run["log_sha256"],
        expected_bytes=run["log_bytes"],
        context="F1 terminal log",
        required_substring=marker,
    )

    run_files = (
        (
            "wandb_offline_record",
            "wandb_offline_record_sha256",
            "wandb_offline_record_bytes",
            "F1 W&B offline record",
        ),
        (
            "resolved_config_path",
            "resolved_config_sha256",
            "resolved_config_bytes",
            "F1 resolved config",
        ),
        (
            "hydra_overrides_path",
            "hydra_overrides_sha256",
            "hydra_overrides_bytes",
            "F1 Hydra overrides",
        ),
    )
    for path_field, sha_field, bytes_field, context in run_files:
        path = _resolved_recorded_file(run[path_field], f"F1.run.{path_field}")
        _require_within(path, output_dir, context)
        _verify_stable_file_digest(
            path,
            expected_sha256=run[sha_field],
            expected_bytes=run[bytes_field],
            context=context,
        )

    checkpoints = _mapping(record["checkpoints"], "F1.checkpoints")
    seen = set()
    for name in ("best", "last"):
        checkpoint = _mapping(checkpoints[name], f"F1.checkpoints.{name}")
        path = _resolved_recorded_file(
            checkpoint["path"], f"F1.checkpoints.{name}.path"
        )
        _require_within(path, output_dir, f"F1 {name} checkpoint")
        digest = _sha256(
            checkpoint["sha256"], f"F1.checkpoints.{name}.sha256"
        )
        byte_count = _positive_int(
            checkpoint["bytes"], f"F1.checkpoints.{name}.bytes"
        )
        identity = (path, digest, byte_count)
        if identity in seen:
            continue
        seen.add(identity)
        _verify_stable_file_digest(
            path,
            expected_sha256=digest,
            expected_bytes=byte_count,
            context=f"F1 {name} checkpoint",
        )


def _reject_any_true_test_split_flag(value: Any, context: str = "F1") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "test_splits_consulted":
                _strict_bool(child, False, f"{context}.test_splits_consulted")
            _reject_any_true_test_split_flag(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_any_true_test_split_flag(child, f"{context}[{index}]")


def _normalized_absolute(path: Path, context: str) -> Path:
    if not path.is_absolute():
        raise F2LaunchGuardError(f"{context} must be absolute")
    return path.resolve(strict=False)


def verify_f2_trigger_attestation(
    attestation_path: Path,
    *,
    expected_attestation_sha256: str,
    repository_root: Path,
    expected_experiment_uid: str,
    expected_experiment_root: Path,
    expected_output_dir: Path,
    expected_launch_marker: Path,
) -> VerifiedF2TriggerAttestation:
    """Verify every prerequisite for the one authorized F2 launch."""

    payload = _load_canonical_json(
        Path(attestation_path), expected_attestation_sha256
    )
    _validate_attestation_shape(payload)

    f1_attestation = _mapping(payload["f1"], "attestation.f1")
    f1_path = _resolve_repository_record(
        Path(repository_root), f1_attestation["record_path"]
    )
    f1_raw = _read_stable_bytes(f1_path, "F1 record")
    f1_sha256 = hashlib.sha256(f1_raw).hexdigest()
    if f1_sha256 != f1_attestation["record_sha256"]:
        raise F2LaunchGuardError(
            "local F1 YAML SHA-256 differs from the trigger attestation"
        )
    record = _load_strict_f1_yaml(f1_raw)
    _validate_frozen_f1_contract(record, f1_attestation["run_uid"])
    failed_gates = _validate_complete_epochs(record)
    _validate_final_checkpoint_and_result(record)
    _validate_recorded_f1_files(record)
    _reject_any_true_test_split_flag(record)

    attested_failed = tuple(
        _sequence(
            _mapping(payload["decision"], "attestation.decision")["failed_gates"],
            "attestation.decision.failed_gates",
        )
    )
    if attested_failed != failed_gates:
        raise F2LaunchGuardError(
            "attested failed_gates do not exactly match recomputed final-epoch gates"
        )

    expected_uid = _nonempty_string(
        expected_experiment_uid, "expected F2 experiment UID"
    )
    if RUN_UID_RE.fullmatch(expected_uid) is None:
        raise F2LaunchGuardError("expected F2 experiment UID is malformed")
    expected_root = _normalized_absolute(
        Path(expected_experiment_root), "expected F2 experiment root"
    )
    expected_output = _normalized_absolute(
        Path(expected_output_dir), "expected F2 output directory"
    )
    expected_marker = _normalized_absolute(
        Path(expected_launch_marker), "expected F2 launch marker"
    )
    if expected_output != expected_root / expected_uid:
        raise F2LaunchGuardError(
            "expected F2 output directory is not experiment_root/experiment_uid"
        )

    prelaunch = _mapping(payload["f2_prelaunch"], "attestation.f2_prelaunch")
    attested_root = _normalized_absolute(
        Path(prelaunch["experiment_root"]), "attested F2 experiment root"
    )
    attested_output = _normalized_absolute(
        Path(prelaunch["output_dir"]), "attested F2 output directory"
    )
    attested_marker = _normalized_absolute(
        Path(prelaunch["launch_marker"]), "attested F2 launch marker"
    )
    if prelaunch["experiment_uid"] != expected_uid:
        raise F2LaunchGuardError("F2 experiment UID differs from the attestation")
    if (
        attested_root != expected_root
        or attested_output != expected_output
        or attested_marker != expected_marker
    ):
        raise F2LaunchGuardError(
            "F2 experiment root/output/launch marker differs from the attestation"
        )
    # Reject the whole dedicated experiment root, not only one UID, so a prior
    # F2 launch under a different UID cannot be hidden from this one-shot route.
    if expected_root.exists():
        raise F2LaunchGuardError(
            "F2 experiment root already exists; F2 may have started early"
        )
    if expected_output.exists():
        raise F2LaunchGuardError("F2 output directory already exists")
    if expected_marker.exists():
        raise F2LaunchGuardError(
            "F2 atomic launch marker already exists; relaunch/resume is forbidden"
        )

    return VerifiedF2TriggerAttestation(
        attestation_sha256=_sha256(
            expected_attestation_sha256, "expected attestation SHA-256"
        ),
        f1_record_sha256=f1_sha256,
        f1_run_uid=f1_attestation["run_uid"],
        failed_gates=failed_gates,
        experiment_uid=expected_uid,
        experiment_root=expected_root,
        output_dir=expected_output,
        launch_marker=expected_marker,
    )


def _validate_authorization_shape(payload: Mapping[str, Any]) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "purpose",
            "authorization_id",
            "source",
            "runtime",
            "artifacts",
            "f1",
            "decision",
            "f2_prelaunch",
        },
        "authorization",
    )
    if payload["schema_version"] != AUTHORIZATION_SCHEMA_VERSION:
        raise F2LaunchGuardError("authorization schema_version is unsupported")
    if payload["purpose"] != AUTHORIZATION_PURPOSE:
        raise F2LaunchGuardError("authorization purpose drifted")
    authorization_id = _plain_string(
        payload["authorization_id"], "authorization.authorization_id"
    )
    if RUN_UID_RE.fullmatch(authorization_id) is None:
        raise F2LaunchGuardError("authorization.authorization_id is malformed")

    source = _mapping(payload["source"], "authorization.source")
    _exact_keys(source, {"git_revision"}, "authorization.source")
    _git_revision(source["git_revision"], "authorization.source.git_revision")

    runtime = _mapping(payload["runtime"], "authorization.runtime")
    _exact_keys(
        runtime,
        {
            "repository_root",
            "navsim_exp_root",
            "openscene_data_root",
            "nuplan_maps_root",
            "navsim_log_root",
            "sensor_blobs_root",
            "qualification_source_root",
        },
        "authorization.runtime",
    )
    for field in runtime:
        value = _plain_string(runtime[field], f"authorization.runtime.{field}")
        if not Path(value).is_absolute():
            raise F2LaunchGuardError(
                f"authorization.runtime.{field} must be absolute"
            )

    artifacts = _mapping(payload["artifacts"], "authorization.artifacts")
    _exact_keys(artifacts, AUTHORIZED_ARTIFACT_KEYS, "authorization.artifacts")
    for name in AUTHORIZED_ARTIFACT_KEYS:
        artifact = _mapping(artifacts[name], f"authorization.artifacts.{name}")
        _exact_keys(
            artifact, {"path", "sha256"}, f"authorization.artifacts.{name}"
        )
        path = _plain_string(
            artifact["path"], f"authorization.artifacts.{name}.path"
        )
        if not Path(path).is_absolute():
            raise F2LaunchGuardError(
                f"authorization.artifacts.{name}.path must be absolute"
            )
        _sha256(
            artifact["sha256"], f"authorization.artifacts.{name}.sha256"
        )

    prelaunch = _mapping(payload["f2_prelaunch"], "authorization.f2_prelaunch")
    _exact_keys(
        prelaunch,
        {
            "started",
            "experiment_name",
            "experiment_uid",
            "experiment_root",
            "output_dir",
            "launch_marker",
        },
        "authorization.f2_prelaunch",
    )
    if prelaunch["experiment_name"] != F2_EXPERIMENT_NAME:
        raise F2LaunchGuardError("authorization F2 experiment_name drifted")
    if prelaunch["experiment_uid"] != F2_EXPERIMENT_UID:
        raise F2LaunchGuardError("authorization F2 experiment_uid drifted")

    # Reuse the exhaustive v1 F1/decision/prelaunch validation rather than
    # maintaining a second interpretation of the completed F1 evidence.
    legacy_prelaunch = {
        key: value for key, value in prelaunch.items() if key != "experiment_name"
    }
    _validate_attestation_shape(
        {
            "schema_version": ATTESTATION_SCHEMA_VERSION,
            "f1": payload["f1"],
            "decision": payload["decision"],
            "f2_prelaunch": legacy_prelaunch,
        }
    )


def _git_executable() -> str:
    system = Path("/usr/bin/git")
    if system.is_file():
        return str(system)
    executable = shutil.which("git")
    if executable is None:
        raise F2LaunchGuardError("git is required for source authentication")
    return executable


def _git_command(repository_root: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        [_git_executable(), "-C", str(repository_root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise F2LaunchGuardError(
            f"git source authentication failed for {' '.join(arguments)}"
            + (f": {detail}" if detail else "")
        )
    try:
        return completed.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise F2LaunchGuardError("git returned non-UTF-8 source evidence") from error


def _authenticate_authorized_source(
    repository_root: Path, expected_revision: str
) -> Path:
    root = _strict_nonsymlink_path(
        repository_root, "authorized repository root", directory=True
    )
    top_level = Path(
        _git_command(root, ("rev-parse", "--show-toplevel"))
    ).resolve(strict=True)
    if os.path.normcase(str(top_level)) != os.path.normcase(str(root)):
        raise F2LaunchGuardError(
            "authorized repository root is not the exact Git top-level"
        )
    expected = _git_revision(expected_revision, "authorized source revision")
    actual = _git_command(root, ("rev-parse", "HEAD"))
    if actual != expected:
        raise F2LaunchGuardError(
            f"live source revision {actual} differs from signed revision {expected}"
        )
    dirty = _git_command(
        root, ("status", "--porcelain=v1", "--untracked-files=all")
    )
    if dirty:
        raise F2LaunchGuardError(
            "repository worktree is not exactly clean at the signed revision"
        )
    submodules = _git_command(root, ("submodule", "status", "--recursive"))
    if any(line.startswith(("+", "-", "U")) for line in submodules.splitlines()):
        raise F2LaunchGuardError(
            "a Git submodule differs from the signed source revision"
        )
    return root


def _reject_ignored_runtime_import_shadows(repository_root: Path) -> None:
    """Reject ignored files that Python/native import could execute."""

    output = _git_command(
        repository_root,
        (
            "ls-files",
            "-z",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            "navsim_v2/navsim",
            "navsim_v2/vjepa2/evals/image_classification_frozen/modelcustom",
            "navsim_v2/vjepa2/src/masks",
            "navsim_v2/vjepa2/src/models",
            "navsim_v2/vjepa2/src/utils",
        ),
    )
    executable_suffixes = (".py", ".pyc", ".pyo", ".pyd", ".so")
    for relative in output.split("\0"):
        relative = relative.strip()
        if relative.lower().endswith(executable_suffixes):
            raise F2LaunchGuardError(
                "ignored import-shadow artifact exists in the F2 runtime: "
                f"{relative}"
            )


def _signed_absolute_path(value: Any, context: str) -> Path:
    raw = _plain_string(value, context)
    path = Path(raw)
    if not path.is_absolute():
        raise F2LaunchGuardError(f"{context} must be absolute")
    normalized = Path(os.path.abspath(str(path)))
    if os.path.normcase(str(normalized)) != os.path.normcase(str(path)):
        raise F2LaunchGuardError(f"{context} must be lexically normalized")
    return normalized


def _read_proc_pseudo_file(path: Path, context: str) -> bytes:
    """Read one procfs pseudo-file through a no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            chunks = []
            while True:
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                if sum(map(len, chunks)) > 1024 * 1024:
                    raise F2LaunchGuardError(f"{context} is unexpectedly large")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except F2LaunchGuardError:
        raise
    except OSError as error:
        raise F2LaunchGuardError(f"cannot read {context}: {path}") from error


def read_linux_boot_id(*, proc_root: Path = Path("/proc")) -> str:
    """Return the canonical Linux boot UUID used to reject PID replay."""

    raw = _read_proc_pseudo_file(
        Path(proc_root) / "sys/kernel/random/boot_id", "Linux boot identity"
    )
    try:
        boot_id = raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise F2LaunchGuardError("Linux boot identity is not ASCII") from error
    if BOOT_ID_RE.fullmatch(boot_id) is None:
        raise F2LaunchGuardError("Linux boot identity is not a canonical UUID")
    return boot_id


def read_linux_process_start_time(
    pid: int, *, proc_root: Path = Path("/proc")
) -> int:
    """Read Linux ``/proc/<pid>/stat`` field 22 without parsing ``comm``."""

    if type(pid) is not int or pid <= 1:
        raise F2LaunchGuardError("launcher PID must be an integer greater than one")
    raw = _read_proc_pseudo_file(
        Path(proc_root) / str(pid) / "stat", f"process {pid} start identity"
    )
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise F2LaunchGuardError("process stat identity is not ASCII") from error
    # ``comm`` is parenthesized and may itself contain whitespace or ')'.
    close = text.rfind(")")
    if close < 0 or close + 2 > len(text):
        raise F2LaunchGuardError("process stat identity is malformed")
    prefix = text[:close]
    if "(" not in prefix:
        raise F2LaunchGuardError("process stat identity is malformed")
    remainder = text[close + 2 :].split()
    # remainder[0] is field 3 (state); remainder[19] is field 22 (starttime).
    if len(remainder) <= 19 or not remainder[19].isdigit():
        raise F2LaunchGuardError("process stat start identity is malformed")
    start_time = int(remainder[19])
    if start_time <= 0:
        raise F2LaunchGuardError("process stat start identity must be positive")
    return start_time


def _require_dedicated_launcher_parent(
    launcher_pid: int,
    repository_root: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> None:
    """Require the claim helper to be a direct child of the reviewed script.

    This is not treated as a cryptographic trust root; the SSHSIG contract is
    that root.  It closes the ordinary ``python -m ... claim`` bypass and makes
    the receipt evidence that the reviewed preflight script reached its final
    claim instruction.
    """

    try:
        expected_script = (
            repository_root
            / "navsim_v2/scripts/training/train_actworld_jepa_f2.sh"
        ).resolve(strict=True)
    except OSError as error:
        raise F2LaunchGuardError(
            "claim helper parent is not the dedicated signed F2 launcher"
        ) from error
    raw = _read_proc_pseudo_file(
        Path(proc_root) / str(launcher_pid) / "cmdline",
        f"launcher process {launcher_pid} command line",
    )
    arguments = [part for part in raw.split(b"\0") if part]
    if not arguments:
        raise F2LaunchGuardError("launcher process command line is empty")
    try:
        decoded = [part.decode("utf-8") for part in arguments]
    except UnicodeDecodeError as error:
        raise F2LaunchGuardError("launcher process command line is not UTF-8") from error
    if "-c" in decoded[1:]:
        raise F2LaunchGuardError("F2 claim cannot originate from bash -c")
    # Bash reserves descriptor 255 for the script it is actively executing.
    # Unlike argv, this identity remains absolute and stable after the launcher
    # changes working directory, so legitimate relative invocation is safe.
    try:
        opened_script = (
            Path(proc_root) / str(launcher_pid) / "fd" / "255"
        ).resolve(strict=True)
    except OSError as error:
        raise F2LaunchGuardError(
            "claim helper parent is not the dedicated signed F2 launcher"
        ) from error
    if os.path.normcase(str(opened_script)) != os.path.normcase(
        str(expected_script)
    ):
        raise F2LaunchGuardError(
            "claim helper parent is not the dedicated signed F2 launcher"
        )


def verify_f2_signed_contract_lightweight(
    authorization_path: Path,
    signature_path: Path,
    *,
    repository_root: Path,
    trusted_public_key: str = TRUSTED_AUTHORITY_SSH_PUBLIC_KEY,
) -> VerifiedF2SignedContract:
    """Authenticate the fixed signed envelope without hashing large artifacts."""

    authorization_file = _strict_nonsymlink_path(
        authorization_path, "F2 launch authorization", directory=False
    )
    signature_file = _strict_nonsymlink_path(
        signature_path, "F2 launch authorization signature", directory=False
    )
    authorization_raw = _read_stable_bytes(
        authorization_file, "F2 launch authorization"
    )
    signature_raw = _read_stable_bytes(
        signature_file, "F2 launch authorization signature"
    )
    verify_detached_ssh_signature(
        authorization_raw,
        signature_raw,
        trusted_public_key=trusted_public_key,
    )
    payload = _parse_canonical_json_bytes(
        authorization_raw, "F2 launch authorization"
    )
    _validate_authorization_shape(payload)

    runtime_payload = _mapping(payload["runtime"], "authorization.runtime")
    signed_repository = _signed_absolute_path(
        runtime_payload["repository_root"],
        "authorization.runtime.repository_root",
    )
    live_repository = _strict_nonsymlink_path(
        repository_root, "live repository root", directory=True
    )
    if os.path.normcase(str(signed_repository)) != os.path.normcase(
        str(live_repository)
    ):
        raise F2LaunchGuardError(
            "live repository root differs from the signed repository root"
        )
    source = _mapping(payload["source"], "authorization.source")
    source_revision = _git_revision(
        source["git_revision"], "authorization.source.git_revision"
    )
    live_repository = _authenticate_authorized_source(
        live_repository, source_revision
    )
    _reject_ignored_runtime_import_shadows(live_repository)

    runtime: Dict[str, Path] = {}
    for name, value in runtime_payload.items():
        signed = _signed_absolute_path(
            value, f"authorization.runtime.{name}"
        )
        runtime[name] = _strict_nonsymlink_path(
            signed,
            f"authorized runtime {name}",
            directory=True,
        )

    artifacts_payload = _mapping(payload["artifacts"], "authorization.artifacts")
    artifacts: Dict[str, AuthorizedArtifact] = {}
    navsim_exp_root = runtime["navsim_exp_root"]
    for name in AUTHORIZED_ARTIFACT_KEYS:
        item = _mapping(
            artifacts_payload[name], f"authorization.artifacts.{name}"
        )
        path = _strict_nonsymlink_path(
            _signed_absolute_path(
                item["path"], f"authorization.artifacts.{name}.path"
            ),
            f"authorized {name}",
            directory=False,
        )
        _require_within(path, navsim_exp_root, f"authorized {name}")
        artifacts[name] = AuthorizedArtifact(
            path=path,
            sha256=_sha256(
                item["sha256"], f"authorization.artifacts.{name}.sha256"
            ),
        )
    if len({artifact.path for artifact in artifacts.values()}) != len(artifacts):
        raise F2LaunchGuardError("two authorized artifact names resolve to one file")

    prelaunch = _mapping(payload["f2_prelaunch"], "authorization.f2_prelaunch")
    experiment_root = _signed_absolute_path(
        prelaunch["experiment_root"], "authorization.f2_prelaunch.experiment_root"
    )
    output_dir = _signed_absolute_path(
        prelaunch["output_dir"], "authorization.f2_prelaunch.output_dir"
    )
    launch_marker = _signed_absolute_path(
        prelaunch["launch_marker"], "authorization.f2_prelaunch.launch_marker"
    )
    expected_experiment_root = navsim_exp_root / F2_EXPERIMENT_NAME
    expected_output = expected_experiment_root / F2_EXPERIMENT_UID
    expected_marker = (
        navsim_exp_root
        / ".actworld_f2_launch_ledger"
        / f"{F2_EXPERIMENT_UID}.started"
    )
    if experiment_root != expected_experiment_root:
        raise F2LaunchGuardError("signed F2 experiment root relation drifted")
    if output_dir != expected_output:
        raise F2LaunchGuardError("signed F2 output directory relation drifted")
    if launch_marker != expected_marker:
        raise F2LaunchGuardError("signed F2 launch marker path drifted")

    return VerifiedF2SignedContract(
        authorization_sha256=hashlib.sha256(authorization_raw).hexdigest(),
        signature_sha256=hashlib.sha256(signature_raw).hexdigest(),
        source_revision=source_revision,
        repository_root=live_repository,
        runtime=runtime,
        artifacts=artifacts,
        experiment_name=F2_EXPERIMENT_NAME,
        experiment_uid=F2_EXPERIMENT_UID,
        experiment_root=experiment_root,
        output_dir=output_dir,
        launch_marker=launch_marker,
    )


def _runtime_claim_payload(
    contract: VerifiedF2SignedContract,
    *,
    boot_id: str,
    launcher_pid: int,
    launcher_start_time_ticks: int,
) -> Mapping[str, Any]:
    return {
        "schema_version": CLAIM_RECEIPT_SCHEMA_VERSION,
        "authorization": {
            "sha256": contract.authorization_sha256,
            "signature_sha256": contract.signature_sha256,
        },
        "source": {
            "git_revision": contract.source_revision,
            "repository_root": str(contract.repository_root),
        },
        "launch": {
            "experiment_name": contract.experiment_name,
            "experiment_uid": contract.experiment_uid,
            "experiment_root": str(contract.experiment_root),
            "output_dir": str(contract.output_dir),
            "launch_marker": str(contract.launch_marker),
        },
        "process": {
            "boot_id": boot_id,
            "launcher_pid": launcher_pid,
            "launcher_start_time_ticks": launcher_start_time_ticks,
        },
    }


def _validate_runtime_claim_shape(payload: Mapping[str, Any]) -> None:
    _exact_keys(
        payload,
        {"schema_version", "authorization", "source", "launch", "process"},
        "runtime claim",
    )
    if payload["schema_version"] != CLAIM_RECEIPT_SCHEMA_VERSION:
        raise F2LaunchGuardError("runtime claim schema_version is unsupported")
    authorization = _mapping(payload["authorization"], "runtime claim.authorization")
    _exact_keys(
        authorization,
        {"sha256", "signature_sha256"},
        "runtime claim.authorization",
    )
    _sha256(authorization["sha256"], "runtime claim authorization SHA-256")
    _sha256(
        authorization["signature_sha256"],
        "runtime claim signature SHA-256",
    )
    source = _mapping(payload["source"], "runtime claim.source")
    _exact_keys(
        source, {"git_revision", "repository_root"}, "runtime claim.source"
    )
    _git_revision(source["git_revision"], "runtime claim source revision")
    _signed_absolute_path(source["repository_root"], "runtime claim repository root")
    launch = _mapping(payload["launch"], "runtime claim.launch")
    _exact_keys(
        launch,
        {
            "experiment_name",
            "experiment_uid",
            "experiment_root",
            "output_dir",
            "launch_marker",
        },
        "runtime claim.launch",
    )
    if launch["experiment_name"] != F2_EXPERIMENT_NAME:
        raise F2LaunchGuardError("runtime claim experiment_name drifted")
    if launch["experiment_uid"] != F2_EXPERIMENT_UID:
        raise F2LaunchGuardError("runtime claim experiment_uid drifted")
    for field in ("experiment_root", "output_dir", "launch_marker"):
        _signed_absolute_path(
            launch[field], f"runtime claim.launch.{field}"
        )
    process = _mapping(payload["process"], "runtime claim.process")
    _exact_keys(
        process,
        {"boot_id", "launcher_pid", "launcher_start_time_ticks"},
        "runtime claim.process",
    )
    if not isinstance(process["boot_id"], str) or BOOT_ID_RE.fullmatch(
        process["boot_id"]
    ) is None:
        raise F2LaunchGuardError("runtime claim boot_id is malformed")
    for field in ("launcher_pid", "launcher_start_time_ticks"):
        if type(process[field]) is not int or process[field] <= 1:
            raise F2LaunchGuardError(f"runtime claim {field} must be > 1")


def _create_claim_receipt_at_marker(
    contract: VerifiedF2SignedContract, payload: bytes
) -> None:
    """Create the ledger and receipt only through no-follow directory FDs."""

    marker = contract.launch_marker
    root = contract.runtime["navsim_exp_root"]
    if marker.parent.parent != root or marker.parent.name != ".actworld_f2_launch_ledger":
        raise F2LaunchGuardError("runtime claim marker escaped signed experiment root")
    if marker.name != f"{contract.experiment_uid}.started":
        raise F2LaunchGuardError("runtime claim marker filename drifted")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        root_fd = os.open(root, directory_flags)
    except OSError as error:
        raise F2LaunchGuardError("cannot open signed NAVSIM experiment root") from error
    ledger_fd: Optional[int] = None
    try:
        try:
            os.mkdir(marker.parent.name, 0o700, dir_fd=root_fd)
            os.fsync(root_fd)
        except FileExistsError:
            pass
        try:
            ledger_fd = os.open(marker.parent.name, directory_flags, dir_fd=root_fd)
        except OSError as error:
            raise F2LaunchGuardError(
                "F2 launch ledger must be a non-symlink directory"
            ) from error
        metadata = os.fstat(ledger_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise F2LaunchGuardError("F2 launch ledger is not a directory")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            receipt_fd = os.open(marker.name, flags, 0o600, dir_fd=ledger_fd)
        except FileExistsError as error:
            raise F2LaunchGuardError(
                "F2 atomic launch marker already exists; relaunch is forbidden"
            ) from error
        except OSError as error:
            raise F2LaunchGuardError("cannot atomically create F2 runtime claim") from error
        try:
            view = memoryview(payload)
            while view:
                written = os.write(receipt_fd, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fsync(receipt_fd)
        finally:
            os.close(receipt_fd)
        os.fsync(ledger_fd)
    finally:
        if ledger_fd is not None:
            os.close(ledger_fd)
        os.close(root_fd)


def create_f2_runtime_claim(
    authorization_path: Path,
    signature_path: Path,
    *,
    repository_root: Path,
    launcher_pid: int,
    trusted_public_key: str = TRUSTED_AUTHORITY_SSH_PUBLIC_KEY,
    proc_root: Path = Path("/proc"),
    require_dedicated_launcher: bool = True,
) -> VerifiedF2RuntimeClaim:
    """Atomically create the canonical receipt immediately before shell exec."""

    if type(launcher_pid) is not int or launcher_pid != os.getppid():
        raise F2LaunchGuardError(
            "claim helper must be a direct child of the declared launcher PID"
        )
    contract = verify_f2_signed_contract_lightweight(
        authorization_path,
        signature_path,
        repository_root=repository_root,
        trusted_public_key=trusted_public_key,
    )
    if contract.experiment_root.exists() or contract.output_dir.exists():
        raise F2LaunchGuardError(
            "F2 output appeared before the runtime claim was admitted"
        )
    if require_dedicated_launcher:
        _require_dedicated_launcher_parent(
            launcher_pid, contract.repository_root, proc_root=proc_root
        )
    boot_id = read_linux_boot_id(proc_root=proc_root)
    start_time = read_linux_process_start_time(launcher_pid, proc_root=proc_root)
    payload = _runtime_claim_payload(
        contract,
        boot_id=boot_id,
        launcher_pid=launcher_pid,
        launcher_start_time_ticks=start_time,
    )
    raw = canonical_attestation_bytes(payload)
    _create_claim_receipt_at_marker(contract, raw)
    return verify_f2_runtime_claim(
        contract.launch_marker,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        signed_contract=contract,
        expected_parent_pid=launcher_pid,
        proc_root=proc_root,
    )


def verify_f2_runtime_claim(
    claim_path: Path,
    *,
    expected_sha256: str,
    signed_contract: VerifiedF2SignedContract,
    expected_parent_pid: Optional[int] = None,
    proc_root: Path = Path("/proc"),
    require_live_parent: bool = True,
) -> VerifiedF2RuntimeClaim:
    """Replay a claim and, at runtime, bind it to the direct torchrun parent."""

    expected_digest = _sha256(expected_sha256, "expected runtime claim SHA-256")
    claim = _strict_nonsymlink_path(
        claim_path, "F2 runtime claim", directory=False
    )
    if claim != signed_contract.launch_marker:
        raise F2LaunchGuardError("runtime claim path differs from signed launch marker")
    raw = _read_stable_bytes(claim, "F2 runtime claim")
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_digest:
        raise F2LaunchGuardError(
            "runtime claim SHA-256 mismatch: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    payload = _parse_canonical_json_bytes(raw, "F2 runtime claim")
    _validate_runtime_claim_shape(payload)
    authorization = _mapping(payload["authorization"], "runtime claim.authorization")
    source = _mapping(payload["source"], "runtime claim.source")
    launch = _mapping(payload["launch"], "runtime claim.launch")
    process = _mapping(payload["process"], "runtime claim.process")
    expected_values = {
        "authorization.sha256": (
            authorization["sha256"], signed_contract.authorization_sha256
        ),
        "authorization.signature_sha256": (
            authorization["signature_sha256"], signed_contract.signature_sha256
        ),
        "source.git_revision": (
            source["git_revision"], signed_contract.source_revision
        ),
        "source.repository_root": (
            source["repository_root"], str(signed_contract.repository_root)
        ),
        "launch.experiment_name": (
            launch["experiment_name"], signed_contract.experiment_name
        ),
        "launch.experiment_uid": (
            launch["experiment_uid"], signed_contract.experiment_uid
        ),
        "launch.experiment_root": (
            launch["experiment_root"], str(signed_contract.experiment_root)
        ),
        "launch.output_dir": (
            launch["output_dir"], str(signed_contract.output_dir)
        ),
        "launch.launch_marker": (
            launch["launch_marker"], str(signed_contract.launch_marker)
        ),
    }
    for field, (actual, expected) in expected_values.items():
        if actual != expected:
            raise F2LaunchGuardError(
                f"runtime claim {field} differs from signed authorization"
            )
    launcher_pid = process["launcher_pid"]
    start_time = process["launcher_start_time_ticks"]
    boot_id = process["boot_id"]
    if require_live_parent:
        parent_pid = os.getppid() if expected_parent_pid is None else expected_parent_pid
        if launcher_pid != parent_pid:
            raise F2LaunchGuardError(
                "runtime claim launcher PID differs from direct torchrun parent"
            )
        if read_linux_boot_id(proc_root=proc_root) != boot_id:
            raise F2LaunchGuardError("runtime claim boot identity is stale")
        live_start = read_linux_process_start_time(parent_pid, proc_root=proc_root)
        if live_start != start_time:
            raise F2LaunchGuardError(
                "runtime claim torchrun parent start identity is stale"
            )
    return VerifiedF2RuntimeClaim(
        path=claim,
        sha256=actual_digest,
        authorization_sha256=authorization["sha256"],
        signature_sha256=authorization["signature_sha256"],
        source_revision=source["git_revision"],
        repository_root=Path(source["repository_root"]),
        experiment_name=launch["experiment_name"],
        experiment_uid=launch["experiment_uid"],
        experiment_root=Path(launch["experiment_root"]),
        output_dir=Path(launch["output_dir"]),
        launch_marker=Path(launch["launch_marker"]),
        boot_id=boot_id,
        launcher_pid=launcher_pid,
        launcher_start_time_ticks=start_time,
    )


def verify_f2_launch_authorization(
    authorization_path: Path,
    signature_path: Path,
    *,
    repository_root: Path,
    trusted_public_key: str = TRUSTED_AUTHORITY_SSH_PUBLIC_KEY,
) -> VerifiedF2LaunchAuthorization:
    """Verify the signed v2 trust root and every named launch input.

    No expected digest or revision is accepted from the caller.  Those values
    come only from canonical bytes authenticated by the embedded public key.
    """

    authorization_file = _strict_nonsymlink_path(
        authorization_path, "F2 launch authorization", directory=False
    )
    signature_file = _strict_nonsymlink_path(
        signature_path, "F2 launch authorization signature", directory=False
    )
    authorization_raw = _read_stable_bytes(
        authorization_file, "F2 launch authorization"
    )
    signature_raw = _read_stable_bytes(
        signature_file, "F2 launch authorization signature"
    )
    verify_detached_ssh_signature(
        authorization_raw,
        signature_raw,
        trusted_public_key=trusted_public_key,
    )
    payload = _parse_canonical_json_bytes(
        authorization_raw, "F2 launch authorization"
    )
    _validate_authorization_shape(payload)

    runtime = _mapping(payload["runtime"], "authorization.runtime")
    signed_repository = _signed_absolute_path(
        runtime["repository_root"], "authorization.runtime.repository_root"
    )
    caller_repository = _strict_nonsymlink_path(
        Path(repository_root), "live repository root", directory=True
    )
    if os.path.normcase(str(signed_repository)) != os.path.normcase(
        str(caller_repository)
    ):
        raise F2LaunchGuardError(
            "live repository root differs from the signed repository root"
        )
    source = _mapping(payload["source"], "authorization.source")
    source_revision = _git_revision(
        source["git_revision"], "authorization.source.git_revision"
    )
    repository = _authenticate_authorized_source(
        caller_repository, source_revision
    )

    navsim_exp_root = _strict_nonsymlink_path(
        _signed_absolute_path(
            runtime["navsim_exp_root"], "authorization.runtime.navsim_exp_root"
        ),
        "authorized NAVSIM experiment root",
        directory=True,
    )
    openscene_data_root = _strict_nonsymlink_path(
        _signed_absolute_path(
            runtime["openscene_data_root"],
            "authorization.runtime.openscene_data_root",
        ),
        "authorized OpenScene data root",
        directory=True,
    )
    nuplan_maps_root = _strict_nonsymlink_path(
        _signed_absolute_path(
            runtime["nuplan_maps_root"],
            "authorization.runtime.nuplan_maps_root",
        ),
        "authorized nuPlan maps root",
        directory=True,
    )
    navsim_log_root = _strict_nonsymlink_path(
        _signed_absolute_path(
            runtime["navsim_log_root"],
            "authorization.runtime.navsim_log_root",
        ),
        "authorized NAVSIM log root",
        directory=True,
    )
    sensor_blobs_root = _strict_nonsymlink_path(
        _signed_absolute_path(
            runtime["sensor_blobs_root"],
            "authorization.runtime.sensor_blobs_root",
        ),
        "authorized sensor blobs root",
        directory=True,
    )
    qualification_source_root = _strict_nonsymlink_path(
        _signed_absolute_path(
            runtime["qualification_source_root"],
            "authorization.runtime.qualification_source_root",
        ),
        "authorized qualification source root",
        directory=True,
    )
    # Production's OpenScene trainval children are symlinks to two different
    # disks.  The authorization therefore names each canonical real root
    # directly; containment under the lexical OpenScene root would be false.

    artifacts_payload = _mapping(payload["artifacts"], "authorization.artifacts")
    artifacts: Dict[str, AuthorizedArtifact] = {}
    for name in AUTHORIZED_ARTIFACT_KEYS:
        item = _mapping(
            artifacts_payload[name], f"authorization.artifacts.{name}"
        )
        expected_sha256 = _sha256(
            item["sha256"], f"authorization.artifacts.{name}.sha256"
        )
        path = _hash_authorized_file(
            _signed_absolute_path(
                item["path"], f"authorization.artifacts.{name}.path"
            ),
            expected_sha256,
            f"authorized {name}",
        )
        _require_within(path, navsim_exp_root, f"authorized {name}")
        artifacts[name] = AuthorizedArtifact(path=path, sha256=expected_sha256)
    if len({artifact.path for artifact in artifacts.values()}) != len(artifacts):
        raise F2LaunchGuardError("two authorized artifact names resolve to one file")

    f1_attestation = _mapping(payload["f1"], "authorization.f1")
    f1_path = _resolve_repository_record(repository, f1_attestation["record_path"])
    f1_raw = _read_stable_bytes(f1_path, "F1 record")
    f1_sha256 = hashlib.sha256(f1_raw).hexdigest()
    if f1_sha256 != f1_attestation["record_sha256"]:
        raise F2LaunchGuardError(
            "local F1 YAML SHA-256 differs from the signed authorization"
        )
    record = _load_strict_f1_yaml(f1_raw)
    _validate_frozen_f1_contract(record, f1_attestation["run_uid"])
    failed_gates = _validate_complete_epochs(record)
    _validate_final_checkpoint_and_result(record)
    _validate_recorded_f1_files(record)
    _reject_any_true_test_split_flag(record)
    attested_failed = tuple(
        _sequence(
            _mapping(payload["decision"], "authorization.decision")["failed_gates"],
            "authorization.decision.failed_gates",
        )
    )
    if attested_failed != failed_gates:
        raise F2LaunchGuardError(
            "signed failed_gates do not exactly match recomputed final-epoch gates"
        )
    lineage = _mapping(record["lineage"], "F1.lineage")
    if lineage["drive_jepa_checkpoint_sha256"] != artifacts[
        "baseline_checkpoint"
    ].sha256:
        raise F2LaunchGuardError(
            "signed baseline checkpoint differs from the completed F1 lineage"
        )
    data = _mapping(record["data"], "F1.data")
    if data["metric_cache_metadata_sha256"] != artifacts["cache_metadata"].sha256:
        raise F2LaunchGuardError(
            "signed cache metadata differs from the completed F1 evidence"
        )

    prelaunch = _mapping(payload["f2_prelaunch"], "authorization.f2_prelaunch")
    experiment_root = _signed_absolute_path(
        prelaunch["experiment_root"], "authorization.f2_prelaunch.experiment_root"
    )
    output_dir = _signed_absolute_path(
        prelaunch["output_dir"], "authorization.f2_prelaunch.output_dir"
    )
    launch_marker = _signed_absolute_path(
        prelaunch["launch_marker"], "authorization.f2_prelaunch.launch_marker"
    )
    expected_experiment_root = navsim_exp_root / F2_EXPERIMENT_NAME
    if experiment_root != expected_experiment_root:
        raise F2LaunchGuardError(
            "signed F2 experiment root is not navsim_exp_root/experiment_name"
        )
    if output_dir != experiment_root / F2_EXPERIMENT_UID:
        raise F2LaunchGuardError(
            "signed F2 output directory is not experiment_root/experiment_uid"
        )
    expected_marker = (
        navsim_exp_root
        / ".actworld_f2_launch_ledger"
        / f"{F2_EXPERIMENT_UID}.started"
    )
    if launch_marker != expected_marker:
        raise F2LaunchGuardError("signed F2 launch marker path drifted")
    if experiment_root.exists():
        raise F2LaunchGuardError(
            "F2 experiment root already exists; F2 may have started early"
        )
    if output_dir.exists():
        raise F2LaunchGuardError("F2 output directory already exists")
    if launch_marker.exists():
        raise F2LaunchGuardError(
            "F2 atomic launch marker already exists; relaunch/resume is forbidden"
        )

    return VerifiedF2LaunchAuthorization(
        authorization_sha256=hashlib.sha256(authorization_raw).hexdigest(),
        signature_sha256=hashlib.sha256(signature_raw).hexdigest(),
        source_revision=source_revision,
        repository_root=repository,
        navsim_exp_root=navsim_exp_root,
        openscene_data_root=openscene_data_root,
        nuplan_maps_root=nuplan_maps_root,
        navsim_log_root=navsim_log_root,
        sensor_blobs_root=sensor_blobs_root,
        qualification_source_root=qualification_source_root,
        artifacts=artifacts,
        f1_record_sha256=f1_sha256,
        f1_run_uid=f1_attestation["run_uid"],
        failed_gates=failed_gates,
        experiment_name=F2_EXPERIMENT_NAME,
        experiment_uid=F2_EXPERIMENT_UID,
        experiment_root=experiment_root,
        output_dir=output_dir,
        launch_marker=launch_marker,
    )


def issue_f2_launch_authorization(
    draft_path: Path,
    authorization_output_path: Path,
    signature_output_path: Path,
    *,
    private_key_path: Path,
    trusted_public_key: str = TRUSTED_AUTHORITY_SSH_PUBLIC_KEY,
) -> Tuple[str, str]:
    """Explicitly canonicalize and sign one authorization without overwrite."""

    draft_file = _strict_nonsymlink_path(
        draft_path, "F2 authorization draft", directory=False
    )
    draft_raw = _read_stable_bytes(draft_file, "F2 authorization draft")
    payload = _decode_strict_json(draft_raw, "F2 authorization draft")
    _validate_authorization_shape(payload)
    canonical = canonical_attestation_bytes(payload)
    authorization_output = Path(authorization_output_path)
    signature_output = Path(signature_output_path)
    if authorization_output == signature_output:
        raise F2LaunchGuardError(
            "authorization and detached signature outputs must differ"
        )
    # Preflight both names, then rely on O_EXCL for the authoritative race check.
    if authorization_output.exists() or signature_output.exists():
        raise F2LaunchGuardError(
            "authorization/signature output already exists; overwrite is forbidden"
        )
    signature = _sign_with_ssh_private_key(
        canonical,
        private_key_path=private_key_path,
        trusted_public_key=trusted_public_key,
    )
    _write_bytes_no_overwrite(
        authorization_output, canonical, "F2 launch authorization"
    )
    _write_bytes_no_overwrite(
        signature_output, signature, "F2 launch authorization signature"
    )
    return hashlib.sha256(canonical).hexdigest(), hashlib.sha256(signature).hexdigest()


def install_f2_launch_authorization(
    authorization_source_path: Path,
    signature_source_path: Path,
    authorization_output_path: Path,
    signature_output_path: Path,
    *,
    repository_root: Path,
    trusted_public_key: str = TRUSTED_AUTHORITY_SSH_PUBLIC_KEY,
) -> Tuple[str, str]:
    """Verify staged signed bytes, then install them with O_EXCL only."""

    authorization_output = Path(authorization_output_path)
    signature_output = Path(signature_output_path)
    if authorization_output == signature_output:
        raise F2LaunchGuardError(
            "authorization and detached signature outputs must differ"
        )
    if authorization_output.exists() or signature_output.exists():
        raise F2LaunchGuardError(
            "authorization/signature output already exists; overwrite is forbidden"
        )
    verified = verify_f2_launch_authorization(
        authorization_source_path,
        signature_source_path,
        repository_root=repository_root,
        trusted_public_key=trusted_public_key,
    )
    authorization_raw = _read_stable_bytes(
        _strict_nonsymlink_path(
            authorization_source_path,
            "staged F2 launch authorization",
            directory=False,
        ),
        "staged F2 launch authorization",
    )
    signature_raw = _read_stable_bytes(
        _strict_nonsymlink_path(
            signature_source_path,
            "staged F2 launch authorization signature",
            directory=False,
        ),
        "staged F2 launch authorization signature",
    )
    if hashlib.sha256(authorization_raw).hexdigest() != verified.authorization_sha256:
        raise F2LaunchGuardError("staged authorization changed after verification")
    if hashlib.sha256(signature_raw).hexdigest() != verified.signature_sha256:
        raise F2LaunchGuardError("staged signature changed after verification")
    _write_bytes_no_overwrite(
        authorization_output,
        authorization_raw,
        "installed F2 launch authorization",
    )
    _write_bytes_no_overwrite(
        signature_output,
        signature_raw,
        "installed F2 launch authorization signature",
    )
    return verified.authorization_sha256, verified.signature_sha256


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue or verify the signed one-shot F2 launch authorization"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser(
        "verify", help="verify the fixed signed authorization and live inputs"
    )
    verify.add_argument("--authorization", required=True, type=Path)
    verify.add_argument("--signature", required=True, type=Path)
    verify.add_argument("--repository-root", required=True, type=Path)
    verify.add_argument(
        "--emit-contract-lines",
        action="store_true",
        help="emit a fixed tab-delimited projection for the dedicated Bash launcher",
    )

    issue = commands.add_parser(
        "issue", help="explicit offline canonicalization and detached signing"
    )
    issue.add_argument("--draft", required=True, type=Path)
    issue.add_argument("--authorization-out", required=True, type=Path)
    issue.add_argument("--signature-out", required=True, type=Path)
    issue.add_argument("--private-key", required=True, type=Path)
    issue.add_argument(
        "--confirm-offline-signing",
        action="store_true",
        help="required acknowledgement; issuance is never implicit",
    )
    install = commands.add_parser(
        "install",
        help="verify staged authorization bytes and install with no overwrite",
    )
    install.add_argument("--authorization-source", required=True, type=Path)
    install.add_argument("--signature-source", required=True, type=Path)
    install.add_argument("--authorization-out", required=True, type=Path)
    install.add_argument("--signature-out", required=True, type=Path)
    install.add_argument("--repository-root", required=True, type=Path)
    install.add_argument(
        "--confirm-no-overwrite-install",
        action="store_true",
        help="required acknowledgement for the one-time external installation",
    )
    claim = commands.add_parser(
        "claim",
        help=(
            "atomically create the canonical runtime claim as a direct child "
            "of the dedicated launcher"
        ),
    )
    claim.add_argument("--authorization", required=True, type=Path)
    claim.add_argument("--signature", required=True, type=Path)
    claim.add_argument("--repository-root", required=True, type=Path)
    claim.add_argument("--launcher-pid", required=True, type=int)
    claim.add_argument(
        "--emit-claim-lines",
        action="store_true",
        help="emit exact tab-delimited claim path and digest for the parent shell",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "issue":
            if not args.confirm_offline_signing:
                raise F2LaunchGuardError(
                    "offline issuance requires --confirm-offline-signing"
                )
            authorization_sha256, signature_sha256 = issue_f2_launch_authorization(
                args.draft,
                args.authorization_out,
                args.signature_out,
                private_key_path=args.private_key,
            )
            print(
                "F2 launch authorization issued without overwrite: "
                f"authorization_sha256={authorization_sha256} "
                f"signature_sha256={signature_sha256}"
            )
            return 0
        if args.command == "install":
            if not args.confirm_no_overwrite_install:
                raise F2LaunchGuardError(
                    "installation requires --confirm-no-overwrite-install"
                )
            authorization_sha256, signature_sha256 = install_f2_launch_authorization(
                args.authorization_source,
                args.signature_source,
                args.authorization_out,
                args.signature_out,
                repository_root=args.repository_root,
            )
            print(
                "F2 launch authorization installed without overwrite: "
                f"authorization_sha256={authorization_sha256} "
                f"signature_sha256={signature_sha256}"
            )
            return 0
        if args.command == "claim":
            claim = create_f2_runtime_claim(
                args.authorization,
                args.signature,
                repository_root=args.repository_root,
                launcher_pid=args.launcher_pid,
            )
            if args.emit_claim_lines:
                print(f"CLAIM_PATH\t{claim.path}")
                print(f"CLAIM_SHA256\t{claim.sha256}")
            else:
                print(
                    "F2 runtime claim created without overwrite: "
                    f"claim_path={claim.path} claim_sha256={claim.sha256} "
                    f"launcher_pid={claim.launcher_pid}"
                )
            return 0
        verified = verify_f2_launch_authorization(
            args.authorization,
            args.signature,
            repository_root=args.repository_root,
        )
    except (F2LaunchGuardError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"F2 launch guard rejected authorization: {error}\n")
    if args.emit_contract_lines:
        for line in verified.contract_lines():
            print(line)
        return 0
    print(
        "F2 launch guard verified signed authorization: "
        f"run_uid={verified.f1_run_uid} "
        f"failed_gates={','.join(verified.failed_gates)} "
        f"authorization_sha256={verified.authorization_sha256} "
        f"source_revision={verified.source_revision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
