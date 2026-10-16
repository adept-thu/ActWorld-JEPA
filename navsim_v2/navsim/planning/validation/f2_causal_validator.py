"""Pure foundations for the preregistered F2 causal navtrain validator.

This module deliberately has no NAVSIM, torch, numpy, checkpoint, or filesystem
runtime dependency.  It freezes the parts of the F2 validation contract that
can be tested with synthetic data:

* canonical held-out-navtrain manifests and their SHA-256 identity;
* exact partition, sensor, and cache-coverage guards;
* deterministic whole-log sharding;
* independent critic/base causal state machines with injected comfort logic;
* token-weighted and micro-pair aggregation, formal gates, and checkpoint rank.

The external adapter is still responsible for loading a checkpoint, running a
full-float32 forward with autocast disabled, producing ``ModelTokenOutput`` for
the original 32 Drive-JEPA proposals, and building the large immutable artifact
represented by ``FrozenTokenArtifact``.  In particular, this module does not
run a model, score a proposal, read NAVSIM data, or claim that the real held-out
artifact exists.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


MANIFEST_SCHEMA_VERSION = "actworld-f2-causal-navtrain-v1"
F2_SPLIT = "navtrain"
F2_PHASE = "development"
F2_PARTITION = "validation"
CANDIDATE_COUNT = 32
TARGET_CHANNEL_NAMES = (
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
TARGET_CHANNEL_COUNT = len(TARGET_CHANNEL_NAMES)
MAX_CONTINUITY_DELTA_US = 550_000
F2_CHECKPOINT_EPOCHS = (0, 1, 2, 3)
F2_MODEL_NUMERIC_MODE = "float32_no_autocast"
F2_DELTA_TANH_SANITY_MAX_ULPS = 4
# The aggregate was computed from float64 simulator metrics before all nine
# channels were independently frozen as float32.  Reconstructing it from the
# first eight frozen channels can therefore differ by several float32 ULPs.
F2_STATIC_AGGREGATE_SANITY_ATOL = 1.0e-6

HARD_FACTOR_NAMES = ("nc", "dac", "tlc", "ddc")

# This is the training-time sensor contract when future latent grounding is
# enabled.  The future frames are used only to define/audit the held-out
# manifest; they never enter online planning inference.
F2_SENSOR_CONTRACT: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("cam_b0", (3,)),
    ("cam_f0", (1, 2, 3, 10, 11)),
    ("cam_l0", (3,)),
    ("cam_l1", ()),
    ("cam_l2", ()),
    ("cam_r0", (3,)),
    ("cam_r1", ()),
    ("cam_r2", ()),
    ("lidar_pc", ()),
)

FORMAL_GATE_THRESHOLDS = MappingProxyType({
    "coverage_fraction": 1.0,
    "fatal_count": 0,
    "future_camera_valid_fraction_min": 0.99,
    "critic_delta_min": 0.005,
    "critic_pairwise_accuracy_min": 0.55,
    "proposal_headroom_closed_min": 0.25,
    "hard_factor_delta_min": -0.001,
    "critic_score_error_excess_max": 0.02,
})


class CausalValidationError(ValueError):
    """Raised when an immutable F2 validation contract is violated."""


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


def _validate_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CausalValidationError(f"{name} must be a 64-character SHA-256")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise CausalValidationError(f"{name} is not hexadecimal")
    return normalized


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CausalValidationError(f"{name} must be a non-empty string")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CausalValidationError(f"{name} must be numeric, not bool")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise CausalValidationError(f"{name} must be numeric") from error
    if not math.isfinite(converted):
        raise CausalValidationError(f"{name} must be finite")
    return converted


def _unit_float(value: Any, name: str) -> float:
    converted = _finite_float(value, name)
    if converted < 0.0 or converted > 1.0:
        raise CausalValidationError(f"{name} must lie in [0, 1]")
    return converted


def _canonical_float32(value: Any, name: str) -> float:
    """Round one finite scalar to the canonical IEEE-754 binary32 value."""

    converted = _finite_float(value, name)
    try:
        rounded = struct.unpack(">f", struct.pack(">f", converted))[0]
    except (OverflowError, struct.error) as error:
        raise CausalValidationError(f"{name} is outside the float32 range") from error
    if not math.isfinite(rounded):
        raise CausalValidationError(f"{name} is outside the float32 range")
    # Collapse signed zero so semantically identical inputs serialize equally.
    return 0.0 if rounded == 0.0 else rounded


def _ordered_float32_bits(value: Any, name: str) -> int:
    """Map one canonical binary32 to an integer ordered by numeric value."""

    canonical = _canonical_float32(value, name)
    bits = struct.unpack(">I", struct.pack(">f", canonical))[0]
    return 0xFFFFFFFF - bits if bits & 0x80000000 else bits + 0x80000000


def _float32_ulp_distance(left: Any, right: Any, name: str) -> int:
    return abs(
        _ordered_float32_bits(left, f"{name} left")
        - _ordered_float32_bits(right, f"{name} right")
    )


def _unit_float32(value: Any, name: str) -> float:
    converted = _canonical_float32(value, name)
    if converted < 0.0 or converted > 1.0:
        raise CausalValidationError(f"{name} must lie in [0, 1]")
    return converted


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalValidationError(f"{name} must be an integer")
    if value < 0:
        raise CausalValidationError(f"{name} must be nonnegative")
    return value


def _require_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise CausalValidationError(
            f"{name} is internally inconsistent: {actual!r} != {expected!r}"
        )


def _unique_strings(values: Iterable[Any], name: str) -> Tuple[str, ...]:
    converted = tuple(_nonempty_string(value, name) for value in values)
    duplicates = sorted(
        value for value, count in Counter(converted).items() if count > 1
    )
    if duplicates:
        raise CausalValidationError(
            f"{name} contains duplicate identifiers: {duplicates[:8]}"
        )
    return converted


def identifier_sha256(values: Iterable[Any]) -> str:
    """Hash sorted identifiers as UTF-8 lines with a final newline."""

    identifiers = sorted(str(value) for value in values)
    payload = (
        ("\n".join(identifiers) + "\n").encode("utf-8")
        if identifiers
        else b""
    )
    return _sha256(payload)


def normalize_sensor_contract(
    contract: Mapping[str, Sequence[int]]
    | Sequence[Tuple[str, Sequence[int]]],
) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    """Return a canonical sensor mapping and reject duplicate indices."""

    items = contract.items() if isinstance(contract, Mapping) else contract
    normalized: Dict[str, Tuple[int, ...]] = {}
    for raw_name, raw_indices in items:
        name = _nonempty_string(raw_name, "sensor name")
        if name in normalized:
            raise CausalValidationError(f"duplicate sensor field {name!r}")
        indices = []
        for index in raw_indices:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise CausalValidationError(
                    f"sensor {name!r} indices must be nonnegative integers"
                )
            indices.append(index)
        if len(indices) != len(set(indices)):
            raise CausalValidationError(f"sensor {name!r} contains duplicates")
        normalized[name] = tuple(indices)
    return tuple(sorted(normalized.items()))


@dataclass(frozen=True)
class ManifestRow:
    """One immutable held-out-navtrain token identity."""

    log_name: str
    time_us: int
    token: str
    future_camera_valid: bool

    def __post_init__(self) -> None:
        _nonempty_string(self.log_name, "log_name")
        _nonempty_string(self.token, "token")
        _nonnegative_int(self.time_us, "time_us")
        if not isinstance(self.future_camera_valid, bool):
            raise CausalValidationError("future_camera_valid must be a bool")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "future_camera_valid": self.future_camera_valid,
            "log_name": self.log_name,
            "time_us": self.time_us,
            "token": self.token,
        }


@dataclass(frozen=True)
class CanonicalManifest:
    """Canonical manifest after all split and coverage guards pass."""

    rows: Tuple[ManifestRow, ...]
    allowlist_logs: Tuple[str, ...]
    cache_metadata_sha256: str
    source_revision: str
    sensor_contract: Tuple[Tuple[str, Tuple[int, ...]], ...] = F2_SENSOR_CONTRACT
    schema_version: str = MANIFEST_SCHEMA_VERSION
    split: str = F2_SPLIT
    phase: str = F2_PHASE
    partition: str = F2_PARTITION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise CausalValidationError("unsupported F2 manifest schema")
        if (self.split, self.phase, self.partition) != (
            F2_SPLIT,
            F2_PHASE,
            F2_PARTITION,
        ):
            raise CausalValidationError(
                "F2 causal manifests are restricted to "
                "navtrain/development/validation"
            )
        object.__setattr__(
            self,
            "cache_metadata_sha256",
            _validate_sha256(
                self.cache_metadata_sha256, "cache_metadata_sha256"
            ),
        )
        _nonempty_string(self.source_revision, "source_revision")
        if normalize_sensor_contract(self.sensor_contract) != F2_SENSOR_CONTRACT:
            raise CausalValidationError("F2 training sensor contract drifted")

        canonical_logs = tuple(sorted(_unique_strings(self.allowlist_logs, "logs")))
        if canonical_logs != self.allowlist_logs:
            raise CausalValidationError("allowlist_logs must be sorted canonically")
        if not canonical_logs:
            raise CausalValidationError("allowlist_logs cannot be empty")

        rows = tuple(self.rows)
        if not rows:
            raise CausalValidationError("manifest cannot be empty")
        canonical_rows = tuple(
            sorted(rows, key=lambda row: (row.log_name, row.time_us, row.token))
        )
        if rows != canonical_rows:
            raise CausalValidationError("manifest rows are not canonically sorted")
        tokens = [row.token for row in rows]
        _unique_strings(tokens, "manifest tokens")
        identities = [(row.log_name, row.time_us, row.token) for row in rows]
        if len(identities) != len(set(identities)):
            raise CausalValidationError("manifest contains duplicate rows")
        manifest_logs = set(row.log_name for row in rows)
        unknown_logs = sorted(manifest_logs - set(canonical_logs))
        if unknown_logs:
            raise CausalValidationError(
                f"manifest rows are outside the held-out log allowlist: {unknown_logs[:8]}"
            )
        missing_logs = sorted(set(canonical_logs) - manifest_logs)
        if missing_logs:
            raise CausalValidationError(
                "held-out log allowlist is not fully represented in the manifest: "
                f"{missing_logs[:8]}"
            )

    @property
    def tokens(self) -> Tuple[str, ...]:
        return tuple(row.token for row in self.rows)

    @property
    def logs(self) -> Tuple[str, ...]:
        return tuple(sorted({row.log_name for row in self.rows}))

    def to_payload(self) -> Dict[str, Any]:
        return {
            "allowlist_logs": list(self.allowlist_logs),
            "cache_metadata_sha256": self.cache_metadata_sha256,
            "evidence": {
                "allowlist_log_count": len(self.allowlist_logs),
                "allowlist_log_sha256": identifier_sha256(self.allowlist_logs),
                "manifest_log_count": len(self.logs),
                "manifest_log_sha256": identifier_sha256(self.logs),
                "token_count": len(self.tokens),
                "token_sha256": identifier_sha256(self.tokens),
            },
            "partition": self.partition,
            "phase": self.phase,
            "rows": [row.to_payload() for row in self.rows],
            "schema_version": self.schema_version,
            "sensor_contract": {
                name: list(indices) for name, indices in self.sensor_contract
            },
            "source_revision": self.source_revision,
            "split": self.split,
        }


def _coverage_guard(
    requested_tokens: Sequence[str],
    available_tokens: Iterable[Any],
    name: str,
) -> None:
    available = set(_unique_strings(available_tokens, name))
    missing = sorted(set(requested_tokens) - available)
    if missing:
        raise CausalValidationError(
            f"{name} does not cover the complete manifest: {missing[:8]}"
        )


def build_canonical_manifest(
    rows: Iterable[ManifestRow],
    *,
    expected_tokens: Iterable[Any],
    sensor_available_tokens: Iterable[Any],
    cache_tokens: Iterable[Any],
    allowlist_logs: Iterable[Any],
    cache_metadata_sha256: str,
    source_revision: str,
    sensor_contract: Mapping[str, Sequence[int]]
    | Sequence[Tuple[str, Sequence[int]]] = F2_SENSOR_CONTRACT,
    split: str = F2_SPLIT,
    phase: str = F2_PHASE,
    partition: str = F2_PARTITION,
) -> CanonicalManifest:
    """Build a canonical manifest and fail on any missing or duplicate input."""

    if (split, phase, partition) != (F2_SPLIT, F2_PHASE, F2_PARTITION):
        raise CausalValidationError(
            "F2 causal manifests are restricted to "
            "navtrain/development/validation"
        )
    normalized_sensor_contract = normalize_sensor_contract(sensor_contract)
    if normalized_sensor_contract != F2_SENSOR_CONTRACT:
        raise CausalValidationError("F2 training sensor contract drifted")

    expected = _unique_strings(expected_tokens, "expected tokens")
    canonical_rows = tuple(
        sorted(tuple(rows), key=lambda row: (row.log_name, row.time_us, row.token))
    )
    row_tokens = tuple(row.token for row in canonical_rows)
    _unique_strings(row_tokens, "manifest tokens")
    missing_rows = sorted(set(expected) - set(row_tokens))
    extra_rows = sorted(set(row_tokens) - set(expected))
    if missing_rows or extra_rows:
        raise CausalValidationError(
            "manifest token set differs from the frozen expected token set: "
            f"missing={missing_rows[:8]}, extra={extra_rows[:8]}"
        )
    _coverage_guard(row_tokens, sensor_available_tokens, "sensor availability")
    _coverage_guard(row_tokens, cache_tokens, "metric cache")

    canonical_logs = tuple(sorted(_unique_strings(allowlist_logs, "logs")))
    return CanonicalManifest(
        rows=canonical_rows,
        allowlist_logs=canonical_logs,
        cache_metadata_sha256=_validate_sha256(
            cache_metadata_sha256, "cache_metadata_sha256"
        ),
        source_revision=_nonempty_string(source_revision, "source_revision"),
        sensor_contract=normalized_sensor_contract,
        split=split,
        phase=phase,
        partition=partition,
    )


def canonical_manifest_bytes(manifest: CanonicalManifest) -> bytes:
    return _canonical_json_bytes(manifest.to_payload())


def manifest_sha256(manifest: CanonicalManifest) -> str:
    return _sha256(canonical_manifest_bytes(manifest))


def manifest_from_bytes(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_tokens: Optional[Iterable[Any]] = None,
    sensor_available_tokens: Optional[Iterable[Any]] = None,
    cache_tokens: Optional[Iterable[Any]] = None,
) -> CanonicalManifest:
    """Parse, canonicalize, and optionally re-audit a frozen manifest."""

    expected_digest = _validate_sha256(expected_sha256, "expected_sha256")
    if _sha256(payload) != expected_digest:
        raise CausalValidationError("manifest SHA-256 mismatch")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CausalValidationError("manifest is not valid canonical JSON") from error
    required_keys = {
        "allowlist_logs",
        "cache_metadata_sha256",
        "evidence",
        "partition",
        "phase",
        "rows",
        "schema_version",
        "sensor_contract",
        "source_revision",
        "split",
    }
    if not isinstance(decoded, dict) or set(decoded) != required_keys:
        raise CausalValidationError("manifest keys do not match the frozen schema")
    try:
        rows = tuple(
            ManifestRow(
                log_name=row["log_name"],
                time_us=row["time_us"],
                token=row["token"],
                future_camera_valid=row["future_camera_valid"],
            )
            for row in decoded["rows"]
        )
        manifest = CanonicalManifest(
            rows=rows,
            allowlist_logs=tuple(decoded["allowlist_logs"]),
            cache_metadata_sha256=decoded["cache_metadata_sha256"],
            source_revision=decoded["source_revision"],
            sensor_contract=normalize_sensor_contract(decoded["sensor_contract"]),
            schema_version=decoded["schema_version"],
            split=decoded["split"],
            phase=decoded["phase"],
            partition=decoded["partition"],
        )
    except (KeyError, TypeError) as error:
        raise CausalValidationError("manifest rows are malformed") from error
    if decoded["evidence"] != manifest.to_payload()["evidence"]:
        raise CausalValidationError("manifest evidence does not match its rows")
    if canonical_manifest_bytes(manifest) != payload:
        raise CausalValidationError("manifest bytes are not canonical")

    if expected_tokens is not None:
        expected = set(_unique_strings(expected_tokens, "expected tokens"))
        actual = set(manifest.tokens)
        if expected != actual:
            raise CausalValidationError("frozen manifest token coverage drifted")
    if sensor_available_tokens is not None:
        _coverage_guard(
            manifest.tokens, sensor_available_tokens, "sensor availability"
        )
    if cache_tokens is not None:
        _coverage_guard(manifest.tokens, cache_tokens, "metric cache")
    return manifest


def rows_by_log(
    rows: Sequence[ManifestRow],
) -> Dict[str, Tuple[ManifestRow, ...]]:
    grouped: Dict[str, list[ManifestRow]] = {}
    for row in rows:
        grouped.setdefault(row.log_name, []).append(row)
    return {
        log_name: tuple(
            sorted(log_rows, key=lambda row: (row.time_us, row.token))
        )
        for log_name, log_rows in sorted(grouped.items())
    }


def deterministic_log_assignments(
    rows: Sequence[ManifestRow], num_shards: int
) -> Tuple[Tuple[str, ...], ...]:
    """Assign whole logs with deterministic longest-first load balancing."""

    if isinstance(num_shards, bool) or not isinstance(num_shards, int):
        raise CausalValidationError("num_shards must be an integer")
    if num_shards < 1:
        raise CausalValidationError("num_shards must be positive")
    counts = Counter(row.log_name for row in rows)
    if not counts:
        raise CausalValidationError("cannot shard an empty manifest")
    buckets: list[list[str]] = [[] for _ in range(num_shards)]
    loads = [0 for _ in range(num_shards)]
    for log_name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        shard = min(range(num_shards), key=lambda index: (loads[index], index))
        buckets[shard].append(log_name)
        loads[shard] += count
    return tuple(tuple(sorted(bucket)) for bucket in buckets)


def _float_tuple(values: Sequence[Any], name: str) -> Tuple[float, ...]:
    return tuple(_finite_float(value, f"{name}[{index}]") for index, value in enumerate(values))


def _float32_tuple(values: Sequence[Any], name: str) -> Tuple[float, ...]:
    return tuple(
        _canonical_float32(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    )


@dataclass(frozen=True)
class FrozenTokenArtifact:
    """Immutable per-token data produced by the external artifact builder.

    ``calibration_states`` must come from the Drive-JEPA ``simulation_v2``
    path used for current-candidate comfort. ``carry_states`` must come from
    the standard sequential PDM simulator and is the state propagated to the
    next token.  They are intentionally separate arrays.
    """

    token: str
    proposal_sha256: str
    base_logits: Tuple[float, ...]
    target_scores: Tuple[Tuple[float, ...], ...]
    calibration_states: Tuple[Any, ...]
    carry_states: Tuple[Any, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.token, "artifact token")
        object.__setattr__(
            self,
            "proposal_sha256",
            _validate_sha256(self.proposal_sha256, "proposal_sha256"),
        )
        base_logits = _float32_tuple(self.base_logits, "base_logits")
        if len(base_logits) != CANDIDATE_COUNT:
            raise CausalValidationError(
                f"base_logits must contain {CANDIDATE_COUNT} candidates"
            )
        object.__setattr__(self, "base_logits", base_logits)

        scores = tuple(
            tuple(
                _unit_float32(
                    value,
                    f"target_scores[{row_index}][{column_index}]",
                )
                for column_index, value in enumerate(row)
            )
            for row_index, row in enumerate(self.target_scores)
        )
        if len(scores) != CANDIDATE_COUNT or any(
            len(row) != TARGET_CHANNEL_COUNT for row in scores
        ):
            raise CausalValidationError(
                "target_scores must have shape [32, 9] in the frozen channel order"
            )
        object.__setattr__(self, "target_scores", scores)

        calibration_states = tuple(self.calibration_states)
        carry_states = tuple(self.carry_states)
        if len(calibration_states) != CANDIDATE_COUNT:
            raise CausalValidationError(
                "calibration_states must contain 32 simulation_v2 states"
            )
        if len(carry_states) != CANDIDATE_COUNT:
            raise CausalValidationError(
                "carry_states must contain 32 standard-simulator states"
            )
        object.__setattr__(self, "calibration_states", calibration_states)
        object.__setattr__(self, "carry_states", carry_states)


@dataclass(frozen=True)
class ModelTokenOutput:
    """Raw current-token output supplied by an external F2 model adapter.

    ``base_logits`` is emitted again for every checkpoint and must match the
    immutable artifact exactly after canonical float32 conversion.  The raw
    F2 ``residual_logits`` and the device-computed ``delta_logits`` are also
    required.  The torch adapter first checks their exact device-side identity
    before converting them to Python values.  This pure boundary then applies
    only a small-ULP cross-runtime sanity check to ``delta`` versus Python
    ``tanh`` and reconstructs utility from the transmitted float32 ``delta``.

    Keeping ``delta_logits`` is essential near cancellation: replacing the
    device delta with Python ``tanh(residual)`` before adding the base can move
    a mathematically valid zero utility by many zero-adjacent float32 ULPs.

    The required numeric-mode declaration is intentional: checking the
    identity within a handful of binary32 ULPs is invalid for a mixed-fp16
    forward.  The external runner must disable autocast for authoritative
    checkpoint evaluation rather than silently widening already-rounded fp16
    outputs.
    """

    token: str
    proposal_sha256: str
    base_logits: Tuple[float, ...]
    residual_logits: Tuple[float, ...]
    delta_logits: Tuple[float, ...]
    utility_logits: Tuple[float, ...]
    numeric_mode: str

    def __post_init__(self) -> None:
        _nonempty_string(self.token, "model output token")
        if self.numeric_mode != F2_MODEL_NUMERIC_MODE:
            raise CausalValidationError(
                "F2 causal model outputs require numeric_mode="
                f"{F2_MODEL_NUMERIC_MODE!r}; found {self.numeric_mode!r}"
            )
        object.__setattr__(
            self,
            "proposal_sha256",
            _validate_sha256(self.proposal_sha256, "proposal_sha256"),
        )
        base_logits = _float32_tuple(self.base_logits, "current base_logits")
        if len(base_logits) != CANDIDATE_COUNT:
            raise CausalValidationError(
                f"current base_logits must contain {CANDIDATE_COUNT} candidates"
            )
        object.__setattr__(self, "base_logits", base_logits)
        residuals = _float32_tuple(
            self.residual_logits, "current residual_logits"
        )
        if len(residuals) != CANDIDATE_COUNT:
            raise CausalValidationError(
                f"current residual_logits must contain {CANDIDATE_COUNT} candidates"
            )
        object.__setattr__(self, "residual_logits", residuals)
        deltas = _float32_tuple(self.delta_logits, "delta_logits")
        if len(deltas) != CANDIDATE_COUNT:
            raise CausalValidationError(
                f"delta_logits must contain {CANDIDATE_COUNT} candidates"
            )
        object.__setattr__(self, "delta_logits", deltas)
        logits = _float32_tuple(self.utility_logits, "utility_logits")
        if len(logits) != CANDIDATE_COUNT:
            raise CausalValidationError(
                f"utility_logits must contain {CANDIDATE_COUNT} candidates"
            )
        delta_sanity_mismatches = []
        utility_mismatches = []
        for index, (base, residual, delta, utility) in enumerate(
            zip(base_logits, residuals, deltas, logits)
        ):
            python_delta = _canonical_float32(
                math.tanh(residual), f"Python tanh residual[{index}]"
            )
            if (
                _float32_ulp_distance(
                    delta, python_delta, f"delta tanh sanity[{index}]"
                )
                > F2_DELTA_TANH_SANITY_MAX_ULPS
            ):
                delta_sanity_mismatches.append(index)
            expected_utility = _canonical_float32(
                base + delta, f"expected utility_logits[{index}]"
            )
            if utility != expected_utility:
                utility_mismatches.append(index)
        if delta_sanity_mismatches:
            raise CausalValidationError(
                "delta_logits are not a small-ULP float32 tanh sanity match for "
                "residual_logits; candidate_indices="
                f"{delta_sanity_mismatches[:8]}"
            )
        if utility_mismatches:
            raise CausalValidationError(
                "utility_logits do not exactly satisfy canonical float32 "
                "base_logits + transmitted delta_logits; candidate_indices="
                f"{utility_mismatches[:8]}"
            )
        object.__setattr__(self, "utility_logits", logits)


def is_continuous(previous_time_us: Optional[int], current_time_us: int) -> bool:
    if previous_time_us is None:
        return False
    delta_time_us = current_time_us - previous_time_us
    return 0 < delta_time_us <= MAX_CONTINUITY_DELTA_US


def stable_sigmoid(logit: float) -> float:
    value = _canonical_float32(logit, "logit")
    if value >= 0.0:
        inverse = math.exp(-value)
        probability = 1.0 / (1.0 + inverse)
    else:
        exponential = math.exp(value)
        probability = exponential / (1.0 + exponential)
    return _canonical_float32(probability, "sigmoid probability")


def stable_argmax(values: Sequence[Any]) -> int:
    """Return the lowest original candidate index among exact maxima."""

    converted = _float32_tuple(values, "argmax values")
    if not converted:
        raise CausalValidationError("cannot select from an empty sequence")
    best_index = 0
    best_value = converted[0]
    for index, value in enumerate(converted[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def micro_pair_counts(
    predicted_scores: Sequence[Any], target_scores: Sequence[Any]
) -> Tuple[int, int]:
    """Count correct/all unequal-target unordered pairs; score ties are wrong."""

    predictions = _float32_tuple(predicted_scores, "pairwise prediction")
    targets = tuple(
        _unit_float32(value, f"pairwise target[{index}]")
        for index, value in enumerate(target_scores)
    )
    if len(predictions) != len(targets):
        raise CausalValidationError("pairwise prediction and target lengths differ")
    correct = 0
    comparable = 0
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            target_delta = targets[left] - targets[right]
            if target_delta == 0.0:
                continue
            comparable += 1
            prediction_delta = predictions[left] - predictions[right]
            if prediction_delta * target_delta > 0.0:
                correct += 1
    return correct, comparable


def causal_true_epdms(
    target_scores: Sequence[Any],
    *,
    two_frame_comfort: Optional[Any],
) -> float:
    """Reconstruct official EPDMS truth from the frozen nine-channel row.

    The frozen aggregate channel intentionally omits two-frame extended
    comfort.  A discontinuous token therefore uses the original 14-weight
    formula, while a continuous branch supplies its own causal comfort value
    and uses the official 16-weight formula.
    """

    scores = tuple(
        _unit_float32(value, f"causal target[{index}]")
        for index, value in enumerate(target_scores)
    )
    if len(scores) != TARGET_CHANNEL_COUNT:
        raise CausalValidationError("causal target must contain nine channels")
    multiplier = math.prod(scores[:4])
    weighted = (
        5.0 * scores[4]
        + 5.0 * scores[5]
        + 2.0 * scores[6]
        + 2.0 * scores[7]
    )
    reconstructed_static = _canonical_float32(
        multiplier * weighted / 14.0,
        "reconstructed static aggregate",
    )
    if abs(scores[8] - reconstructed_static) > F2_STATIC_AGGREGATE_SANITY_ATOL:
        raise CausalValidationError(
            "stored aggregate channel is inconsistent with the official "
            "14-weight formula: "
            f"stored={scores[8]!r}, reconstructed={reconstructed_static!r}, "
            f"atol={F2_STATIC_AGGREGATE_SANITY_ATOL!r}"
        )
    if two_frame_comfort is None:
        value = reconstructed_static
    else:
        comfort = _unit_float32(two_frame_comfort, "two-frame comfort")
        value = multiplier * (weighted + 2.0 * comfort) / 16.0
    return _unit_float32(value, "causal true EPDMS")


def optimistic_oracle_epdms(
    target_scores: Sequence[Sequence[Any]], *, continuous: bool
) -> float:
    """Checkpoint-invariant per-token upper oracle for the frozen proposals."""

    if not isinstance(continuous, bool):
        raise CausalValidationError("oracle continuity flag must be a bool")
    if len(target_scores) != CANDIDATE_COUNT:
        raise CausalValidationError(
            f"oracle target_scores must contain {CANDIDATE_COUNT} candidates"
        )
    comfort = 1.0 if continuous else None
    return max(
        causal_true_epdms(row, two_frame_comfort=comfort)
        for row in target_scores
    )


@dataclass
class _BranchState:
    past_carry_state: Any = None
    last_time_us: Optional[int] = None


@dataclass(frozen=True)
class TokenCausalResult:
    log_name: str
    time_us: int
    token: str
    future_camera_valid: bool
    critic_continuous: bool
    base_continuous: bool
    critic_selected_index: int
    base_selected_index: int
    critic_selected_q: float
    base_selected_q: float
    critic_selected_probability: float
    base_selected_probability: float
    critic_selected_truth: float
    base_selected_truth: float
    critic_target_scores: Tuple[float, ...]
    base_target_scores: Tuple[float, ...]
    oracle_aggregate: float
    pairwise_correct: int
    pairwise_comparable: int

    def __post_init__(self) -> None:
        _nonempty_string(self.log_name, "result log_name")
        _nonempty_string(self.token, "result token")
        _nonnegative_int(self.time_us, "result time_us")
        for name in (
            "future_camera_valid",
            "critic_continuous",
            "base_continuous",
        ):
            if not isinstance(getattr(self, name), bool):
                raise CausalValidationError(f"{name} must be a bool")
        for name, index in (
            ("critic_selected_index", self.critic_selected_index),
            ("base_selected_index", self.base_selected_index),
        ):
            if isinstance(index, bool) or not isinstance(index, int):
                raise CausalValidationError(f"{name} must be an integer")
            if index < 0 or index >= CANDIDATE_COUNT:
                raise CausalValidationError(f"{name} is out of range")
        for name in (
            "critic_selected_q",
            "base_selected_q",
            "critic_selected_probability",
            "base_selected_probability",
            "critic_selected_truth",
            "base_selected_truth",
            "oracle_aggregate",
        ):
            object.__setattr__(
                self,
                name,
                _unit_float32(getattr(self, name), name),
            )
        for name, raw_scores in (
            ("critic_target_scores", self.critic_target_scores),
            ("base_target_scores", self.base_target_scores),
        ):
            scores = tuple(
                _unit_float32(value, f"{name}[{index}]")
                for index, value in enumerate(raw_scores)
            )
            if len(scores) != TARGET_CHANNEL_COUNT:
                raise CausalValidationError(f"{name} must contain nine channels")
            object.__setattr__(self, name, scores)
        _nonnegative_int(self.pairwise_correct, "pairwise_correct")
        _nonnegative_int(self.pairwise_comparable, "pairwise_comparable")
        if self.pairwise_correct > self.pairwise_comparable:
            raise CausalValidationError("pairwise correct count exceeds denominator")
        if self.oracle_aggregate + 1e-12 < max(
            self.critic_selected_truth, self.base_selected_truth
        ):
            raise CausalValidationError("oracle aggregate is below a selected target")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "base_continuous": self.base_continuous,
            "base_selected_index": self.base_selected_index,
            "base_selected_probability": self.base_selected_probability,
            "base_selected_q": self.base_selected_q,
            "base_selected_truth": self.base_selected_truth,
            "base_target_scores": list(self.base_target_scores),
            "critic_continuous": self.critic_continuous,
            "critic_selected_index": self.critic_selected_index,
            "critic_selected_probability": self.critic_selected_probability,
            "critic_selected_q": self.critic_selected_q,
            "critic_selected_truth": self.critic_selected_truth,
            "critic_target_scores": list(self.critic_target_scores),
            "future_camera_valid": self.future_camera_valid,
            "log_name": self.log_name,
            "oracle_aggregate": self.oracle_aggregate,
            "pairwise_comparable": self.pairwise_comparable,
            "pairwise_correct": self.pairwise_correct,
            "time_us": self.time_us,
            "token": self.token,
        }


ComfortFunction = Callable[[str, Any, Any, int], float]
StateCloneFunction = Callable[[Any], Any]


class DualBranchCausalStateMachine:
    """Evaluate one log with independent critic and frozen-base history."""

    def __init__(
        self,
        comfort_function: ComfortFunction,
        *,
        state_clone: StateCloneFunction = copy.deepcopy,
    ) -> None:
        if not callable(comfort_function):
            raise CausalValidationError("comfort_function must be callable")
        if not callable(state_clone):
            raise CausalValidationError("state_clone must be callable")
        self._comfort_function = comfort_function
        self._state_clone = state_clone

    def _branch_scores(
        self,
        branch_name: str,
        logits: Sequence[float],
        state: _BranchState,
        artifact: FrozenTokenArtifact,
        current_time_us: int,
    ) -> Tuple[
        Tuple[float, ...],
        Tuple[float, ...],
        Tuple[float, ...],
        bool,
    ]:
        probabilities = tuple(stable_sigmoid(logit) for logit in logits)
        continuous = is_continuous(state.last_time_us, current_time_us)
        if not continuous:
            state.past_carry_state = None
            truths = tuple(
                causal_true_epdms(row, two_frame_comfort=None)
                for row in artifact.target_scores
            )
            return probabilities, probabilities, truths, False
        if state.past_carry_state is None:
            raise CausalValidationError(
                f"{branch_name} has continuous timing but no prior carry state"
            )
        delta_time_us = current_time_us - int(state.last_time_us)
        comfort = tuple(
            _unit_float32(
                self._comfort_function(
                    branch_name,
                    # Every candidate is conceptually evaluated in parallel
                    # from the same branch snapshot.  Defensive copies make
                    # that true even if an injected simulator adapter mutates
                    # its arguments in place; mutation can otherwise leak
                    # across candidates or from critic into base through the
                    # shared frozen artifact.
                    self._state_clone(state.past_carry_state),
                    self._state_clone(candidate_state),
                    delta_time_us,
                ),
                f"{branch_name} comfort[{index}]",
            )
            for index, candidate_state in enumerate(artifact.calibration_states)
        )
        calibrated = tuple(
            _canonical_float32(
                (14.0 * probability + 2.0 * comfort_value) / 16.0,
                f"{branch_name} calibrated score[{index}]",
            )
            for index, (probability, comfort_value) in enumerate(
                zip(probabilities, comfort)
            )
        )
        truths = tuple(
            causal_true_epdms(row, two_frame_comfort=comfort_value)
            for row, comfort_value in zip(artifact.target_scores, comfort)
        )
        return calibrated, probabilities, truths, True

    def evaluate_log(
        self,
        rows: Sequence[ManifestRow],
        artifacts: Mapping[str, FrozenTokenArtifact],
        model_outputs: Mapping[str, ModelTokenOutput],
    ) -> Tuple[TokenCausalResult, ...]:
        if not rows:
            raise CausalValidationError("cannot evaluate an empty log")
        log_names = {row.log_name for row in rows}
        if len(log_names) != 1:
            raise CausalValidationError("evaluate_log received multiple logs")
        ordered_rows = tuple(sorted(rows, key=lambda row: (row.time_us, row.token)))
        if len({row.token for row in ordered_rows}) != len(ordered_rows):
            raise CausalValidationError("evaluate_log contains duplicate tokens")

        states = {"critic": _BranchState(), "base": _BranchState()}
        results = []
        for row in ordered_rows:
            if row.token not in artifacts:
                raise CausalValidationError(f"missing artifact for token {row.token}")
            if row.token not in model_outputs:
                raise CausalValidationError(
                    f"missing model output for token {row.token}"
                )
            artifact = artifacts[row.token]
            output = model_outputs[row.token]
            if artifact.token != row.token or output.token != row.token:
                raise CausalValidationError("token lookup returned mismatched data")
            if artifact.proposal_sha256 != output.proposal_sha256:
                raise CausalValidationError(
                    f"original proposal order drifted for token {row.token}"
                )
            if artifact.base_logits != output.base_logits:
                mismatched = [
                    index
                    for index, (frozen, current) in enumerate(
                        zip(artifact.base_logits, output.base_logits)
                    )
                    if frozen != current
                ]
                raise CausalValidationError(
                    "frozen Drive-JEPA base logits drifted for token "
                    f"{row.token}; candidate_indices={mismatched[:8]}"
                )

            (
                critic_q,
                critic_probability,
                critic_truth,
                critic_continuous,
            ) = self._branch_scores(
                "critic",
                output.utility_logits,
                states["critic"],
                artifact,
                row.time_us,
            )
            (
                base_q,
                base_probability,
                base_truth,
                base_continuous,
            ) = self._branch_scores(
                "base",
                artifact.base_logits,
                states["base"],
                artifact,
                row.time_us,
            )
            critic_index = stable_argmax(critic_q)
            base_index = stable_argmax(base_q)
            pairwise_correct, pairwise_comparable = micro_pair_counts(
                critic_q, critic_truth
            )
            results.append(
                TokenCausalResult(
                    log_name=row.log_name,
                    time_us=row.time_us,
                    token=row.token,
                    future_camera_valid=row.future_camera_valid,
                    critic_continuous=critic_continuous,
                    base_continuous=base_continuous,
                    critic_selected_index=critic_index,
                    base_selected_index=base_index,
                    critic_selected_q=critic_q[critic_index],
                    base_selected_q=base_q[base_index],
                    critic_selected_probability=critic_probability[critic_index],
                    base_selected_probability=base_probability[base_index],
                    critic_selected_truth=critic_truth[critic_index],
                    base_selected_truth=base_truth[base_index],
                    critic_target_scores=artifact.target_scores[critic_index],
                    base_target_scores=artifact.target_scores[base_index],
                    oracle_aggregate=optimistic_oracle_epdms(
                        artifact.target_scores,
                        continuous=critic_continuous,
                    ),
                    pairwise_correct=pairwise_correct,
                    pairwise_comparable=pairwise_comparable,
                )
            )

            # Clone twice even when both branches choose the same candidate.
            # This prevents a mutable state object from coupling the branches.
            states["critic"].past_carry_state = self._state_clone(
                artifact.carry_states[critic_index]
            )
            states["critic"].last_time_us = row.time_us
            states["base"].past_carry_state = self._state_clone(
                artifact.carry_states[base_index]
            )
            states["base"].last_time_us = row.time_us
        return tuple(results)


def merge_shard_results(
    shard_results: Iterable[Sequence[TokenCausalResult]],
    expected_tokens: Iterable[Any],
) -> Tuple[TokenCausalResult, ...]:
    expected = _unique_strings(expected_tokens, "expected result tokens")
    merged = tuple(result for shard in shard_results for result in shard)
    result_tokens = tuple(result.token for result in merged)
    _unique_strings(result_tokens, "result tokens")
    missing = sorted(set(expected) - set(result_tokens))
    extra = sorted(set(result_tokens) - set(expected))
    if missing or extra:
        raise CausalValidationError(
            f"shard result coverage mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )
    return tuple(sorted(merged, key=lambda row: (row.log_name, row.time_us, row.token)))


def evaluate_manifest_sharded(
    manifest: CanonicalManifest,
    artifacts: Mapping[str, FrozenTokenArtifact],
    model_outputs: Mapping[str, ModelTokenOutput],
    comfort_function: ComfortFunction,
    *,
    num_shards: int,
    state_clone: StateCloneFunction = copy.deepcopy,
) -> Tuple[TokenCausalResult, ...]:
    """Reference orchestration used to prove shard-count invariance.

    This function evaluates shards serially in-process.  A real multi-GPU
    runner should use ``deterministic_log_assignments`` and merge its worker
    outputs with ``merge_shard_results``; model loading/forward remains an
    explicit external TODO.
    """

    expected = set(manifest.tokens)
    artifact_tokens = set(artifacts)
    output_tokens = set(model_outputs)
    if artifact_tokens != expected:
        raise CausalValidationError("artifact token coverage is not exact")
    if output_tokens != expected:
        raise CausalValidationError("model-output token coverage is not exact")

    grouped = rows_by_log(manifest.rows)
    assignments = deterministic_log_assignments(manifest.rows, num_shards)
    shard_outputs = []
    for logs in assignments:
        machine = DualBranchCausalStateMachine(
            comfort_function, state_clone=state_clone
        )
        shard_rows = []
        for log_name in logs:
            shard_rows.extend(
                machine.evaluate_log(grouped[log_name], artifacts, model_outputs)
            )
        shard_outputs.append(tuple(shard_rows))
    return merge_shard_results(shard_outputs, manifest.tokens)


def canonical_evaluation_bytes(results: Sequence[TokenCausalResult]) -> bytes:
    ordered = sorted(results, key=lambda row: (row.log_name, row.time_us, row.token))
    return _canonical_json_bytes([row.to_payload() for row in ordered])


@dataclass(frozen=True)
class AggregateMetrics:
    expected_token_count: int
    observed_token_count: int
    coverage_fraction: float
    fatal_count: int
    future_camera_valid_fraction: float
    critic_selected_score: float
    base_selected_score: float
    oracle_score: float
    critic_delta: float
    critic_oracle_regret: float
    critic_pairwise_accuracy: Optional[float]
    critic_pairwise_correct: int
    critic_pairwise_comparable: int
    critic_score_error: float
    base_score_error: float
    critic_score_error_excess: float
    proposal_headroom_closed: Optional[float]
    hard_factor_deltas: Tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _nonnegative_int(self.expected_token_count, "expected_token_count")
        _nonnegative_int(self.observed_token_count, "observed_token_count")
        _nonnegative_int(self.fatal_count, "fatal_count")
        _nonnegative_int(self.critic_pairwise_correct, "critic_pairwise_correct")
        _nonnegative_int(
            self.critic_pairwise_comparable, "critic_pairwise_comparable"
        )
        if self.expected_token_count < 1:
            raise CausalValidationError("expected_token_count must be positive")
        if self.observed_token_count < 1:
            raise CausalValidationError("observed_token_count must be positive")
        if self.observed_token_count > self.expected_token_count:
            raise CausalValidationError("observed token count exceeds manifest")
        for name in (
            "coverage_fraction",
            "future_camera_valid_fraction",
            "critic_selected_score",
            "base_selected_score",
            "oracle_score",
            "critic_score_error",
            "base_score_error",
        ):
            _unit_float(getattr(self, name), name)
        for name in (
            "critic_delta",
            "critic_oracle_regret",
            "critic_score_error_excess",
        ):
            _finite_float(getattr(self, name), name)
        if self.critic_pairwise_accuracy is not None:
            _unit_float(self.critic_pairwise_accuracy, "critic_pairwise_accuracy")
        if self.proposal_headroom_closed is not None:
            _finite_float(
                self.proposal_headroom_closed, "proposal_headroom_closed"
            )
        hard_factor_deltas = tuple(
            _finite_float(value, f"hard_factor_deltas[{index}]")
            for index, value in enumerate(self.hard_factor_deltas)
        )
        if len(hard_factor_deltas) != 4:
            raise CausalValidationError("hard_factor_deltas must contain four values")
        object.__setattr__(self, "hard_factor_deltas", hard_factor_deltas)

        _require_close(
            self.coverage_fraction,
            self.observed_token_count / self.expected_token_count,
            "coverage_fraction",
        )
        _require_close(
            self.critic_delta,
            self.critic_selected_score - self.base_selected_score,
            "critic_delta",
        )
        _require_close(
            self.critic_oracle_regret,
            self.oracle_score - self.critic_selected_score,
            "critic_oracle_regret",
        )
        _require_close(
            self.critic_score_error_excess,
            self.critic_score_error - self.base_score_error,
            "critic_score_error_excess",
        )
        if self.oracle_score + 1e-12 < max(
            self.critic_selected_score, self.base_selected_score
        ):
            raise CausalValidationError("oracle score is below a selected score")
        if self.critic_pairwise_correct > self.critic_pairwise_comparable:
            raise CausalValidationError("pairwise correct count exceeds denominator")
        if self.critic_pairwise_comparable == 0:
            if self.critic_pairwise_correct != 0:
                raise CausalValidationError("zero pairwise denominator has matches")
            if self.critic_pairwise_accuracy is not None:
                raise CausalValidationError(
                    "pairwise accuracy must be None without comparable pairs"
                )
        else:
            if self.critic_pairwise_accuracy is None:
                raise CausalValidationError(
                    "pairwise accuracy is missing despite comparable pairs"
                )
            _require_close(
                self.critic_pairwise_accuracy,
                self.critic_pairwise_correct / self.critic_pairwise_comparable,
                "critic_pairwise_accuracy",
            )

        headroom_denominator = self.oracle_score - self.base_selected_score
        if headroom_denominator > 0.0:
            if self.proposal_headroom_closed is None:
                raise CausalValidationError(
                    "proposal headroom is missing despite a positive denominator"
                )
            _require_close(
                self.proposal_headroom_closed,
                self.critic_delta / headroom_denominator,
                "proposal_headroom_closed",
            )
        elif self.proposal_headroom_closed is not None:
            raise CausalValidationError(
                "proposal headroom must be None without a positive denominator"
            )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "base_score_error": self.base_score_error,
            "base_selected_score": self.base_selected_score,
            "coverage_fraction": self.coverage_fraction,
            "critic_delta": self.critic_delta,
            "critic_oracle_regret": self.critic_oracle_regret,
            "critic_pairwise_accuracy": self.critic_pairwise_accuracy,
            "critic_pairwise_comparable": self.critic_pairwise_comparable,
            "critic_pairwise_correct": self.critic_pairwise_correct,
            "critic_score_error": self.critic_score_error,
            "critic_score_error_excess": self.critic_score_error_excess,
            "critic_selected_score": self.critic_selected_score,
            "expected_token_count": self.expected_token_count,
            "fatal_count": self.fatal_count,
            "future_camera_valid_fraction": self.future_camera_valid_fraction,
            "hard_factor_deltas": dict(zip(HARD_FACTOR_NAMES, self.hard_factor_deltas)),
            "observed_token_count": self.observed_token_count,
            "oracle_score": self.oracle_score,
            "proposal_headroom_closed": self.proposal_headroom_closed,
        }


def aggregate_results(
    results: Sequence[TokenCausalResult],
    *,
    expected_tokens: Iterable[Any],
    fatal_count: int,
) -> AggregateMetrics:
    """Aggregate metrics with an explicit externally audited fatal count."""

    expected = _unique_strings(expected_tokens, "expected aggregate tokens")
    ordered = tuple(
        sorted(results, key=lambda row: (row.log_name, row.time_us, row.token))
    )
    if not ordered:
        raise CausalValidationError("cannot aggregate zero observed tokens")
    observed_tokens = tuple(row.token for row in ordered)
    _unique_strings(observed_tokens, "observed aggregate tokens")
    unknown = sorted(set(observed_tokens) - set(expected))
    if unknown:
        raise CausalValidationError(f"aggregate contains unknown tokens: {unknown[:8]}")
    if isinstance(fatal_count, bool) or not isinstance(fatal_count, int):
        raise CausalValidationError("fatal_count must be an integer")
    if fatal_count < 0:
        raise CausalValidationError("fatal_count must be nonnegative")

    count = len(ordered)
    critic_scores = [row.critic_selected_truth for row in ordered]
    base_scores = [row.base_selected_truth for row in ordered]
    oracle_scores = [row.oracle_aggregate for row in ordered]
    critic_score = math.fsum(critic_scores) / count
    base_score = math.fsum(base_scores) / count
    oracle_score = math.fsum(oracle_scores) / count
    critic_error = (
        math.fsum(
            abs(row.critic_selected_q - row.critic_selected_truth)
            for row in ordered
        )
        / count
    )
    base_error = (
        math.fsum(
            abs(row.base_selected_q - row.base_selected_truth)
            for row in ordered
        )
        / count
    )
    pairwise_correct = sum(row.pairwise_correct for row in ordered)
    pairwise_comparable = sum(row.pairwise_comparable for row in ordered)
    pairwise_accuracy = (
        pairwise_correct / pairwise_comparable
        if pairwise_comparable > 0
        else None
    )
    headroom_denominator = oracle_score - base_score
    headroom = (
        (critic_score - base_score) / headroom_denominator
        if headroom_denominator > 0.0
        else None
    )
    hard_deltas = tuple(
        math.fsum(
            row.critic_target_scores[index] - row.base_target_scores[index]
            for row in ordered
        )
        / count
        for index in range(4)
    )
    return AggregateMetrics(
        expected_token_count=len(expected),
        observed_token_count=count,
        coverage_fraction=count / len(expected),
        fatal_count=fatal_count,
        future_camera_valid_fraction=(
            sum(int(row.future_camera_valid) for row in ordered) / count
        ),
        critic_selected_score=critic_score,
        base_selected_score=base_score,
        oracle_score=oracle_score,
        critic_delta=critic_score - base_score,
        critic_oracle_regret=oracle_score - critic_score,
        critic_pairwise_accuracy=pairwise_accuracy,
        critic_pairwise_correct=pairwise_correct,
        critic_pairwise_comparable=pairwise_comparable,
        critic_score_error=critic_error,
        base_score_error=base_error,
        critic_score_error_excess=critic_error - base_error,
        proposal_headroom_closed=headroom,
        hard_factor_deltas=hard_deltas,  # type: ignore[arg-type]
    )


def canonical_metrics_bytes(metrics: AggregateMetrics) -> bytes:
    return _canonical_json_bytes(metrics.to_payload())


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    failures: Tuple[str, ...]

    def to_payload(self) -> Dict[str, Any]:
        return {"failures": list(self.failures), "passed": self.passed}


def evaluate_formal_gates(metrics: AggregateMetrics) -> GateDecision:
    """Apply exactly the frozen F2 acceptance gates, with inclusive bounds."""

    failures = []
    thresholds = FORMAL_GATE_THRESHOLDS
    if metrics.coverage_fraction != thresholds["coverage_fraction"]:
        failures.append("coverage_fraction")
    if metrics.fatal_count != thresholds["fatal_count"]:
        failures.append("fatal_count")
    if (
        metrics.future_camera_valid_fraction
        < thresholds["future_camera_valid_fraction_min"]
    ):
        failures.append("future_camera_valid_fraction")
    if metrics.critic_delta < thresholds["critic_delta_min"]:
        failures.append("critic_delta")
    if (
        metrics.critic_pairwise_accuracy is None
        or metrics.critic_pairwise_accuracy
        < thresholds["critic_pairwise_accuracy_min"]
    ):
        failures.append("critic_pairwise_accuracy")
    if (
        metrics.proposal_headroom_closed is None
        or metrics.proposal_headroom_closed
        < thresholds["proposal_headroom_closed_min"]
    ):
        failures.append("proposal_headroom_closed")
    for name, value in zip(HARD_FACTOR_NAMES, metrics.hard_factor_deltas):
        if value < thresholds["hard_factor_delta_min"]:
            failures.append(f"hard_factor_delta_{name}")
    if (
        metrics.critic_score_error_excess
        > thresholds["critic_score_error_excess_max"]
    ):
        failures.append("critic_score_error_excess")
    return GateDecision(passed=not failures, failures=tuple(failures))


@dataclass(frozen=True)
class CheckpointEvaluation:
    """One fully materialized causal validation summary for an epoch."""

    epoch: int
    checkpoint_sha256: str
    manifest_sha256: str
    artifact_sha256: str
    numeric_mode: str
    metrics: AggregateMetrics

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int):
            raise CausalValidationError("epoch must be an integer")
        if self.epoch < 0:
            raise CausalValidationError("epoch must be nonnegative")
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _validate_sha256(self.checkpoint_sha256, "checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            _validate_sha256(self.manifest_sha256, "manifest_sha256"),
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _validate_sha256(self.artifact_sha256, "artifact_sha256"),
        )
        if self.numeric_mode != F2_MODEL_NUMERIC_MODE:
            raise CausalValidationError(
                "checkpoint evaluation requires numeric_mode="
                f"{F2_MODEL_NUMERIC_MODE!r}"
            )
        if not isinstance(self.metrics, AggregateMetrics):
            raise CausalValidationError("metrics must be an AggregateMetrics record")


@dataclass(frozen=True)
class CheckpointRanking:
    """Eligible checkpoints in monitor order and audited ineligible records."""

    eligible: Tuple[CheckpointEvaluation, ...]
    ineligible: Tuple[Tuple[CheckpointEvaluation, GateDecision], ...]

    @property
    def selected(self) -> Optional[CheckpointEvaluation]:
        return self.eligible[0] if self.eligible else None


def rank_checkpoints(
    evaluations: Sequence[CheckpointEvaluation],
) -> CheckpointRanking:
    """Filter by all gates, then maximize causal selected score.

    Exact score ties resolve to the earlier epoch and then checkpoint digest.
    All records must reference the same immutable manifest and artifact.
    """

    epochs = [evaluation.epoch for evaluation in evaluations]
    _unique_strings((str(epoch) for epoch in epochs), "checkpoint epochs")
    actual_epochs = tuple(sorted(epochs))
    if actual_epochs != F2_CHECKPOINT_EPOCHS:
        missing = sorted(set(F2_CHECKPOINT_EPOCHS) - set(actual_epochs))
        extra = sorted(set(actual_epochs) - set(F2_CHECKPOINT_EPOCHS))
        raise CausalValidationError(
            "F2 checkpoint epoch coverage must be exactly 0,1,2,3: "
            f"missing={missing}, extra={extra}"
        )
    checkpoint_hashes = [evaluation.checkpoint_sha256 for evaluation in evaluations]
    _unique_strings(checkpoint_hashes, "checkpoint hashes")
    manifest_hashes = {evaluation.manifest_sha256 for evaluation in evaluations}
    artifact_hashes = {evaluation.artifact_sha256 for evaluation in evaluations}
    if len(manifest_hashes) != 1:
        raise CausalValidationError("checkpoint summaries use different manifests")
    if len(artifact_hashes) != 1:
        raise CausalValidationError("checkpoint summaries use different artifacts")

    reference_metrics = evaluations[0].metrics
    frozen_base_score = _canonical_float32(
        reference_metrics.base_selected_score,
        "frozen base selected score",
    )
    frozen_oracle_score = _canonical_float32(
        reference_metrics.oracle_score,
        "frozen oracle score",
    )
    frozen_base_error = _canonical_float32(
        reference_metrics.base_score_error,
        "frozen base score error",
    )
    frozen_future_valid = _canonical_float32(
        reference_metrics.future_camera_valid_fraction,
        "frozen future-camera validity",
    )
    for evaluation in evaluations[1:]:
        metrics = evaluation.metrics
        if metrics.expected_token_count != reference_metrics.expected_token_count:
            raise CausalValidationError(
                "frozen manifest token count drifted across F2 checkpoints"
            )
        current_base_score = _canonical_float32(
            metrics.base_selected_score,
            f"epoch {evaluation.epoch} base selected score",
        )
        current_oracle_score = _canonical_float32(
            metrics.oracle_score,
            f"epoch {evaluation.epoch} oracle score",
        )
        current_base_error = _canonical_float32(
            metrics.base_score_error,
            f"epoch {evaluation.epoch} base score error",
        )
        current_future_valid = _canonical_float32(
            metrics.future_camera_valid_fraction,
            f"epoch {evaluation.epoch} future-camera validity",
        )
        if current_base_score != frozen_base_score:
            raise CausalValidationError(
                "frozen base selected score drifted across F2 checkpoints"
            )
        if current_oracle_score != frozen_oracle_score:
            raise CausalValidationError(
                "static oracle score drifted across F2 checkpoints"
            )
        if current_base_error != frozen_base_error:
            raise CausalValidationError(
                "frozen base score error drifted across F2 checkpoints"
            )
        if current_future_valid != frozen_future_valid:
            raise CausalValidationError(
                "static future-camera validity drifted across F2 checkpoints"
            )

    eligible = []
    ineligible = []
    for evaluation in evaluations:
        decision = evaluate_formal_gates(evaluation.metrics)
        if decision.passed:
            eligible.append(evaluation)
        else:
            ineligible.append((evaluation, decision))
    eligible.sort(
        key=lambda evaluation: (
            -evaluation.metrics.critic_selected_score,
            evaluation.epoch,
            evaluation.checkpoint_sha256,
        )
    )
    ineligible.sort(key=lambda item: (item[0].epoch, item[0].checkpoint_sha256))
    return CheckpointRanking(
        eligible=tuple(eligible), ineligible=tuple(ineligible)
    )
