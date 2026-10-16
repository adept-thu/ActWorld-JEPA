"""Train-only upper-bound audit for the bounded F2 residual selector.

This module is deliberately pure Python: it does not import torch, NAVSIM
runtime objects, checkpoints, or files.  It answers a narrow question before
an F2 launch: with frozen Drive-JEPA base logits ``b`` and independent
``tanh`` residuals, which of the original 32 proposals can ever be selected
by the static training utility, before evaluation-only two-frame-comfort
calibration?

Although real-valued ``tanh`` has an open range, the deployed F2 numerical
contract is IEEE binary32 (and mixed precision during training).  Finite large
residuals therefore round to exact ``+1`` or ``-1``.  Reachability uses the
actual closed binary32 endpoint controls, rounds ``b + delta`` and the
deployed sigmoid score back to binary32, then applies the
lowest-original-index tie rule.  Candidate ``i`` is reachable exactly when
setting its delta to ``+1`` and every other delta to ``-1`` makes the
score-domain stable argmax return ``i``.  Consequently, an endpoint tie is
reachable only when every tied competitor has a higher index.  This
distinction is recorded explicitly in every candidate result.

The data audit accepts only the authenticated ``navtrain/train`` partition,
uses a frozen deterministic 2-per-log-then-global sample, validates each
32-by-9 static target through the causal validator's official 14-weight EPDMS
reconstruction, and reports analytical upper bounds.  These bounds are not a
checkpoint result and make no claim about held-out, navtest, or navhard
performance.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from navsim.planning.training.f2_qualification_manifest import (
    QualificationManifest,
    QualificationRow,
    qualification_manifest_sha256,
)
from navsim.planning.validation.f2_causal_validator import (
    CANDIDATE_COUNT,
    HARD_FACTOR_NAMES,
    TARGET_CHANNEL_COUNT,
    CausalValidationError,
    causal_true_epdms,
    stable_argmax,
    stable_sigmoid,
)


FEASIBILITY_SCHEMA_VERSION = "actworld-f2-residual-feasibility-v3"
FEASIBILITY_DATASET = "navtrain"
FEASIBILITY_PARTITION = "train"
FEASIBILITY_PROHIBITED_PARTITIONS = ("held_out", "navtest", "navhard")
FEASIBILITY_SAMPLE_SIZE = 2048
FEASIBILITY_PER_LOG_SEED_COUNT = 2
FEASIBILITY_BOOTSTRAP_SEED = 20260824
FEASIBILITY_BOOTSTRAP_REPLICATES = 2000
FEASIBILITY_BOOTSTRAP_CONFIDENCE = 0.95
TANH_DELTA_FLOAT32_RANGE = (-1.0, 1.0)
STABLE_TIE_RULE = "lowest_original_candidate_index"


class F2ResidualFeasibilityError(ValueError):
    """Raised when the train-only feasibility contract cannot be proven."""


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
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
    except (TypeError, ValueError) as error:
        raise F2ResidualFeasibilityError(
            "feasibility payload is not finite canonical JSON"
        ) from error


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise F2ResidualFeasibilityError(f"{name} must be a non-empty string")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise F2ResidualFeasibilityError(f"{name} must be numeric, not bool")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise F2ResidualFeasibilityError(f"{name} must be numeric") from error
    if not math.isfinite(converted):
        raise F2ResidualFeasibilityError(f"{name} must be finite")
    return converted


def _float32(value: Any, name: str) -> float:
    converted = _finite_float(value, name)
    try:
        rounded = struct.unpack(">f", struct.pack(">f", converted))[0]
    except (OverflowError, struct.error) as error:
        raise F2ResidualFeasibilityError(
            f"{name} is outside the float32 range"
        ) from error
    if not math.isfinite(rounded):
        raise F2ResidualFeasibilityError(
            f"{name} is outside the float32 range"
        )
    return 0.0 if rounded == 0.0 else rounded


def _unit_float32(value: Any, name: str) -> float:
    converted = _float32(value, name)
    if converted < 0.0 or converted > 1.0:
        raise F2ResidualFeasibilityError(f"{name} must lie in [0, 1]")
    return converted


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise F2ResidualFeasibilityError(f"{name} must be a nonnegative integer")
    return value


def _mean(values: Iterable[float], name: str) -> float:
    sequence = tuple(_finite_float(value, name) for value in values)
    if not sequence:
        raise F2ResidualFeasibilityError(f"cannot average empty {name}")
    result = math.fsum(sequence) / len(sequence)
    return _finite_float(result, name)


def _normalized_base_logits(values: Sequence[Any]) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise F2ResidualFeasibilityError("base_logits must be a numeric sequence")
    logits = tuple(
        _float32(value, f"base_logits[{index}]")
        for index, value in enumerate(values)
    )
    if len(logits) != CANDIDATE_COUNT:
        raise F2ResidualFeasibilityError(
            f"base_logits must contain exactly {CANDIDATE_COUNT} candidates"
        )
    return logits


def _normalized_static_targets(
    values: Sequence[Sequence[Any]],
) -> Tuple[Tuple[float, ...], ...]:
    if isinstance(values, (str, bytes)):
        raise F2ResidualFeasibilityError(
            "target_scores must be a [32, 9] numeric sequence"
        )
    rows = []
    for row_index, raw_row in enumerate(values):
        if isinstance(raw_row, (str, bytes)):
            raise F2ResidualFeasibilityError(
                f"target_scores[{row_index}] must be a numeric sequence"
            )
        row = tuple(
            _unit_float32(
                value,
                f"target_scores[{row_index}][{column_index}]",
            )
            for column_index, value in enumerate(raw_row)
        )
        if len(row) != TARGET_CHANNEL_COUNT:
            raise F2ResidualFeasibilityError(
                f"target_scores[{row_index}] must contain nine channels"
            )
        try:
            # This is the shared source of truth for M and the frozen aggregate
            # channel.  Comfort is deliberately absent in this static audit.
            causal_true_epdms(row, two_frame_comfort=None)
        except CausalValidationError as error:
            raise F2ResidualFeasibilityError(
                f"target_scores[{row_index}] failed official static EPDMS "
                f"validation: {error}"
            ) from error
        rows.append(row)
    if len(rows) != CANDIDATE_COUNT:
        raise F2ResidualFeasibilityError(
            f"target_scores must contain exactly {CANDIDATE_COUNT} candidates"
        )
    return tuple(rows)


@dataclass(frozen=True)
class CandidateReachability:
    """Exact binary32 endpoint selection envelope for one proposal."""

    candidate_index: int
    base_logit: float
    strongest_competitor_index: int
    strongest_competitor_base_logit: float
    endpoint_selection_margin: float
    reachable: bool
    boundary_kind: str
    endpoint_controls_attainable: bool = True

    def to_payload(self) -> Dict[str, Any]:
        return {
            "base_logit": self.base_logit,
            "boundary_kind": self.boundary_kind,
            "candidate_index": self.candidate_index,
            "reachable": self.reachable,
            "endpoint_controls_attainable": self.endpoint_controls_attainable,
            "endpoint_selection_margin": self.endpoint_selection_margin,
            "strongest_competitor_base_logit": (
                self.strongest_competitor_base_logit
            ),
            "strongest_competitor_index": self.strongest_competitor_index,
        }


@dataclass(frozen=True)
class ResidualReachabilityEnvelope:
    base_logits: Tuple[float, ...]
    base_selected_index: int
    candidates: Tuple[CandidateReachability, ...]
    reachable_candidate_indices: Tuple[int, ...]
    delta_float32_range: Tuple[float, float] = TANH_DELTA_FLOAT32_RANGE
    stable_tie_rule: str = STABLE_TIE_RULE

    def to_payload(self) -> Dict[str, Any]:
        return {
            "base_logits": list(self.base_logits),
            "base_selected_index": self.base_selected_index,
            "candidate_reachability": [
                candidate.to_payload() for candidate in self.candidates
            ],
            "delta_float32_range": list(self.delta_float32_range),
            "reachable_candidate_indices": list(self.reachable_candidate_indices),
            "stable_tie_rule": self.stable_tie_rule,
        }


def analyze_tanh_residual_reachability(
    base_logits: Sequence[Any],
) -> ResidualReachabilityEnvelope:
    """Return exact reachability under deployed binary32 endpoint controls."""

    logits = _normalized_base_logits(base_logits)
    candidates = []
    reachable = []
    for candidate_index, base_logit in enumerate(logits):
        competitor_indices = tuple(
            index for index in range(CANDIDATE_COUNT) if index != candidate_index
        )
        endpoint_utilities = tuple(
            _float32(
                logit + (1.0 if index == candidate_index else -1.0),
                f"candidate[{candidate_index}] endpoint utility[{index}]",
            )
            for index, logit in enumerate(logits)
        )
        endpoint_scores = tuple(
            stable_sigmoid(utility) for utility in endpoint_utilities
        )
        strongest_competitor_index = competitor_indices[0]
        for index in competitor_indices[1:]:
            if endpoint_scores[index] > endpoint_scores[
                strongest_competitor_index
            ]:
                strongest_competitor_index = index
        strongest_competitor_base = logits[strongest_competitor_index]
        endpoint_margin = _float32(
            endpoint_scores[candidate_index]
            - endpoint_scores[strongest_competitor_index],
            f"candidate[{candidate_index}] endpoint selection margin",
        )
        selected_at_endpoints = stable_argmax(endpoint_scores)
        is_reachable = selected_at_endpoints == candidate_index
        if is_reachable and endpoint_margin > 0.0:
            boundary_kind = "strictly_reachable_endpoint_margin"
        elif is_reachable:
            boundary_kind = "reachable_endpoint_tie_lower_index"
        elif endpoint_margin == 0.0:
            boundary_kind = "unreachable_endpoint_tie_higher_index"
        else:
            boundary_kind = "unreachable_dominated"
        if is_reachable:
            reachable.append(candidate_index)
        candidates.append(
            CandidateReachability(
                candidate_index=candidate_index,
                base_logit=base_logit,
                strongest_competitor_index=strongest_competitor_index,
                strongest_competitor_base_logit=strongest_competitor_base,
                endpoint_selection_margin=endpoint_margin,
                reachable=is_reachable,
                boundary_kind=boundary_kind,
            )
        )
    if not reachable:
        # The stable base maximum is always reachable with all residuals equal;
        # reaching this branch indicates an arithmetic or implementation drift.
        raise F2ResidualFeasibilityError(
            "no candidate is reachable under the bounded residual contract"
        )
    return ResidualReachabilityEnvelope(
        base_logits=logits,
        base_selected_index=stable_argmax(
            tuple(stable_sigmoid(logit) for logit in logits)
        ),
        candidates=tuple(candidates),
        reachable_candidate_indices=tuple(reachable),
    )


def _row_sampling_sha256(row: QualificationRow) -> str:
    """Hash the qualified token identity used by both sample stages."""

    return _sha256(
        _canonical_json_bytes(
            {
                "domain": "actworld-f2-feasibility-sampling-row-v1",
                "future_camera_valid": row.future_camera_valid,
                "log_name": row.log_name,
                "time_us": row.time_us,
                "token": row.token,
            }
        )
    )


@dataclass(frozen=True)
class TrainSampleRow:
    log_name: str
    time_us: int
    token: str
    future_camera_valid: bool
    sampling_sha256: str
    selection_stage: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "future_camera_valid": self.future_camera_valid,
            "log_name": self.log_name,
            "sampling_sha256": self.sampling_sha256,
            "selection_stage": self.selection_stage,
            "time_us": self.time_us,
            "token": self.token,
        }


def select_deterministic_navtrain_train_sample(
    qualification: QualificationManifest,
) -> Tuple[TrainSampleRow, ...]:
    """Select the frozen train-only sample; never inspect held-out rows.

    Within each retained train log, rows are ordered by a SHA-256 of their
    canonical identity and the first two are included.  Remaining rows are
    globally SHA-ordered and used to reach 2048.  If fewer than 2048 retained
    train rows exist, every retained train row is selected.
    """

    if type(qualification) is not QualificationManifest:
        raise F2ResidualFeasibilityError(
            "sampling requires an authenticated QualificationManifest"
        )
    if qualification.dataset != FEASIBILITY_DATASET:
        raise F2ResidualFeasibilityError(
            "feasibility sampling is restricted to navtrain"
        )
    if (
        qualification.train.name != FEASIBILITY_PARTITION
        or qualification.held_out.name != "held_out"
    ):
        raise F2ResidualFeasibilityError("qualification partition names drifted")

    train_rows = tuple(qualification.train.rows)
    held_out_tokens = set(qualification.held_out.tokens)
    train_tokens = tuple(row.token for row in train_rows)
    duplicate_tokens = sorted(
        token for token, count in Counter(train_tokens).items() if count > 1
    )
    if duplicate_tokens:
        raise F2ResidualFeasibilityError(
            f"train rows contain duplicate tokens: {duplicate_tokens[:8]}"
        )
    overlap = sorted(set(train_tokens) & held_out_tokens)
    if overlap:
        raise F2ResidualFeasibilityError(
            f"train rows overlap prohibited held_out tokens: {overlap[:8]}"
        )
    if not train_rows:
        raise F2ResidualFeasibilityError("navtrain/train rows cannot be empty")

    identities: Dict[str, QualificationRow] = {}
    for row in train_rows:
        digest = _row_sampling_sha256(row)
        if digest in identities and identities[digest] != row:
            raise F2ResidualFeasibilityError(
                "distinct train rows collide under the sampling SHA-256"
            )
        identities[digest] = row

    rows_by_log: Dict[str, list[Tuple[str, QualificationRow]]] = defaultdict(list)
    for digest, row in identities.items():
        rows_by_log[row.log_name].append((digest, row))

    selected_stage: Dict[str, str] = {}
    for log_name in sorted(rows_by_log):
        ordered = sorted(
            rows_by_log[log_name],
            key=lambda item: (item[0], item[1].time_us, item[1].token),
        )
        for digest, _ in ordered[:FEASIBILITY_PER_LOG_SEED_COUNT]:
            selected_stage[digest] = "per_log_seed"

    target_count = min(FEASIBILITY_SAMPLE_SIZE, len(train_rows))
    if len(selected_stage) > target_count:
        raise F2ResidualFeasibilityError(
            "two-per-log seeds exceed the fixed 2048-row audit budget"
        )
    remaining = sorted(
        (
            (digest, row)
            for digest, row in identities.items()
            if digest not in selected_stage
        ),
        key=lambda item: (item[0], item[1].log_name, item[1].time_us, item[1].token),
    )
    for digest, _ in remaining[: target_count - len(selected_stage)]:
        selected_stage[digest] = "global_sha_fill"

    if len(selected_stage) != target_count:
        raise F2ResidualFeasibilityError(
            "deterministic train sampling did not reach its expected coverage"
        )

    selected = []
    for digest in sorted(selected_stage):
        row = identities[digest]
        if row.token in held_out_tokens:
            raise F2ResidualFeasibilityError(
                f"prohibited held_out token entered sample: {row.token}"
            )
        selected.append(
            TrainSampleRow(
                log_name=row.log_name,
                time_us=row.time_us,
                token=row.token,
                future_camera_valid=row.future_camera_valid,
                sampling_sha256=digest,
                selection_stage=selected_stage[digest],
            )
        )

    # Recheck the per-log minimum independently of the construction above.
    sampled_counts = Counter(row.log_name for row in selected)
    source_counts = Counter(row.log_name for row in train_rows)
    deficient_logs = sorted(
        log_name
        for log_name, count in source_counts.items()
        if sampled_counts[log_name] < min(FEASIBILITY_PER_LOG_SEED_COUNT, count)
    )
    if deficient_logs:
        raise F2ResidualFeasibilityError(
            f"per-log sample coverage failed: {deficient_logs[:8]}"
        )
    return tuple(selected)


@dataclass(frozen=True)
class FrozenStaticTokenInput:
    """Frozen base/target input for one selected train token."""

    log_name: str
    time_us: int
    token: str
    base_logits: Tuple[float, ...]
    target_scores: Tuple[Tuple[float, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_name", _nonempty_string(self.log_name, "log_name"))
        object.__setattr__(self, "token", _nonempty_string(self.token, "token"))
        _nonnegative_int(self.time_us, "time_us")
        object.__setattr__(
            self, "base_logits", _normalized_base_logits(self.base_logits)
        )
        object.__setattr__(
            self,
            "target_scores",
            _normalized_static_targets(self.target_scores),
        )


def _best_index(indices: Sequence[int], scores: Sequence[float]) -> int:
    if not indices:
        raise F2ResidualFeasibilityError("cannot select from an empty candidate set")
    best_index = indices[0]
    best_score = scores[best_index]
    for index in indices[1:]:
        if scores[index] > best_score:
            best_index = index
            best_score = scores[index]
    return best_index


@dataclass(frozen=True)
class TokenFeasibilityResult:
    log_name: str
    time_us: int
    token: str
    future_camera_valid: bool
    sampling_sha256: str
    selection_stage: str
    frozen_base_logits: Tuple[float, ...]
    frozen_target_scores: Tuple[Tuple[float, ...], ...]
    frozen_static_input_sha256: str
    reachability: ResidualReachabilityEnvelope
    static_true_epdms: Tuple[float, ...]
    base_selected_index: int
    base_selected_score: float
    reachable_oracle_index: int
    reachable_oracle_score: float
    unbounded_oracle_index: int
    unbounded_oracle_score: float
    reachable_delta: float
    unbounded_headroom: float
    reachable_headroom_closed: float
    reachable_hard_factor_upper: Tuple[float, float, float, float]
    safe_candidate_count: int
    reachable_safe_candidate_count: int
    has_safe_candidate: bool
    has_reachable_safe_candidate: bool
    safe_candidate_reachability_fraction: Optional[float]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "base_selected": {
                "candidate_index": self.base_selected_index,
                "static_true_epdms": self.base_selected_score,
            },
            "future_camera_valid": self.future_camera_valid,
            "frozen_static_input": {
                "base_logits": list(self.frozen_base_logits),
                "target_scores": [
                    list(candidate) for candidate in self.frozen_target_scores
                ],
            },
            "frozen_static_input_sha256": self.frozen_static_input_sha256,
            "hard_factor_reachable_marginal_upper": dict(
                zip(HARD_FACTOR_NAMES, self.reachable_hard_factor_upper)
            ),
            "log_name": self.log_name,
            "reachability": self.reachability.to_payload(),
            "reachable_delta_upper": self.reachable_delta,
            "reachable_headroom_closed_upper": self.reachable_headroom_closed,
            "reachable_oracle": {
                "candidate_index": self.reachable_oracle_index,
                "static_true_epdms": self.reachable_oracle_score,
            },
            "safety_candidate_coverage": {
                "has_reachable_safe_candidate": (
                    self.has_reachable_safe_candidate
                ),
                "has_safe_candidate": self.has_safe_candidate,
                "reachable_safe_candidate_count": (
                    self.reachable_safe_candidate_count
                ),
                "safe_candidate_count": self.safe_candidate_count,
                "safe_candidate_reachability_fraction": (
                    self.safe_candidate_reachability_fraction
                ),
                "safe_definition": "all_four_hard_channels_exactly_one",
            },
            "sampling_sha256": self.sampling_sha256,
            "selection_stage": self.selection_stage,
            "static_true_epdms": list(self.static_true_epdms),
            "time_us": self.time_us,
            "token": self.token,
            "unbounded_headroom": self.unbounded_headroom,
            "unbounded_oracle": {
                "candidate_index": self.unbounded_oracle_index,
                "static_true_epdms": self.unbounded_oracle_score,
            },
        }


def evaluate_static_token_feasibility(
    frozen: FrozenStaticTokenInput,
    *,
    sample_row: Optional[TrainSampleRow] = None,
) -> TokenFeasibilityResult:
    """Evaluate one frozen 32-by-9 train token without running a model."""

    if type(frozen) is not FrozenStaticTokenInput:
        raise F2ResidualFeasibilityError(
            "token feasibility requires FrozenStaticTokenInput"
        )
    if sample_row is None:
        standalone_row = QualificationRow(
            log_name=frozen.log_name,
            time_us=frozen.time_us,
            token=frozen.token,
            future_camera_valid=True,
        )
        sample_row = TrainSampleRow(
            log_name=frozen.log_name,
            time_us=frozen.time_us,
            token=frozen.token,
            future_camera_valid=True,
            sampling_sha256=_row_sampling_sha256(standalone_row),
            selection_stage="standalone_synthetic",
        )
    if (
        sample_row.log_name != frozen.log_name
        or sample_row.time_us != frozen.time_us
        or sample_row.token != frozen.token
    ):
        raise F2ResidualFeasibilityError(
            "frozen token identity differs from its selected train row"
        )

    reachability = analyze_tanh_residual_reachability(frozen.base_logits)
    reachable_indices = reachability.reachable_candidate_indices
    try:
        truths = tuple(
            causal_true_epdms(row, two_frame_comfort=None)
            for row in frozen.target_scores
        )
    except CausalValidationError as error:
        raise F2ResidualFeasibilityError(
            f"token {frozen.token} failed static EPDMS validation: {error}"
        ) from error

    base_index = reachability.base_selected_index
    reachable_oracle_index = _best_index(reachable_indices, truths)
    unbounded_oracle_index = _best_index(tuple(range(CANDIDATE_COUNT)), truths)
    base_score = truths[base_index]
    reachable_score = truths[reachable_oracle_index]
    unbounded_score = truths[unbounded_oracle_index]
    reachable_delta = _finite_float(
        reachable_score - base_score, "reachable delta"
    )
    unbounded_headroom = _finite_float(
        unbounded_score - base_score, "unbounded headroom"
    )
    if reachable_delta < -1e-12 or unbounded_headroom < -1e-12:
        raise F2ResidualFeasibilityError(
            "oracle score fell below the reachable frozen base selection"
        )
    reachable_delta = max(0.0, reachable_delta)
    unbounded_headroom = max(0.0, unbounded_headroom)
    if unbounded_headroom == 0.0:
        headroom_closed = 1.0
    else:
        headroom_closed = reachable_delta / unbounded_headroom
        if headroom_closed < -1e-12 or headroom_closed > 1.0 + 1e-12:
            raise F2ResidualFeasibilityError(
                "reachable headroom closure is outside [0, 1]"
            )
        headroom_closed = min(1.0, max(0.0, headroom_closed))

    hard_upper = tuple(
        max(frozen.target_scores[index][channel] for index in reachable_indices)
        for channel in range(4)
    )
    safe_indices = tuple(
        index
        for index, target in enumerate(frozen.target_scores)
        if all(target[channel] == 1.0 for channel in range(4))
    )
    reachable_set = set(reachable_indices)
    reachable_safe = tuple(
        index for index in safe_indices if index in reachable_set
    )
    safe_fraction = (
        len(reachable_safe) / len(safe_indices) if safe_indices else None
    )
    frozen_static_input_sha256 = _sha256(
        _canonical_json_bytes(
            {
                "base_logits": list(frozen.base_logits),
                "target_scores": [list(row) for row in frozen.target_scores],
            }
        )
    )

    return TokenFeasibilityResult(
        log_name=frozen.log_name,
        time_us=frozen.time_us,
        token=frozen.token,
        future_camera_valid=sample_row.future_camera_valid,
        sampling_sha256=sample_row.sampling_sha256,
        selection_stage=sample_row.selection_stage,
        frozen_base_logits=frozen.base_logits,
        frozen_target_scores=frozen.target_scores,
        frozen_static_input_sha256=frozen_static_input_sha256,
        reachability=reachability,
        static_true_epdms=truths,
        base_selected_index=base_index,
        base_selected_score=base_score,
        reachable_oracle_index=reachable_oracle_index,
        reachable_oracle_score=reachable_score,
        unbounded_oracle_index=unbounded_oracle_index,
        unbounded_oracle_score=unbounded_score,
        reachable_delta=reachable_delta,
        unbounded_headroom=unbounded_headroom,
        reachable_headroom_closed=headroom_closed,
        reachable_hard_factor_upper=hard_upper,  # type: ignore[arg-type]
        safe_candidate_count=len(safe_indices),
        reachable_safe_candidate_count=len(reachable_safe),
        has_safe_candidate=bool(safe_indices),
        has_reachable_safe_candidate=bool(reachable_safe),
        safe_candidate_reachability_fraction=safe_fraction,
    )


@dataclass(frozen=True)
class FeasibilityAggregate:
    token_count: int
    log_count: int
    base_selected_score: float
    reachable_oracle_score_upper: float
    unbounded_oracle_score: float
    reachable_delta_upper: float
    unbounded_headroom: float
    reachable_headroom_closed_upper: float
    reachable_candidate_fraction: float
    reachable_hard_factor_upper: Tuple[float, float, float, float]
    safe_candidate_token_count: int
    reachable_safe_candidate_token_count: int
    safe_candidate_token_coverage_fraction: Optional[float]
    reachable_safe_token_fraction: float

    def numeric_metrics(self) -> Dict[str, float]:
        metrics = {
            "base_selected_score": self.base_selected_score,
            "reachable_candidate_fraction": self.reachable_candidate_fraction,
            "reachable_delta_upper": self.reachable_delta_upper,
            "reachable_headroom_closed_upper": (
                self.reachable_headroom_closed_upper
            ),
            "reachable_oracle_score_upper": self.reachable_oracle_score_upper,
            "reachable_safe_token_fraction": self.reachable_safe_token_fraction,
            "unbounded_headroom": self.unbounded_headroom,
            "unbounded_oracle_score": self.unbounded_oracle_score,
        }
        metrics.update(
            {
                f"reachable_hard_factor_upper.{name}": value
                for name, value in zip(
                    HARD_FACTOR_NAMES, self.reachable_hard_factor_upper
                )
            }
        )
        return metrics

    def to_payload(self) -> Dict[str, Any]:
        return {
            "base_selected_score": self.base_selected_score,
            "log_count": self.log_count,
            "reachable_candidate_fraction": self.reachable_candidate_fraction,
            "reachable_delta_upper": self.reachable_delta_upper,
            "reachable_hard_factor_upper": dict(
                zip(HARD_FACTOR_NAMES, self.reachable_hard_factor_upper)
            ),
            "reachable_headroom_closed_upper": (
                self.reachable_headroom_closed_upper
            ),
            "reachable_oracle_score_upper": self.reachable_oracle_score_upper,
            "reachable_safe_candidate_token_count": (
                self.reachable_safe_candidate_token_count
            ),
            "reachable_safe_token_fraction": self.reachable_safe_token_fraction,
            "safe_candidate_token_count": self.safe_candidate_token_count,
            "safe_candidate_token_coverage_fraction": (
                self.safe_candidate_token_coverage_fraction
            ),
            "token_count": self.token_count,
            "unbounded_headroom": self.unbounded_headroom,
            "unbounded_oracle_score": self.unbounded_oracle_score,
        }


def _aggregate_token_results(
    results: Sequence[TokenFeasibilityResult],
) -> FeasibilityAggregate:
    if not results:
        raise F2ResidualFeasibilityError("cannot aggregate zero token results")
    token_count = len(results)
    base_score = _mean(
        (result.base_selected_score for result in results), "base selected score"
    )
    reachable_score = _mean(
        (result.reachable_oracle_score for result in results),
        "reachable oracle score",
    )
    unbounded_score = _mean(
        (result.unbounded_oracle_score for result in results),
        "unbounded oracle score",
    )
    reachable_delta = _mean(
        (result.reachable_delta for result in results), "reachable delta"
    )
    unbounded_headroom = _mean(
        (result.unbounded_headroom for result in results), "unbounded headroom"
    )
    headroom_closed = (
        1.0
        if unbounded_headroom == 0.0
        else reachable_delta / unbounded_headroom
    )
    reachable_candidate_fraction = math.fsum(
        len(result.reachability.reachable_candidate_indices) for result in results
    ) / (token_count * CANDIDATE_COUNT)
    hard_upper = tuple(
        _mean(
            (result.reachable_hard_factor_upper[channel] for result in results),
            f"reachable hard factor {HARD_FACTOR_NAMES[channel]}",
        )
        for channel in range(4)
    )
    safe_token_count = sum(result.has_safe_candidate for result in results)
    reachable_safe_token_count = sum(
        result.has_reachable_safe_candidate for result in results
    )
    safe_coverage = (
        reachable_safe_token_count / safe_token_count
        if safe_token_count
        else None
    )
    return FeasibilityAggregate(
        token_count=token_count,
        log_count=len({result.log_name for result in results}),
        base_selected_score=base_score,
        reachable_oracle_score_upper=reachable_score,
        unbounded_oracle_score=unbounded_score,
        reachable_delta_upper=reachable_delta,
        unbounded_headroom=unbounded_headroom,
        reachable_headroom_closed_upper=headroom_closed,
        reachable_candidate_fraction=reachable_candidate_fraction,
        reachable_hard_factor_upper=hard_upper,  # type: ignore[arg-type]
        safe_candidate_token_count=safe_token_count,
        reachable_safe_candidate_token_count=reachable_safe_token_count,
        safe_candidate_token_coverage_fraction=safe_coverage,
        reachable_safe_token_fraction=reachable_safe_token_count / token_count,
    )


@dataclass(frozen=True)
class BootstrapInterval:
    point_estimate: float
    lower: float
    upper: float

    def to_payload(self) -> Dict[str, float]:
        return {
            "lower": self.lower,
            "point_estimate": self.point_estimate,
            "upper": self.upper,
        }


@dataclass(frozen=True)
class LogClusterBootstrap:
    seed: int
    replicate_count: int
    confidence_level: float
    cluster_count: int
    intervals: Mapping[str, BootstrapInterval]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "cluster_count": self.cluster_count,
            "confidence_level": self.confidence_level,
            "intervals": {
                name: self.intervals[name].to_payload()
                for name in sorted(self.intervals)
            },
            "replicate_count": self.replicate_count,
            "resampling_unit": "log_name_cluster",
            "seed": self.seed,
        }


def _bootstrap_draw_index(
    *, seed: int, replicate_index: int, draw_index: int, cluster_count: int
) -> int:
    payload = (
        f"actworld-f2-log-bootstrap-v1\0{seed}\0"
        f"{replicate_index}\0{draw_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % cluster_count


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise F2ResidualFeasibilityError("cannot compute an empty quantile")
    if probability < 0.0 or probability > 1.0:
        raise F2ResidualFeasibilityError("quantile probability is outside [0, 1]")
    position = (len(sorted_values) - 1) * probability
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return math.fsum(
        (
            (1.0 - weight) * sorted_values[lower_index],
            weight * sorted_values[upper_index],
        )
    )


def log_cluster_bootstrap(
    results: Sequence[TokenFeasibilityResult],
    point_estimate: FeasibilityAggregate,
) -> LogClusterBootstrap:
    """Fixed-seed, fixed-count percentile bootstrap over whole log clusters."""

    if not results:
        raise F2ResidualFeasibilityError("bootstrap requires token results")
    duplicate_tokens = sorted(
        token
        for token, count in Counter(result.token for result in results).items()
        if count > 1
    )
    if duplicate_tokens:
        raise F2ResidualFeasibilityError(
            f"bootstrap input contains duplicate tokens: {duplicate_tokens[:8]}"
        )
    recomputed_point = _aggregate_token_results(results)
    if recomputed_point != point_estimate:
        raise F2ResidualFeasibilityError(
            "bootstrap point estimate does not match its token results"
        )
    grouped: Dict[str, list[TokenFeasibilityResult]] = defaultdict(list)
    for result in results:
        grouped[result.log_name].append(result)
    log_names = tuple(sorted(grouped))
    if not log_names:
        raise F2ResidualFeasibilityError("bootstrap requires at least one log")
    for log_name in log_names:
        grouped[log_name].sort(key=lambda result: (result.time_us, result.token))

    point_metrics = point_estimate.numeric_metrics()
    draws: Dict[str, list[float]] = {
        name: [] for name in sorted(point_metrics)
    }
    for replicate_index in range(FEASIBILITY_BOOTSTRAP_REPLICATES):
        replicate_results = []
        for draw_index in range(len(log_names)):
            selected_index = _bootstrap_draw_index(
                seed=FEASIBILITY_BOOTSTRAP_SEED,
                replicate_index=replicate_index,
                draw_index=draw_index,
                cluster_count=len(log_names),
            )
            replicate_results.extend(grouped[log_names[selected_index]])
        replicate_metrics = _aggregate_token_results(
            tuple(replicate_results)
        ).numeric_metrics()
        if set(replicate_metrics) != set(draws):
            raise F2ResidualFeasibilityError(
                "bootstrap metric schema changed between replicates"
            )
        for name, value in replicate_metrics.items():
            draws[name].append(_finite_float(value, f"bootstrap {name}"))

    alpha = (1.0 - FEASIBILITY_BOOTSTRAP_CONFIDENCE) / 2.0
    intervals = {}
    for name in sorted(draws):
        values = sorted(draws[name])
        intervals[name] = BootstrapInterval(
            point_estimate=point_metrics[name],
            lower=_linear_quantile(values, alpha),
            upper=_linear_quantile(values, 1.0 - alpha),
        )
    return LogClusterBootstrap(
        seed=FEASIBILITY_BOOTSTRAP_SEED,
        replicate_count=FEASIBILITY_BOOTSTRAP_REPLICATES,
        confidence_level=FEASIBILITY_BOOTSTRAP_CONFIDENCE,
        cluster_count=len(log_names),
        intervals=intervals,
    )


@dataclass(frozen=True)
class F2ResidualFeasibilityAudit:
    qualification_manifest_sha256: str
    sample_rows: Tuple[TrainSampleRow, ...]
    token_results: Tuple[TokenFeasibilityResult, ...]
    aggregate: FeasibilityAggregate
    bootstrap: LogClusterBootstrap
    schema_version: str = FEASIBILITY_SCHEMA_VERSION

    def to_payload(self) -> Dict[str, Any]:
        return {
            "aggregate": self.aggregate.to_payload(),
            "bootstrap": self.bootstrap.to_payload(),
            "claim_scope": (
                "analytical_train_only_upper_bound_not_checkpoint_or_test_score"
            ),
            "dataset_scope": {
                "dataset": FEASIBILITY_DATASET,
                "partition": FEASIBILITY_PARTITION,
                "prohibited_partitions": list(
                    FEASIBILITY_PROHIBITED_PARTITIONS
                ),
            },
            "qualification_manifest_sha256": (
                self.qualification_manifest_sha256
            ),
            "residual_contract": {
                "candidate_count": CANDIDATE_COUNT,
                "delta_float32_range": list(TANH_DELTA_FLOAT32_RANGE),
                "endpoint_rounding": "finite_torch_tanh_can_equal_plus_or_minus_one",
                "endpoint_tie_rule": (
                    "reachable_only_if_all_tied_competitors_have_higher_index"
                ),
                "selection_domain": (
                    "static_training_utility_before_two_frame_comfort_calibration"
                ),
                "stable_tie_rule": STABLE_TIE_RULE,
            },
            "sample": {
                "per_log_seed_count": FEASIBILITY_PER_LOG_SEED_COUNT,
                "rows": [row.to_payload() for row in self.sample_rows],
                "selection_order": (
                    "per_log_canonical_row_sha256_then_global_canonical_row_sha256"
                ),
                "target_size": FEASIBILITY_SAMPLE_SIZE,
                "token_count": len(self.sample_rows),
            },
            "schema_version": self.schema_version,
            "tokens": [result.to_payload() for result in self.token_results],
        }


def build_train_only_residual_feasibility_audit(
    qualification: QualificationManifest,
    frozen_tokens: Sequence[FrozenStaticTokenInput],
) -> F2ResidualFeasibilityAudit:
    """Build a complete deterministic audit with exact sampled-token coverage."""

    sample_rows = select_deterministic_navtrain_train_sample(qualification)
    if isinstance(frozen_tokens, (str, bytes)):
        raise F2ResidualFeasibilityError(
            "frozen_tokens must be a sequence of token inputs"
        )
    frozen_sequence = tuple(frozen_tokens)
    token_counts = Counter(token.token for token in frozen_sequence)
    duplicates = sorted(token for token, count in token_counts.items() if count > 1)
    if duplicates:
        raise F2ResidualFeasibilityError(
            f"frozen inputs contain duplicate tokens: {duplicates[:8]}"
        )
    frozen_by_token = {token.token: token for token in frozen_sequence}
    expected_tokens = {row.token for row in sample_rows}
    actual_tokens = set(frozen_by_token)
    held_out_tokens = set(qualification.held_out.tokens)
    prohibited = sorted(actual_tokens & held_out_tokens)
    if prohibited:
        raise F2ResidualFeasibilityError(
            f"frozen inputs include prohibited held_out tokens: {prohibited[:8]}"
        )
    missing = sorted(expected_tokens - actual_tokens)
    extra = sorted(actual_tokens - expected_tokens)
    if missing or extra:
        raise F2ResidualFeasibilityError(
            "frozen input coverage differs from the deterministic train sample; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    token_results = []
    for sample_row in sample_rows:
        frozen = frozen_by_token[sample_row.token]
        token_results.append(
            evaluate_static_token_feasibility(frozen, sample_row=sample_row)
        )
    results = tuple(token_results)
    if len(results) != len(sample_rows) or len({r.token for r in results}) != len(results):
        raise F2ResidualFeasibilityError(
            "token evaluation did not preserve exact sample coverage"
        )
    aggregate = _aggregate_token_results(results)
    bootstrap = log_cluster_bootstrap(results, aggregate)
    return F2ResidualFeasibilityAudit(
        qualification_manifest_sha256=qualification_manifest_sha256(qualification),
        sample_rows=sample_rows,
        token_results=results,
        aggregate=aggregate,
        bootstrap=bootstrap,
    )


def canonical_feasibility_audit_bytes(
    audit: F2ResidualFeasibilityAudit,
) -> bytes:
    if type(audit) is not F2ResidualFeasibilityAudit:
        raise F2ResidualFeasibilityError(
            "canonical serialization requires F2ResidualFeasibilityAudit"
        )
    return _canonical_json_bytes(audit.to_payload())


def feasibility_audit_sha256(audit: F2ResidualFeasibilityAudit) -> str:
    return _sha256(canonical_feasibility_audit_bytes(audit))


__all__ = [
    "FEASIBILITY_BOOTSTRAP_CONFIDENCE",
    "FEASIBILITY_BOOTSTRAP_REPLICATES",
    "FEASIBILITY_BOOTSTRAP_SEED",
    "FEASIBILITY_DATASET",
    "FEASIBILITY_PARTITION",
    "FEASIBILITY_PER_LOG_SEED_COUNT",
    "FEASIBILITY_PROHIBITED_PARTITIONS",
    "FEASIBILITY_SAMPLE_SIZE",
    "FEASIBILITY_SCHEMA_VERSION",
    "STABLE_TIE_RULE",
    "TANH_DELTA_FLOAT32_RANGE",
    "BootstrapInterval",
    "CandidateReachability",
    "F2ResidualFeasibilityAudit",
    "F2ResidualFeasibilityError",
    "FeasibilityAggregate",
    "FrozenStaticTokenInput",
    "LogClusterBootstrap",
    "ResidualReachabilityEnvelope",
    "TokenFeasibilityResult",
    "TrainSampleRow",
    "analyze_tanh_residual_reachability",
    "build_train_only_residual_feasibility_audit",
    "canonical_feasibility_audit_bytes",
    "evaluate_static_token_feasibility",
    "feasibility_audit_sha256",
    "log_cluster_bootstrap",
    "select_deterministic_navtrain_train_sample",
]
