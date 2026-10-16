"""Dependency-light leakage guards and evidence for proposal diagnostics."""

from __future__ import annotations

import hashlib
import json
from numbers import Number
from typing import Any, Container, Dict, Iterable, List, Optional, Sequence, Tuple


def check_leakage_guard(split: str, phase: str, partition: str) -> None:
    """Require the one permitted phase/partition pair for each split."""

    allowed = {
        "navtrain": ("development", "validation"),
        "navtest": ("final", "full"),
    }
    if split not in allowed:
        raise ValueError(
            "proposal oracle supports only navtrain and navtest; "
            "navhard remains final selected-trajectory evaluation only"
        )
    expected_phase, expected_partition = allowed[split]
    if (phase, partition) != (expected_phase, expected_partition):
        raise ValueError(
            f"split={split!r} requires proposal_oracle.phase={expected_phase!r} "
            f"and proposal_oracle.partition={expected_partition!r}; use the "
            "fixed navtrain validation partition for iteration and the full "
            "navtest partition only after the checkpoint and hyperparameters "
            "are frozen"
        )


def canonical_identifiers(values: Iterable[Any]) -> List[str]:
    """Return a deterministic, duplicate-free identifier allowlist."""

    return sorted({str(value) for value in values})


def identifier_evidence(
    values: Iterable[Any], *, prefix: str
) -> Dict[str, Any]:
    """Count and hash identifiers using sorted UTF-8 lines.

    The final newline is part of every non-empty digest. Sorting makes the
    evidence independent of filesystem and config iteration order.
    """

    identifiers = sorted(str(value) for value in values)
    payload = (
        ("\n".join(identifiers) + "\n").encode("utf-8")
        if identifiers
        else b""
    )
    return {
        f"{prefix}_count": len(identifiers),
        f"{prefix}_sha256": hashlib.sha256(payload).hexdigest(),
    }


def apply_partition_allowlist(
    scene_filter: Any,
    *,
    partition: str,
    val_logs: Optional[Sequence[str]],
) -> List[str]:
    """Apply the fixed held-out log allowlist required by a partition.

    ``validation`` is always the intersection of the configured scene split
    and ``cfg.val_logs``. ``full`` leaves the configured scene split intact.
    The returned identifiers are the effective log allowlist recorded in the
    diagnostic evidence.
    """

    configured_logs = scene_filter.log_names
    if partition == "validation":
        if val_logs is None:
            raise ValueError(
                "proposal_oracle.partition='validation' requires cfg.val_logs"
            )
        validation_logs = canonical_identifiers(val_logs)
        if not validation_logs:
            raise ValueError(
                "proposal_oracle.partition='validation' requires non-empty "
                "cfg.val_logs"
            )
        if configured_logs is None:
            effective_logs = validation_logs
        else:
            configured_set = set(canonical_identifiers(configured_logs))
            effective_logs = [
                log_name
                for log_name in validation_logs
                if log_name in configured_set
            ]
        if not effective_logs:
            raise RuntimeError(
                "configured scene_filter.log_names and cfg.val_logs have no "
                "logs in common"
            )
        scene_filter.log_names = effective_logs
        return effective_logs
    if partition == "full":
        return (
            canonical_identifiers(configured_logs)
            if configured_logs is not None
            else []
        )
    raise ValueError(
        "proposal_oracle.partition must be 'validation' or 'full'"
    )


def metric_coverage(
    requested_tokens: Sequence[Any],
    metric_tokens: Container[str],
    *,
    require_complete: bool = True,
) -> Tuple[List[int], List[str], List[str], Dict[str, Any]]:
    """Return cache coverage and evidence, rejecting incomplete caches by default."""

    requested = [str(token) for token in requested_tokens]
    if not requested:
        raise RuntimeError("proposal-oracle partition contains no requested tokens")
    valid_pairs = [
        (index, token)
        for index, token in enumerate(requested)
        if token in metric_tokens
    ]
    valid_indices = [index for index, _ in valid_pairs]
    valid_tokens = [token for _, token in valid_pairs]
    missing_tokens = [token for token in requested if token not in metric_tokens]
    evidence = {
        **identifier_evidence(requested, prefix="requested_token"),
        **identifier_evidence(valid_tokens, prefix="valid_token"),
        **identifier_evidence(missing_tokens, prefix="missing_token"),
    }
    if require_complete and missing_tokens:
        raise RuntimeError(
            "proposal-oracle partitions require complete metric-cache coverage; "
            f"evidence={json.dumps(evidence, sort_keys=True)}, "
            f"missing_examples={missing_tokens[:8]}"
        )
    return valid_indices, valid_tokens, missing_tokens, evidence


def attach_partition_evidence(
    rows: List[Dict[str, Any]], evidence: Dict[str, Any]
) -> None:
    """Record immutable partition evidence on every per-scene CSV row."""

    for row in rows:
        overlap = set(row).intersection(evidence)
        if overlap:
            raise ValueError(f"partition evidence would overwrite columns: {overlap}")
        row.update(evidence)


def summarize(
    rows: List[Dict[str, Any]],
    split: str,
    phase: str,
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a JSON-safe aggregate without importing runtime ML dependencies."""

    if not rows:
        raise ValueError("cannot summarize an empty proposal-oracle result")
    evidence_columns = set(evidence)
    mean_columns = [
        column
        for column in rows[0]
        if column not in {"token", "split"}
        and column not in evidence_columns
        and not column.endswith("_index")
        and all(isinstance(row.get(column), Number) for row in rows)
    ]
    return {
        "split": split,
        "phase": phase,
        **evidence,
        "num_scenes": len(rows),
        "leakage_guard": (
            "navtrain/development/validation may guide iteration; "
            "navtest/final/full is reporting only and must not guide checkpoint "
            "or hyperparameter selection"
        ),
        "selection_contract": (
            "single-scene proposal selection without sequential momentum state; "
            "this is a headroom diagnostic, not the official benchmark score"
        ),
        "means": {
            column: sum(float(row[column]) for row in rows) / len(rows)
            for column in mean_columns
        },
    }
