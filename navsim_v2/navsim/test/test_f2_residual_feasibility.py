"""Pure CPU contract tests for the F2 residual feasibility audit."""

from __future__ import annotations

import hashlib
import math

import pytest

from navsim.planning.training.f2_qualification_manifest import (
    QualificationManifest,
    QualificationRow,
    build_qualification_manifest,
    build_qualification_partition,
)
from navsim.planning.validation.f2_residual_feasibility import (
    FEASIBILITY_BOOTSTRAP_REPLICATES,
    FEASIBILITY_BOOTSTRAP_SEED,
    FEASIBILITY_PROHIBITED_PARTITIONS,
    FEASIBILITY_SAMPLE_SIZE,
    F2ResidualFeasibilityError,
    FrozenStaticTokenInput,
    analyze_tanh_residual_reachability,
    build_train_only_residual_feasibility_audit,
    canonical_feasibility_audit_bytes,
    evaluate_static_token_feasibility,
    feasibility_audit_sha256,
    select_deterministic_navtrain_train_sample,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _static_row(
    score: float,
    *,
    hard: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> tuple[float, ...]:
    multiplier = math.prod(hard)
    aggregate = multiplier * score
    return (*hard, score, score, score, score, aggregate)


def _targets(
    scores: tuple[float, ...] | None = None,
    *,
    hard: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> tuple[tuple[float, ...], ...]:
    if scores is None:
        scores = tuple(index / 64.0 for index in range(32))
    assert len(scores) == 32
    return tuple(_static_row(score, hard=hard) for score in scores)


def _manifest(
    train_rows: tuple[QualificationRow, ...],
    held_rows: tuple[QualificationRow, ...] | None = None,
) -> QualificationManifest:
    if held_rows is None:
        held_rows = (QualificationRow("held-log", 1, "held-token", True),)
    train_logs = {row.log_name for row in train_rows}
    held_logs = {row.log_name for row in held_rows}
    train = build_qualification_partition(
        "train",
        train_rows,
        raw_token_count=len(train_rows),
        raw_log_count=len(train_logs),
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=_digest("train-exclusion"),
    )
    held_out = build_qualification_partition(
        "held_out",
        held_rows,
        raw_token_count=len(held_rows),
        raw_log_count=len(held_logs),
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=_digest("held-exclusion"),
    )
    return build_qualification_manifest(
        train,
        held_out,
        config_sha256=_digest("config"),
        cache_sha256=_digest("cache"),
        source_sha256=_digest("source"),
    )


def _frozen(row: QualificationRow, *, offset: int = 0) -> FrozenStaticTokenInput:
    scores = tuple(((index + offset) % 32) / 64.0 for index in range(32))
    return FrozenStaticTokenInput(
        log_name=row.log_name,
        time_us=row.time_us,
        token=row.token,
        base_logits=tuple(index / 100.0 for index in range(32)),
        target_scores=_targets(scores),
    )


def test_float32_endpoint_equality_is_reachable_for_lower_index() -> None:
    envelope = analyze_tanh_residual_reachability((0.0,) + (2.0,) * 31)

    boundary = envelope.candidates[0]
    assert boundary.endpoint_selection_margin == 0.0
    assert boundary.reachable
    assert boundary.boundary_kind == "reachable_endpoint_tie_lower_index"
    assert boundary.endpoint_controls_attainable
    # Finite float32 tanh can saturate to +/-1 and the exact endpoint tie
    # therefore follows the deployed lowest-index rule.
    assert envelope.base_selected_index == 1
    assert 0 in envelope.reachable_candidate_indices


def test_float32_endpoint_equality_is_unreachable_for_higher_index() -> None:
    envelope = analyze_tanh_residual_reachability((2.0,) * 31 + (0.0,))

    boundary = envelope.candidates[31]
    assert boundary.endpoint_selection_margin == 0.0
    assert not boundary.reachable
    assert boundary.boundary_kind == "unreachable_endpoint_tie_higher_index"
    assert 31 not in envelope.reachable_candidate_indices


def test_stable_base_tie_uses_lowest_index_and_all_equal_candidates_reachable() -> None:
    envelope = analyze_tanh_residual_reachability((0.0,) * 32)
    assert envelope.base_selected_index == 0
    assert envelope.reachable_candidate_indices == tuple(range(32))
    assert all(candidate.endpoint_selection_margin > 0.0 for candidate in envelope.candidates)


def test_sigmoid_float32_tie_can_make_logit_winner_unreachable() -> None:
    logits = (1.0, -0.9999999403953552) + (-20.0,) * 30
    envelope = analyze_tanh_residual_reachability(logits)

    # At candidate 1's endpoint, utility[1] exceeds utility[0] by one
    # float32 ULP, but both deployed float32 sigmoid scores round to 0.5.
    boundary = envelope.candidates[1]
    assert boundary.endpoint_selection_margin == 0.0
    assert not boundary.reachable
    assert boundary.boundary_kind == "unreachable_endpoint_tie_higher_index"
    assert 1 not in envelope.reachable_candidate_indices


def test_unreachable_best_target_reduces_reachable_oracle_and_safety_coverage() -> None:
    # Candidate 31 is exactly on an endpoint tie with lower-index competitors. It is the
    # only safe and globally optimal target; every reachable candidate has a
    # first hard factor of 0.5 and static EPDMS 0.1.
    rows = [
        _static_row(0.2, hard=(0.5, 1.0, 1.0, 1.0))
        for _ in range(31)
    ]
    rows.append(_static_row(1.0))
    frozen = FrozenStaticTokenInput(
        log_name="train-log",
        time_us=1,
        token="token",
        base_logits=(2.0,) * 31 + (0.0,),
        target_scores=tuple(rows),
    )

    result = evaluate_static_token_feasibility(frozen)

    assert result.base_selected_index == 0
    assert result.base_selected_score == pytest.approx(0.1)
    assert result.reachable_oracle_index == 0
    assert result.reachable_oracle_score == pytest.approx(0.1)
    assert result.unbounded_oracle_index == 31
    assert result.unbounded_oracle_score == 1.0
    assert result.reachable_delta == 0.0
    assert result.unbounded_headroom == pytest.approx(0.9)
    assert result.reachable_headroom_closed == 0.0
    assert result.reachable_hard_factor_upper == (0.5, 1.0, 1.0, 1.0)
    assert result.safe_candidate_count == 1
    assert result.reachable_safe_candidate_count == 0
    assert result.safe_candidate_reachability_fraction == 0.0


def test_static_target_reuses_official_multiplier_aggregate_validation() -> None:
    rows = list(_targets())
    inconsistent = list(rows[0])
    inconsistent[8] = 0.75
    rows[0] = tuple(inconsistent)

    with pytest.raises(F2ResidualFeasibilityError, match="official static EPDMS"):
        FrozenStaticTokenInput(
            log_name="log",
            time_us=1,
            token="token",
            base_logits=(0.0,) * 32,
            target_scores=tuple(rows),
        )


@pytest.mark.parametrize(
    "bad_logits",
    [
        (float("nan"),) + (0.0,) * 31,
        (float("inf"),) + (0.0,) * 31,
    ],
)
def test_nonfinite_base_logits_fail_closed(bad_logits: tuple[float, ...]) -> None:
    with pytest.raises(F2ResidualFeasibilityError, match="finite"):
        analyze_tanh_residual_reachability(bad_logits)


def test_sampling_is_train_only_two_per_log_then_global_sha() -> None:
    # Exceed the fixed sample size so selection, rather than the small-dataset
    # all-row shortcut, is exercised.  A tiny third log must still retain both
    # of its rows before the global SHA fill.
    train_rows = tuple(
        QualificationRow(
            "train-a" if index < 1024 else "train-b",
            index,
            f"train-token-{index:04d}",
            True,
        )
        for index in range(2048)
    ) + (
        QualificationRow("train-c", 3000, "train-token-2048", True),
        QualificationRow("train-c", 3001, "train-token-2049", True),
    )
    held_rows = (
        QualificationRow("held-log", 0, "held-smallest-looking-token", True),
    )
    qualification = _manifest(train_rows, held_rows)

    first = select_deterministic_navtrain_train_sample(qualification)
    second = select_deterministic_navtrain_train_sample(qualification)

    assert first == second
    assert len(first) == FEASIBILITY_SAMPLE_SIZE
    assert "held-smallest-looking-token" not in {row.token for row in first}
    assert sum(row.log_name == "train-c" for row in first) == 2
    assert sum(row.selection_stage == "per_log_seed" for row in first) == 6
    assert all(row.token.startswith("train-token-") for row in first)


def test_small_train_sample_uses_every_train_row_and_never_held_out() -> None:
    train_rows = (
        QualificationRow("train-a", 2, "train-2", True),
        QualificationRow("train-a", 1, "train-1", True),
        QualificationRow("train-b", 3, "train-3", False),
    )
    qualification = _manifest(train_rows)
    sample = select_deterministic_navtrain_train_sample(qualification)

    assert {row.token for row in sample} == {"train-1", "train-2", "train-3"}
    assert not ({row.token for row in sample} & set(qualification.held_out.tokens))


def test_audit_bootstrap_and_canonical_sha_are_input_order_invariant() -> None:
    train_rows = (
        QualificationRow("log-a", 1, "a-1", True),
        QualificationRow("log-a", 2, "a-2", True),
        QualificationRow("log-b", 1, "b-1", True),
        QualificationRow("log-b", 2, "b-2", False),
    )
    qualification = _manifest(train_rows)
    frozen = tuple(_frozen(row, offset=index) for index, row in enumerate(train_rows))

    first = build_train_only_residual_feasibility_audit(qualification, frozen)
    second = build_train_only_residual_feasibility_audit(
        qualification, tuple(reversed(frozen))
    )

    assert canonical_feasibility_audit_bytes(first) == canonical_feasibility_audit_bytes(second)
    assert feasibility_audit_sha256(first) == feasibility_audit_sha256(second)
    assert first.bootstrap.seed == FEASIBILITY_BOOTSTRAP_SEED
    assert first.bootstrap.replicate_count == FEASIBILITY_BOOTSTRAP_REPLICATES
    assert first.bootstrap.cluster_count == 2
    assert first.bootstrap.intervals == second.bootstrap.intervals
    payload = first.to_payload()
    assert payload["dataset_scope"]["partition"] == "train"
    assert tuple(payload["dataset_scope"]["prohibited_partitions"]) == FEASIBILITY_PROHIBITED_PARTITIONS
    assert payload["claim_scope"].endswith("not_checkpoint_or_test_score")
    assert canonical_feasibility_audit_bytes(first).endswith(b"\n")


def test_bootstrap_is_degenerate_but_valid_for_one_log() -> None:
    train_rows = (
        QualificationRow("only-log", 1, "one", True),
        QualificationRow("only-log", 2, "two", True),
    )
    qualification = _manifest(train_rows)
    audit = build_train_only_residual_feasibility_audit(
        qualification, tuple(_frozen(row) for row in train_rows)
    )
    for interval in audit.bootstrap.intervals.values():
        assert interval.lower == interval.point_estimate == interval.upper


def test_duplicate_missing_extra_and_held_out_inputs_fail_closed() -> None:
    train_rows = (
        QualificationRow("train-log", 1, "train-1", True),
        QualificationRow("train-log", 2, "train-2", True),
    )
    qualification = _manifest(train_rows)
    first = _frozen(train_rows[0])
    second = _frozen(train_rows[1])

    with pytest.raises(F2ResidualFeasibilityError, match="duplicate"):
        build_train_only_residual_feasibility_audit(
            qualification, (first, first, second)
        )
    with pytest.raises(F2ResidualFeasibilityError, match="coverage"):
        build_train_only_residual_feasibility_audit(qualification, (first,))

    extra_row = QualificationRow("train-log", 3, "extra", True)
    with pytest.raises(F2ResidualFeasibilityError, match="coverage"):
        build_train_only_residual_feasibility_audit(
            qualification, (first, second, _frozen(extra_row))
        )

    held = qualification.held_out.rows[0]
    with pytest.raises(F2ResidualFeasibilityError, match="prohibited held_out"):
        build_train_only_residual_feasibility_audit(
            qualification, (first, second, _frozen(held))
        )
