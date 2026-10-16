"""Synthetic contract tests for the F2 causal validator foundation."""

from __future__ import annotations

import hashlib
import math
import struct
import unittest
from dataclasses import replace

from navsim.planning.validation.f2_causal_validator import (
    CANDIDATE_COUNT,
    F2_MODEL_NUMERIC_MODE,
    F2_STATIC_AGGREGATE_SANITY_ATOL,
    F2_SENSOR_CONTRACT,
    FORMAL_GATE_THRESHOLDS,
    AggregateMetrics,
    CausalValidationError,
    CheckpointEvaluation,
    DualBranchCausalStateMachine,
    FrozenTokenArtifact,
    ManifestRow,
    ModelTokenOutput,
    TokenCausalResult,
    aggregate_results,
    build_canonical_manifest,
    causal_true_epdms,
    canonical_evaluation_bytes,
    canonical_manifest_bytes,
    canonical_metrics_bytes,
    deterministic_log_assignments,
    evaluate_formal_gates,
    evaluate_manifest_sharded,
    is_continuous,
    manifest_from_bytes,
    manifest_sha256,
    micro_pair_counts,
    optimistic_oracle_epdms,
    rank_checkpoints,
    stable_argmax,
)


ZERO_SHA = "0" * 64
ONE_SHA = "1" * 64
TWO_SHA = "2" * 64


def _proposal_sha(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _float32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _target_scores() -> tuple[tuple[float, ...], ...]:
    rows = []
    for index in range(CANDIDATE_COUNT):
        aggregate = index / (CANDIDATE_COUNT - 1)
        rows.append(
            (
                1.0,
                1.0,
                1.0,
                1.0,
                aggregate,
                aggregate,
                aggregate,
                aggregate,
                aggregate,
            )
        )
    return tuple(rows)


def _artifact(
    token: str,
    *,
    base_logits: tuple[float, ...] | None = None,
) -> FrozenTokenArtifact:
    if base_logits is None:
        base_logits = tuple(float(index) / 100.0 for index in range(CANDIDATE_COUNT))
    return FrozenTokenArtifact(
        token=token,
        proposal_sha256=_proposal_sha(token),
        base_logits=base_logits,
        target_scores=_target_scores(),
        calibration_states=tuple(
            {"simulator": "v2", "token": token, "index": index}
            for index in range(CANDIDATE_COUNT)
        ),
        carry_states=tuple(
            {
                "simulator": "official",
                "token": token,
                "index": index,
                "seen_by": [],
            }
            for index in range(CANDIDATE_COUNT)
        ),
    )


def _output(
    token: str,
    residual_logits: tuple[float, ...] | None = None,
    *,
    base_logits: tuple[float, ...] | None = None,
) -> ModelTokenOutput:
    if base_logits is None:
        base_logits = tuple(float(index) / 100.0 for index in range(CANDIDATE_COUNT))
    if residual_logits is None:
        residual_logits = tuple(
            float(index) / 90.0 for index in range(CANDIDATE_COUNT)
        )
    delta_logits = tuple(
        _float32(math.tanh(_float32(residual))) for residual in residual_logits
    )
    utility_logits = tuple(
        _float32(_float32(base) + delta)
        for base, delta in zip(base_logits, delta_logits)
    )
    return ModelTokenOutput(
        token=token,
        proposal_sha256=_proposal_sha(token),
        base_logits=base_logits,
        residual_logits=residual_logits,
        delta_logits=delta_logits,
        utility_logits=utility_logits,
        numeric_mode=F2_MODEL_NUMERIC_MODE,
    )


def _manifest(rows: list[ManifestRow]):
    tokens = [row.token for row in rows]
    logs = sorted({row.log_name for row in rows})
    return build_canonical_manifest(
        rows,
        expected_tokens=tokens,
        sensor_available_tokens=tokens,
        cache_tokens=tokens + ["unused-cache-token"],
        allowlist_logs=logs,
        cache_metadata_sha256=ZERO_SHA,
        source_revision="synthetic-fixture",
    )


def _passing_metrics(
    *,
    selected_score: float = 0.90,
    base_score: float = 0.85,
    oracle_score: float = 0.95,
    fatal_count: int = 0,
    coverage_fraction: float = 1.0,
) -> AggregateMetrics:
    return AggregateMetrics(
        expected_token_count=10,
        observed_token_count=10 if coverage_fraction == 1.0 else 9,
        coverage_fraction=coverage_fraction,
        fatal_count=fatal_count,
        future_camera_valid_fraction=1.0,
        critic_selected_score=selected_score,
        base_selected_score=base_score,
        oracle_score=oracle_score,
        critic_delta=selected_score - base_score,
        critic_oracle_regret=oracle_score - selected_score,
        critic_pairwise_accuracy=0.75,
        critic_pairwise_correct=75,
        critic_pairwise_comparable=100,
        critic_score_error=0.08,
        base_score_error=0.07,
        critic_score_error_excess=0.01,
        proposal_headroom_closed=(selected_score - base_score)
        / (oracle_score - base_score),
        hard_factor_deltas=(0.0, 0.0, 0.0, 0.0),
    )


class ManifestContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            ManifestRow("log-b", 1_500_000, "token-b", True),
            ManifestRow("log-a", 1_000_000, "token-a", True),
            ManifestRow("log-a", 1_500_000, "token-c", False),
        ]

    def _build(self, rows):
        return build_canonical_manifest(
            rows,
            expected_tokens=["token-a", "token-b", "token-c"],
            sensor_available_tokens=["token-c", "token-a", "token-b"],
            cache_tokens=["token-b", "token-c", "token-a", "unused"],
            allowlist_logs=["log-b", "log-a"],
            cache_metadata_sha256=ZERO_SHA,
            source_revision="revision",
        )

    def test_manifest_is_canonical_and_hash_stable(self) -> None:
        first = self._build(self.rows)
        second = self._build(list(reversed(self.rows)))
        self.assertEqual(canonical_manifest_bytes(first), canonical_manifest_bytes(second))
        self.assertEqual(manifest_sha256(first), manifest_sha256(second))

        payload = canonical_manifest_bytes(first)
        parsed = manifest_from_bytes(
            payload,
            expected_sha256=manifest_sha256(first),
            expected_tokens=["token-a", "token-b", "token-c"],
            sensor_available_tokens=["token-a", "token-b", "token-c"],
            cache_tokens=["token-a", "token-b", "token-c"],
        )
        self.assertEqual(parsed, first)

    def test_duplicate_missing_partition_sensor_and_cache_are_rejected(self) -> None:
        with self.assertRaisesRegex(CausalValidationError, "duplicate"):
            self._build(self.rows + [self.rows[0]])
        with self.assertRaisesRegex(CausalValidationError, "differs"):
            build_canonical_manifest(
                self.rows[:-1],
                expected_tokens=["token-a", "token-b", "token-c"],
                sensor_available_tokens=["token-a", "token-b", "token-c"],
                cache_tokens=["token-a", "token-b", "token-c"],
                allowlist_logs=["log-a", "log-b"],
                cache_metadata_sha256=ZERO_SHA,
                source_revision="revision",
            )
        with self.assertRaisesRegex(CausalValidationError, "restricted"):
            build_canonical_manifest(
                self.rows,
                expected_tokens=["token-a", "token-b", "token-c"],
                sensor_available_tokens=["token-a", "token-b", "token-c"],
                cache_tokens=["token-a", "token-b", "token-c"],
                allowlist_logs=["log-a", "log-b"],
                cache_metadata_sha256=ZERO_SHA,
                source_revision="revision",
                partition="forbidden",
            )
        drifted_sensor = dict(F2_SENSOR_CONTRACT)
        drifted_sensor["cam_f0"] = (1, 2, 3)
        with self.assertRaisesRegex(CausalValidationError, "sensor contract"):
            build_canonical_manifest(
                self.rows,
                expected_tokens=["token-a", "token-b", "token-c"],
                sensor_available_tokens=["token-a", "token-b", "token-c"],
                cache_tokens=["token-a", "token-b", "token-c"],
                allowlist_logs=["log-a", "log-b"],
                cache_metadata_sha256=ZERO_SHA,
                source_revision="revision",
                sensor_contract=drifted_sensor,
            )
        with self.assertRaisesRegex(CausalValidationError, "sensor availability"):
            build_canonical_manifest(
                self.rows,
                expected_tokens=["token-a", "token-b", "token-c"],
                sensor_available_tokens=["token-a", "token-b"],
                cache_tokens=["token-a", "token-b", "token-c"],
                allowlist_logs=["log-a", "log-b"],
                cache_metadata_sha256=ZERO_SHA,
                source_revision="revision",
            )
        with self.assertRaisesRegex(CausalValidationError, "metric cache"):
            build_canonical_manifest(
                self.rows,
                expected_tokens=["token-a", "token-b", "token-c"],
                sensor_available_tokens=["token-a", "token-b", "token-c"],
                cache_tokens=["token-a", "token-b"],
                allowlist_logs=["log-a", "log-b"],
                cache_metadata_sha256=ZERO_SHA,
                source_revision="revision",
            )
        with self.assertRaisesRegex(CausalValidationError, "duplicate"):
            build_canonical_manifest(
                self.rows,
                expected_tokens=["token-a", "token-b", "token-c"],
                sensor_available_tokens=[
                    "token-a",
                    "token-b",
                    "token-c",
                    "token-c",
                ],
                cache_tokens=["token-a", "token-b", "token-c"],
                allowlist_logs=["log-a", "log-b"],
                cache_metadata_sha256=ZERO_SHA,
                source_revision="revision",
            )
        with self.assertRaisesRegex(CausalValidationError, "duplicate"):
            build_canonical_manifest(
                self.rows,
                expected_tokens=["token-a", "token-b", "token-c"],
                sensor_available_tokens=["token-a", "token-b", "token-c"],
                cache_tokens=["token-a", "token-b", "token-c", "token-c"],
                allowlist_logs=["log-a", "log-b"],
                cache_metadata_sha256=ZERO_SHA,
                source_revision="revision",
            )
        with self.assertRaisesRegex(CausalValidationError, "fully represented"):
            build_canonical_manifest(
                self.rows,
                expected_tokens=["token-a", "token-b", "token-c"],
                sensor_available_tokens=["token-a", "token-b", "token-c"],
                cache_tokens=["token-a", "token-b", "token-c"],
                allowlist_logs=["log-a", "log-b", "missing-log"],
                cache_metadata_sha256=ZERO_SHA,
                source_revision="revision",
            )

    def test_manifest_tamper_and_noncanonical_bytes_are_rejected(self) -> None:
        manifest = self._build(self.rows)
        payload = canonical_manifest_bytes(manifest)
        with self.assertRaisesRegex(CausalValidationError, "SHA-256"):
            manifest_from_bytes(payload + b" ", expected_sha256=manifest_sha256(manifest))
        noncanonical = payload[:-1] + b" \n"
        with self.assertRaisesRegex(CausalValidationError, "canonical"):
            manifest_from_bytes(
                noncanonical,
                expected_sha256=hashlib.sha256(noncanonical).hexdigest(),
            )


class CausalStateMachineTest(unittest.TestCase):
    def test_official_causal_truth_uses_14_or_16_weights_and_multiplier(self) -> None:
        weighted = 5.0 * 0.8 + 5.0 * 0.6 + 2.0 * 0.4 + 2.0 * 0.2
        static = 0.5 * weighted / 14.0
        row = (0.5, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, static)
        self.assertEqual(
            causal_true_epdms(row, two_frame_comfort=None),
            _float32(0.5 * weighted / 14.0),
        )
        self.assertEqual(
            causal_true_epdms(row, two_frame_comfort=0.0),
            _float32(0.5 * weighted / 16.0),
        )
        self.assertEqual(
            causal_true_epdms(row, two_frame_comfort=1.0),
            _float32(0.5 * (weighted + 2.0) / 16.0),
        )
        # Deployment calibrates the predicted probability without the hard
        # multiplier.  q selects; official causal EPDMS supplies the truth.
        deployment_q = _float32((14.0 * 0.5 + 2.0) / 16.0)
        self.assertEqual(deployment_q, _float32(9.0 / 16.0))
        self.assertNotEqual(
            deployment_q,
            causal_true_epdms(row, two_frame_comfort=1.0),
        )

        rounded_row = (*row[:-1], static + 0.5 * F2_STATIC_AGGREGATE_SANITY_ATOL)
        causal_true_epdms(rounded_row, two_frame_comfort=None)
        inconsistent_row = (
            *row[:-1],
            static + 2.0 * F2_STATIC_AGGREGATE_SANITY_ATOL,
        )
        with self.assertRaisesRegex(
            CausalValidationError, "stored aggregate channel is inconsistent"
        ):
            causal_true_epdms(inconsistent_row, two_frame_comfort=None)

        candidates = [row for _ in range(CANDIDATE_COUNT)]
        self.assertEqual(
            optimistic_oracle_epdms(candidates, continuous=False),
            causal_true_epdms(row, two_frame_comfort=None),
        )
        self.assertEqual(
            optimistic_oracle_epdms(candidates, continuous=True),
            causal_true_epdms(row, two_frame_comfort=1.0),
        )

    def test_continuity_boundaries_and_lowest_index_tie(self) -> None:
        self.assertFalse(is_continuous(None, 10))
        self.assertFalse(is_continuous(10, 9))
        self.assertFalse(is_continuous(10, 10))
        self.assertTrue(is_continuous(10, 550_010))
        self.assertFalse(is_continuous(10, 550_011))
        self.assertEqual(stable_argmax([1.0, 2.0, 2.0, 0.0]), 1)
        # Comparisons are defined after canonical float32 conversion, not in
        # the caller's arbitrary Python-double precision.
        self.assertEqual(stable_argmax([1.0, 1.0 + 2.0**-25]), 0)
        self.assertEqual(
            micro_pair_counts(
                [0.0, 1.0], [1.0, 1.0 + 2.0**-25]
            ),
            (0, 0),
        )

    def test_current_checkpoint_base_logit_drift_is_fatal(self) -> None:
        row = ManifestRow("log", 1_000_000, "token", True)
        artifact = _artifact("token")
        drifted = list(artifact.base_logits)
        drifted[7] += 0.01
        output = _output("token", base_logits=tuple(drifted))

        with self.assertRaisesRegex(CausalValidationError, "base logits drifted"):
            DualBranchCausalStateMachine(
                lambda branch, past, current, delta_time_us: 0.0
            ).evaluate_log(
                [row], {"token": artifact}, {"token": output}
            )

    def test_model_output_rejects_miswired_f2_delta_or_utility(self) -> None:
        zeros = tuple(0.0 for _ in range(CANDIDATE_COUNT))
        miswired = list(zeros)
        miswired[7] = 0.25
        with self.assertRaisesRegex(CausalValidationError, "delta_logits"):
            ModelTokenOutput(
                token="token",
                proposal_sha256=_proposal_sha("token"),
                base_logits=zeros,
                residual_logits=zeros,
                delta_logits=tuple(miswired),
                utility_logits=zeros,
                numeric_mode=F2_MODEL_NUMERIC_MODE,
            )

        with self.assertRaisesRegex(CausalValidationError, "do not exactly satisfy"):
            ModelTokenOutput(
                token="token",
                proposal_sha256=_proposal_sha("token"),
                base_logits=zeros,
                residual_logits=zeros,
                delta_logits=zeros,
                utility_logits=tuple(miswired),
                numeric_mode=F2_MODEL_NUMERIC_MODE,
            )

        with self.assertRaisesRegex(CausalValidationError, "numeric_mode"):
            ModelTokenOutput(
                token="token",
                proposal_sha256=_proposal_sha("token"),
                base_logits=zeros,
                residual_logits=zeros,
                delta_logits=zeros,
                utility_logits=zeros,
                numeric_mode="16-mixed",
            )

    def test_model_output_preserves_device_delta_near_cancellation(self) -> None:
        # Captured on both torch 2.1 CPU and CUDA.  Torch produces the first
        # delta while canonical Python math.tanh produces the adjacent float:
        # -0.9884083271026611 vs -0.9884082674980164.  Replacing the transmitted
        # device delta before the final addition would turn the valid zero
        # utility into 5.960464477539063e-08 and fail by many ULPs near zero.
        residual = -2.5724000930786133
        device_delta = -0.9884083271026611
        base = 0.9884083271026611
        repeated_residual = tuple(residual for _ in range(CANDIDATE_COUNT))
        repeated_delta = tuple(device_delta for _ in range(CANDIDATE_COUNT))
        repeated_base = tuple(base for _ in range(CANDIDATE_COUNT))
        zeros = tuple(0.0 for _ in range(CANDIDATE_COUNT))

        output = ModelTokenOutput(
            token="near-cancellation",
            proposal_sha256=_proposal_sha("near-cancellation"),
            base_logits=repeated_base,
            residual_logits=repeated_residual,
            delta_logits=repeated_delta,
            utility_logits=zeros,
            numeric_mode=F2_MODEL_NUMERIC_MODE,
        )
        self.assertEqual(output.utility_logits, zeros)
        self.assertNotEqual(
            _float32(math.tanh(residual)), device_delta
        )

        drifted_utility = list(zeros)
        drifted_utility[3] = _float32(2.0**-149)
        with self.assertRaisesRegex(CausalValidationError, "do not exactly satisfy"):
            ModelTokenOutput(
                token="near-cancellation",
                proposal_sha256=_proposal_sha("near-cancellation"),
                base_logits=repeated_base,
                residual_logits=repeated_residual,
                delta_logits=repeated_delta,
                utility_logits=tuple(drifted_utility),
                numeric_mode=F2_MODEL_NUMERIC_MODE,
            )

    def test_independent_branches_and_dual_simulator_states(self) -> None:
        rows = [
            ManifestRow("log", 1_000_000, "t0", True),
            ManifestRow("log", 1_500_000, "t1", True),
        ]
        base_t0 = [0.0] * CANDIDATE_COUNT
        base_t0[0] = 0.5
        critic_t0 = [0.0] * CANDIDATE_COUNT
        critic_t0[1] = 2.0
        zeros = tuple(0.0 for _ in range(CANDIDATE_COUNT))
        artifacts = {
            "t0": _artifact("t0", base_logits=tuple(base_t0)),
            "t1": _artifact("t1", base_logits=zeros),
        }
        outputs = {
            "t0": _output(
                "t0", tuple(critic_t0), base_logits=tuple(base_t0)
            ),
            "t1": _output("t1", zeros, base_logits=zeros),
        }
        observed_pairs = []

        def comfort(branch, past, current, delta_time_us):
            self.assertEqual(past["simulator"], "official")
            self.assertEqual(current["simulator"], "v2")
            self.assertEqual(delta_time_us, 500_000)
            observed_pairs.append((branch, past["index"], current["index"]))
            if branch == "base" and past["index"] == 0 and current["index"] == 2:
                return 1.0
            if branch == "critic" and past["index"] == 1 and current["index"] == 3:
                return 1.0
            return 0.0

        results = DualBranchCausalStateMachine(comfort).evaluate_log(
            rows, artifacts, outputs
        )
        self.assertEqual(results[0].base_selected_index, 0)
        self.assertEqual(results[0].critic_selected_index, 1)
        self.assertEqual(results[1].base_selected_index, 2)
        self.assertEqual(results[1].critic_selected_index, 3)
        self.assertEqual(
            results[1].base_target_scores, artifacts["t1"].target_scores[2]
        )
        self.assertEqual(
            results[1].critic_target_scores, artifacts["t1"].target_scores[3]
        )
        self.assertEqual(
            results[1].base_selected_truth,
            _float32((14.0 * (2.0 / 31.0) + 2.0) / 16.0),
        )
        self.assertEqual(
            results[1].critic_selected_truth,
            _float32((14.0 * (3.0 / 31.0) + 2.0) / 16.0),
        )
        self.assertEqual(results[1].oracle_aggregate, 1.0)
        self.assertIn(("base", 0, 2), observed_pairs)
        self.assertIn(("critic", 1, 3), observed_pairs)

    def test_same_selected_candidate_still_gets_independent_state_copies(self) -> None:
        rows = [
            ManifestRow("log", 1_000_000, "t0", True),
            ManifestRow("log", 1_500_000, "t1", True),
        ]
        preferred = [0.0] * CANDIDATE_COUNT
        preferred[0] = 2.0
        zeros = tuple(0.0 for _ in range(CANDIDATE_COUNT))
        artifacts = {
            "t0": _artifact("t0", base_logits=tuple(preferred)),
            "t1": _artifact("t1", base_logits=zeros),
        }
        outputs = {
            "t0": _output(
                "t0", tuple(preferred), base_logits=tuple(preferred)
            ),
            "t1": _output("t1", zeros, base_logits=zeros),
        }
        seen_before_mutation = []

        def comfort(branch, past, current, delta_time_us):
            seen_before_mutation.append(
                (
                    branch,
                    current["index"],
                    tuple(past["seen_by"]),
                    tuple(current.get("seen_by", ())),
                )
            )
            past["seen_by"].append((branch, current["index"]))
            current.setdefault("seen_by", []).append(branch)
            return 0.0

        machine = DualBranchCausalStateMachine(comfort)
        first = machine.evaluate_log(rows, artifacts, outputs)
        second = machine.evaluate_log(rows, artifacts, outputs)
        self.assertEqual(
            canonical_evaluation_bytes(first), canonical_evaluation_bytes(second)
        )
        self.assertEqual(len(seen_before_mutation), 4 * CANDIDATE_COUNT)
        self.assertTrue(
            all(
                past_seen == () and current_seen == ()
                for _, _, past_seen, current_seen in seen_before_mutation
            )
        )
        self.assertTrue(
            all(
                "seen_by" not in candidate
                for candidate in artifacts["t1"].calibration_states
            )
        )

    def test_pairwise_target_ties_excluded_and_utility_ties_wrong(self) -> None:
        correct, total = micro_pair_counts([0.0, 0.0, 1.0], [0.0, 0.5, 1.0])
        self.assertEqual((correct, total), (2, 3))
        correct, total = micro_pair_counts([0.0, 0.5, 1.0], [0.0, 0.0, 1.0])
        self.assertEqual((correct, total), (2, 2))


class ShardingAggregationAndRankingTest(unittest.TestCase):
    def test_one_two_eight_shards_are_byte_identical(self) -> None:
        rows = []
        for log_index in range(8):
            for token_index in range(2):
                rows.append(
                    ManifestRow(
                        f"log-{log_index}",
                        1_000_000 + token_index * 500_000,
                        f"token-{log_index}-{token_index}",
                        True,
                    )
                )
        manifest = _manifest(rows)
        artifacts = {row.token: _artifact(row.token) for row in rows}
        outputs = {row.token: _output(row.token) for row in rows}

        def branch_comfort(branch, past, current, delta_time_us):
            self.assertEqual(past["simulator"], "official")
            self.assertEqual(current["simulator"], "v2")
            self.assertEqual(delta_time_us, 500_000)
            if current["index"] == CANDIDATE_COUNT - 1:
                return 0.75 if branch == "critic" else 0.25
            return 0.0

        serialized = []
        metric_payloads = []
        for shard_count in (1, 2, 8):
            results = evaluate_manifest_sharded(
                manifest,
                artifacts,
                outputs,
                branch_comfort,
                num_shards=shard_count,
            )
            serialized.append(canonical_evaluation_bytes(results))
            metrics = aggregate_results(
                results, expected_tokens=manifest.tokens, fatal_count=0
            )
            metric_payloads.append(canonical_metrics_bytes(metrics))
        self.assertEqual(serialized[0], serialized[1])
        self.assertEqual(serialized[0], serialized[2])
        self.assertEqual(metric_payloads[0], metric_payloads[1])
        self.assertEqual(metric_payloads[0], metric_payloads[2])

        assignments = deterministic_log_assignments(manifest.rows, 8)
        self.assertEqual(
            assignments,
            deterministic_log_assignments(tuple(reversed(manifest.rows)), 8),
        )
        assigned_logs = [log_name for shard in assignments for log_name in shard]
        self.assertEqual(len(assigned_logs), len(set(assigned_logs)))
        self.assertEqual(set(assigned_logs), set(manifest.logs))

    @staticmethod
    def _result(token, *, correct, total):
        critic = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.9)
        base = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.8)
        return TokenCausalResult(
            log_name="log",
            time_us=1_000_000 + len(token),
            token=token,
            future_camera_valid=True,
            critic_continuous=False,
            base_continuous=False,
            critic_selected_index=1,
            base_selected_index=0,
            critic_selected_q=0.9,
            base_selected_q=0.8,
            critic_selected_probability=0.9,
            base_selected_probability=0.8,
            critic_selected_truth=0.9,
            base_selected_truth=0.8,
            critic_target_scores=critic,
            base_target_scores=base,
            oracle_aggregate=0.95,
            pairwise_correct=correct,
            pairwise_comparable=total,
        )

    def test_micro_pair_aggregation_and_fatal_coverage_gates(self) -> None:
        results = [
            self._result("a", correct=1, total=1),
            self._result("bb", correct=0, total=3),
        ]
        metrics = aggregate_results(
            results, expected_tokens=["a", "bb"], fatal_count=0
        )
        self.assertEqual(metrics.critic_pairwise_accuracy, 0.25)

        incomplete = aggregate_results(
            results[:1], expected_tokens=["a", "bb"], fatal_count=1
        )
        decision = evaluate_formal_gates(incomplete)
        self.assertFalse(decision.passed)
        self.assertIn("coverage_fraction", decision.failures)
        self.assertIn("fatal_count", decision.failures)

    def test_gt_means_headroom_hard_deltas_and_causal_q_errors(self) -> None:
        critic_rows = (
            (1.0, 0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.9),
            (0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.7),
        )
        base_rows = (
            (0.8, 0.7, 0.9, 0.1, 0.5, 0.5, 0.5, 1.0, 0.5),
            (0.2, 0.5, 0.4, 0.6, 0.5, 0.5, 0.5, 1.0, 0.5),
        )
        results = []
        for index in range(2):
            results.append(
                TokenCausalResult(
                    log_name="log",
                    time_us=1_000_000 + 500_000 * index,
                    token=f"token-{index}",
                    future_camera_valid=index == 0,
                    critic_continuous=index == 1,
                    base_continuous=index == 1,
                    critic_selected_index=1,
                    base_selected_index=0,
                    # Deliberately disagree with raw probabilities.  The
                    # causal q used for selection must also define error.
                    critic_selected_q=(0.2, 0.8)[index],
                    base_selected_q=(0.6, 0.4)[index],
                    critic_selected_probability=(0.99, 0.0)[index],
                    base_selected_probability=(0.1, 1.0)[index],
                    critic_selected_truth=(0.9, 0.7)[index],
                    base_selected_truth=0.5,
                    critic_target_scores=critic_rows[index],
                    base_target_scores=base_rows[index],
                    oracle_aggregate=(1.0, 0.9)[index],
                    pairwise_correct=1,
                    pairwise_comparable=2,
                )
            )

        metrics = aggregate_results(
            results,
            expected_tokens=("token-0", "token-1"),
            fatal_count=0,
        )
        self.assertAlmostEqual(metrics.critic_selected_score, 0.8)
        self.assertAlmostEqual(metrics.base_selected_score, 0.5)
        self.assertAlmostEqual(metrics.oracle_score, 0.95)
        self.assertAlmostEqual(metrics.proposal_headroom_closed, 2.0 / 3.0)
        self.assertAlmostEqual(metrics.critic_score_error, 0.4)
        self.assertAlmostEqual(metrics.base_score_error, 0.1)
        for actual, expected in zip(
            metrics.hard_factor_deltas, (0.0, -0.1, 0.1, -0.1)
        ):
            self.assertAlmostEqual(actual, expected)

    def test_checkpoint_eligibility_then_score_and_epoch_ranking(self) -> None:
        common = {
            "manifest_sha256": ONE_SHA,
            "artifact_sha256": TWO_SHA,
            "numeric_mode": F2_MODEL_NUMERIC_MODE,
        }
        early = CheckpointEvaluation(
            epoch=0,
            checkpoint_sha256="a" * 64,
            metrics=_passing_metrics(selected_score=0.90),
            **common,
        )
        same_score_later = CheckpointEvaluation(
            epoch=1,
            checkpoint_sha256="b" * 64,
            metrics=_passing_metrics(selected_score=0.90),
            **common,
        )
        lower = CheckpointEvaluation(
            epoch=2,
            checkpoint_sha256="c" * 64,
            metrics=_passing_metrics(selected_score=0.89),
            **common,
        )
        higher_but_fatal = CheckpointEvaluation(
            epoch=3,
            checkpoint_sha256="d" * 64,
            metrics=_passing_metrics(selected_score=0.94, fatal_count=1),
            **common,
        )
        evaluations = [higher_but_fatal, lower, same_score_later, early]
        ranking = rank_checkpoints(evaluations)
        self.assertEqual(ranking.selected, early)
        self.assertEqual([item.epoch for item in ranking.eligible], [0, 1, 2])
        self.assertEqual([item[0].epoch for item in ranking.ineligible], [3])

        with self.assertRaisesRegex(CausalValidationError, "epoch coverage"):
            rank_checkpoints(evaluations[:-1])

        base_drift = replace(
            lower,
            metrics=_passing_metrics(
                selected_score=0.90,
                base_score=0.84,
            ),
        )
        with self.assertRaisesRegex(CausalValidationError, "base selected score"):
            rank_checkpoints(
                [higher_but_fatal, base_drift, same_score_later, early]
            )

        oracle_drift = replace(
            lower,
            metrics=_passing_metrics(
                selected_score=0.89,
                oracle_score=0.96,
            ),
        )
        with self.assertRaisesRegex(CausalValidationError, "oracle score"):
            rank_checkpoints(
                [higher_but_fatal, oracle_drift, same_score_later, early]
            )

        base_error_drift = replace(
            lower,
            metrics=replace(
                lower.metrics,
                base_score_error=0.08,
                critic_score_error_excess=0.0,
            ),
        )
        with self.assertRaisesRegex(CausalValidationError, "base score error"):
            rank_checkpoints(
                [higher_but_fatal, base_error_drift, same_score_later, early]
            )

        count_drift = replace(
            lower,
            metrics=replace(
                lower.metrics,
                expected_token_count=20,
                observed_token_count=20,
            ),
        )
        with self.assertRaisesRegex(CausalValidationError, "token count"):
            rank_checkpoints(
                [higher_but_fatal, count_drift, same_score_later, early]
            )

        future_valid_drift = replace(
            lower,
            metrics=replace(
                lower.metrics, future_camera_valid_fraction=0.99
            ),
        )
        with self.assertRaisesRegex(CausalValidationError, "future-camera"):
            rank_checkpoints(
                [higher_but_fatal, future_valid_drift, same_score_later, early]
            )

        # Causal pairwise truth follows the critic branch carry, so its
        # comparable-pair count may legitimately differ between checkpoints.
        denominator_drift = replace(
            lower,
            metrics=replace(
                lower.metrics,
                critic_pairwise_accuracy=0.75,
                critic_pairwise_correct=78,
                critic_pairwise_comparable=104,
            ),
        )
        rank_checkpoints(
            [higher_but_fatal, denominator_drift, same_score_later, early]
        )

    def test_every_formal_gate_and_inclusive_boundaries(self) -> None:
        with self.assertRaises(TypeError):
            FORMAL_GATE_THRESHOLDS["critic_delta_min"] = -1.0

        boundary = AggregateMetrics(
            expected_token_count=100,
            observed_token_count=100,
            coverage_fraction=1.0,
            fatal_count=0,
            future_camera_valid_fraction=0.99,
            critic_selected_score=0.875,
            base_selected_score=0.85,
            oracle_score=0.95,
            critic_delta=0.025,
            critic_oracle_regret=0.075,
            critic_pairwise_accuracy=0.55,
            critic_pairwise_correct=55,
            critic_pairwise_comparable=100,
            critic_score_error=0.09,
            base_score_error=0.07,
            critic_score_error_excess=0.02,
            proposal_headroom_closed=0.25,
            hard_factor_deltas=(-0.001, -0.001, -0.001, -0.001),
        )
        self.assertTrue(evaluate_formal_gates(boundary).passed)

        base = _passing_metrics()
        cases = (
            (
                replace(base, future_camera_valid_fraction=0.98),
                "future_camera_valid_fraction",
            ),
            (_passing_metrics(selected_score=0.854), "critic_delta"),
            (
                replace(
                    base,
                    critic_pairwise_accuracy=0.50,
                    critic_pairwise_correct=50,
                ),
                "critic_pairwise_accuracy",
            ),
            (_passing_metrics(selected_score=0.87), "proposal_headroom_closed"),
            (
                replace(base, hard_factor_deltas=(-0.002, 0.0, 0.0, 0.0)),
                "hard_factor_delta_nc",
            ),
            (
                replace(
                    base,
                    critic_score_error=0.10,
                    critic_score_error_excess=0.03,
                ),
                "critic_score_error_excess",
            ),
        )
        for metrics, expected_failure in cases:
            with self.subTest(expected_failure=expected_failure):
                decision = evaluate_formal_gates(metrics)
                self.assertFalse(decision.passed)
                self.assertIn(expected_failure, decision.failures)


if __name__ == "__main__":
    unittest.main()
