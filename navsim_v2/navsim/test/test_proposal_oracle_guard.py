"""Pure unit tests for proposal-oracle partition and evidence guards."""

import hashlib
import unittest
from types import SimpleNamespace

from navsim.planning.script.proposal_oracle_guard import (
    apply_partition_allowlist,
    attach_partition_evidence,
    check_leakage_guard,
    identifier_evidence,
    metric_coverage,
    summarize,
)


class ProposalOracleGuardTest(unittest.TestCase):
    def test_accepts_only_declared_split_phase_partition_pairs(self) -> None:
        check_leakage_guard("navtrain", "development", "validation")
        check_leakage_guard("navtest", "final", "full")

        rejected = (
            ("navtrain", "development", "full"),
            ("navtrain", "final", "validation"),
            ("navtest", "final", "validation"),
            ("navtest", "development", "full"),
            ("navhard", "final", "full"),
        )
        for split, phase, partition in rejected:
            with self.subTest(split=split, phase=phase, partition=partition):
                with self.assertRaises(ValueError):
                    check_leakage_guard(split, phase, partition)

    def test_validation_partition_intersects_scene_filter_and_val_logs(self) -> None:
        scene_filter = SimpleNamespace(
            log_names=["train-log", "held-b", "held-a", "held-a"]
        )
        effective = apply_partition_allowlist(
            scene_filter,
            partition="validation",
            val_logs=["held-b", "other-held", "held-a"],
        )

        self.assertEqual(effective, ["held-a", "held-b"])
        self.assertEqual(scene_filter.log_names, effective)

        unrestricted_filter = SimpleNamespace(log_names=None)
        unrestricted = apply_partition_allowlist(
            unrestricted_filter,
            partition="validation",
            val_logs=["held-b", "held-a"],
        )
        self.assertEqual(unrestricted, ["held-a", "held-b"])
        self.assertEqual(unrestricted_filter.log_names, unrestricted)

    def test_validation_partition_rejects_missing_or_disjoint_allowlist(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty cfg.val_logs"):
            apply_partition_allowlist(
                SimpleNamespace(log_names=None),
                partition="validation",
                val_logs=[],
            )
        with self.assertRaisesRegex(RuntimeError, "no logs in common"):
            apply_partition_allowlist(
                SimpleNamespace(log_names=["train-log"]),
                partition="validation",
                val_logs=["held-log"],
            )

    def test_full_partition_does_not_apply_val_logs(self) -> None:
        scene_filter = SimpleNamespace(log_names=["test-b", "test-a"])
        effective = apply_partition_allowlist(
            scene_filter,
            partition="full",
            val_logs=["must-not-be-used"],
        )

        self.assertEqual(effective, ["test-a", "test-b"])
        self.assertEqual(scene_filter.log_names, ["test-b", "test-a"])

    def test_digest_and_csv_summary_evidence_are_deterministic(self) -> None:
        allowlist_evidence = identifier_evidence(
            ["held-b", "held-a"], prefix="allowlist_log"
        )
        expected_digest = hashlib.sha256(b"held-a\nheld-b\n").hexdigest()
        self.assertEqual(allowlist_evidence["allowlist_log_count"], 2)
        self.assertEqual(
            allowlist_evidence["allowlist_log_sha256"], expected_digest
        )

        _, _, _, token_evidence = metric_coverage(
            ["token-b", "token-a"], {"token-a", "token-b"}
        )
        evidence = {
            "partition": "validation",
            **allowlist_evidence,
            **token_evidence,
        }
        rows = [
            {
                "token": "token-a",
                "split": "navtrain",
                "selected_index": 0,
                "selected_epdms": 0.75,
            }
        ]
        attach_partition_evidence(rows, evidence)
        for key, value in evidence.items():
            self.assertEqual(rows[0][key], value)

        summary = summarize(
            rows, "navtrain", "development", evidence
        )
        for key, value in evidence.items():
            self.assertEqual(summary[key], value)
        self.assertEqual(summary["requested_token_count"], 2)
        self.assertEqual(summary["valid_token_count"], 2)
        self.assertEqual(summary["missing_token_count"], 0)
        self.assertEqual(summary["means"], {"selected_epdms": 0.75})

    def test_metric_coverage_evidence_exposes_missing_tokens(self) -> None:
        indices, valid, missing, evidence = metric_coverage(
            ["token-c", "token-a", "token-b"],
            {"token-a", "token-c"},
            require_complete=False,
        )

        self.assertEqual(indices, [0, 1])
        self.assertEqual(valid, ["token-c", "token-a"])
        self.assertEqual(missing, ["token-b"])
        self.assertEqual(evidence["requested_token_count"], 3)
        self.assertEqual(evidence["valid_token_count"], 2)
        self.assertEqual(evidence["missing_token_count"], 1)
        self.assertEqual(
            evidence["missing_token_sha256"],
            hashlib.sha256(b"token-b\n").hexdigest(),
        )
        with self.assertRaisesRegex(
            RuntimeError, "complete metric-cache coverage"
        ):
            metric_coverage(
                ["token-c", "token-a", "token-b"],
                {"token-a", "token-c"},
            )

    def test_metric_coverage_accepts_only_complete_requested_partition(self) -> None:
        _, valid, missing, evidence = metric_coverage(
            ["token-b", "token-a"], {"token-a", "token-b", "unused"}
        )

        self.assertEqual(valid, ["token-b", "token-a"])
        self.assertEqual(missing, [])
        self.assertEqual(evidence["requested_token_count"], 2)
        self.assertEqual(evidence["valid_token_count"], 2)
        self.assertEqual(evidence["missing_token_count"], 0)


if __name__ == "__main__":
    unittest.main()
