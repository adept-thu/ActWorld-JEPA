import csv
import json
import tempfile
import unittest
from pathlib import Path

from navsim.planning.validation.navhard_result import (
    COMBINED_TOKEN,
    PUBLISHED_COMPARISON_TOKEN,
    STAGE_ONE_TOKEN,
    NavHardResultError,
    load_navhard_result,
    write_result_record,
)


class NavHardResultTest(unittest.TestCase):
    @staticmethod
    def _write(path: Path, rows) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("token", "valid", "score"))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _rows():
        return [
            {"token": "scene-b", "valid": "True", "score": "0.4"},
            {"token": "scene-a", "valid": "True", "score": "0.8"},
            {"token": STAGE_ONE_TOKEN, "valid": "True", "score": "0.71454"},
            {
                "token": PUBLISHED_COMPARISON_TOKEN,
                "valid": "True",
                "score": "0.528421",
            },
            {"token": COMBINED_TOKEN, "valid": "True", "score": "0.376922"},
        ]

    def test_extracts_stage_two_without_confusing_combined_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "result.csv"
            self._write(csv_path, self._rows())
            result = load_navhard_result(csv_path, expected_scenarios=2)
            self.assertEqual(result.scenario_count, 2)
            self.assertEqual(result.failed_count, 0)
            self.assertAlmostEqual(
                result.published_comparison_stage_two_epdms, 0.528421
            )
            self.assertAlmostEqual(
                result.combined_pseudo_closed_loop_epdms, 0.376922
            )
            self.assertNotEqual(
                result.published_comparison_stage_two_epdms,
                result.combined_pseudo_closed_loop_epdms,
            )

            record_path = Path(directory) / "evidence.json"
            write_result_record(result, record_path)
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["csv_sha256"], result.csv_sha256)
            self.assertEqual(record["scenario_count"], 2)

    def test_rejects_failed_duplicate_or_incomplete_results(self) -> None:
        mutations = {
            "failed": lambda rows: rows.__setitem__(
                0, {"token": "scene-b", "valid": "False", "score": "0.4"}
            ),
            "duplicate": lambda rows: rows.__setitem__(
                1, {"token": "scene-b", "valid": "True", "score": "0.8"}
            ),
            "missing summary": lambda rows: rows.pop(),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                rows = self._rows()
                mutate(rows)
                path = Path(directory) / "result.csv"
                self._write(path, rows)
                with self.assertRaises(NavHardResultError):
                    load_navhard_result(path, expected_scenarios=2)


if __name__ == "__main__":
    unittest.main()
