"""CPU-only contract tests for the F2 NAVSIM comfort adapter."""

from __future__ import annotations

import unittest

import numpy as np

from navsim.planning.validation.f2_navsim_comfort import (
    DriveJepaExtendedComfort,
    F2NavsimComfortError,
)


def _states(offset: float = 0.0) -> np.ndarray:
    states = np.zeros((41, 11), dtype="<f8")
    states[:, 0] = np.arange(41, dtype=np.float64) + offset
    return states


class F2NavsimComfortTest(unittest.TestCase):
    def test_exact_half_second_overlap_and_boolean_result(self) -> None:
        observed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def kernel(
            past: np.ndarray, current: np.ndarray, time_point_s: np.ndarray
        ) -> np.ndarray:
            observed.append((past.copy(), current.copy(), time_point_s.copy()))
            return np.array([True], dtype=np.bool_)

        adapter = DriveJepaExtendedComfort(kernel)
        value = adapter("critic", _states(10.0), _states(20.0), 500_000)

        self.assertEqual(value, 1.0)
        past, current, times = observed[0]
        self.assertEqual(past.shape, (1, 36, 11))
        self.assertEqual(current.shape, (1, 36, 11))
        self.assertEqual(past[0, 0, 0], 15.0)
        self.assertEqual(current[0, -1, 0], 55.0)
        np.testing.assert_array_equal(times, np.arange(36) * 0.1)

    def test_false_is_exact_zero_and_branch_does_not_change_math(self) -> None:
        adapter = DriveJepaExtendedComfort(
            lambda left, right, times: np.array([False], dtype=np.bool_)
        )
        self.assertEqual(adapter("base", _states(), _states(), 550_000), 0.0)

    def test_rejects_noncanonical_continuity_intervals(self) -> None:
        adapter = DriveJepaExtendedComfort(
            lambda left, right, times: np.array([True], dtype=np.bool_)
        )
        for delta in (0, 50_000, 550_001, True, 0.5):
            with self.subTest(delta=delta):
                with self.assertRaises(F2NavsimComfortError):
                    adapter("critic", _states(), _states(), delta)  # type: ignore[arg-type]

    def test_rejects_unknown_branch(self) -> None:
        adapter = DriveJepaExtendedComfort(
            lambda left, right, times: np.array([True], dtype=np.bool_)
        )
        with self.assertRaisesRegex(F2NavsimComfortError, "branch"):
            adapter("shared", _states(), _states(), 500_000)

    def test_rejects_state_shape_dtype_layout_and_nonfinite(self) -> None:
        adapter = DriveJepaExtendedComfort(
            lambda left, right, times: np.array([True], dtype=np.bool_)
        )
        bad_values = (
            np.zeros((40, 11), dtype="<f8"),
            np.zeros((41, 11), dtype="<f4"),
            np.zeros((41, 22), dtype="<f8")[:, ::2],
            np.full((41, 11), np.nan, dtype="<f8"),
        )
        for value in bad_values:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(F2NavsimComfortError):
                    adapter("critic", value, _states(), 500_000)

    def test_rejects_kernel_shape_and_dtype_drift(self) -> None:
        bad_results = (
            np.array(True, dtype=np.bool_),
            np.array([1], dtype=np.int64),
            np.array([True, False], dtype=np.bool_),
        )
        for result in bad_results:
            with self.subTest(result=result):
                adapter = DriveJepaExtendedComfort(
                    lambda left, right, times, value=result: value
                )
                with self.assertRaisesRegex(F2NavsimComfortError, "one boolean"):
                    adapter("critic", _states(), _states(), 500_000)

    def test_wraps_kernel_failures(self) -> None:
        def failed_kernel(
            left: np.ndarray, right: np.ndarray, times: np.ndarray
        ) -> np.ndarray:
            raise AssertionError("synthetic failure")

        adapter = DriveJepaExtendedComfort(failed_kernel)
        with self.assertRaisesRegex(
            F2NavsimComfortError, "AssertionError: synthetic failure"
        ):
            adapter("critic", _states(), _states(), 500_000)


if __name__ == "__main__":
    unittest.main()
