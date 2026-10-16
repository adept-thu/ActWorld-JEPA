"""Regression tests for deterministic NAVSIM scorer-pool recovery."""

import unittest
from concurrent.futures.process import BrokenProcessPool

from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import (
    DriveJEPAAgent,
)


def _score_values(chunk):
    return [point["value"] for point in chunk]


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


class _BrokenPool:
    def __init__(self):
        self.shutdown_calls = []

    def submit(self, function, chunk):
        raise BrokenProcessPool("simulated scorer death")

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))


class _HealthyPool:
    def __init__(self):
        self.submissions = []

    def submit(self, function, chunk):
        self.submissions.append(list(chunk))
        return _ImmediateFuture(function(chunk))

    def shutdown(self, wait=True, cancel_futures=False):
        pass


class ScorePoolRecoveryTest(unittest.TestCase):
    def test_broken_pool_is_rebuilt_once_and_replays_identical_batch(self):
        agent = object.__new__(DriveJEPAAgent)
        broken_pool = _BrokenPool()
        healthy_pool = _HealthyPool()
        agent.persistent_pool = broken_pool
        agent._score_timeout_seconds = 1.0
        agent._score_slow_log_seconds = float("inf")
        agent._create_score_pool = lambda: healthy_pool
        data_points = [
            {"token": f"/cache/token-{index}.xz", "value": index}
            for index in range(5)
        ]

        result = agent._worker_map_with_recovery(
            data_points,
            _score_values,
            "test scoring",
        )

        self.assertEqual(result, [0, 1, 2, 3, 4])
        self.assertEqual(broken_pool.shutdown_calls, [(False, True)])
        self.assertIs(agent.persistent_pool, healthy_pool)
        self.assertEqual(
            healthy_pool.submissions,
            [data_points[0:2], data_points[2:4], data_points[4:5]],
        )


if __name__ == "__main__":
    unittest.main()
