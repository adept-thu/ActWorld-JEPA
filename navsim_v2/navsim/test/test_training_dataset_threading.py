import threading
import time
import unittest
from typing import Optional
from unittest.mock import patch

from torch.utils.data import DataLoader

from navsim.planning.training.dataset import (
    Dataset,
    InProcessBatchPrefetchDataLoader,
    _resolve_in_process_sample_threads,
    resolve_in_process_prefetch_batches,
)


class _RecordingDataset(Dataset):
    def __init__(self, threads: int) -> None:
        self._in_process_sample_threads = threads
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def __len__(self) -> int:
        return 16

    def __getitem__(self, idx: int):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        time.sleep(0.01)
        with self._lock:
            self._active -= 1
        return ({"index": idx}, {"target": idx * 2})


class _BatchRecordingDataset:
    def __init__(self, fail_on_batch: Optional[int] = None) -> None:
        self._lock = threading.Lock()
        self._batch_calls = 0
        self.fail_on_batch = fail_on_batch
        self.second_batch_started = threading.Event()
        self.second_batch_finished = threading.Event()
        self.third_batch_started = threading.Event()
        self.release_second_batch = threading.Event()

    def __len__(self) -> int:
        return 6

    def __getitem__(self, idx: int):
        return idx

    def __getitems__(self, indices):
        with self._lock:
            self._batch_calls += 1
            batch_call = self._batch_calls
        if batch_call == 2:
            self.second_batch_started.set()
            self.release_second_batch.wait(timeout=2.0)
            self.second_batch_finished.set()
        if batch_call == 3:
            self.third_batch_started.set()
        if self.fail_on_batch == batch_call:
            raise RuntimeError("prefetch failure")
        return list(indices)


class TestTrainingDatasetThreading(unittest.TestCase):
    def test_threaded_getitems_preserves_sampler_order(self) -> None:
        dataset = _RecordingDataset(threads=4)

        batch = dataset.__getitems__([7, 2, 9, 1])

        self.assertEqual([item[0]["index"] for item in batch], [7, 2, 9, 1])
        self.assertGreater(dataset.max_active, 1)

    def test_torch_dataloader_uses_batched_getitems_hook(self) -> None:
        dataset = _RecordingDataset(threads=4)
        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=0,
            collate_fn=lambda items: items,
        )

        batch = next(iter(loader))

        self.assertEqual([item[0]["index"] for item in batch], [0, 1, 2, 3])
        self.assertGreater(dataset.max_active, 1)

    def test_single_thread_getitems_is_sequential(self) -> None:
        dataset = _RecordingDataset(threads=1)

        batch = dataset.__getitems__([3, 4])

        self.assertEqual([item[0]["index"] for item in batch], [3, 4])
        self.assertEqual(dataset.max_active, 1)

    def test_thread_count_is_positive_integer(self) -> None:
        with patch.dict("os.environ", {"ACTWORLD_DATASET_THREADS": "8"}):
            self.assertEqual(_resolve_in_process_sample_threads(), 8)
        with patch.dict("os.environ", {"ACTWORLD_DATASET_THREADS": "0"}):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                _resolve_in_process_sample_threads()
        with patch.dict("os.environ", {"ACTWORLD_DATASET_THREADS": "many"}):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                _resolve_in_process_sample_threads()

    def test_one_batch_prefetch_preserves_order_and_overlaps(self) -> None:
        dataset = _BatchRecordingDataset()
        loader = InProcessBatchPrefetchDataLoader(
            dataset,
            batch_size=2,
            num_workers=0,
            collate_fn=lambda items: items,
            prefetch_batches=1,
        )
        iterator = iter(loader)

        self.assertEqual(next(iterator), [0, 1])
        self.assertTrue(dataset.second_batch_started.wait(timeout=1.0))
        dataset.release_second_batch.set()
        self.assertEqual(next(iterator), [2, 3])
        self.assertEqual(next(iterator), [4, 5])
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_prefetch_does_not_materialize_beyond_queue_depth(self) -> None:
        dataset = _BatchRecordingDataset()
        loader = InProcessBatchPrefetchDataLoader(
            dataset,
            batch_size=2,
            num_workers=0,
            collate_fn=lambda items: items,
            prefetch_batches=1,
        )
        iterator = iter(loader)

        self.assertEqual(next(iterator), [0, 1])
        self.assertTrue(dataset.second_batch_started.wait(timeout=1.0))
        dataset.release_second_batch.set()
        self.assertTrue(dataset.second_batch_finished.wait(timeout=1.0))
        self.assertFalse(dataset.third_batch_started.wait(timeout=0.1))

        self.assertEqual(next(iterator), [2, 3])
        self.assertTrue(dataset.third_batch_started.wait(timeout=1.0))
        self.assertEqual(next(iterator), [4, 5])
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_prefetch_propagates_dataset_exception(self) -> None:
        dataset = _BatchRecordingDataset(fail_on_batch=2)
        dataset.release_second_batch.set()
        loader = InProcessBatchPrefetchDataLoader(
            dataset,
            batch_size=2,
            num_workers=0,
            collate_fn=lambda items: items,
            prefetch_batches=1,
        )
        iterator = iter(loader)

        self.assertEqual(next(iterator), [0, 1])
        with self.assertRaisesRegex(RuntimeError, "prefetch failure"):
            next(iterator)

    def test_prefetch_depth_is_zero_or_one(self) -> None:
        with patch.dict(
            "os.environ", {"ACTWORLD_IN_PROCESS_PREFETCH_BATCHES": "0"}
        ):
            self.assertEqual(resolve_in_process_prefetch_batches(), 0)
        with patch.dict(
            "os.environ", {"ACTWORLD_IN_PROCESS_PREFETCH_BATCHES": "1"}
        ):
            self.assertEqual(resolve_in_process_prefetch_batches(), 1)
        for value in ("2", "many"):
            with patch.dict(
                "os.environ", {"ACTWORLD_IN_PROCESS_PREFETCH_BATCHES": value}
            ):
                with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                    resolve_in_process_prefetch_batches()


if __name__ == "__main__":
    unittest.main()
