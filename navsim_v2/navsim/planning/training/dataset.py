from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import logging
import pickle
import gzip
import os
import queue
import sys
import threading

import torch
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

logger = logging.getLogger(__name__)

_PREFETCH_BATCH = object()
_PREFETCH_END = object()
_PREFETCH_ERROR = object()


def _resolve_in_process_sample_threads() -> int:
    """Return the bounded thread count for batched in-process sample loading.

    NAVSIM's ``SceneLoader`` is several GiB per DDP rank.  DataLoader worker
    processes therefore multiply host memory even when their start method is
    ``fork``.  With ``num_workers=0``, PyTorch can instead call a dataset's
    ``__getitems__`` hook once per batch.  The opt-in thread count below lets
    one rank overlap independent scene/image decoding without copying the
    loader or changing the sampler's order.
    """

    raw_value = os.environ.get("ACTWORLD_DATASET_THREADS", "1")
    try:
        threads = int(raw_value)
    except ValueError as error:
        raise ValueError(
            "ACTWORLD_DATASET_THREADS must be a positive integer"
        ) from error
    if threads < 1:
        raise ValueError("ACTWORLD_DATASET_THREADS must be a positive integer")
    return threads


def resolve_in_process_prefetch_batches() -> int:
    """Return the bounded number of batches prepared by one rank-local thread.

    A NAVSIM batch at the formal Stage-1 size can occupy several GiB per rank.
    Depth one is therefore the only admitted overlap: it keeps the exact
    single-process DataLoader and sampler semantics while constructing the
    next batch during the current optimizer step without multiplying the
    multi-GiB ``SceneLoader`` into worker processes.
    """

    raw_value = os.environ.get("ACTWORLD_IN_PROCESS_PREFETCH_BATCHES", "0")
    try:
        batches = int(raw_value)
    except ValueError as error:
        raise ValueError(
            "ACTWORLD_IN_PROCESS_PREFETCH_BATCHES must be 0 or 1"
        ) from error
    if batches not in {0, 1}:
        raise ValueError("ACTWORLD_IN_PROCESS_PREFETCH_BATCHES must be 0 or 1")
    return batches


class _InProcessBatchPrefetchIterator:
    """Prepare one DataLoader batch ahead without changing its yielded order."""

    def __init__(self, source_iterator, depth: int) -> None:
        if depth != 1:
            raise ValueError("in-process batch prefetch depth must be exactly one")
        self._source_iterator = source_iterator
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._finished = False
        self._producer = threading.Thread(
            target=self._produce,
            name="actworld-batch-prefetch",
            daemon=True,
        )
        self._producer.start()

    def _publish(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _wait_for_queue_capacity(self) -> bool:
        """Reserve materialization capacity before asking for another batch.

        ``Queue(maxsize=1)`` only bounds objects already inserted into the
        queue.  If the producer calls ``next`` while that slot is occupied, it
        can materialize a second full batch and then block in ``put``.  Formal
        NAVSIM batches are large enough for that hidden extra batch to violate
        the host-memory gate.  There is only one producer, so waiting for the
        queue to become non-full before ``next`` strictly bounds live prefetched
        batches to the configured depth.
        """

        while not self._stop.is_set():
            if not self._queue.full():
                return True
            self._stop.wait(0.01)
        return False

    def _produce(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._wait_for_queue_capacity():
                    return
                try:
                    batch = next(self._source_iterator)
                except StopIteration:
                    self._publish((_PREFETCH_END, None))
                    return
                except BaseException:
                    self._publish((_PREFETCH_ERROR, sys.exc_info()))
                    return
                if not self._publish((_PREFETCH_BATCH, batch)):
                    return
        finally:
            close = getattr(self._source_iterator, "_shutdown_workers", None)
            if callable(close):
                close()

    def __iter__(self):
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration
        kind, payload = self._queue.get()
        if kind is _PREFETCH_BATCH:
            return payload
        self._finished = True
        self._stop.set()
        if kind is _PREFETCH_END:
            raise StopIteration
        if kind is _PREFETCH_ERROR:
            _, error, traceback = payload
            raise error.with_traceback(traceback)
        raise RuntimeError("unknown in-process prefetch queue item")

    def close(self) -> None:
        self._finished = True
        self._stop.set()

    def __del__(self) -> None:
        self.close()


class InProcessBatchPrefetchDataLoader(torch.utils.data.DataLoader):
    """DataLoader whose rank-local producer overlaps one future batch."""

    def __init__(self, *args, prefetch_batches: int = 1, **kwargs) -> None:
        if prefetch_batches != 1:
            raise ValueError("in-process batch prefetch depth must be exactly one")
        self._in_process_prefetch_batches = prefetch_batches
        super().__init__(*args, **kwargs)

    def __iter__(self):
        return _InProcessBatchPrefetchIterator(
            super().__iter__(), self._in_process_prefetch_batches
        )


def _load_ordered_items(
    dataset: torch.utils.data.Dataset,
    indices: Sequence[int],
    threads: int,
) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    """Load a batch concurrently while preserving the sampler's exact order."""

    ordered_indices = list(indices)
    if threads == 1 or len(ordered_indices) <= 1:
        return [dataset[index] for index in ordered_indices]

    with ThreadPoolExecutor(
        max_workers=min(threads, len(ordered_indices)),
        thread_name_prefix="actworld-sample",
    ) as executor:
        return list(executor.map(dataset.__getitem__, ordered_indices))


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: Optional[List[str]] = None,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)

        if log_names is not None:
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]

        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._in_process_sample_threads = _resolve_in_process_sample_threads()
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            cache_path=self._cache_path,
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
            log_names=self.log_names,
        )
        self.tokens = list(self._valid_cache_paths.keys())

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return self._load_scene_with_token(self.tokens[idx])

    def __getitems__(
        self, indices: Sequence[int]
    ) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Load a DataLoader batch without process-local SceneLoader copies."""

        return _load_ordered_items(
            self, indices, self._in_process_sample_threads
        )

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
                found_caches: List[bool] = []
                for builder in feature_builders + target_builders:
                    data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                    found_caches.append(data_dict_path.is_file())
                if all(found_caches):
                    valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._in_process_sample_threads = _resolve_in_process_sample_threads()

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )

        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """

        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)

        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper function to load feature / targets from cache.
        :param token:  unique identifier of scene to load
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)

    def __len__(self) -> None:
        """
        :return: number of samples to load
        """
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"

            features, targets = self._load_scene_with_token(token)
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input()
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        return (features, targets)

    def __getitems__(
        self, indices: Sequence[int]
    ) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Load a DataLoader batch concurrently and retain sampler ordering."""

        return _load_ordered_items(
            self, indices, self._in_process_sample_threads
        )
