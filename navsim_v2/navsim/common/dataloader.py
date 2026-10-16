from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
from pathlib import Path
import hashlib
import os
import re
import stat

from tqdm import tqdm
import pickle
import lzma

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig
from navsim.planning.metric_caching.metric_cache import MetricCache

FrameList = List[Dict[str, Any]]


@dataclass(frozen=True)
class SensorAvailabilityFilterResult:
    """Summary returned after filtering scenes with unavailable sensor blobs."""

    total_regular_scenes: int
    total_synthetic_scenes: int
    removed_regular_tokens: Tuple[str, ...]
    removed_synthetic_tokens: Tuple[str, ...]

    @property
    def total_scenes(self) -> int:
        """Number of scene entries before filtering."""
        return self.total_regular_scenes + self.total_synthetic_scenes

    @property
    def removed_scenes(self) -> int:
        """Number of scene entries removed by the preflight."""
        return len(self.removed_regular_tokens) + len(self.removed_synthetic_tokens)

    @property
    def retained_scenes(self) -> int:
        """Number of scene entries retained after filtering."""
        return self.total_scenes - self.removed_scenes


def filter_scenes(
    data_path: Path,
    scene_filter: SceneFilter,
    authenticated_raw_inputs: Any | None = None,
) -> Tuple[Dict[str, FrameList], List[str]]:
    """
    Load a set of scenes from dataset, while applying scene filter configuration.
    :param data_path: root directory of log folder
    :param scene_filter: scene filtering configuration class
    :return: dictionary of raw logs format, and list of final frame tokens that can be used to filter synthetic scenes
    """

    def split_list(input_list: List[Any], num_frames: int, frame_interval: int) -> List[List[Any]]:
        """Helper function to split frame list according to sampling specification."""
        return [input_list[i : i + num_frames] for i in range(0, len(input_list), frame_interval)]

    filtered_scenes: Dict[str, Scene] = {}
    # keep track of the final frame tokens which refer to the original scene of potential second stage synthetic scenes
    final_frame_tokens: List[str] = []
    stop_loading: bool = False

    # filter logs
    if authenticated_raw_inputs is not None:
        requested_logs = (
            scene_filter.log_names
            if scene_filter.log_names is not None
            else tuple(entry.log_name for entry in authenticated_raw_inputs.manifest.logs)
        )
        log_files = list(authenticated_raw_inputs.log_paths(requested_logs))
    else:
        log_files = list(data_path.iterdir())
        if scene_filter.log_names is not None:
            log_files = [log_file for log_file in log_files if log_file.name.replace(".pkl", "") in scene_filter.log_names]

    if scene_filter.tokens is not None:
        filter_tokens = True
        tokens = set(scene_filter.tokens)
    else:
        filter_tokens = False

    for log_pickle_path in tqdm(log_files, desc="Loading logs"):

        if authenticated_raw_inputs is None:
            with open(log_pickle_path, "rb") as stream:
                scene_dict_list = pickle.load(stream)
        else:
            scene_dict_list = authenticated_raw_inputs.load_log_pickle(log_pickle_path)
        for frame_list in split_list(scene_dict_list, scene_filter.num_frames, scene_filter.frame_interval):
            # Filter scenes which are too short
            if len(frame_list) < scene_filter.num_frames:
                continue

            # Filter scenes with no route
            if scene_filter.has_route and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0:
                continue

            # Filter by token
            token = frame_list[scene_filter.num_history_frames - 1]["token"]
            if filter_tokens and token not in tokens:
                continue

            filtered_scenes[token] = frame_list
            final_frame_token = frame_list[scene_filter.num_frames - 1]["token"]
            #  TODO: if num_future_frames > proposal_sampling frames, then the final_frame_token index is wrong
            final_frame_tokens.append(final_frame_token)

            if (scene_filter.max_scenes is not None) and (len(filtered_scenes) >= scene_filter.max_scenes):
                stop_loading = True
                break

        if stop_loading:
            break

    return filtered_scenes, final_frame_tokens

def filter_synthetic_scenes(
        data_path: Path,
        scene_filter: SceneFilter,
        stage1_scenes_final_frames_tokens:List[str]
    ) -> Dict[str, Tuple[Path, str]]:
    # Load all the synthetic scenes that belong to the original scenes already loaded
    loaded_scenes: Dict[str, Tuple[Path, str, int]] = {}
    syntethic_scenes_paths = list(data_path.iterdir())

    filter_logs = scene_filter.log_names is not None
    filter_tokens = scene_filter.synthetic_scene_tokens is not None

    for scene_path in tqdm(syntethic_scenes_paths, desc="Loading synthetic scenes"):
        synthetic_scene = Scene.load_from_disk(scene_path, None, None)

        # if a token is requested specifically, we load it even if it is not related to the original scenes loaded
        if filter_tokens and synthetic_scene.scene_metadata.initial_token not in scene_filter.synthetic_scene_tokens:
            continue

        # filter by log names
        log_name = synthetic_scene.scene_metadata.log_name
        if filter_logs and log_name not in scene_filter.log_names:
            continue

        # if we don't filter for tokens explicitly, we load only the synthetic scenes required to run a second stage for the original scenes loaded
        if not filter_tokens and synthetic_scene.scene_metadata.corresponding_orignal_scene not in stage1_scenes_final_frames_tokens:
            continue

        loaded_scenes.update({synthetic_scene.scene_metadata.initial_token: [scene_path, log_name]})

    return loaded_scenes

class SceneLoader:
    """Simple data loader of scenes from logs."""

    def __init__(
        self,
        data_path: Path,
        sensor_blobs_path: Path,
        synthetic_scenes_path: Path,
        scene_filter: SceneFilter,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
        authenticated_raw_inputs: Any | None = None,
    ):
        """
        Initializes the scene data loader.
        :param data_path: root directory of log folder
        :param sensor_blobs_path: root directory of sensor data
        :param scene_filter: dataclass for scene filtering specification
        :param sensor_config: dataclass for sensor loading specification, defaults to no sensors
        """

        if authenticated_raw_inputs is not None:
            from navsim.planning.validation.f2_navtrain_input_manifest import (
                VerifiedNavtrainRawInputs,
            )

            if type(authenticated_raw_inputs) is not VerifiedNavtrainRawInputs:
                raise ValueError(
                    "authenticated_raw_inputs must be a live-verified F2 navtrain manifest"
                )
            authenticated_raw_inputs.require_roots(data_path, sensor_blobs_path)
        self._authenticated_raw_inputs = authenticated_raw_inputs
        self.scene_frames_dicts, stage1_scenes_final_frames_tokens = filter_scenes(
            data_path,
            scene_filter,
            authenticated_raw_inputs=authenticated_raw_inputs,
        )
        self._sensor_blobs_path = sensor_blobs_path
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config

        if scene_filter.include_synthetic_scenes:
            assert synthetic_scenes_path is not None, (
                "Synthetic scenes path cannot be None, when synthetic scenes_filter.include_synthetic_scenes is set to True."
            )
            self.synthetic_scenes = filter_synthetic_scenes(
                data_path=synthetic_scenes_path,
                scene_filter=scene_filter,
                stage1_scenes_final_frames_tokens=stage1_scenes_final_frames_tokens,
            )
            self.synthetic_scenes_tokens = set(self.synthetic_scenes.keys())
        else:
            self.synthetic_scenes = {}
            self.synthetic_scenes_tokens = set()

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.scene_frames_dicts.keys()) + list(self.synthetic_scenes.keys())

    def __len__(self) -> int:
        """
        :return: number for scenes possible to load.
        """
        return len(self.tokens)

    def __getitem__(self, idx) -> str:
        """
        :param idx: index of scene
        :return: unique scene identifier
        """
        return self.tokens[idx]

    @staticmethod
    def _camera_entry(camera_dict: Any, sensor_name: str) -> Any:
        """Return a camera entry while accepting NAVSIM's upper/lower-case keys."""
        if not isinstance(camera_dict, Mapping):
            return None
        return next(
            (value for key, value in camera_dict.items() if str(key).lower() == sensor_name),
            None,
        )

    def _sensor_path_is_file(self, relative_path: Any, cache: Dict[Path, bool]) -> bool:
        """Check a blob path with the same root-joining semantics as scene loading."""
        if relative_path is None or self._sensor_blobs_path is None:
            return False
        authenticated = getattr(self, "_authenticated_raw_inputs", None)
        if authenticated is not None:
            return authenticated.has_sensor_blob(relative_path)
        try:
            path = (self._sensor_blobs_path / Path(relative_path)).absolute()
        except (TypeError, ValueError):
            return False

        if path not in cache:
            try:
                cache[path] = path.is_file()
            except OSError:
                cache[path] = False
        return cache[path]

    def _load_authenticated_sensor_blob(self, relative_path: str) -> bytes:
        authenticated = getattr(self, "_authenticated_raw_inputs", None)
        if authenticated is None:
            raise RuntimeError("SceneLoader has no authenticated raw-input manifest")
        return authenticated.read_sensor_blob(relative_path)

    def _frame_has_available_sensors(
        self,
        frame: Any,
        iteration: int,
        camera_key: str,
        path_cache: Dict[Path, bool],
    ) -> bool:
        """Check all blobs that ``Scene`` would load for one frame."""
        sensor_names = self._sensor_config.get_sensors_at_iteration(iteration)
        if not sensor_names:
            return True
        if not isinstance(frame, Mapping):
            return False

        if "lidar_pc" in sensor_names and not self._sensor_path_is_file(
            frame.get("lidar_path"), path_cache
        ):
            return False

        camera_dict = frame.get(camera_key)
        for sensor_name in sensor_names:
            if sensor_name == "lidar_pc":
                continue
            camera_entry = self._camera_entry(camera_dict, sensor_name)
            if not isinstance(camera_entry, Mapping) or not self._sensor_path_is_file(
                camera_entry.get("data_path"), path_cache
            ):
                return False

        return True

    def _regular_scene_has_available_sensors(
        self, frames: Any, path_cache: Dict[Path, bool]
    ) -> bool:
        """Check the raw frames of a regular NAVSIM scene."""
        if not isinstance(frames, list):
            return False
        return all(
            self._frame_has_available_sensors(frame, index, "cams", path_cache)
            for index, frame in enumerate(frames)
        )

    def _synthetic_scene_has_available_sensors(
        self, scene_path: Path, path_cache: Dict[Path, bool]
    ) -> bool:
        """Check a serialized synthetic scene without decoding its sensor blobs."""
        try:
            with open(scene_path, "rb") as file:
                scene_data = pickle.load(file)
        except (OSError, EOFError, pickle.UnpicklingError):
            return False

        if not isinstance(scene_data, Mapping):
            return False
        frames = scene_data.get("frames")
        if not isinstance(frames, list):
            return False
        return all(
            self._frame_has_available_sensors(frame, index, "camera_dict", path_cache)
            for index, frame in enumerate(frames)
        )

    def filter_scenes_by_sensor_availability(self) -> SensorAvailabilityFilterResult:
        """
        Remove scenes whose configured camera or lidar files are unavailable.

        This is an explicit, mutating preflight intended for dataset construction.
        It is deliberately not called from ``SceneLoader.__init__`` so evaluation
        semantics remain unchanged. Synthetic scenes are checked against their
        serialized frame metadata rather than being treated as regular log scenes.

        :return: counts and removed tokens for regular and synthetic scenes.
        :raises RuntimeError: when every initially loaded scene is removed.
        """
        total_regular_scenes = len(self.scene_frames_dicts)
        total_synthetic_scenes = len(self.synthetic_scenes)
        # navtrain scenes overlap heavily, so each absolute blob path is stat'ed once.
        path_cache: Dict[Path, bool] = {}

        # Avoid opening synthetic pickle files for agents that request no sensors.
        requests_any_sensor = any(bool(value) for value in asdict(self._sensor_config).values())
        if requests_any_sensor:
            removed_regular_tokens = tuple(
                token
                for token, frames in self.scene_frames_dicts.items()
                if not self._regular_scene_has_available_sensors(frames, path_cache)
            )
            removed_synthetic_tokens = tuple(
                token
                for token, (scene_path, _) in self.synthetic_scenes.items()
                if not self._synthetic_scene_has_available_sensors(scene_path, path_cache)
            )
        else:
            removed_regular_tokens = ()
            removed_synthetic_tokens = ()

        for token in removed_regular_tokens:
            del self.scene_frames_dicts[token]
        for token in removed_synthetic_tokens:
            del self.synthetic_scenes[token]
        self.synthetic_scenes_tokens = set(self.synthetic_scenes.keys())

        result = SensorAvailabilityFilterResult(
            total_regular_scenes=total_regular_scenes,
            total_synthetic_scenes=total_synthetic_scenes,
            removed_regular_tokens=removed_regular_tokens,
            removed_synthetic_tokens=removed_synthetic_tokens,
        )
        if result.total_scenes > 0 and result.retained_scenes == 0:
            example_tokens = (removed_regular_tokens + removed_synthetic_tokens)[:5]
            raise RuntimeError(
                "Sensor availability preflight removed all "
                f"{result.total_scenes} scenes (examples: {example_tokens}). "
                f"Check sensor_blobs_path={self._sensor_blobs_path} and the agent SensorConfig."
            )

        return result

    def get_scene_from_token(self, token: str) -> Scene:
        """
        Loads scene given a scene identifier string (token).
        :param token: scene identifier string.
        :return: scene dataclass
        """
        assert token in self.tokens
        if token in self.synthetic_scenes:
            return Scene.load_from_disk(
                file_path=self.synthetic_scenes[token][0],
                sensor_blobs_path=self._sensor_blobs_path,
                sensor_config=self._sensor_config,
            )
        else:
            return Scene.from_scene_dict_list(
                self.scene_frames_dicts[token],
                self._sensor_blobs_path,
                num_history_frames=self._scene_filter.num_history_frames,
                num_future_frames=self._scene_filter.num_future_frames,
                sensor_config=self._sensor_config,
                sensor_blob_loader=(
                    self._load_authenticated_sensor_blob
                    if getattr(self, "_authenticated_raw_inputs", None) is not None
                    else None
                ),
            )

    def get_agent_input_from_token(self, token: str) -> AgentInput:
        """
        Loads agent input given a scene identifier string (token).
        :param token: scene identifier string.
        :return: agent input dataclass
        """
        assert token in self.tokens
        if token in self.synthetic_scenes:
            return Scene.load_from_disk(
                file_path=self.synthetic_scenes[token][0],
                sensor_blobs_path=self._sensor_blobs_path,
                sensor_config=self._sensor_config,
            ).get_agent_input()
        else:
            return AgentInput.from_scene_dict_list(
                self.scene_frames_dicts[token],
                self._sensor_blobs_path,
                num_history_frames=self._scene_filter.num_history_frames,
                sensor_config=self._sensor_config,
                sensor_blob_loader=(
                    self._load_authenticated_sensor_blob
                    if getattr(self, "_authenticated_raw_inputs", None) is not None
                    else None
                ),
            )

    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        """
        Collect tokens for each logs file given filtering.
        :return: dictionary of logs names and tokens
        """
        # generate a dict that contains a list of tokens for each log-name
        tokens_per_logs: Dict[str, List[str]] = {}
        for token, scene_dict_list in self.scene_frames_dicts.items():
            log_name = scene_dict_list[0]["log_name"]
            if tokens_per_logs.get(log_name):
                tokens_per_logs[log_name].append(token)
            else:
                tokens_per_logs.update({log_name: [token]})

        for scene_path, log_name in self.synthetic_scenes.values():
            if tokens_per_logs.get(log_name):
                tokens_per_logs[log_name].append(scene_path.stem)
            else:
                tokens_per_logs.update({log_name: [scene_path.stem]})

        return tokens_per_logs


class MetricCacheLoader:
    """Simple dataloader for metric cache."""

    def __init__(
        self,
        cache_path: Path,
        file_name: str = "metric_cache.pkl",
        *,
        expected_metadata_path: Path | None = None,
        expected_metadata_sha256: str | None = None,
        authenticated_content_manifest: Any | None = None,
    ):
        """
        Initializes the metric cache loader.
        :param cache_path: directory of cache folder
        :param file_name: file name of cached files, defaults to "metric_cache.pkl"
        """

        self._file_name = file_name
        strict_metadata = (expected_metadata_path, expected_metadata_sha256)
        if (strict_metadata[0] is None) != (strict_metadata[1] is None):
            raise ValueError(
                "expected_metadata_path and expected_metadata_sha256 must be supplied together"
            )
        self.metadata_path: Path | None = None
        self.metadata_sha256: str | None = None
        self.metric_cache_paths = self._load_metric_cache_paths(
            Path(cache_path),
            expected_metadata_path=expected_metadata_path,
            expected_metadata_sha256=expected_metadata_sha256,
        )
        self.authenticated_content_manifest = authenticated_content_manifest
        if authenticated_content_manifest is not None:
            # Import lazily so the generic/legacy loader keeps no dependency on
            # the F2 validation stack.  The opaque verifier returns only the
            # qualified train+held_out mapping, with values that remain
            # authenticated when passed to spawned score workers.
            from navsim.planning.validation.f2_metric_cache_manifest import (
                VerifiedMetricCacheManifest,
            )

            if type(authenticated_content_manifest) is not VerifiedMetricCacheManifest:
                raise ValueError(
                    "authenticated_content_manifest must be a live-verified F2 manifest"
                )
            self.metric_cache_paths = authenticated_content_manifest.require_loader_mapping(
                self.metric_cache_paths,
                cache_root=Path(cache_path),
                metadata_path=self.metadata_path,
                metadata_sha256=self.metadata_sha256,
            )

    def _load_metric_cache_paths(
        self,
        cache_path: Path,
        *,
        expected_metadata_path: Path | None,
        expected_metadata_sha256: str | None,
    ) -> Dict[str, Path]:
        """
        Helper function to load all cache file paths from folder.
        :param cache_path: directory of cache folder
        :return: dictionary of token and file path
        """
        metadata_dir = cache_path / "metadata"
        authenticated_metadata_text = None
        if expected_metadata_path is None:
            # Preserve the upstream multi-node/legacy behavior exactly unless
            # an authenticated F2 caller explicitly opts into strict metadata.
            metadata_file = [
                file for file in metadata_dir.iterdir() if ".csv" in str(file)
            ][0]
        else:
            metadata_files = sorted(
                file
                for file in metadata_dir.iterdir()
                if file.suffix.lower() == ".csv"
            )
            if len(metadata_files) != 1:
                raise ValueError(
                    "authenticated metric cache must contain exactly one metadata CSV"
                )
            requested = Path(expected_metadata_path).expanduser()
            try:
                requested_before = requested.lstat()
            except OSError as error:
                raise ValueError(
                    "authenticated metric-cache metadata cannot be stated"
                ) from error
            if stat.S_ISLNK(requested_before.st_mode) or not stat.S_ISREG(
                requested_before.st_mode
            ):
                raise ValueError(
                    "authenticated metric-cache metadata must be a regular non-symlink file"
                )
            metadata_file = metadata_files[0]
            try:
                path_before = metadata_file.lstat()
            except OSError as error:
                raise ValueError(
                    "authenticated metric-cache metadata cannot be stated"
                ) from error
            if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(
                path_before.st_mode
            ):
                raise ValueError(
                    "authenticated metric-cache metadata must not be a symlink"
                )
            if metadata_file.resolve() != requested.resolve():
                raise ValueError(
                    "metric-cache metadata CSV differs from the explicit authenticated path"
                )
            assert expected_metadata_sha256 is not None
            expected = expected_metadata_sha256
            if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                raise ValueError(
                    "expected metric-cache metadata SHA-256 must be exactly "
                    "64 lowercase hexadecimal characters"
                )
            identity = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(metadata_file, flags)
                with os.fdopen(descriptor, "rb", closefd=True) as metadata_stream:
                    opened = os.fstat(metadata_stream.fileno())
                    if (
                        identity(opened) != identity(path_before)
                        or identity(opened) != identity(requested_before)
                    ):
                        raise ValueError(
                            "metric-cache metadata changed before authentication"
                        )
                    digest = hashlib.sha256()
                    for chunk in iter(
                        lambda: metadata_stream.read(1024 * 1024), b""
                    ):
                        digest.update(chunk)
                    actual = digest.hexdigest()
                    if actual != expected:
                        raise ValueError(
                            "metric-cache metadata SHA-256 mismatch: "
                            f"expected={expected}, actual={actual}"
                        )
                    metadata_stream.seek(0)
                    authenticated_metadata_bytes = metadata_stream.read()
                    after_read = os.fstat(metadata_stream.fileno())
                    metadata_stream.seek(0)
                    verification_digest = hashlib.sha256()
                    for chunk in iter(
                        lambda: metadata_stream.read(1024 * 1024), b""
                    ):
                        verification_digest.update(chunk)
                    after_verify = os.fstat(metadata_stream.fileno())
                path_after = metadata_file.lstat()
                requested_after = requested.lstat()
            except ValueError:
                raise
            except OSError as error:
                raise ValueError(
                    "cannot stably authenticate metric-cache metadata"
                ) from error
            verified_actual = verification_digest.hexdigest()
            if verified_actual != expected:
                raise ValueError(
                    "metric-cache metadata changed between authentication and parsing"
                )
            if any(
                identity(value) != identity(opened)
                for value in (
                    after_read,
                    after_verify,
                    path_after,
                    requested_after,
                )
            ):
                raise ValueError(
                    "metric-cache metadata changed while it was authenticated and parsed"
                )
            try:
                authenticated_metadata_text = authenticated_metadata_bytes.decode(
                    "utf-8"
                )
            except UnicodeDecodeError as error:
                raise ValueError(
                    "authenticated metric-cache metadata must be UTF-8"
                ) from error
            self.metadata_sha256 = actual
        self.metadata_path = metadata_file.resolve()
        if authenticated_metadata_text is None:
            with open(str(metadata_file), "r") as f:
                cache_paths = f.read().splitlines()[1:]
        else:
            cache_paths = authenticated_metadata_text.splitlines()[1:]
        metric_cache_dict = {cache_path.split("/")[-2]: cache_path for cache_path in cache_paths}
        return metric_cache_dict

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.metric_cache_paths.keys())

    def __len__(self):
        """
        :return: number for scenes possible to load.
        """
        return len(self.metric_cache_paths)

    def __getitem__(self, idx: int) -> MetricCache:
        """
        :param idx: index of cache to cache to load
        :return: metric cache dataclass
        """
        return self.get_from_token(self.tokens[idx])

    def get_from_token(self, token: str) -> MetricCache:
        """
        Load metric cache from scene identifier
        :param token: unique identifier of scene
        :return: metric cache dataclass
        """
        reference = self.metric_cache_paths[token]
        if self.authenticated_content_manifest is not None:
            from navsim.planning.validation.f2_metric_cache_manifest import (
                load_authenticated_metric_cache,
            )

            metric_cache = load_authenticated_metric_cache(reference)
        else:
            with lzma.open(reference, "rb") as f:
                metric_cache = pickle.load(f)
        return metric_cache

    def to_pickle(self, path: Path) -> None:
        """
        Dumps complete metric cache into pickle.
        :param path: directory of cache folder
        """
        full_metric_cache = {}
        for token in tqdm(self.tokens):
            full_metric_cache[token] = self.get_from_token(token)
        with open(path, "wb") as f:
            pickle.dump(full_metric_cache, f)
