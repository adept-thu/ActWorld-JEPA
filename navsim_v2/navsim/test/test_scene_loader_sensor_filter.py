import pickle
from pathlib import Path

import pytest

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader


def _loader(
    sensor_root: Path,
    sensor_config: SensorConfig,
    regular_scenes=None,
    synthetic_scenes=None,
) -> SceneLoader:
    """Construct a focused loader fixture without reading full NAVSIM logs."""
    loader = SceneLoader.__new__(SceneLoader)
    loader._sensor_blobs_path = sensor_root
    loader._sensor_config = sensor_config
    loader.scene_frames_dicts = regular_scenes or {}
    loader.synthetic_scenes = synthetic_scenes or {}
    loader.synthetic_scenes_tokens = set(loader.synthetic_scenes)
    return loader


def _camera_only_config() -> SensorConfig:
    return SensorConfig(
        cam_f0=[0],
        cam_l0=False,
        cam_l1=False,
        cam_l2=False,
        cam_r0=False,
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=False,
    )


def _regular_frame(camera_path: str):
    return {"cams": {"CAM_F0": {"data_path": camera_path}}, "lidar_path": None}


def _synthetic_pickle(path: Path, camera_path: str) -> None:
    with open(path, "wb") as file:
        pickle.dump(
            {
                "frames": [
                    {
                        "camera_dict": {"cam_f0": {"data_path": camera_path}},
                        "lidar_path": None,
                    }
                ]
            },
            file,
        )


def test_filter_removes_only_regular_scenes_with_missing_requested_sensor(tmp_path: Path) -> None:
    (tmp_path / "present.jpg").write_bytes(b"present")
    loader = _loader(
        tmp_path,
        _camera_only_config(),
        regular_scenes={
            "valid": [_regular_frame("present.jpg")],
            "missing": [_regular_frame("missing.jpg")],
        },
    )

    result = loader.filter_scenes_by_sensor_availability()

    assert loader.tokens == ["valid"]
    assert result.total_scenes == 2
    assert result.removed_regular_tokens == ("missing",)
    assert result.removed_synthetic_tokens == ()
    assert result.retained_scenes == 1


def test_filter_stats_overlapping_blob_path_once(tmp_path: Path, monkeypatch) -> None:
    sensor_path = tmp_path / "shared.jpg"
    sensor_path.write_bytes(b"present")
    loader = _loader(
        tmp_path,
        _camera_only_config(),
        regular_scenes={
            "overlap-a": [_regular_frame("shared.jpg")],
            "overlap-b": [_regular_frame("shared.jpg")],
        },
    )
    original_is_file = Path.is_file
    checked_paths = []

    def counted_is_file(path: Path) -> bool:
        checked_paths.append(path)
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", counted_is_file)

    result = loader.filter_scenes_by_sensor_availability()

    assert result.removed_scenes == 0
    assert checked_paths == [sensor_path.absolute()]


def test_filter_checks_synthetic_metadata_and_updates_token_set(tmp_path: Path) -> None:
    (tmp_path / "present.jpg").write_bytes(b"present")
    valid_pickle = tmp_path / "valid.pkl"
    missing_pickle = tmp_path / "missing.pkl"
    _synthetic_pickle(valid_pickle, "present.jpg")
    _synthetic_pickle(missing_pickle, "missing.jpg")
    loader = _loader(
        tmp_path,
        _camera_only_config(),
        synthetic_scenes={
            "synthetic-valid": (valid_pickle, "log"),
            "synthetic-missing": (missing_pickle, "log"),
        },
    )

    result = loader.filter_scenes_by_sensor_availability()

    assert loader.tokens == ["synthetic-valid"]
    assert loader.synthetic_scenes_tokens == {"synthetic-valid"}
    assert result.removed_synthetic_tokens == ("synthetic-missing",)


def test_filter_does_not_require_unconfigured_sensors_or_open_synthetic_scene(tmp_path: Path) -> None:
    missing_pickle = tmp_path / "does-not-exist.pkl"
    loader = _loader(
        tmp_path,
        SensorConfig.build_no_sensors(),
        regular_scenes={"regular": [{"malformed": "but sensors are disabled"}]},
        synthetic_scenes={"synthetic": (missing_pickle, "log")},
    )

    result = loader.filter_scenes_by_sensor_availability()

    assert set(loader.tokens) == {"regular", "synthetic"}
    assert result.removed_scenes == 0


def test_filter_raises_clear_error_when_all_scenes_are_removed(tmp_path: Path) -> None:
    loader = _loader(
        tmp_path,
        SensorConfig(
            cam_f0=False,
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=True,
        ),
        regular_scenes={"missing-lidar": [{"cams": {}, "lidar_path": "missing.pcd"}]},
    )

    with pytest.raises(RuntimeError, match="removed all 1 scenes"):
        loader.filter_scenes_by_sensor_availability()

    assert loader.tokens == []
