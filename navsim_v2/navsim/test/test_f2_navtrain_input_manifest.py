from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import pytest
from PIL import Image

import navsim.planning.validation.f2_navtrain_input_manifest as raw_manifest_module

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.f2_qualification_manifest import (
    QualificationRow,
    build_qualification_manifest,
    build_qualification_partition,
    canonical_manifest_bytes as canonical_qualification_manifest_bytes,
    qualification_manifest_sha256,
)
from navsim.planning.validation.f2_navtrain_input_manifest import (
    F2NavtrainInputManifestError,
    build_raw_input_manifest,
    cli_main,
    load_and_verify_raw_input_manifest,
    raw_input_manifest_sidecar_path,
    write_raw_input_manifest,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _qualification(log_name: str):
    train = build_qualification_partition(
        "train",
        [QualificationRow(log_name, 3, "token-3", True)],
        raw_token_count=1,
        raw_log_count=1,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=_sha("train-exclusion"),
    )
    held_out = build_qualification_partition(
        "held_out",
        [QualificationRow(log_name, 4, "token-4", True)],
        raw_token_count=1,
        raw_log_count=1,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=_sha("held-exclusion"),
    )
    return build_qualification_manifest(
        train,
        held_out,
        config_sha256=_sha("config"),
        cache_sha256=_sha("cache"),
        source_sha256=_sha("source"),
    )


def _sensor_config() -> SensorConfig:
    return SensorConfig(
        cam_f0=[1, 2, 3, 10, 11],
        cam_l0=[3],
        cam_l1=[],
        cam_l2=[],
        cam_r0=[3],
        cam_r1=[],
        cam_r2=[],
        cam_b0=[3],
        lidar_pc=[],
    )


def _write_image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (color, color, color)).save(path, format="JPEG")


def _raw_frame(log_name: str, index: int, sensor_root: Path) -> dict:
    cameras = {}
    for sensor_index, sensor_name in enumerate(
        ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")
    ):
        relative = f"{sensor_name}/{index}.jpg"
        _write_image(sensor_root / relative, (index + sensor_index) % 255)
        cameras[sensor_name] = {
            "data_path": relative,
            "sensor2lidar_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "sensor2lidar_translation": [0, 0, 0],
            "cam_intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "distortion": [0, 0, 0, 0],
        }
    return {
        "anns": {
            "gt_boxes": [],
            "gt_names": [],
            "gt_velocity_3d": [],
            "instance_tokens": [],
            "track_tokens": [],
        },
        "cams": cameras,
        "driving_command": [1, 0, 0, 0],
        "ego2global_rotation": [1, 0, 0, 0],
        "ego2global_translation": [float(index), 0.0, 0.0],
        "ego_dynamic_state": [0.0, 0.0, 0.0, 0.0],
        "lidar_path": f"LIDAR/{index}.pcd",
        "log_name": log_name,
        "map_location": "sg-one-north",
        "roadblock_ids": ["road"],
        "scene_token": "scene",
        "timestamp": index,
        "token": f"token-{index}",
        "traffic_lights": [],
    }


def _fixture(tmp_path: Path):
    log_name = "qualified-log"
    log_root = tmp_path / "logs" / "trainval"
    sensor_root = tmp_path / "sensor_blobs" / "trainval"
    log_root.mkdir(parents=True)
    sensor_root.mkdir(parents=True)
    frames = [_raw_frame(log_name, index, sensor_root) for index in range(15)]
    (log_root / f"{log_name}.pkl").write_bytes(pickle.dumps(frames))
    qualification = _qualification(log_name)
    qualification_sha = qualification_manifest_sha256(qualification)
    manifest = build_raw_input_manifest(
        log_root=log_root,
        sensor_root=sensor_root,
        qualification=qualification,
        expected_qualification_manifest_sha256=qualification_sha,
        workers=2,
    )
    manifest_path = tmp_path / "raw-input.json"
    manifest_sha = write_raw_input_manifest(manifest_path, manifest)
    return (
        log_root,
        sensor_root,
        qualification,
        qualification_sha,
        manifest,
        manifest_path,
        manifest_sha,
    )


def _load(fixture):
    log_root, sensor_root, qualification, qualification_sha, _, path, digest = fixture
    return load_and_verify_raw_input_manifest(
        path,
        expected_sha256=digest,
        log_root=log_root,
        sensor_root=sensor_root,
        qualification=qualification,
        expected_qualification_manifest_sha256=qualification_sha,
    )


def _write_resigned(path: Path, payload: dict) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    digest = hashlib.sha256(encoded).hexdigest()
    path.write_bytes(encoded)
    raw_input_manifest_sidecar_path(path).write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def test_round_trip_full_preflight_and_same_descriptor_sensor_consumption(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    verified = _load(fixture)

    verified.verify_all(workers=2)
    assert verified.read_sensor_blob("CAM_F0/3.jpg").startswith(b"\xff\xd8")
    assert verified.log_paths(["qualified-log"]) == (
        fixture[0] / "qualified-log.pkl",
    )


def test_verify_cli_performs_full_prelaunch_preflight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture(tmp_path)
    qualification_path = tmp_path / "qualification.json"
    qualification_path.write_bytes(
        canonical_qualification_manifest_bytes(fixture[2])
    )
    result = cli_main(
        [
            "verify",
            "--qualification-manifest",
            str(qualification_path),
            "--qualification-sha256",
            fixture[3],
            "--qualification-config-sha256",
            fixture[2].config_sha256,
            "--qualification-cache-sha256",
            fixture[2].cache_sha256,
            "--qualification-source-sha256",
            fixture[2].source_sha256,
            "--log-root",
            str(fixture[0]),
            "--sensor-root",
            str(fixture[1]),
            "--workers",
            "2",
            "--manifest",
            str(fixture[5]),
            "--manifest-sha256",
            fixture[6],
        ]
    )
    assert result == 0
    assert capsys.readouterr().out.strip() == fixture[6]


def test_scene_loader_consumes_authenticated_bytes_and_rejects_replacement(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    verified = _load(fixture)
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        max_scenes=None,
        log_names=["qualified-log"],
        tokens=["token-3"],
        include_synthetic_scenes=False,
        synthetic_scene_tokens=None,
    )
    loader = SceneLoader(
        data_path=fixture[0],
        sensor_blobs_path=fixture[1],
        synthetic_scenes_path=None,
        scene_filter=scene_filter,
        sensor_config=_sensor_config(),
        authenticated_raw_inputs=verified,
    )
    replaced = fixture[1] / "CAM_F0" / "3.jpg"
    original = replaced.read_bytes()
    replaced.write_bytes(b"X" * len(original))

    with pytest.raises(F2NavtrainInputManifestError, match="SHA-256 mismatch"):
        loader.get_agent_input_from_token("token-3")


@pytest.mark.parametrize("escaped", ("../outside.jpg", "/absolute/outside.jpg"))
def test_builder_rejects_parent_and_absolute_sensor_paths(
    tmp_path: Path, escaped: str
) -> None:
    fixture = _fixture(tmp_path)
    log_path = fixture[0] / "qualified-log.pkl"
    frames = pickle.loads(log_path.read_bytes())
    frames[3]["cams"]["CAM_F0"]["data_path"] = escaped
    log_path.write_bytes(pickle.dumps(frames))

    with pytest.raises(F2NavtrainInputManifestError, match="canonical relative path"):
        build_raw_input_manifest(
            log_root=fixture[0],
            sensor_root=fixture[1],
            qualification=fixture[2],
            expected_qualification_manifest_sha256=fixture[3],
        )


def test_builder_rejects_symlink_component_and_root_ancestor(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = fixture[1] / "real"
    target.mkdir()
    link = fixture[1] / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    log_path = fixture[0] / "qualified-log.pkl"
    frames = pickle.loads(log_path.read_bytes())
    frames[3]["cams"]["CAM_F0"]["data_path"] = "linked/blob.jpg"
    (target / "blob.jpg").write_bytes(b"blob")
    log_path.write_bytes(pickle.dumps(frames))
    with pytest.raises(F2NavtrainInputManifestError, match="symlink"):
        build_raw_input_manifest(
            log_root=fixture[0],
            sensor_root=fixture[1],
            qualification=fixture[2],
            expected_qualification_manifest_sha256=fixture[3],
        )

    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(fixture[0].parent, target_is_directory=True)
    with pytest.raises(F2NavtrainInputManifestError, match="must not traverse a symlink"):
        build_raw_input_manifest(
            log_root=alias_parent / "trainval",
            sensor_root=fixture[1],
            qualification=fixture[2],
            expected_qualification_manifest_sha256=fixture[3],
        )


def test_uppercase_expected_digest_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(F2NavtrainInputManifestError, match="lowercase"):
        load_and_verify_raw_input_manifest(
            fixture[5],
            expected_sha256=fixture[6].upper(),
            log_root=fixture[0],
            sensor_root=fixture[1],
            qualification=fixture[2],
            expected_qualification_manifest_sha256=fixture[3],
        )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_resigned_missing_or_extra_consumed_sensor_entry_is_rejected(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    payload = fixture[4].to_payload()
    entries = payload["roots"]["sensor_blobs"]["entries"]
    if mutation == "missing":
        entries.pop()
    else:
        extra_path = fixture[1] / "UNUSED" / "extra.bin"
        extra_path.parent.mkdir()
        extra_path.write_bytes(b"extra")
        entries.append(
            {
                "byte_size": 5,
                "relative_path": "UNUSED/extra.bin",
                "sha256": hashlib.sha256(b"extra").hexdigest(),
            }
        )
        entries.sort(key=lambda entry: entry["relative_path"])
    path = tmp_path / f"resigned-{mutation}.json"
    digest = _write_resigned(path, payload)

    with pytest.raises(
        F2NavtrainInputManifestError,
        match="missing or extra consumed sensor entries",
    ):
        load_and_verify_raw_input_manifest(
            path,
            expected_sha256=digest,
            log_root=fixture[0],
            sensor_root=fixture[1],
            qualification=fixture[2],
            expected_qualification_manifest_sha256=fixture[3],
        )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_resigned_missing_or_extra_qualified_log_entry_is_rejected(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    payload = fixture[4].to_payload()
    entries = payload["roots"]["logs"]["entries"]
    if mutation == "missing":
        entries.pop()
    else:
        unused = fixture[0] / "unused-log.pkl"
        unused.write_bytes(pickle.dumps([]))
        content = unused.read_bytes()
        entries.append(
            {
                "byte_size": len(content),
                "log_name": "unused-log",
                "relative_path": "unused-log.pkl",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
        entries.sort(key=lambda entry: entry["log_name"])
    path = tmp_path / f"resigned-log-{mutation}.json"
    digest = _write_resigned(path, payload)

    with pytest.raises(
        F2NavtrainInputManifestError,
        match=(
            "raw-input entries cannot be empty"
            if mutation == "missing"
            else "missing or extra qualified log entries"
        ),
    ):
        load_and_verify_raw_input_manifest(
            path,
            expected_sha256=digest,
            log_root=fixture[0],
            sensor_root=fixture[1],
            qualification=fixture[2],
            expected_qualification_manifest_sha256=fixture[3],
        )


def test_replaced_log_fails_before_unpickle(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    verified = _load(fixture)
    log_path = fixture[0] / "qualified-log.pkl"
    original = log_path.read_bytes()
    log_path.write_bytes(os.urandom(len(original)))

    with pytest.raises(F2NavtrainInputManifestError, match="SHA-256 mismatch"):
        verified.load_log_pickle(log_path)


def test_path_replacement_during_same_descriptor_read_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("openat replacement audit is exercised on the Linux runtime")
    fixture = _fixture(tmp_path)
    verified = _load(fixture)
    target = fixture[1] / "CAM_F0" / "3.jpg"
    replacement = target.with_name("replacement.jpg")
    replacement.write_bytes(target.read_bytes())
    original_read = raw_manifest_module.os.read
    replaced = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, count)
        if chunk and not replaced:
            replaced = True
            os.replace(replacement, target)
        return chunk

    monkeypatch.setattr(raw_manifest_module, "_READ_CHUNK_BYTES", 4)
    monkeypatch.setattr(raw_manifest_module.os, "read", racing_read)
    with pytest.raises(F2NavtrainInputManifestError, match="changed while open"):
        verified.read_sensor_blob("CAM_F0/3.jpg")
