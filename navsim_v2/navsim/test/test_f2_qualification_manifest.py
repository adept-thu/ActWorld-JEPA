"""Synthetic tests for the frozen F2 navtrain qualification boundary."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

import navsim.common.dataloader as dataloader_module
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.script.run_training import (
    authenticate_configured_f2_cache_metadata,
    authenticated_f2_training_cache_context,
    build_training_agent,
    build_datasets,
    load_configured_f2_qualification_manifest,
    validate_f2_run_protocol,
)
from navsim.planning.validation.f2_frozen_artifact_io import (
    F2_VJEPA_CHECKPOINT_SHA256,
)
from navsim.planning.training.f2_qualification_manifest import (
    F2QualificationError,
    QualificationManifest,
    QualificationPartition,
    QualificationRow,
    build_qualification_manifest,
    build_qualification_partition,
    canonical_manifest_bytes,
    constrain_scene_filter,
    identifier_sha256,
    manifest_from_bytes,
    qualification_manifest_sha256,
    qualify_scene_loader,
    require_f2_sensor_contract,
    row_identity_sha256,
)
from navsim.planning.validation.f2_causal_validator import F2_SENSOR_CONTRACT
from navsim.planning.validation.f2_metric_cache_manifest import (
    AuthenticatedMetricCacheReference,
    build_metric_cache_manifest,
    write_metric_cache_manifest,
)


CONFIG_SHA = hashlib.sha256(b"synthetic-resolved-f2-config\n").hexdigest()
CACHE_METADATA_BYTES = (
    b"file_name\n"
    b"cache/train-a/metric_cache.pkl\n"
    b"cache/train-b/metric_cache.pkl\n"
    b"cache/held-a/metric_cache.pkl\n"
)
CACHE_SHA = hashlib.sha256(CACHE_METADATA_BYTES).hexdigest()
SOURCE_SHA = hashlib.sha256(b"synthetic-source-provenance\n").hexdigest()
TRAIN_EXCLUSION_SHA = hashlib.sha256(
    b'{"partition":"train","removed":[["rejected","missing_cam_f0"]]}\n'
).hexdigest()
HELD_EXCLUSION_SHA = hashlib.sha256(
    b'{"partition":"held_out","removed":[]}\n'
).hexdigest()


def test_run_training_postpones_annotations_for_python39_import() -> None:
    """PEP 604 annotations must not execute under the official Python 3.9."""

    annotation = authenticate_configured_f2_cache_metadata.__annotations__[
        "metric_cache_loader_factory"
    ]
    assert isinstance(annotation, str)


def _partitions() -> tuple[QualificationPartition, QualificationPartition]:
    # The training audit deliberately has one mixed retained/removed log.  It
    # cannot be represented safely as a log-only allowlist.
    train = build_qualification_partition(
        "train",
        (
            QualificationRow("log-mixed", 200, "train-b", False),
            QualificationRow("log-mixed", 100, "train-a", True),
        ),
        raw_token_count=3,
        raw_log_count=1,
        removed_token_count=1,
        removed_log_count=1,
        mixed_log_count=1,
        exclusion_report_sha256=TRAIN_EXCLUSION_SHA,
    )
    held_out = build_qualification_partition(
        "held_out",
        (QualificationRow("log-held", 300, "held-a", True),),
        raw_token_count=1,
        raw_log_count=1,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=HELD_EXCLUSION_SHA,
    )
    return train, held_out


def _manifest() -> QualificationManifest:
    train, held_out = _partitions()
    return build_qualification_manifest(
        train,
        held_out,
        config_sha256=CONFIG_SHA,
        cache_sha256=CACHE_SHA,
        source_sha256=SOURCE_SHA,
    )


def _write_manifest(path: Path) -> QualificationManifest:
    manifest = _manifest()
    path.write_bytes(canonical_manifest_bytes(manifest))
    return manifest


def _write_cache_content_fixture(
    cache_root: Path,
    metadata: Path,
    qualification: QualificationManifest,
    destination: Path,
) -> tuple[Path, str]:
    for token in (*qualification.train.tokens, *qualification.held_out.tokens):
        cache_path = cache_root / "cache" / token / "metric_cache.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes((token + "-cache\n").encode("utf-8"))
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_bytes(CACHE_METADATA_BYTES)
    qualification_sha = qualification_manifest_sha256(qualification)
    manifest = build_metric_cache_manifest(
        cache_root=cache_root,
        metadata_path=metadata,
        expected_metadata_sha256=CACHE_SHA,
        qualification=qualification,
        expected_qualification_manifest_sha256=qualification_sha,
        workers=2,
    )
    destination = destination.resolve()
    return destination, write_metric_cache_manifest(destination, manifest)


def _content_config(
    qualification_path: Path,
    qualification: QualificationManifest,
    metadata: Path,
    content_path: Path,
    content_sha: str,
) -> dict[str, str]:
    return {
        "manifest_path": str(qualification_path),
        "cache_metadata_path": str(metadata),
        "cache_content_manifest_path": str(content_path),
        "expected_manifest_sha256": qualification_manifest_sha256(qualification),
        "expected_config_sha256": CONFIG_SHA,
        "expected_cache_sha256": CACHE_SHA,
        "expected_cache_content_manifest_sha256": content_sha,
        "expected_source_sha256": SOURCE_SHA,
    }


def test_manifest_is_canonical_deterministic_and_identity_bound() -> None:
    manifest = _manifest()
    payload = canonical_manifest_bytes(manifest)
    decoded = json.loads(payload)

    assert payload.endswith(b"\n")
    assert manifest.train.tokens == ("train-a", "train-b")
    assert decoded["partitions"]["train"]["qualification"] == {
        "exclusion_report_sha256": TRAIN_EXCLUSION_SHA,
        "mixed_log_count": 1,
        "raw_log_count": 1,
        "raw_token_count": 3,
        "removed_log_count": 1,
        "removed_token_count": 1,
    }
    assert decoded["partitions"]["train"]["evidence"][
        "retained_token_sha256"
    ] == identifier_sha256(("train-b", "train-a"))
    expected_identity = hashlib.sha256(
        b"log-mixed\t100\ttrain-a\t1\nlog-mixed\t200\ttrain-b\t0\n"
    ).hexdigest()
    assert row_identity_sha256(manifest.train.rows) == expected_identity

    parsed = manifest_from_bytes(
        payload,
        expected_manifest_sha256=qualification_manifest_sha256(manifest),
        expected_config_sha256=CONFIG_SHA,
        expected_cache_sha256=CACHE_SHA,
        expected_source_sha256=SOURCE_SHA,
    )
    assert parsed == manifest


def test_manifest_rejects_tamper_noncanonical_bytes_and_provenance_drift() -> None:
    manifest = _manifest()
    payload = canonical_manifest_bytes(manifest)
    digest = qualification_manifest_sha256(manifest)

    with pytest.raises(F2QualificationError, match="manifest SHA-256 mismatch"):
        manifest_from_bytes(
            payload + b" ",
            expected_manifest_sha256=digest,
            expected_config_sha256=CONFIG_SHA,
            expected_cache_sha256=CACHE_SHA,
            expected_source_sha256=SOURCE_SHA,
        )

    noncanonical = payload[:-1] + b" \n"
    with pytest.raises(F2QualificationError, match="bytes are not canonical"):
        manifest_from_bytes(
            noncanonical,
            expected_manifest_sha256=hashlib.sha256(noncanonical).hexdigest(),
            expected_config_sha256=CONFIG_SHA,
            expected_cache_sha256=CACHE_SHA,
            expected_source_sha256=SOURCE_SHA,
        )

    with pytest.raises(F2QualificationError, match="config/cache/source digest"):
        manifest_from_bytes(
            payload,
            expected_manifest_sha256=digest,
            expected_config_sha256=hashlib.sha256(b"different-config\n").hexdigest(),
            expected_cache_sha256=CACHE_SHA,
            expected_source_sha256=SOURCE_SHA,
        )

    decoded = json.loads(payload)
    decoded["partitions"]["train"]["evidence"]["retained_token_count"] = 99
    evidence_tamper = (
        json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        + b"\n"
    )
    with pytest.raises(F2QualificationError, match="evidence"):
        manifest_from_bytes(
            evidence_tamper,
            expected_manifest_sha256=hashlib.sha256(evidence_tamper).hexdigest(),
            expected_config_sha256=CONFIG_SHA,
            expected_cache_sha256=CACHE_SHA,
            expected_source_sha256=SOURCE_SHA,
        )


def test_manifest_rejects_overlap_and_inconsistent_qualification_counts() -> None:
    train, held_out = _partitions()
    overlapping = build_qualification_partition(
        "held_out",
        (QualificationRow("log-held", 300, "train-a", True),),
        raw_token_count=1,
        raw_log_count=1,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=HELD_EXCLUSION_SHA,
    )
    with pytest.raises(F2QualificationError, match="partitions overlap"):
        build_qualification_manifest(
            train,
            overlapping,
            config_sha256=CONFIG_SHA,
            cache_sha256=CACHE_SHA,
            source_sha256=SOURCE_SHA,
        )

    with pytest.raises(F2QualificationError, match="raw token count"):
        build_qualification_partition(
            "train",
            train.rows,
            raw_token_count=2,
            raw_log_count=1,
            removed_token_count=1,
            removed_log_count=1,
            mixed_log_count=1,
            exclusion_report_sha256=TRAIN_EXCLUSION_SHA,
        )
    with pytest.raises(F2QualificationError, match="placeholder digest"):
        build_qualification_partition(
            "held_out",
            held_out.rows,
            raw_token_count=1,
            raw_log_count=1,
            removed_token_count=0,
            removed_log_count=0,
            mixed_log_count=0,
            exclusion_report_sha256="0" * 64,
        )
    assert held_out.retained_token_count == 1


def test_scene_filter_uses_explicit_tokens_not_log_partitioning() -> None:
    train, _ = _partitions()
    scene_filter = SceneFilter(
        num_history_frames=1,
        num_future_frames=0,
        log_names=["log-mixed", "log-held"],
        tokens=["train-a", "train-b", "rejected-in-same-log"],
    )
    constrain_scene_filter(scene_filter, train)

    assert scene_filter.tokens == ["train-a", "train-b"]
    assert scene_filter.log_names == ["log-mixed"]

    outside = build_qualification_partition(
        "train",
        (QualificationRow("log-mixed", 1, "outside", True),),
        raw_token_count=1,
        raw_log_count=1,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=TRAIN_EXCLUSION_SHA,
    )
    with pytest.raises(F2QualificationError, match="outside the configured"):
        constrain_scene_filter(scene_filter, outside)


def test_runtime_sensor_config_must_equal_frozen_contract() -> None:
    correct = SensorConfig(
        **{name: list(indices) if indices else False for name, indices in F2_SENSOR_CONTRACT}
    )
    require_f2_sensor_contract(correct)

    drifted = SensorConfig(
        **{
            name: ([0] if name == "cam_f0" else (list(indices) if indices else False))
            for name, indices in F2_SENSOR_CONTRACT
        }
    )
    with pytest.raises(F2QualificationError, match="sensor contract drifted"):
        require_f2_sensor_contract(drifted)


@dataclass
class _Availability:
    total_scenes: int
    removed_scenes: int = 0
    removed_regular_tokens: tuple[str, ...] = ()
    removed_synthetic_tokens: tuple[str, ...] = ()

    @property
    def retained_scenes(self) -> int:
        return self.total_scenes - self.removed_scenes


class _SyntheticSceneLoader:
    """Small dependency-injected SceneLoader stand-in with raw frame identity."""

    def __init__(
        self,
        *,
        scene_filter: SceneFilter,
        rows: dict[str, QualificationRow],
        removed_scenes: int = 0,
        reverse: bool = True,
        **_: Any,
    ) -> None:
        self._scene_filter = scene_filter
        self.synthetic_scenes = {}
        self.synthetic_scenes_tokens = set()
        ordered_tokens = list(scene_filter.tokens or [])
        if reverse:
            ordered_tokens.reverse()
        self.scene_frames_dicts = {
            token: [
                {
                    "log_name": rows[token].log_name,
                    "timestamp": rows[token].time_us,
                    "token": token,
                }
            ]
            for token in ordered_tokens
        }
        self._removed_scenes = removed_scenes
        self.preflight_calls = 0

    @property
    def tokens(self) -> list[str]:
        return list(self.scene_frames_dicts)

    def filter_scenes_by_sensor_availability(self) -> _Availability:
        self.preflight_calls += 1
        return _Availability(len(self.tokens), self._removed_scenes)


def test_loader_requires_exact_set_zero_removal_and_canonical_reindex() -> None:
    train, _ = _partitions()
    rows = {row.token: row for row in train.rows}
    loader = _SyntheticSceneLoader(
        scene_filter=SceneFilter(
            num_history_frames=1,
            num_future_frames=0,
            tokens=list(train.tokens),
        ),
        rows=rows,
    )

    availability = qualify_scene_loader(loader, train)

    assert availability.removed_scenes == 0
    assert loader.preflight_calls == 1
    assert tuple(loader.tokens) == train.tokens

    missing_loader = _SyntheticSceneLoader(
        scene_filter=SceneFilter(
            num_history_frames=1,
            num_future_frames=0,
            tokens=[train.tokens[0]],
        ),
        rows=rows,
    )
    with pytest.raises(F2QualificationError, match="preflight token set"):
        qualify_scene_loader(missing_loader, train)
    assert missing_loader.preflight_calls == 0

    removed_loader = _SyntheticSceneLoader(
        scene_filter=SceneFilter(
            num_history_frames=1,
            num_future_frames=0,
            tokens=list(train.tokens),
        ),
        rows=rows,
        removed_scenes=1,
    )
    with pytest.raises(F2QualificationError, match="removed=1"):
        qualify_scene_loader(removed_loader, train)


def test_loader_rejects_row_identity_drift_before_sensor_preflight() -> None:
    train, _ = _partitions()
    rows = {row.token: row for row in train.rows}
    loader = _SyntheticSceneLoader(
        scene_filter=SceneFilter(
            num_history_frames=1,
            num_future_frames=0,
            tokens=list(train.tokens),
        ),
        rows=rows,
    )
    loader.scene_frames_dicts["train-a"][0]["timestamp"] = 999

    with pytest.raises(F2QualificationError, match="row identity"):
        qualify_scene_loader(loader, train)
    assert loader.preflight_calls == 0


class _Agent:
    _training_stage = "critic"
    _critic_variant = "f2_base_residual"

    @staticmethod
    def get_sensor_config() -> SensorConfig:
        return SensorConfig(
            **{
                name: list(indices) if indices else False
                for name, indices in F2_SENSOR_CONTRACT
            }
        )

    @staticmethod
    def get_feature_builders() -> list[Any]:
        return []

    @staticmethod
    def get_target_builders() -> list[Any]:
        return []


class _Dataset:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _build_cfg() -> Any:
    return OmegaConf.create(
        {
            "f2_qualification": {
                "manifest_path": "synthetic-manifest.json",
                "expected_manifest_sha256": hashlib.sha256(
                    b"synthetic-manifest-expectation\n"
                ).hexdigest(),
                "expected_config_sha256": CONFIG_SHA,
                "expected_cache_sha256": CACHE_SHA,
                "expected_source_sha256": SOURCE_SHA,
            },
            "train_test_split": {
                "scene_filter": {
                    "_target_": "navsim.common.dataclasses.SceneFilter",
                    "_convert_": "all",
                    "num_history_frames": 1,
                    "num_future_frames": 0,
                    "frame_interval": 1,
                    "has_route": True,
                    "max_scenes": None,
                    "log_names": ["log-mixed", "log-held"],
                    "tokens": [
                        "train-a",
                        "train-b",
                        "held-a",
                        "rejected-in-same-log",
                    ],
                    "include_synthetic_scenes": False,
                    "synthetic_scene_tokens": None,
                }
            },
            # The official outer log partition is mandatory, but cannot by
            # itself express the retained/rejected split inside log-mixed.
            "train_logs": ["log-mixed"],
            "val_logs": ["log-held"],
            "navsim_log_path": ".",
            "sensor_blobs_path": ".",
            "synthetic_scenes_path": ".",
            "cache_path": None,
            "force_cache_computation": False,
        }
    )


def test_build_datasets_wires_authenticated_manifest_before_loader_construction() -> None:
    manifest = _manifest()
    rows = {
        row.token: row
        for partition in (manifest.train, manifest.held_out)
        for row in partition.rows
    }
    created_loaders: list[_SyntheticSceneLoader] = []
    loader_calls = []

    def manifest_loader(path: Path, **kwargs: Any) -> QualificationManifest:
        loader_calls.append((path, kwargs))
        return manifest

    def scene_loader_factory(**kwargs: Any) -> _SyntheticSceneLoader:
        loader = _SyntheticSceneLoader(rows=rows, **kwargs)
        created_loaders.append(loader)
        return loader

    authenticated_raw_inputs = object()
    with patch(
        "navsim.planning.script.run_training.authenticate_configured_f2_raw_inputs",
        return_value=authenticated_raw_inputs,
    ):
        train_data, val_data = build_datasets(
            _build_cfg(),
            _Agent(),
            scene_loader_factory=scene_loader_factory,
            dataset_factory=_Dataset,
            qualification_manifest_loader=manifest_loader,
        )

    assert len(loader_calls) == 1
    assert loader_calls[0][0] == Path("synthetic-manifest.json")
    assert loader_calls[0][1]["expected_config_sha256"] == CONFIG_SHA
    assert tuple(created_loaders[0].tokens) == manifest.train.tokens
    assert tuple(created_loaders[1].tokens) == manifest.held_out.tokens
    assert created_loaders[0]._scene_filter.log_names == ["log-mixed"]
    assert created_loaders[1]._scene_filter.log_names == ["log-held"]
    assert train_data.kwargs["scene_loader"] is created_loaders[0]
    assert val_data.kwargs["scene_loader"] is created_loaders[1]


@pytest.mark.parametrize(
    ("train_log", "train_token", "held_log", "held_token"),
    (
        ("log-held", "held-a", "log-mixed", "train-a"),
        ("log-outside", "train-a", "log-held", "held-a"),
    ),
    ids=("swapped-official-partitions", "outside-official-logs"),
)
def test_build_datasets_rejects_swapped_or_outside_manifest_logs(
    train_log: str,
    train_token: str,
    held_log: str,
    held_token: str,
) -> None:
    train = build_qualification_partition(
        "train",
        (QualificationRow(train_log, 100, train_token, True),),
        raw_token_count=1,
        raw_log_count=1,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=TRAIN_EXCLUSION_SHA,
    )
    held_out = build_qualification_partition(
        "held_out",
        (QualificationRow(held_log, 300, held_token, True),),
        raw_token_count=1,
        raw_log_count=1,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=HELD_EXCLUSION_SHA,
    )
    manifest = build_qualification_manifest(
        train,
        held_out,
        config_sha256=CONFIG_SHA,
        cache_sha256=CACHE_SHA,
        source_sha256=SOURCE_SHA,
    )
    cfg = _build_cfg()
    # This is the regression seam: even when the base SceneFilter has no log
    # list, cfg.train_logs/cfg.val_logs remain a mandatory outer boundary.
    cfg.train_test_split.scene_filter.log_names = None

    def manifest_loader(*_: Any, **__: Any) -> QualificationManifest:
        return manifest

    def forbidden_scene_loader(**_: Any) -> Any:
        raise AssertionError("an invalid F2 manifest must fail before SceneLoader")

    with pytest.raises(F2QualificationError, match="outside the configured"):
        build_datasets(
            cfg,
            _Agent(),
            scene_loader_factory=forbidden_scene_loader,
            dataset_factory=_Dataset,
            qualification_manifest_loader=manifest_loader,
        )


def test_configured_manifest_is_required_and_authenticated(tmp_path: Path) -> None:
    cfg = _build_cfg()
    del cfg.f2_qualification.manifest_path
    with pytest.raises(ValueError, match="manifest_path"):
        load_configured_f2_qualification_manifest(cfg)

    path = tmp_path / "qualification.json"
    manifest = _write_manifest(path)
    cfg = _build_cfg()
    cfg.f2_qualification.manifest_path = str(path)
    cfg.f2_qualification.expected_manifest_sha256 = qualification_manifest_sha256(
        manifest
    )
    assert load_configured_f2_qualification_manifest(cfg) == manifest


def test_f2_run_protocol_fails_closed_without_qualification_manifest(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.ckpt"
    baseline.touch()
    metadata = (
        tmp_path
        / "train_metric_cache_v2_pruned"
        / "metadata"
        / "cache.csv"
    )
    metadata.parent.mkdir(parents=True)
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "use_cache_without_dataset": False,
            "resume_checkpoint_path": None,
            "agent": {
                "checkpoint_path": "",
                "lr": 5e-5,
                "config": {
                    "actworld_critic_variant": "f2_base_residual",
                    "actworld_training_stage": "critic",
                    "actworld_future_latent_grounding": True,
                    "actworld_freeze_drive_jepa": True,
                    "baseline_checkpoint_path": str(baseline),
                },
            },
            "trainer": {"params": {"max_epochs": 4}},
        }
    )
    with pytest.raises(ValueError, match="manifest_path"):
        validate_f2_run_protocol(
            cfg, runtime_admission_verifier=lambda _: None
        )

    path = tmp_path / "qualification.json"
    manifest = _write_manifest(path)
    content_path, content_sha = _write_cache_content_fixture(
        metadata.parent.parent,
        metadata,
        manifest,
        tmp_path / "cache-content.json",
    )
    cfg.f2_qualification = _content_config(
        path, manifest, metadata, content_path, content_sha
    )
    with patch(
        "navsim.planning.script.run_training.authenticate_configured_f2_raw_inputs"
    ):
        validate_f2_run_protocol(
            cfg, runtime_admission_verifier=lambda _: None
        )


def test_f2_cache_metadata_is_hashed_from_one_explicit_csv(tmp_path: Path) -> None:
    drive_cache = tmp_path / "Drive-JEPA-cache"
    baseline = drive_cache / "baseline.ckpt"
    baseline.parent.mkdir()
    baseline.touch()
    cache_root = drive_cache / "train_metric_cache_v2_pruned"
    metadata = cache_root / "metadata" / "cache.csv"
    metadata.parent.mkdir(parents=True)
    qualification_path = tmp_path / "qualification.json"
    qualification = _write_manifest(qualification_path)
    content_path, content_sha = _write_cache_content_fixture(
        cache_root,
        metadata,
        qualification,
        tmp_path / "cache-content.json",
    )
    cfg = OmegaConf.create(
        {
            "agent": {"config": {"baseline_checkpoint_path": str(baseline)}},
            "f2_qualification": _content_config(
                qualification_path,
                qualification,
                metadata,
                content_path,
                content_sha,
            ),
        }
    )

    loader = authenticate_configured_f2_cache_metadata(cfg)
    assert loader.metadata_path == metadata.resolve()
    assert loader.metadata_sha256 == CACHE_SHA

    extra = metadata.parent / "shadow.csv"
    extra.write_bytes(b"path\n")
    with pytest.raises(ValueError, match="exactly one"):
        authenticate_configured_f2_cache_metadata(cfg)
    extra.unlink()
    broken = metadata.parent / "broken.csv"
    try:
        broken.symlink_to(metadata.parent / "missing.csv")
    except OSError:
        broken = None
    if broken is not None:
        with pytest.raises(ValueError, match="exactly one"):
            authenticate_configured_f2_cache_metadata(cfg)
        broken.unlink()
    cfg.f2_qualification.expected_cache_sha256 = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(ValueError, match="digest|SHA-256"):
        authenticate_configured_f2_cache_metadata(cfg)


@pytest.mark.parametrize("mutation", ("replace", "rewrite"))
def test_strict_metric_cache_consumes_only_same_fd_authenticated_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    cache_root = tmp_path / "train_metric_cache_v2_pruned"
    metadata = cache_root / "metadata" / "cache.csv"
    metadata.parent.mkdir(parents=True)
    trusted = b"path\n/root/a/metric_cache.pkl\n"
    hostile = b"path\n/root/b/metric_cache.pkl\n"
    assert len(trusted) == len(hostile)
    metadata.write_bytes(trusted)
    expected = hashlib.sha256(trusted).hexdigest()
    real_sha256 = hashlib.sha256
    triggered = False

    class TriggerDigest:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._delegate = real_sha256(*args, **kwargs)

        def update(self, payload: bytes) -> None:
            self._delegate.update(payload)

        def hexdigest(self) -> str:
            nonlocal triggered
            result = self._delegate.hexdigest()
            if not triggered:
                triggered = True
                if mutation == "replace":
                    replacement = tmp_path / "replacement.csv"
                    replacement.write_bytes(hostile)
                    replacement.replace(metadata)
                else:
                    metadata.write_bytes(hostile)
            return result

    monkeypatch.setattr(dataloader_module.hashlib, "sha256", TriggerDigest)
    with pytest.raises(ValueError, match="changed"):
        MetricCacheLoader(
            cache_root,
            expected_metadata_path=metadata,
            expected_metadata_sha256=expected,
        )


def test_strict_metric_cache_rejects_uppercase_metadata_digest(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "train_metric_cache_v2_pruned"
    metadata = cache_root / "metadata" / "cache.csv"
    metadata.parent.mkdir(parents=True)
    payload = b"path\n/root/a/metric_cache.pkl\n"
    metadata.write_bytes(payload)
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        MetricCacheLoader(
            cache_root,
            expected_metadata_path=metadata,
            expected_metadata_sha256=hashlib.sha256(payload).hexdigest().upper(),
        )


def test_configured_f2_cache_rejects_metadata_symlink(tmp_path: Path) -> None:
    drive_cache = tmp_path / "Drive-JEPA-cache"
    baseline = drive_cache / "baseline.ckpt"
    baseline.parent.mkdir()
    baseline.touch()
    metadata = (
        drive_cache
        / "train_metric_cache_v2_pruned"
        / "metadata"
        / "cache.csv"
    )
    metadata.parent.mkdir(parents=True)
    qualification_path = tmp_path / "qualification.json"
    qualification = _write_manifest(qualification_path)
    content_path, content_sha = _write_cache_content_fixture(
        metadata.parent.parent,
        metadata,
        qualification,
        tmp_path / "cache-content.json",
    )
    alias = metadata.parent / "cache-alias.csv"
    try:
        alias.symlink_to(metadata)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    cfg = OmegaConf.create(
        {
            "agent": {"config": {"baseline_checkpoint_path": str(baseline)}},
            "f2_qualification": {
                **_content_config(
                    qualification_path,
                    qualification,
                    metadata,
                    content_path,
                    content_sha,
                ),
                "cache_metadata_path": str(alias),
            },
        }
    )
    with pytest.raises(ValueError, match="symlink"):
        authenticate_configured_f2_cache_metadata(cfg)


def test_f2_agent_construction_is_inside_authenticated_vjepa_context(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "Drive-JEPA-cache" / "baseline.ckpt"
    baseline.parent.mkdir()
    baseline.touch()
    cfg = OmegaConf.create(
        {
            "agent": {
                "config": {
                    "actworld_critic_variant": "f2_base_residual",
                    "baseline_checkpoint_path": str(baseline),
                }
            }
        }
    )
    events: list[str] = []
    cache_paths: dict[str, Path] = {}
    authenticated_cache = SimpleNamespace(metric_cache_paths=cache_paths)

    @contextmanager
    def cache_context(_: Any):
        events.append("cache-enter")
        yield authenticated_cache
        events.append("cache-exit")

    @contextmanager
    def context(repository_root: Path, experiment_root: Path, **kwargs: Any):
        assert (repository_root / "navsim_v2").is_dir()
        assert experiment_root == tmp_path.resolve()
        assert kwargs["expected_vjepa_sha256"] == F2_VJEPA_CHECKPOINT_SHA256
        events.append("vjepa-enter")
        yield
        events.append("vjepa-exit")

    def factory(_: Any) -> Any:
        if cfg.agent.config.actworld_critic_variant == "f2_base_residual":
            assert events == ["cache-enter", "vjepa-enter"]
        else:
            assert events == []
        events.append("construct")
        return SimpleNamespace(train_metric_cache_paths=cache_paths)

    build_training_agent(
        cfg,
        agent_factory=factory,
        vjepa_context_factory=context,
        metric_cache_context_factory=cache_context,
    )
    assert events == [
        "cache-enter",
        "vjepa-enter",
        "construct",
        "vjepa-exit",
        "cache-exit",
    ]

    cfg.agent.config.actworld_critic_variant = "legacy"
    events.clear()
    build_training_agent(
        cfg,
        agent_factory=factory,
        vjepa_context_factory=lambda *args, **kwargs: pytest.fail(
            "legacy construction must not enter the F2 V-JEPA context"
        ),
    )
    assert events == ["construct"]


def test_f2_agent_consumes_authenticated_cache_mapping(
    tmp_path: Path,
) -> None:
    drive_cache = tmp_path / "Drive-JEPA-cache"
    baseline = drive_cache / "baseline.ckpt"
    baseline.parent.mkdir()
    baseline.touch()
    cache_root = drive_cache / "train_metric_cache_v2_pruned"
    metadata = cache_root / "metadata" / "cache.csv"
    metadata.parent.mkdir(parents=True)
    qualification_path = tmp_path / "qualification.json"
    qualification = _write_manifest(qualification_path)
    content_path, content_sha = _write_cache_content_fixture(
        cache_root,
        metadata,
        qualification,
        tmp_path / "cache-content.json",
    )
    cfg = OmegaConf.create(
        {
            "agent": {
                "config": {
                    "actworld_critic_variant": "f2_base_residual",
                    "baseline_checkpoint_path": str(baseline),
                }
            },
            "f2_qualification": _content_config(
                qualification_path,
                qualification,
                metadata,
                content_path,
                content_sha,
            ),
        }
    )

    @contextmanager
    def vjepa_context(*args: Any, **kwargs: Any):
        yield

    def factory(_: Any) -> Any:
        from navsim.agents.drive_jepa_perception_based import drive_jepa_agent

        consumed = drive_jepa_agent.MetricCacheLoader(cache_root)
        return SimpleNamespace(
            train_metric_cache_paths=consumed.metric_cache_paths
        )

    agent = build_training_agent(
        cfg,
        agent_factory=factory,
        vjepa_context_factory=vjepa_context,
    )
    assert set(agent.train_metric_cache_paths) == {
        "train-a",
        "train-b",
        "held-a",
    }
    assert all(
        type(value) is AuthenticatedMetricCacheReference
        for value in agent.train_metric_cache_paths.values()
    )
