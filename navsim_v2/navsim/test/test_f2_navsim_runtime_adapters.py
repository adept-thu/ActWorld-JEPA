from __future__ import annotations

import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

import navsim.planning.validation.f2_navsim_runtime_adapters as runtime_adapters_module

from navsim.agents.drive_jepa_perception_based.drive_jepa_config import (
    DriveJEPAConfig,
)
from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
    DriveJEPAFeatureBuilder,
)
from navsim.planning.training.f2_qualification_manifest import (
    F2_SENSOR_CONTRACT,
    QualificationManifest,
    QualificationPartition,
    QualificationRow,
)
from navsim.planning.validation.f2_causal_model_adapter import (
    checkpoint_file_sha256,
)
from navsim.planning.validation.f2_navsim_runtime_adapters import (
    DRIVE_JEPA_FEATURE_SHAPES,
    OFFICIAL_DRIVE_JEPA_CHECKPOINT_PREFIX,
    F2NavsimRuntimeAdapterError,
    QualifiedHeldOutNavtrainFeatureSource,
    authenticated_f2_vjepa_load_context,
    authenticate_drive_jepa_factory_source,
    build_qualified_held_out_navtrain_feature_source,
    restore_authenticated_drive_jepa_baseline,
)
from navsim.planning.validation.f2_navtrain_input_manifest import (
    VerifiedNavtrainRawInputs,
)
from navsim.planning.validation.f2_causal_validator import (
    deterministic_log_assignments,
)
from navsim.planning.validation.f2_live_provenance import VerifiedF2Provenance


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _partition(name: str, rows: tuple[QualificationRow, ...]) -> QualificationPartition:
    logs = {row.log_name for row in rows}
    return QualificationPartition(
        name=name,
        rows=rows,
        raw_token_count=len(rows),
        raw_log_count=len(logs),
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=_sha(f"{name}-exclusion"),
    )


def _qualification() -> QualificationManifest:
    return QualificationManifest(
        train=_partition(
            "train",
            (QualificationRow("log-train", 50, "train-token", True),),
        ),
        held_out=_partition(
            "held_out",
            (
                QualificationRow("log-a", 100, "held-a", True),
                QualificationRow("log-b", 200, "held-b", False),
            ),
        ),
        config_sha256=_sha("config"),
        cache_sha256=_sha("cache"),
        source_sha256=_sha("source"),
    )


class _TinyDriveJepa(torch.nn.Module):
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(width, dtype=torch.float32))
        self.register_buffer("counter", torch.tensor(3, dtype=torch.int64))

    def forward(self, features: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        offset = features["offset"].reshape(1, 1, 1, 1)
        proposals = torch.zeros(
            (1, 32, 8, 3), dtype=torch.float32, device=self.weight.device
        )
        proposals = proposals + offset + self.weight[0]
        logits = torch.zeros(
            (1, 32, 2), dtype=torch.float32, device=self.weight.device
        )
        logits[..., -1] = torch.arange(
            32, dtype=torch.float32, device=self.weight.device
        )
        return {"proposals": proposals, "pred_logit": logits}


def _checkpoint(path: Path, model: torch.nn.Module) -> str:
    state = {
        OFFICIAL_DRIVE_JEPA_CHECKPOINT_PREFIX + key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    torch.save({"state_dict": state, "epoch": 17}, path)
    return checkpoint_file_sha256(path)


def test_official_baseline_restore_hashes_strips_and_strictly_loads(tmp_path: Path) -> None:
    source = _TinyDriveJepa()
    with torch.no_grad():
        source.weight.copy_(torch.tensor([9.0, 11.0]))
    path = tmp_path / "official.ckpt"
    digest = _checkpoint(path, source)

    restored = restore_authenticated_drive_jepa_baseline(
        path,
        expected_checkpoint_sha256=digest,
        model_factory=_TinyDriveJepa,
    )
    output = restored.forward.infer(
        "held-a", {"offset": torch.tensor([0.25], dtype=torch.float64)}
    )

    assert restored.checkpoint_sha256 == digest
    assert restored.state_tensor_count == 2
    assert restored.factory_source is None
    assert restored.live_provenance is None
    assert restored.is_official_factory is False
    assert restored.forward.model.weight.tolist() == [9.0, 11.0]
    assert output.proposals.shape == (32, 8, 3)
    assert output.proposals[0, 0, 0].item() == pytest.approx(9.25)
    assert output.base_logits.tolist() == pytest.approx(list(range(32)))


def test_official_baseline_hash_mismatch_fails_before_deserialization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "official.ckpt"
    _checkpoint(path, _TinyDriveJepa())
    called = False

    def loader(_: Path) -> Any:
        nonlocal called
        called = True
        raise AssertionError("hash mismatch must fail before torch.load")

    with pytest.raises(F2NavsimRuntimeAdapterError, match="SHA-256 mismatch"):
        restore_authenticated_drive_jepa_baseline(
            path,
            expected_checkpoint_sha256=_sha("different"),
            model_factory=_TinyDriveJepa,
            checkpoint_loader=loader,
        )
    assert called is False


@pytest.mark.parametrize("drift", ["namespace", "missing", "unexpected", "shape", "dtype"])
def test_official_baseline_restore_rejects_all_state_drift(
    tmp_path: Path, drift: str
) -> None:
    model = _TinyDriveJepa()
    state = {
        OFFICIAL_DRIVE_JEPA_CHECKPOINT_PREFIX + key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    if drift == "namespace":
        value = state.pop(OFFICIAL_DRIVE_JEPA_CHECKPOINT_PREFIX + "weight")
        state["wrong.weight"] = value
    elif drift == "missing":
        state.pop(OFFICIAL_DRIVE_JEPA_CHECKPOINT_PREFIX + "weight")
    elif drift == "unexpected":
        state[OFFICIAL_DRIVE_JEPA_CHECKPOINT_PREFIX + "extra"] = torch.zeros(1)
    elif drift == "shape":
        state[OFFICIAL_DRIVE_JEPA_CHECKPOINT_PREFIX + "weight"] = torch.zeros(3)
    else:
        state[OFFICIAL_DRIVE_JEPA_CHECKPOINT_PREFIX + "weight"] = torch.zeros(
            2, dtype=torch.float64
        )
    path = tmp_path / f"{drift}.ckpt"
    torch.save({"state_dict": state}, path)

    with pytest.raises(F2NavsimRuntimeAdapterError):
        restore_authenticated_drive_jepa_baseline(
            path,
            expected_checkpoint_sha256=checkpoint_file_sha256(path),
            model_factory=_TinyDriveJepa,
        )


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_factory_source_authentication_binds_live_bytes_to_full_commit(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "f2-test@example.invalid")
    _git(repository, "config", "user.name", "F2 Test")
    runtime = repository / "runtime.py"
    runtime.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "runtime.py")
    _git(repository, "commit", "-m", "runtime")
    revision = _git(repository, "rev-parse", "HEAD")

    identity = authenticate_drive_jepa_factory_source(
        repository,
        expected_revision=revision,
        paths=("runtime.py",),
    )

    assert identity.revision == revision
    assert identity.paths == ("runtime.py",)
    assert len(identity.path_tree_sha256) == 64

    runtime.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(F2NavsimRuntimeAdapterError, match="differs"):
        authenticate_drive_jepa_factory_source(
            repository,
            expected_revision=revision,
            paths=("runtime.py",),
        )


@pytest.mark.parametrize(
    "shadow_name",
    (
        "shadow.py",
        "shadow.pyc",
        "shadow.cpython-39-x86_64-linux-gnu.so",
    ),
)
def test_factory_source_authentication_rejects_ignored_python_shadows(
    tmp_path: Path, shadow_name: str
) -> None:
    repository = tmp_path / "repo"
    factory = repository / "factory"
    factory.mkdir(parents=True)
    _git(repository, "init")
    _git(repository, "config", "user.email", "f2-test@example.invalid")
    _git(repository, "config", "user.name", "F2 Test")
    (factory / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / ".gitignore").write_text(
        f"factory/{shadow_name}\n", encoding="utf-8"
    )
    _git(repository, "add", ".gitignore", "factory/runtime.py")
    _git(repository, "commit", "-m", "runtime")
    revision = _git(repository, "rev-parse", "HEAD")
    (factory / shadow_name).write_bytes(b"ignored shadow\n")

    with pytest.raises(F2NavsimRuntimeAdapterError, match="ignored files can shadow"):
        authenticate_drive_jepa_factory_source(
            repository,
            expected_revision=revision,
            paths=("factory",),
        )


def test_f2_vjepa_context_loads_authenticated_same_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    experiment_root = tmp_path / "experiment"
    asset = (
        experiment_root
        / "Drive-JEPA-cache"
        / runtime_adapters_module.OFFICIAL_DRIVE_JEPA_VJEPA_NAME
    )
    asset.parent.mkdir(parents=True)
    torch.save({"trusted": torch.tensor([7])}, asset)
    digest = checkpoint_file_sha256(asset)
    monkeypatch.setattr(
        runtime_adapters_module,
        "OFFICIAL_DRIVE_JEPA_VJEPA_SHA256",
        digest,
    )
    observed: list[Any] = []

    with pytest.raises(F2NavsimRuntimeAdapterError, match="changed while loaded"):
        with authenticated_f2_vjepa_load_context(
            repository_root,
            experiment_root,
            expected_vjepa_sha256=digest,
        ):
            original = asset.with_name("authenticated-original.pt")
            asset.rename(original)
            torch.save({"replacement": torch.tensor([99])}, asset)
            import vjepa2.evals.image_classification_frozen.modelcustom.vit_encoder as vit_encoder

            observed.append(vit_encoder.torch.load(asset, map_location="cpu"))

    assert observed[0]["trusted"].tolist() == [7]


def _feature_values() -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in DRIVE_JEPA_FEATURE_SHAPES.items()
    }


class _Loader:
    def __init__(
        self,
        qualification: QualificationManifest,
        rows: tuple[QualificationRow, ...] | None = None,
    ) -> None:
        selected = qualification.held_out.rows if rows is None else rows
        self.tokens = [row.token for row in selected]
        self._sensor_config = {
            name: list(indices) for name, indices in F2_SENSOR_CONTRACT
        }
        self._scene_filter = SimpleNamespace(num_history_frames=4)
        self.scene_frames_dicts = {}
        for row in selected:
            self.scene_frames_dicts[row.token] = [
                {},
                {},
                {},
                {
                    "log_name": row.log_name,
                    "timestamp": row.time_us,
                    "token": row.token,
                },
            ]


class _Dataset:
    def __init__(
        self,
        loader: _Loader,
        *,
        targets: Mapping[str, Any] | None = None,
    ) -> None:
        self._scene_loader = loader
        self._feature_builders = [DriveJEPAFeatureBuilder(DriveJEPAConfig())]
        self._target_builders = []
        self._cache_path = None
        self._targets = dict(targets or {})

    def __len__(self) -> int:
        return len(self._scene_loader.tokens)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return _feature_values(), dict(self._targets)


def _verified(qualification: QualificationManifest) -> VerifiedF2Provenance:
    """Unit-only sealed-type stand-in; production construction is separately closed."""

    provenance = object.__new__(VerifiedF2Provenance)
    object.__setattr__(provenance, "qualification", qualification)
    return provenance


def _source(
    qualification: QualificationManifest,
    *,
    targets: Mapping[str, Any] | None = None,
) -> QualifiedHeldOutNavtrainFeatureSource:
    loader = _Loader(qualification)
    return QualifiedHeldOutNavtrainFeatureSource(
        qualification=qualification,
        live_provenance=_verified(qualification),
        scene_loader=loader,
        feature_dataset=_Dataset(loader, targets=targets),
    )


def test_qualified_feature_source_preserves_manifest_order_identity_and_batch() -> None:
    qualification = _qualification()
    source = _source(qualification)
    row = qualification.held_out.rows[1]

    result = source.load(row)

    assert source.tokens == ("held-a", "held-b")
    assert (result.log_name, result.time_us, result.token) == (
        "log-b",
        200,
        "held-b",
    )
    assert set(result.features) == set(DRIVE_JEPA_FEATURE_SHAPES)
    for name, shape in DRIVE_JEPA_FEATURE_SHAPES.items():
        assert result.features[name].shape == (1,) + shape
        assert result.features[name].dtype == torch.float32
        assert result.features[name].is_contiguous()


@pytest.mark.parametrize("world_size", (1, 2, 8))
def test_concrete_feature_source_accepts_exact_whole_log_rank_views(
    world_size: int,
) -> None:
    qualification = _qualification()
    provenance = _verified(qualification)
    assignments = deterministic_log_assignments(
        qualification.held_out.rows, world_size
    )
    for rank, logs in enumerate(assignments):
        rows = tuple(
            row for log in logs for row in qualification.held_out.rows
            if row.log_name == log
        )
        loader = _Loader(qualification, rows)
        source = QualifiedHeldOutNavtrainFeatureSource(
            qualification=qualification,
            live_provenance=provenance,
            scene_loader=loader,
            feature_dataset=_Dataset(loader),
            rank=rank,
            world_size=world_size,
        )
        assert type(source) is QualifiedHeldOutNavtrainFeatureSource
        assert source.live_provenance is provenance
        assert source.rank == rank
        assert source.world_size == world_size
        assert source.logs == logs
        assert source.tokens == tuple(row.token for row in rows)


@pytest.mark.parametrize("world_size", (1, 2, 8))
def test_production_rank_factory_builds_the_concrete_runner_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
) -> None:
    qualification = _qualification()
    provenance = _verified(qualification)
    rows_by_token = {row.token: row for row in qualification.held_out.rows}

    class _FactoryLoader:
        def __init__(self, *, scene_filter: Any, sensor_config: Any, **_: Any) -> None:
            self._scene_filter = scene_filter
            self._sensor_config = sensor_config
            self.tokens = list(scene_filter.tokens)
            self.synthetic_scenes = {}
            self.scene_frames_dicts = {}
            for token in self.tokens:
                row = rows_by_token[token]
                self.scene_frames_dicts[token] = [
                    {},
                    {},
                    {},
                    {
                        "log_name": row.log_name,
                        "timestamp": row.time_us,
                        "token": row.token,
                    },
                ]

        def filter_scenes_by_sensor_availability(self) -> Any:
            return SimpleNamespace(
                removed_scenes=0,
                total_scenes=len(self.tokens),
                retained_scenes=len(self.tokens),
                removed_regular_tokens=(),
                removed_synthetic_tokens=(),
            )

    class _FactoryDataset:
        def __init__(
            self,
            *,
            scene_loader: Any,
            feature_builders: list[Any],
            target_builders: list[Any],
            cache_path: Any,
            **_: Any,
        ) -> None:
            self._scene_loader = scene_loader
            self._feature_builders = feature_builders
            self._target_builders = target_builders
            self._cache_path = cache_path

        def __len__(self) -> int:
            return len(self._scene_loader.tokens)

        def __getitem__(self, index: int) -> Any:
            del index
            return _feature_values(), {}

    import navsim.common.dataloader as dataloader_module
    import navsim.planning.training.dataset as dataset_module

    monkeypatch.setattr(dataloader_module, "SceneLoader", _FactoryLoader)
    monkeypatch.setattr(dataset_module, "Dataset", _FactoryDataset)
    authenticated_raw_inputs = object.__new__(VerifiedNavtrainRawInputs)
    monkeypatch.setattr(
        VerifiedNavtrainRawInputs,
        "require_roots",
        lambda self, log_root, sensor_root: None,
    )
    log_root = tmp_path / "logs"
    sensor_root = tmp_path / "sensors"
    log_root.mkdir()
    sensor_root.mkdir()
    assignments = deterministic_log_assignments(
        qualification.held_out.rows, world_size
    )
    for rank, logs in enumerate(assignments):
        built = build_qualified_held_out_navtrain_feature_source(
            qualification,
            live_provenance=provenance,
            navsim_log_path=log_root,
            sensor_blobs_path=sensor_root,
            authenticated_raw_inputs=authenticated_raw_inputs,
            rank=rank,
            world_size=world_size,
        )
        assert type(built.source) is QualifiedHeldOutNavtrainFeatureSource
        assert built.source.live_provenance is provenance
        assert built.source.logs == logs


def test_qualified_feature_source_requires_matching_concrete_live_provenance() -> None:
    qualification = _qualification()
    loader = _Loader(qualification)
    dataset = _Dataset(loader)
    with pytest.raises(F2NavsimRuntimeAdapterError, match="concrete sealed"):
        QualifiedHeldOutNavtrainFeatureSource(
            qualification=qualification,
            live_provenance=SimpleNamespace(qualification=qualification),
            scene_loader=loader,
            feature_dataset=dataset,
        )

    different = replace(qualification, source_sha256=_sha("different-source"))
    with pytest.raises(F2NavsimRuntimeAdapterError, match="qualification differs"):
        QualifiedHeldOutNavtrainFeatureSource(
            qualification=qualification,
            live_provenance=_verified(different),
            scene_loader=loader,
            feature_dataset=dataset,
        )

    provenance = _verified(qualification)
    source = QualifiedHeldOutNavtrainFeatureSource(
        qualification=qualification,
        live_provenance=provenance,
        scene_loader=loader,
        feature_dataset=dataset,
    )
    assert source.live_provenance is provenance


def test_qualified_feature_source_rejects_order_identity_sensor_and_target_drift() -> None:
    qualification = _qualification()

    loader = _Loader(qualification)
    loader.tokens.reverse()
    with pytest.raises(F2NavsimRuntimeAdapterError, match="order differs"):
        QualifiedHeldOutNavtrainFeatureSource(
            qualification=qualification,
            live_provenance=_verified(qualification),
            scene_loader=loader,
            feature_dataset=_Dataset(loader),
        )

    loader = _Loader(qualification)
    loader.scene_frames_dicts["held-a"][3]["timestamp"] += 1
    with pytest.raises(F2NavsimRuntimeAdapterError, match="identity drifted"):
        QualifiedHeldOutNavtrainFeatureSource(
            qualification=qualification,
            live_provenance=_verified(qualification),
            scene_loader=loader,
            feature_dataset=_Dataset(loader),
        )

    loader = _Loader(qualification)
    loader._sensor_config["cam_f0"] = [2, 3]
    with pytest.raises(Exception, match="sensor contract drifted"):
        QualifiedHeldOutNavtrainFeatureSource(
            qualification=qualification,
            live_provenance=_verified(qualification),
            scene_loader=loader,
            feature_dataset=_Dataset(loader),
        )

    source = _source(qualification, targets={"trajectory": torch.zeros(8, 3)})
    with pytest.raises(F2NavsimRuntimeAdapterError, match="target"):
        source.load(qualification.held_out.rows[0])

    source = _source(qualification)
    drifted = replace(qualification.held_out.rows[0], time_us=101)
    with pytest.raises(F2NavsimRuntimeAdapterError, match="canonical"):
        source.load(drifted)


def test_qualified_feature_source_rejects_feature_shape_or_key_drift() -> None:
    qualification = _qualification()
    loader = _Loader(qualification)
    dataset = _Dataset(loader)
    source = QualifiedHeldOutNavtrainFeatureSource(
        qualification=qualification,
        live_provenance=_verified(qualification),
        scene_loader=loader,
        feature_dataset=dataset,
    )

    dataset.__class__.__getitem__ = lambda self, index: (
        {**_feature_values(), "camera_feature_1": torch.zeros(3, 128, 512)},
        {},
    )
    with pytest.raises(F2NavsimRuntimeAdapterError, match="shape"):
        source.load(qualification.held_out.rows[0])
