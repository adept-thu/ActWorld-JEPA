"""Synthetic end-to-end tests for the F2 causal shard runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
import torch

from navsim.planning.training.f2_qualification_manifest import (
    F2_SENSOR_CONTRACT,
    QualificationRow,
    build_qualification_manifest,
    build_qualification_partition,
    canonical_manifest_bytes as canonical_qualification_manifest_bytes,
)
from navsim.planning.validation.f2_causal_model_adapter import (
    F2_FROZEN_ACTOR_TENSOR_NAMES,
    F2CausalModelAdapter,
    canonical_original_proposal_sha256,
    checkpoint_file_sha256,
)
from navsim.planning.validation.f2_causal_shard_runner import (
    F2CausalShardRunnerError,
    evaluate_causal_shard,
    merge_causal_world,
    prove_1_2_8_invariance,
)
from navsim.planning.validation.f2_causal_validator import (
    F2_MODEL_NUMERIC_MODE,
    ManifestRow,
    ModelTokenOutput,
    build_canonical_manifest,
    deterministic_log_assignments,
    manifest_sha256,
)
from navsim.planning.validation.f2_frozen_artifact_io import (
    ArtifactBindings,
    CoverageEntry,
    FrozenArtifactChunk,
    FrozenTokenRecord,
    TARGET_CHANNEL_ORDER,
    build_artifact_coverage,
    seal_frozen_artifact,
)
from navsim.planning.validation.f2_navsim_record_builder import TokenFeatures
from navsim.planning.validation.f2_live_provenance import (
    EXPECTED_PROPOSAL_SCHEMA_NAME,
    EXPECTED_SCORER_NAME,
    EXPECTED_SIMULATOR_NAME,
    QUALIFICATION_SOURCE_SCHEMA_VERSION,
    REQUIRED_RUNTIME_SOURCE_PATHS,
    build_runtime_source_manifest,
    canonical_runtime_source_manifest_bytes,
    verify_live_f2_provenance,
)
from navsim.planning.validation.f2_metric_cache_manifest import (
    F2_PRUNED_CACHE_FAMILY,
    build_metric_cache_manifest,
    write_metric_cache_manifest,
)
from navsim.planning.validation.f2_navsim_comfort import (
    DriveJepaExtendedComfort,
    _OFFICIAL_COMFORT_SEAL,
)
from navsim.planning.validation.f2_navsim_runtime_adapters import (
    DRIVE_JEPA_FEATURE_SHAPES,
    OFFICIAL_DRIVE_JEPA_FACTORY_REVISION,
    DriveJepaFactorySourceIdentity,
    QualifiedHeldOutNavtrainFeatureSource,
)
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import (
    DriveJEPAConfig,
)
from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
    DriveJEPAFeatureBuilder,
)


def _sha(label: str) -> str:
    return hashlib.sha256((label + "\n").encode("utf-8")).hexdigest()


CONFIG_SHA = _sha("qualification-config")
RUNTIME_CONFIG_SHA = _sha("runtime-config")
CACHE_TOKENS = ("train-0", "a0", "a1", "b0", "b1", "c0", "c1")
CACHE_METADATA_BYTES = (
    "file_name\n"
    + "".join(f"cache/{token}/metric_cache.pkl\n" for token in CACHE_TOKENS)
).encode("utf-8")
CACHE_SHA = hashlib.sha256(CACHE_METADATA_BYTES).hexdigest()
SOURCE_REVISION = "f2-synthetic-source"
BASELINE_SHA = _sha("baseline")
VJEPA_BYTES = b"vjepa-checkpoint\n"
VJEPA_SHA = hashlib.sha256(VJEPA_BYTES).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


QUALIFICATION_SOURCE_FILE_BYTES = b"qualification-source\n"
QUALIFICATION_SOURCE_BYTES = _canonical_json_bytes(
    {
        "files": [
            {
                "bytes": len(QUALIFICATION_SOURCE_FILE_BYTES),
                "path": "qualification_source.py",
                "sha256": hashlib.sha256(
                    QUALIFICATION_SOURCE_FILE_BYTES
                ).hexdigest(),
            }
        ],
        "foundation_commit": "synthetic-foundation",
        "schema_version": QUALIFICATION_SOURCE_SCHEMA_VERSION,
    }
)
QUALIFICATION_SOURCE_SHA = hashlib.sha256(QUALIFICATION_SOURCE_BYTES).hexdigest()


def _rows() -> tuple[ManifestRow, ...]:
    return (
        ManifestRow("log-a", 100_000, "a0", True),
        ManifestRow("log-a", 600_000, "a1", True),
        ManifestRow("log-b", 100_000, "b0", True),
        ManifestRow("log-b", 600_000, "b1", True),
        ManifestRow("log-c", 100_000, "c0", True),
        ManifestRow("log-c", 600_000, "c1", True),
    )


def _manifest():
    rows = _rows()
    tokens = tuple(row.token for row in rows)
    return build_canonical_manifest(
        rows,
        expected_tokens=tokens,
        sensor_available_tokens=tokens,
        cache_tokens=tokens,
        allowlist_logs=("log-a", "log-b", "log-c"),
        cache_metadata_sha256=CACHE_SHA,
        source_revision=SOURCE_REVISION,
    )


def _qualification(manifest: Any):
    train = build_qualification_partition(
        "train",
        (QualificationRow("log-train", 1, "train-0", True),),
        raw_token_count=1,
        raw_log_count=1,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=_sha("train-exclusions"),
    )
    held = build_qualification_partition(
        "held_out",
        tuple(
            QualificationRow(
                row.log_name, row.time_us, row.token, row.future_camera_valid
            )
            for row in manifest.rows
        ),
        raw_token_count=len(manifest.rows),
        raw_log_count=len(manifest.logs),
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=_sha("held-exclusions"),
    )
    return build_qualification_manifest(
        train,
        held,
        config_sha256=CONFIG_SHA,
        cache_sha256=CACHE_SHA,
        source_sha256=QUALIFICATION_SOURCE_SHA,
    )


def _bindings(
    manifest_sha: str,
    qualification_sha: str,
    runtime_source: Any,
    runtime_source_sha: str,
) -> ArtifactBindings:
    return ArtifactBindings(
        qualification_manifest_sha256=qualification_sha,
        causal_manifest_sha256=manifest_sha,
        baseline_checkpoint_sha256=BASELINE_SHA,
        vjepa_checkpoint_sha256=VJEPA_SHA,
        cache_metadata_sha256=CACHE_SHA,
        metric_cache_manifest_sha256=_sha("metric-cache-manifest"),
        raw_input_manifest_sha256=_sha("raw-input-manifest"),
        qualification_config_sha256=CONFIG_SHA,
        runtime_config_sha256=RUNTIME_CONFIG_SHA,
        qualification_source_manifest_sha256=QUALIFICATION_SOURCE_SHA,
        runtime_source_manifest_sha256=runtime_source_sha,
        source_revision=SOURCE_REVISION,
        proposal_schema_name=EXPECTED_PROPOSAL_SCHEMA_NAME,
        proposal_schema_sha256=runtime_source.component_sha256("proposal_schema"),
        simulator_name=EXPECTED_SIMULATOR_NAME,
        simulator_sha256=runtime_source.component_sha256("dual_simulator"),
        scorer_name=EXPECTED_SCORER_NAME,
        scorer_sha256=runtime_source.component_sha256(
            "training_scorer_and_comfort"
        ),
        channel_order=TARGET_CHANNEL_ORDER,
    )


def _live_bundle(
    tmp_path: Path,
    manifest: Any,
    manifest_sha: str,
    *,
    cache_root_override: Path | None = None,
):
    root = tmp_path / "provenance"
    root.mkdir(exist_ok=True)
    qualification = _qualification(manifest)
    qualification_path = root / "qualification.json"
    causal_path = root / "causal.json"
    qualification_config_path = root / "qualification-config.json"
    runtime_config_path = root / "runtime-config.yaml"
    cache_root = (
        root / F2_PRUNED_CACHE_FAMILY
        if cache_root_override is None
        else cache_root_override
    )
    cache_path = cache_root / "metadata" / "cache.csv"
    cache_manifest_path = root / "cache-content.json"
    baseline_path = root / "baseline.ckpt"
    vjepa_path = root / "vjepa.pt"
    qualification_source_path = root / "qualification-source.json"
    qualification_source_root = root / "qualification-source-root"
    runtime_source_path = root / "runtime-source.json"
    runtime_source_root = root / "runtime-source-root"
    qualification_source_root.mkdir(exist_ok=True)
    runtime_source_root.mkdir(exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for token in CACHE_TOKENS:
        entry_path = cache_root / "cache" / token / "metric_cache.pkl"
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_bytes((token + "-cache\n").encode("utf-8"))
    qualification_path.write_bytes(canonical_qualification_manifest_bytes(qualification))
    from navsim.planning.validation.f2_causal_validator import canonical_manifest_bytes

    causal_path.write_bytes(canonical_manifest_bytes(manifest))
    qualification_config_path.write_bytes(b"qualification-config\n")
    runtime_config_path.write_bytes(b"runtime-config\n")
    cache_path.write_bytes(CACHE_METADATA_BYTES)
    baseline_path.write_bytes(b"baseline\n")
    vjepa_path.write_bytes(VJEPA_BYTES)
    qualification_source_path.write_bytes(QUALIFICATION_SOURCE_BYTES)
    (qualification_source_root / "qualification_source.py").write_bytes(
        QUALIFICATION_SOURCE_FILE_BYTES
    )
    for relative in REQUIRED_RUNTIME_SOURCE_PATHS:
        path = runtime_source_root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode("utf-8"))
    runtime_source = build_runtime_source_manifest(
        runtime_source_root, source_revision=SOURCE_REVISION
    )
    runtime_source_bytes = canonical_runtime_source_manifest_bytes(runtime_source)
    runtime_source_path.write_bytes(runtime_source_bytes)
    qualification_sha = hashlib.sha256(qualification_path.read_bytes()).hexdigest()
    cache_manifest = build_metric_cache_manifest(
        cache_root=cache_root,
        metadata_path=cache_path,
        expected_metadata_sha256=CACHE_SHA,
        qualification=qualification,
        expected_qualification_manifest_sha256=qualification_sha,
        workers=2,
    )
    cache_manifest_sha = write_metric_cache_manifest(
        cache_manifest_path.resolve(), cache_manifest
    )
    bindings = _bindings(
        manifest_sha,
        qualification_sha,
        runtime_source,
        hashlib.sha256(runtime_source_bytes).hexdigest(),
    )
    bindings = replace(
        bindings, metric_cache_manifest_sha256=cache_manifest_sha
    )
    verified = verify_live_f2_provenance(
        bindings=bindings,
        qualification_manifest_path=qualification_path,
        causal_manifest_path=causal_path,
        qualification_config_path=qualification_config_path,
        runtime_config_path=runtime_config_path,
        cache_metadata_path=cache_path,
        metric_cache_manifest_path=cache_manifest_path,
        metric_cache_root=cache_root,
        baseline_checkpoint_path=baseline_path,
        vjepa_checkpoint_path=vjepa_path,
        qualification_source_manifest_path=qualification_source_path,
        qualification_source_root=qualification_source_root,
        runtime_source_manifest_path=runtime_source_path,
        runtime_source_root=runtime_source_root,
    )
    return bindings, verified


def _base() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 32, dtype="<f4")


def _targets() -> np.ndarray:
    candidate = np.arange(32, dtype=np.float32) / np.float32(31.0)
    targets = np.repeat(candidate[:, None], 9, axis=1)
    # The ninth stored channel is the static 14-weight aggregate.  With all
    # eight primitive channels equal to x, the official formula is x**5.
    targets[:, 8] = candidate**5
    return np.ascontiguousarray(targets, dtype="<f4")


def _proposals() -> torch.Tensor:
    return torch.arange(32 * 8 * 3, dtype=torch.float32).reshape(1, 32, 8, 3) / 1000.0


def _record(row: ManifestRow) -> FrozenTokenRecord:
    calibration = np.zeros((32, 41, 11), dtype="<f8")
    carry = np.zeros((32, 41, 11), dtype="<f8")
    calibration[..., 0] = np.arange(32, dtype=np.float64)[:, None]
    carry[..., 0] = np.arange(32, dtype=np.float64)[:, None] + 100.0
    return FrozenTokenRecord(
        token=row.token,
        time_us=row.time_us,
        proposal_sha256=canonical_original_proposal_sha256(_proposals()),
        base=_base(),
        targets=_targets(),
        calibration=calibration,
        carry=carry,
    )


def _artifact(tmp_path: Path, *, cache_root: Path | None = None):
    manifest = _manifest()
    manifest_sha = manifest_sha256(manifest)
    bindings, live_provenance = _live_bundle(
        tmp_path,
        manifest,
        manifest_sha,
        cache_root_override=cache_root,
    )
    coverage = build_artifact_coverage(
        CoverageEntry(row.log_name, row.time_us, row.token) for row in manifest.rows
    )
    path = tmp_path / "sealed-f2.bin"
    chunks = []
    for log_name in ("log-a", "log-b", "log-c"):
        chunks.append(
            FrozenArtifactChunk(
                log_name,
                tuple(_record(row) for row in manifest.rows if row.log_name == log_name),
            )
        )
    descriptor = seal_frozen_artifact(
        path, bindings=bindings, coverage=coverage, chunks=chunks
    )
    return (
        manifest,
        manifest_sha,
        bindings,
        live_provenance,
        coverage,
        path,
        descriptor.root_sha256,
    )


class _QualifiedLoader:
    def __init__(self, rows: tuple[Any, ...]) -> None:
        self.tokens = [row.token for row in rows]
        self._sensor_config = {
            name: list(indices) for name, indices in F2_SENSOR_CONTRACT
        }
        self._scene_filter = SimpleNamespace(num_history_frames=4)
        self.scene_frames_dicts = {
            row.token: [
                {},
                {},
                {},
                {
                    "log_name": row.log_name,
                    "timestamp": row.time_us,
                    "token": row.token,
                },
            ]
            for row in rows
        }


class _QualifiedDataset:
    _features = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in DRIVE_JEPA_FEATURE_SHAPES.items()
    }

    def __init__(self, loader: _QualifiedLoader) -> None:
        self._scene_loader = loader
        self._feature_builders = [DriveJEPAFeatureBuilder(DriveJEPAConfig())]
        self._target_builders: list[Any] = []
        self._cache_path = None

    def __len__(self) -> int:
        return len(self._scene_loader.tokens)

    def __getitem__(self, index: int) -> Any:
        del index
        return self._features, {}


def _feature_source(
    rows: tuple[ManifestRow, ...],
    *,
    live_provenance: Any,
    rank: int = 0,
    world_size: int = 1,
    drift: bool = False,
) -> QualifiedHeldOutNavtrainFeatureSource:
    qualification_rows = tuple(
        row
        for row in live_provenance.qualification.held_out.rows
        if row.token in {item.token for item in rows}
    )
    loader = _QualifiedLoader(qualification_rows)
    source = QualifiedHeldOutNavtrainFeatureSource(
        qualification=live_provenance.qualification,
        live_provenance=live_provenance,
        scene_loader=loader,
        feature_dataset=_QualifiedDataset(loader),
        rank=rank,
        world_size=world_size,
    )
    if drift:
        original_load = source.load

        def drifted_load(row: ManifestRow) -> TokenFeatures:
            item = original_load(row)
            return TokenFeatures(
                token=item.token,
                log_name="wrong-log",
                time_us=item.time_us,
                features=item.features,
            )

        source.load = drifted_load  # type: ignore[method-assign]
    return source


class _TinyF2Model(torch.nn.Module):
    def __init__(self, *, proposal_drift: bool = False) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.proposal_drift = proposal_drift

    def forward(self, features: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        assert set(features) == set(DRIVE_JEPA_FEATURE_SHAPES)
        proposals = _proposals().to(self.anchor.device) + self.anchor * 0.0
        if self.proposal_drift:
            proposals = proposals + 1.0
        base = torch.from_numpy(_base()).to(self.anchor.device)[None] + self.anchor * 0.0
        residual = torch.zeros_like(base)
        delta = torch.tanh(residual)
        utility = base.detach() + delta
        return {
            "proposals": proposals,
            "actworld_original_candidates": proposals.clone(),
            "pred_logit": torch.stack((base - 1.0, base), dim=-1),
            "actworld_base_candidate_logits": base,
            "actworld_residual_logits": residual,
            "actworld_delta_logits": delta,
            "actworld_utility_logits": utility,
        }


def _frozen_actor_tensors(model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
    return {
        name: torch.zeros_like(model.anchor)
        for name in F2_FROZEN_ACTOR_TENSOR_NAMES
    }


def _model_adapter(
    tmp_path: Path,
    *,
    live_provenance: Any | None = None,
    proposal_drift: bool = False,
    checkpoint_root: Path | None = None,
):
    checkpoint = (tmp_path if checkpoint_root is None else checkpoint_root) / "epoch-0.ckpt"
    if not checkpoint.exists():
        torch.save(
            {
                "actworld_training_stage": "critic",
                "actworld_critic_variant": "f2_base_residual",
                "epoch": 0,
                "state_dict": {"anchor": torch.tensor(0.0, dtype=torch.float32)},
            },
            checkpoint,
            _use_new_zipfile_serialization=False,
        )
    adapter = F2CausalModelAdapter.from_checkpoint(
        checkpoint,
        expected_sha256=checkpoint_file_sha256(checkpoint),
        expected_epoch=0,
        model_factory=lambda: _TinyF2Model(proposal_drift=proposal_drift),
        frozen_actor_tensor_getter=_frozen_actor_tensors,
    )
    if live_provenance is not None:
        # Synthetic tests do not instantiate the 307M-parameter production
        # factory.  Mark this already strict-restored tiny adapter with the
        # same private evidence that the production builder must supply.
        adapter._official_factory = True
        adapter._live_provenance = live_provenance
        adapter._factory_source = DriveJepaFactorySourceIdentity(
            repository_root=tmp_path,
            revision=OFFICIAL_DRIVE_JEPA_FACTORY_REVISION,
            path_tree_sha256=_sha("synthetic-factory-source"),
            paths=("synthetic",),
        )
    return adapter


class _ForgedModelAdapter:
    checkpoint_identity = SimpleNamespace(
        sha256=_sha("forged-checkpoint"),
        epoch=0,
        stage="critic",
        variant="f2_base_residual",
    )

    def infer_token(self, token: str, features: Mapping[str, Any]) -> Any:
        raise AssertionError("forged adapter must be rejected before inference")


def _comfort_adapter() -> DriveJepaExtendedComfort:
    def kernel(past: np.ndarray, current: np.ndarray, time_s: np.ndarray) -> np.ndarray:
        assert past.shape == current.shape
        assert time_s.ndim == 1
        return np.ones((past.shape[0],), dtype=np.bool_)

    return DriveJepaExtendedComfort(
        kernel, _official_seal=_OFFICIAL_COMFORT_SEAL
    )


def _run_world(
    tmp_path: Path,
    world_size: int,
    *,
    reader_mode: str = "stream",
    cache_root: Path | None = None,
):
    (
        manifest,
        manifest_sha,
        bindings,
        live_provenance,
        coverage,
        path,
        artifact_sha,
    ) = _artifact(tmp_path, cache_root=cache_root)
    grouped = {log: [] for log in manifest.logs}
    for row in manifest.rows:
        grouped[row.log_name].append(row)
    assignments = deterministic_log_assignments(manifest.rows, world_size)
    shards = []
    for rank, logs in enumerate(assignments):
        assigned = tuple(row for log in logs for row in grouped[log])
        shards.append(
            evaluate_causal_shard(
                manifest=manifest,
                artifact_path=path,
                artifact_bindings=bindings,
                artifact_coverage=coverage,
                expected_artifact_sha256=artifact_sha,
                live_provenance=live_provenance,
                model_adapter=_model_adapter(
                    tmp_path,
                    live_provenance=live_provenance,
                    checkpoint_root=tmp_path.parent,
                ),
                feature_source=_feature_source(
                    assigned,
                    live_provenance=live_provenance,
                    rank=rank,
                    world_size=world_size,
                ),
                comfort_function=_comfort_adapter(),
                rank=rank,
                world_size=world_size,
                reader_mode=reader_mode,
            )
        )
    return manifest, merge_causal_world(manifest, shards)


def test_exact_1_2_8_world_size_invariance(tmp_path: Path) -> None:
    worlds = []
    for world_size in (1, 2, 8):
        run_dir = tmp_path / str(world_size)
        run_dir.mkdir()
        manifest, world = _run_world(
            run_dir,
            world_size,
            cache_root=tmp_path / F2_PRUNED_CACHE_FAMILY,
        )
        assert world.metrics.coverage_fraction == 1.0
        assert world.metrics.fatal_count == 0
        assert world.metrics.critic_delta == 0.0
        assert world.checkpoint_evaluation.epoch == 0
        worlds.append(world)

    selected = prove_1_2_8_invariance(worlds)
    assert selected.world_size == 1
    assert worlds[0].evaluation_sha256 == worlds[1].evaluation_sha256
    assert worlds[0].metrics_sha256 == worlds[2].metrics_sha256


def test_mmap_reader_releases_all_exported_views(tmp_path: Path) -> None:
    _, world = _run_world(tmp_path, 1, reader_mode="mmap")
    assert world.metrics.observed_token_count == 6


def test_rank_preflight_rejects_feature_coverage_and_identity_drift(
    tmp_path: Path,
) -> None:
    manifest, _, bindings, provenance, coverage, path, artifact_sha = _artifact(tmp_path)
    assignments = deterministic_log_assignments(manifest.rows, 2)
    wrong_assignment = tuple(
        row for row in manifest.rows if row.log_name in assignments[1]
    )
    with pytest.raises(F2CausalShardRunnerError, match="rank/world_size"):
        evaluate_causal_shard(
            manifest=manifest,
            artifact_path=path,
            artifact_bindings=bindings,
            artifact_coverage=coverage,
            expected_artifact_sha256=artifact_sha,
            live_provenance=provenance,
            model_adapter=_model_adapter(tmp_path, live_provenance=provenance),
            feature_source=_feature_source(
                wrong_assignment,
                live_provenance=provenance,
                rank=1,
                world_size=2,
            ),
            comfort_function=_comfort_adapter(),
            rank=0,
            world_size=2,
        )

    with pytest.raises(F2CausalShardRunnerError, match="feature row identity"):
        evaluate_causal_shard(
            manifest=manifest,
            artifact_path=path,
            artifact_bindings=bindings,
            artifact_coverage=coverage,
            expected_artifact_sha256=artifact_sha,
            live_provenance=provenance,
            model_adapter=_model_adapter(tmp_path, live_provenance=provenance),
            feature_source=_feature_source(
                manifest.rows, live_provenance=provenance, drift=True
            ),
            comfort_function=_comfort_adapter(),
            rank=0,
            world_size=1,
        )


def test_runtime_provenance_and_proposal_order_fail_closed(tmp_path: Path) -> None:
    manifest, _, bindings, provenance, coverage, path, artifact_sha = _artifact(tmp_path)
    common = dict(
        manifest=manifest,
        artifact_path=path,
        artifact_bindings=bindings,
        artifact_coverage=coverage,
        expected_artifact_sha256=artifact_sha,
        feature_source=_feature_source(
            manifest.rows, live_provenance=provenance
        ),
        comfort_function=_comfort_adapter(),
        rank=0,
        world_size=1,
    )
    with pytest.raises(F2CausalShardRunnerError, match="byte-verified"):
        evaluate_causal_shard(
            **common,
            live_provenance=None,
            model_adapter=_model_adapter(tmp_path),
        )
    with pytest.raises(Exception, match="proposal order drifted"):
        evaluate_causal_shard(
            **common,
            live_provenance=provenance,
            model_adapter=_model_adapter(
                tmp_path,
                live_provenance=provenance,
                proposal_drift=True,
            ),
        )
    with pytest.raises(F2CausalShardRunnerError, match="authenticated F2Causal"):
        evaluate_causal_shard(
            **common,
            live_provenance=provenance,
            model_adapter=_ForgedModelAdapter(),
        )
    with pytest.raises(F2CausalShardRunnerError, match="sealed official F2 factory"):
        evaluate_causal_shard(
            **common,
            live_provenance=provenance,
            model_adapter=_model_adapter(tmp_path),
        )
    with pytest.raises(F2CausalShardRunnerError, match="official authenticated"):
        evaluate_causal_shard(
            **{
                **common,
                "live_provenance": provenance,
                "model_adapter": _model_adapter(
                    tmp_path, live_provenance=provenance
                ),
                "comfort_function": DriveJepaExtendedComfort(
                    lambda past, current, time_s: np.ones((1,), dtype=np.bool_)
                ),
            }
        )


def test_artifact_coverage_must_be_derived_from_manifest(tmp_path: Path) -> None:
    manifest, _, bindings, provenance, _, path, artifact_sha = _artifact(tmp_path)
    wrong_coverage = build_artifact_coverage(
        CoverageEntry(row.log_name, row.time_us, row.token)
        for row in manifest.rows[:-1]
    )
    with pytest.raises(F2CausalShardRunnerError, match="exact causal manifest"):
        evaluate_causal_shard(
            manifest=manifest,
            artifact_path=path,
            artifact_bindings=bindings,
            artifact_coverage=wrong_coverage,
            expected_artifact_sha256=artifact_sha,
            live_provenance=provenance,
            model_adapter=_model_adapter(tmp_path, live_provenance=provenance),
            feature_source=_feature_source(
                manifest.rows, live_provenance=provenance
            ),
            comfort_function=_comfort_adapter(),
            rank=0,
            world_size=1,
        )


def test_merge_rejects_missing_or_duplicate_rank(tmp_path: Path) -> None:
    manifest, _, bindings, provenance, coverage, path, artifact_sha = _artifact(tmp_path)
    assignments = deterministic_log_assignments(manifest.rows, 2)
    grouped = {
        log: tuple(row for row in manifest.rows if row.log_name == log)
        for log in manifest.logs
    }
    logs = assignments[0]
    assigned = tuple(row for log in logs for row in grouped[log])
    shard = evaluate_causal_shard(
        manifest=manifest,
        artifact_path=path,
        artifact_bindings=bindings,
        artifact_coverage=coverage,
        expected_artifact_sha256=artifact_sha,
        live_provenance=provenance,
        model_adapter=_model_adapter(tmp_path, live_provenance=provenance),
        feature_source=_feature_source(
            assigned, live_provenance=provenance, rank=0, world_size=2
        ),
        comfort_function=_comfort_adapter(),
        rank=0,
        world_size=2,
    )
    with pytest.raises(F2CausalShardRunnerError, match="rank coverage"):
        merge_causal_world(manifest, (shard,))
    with pytest.raises(F2CausalShardRunnerError, match="rank coverage"):
        merge_causal_world(manifest, (shard, shard))


def test_world_size_and_rank_are_frozen(tmp_path: Path) -> None:
    manifest, _, bindings, provenance, coverage, path, artifact_sha = _artifact(tmp_path)
    common = dict(
        manifest=manifest,
        artifact_path=path,
        artifact_bindings=bindings,
        artifact_coverage=coverage,
        expected_artifact_sha256=artifact_sha,
        live_provenance=provenance,
        model_adapter=_model_adapter(tmp_path, live_provenance=provenance),
        feature_source=_feature_source(
            manifest.rows, live_provenance=provenance
        ),
        comfort_function=_comfort_adapter(),
    )
    for rank, world_size in ((0, 4), (-1, 1), (1, 1), (True, 1)):
        with pytest.raises(F2CausalShardRunnerError):
            evaluate_causal_shard(
                **common, rank=rank, world_size=world_size  # type: ignore[arg-type]
            )
