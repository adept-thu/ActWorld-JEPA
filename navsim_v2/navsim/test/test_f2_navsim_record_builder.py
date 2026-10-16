"""Synthetic audits for the NAVSIM F2 frozen-record builder foundation."""

from __future__ import annotations

import hashlib
import json
import lzma
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
import torch

import navsim.planning.validation.f2_navsim_runtime_adapters as runtime_adapters_module
import navsim.planning.validation.f2_navsim_record_builder as record_builder_module

from navsim.planning.training.f2_qualification_manifest import (
    QualificationManifest,
    QualificationRow,
    build_qualification_manifest,
    build_qualification_partition,
    canonical_manifest_bytes as canonical_qualification_manifest_bytes,
    qualification_manifest_sha256,
)
from navsim.planning.validation.f2_causal_model_adapter import (
    ORIGINAL_PROPOSAL_HASH_SCHEMA,
    canonical_original_proposal_sha256,
)
from navsim.planning.validation.f2_causal_validator import (
    F2_MODEL_NUMERIC_MODE,
    CanonicalManifest,
    ManifestRow,
    build_canonical_manifest,
    canonical_manifest_bytes as canonical_causal_manifest_bytes,
    manifest_sha256,
)
from navsim.planning.validation.f2_frozen_artifact_io import (
    ArtifactBindings,
    CoverageEntry,
    FrozenArtifactReader,
    TARGET_CHANNEL_ORDER,
    build_artifact_coverage,
)
from navsim.planning.validation.f2_navsim_record_builder import (
    AuthenticatedHeldOutManifests,
    DriveJepaBaselineForward,
    DriveJepaNavsimRecordSemantics,
    F2FrozenRecordBuildError,
    F2_PRUNED_CACHE_FAMILY,
    FrozenNavsimRecordBuilder,
    NavsimTokenSimulation,
    PrunedMetricCacheItem,
    PrunedMetricCacheSource,
    TokenFeatures,
    cli_main,
)
from navsim.planning.validation.f2_live_provenance import (
    EXPECTED_SCORER_NAME,
    EXPECTED_SIMULATOR_NAME,
    QUALIFICATION_SOURCE_SCHEMA_VERSION,
    REQUIRED_RUNTIME_SOURCE_PATHS,
    build_runtime_source_manifest,
    canonical_runtime_source_manifest_bytes,
    verify_live_f2_provenance,
)
from navsim.planning.validation.f2_metric_cache_manifest import (
    build_metric_cache_manifest,
    load_and_verify_metric_cache_manifest,
    write_metric_cache_manifest,
)


def _sha(label: str) -> str:
    return hashlib.sha256((label + "\n").encode("utf-8")).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


QUALIFICATION_SOURCE_FILE_BYTES = b"qualification-source\n"
QUALIFICATION_SOURCE_PAYLOAD = {
    "files": [
        {
            "bytes": len(QUALIFICATION_SOURCE_FILE_BYTES),
            "path": "qualification_source.py",
            "sha256": hashlib.sha256(QUALIFICATION_SOURCE_FILE_BYTES).hexdigest(),
        }
    ],
    "foundation_commit": "synthetic-foundation",
    "schema_version": QUALIFICATION_SOURCE_SCHEMA_VERSION,
}
QUALIFICATION_SOURCE_BYTES = _canonical_json_bytes(QUALIFICATION_SOURCE_PAYLOAD)

CONFIG_SHA = _sha("resolved-f2-config")
RUNTIME_CONFIG_SHA = _sha("runtime-f2-config")
CACHE_TOKENS = ("train-a", "held-a", "held-b", "held-c")
CACHE_METADATA_BYTES = (
    "file_name\n"
    + "".join(f"cache/{token}/metric_cache.pkl\n" for token in CACHE_TOKENS)
).encode("utf-8")
CACHE_SHA = hashlib.sha256(CACHE_METADATA_BYTES).hexdigest()
SOURCE_SHA = hashlib.sha256(QUALIFICATION_SOURCE_BYTES).hexdigest()
BASELINE_SHA = _sha("drive-jepa-baseline-checkpoint")
VJEPA_BYTES = b"vjepa-checkpoint\n"
VJEPA_SHA = hashlib.sha256(VJEPA_BYTES).hexdigest()
TRAIN_EXCLUSION_SHA = _sha("train-qualification-exclusions")
HELD_EXCLUSION_SHA = _sha("held-qualification-exclusions")
SOURCE_REVISION = "synthetic-f2-source-revision"


def _held_rows() -> tuple[QualificationRow, ...]:
    return (
        QualificationRow("log-a", 100, "held-a", True),
        QualificationRow("log-a", 200, "held-b", True),
        QualificationRow("log-b", 100, "held-c", False),
    )


def _qualification(
    held_rows: tuple[QualificationRow, ...] | None = None,
) -> QualificationManifest:
    train = build_qualification_partition(
        "train",
        (QualificationRow("log-train", 10, "train-a", True),),
        raw_token_count=1,
        raw_log_count=1,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=TRAIN_EXCLUSION_SHA,
    )
    held = build_qualification_partition(
        "held_out",
        _held_rows() if held_rows is None else held_rows,
        raw_token_count=3,
        raw_log_count=2,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=HELD_EXCLUSION_SHA,
    )
    return build_qualification_manifest(
        train,
        held,
        config_sha256=CONFIG_SHA,
        cache_sha256=CACHE_SHA,
        source_sha256=SOURCE_SHA,
    )


def _causal() -> CanonicalManifest:
    rows = tuple(
        ManifestRow(
            row.log_name,
            row.time_us,
            row.token,
            row.future_camera_valid,
        )
        for row in _held_rows()
    )
    return build_canonical_manifest(
        rows,
        expected_tokens=(row.token for row in rows),
        sensor_available_tokens=(row.token for row in rows),
        cache_tokens=(row.token for row in rows),
        allowlist_logs=("log-a", "log-b"),
        cache_metadata_sha256=CACHE_SHA,
        source_revision=SOURCE_REVISION,
    )


def _authenticated() -> AuthenticatedHeldOutManifests:
    qualification = _qualification()
    causal = _causal()
    return AuthenticatedHeldOutManifests(
        qualification=qualification,
        qualification_sha256=qualification_manifest_sha256(qualification),
        causal=causal,
        causal_sha256=manifest_sha256(causal),
    )


def _bindings(manifests: AuthenticatedHeldOutManifests) -> ArtifactBindings:
    return ArtifactBindings(
        qualification_manifest_sha256=manifests.qualification_sha256,
        causal_manifest_sha256=manifests.causal_sha256,
        baseline_checkpoint_sha256=BASELINE_SHA,
        vjepa_checkpoint_sha256=VJEPA_SHA,
        cache_metadata_sha256=CACHE_SHA,
        metric_cache_manifest_sha256=_sha("metric-cache-manifest"),
        raw_input_manifest_sha256=_sha("raw-input-manifest"),
        qualification_config_sha256=CONFIG_SHA,
        runtime_config_sha256=RUNTIME_CONFIG_SHA,
        qualification_source_manifest_sha256=SOURCE_SHA,
        runtime_source_manifest_sha256=_sha("runtime-source-placeholder"),
        source_revision=SOURCE_REVISION,
        proposal_schema_name=ORIGINAL_PROPOSAL_HASH_SCHEMA,
        proposal_schema_sha256=_sha("proposal-schema-source"),
        simulator_name=EXPECTED_SIMULATOR_NAME,
        simulator_sha256=_sha("dual-simulator-source"),
        scorer_name=EXPECTED_SCORER_NAME,
        scorer_sha256=_sha("training-pdm-scorer-source"),
        channel_order=TARGET_CHANNEL_ORDER,
    )


def _live_bundle(
    tmp_path: Path, manifests: AuthenticatedHeldOutManifests
) -> tuple[ArtifactBindings, Any, SimpleNamespace]:
    root = tmp_path / "live-provenance"
    root.mkdir(exist_ok=True)
    qualification_path = root / "qualification.json"
    causal_path = root / "causal.json"
    qualification_config_path = root / "qualification-config.json"
    runtime_config_path = root / "runtime-config.yaml"
    cache_root = root / F2_PRUNED_CACHE_FAMILY
    cache_path = cache_root / "metadata" / "cache-metadata.csv"
    cache_manifest_index = 0
    cache_manifest_path = root / "cache-content.json"
    while cache_manifest_path.exists():
        cache_manifest_index += 1
        cache_manifest_path = root / f"cache-content-{cache_manifest_index}.json"
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

    qualification_path.write_bytes(
        canonical_qualification_manifest_bytes(manifests.qualification)
    )
    causal_path.write_bytes(canonical_causal_manifest_bytes(manifests.causal))
    qualification_config_path.write_bytes(b"resolved-f2-config\n")
    runtime_config_path.write_bytes(b"runtime-f2-config\n")
    cache_path.write_bytes(CACHE_METADATA_BYTES)
    baseline_path.write_bytes(b"drive-jepa-baseline-checkpoint\n")
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
    cache_manifest = build_metric_cache_manifest(
        cache_root=cache_root,
        metadata_path=cache_path,
        expected_metadata_sha256=CACHE_SHA,
        qualification=manifests.qualification,
        expected_qualification_manifest_sha256=manifests.qualification_sha256,
        workers=2,
    )
    cache_manifest_sha = write_metric_cache_manifest(
        cache_manifest_path.resolve(), cache_manifest
    )
    bindings = replace(
        _bindings(manifests),
        metric_cache_manifest_sha256=cache_manifest_sha,
        runtime_source_manifest_sha256=hashlib.sha256(runtime_source_bytes).hexdigest(),
        proposal_schema_sha256=runtime_source.component_sha256("proposal_schema"),
        simulator_sha256=runtime_source.component_sha256("dual_simulator"),
        scorer_sha256=runtime_source.component_sha256(
            "training_scorer_and_comfort"
        ),
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
    return bindings, verified, SimpleNamespace(
        qualification=qualification_path,
        causal=causal_path,
        qualification_config=qualification_config_path,
        runtime_config=runtime_config_path,
        cache=cache_path,
        cache_manifest=cache_manifest_path,
        cache_root=cache_root,
        baseline=baseline_path,
        vjepa=vjepa_path,
        qualification_source=qualification_source_path,
        qualification_source_root=qualification_source_root,
        runtime_source=runtime_source_path,
        runtime_source_root=runtime_source_root,
    )


class _MockDriveJepa(torch.nn.Module):
    def __init__(self, *, bad_shape: bool = False) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.125, dtype=torch.float32))
        self.bad_shape = bad_shape
        self.observed: list[dict[str, Any]] = []

    def forward(self, features: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        cpu_autocast = bool(torch.is_autocast_cpu_enabled())
        offset = features["offset"].reshape(1, 1, 1, 1)
        self.observed.append(
            {
                "autocast_cpu": cpu_autocast,
                "dtype": features["offset"].dtype,
                "grad": torch.is_grad_enabled(),
                "training": self.training,
            }
        )
        candidate = torch.arange(32, device=self.bias.device, dtype=torch.float32)
        time = torch.arange(8, device=self.bias.device, dtype=torch.float32)
        proposals = torch.zeros((1, 32, 8, 3), device=self.bias.device)
        proposals[..., 0] = candidate[None, :, None] / 100.0
        proposals[..., 1] = time[None, None, :] / 10.0
        proposals = proposals + offset
        if self.bad_shape:
            proposals = proposals[:, :, :-1]
        pred_logit = torch.zeros((1, 32, 3), device=self.bias.device)
        pred_logit[..., 0] = -10.0
        pred_logit[..., 1] = 99.0
        pred_logit[..., -1] = candidate[None] / 50.0 + self.bias
        return {"proposals": proposals, "pred_logit": pred_logit}


class _SyntheticRestoredBaseline:
    def __init__(self, forward: DriveJepaBaselineForward, live_provenance: Any) -> None:
        self.forward = forward
        self.model = forward.model
        self.checkpoint_sha256 = BASELINE_SHA
        self.live_provenance = live_provenance
        self.is_official_factory = True
        self.factory_source = runtime_adapters_module.DriveJepaFactorySourceIdentity(
            repository_root=Path("."),
            revision=runtime_adapters_module.OFFICIAL_DRIVE_JEPA_FACTORY_REVISION,
            path_tree_sha256=_sha("synthetic-factory-source"),
            paths=("synthetic",),
        )


@pytest.fixture(autouse=True)
def _install_synthetic_restored_baseline_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_adapters_module,
        "RestoredDriveJepaBaseline",
        _SyntheticRestoredBaseline,
    )
    monkeypatch.setattr(
        runtime_adapters_module,
        "QualifiedHeldOutNavtrainFeatureSource",
        _FeatureSource,
    )
    monkeypatch.setattr(
        record_builder_module, "PrunedMetricCacheSource", _CacheSource
    )
    monkeypatch.setattr(
        record_builder_module,
        "DriveJepaNavsimRecordSemantics",
        _SyntheticSemantics,
    )


class _FeatureSource:
    dataset = "navtrain"
    partition = "held_out"

    def __init__(
        self,
        rows: tuple[ManifestRow, ...],
        *,
        fail_token: str | None = None,
        identity_drift: bool = False,
    ) -> None:
        self.tokens = tuple(row.token for row in rows)
        self.fail_token = fail_token
        self.identity_drift = identity_drift
        self.live_provenance: Any = None
        self.calls: list[str] = []

    def load(self, row: ManifestRow) -> TokenFeatures:
        self.calls.append(row.token)
        if row.token == self.fail_token:
            raise OSError("synthetic sensor read failed")
        return TokenFeatures(
            token=row.token,
            log_name="wrong-log" if self.identity_drift else row.log_name,
            time_us=row.time_us,
            features={
                "offset": torch.tensor(
                    [self.tokens.index(row.token) / 1000.0], dtype=torch.float64
                )
            },
        )


class _CacheSource:
    cache_family = F2_PRUNED_CACHE_FAMILY

    def __init__(self, rows: tuple[ManifestRow, ...]) -> None:
        # The real pruned cache is a navtrain superset; extras are permitted.
        self.tokens = tuple(row.token for row in rows) + ("unrelated-navtrain-token",)
        self.calls: list[str] = []

    def load(self, row: ManifestRow) -> PrunedMetricCacheItem:
        self.calls.append(row.token)
        return PrunedMetricCacheItem(
            token=row.token,
            log_name=row.log_name,
            time_us=row.time_us,
            metric_cache=SimpleNamespace(token=row.token),
        )


class _SyntheticSemantics:
    def __init__(self) -> None:
        self.proposals: list[np.ndarray] = []

    def run(self, metric_cache: Any, proposals: np.ndarray) -> NavsimTokenSimulation:
        self.proposals.append(proposals.copy())
        candidate = np.arange(32, dtype=np.float32)
        targets = np.zeros((32, 9), dtype="<f4")
        for channel in range(9):
            targets[:, channel] = (candidate + channel) / 64.0
        calibration = np.zeros((32, 41, 11), dtype="<f8")
        carry = np.zeros((32, 41, 11), dtype="<f8")
        calibration[..., 0] = candidate[:, None]
        carry[..., 0] = candidate[:, None] + 100.0
        return NavsimTokenSimulation(targets, calibration, carry)


def _builder(
    tmp_path: Path,
    *,
    feature_source: _FeatureSource | None = None,
) -> tuple[
    FrozenNavsimRecordBuilder,
    AuthenticatedHeldOutManifests,
    ArtifactBindings,
    _SyntheticRestoredBaseline,
    _FeatureSource,
    _CacheSource,
    _SyntheticSemantics,
]:
    manifests = _authenticated()
    rows = manifests.causal.rows
    features = feature_source or _FeatureSource(rows)
    caches = _CacheSource(rows)
    baseline_forward = DriveJepaBaselineForward(
        model=_MockDriveJepa(),
        authenticated_baseline_sha256=BASELINE_SHA,
    )
    semantics = _SyntheticSemantics()
    bindings, live_provenance, _ = _live_bundle(tmp_path, manifests)
    features.live_provenance = live_provenance
    baseline = _SyntheticRestoredBaseline(baseline_forward, live_provenance)
    return (
        FrozenNavsimRecordBuilder(
            manifests=manifests,
            bindings=bindings,
            live_provenance=live_provenance,
            feature_source=features,
            metric_cache_source=caches,
            restored_baseline=baseline,
            navsim_semantics=semantics,
        ),
        manifests,
        bindings,
        baseline,
        features,
        caches,
        semantics,
    )


def test_baseline_forward_disables_autocast_and_uses_raw_last_logit() -> None:
    model = _MockDriveJepa()
    adapter = DriveJepaBaselineForward(
        model=model,
        authenticated_baseline_sha256=BASELINE_SHA,
    )
    original = {"offset": torch.tensor([0.25], dtype=torch.float64)}
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = adapter.infer("held-a", original)

    assert output.numeric_mode == F2_MODEL_NUMERIC_MODE
    assert output.proposals.shape == (32, 8, 3)
    assert output.base_logits.shape == (32,)
    assert output.base_logits[0].item() == pytest.approx(0.125)
    assert output.base_logits[-1].item() == pytest.approx(31 / 50 + 0.125)
    assert original["offset"].dtype == torch.float64
    assert model.observed == [
        {
            "autocast_cpu": False,
            "dtype": torch.float32,
            "grad": False,
            "training": False,
        }
    ]


def test_baseline_forward_rejects_noncanonical_proposal_shape() -> None:
    adapter = DriveJepaBaselineForward(
        model=_MockDriveJepa(bad_shape=True),
        authenticated_baseline_sha256=BASELINE_SHA,
    )
    with pytest.raises(F2FrozenRecordBuildError, match=r"\(1, 32, 8, 3\)"):
        adapter.infer("held-a", {"offset": torch.tensor([0.0])})


class _RecordingSimulator:
    def __init__(self, addition: float) -> None:
        self.addition = addition
        self.inputs: list[np.ndarray] = []

    def simulate_proposals(self, states: np.ndarray, initial_ego_state: Any) -> np.ndarray:
        self.inputs.append(states.copy())
        return np.ascontiguousarray(states, dtype=np.float64) + self.addition


class _RecordingScorer:
    def __init__(self) -> None:
        self.simulated_agents: list[Any] = []
        self._multi_metrics: np.ndarray | None = None
        self._weighted_metrics: np.ndarray | None = None

    def score_proposals(
        self,
        states: np.ndarray,
        observation: Any,
        centerline: Any,
        route_lane_ids: Any,
        drivable_area_map: Any,
        map_parameters: Any,
        *,
        simulated_agent_detections_tracks: Any,
        human_past_trajectory: Any,
    ) -> np.ndarray:
        self.simulated_agents.append(simulated_agent_detections_tracks)
        candidate = np.arange(33, dtype=np.float64) / 100.0
        self._multi_metrics = np.stack(
            [candidate + channel / 100.0 for channel in range(4)], axis=0
        )
        self._weighted_metrics = np.stack(
            [candidate + (channel + 4) / 100.0 for channel in range(4)], axis=0
        )
        return candidate + 0.08


def test_navsim_semantics_preserves_reference_scoring_and_dual_state_streams() -> None:
    calibration = _RecordingSimulator(10.0)
    carry = _RecordingSimulator(20.0)
    scorer = _RecordingScorer()
    reference_calls: list[Any] = []

    def reference_transform(trajectory: Any, sampling: Any, timepoint: Any) -> np.ndarray:
        reference_calls.append((trajectory, sampling, timepoint))
        states = np.zeros((41, 11), dtype=np.float64)
        states[:, 0] = 0.99
        return states

    def candidate_transform(trajectory: np.ndarray, sampling: Any, ego: Any) -> np.ndarray:
        states = np.zeros((41, 3), dtype=np.float64)
        states[:, 0] = trajectory[0, 0]
        return states

    semantics = DriveJepaNavsimRecordSemantics(
        proposal_sampling="sampling",
        trajectory_factory=lambda value: value.copy(),
        reference_transform=reference_transform,
        candidate_transform=candidate_transform,
        calibration_simulator=calibration,
        carry_simulator=carry,
        scorer=scorer,
        multi_metric_indices=(0, 1, 2, 3),
        weighted_metric_indices=(0, 1, 2, 3),
    )
    metric_cache = SimpleNamespace(
        ego_state=SimpleNamespace(time_point="timepoint"),
        trajectory="cached-pdm-trajectory",
        observation="pruned-observation",
        centerline="centerline",
        route_lane_ids=("lane",),
        drivable_area_map="drivable",
        map_parameters="map-parameters",
        past_human_trajectory="past-human",
    )
    proposals = np.zeros((32, 8, 3), dtype="<f4")
    proposals[:, :, 0] = np.arange(32, dtype=np.float32)[:, None] / 100.0

    result = semantics.run(metric_cache, proposals)

    assert reference_calls == [
        ("cached-pdm-trajectory", "sampling", "timepoint")
    ]
    assert calibration.inputs[0].shape == (33, 41, 11)
    assert calibration.inputs[0][0, 0, 0] == pytest.approx(0.99)
    assert calibration.inputs[0][1, 0, 0] == pytest.approx(0.0)
    assert calibration.inputs[0][32, 0, 0] == pytest.approx(0.31)
    assert carry.inputs[0].shape == (32, 41, 11)
    assert carry.inputs[0][0, 0, 0] == pytest.approx(0.0)
    assert carry.inputs[0][31, 0, 0] == pytest.approx(0.31)
    assert scorer.simulated_agents == [None]
    assert result.targets.shape == (32, 9)
    # Candidate zero is scorer column 1 because column 0 is the PDM reference.
    assert result.targets[0].tolist() == pytest.approx(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    )
    assert result.calibration_states[0, 0, 0] == pytest.approx(10.0)
    assert result.carry_states[0, 0, 0] == pytest.approx(20.0)


def test_official_semantics_wires_training_scorer_and_distinct_simulators() -> None:
    semantics = DriveJepaNavsimRecordSemantics.official()

    assert type(semantics._calibration_simulator).__module__.endswith(
        "simulation_v2.pdm_simulator"
    )
    assert type(semantics._carry_simulator).__module__.endswith(
        "simulation.pdm_simulator"
    )
    assert type(semantics._scorer).__module__.endswith(
        "score_module.train_pdm_scorer"
    )
    assert semantics._multi_indices == (0, 1, 2, 3)
    assert semantics._weighted_indices == (0, 1, 2, 3)


def test_pruned_cache_source_enforces_family_root_and_loaded_row_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / F2_PRUNED_CACHE_FAMILY
    qualification = _authenticated().qualification
    cache_paths = {
        token: root / "cache" / token / "metric_cache.pkl"
        for token in CACHE_TOKENS
    }
    for token, path in cache_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        value = SimpleNamespace(
            log_name="log-a" if token == "held-a" else "unused-log",
            timepoint=SimpleNamespace(time_us=100 if token == "held-a" else 0),
        )
        with lzma.open(path, "wb") as stream:
            pickle.dump(value, stream)
    cache_path = cache_paths["held-a"]
    metadata_path = root / "metadata" / "cache.csv"
    metadata_path.parent.mkdir()
    metadata_path.write_text(
        "path\n" + "".join(path.as_posix() + "\n" for path in cache_paths.values()),
        encoding="utf-8",
    )
    metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    qualification = build_qualification_manifest(
        qualification.train,
        qualification.held_out,
        config_sha256=qualification.config_sha256,
        cache_sha256=metadata_sha,
        source_sha256=qualification.source_sha256,
    )
    qualification_sha = qualification_manifest_sha256(qualification)
    manifest = build_metric_cache_manifest(
        cache_root=root,
        metadata_path=metadata_path,
        expected_metadata_sha256=metadata_sha,
        qualification=qualification,
        expected_qualification_manifest_sha256=qualification_sha,
        workers=2,
    )
    manifest_path = (tmp_path / "cache-content.json").resolve()
    manifest_sha = write_metric_cache_manifest(manifest_path, manifest)
    content_manifest = load_and_verify_metric_cache_manifest(
        manifest_path,
        expected_sha256=manifest_sha,
        cache_root=root,
        metadata_path=metadata_path,
        expected_metadata_sha256=metadata_sha,
        qualification=qualification,
        expected_qualification_manifest_sha256=qualification_sha,
    )
    bindings = replace(
        _bindings(_authenticated()),
        cache_metadata_sha256=metadata_sha,
        metric_cache_manifest_sha256=manifest_sha,
    )

    source = PrunedMetricCacheSource(
        root,
        metadata_path=metadata_path,
        expected_metadata_sha256=metadata_sha,
        bindings=bindings,
        content_manifest=content_manifest,
    )
    item = source.load(ManifestRow("log-a", 100, "held-a", True))

    assert source.tokens == tuple(sorted(CACHE_TOKENS))
    assert item.path == cache_path
    assert item.metric_cache.log_name == "log-a"
    with pytest.raises(F2FrozenRecordBuildError, match="root must be"):
        PrunedMetricCacheSource(
            tmp_path / "navtest_two_stage_metric_cache",
            metadata_path=metadata_path,
            expected_metadata_sha256=metadata_sha,
            bindings=bindings,
            content_manifest=content_manifest,
        )


def test_builder_seals_exact_canonical_coverage_and_candidate_order(tmp_path: Path) -> None:
    builder, manifests, bindings, baseline, features, caches, semantics = _builder(
        tmp_path
    )
    destination = tmp_path / "f2-frozen.bin"

    descriptor = builder.build(destination)

    assert descriptor.record_count == 3
    assert destination.is_file()
    assert not Path(str(destination) + ".partial").exists()
    assert features.calls == ["held-a", "held-b", "held-c"]
    assert caches.calls == features.calls
    assert len(semantics.proposals) == 3
    assert semantics.proposals[0][:, 0, 0].tolist() == pytest.approx(
        (np.arange(32) / 100.0).tolist()
    )
    coverage = build_artifact_coverage(
        CoverageEntry(row.log_name, row.time_us, row.token)
        for row in manifests.causal.rows
    )
    with FrozenArtifactReader.open(
        destination,
        expected_bindings=bindings,
        expected_coverage=coverage,
        mode="stream",
    ) as reader:
        records = list(reader.iter_records(batch_records=1))
    assert [record.token for record in records] == ["held-a", "held-b", "held-c"]
    expected_proposals = baseline.model(  # type: ignore[attr-defined]
        {"offset": torch.tensor([0.0], dtype=torch.float32)}
    )["proposals"][0]
    assert records[0].proposal_sha256 == canonical_original_proposal_sha256(
        expected_proposals
    )
    assert records[0].base[0] == pytest.approx(0.125)
    assert records[0].base[-1] == pytest.approx(31 / 50 + 0.125)
    assert records[0].targets[31, 8] == pytest.approx((31 + 8) / 64.0)
    assert records[0].calibration[31, 0, 0] == pytest.approx(31.0)
    assert records[0].carry[31, 0, 0] == pytest.approx(131.0)


def test_any_token_failure_is_fatal_and_never_publishes_target(tmp_path: Path) -> None:
    manifests = _authenticated()
    features = _FeatureSource(manifests.causal.rows, fail_token="held-b")
    builder, _, _, _, _, caches, _ = _builder(
        tmp_path, feature_source=features
    )
    destination = tmp_path / "failed.bin"

    with pytest.raises(
        F2FrozenRecordBuildError,
        match=r"log='log-a'.*time_us=200.*token='held-b'.*sensor read failed",
    ):
        builder.build(destination)

    assert not destination.exists()
    assert Path(str(destination) + ".partial").is_file()
    assert features.calls == ["held-a", "held-b"]
    assert caches.calls == ["held-a"]


def test_preflight_and_row_identity_fail_closed_before_or_at_exact_token(
    tmp_path: Path,
) -> None:
    builder, manifests, _, _, features, _, _ = _builder(tmp_path)
    features.tokens = features.tokens[:-1]
    destination = tmp_path / "missing.bin"
    with pytest.raises(F2FrozenRecordBuildError, match="exactly equal"):
        builder.build(destination)
    assert not destination.exists()
    assert not Path(str(destination) + ".partial").exists()

    drifted = _FeatureSource(manifests.causal.rows, identity_drift=True)
    builder, _, _, _, _, _, _ = _builder(tmp_path, feature_source=drifted)
    destination = tmp_path / "drifted.bin"
    with pytest.raises(F2FrozenRecordBuildError, match="feature row identity drift"):
        builder.build(destination)
    assert not destination.exists()
    assert Path(str(destination) + ".partial").is_file()


def test_manifest_views_and_artifact_bindings_must_match_exactly(
    tmp_path: Path,
) -> None:
    qualification = _qualification(
        (
            QualificationRow("log-a", 100, "held-a", False),
            QualificationRow("log-a", 200, "held-b", True),
            QualificationRow("log-b", 100, "held-c", False),
        )
    )
    causal = _causal()
    with pytest.raises(F2FrozenRecordBuildError, match="manifest rows differ"):
        AuthenticatedHeldOutManifests(
            qualification=qualification,
            qualification_sha256=qualification_manifest_sha256(qualification),
            causal=causal,
            causal_sha256=manifest_sha256(causal),
        )

    manifests = _authenticated()
    with pytest.raises(F2FrozenRecordBuildError, match="causal manifest"):
        manifests.require_bindings(
            replace(
                _bindings(manifests),
                causal_manifest_sha256=_sha("different-causal-manifest"),
            )
        )
    with pytest.raises(F2FrozenRecordBuildError, match="qualification manifest"):
        manifests.require_bindings(
            replace(
                _bindings(manifests),
                qualification_manifest_sha256=_sha(
                    "different-qualification-manifest"
                ),
            )
        )

    bad_bindings = replace(
        _bindings(manifests),
        baseline_checkpoint_sha256=_sha("different-baseline"),
    )
    manifests.require_bindings(bad_bindings)
    _, verified, _ = _live_bundle(tmp_path, manifests)
    with pytest.raises(F2FrozenRecordBuildError, match="runtime bindings"):
        FrozenNavsimRecordBuilder(
            manifests=manifests,
            bindings=bad_bindings,
            live_provenance=verified,
            feature_source=_FeatureSource(manifests.causal.rows),
            metric_cache_source=_CacheSource(manifests.causal.rows),
            restored_baseline=_SyntheticRestoredBaseline(
                DriveJepaBaselineForward(
                    model=_MockDriveJepa(),
                    authenticated_baseline_sha256=BASELINE_SHA,
                ),
                verified,
            ),
            navsim_semantics=_SyntheticSemantics(),
        )


def _bindings_payload(bindings: ArtifactBindings) -> Mapping[str, Any]:
    return {
        "baseline_checkpoint_sha256": bindings.baseline_checkpoint_sha256,
        "vjepa_checkpoint_sha256": bindings.vjepa_checkpoint_sha256,
        "cache_metadata_sha256": bindings.cache_metadata_sha256,
        "metric_cache_manifest_sha256": bindings.metric_cache_manifest_sha256,
        "raw_input_manifest_sha256": bindings.raw_input_manifest_sha256,
        "channel_order": list(bindings.channel_order),
        "causal_manifest_sha256": bindings.causal_manifest_sha256,
        "proposal_schema_name": bindings.proposal_schema_name,
        "proposal_schema_sha256": bindings.proposal_schema_sha256,
        "scorer_name": bindings.scorer_name,
        "scorer_sha256": bindings.scorer_sha256,
        "simulator_name": bindings.simulator_name,
        "simulator_sha256": bindings.simulator_sha256,
        "qualification_config_sha256": bindings.qualification_config_sha256,
        "qualification_manifest_sha256": bindings.qualification_manifest_sha256,
        "qualification_source_manifest_sha256": (
            bindings.qualification_source_manifest_sha256
        ),
        "runtime_config_sha256": bindings.runtime_config_sha256,
        "runtime_source_manifest_sha256": bindings.runtime_source_manifest_sha256,
        "source_revision": bindings.source_revision,
    }


def _write_canonical_json(path: Path, payload: Any) -> str:
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
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def test_flat_bindings_payload_rejects_uppercase_digest() -> None:
    payload = dict(_bindings_payload(_bindings(_authenticated())))
    payload["causal_manifest_sha256"] = payload[
        "causal_manifest_sha256"
    ].upper()
    with pytest.raises(F2FrozenRecordBuildError, match="lowercase hexadecimal"):
        record_builder_module._bindings_from_payload(payload)


def test_cli_dry_run_authenticates_both_manifests_without_writing_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifests = _authenticated()
    bindings, _, live_paths = _live_bundle(tmp_path, manifests)
    bindings_path = tmp_path / "bindings.json"
    feature_tokens_path = tmp_path / "feature-tokens.txt"
    cache_tokens_path = tmp_path / "cache-tokens.txt"
    bindings_sha = _write_canonical_json(bindings_path, _bindings_payload(bindings))
    tokens = [row.token for row in manifests.causal.rows]
    feature_tokens_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    cache_tokens_path.write_text(
        "\n".join(tokens + ["unrelated-navtrain-token"]) + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "dry-run-only.bin"

    exit_code = cli_main(
        [
            "--qualification-manifest",
            str(live_paths.qualification),
            "--causal-manifest",
            str(live_paths.causal),
            "--qualification-config",
            str(live_paths.qualification_config),
            "--runtime-config",
            str(live_paths.runtime_config),
            "--cache-metadata",
            str(live_paths.cache),
            "--metric-cache-manifest",
            str(live_paths.cache_manifest),
            "--metric-cache-root",
            str(live_paths.cache_root),
            "--baseline-checkpoint",
            str(live_paths.baseline),
            "--vjepa-checkpoint",
            str(live_paths.vjepa),
            "--qualification-source-manifest",
            str(live_paths.qualification_source),
            "--qualification-source-root",
            str(live_paths.qualification_source_root),
            "--runtime-source-manifest",
            str(live_paths.runtime_source),
            "--runtime-source-root",
            str(live_paths.runtime_source_root),
            "--bindings-json",
            str(bindings_path),
            "--bindings-sha256",
            bindings_sha,
            "--feature-tokens",
            str(feature_tokens_path),
            "--cache-tokens",
            str(cache_tokens_path),
            "--destination",
            str(destination),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["artifact_written"] is False
    assert payload["token_count"] == 3
    assert payload["proposal_shape"] == [32, 8, 3]
    assert "remain required" in payload["outer_adapter_blocker"]
    assert not destination.exists()
