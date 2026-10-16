"""CPU-only production-envelope tests for the prelaunch F2 capacity audit."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import pytest

import navsim.planning.validation.f2_feasibility_production as production_module

from navsim.planning.validation.f2_feasibility_production import (
    CAPACITY_GATE_THRESHOLD,
    FEASIBILITY_CODE_PATHS,
    OFFICIAL_DRIVE_JEPA_BASELINE_SHA256,
    PRODUCTION_CLAIM_SCOPE,
    F2FeasibilityProductionError,
    FeasibilityPretrainingConfig,
    build_frozen_static_inputs,
    capacity_gate_passes,
    cli_main,
    parse_pretraining_config_bytes,
    preflight_production_runtime,
    pretraining_config_bytes,
    production_certificate_payload,
    read_pretraining_config,
    read_production_feasibility_audit,
    replay_production_feasibility_audit,
    validate_production_certificate_payload,
    write_pretraining_config,
    write_production_feasibility_audit,
)
from navsim.planning.validation.f2_navsim_runtime_adapters import (
    OFFICIAL_DRIVE_JEPA_CHECKPOINT_NAME,
    OFFICIAL_DRIVE_JEPA_FACTORY_REVISION,
    OFFICIAL_DRIVE_JEPA_VJEPA_NAME,
    OFFICIAL_DRIVE_JEPA_VJEPA_SHA256,
)
from navsim.planning.validation.f2_production_runner import (
    canonical_json_bytes,
    sha256_sidecar_path,
)
from navsim.planning.validation.f2_residual_feasibility import (
    FEASIBILITY_SAMPLE_SIZE,
    F2ResidualFeasibilityAudit,
    TrainSampleRow,
    analyze_tanh_residual_reachability,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _config(tmp_path: Path) -> FeasibilityPretrainingConfig:
    experiment = tmp_path / "experiment"
    return FeasibilityPretrainingConfig(
        qualification_manifest=tmp_path / "qualification.json",
        qualification_config=tmp_path / "qualification.yaml",
        qualification_source_manifest=tmp_path / "qualification-source.json",
        qualification_source_root=tmp_path / "qualification-source",
        raw_input_manifest=tmp_path / "raw-input-content.json",
        metric_cache_metadata=tmp_path / "train_metric_cache_v2_pruned" / "metadata.csv",
        metric_cache_manifest=tmp_path / "metric-cache-content.json",
        metric_cache_root=tmp_path / "train_metric_cache_v2_pruned",
        baseline_checkpoint=tmp_path / OFFICIAL_DRIVE_JEPA_CHECKPOINT_NAME,
        vjepa_checkpoint=(
            experiment / "Drive-JEPA-cache" / OFFICIAL_DRIVE_JEPA_VJEPA_NAME
        ),
        repository_root=tmp_path / "repository",
        experiment_root=experiment,
        navsim_log=tmp_path / "navtrain-logs",
        sensor_blobs=tmp_path / "sensor-blobs",
        qualification_manifest_sha256=_digest("qualification"),
        qualification_config_sha256=_digest("qualification-config"),
        qualification_source_manifest_sha256=_digest("qualification-source"),
        raw_input_manifest_sha256=_digest("raw-input-content"),
        metric_cache_metadata_sha256=_digest("cache-metadata"),
        metric_cache_manifest_sha256=_digest("cache-content"),
        baseline_checkpoint_sha256=OFFICIAL_DRIVE_JEPA_BASELINE_SHA256,
        vjepa_checkpoint_sha256=OFFICIAL_DRIVE_JEPA_VJEPA_SHA256,
        production_code_revision="1" * 40,
    )


_SYNTHETIC_REACHABILITY_PAYLOAD = analyze_tanh_residual_reachability(
    (0.0,) * 32
).to_payload()


class _SyntheticTokenResult:
    def __init__(self, row: TrainSampleRow, reachable_delta_upper: float) -> None:
        self._row = row
        self._delta = struct.unpack(
            ">f", struct.pack(">f", reachable_delta_upper)
        )[0]

    def to_payload(self) -> Dict[str, Any]:
        truths = [0.0] * 32
        truths[1] = self._delta
        target_scores = [
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for _ in range(32)
        ]
        target_scores[1] = [
            1.0,
            1.0,
            1.0,
            1.0,
            self._delta,
            self._delta,
            self._delta,
            self._delta,
            self._delta,
        ]
        frozen_static_input = {
            "base_logits": [0.0] * 32,
            "target_scores": target_scores,
        }
        return {
            "base_selected": {
                "candidate_index": 0,
                "static_true_epdms": 0.0,
            },
            "future_camera_valid": self._row.future_camera_valid,
            "frozen_static_input": frozen_static_input,
            "frozen_static_input_sha256": hashlib.sha256(
                canonical_json_bytes(frozen_static_input)
            ).hexdigest(),
            "hard_factor_reachable_marginal_upper": {
                "nc": 1.0,
                "dac": 1.0,
                "tlc": 1.0,
                "ddc": 1.0,
            },
            "log_name": self._row.log_name,
            "reachability": _SYNTHETIC_REACHABILITY_PAYLOAD,
            "reachable_delta_upper": self._delta,
            "reachable_headroom_closed_upper": 1.0,
            "reachable_oracle": {
                "candidate_index": 1,
                "static_true_epdms": self._delta,
            },
            "safety_candidate_coverage": {
                "has_reachable_safe_candidate": True,
                "has_safe_candidate": True,
                "reachable_safe_candidate_count": 32,
                "safe_candidate_count": 32,
                "safe_candidate_reachability_fraction": 1.0,
                "safe_definition": "all_four_hard_channels_exactly_one",
            },
            "sampling_sha256": self._row.sampling_sha256,
            "selection_stage": self._row.selection_stage,
            "static_true_epdms": truths,
            "time_us": self._row.time_us,
            "token": self._row.token,
            "unbounded_headroom": self._delta,
            "unbounded_oracle": {
                "candidate_index": 1,
                "static_true_epdms": self._delta,
            },
        }


class _SyntheticAggregate:
    def __init__(self, reachable_delta_upper: float) -> None:
        self.reachable_delta_upper = reachable_delta_upper

    def to_payload(self) -> Dict[str, Any]:
        return {
            "reachable_delta_upper": self.reachable_delta_upper,
            "token_count": FEASIBILITY_SAMPLE_SIZE,
        }


class _SyntheticBootstrap:
    def to_payload(self) -> Dict[str, Any]:
        return {"analytical_only": True}


def _synthetic_audit(
    reachable_delta_upper: float = 0.25,
) -> F2ResidualFeasibilityAudit:
    reachable_delta_upper = struct.unpack(
        ">f", struct.pack(">f", reachable_delta_upper)
    )[0]
    rows = tuple(
        TrainSampleRow(
            log_name=f"train-log-{index % 8}",
            time_us=index,
            token=f"train-token-{index:04d}",
            future_camera_valid=True,
            sampling_sha256=_digest(f"sample-{index}"),
            selection_stage="global_sha_fill",
        )
        for index in range(FEASIBILITY_SAMPLE_SIZE)
    )
    return F2ResidualFeasibilityAudit(
        qualification_manifest_sha256=_digest("qualification"),
        sample_rows=rows,
        token_results=tuple(  # type: ignore[arg-type]
            _SyntheticTokenResult(row, reachable_delta_upper) for row in rows
        ),
        aggregate=_SyntheticAggregate(reachable_delta_upper),  # type: ignore[arg-type]
        bootstrap=_SyntheticBootstrap(),  # type: ignore[arg-type]
    )


def _provenance(audit: F2ResidualFeasibilityAudit) -> Dict[str, Any]:
    rows_payload = [row.to_payload() for row in audit.sample_rows]
    row_sha256 = hashlib.sha256(canonical_json_bytes(rows_payload)).hexdigest()
    return {
        "baseline": {
            "checkpoint_sha256": OFFICIAL_DRIVE_JEPA_BASELINE_SHA256,
            "factory_path_tree_sha256": _digest("factory-tree"),
            "factory_revision": OFFICIAL_DRIVE_JEPA_FACTORY_REVISION,
            "state_tensor_count": 1,
            "vjepa_checkpoint_sha256": OFFICIAL_DRIVE_JEPA_VJEPA_SHA256,
        },
        "code_source": {
            "path_tree_sha256": _digest("code-tree"),
            "paths": list(FEASIBILITY_CODE_PATHS),
            "revision": "1" * 40,
        },
        "deterministic_runtime": {
            "cublas_workspace_config": ":4096:8",
            "cuda_matmul_allow_tf32": False,
            "cuda_visible_devices": None,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_version": 8900,
            "deterministic_algorithms": True,
            "device": "cuda:0",
            "device_capability": [8, 9],
            "device_index": 0,
            "device_name": "Synthetic RTX 4090",
            "float32_matmul_precision": "highest",
            "local_rank": None,
            "logical_rank": 0,
            "python_version": "3.9.0",
            "torch_cuda_version": "12.1",
            "torch_version": "2.1.0",
            "world_size": 1,
        },
        "feature_sample": {
            "removed_scenes": 0,
            "row_identity_sha256": row_sha256,
            "token_count": FEASIBILITY_SAMPLE_SIZE,
        },
        "metric_cache": {
            "content_entry_count": FEASIBILITY_SAMPLE_SIZE + 1,
            "content_entries_sha256": _digest("cache-entries"),
            "content_manifest_sha256": _digest("cache-content"),
            "metadata_sha256": _digest("cache-metadata"),
        },
        "qualification": {
            "config_sha256": _digest("qualification-config"),
            "manifest_sha256": audit.qualification_manifest_sha256,
            "source_file_count": 1,
            "source_files_sha256": _digest("qualification-files"),
            "source_manifest_sha256": _digest("qualification-source"),
        },
        "raw_input": {
            "content_manifest_sha256": _digest("raw-input-content"),
            "log_entry_count": 8,
            "sensor_entry_count": FEASIBILITY_SAMPLE_SIZE,
        },
    }


def _certificate(
    value: float = 0.25,
    *,
    pretraining_config_sha256: str = _digest("pretraining-config"),
) -> Dict[str, Any]:
    audit = _synthetic_audit(value)
    return production_certificate_payload(
        audit,
        pretraining_config_sha256=pretraining_config_sha256,
        provenance=_provenance(audit),
    )


def test_pretraining_config_is_canonical_sidecar_bound_and_no_overwrite(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    output = tmp_path / "pretraining-config.json"

    digest = write_pretraining_config(output, config)
    loaded = read_pretraining_config(output)

    assert loaded.config == config
    assert loaded.sha256 == digest
    assert pretraining_config_bytes(config).endswith(b"\n")
    with pytest.raises(F2FeasibilityProductionError, match="overwrite"):
        write_pretraining_config(output, config)


def test_pretraining_config_rejects_noncanonical_duplicate_nan_and_extra_inputs(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    decoded = json.loads(pretraining_config_bytes(config))

    with pytest.raises(F2FeasibilityProductionError, match="duplicate"):
        parse_pretraining_config_bytes(b'{"dataset":"navtrain","dataset":"navtrain"}\n')
    with pytest.raises(F2FeasibilityProductionError, match="non-finite"):
        parse_pretraining_config_bytes(b'{"dataset":NaN}\n')

    decoded["paths"]["causal_manifest"] = (tmp_path / "causal.json").as_posix()
    with pytest.raises(F2FeasibilityProductionError, match="extra"):
        parse_pretraining_config_bytes(canonical_json_bytes(decoded))


def test_config_digests_revisions_and_live_paths_are_strict(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    with pytest.raises(F2FeasibilityProductionError, match="lowercase"):
        replace(
            config,
            qualification_manifest_sha256=config.qualification_manifest_sha256.upper(),
        )
    with pytest.raises(F2FeasibilityProductionError, match="lowercase"):
        replace(config, production_code_revision="A" * 40)

    real_directory = tmp_path / "real-directory"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(F2FeasibilityProductionError, match="symlink"):
        replace(config, repository_root=linked_directory)
    with pytest.raises(F2FeasibilityProductionError, match="symlink"):
        production_module._resolved_directory(
            linked_directory, "repository_root"
        )

    real_file = real_directory / "real.bin"
    real_file.write_bytes(b"authenticated\n")
    linked_file = real_directory / "linked.bin"
    linked_file.symlink_to(real_file)
    with pytest.raises(F2FeasibilityProductionError, match="symlink"):
        production_module._resolved_file(linked_file, "checkpoint")

    decoded = json.loads(pretraining_config_bytes(config))
    decoded["paths"]["split"] = "train"
    with pytest.raises(F2FeasibilityProductionError, match="extra"):
        parse_pretraining_config_bytes(canonical_json_bytes(decoded))


def test_pretraining_config_payload_tamper_fails_sidecar_authentication(
    tmp_path: Path,
) -> None:
    output = tmp_path / "pretraining-config.json"
    write_pretraining_config(output, _config(tmp_path))
    output.write_bytes(output.read_bytes() + b" ")

    with pytest.raises(F2FeasibilityProductionError, match="sidecar"):
        read_pretraining_config(output)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (CAPACITY_GATE_THRESHOLD, False),
        (0.0101, True),
        (0.5, True),
    ],
)
def test_capacity_gate_is_strictly_one_raw_point(
    value: float, expected: bool
) -> None:
    assert capacity_gate_passes(value) is expected
    certificate = _certificate(value)
    assert certificate["capacity_gate"]["passed"] is expected
    assert certificate["capacity_gate"]["threshold"] == 0.01
    assert certificate["capacity_gate"]["unit"] == "raw_score_point"
    assert certificate["claim_scope"] == PRODUCTION_CLAIM_SCOPE


def test_production_certificate_roundtrip_sidecar_and_no_overwrite(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pretraining-config.json"
    config_sha256 = write_pretraining_config(config_path, _config(tmp_path))
    certificate = _certificate(pretraining_config_sha256=config_sha256)
    output = tmp_path / "feasibility-audit.json"

    digest = write_production_feasibility_audit(output, certificate)
    loaded = read_production_feasibility_audit(
        output,
        expected_sha256=digest,
        expected_pretraining_config_path=config_path,
        expected_pretraining_config_sha256=config_sha256,
    )

    assert loaded.sha256 == digest
    assert loaded.payload == certificate
    assert sha256_sidecar_path(output).read_text(encoding="ascii") == digest + "\n"
    with pytest.raises(F2FeasibilityProductionError, match="overwrite"):
        write_production_feasibility_audit(output, certificate)


def test_certificate_tamper_partial_coverage_and_identity_drift_fail_closed() -> None:
    certificate = _certificate()

    gate_tamper = copy.deepcopy(certificate)
    gate_tamper["capacity_gate"]["passed"] = False
    with pytest.raises(F2FeasibilityProductionError, match="capacity gate"):
        validate_production_certificate_payload(gate_tamper)

    partial = copy.deepcopy(certificate)
    partial["audit"]["sample"]["rows"].pop()
    partial["audit_sha256"] = hashlib.sha256(
        canonical_json_bytes(partial["audit"])
    ).hexdigest()
    with pytest.raises(F2FeasibilityProductionError, match="exactly"):
        validate_production_certificate_payload(partial)

    identity = copy.deepcopy(certificate)
    identity["provenance"]["baseline"]["checkpoint_sha256"] = _digest("wrong")
    with pytest.raises(F2FeasibilityProductionError, match="official assets"):
        validate_production_certificate_payload(identity)


def test_certificate_synchronously_resigned_aggregate_and_gate_still_fail() -> None:
    certificate = _certificate(0.25)
    certificate["audit"]["aggregate"]["reachable_delta_upper"] = 0.0
    certificate["capacity_gate"]["value"] = 0.0
    certificate["capacity_gate"]["passed"] = False
    certificate["audit_sha256"] = hashlib.sha256(
        canonical_json_bytes(certificate["audit"])
    ).hexdigest()

    with pytest.raises(F2FeasibilityProductionError, match="token replay"):
        validate_production_certificate_payload(certificate)


def test_certificate_resigned_static_truth_cannot_differ_from_frozen_targets() -> None:
    certificate = _certificate(0.25)
    token = certificate["audit"]["tokens"][0]
    token["frozen_static_input"]["target_scores"][1] = [
        1.0,
        1.0,
        1.0,
        1.0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    ]
    token["frozen_static_input_sha256"] = hashlib.sha256(
        canonical_json_bytes(token["frozen_static_input"])
    ).hexdigest()
    certificate["audit_sha256"] = hashlib.sha256(
        canonical_json_bytes(certificate["audit"])
    ).hexdigest()

    with pytest.raises(F2FeasibilityProductionError, match="static truths differ"):
        validate_production_certificate_payload(certificate)


def test_reader_cross_binds_certificate_provenance_to_external_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pretraining-config.json"
    config_sha256 = write_pretraining_config(config_path, _config(tmp_path))
    certificate = _certificate(pretraining_config_sha256=config_sha256)
    certificate["provenance"]["raw_input"]["content_manifest_sha256"] = _digest(
        "attacker-chosen-raw-input"
    )
    output = tmp_path / "feasibility-audit.json"
    audit_sha256 = write_production_feasibility_audit(output, certificate)

    with pytest.raises(F2FeasibilityProductionError, match="raw-input content manifest"):
        read_production_feasibility_audit(
            output,
            expected_sha256=audit_sha256,
            expected_pretraining_config_path=config_path,
            expected_pretraining_config_sha256=config_sha256,
        )


def test_admission_replay_requires_byte_exact_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "pretraining-config.json"
    config_sha256 = write_pretraining_config(config_path, _config(tmp_path))
    certificate = _certificate(pretraining_config_sha256=config_sha256)
    audit_path = tmp_path / "feasibility-audit.json"
    audit_sha256 = write_production_feasibility_audit(audit_path, certificate)
    monkeypatch.setattr(
        production_module,
        "_compute_production_feasibility_certificate",
        lambda loaded: certificate,
    )

    replayed = replay_production_feasibility_audit(
        config_path=config_path,
        expected_config_sha256=config_sha256,
        audit_path=audit_path,
        expected_audit_sha256=audit_sha256,
    )
    assert replayed.payload == certificate
    assert replayed.sha256 == audit_sha256

    drifted = copy.deepcopy(certificate)
    drifted["provenance"]["deterministic_runtime"]["torch_version"] = "drifted"
    monkeypatch.setattr(
        production_module,
        "_compute_production_feasibility_certificate",
        lambda loaded: drifted,
    )
    with pytest.raises(F2FeasibilityProductionError, match="signed audit bytes"):
        replay_production_feasibility_audit(
            config_path=config_path,
            expected_config_sha256=config_sha256,
            audit_path=audit_path,
            expected_audit_sha256=audit_sha256,
        )


def test_certificate_payload_tamper_fails_sidecar_authentication(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pretraining-config.json"
    config_sha256 = write_pretraining_config(config_path, _config(tmp_path))
    output = tmp_path / "feasibility-audit.json"
    digest = write_production_feasibility_audit(
        output,
        _certificate(pretraining_config_sha256=config_sha256),
    )
    output.write_bytes(output.read_bytes() + b" ")

    with pytest.raises(F2FeasibilityProductionError, match="sidecar"):
        read_production_feasibility_audit(
            output,
            expected_sha256=digest,
            expected_pretraining_config_path=config_path,
            expected_pretraining_config_sha256=config_sha256,
        )


def test_static_input_builder_rejects_non_2048_coverage_before_runtime_access() -> None:
    row = TrainSampleRow(
        log_name="train-log",
        time_us=1,
        token="train-token",
        future_camera_valid=True,
        sampling_sha256=_digest("sample"),
        selection_stage="per_log_seed",
    )
    with pytest.raises(F2FeasibilityProductionError, match="exactly 2048"):
        build_frozen_static_inputs(
            (row,),
            feature_source=object(),
            references={},
            baseline=object(),  # type: ignore[arg-type]
            semantics=object(),  # type: ignore[arg-type]
        )


def test_runtime_preflight_requires_external_config_digest(tmp_path: Path) -> None:
    config_path = tmp_path / "pretraining-config.json"
    write_pretraining_config(config_path, _config(tmp_path))

    with pytest.raises(F2FeasibilityProductionError, match="external expected"):
        preflight_production_runtime(
            config_path=config_path,
            expected_config_sha256=_digest("wrong-config"),
        )


def test_runtime_preflight_replays_source_cache_and_model_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config.baseline_checkpoint.write_bytes(b"checkpoint")
    config_path = tmp_path / "pretraining-config.json"
    digest = write_pretraining_config(config_path, config)
    code_source = SimpleNamespace(
        path_tree_sha256=_digest("code-tree"),
        revision=config.production_code_revision,
    )
    factory_source = SimpleNamespace(
        path_tree_sha256=_digest("factory-tree"),
        revision=config.drive_jepa_factory_revision,
    )
    qualification = object()
    cache = SimpleNamespace(manifest=SimpleNamespace(entries=(1, 2, 3)))
    raw_inputs = SimpleNamespace(
        manifest=SimpleNamespace(logs=(1, 2), sensor_blobs=(1, 2, 3, 4))
    )
    baseline = SimpleNamespace(
        checkpoint_path=config.baseline_checkpoint.resolve(),
        checkpoint_sha256=config.baseline_checkpoint_sha256,
        factory_source=factory_source,
        state_tensor_count=7,
    )
    monkeypatch.setattr(
        production_module,
        "authenticate_feasibility_code_source",
        lambda *args, **kwargs: code_source,
    )
    monkeypatch.setattr(
        production_module,
        "authenticate_drive_jepa_factory_source",
        lambda *args, **kwargs: factory_source,
    )
    monkeypatch.setattr(
        production_module,
        "_load_authenticated_qualification",
        lambda *args, **kwargs: (qualification, ("source-a", "source-b")),
    )
    monkeypatch.setattr(
        production_module,
        "_load_authenticated_metric_cache_manifest",
        lambda *args, **kwargs: cache,
    )
    monkeypatch.setattr(
        production_module,
        "_load_authenticated_raw_inputs",
        lambda *args, **kwargs: raw_inputs,
    )
    monkeypatch.setattr(
        production_module,
        "_restore_official_baseline",
        lambda *args, **kwargs: baseline,
    )

    receipt = preflight_production_runtime(
        config_path=config_path,
        expected_config_sha256=digest,
    )

    assert receipt["config_sha256"] == digest
    assert receipt["metric_cache_entry_count"] == 3
    assert receipt["qualification_source_file_count"] == 2
    assert receipt["baseline_state_tensor_count"] == 7
    assert receipt["raw_input_log_count"] == 2
    assert receipt["raw_input_sensor_count"] == 4


def test_cli_has_no_data_view_selector(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli_main(
            [
                "build-audit",
                "--config",
                str(tmp_path / "config.json"),
                "--output",
                str(tmp_path / "audit.json"),
                "--split",
                "train",
            ]
        )
