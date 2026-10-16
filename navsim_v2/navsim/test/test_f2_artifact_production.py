"""CPU-only synthetic tests for whole-log frozen-artifact production."""

from __future__ import annotations

import hashlib
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from navsim.planning.validation.f2_artifact_production import (
    F2ArtifactProductionError,
    build_rank_artifact_plan,
    merge_rank_artifacts,
    read_merged_artifact_certificate,
    read_rank_artifact_certificate,
    seal_rank_artifact,
)
from navsim.planning.validation.f2_causal_validator import (
    ManifestRow,
    build_canonical_manifest,
    manifest_sha256,
)
from navsim.planning.validation.f2_frozen_artifact_io import (
    ArtifactBindings,
    CoverageEntry,
    FrozenArtifactReader,
    FrozenTokenRecord,
    TARGET_CHANNEL_ORDER,
    build_artifact_coverage,
)
from navsim.planning.validation.f2_production_runner import (
    canonical_json_bytes,
    read_artifact_bindings,
    read_bytes_with_sidecar,
    sha256_sidecar_path,
    write_artifact_bindings,
)


def _sha(label: str) -> str:
    return hashlib.sha256((label + "\n").encode("utf-8")).hexdigest()


def _manifest():
    rows = tuple(
        ManifestRow(f"log-{index}", 100_000, f"token-{index}", True)
        for index in range(8)
    )
    tokens = tuple(row.token for row in rows)
    return build_canonical_manifest(
        rows,
        expected_tokens=tokens,
        sensor_available_tokens=tokens,
        cache_tokens=tokens,
        allowlist_logs=tuple(row.log_name for row in rows),
        cache_metadata_sha256=_sha("cache-metadata"),
        source_revision="synthetic-source",
    )


def _bindings(manifest: Any, *, suffix: str = "") -> ArtifactBindings:
    values: Dict[str, Any] = {
        "qualification_manifest_sha256": _sha("qualification" + suffix),
        "causal_manifest_sha256": manifest_sha256(manifest),
        "baseline_checkpoint_sha256": _sha("baseline" + suffix),
        "vjepa_checkpoint_sha256": _sha("vjepa" + suffix),
        "cache_metadata_sha256": manifest.cache_metadata_sha256,
        "qualification_config_sha256": _sha("qualification-config" + suffix),
        "runtime_config_sha256": _sha("runtime-config" + suffix),
        "qualification_source_manifest_sha256": _sha(
            "qualification-source" + suffix
        ),
        "runtime_source_manifest_sha256": _sha("runtime-source" + suffix),
        "source_revision": manifest.source_revision,
        "proposal_schema_name": "synthetic-proposals",
        "proposal_schema_sha256": _sha("proposal-schema" + suffix),
        "simulator_name": "synthetic-simulator",
        "simulator_sha256": _sha("simulator" + suffix),
        "scorer_name": "synthetic-scorer",
        "scorer_sha256": _sha("scorer" + suffix),
        "channel_order": TARGET_CHANNEL_ORDER,
    }
    # Keep the synthetic constructor compatible with the separately developed
    # per-file cache-inventory binding without assuming its runtime API.
    actual_fields = {item.name for item in fields(ArtifactBindings)}
    if "metric_cache_manifest_sha256" in actual_fields:
        values["metric_cache_manifest_sha256"] = _sha(
            "metric-cache-manifest" + suffix
        )
    if "raw_input_manifest_sha256" in actual_fields:
        values["raw_input_manifest_sha256"] = _sha(
            "raw-input-manifest" + suffix
        )
    return ArtifactBindings(**values)


def _record(row: ManifestRow) -> FrozenTokenRecord:
    return FrozenTokenRecord(
        token=row.token,
        time_us=row.time_us,
        proposal_sha256=_sha("proposal-" + row.token),
        base=np.arange(32, dtype=np.dtype("<f4")),
        targets=np.full((32, 9), 0.5, dtype=np.dtype("<f4")),
        calibration=np.full((32, 41, 11), 1.0, dtype=np.dtype("<f8")),
        carry=np.full((32, 41, 11), 2.0, dtype=np.dtype("<f8")),
    )


def _rank_paths(directory: Path, world_size: int, rank: int) -> Tuple[Path, Path]:
    return (
        directory / f"world{world_size}.rank{rank}.bin",
        directory / f"world{world_size}.rank{rank}.json",
    )


def _build_world(
    directory: Path, manifest: Any, bindings: ArtifactBindings, world_size: int
) -> Tuple[Tuple[Path, ...], Tuple[Path, ...]]:
    artifacts = []
    certificates = []
    for rank in range(world_size):
        artifact, certificate = _rank_paths(directory, world_size, rank)
        seal_rank_artifact(
            causal_manifest=manifest,
            qualification_held_out_rows=manifest.rows,
            bindings=bindings,
            rank=rank,
            world_size=world_size,
            artifact_path=artifact,
            certificate_path=certificate,
            record_factory=_record,
        )
        artifacts.append(artifact)
        certificates.append(certificate)
    return tuple(artifacts), tuple(certificates)


def _rewrite(path: Path, payload: Any) -> None:
    encoded = canonical_json_bytes(payload)
    path.write_bytes(encoded)
    sha256_sidecar_path(path).write_bytes(
        (hashlib.sha256(encoded).hexdigest() + "\n").encode("ascii")
    )


@pytest.mark.parametrize("world_size", (1, 2, 8))
def test_rank_plans_are_exact_whole_log_partitions(world_size: int) -> None:
    manifest = _manifest()
    observed_logs = []
    observed_tokens = []
    for rank in range(world_size):
        plan = build_rank_artifact_plan(
            causal_manifest=manifest,
            qualification_held_out_rows=manifest.rows,
            rank=rank,
            world_size=world_size,
        )
        observed_logs.extend(plan.assigned_logs)
        observed_tokens.extend(row.token for row in plan.rows)
        assert set(row.log_name for row in plan.rows) == set(plan.assigned_logs)
    assert len(observed_logs) == len(set(observed_logs)) == 8
    assert set(observed_logs) == set(manifest.logs)
    assert set(observed_tokens) == set(manifest.tokens)


def test_rank_seal_roundtrip_tamper_and_no_overwrite(tmp_path: Path) -> None:
    manifest = _manifest()
    bindings = _bindings(manifest)
    artifact, certificate = _rank_paths(tmp_path, 2, 0)
    sealed = seal_rank_artifact(
        causal_manifest=manifest,
        qualification_held_out_rows=manifest.rows,
        bindings=bindings,
        rank=0,
        world_size=2,
        artifact_path=artifact,
        certificate_path=certificate,
        record_factory=_record,
    )
    replayed = read_rank_artifact_certificate(
        certificate,
        bindings=bindings,
        causal_manifest=manifest,
        qualification_held_out_rows=manifest.rows,
    )
    assert replayed.descriptor.root_sha256 == sealed.descriptor.root_sha256
    assert replayed.assigned_logs == sealed.assigned_logs
    with pytest.raises(F2ArtifactProductionError, match="overwrite"):
        seal_rank_artifact(
            causal_manifest=manifest,
            qualification_held_out_rows=manifest.rows,
            bindings=bindings,
            rank=0,
            world_size=2,
            artifact_path=artifact,
            certificate_path=certificate,
            record_factory=_record,
        )

    payload, _ = read_bytes_with_sidecar(certificate)
    import json

    decoded = json.loads(payload)
    decoded["fatal_count"] = 1
    _rewrite(certificate, decoded)
    with pytest.raises(F2ArtifactProductionError, match="fatal"):
        read_rank_artifact_certificate(
            certificate,
            bindings=bindings,
            causal_manifest=manifest,
            qualification_held_out_rows=manifest.rows,
        )


def test_binding_file_roundtrip_includes_cache_inventory(tmp_path: Path) -> None:
    bindings = _bindings(_manifest())
    path = tmp_path / "bindings.json"
    write_artifact_bindings(path, bindings)
    assert read_artifact_bindings(path) == bindings
    payload, _ = read_bytes_with_sidecar(path)
    assert b'"metric_cache_manifest_sha256"' in payload


def test_rank_failure_leaves_only_diagnostic_partial(tmp_path: Path) -> None:
    manifest = _manifest()
    bindings = _bindings(manifest)
    artifact, certificate = _rank_paths(tmp_path, 1, 0)

    def failed_factory(row: ManifestRow) -> FrozenTokenRecord:
        if row.token == "token-3":
            raise RuntimeError("synthetic failure")
        return _record(row)

    with pytest.raises(F2ArtifactProductionError, match="sealing failed"):
        seal_rank_artifact(
            causal_manifest=manifest,
            qualification_held_out_rows=manifest.rows,
            bindings=bindings,
            rank=0,
            world_size=1,
            artifact_path=artifact,
            certificate_path=certificate,
            record_factory=failed_factory,
        )
    assert not artifact.exists()
    assert artifact.with_name(artifact.name + ".partial").is_file()
    assert not certificate.exists()


@pytest.mark.parametrize("world_size", (1, 2, 8))
def test_stream_merge_and_replay_for_all_worlds(
    tmp_path: Path, world_size: int
) -> None:
    directory = tmp_path / f"world-{world_size}"
    directory.mkdir()
    manifest = _manifest()
    bindings = _bindings(manifest)
    _, certificates = _build_world(directory, manifest, bindings, world_size)
    merged_artifact = directory / "full.bin"
    merged_certificate = directory / "full.json"
    merged = merge_rank_artifacts(
        causal_manifest=manifest,
        qualification_held_out_rows=manifest.rows,
        bindings=bindings,
        world_size=world_size,
        rank_certificate_paths=certificates,
        artifact_path=merged_artifact,
        certificate_path=merged_certificate,
    )
    assert merged.descriptor.record_count == len(manifest.rows)
    replayed = read_merged_artifact_certificate(
        merged_certificate,
        bindings=bindings,
        causal_manifest=manifest,
        qualification_held_out_rows=manifest.rows,
    )
    assert replayed.descriptor.root_sha256 == merged.descriptor.root_sha256
    coverage = build_artifact_coverage(
        CoverageEntry(row.log_name, row.time_us, row.token) for row in manifest.rows
    )
    with FrozenArtifactReader.open(
        merged_artifact,
        expected_bindings=bindings,
        expected_coverage=coverage,
        mode="stream",
    ) as reader:
        tokens = []
        carry_values = []
        for record in reader.iter_records(batch_records=2):
            tokens.append(record.token)
            carry_values.append(float(record.carry[0, 0, 0]))
        del record
        assert tuple(tokens) == manifest.tokens
        assert all(value == 2.0 for value in carry_values)
    if world_size == 1:
        with pytest.raises(F2ArtifactProductionError, match="overwrite"):
            merge_rank_artifacts(
                causal_manifest=manifest,
                qualification_held_out_rows=manifest.rows,
                bindings=bindings,
                world_size=world_size,
                rank_certificate_paths=certificates,
                artifact_path=merged_artifact,
                certificate_path=merged_certificate,
            )


def test_merge_rejects_partial_duplicate_and_binding_drift(tmp_path: Path) -> None:
    manifest = _manifest()
    bindings = _bindings(manifest)
    _, certificates = _build_world(tmp_path, manifest, bindings, 2)
    with pytest.raises(F2ArtifactProductionError, match="one certificate per rank"):
        merge_rank_artifacts(
            causal_manifest=manifest,
            qualification_held_out_rows=manifest.rows,
            bindings=bindings,
            world_size=2,
            rank_certificate_paths=certificates[:1],
            artifact_path=tmp_path / "partial.bin",
            certificate_path=tmp_path / "partial.json",
        )
    with pytest.raises(F2ArtifactProductionError, match="duplicated"):
        merge_rank_artifacts(
            causal_manifest=manifest,
            qualification_held_out_rows=manifest.rows,
            bindings=bindings,
            world_size=2,
            rank_certificate_paths=(certificates[0], certificates[0]),
            artifact_path=tmp_path / "duplicate.bin",
            certificate_path=tmp_path / "duplicate.json",
        )
    with pytest.raises(F2ArtifactProductionError, match="binding"):
        read_rank_artifact_certificate(
            certificates[0],
            bindings=_bindings(manifest, suffix="-drift"),
            causal_manifest=manifest,
            qualification_held_out_rows=manifest.rows,
        )


def test_merged_certificate_rejects_uppercase_shard_digest(tmp_path: Path) -> None:
    manifest = _manifest()
    bindings = _bindings(manifest)
    _, certificates = _build_world(tmp_path, manifest, bindings, 1)
    merged_certificate = tmp_path / "merged.json"
    merge_rank_artifacts(
        causal_manifest=manifest,
        qualification_held_out_rows=manifest.rows,
        bindings=bindings,
        world_size=1,
        rank_certificate_paths=certificates,
        artifact_path=tmp_path / "merged.bin",
        certificate_path=merged_certificate,
    )
    payload, _ = read_bytes_with_sidecar(merged_certificate)
    import json

    decoded = json.loads(payload)
    decoded["shards"][0]["file_sha256"] = decoded["shards"][0][
        "file_sha256"
    ].upper()
    _rewrite(merged_certificate, decoded)
    with pytest.raises(F2ArtifactProductionError, match="lowercase hexadecimal"):
        read_merged_artifact_certificate(
            merged_certificate,
            bindings=bindings,
            causal_manifest=manifest,
            qualification_held_out_rows=manifest.rows,
        )


def test_certificate_assignment_tamper_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest()
    bindings = _bindings(manifest)
    _, certificates = _build_world(tmp_path, manifest, bindings, 2)
    payload, _ = read_bytes_with_sidecar(certificates[0])
    import json

    decoded = json.loads(payload)
    decoded["rows"][0]["token"] = "forged-token"
    _rewrite(certificates[0], decoded)
    with pytest.raises(F2ArtifactProductionError, match="assignment"):
        read_rank_artifact_certificate(
            certificates[0],
            bindings=bindings,
            causal_manifest=manifest,
            qualification_held_out_rows=manifest.rows,
        )


def test_world_and_qualification_drift_are_rejected() -> None:
    manifest = _manifest()
    with pytest.raises(F2ArtifactProductionError, match="supported"):
        build_rank_artifact_plan(
            causal_manifest=manifest,
            qualification_held_out_rows=manifest.rows,
            rank=0,
            world_size=4,
        )
    drifted = list(manifest.rows)
    drifted[0] = ManifestRow("log-0", 100_000, "different", True)
    with pytest.raises(F2ArtifactProductionError, match="coverage differ"):
        build_rank_artifact_plan(
            causal_manifest=manifest,
            qualification_held_out_rows=drifted,
            rank=0,
            world_size=1,
        )
