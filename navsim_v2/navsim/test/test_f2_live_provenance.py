"""Substitution and byte-tamper tests for the F2 live provenance closure."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import navsim.planning.validation.f2_live_provenance as live_provenance_module
from navsim.planning.training.f2_qualification_manifest import (
    QualificationRow,
    build_qualification_manifest,
    build_qualification_partition,
    canonical_manifest_bytes as canonical_qualification_manifest_bytes,
)
from navsim.planning.validation.f2_causal_validator import (
    ManifestRow,
    build_canonical_manifest,
    canonical_manifest_bytes as canonical_causal_manifest_bytes,
    manifest_sha256,
)
from navsim.planning.validation.f2_frozen_artifact_io import (
    ArtifactBindings,
    TARGET_CHANNEL_ORDER,
)
from navsim.planning.validation.f2_live_provenance import (
    EXPECTED_PROPOSAL_SCHEMA_NAME,
    EXPECTED_SCORER_NAME,
    EXPECTED_SIMULATOR_NAME,
    F2LiveProvenanceError,
    QUALIFICATION_SOURCE_SCHEMA_VERSION,
    REQUIRED_RUNTIME_SOURCE_PATHS,
    VerifiedF2Provenance,
    build_runtime_source_manifest,
    canonical_runtime_source_manifest_bytes,
    verify_live_f2_provenance,
)
from navsim.planning.validation.f2_metric_cache_manifest import (
    build_metric_cache_manifest,
    write_metric_cache_manifest,
)


def _canonical(payload: Any) -> bytes:
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


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path) -> SimpleNamespace:
    source_revision = "synthetic-runtime-revision"
    qualification_config_bytes = b"qualification-config\n"
    runtime_config_bytes = b"runtime-config\n"
    cache_root = tmp_path / "train_metric_cache_v2_pruned"
    cache_paths = {
        token: cache_root / log_name / "unknown" / token / "metric_cache.pkl"
        for token, log_name in (
            ("train-a", "train-log"),
            ("held-a", "log-a"),
            ("held-b", "log-b"),
        )
    }
    for token, cache_path in cache_paths.items():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes((token + "-cache\n").encode("utf-8"))
    cache_bytes = (
        "file_name\n"
        + "".join(path.as_posix() + "\n" for path in cache_paths.values())
    ).encode("utf-8")
    baseline_bytes = b"baseline-checkpoint\n"
    vjepa_bytes = b"vjepa-checkpoint\n"
    qualification_source_file_bytes = b"qualification-generator\n"
    qualification_source_bytes = _canonical(
        {
            "files": [
                {
                    "bytes": len(qualification_source_file_bytes),
                    "path": "generator.py",
                    "sha256": _sha(qualification_source_file_bytes),
                }
            ],
            "foundation_commit": "qualification-foundation",
            "schema_version": QUALIFICATION_SOURCE_SCHEMA_VERSION,
        }
    )
    rows = (
        ManifestRow("log-a", 100, "held-a", True),
        ManifestRow("log-b", 200, "held-b", False),
    )
    causal = build_canonical_manifest(
        rows,
        expected_tokens=(row.token for row in rows),
        sensor_available_tokens=(row.token for row in rows),
        cache_tokens=(row.token for row in rows),
        allowlist_logs=("log-a", "log-b"),
        cache_metadata_sha256=_sha(cache_bytes),
        source_revision=source_revision,
    )
    train = build_qualification_partition(
        "train",
        (QualificationRow("train-log", 1, "train-a", True),),
        raw_token_count=1,
        raw_log_count=1,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=_sha(b"train-exclusion\n"),
    )
    held = build_qualification_partition(
        "held_out",
        tuple(
            QualificationRow(
                row.log_name, row.time_us, row.token, row.future_camera_valid
            )
            for row in rows
        ),
        raw_token_count=2,
        raw_log_count=2,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=_sha(b"held-exclusion\n"),
    )
    qualification = build_qualification_manifest(
        train,
        held,
        config_sha256=_sha(qualification_config_bytes),
        cache_sha256=_sha(cache_bytes),
        source_sha256=_sha(qualification_source_bytes),
    )

    paths = SimpleNamespace(
        qualification=tmp_path / "qualification.json",
        causal=tmp_path / "causal.json",
        qualification_config=tmp_path / "qualification-config.json",
        runtime_config=tmp_path / "runtime-config.yaml",
        cache=cache_root / "metadata" / "cache.csv",
        cache_manifest=tmp_path / "cache-content.json",
        cache_root=cache_root,
        cache_file=cache_paths["held-a"],
        baseline=tmp_path / "baseline.ckpt",
        vjepa=tmp_path / "vjepa.pt",
        qualification_source=tmp_path / "qualification-source.json",
        qualification_source_root=tmp_path / "qualification-source-root",
        runtime_source=tmp_path / "runtime-source.json",
        runtime_source_root=tmp_path / "runtime-source-root",
    )
    paths.qualification_source_root.mkdir()
    paths.runtime_source_root.mkdir()
    paths.cache.parent.mkdir(parents=True)
    paths.qualification.write_bytes(canonical_qualification_manifest_bytes(qualification))
    paths.causal.write_bytes(canonical_causal_manifest_bytes(causal))
    paths.qualification_config.write_bytes(qualification_config_bytes)
    paths.runtime_config.write_bytes(runtime_config_bytes)
    paths.cache.write_bytes(cache_bytes)
    paths.baseline.write_bytes(baseline_bytes)
    paths.vjepa.write_bytes(vjepa_bytes)
    paths.qualification_source.write_bytes(qualification_source_bytes)
    (paths.qualification_source_root / "generator.py").write_bytes(
        qualification_source_file_bytes
    )
    for relative in REQUIRED_RUNTIME_SOURCE_PATHS:
        source_path = paths.runtime_source_root.joinpath(*relative.split("/"))
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes((relative + "\n").encode("utf-8"))
    runtime_source = build_runtime_source_manifest(
        paths.runtime_source_root, source_revision=source_revision
    )
    runtime_source_bytes = canonical_runtime_source_manifest_bytes(runtime_source)
    paths.runtime_source.write_bytes(runtime_source_bytes)
    qualification_sha = _sha(paths.qualification.read_bytes())
    cache_manifest = build_metric_cache_manifest(
        cache_root=paths.cache_root,
        metadata_path=paths.cache,
        expected_metadata_sha256=_sha(cache_bytes),
        qualification=qualification,
        expected_qualification_manifest_sha256=qualification_sha,
        workers=2,
    )
    cache_manifest_sha = write_metric_cache_manifest(
        paths.cache_manifest.resolve(), cache_manifest
    )
    bindings = ArtifactBindings(
        qualification_manifest_sha256=qualification_sha,
        causal_manifest_sha256=manifest_sha256(causal),
        baseline_checkpoint_sha256=_sha(baseline_bytes),
        vjepa_checkpoint_sha256=_sha(vjepa_bytes),
        cache_metadata_sha256=_sha(cache_bytes),
        metric_cache_manifest_sha256=cache_manifest_sha,
        raw_input_manifest_sha256=_sha(b"raw-input-manifest"),
        qualification_config_sha256=_sha(qualification_config_bytes),
        runtime_config_sha256=_sha(runtime_config_bytes),
        qualification_source_manifest_sha256=_sha(qualification_source_bytes),
        runtime_source_manifest_sha256=_sha(runtime_source_bytes),
        source_revision=source_revision,
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
    return SimpleNamespace(
        bindings=bindings,
        qualification=qualification,
        causal=causal,
        runtime_source=runtime_source,
        paths=paths,
    )


def _verify(fixture: SimpleNamespace, *, bindings: ArtifactBindings | None = None):
    paths = fixture.paths
    return verify_live_f2_provenance(
        bindings=fixture.bindings if bindings is None else bindings,
        qualification_manifest_path=paths.qualification,
        causal_manifest_path=paths.causal,
        qualification_config_path=paths.qualification_config,
        runtime_config_path=paths.runtime_config,
        cache_metadata_path=paths.cache,
        metric_cache_manifest_path=paths.cache_manifest,
        metric_cache_root=paths.cache_root,
        baseline_checkpoint_path=paths.baseline,
        vjepa_checkpoint_path=paths.vjepa,
        qualification_source_manifest_path=paths.qualification_source,
        qualification_source_root=paths.qualification_source_root,
        runtime_source_manifest_path=paths.runtime_source,
        runtime_source_root=paths.runtime_source_root,
    )


def test_live_verification_closes_distinct_qualification_and_runtime_sources(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    verified = _verify(fixture)
    assert type(verified) is VerifiedF2Provenance
    assert verified.bindings == fixture.bindings
    assert verified.qualification == fixture.qualification
    assert verified.causal == fixture.causal
    assert (
        fixture.bindings.qualification_source_manifest_sha256
        != fixture.bindings.runtime_source_manifest_sha256
    )
    declared = {item.path for item in verified.runtime_source.files}
    assert set(REQUIRED_RUNTIME_SOURCE_PATHS) <= declared


@pytest.mark.parametrize("manifest_kind", ("qualification", "runtime"))
def test_source_manifests_reject_uppercase_file_digest(
    tmp_path: Path, manifest_kind: str
) -> None:
    fixture = _fixture(tmp_path)
    if manifest_kind == "qualification":
        path = fixture.paths.qualification_source
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["files"][0]["sha256"] = payload["files"][0]["sha256"].upper()
        encoded = _canonical(payload)
        path.write_bytes(encoded)
        with pytest.raises(F2LiveProvenanceError, match="lowercase hexadecimal"):
            live_provenance_module.verify_qualification_source_manifest(
                path,
                expected_sha256=_sha(encoded),
                source_root=fixture.paths.qualification_source_root,
            )
    else:
        path = fixture.paths.runtime_source
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["files"][0]["sha256"] = payload["files"][0]["sha256"].upper()
        encoded = _canonical(payload)
        path.write_bytes(encoded)
        with pytest.raises(F2LiveProvenanceError, match="lowercase hexadecimal"):
            live_provenance_module.load_and_verify_runtime_source_manifest(
                path,
                expected_sha256=_sha(encoded),
                source_root=fixture.paths.runtime_source_root,
            )


def test_verified_provenance_cannot_be_constructed_from_digest_strings(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(F2LiveProvenanceError, match="must come from"):
        VerifiedF2Provenance(
            bindings=fixture.bindings,
            qualification=fixture.qualification,
            causal=fixture.causal,
            metric_cache=None,
            runtime_source=fixture.runtime_source,
            _seal=object(),
        )


@pytest.mark.parametrize(
    "target",
    (
        "qualification",
        "causal",
        "qualification_config",
        "runtime_config",
        "cache",
        "cache_manifest",
        "cache_file",
        "baseline",
        "vjepa",
        "qualification_source",
        "qualification_source_file",
        "runtime_source",
        "runtime_source_file",
    ),
)
def test_every_live_byte_boundary_rejects_tamper(tmp_path: Path, target: str) -> None:
    fixture = _fixture(tmp_path)
    path = getattr(fixture.paths, target, None)
    if target == "qualification_source_file":
        path = fixture.paths.qualification_source_root / "generator.py"
    elif target == "runtime_source_file":
        path = fixture.paths.runtime_source_root.joinpath(
            *REQUIRED_RUNTIME_SOURCE_PATHS[0].split("/")
        )
    assert isinstance(path, Path)
    path.write_bytes(path.read_bytes() + b"tamper")
    if target == "cache_file":
        verified = _verify(fixture)
        reference = verified.metric_cache.authenticated_references()["held-a"]
        with pytest.raises(ValueError):
            with reference.open_authenticated():
                pass
        return
    with pytest.raises((F2LiveProvenanceError, ValueError)):
        _verify(fixture)


@pytest.mark.parametrize(
    "bindings",
    (
        lambda value: replace(
            value,
            qualification_manifest_sha256=value.causal_manifest_sha256,
        ),
        lambda value: replace(
            value,
            causal_manifest_sha256=value.qualification_manifest_sha256,
        ),
        lambda value: replace(
            value,
            qualification_source_manifest_sha256=(
                value.runtime_source_manifest_sha256
            ),
        ),
        lambda value: replace(
            value,
            proposal_schema_sha256=value.simulator_sha256,
        ),
        lambda value: replace(
            value,
            vjepa_checkpoint_sha256=value.baseline_checkpoint_sha256,
        ),
    ),
)
def test_cross_role_digest_substitution_is_rejected(tmp_path: Path, bindings: Any) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises((F2LiveProvenanceError, ValueError)):
        _verify(fixture, bindings=bindings(fixture.bindings))


def test_runtime_manifest_cannot_omit_or_hide_a_source_file(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    undeclared = (
        fixture.paths.runtime_source_root
        / "navsim_v2"
        / "navsim"
        / "planning"
        / "validation"
        / "injected_runtime.py"
    )
    undeclared.write_bytes(b"injected\n")
    with pytest.raises(F2LiveProvenanceError, match="exact-scope coverage"):
        _verify(fixture)


@pytest.mark.parametrize(
    "shadow_name",
    (
        "injected.pyc",
        "injected.pyd",
        "injected.so",
        "injected.cpython-39-x86_64-linux-gnu.so",
    ),
)
def test_runtime_scope_rejects_import_shadow_artifacts(
    tmp_path: Path, shadow_name: str
) -> None:
    fixture = _fixture(tmp_path)
    shadow = (
        fixture.paths.runtime_source_root
        / "navsim_v2"
        / "navsim"
        / "planning"
        / "validation"
        / shadow_name
    )
    shadow.write_bytes(b"ignored-shadow\n")
    with pytest.raises(F2LiveProvenanceError, match="import-shadow"):
        _verify(fixture)


@pytest.mark.parametrize(
    "reader",
    (
        lambda path: live_provenance_module._stable_file_identity(path, "test"),
        lambda path: live_provenance_module._stable_file_payload(path, "test"),
    ),
)
def test_stable_provenance_reader_rejects_equal_size_lstat_open_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: Any
) -> None:
    path = tmp_path / "provenance.bin"
    replacement = tmp_path / "replacement.bin"
    path.write_bytes(b"trusted-bytes")
    replacement.write_bytes(b"hostile-byte!")
    assert path.stat().st_size == replacement.stat().st_size
    real_open = live_provenance_module.os.open
    replaced = False

    def replacing_open(target: Any, flags: int, *args: Any) -> int:
        nonlocal replaced
        if Path(target) == path and not replaced:
            replaced = True
            replacement.replace(path)
        return real_open(target, flags, *args)

    monkeypatch.setattr(live_provenance_module.os, "open", replacing_open)
    with pytest.raises(F2LiveProvenanceError, match="changed before"):
        reader(path)


@pytest.mark.parametrize(
    "reader",
    (
        lambda path: live_provenance_module._stable_file_identity(path, "test"),
        lambda path: live_provenance_module._stable_file_payload(path, "test"),
    ),
)
def test_stable_provenance_reader_rejects_final_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: Any
) -> None:
    path = tmp_path / "provenance.bin"
    replacement = tmp_path / "replacement.bin"
    path.write_bytes(b"trusted-bytes")
    replacement.write_bytes(b"hostile-byte!")
    real_close = live_provenance_module.os.close
    replaced = False

    def replacing_close(descriptor: int) -> None:
        nonlocal replaced
        real_close(descriptor)
        if not replaced:
            replaced = True
            replacement.replace(path)

    monkeypatch.setattr(live_provenance_module.os, "close", replacing_close)
    with pytest.raises(F2LiveProvenanceError, match="changed while"):
        reader(path)
