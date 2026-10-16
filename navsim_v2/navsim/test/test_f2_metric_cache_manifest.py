"""Synthetic security tests for the qualified navtrain cache manifest."""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import navsim.planning.validation.f2_metric_cache_manifest as cache_manifest_module
from navsim.planning.training.f2_qualification_manifest import (
    QualificationRow,
    build_qualification_manifest,
    build_qualification_partition,
    canonical_manifest_bytes as canonical_qualification_manifest_bytes,
)
from navsim.planning.validation.f2_metric_cache_manifest import (
    F2MetricCacheManifestError,
    MetricCacheContentEntry,
    build_metric_cache_manifest,
    canonical_metric_cache_manifest_bytes,
    load_and_verify_metric_cache_manifest,
    load_authenticated_metric_cache,
    metric_cache_manifest_sha256,
    score_authenticated_metric_caches,
    write_metric_cache_manifest,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _qualification() -> Any:
    train = build_qualification_partition(
        "train",
        (QualificationRow("log-train", 1, "train-token", True),),
        raw_token_count=1,
        raw_log_count=1,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=_sha(b"train exclusions\n"),
    )
    held = build_qualification_partition(
        "held_out",
        (QualificationRow("log-held", 2, "held-token", True),),
        raw_token_count=1,
        raw_log_count=1,
        removed_token_count=0,
        removed_log_count=0,
        mixed_log_count=0,
        exclusion_report_sha256=_sha(b"held exclusions\n"),
    )
    return build_qualification_manifest(
        train,
        held,
        config_sha256=_sha(b"qualification config\n"),
        cache_sha256=_sha(b"placeholder replaced by fixture"),
        source_sha256=_sha(b"qualification source\n"),
    )


def _metric_cache(log_name: str, time_us: int, token: str) -> Any:
    return SimpleNamespace(
        log_name=log_name,
        timepoint=SimpleNamespace(time_us=time_us),
        token=token,
    )


def _write_cache(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(path, "wb") as stream:
        pickle.dump(value, stream)


def _fixture(tmp_path: Path, *, overrides: Any = None) -> Any:
    root = tmp_path / "train_metric_cache_v2_pruned"
    paths = {
        "train-token": root / "log-train" / "frame" / "train-token" / "metric_cache.pkl",
        "held-token": root / "log-held" / "frame" / "held-token" / "metric_cache.pkl",
        "unqualified-token": root
        / "log-extra"
        / "frame"
        / "unqualified-token"
        / "metric_cache.pkl",
    }
    values = {
        "train-token": _metric_cache("log-train", 1, "train-token"),
        "held-token": _metric_cache("log-held", 2, "held-token"),
        "unqualified-token": _metric_cache("log-extra", 3, "unqualified-token"),
    }
    if overrides:
        values.update(overrides)
    for token, path in paths.items():
        value = values[token]
        if isinstance(value, bytes):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        else:
            _write_cache(path, value)
    metadata = root / "metadata" / "cache.csv"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        "path\n" + "".join(f"{path.as_posix()}\n" for path in paths.values()),
        encoding="utf-8",
    )
    metadata_sha = _sha(metadata.read_bytes())
    base = _qualification()
    qualification = build_qualification_manifest(
        base.train,
        base.held_out,
        config_sha256=base.config_sha256,
        cache_sha256=metadata_sha,
        source_sha256=base.source_sha256,
    )
    qualification_sha = _sha(canonical_qualification_manifest_bytes(qualification))
    manifest = build_metric_cache_manifest(
        cache_root=root,
        metadata_path=metadata,
        expected_metadata_sha256=metadata_sha,
        qualification=qualification,
        expected_qualification_manifest_sha256=qualification_sha,
        workers=2,
    )
    destination = (tmp_path / "cache-content.json").resolve()
    manifest_sha = write_metric_cache_manifest(destination, manifest)
    return type(
        "Fixture",
        (),
        {
            "root": root,
            "paths": paths,
            "metadata": metadata,
            "metadata_sha": metadata_sha,
            "qualification": qualification,
            "qualification_sha": qualification_sha,
            "manifest": manifest,
            "manifest_path": destination,
            "manifest_sha": manifest_sha,
        },
    )()


def _load(fixture: Any) -> Any:
    return load_and_verify_metric_cache_manifest(
        fixture.manifest_path,
        expected_sha256=fixture.manifest_sha,
        cache_root=fixture.root,
        metadata_path=fixture.metadata,
        expected_metadata_sha256=fixture.metadata_sha,
        qualification=fixture.qualification,
        expected_qualification_manifest_sha256=fixture.qualification_sha,
    )


def test_manifest_selects_exact_qualification_union_and_loads_same_fd(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    assert tuple(entry.token for entry in fixture.manifest.entries) == (
        "held-token",
        "train-token",
    )
    assert "unqualified-token" not in fixture.manifest.by_token
    verified = _load(fixture)
    references = verified.authenticated_references()
    assert set(references) == {"train-token", "held-token"}
    assert load_authenticated_metric_cache(references["held-token"]).token == "held-token"
    verified.verify_all(workers=2)


def test_verify_all_rejects_pickle_identity_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        overrides={
            "held-token": _metric_cache("wrong-log", 999, "held-token"),
        },
    )
    with pytest.raises(F2MetricCacheManifestError, match="differs from qualification"):
        _load(fixture).verify_all(workers=2)


def test_verify_all_rejects_bad_lzma_pickle(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        overrides={"held-token": b"not an lzma-compressed pickle\n"},
    )
    with pytest.raises(F2MetricCacheManifestError, match="complete LZMA/pickle"):
        _load(fixture).verify_all(workers=2)


def test_verify_all_rejects_trailing_second_pickle(tmp_path: Path) -> None:
    payload = lzma.compress(
        pickle.dumps(_metric_cache("log-held", 2, "held-token"))
        + pickle.dumps({"unexpected": "second object"})
    )
    fixture = _fixture(tmp_path, overrides={"held-token": payload})
    with pytest.raises(F2MetricCacheManifestError, match="trailing decompressed"):
        _load(fixture).verify_all(workers=2)


def test_verify_all_rejects_dropped_last_worker_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    real_executor = cache_manifest_module.ThreadPoolExecutor

    class DropLastExecutor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._delegate = real_executor(*args, **kwargs)

        def __enter__(self) -> "DropLastExecutor":
            self._delegate.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self._delegate.__exit__(*args)

        def map(self, function: Any, values: Any) -> Any:
            results = tuple(self._delegate.map(function, values))
            return iter(results[:-1])

    monkeypatch.setattr(cache_manifest_module, "ThreadPoolExecutor", DropLastExecutor)
    with pytest.raises(F2MetricCacheManifestError, match="exact qualification coverage"):
        _load(fixture).verify_all(workers=2)


def test_spawn_worker_scoring_dispatch_cannot_fall_back_to_raw_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    reference = _load(fixture).authenticated_references()["held-token"]
    from navsim.agents.drive_jepa_perception_based.score_module import (
        compute_navsim_score,
    )

    original_loader = compute_navsim_score.load_metric_cache

    def fake_get_scores(arguments: Any) -> list[Any]:
        return [
            compute_navsim_score.load_metric_cache(argument["token"])
            for argument in arguments
        ]

    monkeypatch.setattr(compute_navsim_score, "get_scores", fake_get_scores)
    assert score_authenticated_metric_caches(
        ({"token": reference, "poses": None, "test": True},)
    )[0].token == "held-token"
    assert compute_navsim_score.load_metric_cache is original_loader
    with pytest.raises(F2MetricCacheManifestError, match="non-authenticated"):
        score_authenticated_metric_caches(
            ({"token": reference.path, "poses": None, "test": True},)
        )


def test_pickle_tamper_is_rejected_before_unpickle(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    verified = _load(fixture)
    reference = verified.authenticated_references()["held-token"]
    assert load_authenticated_metric_cache(reference).token == "held-token"
    target = fixture.paths["held-token"]
    payload = bytearray(target.read_bytes())
    payload[len(payload) // 2] ^= 1
    target.write_bytes(payload)
    with pytest.raises(F2MetricCacheManifestError, match="SHA-256 mismatch"):
        load_authenticated_metric_cache(reference)


def test_symlinked_pickle_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = fixture.paths["held-token"]
    replacement = target.with_name("replacement.pkl")
    replacement.write_bytes(target.read_bytes())
    target.unlink()
    try:
        target.symlink_to(replacement)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    verified = _load(fixture)
    with pytest.raises(F2MetricCacheManifestError, match="symlink"):
        load_authenticated_metric_cache(
            verified.authenticated_references()["held-token"]
        )


def test_terminal_metadata_symlink_is_rejected_before_resolve(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    alias = fixture.metadata.with_name("metadata-link.csv")
    try:
        alias.symlink_to(fixture.metadata)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(F2MetricCacheManifestError, match="symlink"):
        build_metric_cache_manifest(
            cache_root=fixture.root,
            metadata_path=alias,
            expected_metadata_sha256=fixture.metadata_sha,
            qualification=fixture.qualification,
            expected_qualification_manifest_sha256=fixture.qualification_sha,
            workers=1,
        )
    with pytest.raises(F2MetricCacheManifestError, match="symlink"):
        load_and_verify_metric_cache_manifest(
            fixture.manifest_path,
            expected_sha256=fixture.manifest_sha,
            cache_root=fixture.root,
            metadata_path=alias,
            expected_metadata_sha256=fixture.metadata_sha,
            qualification=fixture.qualification,
            expected_qualification_manifest_sha256=fixture.qualification_sha,
        )


def test_metadata_path_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "train_metric_cache_v2_pruned"
    metadata = root / "metadata" / "cache.csv"
    metadata.parent.mkdir(parents=True)
    outside = tmp_path / "token" / "metric_cache.pkl"
    _write_cache(outside, {"token": "train-token"})
    metadata.write_text(f"path\n{outside.as_posix()}\n", encoding="utf-8")
    qualification = _qualification()
    with pytest.raises(F2MetricCacheManifestError, match="escapes"):
        build_metric_cache_manifest(
            cache_root=root,
            metadata_path=metadata,
            expected_metadata_sha256=_sha(metadata.read_bytes()),
            qualification=qualification,
            expected_qualification_manifest_sha256=_sha(
                canonical_qualification_manifest_bytes(qualification)
            ),
            workers=1,
        )


def test_missing_qualified_metadata_mapping_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.metadata.write_text(
        "path\n" + fixture.paths["train-token"].as_posix() + "\n",
        encoding="utf-8",
    )
    digest = _sha(fixture.metadata.read_bytes())
    qualification = build_qualification_manifest(
        fixture.qualification.train,
        fixture.qualification.held_out,
        config_sha256=fixture.qualification.config_sha256,
        cache_sha256=digest,
        source_sha256=fixture.qualification.source_sha256,
    )
    with pytest.raises(F2MetricCacheManifestError, match="lacks qualified"):
        build_metric_cache_manifest(
            cache_root=fixture.root,
            metadata_path=fixture.metadata,
            expected_metadata_sha256=digest,
            qualification=qualification,
            expected_qualification_manifest_sha256=_sha(
                canonical_qualification_manifest_bytes(qualification)
            ),
            workers=1,
        )


def test_extra_manifest_token_is_rejected_by_qualification_join(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    extra_path = fixture.paths["unqualified-token"]
    extra = MetricCacheContentEntry(
        token="unqualified-token",
        partition="train",
        relative_path=extra_path.relative_to(fixture.root).as_posix(),
        byte_size=extra_path.stat().st_size,
        sha256=_sha(extra_path.read_bytes()),
    )
    changed = replace(
        fixture.manifest,
        entries=tuple(sorted((*fixture.manifest.entries, extra), key=lambda row: row.token)),
    )
    changed_path = (tmp_path / "changed.json").resolve()
    changed_sha = write_metric_cache_manifest(changed_path, changed)
    with pytest.raises(F2MetricCacheManifestError, match="extra"):
        load_and_verify_metric_cache_manifest(
            changed_path,
            expected_sha256=changed_sha,
            cache_root=fixture.root,
            metadata_path=fixture.metadata,
            expected_metadata_sha256=fixture.metadata_sha,
            qualification=fixture.qualification,
            expected_qualification_manifest_sha256=fixture.qualification_sha,
        )


def test_case_colliding_paths_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = fixture.manifest.entries[0]
    second = fixture.manifest.entries[1]
    with pytest.raises(F2MetricCacheManifestError, match="case-colliding"):
        replace(
            fixture.manifest,
            entries=tuple(
                sorted(
                    (
                        first,
                        replace(
                            second,
                            relative_path=first.relative_path.swapcase(),
                        ),
                    ),
                    key=lambda row: row.token,
                )
            ),
        )


def test_pathname_swap_during_same_fd_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"original")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replaced")
    real_open = cache_manifest_module.os.open
    swapped = False

    def replacing_open(value: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal swapped
        descriptor = real_open(value, flags, *args, **kwargs)
        if Path(value) == path and not swapped:
            swapped = True
            os.replace(replacement, path)
        return descriptor

    monkeypatch.setattr(cache_manifest_module.os, "open", replacing_open)
    with pytest.raises(F2MetricCacheManifestError, match="changed"):
        cache_manifest_module._stable_file_payload(path, "test payload")


def test_parent_directory_swap_cannot_redirect_authenticated_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    reference = _load(fixture).authenticated_references()["held-token"]
    token_directory = fixture.paths["held-token"].parent
    moved_directory = token_directory.with_name("held-token-original")
    hostile_directory = tmp_path / "hostile" / "held-token"
    _write_cache(
        hostile_directory / "metric_cache.pkl",
        _metric_cache("hostile-log", 999, "hostile"),
    )
    real_open = cache_manifest_module.os.open
    swapped = False

    def swapping_open(value: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal swapped
        descriptor = real_open(value, flags, *args, **kwargs)
        if value == "held-token" and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            token_directory.rename(moved_directory)
            try:
                token_directory.symlink_to(hostile_directory, target_is_directory=True)
            except (OSError, NotImplementedError):
                pytest.skip("directory symlinks are unavailable")
        return descriptor

    monkeypatch.setattr(cache_manifest_module.os, "open", swapping_open)
    with pytest.raises(
        F2MetricCacheManifestError,
        match=r"directory (changed before open|path changed)",
    ):
        load_authenticated_metric_cache(reference)


def test_manifest_write_is_no_overwrite(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(FileExistsError):
        write_metric_cache_manifest(fixture.manifest_path, fixture.manifest)
    assert metric_cache_manifest_sha256(fixture.manifest) == fixture.manifest_sha


def test_canonical_manifest_rejects_noncanonical_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = json.loads(canonical_metric_cache_manifest_bytes(fixture.manifest))
    noncanonical = json.dumps(payload, indent=2).encode("utf-8")
    with pytest.raises(F2MetricCacheManifestError, match="not canonical"):
        cache_manifest_module.metric_cache_manifest_from_bytes(
            noncanonical,
            expected_sha256=_sha(noncanonical),
        )


def test_uppercase_digest_is_rejected_without_normalization(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = fixture.manifest.to_payload()
    payload["entries"][0]["sha256"] = payload["entries"][0]["sha256"].upper()
    encoded = cache_manifest_module._canonical_json_bytes(payload)
    with pytest.raises(F2MetricCacheManifestError, match="lowercase"):
        cache_manifest_module.metric_cache_manifest_from_bytes(
            encoded,
            expected_sha256=_sha(encoded),
        )


def test_from_bytes_rejects_non_idempotent_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    payload = fixture.manifest.to_payload()
    payload["entries"][0]["sha256"] = payload["entries"][0]["sha256"].upper()
    encoded = cache_manifest_module._canonical_json_bytes(payload)
    real_sha256 = cache_manifest_module._sha256

    def legacy_normalizing_sha(value: Any, name: str) -> str:
        if isinstance(value, str) and len(value) == 64:
            value = value.lower()
        return real_sha256(value, name)

    monkeypatch.setattr(
        cache_manifest_module, "_sha256", legacy_normalizing_sha
    )
    with pytest.raises(F2MetricCacheManifestError, match="idempotently canonical"):
        cache_manifest_module.metric_cache_manifest_from_bytes(
            encoded,
            expected_sha256=_sha(encoded),
        )
