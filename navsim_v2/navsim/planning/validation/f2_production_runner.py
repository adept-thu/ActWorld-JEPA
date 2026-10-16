"""Fail-closed production persistence and orchestration for F2 validation.

This module deliberately delegates causal math, shard merging, invariance, and
formal checkpoint ranking to the already frozen F2 validation modules.  It
adds the missing production boundary around them:

* strict canonical JSON with duplicate-key and non-finite-number rejection;
* SHA-256 sidecars plus atomic, no-overwrite publication;
* exact rank coverage before a world can be merged;
* re-playable 1/2/8-world invariance certificates;
* exact epoch 0..3 checkpoint ranking from invariant certificates only; and
* one offline-authority-signed production run-root that binds every input used
  by ``evaluate-shard``, ``merge-world``, ``prove-invariance``, and ranking.

All persisted evaluations are restricted to held-out ``navtrain``.  No split
selector is exposed by the API or CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from navsim.planning.validation.f2_causal_shard_runner import (
    CausalShardEvaluation,
    CausalWorldEvaluation,
    F2CausalShardRunnerError,
    SUPPORTED_WORLD_SIZES,
    evaluate_causal_shard,
    merge_causal_world,
    prove_1_2_8_invariance,
)
from navsim.planning.validation.f2_causal_validator import (
    F2_CHECKPOINT_EPOCHS,
    F2_MODEL_NUMERIC_MODE,
    HARD_FACTOR_NAMES,
    AggregateMetrics,
    CanonicalManifest,
    CausalValidationError,
    CheckpointEvaluation,
    CheckpointRanking,
    GateDecision,
    TokenCausalResult,
    aggregate_results,
    canonical_evaluation_bytes,
    canonical_manifest_bytes,
    canonical_metrics_bytes,
    evaluate_formal_gates,
    manifest_from_bytes,
    rank_checkpoints,
)
from navsim.planning.validation.f2_frozen_artifact_io import (
    ArtifactBindings,
    CoverageEntry,
    TARGET_CHANNEL_ORDER,
    build_artifact_coverage,
)
from navsim.planning.validation.f2_live_provenance import verify_live_f2_provenance
from navsim.planning.validation.f2_navsim_comfort import DriveJepaExtendedComfort
from navsim.planning.validation.f2_navtrain_input_manifest import (
    VerifiedNavtrainRawInputs,
    load_and_verify_raw_input_manifest,
)
from navsim.planning.script.f2_launch_guard import (
    AUTHORIZED_ARTIFACT_KEYS,
    TRUSTED_AUTHORITY_SSH_PUBLIC_KEY,
    verify_f2_runtime_claim,
    verify_f2_signed_contract_lightweight,
)
from navsim.planning.validation.f2_metric_cache_manifest import (
    F2_PRUNED_CACHE_FAMILY,
)
from navsim.planning.validation.f2_navsim_runtime_adapters import (
    build_official_f2_causal_model_adapter,
    build_qualified_held_out_navtrain_feature_source,
)


DATASET = "navtrain"
PARTITION = "held_out"
SHARD_SCHEMA_VERSION = "actworld-f2-production-causal-shard-v1"
WORLD_SCHEMA_VERSION = "actworld-f2-production-causal-world-v1"
CHECKPOINT_SCHEMA_VERSION = "actworld-f2-production-invariant-checkpoint-v1"
RANKING_SCHEMA_VERSION = "actworld-f2-production-checkpoint-ranking-v1"
RUN_ROOT_WORLD_SCHEMA_VERSION = "actworld-f2-run-root-causal-world-v1"
RUN_ROOT_CHECKPOINT_SCHEMA_VERSION = (
    "actworld-f2-run-root-invariant-checkpoint-v1"
)
RUN_ROOT_RANKING_SCHEMA_VERSION = "actworld-f2-run-root-checkpoint-ranking-v1"
BINDINGS_SCHEMA_VERSION = "actworld-f2-production-artifact-bindings-v1"
RUNTIME_CONFIG_SCHEMA_VERSION = "actworld-f2-production-runtime-config-v3"
RUN_ROOT_SCHEMA_VERSION = "actworld-f2-production-run-root-v2"
RUN_ROOT_SSH_NAMESPACE = "actworld-f2-production-run-root-v2"
RUN_ROOT_SSH_PRINCIPAL = "actworld-f2-offline-authority"
RUNTIME_RECORD_SCHEMA_VERSION = "actworld-f2-deterministic-runtime-record-v1"
TRAINING_LINEAGE_SCHEMA_VERSION = "actworld-f2-training-lineage-v1"
TRAINING_LINEAGE_FILENAME = "f2_training_lineage.json"
DEVICE_POLICY = "cuda_local_rank_or_explicit"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

_IDENTITY_KEYS = {
    "artifact_sha256",
    "checkpoint_epoch",
    "checkpoint_sha256",
    "config_sha256",
    "manifest_sha256",
    "numeric_mode",
    "source_sha256",
}
_TOKEN_RESULT_KEYS = {
    "base_continuous",
    "base_selected_index",
    "base_selected_probability",
    "base_selected_q",
    "base_selected_truth",
    "base_target_scores",
    "critic_continuous",
    "critic_selected_index",
    "critic_selected_probability",
    "critic_selected_q",
    "critic_selected_truth",
    "critic_target_scores",
    "future_camera_valid",
    "log_name",
    "oracle_aggregate",
    "pairwise_comparable",
    "pairwise_correct",
    "time_us",
    "token",
}
_METRIC_KEYS = {
    "base_score_error",
    "base_selected_score",
    "coverage_fraction",
    "critic_delta",
    "critic_oracle_regret",
    "critic_pairwise_accuracy",
    "critic_pairwise_comparable",
    "critic_pairwise_correct",
    "critic_score_error",
    "critic_score_error_excess",
    "critic_selected_score",
    "expected_token_count",
    "fatal_count",
    "future_camera_valid_fraction",
    "hard_factor_deltas",
    "observed_token_count",
    "oracle_score",
    "proposal_headroom_closed",
}
_RUNTIME_PATH_KEYS = {
    "artifact",
    "baseline_checkpoint",
    "cache_metadata",
    "causal_manifest",
    "experiment_root",
    "metric_cache_manifest",
    "metric_cache_root",
    "navsim_log",
    "nuplan_maps_root",
    "qualification_config",
    "qualification_manifest",
    "qualification_source_manifest",
    "qualification_source_root",
    "raw_input_manifest",
    "repository_root",
    "runtime_source_manifest",
    "runtime_source_root",
    "sensor_blobs",
    "vjepa_checkpoint",
}


class F2ProductionRunnerError(RuntimeError):
    """Raised whenever a production artifact cannot be trusted."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise F2ProductionRunnerError(f"{name} must be a 64-character SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise F2ProductionRunnerError(
            f"{name} must use exactly 64 lowercase hexadecimal characters"
        )
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise F2ProductionRunnerError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _exact_keys(payload: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise F2ProductionRunnerError(f"{name} must be a JSON object")
    actual = set(payload)
    if actual != expected:
        raise F2ProductionRunnerError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return payload


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the sole accepted JSON representation for production files."""

    try:
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
    except (TypeError, ValueError) as error:
        raise F2ProductionRunnerError(
            "production payload is not finite canonical JSON"
        ) from error


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise F2ProductionRunnerError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise F2ProductionRunnerError(
        f"non-finite JSON constant {value!r} is forbidden"
    )


def parse_canonical_json_bytes(payload: bytes, *, name: str) -> Any:
    """Parse JSON and reject duplicate keys, NaN/Inf, and byte drift."""

    if not isinstance(payload, bytes):
        raise F2ProductionRunnerError(f"{name} payload must be bytes")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except F2ProductionRunnerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F2ProductionRunnerError(f"{name} is not valid UTF-8 JSON") from error
    if canonical_json_bytes(decoded) != payload:
        raise F2ProductionRunnerError(f"{name} is not canonical JSON")
    return decoded


def sha256_sidecar_path(path: Path | str) -> Path:
    target = Path(path)
    return target.with_name(target.name + ".sha256")


def _partial_path(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def _fsync_parent(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(str(path.parent), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_bytes_no_overwrite(path: Path | str, payload: bytes) -> str:
    """Atomically publish payload plus digest sidecar without replacement."""

    target = Path(path)
    sidecar = sha256_sidecar_path(target)
    partial = _partial_path(target)
    sidecar_partial = _partial_path(sidecar)
    if not target.parent.is_dir():
        raise F2ProductionRunnerError("output parent directory does not exist")
    occupied = [
        candidate
        for candidate in (target, sidecar, partial, sidecar_partial)
        if candidate.exists()
    ]
    if occupied:
        raise F2ProductionRunnerError(
            "output, sidecar, or partial already exists; overwrite is forbidden: "
            + ", ".join(str(path) for path in occupied)
        )
    digest = _sha256_bytes(payload)
    sidecar_payload = (digest + "\n").encode("ascii")
    sidecar_published = False
    payload_published = False
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        with sidecar_partial.open("xb") as stream:
            stream.write(sidecar_payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(sidecar_partial, sidecar)
        sidecar_published = True
        os.link(partial, target)
        payload_published = True
        _fsync_parent(target)
    except FileExistsError as error:
        raise F2ProductionRunnerError(
            "output appeared during atomic publication; overwrite is forbidden"
        ) from error
    except OSError as error:
        raise F2ProductionRunnerError(
            "atomic hard-link publication failed"
        ) from error
    finally:
        if not payload_published and sidecar_published:
            try:
                if sidecar.exists() and sidecar_partial.exists() and os.path.samefile(
                    sidecar, sidecar_partial
                ):
                    sidecar.unlink()
            except OSError:
                pass
        if payload_published:
            partial.unlink(missing_ok=True)
            sidecar_partial.unlink(missing_ok=True)
            _fsync_parent(target)
    return digest


def write_canonical_json_no_overwrite(path: Path | str, payload: Any) -> str:
    return write_bytes_no_overwrite(path, canonical_json_bytes(payload))


def _file_identity(value: os.stat_result) -> Tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_regular_file(path: Path, *, name: str) -> bytes:
    """Read one non-symlink regular file through an unchanged descriptor."""

    try:
        path_before = path.lstat()
    except OSError as error:
        raise F2ProductionRunnerError(f"cannot stat {name}: {path}") from error
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise F2ProductionRunnerError(f"{name} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            opened = os.fstat(stream.fileno())
            if _file_identity(opened) != _file_identity(path_before):
                raise F2ProductionRunnerError(f"{name} changed before open")
            chunks = []
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                chunks.append(chunk)
            after = os.fstat(stream.fileno())
        path_after = path.lstat()
    except F2ProductionRunnerError:
        raise
    except OSError as error:
        raise F2ProductionRunnerError(f"cannot read stable {name}: {path}") from error
    identity = _file_identity(opened)
    if identity != _file_identity(after) or identity != _file_identity(path_after):
        raise F2ProductionRunnerError(f"{name} changed while being read")
    return b"".join(chunks)


def _sha256_stable_regular_file(path: Path, *, name: str) -> str:
    """Stream-hash a non-symlink file through the same verified descriptor."""

    try:
        path_before = path.lstat()
    except OSError as error:
        raise F2ProductionRunnerError(f"cannot stat {name}: {path}") from error
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise F2ProductionRunnerError(f"{name} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            opened = os.fstat(stream.fileno())
            if _file_identity(opened) != _file_identity(path_before):
                raise F2ProductionRunnerError(f"{name} changed before open")
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        path_after = path.lstat()
    except F2ProductionRunnerError:
        raise
    except OSError as error:
        raise F2ProductionRunnerError(f"cannot hash stable {name}: {path}") from error
    identity = _file_identity(opened)
    if identity != _file_identity(after) or identity != _file_identity(path_after):
        raise F2ProductionRunnerError(f"{name} changed while being hashed")
    return digest.hexdigest()


def read_bytes_with_sidecar(path: Path | str) -> Tuple[bytes, str]:
    target = Path(path)
    sidecar = sha256_sidecar_path(target)
    payload = _read_stable_regular_file(target, name="production file")
    sidecar_payload = _read_stable_regular_file(
        sidecar, name="production SHA sidecar"
    )
    if len(sidecar_payload) != 65 or not sidecar_payload.endswith(b"\n"):
        raise F2ProductionRunnerError("SHA sidecar is not canonical")
    try:
        expected = sidecar_payload[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise F2ProductionRunnerError("SHA sidecar is not ASCII") from error
    expected = _sha256(expected, "sidecar SHA-256")
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise F2ProductionRunnerError("production file SHA-256 differs from sidecar")
    return payload, actual


def _read_canonical_file(path: Path | str, *, name: str) -> Tuple[Any, bytes, str]:
    payload, digest = read_bytes_with_sidecar(path)
    return parse_canonical_json_bytes(payload, name=name), payload, digest


def _identity_payload(evaluation: Any) -> Dict[str, Any]:
    return {
        "artifact_sha256": evaluation.artifact_sha256,
        "checkpoint_epoch": evaluation.checkpoint_epoch,
        "checkpoint_sha256": evaluation.checkpoint_sha256,
        "config_sha256": evaluation.config_sha256,
        "manifest_sha256": evaluation.manifest_sha256,
        "numeric_mode": evaluation.numeric_mode,
        "source_sha256": evaluation.source_sha256,
    }


def _parse_identity(payload: Any) -> Dict[str, Any]:
    identity = _exact_keys(payload, _IDENTITY_KEYS, "evaluation identity")
    epoch = _integer(identity["checkpoint_epoch"], "checkpoint_epoch")
    if epoch not in F2_CHECKPOINT_EPOCHS:
        raise F2ProductionRunnerError("checkpoint epoch must be 0, 1, 2, or 3")
    numeric_mode = identity["numeric_mode"]
    if numeric_mode != F2_MODEL_NUMERIC_MODE:
        raise F2ProductionRunnerError("evaluation numeric mode drifted")
    return {
        "artifact_sha256": _sha256(identity["artifact_sha256"], "artifact"),
        "checkpoint_epoch": epoch,
        "checkpoint_sha256": _sha256(identity["checkpoint_sha256"], "checkpoint"),
        "config_sha256": _sha256(identity["config_sha256"], "config"),
        "manifest_sha256": _sha256(identity["manifest_sha256"], "manifest"),
        "numeric_mode": numeric_mode,
        "source_sha256": _sha256(identity["source_sha256"], "source"),
    }


def _parse_token_result(payload: Any) -> TokenCausalResult:
    row = _exact_keys(payload, _TOKEN_RESULT_KEYS, "causal token result")
    try:
        return TokenCausalResult(
            log_name=row["log_name"],
            time_us=row["time_us"],
            token=row["token"],
            future_camera_valid=row["future_camera_valid"],
            critic_continuous=row["critic_continuous"],
            base_continuous=row["base_continuous"],
            critic_selected_index=row["critic_selected_index"],
            base_selected_index=row["base_selected_index"],
            critic_selected_q=row["critic_selected_q"],
            base_selected_q=row["base_selected_q"],
            critic_selected_probability=row["critic_selected_probability"],
            base_selected_probability=row["base_selected_probability"],
            critic_selected_truth=row["critic_selected_truth"],
            base_selected_truth=row["base_selected_truth"],
            critic_target_scores=tuple(row["critic_target_scores"]),
            base_target_scores=tuple(row["base_target_scores"]),
            oracle_aggregate=row["oracle_aggregate"],
            pairwise_correct=row["pairwise_correct"],
            pairwise_comparable=row["pairwise_comparable"],
        )
    except (CausalValidationError, KeyError, TypeError) as error:
        raise F2ProductionRunnerError("causal token result is malformed") from error


def _parse_metrics(payload: Any) -> AggregateMetrics:
    metrics = _exact_keys(payload, _METRIC_KEYS, "aggregate metrics")
    hard = _exact_keys(
        metrics["hard_factor_deltas"], set(HARD_FACTOR_NAMES), "hard-factor metrics"
    )
    try:
        return AggregateMetrics(
            expected_token_count=metrics["expected_token_count"],
            observed_token_count=metrics["observed_token_count"],
            coverage_fraction=metrics["coverage_fraction"],
            fatal_count=metrics["fatal_count"],
            future_camera_valid_fraction=metrics["future_camera_valid_fraction"],
            critic_selected_score=metrics["critic_selected_score"],
            base_selected_score=metrics["base_selected_score"],
            oracle_score=metrics["oracle_score"],
            critic_delta=metrics["critic_delta"],
            critic_oracle_regret=metrics["critic_oracle_regret"],
            critic_pairwise_accuracy=metrics["critic_pairwise_accuracy"],
            critic_pairwise_correct=metrics["critic_pairwise_correct"],
            critic_pairwise_comparable=metrics["critic_pairwise_comparable"],
            critic_score_error=metrics["critic_score_error"],
            base_score_error=metrics["base_score_error"],
            critic_score_error_excess=metrics["critic_score_error_excess"],
            proposal_headroom_closed=metrics["proposal_headroom_closed"],
            hard_factor_deltas=tuple(hard[name] for name in HARD_FACTOR_NAMES),
        )
    except (CausalValidationError, KeyError, TypeError) as error:
        raise F2ProductionRunnerError("aggregate metrics are malformed") from error


def _local_shard_metrics(
    results: Sequence[TokenCausalResult],
) -> Optional[AggregateMetrics]:
    if not results:
        return None
    return aggregate_results(
        results,
        expected_tokens=(row.token for row in results),
        fatal_count=0,
    )


def _nullable_metrics_bytes(metrics: Optional[AggregateMetrics]) -> bytes:
    return (
        canonical_metrics_bytes(metrics)
        if metrics is not None
        else canonical_json_bytes(None)
    )


def shard_evaluation_payload(evaluation: CausalShardEvaluation) -> Dict[str, Any]:
    if not isinstance(evaluation, CausalShardEvaluation):
        raise F2ProductionRunnerError("evaluation must be CausalShardEvaluation")
    metrics = _local_shard_metrics(evaluation.results)
    return {
        "assigned_logs": list(evaluation.assigned_logs),
        "dataset": DATASET,
        "evaluation_sha256": evaluation.evaluation_sha256,
        "fatal_count": 0,
        "identity": _identity_payload(evaluation),
        "kind": "causal_shard",
        "metrics": metrics.to_payload() if metrics is not None else None,
        "metrics_sha256": _sha256_bytes(_nullable_metrics_bytes(metrics)),
        "partition": PARTITION,
        "rank": evaluation.rank,
        "results": [row.to_payload() for row in evaluation.results],
        "schema_version": SHARD_SCHEMA_VERSION,
        "world_size": evaluation.world_size,
    }


def shard_evaluation_bytes(evaluation: CausalShardEvaluation) -> bytes:
    return canonical_json_bytes(shard_evaluation_payload(evaluation))


def parse_shard_evaluation_bytes(payload: bytes) -> CausalShardEvaluation:
    decoded = parse_canonical_json_bytes(payload, name="causal shard evaluation")
    document = _exact_keys(
        decoded,
        {
            "assigned_logs",
            "dataset",
            "evaluation_sha256",
            "fatal_count",
            "identity",
            "kind",
            "metrics",
            "metrics_sha256",
            "partition",
            "rank",
            "results",
            "schema_version",
            "world_size",
        },
        "causal shard document",
    )
    if (
        document["schema_version"] != SHARD_SCHEMA_VERSION
        or document["kind"] != "causal_shard"
        or document["dataset"] != DATASET
        or document["partition"] != PARTITION
    ):
        raise F2ProductionRunnerError("causal shard schema or partition drifted")
    if document["fatal_count"] != 0 or isinstance(document["fatal_count"], bool):
        raise F2ProductionRunnerError("authoritative shard fatal_count must be zero")
    identity = _parse_identity(document["identity"])
    if not isinstance(document["assigned_logs"], list) or not isinstance(
        document["results"], list
    ):
        raise F2ProductionRunnerError("causal shard rows must be JSON arrays")
    results = tuple(_parse_token_result(row) for row in document["results"])
    try:
        evaluation = CausalShardEvaluation(
            rank=document["rank"],
            world_size=document["world_size"],
            assigned_logs=tuple(document["assigned_logs"]),
            checkpoint_sha256=identity["checkpoint_sha256"],
            checkpoint_epoch=identity["checkpoint_epoch"],
            manifest_sha256=identity["manifest_sha256"],
            artifact_sha256=identity["artifact_sha256"],
            config_sha256=identity["config_sha256"],
            source_sha256=identity["source_sha256"],
            results=results,
            fatal_count=0,
            numeric_mode=identity["numeric_mode"],
        )
    except (F2CausalShardRunnerError, CausalValidationError, TypeError) as error:
        raise F2ProductionRunnerError("causal shard evaluation is invalid") from error
    expected_evaluation = evaluation.evaluation_sha256
    if _sha256(document["evaluation_sha256"], "evaluation") != expected_evaluation:
        raise F2ProductionRunnerError("causal shard evaluation SHA-256 mismatch")
    computed_metrics = _local_shard_metrics(evaluation.results)
    if document["metrics"] is None:
        parsed_metrics = None
    else:
        parsed_metrics = _parse_metrics(document["metrics"])
    if parsed_metrics != computed_metrics:
        raise F2ProductionRunnerError("causal shard local metrics drifted")
    expected_metrics_sha = _sha256_bytes(_nullable_metrics_bytes(computed_metrics))
    if _sha256(document["metrics_sha256"], "metrics") != expected_metrics_sha:
        raise F2ProductionRunnerError("causal shard metrics SHA-256 mismatch")
    return evaluation


def write_shard_evaluation(
    path: Path | str, evaluation: CausalShardEvaluation
) -> str:
    return write_bytes_no_overwrite(path, shard_evaluation_bytes(evaluation))


def read_shard_evaluation(path: Path | str) -> CausalShardEvaluation:
    payload, _ = read_bytes_with_sidecar(path)
    return parse_shard_evaluation_bytes(payload)


def _recomputed_world_metrics(
    evaluation: CausalWorldEvaluation,
) -> AggregateMetrics:
    return aggregate_results(
        evaluation.results,
        expected_tokens=(row.token for row in evaluation.results),
        fatal_count=0,
    )


def world_evaluation_payload(evaluation: CausalWorldEvaluation) -> Dict[str, Any]:
    if not isinstance(evaluation, CausalWorldEvaluation):
        raise F2ProductionRunnerError("evaluation must be CausalWorldEvaluation")
    recomputed = _recomputed_world_metrics(evaluation)
    if canonical_metrics_bytes(recomputed) != canonical_metrics_bytes(evaluation.metrics):
        raise F2ProductionRunnerError("world metrics do not equal canonical token results")
    return {
        "dataset": DATASET,
        "evaluation_sha256": evaluation.evaluation_sha256,
        "fatal_count": 0,
        "identity": _identity_payload(evaluation),
        "kind": "causal_world",
        "metrics": evaluation.metrics.to_payload(),
        "metrics_sha256": evaluation.metrics_sha256,
        "partition": PARTITION,
        "results": [row.to_payload() for row in evaluation.results],
        "schema_version": WORLD_SCHEMA_VERSION,
        "world_size": evaluation.world_size,
    }


def world_evaluation_bytes(evaluation: CausalWorldEvaluation) -> bytes:
    return canonical_json_bytes(world_evaluation_payload(evaluation))


def parse_world_evaluation_bytes(payload: bytes) -> CausalWorldEvaluation:
    decoded = parse_canonical_json_bytes(payload, name="causal world evaluation")
    document = _exact_keys(
        decoded,
        {
            "dataset",
            "evaluation_sha256",
            "fatal_count",
            "identity",
            "kind",
            "metrics",
            "metrics_sha256",
            "partition",
            "results",
            "schema_version",
            "world_size",
        },
        "causal world document",
    )
    if (
        document["schema_version"] != WORLD_SCHEMA_VERSION
        or document["kind"] != "causal_world"
        or document["dataset"] != DATASET
        or document["partition"] != PARTITION
    ):
        raise F2ProductionRunnerError("causal world schema or partition drifted")
    if document["fatal_count"] != 0 or isinstance(document["fatal_count"], bool):
        raise F2ProductionRunnerError("authoritative world fatal_count must be zero")
    if not isinstance(document["results"], list):
        raise F2ProductionRunnerError("causal world results must be a JSON array")
    identity = _parse_identity(document["identity"])
    results = tuple(_parse_token_result(row) for row in document["results"])
    metrics = _parse_metrics(document["metrics"])
    if metrics.fatal_count != 0:
        raise F2ProductionRunnerError("authoritative world metrics contain a fatal")
    try:
        evaluation = CausalWorldEvaluation(
            world_size=document["world_size"],
            checkpoint_sha256=identity["checkpoint_sha256"],
            checkpoint_epoch=identity["checkpoint_epoch"],
            manifest_sha256=identity["manifest_sha256"],
            artifact_sha256=identity["artifact_sha256"],
            config_sha256=identity["config_sha256"],
            source_sha256=identity["source_sha256"],
            results=results,
            metrics=metrics,
            numeric_mode=identity["numeric_mode"],
        )
    except (F2CausalShardRunnerError, CausalValidationError, TypeError) as error:
        raise F2ProductionRunnerError("causal world evaluation is invalid") from error
    if _sha256(document["evaluation_sha256"], "evaluation") != evaluation.evaluation_sha256:
        raise F2ProductionRunnerError("causal world evaluation SHA-256 mismatch")
    if _sha256(document["metrics_sha256"], "metrics") != evaluation.metrics_sha256:
        raise F2ProductionRunnerError("causal world metrics SHA-256 mismatch")
    recomputed = _recomputed_world_metrics(evaluation)
    if canonical_metrics_bytes(recomputed) != canonical_metrics_bytes(metrics):
        raise F2ProductionRunnerError("causal world metrics differ from token results")
    return evaluation


def write_world_evaluation(
    path: Path | str, evaluation: CausalWorldEvaluation
) -> str:
    return write_bytes_no_overwrite(path, world_evaluation_bytes(evaluation))


def read_world_evaluation(path: Path | str) -> CausalWorldEvaluation:
    payload, _ = read_bytes_with_sidecar(path)
    return parse_world_evaluation_bytes(payload)


@dataclass(frozen=True)
class ProductionShardReference:
    """One rank output and its deterministic runtime receipt.

    The file digests authenticate both persisted files.  The separate receipt
    identity digest makes the security-critical rank/world/run-root tuple an
    explicit part of the world certificate instead of relying on an opaque
    self-generated sidecar.
    """

    rank: int
    world_size: int
    shard_file_name: str
    shard_file_sha256: str
    shard_evaluation_sha256: str
    runtime_record_file_name: str
    runtime_record_file_sha256: str
    runtime_identity_sha256: str

    def __post_init__(self) -> None:
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise F2ProductionRunnerError(
                "production shard reference has unsupported world_size"
            )
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or not 0 <= self.rank < self.world_size
        ):
            raise F2ProductionRunnerError(
                "production shard reference rank is outside world_size"
            )
        for name in ("shard_file_name", "runtime_record_file_name"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or Path(value).name != value
                or value in (".", "..")
            ):
                raise F2ProductionRunnerError(
                    "production shard and receipt references must be direct siblings"
                )
        if self.shard_file_name == self.runtime_record_file_name:
            raise F2ProductionRunnerError(
                "production shard and receipt references must be distinct"
            )
        for name in (
            "shard_file_sha256",
            "shard_evaluation_sha256",
            "runtime_record_file_sha256",
            "runtime_identity_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))

    def to_payload(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "runtime_receipt": {
                "file_name": self.runtime_record_file_name,
                "file_sha256": self.runtime_record_file_sha256,
                "identity_sha256": self.runtime_identity_sha256,
            },
            "shard": {
                "evaluation_sha256": self.shard_evaluation_sha256,
                "file_name": self.shard_file_name,
                "file_sha256": self.shard_file_sha256,
            },
            "world_size": self.world_size,
        }


_PRODUCTION_WORLD_SEAL = object()


@dataclass(frozen=True)
class ProductionWorldEvaluation:
    """A world result rooted in signed authority and replayable rank receipts."""

    evaluation: CausalWorldEvaluation
    run_root_sha256: str
    shards: Tuple[ProductionShardReference, ...]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _PRODUCTION_WORLD_SEAL:
            raise F2ProductionRunnerError(
                "production worlds must come from signed run-root replay"
            )
        if not isinstance(self.evaluation, CausalWorldEvaluation):
            raise F2ProductionRunnerError("production world evaluation is malformed")
        object.__setattr__(
            self,
            "run_root_sha256",
            _sha256(self.run_root_sha256, "production run-root"),
        )
        shards = tuple(sorted(self.shards, key=lambda item: item.rank))
        if len(shards) != self.evaluation.world_size or tuple(
            item.rank for item in shards
        ) != tuple(range(self.evaluation.world_size)):
            raise F2ProductionRunnerError(
                "production world certificate rank coverage is incomplete"
            )
        if any(item.world_size != self.evaluation.world_size for item in shards):
            raise F2ProductionRunnerError(
                "production world certificate shard world_size drifted"
            )
        if len({item.shard_file_name for item in shards}) != len(shards) or len(
            {item.runtime_record_file_name for item in shards}
        ) != len(shards):
            raise F2ProductionRunnerError(
                "production world certificate contains duplicate rank files"
            )
        object.__setattr__(self, "shards", shards)

    @property
    def world_size(self) -> int:
        return self.evaluation.world_size

    @property
    def results(self) -> Tuple[TokenCausalResult, ...]:
        return self.evaluation.results

    @property
    def artifact_sha256(self) -> str:
        return self.evaluation.artifact_sha256


def production_world_evaluation_payload(
    certificate: ProductionWorldEvaluation,
) -> Dict[str, Any]:
    if type(certificate) is not ProductionWorldEvaluation:
        raise F2ProductionRunnerError("production world certificate is malformed")
    return {
        "dataset": DATASET,
        "kind": "run_root_causal_world",
        "partition": PARTITION,
        "run_root_sha256": certificate.run_root_sha256,
        "schema_version": RUN_ROOT_WORLD_SCHEMA_VERSION,
        "shards": [item.to_payload() for item in certificate.shards],
        "world_evaluation": world_evaluation_payload(certificate.evaluation),
    }


def production_world_evaluation_bytes(
    certificate: ProductionWorldEvaluation,
) -> bytes:
    return canonical_json_bytes(production_world_evaluation_payload(certificate))


def _parse_production_shard_reference(payload: Any) -> ProductionShardReference:
    value = _exact_keys(
        payload,
        {"rank", "runtime_receipt", "shard", "world_size"},
        "production shard reference",
    )
    shard = _exact_keys(
        value["shard"],
        {"evaluation_sha256", "file_name", "file_sha256"},
        "production shard file reference",
    )
    runtime = _exact_keys(
        value["runtime_receipt"],
        {"file_name", "file_sha256", "identity_sha256"},
        "production runtime receipt reference",
    )
    return ProductionShardReference(
        rank=value["rank"],
        world_size=value["world_size"],
        shard_file_name=shard["file_name"],
        shard_file_sha256=shard["file_sha256"],
        shard_evaluation_sha256=shard["evaluation_sha256"],
        runtime_record_file_name=runtime["file_name"],
        runtime_record_file_sha256=runtime["file_sha256"],
        runtime_identity_sha256=runtime["identity_sha256"],
    )


def _load_manifest(path: Path, expected_sha256: str) -> CanonicalManifest:
    if not path.is_file():
        raise F2ProductionRunnerError(f"causal manifest does not exist: {path}")
    try:
        payload = path.read_bytes()
        manifest = manifest_from_bytes(payload, expected_sha256=expected_sha256)
    except (OSError, CausalValidationError) as error:
        raise F2ProductionRunnerError("causal manifest authentication failed") from error
    if manifest.split != DATASET:
        raise F2ProductionRunnerError("causal manifest is not navtrain")
    return manifest


def merge_world_files(
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    world_size: int,
    shard_paths: Sequence[Path | str],
) -> CausalWorldEvaluation:
    """Read every rank exactly once and invoke the frozen world merger."""

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise F2ProductionRunnerError("world_size must be 1, 2, or 8")
    shards = tuple(read_shard_evaluation(path) for path in shard_paths)
    if len(shards) != world_size:
        raise F2ProductionRunnerError("world merge requires exactly one file per rank")
    ranks = [shard.rank for shard in shards]
    if sorted(ranks) != list(range(world_size)) or any(
        shard.world_size != world_size for shard in shards
    ):
        raise F2ProductionRunnerError("world merge rank coverage is incomplete or duplicated")
    expected_manifest = _sha256(
        expected_manifest_sha256, "expected causal manifest"
    )
    if any(shard.manifest_sha256 != expected_manifest for shard in shards):
        raise F2ProductionRunnerError(
            "shard causal manifest differs from external expectation"
        )
    manifest = _load_manifest(Path(manifest_path), expected_manifest)
    try:
        return merge_causal_world(manifest, shards)
    except (F2CausalShardRunnerError, CausalValidationError) as error:
        raise F2ProductionRunnerError("causal world merge failed") from error


@dataclass(frozen=True)
class WorldFileReference:
    world_size: int
    file_name: str
    file_sha256: str
    evaluation_sha256: str
    metrics_sha256: str

    def __post_init__(self) -> None:
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise F2ProductionRunnerError("world reference has unsupported world_size")
        if (
            not isinstance(self.file_name, str)
            or not self.file_name
            or Path(self.file_name).name != self.file_name
            or self.file_name in (".", "..")
        ):
            raise F2ProductionRunnerError("world reference must be a direct sibling file")
        for name in ("file_sha256", "evaluation_sha256", "metrics_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))

    def to_payload(self) -> Dict[str, Any]:
        return {
            "evaluation_sha256": self.evaluation_sha256,
            "file_name": self.file_name,
            "file_sha256": self.file_sha256,
            "metrics_sha256": self.metrics_sha256,
            "world_size": self.world_size,
        }


_INVARIANCE_SEAL = object()


@dataclass(frozen=True)
class InvariantCheckpointEvaluation:
    """One checkpoint backed by replayed, byte-exact 1/2/8 world files."""

    evaluation: CheckpointEvaluation
    config_sha256: str
    source_sha256: str
    evaluation_sha256: str
    metrics_sha256: str
    worlds: Tuple[WorldFileReference, ...]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _INVARIANCE_SEAL:
            raise F2ProductionRunnerError(
                "invariant checkpoints must come from a replayed 1/2/8 proof"
            )
        if not isinstance(self.evaluation, CheckpointEvaluation):
            raise F2ProductionRunnerError("checkpoint evaluation is malformed")
        object.__setattr__(self, "config_sha256", _sha256(self.config_sha256, "config"))
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "source"))
        object.__setattr__(
            self, "evaluation_sha256", _sha256(self.evaluation_sha256, "evaluation")
        )
        object.__setattr__(self, "metrics_sha256", _sha256(self.metrics_sha256, "metrics"))
        worlds = tuple(sorted(self.worlds, key=lambda item: item.world_size))
        if tuple(item.world_size for item in worlds) != SUPPORTED_WORLD_SIZES:
            raise F2ProductionRunnerError("invariance proof must bind worlds 1, 2, and 8")
        if len({item.file_name for item in worlds}) != 3:
            raise F2ProductionRunnerError("invariance proof world files must be unique")
        if any(item.evaluation_sha256 != self.evaluation_sha256 for item in worlds):
            raise F2ProductionRunnerError("invariance proof evaluation digests differ")
        if any(item.metrics_sha256 != self.metrics_sha256 for item in worlds):
            raise F2ProductionRunnerError("invariance proof metric digests differ")
        if self.metrics_sha256 != _sha256_bytes(
            canonical_metrics_bytes(self.evaluation.metrics)
        ):
            raise F2ProductionRunnerError("checkpoint metrics digest is inconsistent")
        object.__setattr__(self, "worlds", worlds)


def _checkpoint_payload(evaluation: CheckpointEvaluation) -> Dict[str, Any]:
    return {
        "artifact_sha256": evaluation.artifact_sha256,
        "checkpoint_sha256": evaluation.checkpoint_sha256,
        "epoch": evaluation.epoch,
        "manifest_sha256": evaluation.manifest_sha256,
        "metrics": evaluation.metrics.to_payload(),
        "numeric_mode": evaluation.numeric_mode,
    }


def _parse_checkpoint_payload(payload: Any) -> CheckpointEvaluation:
    value = _exact_keys(
        payload,
        {
            "artifact_sha256",
            "checkpoint_sha256",
            "epoch",
            "manifest_sha256",
            "metrics",
            "numeric_mode",
        },
        "checkpoint evaluation",
    )
    try:
        return CheckpointEvaluation(
            epoch=value["epoch"],
            checkpoint_sha256=value["checkpoint_sha256"],
            manifest_sha256=value["manifest_sha256"],
            artifact_sha256=value["artifact_sha256"],
            numeric_mode=value["numeric_mode"],
            metrics=_parse_metrics(value["metrics"]),
        )
    except (CausalValidationError, TypeError) as error:
        raise F2ProductionRunnerError("checkpoint evaluation is invalid") from error


def checkpoint_evaluation_payload(
    proof: InvariantCheckpointEvaluation,
) -> Dict[str, Any]:
    if not isinstance(proof, InvariantCheckpointEvaluation):
        raise F2ProductionRunnerError("checkpoint proof is malformed")
    return {
        "checkpoint_evaluation": _checkpoint_payload(proof.evaluation),
        "dataset": DATASET,
        "evaluation_sha256": proof.evaluation_sha256,
        "identity": {
            "artifact_sha256": proof.evaluation.artifact_sha256,
            "checkpoint_epoch": proof.evaluation.epoch,
            "checkpoint_sha256": proof.evaluation.checkpoint_sha256,
            "config_sha256": proof.config_sha256,
            "manifest_sha256": proof.evaluation.manifest_sha256,
            "numeric_mode": proof.evaluation.numeric_mode,
            "source_sha256": proof.source_sha256,
        },
        "invariance_proved": True,
        "kind": "invariant_checkpoint",
        "metrics_sha256": proof.metrics_sha256,
        "partition": PARTITION,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "world_size": 1,
        "worlds": [item.to_payload() for item in proof.worlds],
    }


def checkpoint_evaluation_bytes(proof: InvariantCheckpointEvaluation) -> bytes:
    return canonical_json_bytes(checkpoint_evaluation_payload(proof))


def _require_sibling(path: Path, output_parent: Path, name: str) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.parent != output_parent.resolve(strict=True):
        raise F2ProductionRunnerError(f"{name} must be a direct sibling of the output")
    return resolved


def prove_invariance_files(
    *,
    world_paths: Sequence[Path | str],
    output_path: Path | str,
) -> InvariantCheckpointEvaluation:
    """Replay worlds 1/2/8 and publish a self-contained sibling certificate."""

    output = Path(output_path)
    if not output.parent.is_dir():
        raise F2ProductionRunnerError("invariance output parent does not exist")
    if len(world_paths) != 3:
        raise F2ProductionRunnerError("invariance proof requires three world files")
    evaluations = []
    references = []
    for raw_path in world_paths:
        path = _require_sibling(Path(raw_path), output.parent, "world file")
        payload, digest = read_bytes_with_sidecar(path)
        evaluation = parse_world_evaluation_bytes(payload)
        evaluations.append(evaluation)
        references.append(
            WorldFileReference(
                world_size=evaluation.world_size,
                file_name=path.name,
                file_sha256=digest,
                evaluation_sha256=evaluation.evaluation_sha256,
                metrics_sha256=evaluation.metrics_sha256,
            )
        )
    try:
        world1 = prove_1_2_8_invariance(evaluations)
    except (F2CausalShardRunnerError, CausalValidationError) as error:
        raise F2ProductionRunnerError("1/2/8 causal invariance proof failed") from error
    proof = InvariantCheckpointEvaluation(
        evaluation=world1.checkpoint_evaluation,
        config_sha256=world1.config_sha256,
        source_sha256=world1.source_sha256,
        evaluation_sha256=world1.evaluation_sha256,
        metrics_sha256=world1.metrics_sha256,
        worlds=tuple(references),
        _seal=_INVARIANCE_SEAL,
    )
    write_bytes_no_overwrite(output, checkpoint_evaluation_bytes(proof))
    return proof


def _parse_world_reference(payload: Any) -> WorldFileReference:
    value = _exact_keys(
        payload,
        {
            "evaluation_sha256",
            "file_name",
            "file_sha256",
            "metrics_sha256",
            "world_size",
        },
        "world file reference",
    )
    return WorldFileReference(
        world_size=value["world_size"],
        file_name=value["file_name"],
        file_sha256=value["file_sha256"],
        evaluation_sha256=value["evaluation_sha256"],
        metrics_sha256=value["metrics_sha256"],
    )


def read_checkpoint_evaluation(
    path: Path | str,
) -> InvariantCheckpointEvaluation:
    """Read a certificate and re-run its referenced invariance proof."""

    certificate_path = Path(path).resolve(strict=True)
    decoded, _, _ = _read_canonical_file(
        certificate_path, name="invariant checkpoint certificate"
    )
    document = _exact_keys(
        decoded,
        {
            "checkpoint_evaluation",
            "dataset",
            "evaluation_sha256",
            "identity",
            "invariance_proved",
            "kind",
            "metrics_sha256",
            "partition",
            "schema_version",
            "world_size",
            "worlds",
        },
        "invariant checkpoint document",
    )
    if (
        document["schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or document["kind"] != "invariant_checkpoint"
        or document["dataset"] != DATASET
        or document["partition"] != PARTITION
        or document["invariance_proved"] is not True
        or document["world_size"] != 1
    ):
        raise F2ProductionRunnerError("checkpoint certificate contract drifted")
    identity = _parse_identity(document["identity"])
    evaluation = _parse_checkpoint_payload(document["checkpoint_evaluation"])
    if (
        identity["checkpoint_epoch"] != evaluation.epoch
        or identity["checkpoint_sha256"] != evaluation.checkpoint_sha256
        or identity["manifest_sha256"] != evaluation.manifest_sha256
        or identity["artifact_sha256"] != evaluation.artifact_sha256
        or identity["numeric_mode"] != evaluation.numeric_mode
    ):
        raise F2ProductionRunnerError("checkpoint certificate identity is inconsistent")
    if not isinstance(document["worlds"], list):
        raise F2ProductionRunnerError("checkpoint worlds must be a JSON array")
    references = tuple(_parse_world_reference(item) for item in document["worlds"])
    worlds = []
    for reference in references:
        world_path = certificate_path.parent / reference.file_name
        payload, digest = read_bytes_with_sidecar(world_path)
        if digest != reference.file_sha256:
            raise F2ProductionRunnerError("referenced world file SHA-256 drifted")
        world = parse_world_evaluation_bytes(payload)
        if (
            world.world_size != reference.world_size
            or world.evaluation_sha256 != reference.evaluation_sha256
            or world.metrics_sha256 != reference.metrics_sha256
        ):
            raise F2ProductionRunnerError("referenced world identity drifted")
        worlds.append(world)
    try:
        world1 = prove_1_2_8_invariance(worlds)
    except (F2CausalShardRunnerError, CausalValidationError) as error:
        raise F2ProductionRunnerError("checkpoint invariance replay failed") from error
    if (
        world1.checkpoint_evaluation != evaluation
        or world1.config_sha256 != identity["config_sha256"]
        or world1.source_sha256 != identity["source_sha256"]
        or world1.evaluation_sha256
        != _sha256(document["evaluation_sha256"], "evaluation")
        or world1.metrics_sha256 != _sha256(document["metrics_sha256"], "metrics")
    ):
        raise F2ProductionRunnerError("checkpoint certificate differs from replayed worlds")
    return InvariantCheckpointEvaluation(
        evaluation=evaluation,
        config_sha256=identity["config_sha256"],
        source_sha256=identity["source_sha256"],
        evaluation_sha256=world1.evaluation_sha256,
        metrics_sha256=world1.metrics_sha256,
        worlds=references,
        _seal=_INVARIANCE_SEAL,
    )


@dataclass(frozen=True)
class ProductionWorldFileReference:
    world_size: int
    file_name: str
    file_sha256: str
    run_root_sha256: str
    evaluation_sha256: str
    metrics_sha256: str

    def __post_init__(self) -> None:
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise F2ProductionRunnerError(
                "production proof world reference has unsupported world_size"
            )
        if (
            not isinstance(self.file_name, str)
            or not self.file_name
            or Path(self.file_name).name != self.file_name
            or self.file_name in (".", "..")
        ):
            raise F2ProductionRunnerError(
                "production proof world must be a direct sibling file"
            )
        for name in (
            "file_sha256",
            "run_root_sha256",
            "evaluation_sha256",
            "metrics_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))

    def to_payload(self) -> Dict[str, Any]:
        return {
            "evaluation_sha256": self.evaluation_sha256,
            "file_name": self.file_name,
            "file_sha256": self.file_sha256,
            "metrics_sha256": self.metrics_sha256,
            "run_root_sha256": self.run_root_sha256,
            "world_size": self.world_size,
        }


_PRODUCTION_INVARIANCE_SEAL = object()


@dataclass(frozen=True)
class ProductionInvariantCheckpointEvaluation:
    proof: InvariantCheckpointEvaluation
    run_root_sha256: str
    worlds: Tuple[ProductionWorldFileReference, ...]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _PRODUCTION_INVARIANCE_SEAL:
            raise F2ProductionRunnerError(
                "production invariance proof must come from recursive world replay"
            )
        if (
            not isinstance(self.proof, InvariantCheckpointEvaluation)
            or self.proof._seal is not _INVARIANCE_SEAL
        ):
            raise F2ProductionRunnerError(
                "production invariance proof contains an untrusted inner proof"
            )
        object.__setattr__(
            self,
            "run_root_sha256",
            _sha256(self.run_root_sha256, "production run-root"),
        )
        worlds = tuple(sorted(self.worlds, key=lambda item: item.world_size))
        if tuple(item.world_size for item in worlds) != SUPPORTED_WORLD_SIZES:
            raise F2ProductionRunnerError(
                "production invariance proof must bind worlds 1, 2, and 8"
            )
        if any(item.run_root_sha256 != self.run_root_sha256 for item in worlds):
            raise F2ProductionRunnerError(
                "production invariance world run-root identity drifted"
            )
        if any(
            item.evaluation_sha256 != self.proof.evaluation_sha256
            or item.metrics_sha256 != self.proof.metrics_sha256
            for item in worlds
        ):
            raise F2ProductionRunnerError(
                "production invariance world evaluation identity drifted"
            )
        object.__setattr__(self, "worlds", worlds)

    @property
    def evaluation(self) -> CheckpointEvaluation:
        return self.proof.evaluation

    @property
    def config_sha256(self) -> str:
        return self.proof.config_sha256

    @property
    def source_sha256(self) -> str:
        return self.proof.source_sha256

    @property
    def evaluation_sha256(self) -> str:
        return self.proof.evaluation_sha256

    @property
    def metrics_sha256(self) -> str:
        return self.proof.metrics_sha256


def production_checkpoint_evaluation_payload(
    certificate: ProductionInvariantCheckpointEvaluation,
) -> Dict[str, Any]:
    if type(certificate) is not ProductionInvariantCheckpointEvaluation:
        raise F2ProductionRunnerError(
            "production checkpoint certificate is malformed"
        )
    proof = certificate.proof
    return {
        "checkpoint_evaluation": _checkpoint_payload(proof.evaluation),
        "dataset": DATASET,
        "evaluation_sha256": proof.evaluation_sha256,
        "identity": {
            "artifact_sha256": proof.evaluation.artifact_sha256,
            "checkpoint_epoch": proof.evaluation.epoch,
            "checkpoint_sha256": proof.evaluation.checkpoint_sha256,
            "config_sha256": proof.config_sha256,
            "manifest_sha256": proof.evaluation.manifest_sha256,
            "numeric_mode": proof.evaluation.numeric_mode,
            "source_sha256": proof.source_sha256,
        },
        "invariance_proved": True,
        "kind": "run_root_invariant_checkpoint",
        "metrics_sha256": proof.metrics_sha256,
        "partition": PARTITION,
        "run_root_sha256": certificate.run_root_sha256,
        "schema_version": RUN_ROOT_CHECKPOINT_SCHEMA_VERSION,
        "worlds": [item.to_payload() for item in certificate.worlds],
    }


def production_checkpoint_evaluation_bytes(
    certificate: ProductionInvariantCheckpointEvaluation,
) -> bytes:
    return canonical_json_bytes(production_checkpoint_evaluation_payload(certificate))


def _parse_production_world_reference(payload: Any) -> ProductionWorldFileReference:
    value = _exact_keys(
        payload,
        {
            "evaluation_sha256",
            "file_name",
            "file_sha256",
            "metrics_sha256",
            "run_root_sha256",
            "world_size",
        },
        "production proof world reference",
    )
    return ProductionWorldFileReference(
        world_size=value["world_size"],
        file_name=value["file_name"],
        file_sha256=value["file_sha256"],
        run_root_sha256=value["run_root_sha256"],
        evaluation_sha256=value["evaluation_sha256"],
        metrics_sha256=value["metrics_sha256"],
    )


_RANKING_SEAL = object()


@dataclass(frozen=True)
class CheckpointFileReference:
    epoch: int
    file_name: str
    file_sha256: str
    checkpoint_sha256: str
    evaluation_sha256: str
    metrics_sha256: str

    def __post_init__(self) -> None:
        if self.epoch not in F2_CHECKPOINT_EPOCHS:
            raise F2ProductionRunnerError("checkpoint reference epoch is outside F2")
        if (
            not isinstance(self.file_name, str)
            or not self.file_name
            or Path(self.file_name).name != self.file_name
        ):
            raise F2ProductionRunnerError(
                "checkpoint reference must be a direct sibling file"
            )
        for name in (
            "file_sha256",
            "checkpoint_sha256",
            "evaluation_sha256",
            "metrics_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))

    def to_payload(self) -> Dict[str, Any]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "epoch": self.epoch,
            "evaluation_sha256": self.evaluation_sha256,
            "file_name": self.file_name,
            "file_sha256": self.file_sha256,
            "metrics_sha256": self.metrics_sha256,
        }


@dataclass(frozen=True)
class ProductionCheckpointRanking:
    ranking: CheckpointRanking
    config_sha256: str
    source_sha256: str
    manifest_sha256: str
    artifact_sha256: str
    inputs: Tuple[CheckpointFileReference, ...]
    evaluation_sha256: str
    metrics_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _RANKING_SEAL:
            raise F2ProductionRunnerError("ranking must come from invariant checkpoints")
        if not isinstance(self.ranking, CheckpointRanking):
            raise F2ProductionRunnerError("ranking result is malformed")
        for name in (
            "config_sha256",
            "source_sha256",
            "manifest_sha256",
            "artifact_sha256",
            "evaluation_sha256",
            "metrics_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        inputs = tuple(sorted(self.inputs, key=lambda item: item.epoch))
        if tuple(item.epoch for item in inputs) != F2_CHECKPOINT_EPOCHS:
            raise F2ProductionRunnerError("ranking input epochs must be exactly 0..3")
        object.__setattr__(self, "inputs", inputs)


def _checkpoint_set_digests(
    evaluations: Sequence[CheckpointEvaluation],
) -> Tuple[str, str]:
    ordered = tuple(sorted(evaluations, key=lambda item: item.epoch))
    evaluation_bytes = canonical_json_bytes(
        [
            {
                "artifact_sha256": item.artifact_sha256,
                "checkpoint_sha256": item.checkpoint_sha256,
                "epoch": item.epoch,
                "manifest_sha256": item.manifest_sha256,
                "numeric_mode": item.numeric_mode,
            }
            for item in ordered
        ]
    )
    metrics_bytes = canonical_json_bytes(
        [item.metrics.to_payload() for item in ordered]
    )
    return _sha256_bytes(evaluation_bytes), _sha256_bytes(metrics_bytes)


def rank_invariant_checkpoints(
    proofs: Sequence[InvariantCheckpointEvaluation],
    *,
    references: Sequence[CheckpointFileReference],
) -> ProductionCheckpointRanking:
    if len(proofs) != 4 or len(references) != 4:
        raise F2ProductionRunnerError("ranking requires four invariant checkpoints")
    if any(
        not isinstance(proof, InvariantCheckpointEvaluation)
        or proof._seal is not _INVARIANCE_SEAL
        for proof in proofs
    ):
        raise F2ProductionRunnerError("ranking accepts invariant checkpoints only")
    by_epoch = {proof.evaluation.epoch: proof for proof in proofs}
    if len(by_epoch) != 4 or tuple(sorted(by_epoch)) != F2_CHECKPOINT_EPOCHS:
        raise F2ProductionRunnerError("ranking epochs must be exactly 0, 1, 2, and 3")
    ordered = tuple(by_epoch[epoch] for epoch in F2_CHECKPOINT_EPOCHS)
    common_fields = (
        "config_sha256",
        "source_sha256",
    )
    for field_name in common_fields:
        if len({getattr(proof, field_name) for proof in ordered}) != 1:
            raise F2ProductionRunnerError(
                f"checkpoint invariant {field_name} drifted across epochs"
            )
    if len({proof.evaluation.manifest_sha256 for proof in ordered}) != 1:
        raise F2ProductionRunnerError("checkpoint manifest identity drifted")
    if len({proof.evaluation.artifact_sha256 for proof in ordered}) != 1:
        raise F2ProductionRunnerError("checkpoint artifact identity drifted")
    try:
        ranking = rank_checkpoints([proof.evaluation for proof in ordered])
    except CausalValidationError as error:
        raise F2ProductionRunnerError("formal checkpoint ranking failed") from error
    evaluation_sha, metrics_sha = _checkpoint_set_digests(
        [proof.evaluation for proof in ordered]
    )
    return ProductionCheckpointRanking(
        ranking=ranking,
        config_sha256=ordered[0].config_sha256,
        source_sha256=ordered[0].source_sha256,
        manifest_sha256=ordered[0].evaluation.manifest_sha256,
        artifact_sha256=ordered[0].evaluation.artifact_sha256,
        inputs=tuple(references),
        evaluation_sha256=evaluation_sha,
        metrics_sha256=metrics_sha,
        _seal=_RANKING_SEAL,
    )


def _ranked_evaluation_payload(
    evaluation: CheckpointEvaluation,
    decision: GateDecision,
) -> Dict[str, Any]:
    return {
        "evaluation": _checkpoint_payload(evaluation),
        "gate": decision.to_payload(),
        "metrics_sha256": _sha256_bytes(canonical_metrics_bytes(evaluation.metrics)),
    }


def checkpoint_ranking_payload(
    result: ProductionCheckpointRanking,
) -> Dict[str, Any]:
    if not isinstance(result, ProductionCheckpointRanking):
        raise F2ProductionRunnerError("production ranking is malformed")
    ranking = result.ranking
    selected = ranking.selected
    return {
        "dataset": DATASET,
        "evaluation_sha256": result.evaluation_sha256,
        "identity": {
            "artifact_sha256": result.artifact_sha256,
            "config_sha256": result.config_sha256,
            "manifest_sha256": result.manifest_sha256,
            "numeric_mode": F2_MODEL_NUMERIC_MODE,
            "source_sha256": result.source_sha256,
        },
        "inputs": [item.to_payload() for item in result.inputs],
        "kind": "checkpoint_ranking",
        "metrics_sha256": result.metrics_sha256,
        "partition": PARTITION,
        "ranking": {
            "eligible": [
                _ranked_evaluation_payload(item, evaluate_formal_gates(item.metrics))
                for item in ranking.eligible
            ],
            "ineligible": [
                _ranked_evaluation_payload(item, decision)
                for item, decision in ranking.ineligible
            ],
            "selected": (
                {
                    "checkpoint_sha256": selected.checkpoint_sha256,
                    "epoch": selected.epoch,
                }
                if selected is not None
                else None
            ),
        },
        "schema_version": RANKING_SCHEMA_VERSION,
    }


def checkpoint_ranking_bytes(result: ProductionCheckpointRanking) -> bytes:
    return canonical_json_bytes(checkpoint_ranking_payload(result))


def production_checkpoint_ranking_payload(
    result: ProductionCheckpointRanking,
    *,
    run_root_sha256: str,
) -> Dict[str, Any]:
    return {
        "dataset": DATASET,
        "kind": "run_root_checkpoint_ranking",
        "partition": PARTITION,
        "ranking": checkpoint_ranking_payload(result),
        "run_root_sha256": _sha256(run_root_sha256, "production run-root"),
        "schema_version": RUN_ROOT_RANKING_SCHEMA_VERSION,
    }


def production_checkpoint_ranking_bytes(
    result: ProductionCheckpointRanking,
    *,
    run_root_sha256: str,
) -> bytes:
    return canonical_json_bytes(
        production_checkpoint_ranking_payload(
            result,
            run_root_sha256=run_root_sha256,
        )
    )


def write_checkpoint_ranking(
    *,
    checkpoint_paths: Sequence[Path | str],
    output_path: Path | str,
) -> ProductionCheckpointRanking:
    output = Path(output_path)
    if not output.parent.is_dir():
        raise F2ProductionRunnerError("ranking output parent does not exist")
    if len(checkpoint_paths) != 4:
        raise F2ProductionRunnerError("ranking requires four checkpoint certificates")
    proofs = []
    references = []
    for raw_path in checkpoint_paths:
        path = _require_sibling(Path(raw_path), output.parent, "checkpoint certificate")
        _, digest = read_bytes_with_sidecar(path)
        proof = read_checkpoint_evaluation(path)
        proofs.append(proof)
        references.append(
            CheckpointFileReference(
                epoch=proof.evaluation.epoch,
                file_name=path.name,
                file_sha256=digest,
                checkpoint_sha256=proof.evaluation.checkpoint_sha256,
                evaluation_sha256=proof.evaluation_sha256,
                metrics_sha256=proof.metrics_sha256,
            )
        )
    result = rank_invariant_checkpoints(proofs, references=references)
    write_bytes_no_overwrite(output, checkpoint_ranking_bytes(result))
    return result


def _parse_checkpoint_reference(payload: Any) -> CheckpointFileReference:
    value = _exact_keys(
        payload,
        {
            "checkpoint_sha256",
            "epoch",
            "evaluation_sha256",
            "file_name",
            "file_sha256",
            "metrics_sha256",
        },
        "checkpoint file reference",
    )
    return CheckpointFileReference(
        epoch=value["epoch"],
        file_name=value["file_name"],
        file_sha256=value["file_sha256"],
        checkpoint_sha256=value["checkpoint_sha256"],
        evaluation_sha256=value["evaluation_sha256"],
        metrics_sha256=value["metrics_sha256"],
    )


def read_checkpoint_ranking(path: Path | str) -> ProductionCheckpointRanking:
    """Read ranking output and replay all four invariant certificates."""

    ranking_path = Path(path).resolve(strict=True)
    decoded, payload, _ = _read_canonical_file(
        ranking_path, name="checkpoint ranking"
    )
    document = _exact_keys(
        decoded,
        {
            "dataset",
            "evaluation_sha256",
            "identity",
            "inputs",
            "kind",
            "metrics_sha256",
            "partition",
            "ranking",
            "schema_version",
        },
        "checkpoint ranking document",
    )
    if (
        document["schema_version"] != RANKING_SCHEMA_VERSION
        or document["kind"] != "checkpoint_ranking"
        or document["dataset"] != DATASET
        or document["partition"] != PARTITION
    ):
        raise F2ProductionRunnerError("checkpoint ranking contract drifted")
    identity = _exact_keys(
        document["identity"],
        {
            "artifact_sha256",
            "config_sha256",
            "manifest_sha256",
            "numeric_mode",
            "source_sha256",
        },
        "ranking identity",
    )
    if identity["numeric_mode"] != F2_MODEL_NUMERIC_MODE:
        raise F2ProductionRunnerError("ranking numeric mode drifted")
    if not isinstance(document["inputs"], list):
        raise F2ProductionRunnerError("ranking inputs must be a JSON array")
    references = tuple(
        _parse_checkpoint_reference(item) for item in document["inputs"]
    )
    proofs = []
    for reference in references:
        certificate_path = ranking_path.parent / reference.file_name
        _, digest = read_bytes_with_sidecar(certificate_path)
        if digest != reference.file_sha256:
            raise F2ProductionRunnerError("ranking checkpoint file SHA-256 drifted")
        proof = read_checkpoint_evaluation(certificate_path)
        if (
            proof.evaluation.epoch != reference.epoch
            or proof.evaluation.checkpoint_sha256 != reference.checkpoint_sha256
            or proof.evaluation_sha256 != reference.evaluation_sha256
            or proof.metrics_sha256 != reference.metrics_sha256
        ):
            raise F2ProductionRunnerError("ranking checkpoint reference drifted")
        proofs.append(proof)
    result = rank_invariant_checkpoints(proofs, references=references)
    if (
        result.artifact_sha256 != _sha256(identity["artifact_sha256"], "artifact")
        or result.config_sha256 != _sha256(identity["config_sha256"], "config")
        or result.manifest_sha256 != _sha256(identity["manifest_sha256"], "manifest")
        or result.source_sha256 != _sha256(identity["source_sha256"], "source")
        or result.evaluation_sha256
        != _sha256(document["evaluation_sha256"], "evaluation")
        or result.metrics_sha256 != _sha256(document["metrics_sha256"], "metrics")
    ):
        raise F2ProductionRunnerError("ranking identity differs from replayed inputs")
    if checkpoint_ranking_bytes(result) != payload:
        raise F2ProductionRunnerError("ranking payload differs from replayed inputs")
    return result


def artifact_bindings_payload(bindings: ArtifactBindings) -> Dict[str, Any]:
    if type(bindings) is not ArtifactBindings:
        raise F2ProductionRunnerError("bindings must be ArtifactBindings")
    return {
        "bindings": bindings.to_payload(),
        "schema_version": BINDINGS_SCHEMA_VERSION,
    }


def artifact_bindings_bytes(bindings: ArtifactBindings) -> bytes:
    return canonical_json_bytes(artifact_bindings_payload(bindings))


def _parse_artifact_bindings_payload(payload: Any) -> ArtifactBindings:
    document = _exact_keys(
        payload, {"bindings", "schema_version"}, "artifact bindings document"
    )
    if document["schema_version"] != BINDINGS_SCHEMA_VERSION:
        raise F2ProductionRunnerError("unsupported artifact bindings schema")
    value = _exact_keys(
        document["bindings"],
        {
            "baseline_checkpoint_sha256",
            "vjepa_checkpoint_sha256",
            "cache_metadata_sha256",
            "metric_cache_manifest_sha256",
            "raw_input_manifest_sha256",
            "channel_order",
            "manifests",
            "proposal_schema",
            "scorer",
            "simulator",
            "source",
        },
        "artifact bindings",
    )
    manifests = _exact_keys(
        value["manifests"], {"causal_sha256", "qualification_sha256"}, "manifests"
    )
    proposal = _exact_keys(
        value["proposal_schema"], {"candidate_count", "name", "sha256"}, "proposal schema"
    )
    scorer = _exact_keys(value["scorer"], {"name", "sha256"}, "scorer")
    simulator = _exact_keys(value["simulator"], {"name", "sha256"}, "simulator")
    source = _exact_keys(
        value["source"],
        {
            "qualification_config_sha256",
            "qualification_source_manifest_sha256",
            "revision",
            "runtime_config_sha256",
            "runtime_manifest_sha256",
        },
        "source identity",
    )
    if proposal["candidate_count"] != 32 or isinstance(
        proposal["candidate_count"], bool
    ):
        raise F2ProductionRunnerError("proposal candidate count drifted")
    if not isinstance(value["channel_order"], list):
        raise F2ProductionRunnerError("channel order must be a JSON array")
    try:
        return ArtifactBindings(
            qualification_manifest_sha256=_sha256(
                manifests["qualification_sha256"], "qualification manifest"
            ),
            causal_manifest_sha256=_sha256(
                manifests["causal_sha256"], "causal manifest"
            ),
            baseline_checkpoint_sha256=_sha256(
                value["baseline_checkpoint_sha256"], "baseline checkpoint"
            ),
            vjepa_checkpoint_sha256=_sha256(
                value["vjepa_checkpoint_sha256"], "V-JEPA checkpoint"
            ),
            cache_metadata_sha256=_sha256(
                value["cache_metadata_sha256"], "cache metadata"
            ),
            metric_cache_manifest_sha256=_sha256(
                value["metric_cache_manifest_sha256"],
                "metric-cache content manifest",
            ),
            raw_input_manifest_sha256=_sha256(
                value["raw_input_manifest_sha256"],
                "raw-input content manifest",
            ),
            qualification_config_sha256=_sha256(
                source["qualification_config_sha256"], "qualification config"
            ),
            runtime_config_sha256=_sha256(
                source["runtime_config_sha256"], "runtime config"
            ),
            qualification_source_manifest_sha256=_sha256(
                source["qualification_source_manifest_sha256"],
                "qualification source manifest",
            ),
            runtime_source_manifest_sha256=_sha256(
                source["runtime_manifest_sha256"], "runtime source manifest"
            ),
            source_revision=source["revision"],
            proposal_schema_name=proposal["name"],
            proposal_schema_sha256=_sha256(
                proposal["sha256"], "proposal schema"
            ),
            simulator_name=simulator["name"],
            simulator_sha256=_sha256(simulator["sha256"], "simulator"),
            scorer_name=scorer["name"],
            scorer_sha256=_sha256(scorer["sha256"], "scorer"),
            channel_order=tuple(value["channel_order"]),
        )
    except (TypeError, ValueError) as error:
        raise F2ProductionRunnerError("artifact bindings are malformed") from error


def write_artifact_bindings(path: Path | str, bindings: ArtifactBindings) -> str:
    return write_bytes_no_overwrite(path, artifact_bindings_bytes(bindings))


def read_artifact_bindings(path: Path | str) -> ArtifactBindings:
    decoded, payload, _ = _read_canonical_file(path, name="artifact bindings")
    bindings = _parse_artifact_bindings_payload(decoded)
    if artifact_bindings_bytes(bindings) != payload:
        raise F2ProductionRunnerError(
            "artifact bindings are not idempotently canonical"
        )
    return bindings


@dataclass(frozen=True)
class RuntimeCheckpoint:
    epoch: int
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if self.epoch not in F2_CHECKPOINT_EPOCHS:
            raise F2ProductionRunnerError("runtime checkpoint epoch is outside F2")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise F2ProductionRunnerError("runtime checkpoint path must be absolute")
        object.__setattr__(self, "sha256", _sha256(self.sha256, "checkpoint"))


@dataclass(frozen=True)
class ProductionRuntimeConfig:
    paths: Tuple[Tuple[str, Path], ...]
    checkpoints: Tuple[RuntimeCheckpoint, ...]
    reader_mode: str
    device_policy: str

    def __post_init__(self) -> None:
        path_map = dict(self.paths)
        if set(path_map) != _RUNTIME_PATH_KEYS or len(path_map) != len(self.paths):
            raise F2ProductionRunnerError("runtime config path keys drifted")
        if any(not path.is_absolute() for path in path_map.values()):
            raise F2ProductionRunnerError("all runtime config paths must be absolute")
        checkpoints = tuple(sorted(self.checkpoints, key=lambda item: item.epoch))
        if tuple(item.epoch for item in checkpoints) != F2_CHECKPOINT_EPOCHS:
            raise F2ProductionRunnerError("runtime config checkpoints must be epochs 0..3")
        if len({item.sha256 for item in checkpoints}) != 4:
            raise F2ProductionRunnerError("runtime checkpoint identities must be unique")
        if self.reader_mode not in ("stream", "mmap"):
            raise F2ProductionRunnerError("runtime reader_mode must be stream or mmap")
        if self.device_policy != DEVICE_POLICY:
            raise F2ProductionRunnerError(
                f"production device policy must be {DEVICE_POLICY}"
            )
        object.__setattr__(self, "paths", tuple(sorted(path_map.items())))
        object.__setattr__(self, "checkpoints", checkpoints)

    @property
    def path_map(self) -> Mapping[str, Path]:
        return dict(self.paths)

    def checkpoint(self, epoch: int) -> RuntimeCheckpoint:
        if epoch not in F2_CHECKPOINT_EPOCHS:
            raise F2ProductionRunnerError("requested checkpoint epoch is outside F2")
        return self.checkpoints[epoch]


def _parse_runtime_config(
    path: Path, *, expected_sha256: str
) -> ProductionRuntimeConfig:
    payload = _read_stable_regular_file(path, name="production runtime config")
    if _sha256_bytes(payload) != _sha256(expected_sha256, "runtime config"):
        raise F2ProductionRunnerError("runtime config SHA-256 differs from bindings")
    decoded = parse_canonical_json_bytes(payload, name="production runtime config")
    document = _exact_keys(
        decoded,
        {
            "checkpoints",
            "dataset",
            "device_policy",
            "partition",
            "paths",
            "reader_mode",
            "schema_version",
        },
        "production runtime config",
    )
    if (
        document["schema_version"] != RUNTIME_CONFIG_SCHEMA_VERSION
        or document["dataset"] != DATASET
        or document["partition"] != PARTITION
    ):
        raise F2ProductionRunnerError("runtime config schema or partition drifted")
    paths = _exact_keys(document["paths"], _RUNTIME_PATH_KEYS, "runtime paths")
    converted_paths = []
    for name, raw_path in paths.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise F2ProductionRunnerError(f"runtime path {name!r} must be a string")
        converted_paths.append((name, Path(raw_path)))
    if not isinstance(document["checkpoints"], list):
        raise F2ProductionRunnerError("runtime checkpoints must be a JSON array")
    checkpoints = []
    for raw_checkpoint in document["checkpoints"]:
        checkpoint = _exact_keys(
            raw_checkpoint, {"epoch", "path", "sha256"}, "runtime checkpoint"
        )
        if not isinstance(checkpoint["path"], str) or not checkpoint["path"]:
            raise F2ProductionRunnerError("runtime checkpoint path must be a string")
        checkpoints.append(
            RuntimeCheckpoint(
                epoch=checkpoint["epoch"],
                path=Path(checkpoint["path"]),
                sha256=checkpoint["sha256"],
            )
        )
    return ProductionRuntimeConfig(
        paths=tuple(converted_paths),
        checkpoints=tuple(checkpoints),
        reader_mode=document["reader_mode"],
        device_policy=document["device_policy"],
    )


@dataclass(frozen=True)
class RunRootCheckpoint:
    """Externally authorized model identity for one formal F2 epoch."""

    epoch: int
    sha256: str

    def __post_init__(self) -> None:
        if self.epoch not in F2_CHECKPOINT_EPOCHS:
            raise F2ProductionRunnerError("run-root checkpoint epoch is outside F2")
        object.__setattr__(self, "sha256", _sha256(self.sha256, "checkpoint"))

    def to_payload(self) -> Dict[str, Any]:
        return {"epoch": self.epoch, "sha256": self.sha256}


def _absolute_authorized_path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise F2ProductionRunnerError(f"{name} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise F2ProductionRunnerError(f"{name} must be an absolute path")
    return path


@dataclass(frozen=True)
class ProductionRunRoot:
    """Canonical identity root authorized outside the production CLI.

    The root is deliberately small.  An offline-authority SSHSIG over its exact
    canonical bytes binds the complete bindings document, runtime
    configuration, raw-input manifest, causal row/log counts, merged-artifact
    certificate and binary root, plus all four model checkpoints.  Paths are
    data inside that authenticated document; a caller cannot replace an input
    by forwarding a matching digest from the input itself.
    """

    bindings_path: Path
    bindings_sha256: str
    launch_authorization_path: Path
    launch_authorization_sha256: str
    launch_authorization_signature_path: Path
    launch_authorization_signature_sha256: str
    runtime_claim_path: Path
    runtime_claim_sha256: str
    training_lineage_path: Path
    training_lineage_sha256: str
    output_dir: Path
    runtime_config_path: Path
    runtime_config_sha256: str
    raw_input_manifest_path: Path
    raw_input_manifest_sha256: str
    causal_manifest_sha256: str
    causal_record_count: int
    causal_log_count: int
    merged_artifact_certificate_path: Path
    merged_artifact_certificate_sha256: str
    artifact_root_sha256: str
    checkpoints: Tuple[RunRootCheckpoint, ...]

    def __post_init__(self) -> None:
        for name in (
            "bindings_path",
            "launch_authorization_path",
            "launch_authorization_signature_path",
            "runtime_claim_path",
            "training_lineage_path",
            "output_dir",
            "runtime_config_path",
            "raw_input_manifest_path",
            "merged_artifact_certificate_path",
        ):
            object.__setattr__(
                self,
                name,
                _absolute_authorized_path(getattr(self, name), name),
            )
        for name in (
            "bindings_sha256",
            "launch_authorization_sha256",
            "launch_authorization_signature_sha256",
            "runtime_claim_sha256",
            "training_lineage_sha256",
            "runtime_config_sha256",
            "raw_input_manifest_sha256",
            "causal_manifest_sha256",
            "merged_artifact_certificate_sha256",
            "artifact_root_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "causal_record_count",
            _integer(self.causal_record_count, "causal record count", minimum=1),
        )
        object.__setattr__(
            self,
            "causal_log_count",
            _integer(self.causal_log_count, "causal log count", minimum=1),
        )
        checkpoints = tuple(sorted(self.checkpoints, key=lambda item: item.epoch))
        if (
            not all(type(item) is RunRootCheckpoint for item in checkpoints)
            or tuple(item.epoch for item in checkpoints) != F2_CHECKPOINT_EPOCHS
        ):
            raise F2ProductionRunnerError(
                "run-root checkpoints must be exactly epochs 0, 1, 2, and 3"
            )
        if len({item.sha256 for item in checkpoints}) != 4:
            raise F2ProductionRunnerError(
                "run-root checkpoint SHA-256 identities must be unique"
            )
        object.__setattr__(self, "checkpoints", checkpoints)

    def checkpoint_sha256(self, epoch: int) -> str:
        if epoch not in F2_CHECKPOINT_EPOCHS:
            raise F2ProductionRunnerError("requested run-root epoch is outside F2")
        return self.checkpoints[epoch].sha256


def production_run_root_payload(root: ProductionRunRoot) -> Dict[str, Any]:
    if type(root) is not ProductionRunRoot:
        raise F2ProductionRunnerError("production run-root is malformed")
    return {
        "bindings": {
            "path": str(root.bindings_path),
            "sha256": root.bindings_sha256,
        },
        "causal_manifest": {
            "log_count": root.causal_log_count,
            "record_count": root.causal_record_count,
            "sha256": root.causal_manifest_sha256,
        },
        "checkpoints": [item.to_payload() for item in root.checkpoints],
        "dataset": DATASET,
        "kind": "production_run_root",
        "launch_authorization": {
            "path": str(root.launch_authorization_path),
            "sha256": root.launch_authorization_sha256,
            "signature_path": str(root.launch_authorization_signature_path),
            "signature_sha256": root.launch_authorization_signature_sha256,
        },
        "merged_artifact": {
            "certificate_path": str(root.merged_artifact_certificate_path),
            "certificate_sha256": root.merged_artifact_certificate_sha256,
            "root_sha256": root.artifact_root_sha256,
        },
        "numeric_mode": F2_MODEL_NUMERIC_MODE,
        "partition": PARTITION,
        "raw_inputs": {
            "manifest_path": str(root.raw_input_manifest_path),
            "manifest_sha256": root.raw_input_manifest_sha256,
        },
        "runtime_claim": {
            "output_dir": str(root.output_dir),
            "path": str(root.runtime_claim_path),
            "sha256": root.runtime_claim_sha256,
        },
        "training_lineage": {
            "path": str(root.training_lineage_path),
            "sha256": root.training_lineage_sha256,
        },
        "runtime_config": {
            "path": str(root.runtime_config_path),
            "sha256": root.runtime_config_sha256,
        },
        "schema_version": RUN_ROOT_SCHEMA_VERSION,
    }


def production_run_root_bytes(root: ProductionRunRoot) -> bytes:
    return canonical_json_bytes(production_run_root_payload(root))


def write_production_run_root(
    path: Path | str, root: ProductionRunRoot
) -> str:
    """Publish a root for a separate authority to review and pin by digest."""

    return write_bytes_no_overwrite(path, production_run_root_bytes(root))


def _parse_run_root_checkpoint(payload: Any) -> RunRootCheckpoint:
    value = _exact_keys(payload, {"epoch", "sha256"}, "run-root checkpoint")
    return RunRootCheckpoint(epoch=value["epoch"], sha256=value["sha256"])


def _parse_production_run_root(payload: Any) -> ProductionRunRoot:
    document = _exact_keys(
        payload,
        {
            "bindings",
            "causal_manifest",
            "checkpoints",
            "dataset",
            "kind",
            "launch_authorization",
            "merged_artifact",
            "numeric_mode",
            "partition",
            "raw_inputs",
            "runtime_claim",
            "training_lineage",
            "runtime_config",
            "schema_version",
        },
        "production run-root document",
    )
    if (
        document["schema_version"] != RUN_ROOT_SCHEMA_VERSION
        or document["kind"] != "production_run_root"
        or document["dataset"] != DATASET
        or document["partition"] != PARTITION
        or document["numeric_mode"] != F2_MODEL_NUMERIC_MODE
    ):
        raise F2ProductionRunnerError("production run-root contract drifted")
    bindings = _exact_keys(
        document["bindings"], {"path", "sha256"}, "run-root bindings"
    )
    runtime = _exact_keys(
        document["runtime_config"],
        {"path", "sha256"},
        "run-root runtime config",
    )
    authorization = _exact_keys(
        document["launch_authorization"],
        {"path", "sha256", "signature_path", "signature_sha256"},
        "run-root launch authorization",
    )
    runtime_claim = _exact_keys(
        document["runtime_claim"],
        {"output_dir", "path", "sha256"},
        "run-root runtime claim",
    )
    training_lineage = _exact_keys(
        document["training_lineage"],
        {"path", "sha256"},
        "run-root training lineage",
    )
    raw_inputs = _exact_keys(
        document["raw_inputs"],
        {"manifest_path", "manifest_sha256"},
        "run-root raw inputs",
    )
    causal = _exact_keys(
        document["causal_manifest"],
        {"log_count", "record_count", "sha256"},
        "run-root causal manifest",
    )
    artifact = _exact_keys(
        document["merged_artifact"],
        {"certificate_path", "certificate_sha256", "root_sha256"},
        "run-root merged artifact",
    )
    if not isinstance(document["checkpoints"], list):
        raise F2ProductionRunnerError("run-root checkpoints must be a JSON array")
    return ProductionRunRoot(
        bindings_path=_absolute_authorized_path(bindings["path"], "bindings path"),
        bindings_sha256=bindings["sha256"],
        launch_authorization_path=_absolute_authorized_path(
            authorization["path"], "launch authorization path"
        ),
        launch_authorization_sha256=authorization["sha256"],
        launch_authorization_signature_path=_absolute_authorized_path(
            authorization["signature_path"],
            "launch authorization signature path",
        ),
        launch_authorization_signature_sha256=authorization["signature_sha256"],
        runtime_claim_path=_absolute_authorized_path(
            runtime_claim["path"], "runtime claim path"
        ),
        runtime_claim_sha256=runtime_claim["sha256"],
        training_lineage_path=_absolute_authorized_path(
            training_lineage["path"], "training lineage path"
        ),
        training_lineage_sha256=training_lineage["sha256"],
        output_dir=_absolute_authorized_path(
            runtime_claim["output_dir"], "runtime claim output directory"
        ),
        runtime_config_path=_absolute_authorized_path(
            runtime["path"], "runtime config path"
        ),
        runtime_config_sha256=runtime["sha256"],
        raw_input_manifest_path=_absolute_authorized_path(
            raw_inputs["manifest_path"], "raw-input manifest path"
        ),
        raw_input_manifest_sha256=raw_inputs["manifest_sha256"],
        causal_manifest_sha256=causal["sha256"],
        causal_record_count=causal["record_count"],
        causal_log_count=causal["log_count"],
        merged_artifact_certificate_path=_absolute_authorized_path(
            artifact["certificate_path"], "merged artifact certificate path"
        ),
        merged_artifact_certificate_sha256=artifact["certificate_sha256"],
        artifact_root_sha256=artifact["root_sha256"],
        checkpoints=tuple(
            _parse_run_root_checkpoint(item) for item in document["checkpoints"]
        ),
    )


def _ssh_keygen_executable() -> str:
    system = Path("/usr/bin/ssh-keygen")
    if system.is_file():
        return str(system)
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise F2ProductionRunnerError(
            "ssh-keygen with SSHSIG support is required for production run-root"
        )
    return executable


def verify_production_run_root_signature(
    message: bytes,
    signature: bytes,
    *,
    trusted_public_key: str = TRUSTED_AUTHORITY_SSH_PUBLIC_KEY,
) -> None:
    """Verify the run-root under its distinct embedded-authority namespace."""

    if not isinstance(message, bytes) or not message:
        raise F2ProductionRunnerError("production run-root signature message is empty")
    if not isinstance(signature, bytes) or (
        not signature.startswith(b"-----BEGIN SSH SIGNATURE-----\n")
        or not signature.endswith(b"-----END SSH SIGNATURE-----\n")
    ):
        raise F2ProductionRunnerError(
            "production run-root signature must be canonical armored SSHSIG"
        )
    if not isinstance(trusted_public_key, str):
        raise F2ProductionRunnerError("trusted production authority key is malformed")
    key_parts = trusted_public_key.strip().split()
    if len(key_parts) != 2 or key_parts[0] not in {"ssh-ed25519", "ssh-rsa"}:
        raise F2ProductionRunnerError("trusted production authority key is malformed")
    normalized_key = " ".join(key_parts)
    with tempfile.TemporaryDirectory(prefix="actworld-f2-run-root-verify-") as temporary:
        root = Path(temporary)
        allowed_signers = root / "allowed_signers"
        signature_file = root / "run-root.sig"
        allowed_signers.write_bytes(
            f"{RUN_ROOT_SSH_PRINCIPAL} {normalized_key}\n".encode("ascii")
        )
        signature_file.write_bytes(signature)
        completed = subprocess.run(
            [
                _ssh_keygen_executable(),
                "-Y",
                "verify",
                "-f",
                str(allowed_signers),
                "-I",
                RUN_ROOT_SSH_PRINCIPAL,
                "-n",
                RUN_ROOT_SSH_NAMESPACE,
                "-s",
                str(signature_file),
            ],
            input=message,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise F2ProductionRunnerError(
            "production run-root authority signature is invalid"
            + (f": {detail}" if detail else "")
        )


def read_production_run_root(
    path: Path | str,
    *,
    signature_path: Path | str,
    expected_sha256: Optional[str] = None,
) -> Tuple[ProductionRunRoot, str]:
    """Authenticate a canonical run-root with the embedded offline authority."""

    decoded, payload, digest = _read_canonical_file(
        path, name="production run-root"
    )
    signature = _read_stable_regular_file(
        Path(signature_path), name="production run-root SSHSIG"
    )
    verify_production_run_root_signature(payload, signature)
    if expected_sha256 is not None and digest != _sha256(
        expected_sha256, "expected production run-root"
    ):
        raise F2ProductionRunnerError(
            "production run-root SHA-256 differs from external expectation"
        )
    root = _parse_production_run_root(decoded)
    if production_run_root_bytes(root) != payload:
        raise F2ProductionRunnerError(
            "production run-root is not idempotently canonical"
        )
    return root, digest


@dataclass(frozen=True)
class VerifiedProductionRunRoot:
    root: ProductionRunRoot
    run_root_sha256: str
    bindings: ArtifactBindings
    runtime_config: ProductionRuntimeConfig
    live_provenance: Any
    raw_inputs: VerifiedNavtrainRawInputs
    merged_artifact: Any


_SIGNED_ARTIFACT_RUNTIME_BINDINGS = {
    "baseline_checkpoint": (
        "baseline_checkpoint",
        "baseline_checkpoint_sha256",
    ),
    "vjepa_checkpoint": ("vjepa_checkpoint", "vjepa_checkpoint_sha256"),
    "cache_metadata": ("cache_metadata", "cache_metadata_sha256"),
    "cache_content_manifest": (
        "metric_cache_manifest",
        "metric_cache_manifest_sha256",
    ),
    "raw_input_manifest": (
        "raw_input_manifest",
        "raw_input_manifest_sha256",
    ),
    "qualification_manifest": (
        "qualification_manifest",
        "qualification_manifest_sha256",
    ),
    "qualification_config": (
        "qualification_config",
        "qualification_config_sha256",
    ),
    "qualification_source_manifest": (
        "qualification_source_manifest",
        "qualification_source_manifest_sha256",
    ),
}

_SIGNED_RUNTIME_CONFIG_BINDINGS = {
    "repository_root": ("repository_root", "runtime_source_root"),
    "navsim_exp_root": ("experiment_root",),
    "navsim_log_root": ("navsim_log",),
    "nuplan_maps_root": ("nuplan_maps_root",),
    "sensor_blobs_root": ("sensor_blobs",),
    "qualification_source_root": ("qualification_source_root",),
}


def _require_signed_launch_evaluation_cross_bindings(
    *,
    signed_contract: Any,
    bindings: ArtifactBindings,
    runtime_paths: Mapping[str, Path],
) -> None:
    """Bind every evaluation input to the training authorization.

    The production run-root is separately signed, but it must not be possible
    to combine a genuine training authorization for asset set A with a
    self-consistent bindings/runtime pair for asset set B.  Feasibility inputs
    are admission-only artifacts and remain transitively authenticated by the
    authorization/claim/run-root chain; every artifact consumed by evaluation
    is bound here by both its exact signed path and digest.
    """

    artifacts = signed_contract.artifacts
    runtime = signed_contract.runtime
    if set(artifacts) != set(AUTHORIZED_ARTIFACT_KEYS):
        raise F2ProductionRunnerError(
            "signed launch artifact key set differs from the production contract"
        )
    expected_runtime_keys = {
        "repository_root",
        "navsim_exp_root",
        "openscene_data_root",
        "nuplan_maps_root",
        "navsim_log_root",
        "sensor_blobs_root",
        "qualification_source_root",
    }
    if set(runtime) != expected_runtime_keys:
        raise F2ProductionRunnerError(
            "signed launch runtime key set differs from the production contract"
        )

    for artifact_name, (
        runtime_path_name,
        binding_digest_name,
    ) in _SIGNED_ARTIFACT_RUNTIME_BINDINGS.items():
        artifact = artifacts[artifact_name]
        if artifact.path != runtime_paths[runtime_path_name]:
            raise F2ProductionRunnerError(
                f"signed launch {artifact_name} path differs from evaluation runtime"
            )
        if artifact.sha256 != getattr(bindings, binding_digest_name):
            raise F2ProductionRunnerError(
                f"signed launch {artifact_name} digest differs from evaluation bindings"
            )

    for signed_runtime_name, runtime_path_names in (
        _SIGNED_RUNTIME_CONFIG_BINDINGS.items()
    ):
        signed_path = runtime[signed_runtime_name]
        for runtime_path_name in runtime_path_names:
            if signed_path != runtime_paths[runtime_path_name]:
                raise F2ProductionRunnerError(
                    f"signed launch {signed_runtime_name} differs from evaluation "
                    f"runtime path {runtime_path_name}"
                )

    expected_cache_root = (
        artifacts["baseline_checkpoint"].path.parent / F2_PRUNED_CACHE_FAMILY
    )
    if runtime_paths["metric_cache_root"] != expected_cache_root:
        raise F2ProductionRunnerError(
            "evaluation metric-cache root differs from the signed training cache family"
        )

    # Scene._build_map_api consumes this module-level value, captured when
    # navsim.common.dataclasses is imported.  Binding only the JSON path would
    # still permit a third, environment-selected map root to drive evaluation.
    from navsim.common import dataclasses as navsim_dataclasses

    imported_maps_root = navsim_dataclasses.NUPLAN_MAPS_ROOT
    if (
        not isinstance(imported_maps_root, str)
        or Path(imported_maps_root) != runtime_paths["nuplan_maps_root"]
    ):
        raise F2ProductionRunnerError(
            "imported NUPLAN_MAPS_ROOT differs from the signed evaluation runtime"
        )


def replay_production_run_root(
    path: Path | str,
    *,
    signature_path: Path | str,
    expected_sha256: Optional[str] = None,
) -> VerifiedProductionRunRoot:
    """Replay every identity and full artifact coverage from one trust root."""

    root, root_digest = read_production_run_root(
        path,
        signature_path=signature_path,
        expected_sha256=expected_sha256,
    )
    _, bindings_digest = read_bytes_with_sidecar(root.bindings_path)
    if bindings_digest != root.bindings_sha256:
        raise F2ProductionRunnerError("run-root bindings SHA-256 drifted")
    bindings = read_artifact_bindings(root.bindings_path)
    if _sha256_bytes(artifact_bindings_bytes(bindings)) != root.bindings_sha256:
        raise F2ProductionRunnerError("run-root bindings bytes are inconsistent")
    if bindings.causal_manifest_sha256 != root.causal_manifest_sha256:
        raise F2ProductionRunnerError(
            "run-root causal manifest differs from artifact bindings"
        )
    if bindings.raw_input_manifest_sha256 != root.raw_input_manifest_sha256:
        raise F2ProductionRunnerError(
            "run-root raw-input manifest differs from artifact bindings"
        )

    runtime_payload = _read_stable_regular_file(
        root.runtime_config_path, name="production runtime config"
    )
    if _sha256_bytes(runtime_payload) != root.runtime_config_sha256:
        raise F2ProductionRunnerError("run-root runtime config SHA-256 drifted")
    if root.runtime_config_sha256 != bindings.runtime_config_sha256:
        raise F2ProductionRunnerError(
            "run-root runtime config differs from artifact bindings"
        )
    config = _parse_runtime_config(
        root.runtime_config_path,
        expected_sha256=root.runtime_config_sha256,
    )
    paths = config.path_map

    try:
        authorization_sha = _sha256_stable_regular_file(
            root.launch_authorization_path,
            name="F2 launch authorization",
        )
        authorization_signature_sha = _sha256_stable_regular_file(
            root.launch_authorization_signature_path,
            name="F2 launch authorization signature",
        )
    except (OSError, ValueError) as error:
        raise F2ProductionRunnerError(
            "run-root launch authorization authentication failed"
        ) from error
    if authorization_sha != root.launch_authorization_sha256:
        raise F2ProductionRunnerError(
            "run-root launch authorization SHA-256 drifted"
        )
    if authorization_signature_sha != root.launch_authorization_signature_sha256:
        raise F2ProductionRunnerError(
            "run-root launch authorization signature SHA-256 drifted"
        )
    try:
        signed_contract = verify_f2_signed_contract_lightweight(
            root.launch_authorization_path,
            root.launch_authorization_signature_path,
            repository_root=paths["repository_root"],
        )
        runtime_claim = verify_f2_runtime_claim(
            root.runtime_claim_path,
            expected_sha256=root.runtime_claim_sha256,
            signed_contract=signed_contract,
            require_live_parent=False,
        )
    except Exception as error:
        raise F2ProductionRunnerError(
            f"run-root signed launch/claim replay failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    if (
        signed_contract.authorization_sha256 != root.launch_authorization_sha256
        or signed_contract.signature_sha256
        != root.launch_authorization_signature_sha256
        or runtime_claim.sha256 != root.runtime_claim_sha256
        or runtime_claim.authorization_sha256 != root.launch_authorization_sha256
        or runtime_claim.signature_sha256
        != root.launch_authorization_signature_sha256
        or signed_contract.output_dir != root.output_dir
        or runtime_claim.output_dir != root.output_dir
        or runtime_claim.launch_marker != root.runtime_claim_path
        or signed_contract.source_revision != bindings.source_revision
        or runtime_claim.source_revision != bindings.source_revision
    ):
        raise F2ProductionRunnerError(
            "run-root launch authorization or runtime claim identity drifted"
        )
    _require_signed_launch_evaluation_cross_bindings(
        signed_contract=signed_contract,
        bindings=bindings,
        runtime_paths=paths,
    )

    expected_lineage_path = root.output_dir / TRAINING_LINEAGE_FILENAME
    if root.training_lineage_path != expected_lineage_path:
        raise F2ProductionRunnerError(
            "run-root training lineage path is outside the signed output directory"
        )
    try:
        lineage_raw = _read_stable_regular_file(
            root.training_lineage_path, name="F2 training lineage"
        )
    except (OSError, ValueError) as error:
        raise F2ProductionRunnerError(
            "run-root training lineage authentication failed"
        ) from error
    if _sha256_bytes(lineage_raw) != root.training_lineage_sha256:
        raise F2ProductionRunnerError("run-root training lineage SHA-256 drifted")
    lineage = _exact_keys(
        parse_canonical_json_bytes(lineage_raw, name="F2 training lineage"),
        {
            "authorization",
            "experiment",
            "runtime_claim",
            "schema_version",
            "source_revision",
            "torchrun",
        },
        "F2 training lineage",
    )
    if lineage["schema_version"] != TRAINING_LINEAGE_SCHEMA_VERSION:
        raise F2ProductionRunnerError("F2 training lineage schema drifted")
    lineage_authorization = _exact_keys(
        lineage["authorization"],
        {"sha256", "signature_sha256"},
        "F2 training lineage authorization",
    )
    lineage_claim = _exact_keys(
        lineage["runtime_claim"],
        {"path", "sha256"},
        "F2 training lineage runtime claim",
    )
    lineage_experiment = _exact_keys(
        lineage["experiment"],
        {"name", "output_dir", "output_directory_identity", "uid"},
        "F2 training lineage experiment",
    )
    lineage_output_identity = _exact_keys(
        lineage_experiment["output_directory_identity"],
        {"device", "inode"},
        "F2 training lineage output directory identity",
    )
    lineage_torchrun = _exact_keys(
        lineage["torchrun"],
        {"boot_id", "pid", "start_time_ticks", "world_size"},
        "F2 training lineage torchrun",
    )
    try:
        lineage_output_stat = root.output_dir.lstat()
    except OSError as error:
        raise F2ProductionRunnerError(
            "signed F2 output directory is unavailable during lineage replay"
        ) from error
    if stat.S_ISLNK(lineage_output_stat.st_mode) or not stat.S_ISDIR(
        lineage_output_stat.st_mode
    ):
        raise F2ProductionRunnerError(
            "signed F2 output directory is not a real directory"
        )
    expected_lineage = {
        "authorization_sha256": root.launch_authorization_sha256,
        "authorization_signature_sha256": (
            root.launch_authorization_signature_sha256
        ),
        "source_revision": bindings.source_revision,
        "claim_path": str(root.runtime_claim_path),
        "claim_sha256": root.runtime_claim_sha256,
        "experiment_name": signed_contract.experiment_name,
        "experiment_uid": signed_contract.experiment_uid,
        "output_dir": str(root.output_dir),
        "output_device": lineage_output_stat.st_dev,
        "output_inode": lineage_output_stat.st_ino,
        "boot_id": runtime_claim.boot_id,
        "pid": runtime_claim.launcher_pid,
        "start_time_ticks": runtime_claim.launcher_start_time_ticks,
        "world_size": 8,
    }
    observed_lineage = {
        "authorization_sha256": _sha256(
            lineage_authorization["sha256"], "lineage authorization"
        ),
        "authorization_signature_sha256": _sha256(
            lineage_authorization["signature_sha256"],
            "lineage authorization signature",
        ),
        "source_revision": lineage["source_revision"],
        "claim_path": lineage_claim["path"],
        "claim_sha256": _sha256(lineage_claim["sha256"], "lineage claim"),
        "experiment_name": lineage_experiment["name"],
        "experiment_uid": lineage_experiment["uid"],
        "output_dir": lineage_experiment["output_dir"],
        "output_device": _integer(
            lineage_output_identity["device"],
            "lineage output directory device",
            minimum=1,
        ),
        "output_inode": _integer(
            lineage_output_identity["inode"],
            "lineage output directory inode",
            minimum=1,
        ),
        "boot_id": lineage_torchrun["boot_id"],
        "pid": _integer(lineage_torchrun["pid"], "lineage launcher PID", minimum=1),
        "start_time_ticks": _integer(
            lineage_torchrun["start_time_ticks"],
            "lineage launcher start time",
            minimum=1,
        ),
        "world_size": _integer(
            lineage_torchrun["world_size"], "lineage world_size", minimum=1
        ),
    }
    if observed_lineage != expected_lineage:
        raise F2ProductionRunnerError(
            "F2 training lineage differs from signed authorization/runtime claim"
        )

    if tuple(item.sha256 for item in config.checkpoints) != tuple(
        item.sha256 for item in root.checkpoints
    ):
        raise F2ProductionRunnerError(
            "runtime checkpoint identities differ from external run-root"
        )
    checkpoint_directory = root.output_dir / "checkpoints"
    try:
        output_stat = root.output_dir.lstat()
        checkpoint_stat = checkpoint_directory.lstat()
        resolved_output = root.output_dir.resolve(strict=True)
        resolved_checkpoint_directory = checkpoint_directory.resolve(strict=True)
    except OSError as error:
        raise F2ProductionRunnerError(
            "signed F2 output/checkpoint directory resolution failed"
        ) from error
    if (
        stat.S_ISLNK(output_stat.st_mode)
        or not stat.S_ISDIR(output_stat.st_mode)
        or stat.S_ISLNK(checkpoint_stat.st_mode)
        or not stat.S_ISDIR(checkpoint_stat.st_mode)
        or resolved_checkpoint_directory.parent != resolved_output
    ):
        raise F2ProductionRunnerError(
            "signed F2 output/checkpoint directory is not a direct real directory"
        )
    resolved_checkpoint_paths = []
    for checkpoint in config.checkpoints:
        try:
            checkpoint_file_stat = checkpoint.path.lstat()
            resolved_checkpoint = checkpoint.path.resolve(strict=True)
            if (
                stat.S_ISLNK(checkpoint_file_stat.st_mode)
                or not stat.S_ISREG(checkpoint_file_stat.st_mode)
                or checkpoint.path.parent != checkpoint_directory
                or resolved_checkpoint.parent != resolved_checkpoint_directory
            ):
                raise F2ProductionRunnerError(
                    f"runtime checkpoint epoch {checkpoint.epoch} is outside the "
                    "signed output_dir/checkpoints directory"
                )
            actual_checkpoint_sha = _sha256_stable_regular_file(
                resolved_checkpoint,
                name=f"runtime checkpoint epoch {checkpoint.epoch}",
            )
            resolved_checkpoint_paths.append(resolved_checkpoint)
        except F2ProductionRunnerError:
            raise
        except (OSError, ValueError) as error:
            raise F2ProductionRunnerError(
                f"runtime checkpoint epoch {checkpoint.epoch} authentication failed"
            ) from error
        if actual_checkpoint_sha != root.checkpoint_sha256(checkpoint.epoch):
            raise F2ProductionRunnerError(
                f"runtime checkpoint epoch {checkpoint.epoch} SHA-256 drifted"
            )
    if len(set(resolved_checkpoint_paths)) != len(F2_CHECKPOINT_EPOCHS):
        raise F2ProductionRunnerError(
            "runtime checkpoint paths are not four unique signed output files"
        )

    try:
        configured_raw_manifest = paths["raw_input_manifest"].resolve(strict=True)
        authorized_raw_manifest = root.raw_input_manifest_path.resolve(strict=True)
    except OSError as error:
        raise F2ProductionRunnerError(
            "run-root raw-input manifest path resolution failed"
        ) from error
    if configured_raw_manifest != authorized_raw_manifest:
        raise F2ProductionRunnerError(
            "runtime raw-input manifest path differs from external run-root"
        )
    try:
        live = verify_live_f2_provenance(
            bindings=bindings,
            qualification_manifest_path=paths["qualification_manifest"],
            causal_manifest_path=paths["causal_manifest"],
            qualification_config_path=paths["qualification_config"],
            runtime_config_path=root.runtime_config_path,
            cache_metadata_path=paths["cache_metadata"],
            metric_cache_manifest_path=paths["metric_cache_manifest"],
            metric_cache_root=paths["metric_cache_root"],
            baseline_checkpoint_path=paths["baseline_checkpoint"],
            vjepa_checkpoint_path=paths["vjepa_checkpoint"],
            qualification_source_manifest_path=paths[
                "qualification_source_manifest"
            ],
            qualification_source_root=paths["qualification_source_root"],
            runtime_source_manifest_path=paths["runtime_source_manifest"],
            runtime_source_root=paths["runtime_source_root"],
        )
    except Exception as error:
        raise F2ProductionRunnerError(
            f"run-root live provenance replay failed: {type(error).__name__}: {error}"
        ) from error
    if live.causal.split != DATASET or live.qualification.dataset != DATASET:
        raise F2ProductionRunnerError("run-root provenance is not held-out navtrain")
    if _sha256_bytes(canonical_manifest_bytes(live.causal)) != root.causal_manifest_sha256:
        raise F2ProductionRunnerError("run-root causal manifest bytes drifted")
    if len(live.causal.rows) != root.causal_record_count:
        raise F2ProductionRunnerError("run-root causal record count drifted")
    if len(live.causal.logs) != root.causal_log_count:
        raise F2ProductionRunnerError("run-root causal log count drifted")

    try:
        raw_inputs = load_and_verify_raw_input_manifest(
            root.raw_input_manifest_path,
            expected_sha256=root.raw_input_manifest_sha256,
            log_root=paths["navsim_log"],
            sensor_root=paths["sensor_blobs"],
            qualification=live.qualification,
            expected_qualification_manifest_sha256=(
                bindings.qualification_manifest_sha256
            ),
        )
        raw_inputs.verify_all(workers=8)
    except Exception as error:
        raise F2ProductionRunnerError(
            f"run-root raw-input replay failed: {type(error).__name__}: {error}"
        ) from error
    if raw_inputs.manifest_sha256 != root.raw_input_manifest_sha256:
        raise F2ProductionRunnerError("run-root raw-input manifest SHA-256 drifted")

    _, certificate_digest = read_bytes_with_sidecar(
        root.merged_artifact_certificate_path
    )
    if certificate_digest != root.merged_artifact_certificate_sha256:
        raise F2ProductionRunnerError(
            "run-root merged artifact certificate SHA-256 drifted"
        )
    try:
        # Local import avoids the intentional artifact-production -> runner
        # dependency at module initialization time.
        from navsim.planning.validation.f2_artifact_production import (
            read_merged_artifact_certificate,
        )

        merged = read_merged_artifact_certificate(
            root.merged_artifact_certificate_path,
            bindings=bindings,
            causal_manifest=live.causal,
            qualification_held_out_rows=live.qualification.held_out.rows,
        )
    except Exception as error:
        raise F2ProductionRunnerError(
            f"merged artifact certificate replay failed: {type(error).__name__}: {error}"
        ) from error
    if merged.descriptor.root_sha256 != root.artifact_root_sha256:
        raise F2ProductionRunnerError("run-root artifact root SHA-256 drifted")
    if merged.descriptor.record_count != root.causal_record_count:
        raise F2ProductionRunnerError("merged artifact record coverage drifted")
    try:
        configured_artifact = paths["artifact"].resolve(strict=True)
        certified_artifact = merged.artifact_path.resolve(strict=True)
    except OSError as error:
        raise F2ProductionRunnerError(
            "run-root artifact path resolution failed"
        ) from error
    if configured_artifact != certified_artifact:
        raise F2ProductionRunnerError(
            "runtime artifact path differs from merged certificate"
        )
    return VerifiedProductionRunRoot(
        root=root,
        run_root_sha256=root_digest,
        bindings=bindings,
        runtime_config=config,
        live_provenance=live,
        raw_inputs=raw_inputs,
        merged_artifact=merged,
    )


@dataclass(frozen=True)
class DeterministicRuntimeState:
    logical_rank: int
    world_size: int
    local_rank: Optional[int]
    device: str
    device_index: int
    cuda_visible_devices: Optional[str]
    torch_version: str
    torch_cuda_version: str
    cudnn_version: Optional[int]
    device_name: str
    device_capability: Tuple[int, int]
    python_version: str = field(default_factory=lambda: sys.version.split()[0])

    def __post_init__(self) -> None:
        if (
            isinstance(self.logical_rank, bool)
            or not isinstance(self.logical_rank, int)
            or isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or not 0 <= self.logical_rank < self.world_size
        ):
            raise F2ProductionRunnerError("deterministic runtime rank is invalid")
        if self.local_rank is not None and (
            isinstance(self.local_rank, bool)
            or not isinstance(self.local_rank, int)
            or self.local_rank < 0
        ):
            raise F2ProductionRunnerError("deterministic runtime LOCAL_RANK is invalid")
        if (
            not isinstance(self.device, str)
            or self.device != f"cuda:{self.device_index}"
            or isinstance(self.device_index, bool)
            or not isinstance(self.device_index, int)
            or self.device_index < 0
        ):
            raise F2ProductionRunnerError("deterministic runtime device is invalid")
        if self.cuda_visible_devices is not None and not isinstance(
            self.cuda_visible_devices, str
        ):
            raise F2ProductionRunnerError("CUDA_VISIBLE_DEVICES record is invalid")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.torch_version,
                self.torch_cuda_version,
                self.device_name,
                self.python_version,
            )
        ):
            raise F2ProductionRunnerError("deterministic runtime version is invalid")
        if self.cudnn_version is not None and (
            isinstance(self.cudnn_version, bool)
            or not isinstance(self.cudnn_version, int)
            or self.cudnn_version <= 0
        ):
            raise F2ProductionRunnerError("cuDNN runtime version is invalid")
        if (
            not isinstance(self.device_capability, tuple)
            or len(self.device_capability) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.device_capability
            )
        ):
            raise F2ProductionRunnerError("CUDA device capability is invalid")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "cuda_matmul_allow_tf32": False,
            "cuda_visible_devices": self.cuda_visible_devices,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_version": self.cudnn_version,
            "deterministic_algorithms": True,
            "device": self.device,
            "device_capability": list(self.device_capability),
            "device_index": self.device_index,
            "device_name": self.device_name,
            "float32_matmul_precision": "highest",
            "local_rank": self.local_rank,
            "logical_rank": self.logical_rank,
            "python_version": self.python_version,
            "torch_cuda_version": self.torch_cuda_version,
            "torch_version": self.torch_version,
            "world_size": self.world_size,
        }


def resolve_production_device(
    *,
    explicit_device: Optional[str],
    environment: Optional[Mapping[str, str]] = None,
) -> Tuple[str, int, Optional[int]]:
    """Resolve a physical process device independently of logical data rank."""

    env = os.environ if environment is None else environment
    raw_local_rank = env.get("LOCAL_RANK")
    local_rank: Optional[int] = None
    if raw_local_rank is not None:
        if not raw_local_rank.isdecimal():
            raise F2ProductionRunnerError("LOCAL_RANK must be a non-negative integer")
        local_rank = int(raw_local_rank)
    if explicit_device is None:
        if local_rank is None:
            raise F2ProductionRunnerError(
                "production evaluation requires LOCAL_RANK or explicit --device"
            )
        index = local_rank
    else:
        if not isinstance(explicit_device, str) or not explicit_device.startswith(
            "cuda:"
        ):
            raise F2ProductionRunnerError(
                "explicit production device must have form cuda:<index>"
            )
        raw_index = explicit_device[5:]
        if not raw_index.isdecimal():
            raise F2ProductionRunnerError(
                "explicit production device must have form cuda:<index>"
            )
        index = int(raw_index)
    return f"cuda:{index}", index, local_rank


def _import_torch_for_runtime() -> Any:
    import torch

    return torch


def initialize_deterministic_runtime(
    *,
    logical_rank: int,
    world_size: int,
    explicit_device: Optional[str],
) -> DeterministicRuntimeState:
    """Initialize the sole accepted CUDA numeric/determinism environment.

    ``CUBLAS_WORKSPACE_CONFIG`` is installed before importing/initializing CUDA.
    An already-created CUDA context is rejected because its cuBLAS policy can
    no longer be proved.  Logical shard rank never participates in device
    selection.
    """

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise F2ProductionRunnerError("world_size must be 1, 2, or 8")
    if (
        isinstance(logical_rank, bool)
        or not isinstance(logical_rank, int)
        or not 0 <= logical_rank < world_size
    ):
        raise F2ProductionRunnerError("logical rank is outside world_size")
    current_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current_cublas is not None and current_cublas != CUBLAS_WORKSPACE_CONFIG:
        raise F2ProductionRunnerError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with deterministic contract"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    device, device_index, local_rank = resolve_production_device(
        explicit_device=explicit_device
    )
    torch = _import_torch_for_runtime()
    if torch.cuda.is_initialized():
        raise F2ProductionRunnerError(
            "CUDA was initialized before deterministic runtime setup"
        )
    if not torch.cuda.is_available():
        raise F2ProductionRunnerError("CUDA is unavailable for production evaluation")
    if device_index >= int(torch.cuda.device_count()):
        raise F2ProductionRunnerError("resolved CUDA device index is unavailable")

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(device_index)
    if not torch.are_deterministic_algorithms_enabled():
        raise F2ProductionRunnerError("PyTorch deterministic algorithms are disabled")
    if torch.get_float32_matmul_precision() != "highest":
        raise F2ProductionRunnerError("float32 matmul precision policy drifted")
    if (
        torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
    ):
        raise F2ProductionRunnerError("CUDA deterministic backend policy drifted")
    cuda_version = getattr(torch.version, "cuda", None)
    if not isinstance(cuda_version, str) or not cuda_version:
        raise F2ProductionRunnerError("PyTorch CUDA runtime version is unavailable")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device_index))
    return DeterministicRuntimeState(
        logical_rank=logical_rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        device_index=device_index,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch_version=str(torch.__version__),
        torch_cuda_version=cuda_version,
        cudnn_version=torch.backends.cudnn.version(),
        device_name=str(torch.cuda.get_device_name(device_index)),
        device_capability=(capability[0], capability[1]),
        python_version=sys.version.split()[0],
    )


def runtime_record_path_for_shard(path: Path | str) -> Path:
    shard = Path(path)
    return shard.with_name(shard.name + ".runtime.json")


def deterministic_runtime_record_payload(
    *,
    state: DeterministicRuntimeState,
    run_root_sha256: str,
    shard_path: Path,
    shard_sha256: str,
    evaluation: CausalShardEvaluation,
) -> Dict[str, Any]:
    if type(state) is not DeterministicRuntimeState:
        raise F2ProductionRunnerError("deterministic runtime state is malformed")
    if not shard_path.is_absolute():
        raise F2ProductionRunnerError("runtime record shard path must be absolute")
    return {
        "dataset": DATASET,
        "identity": {
            "artifact_sha256": evaluation.artifact_sha256,
            "checkpoint_epoch": evaluation.checkpoint_epoch,
            "checkpoint_sha256": evaluation.checkpoint_sha256,
            "manifest_sha256": evaluation.manifest_sha256,
            "numeric_mode": evaluation.numeric_mode,
            "run_root_sha256": _sha256(run_root_sha256, "production run-root"),
        },
        "kind": "deterministic_runtime_record",
        "partition": PARTITION,
        "runtime": state.to_payload(),
        "schema_version": RUNTIME_RECORD_SCHEMA_VERSION,
        "shard": {
            "file_name": shard_path.name,
            "file_sha256": _sha256(shard_sha256, "shard"),
        },
    }


def _validate_deterministic_runtime_payload(payload: Any) -> Mapping[str, Any]:
    expected = {
        "cublas_workspace_config",
        "cuda_matmul_allow_tf32",
        "cuda_visible_devices",
        "cudnn_allow_tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_version",
        "deterministic_algorithms",
        "device",
        "device_capability",
        "device_index",
        "device_name",
        "float32_matmul_precision",
        "local_rank",
        "logical_rank",
        "python_version",
        "torch_cuda_version",
        "torch_version",
        "world_size",
    }
    runtime = _exact_keys(payload, expected, "deterministic runtime")
    fixed = {
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "float32_matmul_precision": "highest",
    }
    if any(runtime[name] != value for name, value in fixed.items()):
        raise F2ProductionRunnerError("deterministic runtime contract drifted")
    state = DeterministicRuntimeState(
        logical_rank=runtime["logical_rank"],
        world_size=runtime["world_size"],
        local_rank=runtime["local_rank"],
        device=runtime["device"],
        device_index=runtime["device_index"],
        cuda_visible_devices=runtime["cuda_visible_devices"],
        torch_version=runtime["torch_version"],
        torch_cuda_version=runtime["torch_cuda_version"],
        cudnn_version=runtime["cudnn_version"],
        device_name=runtime["device_name"],
        device_capability=tuple(runtime["device_capability"]),
        python_version=runtime["python_version"],
    )
    if state.to_payload() != dict(runtime):
        raise F2ProductionRunnerError("deterministic runtime record is inconsistent")
    return runtime


def read_deterministic_runtime_record(
    path: Path | str,
    *,
    expected_run_root_sha256: str,
    shard_path: Path | str,
    evaluation: CausalShardEvaluation,
) -> Mapping[str, Any]:
    record_path = Path(path).resolve(strict=True)
    shard = Path(shard_path).resolve(strict=True)
    decoded, payload, _ = _read_canonical_file(
        record_path, name="deterministic runtime record"
    )
    document = _exact_keys(
        decoded,
        {
            "dataset",
            "identity",
            "kind",
            "partition",
            "runtime",
            "schema_version",
            "shard",
        },
        "deterministic runtime record",
    )
    if (
        document["schema_version"] != RUNTIME_RECORD_SCHEMA_VERSION
        or document["kind"] != "deterministic_runtime_record"
        or document["dataset"] != DATASET
        or document["partition"] != PARTITION
    ):
        raise F2ProductionRunnerError("deterministic runtime record contract drifted")
    identity = _exact_keys(
        document["identity"],
        {
            "artifact_sha256",
            "checkpoint_epoch",
            "checkpoint_sha256",
            "manifest_sha256",
            "numeric_mode",
            "run_root_sha256",
        },
        "runtime record identity",
    )
    expected_identity = {
        "artifact_sha256": evaluation.artifact_sha256,
        "checkpoint_epoch": evaluation.checkpoint_epoch,
        "checkpoint_sha256": evaluation.checkpoint_sha256,
        "manifest_sha256": evaluation.manifest_sha256,
        "numeric_mode": evaluation.numeric_mode,
        "run_root_sha256": _sha256(
            expected_run_root_sha256, "expected production run-root"
        ),
    }
    if dict(identity) != expected_identity:
        raise F2ProductionRunnerError("runtime record identity drifted")
    shard_document = _exact_keys(
        document["shard"], {"file_name", "file_sha256"}, "runtime record shard"
    )
    shard_payload, shard_digest = read_bytes_with_sidecar(shard)
    if (
        shard_document["file_name"] != shard.name
        or _sha256(shard_document["file_sha256"], "runtime record shard")
        != shard_digest
        or shard_evaluation_bytes(evaluation) != shard_payload
    ):
        raise F2ProductionRunnerError("runtime record shard binding drifted")
    runtime = _validate_deterministic_runtime_payload(document["runtime"])
    if runtime["logical_rank"] != evaluation.rank or runtime[
        "world_size"
    ] != evaluation.world_size:
        raise F2ProductionRunnerError("runtime record logical rank drifted")
    if canonical_json_bytes(decoded) != payload:
        raise F2ProductionRunnerError("runtime record is not idempotently canonical")
    return runtime


def _validate_causal_identity_against_run_root(
    evaluation: Any,
    verified: VerifiedProductionRunRoot,
) -> None:
    root = verified.root
    epoch = getattr(evaluation, "checkpoint_epoch", None)
    if epoch is None:
        epoch = getattr(evaluation, "epoch", None)
    if epoch not in F2_CHECKPOINT_EPOCHS:
        raise F2ProductionRunnerError("causal evaluation epoch is outside F2")
    expected = {
        "artifact_sha256": root.artifact_root_sha256,
        "checkpoint_sha256": root.checkpoint_sha256(epoch),
        "config_sha256": verified.bindings.runtime_config_sha256,
        "manifest_sha256": root.causal_manifest_sha256,
        "numeric_mode": F2_MODEL_NUMERIC_MODE,
        "source_sha256": verified.bindings.runtime_source_manifest_sha256,
    }
    observed = {
        "artifact_sha256": getattr(evaluation, "artifact_sha256", None),
        "checkpoint_sha256": getattr(evaluation, "checkpoint_sha256", None),
        "config_sha256": getattr(evaluation, "config_sha256", None),
        "manifest_sha256": getattr(evaluation, "manifest_sha256", None),
        "numeric_mode": getattr(evaluation, "numeric_mode", None),
        "source_sha256": getattr(evaluation, "source_sha256", None),
    }
    if observed != expected:
        raise F2ProductionRunnerError(
            "causal evaluation identity differs from external run-root"
        )


def _validate_proof_identity_against_run_root(
    proof: InvariantCheckpointEvaluation,
    verified: VerifiedProductionRunRoot,
) -> None:
    if not isinstance(proof, InvariantCheckpointEvaluation):
        raise F2ProductionRunnerError("invariance proof is malformed")
    root = verified.root
    evaluation = proof.evaluation
    expected = {
        "artifact_sha256": root.artifact_root_sha256,
        "checkpoint_sha256": root.checkpoint_sha256(evaluation.epoch),
        "config_sha256": verified.bindings.runtime_config_sha256,
        "manifest_sha256": root.causal_manifest_sha256,
        "numeric_mode": F2_MODEL_NUMERIC_MODE,
        "source_sha256": verified.bindings.runtime_source_manifest_sha256,
    }
    observed = {
        "artifact_sha256": evaluation.artifact_sha256,
        "checkpoint_sha256": evaluation.checkpoint_sha256,
        "config_sha256": proof.config_sha256,
        "manifest_sha256": evaluation.manifest_sha256,
        "numeric_mode": evaluation.numeric_mode,
        "source_sha256": proof.source_sha256,
    }
    if observed != expected:
        raise F2ProductionRunnerError(
            "invariance proof identity differs from artifact bindings or run-root"
        )


def _validate_ranking_identity_against_run_root(
    result: ProductionCheckpointRanking,
    verified: VerifiedProductionRunRoot,
) -> None:
    expected = {
        "artifact_sha256": verified.root.artifact_root_sha256,
        "config_sha256": verified.bindings.runtime_config_sha256,
        "manifest_sha256": verified.root.causal_manifest_sha256,
        "source_sha256": verified.bindings.runtime_source_manifest_sha256,
    }
    observed = {
        "artifact_sha256": result.artifact_sha256,
        "config_sha256": result.config_sha256,
        "manifest_sha256": result.manifest_sha256,
        "source_sha256": result.source_sha256,
    }
    if observed != expected:
        raise F2ProductionRunnerError(
            "ranking identity differs from artifact bindings or run-root"
        )


def _validate_complete_metrics_against_run_root(
    metrics: AggregateMetrics,
    root: ProductionRunRoot,
) -> None:
    if (
        metrics.expected_token_count != root.causal_record_count
        or metrics.observed_token_count != root.causal_record_count
        or metrics.fatal_count != 0
        or metrics.coverage_fraction != 1.0
    ):
        raise F2ProductionRunnerError(
            "causal evaluation does not cover the complete run-root manifest"
        )


def _runtime_receipt_identity_sha256(document: Any) -> str:
    value = _exact_keys(
        document,
        {
            "dataset",
            "identity",
            "kind",
            "partition",
            "runtime",
            "schema_version",
            "shard",
        },
        "deterministic runtime record",
    )
    identity = _exact_keys(
        value["identity"],
        {
            "artifact_sha256",
            "checkpoint_epoch",
            "checkpoint_sha256",
            "manifest_sha256",
            "numeric_mode",
            "run_root_sha256",
        },
        "runtime receipt identity",
    )
    runtime = _validate_deterministic_runtime_payload(value["runtime"])
    return _sha256_bytes(
        canonical_json_bytes(
            {
                "identity": dict(identity),
                "logical_rank": runtime["logical_rank"],
                "world_size": runtime["world_size"],
            }
        )
    )


def _authenticate_production_shard(
    shard_path: Path,
    *,
    verified: VerifiedProductionRunRoot,
    output_parent: Path,
) -> Tuple[CausalShardEvaluation, ProductionShardReference]:
    shard_file = _require_sibling(shard_path, output_parent, "production shard")
    shard_payload, shard_digest = read_bytes_with_sidecar(shard_file)
    shard = parse_shard_evaluation_bytes(shard_payload)
    _validate_causal_identity_against_run_root(shard, verified)

    runtime_path = _require_sibling(
        runtime_record_path_for_shard(shard_file),
        output_parent,
        "deterministic runtime receipt",
    )
    runtime_payload, runtime_digest = read_bytes_with_sidecar(runtime_path)
    read_deterministic_runtime_record(
        runtime_path,
        expected_run_root_sha256=verified.run_root_sha256,
        shard_path=shard_file,
        evaluation=shard,
    )
    runtime_document = parse_canonical_json_bytes(
        runtime_payload, name="deterministic runtime receipt"
    )
    reference = ProductionShardReference(
        rank=shard.rank,
        world_size=shard.world_size,
        shard_file_name=shard_file.name,
        shard_file_sha256=shard_digest,
        shard_evaluation_sha256=shard.evaluation_sha256,
        runtime_record_file_name=runtime_path.name,
        runtime_record_file_sha256=runtime_digest,
        runtime_identity_sha256=_runtime_receipt_identity_sha256(runtime_document),
    )
    return shard, reference


def read_production_world_evaluation(
    path: Path | str,
    *,
    verified: VerifiedProductionRunRoot,
) -> ProductionWorldEvaluation:
    """Recursively replay a run-root-bound world down to rank receipts."""

    certificate_path = Path(path).resolve(strict=True)
    decoded, payload, _ = _read_canonical_file(
        certificate_path, name="production world certificate"
    )
    document = _exact_keys(
        decoded,
        {
            "dataset",
            "kind",
            "partition",
            "run_root_sha256",
            "schema_version",
            "shards",
            "world_evaluation",
        },
        "production world certificate",
    )
    if (
        document["schema_version"] != RUN_ROOT_WORLD_SCHEMA_VERSION
        or document["kind"] != "run_root_causal_world"
        or document["dataset"] != DATASET
        or document["partition"] != PARTITION
    ):
        raise F2ProductionRunnerError("production world certificate contract drifted")
    run_root_sha = _sha256(document["run_root_sha256"], "production run-root")
    if run_root_sha != verified.run_root_sha256:
        raise F2ProductionRunnerError(
            "production world certificate run-root identity drifted"
        )
    if not isinstance(document["shards"], list):
        raise F2ProductionRunnerError(
            "production world certificate shards must be a JSON array"
        )
    references = tuple(
        _parse_production_shard_reference(item) for item in document["shards"]
    )
    embedded = parse_world_evaluation_bytes(
        canonical_json_bytes(document["world_evaluation"])
    )
    _validate_causal_identity_against_run_root(embedded, verified)
    shards = []
    replayed_references = []
    for reference in references:
        shard, replayed = _authenticate_production_shard(
            certificate_path.parent / reference.shard_file_name,
            verified=verified,
            output_parent=certificate_path.parent,
        )
        if replayed != reference:
            raise F2ProductionRunnerError(
                "production world shard or runtime receipt reference drifted"
            )
        shards.append(shard)
        replayed_references.append(replayed)
    if tuple(sorted(item.rank for item in shards)) != tuple(
        range(embedded.world_size)
    ):
        raise F2ProductionRunnerError(
            "production world recursive rank coverage is incomplete"
        )
    try:
        replayed_world = merge_causal_world(verified.live_provenance.causal, shards)
    except (F2CausalShardRunnerError, CausalValidationError) as error:
        raise F2ProductionRunnerError(
            "production world recursive shard merge failed"
        ) from error
    if world_evaluation_bytes(replayed_world) != world_evaluation_bytes(embedded):
        raise F2ProductionRunnerError(
            "production world differs from recursively replayed rank shards"
        )
    if len(embedded.results) != verified.root.causal_record_count:
        raise F2ProductionRunnerError("production world result count is incomplete")
    _validate_complete_metrics_against_run_root(embedded.metrics, verified.root)
    certificate = ProductionWorldEvaluation(
        evaluation=embedded,
        run_root_sha256=run_root_sha,
        shards=tuple(replayed_references),
        _seal=_PRODUCTION_WORLD_SEAL,
    )
    if production_world_evaluation_bytes(certificate) != payload:
        raise F2ProductionRunnerError(
            "production world certificate differs from recursive replay"
        )
    return certificate


def merge_production_world_files(
    *,
    run_root_path: Path | str,
    run_root_signature_path: Path | str,
    expected_run_root_sha256: str,
    world_size: int,
    shard_paths: Sequence[Path | str],
    output_path: Path | str,
) -> ProductionWorldEvaluation:
    """Replay signed authority and persist a recursively verifiable world."""

    verified = replay_production_run_root(
        run_root_path,
        signature_path=run_root_signature_path,
        expected_sha256=expected_run_root_sha256,
    )
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise F2ProductionRunnerError("world_size must be 1, 2, or 8")
    output = Path(output_path)
    if not output.parent.is_dir():
        raise F2ProductionRunnerError("production world output parent does not exist")
    if len(shard_paths) != world_size:
        raise F2ProductionRunnerError(
            "production world requires exactly one shard per rank"
        )
    shards = []
    references = []
    for raw_path in shard_paths:
        shard, reference = _authenticate_production_shard(
            Path(raw_path),
            verified=verified,
            output_parent=output.parent,
        )
        if shard.world_size != world_size:
            raise F2ProductionRunnerError("shard world_size differs from merge request")
        shards.append(shard)
        references.append(reference)
    if tuple(sorted(item.rank for item in shards)) != tuple(range(world_size)):
        raise F2ProductionRunnerError(
            "production world rank coverage is incomplete or duplicated"
        )
    try:
        world = merge_causal_world(verified.live_provenance.causal, shards)
    except (F2CausalShardRunnerError, CausalValidationError) as error:
        raise F2ProductionRunnerError("production causal world merge failed") from error
    _validate_causal_identity_against_run_root(world, verified)
    if len(world.results) != verified.root.causal_record_count:
        raise F2ProductionRunnerError("merged world result count is incomplete")
    _validate_complete_metrics_against_run_root(world.metrics, verified.root)
    certificate = ProductionWorldEvaluation(
        evaluation=world,
        run_root_sha256=verified.run_root_sha256,
        shards=tuple(references),
        _seal=_PRODUCTION_WORLD_SEAL,
    )
    write_bytes_no_overwrite(output, production_world_evaluation_bytes(certificate))
    return certificate


def _production_inner_proof(
    worlds: Sequence[ProductionWorldEvaluation],
    references: Sequence[ProductionWorldFileReference],
) -> InvariantCheckpointEvaluation:
    try:
        world1 = prove_1_2_8_invariance(
            [certificate.evaluation for certificate in worlds]
        )
    except (F2CausalShardRunnerError, CausalValidationError) as error:
        raise F2ProductionRunnerError(
            "production 1/2/8 causal invariance proof failed"
        ) from error
    return InvariantCheckpointEvaluation(
        evaluation=world1.checkpoint_evaluation,
        config_sha256=world1.config_sha256,
        source_sha256=world1.source_sha256,
        evaluation_sha256=world1.evaluation_sha256,
        metrics_sha256=world1.metrics_sha256,
        worlds=tuple(
            WorldFileReference(
                world_size=item.world_size,
                file_name=item.file_name,
                file_sha256=item.file_sha256,
                evaluation_sha256=item.evaluation_sha256,
                metrics_sha256=item.metrics_sha256,
            )
            for item in references
        ),
        _seal=_INVARIANCE_SEAL,
    )


def read_production_checkpoint_evaluation(
    path: Path | str,
    *,
    verified: VerifiedProductionRunRoot,
) -> ProductionInvariantCheckpointEvaluation:
    """Replay a production proof, all worlds, ranks, and runtime receipts."""

    certificate_path = Path(path).resolve(strict=True)
    decoded, payload, _ = _read_canonical_file(
        certificate_path, name="production invariant checkpoint certificate"
    )
    document = _exact_keys(
        decoded,
        {
            "checkpoint_evaluation",
            "dataset",
            "evaluation_sha256",
            "identity",
            "invariance_proved",
            "kind",
            "metrics_sha256",
            "partition",
            "run_root_sha256",
            "schema_version",
            "worlds",
        },
        "production invariant checkpoint certificate",
    )
    if (
        document["schema_version"] != RUN_ROOT_CHECKPOINT_SCHEMA_VERSION
        or document["kind"] != "run_root_invariant_checkpoint"
        or document["dataset"] != DATASET
        or document["partition"] != PARTITION
        or document["invariance_proved"] is not True
    ):
        raise F2ProductionRunnerError(
            "production checkpoint certificate contract drifted"
        )
    run_root_sha = _sha256(document["run_root_sha256"], "production run-root")
    if run_root_sha != verified.run_root_sha256:
        raise F2ProductionRunnerError(
            "production checkpoint certificate run-root identity drifted"
        )
    identity = _parse_identity(document["identity"])
    claimed_evaluation = _parse_checkpoint_payload(document["checkpoint_evaluation"])
    if not isinstance(document["worlds"], list):
        raise F2ProductionRunnerError(
            "production checkpoint worlds must be a JSON array"
        )
    references = tuple(
        _parse_production_world_reference(item) for item in document["worlds"]
    )
    worlds = []
    for reference in references:
        world_path = _require_sibling(
            certificate_path.parent / reference.file_name,
            certificate_path.parent,
            "production world certificate",
        )
        _, digest = read_bytes_with_sidecar(world_path)
        if digest != reference.file_sha256:
            raise F2ProductionRunnerError(
                "production proof world certificate SHA-256 drifted"
            )
        world = read_production_world_evaluation(world_path, verified=verified)
        if (
            world.run_root_sha256 != reference.run_root_sha256
            or world.world_size != reference.world_size
            or world.evaluation.evaluation_sha256 != reference.evaluation_sha256
            or world.evaluation.metrics_sha256 != reference.metrics_sha256
        ):
            raise F2ProductionRunnerError(
                "production proof world certificate identity drifted"
            )
        worlds.append(world)
    proof = _production_inner_proof(worlds, references)
    _validate_proof_identity_against_run_root(proof, verified)
    if (
        proof.evaluation != claimed_evaluation
        or proof.evaluation.epoch != identity["checkpoint_epoch"]
        or proof.evaluation.checkpoint_sha256 != identity["checkpoint_sha256"]
        or proof.evaluation.manifest_sha256 != identity["manifest_sha256"]
        or proof.evaluation.artifact_sha256 != identity["artifact_sha256"]
        or proof.evaluation.numeric_mode != identity["numeric_mode"]
        or proof.config_sha256 != identity["config_sha256"]
        or proof.source_sha256 != identity["source_sha256"]
        or proof.evaluation_sha256
        != _sha256(document["evaluation_sha256"], "evaluation")
        or proof.metrics_sha256 != _sha256(document["metrics_sha256"], "metrics")
    ):
        raise F2ProductionRunnerError(
            "production checkpoint differs from recursively replayed worlds"
        )
    _validate_complete_metrics_against_run_root(
        proof.evaluation.metrics, verified.root
    )
    certificate = ProductionInvariantCheckpointEvaluation(
        proof=proof,
        run_root_sha256=run_root_sha,
        worlds=references,
        _seal=_PRODUCTION_INVARIANCE_SEAL,
    )
    if production_checkpoint_evaluation_bytes(certificate) != payload:
        raise F2ProductionRunnerError(
            "production checkpoint certificate differs from recursive replay"
        )
    return certificate


def prove_production_invariance_files(
    *,
    run_root_path: Path | str,
    run_root_signature_path: Path | str,
    expected_run_root_sha256: str,
    world_paths: Sequence[Path | str],
    output_path: Path | str,
) -> ProductionInvariantCheckpointEvaluation:
    """Recursively replay production worlds and persist a run-root proof."""

    verified = replay_production_run_root(
        run_root_path,
        signature_path=run_root_signature_path,
        expected_sha256=expected_run_root_sha256,
    )
    output = Path(output_path)
    if not output.parent.is_dir():
        raise F2ProductionRunnerError("invariance output parent does not exist")
    if len(world_paths) != 3:
        raise F2ProductionRunnerError("invariance proof requires three world files")
    worlds = []
    references = []
    for raw_path in world_paths:
        world_path = _require_sibling(
            Path(raw_path), output.parent, "production world certificate"
        )
        _, digest = read_bytes_with_sidecar(world_path)
        world = read_production_world_evaluation(world_path, verified=verified)
        worlds.append(world)
        references.append(
            ProductionWorldFileReference(
                world_size=world.world_size,
                file_name=world_path.name,
                file_sha256=digest,
                run_root_sha256=world.run_root_sha256,
                evaluation_sha256=world.evaluation.evaluation_sha256,
                metrics_sha256=world.evaluation.metrics_sha256,
            )
        )
    proof = _production_inner_proof(worlds, references)
    _validate_proof_identity_against_run_root(proof, verified)
    _validate_complete_metrics_against_run_root(
        proof.evaluation.metrics, verified.root
    )
    certificate = ProductionInvariantCheckpointEvaluation(
        proof=proof,
        run_root_sha256=verified.run_root_sha256,
        worlds=tuple(references),
        _seal=_PRODUCTION_INVARIANCE_SEAL,
    )
    write_bytes_no_overwrite(
        output, production_checkpoint_evaluation_bytes(certificate)
    )
    return certificate


def rank_production_checkpoint_files(
    *,
    run_root_path: Path | str,
    run_root_signature_path: Path | str,
    expected_run_root_sha256: str,
    checkpoint_paths: Sequence[Path | str],
    output_path: Path | str,
) -> ProductionCheckpointRanking:
    """Recursively replay four proofs/worlds/ranks before formal ranking."""

    verified = replay_production_run_root(
        run_root_path,
        signature_path=run_root_signature_path,
        expected_sha256=expected_run_root_sha256,
    )
    output = Path(output_path)
    if not output.parent.is_dir():
        raise F2ProductionRunnerError("ranking output parent does not exist")
    if len(checkpoint_paths) != 4:
        raise F2ProductionRunnerError("ranking requires four checkpoint certificates")
    certificates = []
    references = []
    for raw_path in checkpoint_paths:
        checkpoint_path = _require_sibling(
            Path(raw_path), output.parent, "production checkpoint certificate"
        )
        _, digest = read_bytes_with_sidecar(checkpoint_path)
        certificate = read_production_checkpoint_evaluation(
            checkpoint_path, verified=verified
        )
        certificates.append(certificate)
        references.append(
            CheckpointFileReference(
                epoch=certificate.evaluation.epoch,
                file_name=checkpoint_path.name,
                file_sha256=digest,
                checkpoint_sha256=certificate.evaluation.checkpoint_sha256,
                evaluation_sha256=certificate.evaluation_sha256,
                metrics_sha256=certificate.metrics_sha256,
            )
        )
    if tuple(sorted(item.evaluation.epoch for item in certificates)) != (
        F2_CHECKPOINT_EPOCHS
    ):
        raise F2ProductionRunnerError("ranking epochs must be exactly 0, 1, 2, and 3")
    result = rank_invariant_checkpoints(
        [item.proof for item in certificates], references=references
    )
    _validate_ranking_identity_against_run_root(result, verified)
    write_bytes_no_overwrite(
        output,
        production_checkpoint_ranking_bytes(
            result, run_root_sha256=verified.run_root_sha256
        ),
    )
    return result


def read_production_checkpoint_ranking(
    path: Path | str,
    *,
    verified: VerifiedProductionRunRoot,
) -> ProductionCheckpointRanking:
    """Replay a persisted ranking through proofs, worlds, shards, and receipts."""

    ranking_path = Path(path).resolve(strict=True)
    decoded, payload, _ = _read_canonical_file(
        ranking_path, name="production checkpoint ranking"
    )
    document = _exact_keys(
        decoded,
        {
            "dataset",
            "kind",
            "partition",
            "ranking",
            "run_root_sha256",
            "schema_version",
        },
        "production checkpoint ranking",
    )
    if (
        document["schema_version"] != RUN_ROOT_RANKING_SCHEMA_VERSION
        or document["kind"] != "run_root_checkpoint_ranking"
        or document["dataset"] != DATASET
        or document["partition"] != PARTITION
    ):
        raise F2ProductionRunnerError("production ranking contract drifted")
    run_root_sha = _sha256(document["run_root_sha256"], "production run-root")
    if run_root_sha != verified.run_root_sha256:
        raise F2ProductionRunnerError("production ranking run-root identity drifted")
    ranking_document = _exact_keys(
        document["ranking"],
        {
            "dataset",
            "evaluation_sha256",
            "identity",
            "inputs",
            "kind",
            "metrics_sha256",
            "partition",
            "ranking",
            "schema_version",
        },
        "embedded checkpoint ranking",
    )
    if (
        ranking_document["schema_version"] != RANKING_SCHEMA_VERSION
        or ranking_document["kind"] != "checkpoint_ranking"
        or ranking_document["dataset"] != DATASET
        or ranking_document["partition"] != PARTITION
        or not isinstance(ranking_document["inputs"], list)
    ):
        raise F2ProductionRunnerError(
            "embedded production ranking contract drifted"
        )
    references = tuple(
        _parse_checkpoint_reference(item) for item in ranking_document["inputs"]
    )
    certificates = []
    for reference in references:
        checkpoint_path = _require_sibling(
            ranking_path.parent / reference.file_name,
            ranking_path.parent,
            "production checkpoint certificate",
        )
        _, digest = read_bytes_with_sidecar(checkpoint_path)
        if digest != reference.file_sha256:
            raise F2ProductionRunnerError(
                "production ranking checkpoint file SHA-256 drifted"
            )
        certificate = read_production_checkpoint_evaluation(
            checkpoint_path, verified=verified
        )
        if (
            certificate.evaluation.epoch != reference.epoch
            or certificate.evaluation.checkpoint_sha256
            != reference.checkpoint_sha256
            or certificate.evaluation_sha256 != reference.evaluation_sha256
            or certificate.metrics_sha256 != reference.metrics_sha256
        ):
            raise F2ProductionRunnerError(
                "production ranking checkpoint reference drifted"
            )
        certificates.append(certificate)
    result = rank_invariant_checkpoints(
        [item.proof for item in certificates], references=references
    )
    _validate_ranking_identity_against_run_root(result, verified)
    if checkpoint_ranking_payload(result) != dict(ranking_document):
        raise F2ProductionRunnerError(
            "production ranking differs from recursively replayed proofs"
        )
    if production_checkpoint_ranking_bytes(
        result, run_root_sha256=run_root_sha
    ) != payload:
        raise F2ProductionRunnerError(
            "production ranking bytes differ from recursive replay"
        )
    return result


def evaluate_production_shard(
    *,
    run_root_path: Path | str,
    run_root_signature_path: Path | str,
    expected_run_root_sha256: str,
    epoch: int,
    rank: int,
    world_size: int,
    output_path: Path | str,
    device: Optional[str] = None,
) -> CausalShardEvaluation:
    """Construct only official adapters and safely persist one rank output.

    Every identity originates in the offline-authority-signed production run-root.
    The merged artifact certificate is replayed before use; the binary's own
    descriptor is never promoted into an expected identity.  Logical ``rank``
    controls data assignment only.  CUDA placement comes from ``LOCAL_RANK``
    or the explicit physical ``device`` argument.
    """

    if world_size not in SUPPORTED_WORLD_SIZES:
        raise F2ProductionRunnerError("world_size must be 1, 2, or 8")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
        raise F2ProductionRunnerError("rank is outside world_size")
    # Authenticate the small root before touching CUDA, then establish the
    # deterministic CUDA contract before any model or feature adapter can
    # create a context.  Full run-root replay follows under that contract.
    _, run_root_digest = read_production_run_root(
        run_root_path,
        signature_path=run_root_signature_path,
        expected_sha256=expected_run_root_sha256,
    )
    runtime_state = initialize_deterministic_runtime(
        logical_rank=rank,
        world_size=world_size,
        explicit_device=device,
    )
    verified = replay_production_run_root(
        run_root_path,
        signature_path=run_root_signature_path,
        expected_sha256=expected_run_root_sha256,
    )
    root = verified.root
    bindings = verified.bindings
    config = verified.runtime_config
    live = verified.live_provenance
    merged = verified.merged_artifact
    paths = config.path_map
    checkpoint = config.checkpoint(epoch)
    if checkpoint.sha256 != root.checkpoint_sha256(epoch):
        raise F2ProductionRunnerError("selected checkpoint differs from run-root")
    output = Path(output_path).resolve(strict=False)
    runtime_record = runtime_record_path_for_shard(output)
    if runtime_record.exists() or sha256_sidecar_path(runtime_record).exists():
        raise F2ProductionRunnerError(
            "deterministic runtime record output already exists"
        )
    try:
        model_adapter = build_official_f2_causal_model_adapter(
            checkpoint.path,
            expected_checkpoint_sha256=checkpoint.sha256,
            expected_epoch=checkpoint.epoch,
            repository_root=paths["repository_root"],
            experiment_root=paths["experiment_root"],
            live_provenance=live,
            device=runtime_state.device,
        )
        feature_build = build_qualified_held_out_navtrain_feature_source(
            live.qualification,
            live_provenance=live,
            navsim_log_path=paths["navsim_log"],
            sensor_blobs_path=paths["sensor_blobs"],
            authenticated_raw_inputs=verified.raw_inputs,
            rank=rank,
            world_size=world_size,
        )
        comfort = DriveJepaExtendedComfort.official()
        evaluation = evaluate_causal_shard(
            manifest=live.causal,
            artifact_path=merged.artifact_path,
            artifact_bindings=bindings,
            artifact_coverage=build_artifact_coverage(
                CoverageEntry(row.log_name, row.time_us, row.token)
                for row in live.causal.rows
            ),
            expected_artifact_sha256=root.artifact_root_sha256,
            live_provenance=live,
            model_adapter=model_adapter,
            feature_source=feature_build.source,
            comfort_function=comfort,
            rank=rank,
            world_size=world_size,
            reader_mode=config.reader_mode,
        )
    except F2ProductionRunnerError:
        raise
    except Exception as error:
        raise F2ProductionRunnerError(
            f"fatal production shard evaluation: {type(error).__name__}: {error}"
        ) from error
    _validate_causal_identity_against_run_root(evaluation, verified)
    shard_digest = write_shard_evaluation(output, evaluation)
    record_payload = deterministic_runtime_record_payload(
        state=runtime_state,
        run_root_sha256=run_root_digest,
        shard_path=output,
        shard_sha256=shard_digest,
        evaluation=evaluation,
    )
    write_bytes_no_overwrite(runtime_record, canonical_json_bytes(record_payload))
    return evaluation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed held-out navtrain F2 causal production runner"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_run_root_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument(
            "--run-root-signature",
            type=Path,
            required=True,
            help="offline authority SSHSIG under the production run-root namespace",
        )
        command.add_argument(
            "--expected-run-root-sha256",
            required=True,
            help="out-of-band lowercase SHA-256 pinned before this invocation",
        )

    evaluate = commands.add_parser("evaluate-shard")
    add_run_root_arguments(evaluate)
    evaluate.add_argument("--epoch", type=int, required=True)
    evaluate.add_argument("--rank", type=int, required=True)
    evaluate.add_argument("--world-size", type=int, required=True)
    evaluate.add_argument(
        "--device",
        help="physical CUDA device (cuda:<index>); otherwise LOCAL_RANK is required",
    )
    evaluate.add_argument("--output", type=Path, required=True)

    merge = commands.add_parser("merge-world")
    add_run_root_arguments(merge)
    merge.add_argument("--world-size", type=int, required=True)
    merge.add_argument("--shard", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)

    prove = commands.add_parser("prove-invariance")
    add_run_root_arguments(prove)
    prove.add_argument("--world-1", type=Path, required=True)
    prove.add_argument("--world-2", type=Path, required=True)
    prove.add_argument("--world-8", type=Path, required=True)
    prove.add_argument("--output", type=Path, required=True)

    rank = commands.add_parser("rank-checkpoints")
    add_run_root_arguments(rank)
    rank.add_argument("--checkpoint", type=Path, action="append", required=True)
    rank.add_argument("--output", type=Path, required=True)
    return parser


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate-shard":
            evaluate_production_shard(
                run_root_path=args.run_root,
                run_root_signature_path=args.run_root_signature,
                expected_run_root_sha256=args.expected_run_root_sha256,
                epoch=args.epoch,
                rank=args.rank,
                world_size=args.world_size,
                output_path=args.output,
                device=args.device,
            )
        elif args.command == "merge-world":
            merge_production_world_files(
                run_root_path=args.run_root,
                run_root_signature_path=args.run_root_signature,
                expected_run_root_sha256=args.expected_run_root_sha256,
                world_size=args.world_size,
                shard_paths=args.shard,
                output_path=args.output,
            )
        elif args.command == "prove-invariance":
            prove_production_invariance_files(
                run_root_path=args.run_root,
                run_root_signature_path=args.run_root_signature,
                expected_run_root_sha256=args.expected_run_root_sha256,
                world_paths=(args.world_1, args.world_2, args.world_8),
                output_path=args.output,
            )
        elif args.command == "rank-checkpoints":
            rank_production_checkpoint_files(
                run_root_path=args.run_root,
                run_root_signature_path=args.run_root_signature,
                expected_run_root_sha256=args.expected_run_root_sha256,
                checkpoint_paths=args.checkpoint,
                output_path=args.output,
            )
        else:  # pragma: no cover - argparse owns command validation.
            raise F2ProductionRunnerError("unknown production command")
    except F2ProductionRunnerError as error:
        parser.exit(2, f"F2 production runner failed: {error}\n")
    return 0


__all__ = (
    "BINDINGS_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "CUBLAS_WORKSPACE_CONFIG",
    "CheckpointFileReference",
    "DEVICE_POLICY",
    "DeterministicRuntimeState",
    "F2ProductionRunnerError",
    "InvariantCheckpointEvaluation",
    "ProductionCheckpointRanking",
    "ProductionInvariantCheckpointEvaluation",
    "ProductionRunRoot",
    "ProductionRuntimeConfig",
    "ProductionShardReference",
    "ProductionWorldEvaluation",
    "ProductionWorldFileReference",
    "RANKING_SCHEMA_VERSION",
    "RUN_ROOT_SCHEMA_VERSION",
    "RUN_ROOT_CHECKPOINT_SCHEMA_VERSION",
    "RUN_ROOT_RANKING_SCHEMA_VERSION",
    "RUN_ROOT_SSH_NAMESPACE",
    "RUN_ROOT_SSH_PRINCIPAL",
    "RUNTIME_CONFIG_SCHEMA_VERSION",
    "RUNTIME_RECORD_SCHEMA_VERSION",
    "RUN_ROOT_WORLD_SCHEMA_VERSION",
    "RunRootCheckpoint",
    "RuntimeCheckpoint",
    "SHARD_SCHEMA_VERSION",
    "WORLD_SCHEMA_VERSION",
    "artifact_bindings_bytes",
    "artifact_bindings_payload",
    "canonical_json_bytes",
    "checkpoint_evaluation_bytes",
    "checkpoint_evaluation_payload",
    "checkpoint_ranking_bytes",
    "checkpoint_ranking_payload",
    "cli_main",
    "deterministic_runtime_record_payload",
    "evaluate_production_shard",
    "initialize_deterministic_runtime",
    "merge_production_world_files",
    "merge_world_files",
    "parse_canonical_json_bytes",
    "parse_shard_evaluation_bytes",
    "parse_world_evaluation_bytes",
    "production_run_root_bytes",
    "production_run_root_payload",
    "production_checkpoint_evaluation_bytes",
    "production_checkpoint_evaluation_payload",
    "production_checkpoint_ranking_bytes",
    "production_checkpoint_ranking_payload",
    "production_world_evaluation_bytes",
    "production_world_evaluation_payload",
    "prove_production_invariance_files",
    "prove_invariance_files",
    "rank_production_checkpoint_files",
    "rank_invariant_checkpoints",
    "read_artifact_bindings",
    "read_bytes_with_sidecar",
    "read_checkpoint_evaluation",
    "read_checkpoint_ranking",
    "read_production_checkpoint_evaluation",
    "read_production_checkpoint_ranking",
    "read_deterministic_runtime_record",
    "read_production_run_root",
    "read_shard_evaluation",
    "read_world_evaluation",
    "read_production_world_evaluation",
    "replay_production_run_root",
    "resolve_production_device",
    "runtime_record_path_for_shard",
    "sha256_sidecar_path",
    "shard_evaluation_bytes",
    "shard_evaluation_payload",
    "world_evaluation_bytes",
    "world_evaluation_payload",
    "verify_production_run_root_signature",
    "write_artifact_bindings",
    "write_bytes_no_overwrite",
    "write_canonical_json_no_overwrite",
    "write_checkpoint_ranking",
    "write_production_run_root",
    "write_shard_evaluation",
    "write_world_evaluation",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
