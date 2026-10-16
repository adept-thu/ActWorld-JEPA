"""Synthetic CPU tests for fail-closed F2 production orchestration."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

from navsim.planning.validation.f2_causal_shard_runner import (
    CausalShardEvaluation,
    CausalWorldEvaluation,
)
from navsim.planning.validation.f2_causal_validator import (
    F2_MODEL_NUMERIC_MODE,
    ManifestRow,
    TokenCausalResult,
    aggregate_results,
    build_canonical_manifest,
    canonical_manifest_bytes,
    deterministic_log_assignments,
    manifest_sha256,
)
from navsim.planning.validation.f2_frozen_artifact_io import ArtifactBindings
from navsim.planning.validation.f2_production_runner import (
    CUBLAS_WORKSPACE_CONFIG,
    DEVICE_POLICY,
    DeterministicRuntimeState,
    F2ProductionRunnerError,
    ProductionRunRoot,
    RunRootCheckpoint,
    RUN_ROOT_SSH_NAMESPACE,
    RUNTIME_CONFIG_SCHEMA_VERSION,
    artifact_bindings_payload,
    canonical_json_bytes,
    deterministic_runtime_record_payload,
    initialize_deterministic_runtime,
    merge_production_world_files,
    merge_world_files,
    parse_canonical_json_bytes,
    prove_production_invariance_files,
    prove_invariance_files,
    rank_production_checkpoint_files,
    read_bytes_with_sidecar,
    read_artifact_bindings,
    read_checkpoint_evaluation,
    read_checkpoint_ranking,
    read_deterministic_runtime_record,
    read_production_run_root,
    read_production_checkpoint_ranking,
    replay_production_run_root,
    read_shard_evaluation,
    read_world_evaluation,
    sha256_sidecar_path,
    shard_evaluation_payload,
    resolve_production_device,
    runtime_record_path_for_shard,
    world_evaluation_payload,
    write_checkpoint_ranking,
    write_artifact_bindings,
    write_bytes_no_overwrite,
    write_production_run_root,
    write_shard_evaluation,
    write_world_evaluation,
    verify_production_run_root_signature,
)


def _sha(label: str) -> str:
    return hashlib.sha256((label + "\n").encode("utf-8")).hexdigest()


MANIFEST_SHA = _sha("manifest")
ARTIFACT_SHA = _sha("artifact")
CONFIG_SHA = _sha("config")
SOURCE_SHA = _sha("source")


def _synthetic_bindings(*, causal_manifest_sha256: str) -> ArtifactBindings:
    return ArtifactBindings(
        qualification_manifest_sha256=_sha("qualification"),
        causal_manifest_sha256=causal_manifest_sha256,
        baseline_checkpoint_sha256=_sha("baseline"),
        vjepa_checkpoint_sha256=_sha("vjepa"),
        cache_metadata_sha256=_sha("cache"),
        metric_cache_manifest_sha256=_sha("metric-cache"),
        raw_input_manifest_sha256=_sha("raw-inputs"),
        qualification_config_sha256=_sha("qualification-config"),
        runtime_config_sha256=CONFIG_SHA,
        qualification_source_manifest_sha256=_sha("qualification-source"),
        runtime_source_manifest_sha256=SOURCE_SHA,
        source_revision="synthetic-source",
        proposal_schema_name="proposal-schema",
        proposal_schema_sha256=_sha("proposal-schema"),
        simulator_name="simulator",
        simulator_sha256=_sha("simulator"),
        scorer_name="scorer",
        scorer_sha256=_sha("scorer"),
    )


def _result(
    *,
    token: str = "token-0",
    log_name: str = "log-0",
    time_us: int = 100_000,
    critic_truth: float = 0.66,
) -> TokenCausalResult:
    target = (0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 1.0, 0.7)
    return TokenCausalResult(
        log_name=log_name,
        time_us=time_us,
        token=token,
        future_camera_valid=True,
        critic_continuous=False,
        base_continuous=False,
        critic_selected_index=0,
        base_selected_index=0,
        critic_selected_q=critic_truth,
        base_selected_q=0.6,
        critic_selected_probability=critic_truth,
        base_selected_probability=0.6,
        critic_selected_truth=critic_truth,
        base_selected_truth=0.6,
        critic_target_scores=target,
        base_target_scores=target,
        oracle_aggregate=0.8,
        pairwise_correct=20,
        pairwise_comparable=20,
    )


def _world(
    world_size: int,
    *,
    epoch: int = 0,
    checkpoint_sha: str | None = None,
    critic_truth: float = 0.66,
) -> CausalWorldEvaluation:
    result = _result(critic_truth=critic_truth)
    metrics = aggregate_results((result,), expected_tokens=(result.token,), fatal_count=0)
    return CausalWorldEvaluation(
        world_size=world_size,
        checkpoint_sha256=checkpoint_sha or _sha(f"checkpoint-{epoch}"),
        checkpoint_epoch=epoch,
        manifest_sha256=MANIFEST_SHA,
        artifact_sha256=ARTIFACT_SHA,
        config_sha256=CONFIG_SHA,
        source_sha256=SOURCE_SHA,
        results=(result,),
        metrics=metrics,
        numeric_mode=F2_MODEL_NUMERIC_MODE,
    )


def _rewrite_canonical(path: Path, payload: Any) -> None:
    encoded = canonical_json_bytes(payload)
    path.write_bytes(encoded)
    sha256_sidecar_path(path).write_bytes(
        (hashlib.sha256(encoded).hexdigest() + "\n").encode("ascii")
    )


def test_strict_json_rejects_duplicate_nan_and_noncanonical() -> None:
    with pytest.raises(F2ProductionRunnerError, match="duplicate"):
        parse_canonical_json_bytes(b'{"x":1,"x":2}\n', name="test")
    with pytest.raises(F2ProductionRunnerError, match="non-finite"):
        parse_canonical_json_bytes(b'{"x":NaN}\n', name="test")
    with pytest.raises(F2ProductionRunnerError, match="not canonical"):
        parse_canonical_json_bytes(b'{ "x": 1 }\n', name="test")


def test_artifact_bindings_reject_uppercase_digest_bytes(tmp_path: Path) -> None:
    bindings = ArtifactBindings(
        qualification_manifest_sha256=_sha("qualification"),
        causal_manifest_sha256=_sha("causal"),
        baseline_checkpoint_sha256=_sha("baseline"),
        vjepa_checkpoint_sha256=_sha("vjepa"),
        cache_metadata_sha256=_sha("cache"),
        metric_cache_manifest_sha256=_sha("cache-content"),
        raw_input_manifest_sha256=_sha("raw-input-content"),
        qualification_config_sha256=_sha("qualification-config"),
        runtime_config_sha256=_sha("runtime-config"),
        qualification_source_manifest_sha256=_sha("qualification-source"),
        runtime_source_manifest_sha256=_sha("runtime-source"),
        source_revision="synthetic-source",
        proposal_schema_name="proposal-schema",
        proposal_schema_sha256=_sha("proposal-schema"),
        simulator_name="simulator",
        simulator_sha256=_sha("simulator"),
        scorer_name="scorer",
        scorer_sha256=_sha("scorer"),
    )
    path = tmp_path / "bindings.json"
    write_artifact_bindings(path, bindings)
    assert read_artifact_bindings(path) == bindings

    tampered = artifact_bindings_payload(bindings)
    tampered["bindings"]["manifests"]["causal_sha256"] = tampered[
        "bindings"
    ]["manifests"]["causal_sha256"].upper()
    _rewrite_canonical(path, tampered)
    with pytest.raises(F2ProductionRunnerError, match="lowercase hexadecimal"):
        read_artifact_bindings(path)


def _production_run_root(
    tmp_path: Path,
    *,
    causal_manifest_sha256: str | None = None,
    causal_record_count: int = 12817,
    causal_log_count: int = 151,
    artifact_root_sha256: str | None = None,
) -> ProductionRunRoot:
    return ProductionRunRoot(
        bindings_path=(tmp_path / "bindings.json").resolve(),
        bindings_sha256=_sha("bindings-file"),
        launch_authorization_path=(tmp_path / "authorization.json").resolve(),
        launch_authorization_sha256=_sha("authorization"),
        launch_authorization_signature_path=(
            tmp_path / "authorization.json.sig"
        ).resolve(),
        launch_authorization_signature_sha256=_sha("authorization-signature"),
        runtime_claim_path=(tmp_path / "runtime-claim.json").resolve(),
        runtime_claim_sha256=_sha("runtime-claim"),
        training_lineage_path=(
            tmp_path / "f2-output" / "f2_training_lineage.json"
        ).resolve(),
        training_lineage_sha256=_sha("training-lineage"),
        output_dir=(tmp_path / "f2-output").resolve(),
        runtime_config_path=(tmp_path / "runtime.json").resolve(),
        runtime_config_sha256=_sha("runtime-file"),
        raw_input_manifest_path=(tmp_path / "raw-inputs.json").resolve(),
        raw_input_manifest_sha256=_sha("raw-inputs"),
        causal_manifest_sha256=causal_manifest_sha256 or _sha("causal-manifest"),
        causal_record_count=causal_record_count,
        causal_log_count=causal_log_count,
        merged_artifact_certificate_path=(
            tmp_path / "merged-artifact.json"
        ).resolve(),
        merged_artifact_certificate_sha256=_sha("merged-certificate"),
        artifact_root_sha256=artifact_root_sha256 or _sha("artifact-root"),
        checkpoints=tuple(
            RunRootCheckpoint(epoch=epoch, sha256=_sha(f"checkpoint-{epoch}"))
            for epoch in range(4)
        ),
    )


def test_run_root_requires_external_digest_and_binds_complete_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import navsim.planning.validation.f2_production_runner as production

    root = _production_run_root(tmp_path)
    path = tmp_path / "run-root.json"
    signature_path = tmp_path / "run-root.json.sig"
    signature_path.write_bytes(b"synthetic-authority-signature")
    verified_messages: list[bytes] = []
    monkeypatch.setattr(
        production,
        "verify_production_run_root_signature",
        lambda message, signature: verified_messages.append(message),
    )
    digest = write_production_run_root(path, root)
    replayed, replayed_digest = read_production_run_root(
        path,
        signature_path=signature_path,
        expected_sha256=digest,
    )
    assert replayed == root
    assert replayed_digest == digest
    assert replayed.causal_record_count == 12817
    assert replayed.causal_log_count == 151
    assert tuple(item.epoch for item in replayed.checkpoints) == (0, 1, 2, 3)
    assert verified_messages == [path.read_bytes()]

    with pytest.raises(F2ProductionRunnerError, match="external expectation"):
        read_production_run_root(
            path,
            signature_path=signature_path,
            expected_sha256=_sha("attacker-self-sign"),
        )

    decoded = json.loads(path.read_text(encoding="utf-8"))
    decoded["merged_artifact"]["root_sha256"] = _sha("replacement-artifact")
    _rewrite_canonical(path, decoded)
    with pytest.raises(F2ProductionRunnerError, match="external expectation"):
        read_production_run_root(
            path,
            signature_path=signature_path,
            expected_sha256=digest,
        )


def test_run_root_rejects_noncanonical_checkpoint_set(tmp_path: Path) -> None:
    root = _production_run_root(tmp_path)
    with pytest.raises(F2ProductionRunnerError, match="exactly epochs"):
        ProductionRunRoot(
            **{
                **root.__dict__,
                "checkpoints": root.checkpoints[:3] + (root.checkpoints[2],),
            }
        )


def test_run_root_replays_launch_claim_and_rejects_checkpoint_outside_signed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import navsim.planning.validation.f2_production_runner as production

    output_dir = (tmp_path / "authorized-output").resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    outside_dir = (tmp_path / "attacker-checkpoints").resolve()
    outside_dir.mkdir()
    checkpoint_paths = []
    checkpoint_digests = []
    for epoch in range(4):
        parent = outside_dir if epoch == 3 else checkpoint_dir
        path = (parent / f"epoch={epoch}.ckpt").resolve()
        payload = f"checkpoint-{epoch}\n".encode("ascii")
        path.write_bytes(payload)
        checkpoint_paths.append(path)
        checkpoint_digests.append(hashlib.sha256(payload).hexdigest())

    runtime_paths = {
        name: str((tmp_path / name).resolve())
        for name in (
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
        )
    }
    runtime_paths["runtime_source_root"] = runtime_paths["repository_root"]
    runtime_paths["metric_cache_root"] = str(
        (tmp_path / "train_metric_cache_v2_pruned").resolve()
    )
    runtime_payload = canonical_json_bytes(
        {
            "checkpoints": [
                {
                    "epoch": epoch,
                    "path": str(checkpoint_paths[epoch]),
                    "sha256": checkpoint_digests[epoch],
                }
                for epoch in range(4)
            ],
            "dataset": "navtrain",
            "device_policy": DEVICE_POLICY,
            "partition": "held_out",
            "paths": runtime_paths,
            "reader_mode": "stream",
            "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
        }
    )
    runtime_path = (tmp_path / "runtime.json").resolve()
    runtime_path.write_bytes(runtime_payload)
    runtime_sha = hashlib.sha256(runtime_payload).hexdigest()

    bindings = replace(
        _synthetic_bindings(causal_manifest_sha256=MANIFEST_SHA),
        runtime_config_sha256=runtime_sha,
    )
    bindings_path = (tmp_path / "bindings.json").resolve()
    bindings_sha = write_artifact_bindings(bindings_path, bindings)
    authorization_path = (tmp_path / "authorization.json").resolve()
    authorization_signature_path = (
        tmp_path / "authorization.json.sig"
    ).resolve()
    authorization_path.write_bytes(b"signed authorization\n")
    authorization_signature_path.write_bytes(b"signed authorization sshsig\n")
    authorization_sha = hashlib.sha256(authorization_path.read_bytes()).hexdigest()
    authorization_signature_sha = hashlib.sha256(
        authorization_signature_path.read_bytes()
    ).hexdigest()
    claim_path = (tmp_path / "runtime-claim.json").resolve()
    claim_sha = _sha("runtime-claim")
    experiment_name = "ActWorld-JEPA-F2"
    experiment_uid = "synthetic-f2-run"
    boot_id = "11111111-2222-3333-4444-555555555555"
    launcher_pid = 4242
    launcher_start_time = 123456
    lineage_path = output_dir / "f2_training_lineage.json"
    output_metadata = output_dir.lstat()
    lineage_raw = canonical_json_bytes(
        {
            "authorization": {
                "sha256": authorization_sha,
                "signature_sha256": authorization_signature_sha,
            },
            "experiment": {
                "name": experiment_name,
                "output_dir": str(output_dir),
                "output_directory_identity": {
                    "device": output_metadata.st_dev,
                    "inode": output_metadata.st_ino,
                },
                "uid": experiment_uid,
            },
            "runtime_claim": {"path": str(claim_path), "sha256": claim_sha},
            "schema_version": "actworld-f2-training-lineage-v1",
            "source_revision": bindings.source_revision,
            "torchrun": {
                "boot_id": boot_id,
                "pid": launcher_pid,
                "start_time_ticks": launcher_start_time,
                "world_size": 8,
            },
        }
    )
    lineage_path.write_bytes(lineage_raw)
    lineage_sha = hashlib.sha256(lineage_raw).hexdigest()

    root = replace(
        _production_run_root(tmp_path),
        bindings_path=bindings_path,
        bindings_sha256=bindings_sha,
        launch_authorization_path=authorization_path,
        launch_authorization_sha256=authorization_sha,
        launch_authorization_signature_path=authorization_signature_path,
        launch_authorization_signature_sha256=authorization_signature_sha,
        runtime_claim_path=claim_path,
        runtime_claim_sha256=claim_sha,
        training_lineage_path=lineage_path,
        training_lineage_sha256=lineage_sha,
        output_dir=output_dir,
        runtime_config_path=runtime_path,
        runtime_config_sha256=runtime_sha,
        causal_manifest_sha256=MANIFEST_SHA,
        raw_input_manifest_sha256=bindings.raw_input_manifest_sha256,
        checkpoints=tuple(
            RunRootCheckpoint(epoch=epoch, sha256=checkpoint_digests[epoch])
            for epoch in range(4)
        ),
    )
    root_path = tmp_path / "production-run-root.json"
    root_signature_path = tmp_path / "production-run-root.json.sig"
    root_signature_path.write_bytes(b"synthetic production root sshsig")
    root_sha = write_production_run_root(root_path, root)
    monkeypatch.setattr(
        production, "verify_production_run_root_signature", lambda message, sig: None
    )
    signed_contract = SimpleNamespace(
        authorization_sha256=authorization_sha,
        signature_sha256=authorization_signature_sha,
        output_dir=output_dir,
        source_revision=bindings.source_revision,
        experiment_name=experiment_name,
        experiment_uid=experiment_uid,
        runtime={
            "repository_root": Path(runtime_paths["repository_root"]),
            "navsim_exp_root": Path(runtime_paths["experiment_root"]),
            "openscene_data_root": (tmp_path / "openscene").resolve(),
            "nuplan_maps_root": Path(runtime_paths["nuplan_maps_root"]),
            "navsim_log_root": Path(runtime_paths["navsim_log"]),
            "sensor_blobs_root": Path(runtime_paths["sensor_blobs"]),
            "qualification_source_root": Path(
                runtime_paths["qualification_source_root"]
            ),
        },
        artifacts={
            "baseline_checkpoint": SimpleNamespace(
                path=Path(runtime_paths["baseline_checkpoint"]),
                sha256=bindings.baseline_checkpoint_sha256,
            ),
            "vjepa_checkpoint": SimpleNamespace(
                path=Path(runtime_paths["vjepa_checkpoint"]),
                sha256=bindings.vjepa_checkpoint_sha256,
            ),
            "cache_metadata": SimpleNamespace(
                path=Path(runtime_paths["cache_metadata"]),
                sha256=bindings.cache_metadata_sha256,
            ),
            "cache_content_manifest": SimpleNamespace(
                path=Path(runtime_paths["metric_cache_manifest"]),
                sha256=bindings.metric_cache_manifest_sha256,
            ),
            "raw_input_manifest": SimpleNamespace(
                path=Path(runtime_paths["raw_input_manifest"]),
                sha256=bindings.raw_input_manifest_sha256,
            ),
            "qualification_manifest": SimpleNamespace(
                path=Path(runtime_paths["qualification_manifest"]),
                sha256=bindings.qualification_manifest_sha256,
            ),
            "qualification_config": SimpleNamespace(
                path=Path(runtime_paths["qualification_config"]),
                sha256=bindings.qualification_config_sha256,
            ),
            "qualification_source_manifest": SimpleNamespace(
                path=Path(runtime_paths["qualification_source_manifest"]),
                sha256=bindings.qualification_source_manifest_sha256,
            ),
            "feasibility_config": SimpleNamespace(
                path=(tmp_path / "feasibility-config.json").resolve(),
                sha256=_sha("feasibility-config"),
            ),
            "feasibility_audit": SimpleNamespace(
                path=(tmp_path / "feasibility-audit.json").resolve(),
                sha256=_sha("feasibility-audit"),
            ),
        },
    )
    runtime_claim = SimpleNamespace(
        sha256=claim_sha,
        authorization_sha256=authorization_sha,
        signature_sha256=authorization_signature_sha,
        output_dir=output_dir,
        launch_marker=claim_path,
        source_revision=bindings.source_revision,
        boot_id=boot_id,
        launcher_pid=launcher_pid,
        launcher_start_time_ticks=launcher_start_time,
    )
    launch_calls: list[str] = []
    monkeypatch.setattr(
        production,
        "verify_f2_signed_contract_lightweight",
        lambda *args, **kwargs: (launch_calls.append("authorization") or signed_contract),
    )
    monkeypatch.setattr(
        production,
        "verify_f2_runtime_claim",
        lambda *args, **kwargs: (launch_calls.append("claim") or runtime_claim),
    )
    import navsim.common.dataclasses as navsim_dataclasses

    monkeypatch.setattr(
        navsim_dataclasses,
        "NUPLAN_MAPS_ROOT",
        runtime_paths["nuplan_maps_root"],
    )
    with pytest.raises(F2ProductionRunnerError, match="outside the signed"):
        replay_production_run_root(
            root_path,
            signature_path=root_signature_path,
            expected_sha256=root_sha,
        )
    assert launch_calls == ["authorization", "claim"]

    monkeypatch.setattr(
        navsim_dataclasses,
        "NUPLAN_MAPS_ROOT",
        str((tmp_path / "attacker-maps").resolve()),
    )
    with pytest.raises(F2ProductionRunnerError, match="imported NUPLAN_MAPS_ROOT"):
        replay_production_run_root(
            root_path,
            signature_path=root_signature_path,
            expected_sha256=root_sha,
        )
    monkeypatch.setattr(
        navsim_dataclasses,
        "NUPLAN_MAPS_ROOT",
        runtime_paths["nuplan_maps_root"],
    )

    mismatched_artifacts = dict(signed_contract.artifacts)
    mismatched_artifacts["baseline_checkpoint"] = SimpleNamespace(
        path=Path(runtime_paths["baseline_checkpoint"]),
        sha256=_sha("authorization-for-asset-set-a"),
    )
    mismatched_contract = SimpleNamespace(
        **{**signed_contract.__dict__, "artifacts": mismatched_artifacts}
    )
    monkeypatch.setattr(
        production,
        "verify_f2_signed_contract_lightweight",
        lambda *args, **kwargs: mismatched_contract,
    )
    with pytest.raises(F2ProductionRunnerError, match="differs from evaluation bindings"):
        replay_production_run_root(
            root_path,
            signature_path=root_signature_path,
            expected_sha256=root_sha,
        )
    monkeypatch.setattr(
        production,
        "verify_f2_signed_contract_lightweight",
        lambda *args, **kwargs: signed_contract,
    )

    substituted_lineage = json.loads(lineage_raw.decode("utf-8"))
    substituted_lineage["runtime_claim"]["sha256"] = _sha("other-claim")
    lineage_path.write_bytes(canonical_json_bytes(substituted_lineage))
    with pytest.raises(F2ProductionRunnerError, match="lineage SHA-256"):
        replay_production_run_root(
            root_path,
            signature_path=root_signature_path,
            expected_sha256=root_sha,
        )
    substituted_root = replace(
        root,
        training_lineage_sha256=hashlib.sha256(
            lineage_path.read_bytes()
        ).hexdigest(),
    )
    substituted_root_path = tmp_path / "substituted-run-root.json"
    substituted_signature_path = tmp_path / "substituted-run-root.json.sig"
    substituted_signature_path.write_bytes(b"synthetic production root sshsig")
    substituted_root_sha = write_production_run_root(
        substituted_root_path, substituted_root
    )
    with pytest.raises(F2ProductionRunnerError, match="differs from signed"):
        replay_production_run_root(
            substituted_root_path,
            signature_path=substituted_signature_path,
            expected_sha256=substituted_root_sha,
        )

    identity_substitution = json.loads(lineage_raw.decode("utf-8"))
    identity_substitution["experiment"]["output_directory_identity"][
        "inode"
    ] += 1
    identity_substitution_raw = canonical_json_bytes(identity_substitution)
    lineage_path.write_bytes(identity_substitution_raw)
    identity_root = replace(
        root,
        training_lineage_sha256=hashlib.sha256(
            identity_substitution_raw
        ).hexdigest(),
    )
    identity_root_path = tmp_path / "identity-substituted-run-root.json"
    identity_signature_path = tmp_path / "identity-substituted-run-root.json.sig"
    identity_signature_path.write_bytes(b"synthetic production root sshsig")
    identity_root_sha = write_production_run_root(
        identity_root_path, identity_root
    )
    with pytest.raises(F2ProductionRunnerError, match="differs from signed"):
        replay_production_run_root(
            identity_root_path,
            signature_path=identity_signature_path,
            expected_sha256=identity_root_sha,
        )
    lineage_path.write_bytes(lineage_raw)

    escaped_lineage = (tmp_path / "attacker-lineage.json").resolve()
    escaped_lineage.write_bytes(lineage_raw)
    escaped_root = replace(root, training_lineage_path=escaped_lineage)
    escaped_root_path = tmp_path / "escaped-run-root.json"
    escaped_signature_path = tmp_path / "escaped-run-root.json.sig"
    escaped_signature_path.write_bytes(b"synthetic production root sshsig")
    escaped_root_sha = write_production_run_root(
        escaped_root_path, escaped_root
    )
    with pytest.raises(F2ProductionRunnerError, match="outside the signed output"):
        replay_production_run_root(
            escaped_root_path,
            signature_path=escaped_signature_path,
            expected_sha256=escaped_root_sha,
        )


def test_run_root_sshsig_uses_distinct_namespace(tmp_path: Path) -> None:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("ssh-keygen with SSHSIG support is unavailable")
    private_key = tmp_path / "authority"
    generated = subprocess.run(
        [
            ssh_keygen,
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            str(private_key),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if generated.returncode != 0:
        pytest.skip("ssh-keygen cannot generate an SSHSIG test key")
    public_key = " ".join(
        (tmp_path / "authority.pub").read_text(encoding="ascii").split()[:2]
    )
    message = canonical_json_bytes({"run_root": "synthetic"})
    message_path = tmp_path / "message.json"
    message_path.write_bytes(message)

    def sign(namespace: str) -> bytes:
        signature_path = Path(str(message_path) + ".sig")
        signature_path.unlink(missing_ok=True)
        completed = subprocess.run(
            [
                ssh_keygen,
                "-Y",
                "sign",
                "-f",
                str(private_key),
                "-n",
                namespace,
                str(message_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0 or not signature_path.is_file():
            pytest.skip("ssh-keygen lacks SSHSIG signing support")
        return signature_path.read_bytes()

    verify_production_run_root_signature(
        message,
        sign(RUN_ROOT_SSH_NAMESPACE),
        trusted_public_key=public_key,
    )
    with pytest.raises(F2ProductionRunnerError, match="signature is invalid"):
        verify_production_run_root_signature(
            message,
            sign("actworld-f2-launch-authorization-v2"),
            trusted_public_key=public_key,
        )


def test_device_resolution_is_independent_of_logical_rank() -> None:
    assert resolve_production_device(
        explicit_device=None, environment={"LOCAL_RANK": "3"}
    ) == ("cuda:3", 3, 3)
    assert resolve_production_device(
        explicit_device="cuda:0", environment={"LOCAL_RANK": "7"}
    ) == ("cuda:0", 0, 7)
    with pytest.raises(F2ProductionRunnerError, match="LOCAL_RANK"):
        resolve_production_device(explicit_device=None, environment={})
    with pytest.raises(F2ProductionRunnerError, match="cuda:<index>"):
        resolve_production_device(
            explicit_device="cuda:rank", environment={"LOCAL_RANK": "0"}
        )


class _FakeCuda:
    def __init__(self, *, initialized: bool = False) -> None:
        self._initialized = initialized
        self.selected: int | None = None

    def is_initialized(self) -> bool:
        return self._initialized

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 8

    def set_device(self, index: int) -> None:
        self.selected = index
        self._initialized = True

    @staticmethod
    def get_device_capability(index: int) -> tuple[int, int]:
        assert index >= 0
        return (8, 9)

    @staticmethod
    def get_device_name(index: int) -> str:
        return f"Fake CUDA {index}"


class _FakeTorch:
    __version__ = "2.fake"

    def __init__(self, *, initialized: bool = False) -> None:
        self.cuda = _FakeCuda(initialized=initialized)
        self.version = SimpleNamespace(cuda="12.fake")
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(
                allow_tf32=True,
                benchmark=True,
                deterministic=False,
                version=lambda: 8900,
            ),
        )
        self.deterministic = False
        self.matmul_precision: str | None = None

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        assert warn_only is False
        self.deterministic = enabled

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.deterministic

    def set_float32_matmul_precision(self, value: str) -> None:
        self.matmul_precision = value

    def get_float32_matmul_precision(self) -> str | None:
        return self.matmul_precision


def test_deterministic_runtime_initializer_sets_and_records_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import navsim.planning.validation.f2_production_runner as production

    fake = _FakeTorch()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(production, "_import_torch_for_runtime", lambda: fake)
    state = initialize_deterministic_runtime(
        logical_rank=7,
        world_size=8,
        explicit_device=None,
    )
    assert state.device == "cuda:2"
    assert state.logical_rank == 7
    assert fake.cuda.selected == 2
    assert fake.deterministic is True
    assert fake.matmul_precision == "highest"
    assert fake.backends.cuda.matmul.allow_tf32 is False
    assert fake.backends.cudnn.allow_tf32 is False
    assert fake.backends.cudnn.benchmark is False
    assert fake.backends.cudnn.deterministic is True
    payload = state.to_payload()
    assert payload["cublas_workspace_config"] == CUBLAS_WORKSPACE_CONFIG
    assert payload["deterministic_algorithms"] is True


def test_deterministic_runtime_rejects_late_cuda_or_cublas_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import navsim.planning.validation.f2_production_runner as production

    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(F2ProductionRunnerError, match="CUBLAS"):
        initialize_deterministic_runtime(
            logical_rank=0, world_size=1, explicit_device=None
        )

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    monkeypatch.setattr(
        production,
        "_import_torch_for_runtime",
        lambda: _FakeTorch(initialized=True),
    )
    with pytest.raises(F2ProductionRunnerError, match="initialized before"):
        initialize_deterministic_runtime(
            logical_rank=0, world_size=1, explicit_device=None
        )


def test_shard_roundtrip_tamper_fatal_and_no_overwrite(tmp_path: Path) -> None:
    result = _result()
    shard = CausalShardEvaluation(
        rank=0,
        world_size=1,
        assigned_logs=(result.log_name,),
        checkpoint_sha256=_sha("checkpoint-0"),
        checkpoint_epoch=0,
        manifest_sha256=MANIFEST_SHA,
        artifact_sha256=ARTIFACT_SHA,
        config_sha256=CONFIG_SHA,
        source_sha256=SOURCE_SHA,
        results=(result,),
    )
    path = tmp_path / "rank0.json"
    digest = write_shard_evaluation(path, shard)
    assert digest == read_bytes_with_sidecar(path)[1]
    assert read_shard_evaluation(path) == shard
    with pytest.raises(F2ProductionRunnerError, match="overwrite"):
        write_shard_evaluation(path, shard)

    tampered = shard_evaluation_payload(shard)
    tampered["evaluation_sha256"] = _sha("forged-evaluation")
    _rewrite_canonical(path, tampered)
    with pytest.raises(F2ProductionRunnerError, match="evaluation SHA-256"):
        read_shard_evaluation(path)

    tampered = shard_evaluation_payload(shard)
    tampered["fatal_count"] = 1
    _rewrite_canonical(path, tampered)
    with pytest.raises(F2ProductionRunnerError, match="fatal_count"):
        read_shard_evaluation(path)

    tampered = shard_evaluation_payload(shard)
    tampered["identity"]["checkpoint_sha256"] = tampered["identity"][
        "checkpoint_sha256"
    ].upper()
    _rewrite_canonical(path, tampered)
    with pytest.raises(F2ProductionRunnerError, match="lowercase hexadecimal"):
        read_shard_evaluation(path)


def test_runtime_record_binds_shard_run_root_and_determinism(tmp_path: Path) -> None:
    result = _result()
    shard = CausalShardEvaluation(
        rank=0,
        world_size=1,
        assigned_logs=(result.log_name,),
        checkpoint_sha256=_sha("checkpoint-0"),
        checkpoint_epoch=0,
        manifest_sha256=MANIFEST_SHA,
        artifact_sha256=ARTIFACT_SHA,
        config_sha256=CONFIG_SHA,
        source_sha256=SOURCE_SHA,
        results=(result,),
    )
    shard_path = (tmp_path / "rank0.json").resolve()
    shard_digest = write_shard_evaluation(shard_path, shard)
    state = DeterministicRuntimeState(
        logical_rank=0,
        world_size=1,
        local_rank=3,
        device="cuda:3",
        device_index=3,
        cuda_visible_devices="4,5,6,7",
        torch_version="2.test",
        torch_cuda_version="12.test",
        cudnn_version=8900,
        device_name="synthetic",
        device_capability=(8, 9),
    )
    run_root_sha = _sha("run-root")
    record_path = runtime_record_path_for_shard(shard_path)
    write_bytes_no_overwrite(
        record_path,
        canonical_json_bytes(
            deterministic_runtime_record_payload(
                state=state,
                run_root_sha256=run_root_sha,
                shard_path=shard_path,
                shard_sha256=shard_digest,
                evaluation=shard,
            )
        ),
    )
    runtime = read_deterministic_runtime_record(
        record_path,
        expected_run_root_sha256=run_root_sha,
        shard_path=shard_path,
        evaluation=shard,
    )
    assert runtime["device"] == "cuda:3"
    assert runtime["logical_rank"] == 0
    with pytest.raises(F2ProductionRunnerError, match="identity drifted"):
        read_deterministic_runtime_record(
            record_path,
            expected_run_root_sha256=_sha("other-run-root"),
            shard_path=shard_path,
            evaluation=shard,
        )


def _merge_manifest():
    rows = (
        ManifestRow("log-a", 100_000, "a0", True),
        ManifestRow("log-b", 100_000, "b0", True),
    )
    tokens = tuple(row.token for row in rows)
    return build_canonical_manifest(
        rows,
        expected_tokens=tokens,
        sensor_available_tokens=tokens,
        cache_tokens=tokens,
        allowlist_logs=("log-a", "log-b"),
        cache_metadata_sha256=_sha("cache"),
        source_revision="synthetic-source",
    )


def test_merge_world_requires_complete_unique_rank_coverage(tmp_path: Path) -> None:
    manifest = _merge_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_manifest_bytes(manifest))
    assignments = deterministic_log_assignments(manifest.rows, 2)
    rows_by_log = {row.log_name: row for row in manifest.rows}
    paths = []
    for rank in range(2):
        log_name = assignments[rank][0]
        row = rows_by_log[log_name]
        result = _result(token=row.token, log_name=row.log_name, time_us=row.time_us)
        shard = CausalShardEvaluation(
            rank=rank,
            world_size=2,
            assigned_logs=assignments[rank],
            checkpoint_sha256=_sha("checkpoint-0"),
            checkpoint_epoch=0,
            manifest_sha256=manifest_sha256(manifest),
            artifact_sha256=ARTIFACT_SHA,
            config_sha256=CONFIG_SHA,
            source_sha256=SOURCE_SHA,
            results=(result,),
        )
        path = tmp_path / f"rank{rank}.json"
        write_shard_evaluation(path, shard)
        paths.append(path)

    with pytest.raises(F2ProductionRunnerError, match="one file per rank"):
        merge_world_files(
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha256(manifest),
            world_size=2,
            shard_paths=paths[:1],
        )
    with pytest.raises(F2ProductionRunnerError, match="incomplete or duplicated"):
        merge_world_files(
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha256(manifest),
            world_size=2,
            shard_paths=(paths[0], paths[0]),
        )
    with pytest.raises(F2ProductionRunnerError, match="external expectation"):
        merge_world_files(
            manifest_path=manifest_path,
            expected_manifest_sha256=_sha("attacker-manifest"),
            world_size=2,
            shard_paths=paths,
        )
    merged = merge_world_files(
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha256(manifest),
        world_size=2,
        shard_paths=paths,
    )
    assert merged.world_size == 2
    assert tuple(row.token for row in merged.results) == ("a0", "b0")


def test_production_merge_replays_external_root_and_runtime_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import navsim.planning.validation.f2_production_runner as production

    manifest = _merge_manifest()
    manifest_path = (tmp_path / "manifest.json").resolve()
    manifest_path.write_bytes(canonical_manifest_bytes(manifest))
    results = tuple(
        _result(token=row.token, log_name=row.log_name, time_us=row.time_us)
        for row in manifest.rows
    )
    shard = CausalShardEvaluation(
        rank=0,
        world_size=1,
        assigned_logs=deterministic_log_assignments(manifest.rows, 1)[0],
        checkpoint_sha256=_sha("checkpoint-0"),
        checkpoint_epoch=0,
        manifest_sha256=manifest_sha256(manifest),
        artifact_sha256=ARTIFACT_SHA,
        config_sha256=CONFIG_SHA,
        source_sha256=SOURCE_SHA,
        results=results,
    )
    shard_path = (tmp_path / "production-rank0.json").resolve()
    shard_digest = write_shard_evaluation(shard_path, shard)
    state = DeterministicRuntimeState(
        logical_rank=0,
        world_size=1,
        local_rank=5,
        device="cuda:5",
        device_index=5,
        cuda_visible_devices=None,
        torch_version="2.test",
        torch_cuda_version="12.test",
        cudnn_version=8900,
        device_name="synthetic",
        device_capability=(8, 9),
    )
    run_root_sha = _sha("production-run-root")
    write_bytes_no_overwrite(
        runtime_record_path_for_shard(shard_path),
        canonical_json_bytes(
            deterministic_runtime_record_payload(
                state=state,
                run_root_sha256=run_root_sha,
                shard_path=shard_path,
                shard_sha256=shard_digest,
                evaluation=shard,
            )
        ),
    )
    root = _production_run_root(
        tmp_path,
        causal_manifest_sha256=manifest_sha256(manifest),
        causal_record_count=2,
        causal_log_count=2,
        artifact_root_sha256=ARTIFACT_SHA,
    )
    verified = SimpleNamespace(
        root=root,
        run_root_sha256=run_root_sha,
        bindings=_synthetic_bindings(
            causal_manifest_sha256=manifest_sha256(manifest)
        ),
        live_provenance=SimpleNamespace(causal=manifest),
        runtime_config=SimpleNamespace(
            path_map={"causal_manifest": manifest_path}
        ),
    )
    monkeypatch.setattr(
        production,
        "replay_production_run_root",
        lambda path, *, signature_path, expected_sha256: verified,
    )
    world_path = tmp_path / "production-world1.json"
    world = merge_production_world_files(
        run_root_path=tmp_path / "externally-pinned.json",
        run_root_signature_path=tmp_path / "externally-pinned.json.sig",
        expected_run_root_sha256=run_root_sha,
        world_size=1,
        shard_paths=(shard_path,),
        output_path=world_path,
    )
    assert len(world.results) == 2
    assert world.artifact_sha256 == root.artifact_root_sha256
    assert world_path.is_file()

    for label, forged_config, forged_source in (
        ("config", _sha("forged-config"), SOURCE_SHA),
        ("source", CONFIG_SHA, _sha("forged-source")),
    ):
        forged_shard = CausalShardEvaluation(
            rank=0,
            world_size=1,
            assigned_logs=deterministic_log_assignments(manifest.rows, 1)[0],
            checkpoint_sha256=_sha("checkpoint-0"),
            checkpoint_epoch=0,
            manifest_sha256=manifest_sha256(manifest),
            artifact_sha256=ARTIFACT_SHA,
            config_sha256=forged_config,
            source_sha256=forged_source,
            results=results,
        )
        forged_path = (tmp_path / f"forged-{label}-rank0.json").resolve()
        forged_digest = write_shard_evaluation(forged_path, forged_shard)
        write_bytes_no_overwrite(
            runtime_record_path_for_shard(forged_path),
            canonical_json_bytes(
                deterministic_runtime_record_payload(
                    state=state,
                    run_root_sha256=run_root_sha,
                    shard_path=forged_path,
                    shard_sha256=forged_digest,
                    evaluation=forged_shard,
                )
            ),
        )
        with pytest.raises(F2ProductionRunnerError, match="external run-root"):
            merge_production_world_files(
                run_root_path=tmp_path / "externally-pinned.json",
                run_root_signature_path=tmp_path / "externally-pinned.json.sig",
                expected_run_root_sha256=run_root_sha,
                world_size=1,
                shard_paths=(forged_path,),
                output_path=tmp_path / f"forged-{label}-world.json",
            )

    tampered_root = _production_run_root(
        tmp_path,
        causal_manifest_sha256=manifest_sha256(manifest),
        causal_record_count=2,
        causal_log_count=2,
        artifact_root_sha256=_sha("replacement-artifact"),
    )
    monkeypatch.setattr(
        production,
        "replay_production_run_root",
        lambda path, *, signature_path, expected_sha256: SimpleNamespace(
            root=tampered_root,
            run_root_sha256=run_root_sha,
            bindings=verified.bindings,
            live_provenance=verified.live_provenance,
            runtime_config=verified.runtime_config,
        ),
    )
    with pytest.raises(F2ProductionRunnerError, match="external run-root"):
        merge_production_world_files(
            run_root_path=tmp_path / "replacement.json",
            run_root_signature_path=tmp_path / "replacement.json.sig",
            expected_run_root_sha256=run_root_sha,
            world_size=1,
            shard_paths=(shard_path,),
            output_path=tmp_path / "replacement-world.json",
        )


def _write_epoch_worlds(
    directory: Path,
    *,
    epoch: int,
    critic_truth: float,
    mismatch_world: int | None = None,
) -> tuple[Path, Path, Path]:
    result = []
    for world_size in (1, 2, 8):
        checkpoint_sha = (
            _sha("different-checkpoint")
            if mismatch_world == world_size
            else _sha(f"checkpoint-{epoch}")
        )
        world = _world(
            world_size,
            epoch=epoch,
            checkpoint_sha=checkpoint_sha,
            critic_truth=critic_truth,
        )
        path = directory / f"epoch{epoch}.world{world_size}.json"
        write_world_evaluation(path, world)
        result.append(path)
    return tuple(result)  # type: ignore[return-value]


def _write_production_epoch_worlds(
    directory: Path,
    *,
    epoch: int,
    critic_truth: float,
    manifest: Any,
    run_root_sha256: str,
) -> tuple[Path, Path, Path]:
    rows_by_log = {row.log_name: row for row in manifest.rows}
    world_paths = []
    for world_size in (1, 2, 8):
        assignments = deterministic_log_assignments(manifest.rows, world_size)
        shard_paths = []
        for rank in range(world_size):
            results = tuple(
                _result(
                    token=rows_by_log[log_name].token,
                    log_name=log_name,
                    time_us=rows_by_log[log_name].time_us,
                    critic_truth=critic_truth,
                )
                for log_name in assignments[rank]
            )
            shard = CausalShardEvaluation(
                rank=rank,
                world_size=world_size,
                assigned_logs=assignments[rank],
                checkpoint_sha256=_sha(f"checkpoint-{epoch}"),
                checkpoint_epoch=epoch,
                manifest_sha256=manifest_sha256(manifest),
                artifact_sha256=ARTIFACT_SHA,
                config_sha256=CONFIG_SHA,
                source_sha256=SOURCE_SHA,
                results=results,
            )
            shard_path = (
                directory / f"epoch{epoch}.world{world_size}.rank{rank}.json"
            ).resolve()
            shard_digest = write_shard_evaluation(shard_path, shard)
            state = DeterministicRuntimeState(
                logical_rank=rank,
                world_size=world_size,
                local_rank=rank,
                device=f"cuda:{rank}",
                device_index=rank,
                cuda_visible_devices=None,
                torch_version="2.test",
                torch_cuda_version="12.test",
                cudnn_version=8900,
                device_name="synthetic",
                device_capability=(8, 9),
            )
            write_bytes_no_overwrite(
                runtime_record_path_for_shard(shard_path),
                canonical_json_bytes(
                    deterministic_runtime_record_payload(
                        state=state,
                        run_root_sha256=run_root_sha256,
                        shard_path=shard_path,
                        shard_sha256=shard_digest,
                        evaluation=shard,
                    )
                ),
            )
            shard_paths.append(shard_path)
        world_path = directory / f"epoch{epoch}.production-world{world_size}.json"
        merge_production_world_files(
            run_root_path=directory / "signed-run-root.json",
            run_root_signature_path=directory / "signed-run-root.json.sig",
            expected_run_root_sha256=run_root_sha256,
            world_size=world_size,
            shard_paths=shard_paths,
            output_path=world_path,
        )
        world_paths.append(world_path)
    return tuple(world_paths)  # type: ignore[return-value]


def test_world_roundtrip_and_invariance_mismatch(tmp_path: Path) -> None:
    paths = _write_epoch_worlds(tmp_path, epoch=0, critic_truth=0.66)
    world = read_world_evaluation(paths[0])
    assert world.world_size == 1
    certificate = tmp_path / "epoch0.invariant.json"
    proof = prove_invariance_files(world_paths=paths, output_path=certificate)
    assert proof.evaluation.epoch == 0
    assert read_checkpoint_evaluation(certificate).evaluation == proof.evaluation

    mismatch_dir = tmp_path / "mismatch"
    mismatch_dir.mkdir()
    mismatch_paths = _write_epoch_worlds(
        mismatch_dir, epoch=0, critic_truth=0.66, mismatch_world=8
    )
    with pytest.raises(F2ProductionRunnerError, match="invariance"):
        prove_invariance_files(
            world_paths=mismatch_paths,
            output_path=mismatch_dir / "invalid.invariant.json",
        )


def test_world_metric_tamper_is_recomputed_from_results(tmp_path: Path) -> None:
    world = _world(1)
    path = tmp_path / "world.json"
    write_world_evaluation(path, world)
    tampered = world_evaluation_payload(world)
    metrics = tampered["metrics"]
    metrics["critic_selected_score"] = 0.7
    metrics["critic_delta"] = 0.7 - metrics["base_selected_score"]
    metrics["critic_oracle_regret"] = metrics["oracle_score"] - 0.7
    metrics["proposal_headroom_closed"] = metrics["critic_delta"] / (
        metrics["oracle_score"] - metrics["base_selected_score"]
    )
    tampered["metrics_sha256"] = hashlib.sha256(
        canonical_json_bytes(metrics)
    ).hexdigest()
    _rewrite_canonical(path, tampered)
    with pytest.raises(F2ProductionRunnerError, match="token results"):
        read_world_evaluation(path)


def test_rank_exact_epochs_replays_proofs_and_persists_selection(
    tmp_path: Path,
) -> None:
    certificates = []
    critic_scores = (0.66, 0.68, 0.67, 0.70)
    for epoch, score in enumerate(critic_scores):
        worlds = _write_epoch_worlds(
            tmp_path, epoch=epoch, critic_truth=score
        )
        certificate = tmp_path / f"epoch{epoch}.invariant.json"
        prove_invariance_files(world_paths=worlds, output_path=certificate)
        certificates.append(certificate)

    ranking_path = tmp_path / "ranking.json"
    ranking = write_checkpoint_ranking(
        checkpoint_paths=certificates, output_path=ranking_path
    )
    assert ranking.ranking.selected is not None
    assert ranking.ranking.selected.epoch == 3
    replayed = read_checkpoint_ranking(ranking_path)
    assert replayed.ranking.selected == ranking.ranking.selected
    assert len(replayed.ranking.eligible) == 4


def test_prove_and_rank_production_wrappers_require_external_root_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import navsim.planning.validation.f2_production_runner as production

    manifest = build_canonical_manifest(
        (ManifestRow("log-0", 100_000, "token-0", True),),
        expected_tokens=("token-0",),
        sensor_available_tokens=("token-0",),
        cache_tokens=("token-0",),
        allowlist_logs=("log-0",),
        cache_metadata_sha256=_sha("cache"),
        source_revision="synthetic-source",
    )
    root = _production_run_root(
        tmp_path,
        causal_manifest_sha256=manifest_sha256(manifest),
        causal_record_count=1,
        causal_log_count=1,
        artifact_root_sha256=ARTIFACT_SHA,
    )
    verified = SimpleNamespace(
        root=root,
        run_root_sha256=_sha("run-root"),
        bindings=_synthetic_bindings(
            causal_manifest_sha256=manifest_sha256(manifest)
        ),
        live_provenance=SimpleNamespace(causal=manifest),
    )
    replay_calls: list[tuple[Path, str]] = []

    def replay(
        path: Path, *, signature_path: Path, expected_sha256: str
    ) -> Any:
        replay_calls.append((Path(path), expected_sha256))
        return verified

    monkeypatch.setattr(production, "replay_production_run_root", replay)
    certificates = []
    for epoch, score in enumerate((0.66, 0.68, 0.67, 0.70)):
        worlds = _write_production_epoch_worlds(
            tmp_path,
            epoch=epoch,
            critic_truth=score,
            manifest=manifest,
            run_root_sha256=_sha("run-root"),
        )
        certificate = tmp_path / f"external-epoch{epoch}.json"
        prove_production_invariance_files(
            run_root_path=tmp_path / "externally-pinned-run-root.json",
            run_root_signature_path=(
                tmp_path / "externally-pinned-run-root.json.sig"
            ),
            expected_run_root_sha256=_sha("run-root"),
            world_paths=worlds,
            output_path=certificate,
        )
        certificates.append(certificate)
    ranking_path = tmp_path / "external-ranking.json"
    ranking = rank_production_checkpoint_files(
        run_root_path=tmp_path / "externally-pinned-run-root.json",
        run_root_signature_path=tmp_path / "externally-pinned-run-root.json.sig",
        expected_run_root_sha256=_sha("run-root"),
        checkpoint_paths=certificates,
        output_path=ranking_path,
    )
    assert ranking.ranking.selected is not None
    assert ranking.ranking.selected.epoch == 3
    replayed_ranking = read_production_checkpoint_ranking(
        ranking_path, verified=verified
    )
    assert replayed_ranking.ranking.selected == ranking.ranking.selected
    assert len(replay_calls) == 17
    assert all(item[1] == _sha("run-root") for item in replay_calls)

    forged_dir = tmp_path / "forged-low-level"
    forged_dir.mkdir()
    forged_worlds = _write_epoch_worlds(
        forged_dir, epoch=0, critic_truth=0.66
    )
    with pytest.raises(F2ProductionRunnerError, match="production world certificate"):
        prove_production_invariance_files(
            run_root_path=tmp_path / "signed-run-root.json",
            run_root_signature_path=tmp_path / "signed-run-root.json.sig",
            expected_run_root_sha256=_sha("run-root"),
            world_paths=forged_worlds,
            output_path=forged_dir / "proof.json",
        )

    low_proof = forged_dir / "low-proof.json"
    prove_invariance_files(world_paths=forged_worlds, output_path=low_proof)
    with pytest.raises(
        F2ProductionRunnerError, match="production invariant checkpoint"
    ):
        rank_production_checkpoint_files(
            run_root_path=tmp_path / "signed-run-root.json",
            run_root_signature_path=tmp_path / "signed-run-root.json.sig",
            expected_run_root_sha256=_sha("run-root"),
            checkpoint_paths=(low_proof, low_proof, low_proof, low_proof),
            output_path=forged_dir / "forged-ranking.json",
        )


def test_ranking_rejects_missing_epoch_and_checkpoint_identity_drift(
    tmp_path: Path,
) -> None:
    certificates = []
    for epoch in range(3):
        worlds = _write_epoch_worlds(tmp_path, epoch=epoch, critic_truth=0.66)
        certificate = tmp_path / f"epoch{epoch}.invariant.json"
        prove_invariance_files(world_paths=worlds, output_path=certificate)
        certificates.append(certificate)
    with pytest.raises(F2ProductionRunnerError, match="epochs"):
        write_checkpoint_ranking(
            checkpoint_paths=(
                certificates[0],
                certificates[1],
                certificates[2],
                certificates[2],
            ),
            output_path=tmp_path / "invalid-ranking.json",
        )

    drift_dir = tmp_path / "drift"
    drift_dir.mkdir()
    worlds = _write_epoch_worlds(
        drift_dir,
        epoch=3,
        critic_truth=0.66,
        mismatch_world=2,
    )
    with pytest.raises(F2ProductionRunnerError, match="invariance"):
        prove_invariance_files(
            world_paths=worlds,
            output_path=drift_dir / "epoch3.invariant.json",
        )


def test_sidecar_tamper_fails_before_json_parse(tmp_path: Path) -> None:
    path = tmp_path / "world.json"
    write_world_evaluation(path, _world(1))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(F2ProductionRunnerError, match="sidecar"):
        read_world_evaluation(path)


def test_proof_rejects_non_sibling_inputs(tmp_path: Path) -> None:
    worlds_dir = tmp_path / "worlds"
    worlds_dir.mkdir()
    paths = _write_epoch_worlds(worlds_dir, epoch=0, critic_truth=0.66)
    with pytest.raises(F2ProductionRunnerError, match="sibling"):
        prove_invariance_files(
            world_paths=paths,
            output_path=tmp_path / "epoch0.invariant.json",
        )
