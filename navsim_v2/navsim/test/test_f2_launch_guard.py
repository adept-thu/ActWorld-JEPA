"""Synthetic fail-closed tests for the one-shot F2 launch authorization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict
from unittest.mock import patch

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from navsim.planning.script.f2_launch_guard import (
    AUTHORIZATION_PURPOSE,
    AUTHORIZATION_SCHEMA_VERSION,
    AUTHORIZATION_SSH_NAMESPACE,
    AUTHORIZED_ARTIFACT_KEYS,
    ATTESTATION_SCHEMA_VERSION,
    F2_EXPERIMENT_NAME,
    F2_EXPERIMENT_UID,
    F1_RECORD_REPOSITORY_PATH,
    LEGACY_LABEL_INTERPRETATION,
    F2LaunchGuardError,
    _authenticate_authorized_source,
    canonical_attestation_bytes,
    create_f2_runtime_claim,
    install_f2_launch_authorization,
    issue_f2_launch_authorization,
    read_expected_attestation_sha256,
    verify_f2_runtime_claim,
    verify_f2_launch_authorization,
    verify_f2_signed_contract_lightweight,
    verify_f2_trigger_attestation,
)
from navsim.planning.script.run_training import (
    F2_FROZEN_HYDRA_VALUES,
    F2_TRAINING_LINEAGE_FILENAME,
    _validate_frozen_f2_hydra_protocol,
    run_f2_train_row_finite_smoke,
    verify_configured_f2_runtime_admission,
)


def _raw_metrics(*, passing: bool = False) -> Dict[str, Any]:
    baseline = 0.8717616200
    selected = 0.9000000000 if passing else 0.8800000000
    return {
        "baseline_score": baseline,
        "best_of_32_score": 0.9576369524,
        "mean_candidate_score": 0.7636889815,
        "critic_selected_score": selected,
        "critic_delta": selected - baseline,
        "critic_oracle_regret": 0.0576369524,
        "critic_pairwise_accuracy": 0.75,
        "critic_score_error": 0.10,
        "baseline_score_error": 0.09,
        "critic_score_error_excess": 0.01,
        "proposal_headroom": 0.0858753324,
        "proposal_headroom_closed_fraction": 0.30,
        "critic_nc_delta": 0.0,
        "critic_dac_delta": 0.0,
        "critic_tlc_delta": 0.0,
        "critic_ddc_delta": 0.0,
        "future_camera_valid_min": 1.0,
        "future_camera_valid_mean": 1.0,
        "future_camera_valid_max": 1.0,
        "future_camera_valid_samples": 19,
        "baseline_reference_drift": 0.0,
        "best_of_32_reference_drift": 0.0,
    }


def _gate_pass(*, passing: bool = False) -> Dict[str, bool]:
    return {
        "critic_selected_score": passing,
        "critic_delta": passing,
        "critic_pairwise_accuracy": True,
        "proposal_headroom_closed": True,
        "critic_score_error_excess": True,
        "nc_delta": True,
        "dac_delta": True,
        "tlc_delta": True,
        "ddc_delta": True,
        "future_camera_valid": True,
        "baseline_and_oracle_drift": True,
        "fatal_or_nonfinite": True,
    }


def _f1_record() -> Dict[str, Any]:
    run_uid = "2026.08.24.20.00.36-stage-a-f1-f58f6b6"
    records = []
    for epoch in range(4):
        records.append(
            {
                "epoch": epoch,
                "status": "failed_stage_a_gates",
                "train_steps": 935,
                "validation_batches": 200,
                "wandb_epoch": epoch,
                "wandb_trainer_global_step": (epoch + 1) * 935 - 1,
                "raw_metrics": _raw_metrics(),
                "gate_pass": _gate_pass(),
                "fatal_error_matches": 0,
            }
        )
    return {
        "schema_version": 1,
        "run": {
            "project": "ActWorld-JEPA",
            "stage": "critic",
            "fallback": "F1",
            "status": "completed_after_final_scheduled_epoch_failure",
            "started_at": "2026-08-24T20:00:36+08:00",
            "completed_at": "2026-08-24T23:00:36+08:00",
            "uid": run_uid,
            "code_revision": "f58f6b6ff850b5c797830dacb524a5bc533e753e",
            "repository_head_at_launch": "04cdf0785f2be5d21baae5ac95afdec4ccf0d130",
            "repository_head_note": "synthetic",
            "torchrun_pid_at_launch": 100,
            "output_dir": "/qualified/f1",
            "log": "/qualified/f1.log",
            "log_bytes": 1,
            "log_sha256": "f" * 64,
            "terminal_log_marker": "Trainer.fit stopped: max_epochs=4 reached",
            "wandb_offline_record": "/qualified/f1.wandb",
            "wandb_offline_record_bytes": 1,
            "wandb_offline_record_sha256": "1" * 64,
            "resolved_config_path": "/qualified/f1/.hydra/config.yaml",
            "resolved_config_bytes": 1,
            "resolved_config_sha256": "a" * 64,
            "hydra_overrides_path": "/qualified/f1/.hydra/overrides.yaml",
            "hydra_overrides_bytes": 1,
            "hydra_overrides_sha256": "b" * 64,
        },
        "lineage": {
            "initialization": "fresh_stage_a",
            "trigger": "primary_epoch_4_failed_any_gate",
            "primary_parent_uid": "primary",
            "resume_checkpoint": None,
            "optimizer_inherited": False,
            "agent_checkpoint_inherited": False,
            "drive_jepa_checkpoint": "/qualified/baseline.ckpt",
            "drive_jepa_checkpoint_sha256": "c" * 64,
        },
        "data": {
            "split": "navtrain",
            "test_splits_consulted": False,
            "metric_scale": "raw_0_to_1",
            "one_reported_point_raw": 0.01,
            "train_samples": 59894,
            "validation_samples": 12817,
            "streaming": True,
            "future_latent_grounding": True,
            "metric_cache": "/qualified/cache",
            "metric_cache_metadata_sha256": "d" * 64,
        },
        "optimization": {
            "seed": 0,
            "accelerator": "8x RTX 4090 48GB",
            "strategy": "ddp",
            "precision": "16-mixed",
            "per_rank_batch_size": 8,
            "global_batch_size": 64,
            "dataloader_workers_per_rank": 2,
            "score_workers_per_rank": 4,
            "optimizer": "AdamW",
            "learning_rate": 0.00005,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "max_epochs": 4,
            "train_steps_per_epoch": 935,
            "validation_batches_per_epoch": 200,
        },
        "frozen_hypothesis": {
            "purpose": "single preregistered F1 hypothesis",
            "code_change_from_primary": False,
            "actworld_rank_loss_weight": 1.0,
            "actworld_rank_temperature": 0.03,
            "unchanged": {
                "actworld_value_loss_weight": 1.0,
                "actworld_latent_loss_weight": 0.25,
                "actworld_future_latent_loss_weight": 0.25,
                "raw_utility_selection": True,
            },
            "hyperparameter_search": "none",
        },
        "selection": {
            "development_split_only": "navtrain",
            "checkpoint_monitor": "val/critic_selected_score_epoch",
            "mode": "max",
            "effective_gates": {
                "baseline_score_reference": 0.8717616200,
                "best_of_32_score_reference": 0.9576369524,
                "critic_delta_min": 0.0214688331,
                "critic_selected_score_min": 0.8932304531,
                "critic_pairwise_accuracy_min": 0.70,
                "proposal_headroom_closed_min": 0.25,
                "critic_score_error_excess_max": 0.02,
                "hard_factor_delta_min": -0.001,
                "future_camera_valid_min": 0.99,
                "baseline_and_oracle_drift_max": 0.001,
                "fatal_or_nonfinite_matches_max": 0,
            },
            "decision": {
                "scheduled_epoch_count": 4,
                "final_scheduled_epoch_index": 3,
                "require_final_scheduled_epoch_gate_conjunction": True,
                "legacy_ambiguous_label": "require_epoch_4_gate_conjunction",
                "legacy_label_interpretation": (
                    "fourth completed epoch, zero-based index 3; "
                    "no index-4 epoch is authorized"
                ),
                "actor_allowed_before_gate_pass": False,
                "on_final_scheduled_epoch_failure": "execute_F2_only",
                "test_evaluation_before_selection": False,
            },
        },
        "epoch_indexing_clarification": {
            "recorded_at": "2026-08-24T21:04:22+08:00",
            "evidence_cutoff": (
                "F1 epoch 0 complete; epoch 1 training in progress; "
                "epoch 1 validation unavailable"
            ),
            "executable_authority": (
                "trainer.params.max_epochs=4 in the frozen resolved config"
            ),
            "lightning_epoch_indices": [0, 1, 2, 3],
            "final_gate_epoch_index": 3,
            "schedule_changed": False,
            "fifth_epoch_authorized": False,
            "note": "fourth completed epoch is zero-based index 3",
        },
        "startup_evidence": {"status": "passed"},
        "epoch_records": records,
        "checkpoints": {
            "best": {
                "epoch": 3,
                "path": "/qualified/f1/checkpoints/last.ckpt",
                "bytes": 1234,
                "sha256": "e" * 64,
            },
            "last": {
                "epoch": 3,
                "global_step": 3740,
                "path": "/qualified/f1/checkpoints/last.ckpt",
                "bytes": 1234,
                "sha256": "e" * 64,
                "training_stage": "critic",
                "state_dict_keys": 587,
                "state_dict_numel": 321251844,
            },
            "selected": None,
        },
        "results": {
            "status": "failed_final_scheduled_epoch_gates",
            "note": "No test split was evaluated.",
        },
    }


class _Evidence:
    def __init__(self, tmp_path: Path) -> None:
        self.repository = tmp_path / "repo"
        self.f1_path = self.repository / F1_RECORD_REPOSITORY_PATH
        self.f1_path.parent.mkdir(parents=True)
        self.record = _f1_record()
        self._write_terminal_files()
        self.experiment_uid = "f2-seed0-preregistered"
        self.experiment_root = tmp_path / "experiments" / "train_actworld_jepa_v2_f2"
        self.output_dir = self.experiment_root / self.experiment_uid
        self.launch_marker = (
            tmp_path / "experiments" / ".actworld_f2_launch_ledger" / f"{self.experiment_uid}.started"
        )
        self.attestation_path = tmp_path / "trigger.json"
        self.digest_path = tmp_path / "trigger.sha256"
        self.attestation: Dict[str, Any] = {}
        self.write_record_and_attestation()

    def _write_terminal_files(self) -> None:
        output = self.repository.parent / "f1-output"
        hydra = output / ".hydra"
        wandb = output / "wandb"
        checkpoints = output / "checkpoints"
        for directory in (hydra, wandb, checkpoints):
            directory.mkdir(parents=True, exist_ok=True)

        marker = "Trainer.fit stopped: max_epochs=4 reached"
        files = {
            "log": (self.repository.parent / "f1.log", ("start\n" + marker + "\n").encode()),
            "wandb_offline_record": (wandb / "run.wandb", b"synthetic-wandb\n"),
            "resolved_config": (hydra / "config.yaml", b"trainer:\n  max_epochs: 4\n"),
            "hydra_overrides": (hydra / "overrides.yaml", b"- trainer.params.max_epochs=4\n"),
            "checkpoint": (checkpoints / "last.ckpt", b"synthetic-checkpoint\n"),
        }
        for _, (path, payload) in files.items():
            path.write_bytes(payload)

        run = self.record["run"]
        run.update(
            {
                "output_dir": str(output.resolve()),
                "log": str(files["log"][0].resolve()),
                "log_bytes": len(files["log"][1]),
                "log_sha256": hashlib.sha256(files["log"][1]).hexdigest(),
                "terminal_log_marker": marker,
                "wandb_offline_record": str(files["wandb_offline_record"][0].resolve()),
                "wandb_offline_record_bytes": len(files["wandb_offline_record"][1]),
                "wandb_offline_record_sha256": hashlib.sha256(
                    files["wandb_offline_record"][1]
                ).hexdigest(),
                "resolved_config_path": str(files["resolved_config"][0].resolve()),
                "resolved_config_bytes": len(files["resolved_config"][1]),
                "resolved_config_sha256": hashlib.sha256(
                    files["resolved_config"][1]
                ).hexdigest(),
                "hydra_overrides_path": str(files["hydra_overrides"][0].resolve()),
                "hydra_overrides_bytes": len(files["hydra_overrides"][1]),
                "hydra_overrides_sha256": hashlib.sha256(
                    files["hydra_overrides"][1]
                ).hexdigest(),
            }
        )
        checkpoint_path, checkpoint_payload = files["checkpoint"]
        checkpoint_fields = {
            "path": str(checkpoint_path.resolve()),
            "bytes": len(checkpoint_payload),
            "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
        }
        self.record["checkpoints"]["best"].update(checkpoint_fields)
        self.record["checkpoints"]["last"].update(checkpoint_fields)

    def write_record_and_attestation(self) -> None:
        self.f1_path.write_text(
            yaml.safe_dump(self.record, sort_keys=False), encoding="utf-8"
        )
        record_sha = hashlib.sha256(self.f1_path.read_bytes()).hexdigest()
        self.attestation = {
            "schema_version": ATTESTATION_SCHEMA_VERSION,
            "f1": {
                "record_path": F1_RECORD_REPOSITORY_PATH,
                "record_sha256": record_sha,
                "run_uid": self.record["run"]["uid"],
                "scheduled_epoch_count": 4,
                "completed_epoch_count": 4,
                "final_scheduled_epoch_index": 3,
                "legacy_label_interpretation": LEGACY_LABEL_INTERPRETATION,
            },
            "decision": {
                "fixed_gate_conjunction_passed": False,
                "failed_gates": ["critic_selected_score", "critic_delta"],
                "authorized_action": "execute_F2_only",
                "test_splits_consulted": False,
            },
            "f2_prelaunch": {
                "started": False,
                "experiment_uid": self.experiment_uid,
                "experiment_root": str(self.experiment_root.resolve()),
                "output_dir": str(self.output_dir.resolve()),
                "launch_marker": str(self.launch_marker.resolve()),
            },
        }
        self.write_attestation()

    def write_attestation(self, *, canonical: bool = True) -> str:
        if canonical:
            raw = canonical_attestation_bytes(self.attestation)
        else:
            raw = (json.dumps(self.attestation, indent=2) + "\n").encode("utf-8")
        self.attestation_path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        self.digest_path.write_text(digest + "\n", encoding="ascii")
        return digest

    def verify(self, *, digest: str | None = None):
        if digest is None:
            digest = read_expected_attestation_sha256(self.digest_path)
        return verify_f2_trigger_attestation(
            self.attestation_path,
            expected_attestation_sha256=digest,
            repository_root=self.repository,
            expected_experiment_uid=self.experiment_uid,
            expected_experiment_root=self.experiment_root,
            expected_output_dir=self.output_dir,
            expected_launch_marker=self.launch_marker,
        )


@pytest.fixture
def evidence(tmp_path: Path) -> _Evidence:
    return _Evidence(tmp_path)


def _run_checked(arguments: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "ActWorld test",
            "GIT_AUTHOR_EMAIL": "actworld-test@example.invalid",
            "GIT_COMMITTER_NAME": "ActWorld test",
            "GIT_COMMITTER_EMAIL": "actworld-test@example.invalid",
        },
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.decode("utf-8", errors="replace"))
    return completed.stdout.decode("utf-8", errors="strict").strip()


class _V2Evidence:
    """Signed synthetic v2 contract whose source/artifacts never use test splits."""

    def __init__(self, tmp_path: Path) -> None:
        if shutil.which("ssh-keygen") is None or shutil.which("git") is None:
            pytest.skip("OpenSSH and Git are required for signed launch tests")
        self.legacy = _Evidence(tmp_path)
        self.repository = self.legacy.repository
        self.navsim_exp_root = tmp_path / "experiments"
        self.navsim_exp_root.mkdir()
        self.openscene_data_root = tmp_path / "momworld-data"
        self.nuplan_maps_root = self.openscene_data_root / "maps"
        self.nuplan_maps_root.mkdir(parents=True)
        # Production uses canonical real roots on separate disks rather than
        # the symlinked children below OPENSCENE_DATA_ROOT.
        self.navsim_log_root = tmp_path / "canonical-navsim-logs" / "trainval"
        self.sensor_blobs_root = tmp_path / "canonical-sensor-blobs" / "trainval"
        self.qualification_source_root = tmp_path / "qualification-source"
        self.navsim_log_root.mkdir(parents=True)
        self.sensor_blobs_root.mkdir(parents=True)
        self.qualification_source_root.mkdir()
        artifact_root = self.navsim_exp_root / "authorized-inputs"
        artifact_root.mkdir()
        self.artifact_paths: Dict[str, Path] = {}
        self.artifact_digests: Dict[str, str] = {}
        for index, name in enumerate(AUTHORIZED_ARTIFACT_KEYS):
            path = artifact_root / f"{name}.bin"
            payload = f"synthetic-{index}-{name}\n".encode("ascii")
            path.write_bytes(payload)
            self.artifact_paths[name] = path.resolve()
            self.artifact_digests[name] = hashlib.sha256(payload).hexdigest()

        self.legacy.record["lineage"]["drive_jepa_checkpoint_sha256"] = (
            self.artifact_digests["baseline_checkpoint"]
        )
        self.legacy.record["data"]["metric_cache_metadata_sha256"] = (
            self.artifact_digests["cache_metadata"]
        )
        self.legacy.write_record_and_attestation()

        _run_checked(["git", "init", "-q"], cwd=self.repository)
        _run_checked(["git", "add", "."], cwd=self.repository)
        _run_checked(
            ["git", "commit", "-q", "-m", "synthetic signed source"],
            cwd=self.repository,
        )
        self.source_revision = _run_checked(
            ["git", "rev-parse", "HEAD"], cwd=self.repository
        )

        self.private_key = tmp_path / "offline-test-ed25519"
        _run_checked(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                "actworld-unit-test",
                "-f",
                str(self.private_key),
            ]
        )
        public_parts = Path(str(self.private_key) + ".pub").read_text(
            encoding="ascii"
        ).split()
        self.public_key = " ".join(public_parts[:2])

        authority_root = tmp_path / "authority"
        authority_root.mkdir()
        self.authorization_path = authority_root / "authorization.json"
        self.signature_path = authority_root / "authorization.json.sig"
        f1 = dict(self.legacy.attestation["f1"])
        decision = dict(self.legacy.attestation["decision"])
        prelaunch = dict(self.legacy.attestation["f2_prelaunch"])
        prelaunch["experiment_name"] = F2_EXPERIMENT_NAME
        self.authorization: Dict[str, Any] = {
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "purpose": AUTHORIZATION_PURPOSE,
            "authorization_id": "synthetic-navtrain-f2-authorization",
            "source": {"git_revision": self.source_revision},
            "runtime": {
                "repository_root": str(self.repository.resolve()),
                "navsim_exp_root": str(self.navsim_exp_root.resolve()),
                "openscene_data_root": str(self.openscene_data_root.resolve()),
                "nuplan_maps_root": str(self.nuplan_maps_root.resolve()),
                "navsim_log_root": str(self.navsim_log_root.resolve()),
                "sensor_blobs_root": str(self.sensor_blobs_root.resolve()),
                "qualification_source_root": str(
                    self.qualification_source_root.resolve()
                ),
            },
            "artifacts": {
                name: {
                    "path": str(self.artifact_paths[name]),
                    "sha256": self.artifact_digests[name],
                }
                for name in AUTHORIZED_ARTIFACT_KEYS
            },
            "f1": f1,
            "decision": decision,
            "f2_prelaunch": prelaunch,
        }
        self.sign()

    def sign(self) -> None:
        self.authorization_path.write_bytes(
            canonical_attestation_bytes(self.authorization)
        )
        if self.signature_path.exists():
            self.signature_path.unlink()
        _run_checked(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(self.private_key),
                "-n",
                AUTHORIZATION_SSH_NAMESPACE,
                str(self.authorization_path),
            ]
        )
        generated = Path(str(self.authorization_path) + ".sig")
        assert generated == self.signature_path
        assert self.signature_path.is_file()

    def verify(self):
        return verify_f2_launch_authorization(
            self.authorization_path,
            self.signature_path,
            repository_root=self.repository,
            trusted_public_key=self.public_key,
        )


@pytest.fixture
def v2_evidence(tmp_path: Path) -> _V2Evidence:
    return _V2Evidence(tmp_path)


def test_valid_four_completed_epochs_authorize_f2(evidence: _Evidence) -> None:
    verified = evidence.verify()
    assert verified.failed_gates == ("critic_selected_score", "critic_delta")
    assert verified.f1_run_uid == evidence.record["run"]["uid"]
    assert verified.output_dir == evidence.output_dir.resolve()


def test_signed_v2_authorization_binds_source_inputs_and_output(
    v2_evidence: _V2Evidence,
) -> None:
    verified = v2_evidence.verify()
    assert verified.source_revision == v2_evidence.source_revision
    assert verified.artifacts["raw_input_manifest"].sha256 == (
        v2_evidence.artifact_digests["raw_input_manifest"]
    )
    assert verified.artifacts["feasibility_audit"].sha256 == (
        v2_evidence.artifact_digests["feasibility_audit"]
    )
    assert verified.output_dir == v2_evidence.legacy.output_dir.resolve()
    lines = verified.contract_lines()
    assert len(lines) == 15 + 2 * len(AUTHORIZED_ARTIFACT_KEYS)
    assert any(line.startswith("SOURCE_REVISION\t") for line in lines)


def test_v2_authorization_cannot_be_modified_without_authority_signature(
    v2_evidence: _V2Evidence,
) -> None:
    v2_evidence.authorization["artifacts"]["raw_input_manifest"]["sha256"] = "0" * 64
    v2_evidence.authorization_path.write_bytes(
        canonical_attestation_bytes(v2_evidence.authorization)
    )
    with pytest.raises(F2LaunchGuardError, match="signature is invalid"):
        v2_evidence.verify()


def test_v2_resigned_wrong_source_revision_still_fails_closed(
    v2_evidence: _V2Evidence,
) -> None:
    v2_evidence.authorization["source"]["git_revision"] = "0" * 40
    v2_evidence.sign()
    with pytest.raises(F2LaunchGuardError, match="differs from signed revision"):
        v2_evidence.verify()


def test_v2_resigned_wrong_raw_input_digest_still_fails_closed(
    v2_evidence: _V2Evidence,
) -> None:
    v2_evidence.authorization["artifacts"]["raw_input_manifest"]["sha256"] = "0" * 64
    v2_evidence.sign()
    with pytest.raises(F2LaunchGuardError, match="raw_input_manifest SHA-256 mismatch"):
        v2_evidence.verify()


def test_v2_signed_raw_root_cannot_traverse_symlink(
    v2_evidence: _V2Evidence,
    tmp_path: Path,
) -> None:
    symlink = tmp_path / "symlinked-navsim-log-root"
    try:
        symlink.symlink_to(v2_evidence.navsim_log_root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    v2_evidence.authorization["runtime"]["navsim_log_root"] = str(symlink)
    v2_evidence.sign()
    with pytest.raises(F2LaunchGuardError, match="must not traverse a symbolic link"):
        v2_evidence.verify()


def test_v2_signed_source_must_remain_globally_clean(
    v2_evidence: _V2Evidence,
) -> None:
    (v2_evidence.repository / "untracked-runtime-shadow.py").write_text(
        "raise RuntimeError('shadow')\n", encoding="utf-8"
    )
    with pytest.raises(F2LaunchGuardError, match="worktree is not exactly clean"):
        v2_evidence.verify()


def test_lightweight_rank_envelope_rejects_tracked_or_untracked_source_race(
    v2_evidence: _V2Evidence,
) -> None:
    v2_evidence.legacy.f1_path.write_bytes(
        v2_evidence.legacy.f1_path.read_bytes() + b"\n"
    )
    with pytest.raises(F2LaunchGuardError, match="worktree is not exactly clean"):
        _lightweight_contract(v2_evidence)


def test_lightweight_rank_envelope_rejects_ignored_import_shadow(
    v2_evidence: _V2Evidence,
) -> None:
    info_exclude = v2_evidence.repository / ".git" / "info" / "exclude"
    info_exclude.write_text("*.pyc\n", encoding="utf-8")
    shadow = v2_evidence.repository / "navsim_v2" / "navsim" / "shadow.pyc"
    shadow.parent.mkdir(parents=True)
    shadow.write_bytes(b"attacker-controlled-bytecode")
    with pytest.raises(F2LaunchGuardError, match="ignored import-shadow"):
        _lightweight_contract(v2_evidence)


def test_lightweight_source_authentication_rejects_submodule_drift(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    revision = "a" * 40

    def fake_git(_root, arguments):
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(repository.resolve())
        if arguments == ("rev-parse", "HEAD"):
            return revision
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if arguments == ("submodule", "status", "--recursive"):
            return "+deadbeef navsim_v2/vendor (heads/main)"
        raise AssertionError(arguments)

    with patch(
        "navsim.planning.script.f2_launch_guard._git_command",
        side_effect=fake_git,
    ), pytest.raises(F2LaunchGuardError, match="submodule differs"):
        _authenticate_authorized_source(repository, revision)


def _lightweight_contract(v2_evidence: _V2Evidence):
    return verify_f2_signed_contract_lightweight(
        v2_evidence.authorization_path,
        v2_evidence.signature_path,
        repository_root=v2_evidence.repository,
        trusted_public_key=v2_evidence.public_key,
    )


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="Linux procfs required")
def test_runtime_claim_is_canonical_parent_bound_and_no_overwrite(
    v2_evidence: _V2Evidence,
) -> None:
    contract = _lightweight_contract(v2_evidence)
    parent_pid = os.getppid()
    claim = create_f2_runtime_claim(
        v2_evidence.authorization_path,
        v2_evidence.signature_path,
        repository_root=v2_evidence.repository,
        launcher_pid=parent_pid,
        trusted_public_key=v2_evidence.public_key,
        require_dedicated_launcher=False,
    )
    assert claim.path == contract.launch_marker
    assert claim.output_dir == contract.output_dir
    assert claim.authorization_sha256 == contract.authorization_sha256
    assert claim.signature_sha256 == contract.signature_sha256
    assert claim.path.read_bytes() == canonical_attestation_bytes(
        json.loads(claim.path.read_text(encoding="utf-8"))
    )
    replay = verify_f2_runtime_claim(
        claim.path,
        expected_sha256=claim.sha256,
        signed_contract=contract,
        expected_parent_pid=parent_pid,
    )
    assert replay.launcher_pid == parent_pid
    with pytest.raises(F2LaunchGuardError, match="already exists"):
        create_f2_runtime_claim(
            v2_evidence.authorization_path,
            v2_evidence.signature_path,
            repository_root=v2_evidence.repository,
            launcher_pid=parent_pid,
            trusted_public_key=v2_evidence.public_key,
            require_dedicated_launcher=False,
        )


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="Linux procfs required")
def test_direct_claim_helper_without_dedicated_launcher_is_rejected(
    v2_evidence: _V2Evidence,
) -> None:
    with pytest.raises(F2LaunchGuardError, match="dedicated signed F2 launcher"):
        create_f2_runtime_claim(
            v2_evidence.authorization_path,
            v2_evidence.signature_path,
            repository_root=v2_evidence.repository,
            launcher_pid=os.getppid(),
            trusted_public_key=v2_evidence.public_key,
        )


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="Linux procfs required")
def test_forged_or_stale_runtime_claim_is_rejected(
    v2_evidence: _V2Evidence,
) -> None:
    contract = _lightweight_contract(v2_evidence)
    claim = create_f2_runtime_claim(
        v2_evidence.authorization_path,
        v2_evidence.signature_path,
        repository_root=v2_evidence.repository,
        launcher_pid=os.getppid(),
        trusted_public_key=v2_evidence.public_key,
        require_dedicated_launcher=False,
    )
    forged = json.loads(claim.path.read_text(encoding="utf-8"))
    forged["process"]["launcher_start_time_ticks"] += 1
    forged_raw = canonical_attestation_bytes(forged)
    claim.path.write_bytes(forged_raw)
    with pytest.raises(F2LaunchGuardError, match="start identity is stale"):
        verify_f2_runtime_claim(
            claim.path,
            expected_sha256=hashlib.sha256(forged_raw).hexdigest(),
            signed_contract=contract,
            expected_parent_pid=os.getppid(),
        )


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="Linux procfs required")
def test_runtime_claim_rejects_symlinked_ledger_parent(
    v2_evidence: _V2Evidence,
    tmp_path: Path,
) -> None:
    external = tmp_path / "external-ledger"
    external.mkdir()
    ledger = v2_evidence.navsim_exp_root / ".actworld_f2_launch_ledger"
    try:
        ledger.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(F2LaunchGuardError, match="non-symlink directory"):
        create_f2_runtime_claim(
            v2_evidence.authorization_path,
            v2_evidence.signature_path,
            repository_root=v2_evidence.repository,
            launcher_pid=os.getppid(),
            trusted_public_key=v2_evidence.public_key,
            require_dedicated_launcher=False,
        )
    assert list(external.iterdir()) == []


def _runtime_rank_inputs(v2_evidence: _V2Evidence):
    contract = _lightweight_contract(v2_evidence)
    claim = create_f2_runtime_claim(
        v2_evidence.authorization_path,
        v2_evidence.signature_path,
        repository_root=v2_evidence.repository,
        launcher_pid=os.getppid(),
        trusted_public_key=v2_evidence.public_key,
        require_dedicated_launcher=False,
    )
    contract.output_dir.mkdir(parents=True)
    cfg = OmegaConf.create(
        {
            "agent": {
                "config": {
                    "baseline_checkpoint_path": str(
                        contract.artifacts["baseline_checkpoint"].path
                    )
                }
            },
            "f2_qualification": {
                "manifest_path": str(
                    contract.artifacts["qualification_manifest"].path
                ),
                "cache_metadata_path": str(
                    contract.artifacts["cache_metadata"].path
                ),
                "cache_content_manifest_path": str(
                    contract.artifacts["cache_content_manifest"].path
                ),
                "raw_input_content_manifest_path": str(
                    contract.artifacts["raw_input_manifest"].path
                ),
                "expected_manifest_sha256": contract.artifacts[
                    "qualification_manifest"
                ].sha256,
                "expected_config_sha256": contract.artifacts[
                    "qualification_config"
                ].sha256,
                "expected_cache_sha256": contract.artifacts[
                    "cache_metadata"
                ].sha256,
                "expected_cache_content_manifest_sha256": contract.artifacts[
                    "cache_content_manifest"
                ].sha256,
                "expected_raw_input_content_manifest_sha256": contract.artifacts[
                    "raw_input_manifest"
                ].sha256,
                "expected_source_sha256": contract.artifacts[
                    "qualification_source_manifest"
                ].sha256,
            },
            "experiment_name": contract.experiment_name,
            "experiment_uid": contract.experiment_uid,
            "output_dir": str(contract.output_dir),
            "navsim_log_path": str(contract.runtime["navsim_log_root"]),
            "sensor_blobs_path": str(contract.runtime["sensor_blobs_root"]),
            "trainer": {
                "params": {"default_root_dir": str(contract.output_dir)}
            },
        }
    )
    environment = {
        "ACTWORLD_F2_AUTHORIZATION_SHA256": contract.authorization_sha256,
        "ACTWORLD_F2_AUTHORIZATION_SIGNATURE_SHA256": contract.signature_sha256,
        "ACTWORLD_F2_SOURCE_REVISION": contract.source_revision,
        "ACTWORLD_F2_OUTPUT_DIR": str(contract.output_dir),
        "ACTWORLD_F2_LAUNCH_MARKER": str(contract.launch_marker),
        "NAVSIM_EXP_ROOT": str(contract.runtime["navsim_exp_root"]),
        "OPENSCENE_DATA_ROOT": str(contract.runtime["openscene_data_root"]),
        "NUPLAN_MAPS_ROOT": str(contract.runtime["nuplan_maps_root"]),
        "ACTWORLD_F2_NAVSIM_LOG_ROOT": str(contract.runtime["navsim_log_root"]),
        "ACTWORLD_F2_SENSOR_BLOBS_ROOT": str(
            contract.runtime["sensor_blobs_root"]
        ),
        "ACTWORLD_F2_QUALIFICATION_SOURCE_ROOT": str(
            contract.runtime["qualification_source_root"]
        ),
        "ACTWORLD_F2_RUNTIME_CLAIM_PATH": str(claim.path),
        "ACTWORLD_F2_RUNTIME_CLAIM_SHA256": claim.sha256,
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
        "GROUP_RANK": "0",
        "RANK": "0",
        "LOCAL_RANK": "0",
    }
    for name, artifact in contract.artifacts.items():
        prefix = f"ACTWORLD_F2_{name.upper()}"
        environment[f"{prefix}_PATH"] = str(artifact.path)
        environment[f"{prefix}_SHA256"] = artifact.sha256
    return contract, claim, cfg, environment


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="Linux procfs required")
def test_rank_runtime_admission_rejects_direct_config_drift_and_replay(
    v2_evidence: _V2Evidence,
) -> None:
    contract, claim, cfg, environment = _runtime_rank_inputs(v2_evidence)
    admitted = verify_configured_f2_runtime_admission(
        cfg,
        authorization_path=v2_evidence.authorization_path,
        signature_path=v2_evidence.signature_path,
        trusted_public_key=v2_evidence.public_key,
        environment=environment,
        parent_pid=os.getppid(),
        repository_root=v2_evidence.repository,
    )
    assert admitted == claim
    lineage = contract.output_dir / F2_TRAINING_LINEAGE_FILENAME
    assert lineage.is_file()

    drifted = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    drifted.experiment_uid = "attacker-run"
    with pytest.raises(ValueError, match="signed authorization"):
        verify_configured_f2_runtime_admission(
            drifted,
            authorization_path=v2_evidence.authorization_path,
            signature_path=v2_evidence.signature_path,
            trusted_public_key=v2_evidence.public_key,
            environment=environment,
            parent_pid=os.getppid(),
            repository_root=v2_evidence.repository,
        )

    with pytest.raises(ValueError, match="duplicate/replayed"):
        verify_configured_f2_runtime_admission(
            cfg,
            authorization_path=v2_evidence.authorization_path,
            signature_path=v2_evidence.signature_path,
            trusted_public_key=v2_evidence.public_key,
            environment=environment,
            parent_pid=os.getppid(),
            repository_root=v2_evidence.repository,
        )


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="Linux procfs required")
@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"ACTWORLD_F2_RUNTIME_CLAIM_PATH": ""}, "claim path"),
        ({"ACTWORLD_F2_AUTHORIZATION_SHA256": "0" * 64}, "environment"),
        ({"WORLD_SIZE": "1"}, "WORLD_SIZE=8"),
        ({"GROUP_RANK": "1"}, "GROUP_RANK=0"),
    ),
)
def test_rank_runtime_admission_rejects_bypass_environment(
    v2_evidence: _V2Evidence,
    mutation: dict[str, str],
    message: str,
) -> None:
    _, _, cfg, environment = _runtime_rank_inputs(v2_evidence)
    environment.update(mutation)
    with pytest.raises(ValueError, match=message):
        verify_configured_f2_runtime_admission(
            cfg,
            authorization_path=v2_evidence.authorization_path,
            signature_path=v2_evidence.signature_path,
            trusted_public_key=v2_evidence.public_key,
            environment=environment,
            parent_pid=os.getppid(),
            repository_root=v2_evidence.repository,
        )


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="Linux procfs required")
def test_rank_runtime_admission_rejects_stale_parent_pid(
    v2_evidence: _V2Evidence,
) -> None:
    _, _, cfg, environment = _runtime_rank_inputs(v2_evidence)
    with pytest.raises(ValueError, match="launcher PID"):
        verify_configured_f2_runtime_admission(
            cfg,
            authorization_path=v2_evidence.authorization_path,
            signature_path=v2_evidence.signature_path,
            trusted_public_key=v2_evidence.public_key,
            environment=environment,
            parent_pid=os.getpid(),
            repository_root=v2_evidence.repository,
        )


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="Linux procfs required")
def test_rank_lineage_rejects_symlink_swapped_output_directory(
    v2_evidence: _V2Evidence,
    tmp_path: Path,
) -> None:
    contract, _, cfg, environment = _runtime_rank_inputs(v2_evidence)
    contract.output_dir.rmdir()
    external = tmp_path / "attacker-output"
    external.mkdir()
    try:
        contract.output_dir.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(ValueError, match="parent chain.*no symlink"):
        verify_configured_f2_runtime_admission(
            cfg,
            authorization_path=v2_evidence.authorization_path,
            signature_path=v2_evidence.signature_path,
            trusted_public_key=v2_evidence.public_key,
            environment=environment,
            parent_pid=os.getppid(),
            repository_root=v2_evidence.repository,
        )
    assert list(external.iterdir()) == []


def test_every_frozen_hydra_value_rejects_config_drift() -> None:
    cfg = OmegaConf.create({})
    for path, expected in F2_FROZEN_HYDRA_VALUES.items():
        OmegaConf.update(cfg, path, expected, merge=False, force_add=True)
    _validate_frozen_f2_hydra_protocol(cfg)

    cfg.agent.config.actworld_f2_collision_bce_weight = 0.999
    with pytest.raises(ValueError, match="collision_bce_weight"):
        _validate_frozen_f2_hydra_protocol(cfg)

    cfg.agent.config.actworld_f2_collision_bce_weight = 1.0
    cfg.dataloader.params.pin_memory = 1
    with pytest.raises(ValueError, match="pin_memory"):
        _validate_frozen_f2_hydra_protocol(cfg)


def test_preclaim_train_row_smoke_requires_finite_actual_loss() -> None:
    class _SmokeAgent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.make_nan = False

        def forward(self, features):
            return {"prediction": features["x"] * self.weight}

        def compute_loss(self, features, targets, prediction):
            loss = (prediction["prediction"] - targets["y"]).square().mean()
            if self.make_nan:
                loss = loss * torch.tensor(float("nan"))
            return {"loss": loss}

        def get_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    agent = _SmokeAgent()
    loader = [({"x": torch.ones(1)}, {"y": torch.zeros(1)})]
    initial_weight = agent.weight.detach().clone()
    run_f2_train_row_finite_smoke(
        agent, loader, device=torch.device("cpu")
    )
    assert not torch.equal(agent.weight.detach(), initial_weight)
    agent.make_nan = True
    with pytest.raises(FloatingPointError, match="non-finite"):
        run_f2_train_row_finite_smoke(
            agent, loader, device=torch.device("cpu")
        )


def test_preclaim_train_row_smoke_rejects_missing_trainable_gradient() -> None:
    class _UnusedParameterAgent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = torch.nn.Parameter(torch.tensor(2.0))
            self.unused = torch.nn.Parameter(torch.tensor(3.0))

        def forward(self, features):
            return {"prediction": features["x"] * self.used}

        def compute_loss(self, features, targets, prediction):
            return {"loss": (prediction["prediction"] - targets["y"]).square().mean()}

        def get_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    loader = [({"x": torch.ones(1)}, {"y": torch.zeros(1)})]
    with pytest.raises(FloatingPointError, match="missing=.*unused"):
        run_f2_train_row_finite_smoke(
            _UnusedParameterAgent(), loader, device=torch.device("cpu")
        )


def test_preclaim_train_row_smoke_rejects_finite_loss_with_nan_gradient() -> None:
    class _FiniteForwardNaNBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            return torch.full_like(gradient, float("nan"))

    class _NaNGradientAgent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, features):
            prediction = features["x"] * self.weight
            return {"prediction": _FiniteForwardNaNBackward.apply(prediction)}

        def compute_loss(self, features, targets, prediction):
            return {"loss": (prediction["prediction"] - targets["y"]).square().mean()}

        def get_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    loader = [({"x": torch.ones(1)}, {"y": torch.zeros(1)})]
    with pytest.raises(FloatingPointError, match="nonfinite=.*weight"):
        run_f2_train_row_finite_smoke(
            _NaNGradientAgent(), loader, device=torch.device("cpu")
        )


def test_v2_offline_issuance_is_explicit_canonical_and_no_overwrite(
    v2_evidence: _V2Evidence,
    tmp_path: Path,
) -> None:
    draft = tmp_path / "draft.json"
    draft.write_text(
        json.dumps(v2_evidence.authorization, indent=2) + "\n", encoding="utf-8"
    )
    output_root = tmp_path / "issued"
    output_root.mkdir()
    authorization = output_root / "authorization.json"
    signature = output_root / "authorization.json.sig"
    authorization_sha, signature_sha = issue_f2_launch_authorization(
        draft,
        authorization,
        signature,
        private_key_path=v2_evidence.private_key,
        trusted_public_key=v2_evidence.public_key,
    )
    assert authorization.read_bytes() == canonical_attestation_bytes(
        v2_evidence.authorization
    )
    assert hashlib.sha256(authorization.read_bytes()).hexdigest() == authorization_sha
    assert hashlib.sha256(signature.read_bytes()).hexdigest() == signature_sha
    with pytest.raises(F2LaunchGuardError, match="overwrite is forbidden"):
        issue_f2_launch_authorization(
            draft,
            authorization,
            signature,
            private_key_path=v2_evidence.private_key,
            trusted_public_key=v2_evidence.public_key,
        )


def test_v2_verified_install_is_no_overwrite(
    v2_evidence: _V2Evidence,
    tmp_path: Path,
) -> None:
    install_root = tmp_path / "fixed-authority-root"
    install_root.mkdir()
    authorization = install_root / "f2_launch_authorization_v2.json"
    signature = install_root / "f2_launch_authorization_v2.json.sig"
    digests = install_f2_launch_authorization(
        v2_evidence.authorization_path,
        v2_evidence.signature_path,
        authorization,
        signature,
        repository_root=v2_evidence.repository,
        trusted_public_key=v2_evidence.public_key,
    )
    assert digests == (
        hashlib.sha256(authorization.read_bytes()).hexdigest(),
        hashlib.sha256(signature.read_bytes()).hexdigest(),
    )
    with pytest.raises(F2LaunchGuardError, match="overwrite is forbidden"):
        install_f2_launch_authorization(
            v2_evidence.authorization_path,
            v2_evidence.signature_path,
            authorization,
            signature,
            repository_root=v2_evidence.repository,
            trusted_public_key=v2_evidence.public_key,
        )


def test_attestation_sha_mismatch_fails_closed(evidence: _Evidence) -> None:
    with pytest.raises(F2LaunchGuardError, match="attestation SHA-256 mismatch"):
        evidence.verify(digest="0" * 64)


def test_noncanonical_json_fails_even_with_matching_sha(evidence: _Evidence) -> None:
    digest = evidence.write_attestation(canonical=False)
    with pytest.raises(F2LaunchGuardError, match="not canonical"):
        evidence.verify(digest=digest)


def test_local_f1_yaml_sha_drift_fails_closed(evidence: _Evidence) -> None:
    evidence.f1_path.write_bytes(evidence.f1_path.read_bytes() + b"\n")
    with pytest.raises(F2LaunchGuardError, match="local F1 YAML SHA-256"):
        evidence.verify()


def test_terminal_log_byte_drift_fails_closed(evidence: _Evidence) -> None:
    log_path = Path(evidence.record["run"]["log"])
    log_path.write_bytes(log_path.read_bytes() + b"late-write\n")
    with pytest.raises(F2LaunchGuardError, match="byte size mismatch"):
        evidence.verify()


def test_terminal_marker_must_exist_in_authenticated_log(evidence: _Evidence) -> None:
    log_path = Path(evidence.record["run"]["log"])
    payload = b"training ended without a Lightning terminal marker\n"
    log_path.write_bytes(payload)
    evidence.record["run"]["log_bytes"] = len(payload)
    evidence.record["run"]["log_sha256"] = hashlib.sha256(payload).hexdigest()
    evidence.write_record_and_attestation()
    with pytest.raises(F2LaunchGuardError, match="terminal marker"):
        evidence.verify()


def test_duplicate_f1_yaml_key_is_rejected(evidence: _Evidence) -> None:
    evidence.f1_path.write_bytes(
        evidence.f1_path.read_bytes() + b"results:\n  status: failed\n  note: duplicate\n"
    )
    evidence.attestation["f1"]["record_sha256"] = hashlib.sha256(
        evidence.f1_path.read_bytes()
    ).hexdigest()
    evidence.write_attestation()
    with pytest.raises(F2LaunchGuardError, match="duplicate key 'results'"):
        evidence.verify()


def test_unknown_f1_field_is_rejected(evidence: _Evidence) -> None:
    evidence.record["unreviewed_trigger"] = True
    evidence.write_record_and_attestation()
    with pytest.raises(F2LaunchGuardError, match="unexpected=.*unreviewed_trigger"):
        evidence.verify()


def test_only_three_complete_epochs_cannot_trigger_f2(evidence: _Evidence) -> None:
    evidence.record["epoch_records"].pop()
    evidence.record["checkpoints"]["last"]["epoch"] = 2
    evidence.record["checkpoints"]["last"]["global_step"] = 2805
    evidence.write_record_and_attestation()
    with pytest.raises(F2LaunchGuardError, match="exactly four complete"):
        evidence.verify()


def test_running_f1_record_cannot_trigger_f2(evidence: _Evidence) -> None:
    evidence.record["run"]["status"] = "running"
    evidence.write_record_and_attestation()
    with pytest.raises(F2LaunchGuardError, match="terminal completion"):
        evidence.verify()


def test_index_four_or_fifth_epoch_is_never_accepted(evidence: _Evidence) -> None:
    evidence.record["selection"]["decision"]["final_scheduled_epoch_index"] = 4
    evidence.record["epoch_indexing_clarification"]["lightning_epoch_indices"] = [0, 1, 2, 3, 4]
    evidence.record["epoch_indexing_clarification"]["final_gate_epoch_index"] = 4
    evidence.record["epoch_indexing_clarification"]["fifth_epoch_authorized"] = True
    evidence.write_record_and_attestation()
    with pytest.raises(F2LaunchGuardError, match="final_scheduled_epoch_index"):
        evidence.verify()


def test_final_fixed_gate_conjunction_pass_forbids_f2(evidence: _Evidence) -> None:
    evidence.record["epoch_records"][3]["raw_metrics"] = _raw_metrics(passing=True)
    evidence.record["epoch_records"][3]["gate_pass"] = _gate_pass(passing=True)
    evidence.write_record_and_attestation()
    with pytest.raises(F2LaunchGuardError, match="passed the fixed-gate conjunction"):
        evidence.verify()


def test_recorded_gate_boolean_is_recomputed_not_trusted(evidence: _Evidence) -> None:
    evidence.record["epoch_records"][3]["gate_pass"]["critic_delta"] = True
    evidence.write_record_and_attestation()
    with pytest.raises(F2LaunchGuardError, match="gate_pass.critic_delta"):
        evidence.verify()


def test_attested_failed_gate_list_must_exactly_match_metrics(evidence: _Evidence) -> None:
    evidence.attestation["decision"]["failed_gates"] = ["critic_delta"]
    evidence.write_attestation()
    with pytest.raises(F2LaunchGuardError, match="do not exactly match"):
        evidence.verify()


def test_any_test_split_consultation_forbids_f2(evidence: _Evidence) -> None:
    evidence.record["data"]["test_splits_consulted"] = True
    evidence.write_record_and_attestation()
    with pytest.raises(F2LaunchGuardError, match="test_splits_consulted"):
        evidence.verify()


def test_attestation_cannot_claim_f2_already_started(evidence: _Evidence) -> None:
    evidence.attestation["f2_prelaunch"]["started"] = True
    evidence.write_attestation()
    with pytest.raises(F2LaunchGuardError, match="started"):
        evidence.verify()


def test_existing_f2_experiment_root_forbids_launch(evidence: _Evidence) -> None:
    evidence.experiment_root.mkdir(parents=True)
    with pytest.raises(F2LaunchGuardError, match="experiment root already exists"):
        evidence.verify()


def test_existing_atomic_launch_marker_forbids_relaunch(evidence: _Evidence) -> None:
    evidence.launch_marker.mkdir(parents=True)
    with pytest.raises(F2LaunchGuardError, match="launch marker already exists"):
        evidence.verify()


def test_run_uid_must_match_f1_yaml(evidence: _Evidence) -> None:
    evidence.attestation["f1"]["run_uid"] = "different-f1-run"
    evidence.write_attestation()
    with pytest.raises(F2LaunchGuardError, match="run UID differs"):
        evidence.verify()


def test_detached_digest_file_is_strict(evidence: _Evidence) -> None:
    digest = evidence.digest_path.read_text(encoding="ascii").strip()
    evidence.digest_path.write_text(f"{digest}  trigger.json\n", encoding="ascii")
    with pytest.raises(F2LaunchGuardError, match="one lowercase SHA-256"):
        read_expected_attestation_sha256(evidence.digest_path)


def test_launcher_is_a_single_explicit_protocol() -> None:
    launcher = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "training"
        / "train_actworld_jepa_f2.sh"
    )
    text = launcher.read_text(encoding="utf-8")
    required_fragments = (
        "--nproc_per_node=8",
        "agent.checkpoint_path=''",
        "agent.lr=0.00005",
        "agent.config.actworld_training_stage=critic",
        "agent.config.actworld_critic_variant=f2_base_residual",
        "agent.config.actworld_future_latent_grounding=true",
        "dataloader.params.batch_size=8",
        "trainer.params.max_epochs=4",
        "trainer.params.strategy=ddp",
        "resume_checkpoint_path=null",
        "cache_path=null",
        "use_cache_without_dataset=false",
        "f2_qualification.expected_manifest_sha256=",
        "f2_qualification.expected_config_sha256=",
        "f2_qualification.expected_cache_sha256=",
        "f2_qualification.expected_raw_input_content_manifest_sha256=",
        "f2_qualification.expected_source_sha256=",
        "f2_qualification.cache_metadata_path=",
        "f2_qualification.raw_input_content_manifest_path=",
        'export PYTHONDONTWRITEBYTECODE="1"',
        "ls-files -z --others --ignored --exclude-standard",
        "ignored import-shadow artifact exists in the F2 runtime",
        "f2_launch_authorization_v2.json",
        "f2_launch_authorization_v2.json.sig",
        "navsim.planning.script.f2_launch_guard verify",
        "--emit-contract-lines",
        "SOURCE_REVISION",
        "RAW_INPUT_MANIFEST_SHA256",
        "NAVSIM_LOG_ROOT",
        "SENSOR_BLOBS_ROOT",
        "QUALIFICATION_SOURCE_ROOT",
        '"navsim_log_path=${NAVSIM_LOG_ROOT}"',
        '"sensor_blobs_path=${SENSOR_BLOBS_ROOT}"',
        "f2_navtrain_input_manifest verify",
        "load_and_verify_metric_cache_manifest",
        "verified.verify_all(workers=16)",
        "FEASIBILITY_AUDIT_SHA256",
        "read_production_feasibility_audit",
        "preflight-runtime",
        "replay-audit",
        "--expected-config-sha256",
        "/usr/bin/git -C \"${REPOSITORY_ROOT}\" status --porcelain=v1 --untracked-files=all",
        "ACTWORLD_F2_PREFLIGHT_SMOKE_ONLY",
        "hydra.output_subdir=null",
        'F2_MIN_OUTPUT_FREE_BYTES="137438953472"',
        "os.statvfs",
        "signed output filesystem has less than 128 GiB free",
        "navsim.planning.script.f2_launch_guard claim",
        "ACTWORLD_F2_RUNTIME_CLAIM_SHA256",
        "--launcher-pid \"$$\"",
        "--emit-claim-lines",
        "--max_restarts=0",
    )
    for fragment in required_fragments:
        assert fragment in text
    assert text.index('export PYTHONDONTWRITEBYTECODE="1"') < text.index(
        'CONTRACT_OUTPUT="$("${ACTWORLD_PYTHON}"'
    )
    assert text.index("os.statvfs") < text.index(
        "navsim.planning.script.f2_launch_guard claim"
    )
    assert "--multirun" not in text
    assert "CACHE_PATH" not in text
    assert "${RESUME_CHECKPOINT" not in text
    assert ": \"${F2_EXPECTED_GIT_HEAD" not in text
    assert ": \"${F2_CACHE_CONTENT_MANIFEST_SHA256" not in text
    assert "ACTWORLD_PYTHON:-" not in text
    assert "mkdir -p -- \"${LAUNCH_LEDGER_ROOT}\"" not in text
    assert text.index("ACTWORLD_F2_PREFLIGHT_SMOKE_ONLY") < text.index(
        "navsim.planning.script.f2_launch_guard claim"
    )
    assert text.index("navsim.planning.script.f2_launch_guard claim") < text.index(
        "exec \"${ACTWORLD_PYTHON}\" -m torch.distributed.run"
    )
    assert text.index("navsim.planning.script.f2_launch_guard verify") < text.index(
        "exec \"${ACTWORLD_PYTHON}\" -m torch.distributed.run"
    )
