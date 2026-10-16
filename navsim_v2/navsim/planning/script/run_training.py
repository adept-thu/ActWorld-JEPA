from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, ContextManager, Mapping, Tuple
from pathlib import Path
import hashlib
import logging
import multiprocessing
import os
import stat
import time

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import (
    CacheOnlyDataset,
    Dataset,
    InProcessBatchPrefetchDataLoader,
    resolve_in_process_prefetch_batches,
)
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.validation.f2_frozen_artifact_io import (
    F2_VJEPA_CHECKPOINT_SHA256,
)
from navsim.planning.training.f2_qualification_manifest import (
    QualificationManifest,
    constrain_scene_filter,
    load_qualification_manifest,
    qualify_scene_loader,
    require_f2_sensor_contract,
)
from navsim.planning.validation.f2_metric_cache_manifest import (
    load_and_verify_metric_cache_manifest,
)
from navsim.planning.validation.f2_navtrain_input_manifest import (
    VerifiedNavtrainRawInputs,
    load_and_verify_raw_input_manifest,
)
from navsim.planning.script.f2_launch_guard import (
    F2LaunchGuardError,
    PRODUCTION_F2_AUTHORIZATION_PATH,
    PRODUCTION_F2_AUTHORIZATION_SIGNATURE_PATH,
    TRUSTED_AUTHORITY_SSH_PUBLIC_KEY,
    VerifiedF2RuntimeClaim,
    canonical_attestation_bytes,
    verify_f2_runtime_claim,
    verify_f2_signed_contract_lightweight,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"
F2_CRITIC_VARIANT = "f2_base_residual"
F2_AGENT_LR = 5e-5
F2_MAX_EPOCHS = 4
F2_PRUNED_CACHE_FAMILY = "train_metric_cache_v2_pruned"
F2_WORLD_SIZE = 8
F2_PREFLIGHT_SMOKE_ENV = "ACTWORLD_F2_PREFLIGHT_SMOKE_ONLY"
F2_NONFORMAL_DIRECT_TRAIN_ENV = "ACTWORLD_F2_NONFORMAL_DIRECT_TRAIN"
F2_NONFORMAL_RESUME_BATCH96_ENV = (
    "ACTWORLD_F2_NONFORMAL_RESUME_BATCH96"
)
F2_NONFORMAL_RESUME_BATCH288_ENV = (
    "ACTWORLD_F2_NONFORMAL_RESUME_BATCH288"
)
F2_NONFORMAL_RESUME_BATCH_SIZE = 96
F2_NONFORMAL_RESUME_BATCH288_SIZE = 288
F2_NONFORMAL_RESUME_NUM_WORKERS = 0
F2_TRAINING_LINEAGE_SCHEMA_VERSION = "actworld-f2-training-lineage-v1"
F2_TRAINING_LINEAGE_FILENAME = "f2_training_lineage.json"
F2_FROZEN_HYDRA_VALUES: Mapping[str, Any] = {
    "agent.checkpoint_path": "",
    "agent.lr": F2_AGENT_LR,
    "agent.config.actworld_training_stage": "critic",
    "agent.config.actworld_stage_a_checkpoint_path": "",
    "agent.config.actworld_critic_variant": F2_CRITIC_VARIANT,
    "agent.config.actworld_freeze_drive_jepa": True,
    "agent.config.actworld_future_latent_grounding": True,
    "agent.config.actworld_f2_aggregate_bce_weight": 1.0,
    "agent.config.actworld_f2_top_regret_margin_weight": 1.0,
    "agent.config.actworld_f2_top_regret_margin": 0.2,
    "agent.config.actworld_f2_safe_unsafe_margin_weight": 1.0,
    "agent.config.actworld_f2_safe_unsafe_margin": 0.5,
    "agent.config.actworld_f2_nine_channel_bce_weight": 1.0,
    "agent.config.actworld_f2_collision_bce_weight": 1.0,
    "agent.config.actworld_f2_historical_latent_weight": 0.25,
    "agent.config.actworld_f2_future_latent_weight": 0.25,
    "agent.config.actworld_score_workers": 4,
    "agent.config.actworld_score_timeout_seconds": 600.0,
    "agent.config.actworld_score_slow_log_seconds": 60.0,
    "cache_path": None,
    "use_cache_without_dataset": False,
    "force_cache_computation": False,
    "resume_checkpoint_path": None,
    "seed": 0,
    "dataloader.params.batch_size": 8,
    "dataloader.params.num_workers": 2,
    "dataloader.params.pin_memory": True,
    "dataloader.params.prefetch_factor": 4,
    "trainer.params.max_epochs": F2_MAX_EPOCHS,
    "trainer.params.check_val_every_n_epoch": 1,
    "trainer.params.val_check_interval": 1.0,
    "trainer.params.limit_train_batches": 1.0,
    "trainer.params.limit_val_batches": 1.0,
    "trainer.params.accelerator": "gpu",
    "trainer.params.strategy": "ddp",
    "trainer.params.precision": "16-mixed",
    "trainer.params.num_nodes": 1,
    "trainer.params.num_sanity_val_steps": 1,
    "trainer.params.fast_dev_run": False,
    "trainer.params.accumulate_grad_batches": 1,
    "trainer.params.gradient_clip_val": 1.0,
    "trainer.params.gradient_clip_algorithm": "norm",
}
_F2_RAW_INPUT_CACHE: dict[tuple[str, ...], VerifiedNavtrainRawInputs] = {}


def _restrict_scene_filter_logs(
    scene_filter: SceneFilter, configured_logs: Any
) -> None:
    """Apply the official train/val log partition before token qualification."""

    if scene_filter.log_names is not None:
        scene_filter.log_names = [
            log_name
            for log_name in scene_filter.log_names
            if log_name in configured_logs
        ]
    else:
        scene_filter.log_names = list(configured_logs)


def load_configured_f2_qualification_manifest(
    cfg: DictConfig,
    *,
    manifest_loader: Callable[..., QualificationManifest] = load_qualification_manifest,
) -> QualificationManifest:
    """Load the explicitly authenticated train/held-out F2 token manifest."""

    manifest_path = OmegaConf.select(cfg, "f2_qualification.manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ValueError(
            "F2 requires f2_qualification.manifest_path to be an explicit file"
        )
    expected = {}
    for field_name in (
        "manifest_sha256",
        "config_sha256",
        "cache_sha256",
        "source_sha256",
    ):
        value = OmegaConf.select(
            cfg, f"f2_qualification.expected_{field_name}"
        )
        if not isinstance(value, str) or not value:
            raise ValueError(
                "F2 requires explicit f2_qualification.expected_"
                f"{field_name}"
            )
        expected[field_name] = value
    return manifest_loader(
        Path(manifest_path),
        expected_manifest_sha256=expected["manifest_sha256"],
        expected_config_sha256=expected["config_sha256"],
        expected_cache_sha256=expected["cache_sha256"],
        expected_source_sha256=expected["source_sha256"],
    )


def authenticate_configured_f2_cache_metadata(
    cfg: DictConfig,
    *,
    metric_cache_loader_factory: Callable[..., Any] | None = None,
) -> Any:
    """Bind the CSV and every qualified pickle consumed by F2 training."""

    metadata_path = OmegaConf.select(cfg, "f2_qualification.cache_metadata_path")
    if not isinstance(metadata_path, str) or not metadata_path:
        raise ValueError(
            "F2 requires f2_qualification.cache_metadata_path to be an explicit CSV"
        )
    expected_sha256 = OmegaConf.select(
        cfg, "f2_qualification.expected_cache_sha256"
    )
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(
            "F2 requires f2_qualification.expected_cache_sha256"
        )
    content_manifest_path = OmegaConf.select(
        cfg, "f2_qualification.cache_content_manifest_path"
    )
    if not isinstance(content_manifest_path, str) or not content_manifest_path:
        raise ValueError(
            "F2 requires f2_qualification.cache_content_manifest_path"
        )
    expected_content_sha256 = OmegaConf.select(
        cfg, "f2_qualification.expected_cache_content_manifest_sha256"
    )
    if not isinstance(expected_content_sha256, str) or not expected_content_sha256:
        raise ValueError(
            "F2 requires f2_qualification.expected_cache_content_manifest_sha256"
        )
    baseline_path = OmegaConf.select(
        cfg, "agent.config.baseline_checkpoint_path", default=""
    )
    if not isinstance(baseline_path, str) or not baseline_path:
        raise ValueError("F2 requires baseline_checkpoint_path before cache binding")
    cache_root = Path(baseline_path).expanduser().resolve().parent / F2_PRUNED_CACHE_FAMILY
    configured_metadata = Path(metadata_path).expanduser()
    configured_metadata_resolved = configured_metadata.resolve()
    if configured_metadata_resolved.parent.parent != cache_root:
        raise ValueError(
            "F2 cache_metadata_path is outside the baseline Drive-JEPA cache root"
        )
    if metric_cache_loader_factory is None:
        from navsim.common.dataloader import MetricCacheLoader

        metric_cache_loader_factory = MetricCacheLoader
    qualification = load_configured_f2_qualification_manifest(cfg)
    verified_content_manifest = load_and_verify_metric_cache_manifest(
        Path(content_manifest_path).expanduser(),
        expected_sha256=expected_content_sha256,
        cache_root=cache_root,
        metadata_path=configured_metadata,
        expected_metadata_sha256=expected_sha256,
        qualification=qualification,
        expected_qualification_manifest_sha256=str(
            OmegaConf.select(cfg, "f2_qualification.expected_manifest_sha256")
        ),
    )
    return metric_cache_loader_factory(
        cache_root,
        expected_metadata_path=configured_metadata,
        expected_metadata_sha256=expected_sha256,
        authenticated_content_manifest=verified_content_manifest,
    )


def authenticate_configured_f2_raw_inputs(
    cfg: DictConfig,
) -> VerifiedNavtrainRawInputs:
    """Bind qualified annotation pickles and every consumed sensor blob."""

    manifest_path = OmegaConf.select(
        cfg, "f2_qualification.raw_input_content_manifest_path"
    )
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ValueError(
            "F2 requires f2_qualification.raw_input_content_manifest_path"
        )
    expected_sha256 = OmegaConf.select(
        cfg, "f2_qualification.expected_raw_input_content_manifest_sha256"
    )
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(
            "F2 requires f2_qualification."
            "expected_raw_input_content_manifest_sha256"
        )
    log_root = Path(str(cfg.navsim_log_path)).expanduser()
    sensor_root = Path(str(cfg.sensor_blobs_path)).expanduser()
    qualification_sha256 = str(
        OmegaConf.select(cfg, "f2_qualification.expected_manifest_sha256")
    )
    cache_key = (
        str(Path(manifest_path).expanduser().absolute()),
        expected_sha256,
        str(log_root.absolute()),
        str(sensor_root.absolute()),
        qualification_sha256,
    )
    cached = _F2_RAW_INPUT_CACHE.get(cache_key)
    if cached is not None:
        cached.require_roots(log_root, sensor_root)
        return cached
    qualification = load_configured_f2_qualification_manifest(cfg)
    verified = load_and_verify_raw_input_manifest(
        Path(manifest_path).expanduser(),
        expected_sha256=expected_sha256,
        log_root=log_root,
        sensor_root=sensor_root,
        qualification=qualification,
        expected_qualification_manifest_sha256=qualification_sha256,
    )
    _F2_RAW_INPUT_CACHE[cache_key] = verified
    return verified


@contextmanager
def authenticated_f2_training_cache_context(
    cfg: DictConfig,
    *,
    metric_cache_loader_factory: Callable[..., Any] | None = None,
) -> Any:
    """Make Drive-JEPA consume the already authenticated F2 CSV mapping."""

    authenticated_loader = authenticate_configured_f2_cache_metadata(
        cfg, metric_cache_loader_factory=metric_cache_loader_factory
    )
    metadata_path = Path(
        str(OmegaConf.select(cfg, "f2_qualification.cache_metadata_path"))
    ).expanduser().resolve()
    expected_cache_root = metadata_path.parent.parent
    from navsim.agents.drive_jepa_perception_based import drive_jepa_agent

    original_loader = drive_jepa_agent.MetricCacheLoader
    calls = 0

    def consume_authenticated_loader(
        cache_path: Any, *args: Any, **kwargs: Any
    ) -> Any:
        nonlocal calls
        if Path(cache_path).expanduser().resolve() != expected_cache_root:
            raise ValueError(
                "F2 Drive-JEPA requested a metric-cache root outside the authenticated cache"
            )
        if args or kwargs:
            raise ValueError(
                "F2 Drive-JEPA must consume the pre-authenticated metric-cache loader"
            )
        calls += 1
        if calls != 1:
            raise ValueError("F2 Drive-JEPA consumed the metric cache more than once")
        return authenticated_loader

    drive_jepa_agent.MetricCacheLoader = consume_authenticated_loader
    try:
        yield authenticated_loader
        if calls != 1:
            raise ValueError(
                "F2 Drive-JEPA did not consume the authenticated metric-cache loader"
            )
    finally:
        drive_jepa_agent.MetricCacheLoader = original_loader


def build_training_agent(
    cfg: DictConfig,
    *,
    agent_factory: Callable[[Any], AbstractAgent] = instantiate,
    vjepa_context_factory: Callable[..., ContextManager[None]] | None = None,
    metric_cache_context_factory: Callable[..., ContextManager[Any]] | None = None,
) -> AbstractAgent:
    """Construct F2 only inside the authenticated same-descriptor V-JEPA context."""

    critic_variant = OmegaConf.select(
        cfg, "agent.config.actworld_critic_variant", default="legacy"
    )
    if critic_variant != F2_CRITIC_VARIANT:
        return agent_factory(cfg.agent)
    baseline_path = Path(
        str(OmegaConf.select(cfg, "agent.config.baseline_checkpoint_path"))
    ).expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[4]
    experiment_root = baseline_path.parent.parent
    if vjepa_context_factory is None:
        from navsim.planning.validation.f2_navsim_runtime_adapters import (
            authenticated_f2_vjepa_load_context,
        )

        vjepa_context_factory = authenticated_f2_vjepa_load_context
    training_stage = OmegaConf.select(
        cfg, "agent.config.actworld_training_stage", default="critic"
    )
    if training_stage == "actor":
        # Stage-B consumes a frozen, already-trained F2 world critic.  It does
        # not use the preregistered Stage-A cache partition, but it must retain
        # the same authenticated V-JEPA backbone load as its parent.
        with vjepa_context_factory(
            repository_root,
            experiment_root,
            expected_vjepa_sha256=F2_VJEPA_CHECKPOINT_SHA256,
        ):
            return agent_factory(cfg.agent)
    if metric_cache_context_factory is None:
        metric_cache_context_factory = authenticated_f2_training_cache_context
    with metric_cache_context_factory(cfg) as authenticated_cache:
        with vjepa_context_factory(
            repository_root,
            experiment_root,
            expected_vjepa_sha256=F2_VJEPA_CHECKPOINT_SHA256,
        ):
            agent = agent_factory(cfg.agent)
        if (
            getattr(agent, "train_metric_cache_paths", None)
            is not authenticated_cache.metric_cache_paths
        ):
            raise ValueError(
                "F2 agent train_metric_cache_paths did not come from authenticated metadata"
            )
        return agent


def validate_training_lineage(cfg: DictConfig) -> None:
    """Reject actor runs that have neither a Stage-A parent nor a full resume."""

    stage = OmegaConf.select(cfg, "agent.config.actworld_training_stage")
    if stage is None:
        return
    stage_a_checkpoint = OmegaConf.select(
        cfg, "agent.config.actworld_stage_a_checkpoint_path", default=""
    )
    stage1_checkpoint = OmegaConf.select(
        cfg, "agent.config.actworld_stage1_checkpoint_path", default=""
    )
    resume_checkpoint = OmegaConf.select(cfg, "resume_checkpoint_path")
    if stage_a_checkpoint and resume_checkpoint:
        raise ValueError(
            "Use either a selective Stage-A parent or a full Lightning resume, "
            "not both"
        )
    if stage == "actor" and not stage_a_checkpoint and not resume_checkpoint:
        raise ValueError(
            "ActWorld actor training requires either "
            "agent.config.actworld_stage_a_checkpoint_path for a fresh run "
            "or resume_checkpoint_path for a full Stage-B resume"
        )
    if stage == "actor" and resume_checkpoint:
        resume_path = Path(str(resume_checkpoint))
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
    if stage == "critic" and stage_a_checkpoint:
        raise ValueError(
            "agent.config.actworld_stage_a_checkpoint_path is only valid for "
            "ActWorld actor training"
        )
    if stage1_checkpoint and resume_checkpoint:
        raise ValueError(
            "Use either the Stage-1 model-only parent or a full Lightning "
            "resume, not both"
        )
    if stage1_checkpoint and stage != "critic":
        raise ValueError(
            "agent.config.actworld_stage1_checkpoint_path is only valid for "
            "selective critic Stage 2"
        )


def _required_environment_value(
    environment: Mapping[str, str], name: str, expected: str
) -> None:
    actual = environment.get(name)
    if actual != expected:
        raise ValueError(
            f"F2 environment {name} differs from the signed authorization"
        )


def _required_config_value(cfg: DictConfig, path: str, expected: Any) -> None:
    actual = OmegaConf.select(cfg, path)
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"F2 config {path} differs from the signed authorization: "
            f"expected {expected!r}, got {actual!r}"
        )


def _open_absolute_directory_nofollow(path: Path) -> tuple[int, os.stat_result]:
    """Walk every absolute path component with openat(O_NOFOLLOW)."""

    path = Path(path)
    lexical = Path(os.path.abspath(str(path)))
    if not path.is_absolute() or lexical != path:
        raise ValueError("F2 signed output_dir must be absolute and normalized")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path.anchor, directory_flags)
        try:
            for component in path.parts[1:]:
                if component in {"", ".", ".."} or "/" in component:
                    raise ValueError("F2 signed output_dir component is unsafe")
                next_descriptor = os.open(
                    component, directory_flags, dir_fd=descriptor
                )
                os.close(descriptor)
                descriptor = next_descriptor
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("F2 signed output_dir is not a directory")
            return descriptor, metadata
        except BaseException:
            os.close(descriptor)
            raise
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(
            "F2 signed output_dir parent chain must contain no symlink"
        ) from error


def _write_training_lineage_no_overwrite(
    output_dir_fd: int, payload: bytes
) -> None:
    """Publish complete rank-0 lineage atomically without replacing a leaf."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    temporary_name = (
        f".{F2_TRAINING_LINEAGE_FILENAME}."
        f"{hashlib.sha256(payload).hexdigest()}.tmp"
    )
    try:
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=output_dir_fd,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                F2_TRAINING_LINEAGE_FILENAME,
                src_dir_fd=output_dir_fd,
                dst_dir_fd=output_dir_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise ValueError(
                "F2 training lineage already exists; "
                "duplicate/replayed training is forbidden"
            ) from error
        finally:
            try:
                os.unlink(temporary_name, dir_fd=output_dir_fd)
            except FileNotFoundError:
                pass
        os.fsync(output_dir_fd)
    except FileExistsError as error:
        raise ValueError(
            "F2 training lineage temporary claim already exists; "
            "duplicate/replayed training is forbidden"
        ) from error
    except OSError as error:
        raise ValueError("could not create the F2 training lineage receipt") from error


def _read_training_lineage_from_fd(output_dir_fd: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(
            F2_TRAINING_LINEAGE_FILENAME, flags, dir_fd=output_dir_fd
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
                raise ValueError("F2 training lineage is not a regular file")
            if metadata.st_size > 64 * 1024:
                raise ValueError("F2 training lineage is unexpectedly large")
            chunks = []
            while True:
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("could not read the F2 training lineage") from error


def _bind_training_lineage(
    claim: VerifiedF2RuntimeClaim,
    *,
    rank: int,
    world_size: int,
) -> Path:
    output_dir_fd, output_metadata = _open_absolute_directory_nofollow(
        claim.output_dir
    )
    payload = {
        "schema_version": F2_TRAINING_LINEAGE_SCHEMA_VERSION,
        "authorization": {
            "sha256": claim.authorization_sha256,
            "signature_sha256": claim.signature_sha256,
        },
        "source_revision": claim.source_revision,
        "runtime_claim": {
            "path": str(claim.path),
            "sha256": claim.sha256,
        },
        "experiment": {
            "name": claim.experiment_name,
            "uid": claim.experiment_uid,
            "output_dir": str(claim.output_dir),
            "output_directory_identity": {
                "device": output_metadata.st_dev,
                "inode": output_metadata.st_ino,
            },
        },
        "torchrun": {
            "boot_id": claim.boot_id,
            "pid": claim.launcher_pid,
            "start_time_ticks": claim.launcher_start_time_ticks,
            "world_size": world_size,
        },
    }
    raw = canonical_attestation_bytes(payload)
    expected_sha256 = hashlib.sha256(raw).hexdigest()
    lineage_path = claim.output_dir / F2_TRAINING_LINEAGE_FILENAME
    try:
        if rank == 0:
            _write_training_lineage_no_overwrite(output_dir_fd, raw)
        deadline = time.monotonic() + 30.0
        while True:
            try:
                observed = _read_training_lineage_from_fd(output_dir_fd)
                break
            except FileNotFoundError:
                if rank == 0 or time.monotonic() >= deadline:
                    raise ValueError(
                        "rank 0 did not create the F2 training lineage"
                    )
                time.sleep(0.05)
        if observed != raw or hashlib.sha256(observed).hexdigest() != expected_sha256:
            raise ValueError("F2 training lineage bytes differ across ranks")
    finally:
        os.close(output_dir_fd)
    return lineage_path


def verify_configured_f2_runtime_admission(
    cfg: DictConfig,
    *,
    authorization_path: Path = PRODUCTION_F2_AUTHORIZATION_PATH,
    signature_path: Path = PRODUCTION_F2_AUTHORIZATION_SIGNATURE_PATH,
    trusted_public_key: str = TRUSTED_AUTHORITY_SSH_PUBLIC_KEY,
    environment: Mapping[str, str] | None = None,
    proc_root: Path = Path("/proc"),
    parent_pid: int | None = None,
    repository_root: Path | None = None,
    require_runtime_claim: bool = True,
) -> VerifiedF2RuntimeClaim | None:
    """Replay the signed launcher contract at every F2 process boundary.

    This intentionally authenticates only the small SSHSIG envelope and claim.
    Baseline/V-JEPA checkpoints, the metric cache, and raw inputs retain their
    existing same-descriptor content authentication paths.
    """

    environment = os.environ if environment is None else environment
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[4]
    try:
        contract = verify_f2_signed_contract_lightweight(
            authorization_path,
            signature_path,
            repository_root=repository_root,
            trusted_public_key=trusted_public_key,
        )
    except F2LaunchGuardError as error:
        raise ValueError(f"F2 signed runtime admission failed: {error}") from error

    # Every Hydra value whose source is the signed envelope is compared again;
    # a direct torchrun invocation cannot substitute a different path/digest or
    # output UID while retaining the valid authorization.
    config_contract = {
        "agent.config.baseline_checkpoint_path": str(
            contract.artifacts["baseline_checkpoint"].path
        ),
        "f2_qualification.manifest_path": str(
            contract.artifacts["qualification_manifest"].path
        ),
        "f2_qualification.cache_metadata_path": str(
            contract.artifacts["cache_metadata"].path
        ),
        "f2_qualification.cache_content_manifest_path": str(
            contract.artifacts["cache_content_manifest"].path
        ),
        "f2_qualification.raw_input_content_manifest_path": str(
            contract.artifacts["raw_input_manifest"].path
        ),
        "f2_qualification.expected_manifest_sha256": contract.artifacts[
            "qualification_manifest"
        ].sha256,
        "f2_qualification.expected_config_sha256": contract.artifacts[
            "qualification_config"
        ].sha256,
        "f2_qualification.expected_cache_sha256": contract.artifacts[
            "cache_metadata"
        ].sha256,
        "f2_qualification.expected_cache_content_manifest_sha256": contract.artifacts[
            "cache_content_manifest"
        ].sha256,
        "f2_qualification.expected_raw_input_content_manifest_sha256": contract.artifacts[
            "raw_input_manifest"
        ].sha256,
        "f2_qualification.expected_source_sha256": contract.artifacts[
            "qualification_source_manifest"
        ].sha256,
        "experiment_name": contract.experiment_name,
        "experiment_uid": contract.experiment_uid,
        "output_dir": str(contract.output_dir),
        "trainer.params.default_root_dir": str(contract.output_dir),
        "navsim_log_path": str(contract.runtime["navsim_log_root"]),
        "sensor_blobs_path": str(contract.runtime["sensor_blobs_root"]),
    }
    for path, expected in config_contract.items():
        _required_config_value(cfg, path, expected)

    environment_contract = {
        "ACTWORLD_F2_AUTHORIZATION_SHA256": contract.authorization_sha256,
        "ACTWORLD_F2_AUTHORIZATION_SIGNATURE_SHA256": contract.signature_sha256,
        "ACTWORLD_F2_SOURCE_REVISION": contract.source_revision,
        "ACTWORLD_F2_OUTPUT_DIR": str(contract.output_dir),
        "ACTWORLD_F2_LAUNCH_MARKER": str(contract.launch_marker),
        "NAVSIM_EXP_ROOT": str(contract.runtime["navsim_exp_root"]),
        "OPENSCENE_DATA_ROOT": str(contract.runtime["openscene_data_root"]),
        "NUPLAN_MAPS_ROOT": str(contract.runtime["nuplan_maps_root"]),
        "ACTWORLD_F2_NAVSIM_LOG_ROOT": str(contract.runtime["navsim_log_root"]),
        "ACTWORLD_F2_SENSOR_BLOBS_ROOT": str(contract.runtime["sensor_blobs_root"]),
        "ACTWORLD_F2_QUALIFICATION_SOURCE_ROOT": str(
            contract.runtime["qualification_source_root"]
        ),
    }
    for name, artifact in contract.artifacts.items():
        prefix = f"ACTWORLD_F2_{name.upper()}"
        environment_contract[f"{prefix}_PATH"] = str(artifact.path)
        environment_contract[f"{prefix}_SHA256"] = artifact.sha256
    for name, expected in environment_contract.items():
        _required_environment_value(environment, name, expected)

    if not require_runtime_claim:
        return None

    world_values = {}
    for name in (
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "RANK",
        "LOCAL_RANK",
    ):
        raw = environment.get(name)
        if raw is None or not raw.isdigit():
            raise ValueError(f"F2 requires numeric torchrun environment {name}")
        world_values[name] = int(raw)
    if world_values["WORLD_SIZE"] != F2_WORLD_SIZE:
        raise ValueError("F2 requires WORLD_SIZE=8")
    if world_values["LOCAL_WORLD_SIZE"] != F2_WORLD_SIZE:
        raise ValueError("F2 requires LOCAL_WORLD_SIZE=8")
    if world_values["GROUP_RANK"] != 0:
        raise ValueError("F2 requires GROUP_RANK=0 on the single authorized node")
    rank = world_values["RANK"]
    local_rank = world_values["LOCAL_RANK"]
    if rank not in range(F2_WORLD_SIZE) or local_rank != rank:
        raise ValueError("F2 requires one-node ranks/local-ranks 0 through 7")

    claim_path = environment.get("ACTWORLD_F2_RUNTIME_CLAIM_PATH")
    claim_sha256 = environment.get("ACTWORLD_F2_RUNTIME_CLAIM_SHA256")
    if claim_path != str(contract.launch_marker):
        raise ValueError("F2 runtime claim path differs from signed launch marker")
    if not isinstance(claim_sha256, str):
        raise ValueError("F2 runtime claim SHA-256 environment is missing")
    direct_parent = os.getppid() if parent_pid is None else parent_pid
    try:
        claim = verify_f2_runtime_claim(
            Path(claim_path),
            expected_sha256=claim_sha256,
            signed_contract=contract,
            expected_parent_pid=direct_parent,
            proc_root=proc_root,
        )
    except F2LaunchGuardError as error:
        raise ValueError(f"F2 runtime claim admission failed: {error}") from error
    _bind_training_lineage(
        claim,
        rank=rank,
        world_size=world_values["WORLD_SIZE"],
    )
    return claim


def _validate_frozen_f2_hydra_protocol(
    cfg: DictConfig,
    *,
    allow_nonformal_resume_batch96: bool = False,
) -> None:
    """Bind every non-path Hydra override to the signed source revision."""

    for path, expected in F2_FROZEN_HYDRA_VALUES.items():
        if allow_nonformal_resume_batch96 and path in {
            "resume_checkpoint_path",
            "dataloader.params.batch_size",
            "dataloader.params.num_workers",
            "dataloader.params.prefetch_factor",
        }:
            continue
        _required_config_value(cfg, path, expected)


def validate_f2_run_protocol(
    cfg: DictConfig,
    *,
    runtime_admission_verifier: Callable[[DictConfig], Any] | None = None,
) -> None:
    """Fail before model construction if the frozen F2 run contract drifted."""

    critic_variant = OmegaConf.select(
        cfg, "agent.config.actworld_critic_variant", default="legacy"
    )
    if critic_variant != F2_CRITIC_VARIANT:
        return

    stage = OmegaConf.select(cfg, "agent.config.actworld_training_stage")
    if stage == "actor":
        if OmegaConf.select(
            cfg, "agent.config.actworld_freeze_drive_jepa", default=False
        ) is not True:
            raise ValueError("F2 Stage-B requires a frozen Drive-JEPA baseline")
        stage_a = OmegaConf.select(
            cfg, "agent.config.actworld_stage_a_checkpoint_path", default=""
        )
        resume = OmegaConf.select(cfg, "resume_checkpoint_path", default=None)
        if not stage_a and resume in (None, ""):
            raise ValueError(
                "F2 Stage-B requires actworld_stage_a_checkpoint_path or "
                "resume_checkpoint_path"
            )
        if stage_a and not Path(stage_a).is_file():
            raise FileNotFoundError(stage_a)
        if resume not in (None, "") and not Path(resume).is_file():
            raise FileNotFoundError(resume)
        return

    if runtime_admission_verifier is None:
        nonformal_direct = (
            os.environ.get(F2_NONFORMAL_DIRECT_TRAIN_ENV) == "1"
        )
        nonformal_resume_batch96 = nonformal_direct and (
            os.environ.get(F2_NONFORMAL_RESUME_BATCH96_ENV) == "1"
        )
        nonformal_resume_batch288 = nonformal_direct and (
            os.environ.get(F2_NONFORMAL_RESUME_BATCH288_ENV) == "1"
        )
        if nonformal_resume_batch96 and nonformal_resume_batch288:
            raise ValueError(
                "F2 non-formal resume batch modes are mutually exclusive"
            )
        nonformal_resume_batch_size = (
            F2_NONFORMAL_RESUME_BATCH_SIZE
            if nonformal_resume_batch96
            else F2_NONFORMAL_RESUME_BATCH288_SIZE
            if nonformal_resume_batch288
            else None
        )
        if nonformal_direct:
            logger.warning(
                "F2 non-formal direct training explicitly enabled: signed "
                "runtime admission and one-shot claim are bypassed; this run "
                "is ineligible for formal paper results"
            )
        else:
            smoke_only = os.environ.get(F2_PREFLIGHT_SMOKE_ENV) == "1"
            verify_configured_f2_runtime_admission(
                cfg, require_runtime_claim=not smoke_only
            )
        _validate_frozen_f2_hydra_protocol(
            cfg,
            allow_nonformal_resume_batch96=(
                nonformal_resume_batch_size is not None
            ),
        )
    else:
        nonformal_resume_batch96 = False
        nonformal_resume_batch288 = False
        nonformal_resume_batch_size = None
        runtime_admission_verifier(cfg)

    if stage != "critic":
        raise ValueError("F2 is restricted to a fresh Stage-A critic run")
    agent_checkpoint = OmegaConf.select(cfg, "agent.checkpoint_path", default="")
    if not isinstance(agent_checkpoint, str) or agent_checkpoint:
        raise ValueError(
            "F2 training requires an empty agent.checkpoint_path and must "
            "start through baseline_checkpoint_path"
        )
    resume_checkpoint = OmegaConf.select(cfg, "resume_checkpoint_path")
    if resume_checkpoint not in (None, ""):
        if nonformal_resume_batch_size is None:
            raise ValueError(
                "F2 preregistration requires a fresh, non-resumed run"
            )
        if not isinstance(resume_checkpoint, str):
            raise ValueError(
                "F2 non-formal large-batch resume requires a checkpoint path"
            )
        if not Path(resume_checkpoint).is_file():
            raise FileNotFoundError(resume_checkpoint)

    if nonformal_resume_batch_size is not None:
        batch_size = OmegaConf.select(cfg, "dataloader.params.batch_size")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size != nonformal_resume_batch_size
        ):
            raise ValueError(
                "F2 non-formal large-batch resume requires "
                "dataloader.params.batch_size="
                f"{nonformal_resume_batch_size}"
            )
        if resume_checkpoint in (None, ""):
            raise ValueError(
                "F2 non-formal large-batch resume requires "
                "resume_checkpoint_path"
            )
        num_workers = OmegaConf.select(
            cfg, "dataloader.params.num_workers"
        )
        if (
            isinstance(num_workers, bool)
            or not isinstance(num_workers, int)
            or num_workers != F2_NONFORMAL_RESUME_NUM_WORKERS
        ):
            raise ValueError(
                "F2 non-formal large-batch resume requires "
                "dataloader.params.num_workers=0"
            )
        prefetch_factor = OmegaConf.select(
            cfg, "dataloader.params.prefetch_factor"
        )
        if prefetch_factor is not None:
            raise ValueError(
                "F2 non-formal large-batch resume requires "
                "dataloader.params.prefetch_factor=null"
            )

    seed = OmegaConf.select(cfg, "seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 0:
        raise ValueError("F2 preregistration requires seed=0")

    lr = OmegaConf.select(cfg, "agent.lr")
    if (
        isinstance(lr, bool)
        or not isinstance(lr, (int, float))
        or float(lr) != F2_AGENT_LR
    ):
        raise ValueError("F2 preregistration requires agent.lr=5e-5")
    max_epochs = OmegaConf.select(cfg, "trainer.params.max_epochs")
    if (
        isinstance(max_epochs, bool)
        or not isinstance(max_epochs, int)
        or max_epochs != F2_MAX_EPOCHS
    ):
        raise ValueError("F2 preregistration requires trainer.params.max_epochs=4")

    if OmegaConf.select(
        cfg,
        "agent.config.actworld_future_latent_grounding",
        default=False,
    ) is not True:
        raise ValueError("F2 training requires future latent grounding")
    if OmegaConf.select(
        cfg, "agent.config.actworld_freeze_drive_jepa", default=False
    ) is not True:
        raise ValueError("F2 requires a frozen Drive-JEPA baseline")

    baseline = OmegaConf.select(
        cfg, "agent.config.baseline_checkpoint_path", default=""
    )
    if not isinstance(baseline, str) or not baseline:
        raise ValueError("F2 requires baseline_checkpoint_path")
    if not Path(baseline).is_file():
        raise FileNotFoundError(Path(baseline))

    cache_only = OmegaConf.select(
        cfg, "use_cache_without_dataset", default=False
    )
    if not isinstance(cache_only, bool):
        raise ValueError("F2 use_cache_without_dataset must be a bool")
    if cache_only:
        # CacheOnlyDataset verifies builder filenames, not payload keys,
        # sensor-frame coverage, or the fixed held-out token manifest.  Until
        # an audited cache index carries those proofs, accepting this path
        # would make future-target completeness unverifiable.
        raise ValueError(
            "F2 forbids use_cache_without_dataset: current cache metadata "
            "cannot prove future-target and held-out-manifest completeness"
        )

    # Authenticate the explicit token-level train/held-out partition before
    # model construction.  build_datasets reopens the same immutable file so
    # no in-memory object can bypass the dataset-construction boundary.
    load_configured_f2_qualification_manifest(cfg)
    authenticate_configured_f2_cache_metadata(cfg)
    authenticate_configured_f2_raw_inputs(cfg)


def build_checkpoint_callback(
    output_dir: str,
    training_stage: str,
    critic_variant: str,
) -> ModelCheckpoint:
    """Build the stage-aware checkpoint policy used by the training entrypoint."""

    # Explicit experiment-only actor snapshots, ranked by external NAVTRAIN.
    actor_interval = int(os.environ.get("ACTWORLD_ACTOR_CHECKPOINT_STEPS", "0"))
    if actor_interval < 0:
        raise ValueError("ACTWORLD_ACTOR_CHECKPOINT_STEPS must be nonnegative")
    if training_stage == "actor" and actor_interval > 0:
        return ModelCheckpoint(
            dirpath=output_dir + "/checkpoints/",
            save_top_k=-1, save_last=True, monitor=None,
            filename="step_{step}", auto_insert_metric_name=False,
            every_n_train_steps=actor_interval, every_n_epochs=0,
            save_on_train_epoch_end=False,
        )

    if training_stage == "critic" and critic_variant == "f2_base_residual":
        # Parallel validation is deliberately non-authoritative for F2.  Keep
        # every epoch so the external causal navtrain validator can rank them
        # without a false in-loop checkpoint monitor.  Disable automatic
        # metric-name insertion so the archive names are exactly epoch_N.ckpt.
        return ModelCheckpoint(
            dirpath=output_dir + "/checkpoints/",
            save_top_k=-1,
            save_last=True,
            monitor=None,
            filename="epoch_{epoch}",
            auto_insert_metric_name=False,
            every_n_epochs=1,
        )

    checkpoint_monitor = (
        "val/critic_selected_score_epoch"
        if training_stage == "critic"
        else "val/score_epoch"
    )
    return ModelCheckpoint(
        dirpath=output_dir + "/checkpoints/",
        save_top_k=1,
        save_last=True,
        monitor=checkpoint_monitor,
        filename="best_{epoch}",
        mode="max",
    )


def build_datasets(
    cfg: DictConfig,
    agent: AbstractAgent,
    *,
    scene_loader_factory: Callable[..., Any] = SceneLoader,
    dataset_factory: Callable[..., Any] = Dataset,
    qualification_manifest_loader: Callable[
        ..., QualificationManifest
    ] = load_qualification_manifest,
) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    strict_f2_sensor_preflight = (
        getattr(agent, "_training_stage", None) == "critic"
        and getattr(agent, "_critic_variant", "legacy") == F2_CRITIC_VARIANT
    )
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    # The official log partition remains a mandatory outer boundary.  F2's
    # authenticated token rows then provide the finer split inside each log;
    # they never broaden train_logs/val_logs.
    _restrict_scene_filter_logs(train_scene_filter, cfg.train_logs)
    _restrict_scene_filter_logs(val_scene_filter, cfg.val_logs)

    qualification_manifest = None
    authenticated_raw_inputs = None
    if strict_f2_sensor_preflight:
        qualification_manifest = load_configured_f2_qualification_manifest(
            cfg, manifest_loader=qualification_manifest_loader
        )
        constrain_scene_filter(train_scene_filter, qualification_manifest.train)
        constrain_scene_filter(val_scene_filter, qualification_manifest.held_out)
        authenticated_raw_inputs = authenticate_configured_f2_raw_inputs(cfg)

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    synthetic_scenes_path = Path(cfg.synthetic_scenes_path)

    sensor_config = agent.get_sensor_config()
    if strict_f2_sensor_preflight:
        require_f2_sensor_contract(sensor_config)
    raw_input_kwargs = (
        {"authenticated_raw_inputs": authenticated_raw_inputs}
        if authenticated_raw_inputs is not None
        else {}
    )

    train_scene_loader = scene_loader_factory(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        synthetic_scenes_path=synthetic_scenes_path,
        scene_filter=train_scene_filter,
        sensor_config=sensor_config,
        **raw_input_kwargs,
    )

    val_scene_loader = scene_loader_factory(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        synthetic_scenes_path=synthetic_scenes_path,
        scene_filter=val_scene_filter,
        sensor_config=sensor_config,
        **raw_input_kwargs,
    )

    for split_name, scene_loader in (
        ("train", train_scene_loader),
        ("val", val_scene_loader),
    ):
        if strict_f2_sensor_preflight:
            assert qualification_manifest is not None
            availability = qualify_scene_loader(
                scene_loader,
                qualification_manifest.partition(split_name),
            )
        else:
            availability = scene_loader.filter_scenes_by_sensor_availability()
        log_method = logger.warning if availability.removed_scenes else logger.info
        log_method(
            "Sensor availability preflight (%s): removed %d/%d scenes "
            "(%d regular, %d synthetic); retained %d",
            split_name,
            availability.removed_scenes,
            availability.total_scenes,
            len(availability.removed_regular_tokens),
            len(availability.removed_synthetic_tokens),
            availability.retained_scenes,
        )
        if availability.removed_scenes:
            removed_tokens = (
                availability.removed_regular_tokens + availability.removed_synthetic_tokens
            )
            logger.warning(
                "Sensor availability preflight (%s) removed token examples: %s",
                split_name,
                removed_tokens[:10],
            )
    train_data = dataset_factory(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = dataset_factory(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


def _move_smoke_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        return {
            key: _move_smoke_value_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_smoke_value_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_smoke_value_to_device(item, device) for item in value]
    return value


def run_f2_train_row_finite_smoke(
    agent: AbstractAgent,
    train_dataloader: DataLoader,
    *,
    device: torch.device | None = None,
) -> None:
    """Run one disposable, single-device F2 optimizer step on a real batch.

    The dedicated launcher invokes this before consuming the signed one-shot
    marker in a separate process and temporary Hydra directory.  It catches
    model/Hydra/critic/backward/optimizer incompatibilities that static input
    and feasibility preflights cannot expose.  This is intentionally not a DDP
    collective smoke: the signed eight-rank launch remains the authority for
    process-group and cross-rank behavior.
    """

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("F2 pre-claim train-row smoke requires CUDA")
        device = torch.device("cuda", 0)
    try:
        batch = next(iter(train_dataloader))
    except StopIteration as error:
        raise RuntimeError("F2 pre-claim smoke found no authenticated train row") from error
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError("F2 train-row smoke batch must be (features, targets)")
    features = _move_smoke_value_to_device(batch[0], device)
    targets = _move_smoke_value_to_device(batch[1], device)
    agent.to(device)
    agent.train()

    trainable = tuple(
        (name, parameter)
        for name, parameter in agent.named_parameters()
        if parameter.requires_grad
    )
    if not trainable:
        raise RuntimeError("F2 pre-claim smoke found no trainable parameters")
    optimizer_config = agent.get_optimizers()
    if isinstance(optimizer_config, torch.optim.Optimizer):
        optimizer = optimizer_config
    elif (
        isinstance(optimizer_config, Mapping)
        and isinstance(optimizer_config.get("optimizer"), torch.optim.Optimizer)
    ):
        optimizer = optimizer_config["optimizer"]
    else:
        raise TypeError(
            "F2 pre-claim smoke requires one optimizer or an optimizer mapping"
        )

    expected_parameter_ids = {id(parameter) for _, parameter in trainable}
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    optimizer_parameter_ids = {id(parameter) for parameter in optimizer_parameters}
    if len(optimizer_parameter_ids) != len(optimizer_parameters):
        raise RuntimeError("F2 pre-claim optimizer contains duplicate parameters")
    if optimizer_parameter_ids != expected_parameter_ids:
        missing = sorted(
            name for name, parameter in trainable
            if id(parameter) not in optimizer_parameter_ids
        )
        unexpected = len(optimizer_parameter_ids - expected_parameter_ids)
        raise RuntimeError(
            "F2 pre-claim optimizer/trainable-parameter mismatch: "
            f"missing={missing}, unexpected_count={unexpected}"
        )

    optimizer.zero_grad(set_to_none=True)
    use_cuda_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=use_cuda_amp,
    ):
        prediction = agent.forward(features)
        loss_result = agent.compute_loss(features, targets, prediction)
    if isinstance(loss_result, dict):
        if "loss" not in loss_result:
            raise RuntimeError("F2 train-row smoke loss dictionary lacks loss")
        loss = loss_result["loss"]
        losses = loss_result.values()
    else:
        loss = loss_result
        losses = (loss_result,)
    observed_tensor = False
    for value in losses:
        if not isinstance(value, torch.Tensor):
            continue
        observed_tensor = True
        if value.numel() == 0 or not torch.isfinite(value).all().item():
            raise FloatingPointError("F2 pre-claim train-row smoke produced non-finite loss")
    if not observed_tensor:
        raise TypeError("F2 pre-claim train-row smoke produced no tensor loss")
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise TypeError("F2 pre-claim smoke primary loss must be one scalar tensor")
    if not loss.requires_grad:
        raise RuntimeError("F2 pre-claim smoke primary loss is not differentiable")

    scaler.scale(loss).backward()
    if use_cuda_amp:
        scaler.unscale_(optimizer)

    missing_gradients = []
    nonfinite_gradients = []
    any_nonzero_gradient = False
    step_probe: tuple[str, torch.nn.Parameter, int, torch.Tensor] | None = None
    for name, parameter in trainable:
        gradient = parameter.grad
        if gradient is None:
            missing_gradients.append(name)
            continue
        gradient_values = (
            gradient.coalesce().values() if gradient.is_sparse else gradient
        )
        if not bool(torch.isfinite(gradient_values).all().item()):
            nonfinite_gradients.append(name)
        if bool(torch.count_nonzero(gradient_values.detach()).item()):
            any_nonzero_gradient = True
            if step_probe is None:
                if gradient.is_sparse:
                    raise TypeError(
                        "F2 pre-claim smoke does not support sparse trainable gradients"
                    )
                probe_index = int(gradient.detach().abs().reshape(-1).argmax().item())
                probe_value = parameter.detach().reshape(-1)[probe_index].clone()
                step_probe = (name, parameter, probe_index, probe_value)
    if missing_gradients or nonfinite_gradients:
        raise FloatingPointError(
            "F2 pre-claim smoke gradient coverage/finite check failed: "
            f"missing={missing_gradients}, nonfinite={nonfinite_gradients}"
        )
    if not any_nonzero_gradient:
        raise RuntimeError("F2 pre-claim smoke produced only zero gradients")
    assert step_probe is not None

    total_norm = torch.nn.utils.clip_grad_norm_(
        tuple(parameter for _, parameter in trainable),
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    if not bool(torch.isfinite(total_norm).item()):
        raise FloatingPointError("F2 pre-claim smoke produced a non-finite grad norm")
    scaler.step(optimizer)
    scaler.update()

    nonfinite_parameters = []
    for name, parameter in trainable:
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            nonfinite_parameters.append(name)
    if nonfinite_parameters:
        raise FloatingPointError(
            "F2 pre-claim optimizer step produced non-finite parameters: "
            f"{nonfinite_parameters}"
        )
    probe_name, probe_parameter, probe_index, probe_before = step_probe
    probe_after = probe_parameter.detach().reshape(-1)[probe_index]
    if torch.equal(probe_before, probe_after):
        raise RuntimeError(
            "F2 pre-claim optimizer step did not change its nonzero-gradient "
            f"probe parameter {probe_name!r}"
        )

    optimizer.zero_grad(set_to_none=True)
    del features, targets, batch, prediction, loss_result, step_probe
    torch.cuda.empty_cache()


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    validate_training_lineage(cfg)
    validate_f2_run_protocol(cfg)

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = build_training_agent(cfg)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    print("Building Datasets")
    in_process_prefetch_batches = resolve_in_process_prefetch_batches()
    train_dataloader_class = (
        InProcessBatchPrefetchDataLoader
        if in_process_prefetch_batches == 1
        else DataLoader
    )
    train_dataloader_kwargs = dict(
        **cfg.dataloader.params, shuffle=True, drop_last=True
    )
    if in_process_prefetch_batches == 1:
        train_dataloader_kwargs["prefetch_batches"] = 1
        logger.info(
            "Enabled rank-local one-batch producer/consumer overlap with "
            "unchanged sampler order"
        )
    train_dataloader = train_dataloader_class(
        train_data, **train_dataloader_kwargs
    )
    print(f"Num training samples: {len(train_data)}")
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False, drop_last=True)
    print(f"Num val samples: {len(val_data)}")

    if (
        OmegaConf.select(
            cfg, "agent.config.actworld_critic_variant", default="legacy"
        )
        == F2_CRITIC_VARIANT
        and os.environ.get(F2_PREFLIGHT_SMOKE_ENV) == "1"
    ):
        logger.info("Running the pre-claim authenticated F2 train-row smoke")
        run_f2_train_row_finite_smoke(agent, train_dataloader)
        logger.info("Pre-claim authenticated F2 train-row smoke passed")
        return

    print('Building wandb logger')
    log_config = {
        'agent': OmegaConf.to_container(cfg.agent, resolve=True),
        'trainer': OmegaConf.to_container(cfg.trainer, resolve=True)
    }
    wandb_dir = Path(cfg.output_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(
        name=cfg.experiment_name,
        save_dir=str(wandb_dir),
        project='navsim',
    )
    wandb_logger.log_hyperparams(log_config)

    training_stage = getattr(agent, "_training_stage", None)
    critic_variant = getattr(agent, "_critic_variant", "legacy")
    checkpoint_cb = build_checkpoint_callback(
        cfg.output_dir, training_stage, critic_variant
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = agent.get_training_callbacks() + [checkpoint_cb, lr_monitor]

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, logger=wandb_logger, callbacks=callbacks)

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.resume_checkpoint_path,
    )


def resolve_multiprocessing_start_method() -> str:
    """Resolve the local child context used by DataLoader workers.

    External ``torchrun`` owns DDP rank creation, while proposal scorers
    explicitly request a spawn context.  This setting therefore controls the
    DataLoader path only.  ``fork`` avoids serializing the multi-gigabyte
    NAVSIM SceneLoader into every worker on Linux.
    """

    method = os.environ.get("ACTWORLD_MP_START_METHOD", "spawn")
    if method not in {"spawn", "fork", "forkserver"}:
        raise ValueError(
            "ACTWORLD_MP_START_METHOD must be spawn, fork, or forkserver"
        )
    return method


if __name__ == "__main__":
    # External torchrun already owns rank creation.  Set the local child
    # context before Hydra, CUDA, scorer pools, or DataLoaders are initialized.
    multiprocessing.set_start_method(
        resolve_multiprocessing_start_method(), force=True
    )
    main()
