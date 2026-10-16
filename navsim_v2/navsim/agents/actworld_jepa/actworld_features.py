"""Feature/target builders for causal ActWorld-JEPA training."""

from typing import Dict, Optional

import cv2
import torch
from torchvision import transforms

from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
    DriveJEPAFeatureBuilder,
    DriveJEPATargetBuilder,
)
from navsim.common.dataclasses import AgentInput

from .actworld_config import ActWorldJEPAConfig


class ActWorldJEPAFeatureBuilder(DriveJEPAFeatureBuilder):
    """Adds one older image so an observed transition can supervise dynamics.

    The online planner still rolls forward from the current two-frame V-JEPA
    representation.  The extra frame is historical (``t-2``), never future
    information, and is used only for the one-step latent-consistency loss.
    """

    def __init__(self, config: ActWorldJEPAConfig):
        super().__init__(config)

    def get_unique_name(self) -> str:
        return "actworld_jepa_feature_v1"

    def _get_camera_feature(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        if len(agent_input.cameras) < 3:
            raise ValueError("ActWorld-JEPA requires at least three history frames")

        result: Dict[str, torch.Tensor] = {}
        # The base model names the newest image ``camera_feature_1`` and the
        # previous image ``camera_feature_2``.  Preserve those checkpoint/cache
        # keys and append ``camera_feature_3`` for the older observed frame.
        for key, camera_index in (
            ("camera_feature_1", -1),
            ("camera_feature_2", -2),
            ("camera_feature_3", -3),
        ):
            image = agent_input.cameras[camera_index].cam_f0.image
            if image is None:
                raise ValueError(
                    f"Missing front-camera history frame at index {camera_index}"
                )
            cropped = image[28:-28]
            resized = cv2.resize(cropped, (512, 256))
            result[key] = transforms.ToTensor()(resized)
        return result


class ActWorldJEPATargetBuilder(DriveJEPATargetBuilder):
    """Adds an optional, training-only terminal camera target.

    The future clip is deliberately a target rather than an agent feature, so
    it cannot enter the online planner input.  With NAVSIM's four history
    frames, scene indices 10 and 11 are the h=7/h=8 observations and together
    form the two-frame V-JEPA clip at the 4 s planning horizon.
    """

    _FUTURE_CLIP_INDICES = (10, 11)
    _CAMERA_SHAPE = (3, 256, 512)

    def __init__(
        self,
        config: ActWorldJEPAConfig,
        enable_future: Optional[bool] = None,
    ):
        super().__init__(config)
        self._enable_future = (
            config.actworld_future_latent_grounding
            if enable_future is None
            else bool(enable_future)
        )

    def get_unique_name(self) -> str:
        # v2 is intentionally incompatible with the trajectory-only target
        # cache: future camera tensors are streamed during training for now.
        return "actworld_jepa_target_v2"

    @staticmethod
    def _front_camera_tensor(image) -> torch.Tensor:
        cropped = image[28:-28]
        resized = cv2.resize(cropped, (512, 256))
        return transforms.ToTensor()(resized)

    def compute_targets(self, scene):
        targets = super().compute_targets(scene)
        if not self._enable_future:
            return targets

        clip = torch.zeros((2, *self._CAMERA_SHAPE), dtype=torch.float32)
        valid = True
        for clip_index, scene_index in enumerate(self._FUTURE_CLIP_INDICES):
            if scene_index >= len(scene.frames):
                valid = False
                continue
            image = scene.frames[scene_index].cameras.cam_f0.image
            if image is None:
                valid = False
                continue
            clip[clip_index] = self._front_camera_tensor(image)

        targets.update(
            {
                "actworld_future_camera_clip": clip,
                "actworld_future_camera_valid": torch.tensor(valid, dtype=torch.bool),
            }
        )
        return targets
