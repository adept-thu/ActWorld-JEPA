"""Cache-separated feature and target builders for ActWorld-JEPA v1."""

from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
    DriveJEPAFeatureBuilder,
    DriveJEPATargetBuilder,
)


class ActWorldJEPAFeatureBuilder(DriveJEPAFeatureBuilder):
    """Use the same two-frame causal input under a distinct cache key."""

    def get_unique_name(self) -> str:
        return "actworld_jepa_v1_feature"


class ActWorldJEPATargetBuilder(DriveJEPATargetBuilder):
    """Use the released v1 trajectory target under a distinct cache key."""

    def get_unique_name(self) -> str:
        return "actworld_jepa_v1_target"


__all__ = ["ActWorldJEPAFeatureBuilder", "ActWorldJEPATargetBuilder"]
