"""Opt-in inference-only optimizations; never modify frozen checkpoints."""
import numpy as np
import torch


def metadata_only_bev(agent_input, training=False):
    if training:
        raise ValueError("Metadata-only feature construction is inference-only")
    matrices, shapes = [], []
    cameras = agent_input.cameras[-1]
    scale = np.eye(4)
    scale[0, 0] *= 0.4
    scale[1, 1] *= 0.4
    for cam in (cameras.cam_b0, cameras.cam_f0, cameras.cam_l0, cameras.cam_r0):
        if cam.image is None:
            continue
        rotation = np.linalg.inv(cam.sensor2lidar_rotation)
        translation = cam.sensor2lidar_translation @ rotation.T
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation.T
        extrinsic[3, :3] = -translation
        viewpad = np.eye(4)
        intrinsic = cam.intrinsics
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        matrices.append(scale @ (viewpad @ extrinsic.T))
        h, w, c = cam.image.shape
        shapes.append((((int(h * 0.4) + 31) // 32) * 32,
                       ((int(w * 0.4) + 31) // 32) * 32, c))
    return {"img_shape": torch.FloatTensor(np.stack(shapes)),
            "lidar2img": torch.FloatTensor(np.stack(matrices))}


def install_metadata_only():
    from navsim.agents.drive_jepa_perception_based import drive_jepa_features
    original = drive_jepa_features._get_bev_feature
    drive_jepa_features._get_bev_feature = metadata_only_bev
    return original
