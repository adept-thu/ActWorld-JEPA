# nuScenes six-second main experiment

This directory contains the frozen round-13 architecture and inference
configuration. The reported result is inference repeat **04** of the same
checkpoint: 6 s L2 **2.3945 m**, six-column mean L2 **1.204533 m**, and mean
box collision **0.763833%**. See [all horizons and paired baseline](../docs/EVALUATION_PROTOCOLS.md).

## Environment and data

The evaluated environment used Python 3.8, PyTorch 1.11.0+cu113,
mmcv-full 1.7.2, mmdet 2.28.2, flash-attn 0.2.2 and fastdtw 0.3.4.
Use a separate environment from NAVSIM. Start at the cloned repository root.
The CUDA toolkit/compiler must match the PyTorch build when building extensions.

```bash
cd nuscenes
conda create -n actworld-nuscenes python=3.8 -y
conda activate actworld-nuscenes
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full==1.7.2 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install -r requirement.txt
pip install flash-attn==0.2.2 fastdtw==0.3.4
cd projects/mmdet3d_plugin/ops
python setup.py build_ext --inplace
cd ../../..
```

The commands leave you in `ActWorld-JEPA/nuscenes`. The dependency versions are
from the evaluated installation; a clean installation on a new CUDA/compiler
combination has not been validated by this release audit.

Prepare nuScenes trainval and six-second annotation info files following
[MomAD](https://github.com/adept-thu/MomAD) / [SparseDrive](https://github.com/swc-17/SparseDrive).
The effective paths are in `projects/configs/actworld_nuscenes.py`; adjust only
data locations for your installation. The validation info is
`data/infos/nuscenes_infos_val_6s.pkl`. Data and annotations are not redistributed.
The four released k-means anchors are already included in `data/kmeans/`.
The original ResNet initialization file belongs at
`ckpt/resnet50-19c8e357.pth` (the full inference checkpoint provides model weights).

## Evaluate

```bash
export ACTWORLD_WEIGHTS=/absolute/path/to/downloaded/weights
export CUDA_VISIBLE_DEVICES=0
bash run_evaluation.sh
```

The launcher preserves one-process distributed inference, seed 41, deterministic
flag and four data workers. Changing execution topology can introduce numerical
differences. No guarantee is made that a fresh run reproduces the selected
repeat's last decimal. Do not scale the metric or use the 1/2/3 s average in
place of the mean over all six horizons.

The inference export retains all 1,414 state tensors exactly; strict loading
against this packaged model was verified on CPU using the compiled extension.
Optimizer states and training metadata were removed. The original baseline
repository is unchanged. Flow matching was already disabled before its unused
branch was removed from this standalone release.
