# nuScenes six-second main experiment

Evaluate ActWorld-JEPA on the nuScenes validation split with six-second planning.
The reported 6 s L2 is **2.3945 m**, mean L2 is **1.2045 m**, and mean box
collision rate is **0.7638%**. See the [complete results](../README.md#nuscenes-planning).

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

The launcher uses one GPU, seed 41, deterministic mode and four data workers.
It saves predictions and reports L2 (meters) and box collision rate (percent)
at 1–6 seconds. The reported mean averages all six horizons.

Use the released configuration and execution settings for evaluation.
Checkpoint checks and additional evaluation details are documented in
[release verification](../docs/RELEASE_VERIFICATION.md) and
[evaluation protocols](../docs/EVALUATION_PROTOCOLS.md).
