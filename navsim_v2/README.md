# NAVSIM v2 main deployments

The shared aligned-actor checkpoint and learned selectors implement the released
NAVTEST model. NAVHARD adds the frozen temporal pair head, temporal head,
channel-wise projection, dense rollout and first-stage arc policy. All these
steps are required to reproduce the main deployment.

| Split | Legacy EPDMS | Corrected EPDMS |
|---|---:|---:|
| NAVTEST (11,989 valid scenes) | 88.1771 | **91.5900** |
| NAVHARD (5,912 valid scenes, two-stage combined) | 39.2045 | **42.7351** |

These score pairs use fixed predictions. The correction recomputes aggregate
terms after human-trajectory filtering; it does not change the model. Read
[evaluation protocols](../docs/EVALUATION_PROTOCOLS.md) before comparing methods.
The NAVHARD first-/second-stage scores are 79.9571 / 53.5421; the combined score
is not their arithmetic average.

## Setup

Use the Drive-JEPA / NAVSIM environment. The evaluated installation used Python
3.9, PyTorch 2.1.0+cu121, torchvision 0.16.0+cu121, pytorch-lightning 2.2.1,
hydra-core 1.2.0, scikit-learn 1.2.2 and timm 1.0.27. In particular, preserve
the scikit-learn version when loading learned selector artifacts.

```bash
# Start in the cloned ActWorld-JEPA repository root.
cd navsim_v2
conda create -n actworld-navsim-v2 python=3.9 -y
conda activate actworld-navsim-v2
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r ../requirements.txt
pip install -e .
pip install pytorch-lightning==2.2.1 hydra-core==1.2.0 scikit-learn==1.2.2 timm==1.0.27
```

Obtain data/maps and matching metric caches through the NAVSIM providers.
NAVHARD requires the reactive **two-stage** split, including synthetic scenes;
a NAVTEST cache is not interchangeable. The frozen reactive evaluator snapshot
is in `navhard_devkit/`, with its original Apache-2.0 notices and the documented
human-filter correction. It is a versioned evaluation snapshot, not a claim
that every file matches the latest upstream NAVSIM release.

Download the shared pretrained V-JEPA asset `vitl_merge_3dataset_e50.pt` from
[LinhanWang/Drive-JEPA](https://huggingface.co/datasets/LinhanWang/Drive-JEPA)
into `$NAVSIM_EXP_ROOT/Drive-JEPA-cache/`. The ActWorld model weights and all
five selector artifacts are at
[shengkaizhang/ActWorld-JEPA](https://huggingface.co/shengkaizhang/ActWorld-JEPA).
Use only trusted pickle/checkpoint files and check their published hashes.

## NAVTEST

```bash
export ACTWORLD_WEIGHTS=/absolute/path/to/weights
export OPENSCENE_DATA_ROOT=/absolute/path/to/navsim-data
export NAVSIM_EXP_ROOT=/absolute/path/to/experiments
export NUPLAN_MAPS_ROOT=/absolute/path/to/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export METRIC_CACHE_PATH=/absolute/path/to/navtest-metric-cache
export CUDA_VISIBLE_DEVICES=0,1
bash run_navtest.sh
```

## NAVHARD

Set the same environment variables, changing data/cache roots to the two-stage
NAVHARD versions, then:

```bash
export OUTPUT_DIR=/absolute/path/to/new-navhard-run
export WORKERS=2 SCORE_WORKERS=8
bash run_navhard.sh
```

The script first captures current-observation candidate representations, then
applies the frozen learned temporal policies, and finally scores the resulting
trajectories. It fails rather than overwrite an existing output directory.
No future labels are read by the prediction/postprocessing steps. Metric caches
are used by the evaluator, separately from prediction generation.

`ops/policies/` contains aggregate NAVTRAIN gate evidence for the fixed
postprocessing policies. Training reference features in the learned selector
are model artifacts, not raw benchmark images. The released best deployment
was chosen during a sequence of benchmark experiments; we do not describe this
release as an untouched-test model-selection protocol.

The `MOMWORLD_*` environment names in score-only execution are compatibility
names inherited from the frozen evaluator. They do not select another model.
`ops/score_only_compat` supplies a fail-closed import shim for an unused
diffusion training dependency; it must not be used to train a diffusion model.
