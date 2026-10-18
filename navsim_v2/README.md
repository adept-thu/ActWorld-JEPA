# NAVSIM v2 main deployments

ActWorld-JEPA uses iPad as its base planner on NAVSIM v2.

The shared aligned-actor checkpoint and learned selectors implement the released
NAVTEST model. NAVHARD adds the frozen temporal pair head, temporal head,
channel-wise projection, dense rollout and first-stage arc policy. All these
steps are required to reproduce the main deployment.

| Split | EPDMS ↑ |
|---|---:|
| NAVTEST | **91.6** |
| NAVHARD (two-stage combined) | **42.7** |

Evaluation uses human-filter-aware score aggregation as described in
[evaluation protocols](../docs/EVALUATION_PROTOCOLS.md). NAVHARD reports a
combined score across its two stages.

## Setup

Create the NAVSIM v2 evaluation environment. The evaluated installation used Python
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
use the matching NAVHARD cache. The required evaluator is included in
`navhard_devkit/`.

Download the shared pretrained V-JEPA asset `vitl_merge_3dataset_e50.pt` from
[LinhanWang/Drive-JEPA](https://huggingface.co/datasets/LinhanWang/Drive-JEPA)
into `$NAVSIM_EXP_ROOT/Drive-JEPA-cache/`. The ActWorld model weights and all
five selector artifacts are at
[shengkaizhang/ActWorld-JEPA](https://huggingface.co/shengkaizhang/ActWorld-JEPA).
Use only trusted pickle/checkpoint files and check their published hashes.

## NAVTEST

Run from `ActWorld-JEPA/navsim_v2/` with the v2 environment active, after
[downloading and verifying the required weights](../docs/GETTING_STARTED.md#download-the-weights).

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

The launcher generates candidate trajectories, applies the released selectors,
and evaluates the final plans. Choose a new `OUTPUT_DIR` for each run.

Final trajectories are saved to `$OUTPUT_DIR/trajectories.pkl`; evaluation
outputs are written to `$OUTPUT_DIR/scored/`. Keep the checkpoint, selectors,
and supplied evaluation configuration together when reproducing the results.
