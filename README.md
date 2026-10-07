# ActWorld-JEPA

Official NAVSIM v1 code release for **ActWorld-JEPA: Action-Conditioned
Joint-Embedding World Modeling for Autonomous End-to-End Driving**.

ActWorld-JEPA starts from the Drive-JEPA perception-based planner and extends
JEPA from visual representation learning into planning. Each candidate
trajectory conditions a latent future rollout, current and predicted-future
features interact at candidate level, and a future-consistent selector ranks
the original Drive-JEPA proposal set. The simulator is not called in the
online planning loop.

## NAVSIM v1 result

The frozen `R383` release produced the following NAVSIM v1 result:

| NC | DAC | DDC | EP | TTC | Comfort | PDMS |
|---:|---:|---:|---:|---:|---:|---:|
| 0.990120 | 0.984192 | 0.944014 | 0.922415 | 0.960563 | 0.999177 | **0.940721** |

The generated submission contains 12,146 trajectories. Its SHA256 is
`ff482967953a5de886b03638c953c24fb7740d3f62ebc15867b515f6fd69998d`.
See [RESULTS.md](RESULTS.md) for the evaluation contract.

## Release scope

This repository intentionally contains only the NAVSIM v1 implementation used
by the best frozen result:

- ActWorld-JEPA model, losses, planning-JEPA rollout, and selection code;
- the required Drive-JEPA perception-based base implementation;
- the NAVSIM v1.1 devkit code and the vendored V-JEPA 2 source dependency;
- the exact R383 submission launcher and artifact builder;
- artifact hashes and a release audit utility.

It does **not** contain datasets, metric caches, raw evaluation CSVs,
per-scene benchmark outputs, credentials, local machine paths, failed
experiments, old source backups, or large learned artifacts. Checkpoints and
frozen scorers are distributed separately; their exact names, sizes, and
hashes are recorded in [ARTIFACTS.md](ARTIFACTS.md).

## Installation

Create the same base environment as Drive-JEPA/NAVSIM v1:

```bash
conda create -n actworld-jepa python=3.9 -y
conda activate actworld-jepa
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cd navsim
pip install -e .
```

Download NAVSIM v1 following the
[official NAVSIM instructions](https://github.com/autonomousvision/navsim/blob/main/docs/install.md)
and the released Drive-JEPA cache/checkpoint bundle from
[LinhanWang/Drive-JEPA](https://huggingface.co/datasets/LinhanWang/Drive-JEPA).
Set:

```bash
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/path/to/maps
export OPENSCENE_DATA_ROOT=/path/to/navsim/data
export NAVSIM_EXP_ROOT=/path/to/navsim/experiments
export VJEPA2_CHECKPOINT="$NAVSIM_EXP_ROOT/Drive-JEPA-cache/vitl_merge_3dataset_e50.pt"
```

## Reproducing R383 inference

Place the seven ActWorld-JEPA artifacts listed in [ARTIFACTS.md](ARTIFACTS.md)
in a local artifact directory. If the R383 wrapper has not been supplied,
build it from the two frozen NAVTRAIN selector artifacts:

```bash
cd navsim
python scripts/artifacts/build_r383_selector.py \
  --r333 /path/to/r333_best_followup_artifact.pkl \
  --r370 /path/to/r370_group_oof_mean_artifact.pkl \
  --output /path/to/r383_r381_guard_artifact.pkl
```

Then export the paths consumed by the final launcher and run it:

```bash
export ACTWORLD_CHECKPOINT=/path/to/last.ckpt
export ACTWORLD_LATENT_VERIFIER=/path/to/ridge_strict_verifier.pkl
export ACTWORLD_CANDIDATE_SCORER=/path/to/multidomain_safe_oracle_set_head.pt
export ACTWORLD_TREE_META_SCORER=/path/to/aggressive_setwise_cascade_scorer.pkl
export ACTWORLD_R383_SELECTOR=/path/to/r383_r381_guard_artifact.pkl
export ACTWORLD_OUTPUT_DIR=/path/to/output

bash scripts/evaluation/eval_actworld_jepa_r383.sh
```

The launcher fails before evaluation when a required file or environment
variable is missing. It uses the official NAVSIM v1 submission schema and
also fails closed on missing, duplicate, or unexpected scene tokens.

## Training protocol

All trainable neural modules and frozen selector estimators in this release
were fitted using NAVTRAIN data. The final global selector weights and
threshold were selected after repeated evaluations using aggregate NAVTEST
metrics. The reported result is therefore test-feedback-tuned rather than an
untouched-test measurement.

## Acknowledgements

This release builds on
[Drive-JEPA](https://github.com/LinhanWang/Drive-JEPA),
[NAVSIM](https://github.com/autonomousvision/navsim), and
[V-JEPA 2](https://github.com/facebookresearch/vjepa2). Their original
licenses are retained. The repository root is Apache-2.0; vendored V-JEPA 2
code retains its MIT license under `navsim/vjepa2/LICENSE`.

## License

Apache License 2.0. See [LICENSE](LICENSE).
