# ActWorld-JEPA

Official NAVSIM v1 code release for **ActWorld-JEPA: Action-Conditioned
Joint-Embedding World Modeling for Autonomous End-to-End Driving**.

ActWorld-JEPA uses iPad as its base planner on NAVSIM and extends
JEPA from visual representation learning into planning. Each candidate
trajectory conditions a latent future rollout, current and predicted-future
features interact at candidate level, and a future-consistent selector ranks
the candidate proposal set. The simulator is not called in the
online planning loop.

This guide is executed from the repository root unless a `cd` is shown.
## Model and checkpoint

The NAVSIM v1 model achieves **90.07 PDMS** on the NAVTEST leaderboard.
Download the model and its four required selectors from
[Hugging Face](https://huggingface.co/shengkaizhang/ActWorld-JEPA).
Follow the [download instructions](GETTING_STARTED.md#download-the-weights)
and verify the files using the published [artifact manifest](../ARTIFACTS.md).

## Installation

Create the NAVSIM v1 evaluation environment:

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

Download the [NAVSIM v1 bundle](https://huggingface.co/shengkaizhang/ActWorld-JEPA/resolve/main/actworld_jepa_navsim_v1_r383_bundle.tar.gz)
and extract its five files into a local artifact directory. Follow the
[download and verification commands](GETTING_STARTED.md#download-the-weights).
Use the extracted model and selectors together.

The installation commands above leave you in `ActWorld-JEPA/navsim`.
Run the commands below from that directory. The launcher **generates predictions**;
it does not submit them to a leaderboard or compute a local score.
Fill in your own submission metadata (do not copy another author's identity).

Then export the paths consumed by the final launcher and run it:

```bash
export ACTWORLD_CHECKPOINT=/path/to/actworld_jepa_navsim_v1_r383.ckpt
export ACTWORLD_LATENT_VERIFIER=/path/to/ridge_strict_verifier.pkl
export ACTWORLD_CANDIDATE_SCORER=/path/to/multidomain_safe_oracle_set_head.pt
export ACTWORLD_TREE_META_SCORER=/path/to/aggressive_setwise_cascade_scorer.pkl
export ACTWORLD_R383_SELECTOR=/path/to/actworld_jepa_navsim_v1_r383_selector.pkl
export ACTWORLD_OUTPUT_DIR=/absolute/path/to/new-output
export CUDA_VISIBLE_DEVICES=0
export ACTWORLD_WORKERS=1
export TEAM_NAME="your-team"
export AUTHORS="Your Name"
export EMAIL="your-email"
export INSTITUTION="your-institution"
export COUNTRY="your-country"

bash scripts/evaluation/eval_actworld_jepa_r383.sh
```

For local scoring, with the matching v1 metric cache (still in `navsim/`):

```bash
python navsim/planning/script/run_pdm_score_from_submission.py \
  train_test_split=navtest \
  submission_file_path="$ACTWORLD_OUTPUT_DIR/submission.pkl" \
  metric_cache_path=/absolute/path/to/v1-metric-cache \
  output_dir="$ACTWORLD_OUTPUT_DIR/local-score"
```

Local scoring is not a leaderboard result. The launcher fails before evaluation when a required file or environment
variable is missing. It uses the official NAVSIM v1 submission schema and
also fails closed on missing, duplicate, or unexpected scene tokens.

The `.ckpt`, `.pt`, and `.pkl` files use Python/PyTorch serialization. Verify
their SHA256 values before use and load them only from this official release.

## Optional inference acceleration

For opt-in acceleration without changing the frozen artifacts, see
[FP32 inference acceleration](NAVSIM_V1_INFERENCE_ACCELERATION.md).
Existing prediction and scoring launchers remain unchanged. Install the runtime
only inside a dedicated worker after its real-input equivalence gate.

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

Apache License 2.0. See [LICENSE](../LICENSE).
