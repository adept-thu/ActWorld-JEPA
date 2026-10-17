# Reproduce a released main model

The commands below assume Linux/Bash and start in the cloned repository root.
Replace `/absolute/path/to/...` with real directories on your computer. The
examples do not download benchmark data or submit anything to a leaderboard.

## Before you start

- Use an NVIDIA GPU with a CUDA-compatible driver. CPU is sufficient for hash
  verification, but full neural inference is intended for GPU execution.
- Install Conda and Git. Keep **three separate environments** for NAVSIM v1,
  NAVSIM v2, and nuScenes. Both NAVSIM trees install a package called `navsim`.
- Allow space for about 3.53 GB of released downloads, plus the extracted R383
  bundle, upstream initialization weights, datasets, caches and outputs.
- We publish frozen inference deployments, not a one-command recipe for
  recreating every checkpoint from training. See the verification record for
  the checks actually performed and the remaining fresh-install limitations.

## Download the weights

No Hugging Face token is required to download this public repository.

```bash
python -m pip install huggingface_hub
export ACTWORLD_WEIGHTS="$(pwd)/weights"
mkdir -p "$ACTWORLD_WEIGHTS"
```

Choose **one** download block:

**NAVSIM v1:**

```bash
hf download shengkaizhang/ActWorld-JEPA \
  actworld_jepa_navsim_v1_r383_bundle.tar.gz --local-dir "$ACTWORLD_WEIGHTS"
python tools/verify_main_artifacts.py "$ACTWORLD_WEIGHTS" --benchmark navsim-v1
mkdir -p "$ACTWORLD_WEIGHTS/r383"
tar -xzf "$ACTWORLD_WEIGHTS/actworld_jepa_navsim_v1_r383_bundle.tar.gz" \
  -C "$ACTWORLD_WEIGHTS/r383"
python tools/verify_artifacts.py --root "$ACTWORLD_WEIGHTS/r383"
```

**NAVSIM v2 NAVTEST:**

```bash
hf download shengkaizhang/ActWorld-JEPA actworld_navsim_v2.ckpt \
  factor005.pkl strict_variant_selector.pkl multicomponent_gate_selector.pkl \
  --local-dir "$ACTWORLD_WEIGHTS"
python tools/verify_main_artifacts.py "$ACTWORLD_WEIGHTS" --benchmark navtest-v2
```

**NAVSIM v2 NAVHARD:**

```bash
hf download shengkaizhang/ActWorld-JEPA actworld_navsim_v2.ckpt \
  factor005.pkl strict_variant_selector.pkl multicomponent_gate_selector.pkl \
  temporal_fcts_pair_head.pkl temporal_fcts_head.pt --local-dir "$ACTWORLD_WEIGHTS"
python tools/verify_main_artifacts.py "$ACTWORLD_WEIGHTS" --benchmark navhard-v2
```

**nuScenes:**

```bash
hf download shengkaizhang/ActWorld-JEPA actworld_nuscenes_r13.pth \
  --local-dir "$ACTWORLD_WEIGHTS"
python tools/verify_main_artifacts.py "$ACTWORLD_WEIGHTS" --benchmark nuscenes
```

The verifier reads bytes only; it does not deserialize model/pickle files.
All eight remote file sizes and LFS SHA256 values were checked against the
release manifest on 2026-10-17. Do not ignore a checksum failure.

## Prepare data and run

| Benchmark | Needed inputs | Next step |
|---|---|---|
| NAVSIM v1 | v1 logs/sensor blobs, maps, upstream V-JEPA initialization; v1 cache for local scoring | [Install and generate the R383 submission](NAVSIM_V1_R383.md#installation) |
| NAVSIM v2 NAVTEST | v2 NAVTEST logs/blobs, maps, matching cache, upstream V-JEPA initialization | [Install and evaluate](../navsim_v2/README.md#setup) |
| NAVSIM v2 NAVHARD | Reactive two-stage logs/blobs including synthetic scenes, matching two-stage cache, maps and V-JEPA initialization | [Install](../navsim_v2/README.md#setup), then [run NAVHARD](../navsim_v2/README.md#navhard) |
| nuScenes | trainval data, six-second annotation infos, released anchors and compiled CUDA extension | [nuScenes setup and evaluation](../nuscenes/README.md) |

Original-provider instructions: [NAVSIM](https://github.com/autonomousvision/navsim),
[Drive-JEPA](https://github.com/linhanwang/Drive-JEPA),
[nuScenes](https://www.nuscenes.org/nuscenes), and
[MomAD](https://github.com/adept-thu/MomAD). Dataset access and licenses remain
with those providers. A checkpoint download does not include the dataset.

The benchmark guide defines the required path variables. Set them in the same
terminal where you run evaluation. Do not point NAVHARD to a NAVTEST cache.

## Read the result correctly

- **v1:** `submission.pkl` is a prediction artifact, not a score. Run the local
  scorer separately; leaderboard submission is a separate user action.
- **v2:** inspect the CSV's aggregate rows and valid-scene counts. Do not take a
  new average over a CSV containing both scene rows and summary rows. NAVHARD's
  combined score is not the average of its stage scores.
- **nuScenes:** report all six 1–6 s columns; collision numbers are percentages.
  The mean here is the arithmetic mean of those six columns, not the 1/2/3 s mean.
- Small inference differences are possible across numerical libraries, devices
  and worker settings. Use the documented evaluation configuration.

## Troubleshooting

| Symptom | Check |
|---|---|
| `No module named navsim` or unexpected agent/config | Activate the benchmark-specific environment and run `python -c "import navsim; print(navsim.__file__)"`; it must point to that benchmark's tree. |
| Missing checkpoint/selector | Download every artifact listed for that row, verify hashes, and use an absolute `ACTWORLD_WEIGHTS` path. |
| Missing `vitl_merge_3dataset_e50.pt` | Follow the upstream initialization download in the NAVSIM guide; this is separate from the ActWorld checkpoint. |
| Pickle/scikit-learn incompatibility | Use the documented scikit-learn 1.2.2 NAVSIM environment and trusted published files. |
| Missing maps, sensors or metric cache | Check the dataset version, split and environment variables. Caches are not interchangeable across benchmarks. |
| CUDA-extension import error in nuScenes | Build the local extension in the nuScenes environment with a matching compiler/CUDA/PyTorch stack. |
| GPU out of memory | Start with one worker; avoid concurrent inference jobs. Report any topology changes when comparing results. |
| Score differs from the table | Check evaluator correction, full scene count, deployment selectors and aggregation before attributing it to model quality. |

## Verification limits

See [release verification](RELEASE_VERIFICATION.md) and
[protocol/provenance notes](EVALUATION_PROTOCOLS.md). Source compilation and
artifact integrity checks are not a substitute for a clean GPU evaluation.
