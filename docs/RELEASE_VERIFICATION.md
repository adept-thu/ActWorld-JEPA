# Release verification

## 2026-10-17 documentation and distribution audit

- All eight Hugging Face artifacts are publicly listed. Their LFS SHA256 and
  byte sizes match `artifact_manifest.json`, including both large NAVSIM files.
- Recomputed all five extracted R383 file hashes. Corrected two transcription
  errors in `ARTIFACTS.md` and `artifacts/manifest.json` (checkpoint and selector).
  Artifact bytes were not changed; the earlier verifier would incorrectly fail
  for these two valid files.
- Added benchmark-specific verification, explicit working directories, separate
  NAVSIM environments, v1 submission metadata and beginner download instructions.
- Aligned the v1 package's license classifier with its existing Apache-2.0
  license. This does not change dataset or pretrained-weight license terms.
- No model parameters, prediction selection or evaluation formulas were changed
  by this documentation update.

## Initial main-model publication

Checks performed for the 2026-10-16 main-model publication:

- **nuScenes model/package compatibility:** the packaged `ActWorldMomADv2SixSecond`
  model loaded all 1,414 tensors with `strict=True` on CPU, using the existing
  compiled CUDA-extension binary for import. No missing/unexpected tensor was
  ignored. A fresh CUDA-extension build is required on another installation.
- **Checkpoint exports:** all 587 NAVSIM v2 tensors and 1,414 nuScenes tensors
  were checked equal to their evaluated training checkpoints. Only state dicts
  are distributed for these exports; optimizer and private training metadata
  are omitted.
- **NAVHARD deployment replay:** the published five-step postprocessing chain
  was run on the saved 5,912 candidate records. All 5,912 output trajectories
  matched the evaluated best deployment exactly; maximum coordinate difference
  was **0**. The final pickle SHA256 was
  `c31f2bf7cab0c215058ad7378239d6af42e058fbd950f4a47505974d00abdc45`,
  identical to the original evaluated prediction file.
- **Actual launch parameters:** the temporal head's runtime overrides were
  recovered from the original launch command, not inferred from a report that
  stored artifact defaults. The released report writer now records both the
  effective policy and the artifact defaults. This changes reporting only.
- **Evaluator correction:** fixed NAVTEST/NAVHARD predictions were rescored;
  corrected and legacy results remain separately labelled.
- **Source audit:** Python compilation, private-path/credential-pattern checks,
  third-party notices and artifact-size/hash checks were performed. Benchmark
  data, raw diagnostic records, metric caches and per-scene evaluation CSVs are
  excluded from the publication.

These checks are not a claim that inference is bitwise deterministic across
all hardware/software versions. The replay check starts from saved candidate
records; it is not a new full GPU inference run of the packaged NAVHARD source.
The nuScenes result is a selected repeat, and R383 leaderboard provenance has
the qualification described in [evaluation protocols](EVALUATION_PROTOCOLS.md).
