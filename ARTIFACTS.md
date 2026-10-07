# Artifact manifest

Large learned files are distributed through the
[NAVSIM v1 R383 release](https://github.com/adept-thu/ActWorld-JEPA/releases/tag/navsim-v1-r383)
rather than committed to Git. Download and verify the release archive:

| Release asset | Bytes | SHA256 |
|---|---:|---|
| `actworld_jepa_navsim_v1_r383_bundle.tar.gz` | 1,767,893,625 | `996774c139f550cac4d4207b6f4b6c56ad29e31eeab5357595459b460bf4a581` |

After extraction, a matching R383 run must use the following files. Verify them with
`python tools/verify_artifacts.py --root /path/to/artifacts`.

| Logical name | Expected file name | Bytes | SHA256 |
|---|---|---:|---|
| ActWorld-JEPA inference checkpoint | `actworld_jepa_navsim_v1_r383.ckpt` | 1,280,935,580 | `203807ec04b0e1966c40a0fa2258a1a1ad11f12b92266b4564f4beb7550ffcaa4` |
| Portable R383 selector | `actworld_jepa_navsim_v1_r383_selector.pkl` | 1,059,413,440 | `e8b2cdf2b42ae812fa3a792464f8c2c1777be4929d941e2ce1424defa8a157220` |
| Latent verifier | `ridge_strict_verifier.pkl` | 50,960 | `2e0974e6ccac28ce7df7d304b0831dc57b136d449502e3f7f91d68ea8aa81e70` |
| Candidate-set scorer | `multidomain_safe_oracle_set_head.pt` | 6,709,028 | `10ee5577c173116cf56b564464a08ac3c63b39e6277074c5d709f21dcea5a2ff` |
| Tree-meta scorer | `aggressive_setwise_cascade_scorer.pkl` | 26,055,736 | `84c78cb80a063fccad203f1743c5af89a5602a9946cd9648572a27271cfae20f` |

The inference checkpoint contains the same 656 model tensors as the evaluated
training checkpoint. Training-only optimizer/callback state and three absolute
server-path strings were removed. The portable selector embeds both frozen
NAVTRAIN estimators; its predictions were checked exactly against the evaluated
selector after replacing path-only metadata with public logical names.

These files use Python/PyTorch serialization and may execute code while being
loaded. Verify the hashes above and load files only from the official release.

The Drive-JEPA V-JEPA 2 initialization checkpoint
`vitl_merge_3dataset_e50.pt` comes from the upstream Drive-JEPA release and is
not an ActWorld-JEPA artifact.
