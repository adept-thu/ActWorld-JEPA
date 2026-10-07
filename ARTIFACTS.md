# Artifact manifest

Large learned files are intentionally not committed to Git. A matching R383
run must use the following files. Verify them with
`python tools/verify_artifacts.py --root /path/to/artifacts`.

| Logical name | Expected file name | Bytes | SHA256 |
|---|---|---:|---|
| ActWorld-JEPA checkpoint | `last.ckpt` | 1,308,588,108 | `b56962ef371ba3e72eb56b13d773d95dac0a3e2b50845d137f376ec80c34fe0e` |
| Latent verifier | `ridge_strict_verifier.pkl` | 50,960 | `2e0974e6ccac28ce7df7d304b0831dc57b136d449502e3f7f91d68ea8aa81e70` |
| Candidate-set scorer | `multidomain_safe_oracle_set_head.pt` | 6,709,028 | `10ee5577c173116cf56b564464a08ac3c63b39e6277074c5d709f21dcea5a2ff` |
| Tree-meta scorer | `aggressive_setwise_cascade_scorer.pkl` | 26,055,736 | `84c78cb80a063fccad203f1743c5af89a5602a9946cd9648572a27271cfae20f` |
| R333 selector dependency | `r333_best_followup_artifact.pkl` | 704,222,961 | `1de2153ebc59d70dab239c45c99c2c657e35a2a8108986c96f3486af7c0cc71b` |
| R370 selector dependency | `r370_group_oof_mean_artifact.pkl` | 355,362,704 | `3f0a1e8dd40a67e2ba46f276092cff7a175394909bac6dcc1a1f1204bf9efff8` |
| R383 selector wrapper | `r383_r381_guard_artifact.pkl` | 199,758 | `97afc49b7cc069ff6372c3796c56fe822ad4a184865e3953b8711f2bede3874d` |

The R383 wrapper is lazy: it stores paths to R333 and R370. The public builder
rewrites those paths for the local machine, so a newly built wrapper will not
have the same SHA256 as the recorded original even though its model and policy
contract are identical. Verify R333 and R370 individually in that case.

The Drive-JEPA V-JEPA 2 initialization checkpoint
`vitl_merge_3dataset_e50.pt` comes from the upstream Drive-JEPA release and is
not an ActWorld-JEPA artifact.
