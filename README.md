<div align="center">

# ActWorld-JEPA
### Action-Conditioned JEPA World Models Meet Latent Planning<br>for End-to-End Autonomous Driving

**Ziying Song · Shengkai Zhang · Caiyan Jia · Lei Yang · Lin Liu · Li Wang · Yadan Luo · Linhan Wang · Chen Lv**

[![Project page](https://img.shields.io/badge/Project-Page-167D7F)](https://adept-thu.github.io/ActWorld-JEPA/)
[![Hugging Face](https://img.shields.io/badge/🤗-Checkpoints-FFD21E)](https://huggingface.co/shengkaizhang/ActWorld-JEPA)
[![License](https://img.shields.io/badge/Code-Apache--2.0-blue)](LICENSE)

</div>

ActWorld-JEPA predicts action-conditioned future representations and uses them
to improve trajectory planning. **PC-JEPA** connects candidate plans to latent
future outcomes; **R³** performs relational residual reranking. The release
contains the frozen main deployments for NAVSIM v1, NAVSIM v2, and nuScenes.

![ActWorld-JEPA framework](docs/assets/framework.png)

## Main results

| Benchmark | Released deployment | Metric | Result |
|:--|:--|:--|--:|
| NAVSIM v1 NAVTEST | R383 | Leaderboard PDMS | **90.07**† |
| NAVSIM v2 NAVTEST | Aligned actor + multicomponent selector | Corrected EPDMS | **91.6** |
| NAVSIM v2 NAVHARD | Frozen two-stage temporal deployment | Corrected combined EPDMS | **42.7** |
| nuScenes validation | Round 13, inference repeat 04 | 6 s L2 ↓ | **2.3945 m** |
| nuScenes validation | Same deployment | Mean 1–6 s L2 / box collision ↓ | **1.2045 m / 0.7638%** |

† The authors identify R383 as the 90.07 leaderboard submission (local 90.05).
The earlier public R383 release records a different local evaluation, 94.0721;
that record is preserved rather than silently relabelled. See
[evaluation protocols](docs/EVALUATION_PROTOCOLS.md) for the distinction.

NAVSIM v2 uses the human-filter aggregation correction discussed in
[official issue #151](https://github.com/autonomousvision/navsim/issues/151).
NAVTEST 91.6 and NAVHARD 42.7 are rescored fixed predictions; legacy results
88.2 and 39.2 are not directly comparable without matching evaluator revisions.
nuScenes repeat 04 is a selected repeated-inference result, not a separate
checkpoint. Exact six-horizon values and paired baselines are documented in
[evaluation protocols](docs/EVALUATION_PROTOCOLS.md).

## Code and checkpoints

| Benchmark | Implementation and instructions | Weights |
|:--|:--|:--|
| NAVSIM v1 | [R383 guide](docs/NAVSIM_V1_R383.md) · root `navsim/` | [Hugging Face](https://huggingface.co/shengkaizhang/ActWorld-JEPA) |
| NAVSIM v2 NAVTEST / NAVHARD | [NAVSIM v2 guide](navsim_v2/README.md) | [Hugging Face](https://huggingface.co/shengkaizhang/ActWorld-JEPA) |
| nuScenes | [nuScenes guide](nuscenes/README.md) | [Hugging Face](https://huggingface.co/shengkaizhang/ActWorld-JEPA) |

The learned selectors and frozen deployment policies are part of the model:
the neural checkpoint alone does not reproduce every reported result.
Datasets, metric caches, raw per-scene benchmark records, optimizer states,
credentials, and server-specific paths are not distributed.

Use separate environments for NAVSIM and the OpenMMLab-based nuScenes code.
Download benchmark data from its original providers and follow their terms.

## Qualitative results

### NAVSIM: scene understanding and planning
![NAVSIM front-view and BEV planning](docs/assets/navsim-main.png)

<details><summary>More NAVSIM examples</summary>

![Additional NAVSIM examples](docs/assets/navsim-more.png)
</details>

### nuScenes: six-second trajectories
![nuScenes planning examples](docs/assets/nuscenes.png)

## Acknowledgements and licensing

We build on [Drive-JEPA](https://github.com/linhanwang/Drive-JEPA),
[NAVSIM](https://github.com/autonomousvision/navsim),
[MomAD](https://github.com/adept-thu/MomAD),
[SparseDrive](https://github.com/swc-17/SparseDrive), and V-JEPA 2.
Project presentation is inspired by MomAD and KineWorld.
ActWorld additions use Apache-2.0; retained third-party components keep their
original licenses. See [third-party notices](THIRD_PARTY_NOTICES.md).

The paper is a manuscript; this page does not claim conference acceptance.
See [CITATION.cff](CITATION.cff) for citation metadata.
