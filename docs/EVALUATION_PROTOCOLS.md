# Evaluation protocols and score provenance

Scores below use percentages for NAVSIM and collision rate, and meters for L2.
They must not be mixed across datasets, splits, or evaluator revisions.

## NAVSIM v1: R383

The authors identify R383 as the submitted model producing leaderboard PDMS
**90.07**, with a reported local result of **90.05**. The earlier public release
records a local score of **94.0721** and a 12,146-trajectory submission hash.
Those records are retained in the release history; they must not be silently
relabelled as 90.05 or treated as the same verified evaluation contract.
The 90.07 figure is author-reported pending attachment of the leaderboard record.

## NAVSIM v2 NAVTEST

The frozen model's saved predictions scored **88.1771** with the legacy
human-filter aggregation and **91.5900** after correcting that aggregation.
The corrected aggregate covers **11,989 valid scenes**. A separate 79-scene
check agreed with the corrected per-scene formula to numerical precision.
This is a reevaluation of fixed predictions, not a 3.4-point model improvement.

The fix addresses stale aggregate scores after human-trajectory exemptions
change component metrics. The multiplicative and weighted terms are recomputed
from the corrected components before producing the final score. See
[NAVSIM issue #151](https://github.com/autonomousvision/navsim/issues/151).
Only results evaluated with matching fixes and splits are directly comparable.

## NAVSIM v2 NAVHARD

The best legacy **39.2045** prediction set was rescored without retraining or
regenerating trajectories, using the same human-filter correction. All **5,912**
scenarios were valid: **450** original first-stage scenes and **5,462** second-stage
synthetic scenes.

| Stage | Corrected EPDMS |
|---|---:|
| Stage 1 | 79.9571 |
| Stage 2 | 53.5421 |
| Combined | 42.7351 |

The combined score follows the two-stage evaluator; it is not the arithmetic
mean of the two displayed stage scores. Its change from 39.2 to 42.7 is a metric
correction, not new training gain. Published legacy scores need the same
reevaluation before claiming a fair numerical ranking against them.

The complete inference pipeline includes learned candidate/temporal selectors
and frozen trajectory postprocessing. A checkpoint alone does not reproduce it.

## nuScenes: round 13, repeat 04

| Metric | 1 s | 2 s | 3 s | 4 s | 5 s | 6 s | Six-column mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| ActWorld L2 (m) | .2780 | .5380 | .8851 | 1.3135 | 1.8181 | 2.3945 | 1.204533 |
| Paired baseline L2 (m) | .2787 | .5398 | .8889 | 1.3202 | 1.8277 | 2.4071 | 1.210400 |
| ActWorld box collision (%) | .012 | .136 | .344 | .723 | 1.334 | 2.034 | .763833 |
| Paired baseline box collision (%) | .000 | .130 | .348 | .750 | 1.372 | 2.106 | .784333 |

Repeat 04 was selected from repeated inference runs of the same round-13
checkpoint. Numerical variability between runs is not an architectural gain.
The paired baseline shares perception outputs with the full model. These are
official-validation results, not the internal held-out training scenes.

Each prediction group must be cloned before CPU evaluation because the
original evaluator changes trajectory coordinates in place. Do not rescale
metrics or substitute a 1/2/3-second average for the six-column mean.
The Flow-matching branch is removed from the released nuScenes head, but it was
already disabled in the baseline configuration; removal alone is not evidence
of a performance gain. These experiments do not establish JEPA's isolated
causal contribution.
