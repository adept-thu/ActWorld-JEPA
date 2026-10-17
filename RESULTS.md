# ActWorld-JEPA results

## NAVSIM

| Benchmark | Metric | Result |
|---|---|---:|
| NAVSIM v1 NAVTEST | Leaderboard PDMS ↑ | **90.07** |
| NAVSIM v2 NAVTEST | EPDMS ↑ | **91.6** |
| NAVSIM v2 NAVHARD | Combined EPDMS ↑ | **42.7** |

NAVSIM v2 uses human-filter-aware aggregation. NAVHARD combines two stages;
its stage scores are 79.9571 and 53.5421. See
[evaluation protocols](docs/EVALUATION_PROTOCOLS.md) for scoring details.

## nuScenes validation

| Metric ↓ | 1 s | 2 s | 3 s | 4 s | 5 s | 6 s | Mean (1–6 s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| L2 (m) | 0.2780 | 0.5380 | 0.8851 | 1.3135 | 1.8181 | **2.3945** | **1.2045** |
| Box collision (%) | 0.012 | 0.136 | 0.344 | 0.723 | 1.334 | 2.034 | **0.7638** |

## Reproduce

The [results and checkpoints table](README.md#results-and-checkpoints) links
each benchmark to its weights, configuration and evaluation guide.
[Evaluation protocols](docs/EVALUATION_PROTOCOLS.md) record metric definitions
and result provenance; [release verification](docs/RELEASE_VERIFICATION.md)
records the checks performed on the released artifacts.
