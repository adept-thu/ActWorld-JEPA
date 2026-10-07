# NAVSIM v1 results

## Frozen release

- Release identifier: `R383`
- Protocol: official NAVSIM v1 submission/evaluation schema
- Number of predicted trajectories: `12,146`
- Submission SHA256:
  `ff482967953a5de886b03638c953c24fb7740d3f62ebc15867b515f6fd69998d`

| Metric | Score |
|---|---:|
| No at-fault collisions (NC) | 0.990120 |
| Drivable area compliance (DAC) | 0.984192 |
| Driving direction compliance (DDC) | 0.944014 |
| Ego progress (EP) | 0.922415 |
| Time to collision (TTC) | 0.960563 |
| Comfort | 0.999177 |
| **PDMS** | **0.940721** |

## Model contract

R383 uses the released ActWorld-JEPA checkpoint and a frozen inference stack:

1. Drive-JEPA produces the candidate set and base proposal scores.
2. Planning-JEPA performs candidate-conditioned future rollout and
   current/future latent interaction.
3. Frozen NAVTRAIN readers estimate future-consistent utility and safety.
4. R383 applies a fixed affine selector contrast `(1.20, -0.20)` and requires
   a minimum predicted gain of `0.0075` before leaving the parent decision.

All learned parameters were fitted on NAVTRAIN.
