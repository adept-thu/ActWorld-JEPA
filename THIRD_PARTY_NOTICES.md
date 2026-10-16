# Third-party notices

ActWorld-JEPA additions use the root Apache-2.0 license. Vendored code retains
its original notices and license; the root license does not replace them.

| Component | Source | License |
|---|---|---|
| NAVSIM | https://github.com/autonomousvision/navsim | Apache-2.0 |
| Drive-JEPA | https://github.com/linhanwang/Drive-JEPA | Apache-2.0 |
| V-JEPA 2 source | https://github.com/facebookresearch/vjepa2 | MIT; see vendored LICENSE |
| SparseDrive | https://github.com/swc-17/SparseDrive | MIT |
| MomAD | https://github.com/adept-thu/MomAD | MIT |

Copies of upstream notices are in `third_party_licenses/`. Components adapted
for this release include the ActWorld agents, benchmark-specific configuration,
trajectory selection/refinement, and the documented NAVSIM v2 human-filter fix.

Dataset downloads, sensor logs, metric caches, and pretrained dependencies are
not relicensed by this repository. Obtain them from their original providers
under their applicable terms. This release does not distribute raw datasets.

Paper figures and qualitative visualizations are supplied by the authors.
Project-page presentation was inspired by MomAD and KineWorld; its HTML/CSS was
written for this project, rather than copying their site source or media.
