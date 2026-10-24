# Opt-in r383 FP32 inference acceleration

This runtime accelerates the frozen NAVSIM v1 r383 inference path without
retraining, changing precision, reducing proposals, or removing selectors.
It is **opt-in**: existing agents and evaluation launchers are unchanged.
Use the five matching artifacts and environment from [the v1 guide](NAVSIM_V1_R383.md).

## Measured deployment

| Method | Dim | Neural parameters (M) | Local PDMS | Plan FPS | Model FPS |
|:--|--:|--:|--:|--:|--:|
| ActWorld-JEPA r383, accelerated | 256 | 324.164 | 94.0518 | 30.9110 | 21.4804 |

Measured on one RTX 4090, FP32, batch size 1. Model and Plan FPS are the
median run throughput across five runs of 64 real NAVTRAIN inputs from 64
distinct logs, after warm-up, using CUDA-synchronized wall-clock timing.
Model FPS includes the perception encoder and deployed selectors, but excludes
CPU feature construction, host-to-device transfer, disk I/O, and simulation.
Plan FPS excludes the perception encoder: **only the timing harness** reuses its
exact outputs. Normal deployment recomputes from each new input and does not
cache predictions. Complete `compute_trajectory` API throughput is **19.2349 FPS**
(median of three runs of eight inputs; includes feature construction and transfer,
excludes disk I/O and simulation). Parameters count deduplicated neural weights,
including frozen/shared components and external neural heads, not tree nodes.

These numbers are not evaluation-pipeline scenarios/second. They are not a
same-hardware comparison with published H20 speeds; in particular, they do not
reach DiffusionDrive's reported 39 FPS / 75 Plan FPS. No fair cross-hardware
speed superiority is claimed.

## Correctness and evaluation scope

The frozen deployment passed bitwise equality of four model outputs
(`trajectory`, `selected_proposal`, `pdm_score`, `proposals`) and API trajectories
on all 64 NAVTRAIN validation inputs. The subsequent complete local NAVTEST
regression finished on 2026-10-24 with 12,146 unique scenarios, zero failures,
finite scores, an unchanged input-token set, and unchanged source/weight hashes.
All seven score components were exactly equal per scenario to the reference.
The measured accelerated PDMS is **94.05183397465883**; it is not substituted
from an earlier run. Machine-readable aggregate evidence and output hashes are
in [the regression record](../artifacts/r383_inference_acceleration.json).

The historical r383 selector weights/threshold used aggregate NAVTEST feedback
for calibration. This is **not an untouched-test result**. The local score does
not replace the **90.07 leaderboard** result reported elsewhere in this release.
No NAVTEST samples were used for acceleration tuning or the deployment gate.

The full regression applies to the archived deployment snapshot. This release
packages the same inference arithmetic with relative imports, temporary native
library builds, and explicit runtime guards. The public planner differs only in
configuration/loading consent checks; its inference/selection method ASTs match
the audited snapshot. Packaging alone is not a new full GPU reproduction: users
must run the equivalence gate on their own installed environment before use.

## Use in a dedicated inference worker

Requirements: Linux, the v1 CUDA/PyTorch environment, and a C compiler named `cc`.
The kernel compiles once per worker using `-fno-fast-math -ffp-contract=off` into
a private temporary directory, not the installed package. Set `TMPDIR` to a
writable scratch filesystem if necessary. No weights or compiled binaries are
included here.

After constructing the released agent, loading its verified artifacts and
calling `initialize()`, move it to the intended GPU and evaluation mode:

```python
import torch
from navsim.agents.actworld_jepa.inference import prepare

torch.backends.cuda.matmul.allow_tf32 = False
# Preserve the original cuDNN precision setting for reference and fast paths.
agent = agent.cuda().eval()  # initialized released agent; FP32, no autocast

# Populate with real AgentInput objects from a fixed NAVTRAIN validation set.
# The audited run used 64 inputs from 64 distinct logs. Never use NAVTEST to
# choose precision, candidates, selector settings, or optimization variants.
report = prepare(agent, validation_inputs=navtrain_validation_inputs)
assert report["exact_outputs"]
trajectory = agent.compute_trajectory(new_agent_input)
```

`prepare` collects reference outputs before installation, verifies metadata-only
feature tensors, installs adapters, then tests exact outputs and API poses.
`pdms_rerun=False` in its return value means **this initialization gate does not
run a simulator**; the separate full regression above does. The adapters retain
the original estimators for checking and fallback.

Important constraints:

- One dedicated agent per worker process, initialized serially. Feature hooks
  are process-global and CUDA capture temporarily stages literal constants;
  do not share that process with training or another model initializer.
- Do not use Python `-O`, autocast, FP16/BF16, or matmul TF32. The measured run
  left the original cuDNN setting unchanged; the gate preserves that setting.
  Do not change parameters,
  device, dtype, selectors, or configuration after installation.
- If installation raises, discard the entire worker, including any partially
  mutated agent, and restart the unmodified inference path. Do not catch and
  continue on that same instance. A second installation is rejected.
- Planner source hashes are pinned. Unknown revisions fail closed. Updating a
  hash requires a fresh source review and equivalence validation, not deleting
  the guard. Existing selector provenance/consent checks remain in force.
- CUDA graphs capture only audited neural regions. Dynamic visible-query
  attention stays eager; its spatial contract is checked on every call.
  New shape signatures receive separate graphs, capped at eight per callable,
  then fall back to eager. Fresh inputs are copied on every replay; returned
  tensors are cloned so later requests cannot overwrite prior outputs.
- Only supported numeric scalar/binary trees are adapted. Categorical HGB and
  nonidentity regression links are rejected; nonfinite forest inputs use the
  original sklearn path. GPU top-k and its tie ordering remain unchanged.

## Tests

From the repository root with the v1 environment installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=navsim \
  python -m pytest -q tests/test_release_contract.py tests/test_inference_acceleration.py
python tools/audit_release.py
```

CPU tests use synthetic toy estimators solely to test numerical/kernel contracts;
they do not fit or tune task models. CUDA graph smoke tests are explicitly
opt-in via `ACTWORLD_TEST_CUDA=1` on a GPU you are authorized to use. These
component tests do not replace the real-input deployment gate or full scoring.
