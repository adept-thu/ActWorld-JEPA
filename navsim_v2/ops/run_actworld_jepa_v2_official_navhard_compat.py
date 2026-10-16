"""Bridge the local ActWorld-JEPA agent to the official reactive NAVHard scorer."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType


OFFICIAL_ROOT = Path(os.environ["OFFICIAL_NAVSIM_ROOT"]).resolve()
V2_ROOT = Path(os.environ["ACTWORLD_V2_ROOT"]).resolve()
V2_AGENTS = V2_ROOT / "navsim/agents"

sys.path.insert(0, str(OFFICIAL_ROOT))
sys.path.append(str(V2_ROOT))

import navsim.agents as agents_package  # noqa: E402
import navsim.common as common_package  # noqa: E402
import navsim.evaluate.pdm_score as official_pdm_score  # noqa: E402
import navsim.planning as planning_package  # noqa: E402
import navsim.planning.simulation.planner.pdm_planner as pdm_planner_package  # noqa: E402
import navsim.planning.training as training_package  # noqa: E402

agents_package.__path__.insert(0, str(V2_AGENTS))
common_package.__path__.append(str(V2_ROOT / "navsim/common"))
planning_package.__path__.append(str(V2_ROOT / "navsim/planning"))
training_package.__path__.append(str(V2_ROOT / "navsim/planning/training"))
# The official reactive runtime predates Drive-JEPA's simulation_v2 helper.
# Keep every official module preferred and expose the local path only as a
# fallback for modules that do not exist in the official package.
pdm_planner_package.__path__.append(
    str(V2_ROOT / "navsim/planning/simulation/planner/pdm_planner")
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


local_pdm_score = _load_module(
    "_actworld_local_pdm_score", V2_ROOT / "navsim/evaluate/pdm_score.py"
)
# Drive-JEPA needs this inference helper, which does not alter the official
# NAVHard scoring function used below.
official_pdm_score.fast_transform_trajectory = (
    local_pdm_score.fast_transform_trajectory
)

# Register the local classes before Hydra resolves the ActWorld config.
from navsim.agents.actworld_jepa.actworld_agent import ActWorldJEPAAgent  # noqa: E402,F401


official = _load_module(
    "navsim.planning.script.run_pdm_score",
    OFFICIAL_ROOT / "navsim/planning/script/run_pdm_score.py",
)


if __name__ == "__main__":
    official.main()
