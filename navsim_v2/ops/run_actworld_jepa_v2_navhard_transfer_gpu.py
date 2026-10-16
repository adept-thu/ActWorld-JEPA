"""Official NAVHARD bridge with CUDA placement and reliable token capture."""

from __future__ import annotations

import importlib.util
import multiprocessing
import os
from pathlib import Path
import sys
import threading

import torch
import torch.nn as nn


bridge_path = Path(os.environ["ACTWORLD_NAVHARD_BASE_BRIDGE"]).resolve()
spec = importlib.util.spec_from_file_location("_actworld_navhard_bridge", bridge_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load compatibility bridge from {bridge_path}")
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)

agent_class = bridge.ActWorldJEPAAgent
original_initialize = agent_class.initialize
original_compute_trajectory = agent_class.compute_trajectory
scene_loader_class = bridge.official.SceneLoader
original_get_agent_input = scene_loader_class.get_agent_input_from_token
input_tokens: dict[int, str] = {}
latest_token: str | None = None
device_lock = threading.Lock()
next_device = 0


def get_agent_input_with_token(self, token: str):
    global latest_token
    agent_input = original_get_agent_input(self, token)
    latest_token = str(token)
    input_tokens[id(agent_input)] = latest_token
    return agent_input


def compute_trajectory_with_token(self, agent_input, *args, **kwargs):
    # Each official process worker evaluates scenes sequentially.  The latest
    # token is therefore a safe fallback when a dataclass wrapper changes the
    # AgentInput object identity between loader and agent boundaries.
    token = input_tokens.pop(id(agent_input), None) or latest_token
    if token is not None:
        self._actworld_selector_token = token
    return original_compute_trajectory(self, agent_input, *args, **kwargs)


def initialize_on_cuda(self) -> None:
    global next_device
    if not torch.cuda.is_available():
        raise RuntimeError("Official NAVHARD evaluation requires CUDA")
    process_identity = multiprocessing.current_process()._identity
    if process_identity:
        logical_device = (process_identity[0] - 1) % torch.cuda.device_count()
    else:
        with device_lock:
            logical_device = next_device % torch.cuda.device_count()
            next_device += 1
    torch.cuda.set_device(logical_device)
    original_initialize(self)
    self.eval()
    for attribute_name in dir(self):
        attribute = getattr(self, attribute_name)
        if isinstance(attribute, nn.Module):
            setattr(self, attribute_name, attribute.cuda())
    self._actworld_navhard_logical_device = logical_device


agent_class.initialize = initialize_on_cuda
agent_class.compute_trajectory = compute_trajectory_with_token
scene_loader_class.get_agent_input_from_token = get_agent_input_with_token


if __name__ == "__main__":
    bridge.official.main()
