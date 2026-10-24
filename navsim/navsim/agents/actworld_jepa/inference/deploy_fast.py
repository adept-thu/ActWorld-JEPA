"""Opt-in r383 FP32 inference runtime, installed only after an equivalence gate.

Use in a dedicated inference worker AFTER initialize(), cuda(), and eval().
No checkpoint/source edits, training, precision changes, or selector removal.
If installation fails, discard this worker/agent and restart the original.
"""
import threading
import inspect
import numpy as np
import torch
from .fast_runtime import install_metadata_only
from .fast_hist import install_tree_adapters
from .graph_prefix import install_neural_prefix_graphs
from .post_cpu import install as install_post_cpu
from .source_contract import validate_source


def prepare(agent, validation_inputs, use_graphs=True, use_cpu_post=True):
    if not __debug__:
        raise RuntimeError("Do not run this assertion-checked adapter with python -O")
    if torch.is_autocast_enabled() or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("Disable autocast and matmul TF32 before the FP32 equivalence gate")
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    if agent.training or not validation_inputs:
        raise ValueError("Need an eval agent and real validation inputs")
    if hasattr(agent, "_fast_runtime_report"):
        raise RuntimeError("Runtime is already installed")
    parameter = next(agent.parameters())
    if parameter.device.type != "cuda" or parameter.dtype != torch.float32:
        raise ValueError("This deployment adapter requires CUDA FP32")
    source_sha256 = validate_source(inspect.getsourcefile(type(agent._pad_model._actworld_planner)))
    keys = ("trajectory", "selected_proposal", "pdm_score", "proposals")
    cpu_features, gpu_features, references = [], [], []
    with torch.no_grad():
        for inp in validation_inputs:
            features = {}
            for builder in agent.get_feature_builders():
                features.update(builder.compute_features(inp))
            cpu_features.append(features)
            features = {k: v.unsqueeze(0).to(parameter.device) for k, v in features.items()}
            gpu_features.append(features)
            output = agent.forward(dict(features))
            references.append({k: output[k].detach().clone() for k in keys})
    original_metadata = install_metadata_only()
    try:
        for idx, inp in enumerate(validation_inputs):
            features = {}
            for builder in agent.get_feature_builders():
                features.update(builder.compute_features(inp))
            assert features.keys() == cpu_features[idx].keys()
            for key in features:
                assert torch.equal(features[key], cpu_features[idx][key]), (idx, key)
    except BaseException:
        from navsim.agents.drive_jepa_perception_based import drive_jepa_features
        drive_jepa_features._get_bev_feature = original_metadata
        raise
    model = agent._pad_model
    adapters = install_tree_adapters(model._actworld_planner, validation_inputs=len(validation_inputs))
    graphs = install_neural_prefix_graphs(model) if use_graphs else []
    if use_cpu_post:
        install_post_cpu(model._actworld_planner)
    with torch.inference_mode():
        for idx, features in enumerate(gpu_features):
            output = agent.forward(dict(features))
            for key in keys:
                if not torch.equal(output[key], references[idx][key]):
                    error = (output[key] - references[idx][key]).abs().max().item()
                    raise RuntimeError(f"Equivalence gate failed: input={idx}, key={key}, error={error}")
        for idx, inp in enumerate(validation_inputs):
            trajectory = agent.compute_trajectory(inp)
            np.testing.assert_array_equal(trajectory.poses, references[idx]["trajectory"][0].cpu().numpy())
    for adapter in adapters:
        assert adapter.validated > 0
        adapter.validate_calls = 0
    forward = agent.forward
    lock = threading.Lock()
    def guarded_forward(features, *args, **kwargs):
        if (agent.training or torch.is_autocast_enabled() or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32 != cudnn_tf32):
            raise RuntimeError("Fast runtime requires eval FP32 and unchanged precision settings")
        with lock, torch.inference_mode():
            return forward(features, *args, **kwargs)
    agent.forward = guarded_forward
    report = dict(validation_inputs=len(validation_inputs), exact_outputs=True,
                  native_models=len(adapters), trees=sum(len(a.roots) for a in adapters),
                  graph_callables=len(graphs), cpu_post=use_cpu_post,
                  precision="FP32", cudnn_allow_tf32=cudnn_tf32, pdms_rerun=False,
                  planner_source_sha256=source_sha256)
    agent._fast_runtime_report = report
    return report
