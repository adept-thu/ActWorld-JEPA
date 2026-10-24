"""Capture audited neural prefixes, retaining the original CPU selector suffix.

AST slicing preserves every original statement and arithmetic operation. Source
hashes pin the exact frozen r383 deployment; unknown versions fail closed.
"""
import ast
import copy
import inspect
from pathlib import Path
import textwrap
import types
import torch
from .cuda_graph_runtime import GraphCall
from .source_contract import validate_source


def split_prefix(instance, boundary_name, exports):
    original = type(instance).forward
    source_path = Path(inspect.getsourcefile(original))
    validate_source(source_path)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    definition = tree.body[0]
    boundary = [i for i, statement in enumerate(definition.body)
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == boundary_name
                    for target in statement.targets)]
    assert len(boundary) == 1
    index = boundary[0]
    prefix = copy.deepcopy(definition)
    prefix.name = "_captured_prefix"
    prefix.body = prefix.body[:index] + [ast.Return(value=ast.Dict(
        keys=[ast.Constant(name) for name in exports],
        values=[ast.Name(id=name, ctx=ast.Load()) for name in exports]))]
    suffix = copy.deepcopy(definition)
    suffix.name = "_graph_forward"
    argument_names = [a.arg for a in definition.args.args if a.arg != "self"]
    keyword_names = [a.arg for a in definition.args.kwonlyargs]
    call = ast.Call(func=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()),
                                      attr="_fast_neural_prefix", ctx=ast.Load()),
                    args=[ast.Name(id=n, ctx=ast.Load()) for n in argument_names],
                    keywords=[ast.keyword(arg=n, value=ast.Name(id=n, ctx=ast.Load())) for n in keyword_names])
    head = [ast.Assign(targets=[ast.Name(id="_prefix_values", ctx=ast.Store())], value=call)]
    head += [ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())], value=ast.Subscript(
        value=ast.Name(id="_prefix_values", ctx=ast.Load()), slice=ast.Constant(n), ctx=ast.Load())) for n in exports]
    suffix.body = head + suffix.body[index:]
    module = ast.fix_missing_locations(ast.Module(body=[prefix, suffix], type_ignores=[]))
    namespace = dict(original.__globals__)
    exec(compile(module, str(source_path) + ":opt_in_graph", "exec"), namespace)
    wrapper = GraphCall(types.MethodType(namespace["_captured_prefix"], instance), type(instance).__name__)
    instance._fast_neural_prefix = wrapper
    instance.forward = types.MethodType(namespace["_graph_forward"], instance)
    return wrapper


def install_neural_prefix_graphs(model):
    assert not model.training
    planner = model._actworld_planner
    assert not planner.training and not planner.config.finite_checks
    assert planner.config.selection_mode == "conservative"
    assert not planner.config.deploy_refined_selected
    # Activation checkpointing is a backward-memory optimization. The frozen
    # ViT had it enabled even in inference, which also prevents CUDA capture.
    for module in model._backbone.modules():
        if hasattr(module, "use_activation_checkpointing"):
            assert not module.training
            module.use_activation_checkpointing = False
    # torchvision Normalize checks its constant std via a GPU->CPU boolean on
    # every call. Validate those constants once, retaining the same clone and
    # in-place subtract/divide sequence, dtype, and broadcast dimensions.
    transforms = model.transform.transforms
    assert len(transforms) == 1 and type(transforms[0]).__name__ == "Normalize"
    normalization = transforms[0]
    assert not normalization.inplace and all(s != 0 for s in normalization.std)
    parameter = next(model.parameters())
    mean = torch.tensor(normalization.mean, device=parameter.device, dtype=parameter.dtype).view(-1, 1, 1)
    std = torch.tensor(normalization.std, device=parameter.device, dtype=parameter.dtype).view(-1, 1, 1)
    def normalize(x):
        assert x.dtype == mean.dtype and x.device == mean.device
        return x.clone().sub_(mean).div_(std)
    model.transform = normalize
    planner_wrapper = split_prefix(planner, "conservative_selection",
        ["scene", "initial_rollout", "initial_evaluation"])
    # Proposal spatial cross-attention uses data-dependent visible-query
    # lengths. Keep it eager; capturing a warm-up visibility mask is incorrect.
    model_wrapper = GraphCall(model._backbone.forward, "image_encoder")
    model._backbone.forward = model_wrapper
    wrappers = [model_wrapper, planner_wrapper]
    for name, module in model.named_modules():
        if type(module).__name__ == "SpatialCrossAttention" and module.full_attention:
            wrapper = GraphCall(module.encoder.forward, name + ".encoder")
            module.encoder.forward = wrapper
            wrappers.append(wrapper)
        elif type(module).__name__ == "MSDeformableAttention3D":
            from .deform_graph import install
            wrappers.append(install(module, name))
    return wrappers
