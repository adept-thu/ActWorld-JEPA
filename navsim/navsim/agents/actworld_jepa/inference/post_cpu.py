"""Gain-set post-selector on CPU, preserving GPU top-k and CPU tree features.

The deployed trees and feature projections already run on CPU. Hoisting the
remaining tiny mask operations avoids repeated GPU launches and synchronization.
Non-gain-set policies retain the original implementation.
"""
import ast
import inspect
import textwrap
import types
import torch


def install(planner):
    original = type(planner)._post_r94_selection
    definition = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    required = sorted({n.slice.value for n in ast.walk(definition)
                       if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                       and n.value.id == "evaluation" and isinstance(n.slice, ast.Constant)
                       and isinstance(n.slice.value, str)})
    top_assignments = [n for n in definition.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "top" for t in n.targets)]
    assert len(top_assignments) == 1
    top_assignments[0].value = ast.Name(id="_fast_top", ctx=ast.Load())
    definition.args.kwonlyargs.append(ast.arg(arg="_fast_top"))
    definition.args.kw_defaults.append(None)
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
                 inspect.getsourcefile(original) + ":post_cpu", "exec"), namespace)
    cpu_function = types.MethodType(namespace[definition.name], planner)
    fallback = planner._post_r94_selection
    projections = {}

    def forward(self, evaluation, baseline, candidates, mother, model=None,
                projection=None, policy=None, feature_names=None):
        if model is None:
            model = self.__dict__.get("post_r94_selector_model")
            projection = self.post_r94_selector_projection
            policy = self.post_r94_selector_policy
            feature_names = self.post_r94_selector_feature_names
        if model is None or policy.get("selection_mode") != "gain_set":
            return fallback(evaluation, baseline, candidates, mother, model,
                            projection, policy, feature_names)
        assert not self.training and baseline.dtype == torch.float32
        # Tie ordering stays on the original GPU implementation.
        top = baseline.topk(k=min(8, baseline.shape[1]), dim=1).indices.cpu()
        values = {key: evaluation[key].detach().cpu() for key in required}
        if id(projection) not in projections:
            projections[id(projection)] = projection.detach().cpu()
        result = cpu_function(values, baseline.detach().cpu(), candidates.detach().cpu(),
                              mother.detach().cpu(), model, projections[id(projection)],
                              policy, feature_names, _fast_top=top)
        return tuple(v.to(baseline.device) for v in result)
    planner._post_r94_selection = types.MethodType(forward, planner)
    return fallback
