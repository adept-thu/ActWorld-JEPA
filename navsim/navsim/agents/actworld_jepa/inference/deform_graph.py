"""Keep the original dynamic spatial-size assertion outside CUDA capture."""
import ast
import inspect
import textwrap
import types
import torch
from .cuda_graph_runtime import GraphCall


def install(module, name):
    original = inspect.unwrap(type(module).forward)
    definition = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    definition.decorator_list = []
    expected = "(spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value"
    matches = [i for i, item in enumerate(definition.body)
               if isinstance(item, ast.Assert) and ast.unparse(item.test) == expected]
    assert len(matches) == 1, "Unknown deformable attention validation contract"
    definition.body.pop(matches[0])
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
                 inspect.getsourcefile(original) + ":graph_body", "exec"), namespace)
    graph = GraphCall(types.MethodType(namespace[definition.name], module), name)
    signature = inspect.signature(original)
    def forward(*args, **kwargs):
        bound = signature.bind(module, *args, **kwargs)
        bound.apply_defaults()
        query = bound.arguments["query"]
        value = bound.arguments["value"]
        value = query if value is None else value
        shapes = bound.arguments["spatial_shapes"]
        assert query.dtype == torch.float32 and value.dtype == torch.float32
        num_value = value.shape[1 if module.batch_first else 0]
        # Same check, on EVERY input, not just the first captured input.
        assert (shapes[:, 0] * shapes[:, 1]).sum() == num_value
        return graph(*args, **kwargs)
    module.forward = forward
    return graph
